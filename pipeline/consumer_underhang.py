from confluent_kafka import Consumer, KafkaException

import os
import json
import datetime
import time
from typing import Dict, Any, List

import pandas as pd
import joblib


class BearingUnderhangConsumer:
    # =========================
    # CONFIG (EDITA AQUI)
    # =========================
    BOOTSTRAP_SERVERS = "127.0.0.1:9092"
    GROUP_ID = "bearing-consumer-underhang"
    AUTO_OFFSET_RESET = "earliest"

    TOPIC_IN = "bearing_underhang_features"

    # Modelo
    UNDERHANG_MODEL_BUNDLE_PATH = "models/underhang/underhang_model.pkl"

    # Features usadas por el submodelo (CSV con columna 'feature')
    RF_FEATURES_CSV = r"C:\Users\Usuario\Desktop\master\1er_semestre\parte2\reto02\models\underhang\features_info\rf_selected_features_for_gridsearch.csv"
    RF_FEATURES_COLNAME = "feature"

    # Datalake root
    DATALAKE_ROOT = "datalake_bearings"

    # Buffers / logs
    BATCH_WRITE_N = 200
    LOG_EVERY_N = 200

    STRICT_REQUIRE_ALL_FEATURES = True
    LOG_MISSING_FEATURES_EVERY_N = 50
    MAX_MISSING_FEATURES_TO_PRINT = 12

    SUBLABELS = ["ball_fault", "cage_fault", "outer_race"]

    def __init__(self) -> None:
        self.consumer = Consumer(
            {
                "bootstrap.servers": self.BOOTSTRAP_SERVERS,
                "group.id": self.GROUP_ID,
                "auto.offset.reset": self.AUTO_OFFSET_RESET,
            }
        )
        self.consumer.subscribe([self.TOPIC_IN])

        self.bundle = joblib.load(self.UNDERHANG_MODEL_BUNDLE_PATH)
        if not isinstance(self.bundle, dict) or "model" not in self.bundle:
            raise ValueError("underhang_model.pkl debe ser dict con al menos key: 'model'.")

        self.model = self.bundle["model"]
        self.label_encoder = self.bundle.get("label_encoder", None)

        # Features desde CSV (no del bundle)
        self.features: List[str] = self._load_features_list(self.RF_FEATURES_CSV, self.RF_FEATURES_COLNAME)
        if not self.features:
            raise ValueError(f"No se han cargado features desde {self.RF_FEATURES_CSV}.")

        sub_order = self.bundle.get("subclass_order", None)
        if isinstance(sub_order, list) and set(map(str, sub_order)) != set(self.SUBLABELS):
            print(f"[WARN] subclass_order del bundle != SUBLABELS esperadas. bundle={sub_order} expected={self.SUBLABELS}")

        self.processed = 0
        self.counts = {s: 0 for s in self.SUBLABELS}
        self._last_write_at = 0
        self._missing_feat_counter = 0
        self._skipped_missing = 0

        self._ensure_dirs()

        # CSV completo underhang
        self.full_csv_path = os.path.join(self.DATALAKE_ROOT, "fallo", "underhang", "underhang.csv")
        self.full_ids = self._load_existing_ids(self.full_csv_path)
        self.buf_full_rows: List[Dict[str, Any]] = []

        # Dedupe y buffers por subtipo
        self.id_sets = {}
        self.buf_rows = {}
        for sub in self.SUBLABELS:
            p = self._path_ids_csv(sub)
            self.id_sets[sub] = self._load_existing_ids(p)
            self.buf_rows[sub] = []

    # ---------- dirs/paths ----------
    def _ensure_dirs(self) -> None:
        base = os.path.join(self.DATALAKE_ROOT, "fallo", "underhang")
        os.makedirs(base, exist_ok=True)
        for sub in self.SUBLABELS:
            os.makedirs(os.path.join(base, sub), exist_ok=True)

    def _path_ids_csv(self, sub: str) -> str:
        return os.path.join(self.DATALAKE_ROOT, "fallo", "underhang", sub, f"{sub}.csv")

    # ---------- features list ----------
    def _load_features_list(self, path_csv: str, colname: str) -> List[str]:
        if not os.path.exists(path_csv):
            raise FileNotFoundError(f"No existe RF_FEATURES_CSV: {path_csv}")
        df = pd.read_csv(path_csv)
        if colname not in df.columns:
            raise ValueError(f"RF_FEATURES_CSV no tiene columna '{colname}'. Columnas: {list(df.columns)}")

        feats = df[colname].astype(str).str.strip().tolist()
        feats = [f for f in feats if f and f.lower() != "nan" and f.lower() != colname.lower()]

        seen = set()
        out = []
        for f in feats:
            if f not in seen:
                seen.add(f)
                out.append(f)
        return out

    # ---------- dedupe IO ----------
    def _load_existing_ids(self, path_csv: str) -> set:
        if not os.path.exists(path_csv):
            return set()
        try:
            df = pd.read_csv(path_csv)
            if "sample_id" not in df.columns:
                return set()
            return set(df["sample_id"].astype(str).tolist())
        except Exception as e:
            print(f"[WARN] No se pudo leer {path_csv} para dedupe: {e}")
            return set()

    def _append_rows_csv(self, path_csv: str, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        df = pd.DataFrame(rows)
        header = not os.path.exists(path_csv)
        df.to_csv(path_csv, mode="a", header=header, index=False)

    # ---------- prepare X (sin imputación) ----------
    def _prepare_X(self, record: Dict[str, Any], sample_id: str) -> pd.DataFrame:
        missing = [f for f in self.features if f not in record]
        if missing:
            self._missing_feat_counter += 1
            if self.LOG_MISSING_FEATURES_EVERY_N > 0 and (self._missing_feat_counter % self.LOG_MISSING_FEATURES_EVERY_N == 0):
                shown = missing[: self.MAX_MISSING_FEATURES_TO_PRINT]
                print(f"[WARN] sample_id={sample_id} faltan {len(missing)} features (UNDERHANG). Ejemplo: {shown}")
            if self.STRICT_REQUIRE_ALL_FEATURES:
                self._skipped_missing += 1
                raise KeyError(f"Faltan {len(missing)} features (UNDERHANG) para sample_id={sample_id}")

        row = {feat: float(record[feat]) for feat in self.features}
        return pd.DataFrame([row], columns=self.features)

    # ---------- main ----------
    def run(self) -> None:
        t0 = time.time()
        try:
            while True:
                msg = self.consumer.poll(1.0)

                if msg is None:
                    print("[CONSUMER-UNDERHANG] msg is None -> terminando.")
                    break

                if msg.error():
                    print(f"[CONSUMER-UNDERHANG-ERROR] {msg.error()}")
                    continue

                record = json.loads(msg.value().decode("utf-8"))
                record["timestamp_consumido_lvl2"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

                if "sample_id" not in record or record.get("sample_id") in [None, ""]:
                    print("[WARN] Mensaje sin sample_id -> skip")
                    continue
                sample_id = str(record["sample_id"]).strip()

                file_path = record.get("file_path", "")
                file_path = "" if file_path is None else str(file_path).strip()

                # CSV completo underhang
                if sample_id not in self.full_ids:
                    self.full_ids.add(sample_id)
                    self.buf_full_rows.append(dict(record))

                # Predict
                try:
                    X = self._prepare_X(record, sample_id)
                except Exception as e:
                    print(f"[SKIP] sample_id={sample_id} no se pudo preparar X (UNDERHANG): {e}")
                    continue

                try:
                    y_pred = self.model.predict(X)[0]
                except Exception as e:
                    print(f"[PREDICT-ERROR] (UNDERHANG) sample_id={sample_id} error={e}")
                    continue

                if self.label_encoder is not None:
                    try:
                        pred_sub = self.label_encoder.inverse_transform([y_pred])[0]
                    except Exception:
                        pred_sub = str(y_pred)
                else:
                    pred_sub = str(y_pred)

                if pred_sub not in self.SUBLABELS:
                    print(f"[WARN] pred_sub fuera de {self.SUBLABELS}: {pred_sub} sample_id={sample_id} -> skip")
                    continue

                if sample_id not in self.id_sets[pred_sub]:
                    self.id_sets[pred_sub].add(sample_id)
                    self.buf_rows[pred_sub].append({"sample_id": sample_id, "file_path": file_path})

                self.counts[pred_sub] += 1
                self.processed += 1

                if (self.processed - self._last_write_at) >= self.BATCH_WRITE_N:
                    self._flush_to_disk()
                    self._last_write_at = self.processed

                if self.LOG_EVERY_N > 0 and (self.processed % self.LOG_EVERY_N == 0):
                    dt = time.time() - t0
                    print(
                        f"[UNDERHANG] processed={self.processed} "
                        f"ball={self.counts['ball_fault']} cage={self.counts['cage_fault']} outer={self.counts['outer_race']} "
                        f"skipped_missing={self._skipped_missing} "
                        f"elapsed_s={dt:.1f}"
                    )

            self._flush_to_disk()

        except KafkaException as e:
            print(f"[KAFKA-EXCEPTION] {e}")
        finally:
            self.consumer.close()
            print("[CONSUMER-UNDERHANG] Cerrado.")

    def _flush_to_disk(self) -> None:
        if self.buf_full_rows:
            self._append_rows_csv(self.full_csv_path, self.buf_full_rows)
            self.buf_full_rows = []

        for sub, rows_buf in self.buf_rows.items():
            if not rows_buf:
                continue
            self._append_rows_csv(self._path_ids_csv(sub), rows_buf)
            self.buf_rows[sub] = []


if __name__ == "__main__":
    app = BearingUnderhangConsumer()
    app.run()
