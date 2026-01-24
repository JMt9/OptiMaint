from confluent_kafka import Consumer, Producer, KafkaException

import os
import json
import datetime
import time
from typing import Dict, Any, List

import pandas as pd
import joblib


class BearingLevel1Consumer:
    # CONFIGURACION
    # Nivel 1 broker (entrada)
    BOOTSTRAP_SERVERS = "127.0.0.1:9092"
    GROUP_ID = "bearing-consumer-lvl1"
    AUTO_OFFSET_RESET = "earliest"
    TOPIC_IN = "bearing_features"

    # Nivel 2 broker (salida over/under)
    LEVEL2_BOOTSTRAP_SERVERS = "127.0.0.1:9092"
    TOPIC_OVERHANG_OUT = "bearing_overhang_features"
    TOPIC_UNDERHANG_OUT = "bearing_underhang_features"
    LEVEL2_CLIENT_ID = "bearing-lvl1-reroute-producer"

    # Modelo principal
    FINAL_MODEL_BUNDLE_PATH = "models/final_model/final_model.pkl"

    # Lista de features usadas en entrenamiento (CSV con columna 'feature')
    RF_FEATURES_CSV = r"C:\Users\Usuario\Desktop\master\1er_semestre\parte2\reto02\models\final_model\features_info\rf_selected_features_for_gridsearch.csv"
    RF_FEATURES_COLNAME = "feature"

    # Datalake root
    DATALAKE_ROOT = "bearings_datalake"

    # Buffers / throughput
    BATCH_WRITE_N = 200
    PRODUCER_FLUSH_EVERY_N = 200
    LOG_EVERY_N = 200

    # Validaciones
    STRICT_REQUIRE_ALL_FEATURES = True
    LOG_MISSING_FEATURES_EVERY_N = 50
    MAX_MISSING_FEATURES_TO_PRINT = 12

    # Labels
    LABELS_6 = [
        "horizontal-misalignment",
        "imbalance",
        "normal",
        "overhang",
        "underhang",
        "vertical-misalignment",
    ]

    LABEL_TO_FOLDER = {
        "horizontal-misalignment": "horizontal_misalignment",
        "vertical-misalignment": "vertical_misalignment",
        "imbalance": "imbalance",
        "normal": "normal",
        "overhang": "overhang",
        "underhang": "underhang",
    }

    def __init__(self) -> None:
        # Consumer (nivel 1)
        self.consumer = Consumer(
            {
                "bootstrap.servers": self.BOOTSTRAP_SERVERS,
                "group.id": self.GROUP_ID,
                "auto.offset.reset": self.AUTO_OFFSET_RESET,
            }
        )
        self.consumer.subscribe([self.TOPIC_IN])

        # Producer (nivel 2)
        self.producer = Producer(
            {
                "bootstrap.servers": self.LEVEL2_BOOTSTRAP_SERVERS,
                "client.id": self.LEVEL2_CLIENT_ID,
            }
        )

        # Load modeL
        self.bundle = joblib.load(self.FINAL_MODEL_BUNDLE_PATH)
        if not isinstance(self.bundle, dict) or "model" not in self.bundle:
            raise ValueError("final_model.pkl debe ser un dict con al menos key: 'model'.")

        self.model = self.bundle["model"]
        self.label_encoder = self.bundle.get("label_encoder", None)

        # Load features list
        self.features: List[str] = self._load_features_list(self.RF_FEATURES_CSV, self.RF_FEATURES_COLNAME)
        if not self.features:
            raise ValueError(f"No se han cargado features desde {self.RF_FEATURES_CSV}.")

        # Aviso si bundle trae otra lista distinta
        bundle_feats = self.bundle.get("features", None)
        if isinstance(bundle_feats, list) and bundle_feats:
            bf = [str(x).strip() for x in bundle_feats]
            if set(bf) != set(self.features):
                only_csv = sorted(set(self.features) - set(bf))
                only_bundle = sorted(set(bf) - set(self.features))
                print("[WARN] Inconsistencia entre features del bundle y del CSV de features.")
                if only_csv:
                    print(f"  - Solo en RF_FEATURES_CSV (n={len(only_csv)}): {only_csv[:12]}")
                if only_bundle:
                    print(f"  - Solo en bundle['features'] (n={len(only_bundle)}): {only_bundle[:12]}")

        self.counts = {lab: 0 for lab in self.LABELS_6}
        self.processed = 0
        self._last_write_at = 0
        self._missing_feat_counter = 0
        self._skipped_missing = 0

        # sets (por sample_id)
        self.id_sets = {
            "horizontal_misalignment": self._load_existing_ids(self._path_class_csv("horizontal_misalignment")),
            "imbalance": self._load_existing_ids(self._path_class_csv("imbalance")),
            "vertical_misalignment": self._load_existing_ids(self._path_class_csv("vertical_misalignment")),
            "normal": self._load_existing_ids(self._path_class_csv("normal")),
            # over/under también son MIN (id+path) en su CSV
            "overhang": self._load_existing_ids(self._path_class_csv("overhang")),
            "underhang": self._load_existing_ids(self._path_class_csv("underhang")),
        }

        # filas min: sample_id + file_path para todas las salidas en disco incluidos over/under
        self.buf_rows_min = {k: [] for k in self.id_sets.keys()}
        self._ensure_dirs()


    # Features list que usa el final_model:
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

    # Paths:
    def _ensure_dirs(self) -> None:
        # normal/
        os.makedirs(os.path.join(self.DATALAKE_ROOT, "normal"), exist_ok=True)

        # fallo/<tipo>/
        base_fallo = os.path.join(self.DATALAKE_ROOT, "fallo")
        for folder in ["horizontal_misalignment", "imbalance", "vertical_misalignment", "overhang", "underhang"]:
            os.makedirs(os.path.join(base_fallo, folder), exist_ok=True)

    def _path_class_csv(self, key_folder: str) -> str:
        # normal va fuera de fallo/
        if key_folder == "normal":
            return os.path.join(self.DATALAKE_ROOT, "normal", "normal.csv")
        return os.path.join(self.DATALAKE_ROOT, "fallo", key_folder, f"{key_folder}.csv")

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

    # Preparamos X (solo features, SIN imputación)
    def _prepare_X(self, record: Dict[str, Any], sample_id: str) -> pd.DataFrame:
        missing = [f for f in self.features if f not in record]
        if missing:
            self._missing_feat_counter += 1
            if (
                self.LOG_MISSING_FEATURES_EVERY_N > 0
                and (self._missing_feat_counter % self.LOG_MISSING_FEATURES_EVERY_N == 0)
            ):
                shown = missing[: self.MAX_MISSING_FEATURES_TO_PRINT]
                print(
                    f"[WARN] sample_id={sample_id} faltan {len(missing)} features del modelo. "
                    f"Ejemplo: {shown}"
                )
            if self.STRICT_REQUIRE_ALL_FEATURES:
                self._skipped_missing += 1
                raise KeyError(f"Faltan {len(missing)} features para sample_id={sample_id}")

        row = {feat: float(record[feat]) for feat in self.features}
        return pd.DataFrame([row], columns=self.features)


    # Kafka reroute (manda FULL payload)
    def delivery_report(self, err, msg):
        if err is not None:
            print(f"[REROUTE-DELIVERY-ERROR] {err}")

    def _reroute(self, topic: str, payload: Dict[str, Any]) -> None:
        j = json.dumps(payload, ensure_ascii=False)
        self.producer.produce(topic=topic, value=j, callback=self.delivery_report)
        self.producer.poll(0)


    # Main
    def run(self) -> None:
        t0 = time.time()
        try:
            while True:
                msg = self.consumer.poll(1.0)

                # si no hay msg, termina
                if msg is None:
                    print("[CONSUMER-LVL1] msg is None -> terminando.")
                    break

                if msg.error():
                    print(f"[CONSUMER-LVL1-ERROR] {msg.error()}")
                    continue

                record = json.loads(msg.value().decode("utf-8"))
                record["timestamp_consumido"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

                if "sample_id" not in record or record.get("sample_id") in [None, ""]:
                    print("[WARN] Mensaje sin sample_id -> skip")
                    continue
                sample_id = str(record["sample_id"]).strip()

                file_path = record.get("file_path", "")
                file_path = "" if file_path is None else str(file_path).strip()

                # predict solo con features seleccionadas
                try:
                    X = self._prepare_X(record, sample_id=sample_id)
                except Exception as e:
                    print(f"[SKIP] sample_id={sample_id} no se pudo preparar X: {e}")
                    continue

                try:
                    y_pred = self.model.predict(X)[0]
                except Exception as e:
                    print(f"[PREDICT-ERROR] sample_id={sample_id} error={e}")
                    continue

                if self.label_encoder is not None:
                    try:
                        pred_label = self.label_encoder.inverse_transform([y_pred])[0]
                    except Exception:
                        pred_label = str(y_pred)
                else:
                    pred_label = str(y_pred)

                if pred_label not in self.LABELS_6:
                    print(f"[WARN] pred_label fuera de las 6 clases: {pred_label} (sample_id={sample_id}) -> skip")
                    continue

                self.counts[pred_label] += 1
                self.processed += 1

                out_min = {"sample_id": sample_id, "file_path": file_path}

                if pred_label == "normal":
                    if sample_id not in self.id_sets["normal"]:
                        self.id_sets["normal"].add(sample_id)
                        self.buf_rows_min["normal"].append(out_min)

                elif pred_label in ["horizontal-misalignment", "imbalance", "vertical-misalignment"]:
                    folder = self.LABEL_TO_FOLDER[pred_label]
                    if sample_id not in self.id_sets[folder]:
                        self.id_sets[folder].add(sample_id)
                        self.buf_rows_min[folder].append(out_min)

                elif pred_label == "overhang":
                    # DISCO: id+path
                    if sample_id not in self.id_sets["overhang"]:
                        self.id_sets["overhang"].add(sample_id)
                        self.buf_rows_min["overhang"].append(out_min)

                    # KAFKA: manda entero
                    out_full = dict(record)
                    out_full["pred_label"] = pred_label
                    self._reroute(self.TOPIC_OVERHANG_OUT, out_full)

                elif pred_label == "underhang":
                    # DISCO: id+path
                    if sample_id not in self.id_sets["underhang"]:
                        self.id_sets["underhang"].add(sample_id)
                        self.buf_rows_min["underhang"].append(out_min)

                    # KAFKA: manda entero
                    out_full = dict(record)
                    out_full["pred_label"] = pred_label
                    self._reroute(self.TOPIC_UNDERHANG_OUT, out_full)

                # flush a disco
                if (self.processed - self._last_write_at) >= self.BATCH_WRITE_N:
                    self._flush_to_disk()
                    self._last_write_at = self.processed

                # flush producer
                if self.PRODUCER_FLUSH_EVERY_N > 0 and (self.processed % self.PRODUCER_FLUSH_EVERY_N == 0):
                    self.producer.flush()

                # log
                if self.LOG_EVERY_N > 0 and (self.processed % self.LOG_EVERY_N == 0):
                    dt = time.time() - t0
                    print(
                        f"[LVL1] processed={self.processed} "
                        f"h={self.counts['horizontal-misalignment']} "
                        f"i={self.counts['imbalance']} "
                        f"n={self.counts['normal']} "
                        f"o={self.counts['overhang']} "
                        f"u={self.counts['underhang']} "
                        f"v={self.counts['vertical-misalignment']} "
                        f"skipped_missing={self._skipped_missing} "
                        f"elapsed_s={dt:.1f}"
                    )

            # final flush
            self._flush_to_disk()
            self.producer.flush()

        except KafkaException as e:
            print(f"[KAFKA-EXCEPTION] {e}")
        finally:
            self.consumer.close()
            try:
                self.producer.flush()
            except Exception:
                pass
            print("[CONSUMER-LVL1] Cerrado.")

    def _flush_to_disk(self) -> None:
        for folder_key, rows_buf in self.buf_rows_min.items():
            if not rows_buf:
                continue
            self._append_rows_csv(self._path_class_csv(folder_key), rows_buf)
            self.buf_rows_min[folder_key] = []


if __name__ == "__main__":
    app = BearingLevel1Consumer()
    app.run()
