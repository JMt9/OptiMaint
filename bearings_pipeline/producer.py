from confluent_kafka import Producer

import csv
import json
import datetime
import time


class BearingFeaturesProducer:
    # CONFIGURACION
    BOOTSTRAP_SERVERS = "127.0.0.1:9092"
    CLIENT_ID = "bearing-producer"
    TOPIC_OUT = "bearing_features"

    # CSV
    INPUT_CSV_PATH = "data/features_kafka.csv"

    # Mapeo de columnas del CSV -> nombres estándar del pipeline
    # CSV:  id, fault, fault2, file, ...features...
    COL_SAMPLE_ID_SRC = "id"
    COL_LABEL_SRC = "fault"
    COL_SUBLABEL_SRC = "fault2"
    COL_FILE_SRC = "file"

    # nombres estándar (lo que esperan los consumers)
    COL_SAMPLE_ID_DST = "sample_id"
    COL_LABEL_DST = "label"
    COL_SUBLABEL_DST = "sublabel"
    COL_FILE_DST = "file_path"

    # Throughput / control
    FLUSH_EVERY_N = 500 # flush cada N mensajes
    SLEEP_EVERY_N = 0
    SLEEP_SECONDS = 0.0

    def __init__(self) -> None:
        conf = {
            "bootstrap.servers": self.BOOTSTRAP_SERVERS,
            "client.id": self.CLIENT_ID,
        }
        self.producer = Producer(conf)
        self._sent = 0
        self._delivered_ok = 0
        self._delivered_err = 0

    def delivery_report(self, err, msg):
        if err is not None:
            self._delivered_err += 1
            print(f"[DELIVERY-ERROR] {err}")
        else:
            self._delivered_ok += 1

    def _normalize_row(self, row: dict) -> dict:
        """
        Normaliza la fila del CSV a los nombres estándar del pipeline:
          id -> sample_id
          fault -> label
          fault2 -> sublabel
          file -> file_path

        Mantiene el resto de columnas (features) tal cual.
        """
        payload = dict(row) # copia completa (incluye features)

        # si no existe la key, deja None
        payload[self.COL_SAMPLE_ID_DST] = payload.pop(self.COL_SAMPLE_ID_SRC, None)
        payload[self.COL_LABEL_DST] = payload.pop(self.COL_LABEL_SRC, None)
        payload[self.COL_SUBLABEL_DST] = payload.pop(self.COL_SUBLABEL_SRC, None)
        payload[self.COL_FILE_DST] = payload.pop(self.COL_FILE_SRC, None)

        # Limpieza mínima (strings)
        if payload[self.COL_SAMPLE_ID_DST] is not None:
            payload[self.COL_SAMPLE_ID_DST] = str(payload[self.COL_SAMPLE_ID_DST]).strip()

        if payload[self.COL_LABEL_DST] is not None:
            payload[self.COL_LABEL_DST] = str(payload[self.COL_LABEL_DST]).strip()

        # sublabel puede venir vacío
        if payload[self.COL_SUBLABEL_DST] is not None:
            payload[self.COL_SUBLABEL_DST] = str(payload[self.COL_SUBLABEL_DST]).strip()

        return payload

    def run(self) -> None:
        t0 = time.time()

        with open(self.INPUT_CSV_PATH, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise ValueError("El CSV no tiene cabecera (fieldnames).")

            # Check de columnas las meta
            required = {self.COL_SAMPLE_ID_SRC, self.COL_LABEL_SRC, self.COL_SUBLABEL_SRC, self.COL_FILE_SRC}
            missing = [c for c in required if c not in reader.fieldnames]
            if missing:
                raise ValueError(
                    f"Faltan columnas esperadas en {self.INPUT_CSV_PATH}: {missing}. "
                    f"Cabecera detectada: {reader.fieldnames}"
                )

            for row in reader:
                payload = self._normalize_row(row)
                payload["timestamp_producer"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

                json_value = json.dumps(payload, ensure_ascii=False)

                self.producer.produce(
                    topic=self.TOPIC_OUT,
                    value=json_value,
                    callback=self.delivery_report,
                )
                self.producer.poll(0)

                self._sent += 1

                if self.FLUSH_EVERY_N > 0 and (self._sent % self.FLUSH_EVERY_N == 0):
                    self.producer.flush()
                    dt = time.time() - t0
                    print(
                        f"[PRODUCER] sent={self._sent} delivered_ok={self._delivered_ok} "
                        f"delivered_err={self._delivered_err} elapsed_s={dt:.1f}"
                    )

                if self.SLEEP_EVERY_N > 0 and (self._sent % self.SLEEP_EVERY_N == 0) and self.SLEEP_SECONDS > 0:
                    time.sleep(self.SLEEP_SECONDS)

        self.producer.flush()
        dt = time.time() - t0
        print(
            f"[PRODUCER-DONE] sent={self._sent} delivered_ok={self._delivered_ok} "
            f"delivered_err={self._delivered_err} elapsed_s={dt:.1f}"
        )


if __name__ == "__main__":
    app = BearingFeaturesProducer()
    app.run()
