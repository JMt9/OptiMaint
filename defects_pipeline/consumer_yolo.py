import torch
from confluent_kafka import Consumer
import numpy as np
from ultralytics import YOLO
import os
from PIL import Image


# Configuración Kafka
consumer_conf = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'consumer_yolo',
    'auto.offset.reset': 'earliest'
}

# Crear el consumer de Kafka con la configuración anterior
consumer = Consumer(consumer_conf)

# Suscribirse al topic de entrada
topic_in = 'imagenes_yolo'
consumer.subscribe([topic_in])


# Crear carpeta de salida
output_dir = r"C:\Users\ireme\PycharmProjects\Reto2\yolo_defect\processed_images"
os.makedirs(output_dir, exist_ok=True)  # crea la carpeta si no existe


# Cargar modelo YOLO
model_yolo = YOLO(r"C:\Users\ireme\PycharmProjects\Reto2\best.pt")
device = 'cpu'
model_yolo.to(device)


# Loop principal
try:
    print(f"YOLO Consumer iniciado. Escuchando topic '{topic_in}'...")

    img_id = 0  # Contador de imágenes procesadas

    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            print(f"Error: {msg.error()}")
            continue

        # Reconstruir imagen desde bytes
        msg_bytes = msg.value()
        img_np = np.frombuffer(msg_bytes, dtype=np.float32).reshape(3, 256, 256)
        img_tensor = torch.tensor(img_np)

        # Normalización para YOLO
        img_yolo = (img_tensor.permute(1, 2, 0) * 255).clamp(0, 255).byte().numpy()
        img_yolo = np.ascontiguousarray(img_yolo)

        # Guardar imagen original con nombre único
        img_name = f"img_{img_id}.jpg"
        img_path = os.path.join(output_dir, img_name)
        img_pil = Image.fromarray(img_yolo)
        img_pil.save(img_path)

        # Pasar la imagen guardada a YOLO para procesar y guardar con boxes
        results = model_yolo.predict(
            source=img_path,
            conf=0.2,
            save=True,
            save_dir=output_dir,
            show=False
        )

        # Debug: número de detecciones
        num_detections = len(results[0].boxes) if hasattr(results[0], 'boxes') else 0
        print(f"Imagen {img_id} evaluada -> {num_detections} objetos detectados")

        img_id += 1  # Incrementar contador

except KeyboardInterrupt:
    print("YOLO Consumer detenido por el usuario.")
finally:
    consumer.close()
    print("YOLO Consumer cerrado.")
