import torch
from confluent_kafka import Consumer, Producer
import torch.nn as nn
from torchvision import models
import numpy as np
import pandas as pd
from datetime import datetime


# Configuración Kafka

consumer_conf = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'consumer_binario',
    'auto.offset.reset': 'earliest'
}

producer_conf = {'bootstrap.servers': 'localhost:9092'}

# Creamos los objetos Consumer y Producer
consumer = Consumer(consumer_conf)
producer = Producer(producer_conf)

# Topics de entrada y salida
topic_in = 'imagenes_dataset'
topic_out = 'imagenes_yolo'

# Suscribirse al topic
consumer.subscribe([topic_in])

# Función callback para comprobar si la entrega del mensaje fue exitosa
def delivery_report(err, msg):
    if err is not None:
        print(f'Delivery failed: {err}')
    else:
        print(f'Mensaje entregado al topic {msg.topic()}')


# Cargar un modelo ResNet18 preentrenado en ImageNet
model_res = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Cambiar la capa final para que sea un clasificador binario (1 salida)
model_res.fc = nn.Linear(512, 1)

# Cargar los pesos entrenados previamente
model_res.load_state_dict(torch.load('model_res_weights.pth'))

# Poner el modelo en modo de evaluación
model_res.eval()

# Definir el dispositivo
device = 'cpu'
model_res.to(device)

# Contador solo de imágenes con error
counter = 0


# Preparar CSV para guardar resultados
results_df = pd.DataFrame(columns=['id_imagen', 'pred', 'probabilidad', 'fecha_hora'])


# Loop principal
try:
    print(f"Consumer iniciado. Escuchando topic '{topic_in}'...")

    img_id = 0  # Contador global de todas las imágenes procesadas

    while True:
        msg = consumer.poll(1.0)
        if msg is None:

            continue
        if msg.error():
            print(f"Error: {msg.error()}")
            continue

        # Reconstruir imagen desde bytes
        msg_bytes = msg.value()
        img_np = np.frombuffer(msg_bytes, dtype=np.float32).reshape(3,256,256)
        img_tensor = torch.tensor(img_np).unsqueeze(0)

        # Predicción con el modelo
        with torch.no_grad():
            output = model_res(img_tensor)
            prob = torch.sigmoid(output)
            pred = (prob > 0.5).int()

        # Guardar todas las imágenes procesadas en CSV
        results_df = pd.concat([results_df, pd.DataFrame({
            'id_imagen': [img_id],
            'pred': [pred.item()],
            'probabilidad': [prob.item()],
            'fecha_hora': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        })], ignore_index=True)

        # Solo enviar al topic YOLO si hay error
        if pred.item() == 1:
            counter += 1
            message_bytes = img_tensor.cpu().numpy().tobytes()

            producer.produce(
                topic_out,
                value=message_bytes,
                callback=delivery_report
            )
            producer.poll(0)
            print(f"Imagen con error enviada a YOLO -> id: {img_id}, contador error: {counter}")

        img_id += 1  # Incrementar contador global de imágenes

except KeyboardInterrupt:
    print("Consumer detenido por el usuario.")
finally:
    consumer.close()
    producer.flush()
    print("Consumer/Producer cerrado.")

    # Guardar CSV con todos los resultados
    results_df.to_csv('resultados_modelo.csv', index=False)
    print("Resultados guardados en 'resultados_modelo.csv'")
