import torch
import time
from confluent_kafka import Producer


# Configuración Kafka

conf = {'bootstrap.servers': 'localhost:9092'}

# Creamos el productor Kafka con la configuración definida
producer = Producer(conf)

# Nombre del topic al que se enviarán los mensajes (imágenes)
topic_name = 'imagenes_dataset'

# Función callback que se llama después de enviar un mensaje.
# Permite verificar si la entrega fue exitosa o hubo algún error.
def delivery_report(err, msg):
    if err is not None:
        print(f'Delivery failed: {err}')
    else:
        print(f'Mensaje entregado al topic {msg.topic()}')


# Cargar dataset
data = torch.load("dataset_test.pt")

# Extraemos las imágenes y etiquetas del dataset
test_images = data['images']
test_labels = data['labels']


# Loop principal para enviar imágenes
image_counter = 0
try:
    print(f"Producer iniciado. Enviando {len(test_images)} imágenes al topic '{topic_name}'...")

    # Iteramos sobre todas las imágenes del dataset
    for i in range(len(test_images)):
        img_tensor = test_images[i]

        # Convertimos el tensor a bytes para enviarlo por Kafka
        message_bytes = img_tensor.cpu().numpy().tobytes()

        # Enviamos el mensaje al topic Kafka
        producer.produce(
            topic_name,
            value=message_bytes,
            callback=delivery_report
        )

        # Llamada no bloqueante para procesar eventos del productor
        producer.poll(0)

        # Mostramos en consola qué imagen se ha enviado
        print(f"Imagen enviada -> id img_{image_counter}")
        image_counter += 1

        # Pequeña pausa entre envíos para no saturar el broker
        time.sleep(0.5)

except KeyboardInterrupt:
    print("Producer detenido por el usuario.")

#Asegura que todos los mensajes pendientes sean enviados antes de cerrar
finally:
    producer.flush()
    print("Producer cerrado.")