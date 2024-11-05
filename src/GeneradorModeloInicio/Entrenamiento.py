import tensorflow as tf
from keras._tf_keras.keras.applications import MobileNetV2
from keras._tf_keras.keras.layers import GlobalAveragePooling2D, Dense
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.utils import Progbar
import tensorflow_datasets as tfds
import numpy as np


# Configuración de parámetros
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# Filtrar el dataset COCO para que solo contenga vehículos
VEHICLE_CLASSES = [3, 6, 8]  # IDs de COCO para "car", "bus", "truck"

def filter_vehicle_classes(example):
    labels = example['objects']['label']
    mask = tf.reduce_any([tf.reduce_any(labels == class_id) for class_id in VEHICLE_CLASSES])
    return mask

# Cargar el dataset y aplicar el filtro
dataset, info = tfds.load("coco/2017", split='train', with_info=True)
vehicle_dataset = dataset.filter(filter_vehicle_classes)

# Preprocesamiento de las imágenes
def preprocess(example):
    image = example['image']
    image = tf.image.resize(image, IMG_SIZE) / 255.0  # Escalar a [0, 1]
    label = example['objects']['label']
    label = tf.reduce_any([label == class_id for class_id in VEHICLE_CLASSES])
    label = tf.cast(label, tf.float32)  # Convertimos a flotante para la clasificación binaria
    return image, label

vehicle_dataset = vehicle_dataset.map(preprocess)
train_dataset = vehicle_dataset.shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Cargar el modelo preentrenado (MobileNetV2) y configurarlo para transfer learning
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # No entrenar las capas preentrenadas

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(1, activation='sigmoid')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo con barra de progreso
steps_per_epoch = int(info.splits['train'].num_examples / BATCH_SIZE)
progbar = Progbar(target=EPOCHS * steps_per_epoch)

print("Entrenando el modelo...")

for epoch in range(EPOCHS):
    print(f'\nEpoch {epoch + 1}/{EPOCHS}')
    for step, (X_batch, y_batch) in enumerate(train_dataset):
        # Entrenar en el lote
        loss, accuracy = model.train_on_batch(X_batch, y_batch)

        # Actualizar la barra de progreso
        progbar.update(epoch * steps_per_epoch + step + 1, values=[('loss', loss), ('accuracy', accuracy)])

print("Entrenamiento completo.")

# Guardar el modelo
model.save("mi_modelo_de_vehiculos.h5")
print("Modelo guardado como 'mi_modelo_de_vehiculos.h5'")
