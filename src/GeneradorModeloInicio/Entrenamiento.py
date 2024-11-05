import tensorflow as tf
from keras._tf_keras.keras.applications import MobileNetV2
from keras._tf_keras.keras.layers import GlobalAveragePooling2D, Dense
from keras._tf_keras.keras.models import Sequential, load_model
from keras._tf_keras.keras.utils import Progbar
import tensorflow_datasets as tfds
import numpy as np
import signal
import sys
import os

# Configuración de parámetros
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
MODEL_PATH = "Modelo/hasby.h5"
CHECKPOINT_PATH = "Modelo/training_checkpoint"

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

# Variables para el control de interrupción y restauración
initial_epoch = 0

# Cargar el modelo desde el archivo guardado si existe
if os.path.exists(MODEL_PATH):
    print(f"\n\nModelo encontrado en '{MODEL_PATH}', cargando para continuar el entrenamiento.\n\n")
    model = load_model(MODEL_PATH)
    # Restaurar el estado de entrenamiento si hay un checkpoint
    checkpoint = tf.train.Checkpoint(optimizer=model.optimizer, model=model, epoch=tf.Variable(1))
    if os.path.exists(CHECKPOINT_PATH + ".index"):
        checkpoint.restore(CHECKPOINT_PATH)
        initial_epoch = int(checkpoint.epoch.numpy())
        print(f"Estado restaurado. Continuando desde la epoch {initial_epoch}.")
else:
    print("No se encontró un modelo guardado. Cargando el modelo preentrenado.")
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # No entrenar las capas preentrenadas

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Configurar el checkpoint para el nuevo modelo
    checkpoint = tf.train.Checkpoint(optimizer=model.optimizer, model=model, epoch=tf.Variable(1))

# Configurar el manejador de señal para guardar el modelo en caso de interrupción
def save_model_on_interrupt(signal, frame):
    print("\nInterrupción recibida, guardando el estado del modelo...")
    checkpoint.epoch.assign(initial_epoch)  # Guardar la epoch actual
    checkpoint.save(CHECKPOINT_PATH)
    model.save(MODEL_PATH)
    print(f"Estado guardado y modelo guardado como '{MODEL_PATH}'")
    sys.exit(0)

signal.signal(signal.SIGINT, save_model_on_interrupt)

# Entrenar el modelo con barra de progreso
steps_per_epoch = int(info.splits['train'].num_examples / BATCH_SIZE)
progbar = Progbar(target=EPOCHS * steps_per_epoch)

print("Entrenando el modelo...")

for epoch in range(initial_epoch, EPOCHS):
    print(f'\nEpoch {epoch + 1}/{EPOCHS}')
    for step, (X_batch, y_batch) in enumerate(train_dataset):
        # Entrenar en el lote
        loss, accuracy = model.train_on_batch(X_batch, y_batch)

        # Actualizar la barra de progreso
        progbar.update(epoch * steps_per_epoch + step + 1, values=[('loss', loss), ('accuracy', accuracy)])

    # Guardar el estado después de cada epoch
    checkpoint.epoch.assign(epoch + 1)
    checkpoint.save(CHECKPOINT_PATH)
    model.save(MODEL_PATH)

print("Entrenamiento completo.")

# Guardar el modelo final
model.save(MODEL_PATH)
print(f"Modelo guardado como '{MODEL_PATH}'")
