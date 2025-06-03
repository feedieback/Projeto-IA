import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ===============================
# 📁 Pré-requisito: Estrutura das pastas
# Dataset deve estar assim:
# dataset/
# ├── open/       # imagens de olhos abertos
# └── closed/     # imagens de olhos fechados
# ===============================

# ✅ Parâmetros
img_size = (64, 64)  # Tamanho da imagem
batch_size = 32
epochs = 10  # Ajuste para mais se desejar

# ✅ Preprocessamento dos dados
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    'dataset',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    'dataset',
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# ✅ Verificar labels
print("Labels:", train_generator.class_indices)

# ✅ Construindo o modelo (CNN simples)
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # saída binária (0 ou 1)
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ✅ Treinamento
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator
)

# ✅ Salvar o modelo normal (opcional)
model.save('modelo_olho_aberto_fechado.h5')

# ===============================
# 🔥 Converter para TensorFlow Lite (.tflite)
# ===============================
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('modelo_olho_aberto_fechado.tflite', 'wb') as f:
    f.write(tflite_model)

print("Modelo .tflite salvo com sucesso!")
