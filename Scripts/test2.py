import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import os

# ===============================
# 🔧 Configurações
# ===============================
IMG_SIZE = (64, 64)  # Tamanho das imagens
BATCH_SIZE = 32
EPOCHS = 15  # Aumente para melhorar

# ===============================
# 📁 Estrutura dos dados:
# dataset/
# ├── open/    (imagens de olhos abertos)
# └── closed/  (imagens de olhos fechados)
# ===============================

# ===============================
# 📈 Pré-processamento e Augmentação
# ===============================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    'dataset',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_data = datagen.flow_from_directory(
    'dataset',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

print("Classes:", train_data.class_indices)

# ===============================
# 🧠 Construindo a Rede Neural CNN
# ===============================
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ===============================
# 🚀 Treinamento
# ===============================
history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=val_data
)

# ===============================
# ✅ Avaliação
# ===============================
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 6))
plt.plot(acc, label='Acurácia Treino')
plt.plot(val_acc, label='Acurácia Validação')
plt.title('Acurácia do Modelo')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(loss, label='Perda Treino')
plt.plot(val_loss, label='Perda Validação')
plt.title('Perda do Modelo')
plt.legend()
plt.show()

# ===============================
# 💾 Salvar modelo
# ===============================
model.save('modelo_olho_aberto_fechado.keras')
print("Modelo salvo como modelo_olho_aberto_fechado.keras")

# ===============================
# 🔥 Exportar para TensorFlow Lite
# ===============================
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('modelo_olho_aberto_fechado.tflite', 'wb') as f:
    f.write(tflite_model)

print("Modelo .tflite salvo com sucesso!")
