import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# ===============================
# 🔧 Configurações
# ===============================
IMG_SIZE = (64, 64)
BATCH_SIZE = 16
EPOCHS = 10  # Número extra de épocas

# ===============================
# 💾 Carregar modelo pré-treinado
# ===============================
model = load_model('modelo_olho.keras')
print("Modelo carregado com sucesso!")

# ===============================
# 📈 Data Augmentation (opcional para melhorar mais)
# ===============================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    'dataset',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    'dataset',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# ===============================
# 🚀 Continuação do treinamento
# ===============================
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)

# ===============================
# 📈 Plot dos resultados
# ===============================
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 6))
plt.plot(acc, label='Acurácia Treino')
plt.plot(val_acc, label='Acurácia Validação')
plt.title('Acurácia após Fine-Tuning')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(loss, label='Perda Treino')
plt.plot(val_loss, label='Perda Validação')
plt.title('Perda após Fine-Tuning')
plt.legend()
plt.show()

# ===============================
# 💾 Salvar novamente
# ===============================
model.save('modelo_olho_finetuned.keras')
print("Modelo salvo como modelo_olho_finetuned.keras")

# ===============================
# 🔥 Exportar novamente para TensorFlow Lite
# ===============================
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('modelo_olho_finetuned.tflite', 'wb') as f:
    f.write(tflite_model)

print("Modelo .tflite salvo com sucesso!")
