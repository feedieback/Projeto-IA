import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Caminho para seu dataset
dataset_path = r"C:\Users\a\Documents\GitHub\Projeto-IA\Dataset"

# Parâmetros
img_size = (64, 64)
batch_size = 32
epochs = 35  # Aumentei um pouco, pois o learning rate é menor

# ===============================
# 🔥 Data Augmentation aprimorado
# ===============================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=5,
    zoom_range=0.05,
    shear_range=0.05,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# ===============================
# 🧠 Modelo CNN melhorado
# ===============================
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(2, activation='softmax')  # 2 classes: aberto e fechado
])

model.compile(optimizer=Adam(learning_rate=0.00005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ===============================
# 🚀 Callbacks
# ===============================
callbacks = [
    EarlyStopping(patience=6, restore_best_weights=True),
    ModelCheckpoint('model_eye_asian_optimized.h5', save_best_only=True)
]

# ===============================
# 🚀 Treinamento
# ===============================
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=callbacks
)

# ===============================
# 💾 Salvar modelo final
# ===============================
model.save("model_eye_asian_optimized_final.h5")
