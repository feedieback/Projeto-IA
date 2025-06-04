import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configurações
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 50

def create_improved_model():
    model = models.Sequential([
        # Primeira camada convolucional
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Segunda camada convolucional
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Terceira camada convolucional
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Quarta camada convolucional
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Camadas densas
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model

def load_and_preprocess_data(data_dir):
    images = []
    labels = []
    
    # Carregar imagens de olhos abertos
    open_eyes_dir = os.path.join(data_dir, 'open')
    for img_name in os.listdir(open_eyes_dir):
        img_path = os.path.join(open_eyes_dir, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(1)  # 1 para olhos abertos
    
    # Carregar imagens de olhos fechados
    closed_eyes_dir = os.path.join(data_dir, 'closed')
    for img_name in os.listdir(closed_eyes_dir):
        img_path = os.path.join(closed_eyes_dir, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(0)  # 0 para olhos fechados
    
    return np.array(images), np.array(labels)

def main():
    # Carregar e preparar dados
    print("Carregando dados...")
    X, y = load_and_preprocess_data('dataset/eyes')
    
    # Normalizar imagens
    X = X.astype('float32') / 255.0
    
    # Dividir em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Data Augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2],
        zoom_range=0.2
    )
    
    # Criar e compilar modelo
    print("Criando modelo...")
    model = create_improved_model()
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        )
    ]
    
    # Treinar modelo
    print("Iniciando treinamento...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=callbacks
    )
    
    # Avaliar modelo
    print("\nAvaliando modelo...")
    test_loss, test_acc, test_auc = model.evaluate(X_test, y_test)
    print(f"\nAcurácia no conjunto de teste: {test_acc:.4f}")
    print(f"AUC no conjunto de teste: {test_auc:.4f}")
    
    # Salvar modelo Keras
    print("\nSalvando modelo Keras...")
    model.save('modelo_olho_improved.keras')
    
    # Converter para TFLite
    print("Convertendo para TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    # Salvar modelo TFLite
    with open('modelo_olho_improved.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print("Processo concluído!")

if __name__ == "__main__":
    main() 