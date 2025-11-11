# cnn_train.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import datetime
import os

# 📁 Ruta al dataset
DATASET_PATH = 'dataset_emociones'

# 📐 Parámetros
#IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 30
# 📊 TensorBoard
log_dir = "logs/30epocas_emociones_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# 🛑 EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 🔄 Aumento de datos + validación
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# 🧠 Modelo CNN con regularización
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(48, 48, 1)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(6, activation='softmax'),
    layers.Dropout(0.5),  # Regularización
    layers.Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 🚀 Entrenamiento
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early_stop, tensorboard_callback]
)

# 💾 Guardar modelo
os.makedirs('app', exist_ok=True)
model.save('modelo_30E_emociones_cnn.h5')
print("✅ Modelo guardado como modelo_30E_emociones_cnn.h5")