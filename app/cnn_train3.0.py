import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
import os
import numpy as np
from PIL import Image
import gc

gc.collect()

# SET DE IMAGENES
DATASET_DIR= r'C:\Users\Andres\Downloads\Datasets\AffectNet\train'

datos_entrenamiento = []
emociones = sorted(os.listdir(DATASET_DIR))
imagenes = []
etiquetas = []

for idx, emocion in enumerate(emociones):
    ruta_clase = os.path.join(DATASET_DIR, emocion) # Une la ruta base con el nombre de la clase
    archivos = [f for f in os.listdir(ruta_clase) if f.endswith(('.jpg', '.png', '.jpeg'))] #Listamos todos los archivos compatibles con las extensiones

    for nombre in archivos:
        ruta_img = os.path.join(ruta_clase, nombre) # Une la ruta de la clase con el nombre del archivo.
        try:
            print("Analizando: ", ruta_img, " etiqueta: ", idx)
            img = Image.open(ruta_img)  # escala de grises
            img = img.resize((96, 96))
            arreglo = np.array(img) # Convierte la imagen en un arreglo NumPy 48x48
            imagenes.append(arreglo) # Concatenamos a el arreglo
            etiquetas.append(idx)  # etiqueta numérica según clase (0-6)
            datos_entrenamiento.append((arreglo, idx))
        except Exception as e:
            print(f"Error al procesar la imagen {ruta_img}: {e}")

# Convierto los arreglos a NumPy
X = np.array(imagenes)               # imágenes
Y = np.array(etiquetas)              # etiquetas numéricas
X = X.reshape(-1, 96, 96, 3)
print("Forma de X:", X.shape)
print("Forma de y:", Y.shape)
print("Emociones:", emociones)
print(len(datos_entrenamiento))

#Normalización da datos
X = np.array(X).astype(float) / 255

############# GENERADOR DE DATOS

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=[0.6, 1.2],
    vertical_flip=True,
    horizontal_flip=True
)

datagen.fit(X)

X, Y = shuffle(X, Y, random_state=42)

X_entrenamiento = X[:20000]
X_validacion = X[20000:]
Y_entrenamiento = Y[:20000]
Y_validacion = Y[20000:]

datagen_entrenamiento = datagen.flow(X_entrenamiento, Y_entrenamiento, batch_size=32)

############# CALLBACKS

reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy',      # o 'val_loss'
    factor=0.5,              # reduce el LR a la mitad
    patience=5,              # espera 5 épocas sin mejora
    min_lr=1e-6,             # no baja más allá de esto
    verbose=1
)

checkpoint = ModelCheckpoint(
    filepath='mejor_modelo.h5',
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

############# TENSORBOARD

#tensorboard = TensorBoard(log_dir='logs/Denso_AffectNet')
#tensorboard = TensorBoard(log_dir='logs/CNN1_AffectNet')
tensorboard = TensorBoard(log_dir='logs/CNN2_AffectNet')

#tensorboard = TensorBoard(log_dir='logs/Denso_AD_AffectNet')
#tensorboard = TensorBoard(log_dir='logs/CNN1_AD_AffectNet')
#tensorboard = TensorBoard(log_dir='logs/CNN2_AD_AffectNet')

############# MODELOS DE ENTRENAMIENTO DE DATOS

# Modelo 1

modeloDenso_AD = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(96, 96, 1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(8, activation='softmax'),
])

# Modelo 2

modeloCNN1_AD = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(96, activation='relu'),
    tf.keras.layers.Dense(8, activation='softmax'),
])

# Modelo 3

modeloCNN2_AD = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 3)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(8, activation='softmax'),
])

############# COMPILACIÓN DE MODELOS

modeloDenso_AD.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

modeloCNN1_AD.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

modeloCNN2_AD.compile(optimizer= Adam(lr=0.001),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

############# ENTRENAMIENTO

"""modeloDenso_AD.fit(X, Y, epochs=100, batch_size=32,
                validation_split=0.15,
                callbacks=[tensorboard, reduce_lr, checkpoint])"""

"""modeloCNN1_AD.fit(X, Y, epochs=100, batch_size=32,
                validation_split=0.15,
                callbacks=[tensorboard, reduce_lr, checkpoint])"""

"""modeloCNN2_AD.fit(X, Y, epochs=100, batch_size=32,
                validation_split=0.15,
                callbacks=[tensorboard, reduce_lr, checkpoint])"""

############# ENTRENAMIENTO CON DATOS AUMENTADOS (AD)

"""modeloDenso_AD.fit(datagen_entrenamiento,
                   epochs=100, batch_size=32,
                   validation_data = (X_validacion, Y_validacion),
                   steps_per_epoch=int(np.ceil(len(X_entrenamiento) / float(32))),
                   validation_steps=int(np.ceil(len(X_validacion) / float(32))),
                   callbacks=[tensorboard, reduce_lr, checkpoint])"""

"""modeloCNN1_AD.fit(datagen_entrenamiento,
                   epochs=100, batch_size=32,
                   validation_data = (X_validacion, Y_validacion),
                   steps_per_epoch=int(np.ceil(len(X_entrenamiento) / float(32))),
                   validation_steps=int(np.ceil(len(X_validacion) / float(32))),
                   callbacks=[tensorboard, reduce_lr, checkpoint])"""

modeloCNN2_AD.fit(datagen_entrenamiento,
                   epochs=10, batch_size=16,
                   validation_data = (X_validacion, Y_validacion),
                   steps_per_epoch=int(np.ceil(len(X_entrenamiento) / float(32))),
                   validation_steps=int(np.ceil(len(X_validacion) / float(32))),
                   callbacks=[tensorboard, reduce_lr, checkpoint])

y_pred = modeloCNN2_AD.predict(X_validacion)
y_pred_classes = np.argmax(y_pred, axis=1)
print(classification_report(Y_validacion, y_pred_classes))

gc.collect()