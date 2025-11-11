import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
import os
import numpy as np
from PIL import Image
import gc

gc.collect()

# SET DE IMAGENES
DATASET_DIR= 'dataset_emociones'

datos_entrenamiento = []
emociones = sorted(os.listdir(DATASET_DIR))
imagenes = []
etiquetas = []
batch_size = 32
n_clases = 7
epochs = 50

for idx, emocion in enumerate(emociones):
    ruta_clase = os.path.join(DATASET_DIR, emocion) # Une la ruta base con el nombre de la clase
    archivos = [f for f in os.listdir(ruta_clase) if f.endswith(('.jpg', '.png', '.jpeg'))] #Listamos todos los archivos compatibles con las extensiones

    for nombre in archivos:
        ruta_img = os.path.join(ruta_clase, nombre) # Une la ruta de la clase con el nombre del archivo.
        img = Image.open(ruta_img).convert('L')  # escala de grises
        img = img.resize((48, 48))
        arreglo = np.array(img) # Convierte la imagen en un arreglo NumPy 48x48
        imagenes.append(arreglo) # Concatenamos a el arreglo
        etiquetas.append(idx)  # etiqueta numérica según clase (0-6)
        datos_entrenamiento.append((arreglo, idx))

# Convierto los arreglos a NumPy
X = np.array(imagenes)               # imágenes
Y = np.array(etiquetas)              # etiquetas numéricas
X = X.reshape(-1, 48, 48, 1)
print("Forma de X:", X.shape)
print("Forma de y:", Y.shape)
print("Emociones:", emociones)
print(len(datos_entrenamiento))

#Normalización da datos
X = np.array(X).astype(float) / 255

############# GENERADOR DE DATOS

datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=[0.7, 1.4],
    vertical_flip=True,
    horizontal_flip=True
)

datagen.fit(X)
X, Y = shuffle(X, Y, random_state=42)

X_entrenamiento = X[:24000]
X_validacion = X[24000:]

Y_entrenamiento = Y[:24000]
Y_validacion = Y[24000:]

datagen_entrenamiento = datagen.flow(X_entrenamiento, Y_entrenamiento, batch_size=batch_size)

datagen_entrenamiento = tf.data.Dataset.from_tensor_slices((X_entrenamiento, Y_entrenamiento))
datagen_entrenamiento = datagen_entrenamiento.shuffle(buffer_size=1000)
datagen_entrenamiento = datagen_entrenamiento.batch(batch_size)
datagen_entrenamiento = datagen_entrenamiento.repeat()
datagen_entrenamiento = datagen_entrenamiento.prefetch(tf.data.AUTOTUNE)

############# CALLBACKS

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',      # o 'val_accuracy'
    factor=0.2,              # reduce el LR a la mitad
    patience=3,              # espera 3 épocas sin mejora
    min_delta=0.0001,
    verbose=1
)

checkpoint = ModelCheckpoint(
    filepath='modeloOptimo.h5',
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

############# TENSORBOARD

#tensorboard = TensorBoard(log_dir='logs/Denso')
#tensorboard = TensorBoard(log_dir='logs/CNN1')
#tensorboard = TensorBoard(log_dir='logs/CNN2')

#tensorboard = TensorBoard(log_dir='logs/Denso_AD')
#tensorboard = TensorBoard(log_dir='logs/CNN1_AD')
#tensorboard = TensorBoard(log_dir='logs/CNN2_AD2')

tensorboard = TensorBoard(log_dir='logs/modeloPRO2')

############# MODELOS DE ENTRENAMIENTO DE DATOS

# Modelo 1

modeloDenso_AD = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(48, 48, 1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(7, activation='softmax'),
])

# Modelo 2

modeloCNN1_AD = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(48, activation='relu'),
    tf.keras.layers.Dense(7, activation='softmax'),
])

# Modelo 3

modeloCNN2_AD = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
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
    tf.keras.layers.Dense(n_clases, activation='softmax'),
])

############# MODELO DE 58.93%

"""modeloCNN2_AD = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(n_clases, activation='softmax'),
])"""

############# COMPILACIÓN DE MODELOS

modeloDenso_AD.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

modeloCNN1_AD.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

modeloCNN2_AD.compile(optimizer=Adam(lr=0.001),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

############# ENTRENAMIENTO

"""modeloDenso.fit(X, y, epochs=100, batch_size=32,
                validation_split=0.15,
                callbacks=[tensorboard])

modeloCNN1.fit(X, y, epochs=100, batch_size=32,
                validation_split=0.15,
                callbacks=[tensorboard])

modeloCNN2.fit(X, y, epochs=100, batch_size=32,
                validation_split=0.15,
                callbacks=[tensorboard])"""

############# ENTRENAMIENTO CON DATOS AUMENTADOS (AD)

"""modeloDenso_AD.fit(datagen_entrenamiento,
                   epochs=100, batch_size=32,
                   validation_data = (X_validacion, Y_validacion),
                   steps_per_epoch=int(np.ceil(len(X_entrenamiento) / float(32))),
                   validation_steps=int(np.ceil(len(X_validacion) / float(32))),
                   callbacks=[tensorboard])"""

"""modeloCNN1_AD.fit(datagen_entrenamiento,
                   epochs=100, batch_size=32,
                   validation_data = (X_validacion, Y_validacion),
                   steps_per_epoch=int(np.ceil(len(X_entrenamiento) / float(32))),
                   validation_steps=int(np.ceil(len(X_validacion) / float(32))),
                   callbacks=[tensorboard])"""

modeloCNN2_AD.fit(datagen_entrenamiento,
                   epochs=epochs, batch_size=batch_size,
                   validation_data = (X_validacion, Y_validacion),
                   steps_per_epoch=int(np.ceil(len(X_entrenamiento) / float(32))),
                   validation_steps=int(np.ceil(len(X_validacion) / float(32))),
                   callbacks=[tensorboard, checkpoint, reduce_lr])

os.makedirs('app', exist_ok=True)
modeloCNN2_AD.save('modelo_PRO.h5')

y_pred = modeloCNN2_AD.predict(X_validacion)
y_pred_classes = np.argmax(y_pred, axis=1)
print(classification_report(Y_validacion, y_pred_classes))

gc.collect()