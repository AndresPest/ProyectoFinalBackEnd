import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.utils import shuffle
from sklearn.utils import class_weight
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from PIL import Image
import gc

gc.collect()

# SET DE IMAGENES
DATASET_DIR= r'C:\Users\Andres\Downloads\Datasets\Mezcla-dataset'

datos_entrenamiento = []
emociones = sorted(os.listdir(DATASET_DIR))
imagenes = []
etiquetas = []
batch_size = 16

for idx, emocion in enumerate(emociones):
    ruta_clase = os.path.join(DATASET_DIR, emocion)
    archivos = [f for f in os.listdir(ruta_clase) if f.endswith(('.jpg', '.png', '.jpeg'))]

    for nombre in archivos:
        ruta_img = os.path.join(ruta_clase, nombre)
        try:
            print("Analizando: ", ruta_img, " etiqueta: ", idx)
            img = Image.open(ruta_img).convert('L')  #escala de grises
            img = img.resize((48, 48))
            arreglo = np.array(img) #convierte la imagen en un arreglo NumPy 48x48
            imagenes.append(arreglo)
            etiquetas.append(idx)  #etiqueta numérica según su emocion (0-6)
            datos_entrenamiento.append((arreglo, idx))
        except Exception as e:
            print(f"Error al procesar la imagen {ruta_img}: {e}")

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
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    vertical_flip=True,
    horizontal_flip=True
)

datagen.fit(X)

X, Y = shuffle(X, Y, random_state=42)

X_entrenamiento = X[:45000]
X_validacion = X[45000:]
Y_entrenamiento = Y[:45000]
Y_validacion = Y[45000:]

datagen_entrenamiento = datagen.flow(X_entrenamiento, Y_entrenamiento, batch_size=batch_size)

#datagen_entrenamiento = tf.data.Dataset.from_tensor_slices((X_entrenamiento, Y_entrenamiento))
#datagen_entrenamiento = datagen_entrenamiento.shuffle(buffer_size=8000)
#datagen_entrenamiento = datagen_entrenamiento.batch(batch_size)
#datagen_entrenamiento = datagen_entrenamiento.repeat()
#datagen_entrenamiento = datagen_entrenamiento.prefetch(tf.data.AUTOTUNE)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=4,
    min_lr=1e-5,
    verbose=1
)

checkpoint = ModelCheckpoint(
    filepath='modelo_checkpoint.h5',
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

################# PRUEBA DE TUNING ###########################

def build_model(hp):
    modelo = tf.keras.models.Sequential()

    filtros_b1 = hp.Choice('filtros_bloque_1', values=[32, 64])
    modelo.add(tf.keras.layers.Conv2D(filtros_b1, (3, 3), activation='elu', input_shape=(48, 48, 1), padding='same'))
    modelo.add(tf.keras.layers.BatchNormalization())
    modelo.add(tf.keras.layers.Conv2D(filtros_b1, (3, 3), activation='elu', padding='same'))
    modelo.add(tf.keras.layers.BatchNormalization())
    modelo.add(tf.keras.layers.MaxPooling2D((2, 2)))
    modelo.add(tf.keras.layers.Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.3, step=0.1)))


    filtros_b2 = hp.Choice('filtros_bloque_2', values=[64, 128])
    modelo.add(tf.keras.layers.Conv2D(filtros_b2, (3, 3), activation='elu', padding='same'))
    modelo.add(tf.keras.layers.BatchNormalization())
    modelo.add(tf.keras.layers.Conv2D(filtros_b2, (3, 3), activation='elu', padding='same'))
    modelo.add(tf.keras.layers.BatchNormalization())
    modelo.add(tf.keras.layers.MaxPooling2D((2, 2)))
    modelo.add(tf.keras.layers.Dropout(hp.Float('dropout_2', min_value=0.1, max_value=0.3, step=0.1)))

    modelo.add(tf.keras.layers.Conv2D(128, (3, 3), activation='elu', padding='same'))
    modelo.add(tf.keras.layers.BatchNormalization())
    modelo.add(tf.keras.layers.Conv2D(128, (3, 3), activation='elu', padding='same'))
    modelo.add(tf.keras.layers.BatchNormalization())
    modelo.add(tf.keras.layers.MaxPooling2D((2, 2)))
    modelo.add(tf.keras.layers.Dropout(hp.Float('dropout_3', min_value=0.2, max_value=0.4, step=0.1)))
    modelo.add(tf.keras.layers.Flatten())

    neuronas_densas = hp.Choice('neuronas_densas', values=[128, 256, 512])
    modelo.add(tf.keras.layers.Dense(neuronas_densas, activation='elu'))
    modelo.add(tf.keras.layers.BatchNormalization())
    modelo.add(tf.keras.layers.Dropout(hp.Float('dropout_denso', min_value=0.2, max_value=0.5, step=0.1)))

    modelo.add(tf.keras.layers.Dense(7, activation='softmax'))

    lr_dinamico = hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4])

    modelo.compile(
        optimizer=Adam(learning_rate=lr_dinamico),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return modelo

tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=20,
    executions_per_trial=1,
    directory='busqueda_hiperparametros',
    project_name='FER_Mejorado2'
)

tuner.search_space_summary()

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True)


print("\n--- BUSQUEDA DE HIPERPARAMETROS ---")
tuner.search(
    datagen_entrenamiento,
    epochs=15,
    validation_data=(X_validacion, Y_validacion),
    steps_per_epoch=len(X_entrenamiento) // batch_size,
    callbacks=[early_stop, reduce_lr]
)

print("\n--- BUSQUEDA FINALIZADA ---")
tuner.results_summary()

mejores_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
La mejor configuración encontrada fue:
- Filtros Bloque 1: {mejores_hp.get('filtros_bloque_1')}
- Filtros Bloque 2: {mejores_hp.get('filtros_bloque_2')}
- Neuronas Densas: {mejores_hp.get('neuronas_densas')}
- Tasa de Aprendizaje: {mejores_hp.get('learning_rate')}
""")

modelo_final = tuner.hypermodel.build(mejores_hp)
modelo_final.summary()
tensorboard_final = TensorBoard(log_dir='logs/CNN_TunerFinal')

checkpoint_final = ModelCheckpoint(
    filepath='modelo_final_ganador.h5',
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

pesos = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(Y_entrenamiento),
    y=Y_entrenamiento
)

pesos_clases = dict(enumerate(pesos))

print("\n--- PESOS POR EMOCION ---")
for clase, peso in pesos_clases.items():
    print(f"Clase {clase}: Peso x{peso:.2f}")

print("\n--- ENTRENAMIENTO DE LA MEJOR COMBINACION ---")

history = modelo_final.fit(
    datagen_entrenamiento,
    epochs=50,
    validation_data=(X_validacion, Y_validacion),
    steps_per_epoch=len(X_entrenamiento) // batch_size,
    callbacks=[tensorboard_final, reduce_lr, checkpoint_final]
)

y_pred = modelo_final.predict(X_validacion)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\n--- REPORTES ---")
print(classification_report(Y_validacion, y_pred_classes, target_names=emociones))

### Matriz de Confusion
cm = confusion_matrix(Y_validacion, y_pred_classes)

plt.figure(figsize=(10, 8))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=emociones, yticklabels=emociones)
plt.title('Matriz de Confusión de Emociones')
plt.ylabel('Etiqueta real de la imagen')
plt.xlabel('Predicción del Modelo')
plt.tight_layout()
plt.show()

### Curvas ROC Y AUC por emocion
Y_val_bin = label_binarize(Y_validacion, classes=[0, 1, 2, 3, 4, 5, 6])
n_classes = Y_val_bin.shape[1]

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(Y_val_bin[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(12, 8))
colores = ['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta']

for i, color in zip(range(n_classes), colores):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC {emociones[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.title('Curvas ROC por Emocion')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()

auc_macro = roc_auc_score(Y_val_bin, y_pred, multi_class='ovr', average='macro')
print(f"\n---> AUC General del Modelo (Macro Average): {auc_macro:.4f} <---")

gc.collect()

##############################################################

############# MODELOS DE ENTRENAMIENTO DE DATOS
# Modelo 3

"""modeloCNN2_AD = tf.keras.models.Sequential([
    # Bloque 1
    tf.keras.layers.Conv2D(32, (3, 3), activation='elu', input_shape=(48, 48, 1), padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, (3, 3), activation='elu', padding='same'), # CAMBIO: Doble Conv2D inicial
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.1), # CAMBIO: Dropout suave temprano

    # Bloque 2
    tf.keras.layers.Conv2D(64, (3, 3), activation='elu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='elu', padding='same'), # CAMBIO: Doble Conv2D
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.15),

    # Bloque 3
    tf.keras.layers.Conv2D(128, (3, 3), activation='elu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3), activation='elu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv2D(256, (3, 3), activation='elu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(256, activation='elu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.25),
    
    tf.keras.layers.Dense(128, activation='elu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(7, activation='softmax'),
])"""

############# MODELO DE 63.52%

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
    tf.keras.layers.Dense(7, activation='softmax'),
])"""

############# COMPILACIÓN DE MODELOS

"""modeloCNN2_AD.compile(optimizer= Adam(lr=0.0005),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

############# TENSORBOARD

#tensorboard = TensorBoard(log_dir='logs/Denso_AD_DatosMezclados')
#tensorboard = TensorBoard(log_dir='logs/CNN1_AD_DatosMezclados')
tensorboard = TensorBoard(log_dir='logs/CNN_TuningDinamico')


############# ENTRENAMIENTO CON DATOS AUMENTADOS (AD)
modeloCNN2_AD.fit(datagen_entrenamiento,
                   epochs=40, batch_size=batch_size,
                   validation_data = (X_validacion, Y_validacion),
                   steps_per_epoch=int(np.ceil(len(X_entrenamiento) // batch_size)),
                   validation_steps=int(np.ceil(len(X_validacion) // batch_size)),
                   callbacks=[tensorboard, reduce_lr, checkpoint])

y_pred = modeloCNN2_AD.predict(X_validacion)
y_pred_classes = np.argmax(y_pred, axis=1)
print(classification_report(Y_validacion, y_pred_classes))

gc.collect()"""