import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

DATASET_PATH = r"C:\Users\Andres\Downloads\Datasets\CREMA-D\AudioWAV"
SAMPLE_RATE = 16000

# Mapeo de emociones
EMOTION_MAP = {
    "FEA": "fear",
    "SAD": "sad",
    "ANG": "angry",
    "DIS": "disgust",
    "HAP": "happy",
    "NEU": "neutral"
}

CLASSES = list(EMOTION_MAP.values())

def extract_mel(y, sr=SAMPLE_RATE, n_mels=96, n_fft=1024, hop_length=256, time_frames=96):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                       hop_length=hop_length, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    if S_db.shape[1] < time_frames:
        pad = time_frames - S_db.shape[1]
        S_db = np.pad(S_db, ((0,0),(0,pad)), mode='constant')
    else:
        S_db = S_db[:, :time_frames]
    X = (S_db - S_db.mean()) / (S_db.std() + 1e-6)
    return X[..., np.newaxis]

X, y = [], []

# Recorrer todos los archivos
for file in os.listdir(DATASET_PATH):
    if file.endswith(".wav"):
        parts = file.split("_")
        emotion_code = parts[2]  # tercer bloque
        emotion_label = EMOTION_MAP.get(emotion_code)

        if emotion_label is None:
            continue  # ignorar si no está en el mapa

        # Mostrar en consola qué archivo se está procesando
        print(f"Analizando {file} → {emotion_label}")

        # Cargar audio
        audio_path = os.path.join(DATASET_PATH, file)
        y_audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

        mel = extract_mel(y_audio)
        X.append(mel)
        y.append(CLASSES.index(emotion_label))

X = np.array(X)
y = tf.keras.utils.to_categorical(y, num_classes=len(CLASSES))

print("Dataset cargado:", X.shape, y.shape)

# Dividir en train/val
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

# Definir modelo CNN
inputs = tf.keras.Input(shape=(96,96,1))
x = tf.keras.layers.Conv2D(32, (3,3), activation='elu')(inputs)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling2D((2,2))(x)

x = tf.keras.layers.Conv2D(64, (3,3), activation='elu', padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling2D((2,2))(x)

x = tf.keras.layers.Conv2D(128, (3,3), activation='elu', padding='same')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling2D((2,2))(x)

x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation='elu')(x)
x = tf.keras.layers.Dropout(0.4)(x)
outputs = tf.keras.layers.Dense(len(CLASSES), activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


modeloCNN_AUDIO = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 1), kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.GlobalAveragePooling2D(),

    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(len(CLASSES), activation='softmax'),
])

modeloCNN_AUDIO.compile(optimizer= Adam(lr=0.001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

############# CALLBACKS

reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy',      # o 'val_loss'
    factor=0.5,              # reduce el LR a la mitad
    patience=5,              # espera 5 épocas sin mejora
    min_lr=1e-6,             # no baja más allá de esto
    verbose=1
)

checkpoint = ModelCheckpoint(
    filepath='models/checkpoint.h5',
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

# Entrenamiento
callbacks = [reduce_lr, checkpoint]

history = modeloCNN_AUDIO.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=50,
                    batch_size=32,
                    callbacks=callbacks)

# Guardar modelo
model.save("models/audio_emotion_cnn.h5")