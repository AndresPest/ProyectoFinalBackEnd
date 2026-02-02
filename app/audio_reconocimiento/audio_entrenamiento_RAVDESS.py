import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

DATASET_PATH = r"C:\Users\Andres\Downloads\Datasets\RAVDESS Emotional speech audio"
SAMPLE_RATE = 16000

# Mapeo de emociones
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise"
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

# Recorrer carpetas de actores
for actor_folder in os.listdir(DATASET_PATH):
    actor_path = os.path.join(DATASET_PATH, actor_folder)
    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):
        if file.endswith(".wav") or file.endswith(".mp4"):
            # Extraer emoción del nombre → tercer número
            parts = file.split("-")
            emotion_code = parts[2]
            emotion_label = EMOTION_MAP[emotion_code]

            print(f"Analizando {file} → {emotion_label}")

            # Cargar audio
            audio_path = os.path.join(actor_path, file)
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

# Entrenamiento
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]

history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=50,
                    batch_size=32,
                    callbacks=callbacks)

# Guardar modelo
model.save("models/audio_emotion_cnn.h5")
