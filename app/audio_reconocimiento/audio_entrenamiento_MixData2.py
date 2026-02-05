import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization, GlobalAveragePooling1D
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers.legacy import Adam
import warnings
warnings.filterwarnings('ignore')

# --- 1. FUNCIONES DE AUMENTO DE DATOS ---
def add_noise(data):
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    return data + noise_amp * np.random.normal(size=data.shape[0])


def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=rate)


def shift(data):
    shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
    return np.roll(data, shift_range)


def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)


# --- 2. EXTRACCIÓN DE CARACTERÍSTICAS (MFCC) ---
def extract_features(data, sr):
    # MFCC (40)
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40).T, axis=0)
    # Chroma (12) - Captura armonía y notas
    chroma = np.mean(librosa.feature.chroma_stft(y=data, sr=sr).T, axis=0)
    # RMS (1) - Captura energía/volumen
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)

    # Combinamos todos: 40 + 12 + 1 = 53 características
    return np.hstack((mfcc, chroma, rms))

def get_features(path):
    data, sr = librosa.load(path, duration=3, offset=0.5)
    data = librosa.util.normalize(data)

    # 1. Original
    res1 = extract_features(data, sr)
    result = np.array(res1)

    # 2. Con ruido
    res2 = extract_features(add_noise(data), sr)
    result = np.vstack((result, res2))

    # 3. Con cambio de tono y velocidad
    res3 = extract_features(pitch(stretch(data), sr), sr)
    result = np.vstack((result, res3))

    return result


# --- 3. CARGA Y PROCESAMIENTO DEL DATASET ---
DATASET_PATH = r'C:\Users\Andres\Downloads\Datasets\Audio_Dataset'
X, Y = [], []

print("Iniciando procesamiento con Augmentation... Esto duplicará/triplicará el dataset.")

for emotion in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, emotion)
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            path = os.path.join(folder_path, filename)

            # Obtener 3 versiones de características por cada audio
            features = get_features(path)

            for f in features:
                X.append(f)
                Y.append(emotion.lower())

# Convertir a arrays de numpy
X = np.array(X)

# --- 4. ESCALAMIENTO (NUEVO Y VITAL) ---
scaler = StandardScaler()
X = scaler.fit_transform(X)
# Guardamos el scaler para usarlo en Angular/Backend después
joblib.dump(scaler, 'models/scaler.joblib')

X = np.expand_dims(X, -1)  # Forma (Total_Muestras, 53, 1)

# Codificación de etiquetas
enc = OneHotEncoder()
y = enc.fit_transform(np.array(Y).reshape(-1, 1)).toarray()

# Split de datos
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(53, 1)),
    BatchNormalization(),
    MaxPooling1D(pool_size=5, strides=2, padding='same'),

    Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'),
    Dropout(0.3),
    MaxPooling1D(pool_size=5, strides=2, padding='same'),

    Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu'),
    GlobalAveragePooling1D(),

    Dense(units=64, activation='relu'), # Aumentamos un poco la capacidad
    Dropout(0.5),
    Dense(units=y.shape[1], activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# --- CALLBACKS MEJORADOS ---
checkpoint = ModelCheckpoint(
    filepath='models/best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

history = model.fit(
    x_train, y_train,
    batch_size=64,
    epochs=70, # Aumentamos épocas ya que el dataset es más complejo
    validation_data=(x_test, y_test),
    callbacks=[checkpoint, reduce_lr],
    shuffle=True
)

y_pred = model.predict(x_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_actual_labels = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_actual_labels, y_pred_labels)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=enc.categories_[0],
            yticklabels=enc.categories_[0],
            cmap='Blues')
plt.xlabel('Predicción')
plt.ylabel('Realidad')
plt.title('Matriz de Confusión - Dataset Unificado')
plt.show()