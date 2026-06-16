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

### Extraemos caracteristicas de audio
def extract_features(data, sr):
    mfcc_full = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)

    mfcc = np.mean(mfcc_full.T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=data, sr=sr).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)

    delta_mfcc = np.mean(librosa.feature.delta(mfcc_full).T, axis=0)
    delta2_mfcc = np.mean(librosa.feature.delta(mfcc_full, order=2).T, axis=0)

    arreglo_completo = np.hstack((mfcc, chroma, rms, delta_mfcc, delta2_mfcc))

    return arreglo_completo

def get_features(path):
    data, sr = librosa.load(path, duration=3, offset=0.5)
    data = librosa.util.normalize(data)

    res1 = extract_features(data, sr)
    result = np.array(res1)

    res2 = extract_features(add_noise(data), sr)
    result = np.vstack((result, res2))

    res3 = extract_features(pitch(stretch(data), sr), sr)
    result = np.vstack((result, res3))

    return result

### Carga del dataset
DATASET_PATH = r'C:\Users\Andres\Downloads\Datasets\Audio_Dataset'
X, Y = [], []

print("Iniciando procesamiento con Augmentation... Esto duplicará/triplicará el dataset.")

for emotion in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, emotion)
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            path = os.path.join(folder_path, filename)

            features = get_features(path)

            for f in features:
                X.append(f)
                Y.append(emotion.lower())

X = np.array(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)
joblib.dump(scaler, 'models/scaler.joblib')

enc = OneHotEncoder()
y = enc.fit_transform(np.array(Y).reshape(-1, 1)).toarray()

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(512, activation='relu', input_shape=(133,)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(units=y.shape[1], activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

### Callbacks para el entrenamiento
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
    epochs=70,
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