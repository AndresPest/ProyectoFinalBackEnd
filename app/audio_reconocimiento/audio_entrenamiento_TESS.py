import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pygame
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers.legacy import Adam
import warnings
warnings.filterwarnings('ignore')

paths = []
labels = []

for dirname, _, filenames in os.walk(r'C:\Users\Andres\Downloads\Datasets\Toronto emotional speech set (TESS)\TESS Toronto emotional speech set data'):
    for filename in filenames:
        paths.append(os.path.join(dirname, filename))
        label = filename.split('_')[-1]
        label = label.split('.')[0]
        labels.append(label.lower())
    if len(paths) == 2800:
        break

print('Dataset is Loaded')
len(paths)

## Create a dataframe
df = pd.DataFrame()
df['speech'] = paths
df['label'] = labels
df.head()


def waveplot(data, sr, emotion):
    plt.figure(figsize=(10, 4))
    plt.title(emotion, size=20)
    librosa.display.waveshow(data, sr=sr)
    plt.show()


def spectogram(data, sr, emotion):
    x = librosa.stft(data)
    xdb = librosa.amplitude_to_db(abs(x))
    plt.figure(figsize=(11, 4))
    plt.title(emotion, size=20)
    librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()
    plt.show()

def reproducir_audio(path):
    pygame.mixer.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        continue

def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

extract_mfcc(df['speech'][0])
X_mfcc = df['speech'].apply(lambda x: extract_mfcc(x))

X = [x for x in X_mfcc]
X = np.array(X)

X = np.expand_dims(X, -1)

enc = OneHotEncoder()
y = enc.fit_transform(df[['label']])
y = y.toarray()

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Forma de x_train: {x_train.shape}")

model = Sequential([
    # Primera capa convolucional
    Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(40, 1)),
    BatchNormalization(),
    MaxPooling1D(pool_size=5, strides=2, padding='same'),

    # Segunda capa convolucional
    Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'),
    Dropout(0.3),  # Para evitar el sobreajuste (overfitting)
    MaxPooling1D(pool_size=5, strides=2, padding='same'),

    # Tercera capa convolucional
    Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu'),
    Flatten(),  # Convertimos a una sola dimensión para las capas densas

    # Capas densas (clasificación)
    Dense(units=32, activation='relu'),
    Dropout(0.3),

    # Capa de salida (el número de neuronas debe ser igual al número de emociones)
    Dense(units=y.shape[1], activation='softmax')
])

model.compile(optimizer= Adam(lr=0.001),
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
callbacks = [checkpoint]

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=50,
                    validation_data=(x_test, y_test),
                    shuffle=True,
                    callbacks=callbacks)
model.save("models/audio_emotion_cnn.h5")

### Matriz de confusion
y_pred = model.predict(x_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_actual_labels = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_actual_labels, y_pred_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=enc.categories_[0], yticklabels=enc.categories_[0])
plt.xlabel('Predicción')
plt.ylabel('Realidad')
plt.show()