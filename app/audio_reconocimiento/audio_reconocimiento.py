from flask import Flask, request, jsonify, Blueprint
from fastapi import FastAPI, UploadFile, File
import soundfile as sf
import io
from .audio_reconocimiento_model import load_model, get_classes
from .audio_reconocimiento_utils import extract_mel
import tensorflow as tf
import numpy as np
import librosa
import joblib
import os

api_audio_reconocimiento = Blueprint('api_audio_reconocimiento', __name__, url_prefix='/api')

MODEL_PATH = r"C:\Users\Andres\PycharmProjects\ProyectoBackend\app\audio_reconocimiento\models\modelo_6443.h5"
scaler = joblib.load(r"C:\Users\Andres\PycharmProjects\ProyectoBackend\app\audio_reconocimiento\models\scaler6443.joblib")
model = tf.keras.models.load_model(MODEL_PATH)

EMOCIONES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


def extract_features(audio_data, sr):
    """
    Sincronizado con el entrenamiento:
    MFCC (40) + Chroma (12) + RMS (1) = 53 total
    """
    # 1. MFCCs (40) - DEBEN IR PRIMERO
    mfcc = np.mean(librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40).T, axis=0)

    # 2. Chroma (12) - DEBE IR DESPUÉS
    chroma = np.mean(librosa.feature.chroma_stft(y=audio_data, sr=sr).T, axis=0)

    # 3. RMS (1) - DEBE IR AL FINAL (En entrenamiento usaste RMS, no ZCR)
    rms = np.mean(librosa.feature.rms(y=audio_data).T, axis=0)

    # Combinar en el orden exacto: (40, 12, 1)
    return np.hstack((mfcc, chroma, rms))

def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    mfcc = np.expand_dims(mfcc, axis=0)
    mfcc = np.expand_dims(mfcc, axis=-1)
    return mfcc

@api_audio_reconocimiento.route('/audio', methods=['POST'])
def predict():
    try:
        print("Archivo recibido...")
        if 'file' not in request.files:
            return jsonify({"error": "No se subió ningún archivo"}), 400

        file = request.files['file']

        # 1. Cargar el audio
        audio_data, sr = librosa.load(io.BytesIO(file.read()), sr=22050)
        audio_data = librosa.util.normalize(audio_data)

        # 2. Extraer características (Asegúrate de que esta función devuelva un array de 53)
        features = extract_features(audio_data, sr)

        # 3. Escalar los datos (Pasa de 1D a 2D para el scaler)
        features_scaled = scaler.transform(features.reshape(1, -1))

        # 4. Crear 'features_final' para la CNN (Añadir dimensión de canal)
        # Aquí es donde se resuelve el error 'Unresolved reference'
        features_final = np.expand_dims(features_scaled, axis=2)

        # 5. Realizar la predicción
        prediction = model.predict(features_final)
        idx = np.argmax(prediction)

        # Lista de emociones en el orden exacto de tu entrenamiento
        EMOCIONES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

        # 6. Devolver el JSON completo para que Angular dibuje las barras
        return jsonify({
            "label": EMOCIONES[idx],
            "confidence": float(prediction[0][idx]),
            "probs": prediction[0].tolist(),  # Importante: convertir a lista
            "classes": EMOCIONES
        })

    except Exception as e:
        import traceback
        traceback.print_exc()  # Esto te dirá en consola si falta algo más
        return jsonify({"error": str(e)}), 500