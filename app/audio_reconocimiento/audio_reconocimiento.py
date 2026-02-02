from flask import Flask, request, jsonify, Blueprint
import soundfile as sf
import io
from .audio_reconocimiento_model import load_model, get_classes
from .audio_reconocimiento_utils import extract_mel
import tensorflow as tf
import numpy as np
import librosa
import os

SAMPLE_RATE = 16000

api_audio_reconocimiento = Blueprint('api_audio_reconocimiento', __name__, url_prefix='/api')

MODEL_PATH = r"C:\Users\Andres\PycharmProjects\ProyectoBackend\app\audio_reconocimiento\models\audio_9964.h5"
model = tf.keras.models.load_model(MODEL_PATH)
LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']

def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    mfcc = np.expand_dims(mfcc, axis=0)
    mfcc = np.expand_dims(mfcc, axis=-1)
    return mfcc

@api_audio_reconocimiento.route('/audio', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No se envió ningún archivo"}), 400

    file = request.files['file']
    file_path = "temp_audio.wav"
    file.save(file_path)

    try:
        # 1. Preprocesar
        features = extract_mfcc(file_path)

        # 2. Predecir
        prediction = model.predict(features)
        predicted_label_index = np.argmax(prediction)
        result = LABELS[predicted_label_index]
        confidence = float(np.max(prediction))

        # 3. Limpiar archivo temporal
        os.remove(file_path)

        return jsonify({
            "emotion": result,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500