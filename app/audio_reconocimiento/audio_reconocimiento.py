from flask import request, jsonify, Blueprint
import io
import tensorflow as tf
import numpy as np
import librosa
import joblib

api_audio_reconocimiento = Blueprint('api_audio_reconocimiento', __name__, url_prefix='/api')

MODEL_PATH = r"C:\Users\Andres\PycharmProjects\ProyectoBackend\app\audio_reconocimiento\models\modelo_7362.h5"
scaler = joblib.load(r"C:\Users\Andres\PycharmProjects\ProyectoBackend\app\audio_reconocimiento\models\scaler7362.joblib")
model = tf.keras.models.load_model(MODEL_PATH)

EMOCIONES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


def extract_features(data, sr):
    mfcc_full = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40)

    mfcc = np.mean(mfcc_full.T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=data, sr=sr).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)

    delta_mfcc = np.mean(librosa.feature.delta(mfcc_full).T, axis=0)
    delta2_mfcc = np.mean(librosa.feature.delta(mfcc_full, order=2).T, axis=0)

    arreglo_completo = np.hstack((mfcc, chroma, rms, delta_mfcc, delta2_mfcc))

    return arreglo_completo

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

        audio_data, sr = librosa.load(io.BytesIO(file.read()), sr=22050)
        audio_data = librosa.util.normalize(audio_data)

        duracion_total = librosa.get_duration(y=audio_data, sr=sr)

        intervalos_sonido = librosa.effects.split(audio_data, top_db=30)
        duracion_sonido = sum([(fin - inicio) / sr for inicio, fin in intervalos_sonido])

        if duracion_total > 0:
            porcentaje_silencio = 100 - ((duracion_sonido / duracion_total) * 100)
        else:
            porcentaje_silencio = 0

        chunk_length = 3 * sr
        predicciones_chunks = []

        if len(audio_data) < chunk_length:
            chunks = [audio_data]
        else:
            chunks = [audio_data[i:i + chunk_length] for i in range(0, len(audio_data), chunk_length)]

            if len(chunks[-1]) < sr:
                chunks.pop()

        for chunk in chunks:
            features = extract_features(chunk, sr)
            features_scaled = scaler.transform(features.reshape(1, -1))
            features_final = np.expand_dims(features_scaled, axis=2)

            pred_probs = model.predict(features_final)[0]
            predicciones_chunks.append(pred_probs)

        probabilidades_promedio = np.mean(predicciones_chunks, axis=0)
        idx = np.argmax(probabilidades_promedio)

        EMOCIONES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

        return jsonify({
            "label": EMOCIONES[idx],
            "confidence": float(probabilidades_promedio[idx]),
            "probs": probabilidades_promedio.tolist(),
            "classes": EMOCIONES,
            "analisis_extra": {
                "duracion_segundos": round(duracion_total, 2),
                "porcentaje_silencio": round(porcentaje_silencio, 2),
                "cantidad_pausas": len(intervalos_sonido) - 1,
                "chunks_analizados": len(chunks)
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500