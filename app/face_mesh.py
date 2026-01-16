from flask import Blueprint, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from app.utils import preparar_desde_base64

api_emociones = Blueprint('api_emocion', __name__, url_prefix='/api')

modelo_emociones = load_model('app/modeloOptimo.h5')

# Lista de clases en el mismo orden que la salida del modelo
clases = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

@api_emociones.route('/emocion', methods=['POST'])
def detectar_emocion():
    datos = request.get_json()
    imagen_b64 = datos.get('imagen')
    if not imagen_b64:
        return jsonify({'error': 'No se recibió imagen'}), 400

    try:
        entrada = preparar_desde_base64(imagen_b64)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    # Predicciones del modelo (vector de probabilidades)
    pred = modelo_emociones.predict(entrada)[0]

    # Índice de la emoción principal
    idx = int(np.argmax(pred))
    emocion = clases[idx]
    confianza = float(pred[idx])

    # Convertir todas las probabilidades a porcentajes
    porcentajes = {clase: round(float(prob) * 100, 2) for clase, prob in zip(clases, pred)}

    return jsonify({
        'emocion': emocion,
        'confianza': round(confianza * 100, 2),
        'porcentajes': porcentajes
    })
