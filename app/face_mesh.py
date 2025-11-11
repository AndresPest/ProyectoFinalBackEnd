from flask import Blueprint, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from app.utils import preparar_desde_base64

api_emociones = Blueprint('api_emocion', __name__, url_prefix='/api')
modelo_emociones = load_model('app/modeloOptimo.h5')
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

    pred = modelo_emociones.predict(entrada)[0]
    idx = np.argmax(pred)
    emocion = clases[idx]
    confianza = float(pred[idx])

    return jsonify({'emocion': emocion, 'confianza': confianza})