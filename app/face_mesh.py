from flask import Blueprint, request, jsonify
import tensorflow as tf
import os
import numpy as np
from app.utils import preparar_desde_base64

api_emociones = Blueprint('api_emocion', __name__, url_prefix='/api')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Carga segura del modelo para evitar errores de optimizador
modelo_emociones = tf.keras.models.load_model('app/modeloOptimo.h5', compile=False)
modelo_emociones.compile(optimizer='adam', loss='categorical_crossentropy')

clases = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


def predecir_core(imagen_b64):
    entrada = preparar_desde_base64(imagen_b64)
    pred = modelo_emociones.predict(entrada)[0]
    idx = int(np.argmax(pred))
    return {
        'emocion': clases[idx],
        'confianza': float(pred[idx]) * 100
    }


@api_emociones.route('/emocion-cnn', methods=['POST'])
def detectar_cnn():
    img = request.get_json().get('imagen')
    if not img: return jsonify({'error': 'No image'}), 400
    res = predecir_core(img)
    return jsonify({**res, 'metodo': 'CNN'})


@api_emociones.route('/emocion-facemesh', methods=['POST'])
def detectar_facemesh():
    datos = request.get_json()
    img = datos.get('imagen')
    # Si no vienen puntos, enviamos lista vacía para que len(puntos) no falle
    puntos = datos.get('puntos', [])

    if not img: return jsonify({'error': 'Missing image.'}), 400

    res = predecir_core(img)
    return jsonify({**res, 'metodo': 'FaceMesh', 'puntos_count': len(puntos)})