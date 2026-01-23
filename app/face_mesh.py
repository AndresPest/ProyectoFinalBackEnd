from flask import Blueprint, request, jsonify
import tensorflow as tf
import os
import numpy as np
from app.utils import preparar_desde_base64

api_emociones = Blueprint('api_emocion', __name__, url_prefix='/api')


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Bloquea logs pesados

# Forzar a usar solo 1 hilo de CPU y nada de memoria extra
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.set_visible_devices([], 'GPU')

# Carga el modelo con una opción para ahorrar RAM
modelo_emociones = tf.keras.models.load_model('app/modeloOptimo.h5', compile=False)


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
