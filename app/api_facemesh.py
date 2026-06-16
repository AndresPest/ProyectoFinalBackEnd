from flask import Blueprint, request, jsonify
import tensorflow as tf
import os
import numpy as np
import cv2
import base64
from app.utils import preparar_desde_base64
from app.facemesh.emotion_processor.main import EmotionRecognitionSystem

api_emociones = Blueprint('api_emocion', __name__, url_prefix='/api')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Carga segura del modelo para evitar errores de optimizador
modelo_emociones = tf.keras.models.load_model('app/modeloOptimo.h5', compile=False)
modelo_emociones.compile(optimizer='adam', loss='categorical_crossentropy')

clases = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

sistema_facemesh = EmotionRecognitionSystem()

def predecir_core(imagen_b64):
    entrada = preparar_desde_base64(imagen_b64)
    pred = modelo_emociones.predict(entrada)[0]
    idx = int(np.argmax(pred))
    return {
        'emocion': clases[idx],
        'confianza': float(pred[idx]) * 100
    }

def decodificar_imagen_original(img_b64):
    try:
        # Decodifica los bytes de base64
        img_bytes = base64.b64decode(img_b64)
        # Convierte los bytes a un array uint8
        np_array = np.frombuffer(img_bytes, dtype=np.uint8)
        # Lee la imagen a color (mantiene tamaño original 1280x720 del canvas)
        imagen_cv2 = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        return imagen_cv2
    except Exception as e:
        print("Error decodificando imagen pura:", str(e))
        return None

@api_emociones.route('/emocion-cnn', methods=['POST'])
def detectar_cnn():
    img = request.get_json().get('imagen')
    if not img: return jsonify({'error': 'No image'}), 400
    res = predecir_core(img)
    return jsonify({**res, 'metodo': 'CNN'})


@api_emociones.route('/emocion-facemesh', methods=['POST'])
def detectar_facemesh():
    datos = request.get_json()
    img_b64 = datos.get('imagen')
    print("entramos")

    if not img_b64:
        return jsonify({'error': 'Missing image.'}), 400

    try:
        if "," in img_b64:
            img_b64 = img_b64.split(",")[1]

        # === USAMOS EL NUEVO DECODIFICADOR A COLOR Y TAMAÑO REAL ===
        imagen_cv2 = decodificar_imagen_original(img_b64)

        if imagen_cv2 is None:
            return jsonify({'error': 'La imagen no pudo ser decodificada por OpenCV.'}), 400

        # Verificamos que las dimensiones ahora sean correctas (ej. 720, 1280, 3)
        print("NUEVAS Dimensiones de la imagen recibida:", imagen_cv2.shape)

        # Forzar tipo de dato estándar uint8
        imagen_cv2 = imagen_cv2.astype('uint8')

        emociones_dict = sistema_facemesh.frame_processing(imagen_cv2)

        # === ENCONTRAR LA EMOCIÓN DOMINANTE ===
        # emociones_dict es ej: {"angry":50, "disgust":25, "fear":65, "sad":70...}
        if isinstance(emociones_dict, dict) and emociones_dict:
            # max(dict, key=dict.get) encuentra la llave con el valor más alto (ej: "sad")
            emocion_dominante = max(emociones_dict, key=emociones_dict.get)
        else:
            emocion_dominante = "unknown"

        return jsonify({
            'metodo': 'FaceMesh',
            'emocion': emocion_dominante,  # Mandará solo el string, ej: "sad"
            'porcentajes': emociones_dict,  # Mantenemos el desglose por si acaso
            'status': 'Procesado exitosamente'
        })

    except Exception as e:
        import traceback
        print("\n" + "=" * 40)
        print("¡ERROR CRÍTICO DETECTADO EN EL PROCESAMIENTO!")
        print(traceback.format_exc())
        print("=" * 40 + "\n")
        return jsonify({'error': f'Error en procesamiento FaceMesh: {str(e)}'}), 500