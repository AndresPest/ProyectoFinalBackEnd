from flask import Blueprint, request, jsonify
import base64, cv2, numpy as np
from app.preprocessing import load_and_preprocess_image_grayscale_b64
from app.gradcam_run import gradcam, overlay_heatmap, activacion_por_capa
from app.model_loader import model

# Haarcascade para recorte de rostro
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

api_gradcam = Blueprint('api_gradcam', __name__, url_prefix='/api')

@api_gradcam.route('/gradcam', methods=['POST'])
def generate_gradcam():
    data = request.json
    img_b64 = data.get("image")

    # Decodificar imagen
    img_bytes = base64.b64decode(img_b64)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Detectar rostro
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return jsonify({"error": "No se detectó ningún rostro"}), 400

    x, y, w, h = faces[0]
    face_img = img[y:y+h, x:x+w]

    # Preprocesar rostro
    _, buffer = cv2.imencode('.jpg', face_img)
    face_b64 = base64.b64encode(buffer).decode('utf-8')
    img_tensor = load_and_preprocess_image_grayscale_b64(face_b64)

    # Generar Grad-CAM sobre la última capa convolucional
    heatmap = gradcam(model, img_tensor, layer_name="conv2d_7")
    overlay = overlay_heatmap(face_img, heatmap)

    _, buffer = cv2.imencode('.jpg', overlay)
    result_b64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({"heatmap": result_b64})

@api_gradcam.route('/activaciones-color', methods=['POST'])
def activaciones_color_endpoint():
    data = request.json
    img_b64 = data.get("image")

    # Decodificar imagen
    img_bytes = base64.b64decode(img_b64)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Detectar rostro con Haarcascade
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return jsonify({"error": "No se detectó ningún rostro"}), 400

    x, y, w, h = faces[0]
    face_img = img[y:y+h, x:x+w]

    # Preprocesar rostro recortado
    _, buffer = cv2.imencode('.jpg', face_img)
    face_b64 = base64.b64encode(buffer).decode('utf-8')
    img_tensor = load_and_preprocess_image_grayscale_b64(face_b64)

    # Obtener activaciones por capa como heatmaps color
    activaciones = activacion_por_capa(model, img_tensor)

    return jsonify({"activaciones": activaciones})
