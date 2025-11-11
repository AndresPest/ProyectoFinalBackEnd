import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image

def preparar_desde_base64(imagen_b64):
    # Decodificar imagen base64
    imagen_bytes = base64.b64decode(imagen_b64)
    imagen_pil = Image.open(BytesIO(imagen_bytes)).convert('RGB')
    imagen_np = np.array(imagen_pil)

    # Convertir a escala de grises para detección
    gris = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2GRAY)

    # Cargar clasificador de rostros
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    rostros = detector.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5)

    if len(rostros) == 0:
        raise ValueError("No se detectó ningún rostro")

    # Tomar el primer rostro detectado
    x, y, w, h = rostros[0]
    rostro = imagen_np[y:y+h, x:x+w]

    # Redimensionar al tamaño esperado por el modelo (por ejemplo, 48x48)
    rostro_redim = cv2.resize(rostro, (48, 48))
    rostro_redim = cv2.cvtColor(rostro_redim, cv2.COLOR_RGB2GRAY)  # si el modelo espera escala de grises
    rostro_redim = rostro_redim / 255.0  # normalizar
    rostro_redim = np.expand_dims(rostro_redim, axis=-1)  # añadir canal si es necesario
    rostro_redim = np.expand_dims(rostro_redim, axis=0)   # añadir batch

    return rostro_redim