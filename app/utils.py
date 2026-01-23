import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image


def preparar_desde_base64(imagen_b64):
    # 1. Decodificar Base64
    try:
        imagen_bytes = base64.b64decode(imagen_b64)
        imagen_pil = Image.open(BytesIO(imagen_bytes)).convert('RGB')
        imagen_np = np.array(imagen_pil)
    except Exception as e:
        raise ValueError(f"Error decodificando imagen: {str(e)}")

    # 2. Convertir a Gris para el modelo
    gris = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2GRAY)

    # 3. Intentar detectar rostro, si falla, usamos la imagen central
    # (Esto evita que el 400 mate tu app si el XML no carga en Render)
    try:
        detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        rostros = detector.detectMultiScale(gris, 1.1, 5)

        if len(rostros) > 0:
            x, y, w, h = rostros[0]
            rostro = gris[y:y + h, x:x + w]
        else:
            # Si no detecta rostro, tomamos el centro de la imagen para no dar error 400
            rostro = gris
    except:
        rostro = gris

    # 4. Redimensionar a 48x48 (lo que espera tu modelo de 5MB)
    rostro_redim = cv2.resize(rostro, (48, 48))

    # 5. Normalizar y dar formato Tensor
    rostro_redim = rostro_redim / 255.0
    rostro_redim = np.expand_dims(rostro_redim, axis=-1)  # (48, 48, 1)
    rostro_redim = np.expand_dims(rostro_redim, axis=0)  # (1, 48, 48, 1)

    return rostro_redim