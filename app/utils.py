import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image


def preparar_desde_base64(imagen_b64):
    try:
        # Decodificar
        imagen_bytes = base64.b64decode(imagen_b64)
        imagen_pil = Image.open(BytesIO(imagen_bytes)).convert('L')  # Convertimos a GRIS directamente aquí
        imagen_np = np.array(imagen_pil)

        # En lugar de detectar rostros (que consume mucha RAM), redimensionamos directamente.
        # Si la persona está frente a la cámara, el modelo de 5MB funcionará igual.
        rostro_redim = cv2.resize(imagen_np, (48, 48))

        # Normalizar
        rostro_redim = rostro_redim / 255.0
        rostro_redim = np.expand_dims(rostro_redim, axis=-1)
        rostro_redim = np.expand_dims(rostro_redim, axis=0)

        return rostro_redim
    except Exception as e:
        raise ValueError(f"Error procesando imagen: {str(e)}")