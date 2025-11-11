import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Ruta del modelo y carpeta de prueba
RUTA_MODELO = 'modelo_100E_emociones_CNN2_AD.h5'
DIRECTORIO = 'logs'
IMG_SIZE = (48, 48)  # Ajusta según tu modelo
COLOR_MODE = 'grayscale'  # Usa 'rgb' si tu modelo acepta 3 canales

# Cargar modelo
modelo = load_model(RUTA_MODELO)

# Obtener clases
clases = list(modelo.clases) if hasattr(modelo, 'clases') else ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Función de carga y preprocesamiento
def procesamiento_imagen(img_ruta):
    img = image.load_img(img_ruta, target_size=IMG_SIZE, color_mode=COLOR_MODE)
    imagen_arreglo = image.img_to_array(img)
    imagen_arreglo = tf.expand_dims(imagen_arreglo, axis=0) / 255.0
    return imagen_arreglo

imagenes = [
    r'C:\Users\Andres\Downloads\Datasets\FER2013\test\neutral\PrivateTest_1844176.jpg',
    r'C:\Users\Andres\Downloads\Datasets\AffectNet\train\anger\image0000182.jpg',
    r'C:\Users\Andres\Downloads\Datasets\AffectNet\train\fear\image0000808.jpg',
    r'C:\Users\Andres\Downloads\Datasets\AffectNet\train\neutral\ffhq_722.png',
    r'C:\Users\Andres\Downloads\Datasets\AffectNet\train\neutral\ffhq_1118.png'
]

# Iterar sobre imágenes en carpeta
for imagen in imagenes:
        img_tensor = procesamiento_imagen(imagen)
        prediccion = modelo.predict(img_tensor)
        prediccion_clase = clases[np.argmax(prediccion)]
        confidence = np.max(prediccion)

        print(f"{imagen}: {prediccion_clase} ({confidence:.2f})")