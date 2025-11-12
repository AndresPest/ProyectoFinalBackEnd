# app/preprocessing.py
import tensorflow as tf
from PIL import Image
from io import BytesIO
import base64

def load_and_preprocess_image_grayscale_b64(imagen_b64):
    imagen_bytes = base64.b64decode(imagen_b64)
    imagen_pil = Image.open(BytesIO(imagen_bytes)).convert('L')
    imagen_pil = imagen_pil.resize((48, 48))
    img_array = tf.keras.preprocessing.image.img_to_array(imagen_pil)
    img_array = tf.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


def preprocess_dataset(x, y, batch_size=32, shuffle=True):
    """Convierte arrays en tf.data.Dataset optimizado."""
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    return dataset