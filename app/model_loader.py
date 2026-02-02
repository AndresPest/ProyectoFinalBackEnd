import tensorflow as tf


import os

# Configuración para que TensorFlow use el mínimo de memoria posible
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Silencia logs pesados
tf.config.set_visible_devices([], 'GPU') # Forzar solo CPU para ahorrar drivers
# Cargar tu modelo
model = tf.keras.models.load_model("app/modeloOptimo.h5")

# Mostrar resumen completo
model.summary()


