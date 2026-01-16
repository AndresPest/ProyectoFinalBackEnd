import tensorflow as tf

# Cargar tu modelo
model = tf.keras.models.load_model("app/modeloOptimo.h5")

# Mostrar resumen completo
model.summary()


