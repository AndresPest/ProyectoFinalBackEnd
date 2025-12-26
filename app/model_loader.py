import tensorflow as tf

# Cargar tu modelo
model = tf.keras.models.load_model("app/modeloOptimo.h5")

# Mostrar resumen completo
model.summary()

# O listar solo los nombres de las capas
for i, layer in enumerate(model.layers):
    print(i, layer.name, layer.output_shape)
