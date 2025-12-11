import tensorflow as tf
import numpy as np
import cv2
import base64

def gradcam(model, img_array, layer_name, class_index=None):
    """
    Genera el mapa de calor Grad-CAM para un modelo Keras/TensorFlow.
    """
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if class_index is None:
            class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8
    return heatmap   # numpy.ndarray

def overlay_heatmap(img, heatmap, alpha=0.4):
    """
    Superpone el heatmap sobre la imagen original.
    """
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return overlay

def activacion_por_capa(model, img_array, capas_interes=None):
    """
    Devuelve una sola imagen por capa convolucional, combinando todos los filtros en un heatmap color.
    """
    if capas_interes is None:
        capas_interes = [layer.name for layer in model.layers if 'conv2d' in layer.name]

    outputs = [model.get_layer(name).output for name in capas_interes]
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=outputs)

    activations = activation_model.predict(img_array)

    resultados = {}
    for layer_name, activation in zip(capas_interes, activations):
        # Combinar todos los filtros en una sola imagen
        mapa = np.mean(activation[0], axis=-1)

        # Normalizar
        mapa = np.maximum(mapa, 0)
        mapa /= np.max(mapa) + 1e-8

        # Convertir a heatmap color
        mapa = cv2.resize(mapa, (96, 96))  # tamaño fijo
        mapa = np.uint8(255 * mapa)
        mapa_color = cv2.applyColorMap(mapa, cv2.COLORMAP_JET)

        # Codificar en base64
        _, buffer = cv2.imencode('.jpg', mapa_color)
        resultados[layer_name] = base64.b64encode(buffer).decode('utf-8')

    return resultados
