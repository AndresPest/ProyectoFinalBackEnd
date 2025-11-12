import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from io import BytesIO
import base64

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, tf.argmax(predictions[0])]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap.numpy()

def superimpose_heatmap(imagen_b64, heatmap, alpha=0.4):
    imagen_bytes = base64.b64decode(imagen_b64)
    imagen_pil = Image.open(BytesIO(imagen_bytes)).convert('RGB')
    imagen_np = np.array(imagen_pil.resize((48, 48)))
    heatmap = cv2.resize(heatmap, (48, 48))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(imagen_np, 1 - alpha, heatmap_color, alpha, 0)
    return superimposed_img
