import tensorflow as tf

MODEL_PATH = "models/audio_9964.h5"
CLASSES = ["angry","disgust","fear","happy","neutral","sad","surprise"]

model = None

def load_model():
    global model
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH)
    return model

def get_classes():
    return CLASSES