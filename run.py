import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from flask import Flask
from flask_cors import CORS

# Importación de tus Blueprints
from app.login.login import api_login
from app.api_facemesh import api_emociones
from app.gradcam_api import api_gradcam
from app.resultados.resultados import api_resultados
from app.audio_reconocimiento.audio_reconocimiento import api_audio_reconocimiento

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})

### Registro de Blueprints
app.register_blueprint(api_emociones)
app.register_blueprint(api_login)
app.register_blueprint(api_gradcam)
app.register_blueprint(api_resultados)
app.register_blueprint(api_audio_reconocimiento)

@app.route('/')
def index():
    return {"status": "Servidor de Tesis Activo", "version": "2.0 - Multi-Modelo"}

if __name__ == '__main__':
    # Puerto para Hugging Face
    port = int(os.environ.get("PORT", 7860))
    app.run(host='0.0.0.0', port=port, debug=False)