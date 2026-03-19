from flask import Flask
from flask_cors import CORS
import os

# Importación de tus Blueprints
from app.login.login import api_login
from app.face_mesh import api_emociones # Este contiene /emocion-cnn y /emocion-facemesh
from app.estres_cuestionario.estres_cuestionario import api_estrescuestionario
from app.gradcam_api import api_gradcam
#from app.audio_reconocimiento.audio_reconocimiento import api_audio_reconocimiento

app = Flask(__name__)

# Configuración de CORS
# Permitimos todos los orígenes (*) para evitar bloqueos durante el desarrollo y despliegue
CORS(app, resources={r"/*": {"origins": "*"}})

# Registro de Blueprints
# Al registrar api_emociones, se habilitan automáticamente las rutas:
# 1. /api/emocion-cnn
# 2. /api/emocion-facemesh
app.register_blueprint(api_emociones)
app.register_blueprint(api_login)
app.register_blueprint(api_estrescuestionario)
app.register_blueprint(api_gradcam)
#app.register_blueprint(api_audio_reconocimiento)

@app.route('/')
def index():
    return {"status": "Servidor de Tesis Activo", "version": "2.0 - Multi-Modelo"}

if __name__ == '__main__':
    # Puerto dinámico para Render o Hugging Face (por defecto 7860)
    port = int(os.environ.get("PORT", 7860))
    app.run(host='0.0.0.0', port=port, debug=False)