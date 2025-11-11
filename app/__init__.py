from flask import Flask
from flask_cors import CORS
"""from app.face_mesh import face_mesh_api"""
from app.face_mesh import api_emociones

def crear_app():
    app = Flask(__name__)
    CORS(app)
    """app.register_blueprint(face_mesh_api, url_prefix='/api')"""
    app.register_blueprint(api_emociones)
    return app