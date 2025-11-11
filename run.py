from flask import Flask
from flask_cors import CORS
from app.login.login import api_login
from app.face_mesh import api_emociones
from app.estres_cuestionario.estres_cuestionario import api_estrescuestionario

from flask import Flask
from flask_cors import CORS
from app.face_mesh import api_emociones

app = Flask(__name__)
CORS(app)
app.register_blueprint(api_emociones)
app.register_blueprint(api_login)
app.register_blueprint(api_estrescuestionario)

if __name__ == '__main__':
    app.run(debug=True)