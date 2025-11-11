from flask import Flask, request, jsonify, Blueprint
from flask_cors import CORS

api_login = Blueprint('api_login', __name__, url_prefix='/api')

@api_login.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    usuario = data.get('usuario')
    password = data.get('password')

    if usuario == 'admin' and password == '1234':
        return jsonify({"status": "ok", "token": "abc123"})
    else:
        return jsonify({"status": "error", "mensaje": "Credenciales inválidas"}), 401