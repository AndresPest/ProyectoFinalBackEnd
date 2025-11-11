from flask import Flask, request, jsonify, Blueprint
from flask_cors import CORS

api_estrescuestionario = Blueprint('api_estrescuestionario', __name__, url_prefix='/api')

@api_estrescuestionario.route('/estrescuestionario', methods=['POST'])
def recibir_puntaje():
    data = request.get_json()
    puntaje = data.get('puntaje')
    nivel_estres = ""
    if(puntaje <= 13):
        nivel_estres = "Bajo nivel de estres percibido"
        return jsonify({"status": "ok", "nivel_estres": nivel_estres, "nivel": puntaje})
    elif(puntaje >= 14 and puntaje <= 26):
        nivel_estres = "Estres moderado"
        return jsonify({"status": "ok", "nivel_estres": nivel_estres, "nivel": puntaje})
    elif(puntaje >= 27 and puntaje <= 40):
        nivel_estres = "Alto nivel de estres percibido"
        return jsonify({"status": "ok", "nivel_estres": nivel_estres, "nivel": puntaje})
    else:
        print("Puntaje recibido:", puntaje)
        nivel_estres = "ERROR"
        return jsonify({"No se pudo calcular el nivel de estres": nivel_estres, "nivel": puntaje})

@api_estrescuestionario.route('/stress', methods=['POST'])
def recibir_respuestas():
    data = request.get_json()
    respuestas = data.get('respuestas', [])
    print("Respuestas recibidas:", respuestas)
    # Aquí podrías procesarlas con tu modelo ML
    return jsonify({"status": "ok", "mensaje": "Respuestas recibidas correctamente"})