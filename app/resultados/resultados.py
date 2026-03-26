from flask import Flask, request, jsonify, Blueprint
from datetime import datetime
from zoneinfo import ZoneInfo

api_resultados = Blueprint('api_resultados', __name__, url_prefix='/api')

@api_resultados.route('/resultados', methods=['POST'])
def recibir_cuestionario():
    data = request.get_json()
    userID = data.get('uid')
    identificador = data.get('identificador')
    sumaTotal = data.get('sumaTotal')
    puntaje = data.get('puntajeFinal')
    categorias = data.get('categorias', [])
    nPreguntasCategoria = data.get('nPreguntasCategoria', [])
    tiempo = data.get('tiempo')
    fecha = data.get('fecha')
    nivel_estres = ""
    mediaCategoria = {}

    # Variables para Miller y Smith
    categoriaVulnerableLv1 = []
    categoriaVulnerableLv2 = []
    categoriaVulnerableLv3 = []

    # Variables para CEAU
    categoriasResaltantes = []
    categoriasAtencion = []

    # Variables para SISCO

    if identificador == "Miller":
        if puntaje >= 0 and puntaje <= 29:
            nivel_estres = "Resistente / Normal"
        elif puntaje >= 30 and puntaje <= 49:
            nivel_estres = "Vulnerable al estrés"
        elif puntaje >= 50 and puntaje <= 75:
            nivel_estres = "Seriamente vulnerable al estrés"
        elif puntaje >= 75:
            nivel_estres = "Extremadamente vulnerable al estrés"

        for categoria, puntajeTotal in categorias.items():
            n_preguntas = nPreguntasCategoria.get(categoria)
            media = puntajeTotal / n_preguntas
            mediaCategoria[categoria] = float("{:.1f}".format(media))

        for categoria, valor in mediaCategoria.items():
            if 2.5 <= valor <= 3.4:
                categoriaVulnerableLv1.append(categoria)
            elif 3.5 <= valor <= 4.7:
                categoriaVulnerableLv2.append(categoria)
            elif 4.8 <= valor <= 5.0:
                categoriaVulnerableLv3.append(categoria)

        resultado_final = {
            "identificador": "Test de Vulnerabilidad al Estrés - L.H. Miller y A.D. Smith",
            "puntaje": puntaje,
            "nivel": nivel_estres,
            "categorias": categorias,
            "categoriaVulnerableLv1": categoriaVulnerableLv1,
            "categoriaVulnerableLv2": categoriaVulnerableLv2,
            "categoriaVulnerableLv3": categoriaVulnerableLv3,
            "tiempo": tiempo,
            "fecha": datetime.now(ZoneInfo("America/Caracas")).isoformat()
        }
        return jsonify(resultado_final)

    elif identificador == "CEAU":

        if puntaje >= 21 and puntaje <= 48:
            nivel_estres = "Estrés Bajo"
        elif puntaje >= 49 and puntaje <= 77:
            nivel_estres = "Estrés Moderado"
        elif puntaje >= 78 and puntaje <= 105:
            nivel_estres = "Estrés Alto / Severo"

        for categoria, puntajeTotal in categorias.items():
            n_preguntas = nPreguntasCategoria.get(categoria)
            media = puntajeTotal / n_preguntas
            mediaCategoria[categoria] = float("{:.1f}".format(media))

        for categoria, valor in mediaCategoria.items():
            if 2.5 <= valor <= 3.7:
                categoriasResaltantes.append(categoria)
            elif 3.8 <= valor <= 5.0:
                categoriasAtencion.append(categoria)

        resultado_final = {
            "identificador": "CEAU - Cuestionario de Estrés Académico en la Universidad",
            "puntaje": puntaje,
            "nivel": nivel_estres,
            "categorias": categorias,
            "categoriasResaltantes": categoriasResaltantes,
            "categoriasAtencion": categoriasAtencion,
            "tiempo": tiempo,
            "fecha": datetime.now(ZoneInfo("America/Caracas")).isoformat()
        }
        return jsonify(resultado_final)

    elif identificador == "SISCO":

        nPreguntasCategoria = {
            "Nivel General": 1,
            "Estresores": 7,
            "Síntomas": 7,
            "Afrontamiento": 7
        }

        puntaje_general = categorias.get("Nivel General", 0)

        if puntaje_general >= 2 and puntaje_general <= 3:
            categoriasResaltantes.append("Nivel General")
            nivel_estres = "Estrés Moderado"
        elif puntaje_general >= 4 and puntaje_general <= 5:
            categoriasAtencion.append("Nivel General")
            nivel_estres = "Estrés Alto"
        else:
            nivel_estres = "Estrés Leve"

        for categoria, puntajeTotal in categorias.items():
            n_preguntas = nPreguntasCategoria.get(categoria, 7)
            media = puntajeTotal / n_preguntas
            mediaCategoria[categoria] = float("{:.1f}".format(media))

        for categoria, valor in mediaCategoria.items():
            if categoria == "Nivel General":
                continue

            if categoria in ["Estresores", "Sintomas"]:
                if 2.5 <= valor <= 3.6:
                    categoriasResaltantes.append(categoria)
                elif 3.7 <= valor <= 5.0:
                    categoriasAtencion.append(categoria)
            elif categoria == "Afrontamiento":
                if 1.0 <= valor <= 2.4:
                    categoriasAtencion.append(categoria)
                elif 2.5 <= valor <= 3.4:
                    categoriasResaltantes.append(categoria)

        puntajeTotal = categorias.get("Estresores", 0) + categorias.get("Sintomas", 0) + categorias.get("Afrontamiento", 0)
        puntajeMedia = float("{:.1f}".format(puntajeTotal / 21))

        resultado_final = {
            "identificador": "SISCO - Inventario Sistémico Cognoscitivista para el estudio del estrés académico",
            "puntaje": puntajeMedia,
            "nivel": nivel_estres,
            "categorias": categorias,
            "categoriasResaltantes": categoriasResaltantes,
            "categoriasAtencion": categoriasAtencion,
            "tiempo": tiempo,
            "fecha": datetime.now(ZoneInfo("America/Caracas")).isoformat()
        }
        print(resultado_final)
        return jsonify(resultado_final)

    elif identificador == "BBS":

        if puntaje >= 8 and puntaje <= 14:
            nivel_estres = "Vulnerable al estrés"
        elif puntaje >= 15 and puntaje <= 22:
            nivel_estres = "Altamente vulnerable al estrés"
        else:
            nivel_estres = "Resistente / Normal"

        for categoria, puntajeTotal in categorias.items():
            n_preguntas = nPreguntasCategoria.get(categoria)
            media = puntajeTotal / n_preguntas
            mediaCategoria[categoria] = float("{:.1f}".format(media))

        for categoria, valor in mediaCategoria.items():
            if 0.4 <= valor <= 0.6:
                categoriasResaltantes.append(categoria)
            elif 0.7 <= valor <= 1.0:
                categoriasAtencion.append(categoria)

        resultado_final = {
            "identificador": "Inventario Sobre Vulnerabilidad al Estrés (Beech, Burns y Sheffield, 1982)",
            "puntaje": puntaje,
            "nivel": nivel_estres,
            "categorias": categorias,
            "categoriasResaltantes": categoriasResaltantes,
            "categoriasAtencion": categoriasAtencion,
            "tiempo": tiempo,
            "fecha": datetime.now(ZoneInfo("America/Caracas")).isoformat()
        }
        return jsonify(resultado_final)

    elif identificador == "ISE":

        if puntaje >= 53 and puntaje <= 75:
            nivel_estres = "Estrés Moderado"
        elif puntaje >= 76 and puntaje <= 97:
            nivel_estres = "Estrés Alto"
        elif puntaje >= 98 and puntaje <= 120:
            nivel_estres = "Estrés Muy Alto"
        else:
            nivel_estres = "Estrés Bajo / Leve"

        for categoria, puntajeTotal in categorias.items():
            n_preguntas = nPreguntasCategoria.get(categoria)
            media = puntajeTotal / n_preguntas
            mediaCategoria[categoria] = float("{:.1f}".format(media))

        for categoria, valor in mediaCategoria.items():
            if 2.5 <= valor <= 3.2:
                categoriasResaltantes.append(categoria)
            elif 3.3 <= valor <= 4.0:
                categoriasAtencion.append(categoria)

        resultado_final = {
            "identificador": "Inventario de Síntomas de Estrés. Segunda versión - Arturo Barraza Macías",
            "puntaje": puntaje,
            "nivel": nivel_estres,
            "categorias": categorias,
            "categoriasResaltantes": categoriasResaltantes,
            "categoriasAtencion": categoriasAtencion,
            "tiempo": tiempo,
            "fecha": datetime.now(ZoneInfo("America/Caracas")).isoformat()
        }
        return jsonify(resultado_final)