import os
import shutil

# 1. Configuración de rutas
# Cambia esta ruta a donde tengas la carpeta raíz de RAVDESS
path_ravdess_raiz = r'C:\Users\Andres\Downloads\Datasets\RAVDESS Emotional speech audio'
# Ruta donde quieres crear las carpetas por emoción
path_destino_emociones = r'C:\Users\Andres\Downloads\Datasets\RAVDESS_Organizado'

# 2. Diccionario basado en tu descripción
ravdess_emociones = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


def organizar_ravdess():
    # Crear la carpeta principal si no existe
    if not os.path.exists(path_destino_emociones):
        os.makedirs(path_destino_emociones)

    contador = 0

    # Caminar por todas las subcarpetas (Actor_01, Actor_02, etc.)
    for root, dirs, files in os.walk(path_ravdess_raiz):
        for filename in files:
            if filename.endswith(".wav"):
                # Separar el nombre por el guion '-'
                # Ejemplo: 03-01-06-01-02-01-12.wav -> ['03', '01', '06', ...]
                partes = filename.split('-')

                if len(partes) >= 3:
                    codigo_emocion = partes[2]  # El tercer elemento
                    nombre_emocion = ravdess_emociones.get(codigo_emocion, "desconocido")

                    # Crear la subcarpeta de la emoción (ej: .../RAVDESS_Organizado/fearful)
                    carpeta_final = os.path.join(path_destino_emociones, nombre_emocion)
                    if not os.path.exists(carpeta_final):
                        os.makedirs(carpeta_final)

                    # Ruta origen y ruta destino
                    origen = os.path.join(root, filename)
                    destino = os.path.join(carpeta_final, filename)

                    # Copiar el archivo
                    shutil.copy2(origen, destino)
                    contador += 1

    print(f"Proceso finalizado. Se han organizado {contador} archivos en {path_destino_emociones}")

if __name__ == "__main__":
    organizar_ravdess()