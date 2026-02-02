import os
import shutil

# 1. Configuración de rutas
# Carpeta donde están todos los archivos .wav de CREMA-D
path_crema_origen = r'C:\Users\Andres\Downloads\Datasets\CREMA-D\AudioWAV'
# Carpeta donde se crearán las subcarpetas por emoción
path_crema_destino = r'C:\Users\Andres\Downloads\Datasets\CREMA_Organizado'

# 2. Diccionario de mapeo (Sigla -> Nombre completo)
crema_emociones = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fearful",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad"
}


def organizar_crema():
    if not os.path.exists(path_crema_destino):
        os.makedirs(path_crema_destino)

    contador = 0

    # Listar los archivos de la carpeta
    for filename in os.listdir(path_crema_origen):
        if filename.endswith(".wav"):
            # Separar por guion bajo: 1001_DFA_SAD_XX.wav -> ['1001', 'DFA', 'SAD', 'XX.wav']
            partes = filename.split('_')

            if len(partes) >= 3:
                sigla_emocion = partes[2]  # Tomamos 'SAD'
                nombre_emocion = crema_emociones.get(sigla_emocion, "desconocido")

                # Crear la subcarpeta si no existe
                ruta_carpeta_final = os.path.join(path_crema_destino, nombre_emocion)
                if not os.path.exists(ruta_carpeta_final):
                    os.makedirs(ruta_carpeta_final)

                # Definir origen y destino
                origen = os.path.join(path_crema_origen, filename)
                destino = os.path.join(ruta_carpeta_final, filename)

                # Copiar archivo
                shutil.copy2(origen, destino)
                contador += 1

    print(f"¡Listo! Se han organizado {contador} archivos en {path_crema_destino}")


if __name__ == "__main__":
    organizar_crema()