El backend está desarrollado en Python con Flask y expone los endpoints que ejecutan los modelos de reconocimiento facial, FaceMesh y voz. Se encuentra desplegado en HuggingFace y utiliza Firebase para el almacenamiento de resultados.

Requisitos previos
Python 3.9 o superior.

Pip actualizado.

Cuenta de Firebase con un proyecto activo y una cuenta de servicio generada (archivo JSON).

Git para clonar el repositorio.

Instalación
Clonar el repositorio:

bash
git clone https://github.com/AndresPest/ProyectoFinalBackEnd.git
cd ProyectoFinalBackEnd
Crear y activar un entorno virtual:

bash
python -m venv venv
# En Windows:
venv\Scripts\activate
# En Linux/macOS:
source venv/bin/activate
Instalar las dependencias:

bash
pip install -r requirements.txt
El archivo requirements.txt incluye Flask, Flask-CORS, TensorFlow 2.10, Keras, Librosa, MediaPipe, OpenCV, Firebase Admin y Gunicorn, entre otras librerías necesarias para el procesamiento de imágenes y audio.

Configurar las credenciales de Firebase:

En la consola de Firebase, ir a Configuración del proyecto → Cuentas de servicio.

Seleccionar Python y generar una nueva clave privada.

Descargar el archivo JSON y colocarlo en la raíz del proyecto o en la ruta que el código espere.

Verificar que la variable de entorno o la ruta al archivo coincida con la configuración del proyecto.

Configurar variables de entorno (opcional, según el despliegue):

PORT: puerto de ejecución (por defecto 7860 para HuggingFace).

CUDA_VISIBLE_DEVICES: se establece en -1 para forzar el uso de CPU si no se cuenta con GPU compatible.

Ejecución local
Para iniciar el servidor en desarrollo:

bash
python run.py
El servidor se levantará en http://localhost:7860 y expondrá los blueprints de login, FaceMesh, resultados, Grad-CAM y reconocimiento de audio.

Despliegue en HuggingFace
El repositorio incluye un archivo Procfile con la instrucción web: gunicorn run:app, lo que permite desplegarlo directamente en HuggingFace Spaces. Solo se debe subir el repositorio al espacio correspondiente y configurar las credenciales de Firebase como secretos del espacio.
