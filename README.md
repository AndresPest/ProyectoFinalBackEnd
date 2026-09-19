API en Python (Flask) para el reconocimiento emocional mediante redes neuronales convolucionales, Google FaceMesh y análisis de voz. Desplegada en HuggingFace y conectada a Firebase Realtime Database.

## Requisitos previos

- Python 3.9 o superior
- Pip actualizado
- Cuenta de Firebase con un proyecto activo y una cuenta de servicio (archivo JSON)
- Git

## Instalación

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/AndresPest/ProyectoFinalBackEnd.git
   cd ProyectoFinalBackEnd
   ```

2. Crear y activar un entorno virtual:
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/macOS
   source venv/bin/activate
   ```

3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

   El archivo `requirements.txt` incluye Flask, Flask-CORS, TensorFlow 2.10, Keras, Librosa, MediaPipe, OpenCV, Firebase Admin y Gunicorn, entre otras.

## Configuración

1. **Credenciales de Firebase**:
   - En la consola de Firebase, ir a **Configuración del proyecto → Cuentas de servicio**.
   - Seleccionar Python y generar una nueva clave privada.
   - Descargar el archivo JSON y colocarlo en la raíz del proyecto o en la ruta indicada por el código.
   - Verificar que la variable de entorno o la ruta al archivo coincida con la configuración.

2. **Variables de entorno** (opcional según el despliegue):
   - `PORT`: puerto de ejecución (por defecto `7860` para HuggingFace).
   - `CUDA_VISIBLE_DEVICES`: establecer en `-1` para forzar el uso de CPU si no hay GPU compatible.

## Ejecución local

```bash
python run.py
```

El servidor se levantará en `http://localhost:7860` y expondrá los blueprints de login, FaceMesh, resultados, Grad-CAM y reconocimiento de audio.

## Despliegue en HuggingFace

El repositorio incluye un `Procfile` con la instrucción:

```
web: gunicorn run:app
```

Sube el repositorio a un Space de HuggingFace y configura las credenciales de Firebase como secretos del espacio.

---
