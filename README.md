# Gallery Organizer

Aplicación web local para indexar, organizar y explorar colecciones de fotografías y vídeos personales.

## Características

- Extracción automática de metadatos EXIF (fecha, cámara, dimensiones) con `exifread` y `pillow`.
- Soporte para formatos HEIC/HEIF mediante `pillow-heif`.
- Detección de imágenes duplicadas mediante hashes visuales y análisis de contenido.
- Agrupación visual y detección de rostros con OpenCV y Scikit-Learn.
- Papelera segura local integrada con `send2trash`.
- Interfaz web servida mediante FastAPI / Jinja2 con soporte de tareas en segundo plano (APScheduler).

## Stack técnico

- **Backend:** Python 3, FastAPI, Uvicorn, SQLAlchemy.
- **Procesamiento de imagen:** OpenCV, Pillow, Scikit-Learn, NumPy.
- **Frontend:** HTML5, CSS y plantillas Jinja2.

## Instalación y ejecución

1. Crear y activar entorno virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: .\venv\Scripts\activate
   ```

2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Iniciar la aplicación:
   ```bash
   uvicorn app_fastapi:app --reload --port 8000
   ```
   O ejecutar en Windows el script lanzador `Iniciar_SmartGallery.bat`.
