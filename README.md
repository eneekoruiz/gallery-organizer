# Gallery Organizer · Visor y Agrupador Facial Local

Aplicación web local de alto rendimiento desarrollada en FastAPI y OpenCV para la indexación, deduplicación, análisis de metadatos EXIF y agrupamiento facial offline de colecciones fotográficas y de vídeo personales.

- **Documentación de arquitectura:** [DeepWiki - gallery-organizer](https://deepwiki.com/eneekoruiz/gallery-organizer)

## Propósito y Valor Técnico

Solución concebida para fotógrafos y usuarios con bibliotecas multimedia masivas que desean organizar, etiquetar y buscar en sus recuerdos familiares con privacidad total, sin enviar sus fotografías ni biomarcadores a nubes de terceros.

- **Indexación Local Ultrarrápida:** Escanea árboles de directorios locales extrayendo metadatos EXIF/XMP (fechas reales de captura, modelos de cámara/lente, aperturas y geolocalización GPS), almacenando el índice en SQLite para búsquedas instantáneas.
- **Detección y Agrupamiento Facial Offline:** Implementa redes neuronales convolucionales con OpenCV: detector YuNet de 1024px para localización facial y extractor SFace para embeddings vectoriales de 128 dimensiones con clustering de centroides múltiples, agrupando automáticamente a las mismas personas a lo largo de los años.
- **Compatibilidad con Formatos de Apple (HEIC/HEIF):** Soporte de lectura nativo para imágenes HEIC generadas por dispositivos iOS mediante `pillow-heif`, generando miniaturas y previsualizaciones WebP al vuelo sin corromper los archivos originales.
- **Detección de Tomas Duplicadas y Ráfagas:** Algoritmos de comparación temporal y visual que detectan disparos en ráfaga y fotos duplicadas para optimizar el almacenamiento.

## Stack Técnico

- **Backend:** Python 3.10+, FastAPI, Uvicorn, Flask (módulos auxiliares)
- **Visión Artificial:** OpenCV (`opencv-python-headless`), YuNet Face Detector, SFace Embeddings
- **Procesamiento de Imagen:** Pillow, `pillow-heif`, NumPy
- **Validación & Datos:** Pydantic, SQLite

## Puesta en Marcha Local

### Prerrequisitos

- Python 3.10 o superior
- Soporte de compilación para librerías C/C++ (incluido en binarios wheels)

### Instalación y Ejecución

```bash
git clone https://github.com/eneekoruiz/gallery-organizer.git
cd gallery-organizer

# Crear y activar entorno virtual
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Iniciar servidor FastAPI
uvicorn app_fastapi:app --host 127.0.0.1 --port 8000 --reload
```
