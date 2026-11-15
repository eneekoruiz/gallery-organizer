# 🔮 Smart Gallery AI — Studio Edition

Organizador y visor inteligente de fotografías y vídeos con inteligencia artificial local, reconocimiento facial biométrico SOTA (State-of-the-Art), clustering anti-chaining con restricciones físicas Cannot-Link, eliminación de duplicados exactos y visor interactivo de nivel estudio inspirado en Apple Photos e Immich.

---

## 🚀 Cómo Iniciar la Aplicación

### Método 1: Doble Clic (Recomendado en Windows)
- Haz doble clic sobre **`Iniciar_SmartGallery.bat`** (o **`Iniciar_SmartGallery_Silencioso.vbs`**).
- La aplicación se iniciará y abrirá automáticamente tu navegador en `http://127.0.0.1:5000`.

### Método 2: Desde la Terminal
```bash
# Iniciar servidor Flask con hot-reload
python app.py

# O ejecutar mediante Gunicorn (Modo Producción / Docker)
gunicorn --bind 0.0.0.0:5000 --workers 1 --threads 4 "app:create_app()"
```

---

## 🧠 Arquitectura Biométrica Forense (SOTA)

### 1. Evaluación de Calidad Facial (FIQA) y Pose 3D
- **Filtro de Calidad y Resolución (`app/services/biometric_engine.py`)**: Evalúa la nitidez mediante la varianza del operador Laplaciano combinada con la distancia interocular mínima (IOD $\ge 20$ px). Rostros microscópicos de fondo o desenfocados son rechazados automáticamente.
- **Estimación de Pose 3D (`solvePnP`)**: Mapea los 5 puntos de referencia contra un modelo antropométrico 3D canónico, obteniendo los ángulos de Tait-Bryan (Yaw, Pitch, Roll). Descarta rotaciones extremas ($|\text{yaw}| > 45^\circ$, $|\text{pitch}| > 35^\circ$) previniendo perfiles distorsionados.

### 2. Test-Time Augmentation (TTA)
- Extracción dual promediando la muestra canónica con su reflexión especular horizontal (flip) y proyectando el vector sobre la hiperesfera unitaria ($S^{D-1}$). Reduce drásticamente el ruido por iluminación asimétrica.

### 3. Clustering Forense Anti-Chaining
- **Restricción Física Cannot-Link (`app/utils/clustering.py`)**: Dos rostros de la misma fotografía o fotograma tienen distancia de repulsión infinita obligatoria. Es físicamente imposible que un clúster agrupe dos caras provenientes de la misma foto.
- **Partición Espectral de Bimodalidad**: Calcula el valor de Fiedler ($\lambda_2$) sobre el Laplaciano simétrico normalizado de cada grupo; si se detecta bimodalidad interna, aplica un corte espectral bipartito (*Normalized Cut*) para bifurcar el clúster.

### 4. Protocolo Tabula Rasa (`scripts/tabula_rasa_pipeline.py`)
Permite auditar, purgar metadatos/cachés obsoletas y reindexar toda la colección sin tocar jamás las fotos originales del usuario:
```bash
# 1. Auditoría del catálogo actual
python scripts/tabula_rasa_pipeline.py --action audit

# 2. Purga y reindexación por lotes desatendida
python scripts/tabula_rasa_pipeline.py --action all --batch-size 50
```

---

## 📸 Formatos Soportados

| Tipo | Extensiones |
| :--- | :--- |
| **Fotografía** | `.jpg`, `.jpeg`, `.png`, `.webp`, `.heic`, `.heif`, `.bmp`, `.tiff` |
| **Vídeo y Live Photos** | `.mp4`, `.mov`, `.avi`, `.mkv`, `.m4v` |

---

## ⌨️ Atajos de Teclado del Visor (Lightbox)

| Tecla | Acción |
| :--- | :--- |
| `→` / `←` | Foto siguiente / Foto anterior |
| `Espacio` | Reproducir / Pausar vídeo |
| `+` / `-` / `0` | Acercar / Alejar / Reiniciar zoom |
| `Ctrl + K` | Búsqueda global Spotlight |
| `Supr` / `Del` | Eliminar archivo actual |
| `Esc` | Cerrar modal activo o visor de fotos |

---

## 🧪 Batería de Pruebas Automatizadas (TDD)

El repositorio cuenta con una suite completa de pruebas unitarias y de estrés en `tests/`:

```bash
# Ejecutar toda la batería de pruebas
python tests/run_all_tests.py
```
- ✅ **100% Tests Pasados en Verde**: 89 pruebas unitarias automatizadas.
- ✅ **Pruebas Biométricas SOTA**: FIQA, TTA, Pose 3D, Fiedler spectral cut, Cannot-Link.
- ✅ **Pruebas de Casos Borde y Estrés**: Archivos nulos/corruptos, micro-caras, formatos especiales (RGBA, WebP, Grayscale) y resiliencia de la cola de tareas en segundo plano.
- ✅ **Benchmark Sub-Segundo**: Búsqueda vectorial facial en 6.24 ms para 20.000 items.
