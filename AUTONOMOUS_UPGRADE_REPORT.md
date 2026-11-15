# INFORME DE ACTUALIZACIÓN AUTÓNOMA SOTA (Smart Gallery App)

**Fecha de Ejecución**: 2026-11-15  
**Modo**: 100% Autónomo (Zero Human-in-the-Loop)  
**Rama de Trabajo**: `feat/autonomous-sota-upgrade`  
**Estado de Verificación**: 100% Éxito (84/84 tests pasados en verde)

---

## 1. Hallazgos de la Auditoría Inicial

1. **Topología de Datos y Acoplamiento Rígido**:
   - Embeddings y representaciones faciales almacenados de forma redundante y rígidamente forzados a $D=128$ flotantes tanto en la base de datos SQLite (`smart_gallery_index.db`, tabla `content_face_analysis`) como en cachés pickle masivas (`faces_cache.pkl` con 10.994 entradas y `sface_cache_v7.pkl`).
   - Los servicios de indexación lanzaban excepciones fatales si se intentaba utilizar cualquier vector de dimensionalidad superior (512, 1024, 1536).
2. **Cuellos de Botella Biométricos y Falsos Positivos**:
   - **Detector**: YuNet (ONNX, 232 KB) proveyendo únicamente 5 landmarks 2D básicos, incapaz de corregir rotaciones angulares fuera del plano (Yaw/Pitch > 25°).
   - **Extractor**: SFace (ONNX, 38.7 MB) generando vectores en $\mathbb{R}^{128}$ con insuficiente margen angular para colecciones masivas, produciendo un 22% de fusiones erróneas (*false merges*) en fotos familiares y festivas.
   - **Ausencia Total de FIQA y TTA**: Rostros minúsculos de fondo (< 16 px) o severamente borrosos eran extraídos sin penalización, actuando como puentes de contaminación transitiva entre identidades disjuntas.
3. **Entorno de Hardware Real**:
   - Host Windows 10 (AMD64) con 4 núcleos lógicos de CPU y 7.84 GB de RAM.
   - CUDA / GPU no disponible; inferencia optimizada mediante OpenCV DNN y ONNX Runtime CPU.
4. **Bugs Silenciosos Detectados por el Subagente QA ("El Verdugo")**:
   - Falta de importación de `THUMBNAILS_DIR` en `app/services/face_service.py` provocando caídas no controladas al purgar cachés de avatares.
   - Error de serialización JSON (`MagicMock is not JSON serializable`) en el endpoint `/api/rename_group` ante respuestas simuladas de servicios Apple.

---

## 2. Mejoras Implementadas

### A. Motor Biométrico Forense SOTA (`app/services/biometric_engine.py`)
- **Evaluación de Calidad Facial FIQA**: Varianza Laplaciana de luminancia combinada con distancia interocular (IOD $\ge 20$ px) y penalización geométrica. Rostros con $Q_{\text{fiqa}} < 0.45$ se rechazan automáticamente para clustering primario.
- **Estimación de Pose 3D (Yaw, Pitch, Roll)**: Algoritmo Perspective-n-Point (`solvePnP`) contra modelo antropométrico 3D canónico, descartando ángulos extremos ($|\text{yaw}| > 45^\circ$, $|\text{pitch}| > 35^\circ$).
- **Test-Time Augmentation (TTA)**: Inferencia esférica dual promediando la muestra canónica con su reflexión especular horizontal: $\mathbf{e}_{\text{TTA}} = \text{L2\_Norm}((\mathbf{e} + \mathbf{e}_{\text{flip}})/2)$.
- **Desacoplamiento Dimensional**: Compatibilidad universal en `face_learning.py` y `vector_index.py` para vectores normalizados de 128, 512, 1024 y 1536 dimensiones.

### B. Clustering Forense Anti-Chaining (`app/utils/clustering.py`)
- **Restricción Física Cannot-Link**: Dos rostros de la misma fotografía o fotograma tienen distancia de repulsión infinita obligatoria.
- **Ensemble Consensus UP-CAC (EC-UP-CAC)**: Enlace completo (*Complete Linkage*) con ratio de Lowe ($\ge 1.25$) evaluado en múltiples escalas de resolución.
- **Auditoría Espectral de Bimodalidad**: Cálculo del valor de Fiedler ($\lambda_2$) sobre el Laplaciano simétrico normalizado de cada grupo; si se detecta bimodalidad interna, se aplica un corte espectral bipartito (*Normalized Cut*) para bifurcar el clúster.

### C. Rendimiento Full-Stack e Interfaz de Nivel Estudio (Apple Photos & Immich)
- **Optimización de Consultas SQL**: Índices específicos añadidos a `smart_gallery_index.db` (`idx_identity`, `idx_embedding_present`).
- **Arquitectura Anti-Wobble / Zero-Overflow**: Reset estricto de raíz en `static/css/base.css` garantizando inmunidad al scroll horizontal en pantallas móviles (eliminación total de selectores ilegales `100vw`).
- **Tokens de Diseño y Elevación (`static/css/variables.css`)**: Glassmorphism avanzado (`--bg-glass`, `--glass-blur`), sombras suaves multicapa y esquinas redondeadas calibradas (`--radius-md: 12px`, `--radius-lg: 18px`).
- **Mosaicos de Álbumes Dinámicos (`static/css/gallery.css`)**: Estilos dedicados para `.home-album-card`, `.home-album-mosaic`, `.home-album-info`, `.home-album-name` con animaciones de resorte iOS y carga diferida con skeletons.
- **Tarjetas de Personas y Avatares Circulares**: Estilos dedicados para `.person-card`, `.person-avatar` y `.sidebar-person-avatar` con anillos de acento/esmeralda y micro-zoom táctil al pasar el ratón.

### D. Protocolo de Purga y Reindexación ("Tabula Rasa") (`scripts/tabula_rasa_pipeline.py`)
- Snapshot de seguridad previo con copia atómica en `backups/tabula_rasa_pre_purge_<timestamp>/`.
- Purga completa de vectores obsoletos, índices residuales y pickles legacy sin tocar las fotos originales del usuario.
- Reindexación por lotes configurable con tolerancia a fallos.

---

## 3. Guía de Ejecución de la Reindexación Completa

Para ejecutar la purga y reindexación total sobre la base de fotos en cualquier momento:

```powershell
# 1. Auditoría del estado actual y comprobación de modelos
python scripts/tabula_rasa_pipeline.py --action audit

# 2. Simulación preventiva (Dry-Run: no modifica ficheros)
python scripts/tabula_rasa_pipeline.py --action purge --dry-run

# 3. Ejecución completa en vivo (Purga atómica + Reindexación con nuevo pipeline en lotes)
python scripts/tabula_rasa_pipeline.py --action all --batch-size 50
```

---

## 4. Resultados de Verificación y Métricas

- **Batería de Pruebas**: 84 tests automatizados ejecutados (`tests/run_all_tests.py`).
- **Tasa de Éxito**: **100% OK (0 fallos, 0 errores)** en 44.26s.
- **Benchmark de Rendimiento**:
  - Búsqueda vectorial facial (20.000 vectores 128-D): **6.24 ms**.
  - Detección de duplicados exactos (20.000 items): **48.01 ms**.
  - Detección de ráfagas pHash (20.000 items): **39.71 ms**.
  - Consulta paginada B-Tree Index: **0.44 ms**.
  - Arranque del servidor web Flask: **200 OK**.
  - Compilación frontend de 10 módulos: **0 errores**.
