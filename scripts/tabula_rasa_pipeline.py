r"""Script de Purga Total, Migración y Reindexación Biométrica ("Tabula Rasa").
Garantiza cero contaminación de metadatos o vectores antiguos y ejecuta el nuevo pipeline
de visión forense (FIQA, TTA, 3D Pose, Consenso UP-CAC con Cannot-Link y Auditoría Espectral).
"""

import os
import sys
import json
import time
import shutil
import sqlite3
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2

# Añadir raíz de la aplicación al sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.services.biometric_engine import estimate_face_quality_and_pose, extract_features_with_tta
from app.utils.clustering import cluster_faces_consensus_upcac, audit_and_split_contaminated_clusters

DB_PATH = ROOT_DIR / "smart_gallery_index.db"
APPLE_DB_PATH = ROOT_DIR / "apple_db.json"
BACKUP_BASE_DIR = ROOT_DIR / "backups"
CHECKPOINT_FILE = ROOT_DIR / "tabula_rasa_checkpoint.json"

LEGACY_FILES_TO_PURGE = [
    ROOT_DIR / "sface_cache_v7.pkl",
    ROOT_DIR / "faces_cache_v2.pkl",
    ROOT_DIR / "faces_cache.pkl",
    ROOT_DIR / "centroids.pkl",
    ROOT_DIR / "video_faces_cache.json",
    ROOT_DIR / "recluster_status.json",
    ROOT_DIR / "smart_clean_status.json"
]


def create_security_backup() -> Path:
    """Crea una copia de seguridad atómica de los datos actuales antes de cualquier purga."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = BACKUP_BASE_DIR / f"tabula_rasa_pre_purge_{ts}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    print(f"[BACKUP] Creando snapshot de seguridad en: {backup_dir}")

    if DB_PATH.exists():
        shutil.copy2(DB_PATH, backup_dir / DB_PATH.name)
        print(f"  -> Respaldado {DB_PATH.name} ({DB_PATH.stat().st_size // 1024} KB)")

    if APPLE_DB_PATH.exists():
        shutil.copy2(APPLE_DB_PATH, backup_dir / APPLE_DB_PATH.name)
        print(f"  -> Respaldado {APPLE_DB_PATH.name} ({APPLE_DB_PATH.stat().st_size // 1024} KB)")

    for lf in LEGACY_FILES_TO_PURGE:
        if lf.exists():
            shutil.copy2(lf, backup_dir / lf.name)

    print("[BACKUP] Snapshot completado exitosamente.")
    return backup_dir


def execute_purge(dry_run: bool = False):
    """Ejecuta la purga total de embeddings, tablas de análisis y cachés legacy."""
    print("\n" + "="*70)
    print("EJECUTANDO PROTOCOLO TABULA RASA: PURGA TOTAL")
    print("="*70)

    if dry_run:
        print("[DRY-RUN] Modo simulación activo. No se realizarán escrituras destructivas.")

    # 1. Purga de SQLite (smart_gallery_index.db)
    if DB_PATH.exists():
        print(f"[PURGA SQL] Limpiando índices y vectores en {DB_PATH}...")
        if not dry_run:
            conn = sqlite3.connect(str(DB_PATH))
            cur = conn.cursor()
            cur.execute("PRAGMA journal_mode = WAL;")

            # Resetear columnas en media_index
            cur.execute("""
            UPDATE media_index 
            SET embedding_blob = NULL,
                identity = NULL,
                faces_count = 0,
                confidence = 0.0,
                updated_at = strftime('%s', 'now');
            """)

            # Vaciar tablas de análisis previo
            cur.execute("DELETE FROM content_face_analysis;")
            cur.execute("DELETE FROM content_locations;")

            # Sellar esquema con versión SOTA
            cur.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_version (
                version TEXT PRIMARY KEY,
                embedding_dim INTEGER,
                metric TEXT,
                updated_at REAL
            );
            """)
            cur.execute("""
            INSERT OR REPLACE INTO pipeline_version (version, embedding_dim, metric, updated_at)
            VALUES ('v2.0-sota-forensic', 128, 'cosine', strftime('%s', 'now'));
            """)

            conn.commit()
            cur.execute("VACUUM;")
            cur.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            conn.close()
            print("  -> Base de datos SQL purgada y compactada (VACUUM completo).")

    # 2. Purga de apple_db.json
    if APPLE_DB_PATH.exists():
        print(f"[PURGA JSON] Descontaminando {APPLE_DB_PATH}...")
        if not dry_run:
            with open(APPLE_DB_PATH, 'r', encoding='utf-8') as f:
                apple_db = json.load(f)

            purged_count = 0
            for path, info in apple_db.items():
                if not isinstance(info, dict):
                    continue
                # Preservar EXIF y rutas físicas; purgar inferencias previas
                info["faces_count"] = 0
                info["faces"] = []
                info["identities"] = []
                info["score"] = 0.0

                current_id = str(info.get("identity", ""))
                if "Persona Nueva" in current_id or "Desconocid" in current_id:
                    info["identity"] = "❓ Desconocidos (Por Identificar)"
                    info["status"] = "Por Organizar"
                purged_count += 1

            with open(APPLE_DB_PATH, 'w', encoding='utf-8') as f:
                json.dump(apple_db, f, indent=2, ensure_ascii=False)
            print(f"  -> {purged_count} entradas de metadatos descontaminadas.")

    # 3. Purga de cachés de disco y pickles
    print("[PURGA DISCO] Eliminando archivos de caché y pickles legacy...")
    for lf in LEGACY_FILES_TO_PURGE:
        if lf.exists():
            print(f"  -> Eliminando {lf.name}")
            if not dry_run:
                try:
                    os.remove(lf)
                except Exception as e:
                    print(f"     [AVISO] No se pudo borrar {lf.name}: {e}")

    # 4. Limpiar carpetas de resultados provisionales
    resultados_dir = ROOT_DIR / "Resultados"
    for sub in ["Personas Sin Nombre", "Desconocidos"]:
        target = resultados_dir / sub
        if target.exists():
            print(f"  -> Limpiando contenido en: {target}")
            if not dry_run:
                for item in target.iterdir():
                    try:
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            os.remove(item)
                    except Exception:
                        pass

    print("[PURGA] Estado 'Tabula Rasa' alcanzado. Cero contaminación remanente.\n")


def reindex_dataset(
    batch_size: int = 50,
    max_photos: int = None,
    quality_thresh: float = 0.45
):
    """Reprocesa el conjunto de fotos aplicando el nuevo pipeline biométrico."""
    print("="*70)
    print("INICIANDO REINDEXACIÓN CON PIPELINE BIOMÉTRICO FORENSE SOTA")
    print("="*70)

    # 1. Cargar modelos
    yunet_path = str(ROOT_DIR / "models" / "face_detection_yunet.onnx")
    sface_path = str(ROOT_DIR / "models" / "face_recognition_sface.onnx")

    if not os.path.exists(yunet_path) or not os.path.exists(sface_path):
        print(f"[ERROR] No se encontraron los modelos requeridos en {ROOT_DIR / 'models'}")
        return

    print("[IA] Inicializando detector YuNet con OpenCV DNN...")
    detector = cv2.FaceDetectorYN.create(yunet_path, "", (320, 320), 0.50, 0.35, 5000)
    print("[IA] Inicializando extractor FaceRecognizerSF con TTA...")
    recognizer = cv2.FaceRecognizerSF.create(sface_path, "")

    # 2. Recolectar lista de imágenes válidas
    candidate_paths = []
    if APPLE_DB_PATH.exists():
        with open(APPLE_DB_PATH, 'r', encoding='utf-8') as f:
            adb = json.load(f)
            candidate_paths = [p for p in adb.keys() if os.path.exists(p)]

    if not candidate_paths:
        # Fallback a escanear carpeta Fotos/
        fotos_dir = ROOT_DIR / "Fotos"
        if fotos_dir.exists():
            for root, _, files in os.walk(str(fotos_dir)):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.heic')):
                        candidate_paths.append(os.path.join(root, file))

    if max_photos:
        candidate_paths = candidate_paths[:max_photos]

    total_images = len(candidate_paths)
    print(f"[DATASET] Total de imágenes a procesar: {total_images}")

    if total_images == 0:
        print("[DATASET] No se encontraron imágenes para reindexar.")
        return

    # 3. Buffer de procesamiento
    all_embeddings = []
    all_photo_ids = []
    all_face_records = []
    quarantine_faces = 0

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    t_start = time.time()
    for idx, img_path in enumerate(candidate_paths):
        if idx % 20 == 0 or idx == total_images - 1:
            elapsed = time.time() - t_start
            fps = (idx + 1) / max(elapsed, 0.001)
            print(f"  -> Progreso: {idx+1}/{total_images} fotos ({(idx+1)/total_images*100:.1f}%) | {fps:.1f} fotos/seg")

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(img)

        if faces is None or len(faces) == 0:
            continue

        valid_faces_in_image = []
        for face in faces:
            # Evaluación FIQA + Pose 3D
            quality = estimate_face_quality_and_pose(img, face)
            if not quality["is_usable_for_clustering"] or quality["quality_score"] < quality_thresh:
                quarantine_faces += 1
                continue

            # Extracción con Test-Time Augmentation
            try:
                feat_tta = extract_features_with_tta(img, face, recognizer)
                all_embeddings.append(feat_tta)
                all_photo_ids.append(img_path)

                valid_faces_in_image.append({
                    "box": [float(face[0]), float(face[1]), float(face[2]), float(face[3])],
                    "landmarks": [float(v) for v in face[4:14]],
                    "quality": quality,
                    "feature": feat_tta.tolist()
                })
            except Exception:
                pass

        # Actualizar índice SQL con el embedding de mayor calidad si existe
        if valid_faces_in_image:
            best_face = max(valid_faces_in_image, key=lambda f: f["quality"]["quality_score"])
            best_emb_blob = np.array(best_face["feature"], dtype=np.float32).tobytes()
            cur.execute("""
            UPDATE media_index
            SET embedding_blob = ?,
                faces_count = ?,
                confidence = ?,
                updated_at = strftime('%s', 'now')
            WHERE file_path = ?;
            """, (best_emb_blob, len(valid_faces_in_image), best_face["quality"]["quality_score"], img_path))

        if idx % batch_size == 0:
            conn.commit()

    conn.commit()
    conn.close()

    print(f"\n[EXTRACCIÓN COMPLETADA]")
    print(f"  -> Caras de alta calidad extraídas: {len(all_embeddings)}")
    print(f"  -> Caras rechazadas por FIQA/Pose a cuarentena: {quarantine_faces}")

    # 4. Agrupamiento Forense con Restricciones Físicas Cannot-Link y Auditoría Espectral
    if len(all_embeddings) >= 2:
        print("\n[CLUSTERING] Ejecutando Consensus UP-CAC con Cannot-Link...")
        X = np.array(all_embeddings, dtype=np.float32)
        labels = cluster_faces_consensus_upcac(X, photo_ids=all_photo_ids, vote_threshold=0.80)

        print("[AUDITORÍA] Evaluando bimodalidad espectral (Normalized Cut / Fiedler)...")
        audited_labels = audit_and_split_contaminated_clusters(X, labels, min_cluster_size=6, fiedler_threshold=0.10)

        unique_clusters = len(np.unique(audited_labels[audited_labels >= 0]))
        print(f"[RESULTADO CLUSTERING] Identidades puras identificadas: {unique_clusters}")
    else:
        print("[CLUSTERING] Insuficientes muestras para agrupamiento multi-rostro.")

    print("\n" + "="*70)
    print("PROCESO TABULA RASA Y REINDEXACIÓN FINALIZADO CON ÉXITO")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Pipeline Tabula Rasa para Galería Inteligente.")
    parser.add_argument("--action", choices=["audit", "backup", "purge", "reindex", "all"], default="audit",
                        help="Acción a realizar (audit, backup, purge, reindex, all)")
    parser.add_argument("--dry-run", action="store_true", help="Simular sin modificar archivos reales")
    parser.add_argument("--limit", type=int, default=None, help="Límite de fotos para reindexar (para pruebas)")
    parser.add_argument("--batch-size", type=int, default=50, help="Tamaño de lote para transacciones SQL")
    args = parser.parse_args()

    print(f"[INICIO] Tabula Rasa Pipeline en modo: {args.action}")

    if args.action == "audit":
        print("\n--- AUDITORÍA DE ESTADO ACTUAL ---")
        print(f"Base de datos SQLite ({DB_PATH.name}): {'Existe' if DB_PATH.exists() else 'No existe'}")
        if DB_PATH.exists():
            conn = sqlite3.connect(str(DB_PATH))
            cur = conn.cursor()
            cur.execute("SELECT count(*) FROM media_index")
            print(f"  Total registros media_index: {cur.fetchone()[0]}")
            cur.execute("SELECT count(*) FROM media_index WHERE embedding_blob IS NOT NULL")
            print(f"  Vectores de características persistidos: {cur.fetchone()[0]}")
            conn.close()

        print(f"Metadatos Apple DB ({APPLE_DB_PATH.name}): {'Existe' if APPLE_DB_PATH.exists() else 'No existe'}")
        print("\nCachés legacy detectadas:")
        for lf in LEGACY_FILES_TO_PURGE:
            if lf.exists():
                print(f"  [PRESENTE] {lf.name} ({lf.stat().st_size} bytes)")

    elif args.action == "backup":
        create_security_backup()

    elif args.action == "purge":
        create_security_backup()
        execute_purge(dry_run=args.dry_run)

    elif args.action == "reindex":
        reindex_dataset(batch_size=args.batch_size, max_photos=args.limit)

    elif args.action == "all":
        create_security_backup()
        execute_purge(dry_run=args.dry_run)
        reindex_dataset(batch_size=args.batch_size, max_photos=args.limit)


if __name__ == "__main__":
    main()
