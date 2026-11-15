# -*- coding: utf-8 -*-
"""
Detector y Clasificador Especializado de Personas de Espaldas (Back-Turned Persons).

Agrupa automáticamente fotos donde aparecen personas de espaldas (cuerpo humano detectado
pero sin rostro frontal visible para el algoritmo biométrico) dentro de:
  Categoría: 📋 Revisión / Por Organizar
  Sub-álbum: 👤 Personas de Espaldas
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.globals import FOTOS_DIR, RESULTADOS_DIR
APPLE_DB_PATH = ROOT_DIR / "apple_db.json"
from app.services.pet_service import detect_pets_and_bodies
from app.utils.file_ops import safe_move_file


def classify_back_turned_photos(dry_run: bool = False, max_photos: int = None, conf_thresh: float = 0.28):
    """
    Escanea fotos sin rostro confirmado en apple_db.json y carpetas físicas,
    detecta personas de espaldas mediante YOLOv8 y las enruta a '👤 Personas de Espaldas'.
    """
    print("=" * 70)
    print("INICIANDO AUDITORÍA Y CLASIFICACIÓN DE PERSONAS DE ESPALDAS")
    print("=" * 70)

    stats = {
        "scanned": 0,
        "classified_espaldas": 0,
        "already_espaldas": 0,
        "skipped_with_faces": 0,
        "errors": 0
    }

    espaldas_dir = RESULTADOS_DIR / "Revision_Interactiva" / "Personas_De_Espaldas"
    if not dry_run:
        espaldas_dir.mkdir(parents=True, exist_ok=True)

    # 1. Procesar apple_db.json si existe
    if APPLE_DB_PATH.exists():
        try:
            with open(APPLE_DB_PATH, "r", encoding="utf-8") as f:
                apple_db = json.load(f)
        except Exception as e:
            print(f"[ERROR] No se pudo leer {APPLE_DB_PATH}: {e}")
            apple_db = {}

        updated_apple_db = False
        candidates = []
        for p_str, info in apple_db.items():
            if not isinstance(info, dict) or info.get("deleted_locally"):
                continue

            ident = str(info.get("identity", "")).strip()
            cat = str(info.get("category", "")).strip()

            if "espalda" in ident.lower():
                stats["already_espaldas"] += 1
                continue

            # Candidatos: Fotos en Sin Rostro o Para Revisar sin caras válidas
            if ident in ("📷 Fotos Sin Rostro", "Sin_Rostro", "❓ Desconocidos (Por Identificar)", "⏳ Pendientes de Organizar") or "sin_rostro" in p_str.lower():
                faces = info.get("faces", [])
                has_named_face = any(
                    f.get("identity") and not str(f.get("identity")).startswith(
                        ("Desconocid", "Falso_Positivo", "Excluid", "Ignorar", "Persona (Sin rostro", "👤 Persona de Espaldas")
                    )
                    for f in faces
                )
                if not has_named_face:
                    candidates.append((p_str, info))

        print(f"[CANDIDATOS] {len(candidates)} fotos candidatas en base de datos.")

        for idx, (p_str, info) in enumerate(candidates):
            if max_photos and idx >= max_photos:
                break

            stats["scanned"] += 1
            if not os.path.exists(p_str):
                continue

            try:
                # Decodificar imagen para detección
                img_bytes = np.fromfile(p_str, dtype=np.uint8)
                img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                if img is None:
                    continue

                h, w = img.shape[:2]
                res = detect_pets_and_bodies(img, conf_thresh=conf_thresh)
                bodies = res.get("bodies", [])

                if bodies:
                    # Se ha detectado cuerpo(s) humano(s) sin rostro frontal
                    best_body = max(bodies, key=lambda b: b.get("confidence", 0.0))
                    bx, by, bw, bh = best_body["x"], best_body["y"], best_body["width"], best_body["height"]

                    head_h = max(25, int(bh * 0.35))
                    head_w = max(25, int(bw * 0.65))
                    head_x = bx + int((bw - head_w) / 2)
                    head_y = by

                    body_face_entry = {
                        "x": head_x,
                        "y": head_y,
                        "width": head_w,
                        "height": head_h,
                        "img_width": w,
                        "img_height": h,
                        "identity": "👤 Persona de Espaldas",
                        "confidence": f"👤 Persona de Espaldas ({best_body['confidence']*100:.1f}%)",
                        "tag_type": "body",
                        "is_body": True,
                        "is_back_turned": True,
                        "is_manual": False
                    }

                    if not dry_run:
                        info["category"] = "📋 Revisión / Por Organizar"
                        info["identity"] = "👤 Personas de Espaldas"
                        info["status"] = "Por Organizar"
                        cur_faces = info.get("faces", [])
                        # Reemplazar o añadir entrada de cuerpo
                        info["faces"] = [body_face_entry]
                        info["faces_count"] = len(info["faces"])
                        dst_p = espaldas_dir / Path(p_str).name
                        if os.path.exists(p_str) and Path(p_str).resolve() != dst_p.resolve():
                            safe_move_file(str(p_str), str(dst_p))
                            info["path"] = str(dst_p)
                        updated_apple_db = True

                    stats["classified_espaldas"] += 1
                    if idx % 10 == 0 or stats["classified_espaldas"] <= 5:
                        print(f"  -> [ESPALDA DETECTADA] {Path(p_str).name} (conf: {best_body['confidence']*100:.1f}%)")

            except Exception as e:
                stats["errors"] += 1

        if updated_apple_db and not dry_run:
            with open(APPLE_DB_PATH, "w", encoding="utf-8") as f:
                json.dump(apple_db, f, indent=2, ensure_ascii=False)
            print(f"[GUARDADO] {APPLE_DB_PATH} actualizado exitosamente.")

    # 2. Procesar directorio físico Sin_Rostro si existe
    phys_sin_rostro = RESULTADOS_DIR / "Revision_Interactiva" / "Sin_Rostro"
    if phys_sin_rostro.exists():
        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
        phys_files = [f for f in phys_sin_rostro.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
        print(f"[DIRECTORIO FÍSICO] {len(phys_files)} archivos en {phys_sin_rostro.name}")

        for idx, fpath in enumerate(phys_files):
            if max_photos and stats["scanned"] >= max_photos:
                break
            stats["scanned"] += 1

            try:
                img_bytes = np.fromfile(str(fpath), dtype=np.uint8)
                img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                if img is None:
                    continue

                res = detect_pets_and_bodies(img, conf_thresh=conf_thresh)
                bodies = res.get("bodies", [])

                if bodies:
                    best_body = max(bodies, key=lambda b: b.get("confidence", 0.0))
                    if not dry_run:
                        safe_move_file(str(fpath), str(espaldas_dir / fpath.name))
                    stats["classified_espaldas"] += 1
                    print(f"  -> [FÍSICO MOVIDO A ESPALDAS] {fpath.name}")
            except Exception:
                stats["errors"] += 1

    print("\n" + "=" * 70)
    print("RESUMEN DE CLASIFICACIÓN 'PERSONAS DE ESPALDAS'")
    print(f"  • Fotos escaneadas: {stats['scanned']}")
    print(f"  • Nuevas clasificadas en '👤 Personas de Espaldas': {stats['classified_espaldas']}")
    print(f"  • Ya existentes de espaldas: {stats['already_espaldas']}")
    print(f"  • Errores / ignoradas: {stats['errors']}")
    print("=" * 70)

    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Clasifica fotos de personas dadas la vuelta / de espaldas.")
    parser.add_argument("--dry-run", action="store_true", help="Simula sin mover ni guardar cambios.")
    parser.add_argument("--limit", type=int, default=None, help="Límite máximo de fotos a analizar.")
    parser.add_argument("--conf", type=float, default=0.28, help="Umbral de confianza para detección de cuerpo YOLO.")
    args = parser.parse_args()

    classify_back_turned_photos(dry_run=args.dry_run, max_photos=args.limit, conf_thresh=args.conf)
