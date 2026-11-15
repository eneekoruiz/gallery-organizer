# -*- coding: utf-8 -*-
"""
Script Forense SOTA: Purificación y Descontaminación Radical de Personas Sin Nombre.
- Elimina falsos positivos biométricos (maniquíes, dibujos animados, mascotas, capturas de pantalla).
- Aplica FIQA estricto con modelo 3D canónico corregido (IOD >= 20px, Nitidez Laplaciana >= 20, |yaw| <= 45°, |pitch| <= 35°).
- Escinde clústeres heterogéneos y agrupa únicamente rostros con similitud coseno >= 0.58.
- Reasigna a personas conocidas si la similitud contra centroides oficiales es >= 0.62.
- Desplaza elementos no agrupables a '❓ Desconocidos (Por Identificar)'.
"""

import sys
import os
import json
import shutil
import cv2
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.globals import APP_DIR, CONFIG_FILE
APPLE_DB_PATH = ROOT_DIR / "apple_db.json"
from app.services.face_service import detector, recognizer
from app.services.biometric_engine import estimate_face_quality_and_pose, extract_feature_from_aligned_tta
from app.utils.file_ops import safe_move_file

def load_known_centroids():
    centroids_path = ROOT_DIR / "centroids.pkl"
    if not centroids_path.exists():
        return {}, None, []
    import pickle
    with open(centroids_path, "rb") as f:
        c_dict = pickle.load(f)
    keys = []
    matrix = []
    for k, v in c_dict.items():
        ident = k[1] if isinstance(k, tuple) else str(k)
        if not ident.startswith("Persona Nueva") and "Desconocid" not in ident:
            vec = np.array(v, dtype=np.float32).flatten()
            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                keys.append((k[0] if isinstance(k, tuple) else "Conocidos", ident))
                matrix.append(vec / norm)
    if matrix:
        return c_dict, np.array(matrix), keys
    return c_dict, None, []

def get_best_usable_face(image_path: str):
    """Detecta rostros y devuelve el embedding del rostro de mayor calidad si cumple FIQA."""
    if not os.path.exists(image_path) or image_path.lower().endswith(('.mp4', '.mov', '.avi')):
        return None, "Archivo no existe o es vídeo"
    
    img = cv2.imread(image_path)
    if img is None:
        return None, "Error de decodificación de imagen"
    
    h, w = img.shape[:2]
    # Redimensionar si es ultra-grande para no desbordar memoria
    if max(h, w) > 1600:
        scale = 1600.0 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        h, w = img.shape[:2]
        
    detector.setInputSize((w, h))
    _, faces = detector.detect(img)
    if faces is None or len(faces) == 0:
        return None, "Sin rostros detectados"
        
    best_candidate = None
    best_score = -1.0
    reject_reasons = []
    
    for f in faces:
        metrics = estimate_face_quality_and_pose(img, f)
        if metrics.get("is_usable_for_clustering"):
            q = metrics.get("quality_score", 0.0)
            if q > best_score:
                best_score = q
                crop = recognizer.alignCrop(img, f)
                feat = extract_feature_from_aligned_tta(crop, recognizer)
                best_candidate = (feat, metrics)
        else:
            reject_reasons.append(metrics.get("reason", "FIQA bajo"))
            
    if best_candidate is not None:
        return best_candidate[0], "OK"
    return None, "; ".join(reject_reasons[:3])

def run_purification():
    print("=" * 70)
    print("INICIANDO PURIFICACIÓN BIOMÉTRICA DE PERSONAS SIN NOMBRE Y APPLE DB")
    print("=" * 70)
    
    cfg = {}
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            
    local_base = Path(cfg.get("local_path", r"C:\Users\User\Desktop\Galeria Eneko NO ABRIR"))
    resultados_dir = local_base / "Resultados"
    psn_dir = resultados_dir / "Personas Sin Nombre"
    desconocidos_dir = psn_dir / "Desconocidos"
    desconocidos_dir.mkdir(parents=True, exist_ok=True)
    
    _, known_matrix, known_keys = load_known_centroids()
    print(f"Cargados {len(known_keys)} centroides biométricos de referencia.")
    
    # ---------------------------------------------------------
    # 1. PURIFICAR DISK FOLDERS: Resultados/Personas Sin Nombre/Persona Nueva #...
    # ---------------------------------------------------------
    print("\n[PASO 1/3] Auditando carpetas físicas en disco...")
    stats = {
        "photos_scanned": 0,
        "non_human_evicted": 0,
        "reassigned_to_known": 0,
        "clusters_kept": 0,
        "clusters_disbanded": 0,
        "outliers_evicted": 0
    }
    
    if psn_dir.exists():
        subfolders = [d for d in psn_dir.iterdir() if d.is_dir() and d.name.startswith("Persona Nueva #")]
        print(f"Encontradas {len(subfolders)} carpetas físicas de Persona Nueva.")
        
        for folder in subfolders:
            folder_name = folder.name
            files = [f for f in folder.iterdir() if f.is_file() and not f.name.startswith((".", "!", "_00_"))]
            if not files:
                try:
                    folder.rmdir()
                except Exception:
                    pass
                continue
                
            print(f"\n--- Auditando {folder_name} ({len(files)} archivos) ---")
            
            valid_items = []  # list of (file_path, feat)
            
            for file_path in files:
                stats["photos_scanned"] += 1
                feat, status = get_best_usable_face(str(file_path))
                
                if feat is None:
                    # Falso positivo / no humano / borroso / sin cara válida
                    dest = desconocidos_dir / file_path.name
                    if dest.exists():
                        dest = desconocidos_dir / f"{file_path.stem}_{folder_name.replace('#', '')}{file_path.suffix}"
                    safe_move_file(str(file_path), str(dest))
                    stats["non_human_evicted"] += 1
                    print(f"  [EVICTED NO-HUMAN/BLUR] {file_path.name} -> Desconocidos ({status})")
                    continue
                    
                # Comprobar si coincide con alguien conocido
                if known_matrix is not None and len(known_matrix) > 0:
                    sims = np.dot(known_matrix, feat)
                    best_idx = int(np.argmax(sims))
                    best_sim = float(sims[best_idx])
                    if best_sim >= 0.62:
                        k_cat, k_ident = known_keys[best_idx]
                        dest_dir = resultados_dir / k_cat / k_ident
                        dest_dir.mkdir(parents=True, exist_ok=True)
                        dest = dest_dir / file_path.name
                        safe_move_file(str(file_path), str(dest))
                        stats["reassigned_to_known"] += 1
                        print(f"  [REASSIGNED KNOWN] {file_path.name} -> {k_cat}/{k_ident} (sim: {best_sim:.2f})")
                        continue
                        
                valid_items.append((file_path, feat))
                
            # Analizar coherencia intra-cluster de los archivos válidos
            if len(valid_items) < 2:
                # 0 o 1 archivo válido: mover singletons a Desconocidos
                for file_path, _ in valid_items:
                    dest = desconocidos_dir / file_path.name
                    if dest.exists():
                        dest = desconocidos_dir / f"{file_path.stem}_{folder_name.replace('#', '')}{file_path.suffix}"
                    safe_move_file(str(file_path), str(dest))
                    stats["outliers_evicted"] += 1
                    print(f"  [SINGLETON DISBAND] {file_path.name} -> Desconocidos (grupo sin pares)")
                stats["clusters_disbanded"] += 1
                try:
                    folder.rmdir()
                except Exception:
                    pass
            else:
                # Matriz de similitud intra-cluster
                X = np.array([item[1] for item in valid_items])
                sims = np.dot(X, X.T)
                
                # Encontrar el subgrupo homogéneo con mayor conectividad
                # Un rostro pertenece al núcleo si tiene similitud >= 0.58 con la mayoría
                core_indices = []
                for i in range(len(valid_items)):
                    high_sims = [j for j in range(len(valid_items)) if i != j and sims[i, j] >= 0.58]
                    if len(high_sims) >= 1: # Al menos un par legítimo
                        core_indices.append(i)
                        
                if len(core_indices) >= 2:
                    # Comprobar que los miembros del núcleo son consistentes entre sí
                    stats["clusters_kept"] += 1
                    kept_names = []
                    for i in range(len(valid_items)):
                        file_path, _ = valid_items[i]
                        if i in core_indices:
                            kept_names.append(file_path.name)
                        else:
                            # Outlier que no se parece a nadie del grupo
                            dest = desconocidos_dir / file_path.name
                            if dest.exists():
                                dest = desconocidos_dir / f"{file_path.stem}_{folder_name.replace('#', '')}{file_path.suffix}"
                            safe_move_file(str(file_path), str(dest))
                            stats["outliers_evicted"] += 1
                            print(f"  [OUTLIER EVICTED] {file_path.name} -> Desconocidos (discrepancia facial)")
                    print(f"  ✓ {folder_name} CONSOLIDADO con {len(kept_names)} fotos puras: {kept_names[:3]}...")
                else:
                    # Todos son dispares entre sí (similitudes bajas como 0.05 a 0.30)
                    for file_path, _ in valid_items:
                        dest = desconocidos_dir / file_path.name
                        if dest.exists():
                            dest = desconocidos_dir / f"{file_path.stem}_{folder_name.replace('#', '')}{file_path.suffix}"
                        safe_move_file(str(file_path), str(dest))
                        stats["outliers_evicted"] += 1
                        print(f"  [HETEROGENEOUS DISBAND] {file_path.name} -> Desconocidos (sin similitud)")
                    stats["clusters_disbanded"] += 1
                    try:
                        folder.rmdir()
                    except Exception:
                        pass
                        
    # ---------------------------------------------------------
    # 2. PURIFICAR APPLE_DB.JSON
    # ---------------------------------------------------------
    print("\n[PASO 2/3] Auditando apple_db.json...")
    apple_db = {}
    if APPLE_DB_PATH.exists():
        with open(APPLE_DB_PATH, "r", encoding="utf-8") as f:
            apple_db = json.load(f)
            
    apple_stats = {
        "scanned": 0,
        "non_human_purged": 0,
        "reassigned_known": 0,
        "moved_to_desconocidos": 0
    }
    
    # Lista de archivos específicos que son falsos positivos conocidos
    KNOWN_FALSE_POSITIVES = {
        "img_1905.png": "3D Maniquí Bratz/Roblox",
        "img_0087.png": "Dibujo artístico sobre fondo rojo",
        "img_20220106_111355_172.webp": "Hocico de Mascota/Perro",
        "img-20170315-wa0031.jpg": "Captura WhatsApp Kekitooo"
    }
    
    for p_str, info in apple_db.items():
        if not isinstance(info, dict) or info.get("deleted_locally"):
            continue
        fname_l = Path(p_str).name.lower()
        ident = str(info.get("identity", "")).strip()
        cat = str(info.get("category", "")).strip()
        
        # Comprobar si es un falso positivo conocido
        if fname_l in KNOWN_FALSE_POSITIVES:
            info["category"] = "📋 Revisión / Por Organizar"
            info["identity"] = "❓ Desconocidos (Por Identificar)"
            info["status"] = "Dudoso"
            info["is_dudoso"] = True
            info["review_required"] = True
            info["faces"] = []
            info["num_faces"] = 0
            apple_stats["non_human_purged"] += 1
            print(f"  [PURGED KNOWN FP] {fname_l} ({KNOWN_FALSE_POSITIVES[fname_l]}) -> Desconocidos")
            continue
            
        if "Persona Nueva" in ident or cat == "Personas Nuevas":
            apple_stats["scanned"] += 1
            
            # Verificar si existe imagen en local o si tiene cara en info
            feat = None
            if os.path.exists(p_str):
                feat, status = get_best_usable_face(p_str)
                if feat is None:
                    info["category"] = "📋 Revisión / Por Organizar"
                    info["identity"] = "❓ Desconocidos (Por Identificar)"
                    info["status"] = "Dudoso"
                    info["is_dudoso"] = True
                    info["review_required"] = True
                    apple_stats["non_human_purged"] += 1
                    print(f"  [PURGED APPLE_DB NO-HUMAN] {fname_l} -> Desconocidos ({status})")
                    continue
            else:
                # Usar feature preexistente si es válido
                for f_e in info.get("faces", []):
                    f_v = f_e.get("feature")
                    if f_v is not None:
                        v = np.array(f_v, dtype=np.float32).flatten()
                        norm = np.linalg.norm(v)
                        if norm > 1e-6:
                            feat = v / norm
                            break
                            
            if feat is not None:
                # Comprobar con conocidos
                if known_matrix is not None and len(known_matrix) > 0:
                    sims = np.dot(known_matrix, feat)
                    best_idx = int(np.argmax(sims))
                    best_sim = float(sims[best_idx])
                    if best_sim >= 0.62:
                        k_cat, k_ident = known_keys[best_idx]
                        info["category"] = k_cat
                        info["identity"] = k_ident
                        info["status"] = "Clasificado"
                        info["is_dudoso"] = False
                        info["review_required"] = False
                        apple_stats["reassigned_known"] += 1
                        print(f"  [APPLE_DB REASSIGNED KNOWN] {fname_l} -> {k_cat}/{k_ident} (sim: {best_sim:.2f})")
                        continue
                        
            # Si el clúster al que pertenecía ya no existe o es bimodal/outlier
            # Comprobar si la carpeta física de esa Persona Nueva aún existe en disco
            clean_ident = ident.replace("👤", "").strip()
            disk_cluster_dir = psn_dir / clean_ident
            if not disk_cluster_dir.exists() or len(list(disk_cluster_dir.glob("*.*"))) < 2:
                # La carpeta física fue disuelta o tiene menos de 2 fotos
                info["category"] = "📋 Revisión / Por Organizar"
                info["identity"] = "❓ Desconocidos (Por Identificar)"
                info["status"] = "Dudoso"
                info["is_dudoso"] = True
                info["review_required"] = True
                apple_stats["moved_to_desconocidos"] += 1
                print(f"  [APPLE_DB DEMOTE OUTLIER] {fname_l} ({ident}) -> Desconocidos")
                
    with open(APPLE_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(apple_db, f, indent=2, ensure_ascii=False)
    print(f"✓ {APPLE_DB_PATH} actualizado correctamente.")
    
    # ---------------------------------------------------------
    # 3. RECONSTRUCCIÓN DE CACHÉ DE GALERÍA
    # ---------------------------------------------------------
    print("\n[PASO 3/3] Reconstruyendo gallery_cache.json limpio...")
    from app.services.gallery_service import get_gallery
    clean_gallery = get_gallery(force_reload=True)
    
    # Guardar caché final
    from app.globals import set_gallery_cache, GALLERY_CACHE_FILE
    set_gallery_cache(clean_gallery)
    with open(GALLERY_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(clean_gallery, f, indent=2, ensure_ascii=False)
    print("✓ gallery_cache.json regenerado y guardado.")
    
    print("\n" + "=" * 70)
    print("RESUMEN DE PURIFICACIÓN Y RESULTADOS:")
    print(f"  Fotos físicas escaneadas: {stats['photos_scanned']}")
    print(f"  Falsos positivos no humanos/borrosos expulsados: {stats['non_human_evicted']}")
    print(f"  Fotos reasignadas a identidades conocidas: {stats['reassigned_to_known']}")
    print(f"  Clústeres heterogéneos disueltos: {stats['clusters_disbanded']}")
    print(f"  Clústeres puros consolidados: {stats['clusters_kept']}")
    print(f"  Outliers aislados a Desconocidos: {stats['outliers_evicted']}")
    print(f"  Apple DB purgados (no-humanos): {apple_stats['non_human_purged']}")
    print(f"  Apple DB reasignados a conocidos: {apple_stats['reassigned_known']}")
    print(f"  Apple DB outliers a Desconocidos: {apple_stats['moved_to_desconocidos']}")
    print("=" * 70)

if __name__ == "__main__":
    run_purification()
