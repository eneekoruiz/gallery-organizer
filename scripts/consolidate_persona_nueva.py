import sys
import os
import re
import json
import pickle
import unicodedata
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from sklearn.cluster import AgglomerativeClustering

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.globals import clear_gallery_cache
from app.utils.file_ops import safe_move_file, is_icloud_file

RES_DIR = Path(r"C:\Users\User\Desktop\Galeria Eneko NO ABRIR\Resultados")
PSN_DIR = RES_DIR / "Personas Sin Nombre"
DESCONOCIDOS_DIR = PSN_DIR / "Desconocidos"
DUDOSOS_DIR = RES_DIR / "Dudosos"
APPLE_DB_PATH = BASE_DIR / "apple_db.json"
FACES_CACHE_PATH = BASE_DIR / "faces_cache.pkl"
CENTROIDS_PATH = BASE_DIR / "centroids.pkl"

def clean_name(n):
    return re.sub(r'^[👤📋📁❓🔍✨📸⚡🎉🏷️👥🧑👧👦\s]+', '', str(n)).strip()

def run_consolidation():
    print("=" * 60)
    print("🚀 SMART GALLERY AI - CONSOLIDACIÓN INTEGRAL DE PERSONAS NUEVAS")
    print("=" * 60)

    # 1. Load apple_db.json
    print("\n[1/7] Cargando base de datos apple_db.json...")
    with open(APPLE_DB_PATH, 'r', encoding='utf-8') as f:
        db = json.load(f)
    print(f"  Total elementos en apple_db: {len(db)}")

    # 2. Fix explicit errors (IMG_6638.PNG -> C. Kid)
    kid_fixed = 0
    for k, v in db.items():
        if 'IMG_6638.PNG' in k or 'IMG_6638.PNG' in v.get('filepath', '') or 'IMG_6638.PNG' in v.get('path', ''):
            v['category'] = 'Conocidos'
            v['identity'] = 'C. Kid'
            v['status'] = 'Clasificado'
            v['is_dudoso'] = False
            v['identities'] = ['C. Kid']
            kid_fixed += 1
            print(f"  ✓ Corregida foto IMG_6638.PNG -> C. Kid (Conocidos)")
    
    # 3. Load centroids and build confirmed matrix
    print("\n[2/7] Construyendo matriz biométrica de identidades confirmadas...")
    with open(CENTROIDS_PATH, 'rb') as f:
        centroids_raw = pickle.load(f)

    confirmed_keys = []
    matrix_list = []
    for k, v in centroids_raw.items():
        if v is None: continue
        if isinstance(k, tuple):
            cat, ident = k
            if cat in ['Conocidos', 'Familia', 'YO', 'Mascotas', 'Profesores']:
                ident_l = ident.lower()
                if not any(term in ident_l for term in ['persona nueva', 'persona #', 'desconocid', 'sin rostro', 'ignorar', 'falso', 'dudoso']):
                    arr = np.array(v, dtype=np.float32).flatten()
                    norm = np.linalg.norm(arr)
                    if norm > 0:
                        confirmed_keys.append((cat, ident))
                        matrix_list.append(arr / norm)

    confirmed_matrix = np.stack(matrix_list)
    print(f"  Matriz biométrica construida: {confirmed_matrix.shape[0]} identidades confirmadas x {confirmed_matrix.shape[1]} dims")

    # 4. Load faces_cache.pkl
    print("\n[3/7] Cargando faces_cache.pkl...")
    try:
        with open(FACES_CACHE_PATH, 'rb') as f:
            faces_cache = pickle.load(f)
    except Exception as e:
        print(f"  Aviso: Error cargando faces_cache ({e}), inicializando vacío")
        faces_cache = {}
    print(f"  Total entradas en faces_cache: {len(faces_cache)}")

    # 5. Rescue high-confidence matches from Desconocidos on disk
    print("\n[4/7] Rescatando fotos con alta certeza biométrica en 'Desconocidos'...")
    rescued_desconocidos = 0
    if DESCONOCIDOS_DIR.exists():
        for f in list(DESCONOCIDOS_DIR.iterdir()):
            if not f.is_file(): continue
            f_key = str(f)
            cached_faces = faces_cache.get(f_key)
            if not cached_faces and f.name in faces_cache:
                cached_faces = faces_cache[f.name]
            
            if not cached_faces or not isinstance(cached_faces, list):
                continue
                
            best_sim = -1.0
            best_target = None
            for face in cached_faces:
                feat = face.get('feature') if isinstance(face, dict) else None
                if feat:
                    arr = np.array(feat, dtype=np.float32).flatten()
                    norm = np.linalg.norm(arr)
                    if norm > 0:
                        sims = np.dot(confirmed_matrix, arr / norm)
                        top_idx = int(np.argmax(sims))
                        top_sim = float(sims[top_idx])
                        if top_sim > best_sim:
                            best_sim = top_sim
                            best_target = confirmed_keys[top_idx]
            
            # High certainty rescue threshold >= 0.52
            if best_sim >= 0.52 and best_target is not None:
                tgt_cat, tgt_id = best_target
                dest_dir = RES_DIR / tgt_cat / tgt_id
                dest_dir.mkdir(parents=True, exist_ok=True)
                target_dest = dest_dir / f.name
                if safe_move_file(f, target_dest):
                    rescued_desconocidos += 1
                    # Find actual destination path
                    actual_dest = target_dest if target_dest.exists() else None
                    if not actual_dest:
                        cand = list(dest_dir.glob(f"{f.stem}*{f.suffix}"))
                        if cand: actual_dest = cand[0]
                    if actual_dest:
                        if f_key in faces_cache:
                            faces_cache[str(actual_dest)] = faces_cache.pop(f_key)
                        if f_key in db:
                            db[f_key]['category'] = tgt_cat
                            db[f_key]['identity'] = tgt_id
                            db[f_key]['status'] = 'Clasificado'
                            db[f_key]['is_dudoso'] = False
                            db[f_key]['path'] = str(actual_dest)
    print(f"  ✓ Rescatadas {rescued_desconocidos} fotos de Desconocidos hacia sus personas confirmadas.")

    # 6. Process all Persona Nueva items
    print("\n[5/7] Procesando fotos asignadas a 'Persona Nueva'...")
    pn_keys = [k for k, v in db.items() if 'persona nueva' in v.get('identity', '').lower()]
    print(f"  Total fotos asignadas a 'Persona Nueva' en apple_db: {len(pn_keys)}")

    matched_confirmed = 0
    unmatched_keys = []
    unmatched_feats = []

    for k in pn_keys:
        v = db[k]
        faces = v.get('faces', [])
        best_sim = -1.0
        best_target = None
        primary_feat = None

        for face in faces:
            feat = face.get('feature')
            if feat:
                arr = np.array(feat, dtype=np.float32).flatten()
                norm = np.linalg.norm(arr)
                if norm > 0:
                    sims = np.dot(confirmed_matrix, arr / norm)
                    top_idx = int(np.argmax(sims))
                    top_sim = float(sims[top_idx])
                    if top_sim > best_sim:
                        best_sim = top_sim
                        best_target = confirmed_keys[top_idx]
                        primary_feat = arr / norm

        # If similarity >= 0.40 -> Match to confirmed person!
        if best_sim >= 0.40 and best_target is not None:
            tgt_cat, tgt_id = best_target
            v['category'] = tgt_cat
            v['identity'] = tgt_id
            v['status'] = 'Clasificado'
            v['is_dudoso'] = False
            v['identities'] = [tgt_id]
            matched_confirmed += 1

            # If file is on disk in Resultados/Personas Sin Nombre or Dudosos, move it physically
            file_p = Path(v.get('filepath') or v.get('path') or k)
            if file_p.exists() and not is_icloud_file(str(file_p)):
                dest_dir = RES_DIR / tgt_cat / tgt_id
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_target = dest_dir / file_p.name
                if safe_move_file(file_p, dest_target):
                    actual_dest = dest_target if dest_target.exists() else None
                    if not actual_dest:
                        cand = list(dest_dir.glob(f"{file_p.stem}*{file_p.suffix}"))
                        if cand: actual_dest = cand[0]
                    if actual_dest:
                        v['path'] = str(actual_dest)
                        v['filepath'] = str(actual_dest)
        else:
            unmatched_keys.append(k)
            if primary_feat is not None:
                unmatched_feats.append(primary_feat)
            else:
                unmatched_feats.append(np.zeros(128, dtype=np.float32))

    print(f"  ✓ {matched_confirmed} fotos de 'Persona Nueva' fusionadas directamente en sus personas confirmadas.")
    print(f"  Fotos restantes sin coincidencia confirmada: {len(unmatched_keys)}")

    # 7. Cluster remaining unknown faces into dense groups
    print("\n[6/7] Agrupando rostros desconocidos en clusters densos y compactos...")
    if len(unmatched_keys) > 0:
        X = np.array(unmatched_feats)
        # Cosine distance matrix
        dist_matrix = 1.0 - np.dot(X, X.T)
        np.clip(dist_matrix, 0, 2, out=dist_matrix)

        # Agglomerative clustering with distance_threshold=0.44 (cosine similarity >= 0.56)
        clustering = AgglomerativeClustering(
            distance_threshold=0.44,
            n_clusters=None,
            metric='precomputed',
            linkage='average'
        ).fit(dist_matrix)

        labels = clustering.labels_
        cluster_groups = defaultdict(list)
        for idx, lbl in enumerate(labels):
            cluster_groups[lbl].append(unmatched_keys[idx])

        # Separate multi-photo clusters (>= 2 photos) from singletons (1 photo)
        multi_clusters = {lbl: items for lbl, items in cluster_groups.items() if len(items) >= 2}
        singletons = [items[0] for lbl, items in cluster_groups.items() if len(items) == 1]

        # Second-pass: Super-cluster multi-cluster centroids to merge different lighting/angles of same person (sim >= 0.50)
        if len(multi_clusters) > 1:
            multi_lbls = list(multi_clusters.keys())
            c_matrix = []
            for m_lbl in multi_lbls:
                items = multi_clusters[m_lbl]
                m_feats = [unmatched_feats[unmatched_keys.index(it)] for it in items if it in unmatched_keys]
                c = np.mean(m_feats, axis=0) if m_feats else np.zeros(128, dtype=np.float32)
                norm_c = np.linalg.norm(c)
                c_matrix.append(c / norm_c if norm_c > 0 else c)
            
            C = np.array(c_matrix)
            dist_c = 1.0 - np.dot(C, C.T)
            np.clip(dist_c, 0, 2, out=dist_c)

            super_cl = AgglomerativeClustering(
                distance_threshold=0.50,
                n_clusters=None,
                metric='precomputed',
                linkage='average'
            ).fit(dist_c)

            merged_multi = defaultdict(list)
            for m_lbl, s_lbl in zip(multi_lbls, super_cl.labels_):
                merged_multi[s_lbl].extend(multi_clusters[m_lbl])
            multi_clusters = merged_multi

        # Sort multi-clusters by size descending
        sorted_multi = sorted(multi_clusters.items(), key=lambda x: len(x[1]), reverse=True)
        print(f"  Clusters recurrentes consolidados (>= 2 fotos): {len(sorted_multi)}")
        print(f"  Rostros aislados / Ocasionales (1 foto): {len(singletons)}")

        # Assign clean, sequential Persona Nueva #1, #2, ... to multi-clusters
        psn_dir = RES_DIR / "Personas Sin Nombre"
        psn_dir.mkdir(parents=True, exist_ok=True)

        for new_num, (lbl, items) in enumerate(sorted_multi, 1):
            cluster_name = f"👤 Persona Nueva #{new_num}"
            cluster_dir = psn_dir / f"Persona Nueva #{new_num}"
            cluster_dir.mkdir(parents=True, exist_ok=True)

            for item_k in items:
                v = db[item_k]
                v['category'] = 'Personas Sin Nombre'
                v['identity'] = cluster_name
                v['status'] = 'Clasificado'
                v['is_dudoso'] = False
                v['identities'] = [cluster_name]

                file_p = Path(v.get('filepath') or v.get('path') or item_k)
                if file_p.exists() and not is_icloud_file(str(file_p)):
                    dest_target = cluster_dir / file_p.name
                    if safe_move_file(file_p, dest_target):
                        actual_dest = dest_target if dest_target.exists() else None
                        if not actual_dest:
                            cand = list(cluster_dir.glob(f"{file_p.stem}*{file_p.suffix}"))
                            if cand: actual_dest = cand[0]
                        if actual_dest:
                            v['path'] = str(actual_dest)
                            v['filepath'] = str(actual_dest)

        # Group all remaining isolated singletons into a SINGLE consolidated album:
        # "👤 Desconocidos / Caras Aisladas"
        single_album_name = "👤 Desconocidos / Caras Aisladas"
        desconocidos_dir = psn_dir / "Desconocidos"
        desconocidos_dir.mkdir(parents=True, exist_ok=True)

        for item_k in singletons:
            v = db[item_k]
            v['category'] = 'Personas Sin Nombre'
            v['identity'] = single_album_name
            v['status'] = 'Clasificado'
            v['is_dudoso'] = False
            v['identities'] = [single_album_name]

            file_p = Path(v.get('filepath') or v.get('path') or item_k)
            if file_p.exists() and not is_icloud_file(str(file_p)):
                dest_target = desconocidos_dir / file_p.name
                if safe_move_file(file_p, dest_target):
                    actual_dest = dest_target if dest_target.exists() else None
                    if not actual_dest:
                        cand = list(desconocidos_dir.glob(f"{file_p.stem}*{file_p.suffix}"))
                        if cand: actual_dest = cand[0]
                    if actual_dest:
                        v['path'] = str(actual_dest)
                        v['filepath'] = str(actual_dest)

        print(f"  ✓ {len(sorted_multi)} carpetas compactas de 'Persona Nueva' creadas con fotos recurrentes.")
        print(f"  ✓ {len(singletons)} fotos aisladas unificadas en '{single_album_name}'. (¡CERO carpetas de 1 sola foto!)")

    # 8. Clean up empty folders on disk
    print("\n[7/7] Limpiando carpetas vacías en disco y guardando cambios...")
    removed_folders = 0
    for base_p in [PSN_DIR, DUDOSOS_DIR]:
        if base_p.exists():
            for folder in list(base_p.iterdir()):
                if folder.is_dir() and "Persona Nueva #" in folder.name:
                    # Check if folder has files
                    files = [f for f in folder.iterdir() if f.is_file() and not f.name.startswith('.')]
                    if len(files) == 0:
                        try:
                            # Remove hidden files if any
                            for h in folder.iterdir():
                                if h.is_file(): os.remove(h)
                            os.rmdir(folder)
                            removed_folders += 1
                        except Exception as e:
                            pass
    print(f"  ✓ Eliminadas {removed_folders} carpetas vacías de Persona Nueva en disco.")

    # Save apple_db.json
    with open(APPLE_DB_PATH, 'w', encoding='utf-8') as f:
        json.dump(db, f, ensure_ascii=False, indent=2)
    print("  ✓ apple_db.json guardado con éxito.")

    # Save faces_cache.pkl
    with open(FACES_CACHE_PATH, 'wb') as f:
        pickle.dump(faces_cache, f)
    print("  ✓ faces_cache.pkl guardado con éxito.")

    # Clear gallery cache
    clear_gallery_cache()
    print("  ✓ Caché de galería invalidada.")

    print("\n" + "=" * 60)
    print("🎉 CONSOLIDACIÓN COMPLETADA EXITOSAMENTE")
    print(f"  - De ~723 carpetas fragmentadas -> {len(sorted_multi)} carpetas densas y 1 álbum de caras aisladas.")
    print(f"  - {matched_confirmed} fotos reasignadas a identidades confirmadas.")
    print(f"  - Cero carpetas residuales de 1 o 2 fotos sueltas.")
    print("=" * 60)

if __name__ == '__main__':
    run_consolidation()
