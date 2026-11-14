import sys
import os
import time
import json
import pickle
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.utils.clustering import cluster_faces_upcac
from app.services.face_learning import normalized_feature, compute_appearance_centroids, HUMAN

def main():
    print("=" * 70)
    print("🚀 RECALCULANDO Y APLICANDO APRENDIZAJE FACIAL CON ALGORITMO UP-CAC")
    print("=" * 70)

    # 1. Cargar base de datos
    db_path = BASE_DIR / 'apple_db.json'
    print("\n[1/5] Cargando apple_db.json...")
    with open(db_path, 'r', encoding='utf-8') as f:
        db = json.load(f)
    print(f"  Total elementos en base de datos: {len(db)}")

    # 2. Recalcular centroides y perfiles de identidades confirmadas
    print("\n[2/5] Recalculando centroides limpios con modelado UP-CAC para identidades conocidas...")
    ident_samples = defaultdict(dict)
    
    for path_str, info in db.items():
        cat = info.get('category', '')
        ident = info.get('identity', '')
        if cat in ['Conocidos', 'Familia', 'YO'] and ident and not any(term in ident.lower() for term in ['persona nueva', 'desconocid', 'sin rostro', 'ignorar', 'dudoso']):
            faces = info.get('faces', [])
            if faces and len(faces) == 1:
                feat = faces[0].get('feature')
                if feat:
                    try:
                        norm_v = normalized_feature(feat)
                        fname = Path(path_str).name
                        ident_samples[ident][fname] = norm_v
                    except Exception:
                        pass

    print(f"  Identidades confirmadas encontradas: {len(ident_samples)}")
    new_known_faces = {}
    total_samples = 0

    for ident, samples in ident_samples.items():
        sample_list = list(samples.values())
        total_samples += len(sample_list)
        
        # Filtro de outliers para identidades con >= 3 fotos
        if len(sample_list) >= 3:
            raw_c = normalized_feature(np.mean(sample_list, axis=0))
            clean_samples = [s for s in sample_list if float(np.dot(raw_c, s)) >= 0.35]
            if not clean_samples:
                clean_samples = sample_list
        else:
            clean_samples = sample_list
            
        ensemble_c = normalized_feature(np.mean(clean_samples, axis=0))
        appearance_c = compute_appearance_centroids(clean_samples, max_centroids=4)
        
        # Determinar categoría
        cat = 'YO' if ident == 'YO' else ('Familia' if ident.startswith('F.') else 'Conocidos')
        
        new_known_faces[ident] = {
            'centroid': ensemble_c,
            'centroids': appearance_c,
            'confirmed_samples': samples,
            'names': list(samples.keys()),
            'cat': cat,
            'provenance': HUMAN
        }

    # Guardar en sface_cache_v7.pkl
    cache_file = BASE_DIR / "sface_cache_v7.pkl"
    with open(cache_file, 'wb') as f:
        pickle.dump(new_known_faces, f)
    print(f"  ✓ sface_cache_v7.pkl actualizado ({len(new_known_faces)} personas, {total_samples} fotos verificadas).")

    # Guardar en centroids.pkl
    centroids_dict = {}
    for ident, data in new_known_faces.items():
        cat = data.get('cat', 'Conocidos')
        centroids_dict[(cat, ident)] = data['centroid']
    with open(BASE_DIR / 'centroids.pkl', 'wb') as f:
        pickle.dump(centroids_dict, f)
    print(f"  ✓ centroids.pkl actualizado ({len(centroids_dict)} identidades).")

    # 3. Re-clustering UP-CAC para fotos en Revisión / Persona Nueva
    print("\n[3/5] Aplicando algoritmo UP-CAC para re-agrupar caras desconocidas/por organizar...")
    unknown_entries = []
    
    for path_str, info in db.items():
        cat = info.get('category', '')
        ident = info.get('identity', '')
        # Candidatos: personas nuevas o desconocidos con 1 rostro detectable
        if 'Revisión' in cat or 'Persona Nueva' in ident or 'Desconocid' in ident or 'Aisladas' in ident:
            faces = info.get('faces', [])
            if faces and len(faces) == 1:
                feat = faces[0].get('feature')
                if feat:
                    try:
                        v = normalized_feature(feat)
                        unknown_entries.append((path_str, info, v))
                    except Exception:
                        pass

    print(f"  Caras no identificadas para agrupar: {len(unknown_entries)}")

    if len(unknown_entries) >= 2:
        embeddings = np.array([e[2] for e in unknown_entries])
        print("  Ejecutando UP-CAC (dt1=0.34, merge=0.60, margin=0.10, reciprocal=True)...")
        labels = cluster_faces_upcac(embeddings, dt1=0.34, merge_sim=0.60, margin=0.10, reciprocal=True)
        
        # Agrupar por cluster
        clusters = defaultdict(list)
        for idx, lbl in enumerate(labels):
            clusters[lbl].append(unknown_entries[idx])

        # Clusters multi-foto (>= 2 fotos) vs Singletons (1 foto)
        multi_clusters = []
        singletons = []
        for lbl, members in clusters.items():
            if len(members) >= 2:
                multi_clusters.append(members)
            else:
                singletons.append(members[0])

        # Ordenar multi-clusters por tamaño descendente
        multi_clusters.sort(key=len, reverse=True)
        print(f"  ✓ Clusters multi-foto formados con alta pureza: {len(multi_clusters)}")
        print(f"  ✓ Fotos aisladas/desconocidos únicos: {len(singletons)}")

        # Asignar nombres limpios 'Persona Nueva #N' a clusters densos
        updated_in_db = 0
        for cluster_idx, members in enumerate(multi_clusters, start=1):
            cluster_name = f"👤 Persona Nueva #{cluster_idx}"
            for p_str, info, _ in members:
                info['category'] = '📋 Revisión / Por Organizar'
                info['identity'] = cluster_name
                info['status'] = 'Clasificado'
                info['is_dudoso'] = False
                info['review_required'] = False
                updated_in_db += 1

        # Todas las caras aisladas (singletons) van al álbum unificado
        for p_str, info, _ in singletons:
            info['category'] = '📋 Revisión / Por Organizar'
            info['identity'] = '👤 Desconocidos / Caras Aisladas'
            info['status'] = 'Clasificado'
            info['is_dudoso'] = False
            info['review_required'] = False
            updated_in_db += 1

        print(f"  ✓ Asignadas {updated_in_db} fotos en {len(multi_clusters)} personas nuevas compactas + 1 álbum de caras aisladas.")

    # 4. Auditoría estricta de 'Fotos Sin Rostro'
    print("\n[4/5] Auditando 'Fotos Sin Rostro' y eliminando dudosas/botones innecesarios...")
    sin_rostro_fixed = 0
    sin_rostro_with_faces = 0

    # Construir matriz de centroides conocidos para rescate
    c_keys = list(new_known_faces.keys())
    c_matrix = np.array([new_known_faces[k]['centroid'] for k in c_keys]) if c_keys else None

    for path_str, info in db.items():
        ident = info.get('identity', '')
        if 'sin rostro' in ident.lower():
            # Invariante 1: CERO dudosas en Sin Rostro
            if info.get('status') == 'Dudoso' or info.get('is_dudoso') or info.get('review_required'):
                info['status'] = 'Clasificado'
                info['is_dudoso'] = False
                info['review_required'] = False
                sin_rostro_fixed += 1

            # Invariante 2: Si tiene caras reales, verificar y reclasificar
            faces = info.get('faces', [])
            if faces and len(faces) == 1 and c_matrix is not None:
                feat = faces[0].get('feature')
                if feat:
                    try:
                        v = normalized_feature(feat)
                        sims = np.dot(c_matrix, v)
                        top_idx = int(np.argmax(sims))
                        top_sim = float(sims[top_idx])
                        top_ident = c_keys[top_idx]
                        if top_sim >= 0.50:
                            # Rescatar a la identidad correcta!
                            dest_cat = new_known_faces[top_ident]['cat']
                            info['category'] = dest_cat
                            info['identity'] = top_ident
                            info['status'] = 'Clasificado'
                            info['is_dudoso'] = False
                            info['review_required'] = False
                            sin_rostro_with_faces += 1
                    except Exception:
                        pass

    print(f"  ✓ Fotos Sin Rostro corregidas de 'Dudoso' a 'Clasificado': {sin_rostro_fixed}")
    print(f"  ✓ Fotos con rostro rescatadas de Sin Rostro a identidades reconocidas: {sin_rostro_with_faces}")

    # Guardar cambios en apple_db.json
    print("\n[5/5] Guardando base de datos y regenerando galería...")
    with open(db_path, 'w', encoding='utf-8') as f:
        json.dump(db, f, indent=2, ensure_ascii=False)
    print("  ✓ apple_db.json guardado con éxito.")

    # Reconstruir gallery_cache.json
    import app.services.gallery_service as gs
    print("  Regenerando gallery_cache.json...")
    gallery_data = gs.get_gallery(force_reload=True)
    with open(BASE_DIR / 'gallery_cache.json', 'w', encoding='utf-8') as f:
        json.dump(gallery_data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ gallery_cache.json regenerado ({len(gallery_data)} categorías).")

    print("\n" + "=" * 70)
    print("🎉 APRENDIZAJE RECALCULADO Y UP-CAC APLICADO EN TODA LA GALERÍA CON ÉXITO")
    print("=" * 70)

if __name__ == '__main__':
    main()
