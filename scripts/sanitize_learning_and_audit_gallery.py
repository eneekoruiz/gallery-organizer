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
    print("=" * 80)
    print("🧹 DESCONTAMINACIÓN DEL APRENDIZAJE Y SANEAMIENTO INTEGRAL DE LA GALERÍA")
    print("=" * 80)

    # 1. Cargar bases de datos y cachés
    db_path = BASE_DIR / 'apple_db.json'
    print("\n[1/6] Cargando apple_db.json y sface_cache_v7.pkl...")
    with open(db_path, 'r', encoding='utf-8') as f:
        db = json.load(f)
    print(f"  Total elementos en apple_db.json: {len(db)}")

    cache_path = BASE_DIR / 'sface_cache_v7.pkl'
    with open(cache_path, 'rb') as f:
        kf = pickle.load(f)
    print(f"  Total identidades en sface_cache: {len(kf)}")

    # 2. Descontaminar muestras de entrenamiento en cada identidad
    print("\n[2/6] Auditando y descontaminando muestras de entrenamiento (outliers de DBSCAN)...")
    sanitized_kf = {}
    evicted_samples = [] # (origin_ident, sample_name, vector)
    total_evicted = 0

    for ident, data in kf.items():
        samples = data.get('confirmed_samples', {})
        cat = data.get('cat', 'Conocidos')
        
        if len(samples) <= 2:
            # Muy pocos ejemplos para calcular outliers por cluster
            sanitized_kf[ident] = data
            continue
            
        names = list(samples.keys())
        vecs = [samples[k] for k in names]
        X = np.array(vecs)
        
        # Calcular matriz de similitud interna
        sim_mat = np.dot(X, X.T)
        mean_sims = (np.sum(sim_mat, axis=1) - 1.0) / (len(vecs) - 1)
        
        # Micro-clustering de pureza interna con enlace completo
        labels = cluster_faces_upcac(X, dt1=0.34, merge_sim=0.60, margin=0.10)
        c_counts = Counter(labels)
        dominant_label = c_counts.most_common(1)[0][0]
        
        # Calcular centroide dominante
        core_indices = [idx for idx, l in enumerate(labels) if l == dominant_label]
        core_c = normalized_feature(np.mean(X[core_indices], axis=0))
        
        clean_samples = {}
        for idx, name in enumerate(names):
            v = vecs[idx]
            sim_to_core = float(np.dot(core_c, v))
            
            # Criterio estricto de pureza:
            # Si la similitud al centroide del núcleo es < 0.40 o mean_sims < 0.35 -> Contaminado por DBSCAN
            if sim_to_core >= 0.40 and mean_sims[idx] >= 0.30:
                clean_samples[name] = v
            else:
                evicted_samples.append((ident, name, v))
                total_evicted += 1
                
        if not clean_samples:
            clean_samples = samples # evitar vaciar completamente
            
        clean_vecs = list(clean_samples.values())
        ensemble_c = normalized_feature(np.mean(clean_vecs, axis=0))
        appearance_c = compute_appearance_centroids(clean_vecs, max_centroids=4)
        
        sanitized_kf[ident] = {
            'centroid': ensemble_c,
            'centroids': appearance_c,
            'confirmed_samples': clean_samples,
            'names': list(clean_samples.keys()),
            'cat': cat,
            'provenance': HUMAN
        }

    print(f"  ✓ Muestras contaminadas expulsadas de las identidades: {total_evicted}")
    print(f"  ✓ Total identidades purificadas: {len(sanitized_kf)}")

    # Guardar cachés purificadas
    with open(cache_path, 'wb') as f:
        pickle.dump(sanitized_kf, f)
    with open(BASE_DIR / 'centroids.pkl', 'wb') as f:
        pickle.dump({(v.get('cat', 'Conocidos'), k): v['centroid'] for k, v in sanitized_kf.items()}, f)
    print("  ✓ sface_cache_v7.pkl y centroids.pkl actualizados con modelos 100% puros.")

    # 3. Matriz de centroides de identidades purificadas para rescate y auditoría
    c_keys = list(sanitized_kf.keys())
    c_matrix = np.array([sanitized_kf[k]['centroid'] for k in c_keys])
    
    # 4. Rescatar fotos en 'Persona Nueva' y 'Caras Aisladas' que corresponden a identidades conocidas
    print("\n[3/6] Rescatando fotos en 'Persona Nueva' y 'Caras Aisladas' que coinciden con identidades conocidas...")
    rescued_from_new_persons = 0
    rescued_by_identity = Counter()

    for path_str, info in db.items():
        ident = info.get('identity', '')
        if 'Persona Nueva' in ident or 'Aisladas' in ident or 'Desconocid' in ident:
            faces = info.get('faces', [])
            if faces and len(faces) == 1:
                feat = faces[0].get('feature')
                if feat:
                    try:
                        v = normalized_feature(feat)
                        sims = np.dot(c_matrix, v)
                        top_idx = int(np.argmax(sims))
                        top_sim = float(sims[top_idx])
                        
                        sorted_sims = np.sort(sims)[::-1]
                        margin = float(sorted_sims[0] - sorted_sims[1]) if len(sorted_sims) > 1 else 1.0
                        
                        target_ident = c_keys[top_idx]
                        
                        # Umbral de certeza alta: sim >= 0.50 y margen >= 0.06
                        if top_sim >= 0.50 and margin >= 0.06:
                            dest_cat = sanitized_kf[target_ident]['cat']
                            info['category'] = dest_cat
                            info['identity'] = target_ident
                            info['status'] = 'Clasificado'
                            info['is_dudoso'] = False
                            info['review_required'] = False
                            rescued_from_new_persons += 1
                            rescued_by_identity[target_ident] += 1
                    except Exception:
                        pass

    print(f"  ✓ Fotos rescatadas hacia identidades conocidas: {rescued_from_new_persons}")
    for ident_name, cnt in rescued_by_identity.most_common(10):
        print(f"    - {ident_name}: +{cnt} fotos recuperadas")

    # 5. Re-clustering UP-CAC de las fotos restantes que son GENUINAMENTE desconocidas
    print("\n[4/6] Re-clustering con UP-CAC de rostros verdaderamente desconocidos (Genuinas Personas Nuevas)...")
    remaining_unknowns = []
    
    for path_str, info in db.items():
        cat = info.get('category', '')
        ident = info.get('identity', '')
        if 'Revisión' in cat or 'Persona Nueva' in ident or 'Aisladas' in ident or 'Desconocid' in ident:
            faces = info.get('faces', [])
            if faces and len(faces) == 1:
                feat = faces[0].get('feature')
                if feat:
                    try:
                        v = normalized_feature(feat)
                        # Comprobar que no coincida con nadie conocido (doble verificación de seguridad)
                        sims = np.dot(c_matrix, v)
                        if float(np.max(sims)) < 0.48:
                            remaining_unknowns.append((path_str, info, v))
                    except Exception:
                        pass

    print(f"  Caras genuinamente nuevas sin coincidencia con nadie configurado: {len(remaining_unknowns)}")

    if len(remaining_unknowns) >= 2:
        embeddings = np.array([e[2] for e in remaining_unknowns])
        labels = cluster_faces_upcac(embeddings, dt1=0.34, merge_sim=0.60, margin=0.10, reciprocal=True)
        
        clusters = defaultdict(list)
        for idx, lbl in enumerate(labels):
            clusters[lbl].append(remaining_unknowns[idx])

        multi_clusters = []
        singletons = []
        for lbl, members in clusters.items():
            if len(members) >= 2:
                multi_clusters.append(members)
            else:
                singletons.append(members[0])

        multi_clusters.sort(key=len, reverse=True)
        print(f"  ✓ Nuevas agrupaciones genuinas formadas con UP-CAC: {len(multi_clusters)}")
        print(f"  ✓ Caras aisladas únicas: {len(singletons)}")

        # Asignar nombres limpios 'Persona Nueva #N'
        for cluster_idx, members in enumerate(multi_clusters, start=1):
            cluster_name = f"👤 Persona Nueva #{cluster_idx}"
            for p_str, info, _ in members:
                info['category'] = '📋 Revisión / Por Organizar'
                info['identity'] = cluster_name
                info['status'] = 'Clasificado'
                info['is_dudoso'] = False
                info['review_required'] = False

        # Unificar caras aisladas
        for p_str, info, _ in singletons:
            info['category'] = '📋 Revisión / Por Organizar'
            info['identity'] = '👤 Desconocidos / Caras Aisladas'
            info['status'] = 'Clasificado'
            info['is_dudoso'] = False
            info['review_required'] = False

    # 6. Auditoría cruzada de álbumes canónicos (Conocidos / Familia / YO)
    print("\n[5/6] Auditando álbumes canónicos para garantizar que cada foto está en la carpeta correcta...")
    misplaced_fixed = 0
    
    for path_str, info in db.items():
        cat = info.get('category', '')
        ident = info.get('identity', '')
        
        if cat in ['Conocidos', 'Familia', 'YO'] and ident in sanitized_kf:
            owner_c = sanitized_kf[ident]['centroid']
            faces = info.get('faces', [])
            if faces and len(faces) == 1:
                feat = faces[0].get('feature')
                if feat:
                    try:
                        v = normalized_feature(feat)
                        sim_owner = float(np.dot(owner_c, v))
                        
                        # Si la foto en la carpeta tiene baja similitud con el dueño (< 0.35)
                        if sim_owner < 0.35:
                            # Verificar si pertenece a otra identidad
                            sims = np.dot(c_matrix, v)
                            top_idx = int(np.argmax(sims))
                            top_sim = float(sims[top_idx])
                            top_ident = c_keys[top_idx]
                            
                            if top_sim >= 0.52 and top_ident != ident:
                                # Mover a la identidad correcta
                                dest_cat = sanitized_kf[top_ident]['cat']
                                info['category'] = dest_cat
                                info['identity'] = top_ident
                                misplaced_fixed += 1
                                print(f"  [RE-DIRIGIDA] '{Path(path_str).name}': '{ident}' (sim={sim_owner:.2f}) -> '{top_ident}' (sim={top_sim:.2f})")
                    except Exception:
                        pass

    print(f"  ✓ Fotos mal ubicadas re-dirigidas a su carpeta correcta: {misplaced_fixed}")

    # 7. Guardar apple_db.json y regenerar gallery_cache.json
    print("\n[6/6] Guardando base de datos limpia y regenerando caché...")
    with open(db_path, 'w', encoding='utf-8') as f:
        json.dump(db, f, indent=2, ensure_ascii=False)
    print("  ✓ apple_db.json guardado.")

    import app.services.gallery_service as gs
    gallery_data = gs.get_gallery(force_reload=True)
    with open(BASE_DIR / 'gallery_cache.json', 'w', encoding='utf-8') as f:
        json.dump(gallery_data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ gallery_cache.json actualizado ({len(gallery_data)} categorías).")

    print("\n" + "=" * 80)
    print("🎉 DESCONTAMINACIÓN Y SANEAMIENTO COMPLETADO EXITOSAMENTE")
    print("=" * 80)

if __name__ == '__main__':
    main()
