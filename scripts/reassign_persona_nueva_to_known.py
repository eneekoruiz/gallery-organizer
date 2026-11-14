import sys
import json
import pickle
from pathlib import Path
import numpy as np

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.utils.clustering import cluster_faces_upcac

def main():
    print("=" * 80)
    print("🎯 REASIGNACIÓN DE PERSONA NUEVA -> IDENTIDADES CONOCIDAS + CLUSTERING UP-CAC")
    print("=" * 80)

    db_path = BASE_DIR / 'apple_db.json'
    with open(db_path, 'r', encoding='utf-8') as f:
        db = json.load(f)

    cache_path = BASE_DIR / 'sface_cache_v7.pkl'
    with open(cache_path, 'rb') as f:
        kf = pickle.load(f)

    c_keys = [k for k in kf.keys() if 'Persona Nueva' not in k]
    c_matrix = np.array([kf[k]['centroid'] for k in c_keys], dtype=np.float32)

    # 1. Reasignar a conocidos
    reassigned = 0
    remaining_pn_paths = []
    remaining_pn_vectors = []

    for path, info in db.items():
        ident = str(info.get('identity', ''))
        if 'Persona Nueva' in ident:
            faces = info.get('faces', [])
            matched = False
            best_match_info = None

            for face in faces:
                feat = face.get('feature')
                if feat is not None:
                    vec = np.array(feat, dtype=np.float32)
                    norm = np.linalg.norm(vec)
                    if norm > 1e-6:
                        vec = vec / norm
                        sims = np.dot(c_matrix, vec)
                        top_idx = int(np.argmax(sims))
                        top_sim = float(sims[top_idx])
                        top_ident = c_keys[top_idx]

                        if top_sim >= 0.63:
                            matched = True
                            best_match_info = (top_ident, top_sim, face)
                            break
                        elif not best_match_info or top_sim > best_match_info[1]:
                            best_match_info = (top_ident, top_sim, face)

            if matched and best_match_info:
                top_ident, top_sim, face = best_match_info
                dest_cat = kf[top_ident].get('cat', 'Conocidos')
                old_ident = info.get('identity')
                info['identity'] = top_ident
                info['category'] = dest_cat
                info['status'] = 'Clasificado'
                info['is_dudoso'] = False
                info['review_required'] = False
                face['identity'] = top_ident
                face['category'] = dest_cat
                reassigned += 1
                print(f"  [REASIGNADA] {Path(path).name} : {old_ident} -> {top_ident} (sim={top_sim:.3f})")
            else:
                # No pertenece a ningún conocido: guardar para clustering UP-CAC
                if best_match_info and best_match_info[2].get('feature') is not None:
                    feat = best_match_info[2].get('feature')
                    v = np.array(feat, dtype=np.float32)
                    v = v / (np.linalg.norm(v) + 1e-6)
                    remaining_pn_paths.append(path)
                    remaining_pn_vectors.append(v)

    print(f"\n[1/3] Fotos de 'Persona Nueva' rescatadas y reasignadas a conocidos: {reassigned}")
    print(f"  Fotos restantes de personas realmente desconocidas: {len(remaining_pn_paths)}")

    # 2. Agrupar las personas desconocidas restantes con el algoritmo ganador UP-CAC
    if len(remaining_pn_vectors) > 1:
        print("\n[2/3] Agrupando rostros no identificados con UP-CAC...")
        X = np.array(remaining_pn_vectors)
        # cluster_faces_upcac para agrupar rostros desconocidos en grupos consistentes
        labels = cluster_faces_upcac(X, dt1=0.38, merge_sim=0.62, margin=0.08, reciprocal=True)
        
        # Asignar nombres limpios '👤 Persona Nueva #1', etc.
        cluster_map = {}
        curr_id = 1
        for lab in labels:
            if lab not in cluster_map:
                cluster_map[lab] = f"👤 Persona Nueva #{curr_id}"
                curr_id += 1

        updated_count = 0
        for path, lab in zip(remaining_pn_paths, labels):
            new_name = cluster_map[lab]
            db[path]['identity'] = new_name
            db[path]['category'] = 'Personas Nuevas'
            db[path]['status'] = 'Clasificado'
            for f in db[path].get('faces', []):
                f['identity'] = new_name
                f['category'] = 'Personas Nuevas'
            updated_count += 1
        print(f"  ✓ {updated_count} fotos agrupadas en {len(cluster_map)} grupos cohesivos con UP-CAC.")

    # 3. Guardar base de datos y sincronizar caché
    print("\n[3/3] Guardando base de datos y regenerando galería...")
    with open(db_path, 'w', encoding='utf-8') as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

    import app.services.gallery_service as gs
    gallery_data = gs.get_gallery(force_reload=True)
    with open(BASE_DIR / 'gallery_cache.json', 'w', encoding='utf-8') as f:
        json.dump(gallery_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("✅ REASIGNACIÓN Y CLUSTERING UP-CAC COMPLETADOS CON ÉXITO")
    print("=" * 80)

if __name__ == '__main__':
    main()
