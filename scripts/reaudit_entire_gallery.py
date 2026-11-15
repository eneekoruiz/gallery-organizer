import sys
import json
import pickle
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.utils.clustering import cluster_faces_consensus_upcac, consolidate_new_persons_upcac

def main():
    print("=" * 80)
    print("🔍 RE-AUDITORÍA INTEGRAL DE LA FOTOTECA: ENSEMBLE EC-UP-CAC + CANNOT-LINK")
    print("=" * 80)

    db_path = BASE_DIR / 'apple_db.json'
    with open(db_path, 'r', encoding='utf-8') as f:
        db = json.load(f)

    cache_path = BASE_DIR / 'centroids.pkl'
    with open(cache_path, 'rb') as f:
        centroids_dict = pickle.load(f)

    known_keys = []
    known_idents = []
    known_vectors = []
    for k, v in centroids_dict.items():
        k_name = k[1] if isinstance(k, tuple) else str(k)
        if not k_name.startswith("Persona Nueva") and "Desconocid" not in k_name:
            known_keys.append(k)
            known_idents.append(k_name)
            norm_v = np.array(v, dtype=np.float32).flatten()
            norm_v = norm_v / (np.linalg.norm(norm_v) + 1e-10)
            known_vectors.append(norm_v)

    c_matrix = np.array(known_vectors, dtype=np.float32)
    print(f"\n[1/4] Cargadas {len(known_idents)} identidades canónicas de referencia.")

    # -------------------------------------------------------------
    # PASO 1: Resolver violaciones Cannot-Link en fotos grupales
    # -------------------------------------------------------------
    print("\n[2/4] Auditando y resolviendo colisiones Cannot-Link en fotos grupales...")
    cl_resolved = 0
    reassigned_alternatives = 0

    for path, info in db.items():
        faces = info.get('faces', [])
        if len(faces) <= 1:
            continue

        # Agrupar rostros por identidad asignada
        ident_faces = defaultdict(list)
        for idx, f in enumerate(faces):
            f_id = f.get('identity', '')
            if f_id and not any(t in f_id.lower() for t in ['desconocid', 'sin rostro', 'ignorar', 'dudoso']):
                ident_faces[f_id].append((idx, f))

        for f_id, f_list in ident_faces.items():
            if len(f_list) > 1:
                # ¡COLISIÓN CANNOT-LINK! Múltiples rostros con la misma identidad en la misma foto.
                # Evaluar similitud exacta de cada rostro con el centroide de f_id
                target_idx = None
                for i_k, name in enumerate(known_idents):
                    if name == f_id:
                        target_idx = i_k
                        break

                evaluated = []
                for idx, f in f_list:
                    feat = f.get('feature')
                    if feat is not None:
                        vec = np.array(feat, dtype=np.float32).flatten()
                        vec = vec / (np.linalg.norm(vec) + 1e-10)
                        sim = float(np.dot(c_matrix[target_idx], vec)) if target_idx is not None else 0.50
                        evaluated.append((sim, idx, f, vec))
                    else:
                        evaluated.append((-1.0, idx, f, None))

                # Ordenar por similitud decreciente: el mejor se queda con la identidad
                evaluated.sort(key=lambda x: x[0], reverse=True)
                winner = evaluated[0]
                colliders = evaluated[1:]

                for sim_c, idx, f, vec in colliders:
                    cl_resolved += 1
                    if vec is not None:
                        # Buscar la mejor alternativa excluyendo la identidad en conflicto
                        sims = np.dot(c_matrix, vec)
                        if target_idx is not None:
                            sims[target_idx] = -1.0 # Excluir la identidad duplicada
                        best_alt_idx = int(np.argmax(sims))
                        best_alt_sim = float(sims[best_alt_idx])
                        best_alt_name = known_idents[best_alt_idx]

                        if best_alt_sim >= 0.63:
                            # Reasignar a la alternativa legítima
                            f['identity'] = best_alt_name
                            f['confidence'] = round(best_alt_sim * 100, 1)
                            reassigned_alternatives += 1
                        else:
                            # Cara desconocida legítima: marcar como nueva persona aislada
                            f['identity'] = f"👤 Persona Nueva (Foto #{cl_resolved})"
                            f['confidence'] = "No identificado"
                    else:
                        f['identity'] = "Desconocido"

    print(f"  ✓ {cl_resolved} colisiones Cannot-Link resueltas con éxito.")
    print(f"  ✓ {reassigned_alternatives} rostros reasignados a su segunda persona legítima.")

    # -------------------------------------------------------------
    # PASO 2: Re-auditoría cruzada de todas las fotos clasificadas
    # -------------------------------------------------------------
    print("\n[3/4] Re-auditoría cruzada de todas las fotos frente a centroides canónicos...")
    cross_fixed = 0
    total_audited = 0

    for path, info in db.items():
        ident = info.get('identity', '')
        cat = info.get('category', '')
        faces = info.get('faces', [])

        if cat in ['Conocidos', 'Familia', 'YO'] and ident and len(faces) == 1:
            total_audited += 1
            f = faces[0]
            feat = f.get('feature')
            if feat is not None and not f.get('is_manual'):
                vec = np.array(feat, dtype=np.float32).flatten()
                vec = vec / (np.linalg.norm(vec) + 1e-10)
                sims = np.dot(c_matrix, vec)
                top_idx = int(np.argmax(sims))
                top_sim = float(sims[top_idx])
                top_name = known_idents[top_idx]

                # Similitud a la identidad actual
                cur_idx = None
                for ik, name in enumerate(known_idents):
                    if name == ident:
                        cur_idx = ik
                        break
                cur_sim = float(sims[cur_idx]) if cur_idx is not None else 0.0

                # Si otra identidad supera a la actual por margen contundente >= 0.08
                if top_name != ident and (top_sim - cur_sim) >= 0.08 and top_sim >= 0.65:
                    dest_cat = "Familia" if top_name.startswith(("F.", "F ")) else ("YO" if top_name == "YO" else "Conocidos")
                    info['identity'] = top_name
                    info['category'] = dest_cat
                    f['identity'] = top_name
                    f['category'] = dest_cat
                    f['confidence'] = round(top_sim * 100, 1)
                    cross_fixed += 1

    print(f"  ✓ {total_audited} fotos auditadas. {cross_fixed} correcciones de clasificación cruzada aplicadas.")

    # -------------------------------------------------------------
    # PASO 3: Consolidación EC-UP-CAC de Personas Nuevas
    # -------------------------------------------------------------
    print("\n[4/4] Ejecutando Ensemble Consensus UP-CAC (EC-UP-CAC) para Personas Nuevas...")
    res_upcac = consolidate_new_persons_upcac(db, centroids_dict)
    print(f"  ✓ EC-UP-CAC completado: {res_upcac['consolidated_unknowns']} fotos agrupadas en {res_upcac['cohesive_groups']} grupos sin falsas fusiones.")

    # Guardar apple_db.json
    print("\nSincronizando apple_db.json y regenerando galería...")
    with open(db_path, 'w', encoding='utf-8') as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

    import app.services.gallery_service as gs
    gallery_data = gs.get_gallery(force_reload=True)
    with open(BASE_DIR / 'gallery_cache.json', 'w', encoding='utf-8') as f:
        json.dump(gallery_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("🎉 RE-AUDITORÍA TOTAL DE LA FOTOTECA COMPLETADA AL 100%")
    print("=" * 80)

if __name__ == '__main__':
    main()
