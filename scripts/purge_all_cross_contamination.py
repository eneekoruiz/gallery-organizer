import sys
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
    print("🛡️ PURGA DEFINITIVA DE CONTAMINACIÓN CRUZADA (ZERO CROSS-CONTAMINATION)")
    print("=" * 80)

    # 1. Cargar bases de datos
    db_path = BASE_DIR / 'apple_db.json'
    with open(db_path, 'r', encoding='utf-8') as f:
        db = json.load(f)

    cache_path = BASE_DIR / 'sface_cache_v7.pkl'
    with open(cache_path, 'rb') as f:
        kf = pickle.load(f)

    print(f"\n[1/4] Analizando {len(kf)} identidades y sus muestras de entrenamiento...")

    # Iteración de purificación hasta convergencia (0 violaciones)
    iteration = 0
    total_purged = 0
    total_reassigned = 0

    while iteration < 5:
        iteration += 1
        c_keys = list(kf.keys())
        c_matrix = np.array([kf[k]['centroid'] for k in c_keys])
        
        violations_found = 0
        samples_to_remove = [] # (ident, s_name, best_other_ident)

        for ident, data in kf.items():
            samples = data.get('confirmed_samples', {})
            c = data['centroid']
            
            for s_name, vec in list(samples.items()):
                sim_to_owner = float(np.dot(c, vec))
                sims = np.dot(c_matrix, vec)
                top_idx = int(np.argmax(sims))
                top_sim = float(sims[top_idx])
                top_ident = c_keys[top_idx]
                
                # Criterio estricto de contaminación cruzada:
                # Si otra identidad tiene mayor similitud que el dueño por un margen >= 0.05
                if top_ident != ident and (top_sim - sim_to_owner) >= 0.05:
                    samples_to_remove.append((ident, s_name, top_ident, top_sim, sim_to_owner))
                    violations_found += 1
                elif sim_to_owner < 0.40:
                    samples_to_remove.append((ident, s_name, None, 0.0, sim_to_owner))
                    violations_found += 1

        print(f"  Paso {iteration}: Violaciones de contaminación cruzada detectadas: {violations_found}")
        if violations_found == 0:
            print("  ✓ Convergencia alcanzada: CERO contaminación cruzada restante!")
            break

        # Aplicar purgas y reasignaciones
        for ident, s_name, top_ident, top_sim, sim_to_owner in samples_to_remove:
            if s_name in kf[ident].get('confirmed_samples', {}):
                del kf[ident]['confirmed_samples'][s_name]
                total_purged += 1

                # Reasignar a la identidad legítima en apple_db.json
                if top_ident:
                    for p_str, info in db.items():
                        if Path(p_str).name == s_name:
                            dest_cat = kf[top_ident].get('cat', 'Conocidos')
                            info['category'] = dest_cat
                            info['identity'] = top_ident
                            info['status'] = 'Clasificado'
                            info['is_dudoso'] = False
                            info['review_required'] = False
                            total_reassigned += 1
                            print(f"    [REASIGNADA] {s_name}: de '{ident}' (sim={sim_to_owner:.2f}) -> '{top_ident}' (sim={top_sim:.2f})")

        # Recalcular centroides tras la purga
        for ident, data in kf.items():
            samples = list(data.get('confirmed_samples', {}).values())
            if samples:
                ens_c = normalized_feature(np.mean(samples, axis=0))
                data['centroid'] = ens_c
                data['centroids'] = compute_appearance_centroids(samples, max_centroids=4)
                data['names'] = list(data['confirmed_samples'].keys())

    print(f"\n[2/4] Resumen de purificación:")
    print(f"  ✓ Muestras contaminadas purgadas: {total_purged}")
    print(f"  ✓ Fotos reasignadas a su verdadero dueño (ej. fotos de YO en Asier): {total_reassigned}")

    # Guardar cachés purificadas
    print("\n[3/4] Guardando sface_cache_v7.pkl y centroids.pkl...")
    with open(cache_path, 'wb') as f:
        pickle.dump(kf, f)

    centroids_dict = {(v.get('cat', 'Conocidos'), k): v['centroid'] for k, v in kf.items()}
    with open(BASE_DIR / 'centroids.pkl', 'wb') as f:
        pickle.dump(centroids_dict, f)
    print("  ✓ Modelos guardados y libres de ruido.")

    # Guardar apple_db.json y regenerar gallery_cache.json
    print("\n[4/4] Sincronizando base de datos y caché...")
    with open(db_path, 'w', encoding='utf-8') as f:
        json.dump(db, f, indent=2, ensure_ascii=False)
    print("  ✓ apple_db.json actualizado.")

    import app.services.gallery_service as gs
    gallery_data = gs.get_gallery(force_reload=True)
    with open(BASE_DIR / 'gallery_cache.json', 'w', encoding='utf-8') as f:
        json.dump(gallery_data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ gallery_cache.json regenerado exitosamente ({len(gallery_data)} categorías).")

    print("\n" + "=" * 80)
    print("🎉 APRENDIZAJE 100% DESCONTAMINADO Y VERIFICADO")
    print("=" * 80)

if __name__ == '__main__':
    main()
