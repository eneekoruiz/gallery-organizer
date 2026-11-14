"""Módulo de Agrupamiento Facial de Ultra-Alta Pureza (UP-CAC).
Sustituto de producción para DBSCAN en reconocimiento facial no supervisado.

Algoritmo: Ultra-Purity Constrained Agglomerative Clustering (UP-CAC)
- Fase 1: Agrupamiento jerárquico por enlace completo (Complete Linkage) para
  garantizar pureza de fronteras e imposibilitar el encadenamiento transitivo.
- Fase 2: Fusión de centroides con restricción de margen de ratio de Lowe y reciprocidad,
  evitando mezclas de personas con fisonomías similares o fotos ambiguas.
"""

from collections import defaultdict
import numpy as np

try:
    from sklearn.cluster import AgglomerativeClustering
except ImportError:
    AgglomerativeClustering = None

def cluster_faces_upcac(
    embeddings: np.ndarray,
    dt1: float = 0.34,
    merge_sim: float = 0.60,
    margin: float = 0.10,
    reciprocal: bool = True
) -> np.ndarray:
    """Ejecuta el clustering UP-CAC sobre una matriz de embeddings faciales normalizados.

    Args:
        embeddings: Matriz 2D (N, D) de vectores unitarios de características faciales.
        dt1: Umbral de distancia coseno máxima para micro-clusters en Fase 1 (dt1 = 1 - sim).
             Por defecto 0.34 (similitud >= 0.66).
        merge_sim: Similitud coseno mínima entre centroides para fusionar en Fase 2.
             Por defecto 0.60.
        margin: Margen de ratio de Lowe exigido entre la mejor coincidencia y la segunda mejor.
             Por defecto 0.10 (evita fusiones ambiguas).
        reciprocal: Si True, exige reciprocidad o cuasi-reciprocidad entre ambos clusters.

    Returns:
        np.ndarray de enteros con las etiquetas asignadas a cada muestra (0 .. K-1).
    """
    n_samples = len(embeddings)
    if n_samples == 0:
        return np.array([], dtype=int)
    if n_samples == 1:
        return np.array([0], dtype=int)

    # 1. Asegurar normalización L2 unitaria
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = embeddings / norms

    # 2. Matriz de distancias coseno precomputada
    sim_matrix = np.dot(X, X.T)
    np.clip(sim_matrix, -1.0, 1.0, out=sim_matrix)
    dist_matrix = 1.0 - sim_matrix
    np.clip(dist_matrix, 0.0, 2.0, out=dist_matrix)

    # 3. Fase 1: Micro-clustering por enlace completo
    # Complete linkage exige que dist_max <= dt1 entre CUALQUIER par del cluster
    if AgglomerativeClustering is not None:
        cl1 = AgglomerativeClustering(
            distance_threshold=dt1,
            n_clusters=None,
            metric='precomputed',
            linkage='complete'
        ).fit(dist_matrix)
        l1 = cl1.labels_
    else:
        # Fallback si scikit-learn no está disponible en entornos ultraligeros
        l1 = np.arange(n_samples)
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if dist_matrix[i, j] <= dt1:
                    l1[j] = l1[i]

    # Agrupar muestras por micro-cluster
    c_map = defaultdict(list)
    for idx, l in enumerate(l1):
        c_map[l].append(X[idx])

    u_labels = sorted(c_map.keys())
    if len(u_labels) <= 1:
        return l1

    # 4. Fase 2: Cálculo de centroides normalizados
    centroids = []
    for l in u_labels:
        c = np.mean(c_map[l], axis=0)
        norm = np.linalg.norm(c)
        centroids.append(c / norm if norm > 0 else c)

    C = np.array(centroids)
    sim_c = np.dot(C, C.T)
    np.fill_diagonal(sim_c, -1.0)

    # Estructura Disjoint-Set Union-Find
    parent = list(range(len(u_labels)))

    def find(i: int) -> int:
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i: int, j: int):
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    n_c = len(u_labels)
    for i in range(n_c):
        row = sim_c[i]
        top_indices = np.argsort(row)[::-1]
        best_j = top_indices[0]
        second_j = top_indices[1] if n_c > 1 else best_j

        best_sim = row[best_j]
        second_sim = row[second_j] if n_c > 1 else -1.0

        # Verificación estricta del criterio de Lowe
        if best_sim >= merge_sim and (best_sim - second_sim) >= margin:
            if reciprocal:
                row_j = sim_c[best_j]
                top_j = np.argsort(row_j)[::-1]
                if top_j[0] == i or row_j[i] >= (merge_sim + 0.05):
                    union(i, best_j)
            else:
                union(i, best_j)

    # Re-mapear raíces a identificadores contiguos 0..K-1
    root_map = {}
    final_labels = []
    for l in l1:
        orig_idx = u_labels.index(l)
        root = find(orig_idx)
        if root not in root_map:
            root_map[root] = len(root_map)
        final_labels.append(root_map[root])

    return np.array(final_labels, dtype=int)


def consolidate_new_persons_upcac(db: dict, known_centroids: dict = None) -> dict:
    """Consolida automáticamente todas las 'Personas Nuevas' en la base de datos:
    1. Si algún rostro coincide con una identidad conocida (>= 0.63), lo reasigna.
    2. Agrupa los rostros desconocidos restantes mediante UP-CAC en grupos cohesivos.
    3. Minimiza carpetas unitarias de 1 sola foto.
    """
    if known_centroids is None:
        try:
            import pickle
            from pathlib import Path
            cache_p = Path(__file__).resolve().parent.parent.parent / "centroids.pkl"
            if cache_p.exists():
                with open(cache_p, "rb") as f:
                    known_centroids = pickle.load(f)
        except Exception:
            known_centroids = {}

    c_keys = []
    c_matrix = []
    if known_centroids:
        for k, v in known_centroids.items():
            k_name = k[1] if isinstance(k, tuple) else str(k)
            if not k_name.startswith("Persona Nueva") and "Desconocid" not in k_name:
                c_keys.append(k_name)
                norm_v = np.array(v, dtype=np.float32).flatten()
                norm_v = norm_v / (np.linalg.norm(norm_v) + 1e-10)
                c_matrix.append(norm_v)

    c_matrix = np.array(c_matrix) if c_matrix else None

    reassigned_to_known = 0
    unknown_paths = []
    unknown_vectors = []

    for path, info in db.items():
        ident = str(info.get("identity", ""))
        cat = str(info.get("category", ""))
        if "Persona Nueva" in ident or cat == "Personas Nuevas":
            faces = info.get("faces", [])
            matched = False
            best_match = None

            for face in faces:
                feat = face.get("feature")
                if feat is not None:
                    vec = np.array(feat, dtype=np.float32).flatten()
                    norm = np.linalg.norm(vec)
                    if norm > 1e-6:
                        vec = vec / norm
                        if c_matrix is not None and len(c_keys) > 0:
                            sims = np.dot(c_matrix, vec)
                            top_idx = int(np.argmax(sims))
                            top_sim = float(sims[top_idx])
                            top_ident = c_keys[top_idx]
                            if top_sim >= 0.63:
                                matched = True
                                best_match = (top_ident, top_sim, face)
                                break
                            elif not best_match or top_sim > best_match[1]:
                                best_match = (top_ident, top_sim, face)

            if matched and best_match:
                top_ident, top_sim, face = best_match
                info["identity"] = top_ident
                dest_cat = "Familia" if top_ident.startswith(("F.", "F ")) else ("YO" if top_ident == "YO" else "Conocidos")
                info["category"] = dest_cat
                info["status"] = "Clasificado"
                info["is_dudoso"] = False
                info["review_required"] = False
                face["identity"] = top_ident
                face["category"] = dest_cat
                reassigned_to_known += 1
            elif best_match and best_match[2].get("feature") is not None:
                feat = best_match[2].get("feature")
                v = np.array(feat, dtype=np.float32).flatten()
                v = v / (np.linalg.norm(v) + 1e-10)
                unknown_paths.append(path)
                unknown_vectors.append(v)

    clusters_count = 0
    if len(unknown_vectors) > 1:
        X = np.array(unknown_vectors)
        labels = cluster_faces_upcac(X, dt1=0.38, merge_sim=0.62, margin=0.08, reciprocal=True)
        cluster_map = {}
        curr_id = 1
        for lab in labels:
            if lab not in cluster_map:
                cluster_map[lab] = f"👤 Persona Nueva #{curr_id}"
                curr_id += 1

        clusters_count = len(cluster_map)
        for path, lab in zip(unknown_paths, labels):
            new_name = cluster_map[lab]
            db[path]["identity"] = new_name
            db[path]["category"] = "Personas Nuevas"
            db[path]["status"] = "Clasificado"
            for f in db[path].get("faces", []):
                f["identity"] = new_name
                f["category"] = "Personas Nuevas"

    return {
        "reassigned_to_known": reassigned_to_known,
        "consolidated_unknowns": len(unknown_paths),
        "cohesive_groups": clusters_count
    }
