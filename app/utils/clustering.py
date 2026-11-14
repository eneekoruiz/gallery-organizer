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
from sklearn.cluster import AgglomerativeClustering

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
    cl1 = AgglomerativeClustering(
        distance_threshold=dt1,
        n_clusters=None,
        metric='precomputed',
        linkage='complete'
    ).fit(dist_matrix)
    l1 = cl1.labels_

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
