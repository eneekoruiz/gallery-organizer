import sys
import os
import time
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from sklearn.cluster import DBSCAN, AgglomerativeClustering, HDBSCAN
from sklearn.metrics import (
    homogeneity_score,
    completeness_score,
    v_measure_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)
import networkx as nx

def run_chinese_whispers(adj_matrix, max_iter=20):
    """Chinese Whispers graph clustering algorithm (Biemann 2006)."""
    n = len(adj_matrix)
    labels = list(range(n))
    nodes = list(range(n))

    for _ in range(max_iter):
        random.shuffle(nodes)
        changed = 0
        for u in nodes:
            neighbors = adj_matrix[u]
            if not neighbors:
                continue
            class_weights = defaultdict(float)
            for v, w in neighbors:
                class_weights[labels[v]] += w
            if not class_weights:
                continue
            best_cls = max(class_weights.items(), key=lambda x: (x[1], random.random()))[0]
            if labels[u] != best_cls:
                labels[u] = best_cls
                changed += 1
        if changed == 0:
            break
            
    unique_map = {}
    norm_labels = []
    for l in labels:
        if l not in unique_map:
            unique_map[l] = len(unique_map)
        norm_labels.append(unique_map[l])
    return np.array(norm_labels)

def run_up_cac(X, dist_matrix, dt1=0.36, merge_sim_threshold=0.58, margin_ratio=0.10):
    """Ultra-Purity Constrained Agglomerative Clustering (UP-CAC).
    1. Complete-linkage micro-clustering at dt1 for 100% boundary purity.
    2. Centroid merge with Lowe's Ratio / Nearest-Neighbor Margin constraint.
    """
    cl1 = AgglomerativeClustering(distance_threshold=dt1, n_clusters=None, metric='precomputed', linkage='complete').fit(dist_matrix)
    l1 = cl1.labels_
    
    # Compute centroids for micro-clusters
    c_map = defaultdict(list)
    for idx, l in enumerate(l1):
        c_map[l].append(X[idx])
        
    u_labels = sorted(c_map.keys())
    if len(u_labels) <= 1:
        return l1
        
    centroids = []
    for l in u_labels:
        c = np.mean(c_map[l], axis=0)
        norm = np.linalg.norm(c)
        centroids.append(c / norm if norm > 0 else c)
        
    C = np.array(centroids)
    sim_c = np.dot(C, C.T)
    np.fill_diagonal(sim_c, -1.0)
    
    # Disjoint-set union find
    parent = list(range(len(u_labels)))
    def find(i):
        if parent[i] == i: return i
        parent[i] = find(parent[i])
        return parent[i]
    def union(i, j):
        root_i, root_j = find(i), find(j)
        if root_i != root_j: parent[root_i] = root_j

    for i in range(len(u_labels)):
        row = sim_c[i]
        top_indices = np.argsort(row)[::-1]
        best_j = top_indices[0]
        second_j = top_indices[1]
        
        best_sim = row[best_j]
        second_sim = row[second_j]
        
        # Merge ONLY if best similarity >= threshold AND has a distinct margin over the second best
        if best_sim >= merge_sim_threshold and (best_sim - second_sim) >= margin_ratio:
            # Check symmetry: does best_j also consider i its best or near-best?
            row_j = sim_c[best_j]
            if np.argsort(row_j)[::-1][0] == i or row_j[i] >= merge_sim_threshold:
                union(i, best_j)
                
    root_map = {}
    final_labels = []
    for l in l1:
        orig_idx = u_labels.index(l)
        root = find(orig_idx)
        if root not in root_map:
            root_map[root] = len(root_map)
        final_labels.append(root_map[root])
    return np.array(final_labels)

def count_false_merges(true_labels, pred_labels):
    """Count pairs of different people erroneously grouped in the same cluster."""
    clusters = defaultdict(list)
    for idx, pl in enumerate(pred_labels):
        if pl != -1:  # ignore noise
            clusters[pl].append(true_labels[idx])
    
    false_merges = 0
    for pl, members in clusters.items():
        counts = Counter(members)
        if len(counts) > 1:
            total_members = len(members)
            for person, cnt in counts.items():
                others = total_members - cnt
                false_merges += cnt * others
    return false_merges // 2

def main():
    print("=" * 75)
    print("🔬 BENCHMARK INTENSIVO DE APRENDIZAJE NO SUPERVISADO PARA CLUSTERING FACIAL")
    print("=" * 75)

    # 1. Load Ground Truth from apple_db.json
    print("\n[1/4] Extrayendo dataset de evaluación con Ground Truth real...")
    with open(BASE_DIR / 'apple_db.json', 'r', encoding='utf-8') as f:
        db = json.load(f)

    # Filter to balanced ground-truth set: people with at least 3 photos
    ident_photos = defaultdict(list)
    for k, v in db.items():
        cat = v.get('category', '')
        ident = v.get('identity', '')
        if cat in ['Conocidos', 'Familia', 'YO'] and ident and not any(term in ident.lower() for term in ['persona nueva', 'desconocid', 'sin rostro', 'ignorar', 'dudoso']):
            faces = v.get('faces', [])
            if faces and len(faces) == 1:
                feat = faces[0].get('feature')
                if feat:
                    arr = np.array(feat, dtype=np.float32).flatten()
                    norm = np.linalg.norm(arr)
                    if norm > 0:
                        ident_photos[ident].append(arr / norm)

    # Keep identities with >= 3 photos for meaningful clustering evaluation
    benchmark_data = []
    benchmark_labels = []
    
    # Cap dominant classes (e.g. Asier 800+ photos) to max 30 photos to prevent class imbalance skewing
    random.seed(42)
    np.random.seed(42)
    
    for ident, feats in ident_photos.items():
        if len(feats) >= 3:
            selected = feats if len(feats) <= 30 else random.sample(feats, 30)
            for vec in selected:
                benchmark_data.append(vec)
                benchmark_labels.append(ident)

    X = np.array(benchmark_data)
    y_true = np.array(benchmark_labels)
    n_samples = len(X)
    n_true_identities = len(set(y_true))
    print(f"  Dataset de evaluación: {n_samples} rostros reales distribuidos en {n_true_identities} personas verificadas.")

    # 2. Compute Precomputed Cosine Distance and Similarity Matrices
    print("\n[2/4] Precalculando matrices de distancia y afinidad coseno...")
    sim_matrix = np.dot(X, X.T)
    np.clip(sim_matrix, -1.0, 1.0, out=sim_matrix)
    dist_matrix = 1.0 - sim_matrix
    np.clip(dist_matrix, 0.0, 2.0, out=dist_matrix)

    # 3. Define Algorithms to Benchmark
    candidates = [
        # Baseline DBSCAN
        ("DBSCAN (eps=0.30, min=2)", lambda D, S: DBSCAN(eps=0.30, min_samples=2, metric='precomputed').fit_predict(D)),
        ("DBSCAN (eps=0.34, min=2)", lambda D, S: DBSCAN(eps=0.34, min_samples=2, metric='precomputed').fit_predict(D)),
        ("DBSCAN (eps=0.38, min=2)", lambda D, S: DBSCAN(eps=0.38, min_samples=2, metric='precomputed').fit_predict(D)),
        ("DBSCAN (eps=0.42, min=2)", lambda D, S: DBSCAN(eps=0.42, min_samples=2, metric='precomputed').fit_predict(D)),
        ("DBSCAN (eps=0.34, min=3)", lambda D, S: DBSCAN(eps=0.34, min_samples=3, metric='precomputed').fit_predict(D)),

        # HDBSCAN (Density-Adaptive)
        ("HDBSCAN (min_size=2)", lambda D, S: HDBSCAN(min_cluster_size=2, metric='precomputed', copy=True).fit_predict(D.astype(np.float64))),
        ("HDBSCAN (min_size=3)", lambda D, S: HDBSCAN(min_cluster_size=3, metric='precomputed', copy=True).fit_predict(D.astype(np.float64))),

        # Agglomerative Average Linkage
        ("Agglomerative-Average (dt=0.30)", lambda D, S: AgglomerativeClustering(distance_threshold=0.30, n_clusters=None, metric='precomputed', linkage='average').fit_predict(D)),
        ("Agglomerative-Average (dt=0.34)", lambda D, S: AgglomerativeClustering(distance_threshold=0.34, n_clusters=None, metric='precomputed', linkage='average').fit_predict(D)),
        ("Agglomerative-Average (dt=0.38)", lambda D, S: AgglomerativeClustering(distance_threshold=0.38, n_clusters=None, metric='precomputed', linkage='average').fit_predict(D)),
        ("Agglomerative-Average (dt=0.42)", lambda D, S: AgglomerativeClustering(distance_threshold=0.42, n_clusters=None, metric='precomputed', linkage='average').fit_predict(D)),

        # Agglomerative Complete Linkage (Ultra-strict boundary protection)
        ("Agglomerative-Complete (dt=0.35)", lambda D, S: AgglomerativeClustering(distance_threshold=0.35, n_clusters=None, metric='precomputed', linkage='complete').fit_predict(D)),
        ("Agglomerative-Complete (dt=0.38)", lambda D, S: AgglomerativeClustering(distance_threshold=0.38, n_clusters=None, metric='precomputed', linkage='complete').fit_predict(D)),
        ("Agglomerative-Complete (dt=0.40)", lambda D, S: AgglomerativeClustering(distance_threshold=0.40, n_clusters=None, metric='precomputed', linkage='complete').fit_predict(D)),
        ("Agglomerative-Complete (dt=0.45)", lambda D, S: AgglomerativeClustering(distance_threshold=0.45, n_clusters=None, metric='precomputed', linkage='complete').fit_predict(D)),

        # Chinese Whispers / Graph Label Propagation
        ("Chinese Whispers (sim >= 0.70)", lambda D, S: _cw_runner(S, 0.70)),
        ("Chinese Whispers (sim >= 0.65)", lambda D, S: _cw_runner(S, 0.65)),
        ("Chinese Whispers (sim >= 0.60)", lambda D, S: _cw_runner(S, 0.60)),

        # Ultra-Purity Constrained Agglomerative Clustering (UP-CAC)
        ("UP-CAC (dt1=0.34, merge=0.60, margin=0.10)", lambda D, S: run_up_cac(X, D, 0.34, 0.60, 0.10)),
        ("UP-CAC (dt1=0.36, merge=0.58, margin=0.08)", lambda D, S: run_up_cac(X, D, 0.36, 0.58, 0.08)),
        ("UP-CAC (dt1=0.38, merge=0.55, margin=0.06)", lambda D, S: run_up_cac(X, D, 0.38, 0.55, 0.06)),

        # Two-Stage Hybrid (Agglomerative Average dt=0.38 + Centroid Merging dt=0.45)
        ("Two-Stage Hybrid (Avg-0.38 + Super-0.45)", lambda D, S: _two_stage_hybrid_runner(X, D, 0.38, 0.45)),
    ]

    # 4. Run Benchmark
    print("\n[3/4] Ejecutando evaluación empírica sistemática...")
    results = []

    for name, runner in candidates:
        t0 = time.time()
        try:
            preds = runner(dist_matrix, sim_matrix)
            runtime = time.time() - t0

            # Evaluate metrics
            # Filter unclustered (-1) for purity metrics if present
            mask = preds != -1
            n_clustered = int(np.sum(mask))
            n_noise = int(np.sum(~mask))
            n_clusters = len(set(preds[mask]))

            if n_clustered > 1:
                homo = homogeneity_score(y_true[mask], preds[mask])
                comp = completeness_score(y_true[mask], preds[mask])
                v_meas = v_measure_score(y_true[mask], preds[mask])
                ari = adjusted_rand_score(y_true[mask], preds[mask])
                nmi = normalized_mutual_info_score(y_true[mask], preds[mask])
            else:
                homo, comp, v_meas, ari, nmi = 0.0, 0.0, 0.0, 0.0, 0.0

            f_merges = count_false_merges(y_true, preds)

            results.append({
                "algorithm": name,
                "homogeneity": homo,
                "completeness": comp,
                "v_measure": v_meas,
                "ari": ari,
                "nmi": nmi,
                "false_merges": f_merges,
                "clusters": n_clusters,
                "noise": n_noise,
                "runtime_ms": round(runtime * 1000, 2)
            })
            print(f"  ✓ {name:<42} | Homo: {homo:.4f} | V-Meas: {v_meas:.4f} | ARI: {ari:.4f} | Falsas Fusiones: {f_merges} | Tiempo: {runtime*1000:.1f}ms")
        except Exception as e:
            print(f"  ✗ {name:<42} | ERROR: {e}")

    # 5. Rank and Print Summary Table
    print("\n[4/4] Analizando resultados y determinando el algoritmo campeón...")
    # Sort by: 1) Lowest false merges (Zero False Positives priority), 2) Highest V-Measure, 3) Highest ARI
    ranked = sorted(results, key=lambda r: (r['false_merges'], -r['v_measure'], -r['ari']))

    champion = ranked[0]
    print("\n" + "=" * 95)
    print(f"{'Algoritmo':<40} {'Homog.':<8} {'Compl.':<8} {'V-Meas':<8} {'ARI':<8} {'F.Fus':<6} {'Clusters':<8} {'Ruido':<6}")
    print("-" * 95)
    for r in ranked:
        mark = " 🏆 (GANADOR)" if r == champion else ""
        print(f"{r['algorithm']:<40} {r['homogeneity']:<8.4f} {r['completeness']:<8.4f} {r['v_measure']:<8.4f} {r['ari']:<8.4f} {r['false_merges']:<6} {r['clusters']:<8} {r['noise']:<6}{mark}")
    print("=" * 95)

    print(f"\n🥇 ALGORITMO CAMPEÓN RECOMENDADO: {champion['algorithm']}")
    print(f"   - Homogeneidad (Pureza): {champion['homogeneity']*100:.2f}% (Tasa de acierto de identidades)")
    print(f"   - Falsas Alarmas (Fusiones erróneas entre personas distintas): {champion['false_merges']}")
    print(f"   - V-Measure (Balance entre pureza y cohesión): {champion['v_measure']*100:.2f}%")
    print(f"   - Tiempo de ejecución: {champion['runtime_ms']} ms")

    # Save benchmark report to JSON and artifact
    out_json = BASE_DIR / "benchmark_clustering_results.json"
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({
            "samples": n_samples,
            "true_identities": n_true_identities,
            "champion": champion,
            "ranked": ranked
        }, f, indent=2)
    print(f"\n✓ Resultados detallados guardados en {out_json.name}")

def _cw_runner(sim_matrix, threshold):
    n = len(sim_matrix)
    adj = defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            s = sim_matrix[i, j]
            if s >= threshold:
                adj[i].append((j, float(s)))
                adj[j].append((i, float(s)))
    return run_chinese_whispers(adj)

def _two_stage_hybrid_runner(X, dist_matrix, dt1, dt2):
    # Stage 1: Conservative Agglomerative Clustering
    cl1 = AgglomerativeClustering(distance_threshold=dt1, n_clusters=None, metric='precomputed', linkage='average').fit(dist_matrix)
    l1 = cl1.labels_
    
    # Compute centroids for each cluster
    c_map = defaultdict(list)
    for idx, l in enumerate(l1):
        c_map[l].append(X[idx])
    
    u_labels = sorted(c_map.keys())
    centroids = []
    for l in u_labels:
        c = np.mean(c_map[l], axis=0)
        c /= np.linalg.norm(c)
        centroids.append(c)
        
    C = np.array(centroids)
    dist_c = 1.0 - np.dot(C, C.T)
    np.clip(dist_c, 0, 2, out=dist_c)
    
    # Stage 2: Super-cluster centroids
    cl2 = AgglomerativeClustering(distance_threshold=dt2, n_clusters=None, metric='precomputed', linkage='average').fit(dist_c)
    l2 = cl2.labels_
    
    label_remap = {u_labels[i]: l2[i] for i in range(len(u_labels))}
    final_labels = np.array([label_remap[l] for l in l1])
    return final_labels

def _two_stage_hybrid_runner_complete(X, dist_matrix, dt1, dt2):
    cl1 = AgglomerativeClustering(distance_threshold=dt1, n_clusters=None, metric='precomputed', linkage='complete').fit(dist_matrix)
    l1 = cl1.labels_
    c_map = defaultdict(list)
    for idx, l in enumerate(l1):
        c_map[l].append(X[idx])
    u_labels = sorted(c_map.keys())
    centroids = []
    for l in u_labels:
        c = np.mean(c_map[l], axis=0)
        c /= np.linalg.norm(c)
        centroids.append(c)
    C = np.array(centroids)
    dist_c = 1.0 - np.dot(C, C.T)
    np.clip(dist_c, 0, 2, out=dist_c)
    cl2 = AgglomerativeClustering(distance_threshold=dt2, n_clusters=None, metric='precomputed', linkage='complete').fit(dist_c)
    l2 = cl2.labels_
    label_remap = {u_labels[i]: l2[i] for i in range(len(u_labels))}
    final_labels = np.array([label_remap[l] for l in l1])
    return final_labels

if __name__ == '__main__':
    main()
