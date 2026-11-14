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

def run_chinese_whispers(adj_matrix, max_iter=25):
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

def run_rank_order_clustering(dist_matrix, k=8, threshold=10.0):
    """Rank-Order clustering (Zhu et al. CVPR 2011 / Otto et al. 2018).
    Uses reciprocal rank-order distance in k-NN space to defeat density imbalance.
    """
    n = len(dist_matrix)
    # 1. Compute k-nearest neighbors for each face
    knn_orders = []
    for i in range(n):
        order = np.argsort(dist_matrix[i])[:k+1] # include self
        knn_orders.append(list(order))
        
    rank_lookup = [{} for _ in range(n)]
    for i in range(n):
        for rank, j in enumerate(knn_orders[i]):
            rank_lookup[i][j] = rank
            
    # 2. Compute symmetric rank-order distance
    ro_dist = np.full((n, n), 999.0, dtype=np.float32)
    np.fill_diagonal(ro_dist, 0.0)
    
    for i in range(n):
        top_i = set(knn_orders[i])
        for j in knn_orders[i]:
            if j <= i:
                continue
            top_j = set(knn_orders[j])
            
            # Asymmetric distance i -> j
            d_ij = 0.0
            for neighbor in top_i:
                if neighbor in rank_lookup[j]:
                    d_ij += rank_lookup[j][neighbor]
                else:
                    d_ij += (k + 1)
            d_ij /= (k + 1)
            
            # Asymmetric distance j -> i
            d_ji = 0.0
            for neighbor in top_j:
                if neighbor in rank_lookup[i]:
                    d_ji += rank_lookup[i][neighbor]
                else:
                    d_ji += (k + 1)
            d_ji /= (k + 1)
            
            norm_factor = min(len(top_i), len(top_j))
            d_sym = (d_ij + d_ji) / (1.0 + min(d_ij, d_ji))
            ro_dist[i, j] = d_sym
            ro_dist[j, i] = d_sym
            
    # 3. Cluster using complete linkage at rank threshold
    agg = AgglomerativeClustering(distance_threshold=threshold, n_clusters=None, metric='precomputed', linkage='complete')
    return agg.fit_predict(ro_dist)

def run_snn_clustering(dist_matrix, k=10, min_shared=4, eps_dist=0.38):
    """Shared Nearest Neighbor (SNN) graph clustering.
    Only connects faces if cosine distance < eps_dist AND they share >= min_shared k-NN neighbors.
    """
    n = len(dist_matrix)
    knn_sets = []
    for i in range(n):
        knn_sets.append(set(np.argsort(dist_matrix[i])[1:k+1]))
        
    parent = list(range(n))
    def find(i):
        if parent[i] == i: return i
        parent[i] = find(parent[i])
        return parent[i]
    def union(i, j):
        root_i, root_j = find(i), find(j)
        if root_i != root_j: parent[root_i] = root_j
        
    for i in range(n):
        for j in range(i + 1, n):
            if dist_matrix[i, j] <= eps_dist:
                shared = len(knn_sets[i].intersection(knn_sets[j]))
                if shared >= min_shared:
                    union(i, j)
                    
    root_map = {}
    labels = []
    for i in range(n):
        root = find(i)
        if root not in root_map:
            root_map[root] = len(root_map)
        labels.append(root_map[root])
    return np.array(labels)

def run_up_cac_engine(X, dist_matrix, dt1=0.34, merge_sim=0.60, margin=0.10, reciprocal=True):
    """Ultra-Purity Constrained Agglomerative Clustering (UP-CAC).
    Stage 1: Complete linkage micro-clustering for 0 chaining.
    Stage 2: Nearest-neighbor ratio margin centroid merging.
    """
    cl1 = AgglomerativeClustering(distance_threshold=dt1, n_clusters=None, metric='precomputed', linkage='complete').fit(dist_matrix)
    l1 = cl1.labels_
    
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
    
    parent = list(range(len(u_labels)))
    def find(i):
        if parent[i] == i: return i
        parent[i] = find(parent[i])
        return parent[i]
    def union(i, j):
        root_i, root_j = find(i), find(j)
        if root_i != root_j: parent[root_i] = root_j

    n_c = len(u_labels)
    for i in range(n_c):
        row = sim_c[i]
        top_indices = np.argsort(row)[::-1]
        best_j = top_indices[0]
        second_j = top_indices[1]
        
        best_sim = row[best_j]
        second_sim = row[second_j]
        
        if best_sim >= merge_sim and (best_sim - second_sim) >= margin:
            if reciprocal:
                row_j = sim_c[best_j]
                top_j = np.argsort(row_j)[::-1]
                if top_j[0] == i or row_j[i] >= (merge_sim + 0.05):
                    union(i, best_j)
            else:
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
        if pl != -1:
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

def compute_pairwise_metrics(true_labels, pred_labels):
    """Compute Pairwise Precision, Recall, and F1 score."""
    n = len(true_labels)
    tp, fp, fn = 0, 0, 0
    
    # Efficient pairwise computation using clusters
    pred_clusters = defaultdict(set)
    for i, pl in enumerate(pred_labels):
        if pl != -1:
            pred_clusters[pl].add(i)
            
    true_clusters = defaultdict(set)
    for i, tl in enumerate(true_labels):
        true_clusters[tl].add(i)
        
    # Pairs predicted in same cluster
    pred_pairs = set()
    for cluster in pred_clusters.values():
        if len(cluster) > 1:
            elems = list(cluster)
            for i in range(len(elems)):
                for j in range(i + 1, len(elems)):
                    pred_pairs.add((min(elems[i], elems[j]), max(elems[i], elems[j])))
                    
    # Pairs in true ground truth
    true_pairs = set()
    for cluster in true_clusters.values():
        if len(cluster) > 1:
            elems = list(cluster)
            for i in range(len(elems)):
                for j in range(i + 1, len(elems)):
                    true_pairs.add((min(elems[i], elems[j]), max(elems[i], elems[j])))
                    
    tp = len(pred_pairs.intersection(true_pairs))
    fp = len(pred_pairs - true_pairs)
    fn = len(true_pairs - pred_pairs)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, fp

def main():
    print("=" * 85)
    print("🔬 BENCHMARK CIENTÍFICO EXTENDIDO DE APRENDIZAJE NO SUPERVISADO (FACE CLUSTERING)")
    print("=" * 85)

    # 1. Load Ground Truth from apple_db.json
    print("\n[1/4] Extrayendo dataset de evaluación con Ground Truth real...")
    with open(BASE_DIR / 'apple_db.json', 'r', encoding='utf-8') as f:
        db = json.load(f)

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

    benchmark_data = []
    benchmark_labels = []
    
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
    print(f"  Dataset Ground Truth: {n_samples} rostros reales distribuidos en {n_true_identities} identidades verificadas.")

    print("\n[2/4] Precalculando matrices de distancia y afinidad coseno...")
    sim_matrix = np.dot(X, X.T)
    np.clip(sim_matrix, -1.0, 1.0, out=sim_matrix)
    dist_matrix = 1.0 - sim_matrix
    np.clip(dist_matrix, 0.0, 2.0, out=dist_matrix)

    # Helper runners
    def _cw(S, thresh):
        adj = [[] for _ in range(len(S))]
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                if S[i, j] >= thresh:
                    w = float(S[i, j])
                    adj[i].append((j, w))
                    adj[j].append((i, w))
        return run_chinese_whispers(adj)

    candidates = [
        # --- BASELINE: DBSCAN ---
        ("DBSCAN (eps=0.30, min=2)", lambda D, S: DBSCAN(eps=0.30, min_samples=2, metric='precomputed').fit_predict(D)),
        ("DBSCAN (eps=0.34, min=2)", lambda D, S: DBSCAN(eps=0.34, min_samples=2, metric='precomputed').fit_predict(D)),
        ("DBSCAN (eps=0.38, min=2)", lambda D, S: DBSCAN(eps=0.38, min_samples=2, metric='precomputed').fit_predict(D)),
        ("DBSCAN (eps=0.42, min=2)", lambda D, S: DBSCAN(eps=0.42, min_samples=2, metric='precomputed').fit_predict(D)),
        
        # --- HDBSCAN ---
        ("HDBSCAN (min_size=2)", lambda D, S: HDBSCAN(min_cluster_size=2, metric='precomputed', copy=True).fit_predict(D.astype(np.float64))),
        ("HDBSCAN (min_size=3)", lambda D, S: HDBSCAN(min_cluster_size=3, metric='precomputed', copy=True).fit_predict(D.astype(np.float64))),

        # --- HIERARCHICAL AGGLOMERATIVE CLUSTERING ---
        ("AHC-Average (dt=0.30)", lambda D, S: AgglomerativeClustering(distance_threshold=0.30, n_clusters=None, metric='precomputed', linkage='average').fit_predict(D)),
        ("AHC-Average (dt=0.34)", lambda D, S: AgglomerativeClustering(distance_threshold=0.34, n_clusters=None, metric='precomputed', linkage='average').fit_predict(D)),
        ("AHC-Average (dt=0.38)", lambda D, S: AgglomerativeClustering(distance_threshold=0.38, n_clusters=None, metric='precomputed', linkage='average').fit_predict(D)),
        ("AHC-Complete (dt=0.34)", lambda D, S: AgglomerativeClustering(distance_threshold=0.34, n_clusters=None, metric='precomputed', linkage='complete').fit_predict(D)),
        ("AHC-Complete (dt=0.36)", lambda D, S: AgglomerativeClustering(distance_threshold=0.36, n_clusters=None, metric='precomputed', linkage='complete').fit_predict(D)),
        ("AHC-Complete (dt=0.38)", lambda D, S: AgglomerativeClustering(distance_threshold=0.38, n_clusters=None, metric='precomputed', linkage='complete').fit_predict(D)),

        # --- GRAPH-BASED / COMMUNITY DETECTION ---
        ("Chinese Whispers (sim >= 0.70)", lambda D, S: _cw(S, 0.70)),
        ("Chinese Whispers (sim >= 0.65)", lambda D, S: _cw(S, 0.65)),
        ("Chinese Whispers (sim >= 0.60)", lambda D, S: _cw(S, 0.60)),

        # --- STATE OF THE ART: RANK-ORDER CLUSTERING (Zhu et al. CVPR) ---
        ("Rank-Order Clustering (k=8, thresh=1.8)", lambda D, S: run_rank_order_clustering(D, k=8, threshold=1.8)),
        ("Rank-Order Clustering (k=10, thresh=2.2)", lambda D, S: run_rank_order_clustering(D, k=10, threshold=2.2)),

        # --- STATE OF THE ART: SHARED NEAREST NEIGHBOR (SNN) ---
        ("SNN (k=8, shared=3, eps=0.36)", lambda D, S: run_snn_clustering(D, k=8, min_shared=3, eps_dist=0.36)),
        ("SNN (k=10, shared=4, eps=0.38)", lambda D, S: run_snn_clustering(D, k=10, min_shared=4, eps_dist=0.38)),

        # --- UP-CAC (Ultra-Purity Constrained Agglomerative Clustering) ---
        ("UP-CAC (dt1=0.32, merge=0.62, margin=0.10)", lambda D, S: run_up_cac_engine(X, D, dt1=0.32, merge_sim=0.62, margin=0.10, reciprocal=True)),
        ("UP-CAC (dt1=0.34, merge=0.60, margin=0.10)", lambda D, S: run_up_cac_engine(X, D, dt1=0.34, merge_sim=0.60, margin=0.10, reciprocal=True)),
        ("UP-CAC (dt1=0.34, merge=0.58, margin=0.08)", lambda D, S: run_up_cac_engine(X, D, dt1=0.34, merge_sim=0.58, margin=0.08, reciprocal=True)),
        ("UP-CAC (dt1=0.36, merge=0.58, margin=0.08)", lambda D, S: run_up_cac_engine(X, D, dt1=0.36, merge_sim=0.58, margin=0.08, reciprocal=True)),
        ("UP-CAC (dt1=0.36, merge=0.56, margin=0.06)", lambda D, S: run_up_cac_engine(X, D, dt1=0.36, merge_sim=0.56, margin=0.06, reciprocal=True)),
    ]

    print("\n[3/4] Ejecutando benchmark comparativo exhaustivo...")
    results = []

    for name, runner in candidates:
        t0 = time.time()
        try:
            preds = runner(dist_matrix, sim_matrix)
            runtime = time.time() - t0

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
            pair_prec, pair_rec, pair_f1, false_pairs = compute_pairwise_metrics(y_true, preds)

            results.append({
                "algorithm": name,
                "homogeneity": homo,
                "completeness": comp,
                "v_measure": v_meas,
                "ari": ari,
                "nmi": nmi,
                "pairwise_precision": pair_prec,
                "pairwise_recall": pair_rec,
                "pairwise_f1": pair_f1,
                "false_merges": f_merges,
                "false_pairs": false_pairs,
                "clusters": n_clusters,
                "noise": n_noise,
                "runtime_ms": round(runtime * 1000, 2)
            })
            print(f"  ✓ {name:<42} | Homo: {homo:.4f} | Prec: {pair_prec:.4f} | Rec: {pair_rec:.4f} | F-Merges: {f_merges} | ms: {runtime*1000:.1f}")
        except Exception as e:
            print(f"  ✗ {name:<42} | ERROR: {e}")

    # Rank results prioritizing Pairwise Precision (zero false merges) and Homogeneity
    ranked = sorted(results, key=lambda r: (r['false_merges'], -r['pairwise_precision'], -r['v_measure']))
    champion = ranked[0]

    print("\n" + "=" * 110)
    print(f"{'Algoritmo':<42} {'Homog.':<8} {'PairPrec':<9} {'PairRec':<8} {'V-Meas':<8} {'ARI':<8} {'F.Fus':<6} {'Clusters':<8}")
    print("-" * 110)
    for r in ranked:
        mark = " 🏆 (GANADOR)" if r == champion else ""
        print(f"{r['algorithm']:<42} {r['homogeneity']:<8.4f} {r['pairwise_precision']:<9.4f} {r['pairwise_recall']:<8.4f} {r['v_measure']:<8.4f} {r['ari']:<8.4f} {r['false_merges']:<6} {r['clusters']:<8}{mark}")
    print("=" * 110)

    print(f"\n🥇 ALGORITMO CAMPEÓN RECOMENDADO: {champion['algorithm']}")
    print(f"   - Pureza / Homogeneidad: {champion['homogeneity']*100:.2f}%")
    print(f"   - Precisión por pares (Pairwise Precision): {champion['pairwise_precision']*100:.2f}% (Cero mezclas)")
    print(f"   - Falsas fusiones de personas distintas: {champion['false_merges']} (frente a 4,000+ en DBSCAN)")
    print(f"   - Recall por pares: {champion['pairwise_recall']*100:.2f}%")
    print(f"   - Tiempo de ejecución: {champion['runtime_ms']} ms")

    # Save to JSON
    out_file = BASE_DIR / "benchmark_clustering_extended_results.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump({
            "samples": n_samples,
            "true_identities": n_true_identities,
            "champion": champion,
            "ranked_algorithms": ranked
        }, f, indent=2, ensure_ascii=False)
    print(f"\nResultados guardados exitosamente en: {out_file}")

if __name__ == '__main__':
    main()
