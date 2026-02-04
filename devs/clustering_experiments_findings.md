# Clustering Experiments: Findings and Recommendations

*Date: January 2026*
*Dataset: Merk X brand associations (n=1805 ideas, 3072-dim embeddings)*

These findings are tentative but useful for informing future development decisions.

---

## 1. Embedding Space Diagnostics

### Pairwise Cosine Similarity Analysis

Before clustering, analyze the raw embedding space to understand the data regime:

- **q90 threshold** classifies data into regimes:
  - Diffuse (q90 < 0.65): spread out, weak clustering signal
  - Mixed (0.65 <= q90 < 0.80): moderate structure
  - Coherent (q90 >= 0.80): tight clusters expected

- **High similarity pair counts** (>0.90, >0.95) indicate potential near-duplicates or very tight clusters

### Density Variation Coefficient (DVC)

DVC = std(d_k) / mean(d_k), where d_k is the distance to the k-th nearest neighbor.

- **k = sqrt(n)** is used for dynamic scaling with dataset size
- DVC interpretation:
  - DVC < 0.25: Uniform density → Agglomerative clustering preferred
  - 0.25 <= DVC < 0.45: Moderate density variation
  - DVC >= 0.45: High density variation → HDBSCAN preferred

### Knee Detection in Ordered kNN Distances

A sharp knee in the ordered k-th nearest neighbor distance plot indicates natural cluster boundaries. Combined with DVC:

- **Sharp knee + high DVC**: Strong signal for HDBSCAN
- **No knee + low DVC**: Data lacks natural density-based structure → consider Agglomerative

---

## 2. HDBSCAN on Precomputed Cosine Distance Matrix

### Experiment

Tested HDBSCAN with `metric="precomputed"` on a cosine distance matrix computed from raw (non-reduced) embeddings.

### Results

- 81 clusters, 31.2% noise
- DBCV: NaN (cannot compute for precomputed distances)
- Significantly worse than UMAP + HDBSCAN (130 clusters, 2.7% noise)

### Conclusion

**The curse of dimensionality is real.** In 3072-dimensional space, density-based clustering struggles because:
- All points appear roughly equidistant
- No clear density peaks emerge
- HDBSCAN cannot find where clusters "end"

Dimensionality reduction is essential for HDBSCAN to work effectively on high-dimensional text embeddings.

---

## 3. Algorithm Selection: HDBSCAN vs Agglomerative

### When to use Agglomerative Clustering

For datasets with:
- **Small n** (< 200-500 ideas)
- **Low DVC** (< 0.25, uniform density)
- **No knee** in ordered kNN distances

Agglomerative with grid search on k (optimizing silhouette or Davies-Bouldin) is safer because:
- It doesn't assume density-based structure
- Grid search externally imposes structure
- More stable with limited data
- Deterministic results

### When to use HDBSCAN

For datasets with:
- **Larger n** (> 500 ideas)
- **High DVC** (> 0.45, variable density)
- **Sharp knee** in kNN distances

HDBSCAN excels when the data naturally has dense regions separated by sparse regions.

### The Middle Ground (0.25 < DVC < 0.45)

This is the challenging regime. Current approach:
- Use knee detection as a tiebreaker
- Sharp knee → HDBSCAN
- No knee → Agglomerative

---

## 4. Dimensionality Reduction: UMAP vs PaCMAP

### Experimental Setup

Both tested with:
- n_components = 10
- n_neighbors = 10 (and 5)
- HDBSCAN: min_cluster_size=5, min_samples=2

### Results Summary (n_neighbors=10)

| Method | Clusters | Noise | DBCV | Low Prob |
|--------|----------|-------|------|----------|
| UMAP + HDBSCAN | 112 | 5.8% | 0.622 | 21.1% |
| PaCMAP + HDBSCAN | 89 | 23.5% | 0.371 | 35.3% |

### Qualitative Analysis

PaCMAP produced **fragmented clusters** for concepts that UMAP kept coherent:
- "duurzaam" split into 4 separate clusters (vs 1 in UMAP)
- "eekhoorn" split into 4 clusters (vs 1 consolidated cluster in UMAP)
- "groen" split into 2 clusters containing just the word itself

Many semantically meaningful items ended up as noise in PaCMAP.

### Why UMAP Outperforms for HDBSCAN Clustering

1. **UMAP's `min_dist=0.0`** creates tight density concentrations that HDBSCAN can exploit. PaCMAP has no equivalent parameter.

2. **Different optimization goals:**
   - UMAP: Preserves local neighborhood structure (good for density-based clustering)
   - PaCMAP: Balances local, mid-range, and global structure (good for visualization)

3. **PaCMAP's global structure preservation works against HDBSCAN** - by spreading points more evenly to show global relationships, it undermines the density peaks HDBSCAN needs.

### Key Insight

**The dimensionality reduction method and clustering algorithm must be matched:**
- UMAP + HDBSCAN: Good match (UMAP creates density structure)
- PaCMAP + HDBSCAN: Mismatch (PaCMAP spreads things out)
- PaCMAP + Agglomerative: Potentially better match (untested)

### When PaCMAP Might Still Be Useful

- When you need **coarser, more holistic clusters** (10-20 themes vs 100+ fine-grained topics)
- Paired with **Agglomerative clustering** instead of HDBSCAN
- For **visualization** where global relationships matter

---

## 5. Production Recommendations

### Default Pipeline

1. **Compute diagnostics**: pairwise similarity, DVC (k=sqrt(n)), kNN knee
2. **Algorithm selection**:
   - Small n + low DVC + no knee → Agglomerative with grid search
   - Large n + high DVC + sharp knee → UMAP + HDBSCAN
3. **Dimensionality reduction**: UMAP with `min_dist=0.0` for HDBSCAN

### UMAP Parameters

- `n_components`: 10 (sufficient for clustering, not visualization)
- `n_neighbors`: 5-15 range, scale with dataset size
- `min_dist`: 0.0 (critical for clustering - creates tight density)
- `metric`: euclidean (on L2-normalized embeddings)

### HDBSCAN Parameters

- `min_cluster_size`: Scale with sqrt(n), range [5, 50]
- `min_samples`: 2-5, controls cluster density threshold
- `metric`: euclidean (on reduced embeddings)

---

## 6. Open Questions

1. **PaCMAP + Agglomerative**: Would this pairing work better for coarse clustering?

2. **Adaptive n_neighbors**: Should UMAP's n_neighbors scale with dataset size or data regime?

3. **Coherent data regime** (q90 >= 0.80): Does the algorithm selection logic hold for very tight, coherent data?

4. **Cross-validation**: How stable are these findings across different datasets and domains?

---

## 7. Tools Created

- `analyze_embeddings.py`: Diagnostic tool for raw embedding space analysis
- `test_hdbscan_setups.py`: Comparison script for UMAP vs PaCMAP with HDBSCAN
