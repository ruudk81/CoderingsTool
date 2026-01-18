# UMAP × Clustering Algorithm Comparison Experiment

## Overview

This experiment explores different combinations of UMAP dimensionality reduction settings and clustering algorithms to find optimal configurations for survey response coding. The goal is to develop a data-driven clustering strategy that adapts based on dataset characteristics.

**Main file**: `umap_clustering_comparison.py`

---

## Context: Why This Experiment?

### The Problem

The production clusterer (`src/utils/clusterer.py`) uses a regime-agnostic approach that doesn't always produce optimal results:

- **Small coherent datasets** (n<100, high q90) often get fragmented into too many clusters
- **HDBSCAN** isn't always the best choice - sometimes Agglomerative or K-means produces better coherence
- **No principled way** to decide which algorithm to use for a given dataset

### Previous Work

1. **Regime-Aware Clustering Plan** (`REGIME_AWARE_CLUSTERING_PLAN.md`):
   - Defined 9 data regimes (R1-R9) based on Size × Structure matrix
   - Proposed regime-specific strategies for parameter selection
   - Identified that q90=0.91 datasets need consolidation, not fine-grained clustering

2. **Experimental Clusterer** (`clusterer NEW.py`):
   - Implemented regime detection and DBCV-primary scoring
   - Added gated final selection with quality thresholds
   - Still primarily HDBSCAN-focused

3. **This Experiment**:
   - Systematic comparison of HDBSCAN vs Agglomerative vs K-means
   - kNN distance elbow analysis to detect density structure
   - Data-driven algorithm selection based on knee detection

---

## Key Discoveries

### 1. Knee Detection for Algorithm Selection

**Core Insight**: The shape of the kNN distance curve reveals whether HDBSCAN is appropriate.

```
If kNN distance curve has a sharp knee early in the curve:
  → Data has density-based structure → HDBSCAN is appropriate

If kNN distance curve is smooth/linear (no sharp knee):
  → Data has uniform density → Use Agglomerative or K-means
```

### 2. Three-Part Knee Validation

A meaningful knee must pass three tests:

1. **Knee Exists**: KneeLocator finds a point
2. **Within Acceptance Bounds**: `K_min <= K <= K_max`
   - `K_min = max(3, 0.5 * sqrt(n))`
   - `K_max = min(4 * sqrt(n), 0.85 * n)`
3. **Sharp Enough**: `slope_after / slope_before >= 3.0`

**Rationale**: A late knee (e.g., K=46 for n=65) indicates uniform density with just outliers at the tail - not good structure for HDBSCAN.

### 3. Search Window vs Acceptance Bounds

**Failed Approach**: Truncating the search window to `[0.25*sqrt(n), 4*sqrt(n)]`
- Forces KneeLocator to find *something* in that range
- Always finds a "meaningful" knee, even in linear curves
- Defeats the purpose of data-driven detection

**Correct Approach**:
- **Search window**: `[1, 0.85*n]` - wide range to find the natural knee
- **Acceptance bounds**: `[0.5*sqrt(n), min(4*sqrt(n), 0.85*n)]` - filter for position
- The bounds act as a *filter*, not a *constraint*

### 4. Coherence is the Key Metric

**Coherence** (mean intra-cluster cosine similarity) is the most meaningful metric for survey response clustering:

| Threshold | Category | Interpretation |
|-----------|----------|----------------|
| < 0.70 | Unacceptable | Clusters too heterogeneous |
| 0.70-0.90 | Low | Marginal quality |
| 0.90-0.95 | Moderate | Acceptable quality |
| >= 0.95 | High | Excellent quality |

**Trade-off**: More clusters = lower coherence per cluster. The "sweet spot" is the highest k where mean coherence remains >= 0.90.

### 5. Algorithm-Specific Metrics

Each clustering algorithm should be evaluated with appropriate metrics:

| Algorithm | Primary Metric | Rationale |
|-----------|---------------|-----------|
| HDBSCAN | DBCV | Density-based, handles non-convex clusters |
| Agglomerative | Silhouette + Davies-Bouldin | Assumes convex/hierarchical clusters |
| K-means | Silhouette + Davies-Bouldin | Assumes convex/spherical clusters |

---

## Emerging Clustering Strategy

Based on experiment results, the following strategy is proposed:

### Decision Flow

```
1. If n < 100:
   → Use Agglomerative + K-means (skip HDBSCAN)
   → Small datasets rarely have meaningful density structure

2. If n >= 100:
   a. Run knee detection on kNN distance curve
   b. If meaningful knee exists (in bounds + sharp):
      → HDBSCAN + K-means (compare results)
   c. If no meaningful knee:
      → Agglomerative + K-means (skip HDBSCAN)

3. Compare results using coherence:
   → Select configuration with highest k while coherence >= 0.90

4. Within each algorithm's grid search:
   → HDBSCAN: Use DBCV for parameter selection
   → Agglomerative/K-means: Use Silhouette + Davies-Bouldin
```

### Why Skip HDBSCAN for Small Datasets?

- Small n means few points to establish density patterns
- Knee detection is unreliable with few data points
- Agglomerative with Ward linkage handles hierarchy well
- K-means provides a stable baseline

---

## Experiment Configuration

### UMAP Grid (9 configurations)
| n_neighbors | n_components | min_dist |
|-------------|--------------|----------|
| 5, 10, 30 | 5, 10, 20 | 0.1 |

### Clustering Algorithms

1. **HDBSCAN**:
   - Grid: `mcs = [sqrt(n), 0.5*sqrt(n), 0.25*sqrt(n)]`, `ms = 0.5 * mcs`
   - Selection: Best DBCV score

2. **Agglomerative**:
   - Linkage: Ward's method
   - k range: 3-15
   - Selection: Best silhouette score

3. **K-means**:
   - k range: 3-15
   - Selection: Best silhouette score

### Total: 27 experiments per dataset (9 UMAP × 3 algorithms)

---

## Output Artifacts

| File | Description |
|------|-------------|
| `exports/umap_clustering_comparison.xlsx` | Full results with all metrics |
| `exports/knn_elbow_plots.png` | Knee detection visualization per UMAP config |
| `exports/coherence_vs_k.png` | Coherence trade-off by number of clusters |

---

## Running the Experiment

```bash
# Activate environment
cd /Users/ruudkooiman/projects/Python_apps/CoderingsTool
source .venv/bin/activate

# Ensure Step 4 embeddings exist in cache
cd src
python pipeline.py  # with RUN_UNTIL_STEP = 4

# Run experiment
python experiments/umap_clustering_comparison.py
```

### Configuration (in file)

```python
# Data source
FILENAME = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
VAR_NAME = "Q20"
SAMPLE_SIZE = 50  # Set to None for full dataset

# UMAP settings
UMAP_NEIGHBORS = [5, 10, 30]
UMAP_COMPONENTS = [5, 10, 20]

# Clustering settings
K_RANGE = range(3, 16)
```

---

## Future Directions

### Phase 1: Validate on Different Dataset Sizes

Test the knee detection strategy on datasets of varying sizes:

- **Small** (n=30-100): Confirm HDBSCAN skipping is appropriate
- **Medium** (n=100-500): Validate knee detection accuracy
- **Large** (n=500+): Check if strategy scales

### Phase 2: Integrate into Production Clusterer

1. Add `choose_cluster_strategy_via_knee()` to `clusterer.py`
2. Implement algorithm selection logic:
   - Small n → Agglomerative/K-means only
   - Meaningful knee → Include HDBSCAN
   - No knee → Skip HDBSCAN
3. Add coherence-based final selection

### Phase 3: Regime-Aware Enhancements

Combine findings with regime-aware clustering plan:

1. Use regime (R1-R9) for initial parameter hints
2. Use knee detection for algorithm selection
3. Use coherence breakdown for quality assessment

### Phase 4: Hierarchical Clustering Exploration

For large datasets (n>500), explore:

- Two-stage clustering (coarse → fine)
- Hierarchical HDBSCAN with cluster_selection_epsilon
- Agglomerative with different linkages (complete, average)

### Phase 5: Representation Model Integration

Combine clustering improvements with representation models:

- c-TF-IDF for cluster keywords (`representation/ctfidf_representation.py`)
- MMR for diversity (`representation/mmr_representation.py`)
- KeyBERT for semantic alignment (`representation/keybert_representation.py`)
- LLM for refinement (`representation/llm_representation.py`)

---

## Technical Notes

### Dependencies Added

```
kneed>=0.8.0  # Knee/elbow detection for clustering strategy
```

### Warning Suppression

```python
# Suppress UMAP n_jobs warning when using random_state
warnings.filterwarnings("ignore", message="n_jobs value .* overridden to 1 by setting random_state")

# Suppress HDBSCAN validity warnings (divide by zero in edge cases)
warnings.filterwarnings("ignore", message="invalid value encountered in divide", module="hdbscan.validity")
```

### Key Functions

| Function | Purpose |
|----------|---------|
| `choose_cluster_strategy_via_knee()` | Knee detection + algorithm recommendation |
| `compute_slope_ratio()` | Validate knee sharpness (slope_after/slope_before) |
| `calculate_coherence_breakdown()` | Per-cluster coherence analysis (4-tier) |
| `generate_knn_elbow_plots()` | Visualization with acceptance bounds |
| `generate_coherence_vs_k_plot()` | Coherence trade-off visualization |

---

## Lessons Learned

1. **Don't force knee detection**: Let KneeLocator search the full range, then filter results
2. **Slope ratio matters**: A detected knee isn't meaningful if the curve is nearly linear
3. **Coherence over separation**: High silhouette with low coherence = meaningless clusters
4. **Size matters**: Small datasets need different strategies than large datasets
5. **Algorithm appropriateness**: HDBSCAN isn't always the best choice

---

## References

- Production clusterer: `src/utils/clusterer.py`
- Experimental clusterer: `src/utils/clusterer NEW.py`
- Regime-aware plan: `src/experiments/REGIME_AWARE_CLUSTERING_PLAN.md`
- Representation models: `src/experiments/representation/`
- Plan file: `/Users/ruudkooiman/.claude/plans/deep-roaming-plum.md`