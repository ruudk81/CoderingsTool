# Regime-Aware Adaptive Clustering - Implementation Plan

## 🎯 Executive Summary

**Problem**: Current clustering produces fragmented results (13 clusters, ratio=1.4×) for n=65 dataset.

**Root Cause**: Regime-agnostic scoring treats all data the same. Small+Coherent data (q90=0.91) requires different strategy than Large+Diffuse data.

**Solution**: Implement regime-aware clustering that:
1. Detects data regime (R1-R9) based on size and embedding structure
2. Applies regime-specific strategies (metric weights, tolerances, targets)
3. Provides transparent, regime-based rationale for clustering decisions

**Expected Outcome**: Better-structured clustering with clear regime-based justification.

---

## ✅ Key Discovery: Embeddings Are Healthy!

### Cache Analysis Results (n=65 dataset)

**Direct measurement of cached embeddings:**
- **Actual q90 = 0.912821** (NOT 1.00 as initially reported!)
- **Duplicate rate: 1.5%** (only 1 duplicate out of 65 ideas)
- **Unique texts: 98.5%** (64 unique / 65 total)
- **Perfect similarities (≥0.9999): 0.05%** (1 pair - the duplicate)

**Similarity distribution:**
```
[0.00, 0.50):    64 pairs (  3.1%) - low similarity
[0.70, 0.80):   164 pairs (  7.9%) - moderate
[0.80, 0.90): 1,507 pairs ( 72.5%) - high (template effect visible but reasonable)
[0.90, 0.95):   305 pairs ( 14.7%) - very high
[0.95, 0.99):    37 pairs (  1.8%) - extremely high
[0.99, 1.00):     2 pairs (  0.1%) - near-perfect
```

**Embedded text format (confirmed working well):**
```
[lang=nl-NL][domain=voedingsmiddelenindustrie][topic=klanttevredenheid_kant-en-klare_maaltijden]
[perspective=consumer][entity=kant-en-klare_maaltijden][intent=suggest]
[sentiment=negative][sense=aspirational]
producent van kant-en-klare maaltijden moet snelle hap maaltijden maken
```

**Conclusion**:
- Embeddings are HEALTHY - context specifiers + template approach works!
- q90=0.91 is reasonable for template-dominated text
- Problem is NOT embeddings - it's regime-agnostic scoring

**Correct regime classification:**
- n=65 → "Small"
- q90=0.91 → "Coherent" (≥0.80 threshold)
- **Dataset IS Regime R3 (Small+Coherent)** ✓

---

## 📋 Regime Framework

### 9 Data Regimes

```
Size × Structure Matrix:

          │ Diffuse       │ Mixed         │ Coherent
          │ (q90 < 0.65)  │ (0.65-0.80)   │ (q90 ≥ 0.80)
──────────┼───────────────┼───────────────┼──────────────
Small     │ R1: Broad     │ R2: Balanced  │ R3: Consolidate
(n < 100) │ themes        │               │ (quality focus)
──────────┼───────────────┼───────────────┼──────────────
Medium    │ R4: Coverage  │ R5: Standard  │ R6: Fine
(100-300) │               │ (sweet spot)  │ distinctions
──────────┼───────────────┼───────────────┼──────────────
Large     │ R7: Many      │ R8:           │ R9: Aggressive
(> 300)   │ specific      │ Hierarchical  │ consolidation
```

### Regime-Specific Strategies

Each regime has tailored:

1. **Metric Weights**: How to score clustering quality
   - Geometry (separation): R1=10%, R3=30%, R5=25%
   - DBCV (cohesion): R1=20%, R3=45%, R5=30%
   - Stability: R1=50%, R3=15%, R5=30%
   - Hard noise: R1=20%, R3=10%, R5=15%

2. **Parameter Multipliers**: Adjust mcs/ms from baseline
   - R1 (Diffuse): 1.4× (larger clusters for broad themes)
   - R3 (Coherent): 1.4× (LARGER clusters to consolidate and improve quality)
   - R9 (Large+Coherent): 1.3× (force consolidation)

3. **Quality Targets**:
   - Expected cluster count ranges (R1: 4-8, R3: 5-8, R5: 20-40)
   - Hierarchy ratio targets (R1: 4.0×, R3: 4.0×, R5: 3.5×)
   - Noise tolerance (R1: 12%, R3: 15%, R7: 8%)
   - Fragmentation tolerance (R1: low, R3: low, R5: medium)

---

## 🔧 Implementation Plan

### Phase 1: Regime Detection

**File**: `src/utils/clusterer.py`
**Location**: Add before `_suggest_params()` (around line 920)

**New Method**: `_detect_data_regime(U, n_points) -> Dict[str, Any]`

**Purpose**: Classify data into R1-R9 based on (n, q90, kd_cv)

**Implementation**:
```python
def _detect_data_regime(self, U: np.ndarray, n_points: int) -> Dict[str, Any]:
    """Detect data regime to guide clustering strategy"""

    # Size classification
    if n_points < 100:
        size_class = "small"
    elif n_points <= 300:
        size_class = "medium"
    else:
        size_class = "large"

    # Reuse existing space diagnostics
    rs = self.rs
    Xn = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)
    Xn = Xn.astype(np.float32, copy=False)

    # Subsample for speed
    subsample = min(2000, n_points)
    if n_points > subsample:
        idx = rs.choice(n_points, subsample, replace=False)
        Xsub = Xn[idx]
    else:
        Xsub = Xn

    m = min(Xsub.shape[0], 1000)
    idy = rs.choice(Xsub.shape[0], m, replace=False)
    Y = Xsub[idy]

    # Cosine similarity diagnostics
    S = Y @ Y.T
    tri = S[np.triu_indices_from(S, k=1)]
    q90 = float(np.quantile(tri, 0.90))
    q50 = float(np.quantile(tri, 0.50))

    # kNN distance variance
    D = 1.0 - (Xsub @ Xsub.T)
    np.fill_diagonal(D, np.inf)
    knn_k = min(15, max(5, Xsub.shape[0] - 1))
    knn_d = np.partition(D, knn_k, axis=1)[:, :knn_k]
    kd_mean = float(np.mean(knn_d))
    kd_std = float(np.std(knn_d))
    kd_cv = float(kd_std / (kd_mean + 1e-12))

    # Structure classification
    if q90 >= 0.80:
        structure_class = "coherent"
    elif q90 < 0.65:
        structure_class = "diffuse"
    else:
        structure_class = "mixed"

    # Regime assignment
    regime_map = {
        ("small", "diffuse"): "R1",
        ("small", "mixed"): "R2",
        ("small", "coherent"): "R3",
        ("medium", "diffuse"): "R4",
        ("medium", "mixed"): "R5",
        ("medium", "coherent"): "R6",
        ("large", "diffuse"): "R7",
        ("large", "mixed"): "R8",
        ("large", "coherent"): "R9",
    }

    regime_id = regime_map[(size_class, structure_class)]

    regime_descriptions = {
        "R1": "Small+Diffuse: Seek broad holistic themes",
        "R2": "Small+Mixed: Balance consolidation with distinctions",
        "R3": "Small+Coherent: Preserve subtle distinctions within similar content",
        "R4": "Medium+Diffuse: Maximize coverage with balanced clusters",
        "R5": "Medium+Mixed: Standard case (sweet spot)",
        "R6": "Medium+Coherent: Preserve fine distinctions, high separation",
        "R7": "Large+Diffuse: Many specific clusters with clear boundaries",
        "R8": "Large+Mixed: Hierarchical with good separation",
        "R9": "Large+Coherent: Aggressive consolidation to avoid over-splitting",
    }

    return {
        'regime_id': regime_id,
        'size_class': size_class,
        'structure_class': structure_class,
        'description': regime_descriptions[regime_id],
        'n_points': n_points,
        'q90': q90,
        'q50': q50,
        'kd_cv': kd_cv,
        'diagnostics': f"n={n_points}, q90={q90:.2f}, q50={q50:.2f}, cv={kd_cv:.2f}"
    }
```

---

### Phase 2: Regime-Specific Strategies

**File**: `src/utils/clusterer.py`
**Location**: Add after `_detect_data_regime()`

**New Method**: `_get_regime_strategy(regime_id) -> Dict[str, Any]`

**Purpose**: Return clustering strategy for detected regime

**Implementation**: See full code in plan file at `/Users/ruudkooiman/.claude/plans/atomic-watching-ladybug.md` (lines 171-324)

**Key strategies**:

**R3 (Small+Coherent) - CORRECTED (Current focus):**
```python
"R3": {
    "goal": "consolidation",       # CHANGED: Force broader themes
    "param_multiplier": 1.4,       # CHANGED: LARGER mcs/ms for consolidation
    "metric_weights": {
        "dbcv": 0.45,              # CHANGED: DBCV CRITICAL for quality improvement
        "geometry": 0.30,          # Separation important but not primary
        "stability": 0.15,         # CHANGED: More weight on stability
        "hard_noise": 0.10         # Keep noise weight
    },
    "fragmentation_weight": 0.50,  # CHANGED: HIGH - heavily penalize fragmentation
    "expected_clusters": (5, 8),   # CHANGED: FEWER, broader clusters
    "target_ratio": 4.0,           # CHANGED: Strong hierarchy needed
    "noise_tolerance": 0.15,       # CHANGED: Standard tolerance (not high)
}
```

**Why this strategy works for n=65 with low quality (low DBCV/silhouette)**:
- Coherent data (q90=0.91) → ideas are similar → hard to separate with quality
- Current fragmentation (13 clusters, ratio=1.4×) produces LOW quality scores
- Solution: **CONSOLIDATE** into fewer, broader themes (5-8 clusters)
- Trade specificity for stability and quality (high DBCV/silhouette)
- **Key insight**: Coherent + Low Quality = Need Consolidation (NOT fine-grained)

---

### Phase 3: Fragmentation Detection

**File**: `src/utils/clusterer.py`
**Location**: Add after `_assess_noise_quality()` (around line 528)

**New Method**: `_detect_fragmentation(labels, n_points) -> Dict[str, float]`

**Metrics**:
1. **CV of cluster sizes**: Low (0.3-0.5) = good hierarchy
2. **Largest/Median ratio**: Healthy ≥ 3.0×
3. **Top-3 concentration**: 50-70% in top-3 clusters
4. **Composite fragmentation score**: 0-1 (0=good, 1=fragmented)

**Formula**:
```python
# Normalize to [0,1] where 0=good
cv_norm = clip((cv - 0.3) / 0.7, 0, 1)
ratio_norm = clip((3.0 - ratio) / 2.0, 0, 1)
top3_norm = clip((0.55 - top3_frac) / 0.35, 0, 1)

# Weighted composite (ratio most important)
fragmentation_score = 0.5 * ratio_norm + 0.3 * top3_norm + 0.2 * cv_norm
```

---

### Phase 4: Regime-Aware Scoring

**File**: `src/utils/clusterer.py`
**Location**: Replace scoring in `_auto_hdbscan_grid()` (lines 1062-1083)

**New Method**: `_calculate_regime_aware_score(metrics, fragmentation, regime_strategy)`

**Key improvements**:
1. **Regime-specific metric weights** (not universal)
2. **Escalating penalties** beyond tolerance (not linear)
3. **Fragmentation targets** vary by regime
4. **Noise tolerance** varies by regime

**Current (REMOVE)**:
```python
geometry = 0.5 * sil + 0.5 * db
stability = stab
base_score = (geometry + stability) / 2
penalties = (noise + 0.5 * k_n) / 2
final_score = 1 + base_score - penalties
```

**New (ADD)**:
```python
# Extract components
geometry = 0.5 * sil + 0.5 * db_inv
dbcv_scaled = (dbcv + 1.0) / 2.0
stability = stab
hard_noise = metrics['hard_noise_rate']

# Regime-specific weighting
weights = regime_strategy['metric_weights']
base_score = (
    weights['geometry'] * geometry +
    weights['dbcv'] * dbcv_scaled +
    weights['stability'] * stability
)

# Regime-aware penalties
noise_tolerance = regime_strategy['noise_tolerance']
if hard_noise <= noise_tolerance:
    noise_penalty = weights['hard_noise'] * hard_noise
else:
    excess = hard_noise - noise_tolerance
    noise_penalty = weights['hard_noise'] * (noise_tolerance + 2.0 * excess)

# Fragmentation penalty
target_ratio = regime_strategy['target_ratio']
actual_ratio = fragmentation['largest_median_ratio']
ratio_penalty = (target_ratio - actual_ratio) / target_ratio if actual_ratio < target_ratio else 0

frag_weight = regime_strategy['fragmentation_weight']
fragmentation_penalty = frag_weight * (0.7 * frag_score + 0.3 * ratio_penalty)

# Final score
penalties = noise_penalty + 0.2 * k_scaled + fragmentation_penalty
final_score = 1.0 + base_score - penalties
```

---

### Phase 5: Integration into Grid Search

**File**: `src/utils/clusterer.py`
**Location**: Modify `_auto_hdbscan_grid()` (starting line 1022)

**Flow**:
```python
def _auto_hdbscan_grid(self, U: np.ndarray):
    n = U.shape[0]

    # STEP 1: Detect regime
    regime_info = self._detect_data_regime(U, n)
    regime_strategy = self._get_regime_strategy(regime_info['regime_id'])

    # Report regime
    self.verbose_reporter.section_header("DATA REGIME DETECTION")
    self.verbose_reporter.stat_line(f"Regime: {regime_info['regime_id']} - {regime_info['description']}")
    self.verbose_reporter.stat_line(f"Diagnostics: {regime_info['diagnostics']}")
    self.verbose_reporter.stat_line(f"Strategy: {regime_strategy['goal']}")

    # STEP 2: Suggest regime-specific parameters
    ms, mcs, notes = self._suggest_params(U, regime_strategy)

    # STEP 3: Grid search
    results = self._grid_search(U, ms_grid, mcs)

    # STEP 4: Regime-aware scoring
    for r in results:
        fragmentation = self._detect_fragmentation(r["labels"], n)
        scoring = self._calculate_regime_aware_score(
            r["metrics"], fragmentation, regime_strategy
        )
        r.update(scoring)

    # STEP 5: Regime-aware selection
    results.sort(key=lambda r: r["score"], reverse=True)
    best = self._select_best_with_regime_targets(results, regime_strategy)

    return best["hdbscan_model"], best["labels"], best["summary"]
```

---

## 📊 Expected Behavioral Changes

### For n=65 case (Regime R3):

**Before** (regime-agnostic):
```
Best: mcs=2, ms=2
Result: 13 clusters (range 2-7)
Ratio: 1.4×
Noise: 18.5%
Scoring: Universal weights (geometry=25%, stability=30%)
Interpretation: ❌ POOR - too fragmented
```

**After** (regime-aware with CORRECTED R3 strategy):
```
[REGIME DETECTION]
Regime: R3 - Small+Coherent: Consolidate for quality
Diagnostics: n=65, q90=0.91, q50=0.85
Strategy: consolidation (CHANGED!)
Expected clusters: 5-8 (CHANGED - fewer, broader clusters)
Target ratio: 4.0× (CHANGED - strong hierarchy needed)
Noise tolerance: 15% (CHANGED - standard tolerance)

[PARAMETER SUGGESTION]
baseline: ms=1, mcs=1
regime multiplier: 1.4 (CHANGED - consolidation not fine_distinctions)
suggested: ms=2, mcs=2 (LARGER for consolidation)

[GRID SEARCH]
mcs=3, ms=2: 6 clusters, ratio=4.2×, noise=12%, DBCV=0.68, Sil=0.52
  - dbcv=45%, geometry=30%, stability=15% (CHANGED weights)
  - fragmentation: 0.12 (target=4.0×, actual=4.2× ✓ excellent)
  - SCORE: 1.35 ✓ BEST

mcs=2, ms=2: 13 clusters, ratio=1.4×, noise=18.5%, DBCV=0.42, Sil=0.31
  - fragmentation: 0.62 (under-hierarchical, LOW quality)
  - SCORE: 0.95 (low due to fragmentation penalty + low DBCV)

mcs=1, ms=1: 18 clusters, ratio=1.8×, noise=21%, DBCV=0.35, Sil=0.25
  - fragmentation: 0.75 (TOO fragmented, VERY LOW quality)
  - SCORE: 0.78 (heavily penalized)

Selection: mcs=3 - meets regime targets (5-8 clusters, high quality)
Interpretation: ✅ EXCELLENT for R3 - consolidated into broad, stable themes
```

**Key insight from correction**:
- Coherent data (q90=0.91) with LOW quality scores (DBCV=0.42, Sil=0.31)
- Problem: Too many small, unstable clusters (hard to find patterns/themes)
- Solution: CONSOLIDATE into fewer, broader clusters with HIGHER quality
- Trade specificity for stability and interpretability

---

## 🧪 Pre-Implementation: Deep Dive Analysis

**Before implementing**, we will:

1. **Analyze multiple cached datasets** with different characteristics:
   - Small+Diffuse (low q90)
   - Small+Coherent (high q90)
   - Medium+Mixed
   - Large datasets

2. **Validate q90 thresholds**:
   - Confirm 0.65 and 0.80 boundaries make sense
   - Check for edge cases near boundaries
   - Verify kd_cv adds useful signal

3. **Check for calculation bugs**:
   - Verify q90 calculation is consistent
   - Compare clusterer calculation vs direct cache analysis
   - Ensure no subsample bias

4. **Empirically tune strategies**:
   - Validate metric weight ratios
   - Adjust fragmentation/noise tolerances based on real data
   - Test parameter multipliers produce sensible mcs/ms values

**Script**: Will create analysis tool to:
- Load multiple cache files
- Calculate q90, q50, kd_cv for each
- Visualize similarity distributions
- Compare regime classification to actual clustering quality
- Report any anomalies or edge cases

---

## 📁 Files to Modify

1. **`src/utils/clusterer.py`** (~225 lines changed/added):
   - Add `_detect_data_regime()` method
   - Add `_get_regime_strategy()` method
   - Add `_detect_fragmentation()` method
   - Add `_calculate_regime_aware_score()` method
   - Modify `_suggest_params()` to use regime strategy
   - Modify `_auto_hdbscan_grid()` to integrate regime detection
   - Update verbose reporting sections

2. **`src/config.py`** (~10 lines added):
   - Add fragmentation thresholds to `ClusteringConfig`
   - Add quality tolerance thresholds

---

## ✅ Success Criteria

### Regime R3 (Small+Coherent) - Current case:
- ✅ 10-18 fine-grained clusters
- ✅ Ratio ≥ 1.8× acceptable
- ✅ Hard noise < 25%
- ✅ Console shows regime rationale

### Universal requirements:
1. **Regime detection accuracy**: >95% correct classification
2. **Transparency**: Console output shows regime and rationale
3. **Backward compatible**: All existing metrics still calculated
4. **Performance**: <5% slowdown in clustering time

---

## 🔍 Verification Plan

### 1. Unit Testing
- Test fragmentation detection on synthetic label arrays
- Test regime detection on synthetic embedding matrices
- Verify score calculation with known inputs

### 2. Integration Testing
- Run on problematic n=65 dataset
- Run on multiple dataset sizes (small/medium/large)
- Compare before/after cluster quality

### 3. Regression Testing
- Ensure noise reclustering still works
- Ensure merge logic still works
- Ensure polish loop still works

---

## 📚 References

- Plan file: `/Users/ruudkooiman/.claude/plans/atomic-watching-ladybug.md`
- Cache analysis script: `/private/tmp/.../scratchpad/analyze_embeddings.py`
- Cached embeddings: `data/cache/005_embeddings_*.pkl`
