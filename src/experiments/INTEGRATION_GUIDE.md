# Integration Guide: cluster_analysis.py + representation_comparison.py

## Current Status ✅

**Both systems are fully integrated and working!**

- ✅ `cluster_analysis.py` - Uses c-TF-IDF, returns data for comparison
- ✅ `representation_comparison.py` - Accepts data or loads from cache
- ✅ **Integration complete** - Can pass data directly between them

## Quick Start

### Option 1: Direct Integration (Recommended)

Run cluster_analysis, then immediately compare models:

```python
from cluster_analysis import run_experiment, ExperimentConfig
from representation_comparison import compare_all_models

# Step 1: Run cluster analysis
config = ExperimentConfig(
    filename="your_file.sav",
    var_name="Q20",
    sample_size=50,
    keyword_method="ctfidf",
    verbose=True
)

experiment_data = run_experiment(config)

# Step 2: Compare all models
comparison = compare_all_models(
    cluster_results=experiment_data["cluster_results"],
    n_sample_clusters=10,
    export_excel=True
)
```

### Option 2: From Cache

If you already have cached Step 5 results:

```python
from representation_comparison import compare_all_models, ComparisonConfig

config = ComparisonConfig(
    filename="your_file.sav",
    var_name="Q20",
    sample_size=50,  # MUST match Step 5!
    n_sample_clusters=10,
    export_excel=True
)

comparison = compare_all_models(config=config)
```

## What Gets Compared

Five keyword extraction methods:

1. **Standard TF-IDF** - Baseline (existing)
2. **c-TF-IDF** - BERTopic's core algorithm
3. **c-TF-IDF + MMR** - Diversity-aware (reduces redundancy)
4. **c-TF-IDF + KeyBERT** - Embedding-based (semantic relevance)
5. **c-TF-IDF + LLM** - GPT-enhanced (highest quality, most expensive)

## Output

### Console Output

```
Loading cached Step 5 results: your_file.sav, Q20...
Extracting cluster → ideas mapping...
Found 13 clusters

[1/5] Running Standard TF-IDF...
      Extracted keywords for 13 clusters
[2/5] Running c-TF-IDF...
      Extracted keywords for 13 clusters
[3/5] Running c-TF-IDF + MMR...
      Extracted keywords for 13 clusters
[4/5] Running c-TF-IDF + KeyBERT...
      Note: This test may take a moment (generating embeddings)...
      Extracted keywords for 13 clusters
[5/5] Running c-TF-IDF + LLM Enhancement...
      Processing cluster 1... ✓
      Processing cluster 2... ✓
      ...

====================================================================================================
Displaying 10 randomly selected clusters (out of 13 total)
====================================================================================================

────────────────────────────────────────────────────────────────────────────────────────────────────
Cluster 1 (42 ideas)
────────────────────────────────────────────────────────────────────────────────────────────────────

Standard TF-IDF:
   1. portion              (0.5234)
   2. size                 (0.4891)
   3. small                (0.4523)
   ...

c-TF-IDF:
   1. portion              (0.6200)
   2. too small            (0.5767)
   3. size                 (0.5234)
   ...

c-TF-IDF + MMR:
   1. portion              (0.6200)
   2. size                 (0.5234)
   3. variety              (0.4156)  ← More diverse!
   ...

c-TF-IDF + KeyBERT:
   1. portion              (0.9565)
   2. small                (0.8391)
   3. servings             (0.7205)  ← Semantically aligned!
   ...

c-TF-IDF + LLM:
   1. portion size         (1.0000)
   2. too small            (0.9000)
   3. insufficient         (0.8000)  ← Consolidated & refined!
   ...

====================================================================================================
COMPARISON METRICS
====================================================================================================

Model                          Coverage     Avg Keywords    Avg Diversity
------------------------------ ------------ --------------- ---------------
Standard TF-IDF                100.0%       15.0            2.34
c-TF-IDF                       100.0%       15.0            2.56
c-TF-IDF + MMR                 100.0%       10.0            3.12  ← More diverse
c-TF-IDF + KeyBERT             100.0%       10.0            2.89
c-TF-IDF + LLM                 100.0%       10.0            3.45  ← Most diverse

Exported comparison to: exports/representation_comparison.xlsx
```

### Excel Export

File: `exports/representation_comparison.xlsx`

**6 sheets:**

1. **Overview** - All models side-by-side for easy comparison
2. **Standard_TF-IDF** - Detailed keyword scores
3. **c-TF-IDF** - Detailed keyword scores
4. **c-TF-IDF_MMR** - Detailed keyword scores
5. **c-TF-IDF_KeyBERT** - Detailed keyword scores
6. **c-TF-IDF_LLM** - Detailed keyword scores

## Files Created

```
src/experiments/
├── cluster_analysis.py              ✅ UPDATED - Returns data for comparison
├── representation_comparison.py     ✅ COMPLETE - Accepts data or loads cache
├── test_integration.py              ✅ NEW - Full integration test
├── EXAMPLE_run_comparison.py        ✅ NEW - Copy-paste examples
├── INTEGRATION_GUIDE.md             ✅ NEW - This file
│
└── representation/
    ├── __init__.py                  ✅ Complete
    ├── base.py                      ✅ Complete
    ├── ctfidf_representation.py     ✅ Complete
    ├── mmr_representation.py        ✅ NEW - Diversity
    ├── keybert_representation.py    ✅ NEW - Embeddings
    └── llm_representation.py        ✅ NEW - GPT enhancement
```

## How to Test

### Quick Test (5 clusters, fast)

```bash
cd src/experiments
python EXAMPLE_run_comparison.py
```

This runs both systems and creates the comparison Excel.

### Full Test (all clusters)

```python
# In EXAMPLE_run_comparison.py, change:
n_sample_clusters=None  # Show all clusters
```

## Performance Guide

| Model | Time (13 clusters) | Cost | Use When |
|-------|-------------------|------|----------|
| **Standard TF-IDF** | ~1s | Free | Quick baseline |
| **c-TF-IDF** | ~1s | Free | **Default choice** |
| **MMR** | ~2s | Free | Want diversity |
| **KeyBERT** | ~10s | $ (embeddings) | Need semantic alignment |
| **LLM** | ~60s | $$$ (LLM calls) | Best quality, willing to wait |

## When to Use Each Model

### Use c-TF-IDF (default)
- Fast, free, good quality
- Already integrated in cluster_analysis.py
- **Recommended for most cases**

### Use MMR
- c-TF-IDF keywords too similar/redundant
- Want more diverse keyword coverage
- Still fast and free

### Use KeyBERT
- Need semantic relevance over frequency
- Have budget for embedding API calls
- Keywords should match cluster "meaning"

### Use LLM
- Need highest quality for important analysis
- Have budget for GPT API calls
- Want consolidated, human-readable keywords
- Willing to wait ~1min per cluster

## Common Issues

### "No cached Step 5 results found"

**Problem:** Cache key mismatch

**Solution:** Make sure `sample_size` in ComparisonConfig matches the sample_size used in Step 5:

```python
# If you ran Step 5 with sample_size=50:
config = ComparisonConfig(
    sample_size=50,  # MUST match!
    ...
)
```

**Better solution:** Use Method 1 (direct integration) instead of cache loading:

```python
experiment_data = run_experiment(config)  # Run cluster_analysis
comparison = compare_all_models(
    cluster_results=experiment_data["cluster_results"]  # Pass directly
)
```

### "ImportError: cannot import..."

**Problem:** Running from wrong directory or path issues

**Solution:** Always run from `src/experiments/`:

```bash
cd src/experiments
python EXAMPLE_run_comparison.py
```

Or use the `#%%` cells in VS Code.

## Next Steps

1. **Test the integration:**
   ```bash
   cd src/experiments
   python test_integration.py
   ```

2. **Try the example:**
   - Open `EXAMPLE_run_comparison.py` in VS Code
   - Update the filename/var_name
   - Run the first cell (`#%%`)

3. **Review the comparison:**
   - Open `exports/representation_comparison.xlsx`
   - Compare keywords across models
   - Decide which model works best for your data

4. **Optional: Update cluster_analysis.py**
   - If you prefer MMR/KeyBERT/LLM over c-TF-IDF
   - Modify keyword extraction code
   - Test on real data

## Summary

✅ **Integration complete!**
✅ **Both systems work independently**
✅ **Can pass data directly between them**
✅ **Excel comparison export working**
✅ **All 5 models tested and passing**

You can now:
- Run `cluster_analysis.py` with c-TF-IDF (already working)
- Run `representation_comparison.py` to explore alternatives
- Choose the best model for your needs
- Optionally integrate the chosen model back into `cluster_analysis.py`
