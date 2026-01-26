# How to Use representation_comparison.py

## Quick Start (Jupyter Notebook / VS Code)

The file has `#%%` cell markers so you can run it interactively in VS Code or Jupyter.

### Step 1: Add a test cell at the end of the file

```python
#%%
# EXAMPLE USAGE - Replace with your actual data
from representation_comparison import compare_all_models, ComparisonConfig

config = ComparisonConfig(
    filename="your_data_file.sav",  # Your SPSS file
    var_name="Q20",                   # Your question variable
    sample_size=50,                   # Sample size used in Step 5
    n_sample_clusters=10,             # How many clusters to display
    export_excel=True,
    verbose=True
)

# Run comparison
results = compare_all_models(
    config=config,
    n_sample_clusters=10,
    export_excel=True
)

print("\n✅ Comparison complete!")
print(f"Results: {len(results['results'])} models compared")
print(f"Clusters: {len(results['clusters'])} clusters analyzed")
```

### Step 2: Run the cell

Click "Run Cell" in VS Code or execute in Jupyter.

## What It Does

1. **Loads cached Step 5 results** from your pipeline run
2. **Extracts cluster → ideas mapping** (with tag stripping)
3. **Runs 5 keyword extraction models**:
   - Standard TF-IDF (baseline)
   - c-TF-IDF (BERTopic)
   - c-TF-IDF + MMR (diversity)
   - c-TF-IDF + KeyBERT (embeddings)
   - c-TF-IDF + LLM (GPT-enhanced)
4. **Displays side-by-side comparison** of keywords
5. **Calculates metrics** (coverage, avg keywords, diversity)
6. **Exports to Excel** with multiple sheets

## Output Structure

### Console Output

```
Loading cached Step 5 results: your_file.sav, Q20...
Extracting cluster → ideas mapping...
Found 13 clusters

[1/5] Running Standard TF-IDF...
[2/5] Running c-TF-IDF...
[3/5] Running c-TF-IDF + MMR...
[4/5] Running c-TF-IDF + KeyBERT...
[5/5] Running c-TF-IDF + LLM Enhancement...

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
   3. quality              (0.4891)
   ...

c-TF-IDF + KeyBERT:
   1. portion              (0.9565)
   2. small                (0.8391)
   3. servings             (0.7205)
   ...

c-TF-IDF + LLM:
   1. portion size         (1.0000)
   2. too small            (0.9000)
   3. insufficient         (0.8000)
   ...

====================================================================================================
COMPARISON METRICS
====================================================================================================

Model                          Coverage     Avg Keywords    Avg Diversity
------------------------------ ------------ --------------- ---------------
Standard TF-IDF                100.0%       15.0            2.34
c-TF-IDF                       100.0%       15.0            2.56
c-TF-IDF + MMR                 100.0%       10.0            3.12
c-TF-IDF + KeyBERT             100.0%       10.0            2.89
c-TF-IDF + LLM                 100.0%       10.0            3.45
```

### Excel Export

File: `exports/representation_comparison.xlsx`

**Sheets:**
1. **Overview** - All models with keywords in one view
2. **Standard_TF-IDF** - Detailed scores for Standard TF-IDF
3. **c-TF-IDF** - Detailed scores for c-TF-IDF
4. **c-TF-IDF_MMR** - Detailed scores for MMR
5. **c-TF-IDF_KeyBERT** - Detailed scores for KeyBERT
6. **c-TF-IDF_LLM** - Detailed scores for LLM

## Integration with cluster_analysis.py

**No integration needed!** The comparison script is standalone and works independently.

However, if you want to **integrate the advanced models into cluster_analysis.py**, you can:

### Option 1: Use cluster_analysis.py as-is (Already working)

`cluster_analysis.py` already uses c-TF-IDF for keywords and passes them to the LLM. This is the validated Phase 1.5 implementation.

### Option 2: Extend cluster_analysis.py with model selection

Add a configuration option to choose which model to use:

```python
@dataclass
class ExperimentConfig:
    # ... existing fields ...

    # NEW: Choose representation model
    representation_model: str = "ctfidf"  # Options: "ctfidf", "mmr", "keybert", "llm"
    mmr_diversity: float = 0.3  # For MMR
    keybert_weight: float = 0.5  # For KeyBERT
    llm_keyword_model: str = "gpt-4.1-mini"  # For LLM enhancement
```

Then update the keyword extraction section to use the selected model.

### Option 3: Run comparison separately

Keep `cluster_analysis.py` focused on c-TF-IDF + LLM descriptions, and use `representation_comparison.py` for exploring different keyword extraction methods.

## Recommended Workflow

1. **First**: Run `cluster_analysis.py` with c-TF-IDF (already working)
   - This gives you baseline cluster descriptions with statistical keywords

2. **Then**: Run `representation_comparison.py` to explore alternatives
   - This shows you how different models compare
   - Helps you decide if you want to switch models

3. **Finally**: If you find a better model, integrate it into `cluster_analysis.py`
   - Update the keyword extraction code
   - Test on real data
   - Compare LLM descriptions quality

## Performance Comparison

| Model | Speed | Cost | Quality | Best For |
|-------|-------|------|---------|----------|
| **Standard TF-IDF** | ⚡⚡⚡ Fast | Free | ★★☆☆☆ | Quick baseline |
| **c-TF-IDF** | ⚡⚡⚡ Fast | Free | ★★★☆☆ | Standard (recommended) |
| **MMR** | ⚡⚡☆ Medium | Free | ★★★★☆ | More diverse keywords |
| **KeyBERT** | ⚡☆☆ Slow | $$ (embeddings) | ★★★★☆ | Semantic relevance |
| **LLM** | ⚡☆☆ Slowest | $$$ (LLM calls) | ★★★★★ | Best quality |

## Current Status

✅ **cluster_analysis.py** - Uses c-TF-IDF (Phase 1.5 complete)
✅ **representation_comparison.py** - Compare all 5 models (Phase 2 complete)
⏳ **Integration** - Optional (you decide based on comparison results)

The systems work independently, so you can:
- Continue using `cluster_analysis.py` as-is
- Run `representation_comparison.py` to explore alternatives
- Integrate later if you find a better model
