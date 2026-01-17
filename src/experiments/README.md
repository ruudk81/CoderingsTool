# Cluster Analysis Experimentation Framework

This directory contains tools for experimenting with TF-IDF keyword extraction and improved cluster descriptions before integrating into the main pipeline.

## Quick Start

```bash
# Activate environment
cd /Users/ruudkooiman/projects/Python_apps/CoderingsTool
source .venv/bin/activate

# Ensure Step 5 cache exists (if not already done)
cd src
# Edit pipeline.py: RUN_UNTIL_STEP = 5, set your dataset/variable
python pipeline.py

# Run experiment
python experiments/cluster_analysis.py
```

## Files

- **`tfidf_analyzer.py`**: TF-IDF keyword extraction utility
  - Configurable n-grams, stopwords, min/max document frequency
  - Supports Dutch and English languages
  - Returns top-k keywords per cluster with scores

- **`cluster_analysis.py`**: Main experiment runner
  - Loads cached Step 5 clustering results
  - Extracts TF-IDF keywords for each cluster
  - Generates LLM descriptions (optionally enhanced with keywords)
  - Displays sample clusters with comparisons to original Step 6 codes
  - Easy configuration switching via `EXPERIMENTS` dict

## Configuration

Edit the `ExperimentConfig` at the bottom of `cluster_analysis.py`:

```python
config = ExperimentConfig(
    # Data source
    filename="M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav",
    var_name="Q20",
    sample_size=50,

    # Choose experiment: "baseline", "bigrams", or "strict_filtering"
    tfidf_config=EXPERIMENTS["bigrams"],

    # LLM settings
    description_model="gpt-4.1",
    use_keywords_in_prompt=True,  # Toggle keyword enhancement

    # Display
    n_sample_clusters=5,
    show_comparisons=True,  # Compare with original Step 6 codes
    verbose=True
)
```

## Predefined Experiments

Switch between experiment configurations by changing `EXPERIMENTS["name"]`:

### `"baseline"` - Unigrams only
```python
TfidfConfig(
    max_features=1000,
    ngram_range=(1, 1),  # Single words only
    top_k_keywords=10
)
```

### `"bigrams"` - Unigrams + bigrams (recommended)
```python
TfidfConfig(
    max_features=2000,
    ngram_range=(1, 2),  # Single words + two-word phrases
    top_k_keywords=15
)
```

### `"strict_filtering"` - More aggressive filtering
```python
TfidfConfig(
    max_features=500,
    min_df=3,      # Keyword must appear in 3+ clusters
    max_df=0.6,    # Exclude very common terms
    top_k_keywords=8
)
```

## Adding New Experiments

Add to the `EXPERIMENTS` dict in `cluster_analysis.py`:

```python
EXPERIMENTS = {
    # ... existing experiments ...

    "my_experiment": TfidfConfig(
        max_features=1500,
        ngram_range=(1, 3),  # Include trigrams
        min_df=2,
        max_df=0.7,
        top_k_keywords=12,
        language="nl",
        custom_stopwords=["extra", "stopwords"]  # Optional
    ),
}
```

Then use: `tfidf_config=EXPERIMENTS["my_experiment"]`

## Output Format

The experiment displays:

```
CLUSTER X (n=Y)
────────────────────────────────────────────────────────

TF-IDF Keywords (top 10):
  1. keyword1 (0.XXX)
  2. keyword2 (0.XXX)
  ...

LLM-Generated Description:
  Theme: [Short thematic label]
  Description: [1-2 sentence description]
  Key Concepts: [List of 3-5 concepts]

Sample Ideas (5 of Y):
  • idea 1
  • idea 2
  ...

Original Step 6 Code (for comparison):
  Code: [Original code name]
  Definition: [Original definition]
```

## Comparing Approaches

### With Keywords vs Without Keywords

Toggle `use_keywords_in_prompt` to see the impact:

```python
# Run 1: With keywords
config.use_keywords_in_prompt = True
run_experiment(config)

# Run 2: Without keywords
config.use_keywords_in_prompt = False
run_experiment(config)
```

### Different TF-IDF Configurations

```python
# Run multiple experiments in sequence
for exp_name in ["baseline", "bigrams", "strict_filtering"]:
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"{'='*80}\n")

    config.tfidf_config = EXPERIMENTS[exp_name]
    run_experiment(config)
```

## Integration Into Pipeline

Once you've validated the best approach:

1. **Move utility to production**:
   - Copy `tfidf_analyzer.py` to `src/utils/`
   - Add unit tests

2. **Update Step 6**:
   - Import `TfidfAnalyzer` in `codeGenerator.py`
   - Extract keywords before theme generation
   - Add keywords to cluster data structure

3. **Update prompts**:
   - Modify prompts in `prompts.py` to include keyword context
   - Test with full pipeline

4. **Add configuration**:
   - Add `TfidfConfig` to `config.py`
   - Make keyword extraction optional via feature flag

## Troubleshooting

**Cache not found**:
```bash
# Make sure you've run pipeline to Step 5 first
cd src
# Edit pipeline.py: RUN_UNTIL_STEP = 5
python pipeline.py
```

**LLM errors**:
- Check that your `.env` has valid API credentials
- Ensure `API_PROVIDER` in `config.py` is set correctly ("openai" or "azure")
- Check model availability in your account

**Import errors**:
- Make sure virtual environment is activated
- Run from `src/` directory: `python experiments/cluster_analysis.py`

## Example Workflow

```bash
# 1. Setup (first time)
cd /Users/ruudkooiman/projects/Python_apps/CoderingsTool
source .venv/bin/activate
cd src

# 2. Generate cache if needed
# Edit pipeline.py: RUN_UNTIL_STEP = 5, configure dataset
python pipeline.py

# 3. Run baseline experiment
# Edit experiments/cluster_analysis.py: tfidf_config=EXPERIMENTS["baseline"]
python experiments/cluster_analysis.py

# 4. Run bigrams experiment
# Edit experiments/cluster_analysis.py: tfidf_config=EXPERIMENTS["bigrams"]
python experiments/cluster_analysis.py

# 5. Compare results
# Review console output, assess keyword quality

# 6. Toggle keyword enhancement
# Edit: use_keywords_in_prompt=False
python experiments/cluster_analysis.py
# Edit: use_keywords_in_prompt=True
python experiments/cluster_analysis.py

# 7. Iterate on TfidfConfig parameters as needed
```

## Notes

- **Fast iteration**: Loading from cache is quick (~1-2 seconds)
- **LLM calls**: Description generation takes ~5-30 seconds depending on cluster count
- **Cost**: Each experiment uses ~8-50 LLM calls (depending on cluster count)
- **Reproducibility**: Random sampling is used for display - rerun for different samples
