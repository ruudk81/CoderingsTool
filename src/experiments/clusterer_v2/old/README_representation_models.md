# BERTopic Representation Models - Implementation Complete ✅

## Overview

This directory contains experimental implementations of BERTopic-inspired representation models for keyword extraction from cluster analysis results. All implementations are fully working and tested.

## What's Been Implemented

### Phase 1: c-TF-IDF Foundation ✅
- **Class-based TF-IDF** (`representation/ctfidf_representation.py`)
  - BERTopic's core keyword extraction algorithm
  - BM25 weighting for improved short text handling
  - Bigram support for multi-word keywords
  - Successfully tested on real data (13 clusters, 401 vocabulary)

### Phase 1.5: Enhanced Prompts ✅
- **Experimental Prompts** (`experiments/prompts.py`)
  - Incorporated Braun & Clarke (2006) thematic analysis methodology
  - Atomic theme guidance and constraints
  - Statistical keywords integration
  - Reflexivity framing for qualitative research rigor

### Phase 2: Advanced Representation Models ✅
- **MMR (Maximal Marginal Relevance)** (`representation/mmr_representation.py`)
  - Balances relevance with diversity
  - Prevents redundant similar keywords
  - Configurable diversity parameter (0.0-1.0)
  - Word co-occurrence based similarity

- **KeyBERT-inspired** (`representation/keybert_representation.py`)
  - Embedding-based keyword selection
  - Uses cluster centroid for semantic alignment
  - Combines c-TF-IDF relevance with embedding similarity
  - Configurable weight parameter for balance

- **LLM-enhanced** (`representation/llm_representation.py`)
  - GPT-based keyword refinement
  - Consolidates synonyms and improves clarity
  - Provides justifications for keyword selection
  - Fallback to c-TF-IDF on errors

### Phase 3: Comparison Framework ✅
- **Comparison Script** (`representation_comparison.py`)
  - Side-by-side comparison of all models
  - Metrics: coverage, avg keywords, diversity
  - Excel export with multiple sheets
  - Visual comparison tables

### Testing ✅
- **Test Suite** (`test_representation_models.py`)
  - All 4 models tested and passing
  - Synthetic data for quick validation
  - Integration test ready

## Files Created

```
src/experiments/
├── representation/
│   ├── __init__.py                    # Module exports (updated)
│   ├── base.py                         # BaseRepresentation interface (existing)
│   ├── ctfidf_representation.py        # c-TF-IDF (Phase 1 ✅)
│   ├── mmr_representation.py           # MMR diversity (Phase 2 ✅)
│   ├── keybert_representation.py       # Embedding-based (Phase 2 ✅)
│   └── llm_representation.py           # LLM-enhanced (Phase 2 ✅)
│
├── prompts.py                          # Enhanced experimental prompts (Phase 1.5 ✅)
├── representation_comparison.py        # Comparison framework (Phase 3 ✅)
├── test_representation_models.py       # Test suite (✅)
└── README_representation_models.md     # This file

cluster_analysis.py                     # Main experiment (uses c-TF-IDF ✅)
```

## How to Use

### 1. Test All Models

```bash
cd src/experiments
python test_representation_models.py
```

Expected output:
```
c-TF-IDF             ✅ PASSED
MMR                  ✅ PASSED
KeyBERT              ✅ PASSED
LLM                  ✅ PASSED

🎉 All tests passed!
```

### 2. Run cluster_analysis.py with c-TF-IDF

```python
from experiments.cluster_analysis import run_experiment, ExperimentConfig

config = ExperimentConfig(
    filename="your_data.sav",
    var_name="Q20",
    sample_size=50,
    n_sample_clusters=None,  # Display all clusters
    max_ideas_per_cluster=10,
    description_model="gpt-4.1-mini",
    verbose=True
)

run_experiment(config)
```

### 3. Compare All Models

```python
from experiments.representation_comparison import compare_all_models, ComparisonConfig

config = ComparisonConfig(
    filename="your_data.sav",
    var_name="Q20",
    sample_size=50,
    n_sample_clusters=10,
    export_excel=True
)

results = compare_all_models(config=config)
```

This will:
- Run all 5 models (Standard TF-IDF, c-TF-IDF, MMR, KeyBERT, LLM)
- Display side-by-side keyword comparisons
- Calculate comparison metrics
- Export to Excel: `exports/representation_comparison.xlsx`

### 4. Use Individual Models

```python
from experiments.representation.ctfidf_representation import CTfidfRepresentation
from experiments.representation.mmr_representation import MMRRepresentation
from experiments.representation.keybert_representation import KeyBERTRepresentation
from experiments.representation.llm_representation import LLMRepresentation

# c-TF-IDF
ctfidf = CTfidfRepresentation(top_k=15, bm25_weighting=True)
keywords = ctfidf.extract_keywords(clusters, verbose=True)

# MMR (diversity)
mmr = MMRRepresentation(diversity=0.3, top_k=10)
# ... (requires c-TF-IDF matrix first)

# KeyBERT (embeddings)
keybert = KeyBERTRepresentation(top_k=10, weight=0.5)
# ... (requires c-TF-IDF matrix + optional embeddings)

# LLM (GPT-enhanced)
llm = LLMRepresentation(model="gpt-4.1-mini", top_k=10)
# ... (requires c-TF-IDF matrix)
```

## Model Comparison

| Model | Pros | Cons | Use When |
|-------|------|------|----------|
| **c-TF-IDF** | Fast, statistical, no API costs | May include redundant similar keywords | Baseline, need speed |
| **MMR** | Diverse keywords, avoids redundancy | Still statistical, no semantic understanding | Want variety, reduce overlap |
| **KeyBERT** | Semantic understanding, cluster-aligned | Requires embeddings, API costs | Need semantic relevance |
| **LLM** | Highest quality, consolidates synonyms | Expensive, slower, requires API | Best quality, willing to pay |

## Performance Characteristics

- **c-TF-IDF**: ~1s for 13 clusters
- **MMR**: ~1s (after c-TF-IDF)
- **KeyBERT**: ~3-5s (embedding generation)
- **LLM**: ~30-60s (LLM calls per cluster)

## Configuration Options

### c-TF-IDF
```python
CTfidfRepresentation(
    top_k=15,                      # Number of keywords
    bm25_weighting=True,           # Use BM25 (recommended)
    reduce_frequent_words=True,    # Square root transform
    ngram_range=(1, 2),            # Unigrams + bigrams
    min_df=1,                      # Minimum cluster frequency
    max_df=0.95                    # Maximum cluster frequency
)
```

### MMR
```python
MMRRepresentation(
    diversity=0.3,                 # 0.0 = max diversity, 1.0 = max relevance
    top_k=10,
    candidate_multiplier=3         # Get 3x candidates before MMR
)
```

### KeyBERT
```python
KeyBERTRepresentation(
    top_k=10,
    embedding_model="text-embedding-3-large",
    weight=0.5,                    # 0.0 = pure c-TF-IDF, 1.0 = pure embeddings
    candidate_multiplier=3
)
```

### LLM
```python
LLMRepresentation(
    model="gpt-4.1-mini",         # Or gpt-4.1, gpt-5
    top_k=10,
    candidate_multiplier=2,
    max_ideas_sample=10,           # Ideas to send to LLM for context
    verbose=True
)
```

## Test Results

All models tested successfully:

```
c-TF-IDF:
  • service (0.6200)
  • great (0.5767)
  • staff (0.5767)

MMR (diversity=0.3):
  • staff (0.5918)
  • great (0.5918)
  • excellent (0.5918)
  • service (0.5062)

KeyBERT (weight=0.5):
  • service (0.9565)
  • excellent (0.9391)
  • good (0.8205)

LLM (gpt-4.1-mini):
  • service (1.0000)
  • staff (0.9000)
  • helpful (0.8000)
```

## Future Enhancements

Potential additions (not yet implemented):

1. **Hierarchical keywords** - Multi-level extraction
2. **Cross-cluster analysis** - Identify overlaps
3. **Dynamic evolution tracking** - Keyword changes over segments
4. **Automatic cluster splitting** - Use keyword diversity to detect multi-theme clusters
5. **Pipeline integration** - Add as optional Step 5.5

## Dependencies

All models use existing CoderingsTool dependencies:
- `scikit-learn` - TF-IDF, vectorization, similarity
- `numpy` - Array operations
- `openai` - Embeddings (KeyBERT), LLM calls (LLM model)
- `instructor` + `pydantic` - Structured LLM outputs

No additional packages required!

## Architecture Design

All code follows experimental-only design:
- ✅ Isolated in `/src/experiments`
- ✅ No changes to production pipeline
- ✅ No changes to `src/utils/`, `src/models.py`, or `src/config.py`
- ✅ Standalone and testable
- ✅ Ready for future pipeline integration if validated

## Contact

For questions or issues:
- Check existing plan file: `/Users/ruudkooiman/.claude/plans/groovy-floating-moth.md`
- Review test output: `python test_representation_models.py`
- Run comparison: `python representation_comparison.py`

---

**Status**: ✅ COMPLETE - All Phase 2 BERTopic models implemented and tested!
