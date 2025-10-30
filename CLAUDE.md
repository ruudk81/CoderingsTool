# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# CoderingsTool Project Context  

## Project Overview

CoderingsTool is a sophisticated text analysis pipeline for processing survey responses from SPSS files. The system performs text preprocessing, quality filtering, embedding generation, clustering, and hierarchical labeling of open-ended survey responses to identify themes and patterns.

## Common Development Commands

### Running the Pipeline
```bash
cd src
python pipeline.py
```

### Running the Web Interface
```bash
cd src
python app.py
# Or use streamlit directly:
streamlit run app.py
```

### Environment Setup
```bash
# Using pip
pip install -r requirements.txt

# Using conda
conda env create -f environment.yml
conda activate coderingtool
```

## Quick Testing Configuration

For rapid validation after code changes, use this standard "cheap test" configuration:

### Standard Test Parameters
```python
QUICK_TEST_CONFIG = {
    "filename": "M250480 Associatiemonitor ASN Bank net databestand.sav",
    "id_column": "DLNMID",
    "var_name": "Qd1_combined",
    "sample_size": 20,  # Small sample for fast testing
}
```

**When to use:**
- After refactoring or code changes
- Before committing changes
- Testing new features
- Verifying bug fixes

**Benefits:**
- **Time**: Minutes instead of hours
- **Cost**: ~$0.10-0.50 instead of $5-20 for full dataset
- **Speed**: Quick validation without burning excessive resources

**Usage:**
```bash
export SAMPLE_SIZE=20
cd src && python pipeline.py
```

**Note:** This is NOT for final validation - always test with full dataset before production use.

## High-Level Architecture

### Pipeline Steps (0-9)
0. **Data Loading** - Import SPSS files via pyreadstat
1. **Preprocessing** - Text normalization, spell checking (Hunspell), finalization
2. **Quality Filtering** - LLM-based filtering of meaningless responses
3. **Idea Extraction** - Segment responses into discrete ideas
4. **Embedding Generation** - OpenAI/Gemini embeddings with caching
5. **Initial Clustering** - UMAP dimensionality reduction + HDBSCAN clustering
6. **Codebook Generation** - 4-chain LLM prompt system for codebook creation
7. **Codebook Refinement** - Hierarchical clustering of generated codes into themes
8. **Code Assignment** - Map codes back to original response segments
9. **Export** - Export to Excel 

### Key Architecture Patterns  

#### Data Models (Pydantic)
- **Hierarchical inheritance**: ResponseModel → PreprocessModel → DescriptiveModel → EmbeddingsModel → ClusterModel → LabelModel
- **Type safety** with numpy array support via custom validators
- **Model conversion** methods (e.g., `to_preprocess_model()`) for pipeline progression

#### Async Processing Patterns
- **Aggressive prompt processing** within model limits as defined in config.py.
- **Tokens** : token bucket + calculation of allowed token usage per minute (TPM).
- **Requests** : allowed rate of requests per minute (RPM).
- **RPM/TPM calculation**  using three probes (define what the “3 probes” are).
- **Rate limiting** : semaphore for concurrency, worker pool size, and smooth throttling via aiolimiter.
- **Adaptive timeouts**  based on p95 of the last 100 samples (samples = observed end-to-end latency).
- **Retry logic**  with exponential backoff via tenacity.

#### Unified Configuration System  
- **Single config.py file** with all configuration classes:
  - `CacheConfig` - Cache management settings
  - `ProcessingConfig` - Processing parameters that affect cache validity
  - `ClusteringConfig` - Automatic clustering parameters
  - `CacheDatabase` - SQLite database operations class
- **Environment variable** support for API keys and paths
- **Default instances** ready to use: `DEFAULT_CACHE_CONFIG`, `DEFAULT_PROCESSING_CONFIG`, `DEFAULT_CLUSTERING_CONFIG`

#### Caching System
- **SQLite-backed** cache with CSV data storage
- **Configuration-aware** cache invalidation
- **Step-by-step** pipeline caching for efficient reruns
- **Version tracking** for embeddings and processing steps

### Key Files and Their Roles

#### Core Pipeline Files
- `pipeline.py` - Main orchestrator with 10-step async pipeline (steps 0-9)
- `config.py` - All configuration classes and defaults
- `models.py` - Pydantic models with numpy array support
- `prompts.py` - LLM prompt templates for various stages

#### Critical Utils
- `cacheManager.py` - SQLite cache operations
- `spellChecker.py` - Hunspell + ai system for correcting spelling mistakes
- `qualityFilter.py` - LLM-based response quality assessment of responses
- `ideaExtractor.py` - LLM-Based system of segmenting multiple responses into single responses as "ideas expressed in light of the survey question"
- `embedder.py`  - Getting Openai/Gemini embeddings of extracted ideas
- `clusterer.py` - UMAP/HDBSCAN clustering implementation
- `codeGenerator.py` - 4-chain prompt system for code generation
- `codeAssigner.py` - LLM-based assigner of codes to individual "ideas expressed" 
- `verboseReporter.py` - Detailed progress reporting system

### Dependencies & Integrations
- **OpenAI API**: Embeddings and completions via instructor
- **Google Gemini**: Alternative embedding provider
- **HDBSCAN + UMAP**: Clustering algorithms
- **Spacy**: NLP processing (nl_core_news_lg, en_core_web_sm)
- **Hunspell**: Multi-language spell checking (Dutch/English)
- **SQLite**: Caching backend
- **Pydantic v2**: Data validation with numpy support
- **Streamlit**: Web interface framework

### Collaboration Workflow
- Our workflow:
  * Users suggests ideas for modification
  * Claude proposes strategies, plans and todo lists
  * User approves
  * Claude implements and adds, commits and pushes modification to the GitHub repo
  * User pulls to test modification locally

### Development Philosophy

**Avoid Backward Compatibility During Development**
- We are in active development, not production
- Backward compatibility creates technical debt and code cluttering
- Complex codebases become impossible to untangle during debugging
- When refactoring, replace old patterns completely - don't maintain both
- Clean, single-path solutions over redundant compatibility layers
- Remove legacy code rather than maintaining parallel implementations

