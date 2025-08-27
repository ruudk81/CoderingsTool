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

### Testing Prompt Chains
```bash
cd src/utils
python promptTester.py
```

### Testing Code Deduplication
```bash
cd src/utils
python codeBookDeduplicator.py
```

### Environment Setup
```bash
# Using pip
pip install -r requirements.txt

# Using conda
conda env create -f environment.yml
conda activate coderingtool
```

## High-Level Architecture

### Pipeline Steps (1-9)
1. **Data Loading** - Import SPSS files via pyreadstat
2. **Preprocessing** - Text normalization, spell checking (Hunspell), finalization
3. **Quality Filtering** - LLM-based filtering of meaningless responses
4. **Idea Extraction** - Segment responses into discrete ideas
5. **Embedding Generation** - OpenAI/Gemini embeddings with caching
6. **Clustering** - UMAP dimensionality reduction + HDBSCAN clustering
7. **Code Generation** - 4-chain LLM prompt system for codebook creation
8. **Theme Identification** - Hierarchical clustering of generated codes
9. **Code Assignment** - Map codes back to original response segments

### Key Architecture Patterns  

#### Data Models (Pydantic)
- **Hierarchical inheritance**: ResponseModel → PreprocessModel → DescriptiveModel → EmbeddingsModel → ClusterModel → LabelModel
- **Type safety** with numpy array support via custom validators
- **Model conversion** methods (e.g., `to_preprocess_model()`) for pipeline progression

#### Async Processing Patterns
- **Consistent async/await** for I/O operations
- **Batch processing** with concurrency limits (aiolimiter) for API calls
- **Instructor library** integration for structured LLM responses
- **Retry logic** with tenacity for API resilience

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
- `pipeline.py` - Main orchestrator with 9-step async pipeline
- `config.py` - All configuration classes and defaults
- `models.py` - Pydantic models with numpy array support
- `prompts.py` - LLM prompt templates for various stages

#### Critical Utils
- `cacheManager.py` - SQLite cache operations
- `codeGenerator.py` - 4-chain prompt system for code generation
- `clusterer.py` - UMAP/HDBSCAN clustering implementation
- `qualityFilter.py` - LLM-based response quality assessment
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

### Code Generation System (codeGenerator.py)
The code generation uses a sophisticated 4-chain prompt system:
1. **Initial Analysis** - Understand cluster themes
2. **Code Generation** - Create initial codebook
3. **Refinement** - Improve code quality and coherence
4. **Finalization** - Ensure consistency and completeness

### Collaboration Workflow
- Our workflow: 
  * Users suggests ideas for modification
  * Claude proposes strategies, plans and todo lists
  * User approves
  * Claude implements and adds, commits and pushes modification to the GitHub repo
  * User pulls to test modification locally

### Important Development Notes
- No formal linting setup - follow existing code style patterns
- Manual testing through verbose reporting and prompt testing
- Extensive backup system in place (check src/utils/old for history)
- Web interface available via Streamlit for interactive testing
- Configuration changes invalidate relevant cache entries automatically

### Prompt Processing Considerations
- Prompts are leading. So pydantic models and processing prompted responses from LLMs down the pipeline in codeGenerator need to facilitate prompts/instructed outputs