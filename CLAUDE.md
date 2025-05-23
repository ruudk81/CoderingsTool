# CoderingsTool Project Context  

## Project Overview

CoderingsTool is a sophisticated text analysis pipeline for processing survey responses from SPSS files. The system performs text preprocessing, quality filtering, embedding generation, clustering, and hierarchical labeling of open-ended survey responses to identify themes and patterns.

### ✅ Working Pipeline (Steps 1-6) 
The core pipeline successfully processes data through 6 complete stages:

1. **✅ Data Loading** - SPSS (.sav) files → ResponseModel objects
2. **✅ Preprocessing** - Text normalization, spell checking, finalization  
3. **✅ Segmentation** - Quality filtering, descriptive code generation
4. **✅ Embeddings** - OpenAI-based code and description embeddings
5. **✅ Clustering** - Two-phase HDBSCAN clustering with LLM-based merging
6. **✅ Hierarchical Labeling** - 3-phase LLM-based Theme→Topic→Keyword hierarchy

### 🔧 Refinement Needed
- **Step 6**: Refine code, investigate opportunities to improve performance in terms of speed, investigate opportunities to improve the quality of the prompts

## Project Structure

```
CoderingsTool/
├── data/                          # Input data files
├── src/
│   ├── __init__.py               # Package initialization
│   ├── app.py                    # Streamlit web application
│   ├── cache_manager.py          # Cache management system
│   ├── config.py                 # **UNIFIED** configuration (all configs merged)
│   ├── models.py                 # Pydantic data models
│   ├── pipeline.py               # Main processing pipeline
│   ├── prompts.py                # LLM prompts
│   ├── ui_text.py                # UI text constants
│   └── utils/                    # **MOVED** from modules/utils/
│       ├── analyze_labels.py
│       ├── clusterMerger.py      
│       ├── cluster_quality.py     
│       ├── clusterer.py
│       ├── csvHandler.py
│       ├── data_io.py
│       ├── embedder.py
│       ├── labeller.py            
│       ├── preprocessor.py
│       ├── qualityFilter.py
│       ├── segmentDescriber.py
│       ├── spellChecker.py
│       ├── textFinalizer.py
│       └── textNormalizer.py
└── requirements.txt
```

## Key Architecture Patterns (Unchanged)

### Data Models (Pydantic)
- **Hierarchical inheritance**: ResponseModel → PreprocessModel → DescriptiveModel → EmbeddingsModel → ClusterModel → LabelModel
- **Type safety** with numpy array support
- **Model conversion** methods for pipeline progression

### Async Processing Patterns
- **Consistent async/await** for I/O operations
- **Batch processing** with concurrency limits for API calls
- **Instructor library** integration for structured LLM responses

### Unified Configuration System  
- **Single config.py file** with all configuration classes:
  - `CacheConfig` - Cache management settings
  - `ProcessingConfig` - Processing parameters that affect cache validity
  - `ClusteringConfig` - Automatic clustering parameters
  - `CacheDatabase` - SQLite database operations class
- **Environment variable** support for API keys and paths
- **Default instances** ready to use: `DEFAULT_CACHE_CONFIG`, `DEFAULT_PROCESSING_CONFIG`, `DEFAULT_CLUSTERING_CONFIG`

### Caching System
- **SQLite-backed** cache with CSV data storage
- **Configuration-aware** cache invalidation
- **Step-by-step** pipeline caching for efficient reruns

### Dependencies & Integrations
- **OpenAI API**: Embeddings and LLM completions via instructor
- **HDBSCAN + UMAP**: Clustering algorithms
- **Spacy**: NLP processing
- **Hunspell**: Multi-language spell checking (Dutch/English)
- **SQLite**: Caching backend
- **Pydantic**: Data validation and models

### Running the Pipeline
```bash
cd src
python pipeline.py
```

### Configuration Usage
All configuration is now centralized:
```python
from config import DEFAULT_CACHE_CONFIG, DEFAULT_PROCESSING_CONFIG, DEFAULT_CLUSTERING_CONFIG
from config import CacheConfig, ProcessingConfig, ClusteringConfig, CacheDatabase
```
