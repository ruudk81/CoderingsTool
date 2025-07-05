# CoderingsTool Project Context  

## Project Overview

CoderingsTool is a sophisticated text analysis pipeline for processing survey responses from SPSS files. The system performs text preprocessing, quality filtering, embedding generation, clustering, and hierarchical labeling of open-ended survey responses to identify themes and patterns.

### Pipeline Progress

Steps 1-5: 
1. **✅ Data Loading** - SPSS (.sav) files → ResponseModel objects
2. **✅ Preprocessing** - Text normalization, spell checking, finalization  
3. **✅ Segmentation** - Quality filtering, descriptive code generation
4. **✅ Embeddings** - OpenAI-based code and description embeddings
5. **✅ Initial clustering** - Two-phase HDBSCAN clustering with LLM-based merging producing micro clusters
6: **🔄 IN PROGRESS** - Hierarchical labbeling


## Key Architecture Patterns  

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

## Development Workflow

### Collaboration Rules
- **Claude develops**: Claude develops, refines and updates code, which will be added, committed and pushed to the GitHub repo
- **User tests**: The user will pull the updates and test code in Spyder, locally
- **Iterative refinement**: Based on test results, Claude will update and improve the code

### Workflow Guidelines
- Add, commit and push modifications to the GitHub repo when completed a task and have modified a .py module or util