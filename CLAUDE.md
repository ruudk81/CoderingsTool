# CoderingsTool Project Context - Updated

## Project Overview

CoderingsTool is a sophisticated text analysis pipeline for processing survey responses from SPSS files. The system performs text preprocessing, quality filtering, embedding generation, clustering, and hierarchical labeling of open-ended survey responses to identify themes and patterns.

## Current Status - December 2024

### ✅ Working Pipeline (Steps 1-5) 
The core pipeline successfully processes data through 5 stages:

1. **✅ Data Loading** - SPSS (.sav) files → ResponseModel objects
2. **✅ Preprocessing** - Text normalization, spell checking, finalization  
3. **✅ Segmentation** - Quality filtering, descriptive code generation
4. **✅ Embeddings** - OpenAI-based code and description embeddings
5. **✅ Clustering** - Two-phase HDBSCAN clustering with LLM-based merging

### ❌ Incomplete (Step 6+)
- **Step 6: Hierarchical Labeling** - Import errors, needs fixing after restructuring

### 🔧 Recently Completed: Major Restructuring

We just completed a comprehensive codebase restructuring to simplify the architecture and eliminate redundancies.

## Recent Achievements (December 2024)

### ✅ 1. Directory Restructuring (COMPLETED)
**Goal**: Simplify directory structure by moving utils up one level and removing modules folder

**Completed Sub-tasks:**
- **✅ 1a**: Verified no actual "modules" are used (only utils)
- **✅ 1b**: Moved `src/modules/utils/` → `src/utils/`
- **✅ 1c**: Removed empty `src/modules/` folder and empty module files
- **✅ 1d**: Updated all import references from `modules.utils` → `utils`
- **✅ 1e**: Fixed hardcoded Windows paths in pipeline.py and app.py

**Results**: Clean directory structure with utils directly under src/

### ✅ 2. File Cleanup (COMPLETED)
**Goal**: Keep only essential files in src/ folder

**Completed Sub-tasks:**
- **✅ 2a**: Identified which .py files in src/ are actually used
- **✅ 2b**: All files in src/ root determined to be needed (no redundant files)
- **✅ 2c**: Cleaned up backup files:
  - Removed `clusterMerger - oud1.py`, `clusterMerger - oud2.py` (old backups)
  - Removed `cluster_quality2.py` (unused alternative implementation)
  - Removed duplicate `clustering_config.py` from src/ root (kept utils version)

**Results**: 8 essential files in src/ root, 20 clean utility files in src/utils/

### ✅ 3. Configuration Consolidation (COMPLETED)
**Goal**: Consolidate all configuration into single config.py file

**Completed Sub-tasks:**
- **✅ 3a**: Moved all config classes into unified `config.py`:
  - `CacheConfig` and `ProcessingConfig` classes from cache_config.py
  - `CacheDatabase` class from cache_database.py
  - `ClusteringConfig` class from utils/clustering_config.py
  - All dependencies and imports properly merged
- **✅ 3b**: Updated all imports across codebase to use unified config
- **✅ 3c**: Removed separate config files after successful migration

**Results**: Single config.py file (647 lines) with all configuration classes and settings

## Current Project Structure

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
│       ├── clusterMerger3.py     # Current cluster merger (backups removed)
│       ├── cluster_quality.py    # Quality metrics (alternative removed)
│       ├── clusterer.py
│       ├── csvHandler.py
│       ├── data_io.py
│       ├── embedder.py
│       ├── labeller.py           # **NEEDS FIXING** after restructuring
│       ├── labeller_phase1_labeller.py
│       ├── labeller_phase2_organizer.py
│       ├── labeller_phase3_summarizer.py
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

### Unified Configuration System (**NEW**)
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

## Next Immediate Priorities

### 4. Cache Storage Verification (NEXT PRIORITY)
**Goal**: Ensure Step 5 clustering output is properly cached after restructuring

**Sub-tasks:**
- **4a**: Verify ClusterModel objects are correctly serialized/deserialized
- **4b**: Check merge_mapping cache storage and retrieval
- **4c**: Validate cache invalidation works correctly for clustering step
- **4d**: Test pipeline runs correctly with new structure

### 5. Fix Step 6 Labeling (HIGH PRIORITY)
**Goal**: Complete the pipeline with working hierarchical labeling

**Sub-tasks:**
- **5a**: Fix import errors in labeller.py after directory restructuring
- **5b**: Either fix existing phase modules or create simplified version
- **5c**: Ensure 3-level hierarchy generation (Themes → Topics → Codes)
- **5d**: Integrate with updated directory structure and unified config

## Development Context

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

## Testing Status

### ✅ Spyder Testing Completed Successfully

**Tested and Confirmed Working:**
1. ✅ All imports work correctly with new structure
2. ✅ Pipeline runs through Step 5 successfully  
3. ✅ Configuration consolidation functions properly
4. ✅ No broken dependencies from the restructuring

**Practical Optimizations Made:**
- **API Rate Limiting**: Reduced `max_concurrent_requests` from 10 → 5 in `clusterMerger.py` to prevent OpenAI API runtime errors
- **File Naming**: Cleaned up `clusterMerger3.py` → `clusterMerger.py` for better naming consistency
- **Pipeline Refinements**: Made improvements to `pipeline.py` (94 lines changed, net cleanup)

**Confirmed Working Components:**
- Directory restructuring (`src/utils/` structure)
- Unified configuration system (`config.py`)
- Import path updates (`modules.utils` → `utils`)
- Cache system and pipeline execution
- API integration with proper rate limiting

## Current Status After Testing

The restructured codebase is **fully functional** and ready for the next development phase. All major restructuring goals have been achieved and tested successfully in the development environment.

**Next Development Focus**: Fix and optimize the labeller (Step 6) to complete the full pipeline.