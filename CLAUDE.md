# CoderingsTool Project Context

## Project Overview

CoderingsTool is a sophisticated text analysis pipeline for processing survey responses from SPSS files. The system performs text preprocessing, quality filtering, embedding generation, clustering, and hierarchical labeling of open-ended survey responses to identify themes and patterns.

## Current Status

### Working Pipeline (Steps 1-5) ✅
The core pipeline successfully processes data through 5 stages:

1. **Data Loading** - SPSS (.sav) files → ResponseModel objects
2. **Preprocessing** - Text normalization, spell checking, finalization  
3. **Segmentation** - Quality filtering, descriptive code generation
4. **Embeddings** - OpenAI-based code and description embeddings
5. **Clustering** - Two-phase HDBSCAN clustering with LLM-based merging

### Broken/Incomplete (Step 6+) ❌
- **Step 6: Hierarchical Labeling** - Import errors, missing phase modules
- **File Organization** - Multiple backup files, inconsistent naming
- **Model Inconsistencies** - Typos and data structure issues

## Architecture Patterns

### Data Models (Pydantic)
- **Hierarchical inheritance**: ResponseModel → PreprocessModel → DescriptiveModel → EmbeddingsModel → ClusterModel → LabelModel
- **Type safety** with numpy array support via `ConfigDict(arbitrary_types_allowed=True)`
- **Model conversion** methods (`to_model()`) for pipeline progression
- **Structured validation** throughout the pipeline

### Async Processing Patterns
- **Consistent async/await** for I/O operations
- **Batch processing** with concurrency limits for API calls
- **Instructor library** integration for structured LLM responses
- **Error handling** and retry logic for resilience

### Caching System
- **SQLite-backed** cache with CSV data storage
- **Configuration-aware** cache invalidation based on processing parameters
- **Atomic operations** with platform compatibility
- **Step-by-step** pipeline caching for efficient reruns

### Configuration Management
- **Centralized config** classes (CacheConfig, ProcessingConfig, ClusteringConfig)
- **Environment variable** support for API keys
- **Parameter validation** and hashing for cache keys

## Current Implementation Details

### Working Components

**Step 1: Data Loading** (`data_io.DataLoader`)
- Loads SPSS files using pandas/pyreadstat
- Extracts specific variables with respondent IDs
- Creates ResponseModel objects with proper validation
- Robust error handling for missing/corrupt data

**Step 2: Preprocessing** (Multiple utilities)
- `textNormalizer`: Unicode normalization, case handling
- `spellChecker`: Hunspell-based Dutch/English correction
- `textFinalizer`: Final cleanup and validation
- Quality filtering integrated early in process

**Step 3: Segmentation** (`qualityFilter` + `segmentDescriber`)
- `qualityFilter.Grader`: LLM-based response quality assessment
- `segmentDescriber`: Segments responses and generates descriptive codes
- Creates DescriptiveModel with segment-level codes and descriptions

**Step 4: Embeddings** (`embedder.Embedder`)
- OpenAI API integration with async batch processing  
- Generates both code embeddings and description embeddings
- Combines embeddings for clustering input
- Token management and rate limiting

**Step 5: Clustering** (`clusterer` + `clusterMerger3`)
- **Phase 5a**: Initial clustering with `clusterer.ClusterGenerator`
  - UMAP dimensionality reduction
  - HDBSCAN clustering for micro-clusters
  - Quality metrics and noise filtering
- **Phase 5b**: Cluster merging with `clusterMerger3.ClusterMerger`
  - LLM-based semantic similarity evaluation
  - Research question-focused merging decisions
  - Merge mapping preservation for traceability

### Broken Components

**Step 6: Hierarchical Labeling** (`labeller.py` - BROKEN)
- Attempts to import missing phase modules:
  - `labeller_phase1_labeller.py` - exists but import fails
  - `labeller_phase2_organizer.py` - exists but import fails  
  - `labeller_phase3_summarizer.py` - exists but import fails
- Complex 4-phase labeling approach that's not fully integrated
- Should create 3-level hierarchy: Themes → Topics → Codes

### Data Model Issues

**Field Inconsistencies:**
- Uses `mirco_cluster` (typo) instead of `micro_cluster` in ClusterModel
- Inconsistent cluster ID storage (sometimes dict, sometimes int)
- Missing proper validation for hierarchical label structures

**Import Path Issues:**
- Hardcoded Windows paths in pipeline.py
- Inconsistent module imports across files
- Missing __init__.py imports for utils modules

## File Organization Issues

### Redundant Files (Need Cleanup)
- `clusterMerger - oud1.py` (old version)
- `clusterMerger - oud2.py` (old version)  
- `clusterMerger3.py` (current working version)
- `cluster_quality.py` vs `cluster_quality2.py`

### Missing Integration
- `ClusteringConfig` class exists but not used in `clusterer.py`
- Hardcoded parameters instead of configuration-driven approach
- Phase labeller modules exist but import incorrectly

## Dependencies & Integrations

### Working Dependencies
- **OpenAI API**: Embeddings and LLM completions via instructor
- **HDBSCAN + UMAP**: Clustering algorithms with sklearn integration
- **Spacy**: NLP processing for text analysis
- **Hunspell**: Multi-language spell checking (Dutch/English)
- **SQLite**: Caching backend with pandas integration
- **Pydantic**: Data validation and model management

### Configuration Variables
- `OPENAI_API_KEY`: Required for embeddings and LLM operations
- File paths: Configurable via config.py
- Language settings: Dutch (nl) and English (en) support
- Batch processing limits: Configurable concurrency and token limits

## Immediate Priorities

### 1. Directory Restructuring (HIGH PRIORITY)
**Goal**: Simplify directory structure by moving utils up one level and removing modules folder

**Sub-tasks:**
- **1a**: Verify no actual "modules" are used (only utils)
- **1b**: Move `src/modules/utils/` → `src/utils/`
- **1c**: Remove empty `src/modules/` folder
- **1d**: Update all import references from `modules.utils` → `utils`
- **1e**: Fix hardcoded Windows paths in pipeline.py

### 2. File Cleanup (HIGH PRIORITY)
**Goal**: Keep only essential files in src/ folder

**Target files to keep:**
- `app.py`, `prompts.py`, `config.py`, `ui_text.py`, `models.py`, `pipeline.py`, `cache_manager.py`

**Sub-tasks:**
- **2a**: Identify which .py files in src/ are actually used
- **2b**: Remove redundant/unused files from src/ root
- **2c**: Clean up backup files (`clusterMerger - oud1.py`, etc.)

### 3. Configuration Consolidation (HIGH PRIORITY)
**Goal**: Consolidate all configuration into single config.py file

**Sub-tasks:**
- **3a**: Move `clustering_config.py` contents into `config.py`
- **3b**: Move `cache_config.py` contents into `config.py` 
- **3c**: Update all imports to use unified config
- **3d**: Remove separate config files after migration

### 4. Cache Storage Verification (MEDIUM PRIORITY)
**Goal**: Ensure Step 5 clustering output is properly cached

**Sub-tasks:**
- **4a**: Verify ClusterModel objects are correctly serialized/deserialized
- **4b**: Check merge_mapping cache storage and retrieval
- **4c**: Validate cache invalidation works correctly for clustering step

### 5. Fix Step 6 Labeling (MEDIUM PRIORITY)
**Goal**: Complete the pipeline with working hierarchical labeling

**Sub-tasks:**
- **5a**: Fix import errors in labeller.py
- **5b**: Either fix existing phase modules or create simplified version
- **5c**: Ensure 3-level hierarchy generation (Themes → Topics → Codes)
- **5d**: Integrate with updated directory structure

### 4. Pipeline Robustness (ONGOING)
- Add comprehensive error handling throughout
- Improve logging and debugging output
- Add validation checks between pipeline steps

## Development Principles

### Code Quality Standards
1. **Modular Design**: Each step as separate, testable component
2. **Type Safety**: Pydantic models with proper validation
3. **Async Efficiency**: Batch processing for external API calls
4. **Configuration-Driven**: Avoid hardcoded parameters
5. **Error Resilience**: Graceful handling of failures with retries

### Testing Strategy
- **Unit tests** for individual utility modules
- **Integration tests** for pipeline steps
- **End-to-end tests** for complete pipeline
- **Cache validation** for data persistence

### Performance Optimization
- **Caching**: Aggressive caching to avoid recomputation
- **Batching**: Efficient API usage with batch processing
- **Async operations**: Non-blocking I/O for external services
- **Memory management**: Efficient handling of large datasets

## Research Context

### Survey Analysis Goal
The tool processes open-ended survey responses to identify themes and patterns. The specific research question (var_lab) guides the clustering and labeling process to ensure meaningful differentiation of response categories.

### Clustering Approach
- **Micro-clusters**: Initial fine-grained HDBSCAN clusters
- **Semantic merging**: LLM-based evaluation of cluster similarity
- **Research question focus**: Merging decisions based on research relevance
- **Hierarchical organization**: 3-level structure for thematic analysis

### Quality Assurance
- **Response filtering**: Remove meaningless/low-quality responses
- **Embedding quality**: Use both code and description embeddings
- **Cluster validation**: Quality metrics to assess clustering effectiveness  
- **Label validation**: Ensure meaningful, mutually exclusive categories

## Next Steps

1. **Immediate**: Fix step 6 labeling to complete the pipeline
2. **Short-term**: Clean up file organization and fix model inconsistencies
3. **Medium-term**: Integrate proper configuration management
4. **Long-term**: Add comprehensive testing and optimization

The codebase demonstrates sophisticated architecture with good separation of concerns, but needs focused effort on completion and cleanup to achieve full functionality.