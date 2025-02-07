# CoderingsTool Project Context

## Project Overview
CoderingsTool is a text analysis pipeline for processing open-ended survey responses from SPSS files. It performs
text preprocessing, quality filtering, embedding generation, clustering, and thematic labeling of qualitative data.

**Current Status: Work in Progress**

## Pipeline Steps and Status

### Step 1: Get Data ✓ (Functional)
- **Current**: Loads survey responses from SPSS (.sav) files
- **Issue**: CSV save/load logic with `if filepath.exists()` is not robust in the pipeline

### Step 2: Preprocess Data ✓ (Functional)
- Text normalization
- Spell checking (Dutch/English)
- Text finalization

### Step 3: Describe and Segment Data ✓ (Functional)
- Quality filtering
- Segment description generation
 
### Step 4: Get Embeddings ✓ (Functional)
- Generates code embeddings
- Generates description embeddings
- Combines embeddings
- Working with OpenAI API
  
### Step 5: Get Clusters ⚠️ (Initial phase)
- Hierarchical clustering (meta/meso/micro)
- **Status**: First implementation exists but needs refinement
- **TODO**: Improve clustering algorithm and parameters

### Step 6: Get Labels ⚠️ (Initial phase)
- Thematic labeling of clusters
- **Status**: Basic implementation, needs significant work
- **TODO**: Refine labeling logic and prompt engineering

### Step 7: Display Results ❌ (Not implemented)
- Visualization of results
- Export functionality
- **Status**: Completely missing
- **TODO**: Implement from scratch (will be done last)

## Known Issues

### CSV Data Caching (RESOLVED)
- ✓ Replaced basic file existence checks with SQLite-backed cache system
- ✓ Implemented atomic writes with temporary files
- ✓ Added cache validation with timestamps and configuration hashes
- ✓ Created command-line options for cache control

### Language Support
- UI language support implemented (Dutch default, English available)
- Data processing currently handles Dutch and English text
- Need to ensure consistency between UI and data processing languages

## Architecture Notes

### Data Flow
1. SPSS file → ResponseModel
2. ResponseModel → PreprocessModel
3. PreprocessModel → DescriptiveModel (with segments)
4. DescriptiveModel → EmbeddingsModel
5. EmbeddingsModel → ClusterModel
6. ClusterModel → LabelModel
7. LabelModel → Display/Export (not implemented)

### Key Files
- `pipeline.py`: Main processing logic (steps 1-6)
- `models.py`: Data models for each stage
- `app.py`: Streamlit interface (step 7 placeholder)
- `modules/utils/`: Processing utilities
- `prompts.py`: Centralized LLM prompts

## Development Priorities

### Phase 1: Cache System ✓ COMPLETED
- ✓ Analyzed pipeline.py CSV handling issues
- ✓ Designed SQLite-based cache strategy
- ✓ Created cache_config.py with configuration classes
- ✓ Implemented cache_database.py for SQLite operations
- ✓ Built cache_manager.py to replace csvHandler
- ✓ Updated pipeline.py to use CacheManager
- ✓ Added command-line arguments for cache control
- ✓ Fixed Windows file locking issues
- ✓ Fixed cache directory location
- ✓ Removed unnecessary migration code
- ⏳ Create tests for cache system (deferred)

### Phase 2: Clustering Improvements (CURRENT)

#### Clustering TODO List:
1. Simplify ClusteringConfig for Automatic Mode ✓
   - Remove manual parameter options ✓
   - Add embedding_type with default "description" ✓
   - Add quality thresholds for automatic decisions ✓
   - Keep only user-relevant options ✓

2. Implement Quality Metrics Module ✓
   - Create cluster_quality.py ✓
   - Calculate silhouette score, noise ratio, mean cluster size ✓
   - Return quality report dictionary ✓

3. Update Clusterer for Automatic Mode ⏳
   - Use simplified config
   - Auto-select parameters based on data size
   - Implement micro-clusters for outliers
   - Calculate and report quality metrics

4. Add Pipeline Integration ⏳
   - Add command-line arguments
   - Save quality metrics to cache
   - Add success/failure reporting

5. Test and Refine ⏳
   - Test with different datasets
   - Adjust automatic parameter selection
   - Fine-tune quality thresholds

### Future Phases
1. Phase 3: Improve labeling system (step 6)
2. Phase 4: Add proper error handling throughout
3. Phase 5: Implement results display (step 7)
4. Phase 6: Add data visualization
5. Phase 7: Create export options

## Environment Requirements
- Python 3.8+
- `OPENAI_API_KEY` environment variable required
- Hunspell dictionaries for Dutch and English

## Testing Notes
- Pipeline can run end-to-end but outputs need verification
- Steps 5 and 6 produce results but quality needs improvement
- No visualization of results yet

## Important Context for Development
- This is an active work in progress
- Core pipeline exists but needs refinement
- UI is set up but not connected to all pipeline functions
- Focus should be on improving existing steps before adding new features

## New Cache System Features
- SQLite database for cache metadata tracking
- Command-line arguments:
  - `--force-recalculate`: Force recalculation of all steps
  - `--force-step STEP_NAME`: Force specific step recalculation
  - `--cleanup`: Clean up old cache files
  - `--stats`: Show cache statistics
- Automatic cache invalidation based on configuration changes
- Atomic file writes to prevent corruption
- Processing time tracking for performance monitoring