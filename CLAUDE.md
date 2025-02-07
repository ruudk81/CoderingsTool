# CoderingsTool Project Context

## Conversation Summary

This document captures the context and progress from our conversation that began with project initialization and evolved through two major phases of development.

### Phase 1: Project Setup and Cache System Implementation

#### Initial Setup
- **Started:** User requested initialization of CoderingsTool project
- **Key Insight:** User emphasized the importance of presenting plans before implementing (after I initially created files without permission)
- **Language Support:** Corrected to use Dutch as default language with English option for UI
- **Created Files:**
  - README.md - Project documentation
  - requirements.txt - Python dependencies  
  - CLAUDE.md - This context file

#### Cache System Development
- **Problem:** Basic CSV file handling with `if filepath.exists()` checks was unreliable
- **Solution:** Implemented sophisticated SQLite-backed cache system
- **Challenges:** Encountered Windows file locking issues requiring multiple iterations
- **Fix:** Disabled atomic writes on Windows platform specifically
- **Architecture:**
  ```python
  # cache_config.py - Fixed cache directory location
  def get_default_cache_dir():
      src_dir = Path(__file__).parent
      return src_dir.parent / "data" / "cache"  # Use project root, not src
  ```
  ```python
  # cache_manager.py - Platform-specific atomic writes
  import platform
  use_atomic = self.config.use_atomic_writes and platform.system() != 'Windows'
  ```

### Phase 2: Clustering System Improvements (Current)

#### User Requirements for Clustering
1. **Only HDBSCAN** - No alternative clustering algorithms
2. **Fully Automatic** - No user parameter tuning required
3. **Default to Description Embeddings** - Not code embeddings
4. **Micro-clusters for Outliers** - Option C from presented choices
5. **No Visualizations** - Quality metrics only
6. **Preserve Core Logic** - Add enhancements on top, don't change existing clustering

#### TODO List Created and Progress
1. **Simplify ClusteringConfig for Automatic Mode** ✓ COMPLETED
   - Removed manual parameter options
   - Added embedding_type with default "description"
   - Added quality thresholds for automatic decisions
   - Kept only user-relevant options

2. **Implement Quality Metrics Module** ✓ COMPLETED
   - Created cluster_quality.py with test section
   - Calculates silhouette score, noise ratio, mean cluster size
   - Combined quality score with weighted components:
     ```python
     weights = {
         'silhouette': 0.4,
         'coverage': 0.3,
         'noise': 0.3  # Negative weight
     }
     ```

3. **Update Clusterer for Automatic Mode** ✓ COMPLETED
   - Added config integration without changing core logic
   - Quality metrics calculation after clustering
   - Retry mechanism with parameter adjustment if quality < 0.7
   - Micro-clustering for outliers if noise ratio > 9%
   - Test section at bottom for validation

4. **Add Pipeline Integration** ⏳ PENDING
   - Add command-line arguments for clustering options
   - Save quality metrics to cache database
   - Add success/failure reporting to user

5. **Test and Refine** ⏳ PENDING
   - Test with different datasets  
   - Adjust automatic parameter selection
   - Fine-tune quality thresholds

#### Key Implementation Details

**ClusteringConfig (clustering_config.py):**
```python
@dataclass
class ClusteringConfig:
    embedding_type: str = "description"  # User default preference
    language: str = "nl"                 # Dutch default
    min_quality_score: float = 0.3       # Minimum acceptable quality
    max_noise_ratio: float = 0.5         # Maximum acceptable noise
```

**Clusterer Updates (clusterer.py):**
- Added config to constructor: `__init__(self, config: ClusteringConfig = None)`
- Automatic parameter selection based on data size
- Quality evaluation after clustering
- Retry mechanism: "If quality is below 0.7, retry with adjusted parameters"
- Micro-clustering: "If noise ratio exceeds 9%, create micro-clusters from outliers"

### Testing Strategy
User explicitly requested: "And after you have done this, we will test clusterer.py. And if it does we will proceed with further refinement."

Current testing approach:
1. Complete implementation of current TODO
2. Test before proceeding to next TODO
3. Get user confirmation before major changes

### Current Status
- ✓ Completed Phase 2: Clustering Improvements
- ✓ Successfully tested automatic clustering with excellent results
- ✓ Quality metrics working correctly (0.881 quality score achieved)
- ✓ Pipeline integration completed with CLI arguments and metrics database
- ✓ Pipeline successfully runs steps 1-5
- 🔜 Next: Phase 3 - Improve labeling system (step 6)

### Files Modified in Phase 2
1. `/workspaces/CoderingsTool/src/modules/utils/clustering_config.py` - NEW
2. `/workspaces/CoderingsTool/src/modules/utils/cluster_quality.py` - NEW
3. `/workspaces/CoderingsTool/src/modules/utils/clusterer.py` - UPDATED
4. `/workspaces/CoderingsTool/src/pipeline.py` - UPDATED (CLI args, metrics)
5. `/workspaces/CoderingsTool/src/cache_database.py` - UPDATED (metrics table)
6. `/workspaces/CoderingsTool/src/cache_manager.py` - UPDATED (metrics methods)
7. `/workspaces/CoderingsTool/CLAUDE.md` - UPDATED

### Key Principles Established
1. **Always present plans before implementing**
2. **Dutch as default language throughout system**
3. **Test incrementally before proceeding**
4. **Keep existing logic intact, add enhancements on top**
5. **Focus on automatic operation without user parameter tuning**

### Next Immediate Step
Test the current clusterer implementation with the quality metrics and automatic mode before proceeding to TODO #4 (Pipeline Integration).

### Pipeline Usage

#### Command-Line Arguments
```bash
# Basic usage
python pipeline.py

# Clustering configuration
python pipeline.py --embedding-type description  # or 'code'
python pipeline.py --language nl                # or 'en'
python pipeline.py --min-quality-score 0.7      # Quality threshold
python pipeline.py --max-noise-ratio 0.1        # Max noise before micro-clustering

# Cache control
python pipeline.py --force-recalculate          # Recalculate all steps
python pipeline.py --force-step clusters        # Recalculate specific step
python pipeline.py --cleanup                    # Clean old cache files
python pipeline.py --stats                      # Show cache statistics
python pipeline.py --show-metrics               # Display clustering metrics

# Combined example
python pipeline.py --embedding-type code --min-quality-score 0.8 --force-step clusters
```

#### Testing clusterer.py Standalone

To test the updated clusterer in Spyder:

1. **Navigate to the correct directory:**
   ```python
   cd /workspaces/CoderingsTool/src/modules/utils/
   ```

2. **Run clusterer.py directly:**
   ```python
   python clusterer.py
   ```

3. **Or run in Spyder:**
   - Open clusterer.py in Spyder
   - The test section at the bottom will run automatically when you execute the file
   - It will:
     - Load embeddings from cache
     - Create a default ClusteringConfig (description embeddings, Dutch)
     - Run the clustering pipeline
     - Display quality metrics
     - Save results to cache
     - Show cluster summaries

4. **What to look for:**
   - Quality metrics output (silhouette score, noise ratio, etc.)
   - Overall quality score (should be between 0 and 1)
   - Number of meta-clusters found
   - The clustering parameters that were automatically selected
   - Any retry attempts if quality was below 0.7

5. **Customizing the test:**
   You can modify the test section to use different settings:
   ```python
   # Use code embeddings instead of descriptions
   config = ClusteringConfig(embedding_type="code")
   
   # Use English language
   config = ClusteringConfig(language="en")
   ```

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
  
### Step 5: Get Clusters ✓ (Significantly improved)
- Hierarchical clustering (meta/meso/micro)
- **Status**: Fully automatic clustering with quality metrics implemented
- **Achievements**: 
  - Automatic parameter selection based on data size
  - Quality metrics (silhouette, noise ratio, coverage)
  - Micro-clustering for outlier handling
  - Retry logic with parameter adjustment
  - Successfully tested: 0.881 quality score

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
- ✓ Created SQLite-backed cache system
- ✓ Fixed Windows file locking issues
- ✓ Added CLI options for cache control
- ✓ Removed migration code (not needed)

### Phase 2: Clustering Improvements ✓ COMPLETED

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

3. Update Clusterer for Automatic Mode ✓
   - Use simplified config ✓
   - Auto-select parameters based on data size ✓
   - Implement micro-clusters for outliers ✓
   - Calculate and report quality metrics ✓

4. Add Pipeline Integration ✓ COMPLETED
   - Added command-line arguments ✓
   - Save quality metrics to cache database ✓
   - Display metrics and warnings ✓
   - Show retry attempts if any ✓
   - Added --show-metrics option ✓

5. Test and Refine ✓ TESTED
   - Successfully tested with 380 embeddings ✓
   - Achieved 0.881 quality score ✓
   - Created 28 meta-clusters from 791 segments ✓
   - Pipeline runs successfully steps 1-5 ✓

#### Test Results Summary:
- **Quality Score**: 0.881 (excellent)
- **Silhouette Score**: 0.666 (good cluster separation)
- **Coverage**: 91.3% (only 8.7% noise)
- **Clusters**: 791 segments → 58 clusters → 28 meta-clusters
- **Performance**: No retries needed, automatic parameters worked well

### Phase 3: Simplified Clustering and Improved Labeling (Current)

#### Clusterer Simplification ✓ COMPLETED
1. **Removed complex logic** - No more retry mechanism or parameter adjustments
2. **Single clustering pass** - Just dimension reduction → clustering → save
3. **Quality metrics for display only** - Calculate and show metrics but make no decisions
4. **Hardcoded parameters** - Reverted to original UMAP and HDBSCAN settings
5. **Only micro-clusters** - No meta or meso clustering in clusterer
6. **Pipeline compatibility** - Updated pipeline to work with simplified clusterer

#### Labeling System Improvements (TODO)
1. **Update labeller to handle clustering hierarchy**
   - Take micro-clusters from simplified clusterer
   - Use LLM to label each cluster
   - Use LLM to identify semantically similar clusters
   - Create merge/remap dictionary
   - Apply merging to create meta-clusters

2. **Smart semantic merging**
   - LLM analyzes cluster labels and content
   - Identifies clusters that should be merged
   - Returns mapping dictionary
   - Preserves cluster IDs for traceability

3. **Domain-aware labeling**
   - Already uses var_lab for context ✓
   - Prompts adapt to survey question ✓

#### Key Changes Made
1. **Clusterer renamed** - `simple_clusterer.py` → `clusterer.py`
2. **Class renamed** - `SimpleClusterGenerator` → `ClusterGenerator`
3. **Pipeline updated** - Works with simplified clusterer, no config needed
4. **Debug output** - Shows micro-clusters instead of meta-clusters

### Future Phases
1. Phase 4: Add proper error handling throughout
2. Phase 5: Implement results display (step 7)
3. Phase 6: Add data visualization
4. Phase 7: Create export options

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

## Current Focus: Simplified Clustering and Smart Labeling
- Simplified clusterer with hardcoded parameters (Phase 3)
- Single-pass clustering with quality metrics for display only
- Labeller handles all semantic intelligence
- LLM-based cluster merging based on semantic similarity
- Domain context derived from var_lab
- Pipeline runs steps 1-5 successfully, step 6 being improved