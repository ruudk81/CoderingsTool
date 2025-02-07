# CoderingsTool Project Context

## Conversation Summary

This document captures the context and progress from our conversation that began with project initialization and evolved through two major phases of development.

### Initial Setup
- **Started:** User requested initialization of CoderingsTool project
- **Key Insight:** User emphasized the importance of presenting plans before implementing (after I initially created files without permission)
- **Language Support:** Corrected to use Dutch as default language with English option for UI
- **Created Files:**
  - README.md - Project documentation
  - requirements.txt - Python dependencies  
  - CLAUDE.md - This context file

### Key Principles Established
1. **Always present plans before implementing**
2. **Dutch as default language throughout system**
3. **Test incrementally before proceeding**
4. **Keep existing logic intact, add enhancements on top**
5. **Focus on automatic operation without user parameter tuning**

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
  
### Step 5: Get Clusters ✓ (Completed)
- Reduce dimensions of embedding with UMAP
- Cluster with HDBSCAN
- Create initial micro-clusters

### Step 6: Merge Similar Clusters ✓ (Completed)
- Merge HDBSCAN's initial micro-clusters using LLM evaluation
- Identify clusters that are not meaningfully differentiated
- Focus on research question relevance
- Preserve traceability to original clusters
- Cache merge mapping for next phase

### TODO
- Step 7a: Label merged micro-clusters
  - Create descriptive labels for the 38 merged clusters
  - Focus on distinctive aspects relevant to research question
  
- Step 7b: Generate hierarchical structure
  - Create meta (themes) and meso (topics) levels
  - Organize micro-clusters into logical groupings
  - Ensure mutual exclusivity within levels


**Guiding Principles**
1. Develop robust code - write modular code, work with structured models including the labeller (see models.py), validate input and output of models (use pydantic) and prompts for LLMs have a response format (use instructor)
2. Adhere to previous code - build classes for each phase, including the labeller
3. Be efficient and quick - batch and run batches async with asyncio if possible

**Job to be Done**
We segmented responses to a survey question and clustered these segments by their code descriptions, particularly: embeddings of these description with HBDscan. We did not constrain segmentation with HBDscan, so we have many clusters. Some of the clusters are not meaningfully differentiated from the viewpoint of the research question. Our task is to merge initial clusters that are not meaningfully differentiated, after which we organise these merged clusters in a 3-level hierarchy. Themes = level-1 node; Topics=level-2 node; Codes= level-3 node. The merged clusters are level-3 node. In completing this job, we cannot lose track of how the merged and new clusters can be mapped back on the original clusters given to us by the clusterer.

**Overall Workflow**

The labeller follows a multi-phase approach to create hierarchical labels:

1. **✅ COMPLETED - Phase 1: Merge initial clusters that are not meaningfully differentiated**
   - Merged initial clusters with high cosine similarity (based on descriptive codes and code descriptions)
   - Used LLM to determine which clusters should be merged based on semantic similarity
   - Applied semantic merging to create differentiated groups
   - Tracked cluster-to-group mappings throughout
   - Reduced from 52 initial clusters to 38 merged clusters (26.9% reduction)
   
   **Key Improvements to LLM Merge Decisions:**
   - Focused the LLM on the research question (var_lab) as the primary context
   - Presented 5 most representative descriptive codes per cluster (based on cosine similarity to centroid)
   - Framed the decision task explicitly: "Are these clusters meaningfully differentiated in how they answer the research question?"
   - Batched multiple cluster pair evaluations for efficiency (reduced API calls)
   - Structured prompt with clear decision criteria focused on the var_lab perspective
   - Achieved semantically meaningful merges (e.g., salt-related, portion size, variety clusters)

2. **Phase 2: Initial Label Generation (NEXT)**
   - Extract descriptive codes and descriptions from the 38 merged clusters
   - Generate initial labels for each merged cluster using LLM
   - Design prompts to focus on the research question context
   - Create distinctive and descriptive labels based on cluster content

3. **Phase 3: Hierarchical Organization**
   - Organize merged clusters into 3-level hierarchy
   - Create meta-level categories (e.g., "1", "2", "3")
   - Create meso-level subcategories (e.g., "1.1", "1.2")
   - Assign micro-level identifiers (e.g., "1.1.1", "1.1.2")
   - Ensure meaningful differentiation at each level
   - Validate mutual exclusivity within levels

4. **Phase 4: Theme Summarization**
   - Generate summaries for each theme
   - Explain how each theme addresses the research question
   - Finalize the hierarchical structure

**Optimization Strategies**

1. **Performance Optimizations**
   - Async LLM calls with concurrency limits
   - Batch processing of clusters (10-20 at a time)
   - Embedding-based pre-filtering to reduce LLM calls
   - Caching for repeated operations

2. **Quality Optimizations**
   - Use centroid embeddings for representative sampling
   - LLM scoring for merging decisions
   - Iterative refinement of labels
   - Context-aware prompting with var_lab

3. **Architecture Patterns**
   - Async/await throughout
   - Configurable settings class
   - Structured data models for each phase
   - Clear separation of concerns


## Architecture Notes


**Key Architectural Patterns from Existing Code**

Based on examination of `segmentDescriber.py`, `embedder.py` and `models.py`:

1. **Class Structure**
   - Main class with configuration in `__init__`
   - Use of config parameters with defaults
   - Clear separation of sync and async methods
   - Builder pattern for complex chains (LangChain)

2. **Async Patterns**
   - Use `AsyncOpenAI` client for async operations
   - `async def` for methods that perform I/O
   - `await` for async calls
   - `asyncio.run()` to run async methods from sync context
   - Batch processing with concurrency control
   - Retry logic for resilience

3. **Data Models (Pydantic)**
   - Inherit from appropriate base models
   - Use `Optional` for nullable fields
   - `ConfigDict(arbitrary_types_allowed=True)` for numpy arrays
   - Clear hierarchy: Base → Submodel → Model
   - Type hints throughout

4. **LLM Integration**
   - Use `instructor` library for structured output
   - Patch OpenAI client: `instructor.from_openai(AsyncOpenAI())`
   - Response models with Pydantic
   - Error handling and retries
   - Token management for batching

5. **Batch Processing**
   - Calculate token budgets
   - Create batches based on size limits
   - Process batches concurrently
   - Maintain index mappings

6. **Prompting**
   - Separate prompts file
   - Template substitution for dynamic values
   - Structured response format
   - Context-aware prompting



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

**✅ COMPLETED: Phase 1 - Cluster Merging**
- Implemented Phase2Merger class in phase2_merger.py
- Created efficient batch processing for LLM cluster comparison
- Developed research question-focused similarity evaluation
- Used representative items based on cosine similarity to centroid
- Created merge groups and remap dictionary
- Applied merging to consolidate similar clusters
- Saved merge mapping (with Dict[int, int] of old-to-new cluster IDs)
- Added merge rationale tracking for transparency
- Achieved significant efficiency through batched LLM calls

**Results Summary:**
- Processed 52 initial micro-clusters
- Merged into 38 final clusters (26.9% reduction)
- Created 8 multi-cluster groups and preserved 30 singleton clusters
- Meaningful merges included:
  - Salt-related clusters (groups of 4 and 3 clusters)
  - Portion size clusters (3 clusters)
  - Variety-focused clusters (3 clusters)
  - Health-related clusters (3 clusters)
  - Vegetable-related clusters (2 clusters)
  - Satisfaction/continuation clusters (2 clusters)
  - Freshness-focused clusters (2 clusters)
- Preserved important distinctions between price vs. affordability, salt vs. sugar reduction, etc.
- Successfully cached merge mapping for next pipeline phase

**TODO 2: Implement Initial Label Generation for Merged Clusters**
- Generate descriptive labels for the 38 merged clusters from Phase 1
- Extract representative content from each merged cluster
- Use LLM to create clear, distinctive labels
- Ensure labels reflect the research question context
- Load merge mapping from cache to work with the updated cluster structure
- Design prompt to focus on the most semantically central content in each merged cluster

**TODO 3: Implement Hierarchical Structure Creation**  
- Organize merged clusters into 3-level hierarchy
- Create meta-level categories (e.g., "1", "2", "3")
- Create meso-level subcategories (e.g., "1.1", "1.2")
- Assign micro-level identifiers (e.g., "1.1.1", "1.1.2")
- Use LLM to ensure logical grouping through var_lab lens

**TODO 4: Implement Label Refinement and Theme Summarization**  
- Ensure labels are mutually exclusive within each level
- Optimize labels for clarity and distinctiveness
- Maintain alignment with var_lab context
- Refine labels to maximize differentiation
- Generate theme summaries explaining research question relevance
- Hierarchical context awareness (meta→meso→micro)
- Original cluster details included for micro-level refinement

**TODO 5: Update Main run_pipeline Method**  
- Execute 4-phase pipeline sequentially
- Phase 1: Merge similar micro-clusters
- Phase 2: Generate labels for merged clusters
- Phase 3: Create 3-level hierarchy
- Phase 4: Refine labels and generate theme summaries
- Convert results to LabelModel format
- Comprehensive progress tracking and statistics
- Processing time measurement

