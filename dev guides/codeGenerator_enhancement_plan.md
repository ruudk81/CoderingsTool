# CodeGenerator Enhancement Plan

## Executive Summary

This document provides an analysis of codeGenerator.py, a critical component (Step 7) in the pipeline that uses a sophisticated 4-chain LLM prompt system to generate qualitative codes from clustered survey responses.

## Current System Overview

### Architecture
- **File**: `src/utils/codeGenerator.py` (~2600+ lines)
- **Main Class**: `InductiveCodeGenerator`
- **Integration**: Called from pipeline.py after clustering (Step 6)
- **Entry Points**: 
  - `design()` - Async method that runs the pipeline
  - `generate_async()` - Returns `CodeGeneratorReasoningResults` with full transparency
  - `generate()` - Sync wrapper for generate_async()

### Core Design Principles
1. **Theme-Based Similarity Processing**: Clusters grouped by theme embeddings (similarity < 0.7 threshold)
2. **Sequential Batch Processing**: Dissimilarity batches run sequentially to ensure SharedCodebook consistency
3. **Aggressive Within-Batch Parallelism**: Concurrent processing within each dissimilarity batch
4. **Version-Based Codebook Management**: Enables concurrent reads with sequential writes
5. **Dual Processing Paths**: Regular (with monitoring) and unlimited (maximum speed) functions

### Pipeline Architecture

The codeGenerator implements a sophisticated multi-stage pipeline:

#### Stage 1: Theme Extraction
- **Method**: `extract_themes()`
- **Prompt**: `CLUSTER_SUMMARY_PROMPT`
- **Processing**: Parallel extraction of 1-2 themes per cluster
- **Output**: `ClusterSummaryOutput` containing `List[ClusterThemeItem]`
- **Data captured**: `step1_inputs`, `step1_summaries`

#### Stage 2: Theme Similarity Analysis
- **Method**: `create_dissimilarity_batches()`
- **Processing**: 
  - Generate embeddings for themes via `SimilarityEngine`
  - Calculate pairwise cosine similarity matrix
  - Greedy batch formation with 0.7 similarity threshold
- **Output**: Sequential list of dissimilarity batches

#### Stage 3: Batch Processing Strategy
- **Method**: `process_batches_sequentially()`
- **Strategy by batch size**:
  - **Large batches (>10)**: Split into sub-batches of max 10, process concurrently
  - **Medium batches (2-10)**: Process all clusters concurrently
  - **Singletons (1)**: Process individually

#### Stage 4: Code Generation Pipeline (Per Cluster)
Each cluster undergoes a 3-step sequential process:

**4a. Candidate Selection** (`CANDIDATE_CODE_SELECTION_PROMPT`)
- Finds nearest codes using embeddings
- Output: `List[CandidateCode]`
- Data captured: `step2_inputs`, `step2_analysis`

**4b. Code Generation** (`CODE_GENERATION_PROMPT`)
- Decides action: create_new, modify_existing, or use_existing
- Output: `CodeRecommendation` with `List[CodingDecision]`
- Data captured: `step3_inputs`, `step3_recommendations`

**4c. Validation** (`VALIDATION_PROMPT`)
- Validates and updates SharedCodebook
- Output: `ValidationResult` with `List[CodeValidation]`
- Data captured: `step4_inputs`, `step4_validations`

### Key Components

#### Core Classes
- **SharedCodebook**: Thread-safe codebook with async locks and version tracking
  - Tracks codebook evolution for cache management
  - Enables concurrent reads during sequential batch updates
  - Caches embeddings per version to avoid recomputation
- **SimilarityEngine**: Handles embeddings and similarity calculations
  - Generates theme embeddings for dissimilarity batching
  - Finds nearest existing codes for candidate selection
- **CodeDesignerAPIClient**: Rate-limited API client with retry logic
  - Implements tenacity retry patterns
  - Precise token tracking for rate limit compliance
- **WorkloadAnalyzer** (from qualityFilter.py): Calculates optimal processing strategies

#### Processing Paths
1. **Regular functions**: Include rate limiting, monitoring, and throttling
   - `_select_candidate_codes()`, `_generate_code()`, `_validate_and_update_codebook()`
2. **Unlimited functions**: Direct API calls for maximum speed
   - `_select_candidate_codes_unlimited()`, `_generate_code_unlimited()`, `_validate_code_unlimited()`

Both paths implement complete data capture for debugging and transparency.

### Performance Optimizations
- **Dissimilarity Batching**: Minimizes codebook conflicts by processing similar themes separately
- **Embedding Caching**: Stores embeddings per codebook version
- **Rate Limiting Strategy**:
  - `Throttler` for request pacing
  - `SlidingWindowMonitor` for API usage tracking
  - Safety factor: 0.95 (aggressive utilization)
  - Concurrent buffer: 3 seconds
- **Parallel Processing**: Aggressive concurrency within batches while maintaining sequential batch coordination

### Error Handling & Monitoring
- **Retry Logic**: Tenacity-based retry for API failures
- **Graceful Degradation**: Partial failures don't stop entire pipeline
- **VerboseReporter**: Real-time progress tracking and statistics
- **Performance Metrics**: Token usage, processing times, batch analytics

### Configuration and Dependencies
- Uses `DEFAULT_CODEDESIGNER_CONFIG` for model settings
- Integrates with OpenAI API via instructor library for structured outputs
- Supports both OpenAI and Gemini embeddings
- Async processing with asyncio, aiolimiter, and custom throttling
- Model-specific rate limits from config

## Enhancement Phases

### Phase 1: Prompt Engineering & Output Format Optimization
**Status: 🔄 Currently In Progress - Investigation Phase**

#### Objectives
- Improve prompt clarity and effectiveness for better LLM responses
- Optimize output formats for cleaner structured data
- Ensure full Pydantic model validation compatibility
- Update all downstream systems to handle new formats
- Enhance prompt parameter handling and construction

#### Implementation Approach
1. User makes changes to prompts and/or output formats
2. Follow the dependency update checklist: `phase1_prompt_change_checklist.md`
3. Validate that displayResults and promptTester still work correctly
4. Test improved prompts and iterate as needed

#### Success Criteria
- All 4 prompts produce expected output formats without errors
- displayResults shows improved data correctly with no display issues
- promptTester can reconstruct and test new prompts successfully
- No regressions in downstream functionality (pipeline, displayResults, promptTester)

### Phase 2: Code Refactoring & Optimization
**Status: ⏸️ On Hold - Pending Phase 1 Completion**

> **Important**: Phase 2 work should not begin until Phase 1 is fully completed and validated. This ensures that any refactoring is done on the finalized prompt structure and output formats.

#### Objectives
- Remove dead code and unused functions
- Consolidate duplicate functionality
- Improve code efficiency and readability
- Optimize performance bottlenecks
- Enhance maintainability

#### Areas to Investigate

**1. Code Duplication Analysis**
- Regular vs unlimited function variants
- Identify shared logic that can be extracted
- Assess the value of maintaining two paths
- Document differences and use cases

**2. Dead Code Identification**
- Unused imports and variables
- Deprecated functions
- Commented-out code blocks
- Unreachable code paths

**3. Performance Bottlenecks**
- Inefficient loops or data structures
- Redundant API calls
- Unnecessary data transformations
- Memory usage patterns

**4. Code Organization**
- Class structure and responsibilities
- Method complexity and length
- Variable naming consistency
- Documentation coverage

**5. Error Handling Patterns**
- Inconsistent error handling
- Missing try-catch blocks
- Error propagation issues
- Logging improvements

#### Expected Outcomes
- Cleaner, more maintainable codebase
- Reduced code complexity
- Improved performance
- Better error handling
- Enhanced developer experience

#### Implementation Approach
1. Create comprehensive code analysis
2. Prioritize refactoring targets
3. Design consolidation strategy
4. Implement changes incrementally
5. Add unit tests for refactored code
6. Performance benchmark before/after
7. Update documentation

### Investigation Timeline

**Phase 1 Investigation**: 2-3 days (Currently Underway)
- Day 1: Prompt analysis and documentation
- Day 2: Design new formats and assess impact
- Day 3: Create implementation plan

**Phase 2 Investigation**: 3-4 days (To Begin After Phase 1 Completion)
- Day 1-2: Code analysis and duplication mapping
- Day 3: Performance profiling
- Day 4: Create refactoring plan

## Phase Dependencies

⚠️ **Critical Dependency**: Phase 2 must not begin until Phase 1 is fully completed. This sequential approach ensures:
- Refactoring is done on stable, finalized prompt structures
- No wasted effort refactoring code that might change due to Phase 1
- Clear separation of concerns between prompt optimization and code optimization
- Easier debugging and testing of each phase independently