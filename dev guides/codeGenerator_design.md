# InductiveCodeGenerator Detailed Implementation Document

## Overview
InductiveCodeGenerator is a sophisticated text analysis system that processes clustered survey ideas through a 4-stage pipeline to generate, validate, and maintain a shared codebook. It uses theme-based similarity batching with sequential batch processing and aggressive parallelism within batches.

## Architecture Overview

### Core Design Principles
1. **Theme-Based Similarity Processing**: Clusters grouped by theme embeddings (similarity < 0.7)
2. **Sequential Batch Processing**: Dissimilarity batches run sequentially for SharedCodebook consistency
3. **Aggressive Within-Batch Parallelism**: Concurrent processing within dissimilarity batches
4. **Version-Based Codebook Management**: Thread-safe SharedCodebook with version tracking
5. **Instructor-Enhanced API Calls**: AsyncOpenAI + instructor for structured outputs

### Main Class
```python
class InductiveCodeGenerator:
    """CodeDesigner: Theme-based similarity processing with 4-stage pipeline"""
    
    def __init__(
        self,
        cluster_results: List[models.ClusterModel],
        starter_codes: List[Dict[str, str]], 
        var_lab: str,
        verbose: bool = False,
        prompt_printer = None,
        config = None,
        **kwargs
    )
```

### Data Models (from models.py)

```python
# Stage 1: Theme Extraction
class ClusterThemeItem(BaseModel):
    theme_id: int = Field(description="Theme identifier (1, 2, etc.)")
    theme_name: str = Field(description="Short phrase clarifying the theme")
    summary: str = Field(description="≤25 words describing cluster contents")

class ClusterSummaryOutput(BaseModel):
    themes: List[ClusterThemeItem] = Field(description="Array of themes extracted from cluster")

# Stage 2: Candidate Selection  
class CandidateCode(BaseModel):
    code: str
    definition: str

# Stage 3: Code Generation
class CodingDecision(BaseModel):
    theme_id: int
    decision: str = Field(description="'create_new', 'modify_existing', or 'use_existing'")
    justification: str
    action_details: ActionDetails

class CodeRecommendation(BaseModel):
    cluster_analysis: ClusterAnalysis
    coding_decisions: List[CodingDecision]
    overall_justification: str

# Stage 4: Validation
class CodeValidation(BaseModel):
    theme_id: int
    decision: str = Field(description="'APPROVE', 'REVISE', 'REJECT', or 'SPLIT'")
    decision_rationale: str
    validated_code: ValidatedCode
    confidence_score: float

class ValidationResult(BaseModel):
    theme_assessment: ThemeAssessment
    code_validations: List[CodeValidation]
    overall_validation: OverallValidation
```

## Pipeline Architecture

### Entry Points
- `design()` → `List[Dict[str, Any]]`: Main async pipeline execution
- `generate_async()` → `CodeGeneratorReasoningResults`: Returns complete reasoning data
- `generate()` → `CodeGeneratorReasoningResults`: Sync wrapper for generate_async()

### Stage 1: Theme Extraction
- **Method**: `extract_themes()`
- **Input**: Clustered ideas from ClusterModel objects
- **Processing**: Parallel theme extraction using `_extract_single_theme()`
- **Prompt**: `CLUSTER_SUMMARY_PROMPT`
- **Output**: `Dict[int, ClusterSummaryOutput]` mapping cluster_id to themes
- **Data Capture**: `step1_inputs`, `step1_summaries`

### Stage 2: Theme Similarity Analysis
- **Method**: `create_dissimilarity_batches()`
- **Input**: Theme data from Stage 1
- **Processing**:
  1. Generate embeddings for themes via `SimilarityEngine`
  2. Calculate pairwise cosine similarity matrix
  3. Greedy batch formation with similarity threshold (0.7)
  4. Report similarity distribution statistics
- **Output**: `List[List[int]]` - Sequential list of dissimilarity batches

### Stage 3: Batch Processing Strategy
- **Method**: `process_batches_sequentially()`
- **Processing Strategy**:
  - **Large batches (>10)**: Split into sub-batches, process concurrently
  - **Medium batches (2-10)**: Process all clusters concurrently  
  - **Singletons (1)**: Process individually
- **Sub-batch creation**: `_create_sub_batches()` for rate limit compliance

### Stage 4: Code Generation Pipeline (Per Cluster)
Each cluster goes through a 3-step sequential process:

#### Step 4a: Candidate Selection
- **Methods**: `_select_candidate_codes()` or `_select_candidate_codes_unlimited()`
- **Processing**:
  1. Find nearest existing codes using embeddings
  2. Apply `CANDIDATE_CODE_SELECTION_PROMPT`
- **Output**: `List[CandidateCode]`
- **Data Capture**: `step2_inputs`, `step2_analysis`

#### Step 4b: Code Generation
- **Methods**: `_generate_code()` or `_generate_code_unlimited()`
- **Processing**: Apply `CODE_GENERATION_PROMPT` with candidates
- **Output**: `CodeRecommendation` with coding decisions
- **Data Capture**: `step3_inputs`, `step3_recommendations`

#### Step 4c: Validation
- **Methods**: `_validate_and_update_codebook()` or `_validate_code_unlimited()`
- **Processing**: 
  1. Apply `VALIDATION_PROMPT`
  2. Update SharedCodebook based on validation
- **Output**: `ValidationResult`
- **Data Capture**: `step4_inputs`, `step4_validations`

## Key Components

### SharedCodebook
```python
class SharedCodebook:
    """Thread-safe shared codebook with async lock and version tracking"""
    
    async def get_current_snapshot(self) -> Tuple[List[Dict[str, str]], int]
    async def add_code_if_new(self, code: str, definition: str) -> Tuple[bool, int]
    async def replace_code(self, original_code: str, new_code: str, new_definition: str) -> Tuple[bool, int]
    async def batch_update(self, new_codes: List[Dict[str, str]], expected_base_version: int) -> bool
    async def cache_embeddings(self, version: int, embeddings: List[np.ndarray])
```

### SimilarityEngine
```python
class SimilarityEngine:
    """Handles embeddings and similarity calculations"""
    
    async def embed_themes(self, themes: Dict[int, ClusterSummaryOutput]) -> Dict[int, np.ndarray]
    def find_nearest_codes(self, query_embedding: np.ndarray, codebook_embeddings: List[np.ndarray], 
                          codebook: List[Dict[str, str]], top_k: int = 5) -> List[Dict]
```

### CodeDesignerAPIClient
```python
class CodeDesignerAPIClient:
    """API client with intelligent retry logic and precise rate limiting"""
    
    async def make_request(self, task_coro, task_info: str, prompt: str = None)
```

### WorkloadAnalyzer (from qualityFilter.py)
```python
def calculate_optimal_strategy(self, total_batches: int, avg_tokens_per_batch: float, 
                             sub_batches_per_batch: int = 1) -> OptimalStrategy
```

## Processing Flow Details

### Cluster Data Preparation
```python
clusters = {}
for result in self.cluster_results:
    ideas_list = result.response_ideas or []
    for idea in ideas_list:
        if idea.initial_cluster is not None and idea.initial_cluster != -1:
            cluster_id = idea.initial_cluster
            if cluster_id not in clusters:
                clusters[cluster_id] = {
                    'cluster_id': cluster_id,
                    'ideas': [],
                    'embeddings': [],
                    'respondent_ids': []
                }
            clusters[cluster_id]['ideas'].append(idea.idea)
            clusters[cluster_id]['embeddings'].append(idea.idea_embedding)
```

### Rate Limiting Strategy
- Uses `WorkloadAnalyzer` to calculate optimal processing rates
- `Throttler` for rate limiting
- `SlidingWindowMonitor` for tracking API usage
- Safety factor: 0.95 (aggressive utilization)
- Concurrent buffer: 3 seconds

### Data Capture for Transparency
All prompt inputs and outputs are captured:
- `step1_inputs/summaries`: Theme extraction data
- `step2_inputs/analysis`: Candidate selection data  
- `step3_inputs/recommendations`: Code generation data
- `step4_inputs/validations`: Validation data

### Performance Optimizations
1. **Batching Strategy**: Dissimilarity batching minimizes codebook conflicts
2. **Parallel Processing**: Aggressive parallelism within batches
3. **Sequential Coordination**: Between batches for consistency
4. **Version-Based Updates**: Allows concurrent reads with sequential writes
5. **Embedding Caching**: Avoids recomputing embeddings for existing codes

## Output Format

### CodeGeneratorReasoningResults
```python
class CodeGeneratorReasoningResults(BaseModel):
    # Raw results
    cluster_results: List[Dict[str, Any]]
    
    # Prompt inputs (for transparency)
    step1_inputs: Dict[int, Dict[str, Any]]
    step2_inputs: Dict[int, Dict[str, Any]]  
    step3_inputs: Dict[int, Dict[str, Any]]
    step4_inputs: Dict[int, Dict[str, Any]]
    
    # Step outputs
    step1_summaries: Dict[int, Dict[str, Any]]
    step2_analysis: Dict[int, List[Dict[str, str]]]
    step3_recommendations: Dict[int, Dict[str, Any]]
    step4_validations: Dict[int, Dict[str, Any]]
    
    # Metadata
    cluster_assignments: Dict[int, Dict[str, Any]]
    stats: Dict[str, Any]
    codebook: List[Dict[str, str]]
    processing_timestamp: str
```

## Implementation Notes

### Two Processing Paths
1. **Regular functions**: Include rate limiting and monitoring
2. **Unlimited functions**: Direct API calls for maximum speed

Both paths capture data properly for debugging and transparency.

### Error Handling
- Tenacity retry logic for API failures
- Graceful degradation on partial failures
- Comprehensive error reporting via VerboseReporter

### Monitoring & Reporting
- Real-time progress tracking
- Similarity distribution statistics
- Performance metrics (tokens, time, rates)
- Batch processing analytics

This architecture maximizes throughput while maintaining codebook consistency through strategic sequential/concurrent processing coordination.