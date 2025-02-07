# Plan to Build the Labeller

## Overview
The ThematicLabeller implements a 4-phase hierarchical labeling system for survey response clusters, creating a structured Theme → Topic → Keyword hierarchy. The implementation in `labeller v2.py` provides a solid foundation that closely aligns with the proposed workflow.

## Current State Analysis

### Strengths of Current Implementation
1. **Proper Pydantic Models**: Well-defined data structures for each phase
2. **LangChain Integration**: Clean chain-based approach for LLM calls
3. **Async Architecture**: Concurrent processing capabilities
4. **Caching System**: Intermediate results can be cached
5. **MapReduce Support**: Handles large numbers of clusters
6. **Retry Logic**: Robust error handling with exponential backoff

### Areas Needing Development
1. **Prompts**: All prompts referenced but not yet created
2. **Configuration Integration**: Config classes need to be added to config.py
3. **Testing**: No test coverage yet
4. **Refinement Phase**: Currently basic, needs LLM-based refinement option
5. **Progress Tracking**: Could be more detailed
6. **Validation**: Input/output validation could be stronger

## Implementation Plan

### Phase 1: Configuration & Prompts Setup

#### 1.1 Update config.py
```python
class LabellerConfig:
    """Configuration for hierarchical labelling"""
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 4000
    seed: int = 42
    api_key: Optional[str] = None
    language: str = "English"
    top_k_representatives: int = 3
    map_reduce_threshold: int = 30
    batch_size: int = 10
    assignment_threshold: float = 0.7
    use_llm_refinement: bool = False  # For Phase 4 enhancement
```

#### 1.2 Create prompts.py entries
Create structured prompts for each phase:
- `PHASE1_FAMILIARIZATION_PROMPT`: Label individual clusters
- `PHASE2_DISCOVERY_SINGLE_PROMPT`: Single-call hierarchy building
- `PHASE2_DISCOVERY_MAP_PROMPT`: MapReduce map step
- `PHASE2_DISCOVERY_REDUCE_PROMPT`: MapReduce reduce step
- `PHASE3_ASSIGNMENT_PROMPT`: Assign clusters to hierarchy
- `PHASE4_REFINEMENT_PROMPT`: Optional LLM refinement

### Phase 2: Core Functionality Enhancements

#### 2.1 Improve Representative Selection
- Add diversity sampling to avoid redundant representatives
- Include outlier detection for better cluster characterization
- Add option to use different similarity metrics

#### 2.2 Enhance MapReduce Logic
- Implement smarter batch creation (by semantic similarity)
- Add intermediate validation between map/reduce phases
- Implement parallel batch processing with controlled concurrency

#### 2.3 Strengthen Assignment Logic
- Implement hierarchical consistency validation
- Add confidence scoring for assignments
- Create fallback strategies for low-confidence assignments

### Phase 3: Integration & Pipeline Updates

#### 3.1 Update pipeline.py
- Add Step 6 for hierarchical labeling
- Integrate with existing cache manager
- Add progress reporting

#### 3.2 Model Compatibility
- Ensure proper conversion from ClusterModel to LabelModel
- Validate Theme/Topic/Keyword field formats
- Add summary generation for each response

### Phase 4: Testing & Validation

#### 4.1 Unit Tests
- Test each phase independently
- Mock LLM responses for deterministic testing
- Test edge cases (empty clusters, single cluster, etc.)

#### 4.2 Integration Tests
- End-to-end pipeline testing
- Cache hit/miss scenarios
- Large dataset handling (MapReduce trigger)

#### 4.3 Validation Suite
- Hierarchy consistency checks
- Assignment coverage validation
- Output format verification

### Phase 5: Advanced Features

#### 5.1 LLM-Based Refinement (Phase 4 Enhancement)
- Implement consistency checking across all labels
- Add mutual exclusivity validation
- Standardize terminology across hierarchy

#### 5.2 Adaptive Thresholding
- Dynamic threshold adjustment based on cluster characteristics
- Confidence-based assignment strategies
- Outlier handling for unclassifiable clusters

#### 5.3 Multi-Language Support
- Enhance prompt templates for language flexibility
- Add language-specific validation
- Test with Dutch and English datasets

### Phase 6: Performance & Optimization

#### 6.1 Concurrency Optimization
- Batch API calls efficiently
- Implement rate limiting
- Add progress bars with tqdm

#### 6.2 Memory Management
- Stream large datasets
- Implement chunked processing
- Optimize embedding storage

#### 6.3 Caching Strategy
- Implement partial result caching
- Add cache versioning
- Create cache cleanup utilities

## Implementation Timeline

### Week 1: Foundation
- [ ] Create all prompt templates
- [ ] Update config.py with LabellerConfig
- [ ] Write comprehensive unit tests
- [ ] Implement basic pipeline integration

### Week 2: Core Features
- [ ] Enhance representative selection
- [ ] Improve MapReduce implementation
- [ ] Strengthen assignment logic
- [ ] Add validation suite

### Week 3: Advanced Features
- [ ] Implement LLM-based refinement
- [ ] Add adaptive thresholding
- [ ] Enhance multi-language support
- [ ] Create integration tests

### Week 4: Polish & Optimization
- [ ] Optimize performance
- [ ] Add comprehensive logging
- [ ] Create documentation
- [ ] Run full pipeline tests

## Key Design Decisions

### 1. Hierarchy Structure
- **Themes**: Integer IDs (1, 2, 3...)
- **Topics**: Float IDs encoding hierarchy (1.1, 1.2, 2.1...)
- **Keywords**: Original micro-cluster IDs preserved

### 2. Assignment Strategy
- Probability-based assignments with configurable thresholds
- Support for "other" category when confidence is low
- Hierarchical consistency enforcement

### 3. Scalability Approach
- Single LLM call for ≤30 clusters (maintains context)
- MapReduce for >30 clusters (handles scale)
- Configurable thresholds for flexibility

### 4. Error Handling
- Graceful degradation with fallback labels
- Retry logic with exponential backoff
- Comprehensive error logging

## Success Metrics

1. **Coverage**: >95% of clusters successfully labeled
2. **Consistency**: Hierarchical assignments logically consistent
3. **Performance**: <5 minutes for 100 clusters
4. **Quality**: Manual validation shows >90% accuracy
5. **Robustness**: Handles edge cases without crashes

## Next Steps

1. **Immediate**: Create prompt templates in prompts.py
2. **Short-term**: Update config.py and write tests
3. **Medium-term**: Enhance core functionality
4. **Long-term**: Implement advanced features and optimizations

This plan provides a clear roadmap for completing the hierarchical labeller while building on the solid foundation already in place.