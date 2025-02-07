# Configuration Migration Plan

## Overview
This document outlines the plan to migrate hardcoded configuration values in the CoderingsTool pipeline to centralized configuration classes in `config.py`.

## Current Status

### ✅ Completed
1. Created configuration classes in `config.py`:
   - `SpellCheckConfig` - For spell checking parameters
   - `QualityFilterConfig` - For quality filtering settings
   - `SegmentationConfig` - For segmentation/description parameters
   - `EmbeddingConfig` - For embedding generation settings

2. Added default instances:
   - `DEFAULT_SPELLCHECK_CONFIG`
   - `DEFAULT_QUALITY_FILTER_CONFIG`
   - `DEFAULT_SEGMENTATION_CONFIG`
   - `DEFAULT_EMBEDDING_CONFIG`

3. Included all parameters identified in the hardcoded params report

### ❌ Pending
1. Refactor utility classes to use configuration objects
2. Update pipeline.py to pass configurations
3. Test the refactored code

## Migration Steps

### Phase 1: Update Utility Classes (High Priority)

#### 1. Update spellChecker.py
```python
# Current
class SpellChecker:
    def __init__(self, verbose=False):
        self.retry_delay = 2  # hardcoded
        self.max_retries = 3  # hardcoded

# Target
class SpellChecker:
    def __init__(self, config: SpellCheckConfig = None, verbose=False):
        self.config = config or DEFAULT_SPELLCHECK_CONFIG
        self.retry_delay = self.config.retry_delay
        self.max_retries = self.config.retries
```

#### 2. Update qualityFilter.py
```python
# Remove GraderConfig class, use QualityFilterConfig instead
# Update Grader class to accept config parameter
```

#### 3. Update segmentDescriber.py
```python
# Update SegmentDescriber to accept SegmentationConfig
# Remove hardcoded values in LangChainPipeline
```

#### 4. Update embedder.py
```python
# Update Embedder to accept EmbeddingConfig
# Use config.batch_size instead of hardcoded 100
# Use config.max_concurrent_requests instead of hardcoded 5
```

### Phase 2: Update Pipeline.py

```python
# Step 2 - Preprocessing
spell_check_config = DEFAULT_SPELLCHECK_CONFIG
spell_checker = spellChecker.SpellChecker(config=spell_check_config, verbose=VERBOSE)

# Step 3 - Quality Filtering
quality_config = DEFAULT_QUALITY_FILTER_CONFIG
grader = qualityFilter.Grader(preprocessed_text, var_lab, config=quality_config, verbose=VERBOSE)

# Step 3 - Segmentation
segmentation_config = DEFAULT_SEGMENTATION_CONFIG
encoder = segmentDescriber.SegmentDescriber(config=segmentation_config, verbose=VERBOSE)

# Step 4 - Embeddings
embedding_config = DEFAULT_EMBEDDING_CONFIG
get_embeddings = embedder.Embedder(config=embedding_config, verbose=VERBOSE)
```

### Phase 3: Advanced Features

1. **Environment Variable Overrides**
```python
# Allow runtime configuration via environment variables
if os.getenv("SPELLCHECK_BATCH_SIZE"):
    DEFAULT_SPELLCHECK_CONFIG.batch_size = int(os.getenv("SPELLCHECK_BATCH_SIZE"))
```

2. **Configuration Profiles**
```python
# Create different configs for dev/prod
DEV_CONFIG = SpellCheckConfig(batch_size=5, max_retries=1)
PROD_CONFIG = SpellCheckConfig(batch_size=20, max_retries=3)
```

3. **Dynamic Configuration**
```python
# Allow users to override specific parameters
custom_config = SpellCheckConfig(
    batch_size=10,
    max_concurrent_requests=3
)
```

## Benefits

1. **Centralized Control**: All configuration in one place
2. **Easy Tuning**: Adjust parameters without code changes
3. **Environment Flexibility**: Different settings for different environments
4. **Better Testing**: Easy to test with different configurations
5. **Documentation**: Clear parameter documentation in config classes
6. **Type Safety**: Dataclass validation ensures correct types
7. **Backwards Compatibility**: Default configs maintain current behavior

## Testing Strategy

1. **Unit Tests**: Test each utility with custom configs
2. **Integration Tests**: Test full pipeline with various configs
3. **Performance Tests**: Verify no performance regression
4. **Validation Tests**: Ensure invalid configs are rejected

## Rollout Plan

1. **Week 1**: Migrate spell checker and quality filter
2. **Week 2**: Migrate segmenter and embedder
3. **Week 3**: Update pipeline and test thoroughly
4. **Week 4**: Documentation and optimization

## Monitoring

After migration, monitor:
- API rate limits with different batch sizes
- Memory usage with different cache sizes
- Processing speed with different concurrency settings
- Error rates with different retry configurations

## Rollback Plan

If issues arise:
1. Default configs preserve current behavior
2. Can temporarily hardcode values in config classes
3. Git history allows reverting to previous version

## Success Criteria

- [ ] All hardcoded values removed from utils
- [ ] All utils accept configuration objects
- [ ] Pipeline uses centralized configs
- [ ] No performance regression
- [ ] Configuration documented
- [ ] Tests pass with various configs