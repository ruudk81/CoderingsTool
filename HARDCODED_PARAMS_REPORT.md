# Hardcoded Configuration Parameters Report

## Summary

This report documents all hardcoded configuration parameters found in the utils files used in steps 1-4 of the CoderingsTool pipeline.

## Findings by File

### 1. utils/data_io.py (Step 1 - Data Loading)
- **Status**: ✅ Clean - No hardcoded configuration values
- All parameters are passed as arguments or use dynamic path resolution

### 2. utils/textNormalizer.py (Step 2 - Text Normalization)
- **Status**: ✅ Already Configurable
- Has `NormalizerConfig` class with defaults:
  - `custom_symbols = "'#%&:;<=>@[\]^_{|}~-"` (line 9)
  - `na_placeholder = "<NA>"` (line 10)
  - `min_length = 1` (line 11)

### 3. utils/spellChecker.py (Step 2 - Spell Checking)
- **Status**: ❌ Multiple Hardcoded Values
- **In GraderConfig class**:
  - `batch_size: int = 20` (line 19)
  - `temperature: float = 0.0` (line 20)
  - `max_tokens: int = 4000` (line 21)
  - `retries: int = 3` (line 23)
- **In SpellChecker class**:
  - `self.retry_delay = 2` (line 35)
  - `self.max_retries = 3` (line 36)
  - `@lru_cache(maxsize=10000)` (line 119)
  - `batch_size=32` for spacy.pipe() (lines 181, 221, 426)
  - `max_batch_size = 5` (line 274)
  - `completion_reserve = 1000` (line 328)
  - `repeated_char_pattern = re.compile(r'^(.)\1{4,}$')` - 5+ chars (line 350)
  - `max_retries=3` for instructor (line 385)
  - `seed = 42` (line 386)
  - `if len(correction_examples) < 10:` (line 476)

### 4. utils/textFinalizer.py (Step 2 - Text Finalization)
- **Status**: ✅ Clean - No hardcoded configuration values
- All logic is pure text processing

### 5. utils/qualityFilter.py (Step 3 - Quality Filtering)
- **Status**: ❌ Multiple Hardcoded Values
- **In GraderConfig class**:
  - `batch_size: int = 20` (line 19)
  - `temperature: float = 0.0` (line 20)
  - `max_tokens: int = 4000` (line 21)
  - `retries: int = 3` (line 23)
- **In Grader class**:
  - `max_retries=3` for instructor (line 69)
  - Quality thresholds: `0.7` for high, `0.4` for medium (lines 128-133)
  - `if ... and len(filtered_examples) < 5:` (line 135)

### 6. utils/segmentDescriber.py (Step 3 - Segmentation & Description)
- **Status**: ❌ Multiple Hardcoded Values
- **In LangChainPipeline class**:
  - `self.retry_delay = 2` (line 35)
  - `self.max_retries = 3` (line 36)
- **In SegmentDescriber class**:
  - `max_tokens: int = 16000` (line 122)
  - `completion_reserve: int = 1000` (line 123)
  - `max_batch_size: int = 5` (line 124)
  - `batch_size=32` for spacy.pipe() (line 181)
  - `'n_jobs': 1` in UMAP params (line 230)
  - `if len(code_examples) < 5` (line 399)

### 7. utils/embedder.py (Step 4 - Embeddings)
- **Status**: ❌ Multiple Hardcoded Values
- **Throughout the class**:
  - `batch_size: int = 100` (lines 28, 51, 223, 256)
  - `max_concurrent: int = 5` (lines 85, 90, 95, 100, 103, 223, 256)
  - First 3 responses for samples (line 143)
  - `model="text-embedding-3-large"` (line 211) - should use config
  
## Configuration Classes Added

To address these hardcoded values, the following configuration classes have been added to `config.py`:

1. **SpellCheckConfig** - Centralizes all spell checking parameters
2. **QualityFilterConfig** - Manages quality filtering thresholds and settings
3. **SegmentationConfig** - Controls segmentation and description generation
4. **EmbeddingConfig** - Handles embedding generation parameters

## Recommendations

1. **Update all utils files** to import and use the new configuration classes
2. **Replace hardcoded values** with references to configuration objects
3. **Allow configuration overrides** through constructor parameters
4. **Consider environment variables** for runtime configuration changes
5. **Document all configuration options** in the README or user guide

## Next Steps

1. Refactor `spellChecker.py` to use `SpellCheckConfig`
2. Refactor `qualityFilter.py` to use `QualityFilterConfig`
3. Refactor `segmentDescriber.py` to use `SegmentationConfig`
4. Refactor `embedder.py` to use `EmbeddingConfig`
5. Update pipeline.py to pass configuration objects to each step
6. Add configuration validation to ensure valid parameter combinations