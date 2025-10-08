# CodeGenerator.py Cleanup Plan

## Overview
This document outlines identified dead code, redundant methods, and overly complex code paths in `src/utils/codeGenerator.py` that could be simplified or removed to improve code maintainability.

## 1. Dead Code to Remove

### 1.1 Commented-out `_bootstrap_probe_theme_extraction` Method
**Location**: Lines 1673-1721  
**Description**: Entire method is commented out but still present in the codebase  
**Status**: Replaced by `probe_call_theme_extraction` method  
**Action**: Remove this entire commented block

### 1.2 Debug Logging Statements
**Location**: Lines 2533-2589 in `_extract_single_theme`  
**Description**: Multiple commented-out debug logging statements that were used during development  
**Examples**:
```python
# if self.verbose_reporter.enabled and ideas:
#     self.verbose_reporter.stat_line(f"DEBUG C{cluster_id}: Input ideas type: {type(ideas[0])}, count: {len(ideas)}")
```
**Action**: Remove all commented debug lines

## 2. Redundant/Duplicate Methods

### 2.1 Embedding Methods Duplication
**Location**: `SimilarityEngine` class  
**Methods**: 
- `_embed_openai_batch()` (line 830)
- `_get_embedding()` (line 851)

**Issue**: These methods duplicate functionality that already exists in the `embedder.py` utility  
**Impact**: Code duplication, maintenance burden  
**Action**: Consider using the existing embedder utility instead of reimplementing

### 2.2 Model Configuration Confusion
**Issue**: Two configuration objects exist:
- `self.config` (CodeDesignerConfig) 
- `self.model_config` (ModelConfig)

**Usage Inconsistency**:
- Some places use `self.config.model`
- Others use `self.model_config.get_model_for_stage('stage_name')`

**Action**: Standardize on one approach for model selection throughout the codebase

## 3. Overly Complex Code Paths

### 3.1 Multi-Theme Cluster Expansion & Redistribution System
**Methods**:
- `expand_multi_theme_clusters()` (lines 2033-2124)
- `redistribute_ideas_to_subthemes()` (lines 2126-2223)  
- `_update_cluster_models_with_redistribution()` (lines 2225-2300)

**Complexity**: 
- Handles rare edge cases where a cluster has multiple themes
- Adds significant complexity for minimal benefit
- Creates sub-clusters and redistributes ideas based on similarity

**Recommendation**: 
1. Make this feature optional via configuration
2. Consider removing if usage statistics show it's rarely triggered
3. Simplify to handle multi-theme clusters without redistribution

### 3.2 Anti-Greedy Redistribution Pattern
**Methods**:
- `_redistribute_to_anti_greedy_pattern()` (line 929)
- `_perform_redistribution()` (lines 1185-1243)
- Related helper methods

**Complexity**:
- Complex logic for balancing batch sizes
- Multiple redistribution attempts with progressive thresholds
- Extensive moveable cluster identification

**Recommendation**: 
- Evaluate if this complexity provides meaningful benefits
- Consider simpler batch size balancing approach
- Remove if performance impact is minimal

### 3.3 Progressive Threshold Similarity Batching
**Location**: `create_dissimilarity_batches()` method  
**Complexity**:
- Multiple nested methods for progressive threshold batching
- Complex similarity matrix calculations
- Multiple threshold levels (0.3, 0.5, 0.7)

**Recommendation**: 
- Simplify to single threshold approach
- Remove progressive threshold logic if benefits are minimal

## 4. Configuration & Parameter Redundancy

### 4.1 Tiktoken Model Mapping
**Location**: Lines 1283-1302  
**Issue**: Complex fallback logic for tiktoken encoding with multiple try-except blocks  
**Current Code**:
```python
try:
    self.encoding = tiktoken.encoding_for_model(self.config.model)
except KeyError:
    tiktoken_model_mapping = {...}
    # Multiple fallback attempts
```
**Action**: Simplify with a direct mapping dictionary and single fallback

### 4.2 Stages Control Parameter
**Parameter**: `stages_to_run`  
**Values**: Only supports 'all' or 'theme_extraction_only'  
**Issue**: String comparison for simple binary choice  
**Action**: Convert to boolean flag: `theme_extraction_only: bool = False`

## 5. Potentially Unused Features

### 5.1 Version Control in SharedCodebook
**Location**: `SharedCodebook` class  
**Features**:
- Complex version management
- Embedding caching per version
- Version history tracking

**Questions**:
- Is version control actually used in practice?
- Could this be simplified to current version only?

**Action**: Evaluate usage and remove if overengineered

## 6. Method Consolidation Opportunities

### 6.1 Sample Methods
**Methods**:
- `_sample_representative_ideas()` (line 2315)
- `_sample_from_subcluster()` (line 2477)

**Opportunity**: Could be consolidated into single sampling method with parameters

### 6.2 Token Measurement
**Current**: Separate token measurement for each stage  
**Opportunity**: Generalize token measurement into single reusable method

## Implementation Priority

### High Priority (Quick Wins)
1. Remove all commented-out code blocks
2. Remove debug statements
3. Convert `stages_to_run` to boolean
4. Simplify tiktoken model mapping

### Medium Priority (Moderate Effort)
1. Consolidate embedding functionality
2. Standardize model configuration approach
3. Consolidate sampling methods

### Low Priority (Major Refactoring)
1. Simplify multi-theme cluster handling
2. Remove anti-greedy redistribution
3. Simplify similarity batching to single threshold
4. Evaluate and potentially remove versioning in SharedCodebook

## Expected Benefits

1. **Code Reduction**: ~300-500 lines of code removal
2. **Complexity Reduction**: Fewer edge cases to maintain
3. **Performance**: Slight improvement from removing unnecessary processing
4. **Maintainability**: Clearer code flow, easier to understand
5. **Testing**: Fewer code paths to test

## Migration Strategy

1. **Phase 1**: Remove dead code and comments (no functional impact)
2. **Phase 2**: Consolidate redundant methods (minimal risk)
3. **Phase 3**: Simplify complex features (requires testing)
4. **Phase 4**: Major refactoring of multi-theme and batching logic

## Notes

- All changes should be tested with existing test suite
- Consider adding feature flags for complex features before removal
- Document any behavioral changes in CHANGELOG.md
- Update related documentation and docstrings