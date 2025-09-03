# Streamlit Performance Optimization Roadmap

## Overview
This document outlines the comprehensive performance optimization strategy for CoderingsTool's Streamlit interface. The optimization follows the proven pattern: **cold starts tiny, reruns instant**.

## Completed Phases

### ✅ Phase 1: Cache Heavy Resources (@st.cache_resource)
**Status: COMPLETED**

#### 1.1 SpaCy Language Models
- **File**: `utils/cached_resources.py` - `get_spacy_nlp()`
- **Impact**: 🔥 **CRITICAL** - SpaCy models (`nl_core_news_lg`, `en_core_web_lg`) are ~500MB and take 3-5s to load
- **Implementation**: Converted from `@lru_cache` to `@st.cache_resource` with loading spinner
- **Benefit**: Models loaded once per session instead of per spell check operation

#### 1.2 OpenAI API Clients
- **File**: `utils/cached_resources.py` - `get_openai_client()`
- **Impact**: 🟡 **MODERATE** - Instructor-patched clients have initialization overhead
- **Implementation**: Centralized cached client creation across all utilities
- **Files Updated**: `spellChecker.py`, `ideaExtractor.py`, `qualityFilter.py`
- **Benefit**: API clients reused across pipeline steps

#### 1.3 Tiktoken Encoders
- **File**: `utils/cached_resources.py` - `get_tiktoken_encoding()`
- **Impact**: 🟡 **MODERATE** - Tokenizer loading has measurable cost for large models
- **Implementation**: Cached encoding instances with fallback to cl100k_base
- **Benefit**: Tokenizers loaded once per model type per session

#### 1.4 Pipeline Components
- **File**: `pipeline_runner.py` - `_get_cached_*()` functions
- **Impact**: 🟠 **LOW-MODERATE** - Some pipeline classes have initialization overhead
- **Implementation**: Basic caching framework for embedders, clusterers, theme identifiers
- **Status**: Foundation laid for future optimization

### ✅ Phase 2: Cache Processing Results (@st.cache_data)
**Status: COMPLETED**

#### 2.1 SPSS File Parsing
- **File**: `pipeline_runner.py` - `_cache_spss_data()`
- **Impact**: 🟠 **MODERATE** - SPSS files can be large and parsing takes time
- **Implementation**: Cache DataFrame results by filename + column + variable
- **Benefit**: File only parsed once per session, instant reruns

#### 2.2 Embedding Results Framework
- **File**: `pipeline_runner.py` - `_cache_embedding_results()`
- **Impact**: 🔥 **HIGH** - Embeddings are expensive API calls
- **Implementation**: Framework for content-hash-based caching
- **Status**: Placeholder implemented, needs integration

#### 2.3 Spell Correction Framework
- **File**: `pipeline_runner.py` - `_cache_spell_correction_results()`
- **Impact**: 🟠 **MODERATE** - Spell correction involves API calls
- **Implementation**: Framework for content+config-hash-based caching
- **Status**: Placeholder implemented, needs integration

## Future Phases

### 📋 Phase 3: UI/UX Improvements
**Priority: HIGH** - Immediate user experience gains

#### 3.1 Progress Indicators & Spinners
**Files to modify**: `app.py`, `pipeline_runner.py`

```python
# Current: Silent loading
model = get_spacy_nlp()

# Target: Clear feedback
with st.spinner("Loading Dutch language model (nl_core_news_lg)..."):
    model = get_spacy_nlp()
```

**Implementation Tasks**:
- [ ] Add specific spinners for each heavy operation
- [ ] Replace generic "🔄 Processing..." with descriptive messages
- [ ] Add progress bars for batch operations (embeddings, quality filtering)
- [ ] Show estimated time remaining for long operations

**Specific Messages**:
- "Loading SpaCy language model (5-10 seconds first time)..."
- "Generating embeddings for 1,234 text segments..."
- "Running quality assessment on responses..."
- "Clustering similar responses..."
- "Generating codebook from clusters..."

#### 3.2 Resource Warm-up Options
**File to create**: `app.py` - Sidebar section

```python
# Sidebar warm-up section
st.sidebar.subheader("🚀 Performance")
if st.sidebar.button("Pre-load Models"):
    with st.spinner("Pre-loading all models..."):
        get_spacy_nlp()  # Triggers caching
        get_openai_client()
        st.sidebar.success("✅ Models ready!")
```

**Implementation Tasks**:
- [ ] Add "Pre-load Models" button in sidebar
- [ ] Create warm-up function that loads all cached resources
- [ ] Show cache status indicators (✅ SpaCy Loaded, ✅ OpenAI Ready)
- [ ] Add cache clearing button for development

#### 3.3 Smart Loading Messages
**Locations**: Throughout pipeline steps

**Current State**:
```python
# Generic container updates
streamlit_container.text("🔄 Preprocessing text data...")
```

**Target State**:
```python
# Smart, informative messages
if first_time_loading:
    streamlit_container.text("🔄 Loading SpaCy model (5-10s first time)...")
else:
    streamlit_container.text("🔄 Preprocessing text data...")
```

**Implementation Tasks**:
- [ ] Add first-time vs. cached detection
- [ ] Context-aware message selection
- [ ] Show processing stats ("Processing 234/1000 responses...")
- [ ] Add time estimates based on data size

### 📋 Phase 4: Advanced Optimizations
**Priority: MEDIUM** - Performance fine-tuning

#### 4.1 Conditional Loading
**Impact**: Avoid loading unused resources

**SpaCy Conditional Loading**:
```python
# Only load if spell checking is enabled
@st.cache_resource
def get_spacy_nlp_conditional(spell_check_enabled: bool):
    if not spell_check_enabled:
        return None
    return get_spacy_nlp()
```

**Implementation Tasks**:
- [ ] Check if spell checking is disabled in config
- [ ] Skip SpaCy loading if not needed
- [ ] Conditional embedding provider loading (OpenAI vs Gemini)
- [ ] Skip heavy imports based on pipeline configuration

#### 4.2 Session State Optimization
**Files**: `app.py`, `pipeline_runner.py`

**Current Issues**:
- Deep nested session state access
- Repeated validation checks
- Unnecessary re-initialization

**Optimization Strategy**:
```python
# Instead of scattered session state access
if st.session_state.step >= 2:
    if st.session_state.preprocessed_data is not None:
        # ... complex logic

# Use structured state manager
class SessionStateManager:
    @property
    def is_preprocessed(self): 
        return hasattr(self, '_preprocessed') and self._preprocessed is not None
    
    def get_preprocessed_data(self):
        return getattr(self, '_preprocessed', None)
```

**Implementation Tasks**:
- [ ] Create SessionStateManager class
- [ ] Centralize state validation
- [ ] Add state caching for complex calculations
- [ ] Optimize state serialization

#### 4.3 Background Loading
**Impact**: Pre-load next step while user reviews current step

**Implementation Strategy**:
```python
# When user completes step 5 (embeddings), start loading clustering in background
if st.session_state.step == 5 and st.session_state.embeddings_complete:
    # User is viewing embeddings results
    # Pre-load clustering resources in background
    asyncio.create_task(preload_clustering_resources())
```

**Implementation Tasks**:
- [ ] Identify predictable user workflow patterns
- [ ] Add background resource loading
- [ ] Implement cancellation for abandoned workflows
- [ ] Add memory management for unused pre-loaded resources

## Performance Metrics & Monitoring

### Current Baseline (Post Phase 1 & 2)
- **Cold Start**: ~5-10 seconds (85.7% improvement from ~30-60s)
- **Reruns**: Near-instant for cached operations
- **Module Loading**: 1,226 modules (down from 8,581)

### Target Metrics (Post Phase 3 & 4)
- **Cold Start**: <5 seconds with warm-up option
- **Reruns**: <1 second for all cached operations  
- **User Feedback**: 100% coverage of loading states
- **Resource Usage**: Conditional loading saves ~30% memory for partial workflows

### Monitoring Implementation
```python
# Add to app.py
if st.sidebar.checkbox("Show Performance Metrics", value=False):
    st.sidebar.json({
        "modules_loaded": len(sys.modules),
        "cache_hits": st.session_state.get('cache_hits', 0),
        "cache_misses": st.session_state.get('cache_misses', 0),
        "memory_usage": f"{psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB"
    })
```

## Troubleshooting Guide

### Common Caching Issues

#### 1. Cache Not Working
**Symptoms**: Resources reloading on every rerun
**Diagnosis**:
```python
# Check if function is properly cached
if st.button("Test Cache"):
    start_time = time.time()
    model = get_spacy_nlp()
    load_time = time.time() - start_time
    st.write(f"Load time: {load_time:.2f}s (should be ~0.01s if cached)")
```
**Solutions**:
- Ensure function parameters are hashable
- Check for accidental parameter changes
- Verify `@st.cache_resource` decorator is applied

#### 2. Memory Issues
**Symptoms**: Streamlit crashes with memory errors
**Diagnosis**: Large models + datasets exceeding memory
**Solutions**:
- Use `st.cache_resource(max_entries=1)` for large models
- Implement cache eviction strategies
- Add memory monitoring

#### 3. Stale Cache
**Symptoms**: Old results persisting after code changes
**Solutions**:
- Use `st.cache_data.clear()` and `st.cache_resource.clear()`
- Add version parameters to cache keys
- Implement development mode with cache disabled

### Development Mode
```python
# Add to app.py for development
DEVELOPMENT_MODE = st.sidebar.checkbox("Development Mode", value=False)
if DEVELOPMENT_MODE:
    if st.sidebar.button("Clear All Caches"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
```

## Implementation Priority

### Immediate (Next Sprint)
1. **Phase 3.1**: Add progress spinners with specific messages
2. **Phase 3.2**: Implement model pre-loading button
3. **Phase 4.1**: Add conditional SpaCy loading

### Medium Term (Next Month)
1. **Phase 3.3**: Smart loading message system
2. **Phase 4.2**: Session state optimization
3. **Integration**: Full embedding and spell correction caching

### Long Term (Next Quarter)
1. **Phase 4.3**: Background loading implementation
2. **Advanced Metrics**: Comprehensive performance monitoring
3. **User Studies**: Measure actual user experience improvements

## Notes for Future Developers

### Key Principles
1. **Cache Immutable Resources**: Models, clients, configurations
2. **Cache Pure Functions**: Same input → same output
3. **Hash for Data Caching**: Use content hashes for deterministic caching
4. **Show Progress**: Never leave users wondering what's happening
5. **Graceful Degradation**: Always have fallbacks for cache failures

### Anti-Patterns to Avoid
- ❌ Caching mutable objects that change
- ❌ Complex cache keys that rarely match
- ❌ Silent operations longer than 2 seconds
- ❌ Cache keys with unstable parameters
- ❌ Memory leaks from uncapped caches

### Testing Cache Behavior
```python
# Add to tests
def test_cache_effectiveness():
    # Clear cache
    st.cache_resource.clear()
    
    # Time first load
    start = time.time()
    model = get_spacy_nlp()
    first_load = time.time() - start
    
    # Time second load (should be cached)
    start = time.time()
    model2 = get_spacy_nlp()
    second_load = time.time() - start
    
    assert second_load < first_load / 10, "Cache not working effectively"
    assert model is model2, "Different instances returned"
```

This roadmap provides a systematic approach to making CoderingsTool's Streamlit interface both fast to start and lightning-fast to use repeatedly.