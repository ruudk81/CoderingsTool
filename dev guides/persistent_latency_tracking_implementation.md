# Persistent Latency Tracking Implementation Guide

## Overview

This document outlines the implementation plan for adding persistent latency tracking to all LLM-processing utilities in CoderingsTool. The system will learn from actual API response times across sessions, enabling better concurrency optimization and timeout calculations.

### Problem Statement

Currently, all LLM utilities (spellChecker, ideaExtractor, codeGenerator, themeIdentifier, codeAssigner) use hardcoded latency assumptions (2.0 seconds) for:
- Concurrency calculations (`rpm_concurrency = limits.rpm * latency`)
- Worker pool sizing (`num_workers = throughput * latency * 2.0`)
- Timeout values (fixed timeouts, not adaptive)

**Impact of Wrong Assumptions:**
- **If actual latency is 0.8s** (faster than assumed): Underutilized concurrency, slower processing
- **If actual latency is 3.5s** (slower than assumed): Over-aggressive concurrency, rate limit violations
- **Learning is lost between sessions**: Every new processing run starts from scratch

### Current State Analysis

#### Session-Only Learning
Each utility has `LatencyTracker` that learns **within** a session but loses knowledge **between** sessions:

```python
# Session 1: Tuesday 9am
spell_checker = SpellChecker()  # Fresh LatencyTracker(default=2.0s)
# Learns actual latency is 1.2s
# Processing completes → learning lost

# Session 2: Tuesday 2pm  
spell_checker = SpellChecker()  # Fresh LatencyTracker(default=2.0s) again!
# Has to relearn everything from scratch
```

#### What Defines a "Session"
- One complete pipeline run (`python pipeline.py`)
- One dataset processing via web interface
- One direct API call to any utility
- Essentially: whenever a new utility instance is created

## Implementation Plan

### Core Architecture

#### 1. Generic Cache Integration
Extend `CacheManager` to handle latency data for any utility:

```python
# Generic methods in CacheManager
def save_latency_data(self, utility_name: str, model_name: str, latency_data: dict) -> bool
def load_latency_data(self, utility_name: str, model_name: str) -> Optional[dict]
def get_latency_cache_path(self, utility_name: str, model_name: str) -> Path

# Storage structure
cache/latency_data/
├── spellChecker_gpt-4o-mini.pkl
├── ideaExtractor_gpt-4o-mini.pkl  
├── codeGenerator_gpt-4o-mini.pkl
├── themeIdentifier_gpt-4o-mini.pkl
└── codeAssigner_gpt-4o-mini.pkl
```

#### 2. Enhanced LatencyTracker
Make `LatencyTracker` cache-aware and utility-agnostic:

```python
class LatencyTracker:
    def __init__(self, cache_manager=None, utility_name=None, model_name=None, alpha=0.1):
        self.cache_manager = cache_manager
        self.utility_name = utility_name
        self.model_name = model_name
        self.alpha = alpha
        
        # Load from cache if available
        if cache_manager and utility_name and model_name:
            cached_data = cache_manager.load_latency_data(utility_name, model_name)
            if cached_data:
                self.ema = cached_data.get('ema')
                recent_values = cached_data.get('recent_values', [])
                self.values = deque(recent_values, maxlen=100)
            else:
                self.ema = None
                self.values = deque(maxlen=100)
        else:
            self.ema = None
            self.values = deque(maxlen=100)
```

#### 3. Cache Data Structure
```python
{
    'ema': 1.23,                    # Current EMA value
    'recent_values': [1.1, 1.4, ...], # Last 20 latency samples for P95
    'sample_count': 150,             # Total samples collected
    'last_updated': 1703847600.123,  # Timestamp
    'model_config_hash': 'abc123',   # Optional: detect model config changes
    'avg_tokens_per_request': 850,   # Help with concurrency calculations
    'utility_version': '1.0'         # Track compatibility
}
```

## SpellChecker Implementation (Phase 1)

### Step 1: Extend CacheManager

Add latency-specific methods to `src/utils/cacheManager.py`:

```python
def save_latency_data(self, utility_name: str, model_name: str, latency_data: dict) -> bool:
    """Save latency tracking data for any utility + model combination"""
    try:
        cache_path = self.get_latency_cache_path(utility_name, model_name)
        
        # Ensure latency_data directory exists
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        latency_data['last_updated'] = time.time()
        
        # Save using pickle
        with open(cache_path, 'wb') as f:
            pickle.dump(latency_data, f)
            
        logger.debug(f"Saved latency data for {utility_name}:{model_name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save latency data: {e}")
        return False

def load_latency_data(self, utility_name: str, model_name: str) -> Optional[dict]:
    """Load latency tracking data for any utility + model combination"""
    try:
        cache_path = self.get_latency_cache_path(utility_name, model_name)
        
        if not cache_path.exists():
            logger.debug(f"No latency cache found for {utility_name}:{model_name}")
            return None
            
        with open(cache_path, 'rb') as f:
            latency_data = pickle.load(f)
            
        # Check age (optional: expire after 7 days)
        if 'last_updated' in latency_data:
            age_days = (time.time() - latency_data['last_updated']) / 86400
            if age_days > 7:
                logger.info(f"Latency cache expired for {utility_name}:{model_name} ({age_days:.1f} days)")
                cache_path.unlink()
                return None
                
        logger.debug(f"Loaded latency data for {utility_name}:{model_name}")
        return latency_data
        
    except Exception as e:
        logger.error(f"Failed to load latency data: {e}")
        return None

def get_latency_cache_path(self, utility_name: str, model_name: str) -> Path:
    """Generate cache path for latency data"""
    # Sanitize model name for filesystem
    safe_model = model_name.replace('/', '_').replace(':', '_')
    filename = f"{utility_name}_{safe_model}.pkl"
    return self.config.cache_dir / "latency_data" / filename
```

### Step 2: Update LatencyTracker

Modify the existing `LatencyTracker` in `src/utils/spellChecker.py`:

```python
class LatencyTracker:
    """Simple EMA tracker for latencies with cache persistence"""
    def __init__(self, cache_manager=None, utility_name=None, model_name=None, alpha=0.1):
        self.cache_manager = cache_manager
        self.utility_name = utility_name
        self.model_name = model_name
        self.alpha = alpha
        
        # Load from cache if available
        if cache_manager and utility_name and model_name:
            cached_data = cache_manager.load_latency_data(utility_name, model_name)
            if cached_data:
                self.ema = cached_data.get('ema')
                recent_values = cached_data.get('recent_values', [])
                self.values = deque(recent_values, maxlen=100)
                logger.info(f"Loaded {len(recent_values)} latency samples for {utility_name}:{model_name}, EMA: {self.ema:.3f}s")
            else:
                self.ema = None
                self.values = deque(maxlen=100)
                logger.info(f"No cached latency data for {utility_name}:{model_name}, using defaults")
        else:
            self.ema = None
            self.values = deque(maxlen=100)
    
    def add(self, value):
        """Add latency measurement and persist to cache"""
        self.values.append(value)
        if self.ema is None:
            self.ema = value
        else:
            self.ema = self.alpha * value + (1 - self.alpha) * self.ema
        
        # Persist to cache every 10 samples to avoid excessive I/O
        if len(self.values) % 10 == 0:
            self._persist_to_cache()
    
    def _persist_to_cache(self):
        """Save current state to cache"""
        if self.cache_manager and self.utility_name and self.model_name:
            latency_data = {
                'ema': self.ema,
                'recent_values': list(self.values)[-20:],  # Keep last 20 samples
                'sample_count': len(self.values),
                'utility_version': '1.0'
            }
            self.cache_manager.save_latency_data(self.utility_name, self.model_name, latency_data)
    
    def get_timeout(self, est_tokens, margin=1.5):
        """Calculate timeout based on EMA and token count"""
        if not self.values:
            return 30.0  # Default 30s
        
        # Use P95 latency as base
        p95 = np.percentile(list(self.values), 95)
        # Simple linear scaling with token count
        token_factor = est_tokens / 1000
        timeout = p95 + (token_factor * 0.1)
        # Apply margin and bounds
        return max(15.0, min(60.0, timeout * margin))
    
    def get_avg_latency(self):
        """Get average latency for concurrency calculations"""
        if not self.values:
            return 2.0  # Default 2s
        return self.ema if self.ema is not None else 2.0
```

### Step 3: Update SpellChecker Constructor

Modify `SpellChecker.__init__()` to accept and use cache_manager:

```python
class SpellChecker:
    def __init__(self, config: SpellCheckConfig = None, model_config: ModelConfig = None, 
                 openai_api_key: Optional[str] = None, cache_manager = None, 
                 verbose: bool = False, prompt_printer = None, verbose_reporter: Optional['VerboseReporter'] = None):
        # ... existing initialization ...
        
        # Initialize latency tracker with cache persistence
        self.latency_tracker = LatencyTracker(
            cache_manager=cache_manager,
            utility_name="spellChecker", 
            model_name=self.model
        )
        
        # ... rest of initialization ...
```

### Step 4: Update Pipeline Integration

Ensure pipeline passes `cache_manager` to `SpellChecker`:

```python
# In pipeline.py, wherever SpellChecker is created:
spell_checker = SpellChecker(
    config=spell_config,
    model_config=model_config,
    cache_manager=cache_manager,  # Add this line
    verbose=verbose,
    verbose_reporter=verbose_reporter
)
```

## Generic Design Pattern (Other Utilities)

### Implementation Template

For any LLM utility (ideaExtractor, codeGenerator, etc.), follow this pattern:

#### 1. Add LatencyTracker to Utility

```python
class YourUtility:
    def __init__(self, ..., cache_manager=None, ...):
        # ... existing init ...
        
        self.latency_tracker = LatencyTracker(
            cache_manager=cache_manager,
            utility_name="yourUtility",  # Unique name
            model_name=self.model
        )
```

#### 2. Track API Latency

```python
async def your_api_method(self, ...):
    # Track API call latency
    api_start_time = time.time()
    
    # Make API call with dynamic timeout
    dynamic_timeout = self.latency_tracker.get_timeout(tokens_needed)
    response = await self.client.chat.completions.create(
        # ... api parameters ...
        timeout=dynamic_timeout
    )
    
    # Record latency for future calculations
    api_latency = time.time() - api_start_time
    self.latency_tracker.add(api_latency)
```

#### 3. Use Dynamic Latency in Concurrency

```python
def calculate_concurrency(self):
    # Use learned latency instead of hardcoded 2.0
    dynamic_latency = self.latency_tracker.get_avg_latency()
    
    rpm_concurrency = limits.requests_per_minute * HEADROOM / 60 * dynamic_latency
    tpm_concurrency = (limits.tokens_per_minute * HEADROOM / avg_tokens) * dynamic_latency
    
    return min(rpm_concurrency, tpm_concurrency, MAX_CONCURRENCY)
```

### Utility-Specific Cache Keys

Each utility gets its own cache namespace:

| Utility | Cache Key | Purpose |
|---------|-----------|---------|
| spellChecker | `spellChecker_gpt-4o-mini` | Spell correction API patterns |
| ideaExtractor | `ideaExtractor_gpt-4o-mini` | Idea segmentation API patterns |
| codeGenerator | `codeGenerator_gpt-4o-mini` | Code generation API patterns |
| themeIdentifier | `themeIdentifier_gpt-4o-mini` | Theme identification patterns |
| codeAssigner | `codeAssigner_gpt-4o-mini` | Code assignment patterns |

## Testing Strategy

### Verification Steps

1. **First Session Test:**
   ```python
   # Session 1: Fresh start
   checker = SpellChecker(cache_manager=cache_manager)
   # Verify: starts with default 2.0s latency
   assert checker.latency_tracker.get_avg_latency() == 2.0
   
   # Process some data, learn latency
   await checker.spell_check_async(test_data)
   learned_latency = checker.latency_tracker.get_avg_latency()
   # Verify: learned something different
   assert learned_latency != 2.0
   ```

2. **Persistence Test:**
   ```python
   # Session 2: New instance
   checker2 = SpellChecker(cache_manager=cache_manager)
   # Verify: starts with learned latency from session 1
   assert abs(checker2.latency_tracker.get_avg_latency() - learned_latency) < 0.1
   ```

3. **Cache File Test:**
   ```bash
   # Verify cache file exists
   ls cache/latency_data/spellChecker_gpt-4o-mini.pkl
   
   # Verify content structure
   python -c "
   import pickle
   data = pickle.load(open('cache/latency_data/spellChecker_gpt-4o-mini.pkl', 'rb'))
   print(data.keys())  # Should include: ema, recent_values, sample_count
   "
   ```

### Performance Validation

Monitor improvements over time:

```python
# Before implementation
print(f"Cold start concurrency: {get_concurrency(default_latency=2.0)}")

# After implementation  
print(f"Warm start concurrency: {get_concurrency(learned_latency=1.2)}")

# Expected: Better-tuned concurrency, faster processing
```

## Future Enhancements

### Cross-Utility Analytics

Once all utilities have latency tracking:

```python
def get_performance_report(cache_manager):
    """Generate performance analysis across all utilities"""
    utilities = ['spellChecker', 'ideaExtractor', 'codeGenerator', 'themeIdentifier', 'codeAssigner']
    models = ['gpt-4o-mini', 'gpt-4o', 'claude-3.5-sonnet']
    
    report = {}
    for utility in utilities:
        for model in models:
            data = cache_manager.load_latency_data(utility, model)
            if data:
                report[f"{utility}_{model}"] = {
                    'avg_latency': data.get('ema'),
                    'sample_count': data.get('sample_count'),
                    'last_updated': data.get('last_updated')
                }
    return report
```

### Alerting and Monitoring

```python
def detect_performance_regression(cache_manager, utility_name, model_name):
    """Alert on significant latency increases"""
    data = cache_manager.load_latency_data(utility_name, model_name)
    if data and len(data.get('recent_values', [])) > 10:
        recent_avg = np.mean(data['recent_values'][-10:])
        overall_avg = data.get('ema')
        
        if recent_avg > overall_avg * 1.5:  # 50% increase
            logger.warning(f"Performance regression detected: {utility_name}:{model_name}")
            logger.warning(f"Recent: {recent_avg:.2f}s, Historical: {overall_avg:.2f}s")
```

### Model Comparison

```python
def compare_model_performance():
    """Compare latency across different models for same utility"""
    models = ['gpt-4o-mini', 'gpt-4o', 'claude-3.5-sonnet']
    
    for utility in ['spellChecker', 'ideaExtractor']:
        print(f"\n{utility} Performance:")
        for model in models:
            data = cache_manager.load_latency_data(utility, model)
            if data:
                print(f"  {model}: {data.get('ema', 0):.2f}s avg")
```

## Implementation Timeline

### Phase 1: SpellChecker (Week 1)
- [ ] Extend CacheManager with latency methods
- [ ] Update LatencyTracker for persistence
- [ ] Modify SpellChecker constructor
- [ ] Update pipeline integration
- [ ] Test persistence across sessions

### Phase 2: Other Utilities (Week 2-3) 
- [ ] Apply pattern to ideaExtractor
- [ ] Apply pattern to codeGenerator
- [ ] Apply pattern to themeIdentifier
- [ ] Apply pattern to codeAssigner
- [ ] Validate all utilities track independently

### Phase 3: Analytics (Week 4)
- [ ] Implement cross-utility reporting
- [ ] Add performance monitoring
- [ ] Create alerting for regressions
- [ ] Document usage patterns

## Success Metrics

- **Faster Processing**: Reduced time to optimal concurrency (immediate vs learning period)
- **Better Resource Utilization**: Concurrency matches actual API performance
- **Reduced Rate Limit Violations**: Better timeout and concurrency calculations
- **Cross-Session Consistency**: Same performance characteristics across sessions
- **Utility-Specific Optimization**: Each utility optimized for its specific patterns

## Code Examples Reference

### Basic Usage Pattern
```python
# Initialize utility with cache support
utility = SpellChecker(cache_manager=cache_manager)

# Latency tracking happens automatically
await utility.spell_check_async(data)

# Future sessions benefit from learned latency
utility2 = SpellChecker(cache_manager=cache_manager)  # Starts with learned values
```

### Manual Cache Management
```python
# Check current latency data
data = cache_manager.load_latency_data("spellChecker", "gpt-4o-mini")
if data:
    print(f"Average latency: {data['ema']:.2f}s")
    print(f"Samples collected: {data['sample_count']}")

# Clear cache (force relearning)
cache_path = cache_manager.get_latency_cache_path("spellChecker", "gpt-4o-mini")
if cache_path.exists():
    cache_path.unlink()
```

---

*This implementation provides a foundation for intelligent, adaptive LLM processing that learns from experience and improves performance over time.*