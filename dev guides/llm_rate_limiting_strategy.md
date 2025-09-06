# LLM Rate Limiting Strategy Guide

## Executive Summary

This guide documents our hybrid rate limiting strategy for making efficient API calls to Large Language Models (LLMs) while respecting rate limits. After extensive analysis and iteration, we've developed a minimal but robust approach that combines the simplicity of `AsyncLimiter` with token-aware capacity tracking.

### Key Principles
1. **Start simple, add complexity only when needed**
2. **Let the HTTP client handle connection pooling efficiently**
3. **Track both request and token limits to prevent 429 errors**
4. **Use smart waiting instead of busy polling**
5. **Include output tokens in capacity planning**
6. **Dynamic batch sizing based on token budgets**
7. **Workload assessment before applying rate limiting**
8. **Proper error recovery with cooldown periods**

### When to Use This Pattern
Apply this rate limiting strategy to any utility or module that:
- Makes API calls to OpenAI, Gemini, or similar LLM providers
- Processes batches of requests
- Needs to maximize throughput while avoiding rate limit errors

## Core Concepts

### Understanding Rate Limits
LLM providers typically enforce two types of limits:
- **RPM (Requests Per Minute)**: Maximum number of API calls
- **TPM (Tokens Per Minute)**: Maximum tokens processed (input + output)

Example limits from config.py:
```python
"gpt-4.1-mini": OpenAIRateLimits(
    tokens_per_minute=4_000_000,  # Conservative for cross-tier compatibility
    requests_per_minute=5_000,
)
```

### Sliding Windows vs Hard Minutes
Most providers use sliding windows (rolling 60-second periods) rather than calendar minutes. This means:
- ❌ Wrong: Fire 5000 requests at minute 0, wait until minute 1
- ✅ Right: Smooth request distribution over time

### Why Both Constraints Matter
- **Small requests**: Limited by RPM (many requests, few tokens each)
- **Large requests**: Limited by TPM (few requests, many tokens each)
- **Mixed workloads**: Need to track both to avoid hitting either limit

## Implementation Strategy

### Architecture Overview
Our hybrid approach uses three layers of control:
1. **AsyncLimiter**: Controls request launch rate (RPM)
2. **TokenBucket**: Tracks token consumption (TPM)  
3. **Semaphore**: Limits concurrent connections (transport/system limits)

### Core Components

#### 1. TokenBucket Class (~25 lines)
```python
class TokenBucket:
    def __init__(self, tokens_per_minute):
        self.tpm = tokens_per_minute
        self.available = tokens_per_minute
        self.last_update = time.monotonic()  # Use monotonic to avoid clock issues
        self.lock = asyncio.Lock()
    
    async def acquire(self, tokens_needed):
        async with self.lock:
            # Regenerate tokens based on time elapsed
            now = time.monotonic()
            elapsed = now - self.last_update
            self.available = min(self.tpm, self.available + (self.tpm * elapsed / 60))
            self.last_update = now
            
            # Calculate wait time if not enough tokens (avoid busy polling)
            if self.available < tokens_needed:
                deficit = tokens_needed - self.available
                wait_seconds = deficit * 60 / self.tpm
                await asyncio.sleep(wait_seconds)
                # Recalculate after sleep
                now = time.monotonic()
                elapsed = wait_seconds
                self.available = min(self.tpm, self.available + (self.tpm * elapsed / 60))
                self.last_update = now
            
            # Consume tokens
            self.available -= tokens_needed
```

#### 2. Rate Limiter Setup
```python
from aiolimiter import AsyncLimiter
from config import get_openai_rate_limits

# Get limits for the model
limits = get_openai_rate_limits(model_name)
HEADROOM = 0.8  # Use 80% of limits for safety

# Create rate limiters
rpm_limiter = AsyncLimiter(limits.requests_per_minute * HEADROOM / 60, 1)
token_bucket = TokenBucket(limits.tokens_per_minute * HEADROOM)

# Transport limit (HTTP/2 streams, system resources)
semaphore = asyncio.Semaphore(100)
```

#### 3. Token Counting
```python
def count_tokens_with_output(request, model):
    """Count total tokens including estimated output"""
    # Count input tokens (already exists in most utilities)
    input_tokens = count_input_tokens(request, model)
    
    # Estimate output tokens
    if hasattr(request, 'max_tokens'):
        output_tokens = request.max_tokens
    else:
        # Use p95 estimate for the model/task type
        output_tokens = get_p95_output_estimate(model, task_type)
    
    return input_tokens + output_tokens
```

#### 4. Process Request Function
```python
async def process_request_with_rate_limiting(request, model):
    """Process a single request with rate limiting"""
    # Count tokens including output estimate
    total_tokens = count_tokens_with_output(request, model)
    
    # Wait for rate limits
    async with rpm_limiter:                    # RPM check
        await token_bucket.acquire(total_tokens)  # TPM check
        async with semaphore:                     # Transport limit
            # Use existing retry logic (tenacity)
            return await call_api_with_retry(request)
```

### Complete Minimal Implementation Example
```python
import asyncio
import time
from aiolimiter import AsyncLimiter
from tenacity import retry, retry_if_exception_type, wait_exponential_jitter
from config import get_openai_rate_limits

class TokenBucket:
    """Token bucket for TPM rate limiting"""
    def __init__(self, tokens_per_minute):
        self.tpm = tokens_per_minute
        self.available = tokens_per_minute
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()
    
    async def acquire(self, tokens_needed):
        async with self.lock:
            # Regenerate tokens
            now = time.monotonic()
            elapsed = now - self.last_update
            self.available = min(self.tpm, self.available + (self.tpm * elapsed / 60))
            self.last_update = now
            
            # Wait if needed (smart waiting, no polling)
            if self.available < tokens_needed:
                wait_seconds = (tokens_needed - self.available) * 60 / self.tpm
                await asyncio.sleep(wait_seconds)
                now = time.monotonic()
                self.available = min(self.tpm, self.available + (self.tpm * wait_seconds / 60))
                self.last_update = now
            
            self.available -= tokens_needed

class RateLimitedProcessor:
    """Manages rate limiting for LLM API calls"""
    def __init__(self, model_name, headroom=0.8):
        limits = get_openai_rate_limits(model_name)
        
        # Setup rate limiters
        self.rpm_limiter = AsyncLimiter(limits.requests_per_minute * headroom / 60, 1)
        self.token_bucket = TokenBucket(limits.tokens_per_minute * headroom)
        self.semaphore = asyncio.Semaphore(100)  # Transport limit
        
        self.model = model_name
    
    @retry(
        retry=retry_if_exception_type(RateLimitError),
        wait=wait_exponential_jitter(initial=1, max=30),
        stop=stop_after_attempt(3)
    )
    async def call_api(self, request):
        """Make the actual API call with retries"""
        # Your API call logic here
        return await client.create_completion(request)
    
    async def process_request(self, request):
        """Process a single request with rate limiting"""
        # Count tokens
        total_tokens = self.count_tokens_with_output(request)
        
        # Apply rate limits
        async with self.rpm_limiter:
            await self.token_bucket.acquire(total_tokens)
            async with self.semaphore:
                return await self.call_api(request)
    
    def count_tokens_with_output(self, request):
        """Count input + estimated output tokens"""
        input_tokens = count_input_tokens(request, self.model)
        output_tokens = getattr(request, 'max_tokens', 500)  # Default estimate
        return input_tokens + output_tokens

# Usage
processor = RateLimitedProcessor("gpt-4.1-mini")
results = await asyncio.gather(*[
    processor.process_request(req) for req in requests
])
```

## Advanced Features

### Dynamic Batch Sizing
Instead of fixed batch sizes, calculate optimal batches based on token budgets:

```python
def calculate_optimal_batch_size(tasks, model_config, headroom=0.8):
    """Calculate how many tasks fit in one API call"""
    # Get token limits
    limits = get_openai_rate_limits(model)
    max_tokens_per_request = limits.tokens_per_minute * headroom / 60  # Per-second budget
    
    # Estimate tokens per task
    sample_tasks = tasks[:5]
    avg_task_tokens = estimate_task_tokens(sample_tasks)
    prompt_base_tokens = count_prompt_template_tokens()
    
    # Calculate max tasks that fit
    available_for_tasks = max_tokens_per_request - prompt_base_tokens
    max_tasks_per_batch = int(available_for_tasks / avg_task_tokens)
    
    return max(1, min(max_tasks_per_batch, len(tasks)))
```

### Workload Assessment
Check if the entire workload fits within limits before batching:

```python
async def process_with_smart_batching(tasks, model):
    """Process tasks with intelligent batching"""
    # Assess total workload
    total_tokens = estimate_total_tokens(tasks)
    total_requests = 1 if len(tasks) <= max_batch_size else ceil(len(tasks) / max_batch_size)
    
    # Get limits
    limits = get_openai_rate_limits(model)
    
    # If everything fits in one minute, process all at once
    if total_tokens < limits.tokens_per_minute * 0.8 and total_requests < limits.requests_per_minute * 0.8:
        # Fire all tasks in minimal batches
        return await process_all_immediate(tasks)
    else:
        # Use rate limiting for large workloads
        return await process_with_rate_limiting(tasks)
```

### Error Recovery with Cooldown
Implement proper cooldown after rate limit errors:

```python
class RateLimitTracker:
    def __init__(self):
        self.last_rate_limit_time = 0
        self.cooldown_seconds = 15
    
    def check_cooldown(self):
        """Check if we're still in cooldown period"""
        time_since_error = time.monotonic() - self.last_rate_limit_time
        if time_since_error < self.cooldown_seconds:
            remaining = self.cooldown_seconds - time_since_error
            return True, remaining
        return False, 0
    
    def record_rate_limit(self):
        """Record when rate limit was hit"""
        self.last_rate_limit_time = time.monotonic()

# In retry logic
@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential_jitter(initial=1, max=30),
    before_sleep=lambda retry_state: rate_limit_tracker.record_rate_limit()
)
```

### Request Rate Capping
Add micro-delays to cap maximum request rate:

```python
async def process_with_rate_cap(tasks):
    """Process with maximum 1000 requests per second"""
    MIN_REQUEST_INTERVAL = 0.001  # 1ms = 1000 req/sec max
    
    last_request_time = 0
    for task in tasks:
        # Ensure minimum interval between requests
        current_time = time.monotonic()
        time_since_last = current_time - last_request_time
        if time_since_last < MIN_REQUEST_INTERVAL:
            await asyncio.sleep(MIN_REQUEST_INTERVAL - time_since_last)
        
        # Process request
        await process_request(task)
        last_request_time = time.monotonic()
```

## Design Decisions

### Why Hybrid Over Pure OpenAI Approach
1. **Simpler code**: ~40 lines vs ~200 lines
2. **Cleaner abstractions**: AsyncLimiter handles RPM elegantly
3. **Better separation**: Each component has one responsibility
4. **Easier to test**: Modular design with clear interfaces

### Trade-offs We Accepted
1. **No token refunding**: If a request fails, we don't reclaim tokens
   - Acceptable because our retry logic is robust
   - Complexity not worth it for occasional failures

2. **Fixed semaphore**: Not adaptive based on workload
   - 100 concurrent is reasonable for most systems
   - HTTP client provides additional natural throttling

3. **Simple token estimation**: Use max_tokens or fixed estimates
   - More accurate than ignoring output tokens
   - Simpler than tracking actual usage

### When to Add Complexity
Only add these features if you observe specific issues:
- **Per-model buckets**: Only if using multiple models with different limits
- **Token refunding**: Only if seeing high failure rates
- **Adaptive concurrency**: Only if 100 concurrent causes issues
- **Exact token tracking**: Only if estimates cause frequent 429s

## Rollout Plan

### Phase 1: High-Volume Utilities
1. **spellChecker.py** - Highest volume, most critical
2. **qualityFilter.py** - High volume, benefits from consistency
3. **codeGenerator.py** - Complex prompts, variable token usage

### Phase 2: Other LLM Utilities
- segmentation.py
- hierarchicalLabeller.py
- codeAssignment.py

### Implementation Steps
1. Add `TokenBucket` class to a shared utility module
2. Create `RateLimitedProcessor` base class
3. Update each utility to use the new rate limiting
4. Test with realistic workloads
5. Monitor for 429 errors and adjust headroom if needed

### Testing Approach
```python
# Test different scenarios
async def test_rate_limiting():
    processor = RateLimitedProcessor("gpt-4.1-mini", headroom=0.8)
    
    # Test 1: Small burst (should complete quickly)
    small_requests = [create_request(tokens=100) for _ in range(10)]
    start = time.time()
    await asyncio.gather(*[processor.process_request(r) for r in small_requests])
    assert time.time() - start < 2  # Should be fast
    
    # Test 2: Large requests (should respect TPM)
    large_requests = [create_request(tokens=10000) for _ in range(100)]
    # Should take ~1 minute due to TPM limits
    
    # Test 3: Many small requests (should respect RPM)
    many_requests = [create_request(tokens=10) for _ in range(5000)]
    # Should take ~1 minute due to RPM limits
```

### Monitoring
Key metrics to track:
- 429 error rate (target: <0.1%)
- Average wait time per request
- Token utilization (actual vs limit)
- Request throughput (requests/minute achieved)

## Future Enhancements

### Near-term (if needed)
1. **Response header parsing**: Extract actual rate limit info from headers
2. **Warmup logic**: Start conservative, increase to target rate
3. **Priority queuing**: Process urgent requests first

### Long-term (probably not needed)
1. **Cross-process coordination**: Share rate limits across workers
2. **Predictive throttling**: ML-based rate prediction
3. **Cost optimization**: Balance speed vs API costs

## Conclusion

This hybrid rate limiting strategy provides:
- ✅ Production-ready robustness
- ✅ Minimal code complexity
- ✅ Efficient API utilization
- ✅ Easy integration with existing code

Start with this implementation, monitor actual performance, and only add complexity where real issues arise. The beauty of this approach is that it's simple enough to understand at a glance, yet sophisticated enough to handle real-world LLM API constraints effectively.