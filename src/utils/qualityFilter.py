import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import asyncio
import time
import logging
from typing import Dict, List, Optional, Union

from aiolimiter import AsyncLimiter
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter
from openai import RateLimitError, APIConnectionError, APITimeoutError, InternalServerError
from instructor.exceptions import InstructorRetryException

# === MODELS ========================================================================================================
import models

# === CONFIG ========================================================================================================
from config import OPENAI_API_KEY, DEFAULT_LANGUAGE, ModelConfig, QualityFilterConfig, DEFAULT_QUALITY_FILTER_CONFIG, get_openai_rate_limits
from prompts import GRADER_INSTRUCTIONS

# === UTILS ========================================================================================================
from .verboseReporter import VerboseReporter, ProcessingStats
from .cached_resources import get_openai_client, get_tiktoken_encoding

try:
    import nest_asyncio #for Spyder
    nest_asyncio.apply()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# === RATE LIMITING (PROVEN THREE-LAYER PATTERN FROM SPELLCHECKER) ========================================================================================================
class TokenBucket:
    
    def __init__(self, tokens_per_minute):
        self.tpm = tokens_per_minute
        self.available = tokens_per_minute
        self.last_update = time.monotonic()  # Use monotonic to avoid clock issues
        self.lock = asyncio.Lock()
    
    async def acquire(self, tokens_needed):
        """Acquire tokens, waiting if necessary"""
        while True:
            async with self.lock:
                # Regenerate tokens based on time elapsed
                now = time.monotonic()
                elapsed = now - self.last_update
                self.available = min(self.tpm, self.available + (self.tpm * elapsed / 60))
                self.last_update = now
                
                # Check if we have enough tokens
                if self.available >= tokens_needed:
                    # Consume tokens and exit
                    self.available -= tokens_needed
                    logger.debug(f"Token bucket: consumed {tokens_needed}, {self.available:.0f} remaining")
                    return
                
                # Calculate wait time if not enough tokens
                deficit = tokens_needed - self.available
                wait_seconds = deficit * 60 / self.tpm
            
            # CRITICAL FIX: Release lock before sleeping
            if wait_seconds > 1.0:  # Only log significant waits
                print(f"[RATE LIMIT] Token bucket waiting {wait_seconds:.1f}s for {tokens_needed} tokens (deficit: {deficit:.0f})")
            await asyncio.sleep(wait_seconds)
            # Loop back to reacquire lock and check again


class RateLimitTracker:
    """Track rate limit errors and enforce cooldown periods"""
    
    def __init__(self, cooldown_seconds=15):
        self.last_rate_limit_time = 0
        self.cooldown_seconds = cooldown_seconds
    
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
        logger.warning(f"Rate limit error recorded, entering {self.cooldown_seconds}s cooldown")


class DynamicTimeoutManager:
    """Dynamic timeout management based on task complexity and learned patterns"""
    
    def __init__(self, base_timeout=30, min_timeout=15, max_timeout=120):
        self.base_timeout = base_timeout
        self.min_timeout = min_timeout
        self.max_timeout = max_timeout
        
        # Learning data
        self.response_times = []  # Store actual response times
        self.token_to_time_ratio = 0.02  # Initial estimate: 20ms per token
        self.complexity_patterns = {}  # Cache timeout patterns by response characteristics
        
        # Baseline measurement
        self.baseline_measured = False
        self.baseline_tasks_needed = 10
        self.baseline_times = []
        
    def calculate_timeout(self, token_count: int, response_length: int = None) -> float:
        """Calculate dynamic timeout based on task complexity"""
        
        if not self.baseline_measured:
            # Use conservative timeout during baseline measurement
            return self.max_timeout
        
        # Base calculation from learned token-to-time ratio
        estimated_time = token_count * self.token_to_time_ratio
        
        # Add buffer based on response complexity
        complexity_factor = 1.0
        if response_length:
            if response_length > 500:
                complexity_factor = 1.5  # Long responses need more processing time
            elif response_length < 50:
                complexity_factor = 0.8  # Short responses process faster
        
        # Calculate timeout with safety margin
        timeout = estimated_time * complexity_factor * 3  # 3x safety margin
        
        # Enforce bounds
        return max(self.min_timeout, min(self.max_timeout, timeout))
    
    def record_response_time(self, token_count: int, response_time: float, response_length: int = None):
        """Record actual response time to improve estimates"""
        self.response_times.append(response_time)
        
        # Update baseline measurement
        if not self.baseline_measured:
            self.baseline_times.append(response_time)
            if len(self.baseline_times) >= self.baseline_tasks_needed:
                avg_baseline = sum(self.baseline_times) / len(self.baseline_times)
                self.base_timeout = max(self.min_timeout, avg_baseline * 2)  # 2x average as base
                self.baseline_measured = True
                logger.info(f"Dynamic timeout baseline established: {self.base_timeout:.1f}s from {len(self.baseline_times)} samples")
        
        # Update token-to-time ratio (rolling average)
        if token_count > 0:
            observed_ratio = response_time / token_count
            self.token_to_time_ratio = (self.token_to_time_ratio * 0.9) + (observed_ratio * 0.1)
        
        # Cache complex patterns
        if response_length:
            pattern_key = f"{response_length//100}00"  # Group by hundreds
            if pattern_key not in self.complexity_patterns:
                self.complexity_patterns[pattern_key] = []
            self.complexity_patterns[pattern_key].append(response_time)
            
            # Keep only recent patterns (last 50 per group)
            if len(self.complexity_patterns[pattern_key]) > 50:
                self.complexity_patterns[pattern_key] = self.complexity_patterns[pattern_key][-50:]

    def get_stats(self):
        """Get timeout manager statistics"""
        return {
            'baseline_measured': self.baseline_measured,
            'base_timeout': self.base_timeout,
            'token_to_time_ratio': self.token_to_time_ratio,
            'total_responses_recorded': len(self.response_times),
            'avg_response_time': sum(self.response_times) / len(self.response_times) if self.response_times else 0,
            'complexity_patterns_learned': len(self.complexity_patterns)
        }


class SmartConcurrencyManager:
    """Smart concurrency management with real-time timeout monitoring"""
    
    def __init__(self, initial_limit=100, min_limit=20, max_limit=100):
        self.current_limit = initial_limit
        self.min_limit = min_limit
        self.max_limit = max_limit
        
        # CRITICAL FIX: Store shared semaphore as instance variable
        self.semaphore = asyncio.Semaphore(initial_limit)
        
        # Monitoring data
        self.recent_timeouts = []  # Track recent timeout events
        self.timeout_window = 60  # seconds
        self.adjustment_cooldown = 30  # seconds between adjustments
        self.last_adjustment = 0
        
        # Performance tracking
        self.requests_completed = 0
        self.timeouts_occurred = 0
        
    def record_timeout(self):
        """Record a timeout occurrence"""
        now = time.monotonic()
        self.recent_timeouts.append(now)
        self.timeouts_occurred += 1
        
        # Clean old timeouts outside window
        cutoff = now - self.timeout_window
        self.recent_timeouts = [t for t in self.recent_timeouts if t > cutoff]
        
        # Check if adjustment needed
        self._check_adjustment_needed()
    
    def record_success(self):
        """Record a successful completion"""
        self.requests_completed += 1
        
        # Check if we can increase concurrency
        self._check_adjustment_needed()
    
    def _check_adjustment_needed(self):
        """Check if concurrency adjustment is needed"""
        now = time.monotonic()
        
        # Cooldown check
        if now - self.last_adjustment < self.adjustment_cooldown:
            return
        
        # Calculate recent timeout rate
        recent_timeout_count = len(self.recent_timeouts)
        total_recent = self.requests_completed + self.timeouts_occurred
        
        if total_recent < 10:  # Need minimum data
            return
        
        timeout_rate = recent_timeout_count / min(total_recent, 100)  # Look at last 100 requests
        
        old_limit = self.current_limit
        
        if timeout_rate > 0.15:  # >15% timeout rate - reduce concurrency
            self.current_limit = max(self.min_limit, int(self.current_limit * 0.7))
            logger.info(f"High timeout rate ({timeout_rate:.1%}), reducing concurrency: {old_limit} → {self.current_limit}")
            self._resize_semaphore(self.current_limit)
            self.last_adjustment = now
            
        elif timeout_rate < 0.05 and self.current_limit < self.max_limit:  # <5% timeout rate - can increase
            self.current_limit = min(self.max_limit, int(self.current_limit * 1.2))
            logger.info(f"Low timeout rate ({timeout_rate:.1%}), increasing concurrency: {old_limit} → {self.current_limit}")
            self._resize_semaphore(self.current_limit)
            self.last_adjustment = now
    
    def _resize_semaphore(self, new_limit):
        """Resize the shared semaphore to new limit"""
        # Create new semaphore with new limit
        self.semaphore = asyncio.Semaphore(new_limit)
    
    def get_current_semaphore(self):
        """Get the shared semaphore"""
        return self.semaphore
    
    def get_stats(self):
        """Get concurrency manager statistics"""
        total_requests = self.requests_completed + self.timeouts_occurred
        return {
            'current_limit': self.current_limit,
            'total_requests': total_requests,
            'timeouts_occurred': self.timeouts_occurred,
            'timeout_rate': self.timeouts_occurred / total_requests if total_requests > 0 else 0,
            'recent_timeouts': len(self.recent_timeouts)
        }


class TokenUsageLearner:
    """Learn and improve token estimation accuracy over time"""
    
    def __init__(self):
        self.estimation_records = []  # (estimated, actual) pairs
        self.current_ratio = 0.15  # Initial 15% output estimate
        self.min_ratio = 0.10
        self.max_ratio = 0.50
        
    def record_usage(self, estimated_tokens: int, actual_tokens: int):
        """Record actual vs estimated token usage"""
        if estimated_tokens > 0 and actual_tokens > 0:
            self.estimation_records.append((estimated_tokens, actual_tokens))
            
            # Keep only recent records (last 1000)
            if len(self.estimation_records) > 1000:
                self.estimation_records = self.estimation_records[-1000:]
            
            # Update ratio based on recent accuracy
            if len(self.estimation_records) >= 10:
                recent_records = self.estimation_records[-50:]  # Use last 50 for ratio calculation
                
                # Calculate actual output ratio
                total_estimated_input = 0
                total_actual_output = 0
                
                for est, actual in recent_records:
                    # Estimated input was est / (1 + current_ratio)
                    estimated_input = est / (1 + self.current_ratio)
                    actual_output = actual - estimated_input
                    
                    total_estimated_input += estimated_input
                    total_actual_output += max(0, actual_output)
                
                if total_estimated_input > 0:
                    observed_ratio = total_actual_output / total_estimated_input
                    # Smooth update
                    self.current_ratio = (self.current_ratio * 0.8) + (observed_ratio * 0.2)
                    # Enforce bounds
                    self.current_ratio = max(self.min_ratio, min(self.max_ratio, self.current_ratio))
    
    def get_current_ratio(self) -> float:
        """Get current output token estimation ratio"""
        return self.current_ratio
    
    def get_accuracy_stats(self):
        """Get estimation accuracy statistics"""
        if len(self.estimation_records) < 10:
            return {'samples': len(self.estimation_records), 'accuracy': 'insufficient_data'}
        
        recent = self.estimation_records[-100:]  # Last 100 records
        errors = []
        
        for estimated, actual in recent:
            error = abs(estimated - actual) / actual if actual > 0 else 0
            errors.append(error)
        
        avg_error = sum(errors) / len(errors) if errors else 0
        
        return {
            'samples': len(self.estimation_records),
            'current_ratio': self.current_ratio,
            'avg_estimation_error': avg_error,
            'accuracy': 1.0 - avg_error if avg_error < 1.0 else 0.0
        }


class Grader:
    def __init__(
        self, 
        responses: List[models.PreprocessedModel], 
        var_lab: str,
        config: Optional[QualityFilterConfig] = None,
        model_config: Optional[ModelConfig] = None,
        verbose: bool = False,
        prompt_printer = None):
        
        self.responses = responses
        self.question = var_lab
        self.config = config or DEFAULT_QUALITY_FILTER_CONFIG
        self.model_config = model_config or ModelConfig()
        self.model = self.model_config.get_model_for_stage('quality_filter')
        self.grader_instructions = GRADER_INSTRUCTIONS 
        self._results: List[models.QualityFilteredModel] = []
        self.verbose_reporter = VerboseReporter(verbose, capture_logging=True)
        self._stats = ProcessingStats()
        self.prompt_printer = prompt_printer
        
        # Initialize tokenizer for token counting (cached)
        self.encoding = get_tiktoken_encoding(self.model)
        
        # Instructor-patched async OpenAI client for structured output (cached)
        self.client = get_openai_client(OPENAI_API_KEY)
        
        # Rate limiting setup (proven three-layer pattern from spellChecker.py)
        limits = get_openai_rate_limits(self.model)
        HEADROOM = 0.8
        
        self.rpm_limiter = AsyncLimiter(int(limits.requests_per_minute * HEADROOM), 60)
        self.token_bucket = TokenBucket(limits.tokens_per_minute * HEADROOM)
        
        # Smart management systems
        self.timeout_manager = DynamicTimeoutManager(base_timeout=30, min_timeout=15, max_timeout=120)
        self.concurrency_manager = SmartConcurrencyManager(initial_limit=100, min_limit=20, max_limit=100)
        self.token_learner = TokenUsageLearner()
        
        # Rate limit tracking
        self.rate_limit_tracker = RateLimitTracker(cooldown_seconds=15)
        
        # Enhanced stats tracking
        self.stats = {
            'tasks_processed': 0,
            'tasks_successful': 0,
            'tasks_failed': 0,
            'llm_calls_made': 0,
            'llm_calls_successful': 0,
            'llm_calls_failed': 0,
            'processing_time': 0.0,
            'timeouts_occurred': 0,
            'avg_response_time': 0.0,
            'dynamic_timeout_adjustments': 0,
            'concurrency_adjustments': 0
        }

    def _prepare_individual_tasks(self) -> List[Dict]:
        """Prepare individual tasks for processing (individual task approach)"""
        items_to_process = [r for r in self.responses if r.quality_filter_code is None]
        
        tasks = []
        for i, response in enumerate(items_to_process):
            tasks.append({
                'task_id': response.respondent_id,
                'response_text': response.response,
                'original_response': response
            })
        
        return tasks

    def _build_individual_prompt(self, var_lab: str, response_id: str, response_text: str) -> str:
        """Build prompt for individual response assessment"""
        responses_text = f"respondent_id: {response_id}, response: \"{response_text}\""
        return self.grader_instructions.format(
            language=DEFAULT_LANGUAGE,
            var_lab=var_lab,
            responses=responses_text
        )
    
    def count_task_tokens(self, task: Dict) -> int:
        """Count tokens with output estimates (critical for TPM limiting)"""
        prompt = self._build_individual_prompt(self.question, task['task_id'], task['response_text'])
        
        input_tokens = len(self.encoding.encode(prompt))
        
        # Use dynamic output estimation from token learner
        output_ratio = self.token_learner.get_current_ratio()
        estimated_output_tokens = max(50, int(input_tokens * output_ratio))
        
        return input_tokens + estimated_output_tokens

    @retry(
        retry=retry_if_exception_type((
            RateLimitError,          # OpenAI rate limits
            APIConnectionError,      # Connection issues
            APITimeoutError,         # OpenAI timeout errors
            InternalServerError,     # Server-side issues
            InstructorRetryException, # Instructor retry failures
            asyncio.TimeoutError,    # Our asyncio timeout
            ConnectionError,         # Network connection errors
            TimeoutError             # General timeout errors
        )),
        wait=wait_exponential_jitter(initial=2, max=60),  # Longer backoff for connection issues
        stop=stop_after_attempt(5),  # More attempts for network issues
        reraise=True
    )
    async def process_individual_task(self, task: Dict, task_index: int) -> models.QualityFilteredModel:
        """Process individual quality assessment task with dynamic timeout and monitoring"""
        
        tokens_needed = self.count_task_tokens(task)
        
        # Fix data type bug: safely handle float/NaN/None values in response_text
        response_text = task.get('response_text', '')
        if isinstance(response_text, (float, int)):
            # Handle NaN, inf, or numeric values
            if str(response_text).lower() in ['nan', 'inf', '-inf']:
                response_text = ''
            else:
                response_text = str(response_text)
        elif response_text is None:
            response_text = ''
        
        response_length = len(response_text) if response_text else 0
        
        # Using fixed 30-second timeout (simplified from dynamic approach)
        
        # Get current semaphore from concurrency manager (adapts based on timeout rates)
        current_semaphore = self.concurrency_manager.get_current_semaphore()
        
        # Three-layer rate limiting (EXACT ORDER from spellChecker.py)
        async with self.rpm_limiter:                    # 1. RPM check first
            await self.token_bucket.acquire(tokens_needed)  # 2. TPM check second
            async with current_semaphore:                   # 3. Smart transport limit last
                
                task_start_time = time.monotonic()
                
                try:
                    self.stats['llm_calls_made'] += 1
                    
                    # Build prompt for individual task
                    prompt = self._build_individual_prompt(
                        self.question, 
                        task['task_id'], 
                        task['response_text']
                    )
                    
                    # Capture prompt for the first task only
                    if self.prompt_printer and task_index == 0:
                        self.prompt_printer.capture_prompt(
                            step_name="quality_filter",
                            utility_name="QualityFilter",
                            prompt_content=prompt,
                            prompt_type="quality_assessment",
                            metadata={
                                "model": self.model,
                                "var_lab": self.question,
                                "language": DEFAULT_LANGUAGE,
                                "individual_task": True,
                                "fixed_timeout": 30,
                                "estimated_tokens": tokens_needed
                            }
                        )
                    
                    # Use instructor client for structured output with fixed 30s timeout
                    response = await asyncio.wait_for(
                        self.client.chat.completions.create(
                            model=self.model,
                            response_model=List[models.QualityFilteredModel],
                            messages=[{"role": "user", "content": prompt}],
                            temperature=self.config.temperature,
                            max_tokens=self.config.max_tokens,
                            seed=self.model_config.seed
                        ),
                        timeout=30  # Fixed 30-second timeout
                    )
                    
                    # Record successful completion and timing
                    task_end_time = time.monotonic()
                    response_time = task_end_time - task_start_time
                    
                    self.stats['llm_calls_successful'] += 1
                    
                    # Record performance data for learning
                    self.timeout_manager.record_response_time(tokens_needed, response_time, response_length)
                    self.concurrency_manager.record_success()
                    
                    # Record actual token usage if available (for learning)
                    if hasattr(response, 'usage') and response.usage:
                        actual_tokens = response.usage.total_tokens
                        self.token_learner.record_usage(tokens_needed, actual_tokens)
                    
                    # Update stats
                    self.stats['avg_response_time'] = (
                        (self.stats['avg_response_time'] * (self.stats['llm_calls_successful'] - 1) + response_time) /
                        self.stats['llm_calls_successful']
                    )
                    
                    # Extract single result from list response
                    if response and len(response) > 0:
                        return response[0]  # Single task = single result
                    else:
                        # Fallback - no valid response
                        return self.create_fallback_response(task)
                        
                except asyncio.TimeoutError:
                    task_end_time = time.monotonic()
                    response_time = task_end_time - task_start_time
                    
                    logger.warning(f"Task {task['task_id']} timed out after 30s")
                    
                    # Record timeout for monitoring and adjustment
                    self.concurrency_manager.record_timeout()
                    self.stats['llm_calls_failed'] += 1
                    self.stats['timeouts_occurred'] += 1
                    
                    return self.create_fallback_response(task)
                    
                except RateLimitError as e:
                    logger.warning(f"Task {task['task_id']} hit rate limit: {e}")
                    self.stats['llm_calls_failed'] += 1
                    # CRITICAL FIX: Re-raise for tenacity retry
                    raise
                    
                except (APIConnectionError, ConnectionError) as e:
                    logger.warning(f"Task {task['task_id']} connection failed: {e}")
                    self.stats['llm_calls_failed'] += 1
                    # CRITICAL FIX: Re-raise for tenacity retry
                    raise
                    
                except (APITimeoutError, TimeoutError) as e:
                    logger.warning(f"Task {task['task_id']} request timeout: {e}")
                    self.stats['llm_calls_failed'] += 1
                    # CRITICAL FIX: Re-raise for tenacity retry
                    raise
                    
                except InternalServerError as e:
                    logger.warning(f"Task {task['task_id']} server error: {e}")
                    self.stats['llm_calls_failed'] += 1
                    # CRITICAL FIX: Re-raise for tenacity retry
                    raise
                    
                except InstructorRetryException as e:
                    logger.warning(f"Task {task['task_id']} instructor retry failed: {e}")
                    self.stats['llm_calls_failed'] += 1
                    # CRITICAL FIX: Re-raise for tenacity retry
                    raise
                    
                except ValueError as e:
                    # Data type or parsing errors
                    logger.error(f"Task {task['task_id']} data error: {e}")
                    self.stats['llm_calls_failed'] += 1
                    return self.create_fallback_response(task)
                    
                except Exception as e:
                    # Catch-all for unexpected errors
                    logger.error(f"Task {task['task_id']} unexpected error [{type(e).__name__}]: {e}")
                    self.stats['llm_calls_failed'] += 1
                    return self.create_fallback_response(task)
    
    def create_fallback_response(self, task: Dict) -> models.QualityFilteredModel:
        """Create fallback response for failed tasks"""
        original = task['original_response']
        # Return as meaningful response (conservative fallback)
        original.quality_filter = False
        original.quality_filter_code = 0
        return original

    async def process_all_tasks_async(self, tasks: List[Dict]) -> List[models.QualityFilteredModel]:
        """Process all tasks using proven individual task approach (following spellChecker pattern)"""
        
        if not tasks:
            return []
        
        # Rate limiting setup logging (following spellChecker pattern)
        limits = get_openai_rate_limits(self.model)
        HEADROOM = 0.8
        
        print("[SMART RATE LIMITING SETUP]")
        print(f"- Model: {self.model}")
        print(f"- RPM limit: {limits.requests_per_minute:,} ({limits.requests_per_minute * HEADROOM:,.0f} with headroom)")
        print(f"- TPM limit: {limits.tokens_per_minute:,} ({limits.tokens_per_minute * HEADROOM:,.0f} with headroom)")
        print(f"- Processing {len(tasks):,} individual tasks")
        print(f"- Initial concurrent limit: {self.concurrency_manager.current_limit}")
        print(f"- Dynamic timeout: {self.timeout_manager.base_timeout:.1f}s base (adaptive)")
        print(f"- Token estimation: {self.token_learner.current_ratio:.1%} output ratio")
        
        # Create individual task coroutines (NOT batches)
        task_coroutines = [
            self.process_individual_task(task, i) 
            for i, task in enumerate(tasks)
        ]
        
        # Process with protected gathering (CRITICAL)
        print(f"Processing tasks... 0/{len(tasks)} (0.0%)")
        start_time = time.time()
        
        results = await asyncio.gather(*task_coroutines, return_exceptions=True)
        
        processing_time = time.time() - start_time
        
        # Handle results safely (follows spellChecker.py pattern)
        processed_results = []
        successful_tasks = 0
        failed_tasks = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {i} failed with exception: {result}")
                fallback = self.create_fallback_response(tasks[i])
                processed_results.append(fallback)
                failed_tasks += 1
                self.stats['tasks_failed'] += 1
            else:
                processed_results.append(result)
                successful_tasks += 1
                self.stats['tasks_successful'] += 1
        
        # Statistics and reporting (CoderingsTool style)
        success_rate = (successful_tasks / len(tasks)) * 100
        self.stats['processing_time'] = processing_time
        self.stats['tasks_processed'] = len(tasks)
        
        print(f"Processing tasks... {len(tasks)}/{len(tasks)} (100.0%)")
        print(f"• Successful: {successful_tasks}")
        print(f"• Failed: {failed_tasks}")
        print(f"• Success rate: {success_rate:.1f}%")
        print(f"• Timeouts: {self.stats['timeouts_occurred']}")
        print(f"• Avg response time: {self.stats['avg_response_time']:.2f}s")
        
        if processing_time > 1:
            rate = len(tasks) / processing_time
            print(f"• Processing rate: {rate:.1f} tasks/sec")
        
        # Enhanced monitoring statistics
        timeout_stats = self.timeout_manager.get_stats()
        concurrency_stats = self.concurrency_manager.get_stats()
        token_stats = self.token_learner.get_accuracy_stats()
        
        print(f"[SMART MONITORING RESULTS]")
        print(f"• Timeout baseline: {'Established' if timeout_stats['baseline_measured'] else 'Learning'}")
        print(f"• Final concurrent limit: {concurrency_stats['current_limit']} (started at 100)")
        if token_stats['accuracy'] != 'insufficient_data':
            print(f"• Token estimation accuracy: {token_stats['accuracy']:.1%}")
        
        return processed_results

    def grade(self) -> List[models.QualityFilteredModel]:
        self._stats.start_timing()
        self._stats.input_count = len(self.responses)
        
        # Separate items that need LLM evaluation from pre-filtered items
        items_to_process = [r for r in self.responses if r.quality_filter_code is None]
        pre_filtered_items = [r for r in self.responses if r.quality_filter_code is not None]
        
        self.verbose_reporter.step_start("Quality Assessment")
        limits = get_openai_rate_limits(self.model)
        self.verbose_reporter.stat_line(f"Model: {self.model} (Limits: {limits.requests_per_minute} RPM, {limits.tokens_per_minute:,} TPM)")
        self.verbose_reporter.stat_line(f"Items needing LLM evaluation: {len(items_to_process)}")
        self.verbose_reporter.stat_line(f"Pre-filtered items: {len(pre_filtered_items)}")
        
        # Process items that need LLM evaluation using individual task approach
        if items_to_process:
            # Prepare individual tasks
            tasks = self._prepare_individual_tasks()
            
            # Process with proven individual task strategy
            if nest_asyncio:
                nest_asyncio.apply()
            llm_results = asyncio.run(self.process_all_tasks_async(tasks))
            
            # Store LLM results
            self._results = llm_results
        else:
            self.verbose_reporter.stat_line("No items require LLM evaluation")
        
        # Create mapping from respondent_id to LLM results for efficient lookup
        llm_results_map = {result.respondent_id: result for result in self._results}
        
        # Merge results: combine pre-filtered items with LLM results in original order
        merged_results = []
        for original_item in self.responses:
            if original_item.quality_filter_code is not None:
                # Keep pre-filtered item as-is
                merged_results.append(original_item)
            else:
                # Use LLM result if available, otherwise keep original
                if original_item.respondent_id in llm_results_map:
                    merged_results.append(llm_results_map[original_item.respondent_id])
                else:
                    # Fallback: mark as unprocessed (shouldn't happen normally)
                    original_item.quality_filter = False
                    original_item.quality_filter_code = 0  # Assume meaningful if LLM failed
                    merged_results.append(original_item)
        
        # Update self._results to the merged list for statistics and filtering
        self._results = merged_results
        
        # Calculate quality statistics
        quality_counts = {"high": 0, "medium": 0, "low": 0}
        filtered_examples = []
        
        for result in self._results:
            if hasattr(result, 'quality_score'):
                if result.quality_score >= self.config.high_quality_threshold:
                    quality_counts["high"] += 1
                elif result.quality_score >= self.config.medium_quality_threshold:
                    quality_counts["medium"] += 1
                else:
                    quality_counts["low"] += 1
            
            # Only show examples for user-missing (97) or no-answer (99) codes, not system-missing (98)
            if (result.quality_filter and 
                len(filtered_examples) < self.config.max_filter_examples and
                result.quality_filter_code is not None and
                (result.quality_filter_code % 100 == 97 or result.quality_filter_code % 100 == 99)):
                filtered_examples.append(f'"{result.response}" (quality filter: meaningless)')
        
        self._stats.output_count = len([r for r in self._results if not r.quality_filter])
        self._stats.end_timing()
        
        # Report statistics
        total = len(self._results)
        filtered_count = sum(1 for r in self._results if r.quality_filter)
        llm_processed = len(items_to_process)
        
        self.verbose_reporter.stat_line(f"Total responses: {total}")
        self.verbose_reporter.stat_line(f"LLM processed: {llm_processed}")
        self.verbose_reporter.stat_line(f"Pre-filtered: {len(pre_filtered_items)}")
        
        if quality_counts["high"] > 0:
            self.verbose_reporter.stat_line(f"High quality: {quality_counts['high']} responses ({quality_counts['high']/llm_processed*100:.1f}% of LLM processed)" if llm_processed > 0 else "High quality: 0 responses")
        if quality_counts["medium"] > 0:
            self.verbose_reporter.stat_line(f"Medium quality: {quality_counts['medium']} responses ({quality_counts['medium']/llm_processed*100:.1f}% of LLM processed)" if llm_processed > 0 else "Medium quality: 0 responses")
        if quality_counts["low"] > 0:
            self.verbose_reporter.stat_line(f"Low quality: {quality_counts['low']} responses ({quality_counts['low']/llm_processed*100:.1f}% of LLM processed)" if llm_processed > 0 else "Low quality: 0 responses")
        
        self.verbose_reporter.stat_line(f"Total filtered out: {filtered_count} responses ({filtered_count/total*100:.1f}%)")
        
        # Show filtered examples
        if filtered_examples:
            self.verbose_reporter.sample_list("Sample filtered responses", filtered_examples)
        
        self.verbose_reporter.step_complete("Quality filtering completed")
        
        return self._results

    def filter(self) -> List[models.QualityFilteredModel]:
        return [r for r in self._results if not r.quality_filter]

    def summary(self) -> Dict[str, Union[int, float]]:
        total = len(self._results)
        meaningless = sum(1 for r in self._results if r.quality_filter)
        meaningful = total - meaningless
        
        # Count items by how they were processed
        llm_processed = sum(1 for r in self._results if hasattr(r, 'quality_score'))
        pre_filtered = total - llm_processed

        return {
            "total_responses": total,
            "meaningful_responses": meaningful,
            "meaningless_responses": meaningless,
            "meaningful_percentage": round((meaningful / total) * 100, 2) if total > 0 else 0,
            "llm_processed": llm_processed,
            "pre_filtered": pre_filtered
        }