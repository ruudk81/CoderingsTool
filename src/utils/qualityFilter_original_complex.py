import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import asyncio
import time
import statistics
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from collections import deque

# import instructor
# from openai import AsyncOpenAI, RateLimitError
# import tiktoken
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from asyncio_throttle import Throttler

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

async_client = get_openai_client(OPENAI_API_KEY)


@dataclass
class OptimalStrategy:
    """Evidence-based optimal processing strategy for quality filtering"""
    target_time_seconds: float
    launch_rate_per_second: float
    concurrent_limit: int
    bottleneck_type: str
    total_requests: int
    total_tokens: int
    safety_factor: float
    sub_batch_size: int


class WorkloadAnalyzer:
    """Analyzes workload and calculates optimal processing strategy"""
    
    def __init__(self, model_name: str, encoding):
        self.model_name = model_name
        self.encoding = encoding
    
    def measure_token_usage(self, sample_batches: List[List[tuple]], base_prompt_template: str, var_lab: str) -> float:
        """Measure actual token usage from real batch prompts"""
        if not sample_batches:
            return 1500  # Conservative fallback
        
        token_counts = []
        for batch in sample_batches[:3]:  # Sample first 3 batches
            responses_text = "\n".join(f"respondent_id: {rid}, response: \"{response}\"" for _, rid, response in batch)
            prompt = base_prompt_template.format(
                language=DEFAULT_LANGUAGE,
                var_lab=var_lab,
                responses=responses_text
            )
            prompt_tokens = len(self.encoding.encode(prompt))
            # Estimate completion tokens (typically 20-30% of prompt for structured output)
            completion_tokens = int(prompt_tokens * 0.25)
            total_tokens = prompt_tokens + completion_tokens
            token_counts.append(total_tokens)
        
        return statistics.mean(token_counts) if token_counts else 1500
    
    def calculate_optimal_strategy(self, total_batches: int, avg_tokens_per_batch: float, sub_batches_per_batch: int = 1) -> OptimalStrategy:
        """Calculate evidence-based strategy with rate smoothing"""
        # Get API limits from config
        rate_limits = get_openai_rate_limits(self.model_name)
        
        # Calculate total resource requirements - NO MULTIPLIER for sequential steps
        # For codeGenerator: sub_batches_per_batch should be 1 (sequential steps don't multiply load)
        total_requests = total_batches  # Each cluster = 1 request active at a time
        total_tokens = total_requests * avg_tokens_per_batch
        
        # Calculate optimal sustained rate (what we can maintain)
        optimal_sustained_rate = rate_limits.requests_per_minute / 60  # req/sec
        optimal_sustained_tokens = rate_limits.tokens_per_minute / 60  # tokens/sec
        
        # Find bottleneck for sustained rate
        time_by_requests = total_requests / optimal_sustained_rate
        time_by_tokens = total_tokens / optimal_sustained_tokens
        
        # Use evidence-based approach: plan for sustained rate
        bottleneck_time = max(time_by_requests, time_by_tokens)
        bottleneck_type = 'tokens' if time_by_tokens > time_by_requests else 'requests'
        
        # Apply aggressive utilization (95% for Phase 1)
        safety_factor = 0.95
        target_time = bottleneck_time / safety_factor
        
        # Calculate aggressive launch rate
        optimal_launch_rate = total_requests / target_time
        
        # Aggressive concurrent limit (3-second burst capacity instead of 5)
        concurrent_limit = int(optimal_launch_rate * 3)
        
        return OptimalStrategy(
            target_time_seconds=target_time,
            launch_rate_per_second=optimal_launch_rate,
            concurrent_limit=concurrent_limit,
            bottleneck_type=bottleneck_type,
            total_requests=total_requests,
            total_tokens=total_tokens,
            safety_factor=safety_factor,
            sub_batch_size=sub_batches_per_batch
        )


class SlidingWindowMonitor:
    """Real-time monitoring of API usage with sliding windows"""
    
    def __init__(self, rpm_limit: int, tpm_limit: int, window_seconds: int = 60):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.window_seconds = window_seconds
        
        # Thread-safe tracking across all concurrent operations
        self._lock = asyncio.Lock()
        self.requests_window = deque()  # timestamps
        self.tokens_window = deque()    # (timestamp, token_count) tuples
        
        # Statistics
        self.total_requests = 0
        self.total_tokens = 0
        self.start_time = time.time()
    
    def _cleanup_windows(self):
        """Remove entries older than window_seconds"""
        cutoff_time = time.time() - self.window_seconds
        
        # Clean requests window
        while self.requests_window and self.requests_window[0] < cutoff_time:
            self.requests_window.popleft()
        
        # Clean tokens window
        while self.tokens_window and self.tokens_window[0][0] < cutoff_time:
            self.tokens_window.popleft()
    
    async def can_proceed(self, estimated_tokens: int = 0) -> bool:
        """Check if we can make request within 99% of 60-second limits (Phase 3 maximum aggression)"""
        async with self._lock:
            self._cleanup_windows()
            
            # Calculate current usage in 60-second window
            current_rpm = len(self.requests_window)
            current_tpm = sum(tokens for _, tokens in self.tokens_window)
            
            # Maximum aggression: use 99% of limits (Phase 3)
            would_exceed_rpm = (current_rpm + 1) > (self.rpm_limit * 0.99)
            would_exceed_tpm = (current_tpm + estimated_tokens) > (self.tpm_limit * 0.99)
            
            return not (would_exceed_rpm or would_exceed_tpm)
    
    async def record_request(self, tokens_used: int):
        """Record a completed API request (async for thread safety)"""
        async with self._lock:
            now = time.time()
            self.requests_window.append(now)
            self.tokens_window.append((now, tokens_used))
            
            self.total_requests += 1
            self.total_tokens += tokens_used
            
            self._cleanup_windows()
    
    async def get_current_utilization(self) -> Dict:
        """Get current resource utilization (async for thread safety)"""
        async with self._lock:
            self._cleanup_windows()
            
            current_rpm = len(self.requests_window)
            current_tpm = sum(tokens for _, tokens in self.tokens_window)
            
            return {
                'current_rpm': current_rpm,
                'current_tpm': current_tpm,
                'rpm_utilization': current_rpm / self.rpm_limit,
                'tpm_utilization': current_tpm / self.tpm_limit,
                'rpm_remaining': self.rpm_limit - current_rpm,
                'tpm_remaining': self.tpm_limit - current_tpm,
                'total_requests': self.total_requests,
                'total_tokens': self.total_tokens,
                'elapsed_time': time.time() - self.start_time
            }


class SmartAPIClient:
    """API client with intelligent retry logic and precise rate limiting"""
    
    def __init__(self, throttler: Throttler, monitor: SlidingWindowMonitor, config: QualityFilterConfig, 
                 encoding, model_config: ModelConfig, verbose_reporter: VerboseReporter):
        self.throttler = throttler
        self.monitor = monitor
        self.config = config
        self.client = async_client
        self.model_config = model_config
        self.model = self.model_config.get_model_for_stage('quality_filter')
        self.encoding = encoding
        self.verbose_reporter = verbose_reporter
    
    @retry(
        retry=retry_if_exception_type(RateLimitError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    async def make_request(self, prompt: str, batch_info: str) -> List[models.QualityFilteredModel]:
        """Make API request with intelligent retry and rate limiting"""
        
        # Apply precision rate limiting
        async with self.throttler:
            try:
                # Make the API call
                response = await self.client.chat.completions.create(
                    model=self.model,
                    response_model=List[models.QualityFilteredModel],
                    max_retries=0,  # Let tenacity handle retries
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    seed=self.model_config.seed
                )
                
                # Record successful request with accurate token count
                estimated_tokens = len(self.encoding.encode(prompt))
                await self.monitor.record_request(estimated_tokens)
                
                return response
                
            except Exception as e:
                self.verbose_reporter.error(f"API request failed for {batch_info}: {str(e)}")
                raise


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
        
        # Initialize tokenizer for batch size calculation (cached)
        self.encoding = get_tiktoken_encoding(self.model)
        
        # Initialize workload analyzer
        self.workload_analyzer = WorkloadAnalyzer(self.model, self.encoding)
        
        # Get rate limits
        rate_limits = get_openai_rate_limits(self.model)
        self.rpm_limit = rate_limits.requests_per_minute
        self.tpm_limit = rate_limits.tokens_per_minute

    # Removed: _calculate_token_budget and _create_sub_batches (no longer needed with individual task approach)

    def _batch(self) -> List[List[tuple]]:
        """Create token-aware batches with pre-calculated token caching"""
        # Only batch items that need LLM evaluation (quality_filter_code = None)
        items_to_process = [r for r in self.responses if r.quality_filter_code is None]
        
        if not items_to_process:
            return []
        
        token_budget = self._calculate_token_budget()
        
        # Pre-calculate and cache token counts for responses
        response_tokens = []
        for r in items_to_process:
            response_text = f"respondent_id: {r.respondent_id}, response: \"{r.response}\""
            tokens = len(self.encoding.encode(response_text))
            response_tokens.append(tokens)
        
        # Calculate adaptive batch size based on average response length
        avg_tokens = sum(response_tokens) / max(1, len(response_tokens))
        adaptive_max_batch = min(self.config.batch_size, max(1, int(token_budget / max(1, avg_tokens))))
        
        batches = []
        current_batch = []
        current_tokens = 0
        
        for i, (response, tokens) in enumerate(zip(items_to_process, response_tokens)):
            indexed_item = (i, response.respondent_id, response.response)
            
            # Handle oversized responses
            if tokens > token_budget and not current_batch:
                self.verbose_reporter.warning(f"Response from {response.respondent_id} exceeds token budget ({tokens} > {token_budget})")
                batches.append([indexed_item])
                continue
            
            # Check if adding this response would exceed limits
            if (current_tokens + tokens > token_budget or 
                len(current_batch) >= adaptive_max_batch):
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_tokens = 0
            
            current_batch.append(indexed_item)
            current_tokens += tokens
        
        if current_batch:
            batches.append(current_batch)
        
        return batches

    def _build_prompt(self, var_lab: str, batch: List[tuple]) -> str:
        responses_text = "\n".join(f"respondent_id: {rid}, response: \"{response}\"" for _, rid, response in batch)
        return self.grader_instructions.format(
            language=DEFAULT_LANGUAGE,
            var_lab=var_lab,
            responses=responses_text)

    async def _process_sub_batch(self, sub_batch: List[tuple], batch_index: int, sub_batch_index: int, 
                                 api_client: SmartAPIClient) -> List[models.QualityFilteredModel]:
        """Process a single sub-batch of responses with smart retry logic"""
        prompt = self._build_prompt(self.question, sub_batch)
        
        # Capture prompt only for the first sub-batch of the first batch
        if self.prompt_printer and batch_index == 0 and sub_batch_index == 0:
            self.prompt_printer.capture_prompt(
                step_name="quality_filter",
                utility_name="QualityFilter",
                prompt_content=prompt,
                prompt_type="quality_assessment",
                metadata={
                    "model": self.model,
                    "var_lab": self.question,
                    "language": DEFAULT_LANGUAGE,
                    "sub_batch_size": len(sub_batch),
                    "batch_number": batch_index + 1,
                    "sub_batch_number": sub_batch_index + 1
                }
            )
        
        try:
            batch_info = f"batch {batch_index + 1}, sub-batch {sub_batch_index + 1}"
            response_data = await api_client.make_request(prompt, batch_info)
            return response_data
        except Exception as e:
            self.verbose_reporter.error(f"Sub-batch {sub_batch_index + 1} of batch {batch_index + 1} processing failed: {str(e)}")
            # Return empty results for failed sub-batch
            return []

    async def _process_with_optimal_strategy(self, batches: List[List[tuple]]) -> List[models.QualityFilteredModel]:
        """Process all batches using evidence-based optimal strategy"""
        
        # Calculate sub-batches for each batch
        sub_batch_size = 5
        total_sub_batches = sum(len(self._create_sub_batches(batch, sub_batch_size)) for batch in batches)
        
        # Analyze workload and calculate optimal strategy
        avg_tokens = self.workload_analyzer.measure_token_usage(
            batches[:3], self.grader_instructions, self.question
        )
        strategy = self.workload_analyzer.calculate_optimal_strategy(
            len(batches), avg_tokens, sub_batch_size
        )
        
        # Show optimal strategy
        self.verbose_reporter.stat_line(f"Optimal strategy: {strategy.launch_rate_per_second:.1f} req/s, max {strategy.concurrent_limit} concurrent")
        self.verbose_reporter.stat_line(f"Processing {len(self.responses)} responses in {len(batches)} batches ({total_sub_batches} sub-batches)...")
        
        # Initialize precision throttler and monitor
        throttler = Throttler(rate_limit=strategy.launch_rate_per_second, period=1.0)
        monitor = SlidingWindowMonitor(self.rpm_limit, self.tpm_limit)
        api_client = SmartAPIClient(throttler, monitor, self.config, self.encoding, 
                                   self.model_config, self.verbose_reporter)
        
        # Create all tasks with throttling
        all_tasks = []
        for batch_idx, batch in enumerate(batches):
            sub_batches = self._create_sub_batches(batch, sub_batch_size)
            for sub_batch_idx, sub_batch in enumerate(sub_batches):
                task = asyncio.create_task(
                    self._process_sub_batch(sub_batch, batch_idx, sub_batch_idx, api_client)
                )
                all_tasks.append(task)
        
        # Process results as they complete
        all_results = []
        completed = 0
        
        for coro in asyncio.as_completed(all_tasks):
            result = await coro
            all_results.extend(result)
            completed += 1
            
            # Progress reporting
            if completed % 10 == 0 or completed == len(all_tasks):
                self.verbose_reporter.progress_line(completed, len(all_tasks), "sub-batches")
        
        # Final stats
        final_stats = await monitor.get_current_utilization()
        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line(f"Completed in {final_stats['elapsed_time']:.1f}s " +
                                          f"(RPM: {final_stats['rpm_utilization']:.0%}, " +
                                          f"TPM: {final_stats['tpm_utilization']:.0%} utilization)")
        
        return all_results

    def grade(self) -> List[models.QualityFilteredModel]:
        self._stats.start_timing()
        self._stats.input_count = len(self.responses)
        
        # Separate items that need LLM evaluation from pre-filtered items
        items_to_process = [r for r in self.responses if r.quality_filter_code is None]
        pre_filtered_items = [r for r in self.responses if r.quality_filter_code is not None]
        
        self.verbose_reporter.step_start("Quality Assessment")
        self.verbose_reporter.stat_line(f"Model: {self.model} (Limits: {self.rpm_limit} RPM, {self.tpm_limit:,} TPM)")
        self.verbose_reporter.stat_line(f"Items needing LLM evaluation: {len(items_to_process)}")
        self.verbose_reporter.stat_line(f"Pre-filtered items: {len(pre_filtered_items)}")
        
        # Only process items that need LLM evaluation
        if items_to_process:
            # Temporarily replace self.responses for batching
            original_responses = self.responses
            self.responses = items_to_process
            
            # Get batches
            batches = self._batch()
            
            # Process with optimal strategy
            if nest_asyncio:
                nest_asyncio.apply()
            llm_results = asyncio.run(self._process_with_optimal_strategy(batches))
            
            # Restore original responses
            self.responses = original_responses
            
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