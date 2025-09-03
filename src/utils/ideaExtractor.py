import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import asyncio
import time
import statistics
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from collections import deque

import nest_asyncio
import instructor
from openai import AsyncOpenAI, RateLimitError
import tiktoken
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from asyncio_throttle import Throttler

# === MODELS ========================================================================================================
from pydantic import BaseModel
import models

# === CONFIG ========================================================================================================
from config import OPENAI_API_KEY, DEFAULT_LANGUAGE, ModelConfig, SegmentationConfig, DEFAULT_SEGMENTATION_CONFIG, get_openai_rate_limits
from prompts import IDEA_EXTRACTION_PROMPT

# === UTILS ========================================================================================================
from .verboseReporter import VerboseReporter, ProcessingStats
from .cached_resources import get_openai_client, get_tiktoken_encoding

async_client = get_openai_client(OPENAI_API_KEY)


@dataclass
class OptimalStrategy:
    """Evidence-based optimal processing strategy for idea extraction"""
    target_time_seconds: float
    launch_rate_per_second: float
    concurrent_limit: int
    bottleneck_type: str
    total_requests: int
    total_tokens: int
    safety_factor: float


class WorkloadAnalyzer:
    """Analyzes workload and calculates optimal processing strategy for individual response processing"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.encoding = get_tiktoken_encoding(model_name)
    
    def measure_token_usage(self, sample_prompts: List[str], num_samples: int = 10) -> float:
        """Measure actual token usage from real prompts"""
        if not sample_prompts:
            return 1500  # Conservative fallback
        
        # Sample random prompts if we have many
        sample_size = min(num_samples, len(sample_prompts))
        sampled_prompts = sample_prompts[:sample_size]
        
        token_counts = []
        for prompt in sampled_prompts:
            # Count prompt tokens
            prompt_tokens = len(self.encoding.encode(prompt))
            # Estimate completion tokens (typically 20-30% of prompt for idea extraction)
            completion_tokens = int(prompt_tokens * 0.25)
            total_tokens = prompt_tokens + completion_tokens
            token_counts.append(total_tokens)
        
        return statistics.mean(token_counts)
    
    def calculate_optimal_strategy(self, total_responses: int, avg_tokens_per_request: float) -> OptimalStrategy:
        """Calculate mathematically optimal processing strategy"""
        # Get API limits from config
        rate_limits = get_openai_rate_limits(self.model_name)
        
        # Calculate total resource requirements
        total_requests = total_responses
        total_tokens = total_responses * avg_tokens_per_request
        
        # Calculate minimum time based on constraints
        time_by_requests = total_requests / rate_limits.requests_per_minute * 60
        time_by_tokens = total_tokens / rate_limits.tokens_per_minute * 60
        
        # Find bottleneck and minimum time
        bottleneck_time = max(time_by_requests, time_by_tokens)
        bottleneck_type = 'tokens' if time_by_tokens > time_by_requests else 'requests'
        
        # Apply safety factor (use 95% of capacity like codeAssigner)
        safety_factor = 0.95
        target_time = bottleneck_time / safety_factor
        
        # Calculate optimal launch rate
        optimal_launch_rate = total_requests / target_time
        
        # Calculate concurrent request limit (3 seconds of buffer like codeAssigner)
        concurrent_limit = int(optimal_launch_rate * 3)
        
        return OptimalStrategy(
            target_time_seconds=target_time,
            launch_rate_per_second=optimal_launch_rate,
            concurrent_limit=concurrent_limit,
            bottleneck_type=bottleneck_type,
            total_requests=total_requests,
            total_tokens=total_tokens,
            safety_factor=safety_factor
        )


class SlidingWindowMonitor:
    """Real-time monitoring of API usage with sliding windows"""
    
    def __init__(self, rpm_limit: int, tpm_limit: int, window_seconds: int = 60):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.window_seconds = window_seconds
        
        # Sliding windows for tracking usage
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
    
    def record_request(self, tokens_used: int):
        """Record a completed API request"""
        now = time.time()
        self.requests_window.append(now)
        self.tokens_window.append((now, tokens_used))
        
        self.total_requests += 1
        self.total_tokens += tokens_used
        
        self._cleanup_windows()
    
    def get_current_utilization(self) -> Dict:
        """Get current resource utilization"""
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
    
    def __init__(self, throttler: Throttler, monitor: SlidingWindowMonitor, config: SegmentationConfig, 
                 encoding, model_config: ModelConfig, verbose_reporter: VerboseReporter):
        self.throttler = throttler
        self.monitor = monitor
        self.config = config
        self.client = async_client
        self.model_config = model_config
        self.model = self.model_config.get_model_for_stage('segmentation')
        self.encoding = encoding
        self.verbose_reporter = verbose_reporter
    
    @retry(
        retry=retry_if_exception_type(RateLimitError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    async def make_request(self, prompt: str, respondent_id: str) -> List:
        """Make API request with intelligent retry and rate limiting"""
        
        # Apply precision rate limiting
        async with self.throttler:
            try:
                # Make the API call
                response = await self.client.chat.completions.create(
                    model=self.model,
                    response_model=List[IdeaResponse],
                    max_retries=0,  # Let tenacity handle retries
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    seed=self.model_config.seed
                )
                
                # Record successful request with accurate token count
                estimated_tokens = len(self.encoding.encode(prompt))
                self.monitor.record_request(estimated_tokens)
                
                return response
                
            except Exception as e:
                self.verbose_reporter.error(f"API request failed for respondent {respondent_id}: {str(e)}")
                raise


class IdeaResponse(BaseModel):
    respondent_id: str
    idea_id: str
    idea: str


class IdeaExtractor:
    def __init__(
        self,
        responses: List[models.QualityFilteredModel],
        var_lab: str,
        config: Optional[SegmentationConfig] = None,
        model_config: Optional[ModelConfig] = None,
        verbose: bool = False,
        prompt_printer=None):
        
        self.responses = responses
        self.var_lab = var_lab
        self.config = config or DEFAULT_SEGMENTATION_CONFIG
        self.model_config = model_config or ModelConfig()
        self.model = self.model_config.get_model_for_stage('segmentation')
        self.language = DEFAULT_LANGUAGE
        self._results: List[models.IdeasExtractedModel] = []
        self.verbose_reporter = VerboseReporter(verbose, capture_logging=True)
        self._stats = ProcessingStats()
        self.prompt_printer = prompt_printer
        self._captured_prompt = False
        
        # Initialize components for optimal strategy
        self.workload_analyzer = WorkloadAnalyzer(self.model)
        
        # Initialize rate limits and monitoring
        rate_limits = get_openai_rate_limits(self.model)
        self.rpm_limit = rate_limits.requests_per_minute
        self.tpm_limit = rate_limits.tokens_per_minute
        
        # Initialize tokenizer for batch size calculation (cached)
        self.encoding = get_tiktoken_encoding(self.model)

    def _build_prompt(self, respondent_id: str, response: str) -> str:
        """Build prompt for a single response"""
        return IDEA_EXTRACTION_PROMPT.format(
            var_lab=self.var_lab,
            language=self.language,
            respondent_id=respondent_id,
            response=response
        )

    async def _process_single_response(self, response_data: tuple, api_client: SmartAPIClient) -> models.IdeasExtractedModel:
        """Process a single response and extract ideas using smart API client"""
        idx, respondent_id, response_text = response_data
        
        try:
            # Create prompt
            prompt = self._build_prompt(respondent_id, response_text)
            
            # Capture prompt for debugging if enabled
            if self.prompt_printer and not self._captured_prompt:
                self.prompt_printer.capture_prompt(
                    step_name="idea_extraction",
                    utility_name="IdeaExtractor",
                    prompt_content=prompt,
                    prompt_type="idea_extraction",
                    metadata={
                        "model": self.model,
                        "var_lab": self.var_lab,
                        "language": self.language,
                        "respondent_id": respondent_id
                    }
                )
                self._captured_prompt = True
            
            # Make API call through smart client
            response_data_list = await api_client.make_request(prompt, respondent_id)
            
            # Process response - array of IdeaResponse objects
            ideas = []
            for i, idea_response in enumerate(response_data_list):
                if idea_response.idea:
                    ideas.append(models.IdeasExtractedSubmodel(
                        idea_id=f"{respondent_id}_{i+1}",
                        idea=idea_response.idea
                    ))
            
            return models.IdeasExtractedModel(
                respondent_id=respondent_id,
                response=response_text,
                quality_filter=self.responses[idx].quality_filter,
                quality_filter_code=self.responses[idx].quality_filter_code,
                response_ideas=ideas,
                idea_count=len(ideas)
            )
            
        except Exception as e:
            self.verbose_reporter.error(f"Processing failed for respondent {respondent_id}: {str(e)}")
            # Return error result
            return models.IdeasExtractedModel(
                respondent_id=respondent_id,
                response=response_text,
                quality_filter=self.responses[idx].quality_filter,
                quality_filter_code=self.responses[idx].quality_filter_code,
                response_ideas=[
                    models.IdeasExtractedSubmodel(
                        idea_id=f"{respondent_id}_1",
                        idea="PROCESSING_ERROR"
                    )
                ],
                idea_count=1
            )

    async def _process_with_optimal_strategy(self, all_responses: List[tuple]) -> List[models.IdeasExtractedModel]:
        """Process all responses using evidence-based optimal strategy (like codeAssigner)"""
        
        # Step 1: Analyze workload and calculate optimal strategy
        sample_prompts = [self._build_prompt(resp[1], resp[2]) for resp in all_responses[:10]]
        avg_tokens = self.workload_analyzer.measure_token_usage(sample_prompts)
        strategy = self.workload_analyzer.calculate_optimal_strategy(len(all_responses), avg_tokens)
        
        # Show optimal strategy
        self.verbose_reporter.stat_line(f"Model: {self.model} (Limits: {self.rpm_limit} RPM, {self.tpm_limit:,} TPM)")
        self.verbose_reporter.stat_line(f"Optimal strategy: {strategy.launch_rate_per_second:.1f} req/s, max {strategy.concurrent_limit} concurrent")
        self.verbose_reporter.stat_line(f"Processing {len(all_responses)} responses with individual API calls...")
        
        # Step 2: Initialize precision throttler and monitor
        throttler = Throttler(rate_limit=strategy.launch_rate_per_second, period=1.0)
        monitor = SlidingWindowMonitor(self.rpm_limit, self.tpm_limit)
        api_client = SmartAPIClient(throttler, monitor, self.config, self.workload_analyzer.encoding, 
                                   self.model_config, self.verbose_reporter)
        
        # Step 3: Launch all requests with precision timing (like codeAssigner)
        # Create all tasks - throttler handles the timing
        tasks = [
            asyncio.create_task(self._process_single_response(response_data, api_client))
            for response_data in all_responses
        ]
        
        # Monitor progress
        all_results = []
        completed = 0
        
        # Process results as they complete (like codeAssigner)
        for coro in asyncio.as_completed(tasks):
            result = await coro
            all_results.append(result)
            completed += 1
            
            if completed % 50 == 0 or completed == len(all_responses):
                self.verbose_reporter.progress_line(completed, len(all_responses), "responses")
        
        # Final stats
        final_stats = monitor.get_current_utilization()
        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line(f"Completed in {final_stats['elapsed_time']:.1f}s " +
                                          f"(RPM: {final_stats['rpm_utilization']:.0%}, " +
                                          f"TPM: {final_stats['tpm_utilization']:.0%} utilization)")
        
        return all_results

    def extract(self) -> List[models.IdeasExtractedModel]:
        """Main method to extract ideas from responses using optimal strategy"""
        self._stats.start_timing()
        self._stats.input_count = len(self.responses)
        
        self.verbose_reporter.step_start("Idea Extraction", emoji="💡")
        
        # self.verbose_reporter.empty_line()
        # self.verbose_reporter.stat_line("Idea extraction configuration:")
        # self.verbose_reporter.stat_line(f"  • Model: {self.model}")
        # self.verbose_reporter.stat_line(f"  • Temperature: {self.config.temperature}")
        # self.verbose_reporter.empty_line()
        
        if not self.responses:
            self.verbose_reporter.stat_line("No responses to process")
            return []
        
        # Prepare all responses for processing (individual, like codeAssigner)
        all_responses = [(i, resp.respondent_id, resp.response) for i, resp in enumerate(self.responses)]
        
        # Process with optimal strategy
        if nest_asyncio:
            nest_asyncio.apply()
        self._results = asyncio.run(self._process_with_optimal_strategy(all_responses))
        
        # Ensure all responses are accounted for
        result_ids = {r.respondent_id for r in self._results}
        for response in self.responses:
            if response.respondent_id not in result_ids:
                # Add missing responses with error marker
                self._results.append(models.IdeasExtractedModel(
                    respondent_id=response.respondent_id,
                    response=response.response,
                    quality_filter=response.quality_filter,
                    quality_filter_code=response.quality_filter_code,
                    response_ideas=[
                        models.IdeasExtractedSubmodel(
                            idea_id=f"{response.respondent_id}_1",
                            idea="NOT_PROCESSED"
                        )
                    ],
                    idea_count=1
                ))
        
        self._stats.output_count = len(self._results)
        self._stats.end_timing()
        
        # Calculate statistics
        unique_ideas = set()
        multi_idea_responses = 0
        total_idea_length = 0
        idea_count = 0
        
        # Collect response examples with all their ideas
        response_examples = []
        for resp in self._results:
            if resp.response_ideas and len(resp.response_ideas) > 0:
                if len(resp.response_ideas) > 1:
                    multi_idea_responses += 1
                
                valid_ideas = []
                for idea in resp.response_ideas:
                    if idea.idea and idea.idea not in ["NA", "PROCESSING_ERROR", "NOT_PROCESSED"]:
                        unique_ideas.add(idea.idea)
                        idea_words = idea.idea.split()
                        total_idea_length += len(idea_words)
                        idea_count += 1
                        valid_ideas.append(idea.idea)
                
                # Collect complete response examples
                if valid_ideas and len(response_examples) < self.config.max_code_examples:
                    response_examples.append({
                        'response': resp.response,
                        'ideas': valid_ideas
                    })
        
        # Report statistics
        self.verbose_reporter.stat_line(f"Total responses processed: {len(self._results)}")
        self.verbose_reporter.stat_line(f"Total ideas extracted: {idea_count}")
        self.verbose_reporter.stat_line(f"Unique ideas identified: {len(unique_ideas)}")
        if multi_idea_responses > 0:
            single_idea_responses = len([r for r in self._results if r.response_ideas and len(r.response_ideas) == 1])
            self.verbose_reporter.stat_line(f"Single idea responses: {single_idea_responses} ({single_idea_responses/len(self._results)*100:.1f}%)")
            self.verbose_reporter.stat_line(f"Multiple idea responses: {multi_idea_responses} ({multi_idea_responses/len(self._results)*100:.1f}%)")
        
        # Show idea examples with enhanced format
        if response_examples:
            print("\n📋 Sample extracted ideas:")
            for example in response_examples:
                print(f'  • "{example["response"]}"')
                for idea in example['ideas']:
                    print(f'    → "{idea}"')
                if example != response_examples[-1]:
                    print()
        
        self.verbose_reporter.step_complete("Idea extraction completed")
        
        return self._results

    def summary(self) -> Dict[str, Union[int, float]]:
        """Generate summary statistics"""
        total = len(self._results)
        processed = sum(1 for r in self._results 
                       if r.response_ideas and 
                       not any(idea.idea in ["PROCESSING_ERROR", "NOT_PROCESSED"] 
                              for idea in r.response_ideas))
        failed = total - processed
        
        total_ideas = sum(r.idea_count for r in self._results)
        unique_ideas = len(set(idea.idea for r in self._results 
                              for idea in r.response_ideas 
                              if idea.idea not in ["NA", "PROCESSING_ERROR", "NOT_PROCESSED"]))
        
        return {
            "total_responses": total,
            "processed_responses": processed,
            "failed_responses": failed,
            "success_rate": round((processed / total) * 100, 2) if total > 0 else 0,
            "total_ideas": total_ideas,
            "unique_ideas": unique_ideas,
            "avg_ideas_per_response": round(total_ideas / total, 2) if total > 0 else 0
        }