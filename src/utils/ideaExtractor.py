import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import asyncio
import time
import statistics
import itertools
import logging
from typing import Dict, List, Optional, Union, Literal
from dataclasses import dataclass
from collections import deque
import numpy as np

import nest_asyncio
#import instructor
from openai import RateLimitError, APIConnectionError, APITimeoutError, InternalServerError #AsyncOpenAI
#import tiktoken
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type
from instructor.exceptions import InstructorRetryException
from aiolimiter import AsyncLimiter

logger = logging.getLogger(__name__)

# === MODELS ========================================================================================================
from pydantic import BaseModel, Field, field_validator
import models

# === CONFIG ========================================================================================================
from config import OPENAI_API_KEY, DEFAULT_LANGUAGE, ModelConfig, SegmentationConfig, DEFAULT_SEGMENTATION_CONFIG, ProcessingConfig, DEFAULT_PROCESSING_CONFIG, get_openai_rate_limits
from prompts import IDEA_EXTRACTION_PROMPT, EXTRACT_SUBJECT

# === UTILS ========================================================================================================
from .verboseReporter import VerboseReporter, ProcessingStats
from .cached_resources import get_openai_client, get_tiktoken_encoding

async_client = get_openai_client(OPENAI_API_KEY)

# ============================================================================
# RATE LIMITING CLASSES (upgraded to match qualityFilter.py)
# ============================================================================

class TokenBucket:
    """Simple token bucket for TPM limiting"""
    def __init__(self, tokens_per_minute):
        self.tpm = tokens_per_minute
        self.available = tokens_per_minute
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()
    
    async def acquire(self, tokens_needed):
        """Acquire tokens, returning wait time if not available"""
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            # Regenerate tokens based on time elapsed
            self.available = min(self.tpm, self.available + (self.tpm * elapsed / 60))
            self.last_update = now
            
            if self.available >= tokens_needed:
                self.available -= tokens_needed
                return True
            else:
                # Calculate wait time
                deficit = tokens_needed - self.available
                wait_seconds = deficit * 60 / self.tpm
                return wait_seconds
    
    async def wait_and_acquire(self, tokens_needed):
        """Wait if necessary and acquire tokens"""
        while True:
            result = await self.acquire(tokens_needed)
            if result is True:
                return
            else:
                # result is wait_seconds
                await asyncio.sleep(result)
    
    async def reconcile(self, delta_tokens):
        """Reconcile actual vs estimated tokens"""
        if delta_tokens < 0:
            async with self.lock:
                self.available = min(self.tpm, self.available - delta_tokens)


class LatencyTracker:
    """Simple EMA tracker for latencies"""
    def __init__(self, processing_config: Optional[ProcessingConfig] = None):
        self.processing_config = processing_config or DEFAULT_PROCESSING_CONFIG
        self.ema = None
        self.alpha = self.processing_config.latency_tracker_ema_alpha
        self.values = deque(maxlen=self.processing_config.latency_tracker_samples_window)
    
    def add(self, value):
        """Add a latency measurement"""
        self.values.append(value)
        if self.ema is None:
            self.ema = value
        else:
            self.ema = self.alpha * value + (1 - self.alpha) * self.ema
    
    def get_timeout(self, est_tokens):
        """Calculate timeout based on EMA and token count with configurable bounds"""
        config = self.processing_config
        if not self.values:
            return max(config.adaptive_timeout_min_seconds, 30.0)

        # Use P95 latency as base
        p95 = np.percentile(list(self.values), 95)
        # Simple linear scaling with token count
        token_factor = est_tokens / 1000
        timeout = p95 + (token_factor * 0.1)
        # Apply margin and configurable bounds
        return max(config.adaptive_timeout_min_seconds, min(config.adaptive_timeout_max_seconds, timeout * config.adaptive_timeout_margin))
    
    def get_avg_latency(self):
        """Get average latency for concurrency calculations"""
        if not self.values:
            return 2.0  # Default 2s
        return self.ema if self.ema is not None else 2.0


# === BOOTSTRAP MEASUREMENT SYSTEM ========================================================================================================

@dataclass
class ApiLimits:
    """API limits structure for bootstrap calculations"""
    tokens_per_minute: int
    requests_per_minute: int


def compute_optimal_concurrency(limits: ApiLimits, latency_seconds: float, avg_tokens: float, processing_config: Optional[ProcessingConfig] = None, cap: Optional[int] = None, min_conc: Optional[int] = None, headroom: Optional[float] = None) -> int:
    """Compute optimal concurrency using Little's Law"""
    config = processing_config or DEFAULT_PROCESSING_CONFIG
    cap = cap if cap is not None else config.concurrency_cap_default
    min_conc = min_conc if min_conc is not None else config.concurrency_min_default
    headroom = headroom if headroom is not None else config.rate_limit_headroom

    latency_seconds = max(float(latency_seconds or 0.5), 0.05)
    avg_tokens = max(float(avg_tokens or 1.0), 1.0)

    rpm_throughput = limits.requests_per_minute * headroom / 60
    tpm_throughput = limits.tokens_per_minute * headroom / avg_tokens / 60
    candidates = [rpm_throughput, tpm_throughput]
    allowed_rps = max(min(candidates), 0.0)
    target = allowed_rps * latency_seconds   # Little's Law

    return int(max(min(target, cap), min_conc))


async def bootstrap_measure_async(call_fn, n_probes: int = 3):
    """Run n_probes serial calls and return (avg_latency_s, avg_tokens). call_fn() -> usage dict."""
    latencies, tokens = [], []
    for _ in range(n_probes):
        t0 = time.perf_counter()
        usage = await call_fn()  # Let tenacity handle timeouts and retries
        t1 = time.perf_counter()
        latencies.append(max(t1 - t0, 0.001))
        pt = int(usage.get("prompt_tokens", 0))
        ct = int(usage.get("completion_tokens", 0))
        tokens.append(max(pt + ct, 1))
    return sum(latencies)/len(latencies), sum(tokens)/len(tokens)


# === PYDANTIC MODELS ========================================================================================================

class SubjectExtractionResponse(BaseModel):
    """Response model for subject/actor extraction"""
    decision: Literal["CANONICAL_SUBJECT", "CANONICAL_ACTOR"] = Field(default="CANONICAL_SUBJECT", description="Whether to use subject or actor phrasing")
    canonical_term: str = Field(description="The canonical subject or actor as a single word or short phrase")
    canonical_phrasing: str = Field(description="Template with canonical term and verb/state")

class IdeaResponse(BaseModel):
    """Pydantic model for idea extraction response - matches prompt output format"""
    respondent_id: str = Field(alias_choices=['respondent_id', 'respond_id', 'respondent', 'respondrespondent_id'])
    idea_id: str = Field(default="1", alias_choices=['idea_id', 'id'])
    idea: str = Field(description="The extracted idea text", alias_choices=['idea', 'content'])

    # Class variable to hold expected template prefix (set dynamically per extraction)
    _expected_template_prefix: str = ""

    @field_validator('idea')
    @classmethod
    def validate_template_compliance(cls, v: str) -> str:
        """Ensure idea follows the required phrasing template"""
        if not v or v in ["", "NA", "N/A"]:
            return v

        # Get expected prefix (set by extractor before calling API)
        expected_prefix = getattr(cls, '_expected_template_prefix', '')

        if expected_prefix and not v.startswith(expected_prefix):
            # Validation fails - instructor will retry with error message
            raise ValueError(
                f"Idea must start with the required template prefix: '{expected_prefix}'. "
                f"Got: '{v}'. "
                f"Please reformulate to match the template exactly."
            )

        return v


# === MAIN IDEA EXTRACTOR CLASS ========================================================================================================

class IdeaExtractor:
    def __init__(
        self,
        responses: List[models.QualityFilteredModel],
        var_lab: str,
        config: Optional[SegmentationConfig] = None,
        model_config: Optional[ModelConfig] = None,
        processing_config: Optional[ProcessingConfig] = None,
        verbose: bool = False,
        prompt_printer=None,
        verbose_reporter: Optional['VerboseReporter'] = None):

        self.responses = responses
        self.var_lab = var_lab
        self.config = config or DEFAULT_SEGMENTATION_CONFIG
        self.model_config = model_config or ModelConfig()
        self.processing_config = processing_config or DEFAULT_PROCESSING_CONFIG
        self.model = self.model_config.get_model_for_stage('segmentation')
        self.language = DEFAULT_LANGUAGE
        self._results: List[models.IdeasExtractedModel] = []
        self.verbose_reporter = verbose_reporter or VerboseReporter(verbose, capture_logging=True)
        self._stats = ProcessingStats()
        self.prompt_printer = prompt_printer
        self._captured_prompt = False

        # Initialize tokenizer for token estimation (cached)
        self.encoding = get_tiktoken_encoding(self.model)

        # Initialize OpenAI client (cached)
        self.client = get_openai_client(OPENAI_API_KEY)

        # Rate limiting setup
        limits = get_openai_rate_limits(self.model)

        # Token bucket for TPM limiting
        self.tpm_bucket = TokenBucket(limits.tokens_per_minute * self.processing_config.rate_limit_headroom)
        
        # Adaptive token estimation (following qualityFilter strategy)
        self.input_token_history = deque(maxlen=3)  # First 3 input token counts
        self.output_token_history = deque(maxlen=5)  # First 5 output token counts
        self.estimation_errors = deque(maxlen=50)  # Track accuracy
        self.first_prompt_tokens = None  # Cache first prompt calculation
        
        # Rolling average of actual total tokens for comparison
        self.actual_total_tokens = deque(maxlen=50)  # Track actual total usage
        
        # Latency tracking
        self.latency_tracker = LatencyTracker(processing_config=self.processing_config)
        
        # Calculate initial average tokens estimate
        self.avg_tokens = self._calculate_avg_tokens()
        
        # Rate limiting components (will be initialized after bootstrap)
        self.rate_limiter = None
        self.semaphore = None
        self.optimal_concurrency = None
        
        # Stats (matching qualityFilter pattern)
        self.stats = {
            'tasks_processed': 0,
            'tasks_successful': 0,
            'tasks_failed': 0,
            'retries': 0,
            'rate_limits': 0,
            'timeouts': 0
        }
        
        # Cache for subject extraction to avoid redundant LLM calls
        self._subject_cache = {}

    def _calculate_avg_tokens(self) -> int:
        """Calculate average tokens per request for rate limiting"""
        if not self.responses:
            return 1500  # Conservative fallback
        
        # Sample up to 10 responses for token estimation
        sample_size = min(10, len(self.responses))
        sample_responses = self.responses[:sample_size]
        
        token_counts = []
        # For initial token estimation, use placeholder values
        # The actual subject will be extracted during processing
        placeholder_canonical_phrasing = "Canonical form: the subject"
        placeholder_phrasing_template = "Template: '[the subject] [should/needs to/must/is/are] [property or outcome]'"
        
        for response in sample_responses:
            prompt = self._build_prompt(
                response.respondent_id, 
                response.response, 
                placeholder_canonical_phrasing,
                placeholder_phrasing_template
            )
            prompt_tokens = len(self.encoding.encode(prompt))
            # Estimate completion tokens (25% of prompt for idea extraction)
            completion_tokens = int(prompt_tokens * 0.25)
            token_counts.append(prompt_tokens + completion_tokens)
        
        return int(statistics.mean(token_counts)) if token_counts else 1500

    async def _extract_subject(self, survey_question: str) -> SubjectExtractionResponse:
        """Extract canonical subject/actor from survey question with caching"""

        if survey_question in self._subject_cache:
            return self._subject_cache[survey_question]
        
        try:
            
            prompt = EXTRACT_SUBJECT.format(language=self.language, survey_question=survey_question)
            
            response = await self.client.chat.completions.create(
                model=self.model,
                response_model=SubjectExtractionResponse,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,  # Use deterministic temperature for consistency
                seed=self.model_config.seed)

            # Capture subject extraction prompt if prompt_printer enabled
            if self.prompt_printer:
                self.prompt_printer.capture_prompt(
                    step_name="idea_extraction_subject",
                    utility_name="IdeaExtractor",
                    prompt_content=prompt,
                    prompt_type="subject_extraction",
                    metadata={
                        "model": self.model,
                        "survey_question": survey_question,
                        "language": self.language
                    }
                )

            self._subject_cache[survey_question] = response

            if self.verbose_reporter.enabled:
                self.verbose_reporter.stat_line(f"Extracted canonical_term: '{response.canonical_term}'")
                self.verbose_reporter.stat_line(f"Extracted canonical_phrasing: '{response.canonical_phrasing}'")
            
            return response
            
        except Exception as e:
            logger.warning(f"Subject extraction failed: {e}. Using fallback.")
            # Fallback: extract a reasonable default from the question
            fallback = SubjectExtractionResponse(
                decision="CANONICAL_SUBJECT",
                canonical_term="the subject",
                canonical_phrasing="the subject [ATTRIBUTE_OR_ACTION]"
            )
            self._subject_cache[survey_question] = fallback
            return fallback

    def _build_prompt(self, respondent_id: str, response: str, canonical_phrasing: str, phrasing_template: str) -> str:
        """Build prompt for a single response"""
        return IDEA_EXTRACTION_PROMPT.format(
            var_lab=self.var_lab,
            subject=canonical_phrasing,
            phrasing_template=phrasing_template,
            language=self.language,
            respondent_id=respondent_id,
            response=response
        )
    
    def estimate_tokens(self, prompt: str) -> int:
        """Estimate total tokens using adaptive strategy (matching qualityFilter)"""
        actual_input_tokens = len(self.encoding.encode(prompt))
        
        # Input estimation: first prompt + 15%, then average of first 3
        if self.first_prompt_tokens is None:
            # First prompt: use actual + 15% margin
            self.first_prompt_tokens = actual_input_tokens
            estimated_input = int(actual_input_tokens * 1.15)
        elif len(self.input_token_history) < 3:
            # Still collecting data: use actual + 15%
            estimated_input = int(actual_input_tokens * 1.15)
        else:
            # Use average of first 3 actual inputs
            avg_input = sum(self.input_token_history) / len(self.input_token_history)
            estimated_input = int(avg_input)
        
        # Track input tokens for learning
        if len(self.input_token_history) < 3:
            self.input_token_history.append(actual_input_tokens)
        
        # Output estimation: 25% of input, then average of first 5 responses
        if len(self.output_token_history) < 5:
            # Use 25% of input as estimate (idea extraction typically has more output than qualityFilter)
            estimated_output = int(estimated_input * 0.25)
        else:
            # Use average of first 5 actual outputs
            avg_output = sum(self.output_token_history) / len(self.output_token_history)
            estimated_output = int(avg_output)
        
        # Ensure we don't exceed max_tokens
        estimated_output = min(self.config.max_tokens, estimated_output)
        
        total_estimate = estimated_input + estimated_output
        
        return total_estimate
    
    def get_token_estimation_stats(self) -> dict:
        """Get token estimation accuracy statistics (matching qualityFilter)"""
        if not self.estimation_errors:
            return {"status": "collecting_data", "samples": 0}
        
        avg_error = sum(self.estimation_errors) / len(self.estimation_errors)
        avg_input = sum(self.input_token_history) / len(self.input_token_history) if self.input_token_history else 0
        avg_output = sum(self.output_token_history) / len(self.output_token_history) if self.output_token_history else 0
        avg_actual_total = sum(self.actual_total_tokens) / len(self.actual_total_tokens) if self.actual_total_tokens else 0
        
        return {
            "status": "learning",
            "samples": len(self.estimation_errors),
            "avg_estimation_error": avg_error,
            "avg_input_tokens": avg_input,
            "avg_output_tokens": avg_output,
            "avg_actual_total_tokens": avg_actual_total,
            "initial_avg_tokens": self.avg_tokens,
            "input_samples": len(self.input_token_history),
            "output_samples": len(self.output_token_history),
            "actual_samples": len(self.actual_total_tokens)
        }
    
    def get_token_bucket_status(self) -> dict:
        """Get current token bucket status (matching qualityFilter)"""
        available_pct = (self.tpm_bucket.available / self.tpm_bucket.tpm) * 100
        
        # Calculate real utilization based on consumption rate vs capacity
        if len(self.actual_total_tokens) >= 10:
            # Use actual consumption rate over last 10 samples
            recent_avg = sum(list(self.actual_total_tokens)[-10:]) / 10
            # Convert to per-second rate (assuming ~2s per request for rough estimate)
            consumption_rate_per_sec = recent_avg / 2.0
            # Calculate utilization as percentage of per-second capacity
            real_utilization_pct = (consumption_rate_per_sec / (self.tpm_bucket.tpm / 60)) * 100
        else:
            # Fallback to bucket level method for early samples
            real_utilization_pct = 100 - available_pct
        
        return {
            "available_tokens": int(self.tpm_bucket.available),
            "capacity": self.tpm_bucket.tpm,
            "utilization_pct": real_utilization_pct,
            "low_tokens": available_pct < 10,
            "consumption_rate": consumption_rate_per_sec if len(self.actual_total_tokens) >= 10 else 0
        }
    
    @retry(
        retry=retry_if_exception_type((
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            InstructorRetryException,
            asyncio.TimeoutError
        )),
        wait=wait_exponential_jitter(initial=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True
    )
    async def probe_call_no_structured(self, task_dict):
        """Probe call without structured output for bootstrap measurement"""
        # For probe calls, use placeholder values to avoid extra API calls
        placeholder_canonical_phrasing = "Canonical subject: the subject"
        placeholder_phrasing_template = "SUBJECT template: '[the subject] [should/needs to/must/is/are] [property or outcome]'"
        
        prompt = self._build_prompt(
            task_dict['respondent_id'], 
            task_dict['response'], 
            placeholder_canonical_phrasing,
            placeholder_phrasing_template
        )
        
        # For probes: avoid response_model so we can read .usage
        resp = await asyncio.wait_for(
            self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                seed=self.model_config.seed
            ),
            timeout=30.0  # Conservative timeout for bootstrap
        )

        u = getattr(resp, "usage", None)
        return {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens}

    @retry(
        retry=retry_if_exception_type((
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            InstructorRetryException,
            asyncio.TimeoutError
        )),
        wait=wait_exponential_jitter(initial=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True
    )
    async def process_task(self, task: Dict) -> models.IdeasExtractedModel:
        """Process a single idea extraction task (following qualityFilter pattern)"""
        task_start = time.perf_counter()
        
        try:
            # Extract subject first (will be cached after first call)
            subject_response = await self._extract_subject(self.var_lab)
            subject = subject_response.canonical_term
            phrasing_template = subject_response.canonical_phrasing

            # Extract the template prefix (everything before [ATTRIBUTE_OR_ACTION])
            template_prefix = phrasing_template.split('[ATTRIBUTE_OR_ACTION]')[0].strip() if '[ATTRIBUTE_OR_ACTION]' in phrasing_template else phrasing_template

            # Set the expected prefix for Pydantic validation
            IdeaResponse._expected_template_prefix = template_prefix

            # Build prompt with subject
            prompt = self._build_prompt(task['respondent_id'], task['response'], subject, phrasing_template)
            
            # Capture prompt for debugging if enabled (first time only)
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
                        "respondent_id": task['respondent_id'],
                        "canonical_term_used": subject,
                        "canonical_phrasing_used": phrasing_template,
                        "cache_hit": self.var_lab in self._subject_cache
                    }
                )
                self._captured_prompt = True
            
            # Estimate tokens using adaptive strategy (matching qualityFilter)
            est_tokens = self.estimate_tokens(prompt)
            
            # Log estimation for first few tasks
            if task.get('task_index', 0) < 5:
                logger.info(f"[ESTIMATION DEBUG] Task {task.get('task_index', 0)}: estimated {est_tokens} tokens")
            
            # Track task processing
            self.stats['tasks_processed'] += 1
            
            # TPM bucket for token limiting
            await self.tpm_bucket.wait_and_acquire(est_tokens)
            
            # Calculate dynamic timeout BEFORE rate limiting for progressive learning
            timeout = self.latency_tracker.get_timeout(
                est_tokens,
                min_timeout=self.config.minimum_timeout_seconds,
                max_timeout=self.config.maximum_timeout_seconds)
            
            # Unified rate limiting and semaphore
            async with self.semaphore:
                async with self.rate_limiter:
                    
                    # Make API call with Pydantic validation and retries
                    response = await asyncio.wait_for(
                        self.client.chat.completions.create(
                            model=self.model,
                            response_model=List[IdeaResponse],
                            messages=[{"role": "user", "content": prompt}],
                            temperature=self.config.temperature,
                            max_tokens=self.config.max_tokens,
                            seed=self.model_config.seed,
                            max_retries=3  # Instructor will retry on ValidationError
                        ),
                        timeout=timeout
                    )
                    
                    # Record latency
                    latency = time.perf_counter() - task_start
                    self.latency_tracker.add(latency)
                    
                    # Track actual token usage for learning and reconciliation (matching qualityFilter)
                    if hasattr(response, '_raw_response'):
                        usage = response._raw_response.usage
                        if usage:
                            actual_total_tokens = usage.total_tokens
                            actual_output_tokens = usage.completion_tokens
                            actual_input_tokens = usage.prompt_tokens
                            
                            # Update token histories for estimation learning (following qualityFilter pattern)
                            if len(self.input_token_history) < 3:
                                self.input_token_history.append(actual_input_tokens)
                            if len(self.output_token_history) < 5:
                                self.output_token_history.append(actual_output_tokens)
                            
                            # Track actual total tokens for rolling average
                            self.actual_total_tokens.append(actual_total_tokens)
                            
                            # Track estimation accuracy
                            estimation_error = abs(actual_total_tokens - est_tokens)
                            self.estimation_errors.append(estimation_error)
                            
                            # Reconcile token difference with bucket
                            delta = actual_total_tokens - est_tokens
                            if task.get('task_index', 0) < 5:
                                print(f"[DEBUG] Task {task.get('task_index', 0)}: Reconciling {delta} tokens (actual: {actual_total_tokens}, estimated: {est_tokens})")
                            await self.tpm_bucket.reconcile(delta)
                    
                    # Process response - array of IdeaResponse objects
                    ideas = []
                    for i, idea_response in enumerate(response):
                        # Handle missing or empty ideas with validation
                        idea_text = idea_response.idea.strip() if idea_response.idea else ""
                        if idea_text and idea_text not in ["", "NA", "N/A"]:
                            # Use the idea_id from response if available, otherwise generate
                            response_idea_id = getattr(idea_response, 'idea_id', None) or str(i+1)
                            ideas.append(models.IdeasExtractedSubmodel(
                                idea_id=f"{task['respondent_id']}_{response_idea_id}",
                                idea=idea_text
                            ))
                    
                    # Extract result
                    if ideas:
                        self.stats['tasks_successful'] += 1
                        return models.IdeasExtractedModel(
                            respondent_id=task['respondent_id'],
                            response=task['response'],
                            quality_filter=task.get('quality_filter', True),
                            quality_filter_code=task.get('quality_filter_code', 0),
                            response_ideas=ideas,
                            idea_count=len(ideas)
                        )
                    else:
                        return self.create_fallback_response(task)
                    
        except asyncio.TimeoutError:
            self.stats['timeouts'] += 1
            logger.warning(f"Task {task['respondent_id']} timed out")
            raise  # Let tenacity retry
            
        except RateLimitError:
            self.stats['rate_limits'] += 1
            logger.warning(f"Task {task['respondent_id']} hit rate limit")
            raise  # Let tenacity retry
            
        except Exception as e:
            logger.error(f"Task {task['respondent_id']} failed: {type(e).__name__}: {e}")
            raise  # Let tenacity retry
    
    def create_fallback_response(self, task: Dict) -> models.IdeasExtractedModel:
        """Create fallback response for failed tasks"""
        return models.IdeasExtractedModel(
            respondent_id=task['respondent_id'],
            response=task['response'],
            quality_filter=task.get('quality_filter', True),
            quality_filter_code=task.get('quality_filter_code', 0),
            response_ideas=[
                models.IdeasExtractedSubmodel(
                    idea_id=f"{task['respondent_id']}_1",
                    idea="PROCESSING_ERROR"
                )
            ],
            idea_count=1
        )

    async def worker(self, queue: asyncio.Queue, results: List):
        """Worker coroutine that processes tasks from queue"""
        while True:
            task = None
            try:
                task = await queue.get()
                if task is None:  # Sentinel to stop worker
                    break
                
                task_index, task_data = task
                result = await self.process_task(task_data)
                results[task_index] = result
                
            except Exception as e:
                # After all retries failed
                logger.error(f"Task failed after retries: {e}")
                self.stats['tasks_failed'] += 1
                if task is not None:
                    task_index, task_data = task
                    results[task_index] = self.create_fallback_response(task_data)
            finally:
                # Always mark task as done to prevent hanging
                if task is not None:
                    queue.task_done()

    async def process_all_tasks_async(self, tasks: List[Dict]) -> List[models.IdeasExtractedModel]:
        """Process all tasks using queue + workers pattern with bootstrap measurement"""
        if not tasks:
            return []
        
        # Setup
        limits = get_openai_rate_limits(self.model)

        # Bootstrap measurement with probe calls (following qualityFilter pattern)
        sample_tasks = tasks[:min(3, len(tasks))]
        if len(sample_tasks) < 3:
            # Duplicate tasks if we have fewer than 3
            sample_tasks = sample_tasks * 3
            sample_tasks = sample_tasks[:3]
        
        self.verbose_reporter.step_start("Idea Extraction", emoji="💡")
        
        # Extract subject once at the beginning to cache it
        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line("Extracting canonical subject/actor from survey question...")
        
        # This will cache the subject for all subsequent calls
        await self._extract_subject(self.var_lab)
        
        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line("Running bootstrap measurement (3 probe calls)...")
        
        start_time = time.time()
        task_cycle = itertools.cycle(sample_tasks)
        
        async def probe_with_different_tasks():
            return await self.probe_call_no_structured(next(task_cycle))
        
        avg_latency_s, avg_tokens = await bootstrap_measure_async(probe_with_different_tasks, n_probes=3)
        
        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line(f"Probe time: {time.time() - start_time:.3f}s")
            self.verbose_reporter.stat_line(f"Bootstrap results: {avg_latency_s:.3f}s avg latency, {avg_tokens:.0f} avg tokens")
        
        # Initialize latency tracker with bootstrap measurements
        for i in range(3):  # Add 3 samples to get started
            self.latency_tracker.add(avg_latency_s)
        
        # Update avg_tokens with bootstrap measurement
        self.avg_tokens = int(avg_tokens)
        
        # Calculate optimal concurrency using Little's Law (matching qualityFilter)
        api_limits = ApiLimits(limits.tokens_per_minute, limits.requests_per_minute)
        Little = compute_optimal_concurrency(api_limits, avg_latency_s, avg_tokens, processing_config=self.processing_config, cap=self.processing_config.concurrency_cap_permissive, min_conc=self.processing_config.concurrency_min_permissive)
        optimal = min(300, max(Little, 100))  # constrained to range 100-300

        # Initialize rate limiting components
        arrival_rate = min(
            limits.requests_per_minute * self.processing_config.rate_limit_headroom / 60,
            limits.tokens_per_minute * self.processing_config.rate_limit_headroom / avg_tokens / 60
        )

        if arrival_rate < 1:
            self.rate_limiter = AsyncLimiter(1, time_period=1/arrival_rate)
        else:
            self.rate_limiter = AsyncLimiter(int(arrival_rate), time_period=1.0)

        self.semaphore = asyncio.Semaphore(min(len(tasks), optimal))
        self.optimal_concurrency = min(len(tasks), optimal)

        print("[RATE LIMITING SETUP]")
        print(f"- Model: {self.model}")
        print(f"- RPM limit: {limits.requests_per_minute:,} ({limits.requests_per_minute * self.processing_config.rate_limit_headroom:,.0f} with headroom)")
        print(f"- TPM limit: {limits.tokens_per_minute:,} ({limits.tokens_per_minute * self.processing_config.rate_limit_headroom:,.0f} with headroom)")
        print(f"- Bootstrap measured avg_tokens: {self.avg_tokens}")

        # Show expected throughput breakdown
        rpm_throughput = limits.requests_per_minute * self.processing_config.rate_limit_headroom / 60
        tpm_throughput = limits.tokens_per_minute * self.processing_config.rate_limit_headroom / self.avg_tokens / 60
        bottleneck = "RPM" if rpm_throughput < tpm_throughput else "TPM"
        print(f"- Expected throughput: {min(rpm_throughput, tpm_throughput):.1f}/s ({bottleneck} limited)")
        print(f"- Optimal by Little's law: {Little}")
        print(f"- Constrained optimum: {optimal} (min=100, max=300)")
        
        print(f"- Processing {len(tasks):,} tasks")
        
        # Calculate number of workers
        expected_throughput = min(rpm_throughput, tpm_throughput) 
        num_workers = min(200, max(50, int(expected_throughput * avg_latency_s * 2.0)))
       
        print(f"Workers launched: (concurrent subroutines): {num_workers}")
        print(f"API calls in flight (concurrency ceiling/semaphore): {self.optimal_concurrency}")
        
        # Create queue and results list
        queue = asyncio.Queue()
        results = [None] * len(tasks)
        
        # Add tasks to queue with result indices
        for i, task in enumerate(tasks):
            task['result_index'] = i
            task['task_index'] = i
            await queue.put((i, task))
        
        # Start workers
        workers = []
        for _ in range(num_workers):
            w = asyncio.create_task(self.worker(queue, results))
            workers.append(w)
        
        # Progress monitoring with diagnostics (matching qualityFilter)
        start_time = time.time()
        last_report = start_time
        last_diagnostics = start_time
        
        while not queue.empty():
            await asyncio.sleep(1)
            now = time.time()
            
            # Regular progress report every 5s
            if now - last_report >= 5:
                completed = self.stats['tasks_processed']
                remaining = queue.qsize()
                elapsed = now - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                
                print(f"Progress: {completed}/{len(tasks)} ({completed/len(tasks)*100:.1f}%), "
                      f"Rate: {rate:.1f}/s, Queue: {remaining}")
                last_report = now
            
            # Diagnostic report every 30s (if verbose)
            if self.verbose_reporter.enabled and now - last_diagnostics >= 30:
                bucket_status = self.get_token_bucket_status()
                token_stats = self.get_token_estimation_stats()
                
                # Token bucket diagnostics
                if bucket_status['low_tokens']:
                    self.verbose_reporter.stat_line(f"⚠️ Token bucket low: {bucket_status['available_tokens']:,} tokens ({bucket_status['utilization_pct']:.1f}% utilized)")
                
                # Token estimation diagnostics
                if token_stats['status'] == 'learning' and token_stats['samples'] >= 5:
                    self.verbose_reporter.stat_line(f"Token estimation: {token_stats['avg_estimation_error']:.0f} avg error, "
                                                  f"Input: {token_stats['avg_input_tokens']:.0f} avg ({token_stats['input_samples']}/3), "
                                                  f"Output: {token_stats['avg_output_tokens']:.0f} avg ({token_stats['output_samples']}/5)")
                
                last_diagnostics = now
        
        # Wait for all tasks to complete
        await queue.join()
        
        # Stop workers
        for _ in workers:
            await queue.put(None)
        await asyncio.gather(*workers)
        
        # Final stats with diagnostics (matching qualityFilter)
        elapsed = time.time() - start_time
        print(f"\nCompleted {len(tasks)} tasks in {elapsed:.1f}s")
        print(f"- Successful: {self.stats['tasks_successful']}")
        print(f"- Failed: {self.stats['tasks_failed']}")
        print(f"- Rate limits: {self.stats['rate_limits']}")
        print(f"- Timeouts: {self.stats['timeouts']}")
        print(f"- Average: {elapsed/len(tasks):.2f}s/task")
        
        # Final diagnostic summary (if verbose)
        if self.verbose_reporter.enabled:
            token_stats = self.get_token_estimation_stats()
            bucket_status = self.get_token_bucket_status()
            
            if token_stats['status'] == 'learning':
                accuracy = max(0, 100 - (token_stats['avg_estimation_error'] / max(1, token_stats['avg_input_tokens'] + token_stats['avg_output_tokens']) * 100))
                self.verbose_reporter.stat_line(f"Token estimation accuracy: {accuracy:.1f}% (avg error: {token_stats['avg_estimation_error']:.0f} tokens)")
                self.verbose_reporter.stat_line(f"Learned averages - Input: {token_stats['avg_input_tokens']:.0f}, Output: {token_stats['avg_output_tokens']:.0f}")
                
                # Final comparison of initial vs learned token usage
                if token_stats['actual_samples'] >= 10:
                    actual_avg = token_stats['avg_actual_total_tokens']
                    initial_avg = token_stats['initial_avg_tokens']
                    difference = actual_avg - initial_avg
                    
                    # Calculate what throughput would have been with perfect estimation
                    limits = get_openai_rate_limits(self.model)
                    optimal_throughput = limits.tokens_per_minute * self.processing_config.rate_limit_headroom / actual_avg / 60
                    initial_throughput = limits.tokens_per_minute * self.processing_config.rate_limit_headroom / initial_avg / 60
                    
                    self.verbose_reporter.stat_line(f"Token usage summary: Initial {initial_avg:.0f} → Actual {actual_avg:.0f} "
                                                  f"({difference:+.0f} tokens, {difference/initial_avg*100:+.1f}%)")
                    self.verbose_reporter.stat_line(f"Throughput analysis: Expected {initial_throughput:.1f}/s → Optimal {optimal_throughput:.1f}/s with perfect estimation")
        
        return results

    def extract(self) -> List[models.IdeasExtractedModel]:
        """Main method to extract ideas from responses using bootstrap measurement and unified processing"""
        self._stats.start_timing()
        self._stats.input_count = len(self.responses)
        
        if not self.responses:
            self.verbose_reporter.stat_line("No responses to process")
            return []
        
        # Prepare tasks for async processing
        tasks = []
        for response in self.responses:
            tasks.append({
                'respondent_id': response.respondent_id,
                'response': response.response,
                'quality_filter': response.quality_filter,
                'quality_filter_code': response.quality_filter_code
            })
        
        # Process with bootstrap measurement and unified rate limiting
        if nest_asyncio:
            nest_asyncio.apply()
        self._results = asyncio.run(self.process_all_tasks_async(tasks))
        
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

        # Store statistics as instance attributes for app display
        single_idea_responses = len([r for r in self._results if r.response_ideas and len(r.response_ideas) == 1]) if multi_idea_responses > 0 else 0
        self.stats = {
            'total_responses': len(self._results),
            'total_ideas': idea_count,
            'unique_ideas': len(unique_ideas),
            'single_idea_responses': single_idea_responses,
            'multi_idea_responses': multi_idea_responses,
            'single_idea_percentage': (single_idea_responses / len(self._results) * 100) if len(self._results) > 0 and multi_idea_responses > 0 else 0,
            'multi_idea_percentage': (multi_idea_responses / len(self._results) * 100) if len(self._results) > 0 else 0
        }

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