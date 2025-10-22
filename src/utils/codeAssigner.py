import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import asyncio
import time
import logging
import itertools
from typing import Dict, List, Optional #, Union
from dataclasses import dataclass
from collections import deque
import numpy as np

from openai import RateLimitError, APIConnectionError, APITimeoutError, InternalServerError
#import tiktoken
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type
from instructor.exceptions import InstructorRetryException
from aiolimiter import AsyncLimiter

logger = logging.getLogger(__name__)

# === MODELS ========================================================================================================
from pydantic import BaseModel
import models

# === CONFIG ========================================================================================================
from config import OPENAI_API_KEY, DEFAULT_LANGUAGE, ModelConfig, CodeAssignmentConfig, DEFAULT_CODE_ASSIGNMENT_CONFIG, ProcessingConfig, DEFAULT_PROCESSING_CONFIG, get_openai_rate_limits
from prompts import CODE_ASSIGNMENT_PROMPT

# === UTILS ========================================================================================================
from .verboseReporter import VerboseReporter, ProcessingStats
from .embedder import Embedder
from .cached_resources import get_openai_client, get_tiktoken_encoding

try:
    import nest_asyncio  # for Spyder
    nest_asyncio.apply()
except ImportError:
    pass

# === RATE LIMITING HELPER CLASSES ========================================================================================================

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
        logger.debug(f"[TOKEN BUCKET] Requesting {tokens_needed} tokens")
        
        while True:
            result = await self.acquire(tokens_needed)
            if result is True:
                logger.debug(f"[TOKEN BUCKET] Acquired {tokens_needed} tokens, {self.available:.0f} remaining")
                return
            else:
                # result is wait_seconds
                logger.debug(f"[TOKEN BUCKET] Insufficient tokens, waiting {result:.1f}s")
                await asyncio.sleep(result)
    
    async def reconcile(self, delta_tokens):
        """Reconcile actual vs estimated tokens"""
        # If we overestimated (delta < 0), return tokens
        # If we underestimated (delta > 0), we already used them
        if delta_tokens < 0:
            async with self.lock:
                old_available = self.available
                self.available = min(self.tpm, self.available - delta_tokens)
                logger.debug(f"[TOKEN BUCKET] Reconciled {-delta_tokens} tokens back, {old_available:.0f} → {self.available:.0f}")
        else:
            logger.debug(f"[TOKEN BUCKET] No reconciliation needed for +{delta_tokens} tokens (underestimated)")


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
        # Assume ~100ms per 1000 tokens as baseline
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

class CodeAssignmentResponse(BaseModel):
    idea_id: str
    idea: str
    assigned_codes: List[str]
    assignment_confidence: float
    assignment_rationale: str
    assigned_themes: Optional[List[str]] = None


class EmbeddingLoader:
    """Utility class for loading and managing embeddings from cache"""
    
    @staticmethod
    def load_idea_embeddings_from_cache(cache_manager, filename):
        """Load idea embeddings from cache step 'embeddings'"""
        embeddings_results = cache_manager.load_from_cache(
            filename, "embeddings", models.EmbeddingsModel
        )
        
        if not embeddings_results:
            return []
        
        # Extract all ideas with their embeddings
        ideas_with_embeddings = []
        for result in embeddings_results:
            if result.response_ideas:
                for idea in result.response_ideas:
                    if idea.idea_embedding is not None:
                        ideas_with_embeddings.append({
                            'idea': idea.idea,
                            'idea_id': idea.idea_id,
                            'embedding': idea.idea_embedding,
                            'respondent_id': result.respondent_id
                        })
        
        return ideas_with_embeddings
    
    @staticmethod
    def format_codes_for_embedding(enriched_codebook):
        """Format enriched codebook entries for embedding generation"""
        # Use definitions only to match idea embedding format (just text)
        return [code.definition for code in enriched_codebook]


class CodeAssigner:
    """
    Simplified code assignment with direct LLM processing.
    LLM sees all codes in codebook instead of similarity-filtered subset.
    """
    
    def __init__(
        self,
        cluster_models: List[models.ClusterModel],
        codebook: List[models.Codebook],
        var_lab: str,
        code_to_theme_mapping: Optional[Dict[str, str]] = None,
        cached_idea_embeddings: Optional[List[Dict]] = None,
        config: Optional[CodeAssignmentConfig] = None,
        model_config: Optional[ModelConfig] = None,
        processing_config: Optional[ProcessingConfig] = None,
        verbose: bool = False,
        prompt_printer = None):

        self.cluster_models = cluster_models
        self.codebook = codebook
        self.var_lab = var_lab
        self.config = config or DEFAULT_CODE_ASSIGNMENT_CONFIG
        self.model_config = model_config or ModelConfig()
        self.processing_config = processing_config or DEFAULT_PROCESSING_CONFIG
        self.model = self.model_config.get_model_for_stage('code_assignment')
        self.language = DEFAULT_LANGUAGE
        self._results: List[models.CodeAssignedModel] = []
        self.verbose_reporter = VerboseReporter(verbose, capture_logging=True)
        self.prompt_printer = prompt_printer
        self._captured_prompt = False

        # Cache for idea embeddings if provided (for compatibility)
        self._cached_idea_embeddings = cached_idea_embeddings

        # Code embedding cache and embedder
        self._code_embeddings = None
        self.embedder = Embedder(model_config=self.model_config)

        # Theme mapping for code-to-theme assignments
        self.code_to_theme_mapping = code_to_theme_mapping or {}

        # Initialize tokenizer for token counting (cached)
        self.encoding = get_tiktoken_encoding(self.model)

        # Instructor-patched async OpenAI client for structured output (cached)
        self.client = get_openai_client(OPENAI_API_KEY)

        # Rate limiting setup
        limits = get_openai_rate_limits(self.model)

        # Token bucket for TPM limiting
        self.tpm_bucket = TokenBucket(limits.tokens_per_minute * self.processing_config.rate_limit_headroom)
        
        # Progressive token estimation (following qualityFilter.py pattern)
        self.input_token_history = deque(maxlen=3)  # First 3 input token counts
        self.output_token_history = deque(maxlen=5)  # First 5 output token counts
        self.estimation_errors = deque(maxlen=50)  # Track accuracy
        self.first_prompt_tokens = None  # Cache first prompt calculation
        
        # Rolling average of actual total tokens for comparison
        self.actual_total_tokens = deque(maxlen=50)  # Track actual total usage
        
        # Latency tracking
        self.latency_tracker = LatencyTracker(processing_config=self.processing_config)
        
        # Calculate initial average tokens estimate for bootstrapping
        self.avg_tokens = self._calculate_avg_tokens()
        
        # Rate limiting components (will be initialized after bootstrap)
        self.rate_limiter = None
        self.semaphore = None
        self.optimal_concurrency = None
        
        # Stats
        self._stats = ProcessingStats()
        self.stats = {
            'tasks_processed': 0,
            'tasks_successful': 0,
            'tasks_failed': 0,
            'retries': 0,
            'rate_limits': 0,
            'timeouts': 0
        }
        
        self.verbose_reporter.stat_line(f"Model: {self.model}")
        self.verbose_reporter.stat_line(f"API Limits: {limits.requests_per_minute} RPM, {limits.tokens_per_minute:,} TPM")

    def _calculate_avg_tokens(self) -> int:
        """Calculate average token count for code assignment requests"""
        if not self.cluster_models:
            return 1500  # Default estimate
        
        sample_size = min(10, len(self.cluster_models))
        total_tokens = 0
        sample_count = 0
        
        for i in range(sample_size):
            model = self.cluster_models[i]
            if hasattr(model, 'response_ideas') and model.response_ideas:
                for idea in model.response_ideas:
                    if hasattr(idea, 'idea') and idea.idea:
                        # Create sample prompt
                        prompt = self._create_prompt_for_estimation(idea.idea_id, idea.idea)
                        total_tokens += len(self.encoding.encode(prompt))
                        sample_count += 1
                        break  # Only sample first idea per model
        
        if sample_count == 0:
            return 1500  # Fallback
        
        avg_input = total_tokens / sample_count
        # Assume 15% output ratio initially
        return int(avg_input * 1.15)
    
    def _create_prompt_for_estimation(self, idea_id: str, idea_text: str) -> str:
        """Create a sample prompt for token estimation (simplified version)"""
        # Use first few codes for estimation
        sample_codes = self.codebook[:min(5, len(self.codebook))]
        candidate_codes_text = "\n".join([
            f"Code label: {code.code}\nCode description: {code.definition}\n" 
            for code in sample_codes
        ])
        
        return CODE_ASSIGNMENT_PROMPT.format(
            language=self.language,
            var_lab=self.var_lab,
            idea_id=idea_id,
            idea_text=idea_text,
            candidate_codes=candidate_codes_text
        )
    
    def estimate_tokens(self, prompt: str) -> int:
        """Estimate total tokens using adaptive strategy (following qualityFilter.py)"""
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
        
        # Output estimation: 15% of input, then average of first 5 responses
        if len(self.output_token_history) < 5:
            # Use 15% of input as estimate
            estimated_output = int(estimated_input * 0.15)
        else:
            # Use average of first 5 actual outputs
            avg_output = sum(self.output_token_history) / len(self.output_token_history)
            estimated_output = int(avg_output)
        
        # Ensure we don't exceed max_tokens
        estimated_output = min(self.config.max_tokens, estimated_output)
        
        total_estimate = estimated_input + estimated_output
        
        return total_estimate

    async def _get_code_embeddings(self) -> np.ndarray:
        """Generate or retrieve cached embeddings for all codes in codebook"""
        #print(f"[DEBUG] _get_code_embeddings called")
        
        if self._code_embeddings is not None:
            #print(f"[DEBUG] Using cached code embeddings")
            return self._code_embeddings
        
        #print(f"[DEBUG] Creating embeddings for {len(self.codebook)} codes")
        self.verbose_reporter.stat_line(f"Generating embeddings for {len(self.codebook)} codes...")
        
        try:
            # Create temporary models for embedding generation
            temp_models = []
            #print(f"[DEBUG] Creating {len(self.codebook)} temp models...")
            
            for i, code in enumerate(self.codebook):
                # Create a simple model with the code definition as response text
                temp_model = models.EmbeddingsModel(
                    respondent_id=f"code_{i}",
                    response=code.definition,  # Use definition for embedding
                    response_ideas=[models.EmbeddingsSubmodel(
                        idea_id="1",
                        idea=code.definition
                    )]
                )
                temp_models.append(temp_model)
            
            #print(f"[DEBUG] Created {len(temp_models)} temp models, calling embedder...")
            
            # Generate embeddings using async method directly
            embedded_codes = await self.embedder._process_embeddings_with_id_tracking(temp_models)
            #print(f"[DEBUG] Embedder returned {len(embedded_codes)} embedded codes")
        
            # Extract embeddings array
            embeddings = []
            #print(f"[DEBUG] Extracting embeddings from {len(embedded_codes)} models...")
            
            for i, model in enumerate(embedded_codes):
                if hasattr(model, 'response_ideas') and model.response_ideas and len(model.response_ideas) > 0:
                    embedding = model.response_ideas[0].idea_embedding
                    if embedding is not None:
                        embeddings.append(embedding)
                        #print(f"[DEBUG] Model {i}: Got embedding of shape {embedding.shape}")
                    else:
                        embeddings.append(np.zeros(1536))  # text-embedding-3-large dimension
                        #print(f"[DEBUG] Model {i}: No embedding, using zeros")
                else:
                    embeddings.append(np.zeros(1536))
                    #print(f"[DEBUG] Model {i}: No response_ideas, using zeros")
            
            #print(f"[DEBUG] Extracted {len(embeddings)} embeddings, creating array...")
            self._code_embeddings = np.array(embeddings)
            #print(f"[DEBUG] Code embeddings array shape: {self._code_embeddings.shape}")
            return self._code_embeddings
            
        except Exception as e:
            #print(f"[DEBUG] ERROR in _get_code_embeddings: {type(e).__name__}: {e}")
            import traceback
            #print(f"[DEBUG] Embedding traceback: {traceback.format_exc()}")
            # Return zero embeddings as fallback
            self._code_embeddings = np.zeros((len(self.codebook), 1536))
            return self._code_embeddings

    def _find_similar_codes(self, idea_embedding: np.ndarray, top_k: int = 10) -> List[models.Codebook]:
        """Find the top_k most similar codes to an idea based on embedding similarity"""
        if self._code_embeddings is None:
            raise ValueError("Code embeddings not initialized. Call _get_code_embeddings first.")
        
        # Calculate cosine similarity
        similarities = cosine_similarity([idea_embedding], self._code_embeddings)[0]
        
        # Get top_k most similar indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return corresponding codes
        return [self.codebook[i] for i in top_indices]


    def _assign_themes_to_codes(self, assigned_codes: List[str]) -> List[str]:
        """Map assigned codes to their themes using cached mapping"""
        themes = []
        for code in assigned_codes:
            theme = self.code_to_theme_mapping.get(code)
            if theme and theme not in themes:
                themes.append(theme)
        return themes

    def _extract_all_ideas(self) -> List[tuple]:
        """Extract all individual ideas for processing with embeddings"""
        # Use cached embeddings if provided (for compatibility)
        if self._cached_idea_embeddings:
            all_ideas = []
            for cached_idea in self._cached_idea_embeddings:
                all_ideas.append((
                    cached_idea['respondent_id'],
                    cached_idea['idea_id'],
                    cached_idea['idea'],
                    cached_idea['embedding']
                ))
            self.verbose_reporter.stat_line(f"Using {len(all_ideas)} cached ideas with embeddings")
            return all_ideas
        
        # Otherwise extract from cluster models
        all_ideas = []
        
        for model in self.cluster_models:
            if hasattr(model, 'response_ideas') and model.response_ideas:
                for idea_submodel in model.response_ideas:
                    if hasattr(idea_submodel, 'idea_embedding') and idea_submodel.idea_embedding is not None:
                        all_ideas.append((
                            model.respondent_id,
                            idea_submodel.idea_id,
                            idea_submodel.idea,
                            idea_submodel.idea_embedding
                        ))
                    else:
                        self.verbose_reporter.stat_line(f"Warning: No embedding for idea {idea_submodel.idea_id}")
            else:
                self.verbose_reporter.stat_line(f"Warning: No response_ideas found for respondent {model.respondent_id}")
        
        return all_ideas

    def _create_prompt(self, idea_id: str, idea_text: str, idea_embedding: np.ndarray) -> str:
        """Create prompt for a single idea with most similar codes"""
        # Find most similar codes
        similar_codes = self._find_similar_codes(idea_embedding, top_k=self.config.top_k_similar_codes)
        
        # Format candidate codes for prompt
        candidate_codes_text = "\n".join([
            f"Code label: {code.code}\nCode description: {code.definition}\n" 
            for code in similar_codes
        ])
        
        # Create prompt
        prompt = CODE_ASSIGNMENT_PROMPT.format(
            language=self.language,
            var_lab=self.var_lab,
            idea_id=idea_id,
            idea_text=idea_text,
            candidate_codes=candidate_codes_text
        )
        
        return prompt
    
    async def probe_call_no_structured(self, task_dict):
        """Probe call without structured output for bootstrap measurement"""
        idea_data = task_dict['idea_data']
        respondent_id, idea_id, idea_text, idea_embedding = idea_data
        
        prompt = self._create_prompt(idea_id, idea_text, idea_embedding)
        
        # For probes: avoid response_model so we can read .usage
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            seed=self.model_config.seed
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
    async def process_task(self, task: Dict) -> CodeAssignmentResponse:
        """Process a single code assignment task"""
        task_start = time.perf_counter()
        
        try:
            idea_data = task['idea_data']
            respondent_id, idea_id, idea_text, idea_embedding = idea_data
            
            # Build prompt
            prompt = self._create_prompt(idea_id, idea_text, idea_embedding)
            
            # Estimate tokens
            est_tokens = self.estimate_tokens(prompt)
            
            # Log estimation for first few tasks
            if task.get('task_index', 0) < 5:
                logger.info(f"[ESTIMATION DEBUG] Task {task.get('task_index', 0)}: estimated {est_tokens} tokens")
            
            # Capture prompt for first task
            if self.prompt_printer and task.get('task_index', 0) == 0:
                self.prompt_printer.capture_prompt(
                    step_name="code_assignment",
                    utility_name="CodeAssigner",
                    prompt_content=prompt,
                    prompt_type="code_assignment",
                    metadata={
                        "model": self.model,
                        "var_lab": self.var_lab,
                        "language": self.language,
                        "estimated_tokens": est_tokens,
                        "idea_id": idea_id
                    }
                )
           
            # TPM bucket for token limiting
            await self.tpm_bucket.wait_and_acquire(est_tokens)
            
            # Calculate dynamic timeout BEFORE rate limiting for progressive learning
            timeout = self.latency_tracker.get_timeout(est_tokens)
            
            # Unified rate limiting and semaphore
            async with self.semaphore:
                async with self.rate_limiter:
                    
                    # Make API call
                    response = await asyncio.wait_for(
                        self.client.chat.completions.create(
                            model=self.model,
                            response_model=CodeAssignmentResponse,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=self.config.temperature,
                            max_tokens=self.config.max_tokens,
                            seed=self.model_config.seed
                        ),
                        timeout=timeout
                    )
                    
                    # Record latency
                    latency = time.perf_counter() - task_start
                    self.latency_tracker.add(latency)
                    
                    # Track actual token usage for learning and reconciliation
                    if hasattr(response, '_raw_response'):
                        usage = response._raw_response.usage
                        if usage:
                            actual_total_tokens = usage.total_tokens
                            actual_output_tokens = usage.completion_tokens
                            
                            # Update output token history for estimation learning
                            if len(self.output_token_history) < 5:
                                self.output_token_history.append(actual_output_tokens)
                            
                            # Track actual total tokens for rolling average
                            self.actual_total_tokens.append(actual_total_tokens)
                            
                            # Track estimation accuracy
                            estimation_error = abs(actual_total_tokens - est_tokens)
                            self.estimation_errors.append(estimation_error)
                            
                            # Reconcile token difference with bucket
                            delta = actual_total_tokens - est_tokens
                            # if task.get('task_index', 0) < 5:
                            #     print(f"[DEBUG] Task {task.get('task_index', 0)}: Reconciling {delta} tokens (actual: {actual_total_tokens}, estimated: {est_tokens})")
                            await self.tpm_bucket.reconcile(delta)
                    
                    # Add theme assignments
                    assigned_themes = self._assign_themes_to_codes(response.assigned_codes)
                    response.assigned_themes = assigned_themes
                    
                    self.stats['tasks_successful'] += 1
                    return response
                    
        except asyncio.TimeoutError:
            self.stats['timeouts'] += 1
            logger.warning(f"Task {task['task_id']} timed out")
            raise  # Let tenacity retry
            
        except RateLimitError:
            self.stats['rate_limits'] += 1
            logger.warning(f"Task {task['task_id']} hit rate limit")
            raise  # Let tenacity retry
            
        except Exception as e:
            logger.error(f"Task {task['task_id']} failed: {type(e).__name__}: {e}")
            raise  # Let tenacity retry
    
    def create_fallback_response(self, task: Dict) -> CodeAssignmentResponse:
        """Create fallback response for failed tasks"""
        idea_data = task['idea_data']
        respondent_id, idea_id, idea_text, idea_embedding = idea_data
        
        # Return fallback response (first available code)
        fallback_code = self.codebook[0].code if self.codebook else "Unknown"
        fallback_themes = self._assign_themes_to_codes([fallback_code]) if fallback_code != "Unknown" else []
        
        return CodeAssignmentResponse(
            idea_id=idea_id,
            idea=idea_text,
            assigned_codes=[fallback_code],
            assigned_themes=fallback_themes,
            assignment_confidence=0.1,
            assignment_rationale=f"Processing failed, using fallback code"
        )
    
    async def worker(self, queue: asyncio.Queue, results: List):
        """Worker coroutine that processes tasks from queue"""
        worker_id = id(asyncio.current_task())
        #print(f"[DEBUG] Worker {worker_id} started")
        task_count = 0
        
        while True:
            try:
                task = await queue.get()
                if task is None:  # Sentinel
                    #print(f"[DEBUG] Worker {worker_id} received sentinel, processed {task_count} tasks")
                    break
                
                task_count += 1
                #print(f"[DEBUG] Worker {worker_id} processing task {task_count}: {task.get('task_id', 'unknown')}")
                
                try:
                    result = await self.process_task(task)
                    results[task['result_index']] = result
                    #print(f"[DEBUG] Worker {worker_id} task {task_count} SUCCESS")
                except Exception as e:
                    # After all retries failed
                    #print(f"[DEBUG] Worker {worker_id} task {task_count} FAILED: {type(e).__name__}: {e}")
                    logger.error(f"Task {task['task_id']} failed after retries: {type(e).__name__}: {e}")
                    import traceback
                    logger.error(f"Full traceback: {traceback.format_exc()}")
                    self.stats['tasks_failed'] += 1
                    results[task['result_index']] = self.create_fallback_response(task)
                finally:
                    self.stats['tasks_processed'] += 1
                    queue.task_done()
                    
            except Exception as e:
                logger.error(f"Worker error: {type(e).__name__}: {e}")
                import traceback
                logger.error(f"Worker traceback: {traceback.format_exc()}")
                break

    async def _process_single_idea(self, idea_data: tuple) -> CodeAssignmentResponse:
        """Deprecated - use process_task instead"""
        # This method is now deprecated - it's kept for compatibility
        # All processing now goes through the worker queue pattern
        pass

    def _merge_results_into_models(self, assignment_results: List[CodeAssignmentResponse]) -> List[models.CodeAssignedModel]:
        """Merge assignment results back into model structure"""
        
        # If using cached embeddings, create simple models from assignments
        if self._cached_idea_embeddings and not self.cluster_models:
            coded_models = []
            
            # Group assignments by respondent_id
            respondent_assignments = {}
            for result in assignment_results:
                # Extract respondent_id from the cached data
                for cached_idea in self._cached_idea_embeddings:
                    if cached_idea['idea_id'] == result.idea_id:
                        resp_id = cached_idea['respondent_id']
                        if resp_id not in respondent_assignments:
                            respondent_assignments[resp_id] = []
                        respondent_assignments[resp_id].append(result)
                        break
            
            # Create CodeAssignedModel for each respondent
            for resp_id, assignments in respondent_assignments.items():
                assigned_ideas = []
                for assignment in assignments:
                    assigned_idea = models.AssignedIdeaSubmodel(
                        idea_id=assignment.idea_id,
                        idea=assignment.idea,
                        assigned_codes=assignment.assigned_codes,
                        assignment_confidence=assignment.assignment_confidence,
                        assignment_rationale=assignment.assignment_rationale,
                        assigned_themes=assignment.assigned_themes
                    )
                    assigned_ideas.append(assigned_idea)
                
                coded_model = models.CodeAssignedModel(
                    respondent_id=resp_id,
                    response='',  # We don't have the full response text
                    response_ideas=assigned_ideas
                )
                coded_models.append(coded_model)
            
            return coded_models
        
        # Original logic for cluster models
        # Create lookup for assignments by idea_id
        assignments_lookup = {result.idea_id: result for result in assignment_results}
        
        coded_models = []
        
        for original_model in self.cluster_models:
            # Convert to CodeAssignedModel
            coded_model = original_model.to_model(models.CodeAssignedModel)
            
            # Update response_ideas with assignments
            if coded_model.response_ideas:
                updated_ideas = []
                for idea_submodel in coded_model.response_ideas:
                    # Convert to AssignedIdeaSubmodel
                    assigned_idea = models.AssignedIdeaSubmodel(
                        idea_id=idea_submodel.idea_id,
                        idea=idea_submodel.idea,
                        initial_cluster=getattr(idea_submodel, 'initial_cluster', None),
                        expanded_cluster=getattr(idea_submodel, 'expanded_cluster', None),
                        idea_embedding=getattr(idea_submodel, 'idea_embedding', None)
                    )
                    
                    # Add assignment data if available
                    if idea_submodel.idea_id in assignments_lookup:
                        assignment = assignments_lookup[idea_submodel.idea_id]
                        assigned_idea.assigned_codes = assignment.assigned_codes
                        assigned_idea.assigned_themes = assignment.assigned_themes
                        assigned_idea.assignment_confidence = assignment.assignment_confidence
                        assigned_idea.assignment_rationale = assignment.assignment_rationale
                    else:
                        # Fallback if no assignment found
                        assigned_idea.assigned_codes = ["Unassigned"]
                        assigned_idea.assigned_themes = []
                        assigned_idea.assignment_confidence = 0.0
                        assigned_idea.assignment_rationale = "No assignment found"
                    
                    updated_ideas.append(assigned_idea)
                
                coded_model.response_ideas = updated_ideas
            
            coded_models.append(coded_model)
        
        return coded_models

    async def process_all_tasks_async(self, tasks: List[Dict]) -> List[CodeAssignmentResponse]:
        """Process all tasks using queue + workers pattern with bootstrap measurement"""
        if not tasks:
            return []
        
        try:
            #print(f"[DEBUG] Starting process_all_tasks_async with {len(tasks)} tasks")
            
            # Setup
            limits = get_openai_rate_limits(self.model)

            # Initialize code embeddings first
            #print(f"[DEBUG] Initializing code embeddings...")
            await self._get_code_embeddings()
            #print(f"[DEBUG] Code embeddings initialized successfully")
            
            # Bootstrap measurement with probe calls (following qualityFilter.py pattern)
            sample_tasks = tasks[:min(3, len(tasks))]
            if len(sample_tasks) < 3:
                # Duplicate tasks if we have fewer than 3
                sample_tasks = sample_tasks * 3
                sample_tasks = sample_tasks[:3]
            
            self.verbose_reporter.step_start("Code Assignment")
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
        
            # Calculate optimal concurrency using Little's Law
            api_limits = ApiLimits(limits.tokens_per_minute, limits.requests_per_minute)
            Little = compute_optimal_concurrency(api_limits, avg_latency_s, avg_tokens, processing_config=self.processing_config, cap=self.processing_config.concurrency_cap_permissive, min_conc=self.processing_config.concurrency_min_permissive)
            optimal = min(300, max(Little, 100)) # constrained to range 100-300

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
            
            print(f"- Workers launched: (concurrent subroutines): {num_workers}")
            print(f"- API calls in flight (concurrency ceiling/semaphore): {self.optimal_concurrency}")
            
            # Create queue and results list
            queue = asyncio.Queue()
            results = [None] * len(tasks)
            
            # Add tasks to queue with result indices
            for i, task in enumerate(tasks):
                task['result_index'] = i
                task['task_index'] = i
                task['task_id'] = task['idea_data'][1]  # idea_id
                await queue.put(task)
        
            # Start workers
            workers = []
            #print(f"[DEBUG] Starting {num_workers} workers...")
            for i in range(num_workers):
                w = asyncio.create_task(self.worker(queue, results))
                workers.append(w)
            #print(f"[DEBUG] All {len(workers)} workers started")
        
            # Progress monitoring
            start_time = time.time()
            last_report = start_time
        
            #print(f"[DEBUG] Starting progress monitoring, queue size: {queue.qsize()}")
        
            # Monitor progress until all tasks are processed
            while self.stats['tasks_processed'] < len(tasks):
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
            
                # Check if queue is empty but not all tasks processed (potential deadlock)
                if queue.empty() and self.stats['tasks_processed'] < len(tasks):
                    #print(f"[DEBUG] Queue empty but only {self.stats['tasks_processed']}/{len(tasks)} processed")
                    break
        
            #print("[DEBUG] Progress monitoring complete, waiting for queue.join()")
        
            # Wait for all tasks to complete
            await queue.join()
        
            #print("[DEBUG] Queue.join() complete")
        
            # Stop workers
            for _ in workers:
                await queue.put(None)
            await asyncio.gather(*workers)
        
            # Final stats
            elapsed = time.time() - start_time
            print(f"\nCompleted {len(tasks)} tasks in {elapsed:.1f}s")
            print(f"- Successful: {self.stats['tasks_successful']}")
            print(f"- Failed: {self.stats['tasks_failed']}")
            print(f"- Rate limits: {self.stats['rate_limits']}")
            print(f"- Timeouts: {self.stats['timeouts']}")
            print(f"- Average: {elapsed/len(tasks):.2f}s/task")
        
            return results
        
        except Exception as e:
            #print(f"[DEBUG] CRITICAL ERROR in process_all_tasks_async: {type(e).__name__}: {e}")
            import traceback
            #print(f"[DEBUG] Full traceback: {traceback.format_exc()}")
            # Return fallback responses for all tasks
            fallback_results = []
            for task in tasks:
                fallback_results.append(self.create_fallback_response(task))
            return fallback_results

    def _prepare_individual_tasks(self, all_ideas: List[tuple]) -> List[Dict]:
        """Prepare individual tasks for processing"""
        tasks = []
        for i, idea_data in enumerate(all_ideas):
            tasks.append({
                'idea_data': idea_data
            })
        return tasks
    
    async def assign_codes(self) -> List[models.CodeAssignedModel]:
        """Main method to assign codes using standardized processing patterns"""
        self._stats.start_timing()
        
        # Extract all ideas
        all_ideas = self._extract_all_ideas()
        total_ideas = len(all_ideas)
        
        if total_ideas == 0:
            self.verbose_reporter.stat_line("No ideas found for code assignment")
            return []
        
        limits = get_openai_rate_limits(self.model)
        self.verbose_reporter.stat_line(f"Model: {self.model} (Limits: {limits.requests_per_minute} RPM, {limits.tokens_per_minute:,} TPM)")
        self.verbose_reporter.stat_line(f"Processing {total_ideas} ideas with {len(self.codebook)} available codes")
        
        # Prepare tasks
        tasks = self._prepare_individual_tasks(all_ideas)
        
        # Process with queue + workers pattern
        if nest_asyncio:
            nest_asyncio.apply()
        all_results = await self.process_all_tasks_async(tasks)
        
        # Merge results back into model structure
        self._results = self._merge_results_into_models(all_results)
        
        # Report summary
        if all_results:
            valid_results = [r for r in all_results if r is not None]
            if valid_results:
                avg_confidence = np.mean([r.assignment_confidence for r in valid_results])
                high_confidence = sum(1 for r in valid_results if r.assignment_confidence >= 0.7)
                low_confidence = sum(1 for r in valid_results if r.assignment_confidence < 0.5)
                
                self.verbose_reporter.summary("CODE ASSIGNMENT COMPLETED", {
                    "Total ideas processed": len(valid_results),
                    "Average confidence": f"{avg_confidence:.2f}",
                    "High confidence (≥0.7)": high_confidence,
                    "Low confidence (<0.5)": low_confidence
                })
        
        return self._results

    def assign(self) -> List[models.CodeAssignedModel]:
        """Synchronous wrapper for assign_codes"""
        if nest_asyncio:
            nest_asyncio.apply()
        
        return asyncio.run(self.assign_codes())
