import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import asyncio
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
#import json
from collections import deque
import itertools
import logging
from dataclasses import dataclass
import difflib

from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError, InternalServerError
from pydantic import BaseModel, ConfigDict, RootModel
import tiktoken
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type #wait_exponential
from pydantic import ValidationError
from aiolimiter import AsyncLimiter

import umap
from hdbscan import HDBSCAN
from sklearn.preprocessing import normalize 
from sklearn.metrics.pairwise import cosine_similarity

# === CONFIG & MODELS ========================================================================================================
from models import ClusterModel  
from config import OPENAI_API_KEY, DEFAULT_LANGUAGE, ModelConfig, DEFAULT_CODEDESIGNER_CONFIG, ProcessingConfig, DEFAULT_PROCESSING_CONFIG, get_openai_rate_limits
from prompts import CLUSTER_SUMMARY_PROMPT, CODING_DECISION_PROMPT, CODE_CREATION_PROMPT,CODING_MODIFICATION_PROMPT, VALIDATION_PROMPT
from .verboseReporter import VerboseReporter

try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

client = OpenAI()

logger = logging.getLogger(__name__)

EXTRA_VERBOSE = False
if EXTRA_VERBOSE:  
    logging.basicConfig(level=logging.INFO)
else:
    logging.basicConfig(level=logging.CRITICAL)    
    

import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*n_jobs value 1 overridden to 1 by setting random_state.*",
    category=UserWarning,
    module=r"umap\.umap_" )
               
# ============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUTS
# ============================================================================

"""Prompt 1 : Theme Extraction"""
class ClusterThemeItem(BaseModel):
    theme_id: int 
    theme_label: str 
    theme_clarification: str  

class ClusterSummaryItem(BaseModel):
    analysis: str  
    extracted_themes: List[ClusterThemeItem] 

class ClusterSummaryOutput(RootModel[Dict[str, ClusterSummaryItem]]):
    root: Dict[str, ClusterSummaryItem]

"""Prompt 2 : Coding Decision"""
class MatchedCandidate(BaseModel):
    code: str
    definition: str
    model_config = ConfigDict(arbitrary_types_allowed=True)

class CodingDecision(BaseModel):
    theme_number: int
    theme_name: str 
    matched_candidates: List[MatchedCandidate]
    decision: str  # use | modify | create
    source_code: Optional[str] = None   
    justification: str
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

class CodingDecisionOutput(BaseModel):
    coding_decision: CodingDecision
    model_config = ConfigDict(arbitrary_types_allowed=True)

"""Prompt 3 : Code Generation"""
class GeneratedCode(BaseModel):
    theme_number: int
    theme_name: str
    source_code: Optional[str] = None
    code_label: str
    code_definition: str
    model_config = ConfigDict(arbitrary_types_allowed=True)

class CodeGenerationOutput(BaseModel):
    generated_code: GeneratedCode
    model_config = ConfigDict(arbitrary_types_allowed=True)

"""Prompt 4 : Validation of code label and description"""
class ValidatedCode(BaseModel):
    code: str
    definition: str
    model_config = ConfigDict(arbitrary_types_allowed=True)

class OriginalRecommendation(BaseModel):
    code: str
    definition: str
    model_config = ConfigDict(arbitrary_types_allowed=True)

class CodeValidation(BaseModel):
    theme_number: int
    theme_name: str 
    original_recommendation: OriginalRecommendation
    verdict: str  # APPROVE | REJECT (renamed from 'decision')
    decision_rationale: str
    validated_decision: str  # use | modify | create (NEW - final decision)
    source_code: Optional[str] = None  # NEW - exact candidate code name if use/modify, or null if create
    validated_code: ValidatedCode
    model_config = ConfigDict(arbitrary_types_allowed=True)

class ValidationResult(BaseModel):
    code_validation: CodeValidation
    model_config = ConfigDict(arbitrary_types_allowed=True)

 
"""Codebook with reasoning"""
class CodeGeneratorReasoningResults(BaseModel):
    cluster_results: List[Dict[str, Any]]  # Raw results from each cluster
    
    step1_inputs: Dict[Union[int, str], Dict[str, Any]] = {}  # What Prompt 1 received
    step2_inputs: Dict[Union[int, str], Dict[str, Any]] = {}  # What Prompt 2 received
    step3_inputs: Dict[Union[int, str], Dict[str, Any]] = {}  # What Prompt 3 received
    step4_inputs: Dict[Union[int, str], Dict[str, Any]] = {}  # What Prompt 4 received
    step3_validation_warnings: Dict[Union[int, str], List[Dict[str, Any]]] = {}  # Validation warnings
    
    step1_summaries: Dict[Union[int, str], Dict[str, Any]]  # ClusterThemeAnalysis: {cluster_summary, themes[]}
    step2_analysis: Dict[Union[int, str], Dict[str, Any]]  # CodingDecisionOutput: {coding_decisions[]}
    step3_recommendations: Dict[Union[int, str], Dict[str, Any]]  # CodeGenerationOutput: {generated_codes[]}  
    step4_validations: Dict[Union[int, str], Dict[str, Any]]  # ValidationResult: {code_validations[]}
    step4_validated_codes: Dict[Union[int, str], Dict[str, Any]] = {}  # Final validated codes from Step 4
    
    stats: Dict[str, Any]
    generator_version: str
    var_lab: str
    total_clusters: int
    total_ideas: int
    processing_timestamp: str
    
    cluster_assignments: Dict[Union[int, str], Dict[str, Any]]
    codebook: List[Dict[str, str]]   
    cluster_data: Dict[Union[int, str], Dict[str, Any]]   
    validation_details: Optional[Dict[Union[int, str], Any]] = None
    redistribution_stats: Optional[Dict[str, Any]] = None  # Statistics from idea redistribution
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def get(self, key: str, default=None):
        return getattr(self, key, default)

# ============================================================================
#  RATE LIMITING CLASSES 
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


# ============================================================================
# PROCESSING UTILITY FUNCTIONS 
# ============================================================================

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


def normalize_usage(u) -> dict:
    """Normalize OpenAI API usage data to handle different naming conventions"""
    if not u:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    # Handle both dict-like and Pydantic model objects (ResponseUsage)
    def safe_get(obj, key, default=None):
        """Safely get value from dict or Pydantic model"""
        if hasattr(obj, 'get'):
            return obj.get(key, default)  # Dict-like access
        else:
            return getattr(obj, key, default)  # Attribute access for Pydantic models
    
    # Primary names (Responses API)
    input_tok = safe_get(u, "input_tokens")
    output_tok = safe_get(u, "output_tokens")
    
    # Back-compat aliases (Chat Completions API)
    if input_tok is None:
        input_tok = safe_get(u, "prompt_tokens", 0)
    if output_tok is None:
        output_tok = safe_get(u, "completion_tokens", 0)
    
    total = safe_get(u, "total_tokens", (input_tok or 0) + (output_tok or 0))
    
    # Optional breakdowns sometimes present on reasoning models
    details = safe_get(u, "output_tokens_details") or {}
    if details and hasattr(details, 'get'):
        reasoning_tok = details.get("reasoning_tokens")
    elif details:
        reasoning_tok = getattr(details, "reasoning_tokens", None)
    else:
        reasoning_tok = safe_get(u, "reasoning_tokens")
    
    result = {
        "prompt_tokens": input_tok or 0,
        "completion_tokens": output_tok or 0,
        "total_tokens": total or 0,
    }
    
    # Add reasoning tokens if present
    if reasoning_tok is not None:
        result["reasoning_tokens"] = reasoning_tok
    
    return result


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

# ============================================================================
# ASYNC API WRAPPERS & ERROR HANDLING
# ============================================================================

class RetryableError(Exception):
    """Retryable API errors for tenacity"""
    pass

class JSONValidationError(Exception):
    """JSON validation errors that should trigger retries with enhanced prompts"""
    pass

def extract_json_from_markdown(text: str) -> str:
    """Extract JSON from markdown code blocks if present"""
    # Check for ```json ... ``` pattern
    if text.strip().startswith('```json') and text.strip().endswith('```'):
        # Extract content between the markers
        lines = text.strip().split('\n')
        if len(lines) >= 3:  # Must have at least opening, content, closing
            json_content = '\n'.join(lines[1:-1])
            return json_content
    # Check for ``` ... ``` pattern (without json identifier)
    elif text.strip().startswith('```') and text.strip().endswith('```'):
        lines = text.strip().split('\n')
        if len(lines) >= 3:
            json_content = '\n'.join(lines[1:-1])
            return json_content
    # Return original text if no markdown blocks found
    return text

async def async_responses_create_with_json_retry(
    model: str, 
    prompt: str, 
    response_model, 
    reasoning_effort: str = "minimal", 
    text_verbosity: str = "low", 
    semaphore = None,
    rate_limiter = None,
    tpm_bucket = None,
    latency_tracker = None,
    config = None,
    max_retries: int = 3,
    timeout: float = 30.0
    ):
    """Async wrapper with JSON validation retry logic"""
    
    base_prompt = prompt
    
    for attempt in range(max_retries):
        try:
            # Get raw response
            resp = await async_responses_create_with_unified_limits(
                model=model,
                prompt=prompt,
                semaphore=semaphore,
                rate_limiter=rate_limiter,
                tpm_bucket=tpm_bucket,
                latency_tracker=latency_tracker,
                config=config,
                reasoning_effort=reasoning_effort,
                text_verbosity=text_verbosity
            )
            
            # Extract JSON from markdown if present
            raw_text = resp.output_text
            json_text = extract_json_from_markdown(raw_text)
            
            # Log if markdown unwrapping occurred
            if json_text != raw_text:
                logger.info(f"[JSON EXTRACTION] Unwrapped JSON from markdown code block for model {model}")
            
            # Try to parse JSON
            if hasattr(response_model, 'model_validate_json'):
                # Special handling for ClusterSummaryOutput - LLM might return array instead of object
                if response_model == ClusterSummaryOutput:
                    import json
                    try:
                        parsed_json = json.loads(json_text)
                        # Handle various LLM response formats
                        if isinstance(parsed_json, list):
                            if len(parsed_json) == 1 and isinstance(parsed_json[0], dict):
                                # Case: [{"cluster_id": {...}}] -> {"cluster_id": {...}}
                                parsed_json = parsed_json[0]
                            elif len(parsed_json) > 1:
                                # Case: [{"cluster_id": {...}}, {...}] - merge all dictionaries
                                merged_dict = {}
                                for item in parsed_json:
                                    if isinstance(item, dict):
                                        merged_dict.update(item)
                                parsed_json = merged_dict
                        # Now validate with the corrected structure
                        response = response_model.model_validate(parsed_json)
                    except (json.JSONDecodeError, ValidationError):
                        # Fall back to original method if preprocessing fails
                        response = response_model.model_validate_json(json_text)
                else:
                    response = response_model.model_validate_json(json_text)
                    
                if hasattr(response, 'root'):
                    return response.root
                return response
            else:
                # Fallback for models without model_validate_json
                return response_model(json_text)
                
        except ValidationError as e:
            error_msg = str(e)
            
            # Check if this is a retryable JSON error
            if attempt < max_retries - 1:  # Don't retry on last attempt
                if "control character" in error_msg.lower() or "invalid json" in error_msg.lower():
                    # Check if this is a GPT-4 model for specific handling
                    is_gpt4 = model.startswith('gpt-4')
                    
                    # Enhance prompt for retry based on error type
                    if "control character" in error_msg.lower():
                        prompt = base_prompt + "\n\nIMPORTANT: Return valid JSON only. Use standard ASCII characters. Avoid any control characters or special Unicode symbols in your response."
                    elif "expected" in error_msg.lower() and ("," in error_msg or "}" in error_msg):
                        prompt = base_prompt + "\n\nIMPORTANT: Return valid JSON with proper syntax. Ensure all objects have correct comma placement and closing braces."
                    elif is_gpt4 and "expected value at line 1 column 1" in error_msg:
                        # GPT-4 specific - likely wrapped in markdown
                        prompt = base_prompt + "\n\nIMPORTANT: Return ONLY raw JSON without any markdown formatting or code blocks. Do NOT wrap the JSON in ```json``` tags or any other formatting."
                    else:
                        prompt = base_prompt + "\n\nIMPORTANT: Return only valid, well-formed JSON. Check your syntax carefully. Do not include any markdown formatting or code blocks."
                    
                    continue  # Retry with enhanced prompt
            
            # Re-raise if not retryable or max retries reached
            raise e
        except Exception as e:
            # Non-JSON errors should not be retried here
            raise e
    
    # Should never reach here, but just in case
    raise JSONValidationError(f"Failed to get valid JSON after {max_retries} attempts")

@retry(
    reraise=True,
    stop=stop_after_attempt(6),
    wait=wait_exponential_jitter(initial=0.5, max=8),
    retry=retry_if_exception_type(RetryableError)
)
def _sync_responses_create(model: str, prompt: str, reasoning_effort: str = "minimal", text_verbosity: str = "low", timeout: float = 30.0, raw_input: bool = False):
    """Sync wrapper for responses.create with retry logic and adaptive timeout"""
    try:
        # Import ModelConfig here to avoid circular imports
        from config import ModelConfig
        
        # Check if this is a GPT-5 reasoning model
        model_config = ModelConfig()
        model_type = model_config.MODEL_TYPES.get(model, "chat")
        
        # Build request parameters based on model type
        request_params = {
            "model": model,
            "input": prompt if raw_input else [{"role": "user", "content": prompt}],
            "timeout": timeout  # Add adaptive timeout
        }
        
        # Only add reasoning parameters for GPT-5 models
        if model_type == "reasoning":
            request_params["text"] = {"verbosity": text_verbosity}
            request_params["reasoning"] = {"effort": reasoning_effort}
        
        return client.responses.create(**request_params)
    except asyncio.TimeoutError as e:
        logger.error(f"[API TIMEOUT] Request timed out after {timeout:.1f}s - {str(e)}")
        raise RetryableError(str(e)) from e
    except RateLimitError as e:
        logger.error(f"[RATE LIMIT] 429 error - {str(e)}")
        raise RetryableError(str(e)) from e
    except APITimeoutError as e:
        logger.error(f"[API TIMEOUT] OpenAI timeout - {str(e)}")
        raise RetryableError(str(e)) from e
    except APIConnectionError as e:
        logger.error(f"[CONNECTION ERROR] Network/connection issue - {str(e)}")
        raise RetryableError(str(e)) from e
    except InternalServerError as e:
        logger.error(f"[INTERNAL SERVER ERROR] 5xx error - {str(e)}")
        raise RetryableError(str(e)) from e
    except Exception as e:
        logger.error(f"[UNKNOWN ERROR] {type(e).__name__}: {str(e)}")
        raise  # Re-raise non-retryable errors immediately

async def async_responses_create(model: str, prompt: str, reasoning_effort: str = "minimal", text_verbosity: str = "low", timeout: float = 30.0, raw_input: bool = False):
    """Async wrapper using asyncio.to_thread for true concurrency with adaptive timeout"""
    return await asyncio.to_thread(_sync_responses_create, model, prompt, reasoning_effort, text_verbosity, timeout, raw_input)

async def async_responses_create_with_unified_limits(
    model: str, 
    prompt: str, 
    semaphore: asyncio.Semaphore,
    rate_limiter: AsyncLimiter,
    tpm_bucket: TokenBucket,
    latency_tracker: LatencyTracker,
    config,
    reasoning_effort: str = "minimal", 
    text_verbosity: str = "low"
):
    """Async wrapper with unified rate limiting (following qualityFilter.py patterns)"""
    # Estimate tokens for this request
    import tiktoken
    try:
        encoding = tiktoken.encoding_for_model(model)
        tokens_needed = int(len(encoding.encode(prompt)) * 1.2)  # Add 20% for output
    except:
        tokens_needed = 400  # Fallback estimate
    
    # Calculate adaptive timeout before rate limiting
    timeout_seconds = latency_tracker.get_timeout(tokens_needed)
    
    # STANDARDIZED RATE LIMITING PATTERN
    await tpm_bucket.wait_and_acquire(tokens_needed)
    async with semaphore:
        async with rate_limiter:
            response = await async_responses_create(model, prompt, reasoning_effort, text_verbosity, timeout_seconds)
            
            # Token reconciliation (if possible)
            if hasattr(response, 'usage') and response.usage:
                actual_tokens = response.usage.total_tokens
                delta = actual_tokens - tokens_needed
                await tpm_bucket.reconcile(delta)
            
            return response

# ============================================================================
# SHARED CODEBOOK 
# ============================================================================

class SharedCodebook:
    """Thread-safe shared codebook with async lock and version tracking"""
    
    def __init__(self, initial_codes: List[Dict[str, str]], max_cached_versions: int = 5):
        self._codes = initial_codes.copy()
        self._lock = asyncio.Lock()
        self._version = 0
        self._update_log = []
        self._embedding_cache = {}
        self._max_cached_versions = max_cached_versions
    
    async def get_current_snapshot(self) -> Tuple[List[Dict[str, str]], int]:
        """Get current codes and version atomically"""
        async with self._lock:
            return self._codes.copy(), self._version
    
    async def add_code_if_new(self, code: str, definition: str, cluster_id: Optional[Union[int, str]] = None) -> Tuple[bool, int]:
        """Add a new code if it doesn't exist, return (added, new_version)"""
        async with self._lock:
            # Check if code already exists
            for existing in self._codes:
                if existing['code'].lower() == code.lower():
                    return False, self._version
            
            # Add new code with cluster origin tracking
            code_entry = {'code': code, 'definition': definition}
            if cluster_id is not None:
                code_entry['source_cluster_id'] = str(cluster_id)
            
            self._codes.append(code_entry)
            self._version += 1
            self._update_log.append({
                'version': self._version,
                'action': 'add',
                'code': code,
                'cluster_id': cluster_id,
                'timestamp': time.time()
            })
            return True, self._version
    
    async def replace_code(self, original_code: str, new_code: str, new_definition: str, cluster_id: Optional[Union[int, str]] = None) -> Tuple[bool, int]:
        """Replace an existing code with a modified version, return (replaced, new_version)"""
        async with self._lock:
            # Find and replace the original code
            for i, existing in enumerate(self._codes):
                if existing['code'].lower() == original_code.lower():
                    # Preserve existing cluster_id if none provided, otherwise use new one
                    existing_cluster_id = existing.get('source_cluster_id') if cluster_id is None else str(cluster_id)
                    
                    replacement_entry = {'code': new_code, 'definition': new_definition}
                    if existing_cluster_id is not None:
                        replacement_entry['source_cluster_id'] = existing_cluster_id
                    
                    self._codes[i] = replacement_entry
                    self._version += 1
                    self._update_log.append({
                        'version': self._version,
                        'action': 'replace',
                        'original_code': original_code,
                        'new_code': new_code,
                        'cluster_id': cluster_id,
                        'timestamp': time.time()
                    })
                    return True, self._version
            
            # Original code not found - fail gracefully instead of creating duplicate
            self._update_log.append({
                'version': self._version,
                'action': 'replace_failed',
                'original_code': original_code,
                'new_code': new_code,
                'cluster_id': cluster_id,
                'timestamp': time.time(),
                'reason': 'original_code_not_found'
            })
            return False, self._version
    
    async def get_version_info(self) -> Dict[str, Any]:
        """Get codebook version information"""
        async with self._lock:
            return {
                'version': self._version,
                'total_codes': len(self._codes),
                'recent_updates': self._update_log[-5:] if self._update_log else []
            }
    
    async def get_embeddings_for_version(self, version: int) -> Optional[List[np.ndarray]]:
        """Get cached embeddings for a specific version"""
        async with self._lock:
            return self._embedding_cache.get(version)
    
    async def cache_embeddings(self, version: int, embeddings: List[np.ndarray]):
        """Cache embeddings for a version with memory management"""
        async with self._lock:
            self._embedding_cache[version] = embeddings
            
            # Memory management: keep only recent versions
            if len(self._embedding_cache) > self._max_cached_versions:
                # Remove oldest cached version
                oldest_version = min(self._embedding_cache.keys())
                del self._embedding_cache[oldest_version]
    
    def _normalize_code_name(self, code: str) -> str:
        """Normalize code name for consistent comparison"""
        return code.strip().lower().replace('-', ' ').replace('_', ' ')
    
    def _is_duplicate(self, code1: str, code2: str) -> bool:
        """Check if two codes are duplicates using normalized comparison"""
        norm1 = self._normalize_code_name(code1)
        norm2 = self._normalize_code_name(code2)
        return norm1 == norm2
    
    async def batch_update(self, new_codes: List[Dict[str, str]], expected_base_version: int) -> bool:
        """Phase 3: Batch update multiple codes atomically with comprehensive duplicate detection"""
        async with self._lock:
            # Version conflict check with retry logic
            if self._version != expected_base_version:
                version_conflict_info = {
                    'expected_version': expected_base_version,
                    'actual_version': self._version,
                    'codes_to_add': len(new_codes),
                    'timestamp': time.time()
                }
                
                # Log version conflict
                self._update_log.append({
                    'version': self._version,
                    'action': 'version_conflict_detected',
                    **version_conflict_info
                })
                
                # For safety, validate that proceeding won't cause issues
                # Check if any of the new codes would conflict with recent additions
                recent_adds = [log for log in self._update_log[-10:] 
                              if log.get('action') in ['add', 'batch_add'] and 
                              log.get('version', 0) > expected_base_version]
                
                potential_conflicts = 0
                for code_dict in new_codes:
                    for recent_add in recent_adds:
                        if self._is_duplicate(code_dict['code'], recent_add.get('code', '')):
                            potential_conflicts += 1
                
                if potential_conflicts > 0:
                    self._update_log.append({
                        'version': self._version,
                        'action': 'version_conflict_blocked',
                        'potential_conflicts': potential_conflicts,
                        **version_conflict_info
                    })
                    return False  # Refuse to proceed with conflicting update
                else:
                    self._update_log.append({
                        'version': self._version,
                        'action': 'version_conflict_resolved',
                        'reason': 'no_potential_conflicts',
                        **version_conflict_info
                    })
            
            # Batch add all new codes with enhanced duplicate detection
            added_count = 0
            duplicates_merged = 0  # Count of cluster IDs merged into existing codes
            
            for code_dict in new_codes:
                code = code_dict['code']
                definition = code_dict['definition']
                cluster_id = code_dict.get('cluster_id', 'unknown')
                
                # Enhanced duplicate check - normalize and compare
                is_duplicate = False
                duplicate_of = None
                duplicate_entry = None

                for existing in self._codes:
                    if self._is_duplicate(existing['code'], code):
                        is_duplicate = True
                        duplicate_of = existing['code']
                        duplicate_entry = existing  # Keep reference to merge into
                        break

                if is_duplicate:
                    # MERGE cluster IDs instead of preventing duplicate
                    if cluster_id and cluster_id != 'unknown':
                        existing_clusters = duplicate_entry.get('source_cluster_id', '')
                        if existing_clusters:
                            # Append new cluster ID to existing comma-separated list
                            duplicate_entry['source_cluster_id'] = f"{existing_clusters},{cluster_id}"
                        else:
                            # First cluster ID for this code
                            duplicate_entry['source_cluster_id'] = str(cluster_id)

                    duplicates_merged += 1
                    self._update_log.append({
                        'version': self._version,
                        'action': 'cluster_id_merged',
                        'code': code,
                        'cluster_id': cluster_id,
                        'merged_into': duplicate_of,
                        'timestamp': time.time()
                    })
                else:
                    # New code - add normally
                    code_entry = {'code': code, 'definition': definition}
                    if cluster_id and cluster_id != 'unknown':
                        code_entry['source_cluster_id'] = str(cluster_id)

                    self._codes.append(code_entry)
                    added_count += 1
                    self._update_log.append({
                        'version': self._version + 1,
                        'action': 'batch_add',
                        'code': code,
                        'cluster_id': cluster_id,
                        'timestamp': time.time()
                    })
            
            # Update version once for the entire batch
            if added_count > 0:
                self._version += 1
            
            # Log summary if cluster IDs were merged
            if duplicates_merged > 0:
                self._update_log.append({
                    'version': self._version,
                    'action': 'batch_summary',
                    'added_count': added_count,
                    'cluster_ids_merged': duplicates_merged,
                    'timestamp': time.time()
                })
            
            return added_count > 0


# ============================================================================
# THEME-BASED SIMILARITY ENGINE
# ============================================================================

class SimilarityEngine:
    """Handles theme-based similarity calculation and batch formation"""
    
    def __init__(self, similarity_threshold: float = 0.7, verbose_reporter: VerboseReporter = None):
        self.similarity_threshold = similarity_threshold
        self.verbose_reporter = verbose_reporter or VerboseReporter(False)
    
    async def embed_themes(self, themes: Dict[str, ClusterSummaryOutput]) -> Dict[str, np.ndarray]:
        """Generate embeddings for all theme names using efficient batch processing (like embedder.py)"""
        self.verbose_reporter.step_start("Theme Embedding")
        self.verbose_reporter.stat_line(f"Processing {len(themes)} theme names in batches")
        
        # Prepare data for batch processing
        cluster_ids = list(themes.keys())
        # Get theme_labels - themes[cid] is ClusterSummaryOutput with root dict
        theme_names = []
        for cid in cluster_ids:
            theme_output = themes[cid]
            try:
                # Access the ClusterSummaryItem from the root dict
                cluster_summary_items = list(theme_output.root.values())
                if cluster_summary_items and cluster_summary_items[0].extracted_themes:
                    theme_names.append(cluster_summary_items[0].extracted_themes[0].theme_label)
                else:
                    theme_names.append("Unknown")
            except (AttributeError, IndexError):
                theme_names.append("Unknown")   
                      
        # Process in batches (OpenAI supports up to 2048 inputs per call, but use smaller batches for reliability)
        batch_size = 100
        theme_embeddings = {}
        
        try:
            for i in range(0, len(theme_names), batch_size):
                batch_statements = theme_names[i:i + batch_size]
                batch_ids = cluster_ids[i:i + batch_size]
                
                self.verbose_reporter.stat_line(f"Processing batch {i//batch_size + 1}/{(len(theme_names) + batch_size - 1)//batch_size} ({len(batch_statements)} themes)")
                
                # Use efficient batch embedding like embedder.py
                batch_embeddings = await self._embed_openai_batch(batch_statements)
                
                # Map embeddings back to cluster IDs
                for cluster_id, embedding in zip(batch_ids, batch_embeddings):
                    theme_embeddings[cluster_id] = embedding
                    
        except Exception as e:
            self.verbose_reporter.error(f"Batch embedding failed: {e}")
            # Fallback to individual processing if batch fails
            self.verbose_reporter.warning("Falling back to individual embedding processing")
            
            for cluster_id, theme in themes.items():
                try:
                    # Extract theme_label from ClusterSummaryOutput structure
                    cluster_summary_items = list(theme.root.values())
                    if cluster_summary_items and cluster_summary_items[0].extracted_themes:
                        theme_label = cluster_summary_items[0].extracted_themes[0].theme_label
                    else:
                        theme_label = "Unknown"
                    embedding = await self._get_embedding(theme_label)
                    theme_embeddings[cluster_id] = embedding
                except Exception as individual_error:
                    self.verbose_reporter.error(f"Failed to embed theme for cluster {cluster_id}: {individual_error}")
                    # Use zero vector as fallback
                    theme_embeddings[cluster_id] = np.zeros(1536)
        
        self.verbose_reporter.stat_line(f"Generated embeddings for {len(theme_embeddings)} themes")
        self.verbose_reporter.step_complete("Theme Embedding")
        return theme_embeddings
    
    async def _embed_openai_batch(self, batch_texts: List[str]) -> List[np.ndarray]:
        """Batch embedding with retry logic"""
        client = OpenAI(api_key=OPENAI_API_KEY)
       
        for attempt in range(3):
            try:
                response = client.embeddings.create(
                    input=batch_texts,
                    #model="text-embedding-3-large"
                    model=DEFAULT_CODEDESIGNER_CONFIG.embedding_model
                )
                return [np.array(item.embedding, dtype=np.float32) for item in response.data]
                
            except Exception as e:
                if attempt == 2:  # Last attempt
                    raise
                # Exponential backoff
                wait_time = 0.8 * (2 ** attempt)
                self.verbose_reporter.warning(f"Embedding batch failed (attempt {attempt + 1}), retrying in {wait_time}s: {str(e)[:100]}")
                await asyncio.sleep(wait_time)

    async def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for single text"""
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.embeddings.create(
            #model="text-embedding-3-large",
            model=DEFAULT_CODEDESIGNER_CONFIG.embedding_model,
            input=text
        )
        return np.array(response.data[0].embedding)
    
    def create_dissimilarity_batches(self, theme_embeddings: Dict[str, np.ndarray], themes: Dict = None) -> List[List[str]]:
        """Create batches using Progressive Dispersion with Conflict Set Tracking"""
        self.verbose_reporter.step_start("Progressive Dispersion Batching")
        
        cluster_ids = list(theme_embeddings.keys())
        embeddings_matrix = np.array([theme_embeddings[cid] for cid in cluster_ids])
        
        # Calculate pairwise similarity matrix
        similarity_matrix = cosine_similarity(embeddings_matrix)
        
        # Report similarity distribution (keep for comparison)
        self._report_similarity_distribution(similarity_matrix)
        
        # Report hierarchical dissimilarity batching strategy  
        progressive_thresholds = [0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8]
        self.verbose_reporter.stat_line(f"Using hierarchical dissimilarity batching with max 0.80 similarity: {' → '.join(map(str, progressive_thresholds))}")
        
        # Hierarchical Dissimilarity Batching
        # Create batches in order of increasing similarity tolerance
        batches = []
        unassigned_indices = set(range(len(cluster_ids)))
        
        for batch_num, threshold in enumerate(progressive_thresholds):
            if not unassigned_indices:
                break
                
            # Create batch with themes having max similarity < threshold
            batch_indices = self._extract_similarity_constrained_batch(
                similarity_matrix, list(unassigned_indices), threshold
            )
            
            if batch_indices:
                # Convert to cluster_ids and add to batches
                batch_cluster_ids = [cluster_ids[i] for i in batch_indices]
                batches.append(batch_cluster_ids)
                
                # Remove assigned themes
                unassigned_indices -= set(batch_indices)
                
                self.verbose_reporter.stat_line(f"Batch {batch_num + 1} (threshold < {threshold}): {len(batch_indices)} themes")
            else:
                self.verbose_reporter.stat_line(f"Batch {batch_num + 1} (threshold < {threshold}): 0 themes (skipped)")
        
        # Handle any remaining themes that couldn't be batched within 0.85 similarity constraint
        if unassigned_indices:
            # Try to create final batches still respecting max 0.85 similarity
            remaining_themes = list(unassigned_indices)
            while remaining_themes:
                # Extract one more batch at 0.85 threshold
                final_batch_indices = self._extract_similarity_constrained_batch(
                    similarity_matrix, remaining_themes, 0.85
                )
                if final_batch_indices:
                    final_batch_cluster_ids = [cluster_ids[i] for i in final_batch_indices]
                    batches.append(final_batch_cluster_ids)
                    # Remove assigned themes
                    for idx in final_batch_indices:
                        remaining_themes.remove(idx)
                    self.verbose_reporter.stat_line(f"Additional batch (max 0.85): {len(final_batch_indices)} themes")
                else:
                    # Force remaining themes into singletons if they can't be grouped at 0.85
                    for idx in remaining_themes:
                        singleton_cluster_id = [cluster_ids[idx]]
                        batches.append(singleton_cluster_id)
                        self.verbose_reporter.stat_line(f"Singleton batch: {cluster_ids[idx]} (couldn't group at 0.85)")
                    break
        
        # Apply anti-greedy redistribution to balance batch sizes
        batches = self._redistribute_to_anti_greedy_pattern(batches, similarity_matrix, cluster_ids, progressive_thresholds)
        
        # Report final batch statistics
        for batch_idx, batch_cluster_ids in enumerate(batches):
            # Convert cluster_ids back to indices for similarity calculation
            batch_indices = [cluster_ids.index(cid) for cid in batch_cluster_ids]
            
            # Calculate batch statistics
            batch_similarities = []
            for i, idx_i in enumerate(batch_indices):
                for j in range(i + 1, len(batch_indices)):
                    idx_j = batch_indices[j]
                    batch_similarities.append(similarity_matrix[idx_i, idx_j])
            
            avg_sim = np.mean(batch_similarities) if batch_similarities else 0
            max_sim = max(batch_similarities) if batch_similarities else 0
            
            self.verbose_reporter.stat_line(
                f"Final Batch {batch_idx + 1}: {len(batch_cluster_ids)} themes, "
                f"avg_sim={avg_sim:.3f}, max_sim={max_sim:.3f}"
            )
        
        # VERBOSE: Final batch assignments summary
        self.verbose_reporter.stat_line("\n=== FINAL BATCH ASSIGNMENTS ===")
        for batch_idx, batch_cluster_ids in enumerate(batches):
            theme_labels = []
            if themes:
                for cid in batch_cluster_ids:
                    if cid in themes:
                        theme_output = themes[cid]
                        cluster_summary_items = list(theme_output.root.values())
                        if cluster_summary_items and cluster_summary_items[0].extracted_themes:
                            theme_label = cluster_summary_items[0].extracted_themes[0].theme_label
                            theme_labels.append(f"C{cid}: '{theme_label}'")
                        else:
                            theme_labels.append(f"C{cid}: 'unknown'")
                    else:
                        theme_labels.append(f"C{cid}: unknown")
            else:
                theme_labels = [f"C{cid}" for cid in batch_cluster_ids]
            
            #self.verbose_reporter.stat_line(f"Batch {batch_idx + 1}: {', '.join(theme_labels)}")
            self.verbose_reporter.stat_line(f"Batch {batch_idx + 1}:\n" + "\n".join(theme_labels))
            
        
        # Final reporting
        self._report_batch_quality(batches, similarity_matrix, cluster_ids)
        self.verbose_reporter.step_complete("Progressive Dispersion Batching")
        return batches
    
    def _report_similarity_distribution(self, similarity_matrix: np.ndarray):
        """Report similarity distribution statistics"""
        n_clusters = similarity_matrix.shape[0]
        similarities = similarity_matrix[np.triu_indices(n_clusters, k=1)]
        
        self.verbose_reporter.stat_line(f"Analyzing similarity distribution for {n_clusters} themes:")
        
        thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        for threshold in thresholds:
            count = np.sum(similarities < threshold)
            percentage = count / len(similarities) * 100 if len(similarities) > 0 else 0
            self.verbose_reporter.stat_line(
                f"  Similarity < {threshold}: {count} pairs ({percentage:.1f}%)"
            )
    
    def _report_batch_quality(self, batches, similarity_matrix, cluster_ids):
        """Report detailed batch quality metrics"""
        self.verbose_reporter.empty_line()
        self.verbose_reporter.stat_line("=== Batch Quality Report ===")
        self.verbose_reporter.stat_line(f"Total batches created: {len(batches)}")
        
        # Check if any high-similarity pairs ended up in same batch
        violations = 0
        conflict_threshold = 0.9
        
        for batch_idx, batch in enumerate(batches):
            batch_indices = [cluster_ids.index(cid) for cid in batch]
            for i, idx_i in enumerate(batch_indices):
                for j in range(i + 1, len(batch_indices)):
                    idx_j = batch_indices[j]
                    if similarity_matrix[idx_i, idx_j] > conflict_threshold:
                        violations += 1
                        self.verbose_reporter.warning(
                            f"High similarity ({similarity_matrix[idx_i, idx_j]:.3f}) "
                            f"in batch {batch_idx + 1}"
                        )
        
        if violations == 0:
            self.verbose_reporter.stat_line("✓ No high-similarity conflicts in any batch")
        else:
            self.verbose_reporter.warning(f"⚠ Found {violations} high-similarity pairs in same batch")
        
        # Report batch size distribution
        batch_sizes = [len(batch) for batch in batches]
        self.verbose_reporter.stat_line(f"Batch sizes: min={min(batch_sizes)}, max={max(batch_sizes)}, avg={np.mean(batch_sizes):.1f}")
    
    def create_sub_batches(self, batch: List[int], max_size: int = 10) -> List[List[int]]:
        """Split large batch into sub-batches for rate limiting"""
        if len(batch) <= max_size:
            return [batch]
        
        sub_batches = []
        for i in range(0, len(batch), max_size):
            sub_batch = batch[i:i + max_size]
            sub_batches.append(sub_batch)
        
        return sub_batches
    
    def _extract_similarity_constrained_batch(self, similarity_matrix: np.ndarray, available_indices: List[int], max_similarity_threshold: float) -> List[int]:
        """Extract single largest batch from available indices where max similarity < threshold"""
        if not available_indices:
            return []
        
        # Greedy approach: start with first theme, add compatible themes
        batch_indices = [available_indices[0]]
        remaining = set(available_indices[1:])
        
        # Keep adding themes that don't violate similarity constraint
        added_theme = True
        while added_theme and remaining:
            added_theme = False
            
            # Try each remaining theme
            for candidate_idx in list(remaining):
                # Check if candidate is compatible with all themes in current batch
                can_add = True
                for batch_member_idx in batch_indices:
                    if similarity_matrix[candidate_idx, batch_member_idx] >= max_similarity_threshold:
                        can_add = False
                        break
                
                if can_add:
                    batch_indices.append(candidate_idx)
                    remaining.remove(candidate_idx)
                    added_theme = True
                    break  # Start over to find next compatible theme
        
        return batch_indices

    def _redistribute_to_anti_greedy_pattern(self, batches: List[List[str]], 
                                           similarity_matrix: np.ndarray, 
                                           cluster_ids: List[str], 
                                           progressive_thresholds: List[float]) -> List[List[str]]:
        """Redistribute clusters to achieve anti-greedy pattern (increasing batch sizes)"""
        if not batches or len(batches) <= 1:
            return batches
            
        self.verbose_reporter.step_start("Anti-Greedy Batch Redistribution")
        
        # Calculate original distribution
        original_sizes = [len(batch) for batch in batches]
        total_clusters = sum(original_sizes)
        
        self.verbose_reporter.stat_line(f"Original batch sizes: {original_sizes} (total: {total_clusters})")
        
        # Calculate target anti-greedy distribution (increasing sizes)
        target_sizes = self._calculate_anti_greedy_targets(total_clusters, len(batches))
        self.verbose_reporter.stat_line(f"Target batch sizes: {target_sizes}")
        
        # Identify moveable clusters between adjacent batches
        moveable_clusters = self._identify_moveable_clusters(batches, similarity_matrix, cluster_ids, progressive_thresholds)
        
        # Perform redistribution
        redistributed_batches = self._perform_redistribution(batches, target_sizes, moveable_clusters, similarity_matrix, cluster_ids, progressive_thresholds)
        
        # Report results
        final_sizes = [len(batch) for batch in redistributed_batches]
        self.verbose_reporter.stat_line(f"Final batch sizes: {final_sizes}")
        
        moved_count = sum(abs(final_sizes[i] - original_sizes[i]) for i in range(len(final_sizes))) // 2
        self.verbose_reporter.stat_line(f"Clusters redistributed: {moved_count}")
        
        self.verbose_reporter.step_complete("Anti-Greedy Batch Redistribution")
        
        return redistributed_batches

    def _calculate_anti_greedy_targets(self, total_clusters: int, num_batches: int) -> List[int]:
        """Calculate target batch sizes for anti-greedy pattern (increasing sizes)"""
        if num_batches <= 1:
            return [total_clusters]
        
        # Create increasing pattern: start small, end large
        # Use arithmetic progression with positive common difference
        base_size = total_clusters // num_batches
        remainder = total_clusters % num_batches
        
        # Create increasing sequence
        targets = []
        adjustment = -(num_batches - 1) // 2  # Start below average
        
        for i in range(num_batches):
            target = base_size + adjustment
            if i < remainder:  # Distribute remainder across later batches
                target += 1
            targets.append(max(1, target))  # Ensure minimum size of 1
            adjustment += 1  # Increase for next batch
        
        # Adjust if total doesn't match (due to minimum size constraints)
        current_total = sum(targets)
        if current_total != total_clusters:
            # Add/remove from last batch
            targets[-1] += (total_clusters - current_total)
        
        return targets

    def _identify_moveable_clusters(self, batches: List[List[str]], 
                                  similarity_matrix: np.ndarray,
                                  cluster_ids: List[str],
                                  progressive_thresholds: List[float]) -> Dict[int, Dict[int, List[str]]]:
        """Identify clusters that can be moved between adjacent batches while respecting similarity constraints"""
        moveable = {}  # {from_batch: {to_batch: [cluster_ids]}}
        
        for from_idx in range(len(batches) - 1):  # Don't check last batch
            to_idx = from_idx + 1
            #from_threshold = progressive_thresholds[from_idx] if from_idx < len(progressive_thresholds) else 0.85
            to_threshold = progressive_thresholds[to_idx] if to_idx < len(progressive_thresholds) else 0.85
            
            # Find clusters in from_batch that could move to to_batch
            moveable_to_next = []
            
            for cluster_id in batches[from_idx]:
                # Check if this cluster could fit in the next batch without violating similarity constraints
                if self._can_cluster_join_batch(cluster_id, batches[to_idx], similarity_matrix, cluster_ids, to_threshold):
                    moveable_to_next.append(cluster_id)
            
            if moveable_to_next:
                if from_idx not in moveable:
                    moveable[from_idx] = {}
                moveable[from_idx][to_idx] = moveable_to_next
        
        return moveable

    def _can_cluster_join_batch(self, cluster_id: str, target_batch: List[str], 
                              similarity_matrix: np.ndarray, cluster_ids: List[str], 
                              threshold: float) -> bool:
        """Check if a cluster can join a batch without violating similarity constraints"""
        if not target_batch:
            return True
        
        try:
            cluster_idx = cluster_ids.index(cluster_id)
        except ValueError:
            return False
        
        # Check similarity with all clusters in target batch
        for target_cluster_id in target_batch:
            try:
                target_idx = cluster_ids.index(target_cluster_id)
                similarity = similarity_matrix[cluster_idx, target_idx]
                if similarity >= threshold:
                    return False  # Would violate similarity constraint
            except ValueError:
                continue
        
        return True

    def _perform_redistribution(self, batches: List[List[str]], target_sizes: List[int],
                              moveable_clusters: Dict[int, Dict[int, List[str]]],
                              similarity_matrix: np.ndarray, cluster_ids: List[str],
                              progressive_thresholds: List[float]) -> List[List[str]]:
        """Perform the actual redistribution of clusters"""
        # Create mutable copies
        redistributed = [batch.copy() for batch in batches]
        #current_sizes = [len(batch) for batch in redistributed]
        
        # Redistribute from early batches to later batches
        for from_idx in range(len(redistributed) - 1):
            if from_idx not in moveable_clusters:
                continue
                
            current_size = len(redistributed[from_idx])
            target_size = target_sizes[from_idx]
            
            # If this batch is larger than target, try to move clusters to later batches
            if current_size > target_size:
                excess = current_size - target_size
                
                # Try to move to each possible later batch
                for to_idx, moveable_list in moveable_clusters[from_idx].items():
                    if excess <= 0:
                        break
                    
                    current_to_size = len(redistributed[to_idx])
                    target_to_size = target_sizes[to_idx]
                    
                    # If target batch has room, move some clusters
                    if current_to_size < target_to_size:
                        can_accept = target_to_size - current_to_size
                        to_move = min(excess, can_accept, len(moveable_list))
                        
                        # Move clusters
                        for i in range(to_move):
                            cluster_to_move = moveable_list[i]
                            if cluster_to_move in redistributed[from_idx]:
                                # Verify one more time before moving
                                if self._can_cluster_join_batch(cluster_to_move, redistributed[to_idx],
                                                              similarity_matrix, cluster_ids,
                                                              progressive_thresholds[to_idx] if to_idx < len(progressive_thresholds) else 0.85):
                                    redistributed[from_idx].remove(cluster_to_move)
                                    redistributed[to_idx].append(cluster_to_move)
                                    excess -= 1
        
        return redistributed


# ============================================================================
# MAIN CODEDESIGNER CLASS
# ============================================================================

class InductiveCodeGenerator:
    """CodeDesigner: Theme-based similarity processing with 4-stage pipeline"""
    
    def __init__(
        self,
        cluster_results: List[ClusterModel],
        starter_codes: List[Dict[str, str]],
        var_lab: str,
        verbose: bool = False,
        verbose_detailed: bool = False,
        prompt_printer = None,
        config = None,
        processing_config: Optional[ProcessingConfig] = None,
        verbose_reporter: Optional['VerboseReporter'] = None,
        stages_to_run: str = 'all',  # 'all' or 'theme_extraction_only'
        **kwargs  # For backward compatibility
    ):
        self.cluster_results = cluster_results
                
        self.starter_codes = starter_codes
        self.var_lab = var_lab
        self.verbose = verbose
        self.verbose_detailed = verbose_detailed
        self.prompt_printer = prompt_printer
        self.config = config or DEFAULT_CODEDESIGNER_CONFIG
        self.processing_config = processing_config or DEFAULT_PROCESSING_CONFIG
        self.stages_to_run = stages_to_run
        
        # Sampling statistics tracking
        self.sampling_stats = {
            'clusters_processed': 0,
            'clusters_sampled': 0,
            'total_original_ideas': 0,
            'total_sampled_ideas': 0
        }
        
        # Initialize components
        self.model_config = kwargs.get('model_config') or ModelConfig()
        self.verbose_reporter = verbose_reporter or VerboseReporter(verbose, capture_logging=True)
        
        # Initialize config-aware concurrency control
        self.concurrency_semaphore = asyncio.Semaphore(self.config.async_concurrency_limit)
        
        # Initialize embedding client (sync client is used via async_responses_create wrapper)
        self.embedding_client = OpenAI()
        
        # Initialize tokenizer with proper model mapping
        try:
            self.encoding = tiktoken.encoding_for_model(self.config.model)
        except KeyError:
            # Map newer model names to their tiktoken-compatible equivalents
            tiktoken_model_mapping = {
                'gpt-4.1-mini': 'gpt-4o-mini',
                'gpt-4.1': 'gpt-4o', 
                'gpt-4.1-turbo': 'gpt-4o'}
            
            tiktoken_model = tiktoken_model_mapping.get(self.config.model) 
            if tiktoken_model:
                try:
                    self.encoding = tiktoken.encoding_for_model(tiktoken_model)
                    # This is expected behavior, not a fallback
                except KeyError:
                    # Only this is truly a fallback
                    self.encoding = tiktoken.get_encoding("cl100k_base")
                    self.verbose_reporter.warning(f"Fallback to cl100k_base encoding for {self.config.model}")
            else:
                self.encoding = tiktoken.get_encoding("cl100k_base")
                self.verbose_reporter.warning(f"Fallback to cl100k_base encoding for {self.config.model}")
        
        # Initialize unified rate limiting system (following qualityFilter.py patterns)
        rate_limits = get_openai_rate_limits(self.config.model)

        # Initialize bootstrap attributes (will be populated by async_initialize)
        self.bootstrap_latency = None
        self.bootstrap_tokens = None
        self._bootstrap_completed = False

        # Calculate average tokens for rate limiting (fallback until bootstrap completes)
        self.avg_tokens = self._calculate_avg_tokens()

        # Create unified rate limiting system
        self.tpm_bucket = TokenBucket(rate_limits.tokens_per_minute * self.processing_config.rate_limit_headroom)
        
        # Add AsyncLimiter for unified rate limiting (following qualityFilter pattern)
        arrival_rate = min(
            rate_limits.requests_per_minute * self.processing_config.rate_limit_headroom / 60,
            rate_limits.tokens_per_minute * self.processing_config.rate_limit_headroom / self.avg_tokens / 60
        )
        if arrival_rate < 1:
            self.rate_limiter = AsyncLimiter(1, time_period=1/arrival_rate)
        else:
            self.rate_limiter = AsyncLimiter(int(arrival_rate), time_period=1.0)

        # Calculate optimal concurrency based on API limits and latency
        rpm_concurrency = rate_limits.requests_per_minute * self.processing_config.rate_limit_headroom / 60 * 2.0  # Assume 2s avg latency
        tpm_concurrency = (rate_limits.tokens_per_minute * self.processing_config.rate_limit_headroom / self.avg_tokens) * 2.0
        optimal_concurrency = min(rpm_concurrency, tpm_concurrency, self.config.async_concurrency_limit)

        self.concurrency_semaphore = asyncio.Semaphore(int(optimal_concurrency))
        
        # Store limits for reporting
        self.rpm_limit = rate_limits.requests_per_minute
        self.tpm_limit = rate_limits.tokens_per_minute
        
        # Initialize latency tracking (following qualityFilter pattern)
        self.latency_tracker = LatencyTracker()
        
        # Bootstrap measurement attributes (from qualityFilter.py)
        # Adaptive token estimation
        self.input_token_history = deque(maxlen=3)  # First 3 input token counts
        self.output_token_history = deque(maxlen=5)  # First 5 output token counts
        self.estimation_errors = deque(maxlen=50)  # Track accuracy
        self.first_prompt_tokens = None  # Cache first prompt calculation
        self.actual_total_tokens = deque(maxlen=50)  # Track actual total usage
        
        # Initialize components
        self.similarity_engine = SimilarityEngine(similarity_threshold=self.config.similarity_threshold, verbose_reporter=self.verbose_reporter)
        self.shared_codebook = SharedCodebook(starter_codes)
        
        # Results storage
        self._results = []
        self._processing_stats = {}
        
        # Initialize prompt tracking for CodeGeneratorReasoningResults
        self.step1_inputs = {}           # Theme extraction inputs
        self.step2_inputs = {}           # Candidate code selection inputs
        self.step3_inputs = {}           # Code generation inputs
        self.step4_inputs = {}           # Validation inputs
        self.step1_summaries = {}        # Theme extraction results
        self.step2_analysis = {}         # Candidate codes
        self.step3_recommendations = {}  # Code generation results
        self.step4_validations = {}      # Validation results
        self.step4_validated_codes = {}  # Final validated codes
        self.cluster_assignments = {}    # Cluster-to-code mappings
        
        # Initialize modification leak collection for race condition recovery
        self.modification_leaks = []      # Failed MODIFY operations for retry
        
        # Initialize redistribution statistics
        self._redistribution_stats = {
            'clusters_redistributed': [],
            'redistribution_details': {}
        }

    async def async_initialize(self):
        """Initialize bootstrap measurement and update rate limiting with real API performance data"""
        if self._bootstrap_completed:
            return
        
        # Bootstrap is needed for all modes to properly configure rate limiting
        
        # Bootstrap measurement for real API performance data (following qualityFilter.py pattern)
        if self.cluster_results and len(self.cluster_results) > 0:
            try:
                if self.verbose_reporter.enabled:
                    self.verbose_reporter.stat_line("Running bootstrap measurement (3 probe calls)...")
                
                # Prepare sample clusters for probing
                sample_clusters = self.cluster_results[:min(1, len(self.cluster_results))]
                if len(sample_clusters) < 3:
                    # Duplicate clusters if we have fewer than 3
                    sample_clusters = sample_clusters * 3
                    sample_clusters = sample_clusters[:3]
                
                cluster_cycle = itertools.cycle(sample_clusters)
                
                async def probe_bootstrap_call():
                    cluster_data = next(cluster_cycle)
                    ideas_list = cluster_data.response_ideas or []
                    ideas_strings = [str(idea) for idea in ideas_list[:10]]  # Convert to strings
                    
                    # Create probe task with proper structure matching probe method expectations
                    probe_task = {
                        'cluster_id': 'bootstrap_probe',
                        'ideas': ideas_strings  # Use 'ideas' key that probe method expects
                    }
                    return await self.probe_call_theme_extraction(probe_task)
                
                # Run bootstrap measurement
                start_bootstrap = time.time()
                self.bootstrap_latency, self.bootstrap_tokens = await bootstrap_measure_async(
                    probe_bootstrap_call, n_probes=3
                )
                bootstrap_time = time.time() - start_bootstrap
                
                if self.verbose_reporter.enabled:
                    self.verbose_reporter.stat_line(f"Bootstrap results: {self.bootstrap_latency:.3f}s avg latency, {self.bootstrap_tokens:.0f} avg tokens ({bootstrap_time:.1f}s)")
                
                # Update avg_tokens with bootstrap data
                self.avg_tokens = int(self.bootstrap_tokens)
                
                # Initialize LatencyTracker with bootstrap values (3 samples for stability)
                for _ in range(3):
                    self.latency_tracker.add(self.bootstrap_latency)
                
                # Initialize progressive token estimation with bootstrap
                self.first_prompt_tokens = int(self.bootstrap_tokens * 0.85)  # Input portion
                
                self._bootstrap_completed = True
                
            except Exception as e:
                if self.verbose_reporter.enabled:
                    self.verbose_reporter.warning(f"Bootstrap measurement failed: {e}")
                # Use fallback values
                self.bootstrap_latency = 2.0
                self.bootstrap_tokens = float(self.avg_tokens)
                self._bootstrap_completed = True
        else:
            # No clusters available for bootstrap - use static fallback
            self.bootstrap_latency = 2.0
            self.bootstrap_tokens = float(self.avg_tokens)
            self._bootstrap_completed = True
        
        # Update rate limiting with bootstrap data
        await self._update_rate_limiting_with_bootstrap()

    async def _update_rate_limiting_with_bootstrap(self):
        """Update rate limiting components using bootstrap measurement data"""
        if not self._bootstrap_completed:
            return

        rate_limits = get_openai_rate_limits(self.config.model)

        # Recalculate arrival_rate using bootstrap-measured tokens
        arrival_rate = min(
            rate_limits.requests_per_minute * self.processing_config.rate_limit_headroom / 60,
            rate_limits.tokens_per_minute * self.processing_config.rate_limit_headroom / self.avg_tokens / 60
        )
        
        # Update AsyncLimiter with new arrival rate
        if arrival_rate < 1:
            self.rate_limiter = AsyncLimiter(1, time_period=1/arrival_rate)
        else:
            self.rate_limiter = AsyncLimiter(int(arrival_rate), time_period=1.0)
        
        # Calculate optimal concurrency using Little's Law with bootstrap-measured latency
        optimal_concurrency = compute_optimal_concurrency(
            ApiLimits(rate_limits.tokens_per_minute, rate_limits.requests_per_minute),
            self.bootstrap_latency,
            self.avg_tokens,
            self.processing_config,
            cap=self.config.async_concurrency_limit,
            headroom=self.processing_config.rate_limit_headroom
        )
        
        # Update concurrency semaphore
        self.concurrency_semaphore = asyncio.Semaphore(min(len(self.cluster_results), max(optimal_concurrency, 100)))
        
        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line(f"Rate limiting updated - arrival rate: {arrival_rate:.2f}/s, concurrency: {optimal_concurrency}")

    def _calculate_avg_tokens(self) -> int:
        """Calculate average token count for requests (following qualityFilter.py pattern)"""
        # Sample a few clusters to estimate average token usage
        sample_size = min(5, len(self.cluster_results))
        if sample_size == 0:
            return 400  # Default estimate for code generation tasks
        
        total_tokens = 0
        for i in range(sample_size):
            cluster_result = self.cluster_results[i]
            ideas_list = cluster_result.response_ideas or []
            ideas_text = "\n".join([f"- {idea}" for idea in ideas_list[:10]])  # Sample first 10 ideas
            
            # Estimate tokens for typical cluster summary prompt
            sample_prompt = CLUSTER_SUMMARY_PROMPT.format(
                cluster_id="sample",
                survey_question=self.var_lab or "sample question",
                language=DEFAULT_LANGUAGE,
                cluster_text=ideas_text
            )
            total_tokens += len(self.encoding.encode(sample_prompt))
        
        avg_input = total_tokens / sample_size
        # Assume 20% output ratio for code generation (more complex than quality filtering)
        return int(avg_input * 1.2)

    def estimate_tokens(self, prompt: str) -> int:
        """Estimate total tokens using adaptive strategy (from qualityFilter.py)"""
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
        
        # Output estimation: 20% of input for reasoning models, then average of first 5 responses
        if len(self.output_token_history) < 5:
            # Use 20% of input as estimate (higher for reasoning models)
            estimated_output = int(estimated_input * 0.20)
        else:
            # Use average of first 5 actual outputs
            avg_output = sum(self.output_token_history) / len(self.output_token_history)
            estimated_output = int(avg_output)
        
        # Ensure we don't exceed max_tokens if configured
        if hasattr(self.config, 'max_tokens') and self.config.max_tokens:
            estimated_output = min(self.config.max_tokens, estimated_output)
        
        total_estimate = estimated_input + estimated_output
        
        return total_estimate

    def get_token_bucket_status(self) -> dict:
        """Get current token bucket status with utilization calculation"""
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
            'available_tokens': int(self.tpm_bucket.available),
            'capacity': int(self.tpm_bucket.tpm),
            'available_pct': available_pct,
            'utilization_pct': real_utilization_pct,
            'low_tokens': self.tpm_bucket.available < self.tpm_bucket.tpm * 0.2
        }
    
    def get_token_estimation_stats(self) -> dict:
        """Get token estimation accuracy statistics"""
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

    def update_token_usage(self, estimated_tokens: int, actual_usage: Dict):
        """Update token history with actual usage data for progressive learning"""
        if not actual_usage:
            return
            
        # Extract token counts
        actual_input = actual_usage.get('prompt_tokens', 0)
        actual_output = actual_usage.get('completion_tokens', 0)
        actual_total = actual_usage.get('total_tokens', actual_input + actual_output)
        
        # Update output token history (first 5 samples)
        if len(self.output_token_history) < 5 and actual_output > 0:
            self.output_token_history.append(actual_output)
        
        # Track actual total tokens
        if actual_total > 0:
            self.actual_total_tokens.append(actual_total)
        
        # Calculate and track estimation error
        if estimated_tokens > 0 and actual_total > 0:
            error = abs(estimated_tokens - actual_total) / actual_total
            self.estimation_errors.append(error)

    async def probe_call_theme_extraction(self, cluster_data: Dict) -> Dict:
        """Probe call with EXACT same structure as production for accurate bootstrap measurement"""
        # Extract sample ideas for probe - same logic as production
        ideas = cluster_data.get('ideas', ['sample idea'])[:10]  # Limit to 10 ideas for probe
        ideas_text = "\n".join([f"- {idea}" for idea in ideas])
        
        # Build prompt - EXACT same as production
        prompt = CLUSTER_SUMMARY_PROMPT.format(
            cluster_id=cluster_data.get('cluster_id', 'probe'),
            survey_question=self.var_lab,
            language=DEFAULT_LANGUAGE,
            cluster_text=ideas_text
        )
        
        # Use EXACT same adaptive timeout as production
        adaptive_timeout = self._get_adaptive_timeout()
        
        try:
            # Use EXACT same wrapper as production with structured output
            response = await async_responses_create_with_json_retry(
                model=self.model_config.get_model_for_stage('theme_extraction'),  # Same model selection
                prompt=prompt,
                response_model=ClusterSummaryOutput,  # Same structured output
                reasoning_effort=self.model_config.get_reasoning_effort_for_stage('theme_extraction'),
                text_verbosity=self.model_config.get_text_verbosity_for_stage('theme_extraction'),
                semaphore=self.concurrency_semaphore,  # Same rate limiting
                rate_limiter=self.rate_limiter,        # Same rate limiting  
                tpm_bucket=self.tpm_bucket,           # Same rate limiting
                latency_tracker=self.latency_tracker, # Same timing tracking
                config=self.config,
                timeout=adaptive_timeout              # Same adaptive timeout
            )
            
            # Extract usage from structured response
            usage_data = getattr(response, "usage", None)
            if hasattr(response, 'response') and hasattr(response.response, 'usage'):
                usage_data = response.response.usage
            
            normalized_usage = normalize_usage(usage_data)
            
            # If no real usage data was available, normalized_usage will have zeros
            if normalized_usage["total_tokens"] == 0:
                # Fallback with realistic structured output token counts
                return {"prompt_tokens": 400, "completion_tokens": 150, "total_tokens": 550}
            
            return normalized_usage
            
        except Exception as e:
            # Fallback with realistic structured output token counts on error
            if self.verbose_reporter.enabled:
                self.verbose_reporter.warning(f"Bootstrap probe failed: {e}")
            return {"prompt_tokens": 400, "completion_tokens": 150, "total_tokens": 550}
   
    def _get_adaptive_timeout(self) -> float:
        """Get adaptive timeout based on latency tracking (following qualityFilter pattern)"""
        if self.latency_tracker.ema is None:
            return 30.0  # Default timeout
        
        # Use 95th percentile + 50% margin for timeouts
        if len(self.latency_tracker.values) >= 10:
            import numpy as np
            p95 = np.percentile(self.latency_tracker.values, 95)
            return min(max(p95 * 1.5, 15.0), 120.0)  # Between 15s and 120s
        else:
            # Use EMA + 50% margin for early stages
            return min(max(self.latency_tracker.ema * 1.5, 15.0), 60.0)  # Between 15s and 60s
    
    def _capture_prompt_params(self, cluster_id: Union[int, str], step: str, **kwargs):
        """Capture exact parameters used in prompt.format() for debugging/testing"""
        # Convert cluster_id to string for consistent dict key format
        key = str(cluster_id)
        if step == "step1":
            self.step1_inputs[key] = kwargs
        elif step == "step2":
            self.step2_inputs[key] = kwargs
        elif step == "step3":
            self.step3_inputs[key] = kwargs
        elif step == "step4":
            self.step4_inputs[key] = kwargs
    
    def _get_theme_id(self, theme_data) -> int:
        """Extract theme_id from theme data structure"""
        if hasattr(theme_data, 'root'):
            # Handle ClusterSummaryOutput structure
            cluster_summary_items = list(theme_data.root.values())
            if cluster_summary_items and cluster_summary_items[0].extracted_themes:
                theme = cluster_summary_items[0].extracted_themes[0]
                return theme.theme_id
        elif hasattr(theme_data, 'root') and theme_data.root:
            # Handle ThemeExtractionResult structure  
            theme = theme_data.root[0]
            return theme.theme_id
        return 1  # Default fallback
    
    def _find_closest_code(self, llm_code_name: str, available_codes: List[str], threshold: float = 0.8) -> Optional[str]:
        """Find closest matching code using fuzzy string matching"""
        if not llm_code_name or not available_codes:
            return None
        
        # Try exact match first (fastest)
        if llm_code_name in available_codes:
            return llm_code_name
        
        # Case-insensitive exact match
        for code in available_codes:
            if llm_code_name.lower() == code.lower():
                return code
        
        # Fuzzy matching with difflib
        matches = difflib.get_close_matches(
            llm_code_name, 
            available_codes, 
            n=1, 
            cutoff=threshold
        )
        
        return matches[0] if matches else None
    
    def _format_theme_for_prompt(self, theme_data) -> str:
        if hasattr(theme_data, 'root'):
            # Handle ClusterSummaryOutput structure
            cluster_summary_items = list(theme_data.root.values())
            if cluster_summary_items and cluster_summary_items[0].extracted_themes:
                theme = cluster_summary_items[0].extracted_themes[0]
                return f"{theme.theme_label}"
        elif hasattr(theme_data, 'root') and theme_data.root:
            # Handle ThemeExtractionResult structure  
            theme = theme_data.root[0]
            return f"Theme number: {theme.theme_id}\nTheme name: {theme.theme_label}\nTheme description: {theme.theme_clarification}"
        return "Unknown theme"
    
    def _get_theme_statement(self, theme_data) -> str:  
        """Safely get theme statement from theme data"""
        if hasattr(theme_data, 'root'):
            # Handle ClusterSummaryOutput structure
            cluster_summary_items = list(theme_data.root.values())
            if cluster_summary_items and cluster_summary_items[0].extracted_themes:
                return cluster_summary_items[0].extracted_themes[0].theme_clarification
        elif hasattr(theme_data, 'root') and theme_data.root:
            # Handle ThemeExtractionResult structure
            return theme_data.root[0].theme_clarification
        return "Unknown theme"
    
    def _get_theme_name(self, theme_data) -> str:  
        """Safely get theme name from theme data"""
        if hasattr(theme_data, 'root'):
            # Handle ClusterSummaryOutput structure
            cluster_summary_items = list(theme_data.root.values())
            if cluster_summary_items and cluster_summary_items[0].extracted_themes:
                theme = cluster_summary_items[0].extracted_themes[0]
                return f"{theme.theme_label}"
        elif hasattr(theme_data, 'root') and theme_data.root:
            # Handle ThemeExtractionResult structure  
            theme = theme_data.root[0]
            return f"{theme.theme_label}"
        return "Unknown theme"
    
    def _get_theme_description(self, theme_data) -> str:  
        """Safely get theme description from theme data"""
        if hasattr(theme_data, 'root'):
            # Handle ClusterSummaryOutput structure
            cluster_summary_items = list(theme_data.root.values())
            if cluster_summary_items and cluster_summary_items[0].extracted_themes:
                theme = cluster_summary_items[0].extracted_themes[0]
                return  f"{theme.theme_clarification}"
        elif hasattr(theme_data, 'root') and theme_data.root:
            # Handle ThemeExtractionResult structure  
            theme = theme_data.root[0]
            return f"{theme.theme_clarification}"
        return "No theme desciption"

    def _calculate_cosine_similarities(
        self,
        theme_embedding: np.ndarray,
        candidate_codes: List[Dict[str, str]],
        all_codes: List[Dict[str, str]],
        code_embeddings: List[np.ndarray]
    ) -> Dict[str, float]:
       
        similarities = {}

        for candidate in candidate_codes:
            # Find index of this code in the full codebook
            try:
                code_idx = next(
                    i for i, c in enumerate(all_codes)
                    if c['code'] == candidate['code']
                )
            except StopIteration:
                # Code not found (shouldn't happen, but handle gracefully)
                similarities[candidate['code']] = 0.0
                continue

            # Calculate cosine similarity
            similarity = cosine_similarity(
                theme_embedding.reshape(1, -1),
                code_embeddings[code_idx].reshape(1, -1)
            )[0][0]

            similarities[candidate['code']] = round(float(similarity), 3)

        return similarities

    def _format_codes_with_cosine(
        self,
        candidate_codes: List[Dict[str, str]],
        cosine_scores: Dict[str, float]
    ) -> str:
       
        formatted_lines = []

        for code in candidate_codes:
            code_label = code['code']
            cosine = cosine_scores.get(code_label, 0.0)
            line = f"- {code_label} (cosine: {cosine:.2f})"
            formatted_lines.append(line)

        return "\n".join(formatted_lines)


    def extract_cluster_data(self) -> Dict[Union[int, str], Dict[str, Any]]:
        """Extract cluster data from ClusterModel objects using expanded_cluster when available"""
        clusters = {}
        
        for result in self.cluster_results:
            ideas_list = result.response_ideas or []
            
            for idea in ideas_list:
                # Use expanded_cluster if available, otherwise fall back to initial_cluster
                cluster_id = idea.expanded_cluster if idea.expanded_cluster is not None else str(idea.initial_cluster) if idea.initial_cluster is not None else None
                
                if cluster_id is not None and cluster_id != "-1":
                    # Create cluster entry if it doesn't exist
                    if cluster_id not in clusters:
                        clusters[cluster_id] = {
                            'cluster_id': cluster_id,
                            'ideas': [],
                            'embeddings': [],
                            'respondent_ids': []
                        }
                    
                    # Add idea data - store the full object to preserve embeddings
                    clusters[cluster_id]['ideas'].append(idea)
                    clusters[cluster_id]['respondent_ids'].append(idea.idea_id)  # Using idea_id as respondent identifier
                    
                    # Add embedding if available (kept for backward compatibility)
                    if hasattr(idea, 'idea_embedding') and idea.idea_embedding is not None:
                        clusters[cluster_id]['embeddings'].append(idea.idea_embedding)
        
        # Filter out empty clusters
        return {cid: cdata for cid, cdata in clusters.items() if len(cdata['embeddings']) > 0}
    
    def _format_theme_only_results(self, themes: Dict[int, ClusterSummaryOutput]) -> List[Dict[str, Any]]:
        """Format theme extraction results for early return without 4-prompt chain processing"""
        results = []
        for cluster_id, theme_output in themes.items():
            # Extract theme information from ClusterSummaryOutput
            cluster_summary_items = list(theme_output.root.values())
            if cluster_summary_items and cluster_summary_items[0].extracted_themes:
                for theme in cluster_summary_items[0].extracted_themes:
                    results.append({
                        'cluster_id': cluster_id,
                        'theme_id': theme.theme_id,
                        'theme_label': theme.theme_label,
                        'theme_clarification': theme.theme_clarification,
                        'stage': 'theme_extraction_only'
                    })
        return results
    
    async def extract_themes(self, clusters: Dict[int, Dict[str, Any]]) -> Dict[int, ClusterSummaryOutput]:
        """Stage 1: Extract themes from all clusters using queue-worker pattern with proper rate limiting"""
        if not clusters:
            return {}
        
        self.verbose_reporter.step_start("Theme Extraction")
        self.verbose_reporter.stat_line(f"Processing {len(clusters)} clusters")
        
        # Get rate limits
        rate_limits = get_openai_rate_limits(self.config.model)

        # Calculate number of workers based on rate limits and latency
        rpm_throughput = rate_limits.requests_per_minute * self.processing_config.rate_limit_headroom / 60
        tpm_throughput = rate_limits.tokens_per_minute * self.processing_config.rate_limit_headroom / self.avg_tokens / 60
        expected_throughput = min(rpm_throughput, tpm_throughput)
        
        # Get average latency for worker calculation
        avg_latency_s = self.latency_tracker.get_avg_latency()
        num_workers = min(200, max(50, int(expected_throughput * avg_latency_s * 2.0)))
        
        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line("Theme extraction setup:")
            self.verbose_reporter.stat_line(f"- Workers: {num_workers} concurrent subroutines")
            self.verbose_reporter.stat_line(f"- Semaphore limit: {self.concurrency_semaphore._value} (API calls in flight)")
            self.verbose_reporter.stat_line(f"- Rate limits: {rpm_throughput:.1f} req/s, {tpm_throughput:.1f} tok/s")
        
        # Create queue and results dict
        queue = asyncio.Queue()
        theme_results = {}
        failed_clusters = []
        
        # Add tasks to queue
        for cluster_id, cluster_data in clusters.items():
            await queue.put({
                'cluster_id': cluster_id,
                'ideas': cluster_data['ideas']
            })
        
        # Create worker tasks
        async def worker():
            while True:
                try:
                    task = await queue.get()
                    if task is None:  # Sentinel value
                        break
                    
                    try:
                        result = await self._extract_single_theme(task['cluster_id'], task['ideas'])
                        theme_results[task['cluster_id']] = result
                    except Exception as e:
                        # Sanitize exception message for Windows console
                        error_msg = str(e).replace('🤖', '[BOT]').replace('\uFE0F', '')
                        self.verbose_reporter.error(f"Theme extraction failed for cluster {task['cluster_id']}: {error_msg}")
                        failed_clusters.append(task['cluster_id'])
                    finally:
                        queue.task_done()
                        
                except Exception as e:
                    logger.error(f"Worker error in theme extraction: {e}")
                    break
        
        # Start workers
        workers = []
        for _ in range(num_workers):
            w = asyncio.create_task(worker())
            workers.append(w)
        
        # Progress monitoring with diagnostics
        start_time = time.time()
        last_report = start_time
        last_diagnostics = start_time
        initial_queue_size = queue.qsize()
        
        while not queue.empty():
            await asyncio.sleep(1)
            now = time.time()
            
            # Progress report every 5s
            if now - last_report >= 5:
                completed = initial_queue_size - queue.qsize()
                remaining = queue.qsize()
                elapsed = now - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                
                if self.verbose_reporter.enabled:
                    self.verbose_reporter.stat_line(
                        f"Progress: {completed}/{initial_queue_size} ({completed/initial_queue_size*100:.1f}%), "
                        f"Rate: {rate:.1f}/s, Queue: {remaining}"
                    )
                last_report = now
            
            # Diagnostic report every 30s (if verbose)
            if self.verbose_reporter.enabled and now - last_diagnostics >= 30:
                # Token bucket diagnostics
                bucket_status = self.get_token_bucket_status()
                if bucket_status['low_tokens']:
                    self.verbose_reporter.stat_line(
                        f"⚠️ Token bucket low: {bucket_status['available_tokens']:,} tokens "
                        f"({bucket_status['utilization_pct']:.1f}% utilized)"
                    )
                
                # Token estimation diagnostics  
                token_stats = self.get_token_estimation_stats()
                if token_stats['status'] == 'learning' and token_stats['samples'] >= 5:
                    self.verbose_reporter.stat_line(
                        f"Token estimation: {token_stats['avg_estimation_error']:.0f} avg error, "
                        f"Input: {token_stats['avg_input_tokens']:.0f} avg ({token_stats['input_samples']}/3), "
                        f"Output: {token_stats['avg_output_tokens']:.0f} avg ({token_stats['output_samples']}/5)"
                    )
                
                # Latency tracking diagnostics
                if len(self.latency_tracker.values) >= 10:
                    p95_latency = np.percentile(list(self.latency_tracker.values), 95)
                    self.verbose_reporter.stat_line(
                        f"Latency: {self.latency_tracker.get_avg_latency():.1f}s avg, {p95_latency:.1f}s P95"
                    )
                
                last_diagnostics = now
        
        # Wait for all tasks to complete
        await queue.join()
        
        # Send sentinel values to stop workers
        for _ in workers:
            await queue.put(None)
        
        # Wait for workers to finish
        await asyncio.gather(*workers)
        
        self.verbose_reporter.stat_line(f"Extracted {len(theme_results)} themes successfully")
        if failed_clusters:
            self.verbose_reporter.stat_line(f"Failed clusters: {len(failed_clusters)}")
        
        # Report sampling summary
        if self.sampling_stats['clusters_sampled'] > 0:
            reduction_ratio = (self.sampling_stats['total_original_ideas'] - self.sampling_stats['total_sampled_ideas']) / self.sampling_stats['total_original_ideas'] * 100
            self.verbose_reporter.stat_line(f"Idea Sampling Summary: {self.sampling_stats['clusters_sampled']}/{self.sampling_stats['clusters_processed']} clusters sampled, {self.sampling_stats['total_original_ideas']}→{self.sampling_stats['total_sampled_ideas']} ideas ({reduction_ratio:.1f}% reduction)")
        else:
            self.verbose_reporter.stat_line(f"Idea Sampling: No large clusters found, all {self.sampling_stats['clusters_processed']} clusters used complete idea sets")
            
        self.verbose_reporter.step_complete("Theme Extraction")
        return theme_results
    
    def expand_multi_theme_clusters(self, themes: Dict[Union[int, str], ClusterSummaryOutput], clusters: Dict[Union[int, str], Dict[str, Any]]) -> Tuple[Dict[str, ClusterSummaryOutput], Dict[str, Dict[str, Any]], Dict[int, List[str]]]:
        """Expand multi-theme clusters into sub-clusters for independent processing
        Returns: (expanded_themes, expanded_clusters, multi_theme_mapping)
        """
        self.verbose_reporter.step_start("Multi-Theme Cluster Expansion")
        
        expanded_themes = {}
        expanded_clusters = {}
        multi_theme_mapping = {}  # Maps original cluster_id to list of sub_cluster_ids
        
        # Also expand step1_summaries and step1_inputs to match the new sub-cluster structure
        expanded_step1_summaries = {}
        expanded_step1_inputs = {}
        

        for cluster_id, theme_data in themes.items():
            # Skip already-expanded clusters
            if isinstance(cluster_id, str) and '-' in str(cluster_id):
                # This is already a sub-cluster, add it as-is
                string_cluster_id = str(cluster_id)
                expanded_themes[string_cluster_id] = theme_data
                
                # Handle cluster data if available
                if cluster_id in clusters:
                    expanded_clusters[string_cluster_id] = clusters[cluster_id].copy()
                
                # Handle step1_summaries for consistency (same as single-theme logic)
                cluster_summary_items = list(theme_data.root.values())
                if cluster_summary_items and cluster_summary_items[0].extracted_themes:
                    theme_item = cluster_summary_items[0].extracted_themes[0]
                    expanded_step1_summaries[string_cluster_id] = {
                        'analysis': cluster_summary_items[0].analysis,
                        'cluster_summary': theme_item.theme_clarification,
                        'themes': cluster_summary_items[0].extracted_themes,
                        'theme_id': theme_item.theme_id,
                        'theme_label': theme_item.theme_label,
                        'theme_description': theme_item.theme_clarification
                    }
                
                # Handle step1_inputs
                if str(cluster_id) in self.step1_inputs:
                    expanded_step1_inputs[string_cluster_id] = self.step1_inputs[str(cluster_id)]
                
                continue
            # theme_data is a ClusterSummaryOutput, .root is a dict with cluster_id as key
            # Get the ClusterSummaryItem and check its extracted_themes
            cluster_summary_items = list(theme_data.root.values())
            if cluster_summary_items and len(cluster_summary_items[0].extracted_themes) > 1:
                # Multi-theme cluster: create sub-clusters
                extracted_themes = cluster_summary_items[0].extracted_themes
                self.verbose_reporter.stat_line(f"Expanding cluster {cluster_id} into {len(extracted_themes)} sub-clusters")
                
                sub_cluster_ids = []
                for i, theme_item in enumerate(extracted_themes, 1):
                    sub_cluster_id = f"{cluster_id}-{i}"
                    sub_cluster_ids.append(sub_cluster_id)
                    original_analysis = cluster_summary_items[0].analysis
                    cluster_summary_item = ClusterSummaryItem(
                        analysis=original_analysis,
                        extracted_themes=[theme_item]
                    )
                    # Create ClusterSummaryOutput with proper structure
                    single_theme_data = ClusterSummaryOutput(root={sub_cluster_id: cluster_summary_item})
                    expanded_themes[sub_cluster_id] = single_theme_data
                    
                    # Temporarily duplicate cluster data - will be redistributed later
                    if cluster_id in clusters:
                        expanded_clusters[sub_cluster_id] = clusters[cluster_id].copy()
                    
                    # Create step1_summary for this sub-cluster with only its single theme (matching single-theme structure)
                    expanded_step1_summaries[sub_cluster_id] = {
                        'analysis': cluster_summary_items[0].analysis,
                        'cluster_summary': theme_item.theme_clarification,
                        'themes': [theme_item],
                        'theme_id': theme_item.theme_id,
                        'theme_label': theme_item.theme_label,
                        'theme_description': theme_item.theme_clarification
                    } 
                    # Duplicate step1_inputs for each sub-cluster to maintain key alignment
                    if str(cluster_id) in self.step1_inputs:
                        expanded_step1_inputs[sub_cluster_id] = self.step1_inputs[str(cluster_id)].copy()
                
                # Track multi-theme mapping for later redistribution
                multi_theme_mapping[cluster_id] = sub_cluster_ids
                    
            else:
                # Single-theme cluster: keep as-is but convert to string ID for consistency
                string_cluster_id = str(cluster_id)
                expanded_themes[string_cluster_id] = theme_data
                
                if cluster_id in clusters:
                    expanded_clusters[string_cluster_id] = clusters[cluster_id].copy()
                
                # Also convert step1_summaries to string ID
                # Get the theme data from the single-theme cluster
                cluster_summary_items = list(theme_data.root.values())
                if cluster_summary_items and cluster_summary_items[0].extracted_themes:
                    theme_item = cluster_summary_items[0].extracted_themes[0]  # First (and only) theme
                    expanded_step1_summaries[string_cluster_id] = {
                        'analysis': cluster_summary_items[0].analysis,
                        'cluster_summary': theme_item.theme_clarification,
                        'themes': cluster_summary_items[0].extracted_themes,
                        'theme_id': theme_item.theme_id,
                        'theme_label': theme_item.theme_label,
                        'theme_description': theme_item.theme_clarification
                    }
                    
                # Also convert step1_inputs to string ID for consistency
                if str(cluster_id) in self.step1_inputs:
                    expanded_step1_inputs[string_cluster_id] = self.step1_inputs[str(cluster_id)]
        
        # Replace the original step1_summaries and step1_inputs with the expanded versions
        self.step1_summaries = expanded_step1_summaries
        self.step1_inputs = expanded_step1_inputs
        
        self.verbose_reporter.stat_line(f"Expanded {len(themes)} clusters into {len(expanded_themes)} processing units")
        if multi_theme_mapping:
            self.verbose_reporter.stat_line(f"Found {len(multi_theme_mapping)} multi-theme clusters for later redistribution")
        self.verbose_reporter.step_complete("Multi-Theme Cluster Expansion")
        
        return expanded_themes, expanded_clusters, multi_theme_mapping

    def _create_expanded_cluster_to_theme_mapping(self) -> Dict[str, str]:
        """
        Create mapping from expanded_cluster IDs to theme labels.

        Uses step1_summaries which contains theme extraction results where:
        - Single-theme clusters: cluster_id → one theme
        - Multi-theme clusters: sub-cluster IDs (e.g., "12-1", "12-2") → individual themes

        Returns:
            Dict mapping expanded_cluster ID (e.g., "12-1") to theme_label (e.g., "Customer Service")
        """
        mapping = {}
        for cluster_id, summary in self.step1_summaries.items():
            # cluster_id is already the expanded_cluster ID after expansion
            theme_label = summary.get('theme_label', '')
            if theme_label:
                mapping[str(cluster_id)] = theme_label

        return mapping

    async def redistribute_ideas_to_subthemes(self, original_cluster_id: int, sub_cluster_ids: List[str],  original_cluster_data: Dict, sub_themes: Dict[str, ClusterSummaryOutput],theme_embeddings: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """Redistribute ideas from original cluster to sub-clusters based on embedding similarity"""
        #self.verbose_reporter.step_start(f"Redistributing ideas for cluster {original_cluster_id}")
        
        # Initialize empty cluster data for each sub-cluster
        redistributed_clusters = {
            sub_id: {
                'cluster_id': sub_id,
                'ideas': [],
                'embeddings': [],
                'respondent_ids': []
            } for sub_id in sub_cluster_ids
        }
        
        # Track redistribution statistics
        redistribution_detail = {
            'original_cluster_id': original_cluster_id,
            'sub_clusters': sub_cluster_ids,
            'original_idea_count': len(original_cluster_data.get('ideas', [])),
            'redistribution': {},
            'similarity_scores': [],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Process each idea
        ideas = original_cluster_data.get('ideas', [])
        embeddings = original_cluster_data.get('embeddings', [])
        respondent_ids = original_cluster_data.get('respondent_ids', [])
        
        if not embeddings or len(embeddings) != len(ideas):
            self.verbose_reporter.warning(f"Missing or mismatched embeddings for cluster {original_cluster_id}, using duplication fallback")
            # Fallback: duplicate all ideas to all sub-clusters
            for sub_id in sub_cluster_ids:
                redistributed_clusters[sub_id] = original_cluster_data.copy()
            return redistributed_clusters
        
        # Calculate similarities and assign ideas
        for idx, (idea, idea_embedding, respondent_id) in enumerate(zip(ideas, embeddings, respondent_ids)):
            # Calculate similarity to each sub-theme
            similarities = []
            for sub_id in sub_cluster_ids:
                if sub_id in theme_embeddings:
                    theme_embedding = theme_embeddings[sub_id]
                    # Calculate cosine similarity
                    similarity = cosine_similarity(
                        idea_embedding.reshape(1, -1),
                        theme_embedding.reshape(1, -1)
                    )[0, 0]
                    similarities.append((sub_id, similarity))
                else:
                    similarities.append((sub_id, 0.0))
            
            # Find best matching sub-theme
            if similarities:
                best_sub_id, best_similarity = max(similarities, key=lambda x: x[1])
                
                # Assign idea to best matching sub-cluster
                redistributed_clusters[best_sub_id]['ideas'].append(idea)
                redistributed_clusters[best_sub_id]['embeddings'].append(idea_embedding)
                redistributed_clusters[best_sub_id]['respondent_ids'].append(respondent_id)
                
                # Track similarity scores for statistics
                redistribution_detail['similarity_scores'].append({
                    'idea_idx': idx,
                    'assigned_to': best_sub_id,
                    'similarity': float(best_similarity),
                    'all_similarities': {sub_id: float(sim) for sub_id, sim in similarities}
                })
        
        # Calculate final redistribution statistics
        for sub_id in sub_cluster_ids:
            count = len(redistributed_clusters[sub_id]['ideas'])
            avg_similarity = 0.0
            if count > 0:
                # Calculate average similarity for ideas assigned to this sub-cluster
                sub_similarities = [
                    score['similarity'] 
                    for score in redistribution_detail['similarity_scores'] 
                    if score['assigned_to'] == sub_id
                ]
                avg_similarity = np.mean(sub_similarities) if sub_similarities else 0.0
            
            redistribution_detail['redistribution'][sub_id] = {
                'count': count,
                'avg_similarity': float(avg_similarity)
            }
        
        # Log redistribution summary
        #counts = [redistribution_detail['redistribution'][sub_id]['count'] for sub_id in sub_cluster_ids]
        #self.verbose_reporter.stat_line(f"Redistributed cluster {original_cluster_id}: {redistribution_detail['original_idea_count']} ideas → {counts} across {len(sub_cluster_ids)} sub-themes")
        
        # Store detailed statistics
        self._redistribution_stats['clusters_redistributed'].append(original_cluster_id)
        self._redistribution_stats['redistribution_details'][str(original_cluster_id)] = redistribution_detail
        
        #self.verbose_reporter.step_complete(f"Idea redistribution for cluster {original_cluster_id}")
        
        return redistributed_clusters
    
    async def _update_cluster_models_with_redistribution(self, multi_theme_mapping: Dict[int, List[str]], original_clusters: Dict, themes: Dict[str, ClusterSummaryOutput],theme_embeddings: Dict[str, np.ndarray]):
        """Update ClusterModel objects with expanded_cluster and cluster_theme assignments based on similarity"""

        # Create expanded_cluster → theme_label mapping
        expanded_cluster_to_theme = self._create_expanded_cluster_to_theme_mapping()

        # For each multi-theme cluster
        for orig_cluster_id, sub_cluster_ids in multi_theme_mapping.items():
            if orig_cluster_id not in original_clusters:
                continue
                
            # Get original cluster data
            original_cluster_data = original_clusters[orig_cluster_id]
            
            # Redistribute ideas to get assignments
            redistributed_data = await self.redistribute_ideas_to_subthemes(
                orig_cluster_id,
                sub_cluster_ids,
                original_cluster_data,
                themes,
                theme_embeddings
            )
            
            # Create a mapping of idea_id to expanded_cluster
            idea_to_expanded_cluster = {}
            for sub_id, sub_data in redistributed_data.items():
                for respondent_id in sub_data['respondent_ids']:
                    idea_to_expanded_cluster[respondent_id] = sub_id
            
            
            # Update ClusterModel objects
            updated_ideas_count = 0
            #sample_idea_ids = []
            total_ideas_checked = 0
            matching_cluster_ideas = 0
            
            
            for result in self.cluster_results:
                if result.response_ideas:
                    for idea in result.response_ideas:
                        total_ideas_checked += 1
                        
                        
                        # Check if this idea belongs to the original cluster - ensure type consistency
                        if idea.initial_cluster == orig_cluster_id or str(idea.initial_cluster) == str(orig_cluster_id):
                            matching_cluster_ideas += 1
                            
                            # Find its expanded cluster assignment
                            if idea.idea_id in idea_to_expanded_cluster:
                                idea.expanded_cluster = idea_to_expanded_cluster[idea.idea_id]
                                updated_ideas_count += 1

                                # Set cluster_theme from mapping
                                if idea.expanded_cluster in expanded_cluster_to_theme:
                                    idea.cluster_theme = expanded_cluster_to_theme[idea.expanded_cluster]
            
        
        # For single-theme clusters, set expanded_cluster to string version of initial_cluster
        single_theme_updates = 0
        for result in self.cluster_results:
            if result.response_ideas:
                for idea in result.response_ideas:
                    if idea.expanded_cluster is None and idea.initial_cluster is not None:
                        # Single-theme cluster: expanded_cluster = str(initial_cluster)
                        # Check both integer and string forms for type consistency
                        if (idea.initial_cluster not in multi_theme_mapping and
                            str(idea.initial_cluster) not in multi_theme_mapping):
                            idea.expanded_cluster = str(idea.initial_cluster)
                            single_theme_updates += 1

                            # Set cluster_theme for single-theme clusters
                            if idea.expanded_cluster in expanded_cluster_to_theme:
                                idea.cluster_theme = expanded_cluster_to_theme[idea.expanded_cluster]
        
    #########################################################################################################
    # IDEA SAMPLING METHODS — UMAP(10D) + HDBSCAN (euclidean), noise excluded
    #########################################################################################################
    
    def _sample_representative_ideas(self, ideas: List, max_ideas: int = None) -> List[str]:
        
        """Return up to max_ideas that are balanced across sub-clusters (HDBSCAN) or
        the 'best' representatives by centroid similarity, depending on config.
    
        Behaviour:
          - If n <= max_ideas (or n <= 30), return all (no clustering).
          - Else:
              UMAP (10D, metric='cosine') -> HDBSCAN (euclidean).
              Exclude noise (-1). Allocate ∝ cluster size (stable rounding).
              Within each sub-cluster: random sample.  (idea_sampling_mode='balanced')
              Or pick global top-k by cosine-to-centroid. (idea_sampling_mode='best')
        """
        
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        import random
    
        # Use config value if not specified
        if max_ideas is None:
            max_ideas = self.config.max_ideas_per_cluster
    
        # Prepare inputs
        original_count = len(ideas)
        self.sampling_stats['clusters_processed'] += 1
        self.sampling_stats['total_original_ideas'] += original_count
    
        # Helper: normalize to texts + embeddings
        idea_texts: List[str] = []
        embeddings: List[np.ndarray] = []
    
        if ideas and isinstance(ideas[0], str):
            idea_texts = list(ideas)
            # no embeddings available in this branch
        else:
            for idea in ideas:
                txt = idea.idea if hasattr(idea, "idea") else str(idea)
                idea_texts.append(txt)
                if hasattr(idea, "idea_embedding") and idea.idea_embedding is not None:
                    embeddings.append(np.asarray(idea.idea_embedding, dtype=np.float32))
                else:
                    embeddings.append(None)
    
        n = len(idea_texts)
    
        # Early exit: small clusters — keep existing behaviour (no clustering/noise filtering)
        if n <= max_ideas or n <= 30:
            self.sampling_stats['clusters_sampled'] += 1
            self.sampling_stats['total_sampled_ideas'] += n
            if hasattr(self, 'verbose_reporter') and self.verbose_reporter.enabled:
                self.verbose_reporter.stat_line(f"Idea Sampling (Small/Direct): {n}→{n} (no clustering)")
            return idea_texts
    
        # If we cannot cluster (missing embeddings), use a simple spacing or random fallback
        have_dense_embeddings = all(e is not None for e in embeddings)
        if not have_dense_embeddings:
            k = min(max_ideas, n)
            sampled = random.sample(idea_texts, k)
            self.sampling_stats['clusters_sampled'] += 1
            self.sampling_stats['total_sampled_ideas'] += len(sampled)
            if hasattr(self, 'verbose_reporter') and self.verbose_reporter.enabled:
                self.verbose_reporter.stat_line(
                    f"Idea Sampling (Fallback: no embeddings): {n}→{len(sampled)} (random)"
                )
            return sampled
    
        emb = np.vstack(embeddings).astype(np.float32)
        L2_emb = normalize(emb, norm="l2", copy=False)
        
        reducer = umap.UMAP(n_components=10, n_neighbors=5, metric="cosine", random_state=42)
        emb_10 = reducer.fit_transform(L2_emb)
    
        # Heuristic: min_cluster_size grows sublinearly with n
        min_cluster_size = max(5, int(np.sqrt(n)))
        hdb = HDBSCAN(min_cluster_size=min_cluster_size,
                      min_samples=None,
                      metric="euclidean",
                      cluster_selection_method="eom",
                      allow_single_cluster=False)
        labels = hdb.fit_predict(emb_10)
    
        # Build clusters excluding noise (-1)
        clusters: Dict[int, List[int]] = {}
        for i, lbl in enumerate(labels):
            if lbl == -1:
                continue  # exclude noise completely
            clusters.setdefault(int(lbl), []).append(i)
    
        total_non_noise = sum(len(v) for v in clusters.values())
    
        # If HDBSCAN yielded only noise or a degenerate result, fall back to centroid top-k
        if total_non_noise == 0:
            centroid = emb.mean(axis=0, keepdims=True)
            sims = cosine_similarity(emb, centroid).ravel()
            top_idx = np.argsort(sims)[-max_ideas:][::-1]
            picked = [idea_texts[i] for i in top_idx]
            self.sampling_stats['clusters_sampled'] += 1
            self.sampling_stats['total_sampled_ideas'] += len(picked)
            if hasattr(self, 'verbose_reporter') and self.verbose_reporter.enabled:
                self.verbose_reporter.stat_line(
                    f"Idea Sampling (Fallback: all-noise): {n}→{len(picked)} (centroid top-k)"
                )
            return picked
    
        # Allocation ∝ cluster size (stable rounding; may allocate 0 if there are more clusters than budget)
        budget = min(max_ideas, total_non_noise)
    
        sizes = {cid: len(idxs) for cid, idxs in clusters.items()}
        total = float(total_non_noise)
    
        # base quota + fractional remainder for stable rounding
        raw = {cid: (sizes[cid] / total) * budget for cid in sizes}
        base = {cid: int(np.floor(raw[cid])) for cid in sizes}
        remainder = budget - sum(base.values())
    
        if remainder > 0:
            order = sorted(sizes.keys(), key=lambda c: (raw[c] - base[c], sizes[c]), reverse=True)
            for cid in order[:remainder]:
                base[cid] += 1
    
        allocation = base  # final per-cluster k
    
        # Sample within each cluster
        sampled_indices: List[int] = []
        for cid, idxs in clusters.items():
            k = min(len(idxs), allocation.get(cid, 0))
            if k > 0:
                sampled_indices.extend(random.sample(idxs, k))
    
        # Safety: cap to budget and map to texts
        sampled_indices = sampled_indices[:budget]
        sampled_texts = [idea_texts[i] for i in sampled_indices]
    
        # Stats + verbose
        self.sampling_stats['clusters_sampled'] += 1
        self.sampling_stats['total_sampled_ideas'] += len(sampled_texts)
    
        if hasattr(self, 'verbose_reporter') and self.verbose_reporter.enabled:
            alloc_str = ", ".join(f"{cid}:{allocation[cid]}" for cid in sorted(allocation))
            self.verbose_reporter.stat_line(
                f"Idea Sampling (Balanced/HDBSCAN): {n}→{len(sampled_texts)} "
                f"clusters={len(clusters)} (noise excluded), allocation: [{alloc_str}]"
            )
    
        return sampled_texts
    
    #########################################################################################################
    # Stage 1: Prompt Formatting & LLM Calling  for THEME EXTRACTION/CLUSTER SUMMARIES -  
    #########################################################################################################
    
    async def _extract_single_theme(self, cluster_id: Union[int, str], ideas: List[str]):
        """Extract theme for single cluster using instructor"""
        
        # Sample representative ideas if cluster is too large
        sampled_ideas = self._sample_representative_ideas(ideas)
        ideas_text = "\n".join([f"- {idea}" for idea in sampled_ideas])
        
        # # Prepare exact parameters for prompt
        params = {
            'cluster_id': str(cluster_id),  # Convert to string as prompt expects string
            'survey_question': self.var_lab,
            'language': DEFAULT_LANGUAGE,
            'cluster_text': ideas_text
        }
        
        prompt = CLUSTER_SUMMARY_PROMPT.format(**params)
        
        
        # Capture exact parameters used in prompt construction
        params_for_capture = {k: v for k, v in params.items() if k != 'cluster_id'} 
        self._capture_prompt_params(cluster_id, "step1", **params_for_capture)  
   
        
        # Capture prompt with prompt_printer if available
        if self.prompt_printer:
            self.prompt_printer.capture_prompt(
                step_name="theme_extraction",
                utility_name="codeGenerator",
                prompt_content=prompt,
                prompt_type="cluster_summary",
                metadata={
                    "cluster_id": cluster_id,
                    "model": self.config.model,
                    "ideas_count": len(ideas)
                }
            )
        
        try:
            # Use async wrapper with JSON retry logic and adaptive timeout
            adaptive_timeout = self._get_adaptive_timeout()
            
            # # Debug: About to make API call
            # if self.verbose_reporter.enabled:
            #     self.verbose_reporter.stat_line(f"DEBUG C{cluster_id}: Starting API call with timeout {adaptive_timeout:.1f}s")
            
            response = await async_responses_create_with_json_retry(
                model=self.model_config.get_model_for_stage('theme_extraction'),
                prompt=prompt,
                response_model=ClusterSummaryOutput,
                reasoning_effort=self.model_config.get_reasoning_effort_for_stage('theme_extraction'),
                text_verbosity=self.model_config.get_text_verbosity_for_stage('theme_extraction'),
                semaphore=self.concurrency_semaphore,  
                rate_limiter=self.rate_limiter,
                tpm_bucket=self.tpm_bucket,
                latency_tracker=self.latency_tracker,
                config=self.config,
                timeout=adaptive_timeout
            )
            
            # Handle ClusterSummaryOutput response from CLUSTER_SUMMARY_PROMPT
            # The response is a dictionary with cluster_id as key (RootModel.root was automatically extracted)
            if hasattr(response, '__await__'):
                self.verbose_reporter.error(f"Response is still a coroutine for cluster {cluster_id}: {type(response)}")
                return None
            
            # The response should be a dictionary (since async_responses_create_with_json_retry extracts .root)
            if isinstance(response, dict):
                # The response is already the root dictionary from ClusterSummaryOutput
                # Create a proper ClusterSummaryOutput object
                result = ClusterSummaryOutput(root=response)
                
                # Capture theme extraction results for transparency
                cluster_key = str(cluster_id)
                if cluster_key in response:
                    cluster_summary_item = response[cluster_key]
                    if cluster_summary_item.extracted_themes:
                        first_theme = cluster_summary_item.extracted_themes[0]
                        self.step1_summaries[cluster_id] = {
                            'analysis': cluster_summary_item.analysis,
                            'cluster_summary': first_theme.theme_clarification,
                            'themes': cluster_summary_item.extracted_themes,
                        }
                        
                return result
            else:
                self.verbose_reporter.error(f"Unexpected response type for cluster {cluster_id}: {type(response)}")
                return None
            
        except Exception as e:
            # Sanitize exception message for Windows console
            error_msg = str(e).replace('🤖', '[BOT]').replace('\uFE0F', '')
            self.verbose_reporter.error(f"Theme extraction failed for cluster {cluster_id}: {error_msg}")
            return None
        
    async def _measure_code_generation_tokens(self, clusters: Dict[int, Dict[str, Any]], themes: Dict[int, ClusterSummaryOutput]) -> Dict[str, float]:
        """Measure real token usage for all 3 code generation steps (like qualityFilter approach)"""
        self.verbose_reporter.step_start("Code Generation Token Measurement")
        
        # Sample first 5-10 clusters for measurement (balance accuracy vs speed)
        sample_size = min(8, len(clusters))
        sample_items = list(clusters.items())[:sample_size]
        
        token_measurements = {
            'candidate_selection': [],
            'code_generation': [], 
            'validation': []
        }
        
        # Get current codebook state for realistic measurements
        current_codes, version = await self.shared_codebook.get_current_snapshot()
        
        for cluster_id, cluster_data in sample_items:
            if cluster_id not in themes:
                continue
                
            theme_data = themes[cluster_id]
            #ideas_text = "\n".join([f"- {idea}" for idea in cluster_data['ideas']])
            
            # PROMPT2 Candidate Selection prompt
            codes_text = "\n".join([f"-{code['code']}" for code in current_codes[:5]])  # Limit like the real implementation
            
            # Get theme_id for the measurement prompt
            theme_id = self._get_theme_id(theme_data)
            
            candidate_prompt = CODING_DECISION_PROMPT.format(
                survey_question=self.var_lab,
                language=DEFAULT_LANGUAGE,
                theme_name=self._get_theme_name(theme_data),
                theme_description=self._get_theme_statement(theme_data),
                code_text=codes_text,
                theme_id=theme_id
            )
            candidate_tokens = len(self.encoding.encode(candidate_prompt)) + 200  # + completion estimate
            token_measurements['candidate_selection'].append(candidate_tokens)

            code_gen_prompt = CODE_CREATION_PROMPT.format(
                language=DEFAULT_LANGUAGE,
                survey_question=self.var_lab,
                theme_name=self._get_theme_name(theme_data),
                theme_description=self._get_theme_statement(theme_data),
                theme_id=theme_id,
                cluster_summary=self._get_theme_name(theme_data)
            )
            code_gen_tokens = len(self.encoding.encode(code_gen_prompt)) + 150  # + completion estimate
            token_measurements['code_generation'].append(code_gen_tokens)
            
            # PROMPT4 Validation prompt
            validation_codes_text = "\n".join([f"-{code['code']}" for code in current_codes[:5]])  # Limited like real implementation
            
            validation_prompt = VALIDATION_PROMPT.format(
                language=DEFAULT_LANGUAGE,
                survey_question=self.var_lab,
                theme_name=self._get_theme_name(theme_data),
                theme_description=self._get_theme_statement(theme_data),
                code_text=validation_codes_text,
                step3_recommendation='{"generated_code": {"theme_number": 1, "theme_name": "Example theme", "code_label": "Example code", "code_definition": "Example definition"}}',
                theme_id=theme_id,
                cluster_summary=self._get_theme_name(theme_data),
                source_code="Null"
            )
            validation_tokens = len(self.encoding.encode(validation_prompt)) + 100  # + completion estimate
            token_measurements['validation'].append(validation_tokens)
        
        # Calculate averages
        import statistics
        measured_averages = {}
        for step, token_list in token_measurements.items():
            if token_list:
                avg_tokens = statistics.mean(token_list)
                measured_averages[step] = avg_tokens
                self.verbose_reporter.stat_line(f"Measured {step} token usage: {avg_tokens:.0f} tokens/request (from {len(token_list)} samples)")
            else:
                # Fallback estimates
                fallback_tokens = {'candidate_selection': 1200, 'code_generation': 1000, 'validation': 900}
                measured_averages[step] = fallback_tokens[step]
        
        self.verbose_reporter.step_complete("Code Generation Token Measurement")
        return measured_averages
    
    async def process_batches_sequentially(self, dissimilarity_batches: List[List[str]], 
                                         clusters: Dict, themes: Dict, 
                                         theme_embeddings: Dict) -> List[Dict[str, Any]]:
        """Process Level 1 batches sequentially, Level 2 batches concurrently with staggering"""
        self.verbose_reporter.step_start("Two-Level Batch Processing")
        self.verbose_reporter.stat_line(f"Processing {len(dissimilarity_batches)} Level 1 batches sequentially")
        
        # Store theme embeddings for use in candidate selection
        self._theme_embeddings_cache = theme_embeddings
        
        # Measure token usage for code generation steps
        self.code_gen_token_measurements = await self._measure_code_generation_tokens(clusters, themes)
        
        # Calculate global rate limiting strategy
        composite_tokens = (
            self.code_gen_token_measurements.get('candidate_selection', 1200) +
            self.code_gen_token_measurements.get('code_generation', 1000) +
            self.code_gen_token_measurements.get('validation', 900)
        )
        
        total_clusters = sum(len(batch) for batch in dissimilarity_batches)
        self.verbose_reporter.stat_line(f"Total clusters to process: {total_clusters}")
        self.verbose_reporter.stat_line(f"Composite tokens per cluster: {composite_tokens}")
        
        all_results = []
        total_batches = len(dissimilarity_batches)
        
        for batch_idx, dissimilarity_batch in enumerate(dissimilarity_batches):
            self.verbose_reporter.step_start(f"Level 1 Batch {batch_idx + 1}/{total_batches}")
            self.verbose_reporter.stat_line(f"Processing {len(dissimilarity_batch)} clusters")
            
            # Create Level 2 sub-batches
            if len(dissimilarity_batch) > self.config.max_sub_batch_size:
                sub_batches = self.similarity_engine.create_sub_batches(dissimilarity_batch, self.config.max_sub_batch_size)
            else:
                sub_batches = [dissimilarity_batch]  # Single sub-batch
             
            # Process Level 2 sub-batches concurrently with staggering
            sub_batch_tasks = []
            for i, sub_batch in enumerate(sub_batches):
                stagger_delay = i * 0.5  # 500ms between sub-batch starts for smooth distribution
                task = self._process_sub_batch_with_stagger(
                    sub_batch, clusters, themes, stagger_delay
                )
                sub_batch_tasks.append(task)
            
            # Wait for ALL sub-batches to complete before next Level 1 batch
            batch_results = await asyncio.gather(*sub_batch_tasks)
            
            # Flatten and collect results
            for sub_batch_result in batch_results:
                all_results.extend(sub_batch_result)
            
            # Level 1 batch complete - codebook is updated
            version_info = await self.shared_codebook.get_version_info()
            self.verbose_reporter.stat_line(f"Codebook version: {version_info['version']}, total codes: {version_info['total_codes']}")
            self.verbose_reporter.step_complete(f"Level 1 Batch {batch_idx + 1} completed")
            # Next Level 1 batch starts immediately (if API limits allow)
        
        self.verbose_reporter.step_complete("Two-Level Batch Processing")
        
        # Process modification leaks recovery batch if any were collected
        if self.modification_leaks:
            if self.config.enable_concurrent_leak_recovery:
                recovery_results = await self._process_modification_leaks_batch_concurrent(clusters, themes, theme_embeddings)
            else:
                recovery_results = await self._process_modification_leak_recovery(clusters, themes, theme_embeddings)
            all_results.extend(recovery_results)
        
        return all_results
    
    async def _process_modification_leak_recovery(self, clusters: Dict, themes: Dict, theme_embeddings: Dict) -> List[Dict[str, Any]]:
        """Process modification leaks sequentially to avoid race conditions"""
        self.verbose_reporter.step_start("Modification Leak Recovery")
        self.verbose_reporter.stat_line(f"Processing {len(self.modification_leaks)} modification leaks sequentially")
        
        recovery_results = []
        recovery_stats = {'resolved': 0, 'failed': 0, 'changed_decision': 0}
        
        for leak_data in self.modification_leaks:
            cluster_id = leak_data['cluster_id']
            original_result = leak_data.get('full_result')
            
            if not original_result:
                self.verbose_reporter.error(f"C{cluster_id}: Recovery failed - missing original result data")
                recovery_stats['failed'] += 1
                continue
            
            try:
                self.verbose_reporter.stat_line(f"C{cluster_id}: Recovering modification leak - re-running candidate selection")
                
                # Re-run candidate selection with current codebook state
                cluster_data = clusters.get(cluster_id, {})
                theme_data = themes.get(cluster_id)
                
                if not cluster_data or not theme_data:
                    self.verbose_reporter.error(f"C{cluster_id}: Recovery failed - missing cluster or theme data")
                    recovery_stats['failed'] += 1
                    continue
                
                # Re-run the full processing pipeline for this cluster
                result = await self._process_single_cluster_comprehensive(cluster_id, cluster_data, theme_data, theme_embeddings.get(cluster_id))
                
                if result:
                    # Check if the new decision is different (due to codebook changes)
                    if result.get('candidate_selection') and hasattr(result['candidate_selection'], 'coding_decision'):
                        new_decision = result['candidate_selection'].coding_decision.decision.lower()
                        
                        if new_decision != 'modify':
                            self.verbose_reporter.stat_line(f"C{cluster_id}: Decision changed from MODIFY to {new_decision.upper()} after recovery")
                            recovery_stats['changed_decision'] += 1
                        else:
                            self.verbose_reporter.stat_line(f"C{cluster_id}: MODIFY decision maintained after recovery")
                            recovery_stats['resolved'] += 1
                    
                    recovery_results.append(result)
                    
                    # Apply codebook updates for this single cluster immediately
                    await self._merge_codebook_updates([result], None)
                    
                else:
                    recovery_stats['failed'] += 1
                    
            except Exception as e:
                self.verbose_reporter.error(f"C{cluster_id}: Recovery failed with error: {e}")
                recovery_stats['failed'] += 1
        
        # Report recovery results
        total_processed = sum(recovery_stats.values())
        if total_processed > 0:
            self.verbose_reporter.stat_line(f"Recovery summary: RESOLVED={recovery_stats['resolved']}, DECISION_CHANGED={recovery_stats['changed_decision']}, FAILED={recovery_stats['failed']}")
        
        # Clear modification leaks after processing
        self.modification_leaks.clear()
        
        self.verbose_reporter.step_complete("Modification Leak Recovery")
        return recovery_results
    
    async def _process_modification_leaks_batch_concurrent(self, clusters: Dict, themes: Dict, theme_embeddings: Dict) -> List[Dict[str, Any]]:
        """Process modification leaks concurrently using batch processing patterns"""
        self.verbose_reporter.step_start("Concurrent Modification Leak Recovery")
        self.verbose_reporter.stat_line(f"Processing {len(self.modification_leaks)} modification leaks in concurrent batches")
        
        if not self.modification_leaks:
            return []
        
        # Group modification leaks into batches for concurrent processing
        leak_batches = self._create_modification_leak_batches(self.modification_leaks)
        self.verbose_reporter.stat_line(f"Created {len(leak_batches)} batches for concurrent processing")
        
        all_recovery_results = []
        all_recovery_stats = {'resolved': 0, 'failed': 0, 'changed_decision': 0}
        
        # Process batches sequentially (level 1), each batch processed concurrently (level 2)
        for batch_idx, leak_batch in enumerate(leak_batches, 1):
            self.verbose_reporter.stat_line(f"[START] Recovery Batch {batch_idx}/{len(leak_batches)} - Processing {len(leak_batch)} leaks concurrently")
            
            # Process this batch of leaks concurrently
            batch_results, batch_stats = await self._process_leak_batch_concurrent(
                leak_batch, clusters, themes, theme_embeddings, batch_idx
            )
            
            # Accumulate results and stats
            all_recovery_results.extend(batch_results)
            for key in all_recovery_stats:
                all_recovery_stats[key] += batch_stats.get(key, 0)
                
            self.verbose_reporter.stat_line(f"[COMPLETE] Recovery Batch {batch_idx}/{len(leak_batches)} - Results: {len(batch_results)} recovered, Stats: {batch_stats}")
        
        # Apply all successful recovery results atomically to the codebook
        if all_recovery_results:
            self.verbose_reporter.stat_line(f"Applying {len(all_recovery_results)} recovery results to codebook atomically")
            await self._merge_codebook_updates(all_recovery_results, None)
        
        # Report overall recovery results
        total_processed = sum(all_recovery_stats.values())
        if total_processed > 0:
            self.verbose_reporter.stat_line(f"Concurrent recovery summary: RESOLVED={all_recovery_stats['resolved']}, DECISION_CHANGED={all_recovery_stats['changed_decision']}, FAILED={all_recovery_stats['failed']}")
        
        # Clear modification leaks after processing
        self.modification_leaks.clear()
        
        self.verbose_reporter.step_complete("Concurrent Modification Leak Recovery")
        return all_recovery_results
    
    def _create_modification_leak_batches(self, leaks: List[Dict]) -> List[List[Dict]]:
        """Create batches of modification leaks for concurrent processing"""
        # Configurable batching strategy aligned with existing batch processing
        batch_size = min(self.config.modification_leak_batch_size, len(leaks))
        
        batches = []
        for i in range(0, len(leaks), batch_size):
            batch = leaks[i:i + batch_size]
            batches.append(batch)
        
        return batches
    
    async def _process_leak_batch_concurrent(self, leak_batch: List[Dict], clusters: Dict, themes: Dict, theme_embeddings: Dict, batch_idx: int) -> Tuple[List[Dict], Dict]:
        """Process a single batch of modification leaks concurrently"""
        # Use existing concurrency control patterns
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)  # Reuse existing config
        
        async def process_single_leak(leak_data: Dict) -> Optional[Dict]:
            async with semaphore:
                cluster_id = leak_data['cluster_id']
                
                try:
                    # Validate leak data
                    original_result = leak_data.get('full_result')
                    if not original_result:
                        self.verbose_reporter.error(f"C{cluster_id}: Recovery failed - missing original result data")
                        return None
                    
                    # Get cluster and theme data
                    cluster_data = clusters.get(cluster_id, {})
                    theme_data = themes.get(cluster_id)
                    
                    if not cluster_data or not theme_data:
                        self.verbose_reporter.error(f"C{cluster_id}: Recovery failed - missing cluster or theme data")
                        return None
                    
                    if self.verbose_detailed:
                        self.verbose_reporter.stat_line(f"C{cluster_id}: Batch {batch_idx} - Recovering modification leak concurrently")
                    
                    # Re-run the full processing pipeline for this cluster
                    result = await self._process_single_cluster_comprehensive(cluster_id, cluster_data, theme_data, theme_embeddings.get(cluster_id))
                    
                    return result
                    
                except Exception as e:
                    self.verbose_reporter.error(f"C{cluster_id}: Batch {batch_idx} - Recovery failed with error: {e}")
                    return None
        
        # Process all leaks in this batch concurrently
        leak_tasks = [process_single_leak(leak) for leak in leak_batch]
        leak_results = await asyncio.gather(*leak_tasks, return_exceptions=True)
        
        # Collect successful results and compute stats
        batch_results = []
        batch_stats = {'resolved': 0, 'failed': 0, 'changed_decision': 0}
        
        for i, result in enumerate(leak_results):
            cluster_id = leak_batch[i]['cluster_id']
            
            if isinstance(result, Exception):
                self.verbose_reporter.error(f"C{cluster_id}: Batch {batch_idx} - Recovery failed with exception: {result}")
                batch_stats['failed'] += 1
            elif result is None:
                batch_stats['failed'] += 1
            else:
                # Check if the new decision is different (due to codebook changes)
                if result.get('candidate_selection') and hasattr(result['candidate_selection'], 'coding_decision'):
                    new_decision = result['candidate_selection'].coding_decision.decision.lower()
                    
                    if new_decision != 'modify':
                        if self.verbose_detailed:
                            self.verbose_reporter.stat_line(f"C{cluster_id}: Batch {batch_idx} - Decision changed from MODIFY to {new_decision.upper()}")
                        batch_stats['changed_decision'] += 1
                    else:
                        if self.verbose_detailed:
                            self.verbose_reporter.stat_line(f"C{cluster_id}: Batch {batch_idx} - MODIFY decision maintained")
                        batch_stats['resolved'] += 1
                
                batch_results.append(result)
        
        return batch_results, batch_stats
    
    async def _process_sub_batch_with_stagger(self, sub_batch: List[str], clusters: Dict, themes: Dict, 
                                            stagger_delay: float) -> List[Dict[str, Any]]:
        """Process sub-batch with optimized concurrency, rate limiting, and bootstrap measurement"""
        
        # Apply stagger delay for smooth distribution
        if stagger_delay > 0:
            await asyncio.sleep(stagger_delay)
        
        if not sub_batch:
            return []
        
        # Use shared bootstrap data for cluster processing chain (Prompts 2-4)
        limits = get_openai_rate_limits(self.config.model)

        # Ensure bootstrap measurement has been completed
        if not self._bootstrap_completed:
            self.verbose_reporter.warning("Bootstrap measurement not completed for cluster processing - using fallback")

        # Calculate stage-specific optimal concurrency for the 3-prompt chain
        # Use bootstrap data but adjust for the complexity of the 3-prompt chain
        chain_latency = self.bootstrap_latency * 3  # Approximate latency for 3-prompt sequence
        api_limits = ApiLimits(limits.tokens_per_minute, limits.requests_per_minute)
        Little = compute_optimal_concurrency(api_limits, chain_latency, self.avg_tokens, self.processing_config, cap=self.processing_config.concurrency_cap_default, min_conc=self.processing_config.concurrency_min_conservative)
        optimal = min(100, max(Little, 20))  # Constrained for complex chain processing

        # Create stage-specific rate limiting (adjusted for 3-prompt chain)
        arrival_rate = min(
            limits.requests_per_minute * self.processing_config.rate_limit_headroom / 60 / 3,  # Divided by 3 for 3-prompt chain
            limits.tokens_per_minute * self.processing_config.rate_limit_headroom / self.avg_tokens / 60
        )
        
        limiter = AsyncLimiter(arrival_rate, 1)
        semaphore = asyncio.Semaphore(optimal)
        tpm_bucket = self.tpm_bucket  # Use shared TPM bucket
        
        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line(f"Cluster chain setup: {optimal} concurrent, adjusted latency {chain_latency:.3f}s (3x bootstrap)")
            self.verbose_reporter.stat_line(f"Chain arrival rate: {arrival_rate:.1f}/s (RPM/3 for 3-prompt sequence)")
        
        # Get current codebook snapshot for consistent processing
        codebook_snapshot, base_version = await self.shared_codebook.get_current_snapshot()
        
        # PRE-COMPUTE: Ensure codebook embeddings exist for this version before cluster processing
        # This prevents multiple clusters from generating the same embeddings simultaneously
        cached_embeddings = await self.shared_codebook.get_embeddings_for_version(base_version)
        if cached_embeddings is None and codebook_snapshot:
            code_texts = [f"{code['code']}" for code in codebook_snapshot]
            try:
                code_embeddings = await self.similarity_engine._embed_openai_batch(code_texts)
                await self.shared_codebook.cache_embeddings(base_version, code_embeddings)
                self.verbose_reporter.stat_line(f"Cached embeddings for version {base_version}")
            except Exception as e:
                self.verbose_reporter.error(f"Failed to pre-compute embeddings for version {base_version}: {e}")
        
        # Optimized cluster processing with rate limiting
        async def process_cluster_with_limits(cluster_id):
            async with semaphore, limiter:
                # Build composite prompt for token estimation (3-prompt chain)
                # Note: For Stage 2, we estimate based on the most token-heavy prompt (typically candidate selection)
                if cluster_id in clusters and cluster_id in themes:
                    #cluster_data = clusters[cluster_id]
                    theme_data = themes[cluster_id]
                    
                    # Estimate using candidate selection prompt as representative
                    # Use proper template parameters that match CODING_DECISION_PROMPT
                    theme_id = self._get_theme_id(theme_data)
                    sample_prompt = CODING_DECISION_PROMPT.format(
                        survey_question=self.var_lab,
                        language=DEFAULT_LANGUAGE,
                        theme_name=self._get_theme_name(theme_data),
                        theme_description=self._get_theme_statement(theme_data),
                        code_text="",  # Empty for estimation
                        theme_id=theme_id
                    )
                    
                    # Use progressive estimation for the chain (multiply by 3 for 3 prompts)
                    estimated_tokens = self.estimate_tokens(sample_prompt) * 3
                else:
                    # Fallback to average if cluster data not available
                    estimated_tokens = int(self.avg_tokens)
                
                await tpm_bucket.wait_and_acquire(estimated_tokens)
                
                # Track latency for the full chain
                start_time = time.perf_counter()
                try:
                    result = await self._process_single_cluster(
                        cluster_id, clusters, themes, codebook_snapshot, base_version
                    )
                    # Track latency for continuous optimization
                    latency = time.perf_counter() - start_time
                    self.latency_tracker.add(latency)
                    return result
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    self.verbose_reporter.error(f"Cluster processing failed for {cluster_id}: {e}")
                    self.verbose_reporter.error(f"Full traceback: {tb}")
                    return None
        
        # Create optimized tasks
        cluster_tasks = [
            process_cluster_with_limits(cluster_id) 
            for cluster_id in sub_batch
        ]
        
        # Process with optimized concurrency and gather results  
        results = []
        completed_results = await asyncio.gather(*cluster_tasks, return_exceptions=True)
        
        for result in completed_results:
            if isinstance(result, Exception):
                import traceback
                tb = ''.join(traceback.format_exception(type(result), result, result.__traceback__))
                self.verbose_reporter.error(f"Cluster task failed: {result}")
                self.verbose_reporter.error(f"Exception type: {type(result).__name__}")
                self.verbose_reporter.error(f"Full exception traceback: {tb}")
                continue
            if result is not None:
                results.append(result)
        
        # Update SharedCodebook with any new codes from this sub-batch
        if results:
            await self._merge_codebook_updates(results, base_version)
            
            # POST-PROCESS: Generate embeddings for any new codes added during batch processing
            updated_codes, new_version = await self.shared_codebook.get_current_snapshot()
            if new_version > base_version:
                # Codebook was updated during processing, ensure embeddings exist for new version
                cached_embeddings = await self.shared_codebook.get_embeddings_for_version(new_version)
                if cached_embeddings is None:
                    # Generate embeddings for all codes in the updated codebook
                    if self.verbose_detailed: 
                        self.verbose_reporter.stat_line(f"Post-processing: Generating embeddings for updated codebook (version {new_version})")
                    #code_texts = [f"{code['code']}: {code['definition']}" for code in updated_codes]
                    code_texts = [f"{code['code']}" for code in updated_codes]
                    try:
                        code_embeddings = await self.similarity_engine._embed_openai_batch(code_texts)
                        await self.shared_codebook.cache_embeddings(new_version, code_embeddings)
                        self.verbose_reporter.stat_line(f"Cached embeddings for updated version {new_version}")
                    except Exception as e:
                        self.verbose_reporter.error(f"Failed to generate embeddings for updated codebook (version {new_version}): {e}")
        
        return results
    
    async def _process_single_cluster_comprehensive(self, cluster_id: str, cluster_data: Dict, theme_data, theme_embedding) -> Optional[Dict[str, Any]]:
        """Process single cluster comprehensively for modification leak recovery"""
        
        try:
            # Step 1: Get current codebook for candidate selection (ensures latest codes are visible)
            current_codes, _ = await self.shared_codebook.get_current_snapshot()
            nearest_codes = await self._find_nearest_codes_by_theme(cluster_id, theme_data, current_codes, k=5)
            
            # Step 2: Candidate selection - re-run with current codebook state
            candidate_selection = await self._select_candidate_codes(cluster_id, cluster_data, theme_data, nearest_codes)
            
            if not candidate_selection or not hasattr(candidate_selection, 'coding_decision'):
                self.verbose_reporter.error(f"C{cluster_id}: Recovery failed - no candidate selection result")
                return None
            
            decision = candidate_selection.coding_decision.decision.lower()
            
            # OPTIMIZATION: Skip Steps 3 & 4 for USE decisions
            if decision == "use":
                if self.verbose_detailed:
                    self.verbose_reporter.stat_line(f"C{cluster_id}: USE decision detected in recovery - skipping code generation and validation")
                
                # Return minimal result structure for USE decisions
                return {
                    'cluster_id': cluster_id,
                    'candidate_selection': candidate_selection,
                    'optimization': 'use_early_return'
                }
            
            # Step 3: Code generation for CREATE/MODIFY decisions
            code_generation = await self._generate_code(cluster_id, cluster_data, theme_data, candidate_selection)
            
            if not code_generation:
                self.verbose_reporter.error(f"C{cluster_id}: Recovery failed - no code generation result")
                return None
            
            # Step 4: Validation
            validation = await self._validate_code(cluster_id, cluster_data, theme_data, code_generation, nearest_codes)
            
            if not validation:
                self.verbose_reporter.error(f"C{cluster_id}: Recovery failed - no validation result")
                return None
            
            # Return complete result structure
            return {
                'cluster_id': cluster_id,
                'candidate_selection': candidate_selection,
                'code_generation': code_generation,
                'validation': validation
            }
            
        except Exception as e:
            self.verbose_reporter.error(f"C{cluster_id}: Recovery processing failed with error: {e}")
            return None
   
    async def _find_nearest_codes_by_theme(self, cluster_id: Union[int, str], theme_data, 
                                          current_codes: List[Dict[str, str]], k: int = 5) -> List[Dict[str, str]]:
        """Find k nearest codes to themes using cosine similarity - handles multiple themes per cluster"""
        if not current_codes:
            return []
        
        # Handle multiple themes per cluster (aggregate approach)
        all_nearest_codes = []
        
        # Check if theme_data has multiple themes
        if hasattr(theme_data, 'root') and isinstance(theme_data.root, list) and len(theme_data.root) > 1:
            # Multiple themes: get k codes for each theme and aggregate
            for theme_item in theme_data.root:
                theme_embedding = await self._get_theme_embedding_for_item(cluster_id, theme_item)
                if theme_embedding is not None:
                    nearest_codes = await self._get_nearest_codes_by_embedding(theme_embedding, current_codes, k)
                    all_nearest_codes.extend(nearest_codes)
        else:
            # Single theme: use cached embedding (backward compatibility)
            if not hasattr(self, '_theme_embeddings_cache'):
                self.verbose_reporter.error(f"No theme embeddings found for cluster {cluster_id}")
                return []
                
            theme_embedding = self._theme_embeddings_cache.get(cluster_id)
            if theme_embedding is None:
                self.verbose_reporter.error(f"No theme embedding found for cluster {cluster_id}")
                return []
            
            all_nearest_codes = await self._get_nearest_codes_by_embedding(theme_embedding, current_codes, k)
        
        # Deduplicate by code name while preserving order
        seen_codes = set()
        deduplicated_codes = []
        for code in all_nearest_codes:
            code_key = code['code']
            if code_key not in seen_codes:
                seen_codes.add(code_key)
                deduplicated_codes.append(code)
        
        return deduplicated_codes
    
    async def _get_theme_embedding_for_item(self, cluster_id: Union[int, str], theme_item) -> Optional[np.ndarray]:
        """Get embedding for a specific theme item"""
        try:
            # Generate embedding for this specific theme
            theme_text = theme_item.theme_clarification  # Using theme_clarification instead of theme_statement
            embedding = await self.similarity_engine._get_embedding(theme_text)
            return embedding
        except Exception as e:
            self.verbose_reporter.error(f"Failed to embed theme '{theme_item.theme_clarification}' for cluster {cluster_id}: {e}")
            return None
    
    async def _get_nearest_codes_by_embedding(self, theme_embedding: np.ndarray, 
                                            current_codes: List[Dict[str, str]], k: int) -> List[Dict[str, str]]:
        """Get k nearest codes to a theme embedding using cosine similarity"""
        # Get codebook version
        _, version = await self.shared_codebook.get_current_snapshot()
        
        # Check for cached code embeddings
        code_embeddings = await self.shared_codebook.get_embeddings_for_version(version)
        
        if code_embeddings is None:
            # Generate embeddings for all codes
            #self.verbose_reporter.stat_line(f"Generating embeddings for {len(current_codes)} codes (version {version})")
            
            # Format codes for embedding (same format as old codeGenerator)
            #code_texts = [f"{code['code']}: {code['definition']}" for code in current_codes]
            code_texts = [f"{code['code']}" for code in current_codes]
            
            # Batch embed all codes
            try:
                code_embeddings = await self.similarity_engine._embed_openai_batch(code_texts)
                # Cache the embeddings
                await self.shared_codebook.cache_embeddings(version, code_embeddings)
            except Exception as e:
                self.verbose_reporter.error(f"Failed to generate code embeddings: {e}")
                return []
        
        # Calculate cosine similarities
        code_embeddings_array = np.array(code_embeddings)
        theme_embedding_array = theme_embedding.reshape(1, -1)
        similarities = cosine_similarity(theme_embedding_array, code_embeddings_array)[0]
        
        # Get top k indices
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        # Filter by similarity threshold and return the nearest codes
        nearest_codes = []
        min_similarity_threshold = 0.3  # Only consider codes with at least 30% similarity
        for idx in top_k_indices:
            if idx < len(current_codes) and similarities[idx] >= min_similarity_threshold:
                nearest_codes.append(current_codes[idx])
        
        # Detailed verbodse: similarity score nearest codes to theme embedding
        if self.verbose_detailed and nearest_codes:
            similarity_values = [round(float(similarities[idx]), 3) for idx in top_k_indices]
            codes_with_scores = [f"{code['code']} ({score})" for code, score in zip(nearest_codes, similarity_values)]
            self.verbose_reporter.stat_line(f"Found {len(nearest_codes)} nearest codes with similarities: {codes_with_scores}")
        
        
        return nearest_codes

    
    async def design(self) -> List[Dict[str, Any]]:
        """Main method: Run complete 4-stage CodeDesigner pipeline with comprehensive error handling"""
        start_time = time.time()
        
        # Initialize bootstrap measurement and rate limiting with real API performance data
        await self.async_initialize()
        
        try:
            # Initialize processing statistics
            self._processing_stats = {
                'start_time': start_time,
                'clusters_found': 0,
                'themes_extracted': 0,
                'themes_embedded': 0,
                'batches_created': 0,
                'clusters_processed': 0,
                'codes_added': 0,
                'codes_modified': 0,
                'validation_failures': 0,
                'api_errors': 0,
                'stage_times': {}
                }
            
            # Stage 0: Extract cluster data with error recovery
            stage_start = time.time()
            try:
                clusters = self.extract_cluster_data()
                self._processing_stats['clusters_found'] = len(clusters)
                self.verbose_reporter.stat_line(f"Extracted data from {len(clusters)} clusters")
                
                if not clusters:
                    self.verbose_reporter.warning("No valid clusters found - check input data")
                    return []
                    
            except Exception as e:
                self.verbose_reporter.error(f"Failed to extract cluster data: {e}")
                return []
                
            self._processing_stats['stage_times']['data_extraction'] = time.time() - stage_start
            
            # Stage 1: Theme Extraction with comprehensive error handling
            stage_start = time.time()
            try:
                themes = await self.extract_themes(clusters)
                self._processing_stats['themes_extracted'] = len(themes)
                
                if not themes:
                    self.verbose_reporter.warning("No themes extracted - check cluster content or API connectivity")
                    return []
                    
            except Exception as e:
                self.verbose_reporter.error(f"Critical failure in theme extraction: {e}")
                # Attempt graceful degradation
                self.verbose_reporter.warning("Attempting to continue with partial results...")
                themes = {}
                
            self._processing_stats['stage_times']['theme_extraction'] = time.time() - stage_start
            
            # Store original clusters before expansion for later redistribution
            original_clusters = clusters.copy()
            
            # Stage 1.5: Expand multi-theme clusters into sub-clusters
            themes, clusters, multi_theme_mapping = self.expand_multi_theme_clusters(themes, clusters)
            
            # Early return for theme extraction only mode
            if self.stages_to_run == 'theme_extraction_only':
                self.verbose_reporter.info("Theme extraction only mode - returning themes without further processing")
                return self._format_theme_only_results(themes)
            
            # Stage 2: Theme Embedding with fallback handling
            stage_start = time.time()
            try:
                theme_embeddings = await self.similarity_engine.embed_themes(themes)
                self._processing_stats['themes_embedded'] = len(theme_embeddings)
                
                if not theme_embeddings:
                    self.verbose_reporter.warning("No theme embeddings generated - check embedding API")
                    return []
                    
            except Exception as e:
                self.verbose_reporter.error(f"Critical failure in theme embedding: {e}")
                return []
                
            self._processing_stats['stage_times']['theme_embedding'] = time.time() - stage_start
            
            # Stage 2.5: Redistribute ideas for multi-theme clusters based on embeddings
            if multi_theme_mapping:
                stage_start = time.time()
                self.verbose_reporter.step_start("Idea Redistribution for Multi-Theme Clusters")
                
                # Update ClusterModel objects with expanded_cluster assignments
                await self._update_cluster_models_with_redistribution(
                    multi_theme_mapping, 
                    original_clusters, 
                    themes, 
                    theme_embeddings
                )
                
                # Re-extract cluster data now that models have been updated
                clusters = self.extract_cluster_data()
                
                self.verbose_reporter.step_complete("Idea Redistribution")
                self._processing_stats['stage_times']['idea_redistribution'] = time.time() - stage_start
            
            # Stage 3: Similarity-Based Batching with validation
            stage_start = time.time()
            try:
                dissimilarity_batches = self.similarity_engine.create_dissimilarity_batches(theme_embeddings, themes)
                self._processing_stats['batches_created'] = len(dissimilarity_batches)
                
                if not dissimilarity_batches:
                    self.verbose_reporter.warning("No batches created - all themes may be too similar")
                    # Create single large batch as fallback
                    dissimilarity_batches = [list(theme_embeddings.keys())]
                    
            except Exception as e:
                self.verbose_reporter.error(f"Failure in batch creation: {e}")
                # Fallback: process all clusters individually
                dissimilarity_batches = [[cid] for cid in theme_embeddings.keys()]
                
            self._processing_stats['stage_times']['batch_creation'] = time.time() - stage_start
            
            # Stage 4: Sequential Batch Processing with error recovery
            stage_start = time.time()
            try:
                all_results = await self.process_batches_sequentially(
                    dissimilarity_batches, clusters, themes, theme_embeddings
                )
                self._processing_stats['clusters_processed'] = len(all_results)
                
            except Exception as e:
                # Enhanced error logging to identify exact failure point
                error_msg = str(e).strip()
                self.verbose_reporter.error(f"Critical failure in batch processing: {repr(error_msg)}")
                self.verbose_reporter.error(f"Error type: {type(e).__name__}")
                self.verbose_reporter.error(f"Error length: {len(error_msg)}")
                
                # Print the full stack trace to understand where exactly this is failing
                import traceback
                self.verbose_reporter.error("Full traceback:")
                for line in traceback.format_exc().split('\n'):
                    if line.strip():
                        self.verbose_reporter.error(f"  {line}")
                        
                all_results = []
                
            self._processing_stats['stage_times']['batch_processing'] = time.time() - stage_start
            
            # Final statistics and validation
            processing_time = time.time() - start_time
            self._processing_stats['total_time'] = processing_time
            
            # Final codebook statistics
            try:
                final_version_info = await self.shared_codebook.get_version_info()
                self._processing_stats['final_codebook_version'] = final_version_info['version']
                self._processing_stats['final_codebook_size'] = final_version_info['total_codes']
                self.verbose_reporter.stat_line(f"Final codebook: version {final_version_info['version']}, {final_version_info['total_codes']} codes")
            except Exception as e:
                self.verbose_reporter.error(f"Failed to get final codebook stats: {e}")
            
            # Comprehensive final reporting
            self._generate_final_report(processing_time, len(all_results))
            
            self.verbose_reporter.step_complete("CodeDesigner Pipeline")
            
            self._results = all_results
            return all_results
            
        except Exception as e:
            # Ultimate fallback error handling
            self.verbose_reporter.error(f"Critical pipeline failure: {e}")
            processing_time = time.time() - start_time
            self._processing_stats['total_time'] = processing_time
            self._processing_stats['critical_failure'] = str(e)
            
            self.verbose_reporter.step_complete("CodeDesigner Pipeline (FAILED)")
            return []
    
    def _generate_final_report(self, processing_time: float, clusters_processed: int):
        """Generate comprehensive final processing report"""
        self.verbose_reporter.step_start("Final Processing Report")
        
        # Performance metrics
        self.verbose_reporter.stat_line(f"Total processing time: {processing_time:.1f}s")
        self.verbose_reporter.stat_line(f"Clusters processed: {clusters_processed}")
        
        if 'stage_times' in self._processing_stats:
            self.verbose_reporter.stat_line("Stage breakdown:")
            for stage, duration in self._processing_stats['stage_times'].items():
                percentage = (duration / processing_time) * 100 if processing_time > 0 else 0
                self.verbose_reporter.stat_line(f"  {stage}: {duration:.1f}s ({percentage:.1f}%)")
        
        # Processing efficiency
        if processing_time > 0:
            clusters_per_second = clusters_processed / processing_time
            self.verbose_reporter.stat_line(f"Processing rate: {clusters_per_second:.2f} clusters/second")
        
        # Success rates
        clusters_found = self._processing_stats.get('clusters_found', 0)
        themes_extracted = self._processing_stats.get('themes_extracted', 0)
        
        if clusters_found > 0:
            theme_success_rate = (themes_extracted / clusters_found) * 100
            processing_success_rate = (clusters_processed / clusters_found) * 100
            
            self.verbose_reporter.stat_line(f"Theme extraction success: {themes_extracted}/{clusters_found} ({theme_success_rate:.1f}%)")
            self.verbose_reporter.stat_line(f"Overall processing success: {clusters_processed}/{clusters_found} ({processing_success_rate:.1f}%)")
        
        # Codebook growth and decision statistics
        initial_codes = len(self.starter_codes)
        final_codes = self._processing_stats.get('final_codebook_size', initial_codes)
        codes_added = final_codes - initial_codes
        
        self.verbose_reporter.stat_line(f"Codebook growth: {initial_codes} → {final_codes} (+{codes_added} codes)")
        
        # Decision tracking statistics
        codes_used = self._processing_stats.get('codes_used', 0)
        codes_modified = self._processing_stats.get('codes_modified', 0)
        codes_created = self._processing_stats.get('codes_added', 0)
        total_decisions = codes_used + codes_modified + codes_created
        
        if total_decisions > 0:
            self.verbose_reporter.stat_line("Decision breakdown:")
            self.verbose_reporter.stat_line(f"  USE decisions: {codes_used} ({codes_used/total_decisions*100:.1f}%)")
            self.verbose_reporter.stat_line(f"  MODIFY decisions: {codes_modified} ({codes_modified/total_decisions*100:.1f}%)")
            self.verbose_reporter.stat_line(f"  CREATE decisions: {codes_created} ({codes_created/total_decisions*100:.1f}%)")
        
        # Quality indicators
        validation_failures = self._processing_stats.get('validation_failures', 0)
        api_errors = self._processing_stats.get('api_errors', 0)
        
        if validation_failures > 0 or api_errors > 0:
            self.verbose_reporter.stat_line(f"Issues encountered: {validation_failures} validation failures, {api_errors} API errors")
        else:
            self.verbose_reporter.stat_line("Processing completed without major issues")
        
        self.verbose_reporter.step_complete("Final Processing Report")
    
    def generate(self) -> CodeGeneratorReasoningResults:
        """Generate codes and return complete reasoning results"""
        return asyncio.run(self.generate_async())
    
    async def generate_async(self) -> CodeGeneratorReasoningResults:
        """Async method for code generation - returns proper CodeGeneratorReasoningResults"""
        # Run the design pipeline
        results = await self.design()
        
        # Extract deduplicated codebook from SharedCodebook
        final_codes, version = await self.shared_codebook.get_current_snapshot()
        
        # Get raw cluster data for stats calculations
        cluster_data = self._prepare_cluster_data_for_results()
        
        
        # Convert to CodeGeneratorReasoningResults format
        return CodeGeneratorReasoningResults(
            # Raw cluster results
            cluster_results=results,
            
            step1_inputs=self.step1_inputs,
            step2_inputs=self.step2_inputs,   
            step3_inputs=self.step3_inputs,
            step4_inputs=self.step4_inputs,
            
            step1_summaries=self.step1_summaries,
            step2_analysis=self.step2_analysis,  
            step3_recommendations=self.step3_recommendations,
            step4_validations=self.step4_validations,
            step4_validated_codes=self.step4_validated_codes,
            
            # Processing metadata
            stats=self.summary(),
            generator_version="codeGenerator_4 chain prompt",
            var_lab=self.var_lab,
            total_clusters=len(self.cluster_assignments),
            total_ideas=sum(len(cluster_data.get('ideas', [])) for cluster_data in cluster_data.values()) if cluster_data else 0,
            processing_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            
            # Cluster assignments for cross-reference
            cluster_assignments=self.cluster_assignments,
            
            # New fields for alignment with old codeGenerator
            codebook=final_codes,  # Final deduplicated codebook from SharedCodebook
            cluster_data=cluster_data,  # Raw cluster data for stats calculations
            validation_details=self.step4_validations,  # Detailed validation results
            redistribution_stats=self._redistribution_stats if self._redistribution_stats['clusters_redistributed'] else None
        )
    
    def get_results(self) -> List[Dict[str, Any]]:
        """Get processing results"""
        return self._results
    
    def _prepare_cluster_data_for_results(self) -> Dict[Union[int, str], Dict[str, Any]]:
        """Prepare cluster data from cluster_results using expanded_cluster when available"""
        clusters = {}
        
        for result in self.cluster_results:
            ideas_list = result.response_ideas or []
            
            for idea in ideas_list:
                # Use expanded_cluster if available, otherwise fall back to initial_cluster
                cluster_id = idea.expanded_cluster if idea.expanded_cluster is not None else str(idea.initial_cluster) if idea.initial_cluster is not None else None
                
                if cluster_id is not None and cluster_id != "-1":
                    if cluster_id not in clusters:
                        clusters[cluster_id] = {
                            'cluster_id': cluster_id,
                            'ideas': [],
                            'embeddings': [],
                            'respondent_ids': []
                        }
                    
                    # Add idea data - store the full object to preserve embeddings
                    clusters[cluster_id]['ideas'].append(idea)
                    clusters[cluster_id]['respondent_ids'].append(idea.idea_id)  # Using idea_id as respondent identifier
                    
                    # Add embedding if available (kept for backward compatibility)
                    if hasattr(idea, 'idea_embedding') and idea.idea_embedding is not None:
                        clusters[cluster_id]['embeddings'].append(idea.idea_embedding)
        
        return clusters
    

    async def _process_single_cluster(self, cluster_id: str, clusters: Dict, themes: Dict, codebook_snapshot: List[Dict], base_version: int) -> Optional[Dict[str, Any]]:
        """Process single cluster with unlimited concurrency - no artificial limits (Phase 3)"""
        
        if cluster_id not in themes or cluster_id not in clusters:
            return None
        
        cluster_data = clusters[cluster_id]
        theme_data = themes[cluster_id]
        
        try:
            # Step 1: Get current codebook for candidate selection (ensures latest codes are visible)
            current_codes, _ = await self.shared_codebook.get_current_snapshot()
            nearest_codes = await self._find_nearest_codes_by_theme(cluster_id, theme_data, current_codes, k=5)
            
            # Step 1: Candidate selection - pure unlimited call
            step1_start = time.time()
            candidate_selection = await self._select_candidate_codes(cluster_id, cluster_data, theme_data, nearest_codes)
            step1_duration = time.time() - step1_start
            
            # Check decision from Step 1 to optimize processing
            decision = None
            if candidate_selection and hasattr(candidate_selection, 'coding_decision'):
                decision = candidate_selection.coding_decision.decision.lower()
            
            # OPTIMIZATION: Skip Steps 2 & 3 for USE decisions
            if decision == "use":
                if self.verbose_detailed:
                    self.verbose_reporter.stat_line(f"C{cluster_id}: USE decision detected - skipping code generation and validation steps")
                
                # Populate step3_recommendations using step2 data for reporting consistency
                coding_decision = candidate_selection.coding_decision
                
                # Find the selected candidate from matched_candidates
                selected_candidate = None
                if coding_decision.matched_candidates:
                    for candidate in coding_decision.matched_candidates:
                        if candidate.code == coding_decision.source_code:
                            selected_candidate = candidate
                            break
                
                # Use EXACT same format as real step3_recommendations
                self.step3_recommendations[cluster_id] = {
                    'coding_proposal': 'USE',
                    'source_code': coding_decision.source_code,
                    'code_label_proposal': selected_candidate.code if selected_candidate else coding_decision.source_code,
                    'code_definition_proposal': selected_candidate.definition if selected_candidate else "Existing code definition"
                }
                
                # For USE decisions, return early with minimal result structure
                return {
                    'cluster_id': cluster_id,
                    'theme_name': self._get_theme_name(theme_data),
                    'theme_description': self._get_theme_statement(theme_data),
                    'ideas_count': len(cluster_data['ideas']),
                    'candidate_selection': candidate_selection,
                    'code_generation': None,  # Skipped for USE decisions
                    'validation': None,       # Skipped for USE decisions
                    'final_code': candidate_selection.coding_decision.source_code if hasattr(candidate_selection.coding_decision, 'source_code') else None,
                    'final_definition': None,  # Will be populated from existing codebook during merge
                    'base_version': base_version,
                    'timing': {
                        'step1_duration': step1_duration,
                        'step2_duration': 0.0,  # Skipped
                        'step3_duration': 0.0   # Skipped
                    },
                    'optimization': 'use_early_return'  # Flag to indicate this was optimized
                }
            
            # For CREATE and MODIFY decisions, continue with full 3-step pipeline
            # Step 2: Code generation - pure unlimited call
            step2_start = time.time()
            code_generation = await self._generate_code(
                cluster_id, cluster_data, theme_data, candidate_selection
            )
            step2_duration = time.time() - step2_start
            
            # Step 3: Validation - pure unlimited call
            step3_start = time.time()
            validation = await self._validate_code(cluster_id, cluster_data, theme_data, code_generation, nearest_codes)
            step3_duration = time.time() - step3_start
            
            # Extract final code/definition from validation
            final_code = None
            final_definition = None
            if validation and hasattr(validation, 'code_validation') and validation.code_validation:
                code_validation = validation.code_validation
                if code_validation.validated_code:
                    final_code = code_validation.validated_code.code
                    final_definition = code_validation.validated_code.definition
            
            
            return {
                'cluster_id': cluster_id,
                'theme_name': self._get_theme_name(theme_data),
                'theme_description': self._get_theme_statement(theme_data),
                'ideas_count': len(cluster_data['ideas']),
                'candidate_selection': candidate_selection,
                'code_generation': code_generation,
                'validation': validation,
                'final_code': final_code,
                'final_definition': final_definition,
                'base_version': base_version,
                'timing': {
                    'step1_duration': step1_duration,
                    'step2_duration': step2_duration,
                    'step3_duration': step3_duration
                }
            }
            
        except Exception as e:
            import traceback
            self.verbose_reporter.error(f"Pipeline failed for cluster {cluster_id}: {e}")
            self.verbose_reporter.error(f"Full traceback: {traceback.format_exc()}")
            return None
    
    async def _merge_codebook_updates(self, results: List[Dict[str, Any]], base_version: int):
        """Merge all codebook updates from sub-batch atomically - respects USE/MODIFY/CREATE decisions"""
        
        # Collect updates by decision type
        create_codes = []
        modify_operations = []
        #use_count = 0
        decision_stats = {'use': 0, 'modify': 0, 'create': 0, 'errors': 0, 'modification_leaks': 0}
        
        for result in results:
            cluster_id = result.get('cluster_id', 'unknown')
            
            # Debug: Check result structure
            if not result:
                self.verbose_reporter.error(f"C{cluster_id}: Empty result in merge_codebook_updates")
                decision_stats['errors'] += 1
                continue
            
            # Must have candidate_selection for all decisions
            if not result.get('candidate_selection'):
                self.verbose_reporter.error(f"C{cluster_id}: Missing candidate_selection in result")
                decision_stats['errors'] += 1
                continue
            
            candidate_selection = result['candidate_selection']
            
            # Get the decision from step 1 (candidate_selection)
            decision_info = None
            if hasattr(candidate_selection, 'coding_decision'):
                decision_info = candidate_selection.coding_decision
            else:
                self.verbose_reporter.error(f"C{cluster_id}: Missing candidate_selection coding_decision")
                decision_stats['errors'] += 1
                continue
            
            decision = decision_info.decision.lower()
            
            # Handle USE decisions (optimized path - no validation/code_generation)
            if decision == "use":
                decision_stats['use'] += 1
                if result.get('optimization') == 'use_early_return':
                    self.verbose_reporter.stat_line(f"C{cluster_id}: USE decision (optimized) - no codebook update for '{decision_info.source_code or 'unknown code'}'")
                else:
                    self.verbose_reporter.stat_line(f"C{cluster_id}: USE decision - no codebook update for '{decision_info.source_code or 'unknown code'}'")
                continue  # USE decisions don't modify the codebook
            
            # For CREATE and MODIFY decisions, we need validation and code_generation
            if not result.get('validation') or not result.get('code_generation'):
                self.verbose_reporter.error(f"C{cluster_id}: {decision.upper()} decision missing validation or code_generation")
                decision_stats['errors'] += 1
                continue
                
            validation = result['validation']
            code_generation = result['code_generation']
            
            if not hasattr(validation, 'code_validation') or not hasattr(code_generation, 'generated_code'):
                self.verbose_reporter.error(f"C{cluster_id}: Missing code_validation or generated_code for {decision.upper()} decision")
                decision_stats['errors'] += 1
                continue
            
            # Process the single validation with its corresponding decision
            code_validation = validation.code_validation
            # Use the single generated code
            generated_code = code_generation.generated_code
            
            # CRITICAL CHANGE: Use Prompt 4's final decision instead of Prompt 2's decision
            if hasattr(code_validation, 'validated_decision') and code_validation.validated_decision:
                final_decision = code_validation.validated_decision.lower()
                # Get source_code from validation (Prompt 4) if available, fallback to Prompt 2
                final_source_code = code_validation.source_code if hasattr(code_validation, 'source_code') and code_validation.source_code else decision_info.source_code
            else:
                # Fallback to Prompt 2's decision if Prompt 4 validation failed
                final_decision = decision
                final_source_code = decision_info.source_code
                self.verbose_reporter.warning(f"C{cluster_id}: Using Prompt 2 decision as fallback - Prompt 4 validation incomplete")
            
            if code_validation.validated_code and generated_code and decision_info:
                validated_code = code_validation.validated_code
                
                # Log both decisions for transparency
                if final_decision != decision:
                    self.verbose_reporter.stat_line(f"C{cluster_id}: Prompt 4 overrode Prompt 2: {decision.upper()} → {final_decision.upper()}")
                
                if final_decision == "create":
                    create_codes.append({
                        'code': validated_code.code,
                        'definition': validated_code.definition,
                        'cluster_id': cluster_id
                    })
                    decision_stats['create'] += 1
                    self.verbose_reporter.stat_line(f"C{cluster_id}: FINAL CREATE decision - will add '{validated_code.code}'")
                
                elif final_decision == "modify" and final_source_code:
                    modify_operations.append({
                        'original_code': final_source_code,
                        'new_code': validated_code.code,
                        'new_definition': validated_code.definition,
                        'cluster_id': cluster_id,
                        'full_result': result  # Store complete result for potential recovery
                    })
                    decision_stats['modify'] += 1
                    self.verbose_reporter.stat_line(f"C{cluster_id}: FINAL MODIFY decision - will replace '{final_source_code}' with '{validated_code.code}'")
                
                elif final_decision == "use":
                    # Prompt 4 decided to USE existing code (override CREATE/MODIFY from Prompt 2)
                    decision_stats['use'] += 1
                    self.verbose_reporter.stat_line(f"C{cluster_id}: FINAL USE decision (override) - no codebook update for '{final_source_code or validated_code.code}'")
                
                else:
                    self.verbose_reporter.error(f"C{cluster_id}: Unknown decision '{final_decision}' or missing source_code for modify")
                    decision_stats['errors'] += 1
            else:
                self.verbose_reporter.error(f"C{cluster_id}: Missing validated_code, generated_code, or decision_info for {decision.upper()} decision")
                decision_stats['errors'] += 1
        
        # Execute batch operations with fresh version checking
        updates_made = False
        
        # Process CREATE operations with fresh base version
        if create_codes:
            # Get fresh snapshot to ensure atomic operation
            _, current_version = await self.shared_codebook.get_current_snapshot()
            self.verbose_reporter.stat_line(f"Batch adding {len(create_codes)} new codes to SharedCodebook")
            await self.shared_codebook.batch_update(create_codes, current_version)
            updates_made = True
        
        # Process MODIFY operations individually with validation
        for modify_op in modify_operations:
            # Get fresh snapshot before each modify to ensure consistency
            current_codes, current_version = await self.shared_codebook.get_current_snapshot()
            
            replaced, new_version = await self.shared_codebook.replace_code(
                modify_op['original_code'],
                modify_op['new_code'],
                modify_op['new_definition'],
                modify_op['cluster_id']
            )
            if replaced:
                updates_made = True
                self.verbose_reporter.stat_line(f"C{modify_op['cluster_id']}: Replaced '{modify_op['original_code']}' with '{modify_op['new_code']}'")

                # Update cluster_results: change any cluster with old code to new code
                for cr in self.cluster_results:
                    if cr.get('final_code') == modify_op['original_code']:
                        cr['final_code'] = modify_op['new_code']
                        if self.verbose_detailed:
                            self.verbose_reporter.stat_line(f"  Updated C{cr.get('cluster_id')}'s final_code to '{modify_op['new_code']}'")

                # Post-MODIFY validation: Check if new code creates duplicate
                final_codes, _ = await self.shared_codebook.get_current_snapshot()
                duplicate_count = sum(1 for code in final_codes 
                                    if self.shared_codebook._is_duplicate(code['code'], modify_op['new_code']))
                
                if duplicate_count > 1:
                    self.verbose_reporter.error(f"C{modify_op['cluster_id']}: MODIFY created duplicate! '{modify_op['new_code']}' now exists {duplicate_count} times")
                
            else:
                # This is a modification leak - race condition where source code was already replaced
                decision_stats['modification_leaks'] += 1
                self.verbose_reporter.stat_line(f"C{modify_op['cluster_id']}: MODIFICATION LEAK - '{modify_op['original_code']}' already replaced by concurrent operation")
                
                # Collect modification leak data for recovery batch processing
                leak_data = {
                    'cluster_id': modify_op['cluster_id'],
                    'original_code': modify_op['original_code'], 
                    'new_code': modify_op['new_code'],
                    'new_definition': modify_op['new_definition'],
                    'full_result': modify_op.get('full_result'),  # Complete result for recovery
                    'reason': 'concurrent_modification',
                    'timestamp': time.time()
                }
                self.modification_leaks.append(leak_data)
        
        # Report decision statistics
        total_decisions = sum(decision_stats.values())
        if total_decisions > 0:
            self.verbose_reporter.stat_line(f"Decision summary: USE={decision_stats['use']}, MODIFY={decision_stats['modify']}, CREATE={decision_stats['create']}, ERRORS={decision_stats['errors']}, MODIFICATION_LEAKS={decision_stats['modification_leaks']}")
            
            # Track in processing stats for global reporting
            self._processing_stats['codes_used'] = self._processing_stats.get('codes_used', 0) + decision_stats['use']
            self._processing_stats['codes_modified'] = self._processing_stats.get('codes_modified', 0) + decision_stats['modify']  
            self._processing_stats['codes_added'] = self._processing_stats.get('codes_added', 0) + decision_stats['create']
        
        if not updates_made:
            self.verbose_reporter.stat_line("No codebook updates needed from this sub-batch")
    
    #########################################################################################################
    # Stage 2: Prompt Formatting & LLM Calling for CANDIDATE CODE SELECTION  
    #########################################################################################################
    
    async def _select_candidate_codes(self, cluster_id: Union[int, str], cluster_data: Dict, theme_data, nearest_codes: List[Dict]):
        """Select candidate codes with unlimited concurrency - pure API call"""
        try:
            # Build prompt directly
            theme_id = self._get_theme_id(theme_data)
            theme_name = self._get_theme_name(theme_data)
            theme_description = self._get_theme_description(theme_data)

            # Get embeddings for similarity calculation
            theme_embedding = self._theme_embeddings_cache.get(cluster_id)
            current_codes, version = await self.shared_codebook.get_current_snapshot()
            code_embeddings = await self.shared_codebook.get_embeddings_for_version(version)

            # Calculate similarity metrics if embeddings are available
            if theme_embedding is not None and code_embeddings is not None:
                # Calculate cosine similarities for nearest_codes
                cosine_scores = self._calculate_cosine_similarities(
                    theme_embedding=theme_embedding,
                    candidate_codes=nearest_codes[:20],
                    all_codes=current_codes,
                    code_embeddings=code_embeddings
                )

                # Format codes with cosine similarity
                codes_text = self._format_codes_with_cosine(
                    candidate_codes=nearest_codes[:20],
                    cosine_scores=cosine_scores
                )
            else:
                # Fallback: use simple format without metrics
                codes_text = "\n".join([f"-{code['code']}" for code in nearest_codes[:20]])
                if self.verbose_detailed:
                    self.verbose_reporter.stat_line(f"C{cluster_id}: STEP2 - No embeddings available, using simple code format")

            # Prepare exact parameters for prompt
            params = {
                "survey_question": self.var_lab,
                "language": DEFAULT_LANGUAGE,
                "theme_name": theme_name,
                "theme_description": theme_description,
                "code_text": codes_text,
                "theme_id": theme_id
            }

            prompt = CODING_DECISION_PROMPT.format(**params)
            
            # Capture exact parameters used in prompt construction
            self._capture_prompt_params(cluster_id, "step2", **params)
            
            if self.verbose_detailed: 
                self.verbose_reporter.stat_line(f"C{cluster_id}: STEP1 - Starting candidate selection API call")
                self.verbose_reporter.stat_line(f"C{cluster_id}: STEP1 - Prompt length: {len(prompt)} chars")
                self.verbose_reporter.stat_line(f"C{cluster_id}: STEP1 - Available codes: {len(nearest_codes)}")
            
            # Use async wrapper with JSON retry logic and adaptive timeout
            adaptive_timeout = self._get_adaptive_timeout()
            response = await async_responses_create_with_json_retry(
                model=self.model_config.get_model_for_stage('candidate_selection'),
                prompt=prompt,
                response_model=CodingDecisionOutput,
                reasoning_effort=self.model_config.get_reasoning_effort_for_stage('candidate_selection'),
                text_verbosity=self.model_config.get_text_verbosity_for_stage('candidate_selection'),
                semaphore=self.concurrency_semaphore,
                rate_limiter=self.rate_limiter,
                tpm_bucket=self.tpm_bucket,
                latency_tracker=self.latency_tracker,
                config=self.config,
                timeout=adaptive_timeout
            )
            
            # FUZZY MATCHING: Correct source_code if needed
            if response and response.coding_decision.source_code:
                # Get current codebook codes
                current_codes, _ = await self.shared_codebook.get_current_snapshot()
                available_code_names = [code['code'] for code in current_codes]
                
                # Apply fuzzy matching to source_code
                corrected_source = self._find_closest_code(
                    response.coding_decision.source_code,
                    available_code_names
                )
                
                if corrected_source != response.coding_decision.source_code:
                    if self.verbose_detailed:
                        self.verbose_reporter.stat_line(
                            f"C{cluster_id}: STEP1 - Corrected source_code: '{response.coding_decision.source_code}' → '{corrected_source}'"
                        )
                    # Update the response object
                    response.coding_decision.source_code = corrected_source
            
            # # Capture step2_analysis - the actual coding decisions used in pipeline
            if response:
                self.step2_analysis[cluster_id] = {
                    "coding_decision": {
                        "theme_number": response.coding_decision.theme_number,
                        "theme_name": response.coding_decision.theme_name,
                        "decision": response.coding_decision.decision,
                        "source_code": response.coding_decision.source_code,  # Now corrected
                        "justification": response.coding_decision.justification,
                        "matched_candidates": [
                            {"code": candidate.code, "definition": candidate.definition}
                            for candidate in response.coding_decision.matched_candidates
                        ]
                    }
                }
            
            return response
            
        except Exception as e:
            # Error logging with context
            error_msg = str(e).strip()
            self.verbose_reporter.error(f"C{cluster_id}: STEP1 - Candidate selection failed")
            self.verbose_reporter.error(f"C{cluster_id}: STEP1 - Error type: {type(e).__name__}")
            self.verbose_reporter.error(f"C{cluster_id}: STEP1 - Error message: '{error_msg}' (length: {len(error_msg)})")
            if error_msg == '\n' or error_msg == '':
                self.verbose_reporter.error(f"C{cluster_id}: STEP1 - EMPTY/NEWLINE ERROR DETECTED - API likely returned malformed response")
            return []
    
    #########################################################################################################
    # Stage 3: Prompt Formatting & LLM Calling for GENERATE CODES
    #########################################################################################################
    
    async def _generate_code(self, cluster_id: Union[int, str], cluster_data: Dict, theme_data, candidate_selection):
        """Generate code with unlimited concurrency - pure API call"""
        try:
            if candidate_selection and hasattr(candidate_selection, 'coding_decision'):
                coding_decision_obj = candidate_selection.coding_decision
                decision = coding_decision_obj.decision.upper()
                source_code = coding_decision_obj.source_code
                
                # Fallback logic: if MODIFY decision but no source_code, fall back to CREATE
                if decision == "MODIFY" and (not source_code or source_code.lower() in ['null', 'none', '']):
                    if self.verbose_detailed:
                        self.verbose_reporter.warning(f"C{cluster_id}: STEP2 - MODIFY decision without source_code, falling back to CREATE")
                    decision = "CREATE"
                    source_code = "null"
                    CODING_GENERATION_PROMPT = CODE_CREATION_PROMPT
                elif decision == "MODIFY":
                    CODING_GENERATION_PROMPT = CODING_MODIFICATION_PROMPT
                else:
                    CODING_GENERATION_PROMPT = CODE_CREATION_PROMPT
                
            theme_id = self._get_theme_id(theme_data) 
            theme_name = self._get_theme_name(theme_data)
            theme_description = self._get_theme_description(theme_data)
           
            # Prepare exact parameters for prompt
            params = {
                "language": DEFAULT_LANGUAGE,
                "survey_question": self.var_lab,
                "theme_name": theme_name,
                "theme_description": theme_description,
                "coding_decision": decision,
                "theme_id": theme_id,
                "cluster_summary": theme_name,
                "source_code": source_code
            }
            
            prompt = CODING_GENERATION_PROMPT.format(**params)
            
            # Capture exact parameters used in prompt construction
            self._capture_prompt_params(cluster_id, "step3", **params)
            
            if self.verbose_detailed:
                self.verbose_reporter.stat_line(f"C{cluster_id}: STEP2 - Starting code generation API call")
                self.verbose_reporter.stat_line(f"C{cluster_id}: STEP2 - Prompt length: {len(prompt)} chars")
                self.verbose_reporter.stat_line(f"C{cluster_id}: STEP2 - Candidate codes: {len(candidate_selection.coding_decision.matched_candidates) if candidate_selection and hasattr(candidate_selection, 'coding_decision') else 0}")
            
            # Use async wrapper with JSON retry logic and adaptive timeout
            adaptive_timeout = self._get_adaptive_timeout()
            response = await async_responses_create_with_json_retry(
                model=self.model_config.get_model_for_stage('code_recommendation'),
                prompt=prompt,
                response_model=CodeGenerationOutput,
                reasoning_effort=self.model_config.get_reasoning_effort_for_stage('code_recommendation'),
                text_verbosity=self.model_config.get_text_verbosity_for_stage('code_recommendation'),
                semaphore=self.concurrency_semaphore,
                rate_limiter=self.rate_limiter,
                tpm_bucket=self.tpm_bucket,
                latency_tracker=self.latency_tracker,
                config=self.config,
                timeout=adaptive_timeout
            )
                
            # Capture step3_recommendations (code generation results)
            if response and hasattr(response, 'generated_code'):
                self.step3_recommendations[cluster_id] = {
                #'theme_number': response.generated_code.theme_number,
                #'theme_name': response.generated_code.theme_name,
                'coding_proposal': decision,
                **({'source_code': response.generated_code.source_code} if decision.lower() in ("use", "modify") else {}),
                'code_label_proposal': response.generated_code.code_label,
                'code_definition_proposal': response.generated_code.code_definition
            }
            
            return response
            
        except Exception as e:
            # Error logging with context
            error_msg = str(e).strip()
            self.verbose_reporter.error(f"C{cluster_id}: STEP2 - Code generation failed")
            self.verbose_reporter.error(f"C{cluster_id}: STEP2 - Error type: {type(e).__name__}")
            self.verbose_reporter.error(f"C{cluster_id}: STEP2 - Error message: '{error_msg}' (length: {len(error_msg)})")
            if error_msg == '\n' or error_msg == '':
                self.verbose_reporter.error(f"C{cluster_id}: STEP2 - EMPTY/NEWLINE ERROR DETECTED - API likely returned malformed response")
            return None
    
    
    #########################################################################################################
    # Stage 4: Prompt Formatting & LLM Calling for VALIDATIONN
    #########################################################################################################

    async def _validate_code(self, cluster_id: Union[int, str], cluster_data: Dict, theme_data, code_generation, nearest_codes):
        """Validate code with unlimited concurrency - pure API call"""
        try:
            if len(nearest_codes) > 0:            
                validation_codes_text = "\n".join([f"-{code['code']}" for code in nearest_codes[:10]])
            else:
                validation_codes_text = "No existing codes in the codebook"
            
            theme_id = self._get_theme_id(theme_data) 
            theme_name = self._get_theme_name(theme_data)
            theme_description = self._get_theme_description(theme_data)
            
            #step3_recommendation_json = str(code_generation.model_dump_json(indent=2)) if code_generation else "No recommendations"
            if code_generation:
                step3_recommendation = self.step3_recommendations.get(cluster_id, {})
                if step3_recommendation:
                    code_to_modify_str = ('' if step3_recommendation.get('coding_proposal', 'unknown').lower() != "modify" else f"-Code to modify: {step3_recommendation.get('source_code', 'None')}\n")
                    step3_recommendation_text = (
                        f"-{step3_recommendation.get('coding_proposal', 'unknown')} code\n"
                        f"{code_to_modify_str}"
                        f"-Proposed new label: {step3_recommendation.get('code_label_proposal', 'unknown')}\n"
                        f"-With the following description: {step3_recommendation.get('code_definition_proposal', 'unknown')}\n"
                    )
            else:
                step3_recommendation_text = "No recommendations"
            
            source_code = self.step3_recommendations.get(cluster_id, {}).get('source_code', 'null')
            # Prepare exact parameters for prompt
            params = {
                'language': DEFAULT_LANGUAGE,
                'survey_question': self.var_lab,
                'code_text': validation_codes_text,
                "theme_name": theme_name,
                "theme_description": theme_description,
                'step3_recommendation': step3_recommendation_text,
                'theme_id': theme_id,
                'cluster_summary': theme_name,
                "source_code": source_code
            }
            
            prompt = VALIDATION_PROMPT.format(**params)
            
            # Capture exact parameters used in prompt construction
            self._capture_prompt_params(cluster_id, "step4", **params)
            
                      # Use async wrapper with JSON retry logic and adaptive timeout
            adaptive_timeout = self._get_adaptive_timeout()
            response = await async_responses_create_with_json_retry(
                model=self.model_config.get_model_for_stage('recommendation_validation'),
                prompt=prompt,
                response_model=ValidationResult,
                reasoning_effort=self.model_config.get_reasoning_effort_for_stage('recommendation_validation'),
                text_verbosity=self.model_config.get_text_verbosity_for_stage('recommendation_validation'),
                semaphore=self.concurrency_semaphore,
                rate_limiter=self.rate_limiter,
                tpm_bucket=self.tpm_bucket,
                latency_tracker=self.latency_tracker,
                config=self.config,
                timeout=adaptive_timeout
            )
         
            if response and hasattr(response, 'code_validation') and response.code_validation.source_code:
                # Get current codebook codes
                current_codes, _ = await self.shared_codebook.get_current_snapshot()
                available_code_names = [code['code'] for code in current_codes]
                
                # Apply fuzzy matching to validation source_code
                corrected_source = self._find_closest_code(
                    response.code_validation.source_code,
                    available_code_names
                )
                
                if corrected_source != response.code_validation.source_code:
                    if self.verbose_detailed:
                        self.verbose_reporter.stat_line(
                            f"C{cluster_id}: STEP3 - Corrected validation source_code: '{response.code_validation.source_code}' → '{corrected_source}'"
                        )
                    # Update the response object
                    response.code_validation.source_code = corrected_source
            
            # Capture step4_validations
            if response and hasattr(response, 'code_validation'):
                self.step4_validations[cluster_id] = {
                    'code_validation': {
                        'theme_number': response.code_validation.theme_number,
                        'theme_name': response.code_validation.theme_name,
                        'original_recommendation': {
                            'code': response.code_validation.original_recommendation.code,
                            'definition': response.code_validation.original_recommendation.definition
                        },
                        'verdict': response.code_validation.verdict,  # APPROVE/REJECT (renamed from 'decision')
                        'decision_rationale': response.code_validation.decision_rationale,
                        'validated_decision': response.code_validation.validated_decision,  # USE/MODIFY/CREATE (NEW)
                        'source_code': response.code_validation.source_code,  # NEW
                        'validated_code': {
                            'code': response.code_validation.validated_code.code,
                            'definition': response.code_validation.validated_code.definition
                        }
                    }
                }
            
            
            return response
            
        except Exception as e:
            # Enhanced error logging with context
            error_msg = str(e).strip()
            self.verbose_reporter.error(f"C{cluster_id}: STEP3 - Code validation failed")
            self.verbose_reporter.error(f"C{cluster_id}: STEP3 - Error type: {type(e).__name__}")
            self.verbose_reporter.error(f"C{cluster_id}: STEP3 - Error message: '{error_msg}' (length: {len(error_msg)})")
            if error_msg == '\n' or error_msg == '':
                self.verbose_reporter.error(f"C{cluster_id}: STEP3 - EMPTY/NEWLINE ERROR DETECTED - API likely returned malformed response")
            return None
    

    def summary(self) -> Dict[str, Any]:
        """Get comprehensive processing summary statistics"""
        clusters_found = self._processing_stats.get('clusters_found', 0)
        themes_extracted = self._processing_stats.get('themes_extracted', 0)
        clusters_processed = len(self._results)
        
        # Calculate success rates
        theme_success_rate = (themes_extracted / clusters_found * 100) if clusters_found > 0 else 0
        processing_success_rate = (clusters_processed / clusters_found * 100) if clusters_found > 0 else 0
        
        # Calculate processing efficiency
        total_time = self._processing_stats.get('total_time', 0)
        processing_rate = (clusters_processed / total_time) if total_time > 0 else 0
        
        # Codebook statistics
        initial_codes = len(self.starter_codes)
        final_codes = self._processing_stats.get('final_codebook_size', initial_codes)
        
        return {
            # Core metrics
            'total_clusters_processed': clusters_processed,
            'clusters_found': clusters_found,
            'themes_extracted': themes_extracted,
            'processing_success_rate': round(processing_success_rate, 2),
            'theme_extraction_success_rate': round(theme_success_rate, 2),
            
            # Performance metrics
            'total_processing_time_seconds': round(total_time, 2),
            'processing_rate_clusters_per_second': round(processing_rate, 3),
            'stage_times': self._processing_stats.get('stage_times', {}),
            
            # Configuration
            'similarity_threshold': self.config.similarity_threshold,
            'model_used': self.config.model,
            'max_sub_batch_size': self.config.max_sub_batch_size,
            
            # Codebook evolution
            'initial_codebook_size': initial_codes,
            'final_codebook_size': final_codes,
            'codebook_growth': final_codes - initial_codes,
            'final_codebook_version': self._processing_stats.get('final_codebook_version', 0),
            
            # Quality indicators
            'batches_created': self._processing_stats.get('batches_created', 0),
            'themes_embedded': self._processing_stats.get('themes_embedded', 0),
            'validation_failures': self._processing_stats.get('validation_failures', 0),
            'api_errors': self._processing_stats.get('api_errors', 0),
            
            # Pipeline health
            'pipeline_completed_successfully': 'critical_failure' not in self._processing_stats,
            'critical_failure': self._processing_stats.get('critical_failure'),
            
            # Raw processing stats
            'raw_processing_stats': self._processing_stats
        }