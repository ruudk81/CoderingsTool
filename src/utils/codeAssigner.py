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

from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError, InternalServerError
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type
from instructor.exceptions import InstructorRetryException
from aiolimiter import AsyncLimiter
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# === MODELS ========================================================================================================
from pydantic import BaseModel, field_validator
import models

# === CONFIG ========================================================================================================
from config import OPENAI_API_KEY, DEFAULT_LANGUAGE, ModelConfig, CodeAssignmentConfig, DEFAULT_CODE_ASSIGNMENT_CONFIG, ProcessingConfig, DEFAULT_PROCESSING_CONFIG, get_openai_rate_limits, GENERAL_CODE_LABELS, MISCELLANEOUS_CODE_LABELS
from prompts import DEFAULT_CODE_EVALUATION_PROMPT, FALLBACK_CODE_ASSIGNMENT_PROMPT

# === UTILS ========================================================================================================
from .verboseReporter import VerboseReporter, ProcessingStats
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

class DefaultCodeEvaluationResponse(BaseModel):
    """Stage 1: Evaluating default code from cluster"""
    idea_id: str
    confidence: float
    rationale: str

    @field_validator('confidence', mode='before')
    @classmethod
    def coerce_confidence(cls, v):
        """Coerce string numbers to float (common with LLM JSON output)"""
        if isinstance(v, str):
            return float(v)
        return v

class FallbackCodeAssignmentResponse(BaseModel):
    """Stage 2: Selecting from all codes"""
    idea_id: str
    assigned_codes: List[str]
    assignment_confidence: float
    assignment_rationale: str

    @field_validator('assignment_confidence', mode='before')
    @classmethod
    def coerce_confidence(cls, v):
        """Coerce string numbers to float (common with LLM JSON output)"""
        if isinstance(v, str):
            return float(v)
        return v


class CodeAssigner:
    """
    Two-stage code assignment using embedding-based similarity filtering.
    Stage 2 presents top-10 most similar codes instead of entire codebook.
    """
    
    def __init__(
        self,
        cluster_models: List[models.ClusterModel],
        codebook: List[models.Codebook],
        var_lab: str,
        code_to_theme_mapping: Optional[Dict[str, str]] = None,
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

        # Theme mapping for code-to-theme assignments
        self.code_to_theme_mapping = code_to_theme_mapping or {}

        # Initialize tokenizer for token counting (cached)
        self.encoding = get_tiktoken_encoding(self.model)

        # Instructor-patched async OpenAI client for structured output (cached)
        self.client = get_openai_client(OPENAI_API_KEY)

        # Embedding client for code similarity (plain OpenAI client)
        self.embedding_client = OpenAI(api_key=OPENAI_API_KEY)
        self.embedding_model = self.model_config.embedding_model

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
            'timeouts': 0,
            'error_types': {}  # Track error types: {error_type: count}
        }

        # Two-stage assignment stats
        self.stage_1_calls = 0
        self.stage_2_calls = 0
        self.used_default_count = 0
        self.used_fallback_count = 0

        # Prompt/Response logging for debugging
        self.prompt_responses = []
        self.last_prompt = ""  # Track the last prompt used for assignment
        self.verbose = verbose

        # Build cluster→codes mapping
        self.cluster_to_codes = self._build_cluster_code_mapping()

        # Code embeddings for similarity filtering (lazy-load on first use)
        self._code_embeddings = None

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

    def _build_cluster_code_mapping(self) -> Dict[str, List[models.Codebook]]:
        """Build mapping from expanded_cluster ID to codes generated from that cluster"""
        from collections import defaultdict
        mapping = defaultdict(list)
        merged_codes_count = 0

        for code in self.codebook:
            if hasattr(code, 'source_cluster') and code.source_cluster:
                # Split comma-separated cluster IDs (e.g., "8,11,23" → ["8", "11", "23"])
                cluster_ids = str(code.source_cluster).split(',')

                # Log if multiple clusters share this code
                if len(cluster_ids) > 1:
                    merged_codes_count += 1
                    # if self.verbose:
                    #     cluster_list = [c.strip() for c in cluster_ids]
                    #     #self.verbose_reporter.info(f"  Code '{code.code[:50]}...' mapped to {len(cluster_list)} clusters: {cluster_list}")

                # Create mapping for each individual cluster ID
                for cluster_id in cluster_ids:
                    cluster_id = cluster_id.strip()  # Remove whitespace
                    if cluster_id:  # Skip empty strings
                        mapping[cluster_id].append(code)

        # Convert to regular dict and log stats
        cluster_dict = dict(mapping)
        if self.verbose:
            total_clusters_with_codes = len(cluster_dict)
            avg_codes_per_cluster = sum(len(codes) for codes in cluster_dict.values()) / total_clusters_with_codes if total_clusters_with_codes > 0 else 0
            self.verbose_reporter.stat_line(f"Cluster→Code mapping: {total_clusters_with_codes} clusters, avg {avg_codes_per_cluster:.1f} codes/cluster")
            if merged_codes_count > 0:
                self.verbose_reporter.stat_line(f"  {merged_codes_count} codes shared across multiple clusters")

        return cluster_dict

    def _create_prompt_for_estimation(self, idea_id: str, idea_text: str) -> str:
        """Create prompt for token estimation using top-k codes"""
        top_k = self.config.top_k_similar_codes
        all_codes_text = "\n".join([
            f"Code: {code.code}\nDefinition: {code.definition}\n"
            for code in self.codebook[:top_k]
        ])

        unknown_label = MISCELLANEOUS_CODE_LABELS.get(self.language, "Other")

        return FALLBACK_CODE_ASSIGNMENT_PROMPT.format(
            language=self.language,
            var_lab=self.var_lab,
            idea_id=idea_id,
            idea_text=idea_text,
            default_confidence=0.0,
            all_codes=all_codes_text,
            unknown_label=unknown_label
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

    def _assign_themes_to_codes(self, assigned_codes: List[str]) -> List[str]:
        """Map assigned codes to their themes using cached mapping"""
        themes = []
        for code in assigned_codes:
            theme = self.code_to_theme_mapping.get(code)
            if theme and theme not in themes:
                themes.append(theme)
        return themes

    def _build_general_codes(self) -> List[Dict[str, any]]:
        """Build synthetic general codes for theme and category fallbacks"""
        general_codes = []
        general_label = GENERAL_CODE_LABELS.get(self.language, "overall")

        # Theme-level general codes
        unique_themes = set()
        for code in self.codebook:
            if hasattr(code, 'theme') and code.theme:
                theme_desc = getattr(code, 'theme_description', code.theme)
                unique_themes.add((code.theme, theme_desc))

        for theme, theme_desc in unique_themes:
            # Collect specific codes in this theme for exclusion examples
            specific_codes_in_theme = [code.code for code in self.codebook
                                      if hasattr(code, 'theme') and code.theme == theme]

            general_codes.append({
                'code': f"{theme} - {general_label}",
                'definition': f"Algemene verwijzing naar thema '{theme}': {theme_desc}",
                'inclusion_examples': [
                    f"Algemene of vage verwijzing naar {theme}",
                    f"Niet-specifieke uitspraak over {theme}",
                    f"Vaag verband met {theme} zonder concrete details"
                ],
                'exclusion_examples': specific_codes_in_theme,
                'near_neighbor_label': "Specifieke codes binnen dit thema",
                'tell_apart_rule': f"Gebruik deze algemene code alleen als geen enkele specifieke code past. Specifieke codes in dit thema: {', '.join(specific_codes_in_theme[:3])}{'...' if len(specific_codes_in_theme) > 3 else ''}",
                'type': 'theme_general'
            })

        # Category-level general codes (if 3-level hierarchy exists)
        unique_categories = set()
        for code in self.codebook:
            if hasattr(code, 'category') and code.category:
                cat_desc = getattr(code, 'category_description', code.category)
                theme = getattr(code, 'theme', '')
                unique_categories.add((code.category, cat_desc, theme))

        for category, cat_desc, theme in unique_categories:
            # Collect specific codes in this category for exclusion examples
            specific_codes_in_category = [code.code for code in self.codebook
                                         if hasattr(code, 'category') and code.category == category]

            general_codes.append({
                'code': f"{category} - {general_label}",
                'definition': f"Algemene verwijzing naar categorie '{category}' binnen {theme}: {cat_desc}",
                'inclusion_examples': [
                    f"Algemene of vage verwijzing naar {category}",
                    f"Niet-specifieke uitspraak over {category} binnen {theme}",
                    f"Vaag verband met {category} zonder concrete details"
                ],
                'exclusion_examples': specific_codes_in_category,
                'near_neighbor_label': "Specifieke codes binnen deze categorie",
                'tell_apart_rule': f"Gebruik deze algemene code alleen als geen enkele specifieke code past. Specifieke codes in deze categorie: {', '.join(specific_codes_in_category[:3])}{'...' if len(specific_codes_in_category) > 3 else ''}",
                'type': 'category_general'
            })

        return general_codes

    def _generate_code_embeddings(self) -> np.ndarray:
        """Generate embeddings for all codes (code + definition)"""
        code_texts = [f"Code: {code.code}. Definition: {code.definition}"
                      for code in self.codebook]

        embeddings = []
        for text in code_texts:
            response = self.embedding_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            embeddings.append(response.data[0].embedding)

        return np.array(embeddings)

    @property
    def code_embeddings(self):
        """Lazy-load code embeddings on first use"""
        if self._code_embeddings is None:
            self._code_embeddings = self._generate_code_embeddings()
        return self._code_embeddings

    def _find_similar_codes(self, idea_embedding: np.ndarray, top_k: int = 10) -> List[models.Codebook]:
        """Find top-k most similar codes using cosine similarity"""
        similarities = cosine_similarity([idea_embedding], self.code_embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.codebook[i] for i in top_indices]

    def _format_examples_list(self, examples: Optional[List[str]]) -> str:
        """Format examples list for prompt display"""
        if not examples:
            return "No specific examples provided"
        return "\n".join([f"  • {ex}" for ex in examples])

    def _extract_all_ideas(self) -> List[tuple]:
        """Extract all individual ideas for processing with expanded_cluster info"""
        all_ideas = []

        for model in self.cluster_models:
            if hasattr(model, 'response_ideas') and model.response_ideas:
                for idea_submodel in model.response_ideas:
                    if hasattr(idea_submodel, 'idea_embedding') and idea_submodel.idea_embedding is not None:
                        # Extract expanded_cluster (fallback to initial_cluster if not available)
                        expanded_cluster = getattr(idea_submodel, 'expanded_cluster', None) or \
                                         getattr(idea_submodel, 'initial_cluster', None)

                        all_ideas.append((
                            model.respondent_id,
                            idea_submodel.idea_id,
                            idea_submodel.idea,
                            idea_submodel.idea_embedding,
                            expanded_cluster
                        ))
                    else:
                        self.verbose_reporter.stat_line(f"Warning: No embedding for idea {idea_submodel.idea_id}")
            else:
                self.verbose_reporter.stat_line(f"Warning: No response_ideas found for respondent {model.respondent_id}")

        return all_ideas

    def _create_prompt(self, idea_id: str, idea_text: str) -> str:
        """Create prompt for probe calls using Stage 2 (all codes) prompt"""
        # Format all codes
        all_codes_text = "\n".join([
            f"Code: {code.code}\nDefinition: {code.definition}\n"
            for code in self.codebook
        ])

        unknown_label = MISCELLANEOUS_CODE_LABELS.get(self.language, "Other")

        prompt = FALLBACK_CODE_ASSIGNMENT_PROMPT.format(
            language=self.language,
            var_lab=self.var_lab,
            idea_id=idea_id,
            idea_text=idea_text,
            default_confidence=0.0,
            all_codes=all_codes_text,
            unknown_label=unknown_label
        )

        return prompt
    
    async def probe_call_no_structured(self, task_dict):
        """Probe call without structured output for bootstrap measurement"""
        idea_data = task_dict['idea_data']
        respondent_id, idea_id, idea_text, idea_embedding, expanded_cluster = idea_data

        prompt = self._create_prompt(idea_id, idea_text)
        
        # For probes: avoid response_model so we can read .usage
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            seed=self.model_config.seed
        )

        u = getattr(resp, "usage", None)
        return {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens}

    async def evaluate_default_code(self, idea_id: str, idea_text: str, default_code: models.Codebook):
        """Stage 1: Evaluate how well the default code from cluster fits the idea

        Returns:
            tuple: (DefaultCodeEvaluationResponse, str) - response and prompt used
        """

        prompt = DEFAULT_CODE_EVALUATION_PROMPT.format(
            language=self.language,
            var_lab=self.var_lab,
            idea_id=idea_id,
            idea_text=idea_text,
            default_code=default_code.code,
            default_definition=default_code.definition,
            inclusion_examples=self._format_examples_list(default_code.inclusion_examples),
            exclusion_examples=self._format_examples_list(default_code.exclusion_examples),
            near_neighbor_label=default_code.near_neighbor_label or "Unknown",
            tell_apart_rule=default_code.tell_apart_rule or "N/A"
        )

        self.last_prompt = prompt  # Store for backward compatibility

        # Estimate tokens and acquire from TPM bucket
        est_tokens = self.estimate_tokens(prompt)
        await self.tpm_bucket.wait_and_acquire(est_tokens)

        # Calculate adaptive timeout
        timeout = self.latency_tracker.get_timeout(est_tokens)

        # Unified rate limiting with semaphore and rate limiter
        async with self.semaphore:
            async with self.rate_limiter:
                start_time = time.perf_counter()

                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.model,
                        response_model=DefaultCodeEvaluationResponse,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        seed=self.model_config.seed
                    ),
                    timeout=timeout
                )

                # Track latency for adaptive timeout adjustment
                latency = time.perf_counter() - start_time
                self.latency_tracker.add(latency)

                # Token reconciliation: reconcile actual vs estimated
                if hasattr(response, '_raw_response'):
                    usage = response._raw_response.usage
                    if usage:
                        actual_total_tokens = usage.total_tokens
                        delta = actual_total_tokens - est_tokens
                        await self.tpm_bucket.reconcile(delta)

        self.stage_1_calls += 1
        return response, prompt

    async def assign_from_all_codes(self, idea_id: str, idea_text: str, idea_embedding: np.ndarray, default_confidence: float):

        # Build list of codes: top-10 similar + general + unknown
        all_codes_list = []

        # 1. Add top-10 most similar codes using embeddings
        top_k = self.config.top_k_similar_codes
        similar_codes = self._find_similar_codes(idea_embedding, top_k=top_k)

        for code in similar_codes:
            all_codes_list.append({
                'code': code.code,
                'definition': code.definition,
                'inclusion_examples': code.inclusion_examples,
                'exclusion_examples': code.exclusion_examples,
                'near_neighbor_label': code.near_neighbor_label,
                'tell_apart_rule': code.tell_apart_rule,
                'type': 'specific'
            })

        # 2. Add general codes (theme/category level)
        all_codes_list.extend(self._build_general_codes())

        # 3. Add unknown fallback
        unknown_label = MISCELLANEOUS_CODE_LABELS.get(self.language, "Other")
        all_codes_list.append({
            'code': unknown_label,
            'definition': "Geen duidelijke relatie met thema's in codebook",
            'inclusion_examples': None,
            'exclusion_examples': None,
            'near_neighbor_label': None,
            'tell_apart_rule': None,
            'type': 'unknown'
        })

        # Format all codes for prompt
        all_codes_text = "\n".join([
            f"Code: {c['code']}\n"
            f"Definition: {c['definition']}\n"
            f"Include when: {self._format_examples_list(c.get('inclusion_examples'))}\n"
            f"Exclude when: {self._format_examples_list(c.get('exclusion_examples'))}\n"
            f"Boundary: Differs from '{c.get('near_neighbor_label') or 'N/A'}' - {c.get('tell_apart_rule') or 'N/A'}\n"
            for c in all_codes_list
        ])

        prompt = FALLBACK_CODE_ASSIGNMENT_PROMPT.format(
            language=self.language,
            var_lab=self.var_lab,
            idea_id=idea_id,
            idea_text=idea_text,
            default_confidence=default_confidence,
            all_codes=all_codes_text,
            unknown_label=unknown_label
        )

        self.last_prompt = prompt  # Store for backward compatibility

        # Estimate tokens and acquire from TPM bucket
        est_tokens = self.estimate_tokens(prompt)
        await self.tpm_bucket.wait_and_acquire(est_tokens)

        # Calculate adaptive timeout
        timeout = self.latency_tracker.get_timeout(est_tokens)

        # Unified rate limiting with semaphore and rate limiter
        async with self.semaphore:
            async with self.rate_limiter:
                start_time = time.perf_counter()

                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.model,
                        response_model=FallbackCodeAssignmentResponse,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        seed=self.model_config.seed
                    ),
                    timeout=timeout
                )

                # Track latency for adaptive timeout adjustment
                latency = time.perf_counter() - start_time
                self.latency_tracker.add(latency)

                # Token reconciliation: reconcile actual vs estimated
                if hasattr(response, '_raw_response'):
                    usage = response._raw_response.usage
                    if usage:
                        actual_total_tokens = usage.total_tokens
                        delta = actual_total_tokens - est_tokens
                        await self.tpm_bucket.reconcile(delta)

        self.stage_2_calls += 1
        return response, prompt

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
        """Two-stage code assignment: evaluate default from cluster, fallback to all codes if needed"""
        #task_start = time.perf_counter()

        try:
            idea_data = task['idea_data']
            respondent_id, idea_id, idea_text, idea_embedding, expanded_cluster = idea_data

            # Metadata for tracking default vs fallback
            metadata = {
                'used_default': False,
                'fallback_triggered': False,
                'default_confidence': None,
                'expanded_cluster': expanded_cluster
            }

            # Local variable to capture the prompt for this specific task (avoid race conditions)
            prompt_used = ""

            # Get default code(s) from this idea's cluster
            default_codes = self.cluster_to_codes.get(str(expanded_cluster), [])

            if not default_codes:
                # No codes from this cluster - go straight to fallback (Stage 2)
                metadata['fallback_triggered'] = True
                stage_2_result, prompt_used = await self.assign_from_all_codes(idea_id, idea_text, idea_embedding, default_confidence=0.0)

                assigned_code = stage_2_result.assigned_codes[0]
                confidence = stage_2_result.assignment_confidence
                rationale = f"No default code available. {stage_2_result.assignment_rationale}"
                self.used_fallback_count += 1

            else:
                # Stage 1: Evaluate default code from cluster
                default_code = default_codes[0]  # Use first code from cluster
                stage_1_result, prompt_used = await self.evaluate_default_code(idea_id, idea_text, default_code)

                metadata['default_confidence'] = stage_1_result.confidence

                if stage_1_result.confidence >= 0.7:
                    # Use default code
                    metadata['used_default'] = True
                    assigned_code = default_code.code
                    confidence = stage_1_result.confidence
                    rationale = stage_1_result.rationale
                    self.used_default_count += 1

                else:
                    # Stage 2: Fallback to top-10 similar codes
                    metadata['fallback_triggered'] = True
                    stage_2_result, prompt_used = await self.assign_from_all_codes(idea_id, idea_text, idea_embedding, stage_1_result.confidence)

                    assigned_code = stage_2_result.assigned_codes[0]
                    confidence = stage_2_result.assignment_confidence
                    rationale = f"Default: {stage_1_result.rationale} | Fallback: {stage_2_result.assignment_rationale}"
                    self.used_fallback_count += 1

            # Create response
            response = CodeAssignmentResponse(
                idea_id=idea_id,
                idea=idea_text,
                assigned_codes=[assigned_code],
                assignment_confidence=confidence,
                assignment_rationale=rationale
            )

            # Add theme mapping
            response.assigned_themes = self._assign_themes_to_codes([assigned_code])

            # Capture for debugging (only if verbose)
            if self.verbose:
                self.prompt_responses.append({
                    'prompt': prompt_used,  # Use local variable to avoid race conditions with concurrent tasks
                    'respondent_id': respondent_id,
                    'idea_id': idea_id,
                    'idea_text': idea_text,
                    'expanded_cluster': expanded_cluster,
                    'assigned_codes': [assigned_code],
                    'confidence': confidence,
                    'rationale': rationale,
                    'metadata': metadata
                })

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
        respondent_id, idea_id, idea_text, idea_embedding, expanded_cluster = idea_data
        
        # Return fallback response (first available code)
        fallback_code = self.codebook[0].code if self.codebook else "Unknown"
        fallback_themes = self._assign_themes_to_codes([fallback_code]) if fallback_code != "Unknown" else []
        
        return CodeAssignmentResponse(
            idea_id=idea_id,
            idea=idea_text,
            assigned_codes=[fallback_code],
            assigned_themes=fallback_themes,
            assignment_confidence=0.1,
            assignment_rationale="Processing failed, using fallback code"
        )
    
    async def worker(self, queue: asyncio.Queue, results: List):
        """Worker coroutine that processes tasks from queue"""
        #worker_id = id(asyncio.current_task())
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
                    error_type = type(e).__name__
                    error_msg = str(e)

                    # Track error types
                    if error_type not in self.stats['error_types']:
                        self.stats['error_types'][error_type] = {'count': 0, 'sample_messages': []}
                    self.stats['error_types'][error_type]['count'] += 1
                    # Store up to 3 sample error messages per type
                    if len(self.stats['error_types'][error_type]['sample_messages']) < 3:
                        self.stats['error_types'][error_type]['sample_messages'].append(error_msg[:200])

                    logger.error(f"Task {task['task_id']} failed after retries: {error_type}: {e}")
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
                        cluster_theme=getattr(idea_submodel, 'cluster_theme', None),
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

    def get_random_samples(self, n: int = 3, seed: int = None) -> List[Dict]:
        """
        Get n random prompt/response samples for inspection.

        Args:
            n: Number of samples to return (default 3)
            seed: Random seed for reproducibility (default None)

        Returns:
            List of dictionaries containing sample data
        """
        if not self.prompt_responses:
            return []

        # Use numpy random for consistent behavior
        rng = np.random.default_rng(seed)

        # Sample without replacement (or all if n > total)
        n_samples = min(n, len(self.prompt_responses))
        indices = rng.choice(len(self.prompt_responses), size=n_samples, replace=False)

        samples = [self.prompt_responses[i] for i in indices]
        return samples

    def print_samples(self, samples: List[Dict]):
        """Pretty-print samples for inspection"""
        if not samples:
            print("\n⚠️ No samples available (verbose mode may be disabled)")
            return

        print(f"\n{'='*80}")
        print(f"RANDOM CODE ASSIGNMENT SAMPLES (n={len(samples)})")
        print(f"{'='*80}")

        for i, sample in enumerate(samples, 1):
            print(f"\n{'─'*80}")
            print(f"SAMPLE #{i}")
            print(f"{'─'*80}")
            print(f"Respondent ID: {sample['respondent_id']}")
            print(f"Idea ID: {sample['idea_id']}")
            print("\nIdea Text:")
            print(f"  {sample['idea_text']}")
            print(f"\nAssigned Codes: {', '.join(sample['assigned_codes'])}")
            print(f"Assigned Themes: {', '.join(sample['assigned_themes']) if sample['assigned_themes'] else 'None'}")
            print(f"Confidence: {sample['confidence']:.2f}")
            print("\nRationale:")
            print(f"  {sample['rationale']}")
            print(f"\n{'─'*40}")
            print("FULL PROMPT:")
            print(f"{'─'*40}")
            print(sample['prompt'])
            print(f"{'─'*80}\n")

    def print_assignment_stats(self):
        """Print detailed stats about default vs fallback usage"""
        total = self.used_default_count + self.used_fallback_count

        if total == 0:
            print("\n⚠️ No assignment stats available")
            return

        default_pct = (self.used_default_count / total) * 100
        fallback_pct = (self.used_fallback_count / total) * 100

        print(f"\n{'='*80}")
        print("CODE ASSIGNMENT STRATEGY BREAKDOWN")
        print(f"{'='*80}")
        print(f"Total ideas processed: {total}")
        print("")
        print(f"Used default (cluster code): {self.used_default_count} ({default_pct:.1f}%)")
        print(f"Used fallback (all codes):   {self.used_fallback_count} ({fallback_pct:.1f}%)")
        print("")
        print("API calls:")
        print(f"  Stage 1 (evaluate default): {self.stage_1_calls}")
        print(f"  Stage 2 (fallback):         {self.stage_2_calls}")
        print(f"  Total API calls:            {self.stage_1_calls + self.stage_2_calls}")
        print(f"  Avg calls per idea:         {(self.stage_1_calls + self.stage_2_calls) / total:.2f}")
        print(f"{'='*80}\n")

    async def process_all_tasks_async(self, tasks: List[Dict]) -> List[CodeAssignmentResponse]:
        """Process all tasks using queue + workers pattern with bootstrap measurement"""
        if not tasks:
            return []
        
        try:
            #print(f"[DEBUG] Starting process_all_tasks_async with {len(tasks)} tasks")
            
            # Setup
            limits = get_openai_rate_limits(self.model)

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

            # Report error types if any failures occurred
            if self.stats['error_types']:
                print(f"\nError Types ({len(self.stats['error_types'])} unique):")
                for error_type, error_data in sorted(self.stats['error_types'].items(),
                                                     key=lambda x: x[1]['count'], reverse=True):
                    print(f"  - {error_type}: {error_data['count']} occurrences")
                    if error_data['sample_messages']:
                        print("    Sample errors:")
                        for i, msg in enumerate(error_data['sample_messages'], 1):
                            print(f"      {i}. {msg}")
        
            return results
        
        except Exception as e:
            logger.error(f"[CRITICAL ERROR] process_all_tasks_async failed: {type(e).__name__}: {e}")
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            print(f"\n❌ CODE ASSIGNMENT FAILED: {type(e).__name__}: {e}")
            print("Returning fallback responses for all tasks...\n")
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
