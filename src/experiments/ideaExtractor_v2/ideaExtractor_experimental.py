"""
Experimental IdeaExtractor for ideaExtractor_v2

This is a COPY of src/utils/ideaExtractor.py that can be modified freely
for experimentation without affecting production code.

KEY CHANGE: Imports prompts from local experiments/ideaExtractor_v2/prompts.py
instead of production src/prompts.py

COPIED FROM: src/utils/ideaExtractor.py
"""

import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import asyncio
import random
import re
import time
import statistics
import itertools
import logging
import unicodedata
from typing import Dict, List, Optional, Union, Literal, Tuple
from dataclasses import dataclass
from collections import deque
import numpy as np

import nest_asyncio
from openai import RateLimitError, APIConnectionError, APITimeoutError, InternalServerError
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type
from instructor.exceptions import InstructorRetryException
from aiolimiter import AsyncLimiter

logger = logging.getLogger(__name__)

# === MODELS ========================================================================================================
from pydantic import BaseModel, Field, field_validator
import models

# === CONFIG ========================================================================================================
from config import OPENAI_API_KEY, DEFAULT_LANGUAGE, ModelConfig, SegmentationConfig, DEFAULT_SEGMENTATION_CONFIG, ProcessingConfig, DEFAULT_PROCESSING_CONFIG, FALLBACK_TPM, FALLBACK_RPM
from utils.llm import create_client, llm_create_async, ProbeResponse, RateLimits, extract_rate_limits_from_response

# === EXPERIMENTAL PROMPTS (KEY CHANGE) =============================================================================
# Import from local experimental prompts instead of production
from experiments.ideaExtractor_v2.prompts import (
    CONSOLIDATE_SPECIFIERS_GROUP1, CONSOLIDATE_SPECIFIERS_GROUP2,
    CONTEXT_SPECIFIER_PROMPT1, CONTEXT_SPECIFIER_PROMPT2,
    TAXONOMY_CHUNK_SCORING_PROMPT, TAXONOMY_CONSOLIDATION_PROMPT,
    TAXONOMY_AWARE_SUBJECT_PROMPT, TAXONOMY_ENRICHED_EXTRACTION_PROMPT,
    TAXONOMY_AXIS_DESCRIPTIONS
)

# === UTILS ========================================================================================================
from utils.verboseReporter import VerboseReporter, ProcessingStats
from utils.cached_resources import get_openai_client, get_tiktoken_encoding

async_client = get_openai_client(OPENAI_API_KEY)


# === CONSTANTS ========================================================================================================
INPUT_HISTORY_MAXLEN = 3              # EMA input token history window
OUTPUT_HISTORY_MAXLEN = 5             # EMA output token history window
ERROR_WINDOW_SIZE = 50                # Token estimation error tracking window
DEFAULT_TIMEOUT_SECONDS = 30.0        # Default timeout when no latency data
DEFAULT_LATENCY_SECONDS = 2.0         # Default latency estimate
PROGRESS_REPORT_INTERVAL = 5          # Seconds between progress reports
DIAGNOSTIC_INTERVAL = 30              # Seconds between diagnostic reports
MAX_TOKEN_ACQUIRE_ATTEMPTS = 1000     # Max attempts to acquire tokens before failing
BOOTSTRAP_TIMEOUT_SECONDS = 30.0      # Timeout for bootstrap probe calls
DEFAULT_AVG_TOKENS = 1500             # Default token estimate fallback
SAMPLE_SIZE_FOR_TOKEN_ESTIMATION = 10 # Sample size for initial token calculation
GENERIC_SPECIFIER_SAMPLE_MIN = 50     # Min samples for generic specifiers
GENERIC_SPECIFIER_SAMPLE_MAX = 1000   # Max samples for generic specifiers
GENERIC_SPECIFIER_CHUNK_SIZE = 100    # Chunk size for specifier extraction
MAX_SPECIFIER_WORKERS = 10            # Max workers for specifier extraction


# === RATE LIMITING CLASSES  ========================================================================================================
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
        attempts = 0
        while attempts < MAX_TOKEN_ACQUIRE_ATTEMPTS:
            attempts += 1
            result = await self.acquire(tokens_needed)
            if result is True:
                return
            else:
                # result is wait_seconds
                await asyncio.sleep(result)

        raise RuntimeError(f"Failed to acquire {tokens_needed} tokens after {MAX_TOKEN_ACQUIRE_ATTEMPTS} attempts")

    async def reconcile(self, delta_tokens: int) -> None:
        """Reconcile actual token usage against estimate."""
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
            return max(config.adaptive_timeout_min_seconds, DEFAULT_TIMEOUT_SECONDS)

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
            return DEFAULT_LATENCY_SECONDS
        return self.ema if self.ema is not None else DEFAULT_LATENCY_SECONDS


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

class GenericSpecifierGroup1Response(BaseModel):
    """Group 1: Speaker characteristics"""
    lang: str = Field(description="Language/dialect code (e.g., nl-NL, en-US)")
    perspective: str = Field(description="Stakeholder viewpoint (e.g., consumer, employee)")
    intent: str = Field(description="Purpose of responses (e.g., evaluate, describe)")

class GenericSpecifierGroup2Response(BaseModel):
    """Group 2: Subject matter"""
    domain: str = Field(description="Industry domain (e.g., finance, healthcare)")
    topic: str = Field(description="Subject matter (e.g., brand_association, customer_service)")
    entity: str = Field(description="Main entity (e.g., merk_x, company_name)")

# === TAXONOMY PYDANTIC MODELS ========================================================================================================
class TaxonomyChunkScore(BaseModel):
    """Per-chunk taxonomy axis scoring (used in parallel phase)."""
    axis: str = Field(description="Axis code: WHAT, WHY, HOW, WHO, SENTIMENT, WHEN, WHERE")
    score: float = Field(description="Relevance score 0.0-1.0")
    evidence_count: int = Field(default=0, description="How many responses in this chunk support this axis")
    sample_phrases: List[str] = Field(default_factory=list, description="2-3 example phrases from this chunk")


class TaxonomyChunkResponse(BaseModel):
    """LLM response for single chunk taxonomy scoring."""
    axis_scores: List[TaxonomyChunkScore] = Field(description="Scores for all 7 axes")


class TaxonomyConsolidatedResponse(BaseModel):
    """Consolidated taxonomy selection after merging all chunks."""
    primary_axis: str = Field(description="Selected primary axis code")
    primary_axis_rationale: str = Field(description="Why this axis was chosen")
    primary_axis_score: float = Field(default=0.0, description="Weighted score across chunks")
    primary_axis_description: str = Field(
        default="",
        description="1-2 sentence context-specific description of this axis tailored to the survey question and domain"
    )
    secondary_axis: Optional[str] = Field(default=None, description="Optional orthogonal axis")
    secondary_axis_rationale: Optional[str] = Field(default=None, description="Why secondary axis adds value")
    all_axis_scores: List[TaxonomyChunkScore] = Field(default_factory=list, description="Final consolidated scores")


class TaxonomyEnrichedIdeaResponse(BaseModel):
    """Extended idea response with taxonomy phrase."""
    respondent_id: str = Field(alias_choices=['respondent_id', 'respond_id', 'respondent'])
    idea_id: str = Field(default="1", alias_choices=['idea_id', 'id'])
    idea: str = Field(description="The extracted idea text following template")
    taxonomy_phrase: str = Field(default="", description="2-4 word categorization phrase for clustering")
    sentiment: Literal["positive", "negative", "neutral"] = Field(default="neutral", description="Sentiment: positive, negative, or neutral")
    sense: Literal["factual", "evaluative", "aspirational", "experiential"] = Field(default="factual", description="Sense: factual, evaluative, aspirational, experiential")

    # Class variable to hold expected template prefix (set dynamically per extraction)
    _expected_template_prefix: str = ""

    @field_validator('sentiment', mode='before')
    @classmethod
    def normalize_sentiment(cls, v: str) -> str:
        """Normalize invalid sentiment values to valid 3-class options."""
        if not isinstance(v, str):
            return "neutral"
        v_lower = v.lower().strip()
        if v_lower in ("positive", "negative", "neutral"):
            return v_lower
        # Tie-break: map invalid values to neutral
        if v_lower in ("mixed", "evaluative", "aspirational"):
            return "neutral"
        return "neutral"

    @field_validator('sense', mode='before')
    @classmethod
    def normalize_sense(cls, v: str) -> str:
        """Normalize invalid sense values."""
        if not isinstance(v, str):
            return "factual"
        v_lower = v.lower().strip()
        if v_lower in ("factual", "evaluative", "aspirational", "experiential"):
            return v_lower
        return "factual"

    @field_validator('taxonomy_phrase', mode='before')
    @classmethod
    def normalize_taxonomy_phrase(cls, v: str) -> str:
        """Normalize taxonomy_phrase: lowercase, no punctuation, trimmed."""
        if not isinstance(v, str):
            return ""
        # Lowercase and strip
        v = v.lower().strip()
        # Remove trailing punctuation (keep internal hyphens)
        v = v.rstrip('.,;:!?')
        return v

    @field_validator('idea')
    @classmethod
    def validate_template_compliance(cls, v: str) -> str:
        """Ensure idea follows the required phrasing template"""
        if not v or v in ["", "NA", "N/A"]:
            return v

        expected_prefix = getattr(cls, '_expected_template_prefix', '')

        if expected_prefix and not v.startswith(expected_prefix):
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

        # Initialize OpenAI client with instructor (supports OpenAI and Azure)
        self.client = create_client(self.model, async_mode=True)

        # Rate limiting setup - use fallback values for initial setup
        self.rate_limits = RateLimits(
            tokens_per_minute=FALLBACK_TPM,
            requests_per_minute=FALLBACK_RPM,
            tokens_per_day=FALLBACK_TPM * 60 * 24
        )

        # Token bucket for TPM limiting
        self.tpm_bucket = TokenBucket(self.rate_limits.tokens_per_minute * self.processing_config.rate_limit_headroom)

        # Adaptive token estimation
        self.input_token_history = deque(maxlen=INPUT_HISTORY_MAXLEN)
        self.output_token_history = deque(maxlen=OUTPUT_HISTORY_MAXLEN)
        self.estimation_errors = deque(maxlen=ERROR_WINDOW_SIZE)
        self.first_prompt_tokens = None

        # Rolling average of actual total tokens
        self.actual_total_tokens = deque(maxlen=ERROR_WINDOW_SIZE)

        # Latency tracking
        self.latency_tracker = LatencyTracker(processing_config=self.processing_config)

        # Cache for subject extraction
        self._subject_cache = {}

        # Generic specifiers (must be initialized before _calculate_avg_tokens)
        self.generic_specifiers = {}

        # Template prefix for embedding
        self.template_prefix = None

        # Taxonomy axis (must be initialized before _calculate_avg_tokens)
        self.taxonomy_axis = None
        self.secondary_axis = None
        self.taxonomy_rationale = None
        self.taxonomy_axis_description = None  # Dynamic context-specific description
        self.taxonomy_sample_phrases = []  # Sample phrases from responses for dynamic examples

        # Calculate initial average tokens estimate
        self.avg_tokens = self._calculate_avg_tokens()

        # Rate limiting components (will be initialized after bootstrap)
        self.rate_limiter = None
        self.semaphore = None
        self.optimal_concurrency = None

        # Stats
        self.stats = {
            'tasks_processed': 0,
            'tasks_successful': 0,
            'tasks_failed': 0,
            'retries': 0,
            'rate_limits': 0,
            'timeouts': 0
        }

    def _calculate_avg_tokens(self) -> int:
        """Calculate average tokens per request for rate limiting.

        Uses taxonomy-enriched prompt with default values for estimation.
        """
        if not self.responses:
            return DEFAULT_AVG_TOKENS

        sample_size = min(SAMPLE_SIZE_FOR_TOKEN_ESTIMATION, len(self.responses))
        sample_responses = self.responses[:sample_size]

        # Store original values to restore after estimation
        original_taxonomy_axis = self.taxonomy_axis
        original_taxonomy_axis_description = self.taxonomy_axis_description
        original_taxonomy_sample_phrases = self.taxonomy_sample_phrases
        original_generic_specifiers = self.generic_specifiers

        # Set defaults for token estimation
        self.taxonomy_axis = "WHAT"
        self.taxonomy_axis_description = "general concepts and ideas"
        self.taxonomy_sample_phrases = []
        self.generic_specifiers = {}

        placeholder_subject = "the subject"
        placeholder_phrasing_template = "the subject is [ATTRIBUTE_OR_ACTION]"

        token_counts = []
        for response in sample_responses:
            prompt = self._build_taxonomy_enriched_prompt(
                response.respondent_id,
                response.response,
                placeholder_subject,
                placeholder_phrasing_template
            )
            prompt_tokens = len(self.encoding.encode(prompt))
            completion_tokens = int(prompt_tokens * 0.25)
            token_counts.append(prompt_tokens + completion_tokens)

        # Restore original values
        self.taxonomy_axis = original_taxonomy_axis
        self.taxonomy_axis_description = original_taxonomy_axis_description
        self.taxonomy_sample_phrases = original_taxonomy_sample_phrases
        self.generic_specifiers = original_generic_specifiers

        return int(statistics.mean(token_counts)) if token_counts else DEFAULT_AVG_TOKENS

    async def _extract_taxonomy_aware_subject(
        self,
        survey_question: str,
        taxonomy_axis: str,
        secondary_axis: Optional[str] = None
    ) -> SubjectExtractionResponse:
        """Extract canonical subject with axis-aware template generation.

        This method generates a phrasing template that is shaped by the selected
        taxonomy axis, ensuring ideas are normalized along a consistent dimension.

        Args:
            survey_question: The original survey question
            taxonomy_axis: Primary taxonomy axis (WHAT, WHY, HOW, etc.)
            secondary_axis: Optional secondary orthogonal axis

        Returns:
            SubjectExtractionResponse with axis-aware canonical_phrasing
        """
        # Cache key includes taxonomy axis
        cache_key = f"{survey_question}_{taxonomy_axis}"
        if cache_key in self._subject_cache:
            return self._subject_cache[cache_key]

        try:
            # Use dynamic description if available, fall back to static dict
            axis_description = getattr(self, 'taxonomy_axis_description', None) or \
                TAXONOMY_AXIS_DESCRIPTIONS.get(taxonomy_axis, "general concepts and ideas")

            prompt = TAXONOMY_AWARE_SUBJECT_PROMPT.format(
                language=self.language,
                survey_question=survey_question,
                primary_axis=taxonomy_axis,
                primary_axis_description=axis_description,
                secondary_axis=secondary_axis or "None"
            )

            response = await llm_create_async(
                client=self.client,
                model=self.model,
                response_model=SubjectExtractionResponse,
                prompt=prompt,
                temperature=0.0
            )

            # Capture prompt if enabled
            if self.prompt_printer:
                self.prompt_printer.capture_prompt(
                    step_name="idea_extraction_taxonomy_subject",
                    utility_name="IdeaExtractor",
                    prompt_content=prompt,
                    prompt_type="taxonomy_aware_subject_extraction",
                    metadata={
                        "model": self.model,
                        "survey_question": survey_question,
                        "taxonomy_axis": taxonomy_axis,
                        "secondary_axis": secondary_axis,
                        "language": self.language
                    }
                )

            self._subject_cache[cache_key] = response

            if self.verbose_reporter.enabled:
                self.verbose_reporter.stat_line(f"Taxonomy-aware subject extraction:")
                self.verbose_reporter.stat_line(f"  Axis: {taxonomy_axis} ({axis_description})")
                self.verbose_reporter.stat_line(f"  Canonical term: '{response.canonical_term}'")
                self.verbose_reporter.stat_line(f"  Canonical phrasing: '{response.canonical_phrasing}'")

            return response

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"Taxonomy-aware subject extraction failed: {e}. Using fallback.", exc_info=True)
            # Generate axis-appropriate fallback template
            axis_templates = {
                "WHAT": "the subject has [ATTRIBUTE_OR_ACTION]",
                "WHY": "the subject should achieve [ATTRIBUTE_OR_ACTION]",
                "HOW": "the subject should [ATTRIBUTE_OR_ACTION]",
                "WHO": "the stakeholder needs [ATTRIBUTE_OR_ACTION]",
                "SENTIMENT": "the subject is [ATTRIBUTE_OR_ACTION]",
                "WHEN": "the subject should [ATTRIBUTE_OR_ACTION]",
                "WHERE": "the subject at [ATTRIBUTE_OR_ACTION]"
            }
            fallback_template = axis_templates.get(taxonomy_axis, "the subject [ATTRIBUTE_OR_ACTION]")

            fallback = SubjectExtractionResponse(
                decision="CANONICAL_SUBJECT",
                canonical_term="the subject",
                canonical_phrasing=fallback_template
            )
            self._subject_cache[cache_key] = fallback
            return fallback

    async def _consolidate_specifiers(self, group: int, chunk_results: List[Dict]) -> Dict[str, str]:
        """Consolidate specifier results from multiple chunks using LLM."""
        if hasattr(self, 'verbose_reporter') and self.verbose_reporter.enabled:
            group_name = "Group1 (lang/perspective/intent)" if group == 1 else "Group2 (domain/topic/entity)"
            self.verbose_reporter.stat_line(f"  Consolidating {len(chunk_results)} {group_name} results via LLM...")

        formatted_results = []
        for idx, result in enumerate(chunk_results, 1):
            response_obj = result['response']
            if group == 1:
                formatted_results.append(
                    f"Chunk {idx}:\n"
                    f"  - lang: {response_obj.lang}\n"
                    f"  - perspective: {response_obj.perspective}\n"
                    f"  - intent: {response_obj.intent}"
                )
            else:
                formatted_results.append(
                    f"Chunk {idx}:\n"
                    f"  - domain: {response_obj.domain}\n"
                    f"  - topic: {response_obj.topic}\n"
                    f"  - entity: {response_obj.entity}"
                )

        chunk_results_text = "\n\n".join(formatted_results)

        if group == 1:
            prompt = CONSOLIDATE_SPECIFIERS_GROUP1.format(
                survey_question=self.var_lab,
                chunk_results=chunk_results_text
            )
            response_model = GenericSpecifierGroup1Response
        else:
            prompt = CONSOLIDATE_SPECIFIERS_GROUP2.format(
                survey_question=self.var_lab,
                chunk_results=chunk_results_text
            )
            response_model = GenericSpecifierGroup2Response

        async with self.semaphore:
            await self.tpm_bucket.acquire(2000)
            await self.rate_limiter.acquire()

            response = await llm_create_async(
                client=self.client,
                model=self.model,
                response_model=response_model,
                prompt=prompt,
                temperature=0.0
            )

            await self.tpm_bucket.reconcile(0)

        if hasattr(self, 'verbose_reporter') and self.verbose_reporter.enabled:
            if group == 1:
                self.verbose_reporter.stat_line(
                    f"    Consolidated: lang={response.lang}, perspective={response.perspective}, intent={response.intent}"
                )
            else:
                self.verbose_reporter.stat_line(
                    f"    Consolidated: domain={response.domain}, topic={response.topic}, entity={response.entity}"
                )

        if group == 1:
            return {
                "lang": response.lang,
                "perspective": response.perspective,
                "intent": response.intent
            }
        else:
            return {
                "domain": response.domain,
                "topic": response.topic,
                "entity": response.entity
            }

    async def _consolidate_taxonomy(self, chunk_results: List[Dict]) -> TaxonomyConsolidatedResponse:
        """Consolidate taxonomy scores from chunks to select primary + secondary axis.

        Always calls LLM consolidation to generate context-specific axis description,
        even for single chunks.

        Args:
            chunk_results: List of dicts with 'response' containing TaxonomyChunkResponse

        Returns:
            TaxonomyConsolidatedResponse with selected axes and context-specific description
        """
        # Format chunk results for consolidation prompt
        formatted_results = []
        for idx, result in enumerate(chunk_results):
            chunk_response = result['response']
            scores_text = "\n".join([
                f"  - {s.axis}: {s.score:.2f} (evidence: {s.evidence_count})"
                for s in chunk_response.axis_scores
            ])
            formatted_results.append(f"Chunk {idx + 1}:\n{scores_text}")

        prompt = TAXONOMY_CONSOLIDATION_PROMPT.format(
            language=self.language,
            survey_question=self.var_lab,
            chunk_results="\n\n".join(formatted_results)
        )

        async with self.semaphore:
            await self.tpm_bucket.acquire(2000)
            await self.rate_limiter.acquire()

            response = await llm_create_async(
                client=self.client,
                model=self.model,
                response_model=TaxonomyConsolidatedResponse,
                prompt=prompt,
                temperature=0.0
            )

            await self.tpm_bucket.reconcile(0)

        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line(f"  Taxonomy consolidated:")
            self.verbose_reporter.stat_line(f"    Primary: {response.primary_axis} (score: {response.primary_axis_score:.2f})")
            self.verbose_reporter.stat_line(f"    Rationale: {response.primary_axis_rationale[:100]}...")
            if response.secondary_axis:
                self.verbose_reporter.stat_line(f"    Secondary: {response.secondary_axis}")

        return response

    def _select_axis_from_single_chunk(self, chunk_response: TaxonomyChunkResponse) -> TaxonomyConsolidatedResponse:
        """Select primary axis from a single chunk's scores (no consolidation needed).

        Args:
            chunk_response: TaxonomyChunkResponse with axis_scores

        Returns:
            TaxonomyConsolidatedResponse with selected axes
        """
        if not chunk_response.axis_scores:
            # Fallback if no scores
            return TaxonomyConsolidatedResponse(
                primary_axis="WHAT",
                primary_axis_rationale="Fallback - no axis scores available",
                primary_axis_score=0.5,
                secondary_axis=None,
                secondary_axis_rationale=None,
                all_axis_scores=[]
            )

        # Sort by score descending
        sorted_axes = sorted(chunk_response.axis_scores, key=lambda x: x.score, reverse=True)
        primary = sorted_axes[0]

        # Check for orthogonal secondary axis (SENTIMENT is often orthogonal)
        secondary = None
        secondary_rationale = None
        if len(sorted_axes) > 1:
            # SENTIMENT is orthogonal to most other axes
            for axis in sorted_axes[1:]:
                if axis.axis == "SENTIMENT" and axis.score >= 0.3:
                    secondary = axis.axis
                    secondary_rationale = f"SENTIMENT is orthogonal to {primary.axis} with score {axis.score:.2f}"
                    break
                # Also consider other high-scoring orthogonal pairs
                elif axis.score >= 0.5 and self._are_axes_orthogonal(primary.axis, axis.axis):
                    secondary = axis.axis
                    secondary_rationale = f"{axis.axis} is orthogonal to {primary.axis} with score {axis.score:.2f}"
                    break

        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line(f"  Taxonomy (single chunk selection):")
            self.verbose_reporter.stat_line(f"    Primary: {primary.axis} (score: {primary.score:.2f})")
            if secondary:
                self.verbose_reporter.stat_line(f"    Secondary: {secondary}")

        return TaxonomyConsolidatedResponse(
            primary_axis=primary.axis,
            primary_axis_rationale=f"Highest scoring axis in single chunk with score {primary.score:.2f}",
            primary_axis_score=primary.score,
            secondary_axis=secondary,
            secondary_axis_rationale=secondary_rationale,
            all_axis_scores=sorted_axes
        )

    def _are_axes_orthogonal(self, axis1: str, axis2: str) -> bool:
        """Check if two taxonomy axes are orthogonal (independent dimensions).

        Args:
            axis1: First axis code
            axis2: Second axis code

        Returns:
            True if axes are orthogonal
        """
        # Define orthogonal pairs - axes that measure different dimensions
        orthogonal_pairs = {
            ("WHAT", "SENTIMENT"),
            ("WHY", "SENTIMENT"),
            ("HOW", "SENTIMENT"),
            ("WHO", "SENTIMENT"),
            ("WHEN", "SENTIMENT"),
            ("WHERE", "SENTIMENT"),
            ("WHAT", "WHO"),
            ("WHAT", "WHEN"),
            ("WHAT", "WHERE"),
            ("WHY", "HOW"),
            ("WHO", "WHEN"),
            ("WHO", "WHERE"),
        }
        pair = tuple(sorted([axis1, axis2]))
        return pair in orthogonal_pairs or (pair[1], pair[0]) in orthogonal_pairs

    async def _extract_generic_specifiers(self) -> Tuple[Dict[str, str], TaxonomyConsolidatedResponse]:
        """Extract context specifiers AND taxonomy axis from response sample.

        Returns:
            Tuple of (context_specifiers dict, TaxonomyConsolidatedResponse)
        """
        sample_size = min(GENERIC_SPECIFIER_SAMPLE_MAX, max(int(0.2 * len(self.responses)), GENERIC_SPECIFIER_SAMPLE_MIN))
        sample = random.sample(self.responses, min(sample_size, len(self.responses)))

        chunk_size = GENERIC_SPECIFIER_CHUNK_SIZE
        chunks = [sample[i:i+chunk_size] for i in range(0, len(sample), chunk_size)]

        self.verbose_reporter.stat_line(f"Context + Taxonomy: {len(sample)} samples, {len(chunks)} chunks")

        tasks = []
        for chunk_idx, chunk in enumerate(chunks):
            chunk_text = "\n".join([f"- {r.response}" for r in chunk])

            # Group 1: lang/perspective/intent
            tasks.append({
                'task_id': f"group1_chunk{chunk_idx}",
                'group': 1,
                'chunk_idx': chunk_idx,
                'chunk_text': chunk_text,
                'chunk_size': len(chunk)
            })

            # Group 2: domain/topic/entity
            tasks.append({
                'task_id': f"group2_chunk{chunk_idx}",
                'group': 2,
                'chunk_idx': chunk_idx,
                'chunk_text': chunk_text,
                'chunk_size': len(chunk)
            })

            # Group 3: Taxonomy axis scoring (NEW)
            tasks.append({
                'task_id': f"taxonomy_chunk{chunk_idx}",
                'group': 3,
                'chunk_idx': chunk_idx,
                'chunk_text': chunk_text,
                'chunk_size': len(chunk)
            })

        results = await self._process_generic_specifier_tasks(tasks)

        group1_results = [r for r in results if r['group'] == 1]
        group2_results = [r for r in results if r['group'] == 2]
        taxonomy_results = [r for r in results if r['group'] == 3]  # NEW

        if self.verbose_reporter.enabled and group1_results and group2_results:
            self.verbose_reporter.stat_line(f"  Chunk-level results:")
            for r in group1_results:
                self.verbose_reporter.stat_line(
                    f"    Chunk {r['chunk_idx']+1} (Group1): "
                    f"lang={r['response'].lang}, perspective={r['response'].perspective}, intent={r['response'].intent}"
                )
            for r in group2_results:
                self.verbose_reporter.stat_line(
                    f"    Chunk {r['chunk_idx']+1} (Group2): "
                    f"domain={r['response'].domain}, topic={r['response'].topic}, entity={r['response'].entity}"
                )
            # NEW: Log taxonomy chunk results
            for r in taxonomy_results:
                top_axes = sorted(r['response'].axis_scores, key=lambda x: x.score, reverse=True)[:3]
                top_str = ", ".join([f"{a.axis}={a.score:.2f}" for a in top_axes])
                self.verbose_reporter.stat_line(
                    f"    Chunk {r['chunk_idx']+1} (Taxonomy): Top 3: {top_str}"
                )

        # Handle missing results with fallbacks
        if not group1_results or not group2_results:
            self.verbose_reporter.stat_line(f"  Warning: Generic specifier extraction failed ({len(group1_results)} group1, {len(group2_results)} group2 results)")

            lang_code = "nl-NL" if "dutch" in self.language.lower() or "nederlands" in self.language.lower() else "en-US"

            context_result = {
                "lang": lang_code,
                "perspective": "consumer",
                "intent": "evaluate",
                "domain": "general",
                "topic": "feedback",
                "entity": "unknown"
            }
            self.verbose_reporter.stat_line(f"  Using fallback context defaults: {context_result}")

            # Fallback taxonomy
            taxonomy_result = TaxonomyConsolidatedResponse(
                primary_axis="WHAT",
                primary_axis_rationale="Fallback default - no taxonomy results available",
                primary_axis_score=0.5,
                secondary_axis=None,
                secondary_axis_rationale=None,
                all_axis_scores=[]
            )
            return context_result, taxonomy_result

        # Consolidate Group 1
        if len(group1_results) == 1:
            if self.verbose_reporter.enabled:
                self.verbose_reporter.stat_line(f"  Single chunk - skipping consolidation for Group1")
            group1_consolidated = {
                "lang": group1_results[0]['response'].lang,
                "perspective": group1_results[0]['response'].perspective,
                "intent": group1_results[0]['response'].intent
            }
        else:
            group1_consolidated = await self._consolidate_specifiers(1, group1_results)

        # Consolidate Group 2
        if len(group2_results) == 1:
            if self.verbose_reporter.enabled:
                self.verbose_reporter.stat_line(f"  Single chunk - skipping consolidation for Group2")
            group2_consolidated = {
                "domain": group2_results[0]['response'].domain,
                "topic": group2_results[0]['response'].topic,
                "entity": group2_results[0]['response'].entity
            }
        else:
            group2_consolidated = await self._consolidate_specifiers(2, group2_results)

        # NEW: Consolidate Taxonomy
        if not taxonomy_results:
            self.verbose_reporter.stat_line(f"  Warning: No taxonomy results - using fallback")
            taxonomy_consolidated = TaxonomyConsolidatedResponse(
                primary_axis="WHAT",
                primary_axis_rationale="Fallback default - no taxonomy results available",
                primary_axis_score=0.5,
                secondary_axis=None,
                secondary_axis_rationale=None,
                all_axis_scores=[]
            )
        else:
            taxonomy_consolidated = await self._consolidate_taxonomy(taxonomy_results)

        context_result = {**group1_consolidated, **group2_consolidated}

        self.verbose_reporter.stat_line(f"  Context results: {context_result}")
        self.verbose_reporter.stat_line(f"  Taxonomy: primary={taxonomy_consolidated.primary_axis}, secondary={taxonomy_consolidated.secondary_axis}")
        return context_result, taxonomy_consolidated

    async def _process_generic_specifier_tasks(self, tasks: List[Dict]) -> List[Dict]:
        queue = asyncio.Queue()
        results = []

        for task in tasks:
            await queue.put(task)

        num_workers = min(MAX_SPECIFIER_WORKERS, len(tasks))
        for _ in range(num_workers):
            await queue.put(None)

        workers = [
            asyncio.create_task(self._generic_specifier_worker(queue, results))
            for _ in range(num_workers)
        ]

        await asyncio.gather(*workers)

        return results

    async def _generic_specifier_worker(self, queue: asyncio.Queue, results: List):
        """Worker for processing generic specifier AND taxonomy tasks."""
        while True:
            task = await queue.get()
            if task is None:
                break

            try:
                if self.semaphore is None or self.rate_limiter is None:
                    raise RuntimeError(
                        f"Rate limiters not initialized before worker started. "
                        f"semaphore={self.semaphore}, rate_limiter={self.rate_limiter}"
                    )

                async with self.semaphore:
                    await self.tpm_bucket.acquire(DEFAULT_AVG_TOKENS)
                    await self.rate_limiter.acquire()

                    if task['group'] == 1:
                        # Group 1: lang/perspective/intent
                        prompt = CONTEXT_SPECIFIER_PROMPT1.format(
                            language=self.language,
                            survey_question=self.var_lab,
                            chunk_responses=task['chunk_text'],
                            chunk_size=task['chunk_size']
                        )
                        response_model = GenericSpecifierGroup1Response
                    elif task['group'] == 2:
                        # Group 2: domain/topic/entity
                        prompt = CONTEXT_SPECIFIER_PROMPT2.format(
                            language=self.language,
                            survey_question=self.var_lab,
                            chunk_responses=task['chunk_text'],
                            chunk_size=task['chunk_size']
                        )
                        response_model = GenericSpecifierGroup2Response
                    else:  # group == 3: Taxonomy
                        # Group 3: Taxonomy axis scoring (NEW)
                        prompt = TAXONOMY_CHUNK_SCORING_PROMPT.format(
                            language=self.language,
                            survey_question=self.var_lab,
                            chunk_responses=task['chunk_text'],
                            chunk_size=task['chunk_size']
                        )
                        response_model = TaxonomyChunkResponse

                    response = await llm_create_async(
                        client=self.client,
                        model=self.model,
                        response_model=response_model,
                        prompt=prompt,
                        temperature=0.0
                    )

                    await self.tpm_bucket.reconcile(0)

                    results.append({
                        'task_id': task['task_id'],
                        'group': task['group'],
                        'chunk_idx': task['chunk_idx'],
                        'response': response
                    })

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"Generic specifier task {task['task_id']} failed: {e}", exc_info=True)
                self.verbose_reporter.stat_line(f"Generic specifier task {task['task_id']} failed: {e}")
            finally:
                queue.task_done()

    def _build_taxonomy_enriched_prompt(
        self,
        respondent_id: str,
        response: str,
        subject: str,
        phrasing_template: str
    ) -> str:
        """Build taxonomy-enriched prompt for idea extraction.

        Uses TAXONOMY_ENRICHED_EXTRACTION_PROMPT which includes taxonomy axis context
        and requests taxonomy_phrase for each extracted idea.

        Args:
            respondent_id: Respondent identifier
            response: The response text to extract ideas from
            subject: Canonical subject/term
            phrasing_template: Template for phrasing ideas

        Returns:
            Formatted prompt string
        """
        # Use dynamic description if available, fall back to static dict
        taxonomy_axis_description = getattr(self, 'taxonomy_axis_description', None) or \
            TAXONOMY_AXIS_DESCRIPTIONS.get(self.taxonomy_axis, "General categorization")

        # Format sample phrases as bullet list (from actual responses)
        if self.taxonomy_sample_phrases:
            sample_phrases_formatted = "\n".join([
                f"   - \"{phrase}\"" for phrase in self.taxonomy_sample_phrases
            ])
        else:
            # Fallback to generic examples if no sample phrases available
            fallback_examples = {
                "WHAT": ["product quality", "service reliability", "price value", "feature completeness"],
                "WHY": ["improve efficiency", "reduce costs", "increase satisfaction", "enhance experience"],
                "HOW": ["streamline process", "automate tasks", "simplify interface", "add functionality"],
                "WHO": ["customer needs", "employee support", "management direction"],
                "SENTIMENT": ["positive experience", "negative feedback", "mixed feelings"],
                "WHEN": ["urgent priority", "future planning", "immediate action"],
                "WHERE": ["online channel", "physical location", "mobile app"]
            }
            examples = fallback_examples.get(self.taxonomy_axis, ["general topic", "specific aspect"])
            sample_phrases_formatted = "\n".join([f"   - \"{ex}\"" for ex in examples])

        return TAXONOMY_ENRICHED_EXTRACTION_PROMPT.format(
            var_lab=self.var_lab,
            taxonomy_axis=self.taxonomy_axis,
            taxonomy_axis_description=taxonomy_axis_description,
            domain=self.generic_specifiers.get('domain', 'general'),
            topic=self.generic_specifiers.get('topic', 'feedback'),
            entity=self.generic_specifiers.get('entity', 'unknown'),
            subject=subject,
            phrasing_template=phrasing_template,
            language=self.language,
            respondent_id=respondent_id,
            response=response,
            sample_phrases_formatted=sample_phrases_formatted
        )

    def estimate_tokens(self, prompt: str) -> int:
        """Estimate total tokens using adaptive strategy"""
        actual_input_tokens = len(self.encoding.encode(prompt))

        if self.first_prompt_tokens is None:
            self.first_prompt_tokens = actual_input_tokens
            estimated_input = int(actual_input_tokens * 1.15)
        elif len(self.input_token_history) < 3:
            estimated_input = int(actual_input_tokens * 1.15)
        else:
            avg_input = sum(self.input_token_history) / len(self.input_token_history)
            estimated_input = int(avg_input)

        if len(self.input_token_history) < 3:
            self.input_token_history.append(actual_input_tokens)

        if len(self.output_token_history) < 5:
            estimated_output = int(estimated_input * 0.25)
        else:
            avg_output = sum(self.output_token_history) / len(self.output_token_history)
            estimated_output = int(avg_output)

        estimated_output = min(self.config.max_tokens, estimated_output)

        total_estimate = estimated_input + estimated_output

        return total_estimate

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

    def get_token_bucket_status(self) -> dict:
        """Get current token bucket status"""
        available_pct = (self.tpm_bucket.available / self.tpm_bucket.tpm) * 100

        if len(self.actual_total_tokens) >= 10:
            recent_avg = sum(list(self.actual_total_tokens)[-10:]) / 10
            consumption_rate_per_sec = recent_avg / 2.0
            real_utilization_pct = (consumption_rate_per_sec / (self.tpm_bucket.tpm / 60)) * 100
        else:
            real_utilization_pct = 100 - available_pct
            consumption_rate_per_sec = 0

        return {
            "available_tokens": int(self.tpm_bucket.available),
            "capacity": self.tpm_bucket.tpm,
            "utilization_pct": real_utilization_pct,
            "low_tokens": available_pct < 10,
            "consumption_rate": consumption_rate_per_sec
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
        """Probe call without structured output for bootstrap measurement.

        Uses taxonomy-enriched prompt with default values.
        """
        # Store original values
        original_taxonomy_axis = self.taxonomy_axis
        original_taxonomy_axis_description = self.taxonomy_axis_description
        original_taxonomy_sample_phrases = self.taxonomy_sample_phrases
        original_generic_specifiers = self.generic_specifiers

        # Set defaults for bootstrap probing
        self.taxonomy_axis = self.taxonomy_axis or "WHAT"
        self.taxonomy_axis_description = self.taxonomy_axis_description or "general concepts and ideas"
        self.taxonomy_sample_phrases = self.taxonomy_sample_phrases or []
        if not self.generic_specifiers:
            self.generic_specifiers = {}

        placeholder_subject = "the subject"
        placeholder_phrasing_template = "the subject is [ATTRIBUTE_OR_ACTION]"

        prompt = self._build_taxonomy_enriched_prompt(
            task_dict['respondent_id'],
            task_dict['response'],
            placeholder_subject,
            placeholder_phrasing_template
        )

        # Restore original values
        self.taxonomy_axis = original_taxonomy_axis
        self.taxonomy_axis_description = original_taxonomy_axis_description
        self.taxonomy_sample_phrases = original_taxonomy_sample_phrases
        self.generic_specifiers = original_generic_specifiers

        resp = await asyncio.wait_for(
            llm_create_async(
                client=self.client,
                model=self.model,
                prompt=prompt,
                response_model=ProbeResponse,
                temperature=self.config.temperature,
                track_usage=False,
            ),
            timeout=BOOTSTRAP_TIMEOUT_SECONDS
        )

        u = getattr(resp, "_raw_response", None)
        if u:
            u = getattr(u, "usage", None)
        if not u:
            u = getattr(resp, "usage", None)
        input_tokens = getattr(u, "input_tokens", 0) or getattr(u, "prompt_tokens", 0)
        output_tokens = getattr(u, "output_tokens", 0) or getattr(u, "completion_tokens", 0)
        return {"prompt_tokens": input_tokens, "completion_tokens": output_tokens}

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
        """Process a single idea extraction task"""
        task_start = time.perf_counter()

        try:
            # Use taxonomy-aware subject extraction
            subject_response = await self._extract_taxonomy_aware_subject(
                self.var_lab,
                self.taxonomy_axis or "WHAT",
                self.secondary_axis
            )

            subject = subject_response.canonical_term
            phrasing_template = subject_response.canonical_phrasing

            template_prefix = phrasing_template.split('[ATTRIBUTE_OR_ACTION]')[0].strip() if '[ATTRIBUTE_OR_ACTION]' in phrasing_template else phrasing_template

            if self.template_prefix is None:
                self.template_prefix = template_prefix

            TaxonomyEnrichedIdeaResponse._expected_template_prefix = template_prefix

            # Build taxonomy-enriched prompt
            prompt = self._build_taxonomy_enriched_prompt(
                task['respondent_id'],
                task['response'],
                subject,
                phrasing_template
            )

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

            est_tokens = self.estimate_tokens(prompt)

            if task.get('task_index', 0) < 5:
                logger.info(f"[ESTIMATION DEBUG] Task {task.get('task_index', 0)}: estimated {est_tokens} tokens")

            self.stats['tasks_processed'] += 1

            timeout = self.latency_tracker.get_timeout(est_tokens)

            async with self.semaphore:
                await self.tpm_bucket.wait_and_acquire(est_tokens)
                async with self.rate_limiter:
                    response = await asyncio.wait_for(
                        llm_create_async(
                            client=self.client,
                            model=self.model,
                            response_model=List[TaxonomyEnrichedIdeaResponse],
                            prompt=prompt,
                            temperature=self.config.temperature,
                            max_tokens=self.config.max_tokens,
                            max_retries=3
                        ),
                        timeout=timeout
                    )

                    latency = time.perf_counter() - task_start
                    self.latency_tracker.add(latency)

                    usage = getattr(response, '_raw_response', None)
                    if usage:
                        usage = getattr(usage, 'usage', None)
                    if not usage:
                        usage = getattr(response, 'usage', None)

                    if usage:
                        actual_input_tokens = getattr(usage, 'input_tokens', 0) or getattr(usage, 'prompt_tokens', 0)
                        actual_output_tokens = getattr(usage, 'output_tokens', 0) or getattr(usage, 'completion_tokens', 0)
                        actual_total_tokens = getattr(usage, 'total_tokens', 0) or (actual_input_tokens + actual_output_tokens)

                        if len(self.input_token_history) < 3:
                            self.input_token_history.append(actual_input_tokens)
                        if len(self.output_token_history) < 5:
                            self.output_token_history.append(actual_output_tokens)

                        self.actual_total_tokens.append(actual_total_tokens)

                        estimation_error = abs(actual_total_tokens - est_tokens)
                        self.estimation_errors.append(estimation_error)

                        delta = actual_total_tokens - est_tokens
                        await self.tpm_bucket.reconcile(delta)

                    ideas = []
                    for i, idea_response in enumerate(response):
                        normalized = self._normalize_idea_text(idea_response.idea) if idea_response.idea else ""
                        if normalized and normalized not in ["", "NA", "N/A"]:
                            # Extract taxonomy_phrase, sentiment, sense as separate fields
                            taxonomy_phrase = getattr(idea_response, 'taxonomy_phrase', "") or ""
                            sentiment = getattr(idea_response, 'sentiment', "neutral") or "neutral"
                            sense = getattr(idea_response, 'sense', "factual") or "factual"

                            # Clean idea text (metadata stored in separate fields)
                            idea_text = self._format_idea_text(normalized)
                            response_idea_id = getattr(idea_response, 'idea_id', None) or str(i+1)
                            ideas.append(models.IdeasExtractedSubmodel(
                                idea_id=f"{task['respondent_id']}_{response_idea_id}",
                                idea=idea_text,
                                taxonomy_phrase=taxonomy_phrase,
                                sentiment=sentiment,
                                sense=sense
                            ))

                    if ideas:
                        self.stats['tasks_successful'] += 1
                        return models.IdeasExtractedModel(
                            respondent_id=task['respondent_id'],
                            response=task['response'],
                            quality_filter=task.get('quality_filter', True),
                            quality_filter_code=task.get('quality_filter_code', 0),
                            response_ideas=ideas,
                            idea_count=len(ideas),
                            template_prefix=self.template_prefix
                        )
                    else:
                        return self.create_fallback_response(task)

        except asyncio.TimeoutError:
            self.stats['timeouts'] += 1
            logger.warning(f"Task {task['respondent_id']} timed out")
            raise

        except RateLimitError:
            self.stats['rate_limits'] += 1
            logger.warning(f"Task {task['respondent_id']} hit rate limit")
            raise

        except Exception as e:
            logger.error(f"Task {task['respondent_id']} failed: {type(e).__name__}: {e}")
            raise

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
            idea_count=1,
            template_prefix=self.template_prefix
        )

    def _normalize_idea_text(self, text: str) -> str:
        if not text:
            return ""

        text = unicodedata.normalize('NFC', text)
        text = text.strip()
        text = ' '.join(text.split())

        zero_width_chars = ['\u200b', '\u200c', '\u200d', '\ufeff']
        for char in zero_width_chars:
            text = text.replace(char, '')

        return text

    def _format_idea_text(self, normalized_text: str) -> str:
        """Return clean idea text.

        Metadata (taxonomy_phrase, sentiment, sense) are stored as separate fields
        on the IdeasExtractedSubmodel.

        Args:
            normalized_text: The normalized idea text

        Returns:
            Clean idea text
        """
        return normalized_text

    def build_extraction_metadata(self, filename: str = "", var_name: str = "") -> 'models.ExtractionMetadata':
        """Build ExtractionMetadata from extracted context specifiers and taxonomy info.

        This creates a single metadata object that captures all extraction-level
        information that applies to the entire dataset (not per-idea).

        Args:
            filename: The source data filename
            var_name: The variable name being extracted

        Returns:
            ExtractionMetadata instance with all fields populated
        """
        from datetime import datetime

        return models.ExtractionMetadata(
            # File/variable info
            filename=filename,
            var_name=var_name,
            var_lab=self.var_lab,

            # Template
            template_prefix=self.template_prefix or "",

            # Context specifiers (6 fields)
            lang=self.generic_specifiers.get('lang', ''),
            domain=self.generic_specifiers.get('domain', ''),
            topic=self.generic_specifiers.get('topic', ''),
            perspective=self.generic_specifiers.get('perspective', ''),
            entity=self.generic_specifiers.get('entity', ''),
            intent=self.generic_specifiers.get('intent', ''),

            # Taxonomy axis info
            taxonomy_primary_axis=self.taxonomy_axis or '',
            taxonomy_secondary_axis=self.secondary_axis,
            taxonomy_rationale=self.taxonomy_rationale or '',
            taxonomy_axis_description=self.taxonomy_axis_description or '',
            taxonomy_sample_phrases=self.taxonomy_sample_phrases or [],

            # Timestamp
            extraction_timestamp=datetime.now().isoformat()
        )

    async def _fetch_rate_limits_from_api(self) -> RateLimits:
        """Make a minimal API call to fetch rate limits from response headers."""
        from openai import AsyncOpenAI
        from config import API_PROVIDER, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_NAME

        if API_PROVIDER == "azure":
            client = AsyncOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                base_url=f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}/",
                default_query={"api-version": "2024-10-21"},
            )
            model = AZURE_OPENAI_DEPLOYMENT_NAME
        else:
            client = AsyncOpenAI(api_key=OPENAI_API_KEY)
            model = self.model

        response = await client.chat.completions.with_raw_response.create(
            model=model,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5
        )

        return extract_rate_limits_from_response(response)

    def _initialize_rate_limiters(self, avg_latency_s: float, avg_tokens: int, limits, num_tasks: int) -> int:

        api_limits = ApiLimits(limits.tokens_per_minute, limits.requests_per_minute)
        little_law_concurrency = compute_optimal_concurrency(
            api_limits, avg_latency_s, avg_tokens,
            processing_config=self.processing_config,
            cap=self.processing_config.concurrency_cap_permissive,
            min_conc=self.processing_config.concurrency_min_permissive
        )
        max_concurrency = self.processing_config.concurrency_cap_default
        min_concurrency = self.processing_config.concurrency_min_default
        optimal = min(max_concurrency, max(little_law_concurrency, min_concurrency))

        arrival_rate = min(
            limits.requests_per_minute * self.processing_config.rate_limit_headroom / 60,
            limits.tokens_per_minute * self.processing_config.rate_limit_headroom / avg_tokens / 60)

        if arrival_rate < 1:
            self.rate_limiter = AsyncLimiter(1, time_period=1/arrival_rate)
        else:
            self.rate_limiter = AsyncLimiter(int(arrival_rate), time_period=1.0)

        self.semaphore = asyncio.Semaphore(min(num_tasks, optimal))
        self.optimal_concurrency = min(num_tasks, optimal)

        return optimal

    async def worker(self, queue: asyncio.Queue, results: List):
        """Worker coroutine that processes tasks from queue"""
        while True:
            task = None
            try:
                task = await queue.get()
                if task is None:
                    break

                task_index, task_data = task
                result = await self.process_task(task_data)
                results[task_index] = result

            except Exception as e:
                logger.error(f"Task failed after retries: {e}")
                self.stats['tasks_failed'] += 1
                if task is not None:
                    task_index, task_data = task
                    results[task_index] = self.create_fallback_response(task_data)
            finally:
                if task is not None:
                    queue.task_done()

    async def process_all_tasks_async(self, tasks: List[Dict]) -> List[models.IdeasExtractedModel]:
        """Process all tasks using queue + workers pattern with bootstrap measurement"""
        if not tasks:
            return []

        self.verbose_reporter.step_start("Idea Extraction", emoji="💡")

        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line("Fetching rate limits from API...")

        limits = await self._fetch_rate_limits_from_api()

        if limits.tokens_per_minute == 0 or limits.requests_per_minute == 0:
            if self.verbose_reporter.enabled:
                self.verbose_reporter.stat_line(f"Warning: Using fallback rate limits (TPM={FALLBACK_TPM}, RPM={FALLBACK_RPM})")
            limits = RateLimits(
                tokens_per_minute=FALLBACK_TPM,
                requests_per_minute=FALLBACK_RPM,
                tokens_per_day=FALLBACK_TPM * 60 * 24
            )
        else:
            if self.verbose_reporter.enabled:
                self.verbose_reporter.stat_line(f"Fetched from API: TPM={limits.tokens_per_minute:,}, RPM={limits.requests_per_minute:,}")

        self.rate_limits = limits

        sample_tasks = tasks[:min(3, len(tasks))]
        if len(sample_tasks) < 3:
            sample_tasks = sample_tasks * 3
            sample_tasks = sample_tasks[:3]

        conservative_latency = 2.0
        conservative_tokens = self.avg_tokens
        self._initialize_rate_limiters(conservative_latency, conservative_tokens, limits, num_tasks=20)
        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line(f"Initialized conservative rate limiters (latency={conservative_latency}s, tokens={conservative_tokens})\n")

        # Extract context specifiers AND taxonomy axis in parallel chunks
        self.verbose_reporter.stat_line("Extracting context specifiers and taxonomy axis...")
        self.generic_specifiers, taxonomy_result = await self._extract_generic_specifiers()

        # Store taxonomy axis info for use in idea extraction
        self.taxonomy_axis = taxonomy_result.primary_axis
        self.secondary_axis = taxonomy_result.secondary_axis
        self.taxonomy_rationale = taxonomy_result.primary_axis_rationale
        self.taxonomy_axis_description = taxonomy_result.primary_axis_description  # Dynamic context-specific description

        # Extract sample_phrases from the primary axis for dynamic examples in extraction prompt
        self.taxonomy_sample_phrases = []
        for axis_score in taxonomy_result.all_axis_scores:
            if axis_score.axis == self.taxonomy_axis and axis_score.sample_phrases:
                self.taxonomy_sample_phrases = axis_score.sample_phrases[:6]  # Limit to 6 examples
                break

        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line(f"\nTaxonomy axis selected: {self.taxonomy_axis}")
            if self.taxonomy_axis_description:
                self.verbose_reporter.stat_line(f"Description: {self.taxonomy_axis_description}")
            if self.secondary_axis:
                self.verbose_reporter.stat_line(f"Secondary axis: {self.secondary_axis}")
            if self.taxonomy_sample_phrases:
                self.verbose_reporter.stat_line(f"Sample phrases: {self.taxonomy_sample_phrases[:3]}...")

        # Generate taxonomy-aware subject and template
        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line("\nGenerating taxonomy-aware phrasing template...")

        await self._extract_taxonomy_aware_subject(
            self.var_lab,
            self.taxonomy_axis,
            self.secondary_axis
        )

        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line("\nRunning bootstrap measurement (3 probe calls)...")

        start_time = time.time()
        task_cycle = itertools.cycle(sample_tasks)

        async def probe_with_different_tasks():
            return await self.probe_call_no_structured(next(task_cycle))

        avg_latency_s, avg_tokens = await bootstrap_measure_async(probe_with_different_tasks, n_probes=3)

        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line(f"Probe time: {time.time() - start_time:.3f}s")
            self.verbose_reporter.stat_line(f"Bootstrap results: {avg_latency_s:.3f}s avg latency, {avg_tokens:.0f} avg tokens")

        for i in range(3):
            self.latency_tracker.add(avg_latency_s)

        self.avg_tokens = int(avg_tokens)

        optimal = self._initialize_rate_limiters(avg_latency_s, avg_tokens, limits, len(tasks))

        api_limits = ApiLimits(limits.tokens_per_minute, limits.requests_per_minute)
        little_law_concurrency = compute_optimal_concurrency(
            api_limits, avg_latency_s, avg_tokens,
            processing_config=self.processing_config,
            cap=self.processing_config.concurrency_cap_permissive,
            min_conc=self.processing_config.concurrency_min_permissive
        )

        print("\nRATE LIMITING SETUP - Bootstrap Optimized")
        print(f"- Model: {self.model}")
        print(f"- RPM limit: {limits.requests_per_minute:,} ({limits.requests_per_minute * self.processing_config.rate_limit_headroom:,.0f} with headroom)")
        print(f"- TPM limit: {limits.tokens_per_minute:,} ({limits.tokens_per_minute * self.processing_config.rate_limit_headroom:,.0f} with headroom)")
        print(f"- Bootstrap measured avg_tokens: {self.avg_tokens}")

        rpm_throughput = limits.requests_per_minute * self.processing_config.rate_limit_headroom / 60
        tpm_throughput = limits.tokens_per_minute * self.processing_config.rate_limit_headroom / self.avg_tokens / 60
        bottleneck = "RPM" if rpm_throughput < tpm_throughput else "TPM"
        print(f"- Expected throughput: {min(rpm_throughput, tpm_throughput):.1f}/s ({bottleneck} limited)")
        print(f"- Optimal by Little's law: {little_law_concurrency}")
        min_concurrency = self.processing_config.concurrency_min_default
        max_concurrency = self.processing_config.concurrency_cap_default
        print(f"- Constrained optimum: {optimal} (min={min_concurrency}, max={max_concurrency})")

        print(f"- Processing {len(tasks):,} tasks")

        expected_throughput = min(rpm_throughput, tpm_throughput)
        max_workers = self.processing_config.max_workers if hasattr(self.processing_config, 'max_workers') else 200
        min_workers = self.processing_config.min_workers if hasattr(self.processing_config, 'min_workers') else 50
        num_workers = min(max_workers, max(min_workers, int(expected_throughput * avg_latency_s * 2.0)))

        print(f"\nWorkers launched: (concurrent subroutines): {num_workers}")
        print(f"API calls in flight (concurrency ceiling/semaphore): {self.optimal_concurrency}")

        queue = asyncio.Queue()
        results = [None] * len(tasks)

        for i, task in enumerate(tasks):
            task['result_index'] = i
            task['task_index'] = i
            await queue.put((i, task))

        workers = []
        for _ in range(num_workers):
            w = asyncio.create_task(self.worker(queue, results))
            workers.append(w)

        start_time = time.time()
        last_report = start_time
        last_diagnostics = start_time

        while not queue.empty():
            await asyncio.sleep(1)
            now = time.time()

            if now - last_report >= PROGRESS_REPORT_INTERVAL:
                completed = self.stats['tasks_processed']
                remaining = queue.qsize()
                elapsed = now - start_time
                rate = completed / elapsed if elapsed > 0 else 0

                print(f"Progress: {completed}/{len(tasks)} ({completed/len(tasks)*100:.1f}%), "
                      f"Rate: {rate:.1f}/s, Queue: {remaining}")
                last_report = now

            if self.verbose_reporter.enabled and now - last_diagnostics >= DIAGNOSTIC_INTERVAL:
                bucket_status = self.get_token_bucket_status()
                token_stats = self.get_token_estimation_stats()

                if bucket_status['low_tokens']:
                    self.verbose_reporter.stat_line(f"⚠️ Token bucket low: {bucket_status['available_tokens']:,} tokens ({bucket_status['utilization_pct']:.1f}% utilized)")

                if token_stats['status'] == 'learning' and token_stats['samples'] >= 5:
                    self.verbose_reporter.stat_line(f"Token estimation: {token_stats['avg_estimation_error']:.0f} avg error, "
                                                  f"Input: {token_stats['avg_input_tokens']:.0f} avg ({token_stats['input_samples']}/3), "
                                                  f"Output: {token_stats['avg_output_tokens']:.0f} avg ({token_stats['output_samples']}/5)")

                last_diagnostics = now

        await queue.join()

        for _ in workers:
            await queue.put(None)
        await asyncio.gather(*workers)

        elapsed = time.time() - start_time
        print(f"\nCompleted {len(tasks)} tasks in {elapsed:.1f}s")
        print(f"- Successful: {self.stats['tasks_successful']}")
        print(f"- Failed: {self.stats['tasks_failed']}")
        print(f"- Rate limits: {self.stats['rate_limits']}")
        print(f"- Timeouts: {self.stats['timeouts']}")
        print(f"- Average: {elapsed/len(tasks):.2f}s/task")

        if self.verbose_reporter.enabled:
            token_stats = self.get_token_estimation_stats()
            bucket_status = self.get_token_bucket_status()

            if token_stats['status'] == 'learning':
                accuracy = max(0, 100 - (token_stats['avg_estimation_error'] / max(1, token_stats['avg_input_tokens'] + token_stats['avg_output_tokens']) * 100))
                self.verbose_reporter.stat_line(f"Token estimation accuracy: {accuracy:.1f}% (avg error: {token_stats['avg_estimation_error']:.0f} tokens)")
                self.verbose_reporter.stat_line(f"Learned averages - Input: {token_stats['avg_input_tokens']:.0f}, Output: {token_stats['avg_output_tokens']:.0f}")

                if token_stats['actual_samples'] >= 10:
                    actual_avg = token_stats['avg_actual_total_tokens']
                    initial_avg = token_stats['initial_avg_tokens']
                    difference = actual_avg - initial_avg

                    optimal_throughput = self.rate_limits.tokens_per_minute * self.processing_config.rate_limit_headroom / max(actual_avg, 1) / 60
                    initial_throughput = self.rate_limits.tokens_per_minute * self.processing_config.rate_limit_headroom / max(initial_avg, 1) / 60
                    pct_change = (difference / initial_avg * 100) if initial_avg > 0 else 0

                    self.verbose_reporter.stat_line(f"Token usage summary: Initial {initial_avg:.0f} → Actual {actual_avg:.0f} "
                                                  f"({difference:+.0f} tokens, {pct_change:+.1f}%)")
                    self.verbose_reporter.stat_line(f"Throughput analysis: Expected {initial_throughput:.1f}/s → Optimal {optimal_throughput:.1f}/s with perfect estimation")

        return results

    def extract(self) -> List[models.IdeasExtractedModel]:
        """Main method to extract ideas from responses using bootstrap measurement and unified processing"""
        self._stats.start_timing()
        self._stats.input_count = len(self.responses)

        if not self.responses:
            self.verbose_reporter.stat_line("No responses to process")
            return []

        tasks = []
        for response in self.responses:
            tasks.append({
                'respondent_id': response.respondent_id,
                'response': response.response,
                'quality_filter': response.quality_filter,
                'quality_filter_code': response.quality_filter_code
            })

        if nest_asyncio:
            nest_asyncio.apply()
        self._results = asyncio.run(self.process_all_tasks_async(tasks))

        self._stats.output_count = len(self._results)
        self._stats.end_timing()

        unique_ideas = set()
        multi_idea_responses = 0
        total_idea_length = 0
        idea_count = 0

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

                if valid_ideas and len(response_examples) < self.config.max_code_examples:
                    response_examples.append({
                        'response': resp.response,
                        'ideas': valid_ideas
                    })

        self.verbose_reporter.stat_line(f"Total responses processed: {len(self._results)}")
        self.verbose_reporter.stat_line(f"Total ideas extracted: {idea_count}")
        self.verbose_reporter.stat_line(f"Unique ideas identified: {len(unique_ideas)}")
        if multi_idea_responses > 0:
            single_idea_responses = len([r for r in self._results if r.response_ideas and len(r.response_ideas) == 1])
            self.verbose_reporter.stat_line(f"Single idea responses: {single_idea_responses} ({single_idea_responses/len(self._results)*100:.1f}%)")
            self.verbose_reporter.stat_line(f"Multiple idea responses: {multi_idea_responses} ({multi_idea_responses/len(self._results)*100:.1f}%)")

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

        if response_examples:
            print("\n📋 Sample extracted ideas:")
            for example in response_examples:
                print(f'  • "{example["response"]}"')
                for idea in example['ideas']:
                    cleaned_idea = re.sub(r"\[.*?\]", "", idea)
                    cleaned_idea = re.sub(r"\s+", " ", cleaned_idea).strip()
                    print(f'    → "{cleaned_idea}"')
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
