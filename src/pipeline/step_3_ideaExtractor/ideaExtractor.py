
"""
IdeaExtractor — dimension-based idea extraction from survey responses.

Step-3-specific business logic:
- Context extraction (language, sector, perspective, intent, entity, topic)
- Primary dimension selection (10 MECE dimensions via decision tree)
- Domain discovery (5-15 MECE domains per dimension)
- Per-response idea extraction with abstraction ladder

Bulk processing is delegated to SmoothRequester (utils/smoothRequester.py),
which handles rate pacing, concurrency control, workers, monitoring, and retry.
"""

# === MODULES ========================================================================================================
import asyncio
import random
import statistics
import logging
from typing import Dict, List, Optional, Tuple
from collections import deque

import nest_asyncio
from aiolimiter import AsyncLimiter

logger = logging.getLogger(__name__)

# === MODELS ========================================================================================================
import models

# === CONFIG ========================================================================================================
from config import DEFAULT_LANGUAGE, ModelConfig, ProcessingConfig, DEFAULT_PROCESSING_CONFIG, FALLBACK_TPM, FALLBACK_RPM, get_reasoning_params
from pipeline.step_3_ideaExtractor.config_ideaExtractor import IdeaExtractionConfig, DEFAULT_IDEA_EXTRACTION_CONFIG
from utils.llm import create_client, llm_create_async, RateLimits, fetch_rate_limits, token_tracker
from utils.perfModel import perf_model

# === PROMPTS (builders + response models) =========================================================================
from pipeline.step_3_ideaExtractor.prompts_ideaExtractor import (
    STANDING_BARE_KEY,
    STANDING_OTHER_KEY,
    build_context_specifier_group1_prompt,
    build_context_specifier_group2_prompt,
    build_consolidate_specifiers_group1_prompt,
    build_consolidate_specifiers_group2_prompt,
    build_primary_dimension_decision_tree_prompt,
    build_primary_dimension_consolidation_prompt,
    build_domain_discovery_prompt,
    build_domain_consolidation_prompt,
    build_taxonomy_enriched_extraction_prompt,
    GenericSpecifierGroup1Response,
    GenericSpecifierGroup2Response,
    PrimaryDimensionChunkResponse,
    PrimaryDimensionConsolidatedResponse,
    DomainItem,
    DomainChunkResponse,
    DomainConsolidatedResponse,
    create_extraction_model,
    consolidate_primary_dimension_by_majority,
    build_orthogonalize_domains_prompt,
    ReformulatedDomains,
    build_standing_labels_prompt,
    StandingLabelsResponse,
)

# === DIMENSION DATA ===============================================================================================
from pipeline.step_3_ideaExtractor.dimension_data import get_dimension, DimensionDefinition

# === STEP-SPECIFIC CONFIG =============================================================================================
from pipeline.step_3_ideaExtractor.config_ideaExtractor import (
    DEFAULT_TOKEN_HISTORY_CONFIG,
    DEFAULT_TIKTOKEN_OFFSET_CONFIG,
    DEFAULT_TIMEOUT_CONFIG,
    DEFAULT_BOOTSTRAP_CONFIG,
    DEFAULT_WARM_UP_CONFIG,
    DEFAULT_SPECIFIER_CONFIG,
    DEFAULT_DOMAIN_DISCOVERY_CONFIG,
)

# === SMOOTH REQUESTER (orchestrator for bulk API processing) ==========================================
from utils.smoothRequester import (
    SmoothRequester,
    # Building blocks used for the context extraction phases (1-3)
    TokenBucket, ConcurrencyGate, LatencyTracker, TiktokenOffsetLearner,
)

# === UTILS ========================================================================================================
from utils.verboseReporter import VerboseReporter, ProcessingStats
from utils.cachedResources import get_tiktoken_encoding
from utils.embedder import SharedEmbedder, find_representative_samples

# Post-extraction domain orthogonalization (one-shot reformulation; no reassignment)
ENABLE_DOMAIN_ORTHOGONALIZE = True
ORTHOGONALIZE_TOP_N = 8   # representative exemplars per domain fed to the reformulation



# === CONSTANTS =========================================================================
# Token history windows
INPUT_HISTORY_MAXLEN = DEFAULT_TOKEN_HISTORY_CONFIG.input_history_maxlen
OUTPUT_HISTORY_MAXLEN = DEFAULT_TOKEN_HISTORY_CONFIG.output_history_maxlen
OUTPUT_RATIO_HISTORY_MAXLEN = DEFAULT_TOKEN_HISTORY_CONFIG.output_ratio_history_maxlen
DEFAULT_OUTPUT_RATIO = DEFAULT_TOKEN_HISTORY_CONFIG.default_output_ratio
ERROR_WINDOW_SIZE = DEFAULT_TOKEN_HISTORY_CONFIG.error_window_size

# Tiktoken offset (for context extraction token estimation)
TIKTOKEN_API_OFFSET_DEFAULT = DEFAULT_TIKTOKEN_OFFSET_CONFIG.api_offset_default

# Timeouts (for context extraction rate limiting)
TIMEOUT_FLOOR_SECONDS = DEFAULT_TIMEOUT_CONFIG.timeout_floor_seconds
DEFAULT_TIMEOUT_SECONDS = DEFAULT_TIMEOUT_CONFIG.default_timeout_seconds
DEFAULT_LATENCY_SECONDS = DEFAULT_TIMEOUT_CONFIG.default_latency_seconds

# Bootstrap (for initial token estimation)
DEFAULT_AVG_TOKENS = DEFAULT_BOOTSTRAP_CONFIG.default_avg_tokens
SAMPLE_SIZE_FOR_TOKEN_ESTIMATION = DEFAULT_BOOTSTRAP_CONFIG.sample_size_for_token_estimation

# Sampling seed for the context-specifier sample. Unseeded, two runs on the same
# data drew different responses, so any comparison between runs measured the draw
# rather than the change. Seeded, the sample is a property of the dataset.
SAMPLING_SEED = 20260728

# Generic specifier settings (context extraction phases 1-2)
GENERIC_SPECIFIER_SAMPLE_MIN = DEFAULT_SPECIFIER_CONFIG.sample_min
GENERIC_SPECIFIER_SAMPLE_MAX = DEFAULT_SPECIFIER_CONFIG.sample_max
GENERIC_SPECIFIER_CHUNK_SIZE = DEFAULT_SPECIFIER_CONFIG.chunk_size
MAX_SPECIFIER_WORKERS = DEFAULT_SPECIFIER_CONFIG.max_workers

# Domain discovery (phase 3) — reads EVERY response, not the specifier sample.
DOMAIN_CHUNK_SIZE_MIN = DEFAULT_DOMAIN_DISCOVERY_CONFIG.chunk_size_min
DOMAIN_CHUNK_SIZE_MAX = DEFAULT_DOMAIN_DISCOVERY_CONFIG.chunk_size_max
DOMAIN_TARGET_CHUNKS = DEFAULT_DOMAIN_DISCOVERY_CONFIG.target_chunks
DOMAIN_CHUNK_OVERLAP = DEFAULT_DOMAIN_DISCOVERY_CONFIG.chunk_overlap









# === MAIN IDEA EXTRACTOR CLASS ========================================================================================================
class IdeaExtractor:
    def __init__(
        self,
        responses: List[models.QualityFilteredModel],
        var_lab: str,
        config: Optional[IdeaExtractionConfig] = None,
        model_config: Optional[ModelConfig] = None,
        processing_config: Optional[ProcessingConfig] = None,
        verbose: bool = False,
        prompt_printer=None,
        verbose_reporter: Optional['VerboseReporter'] = None,
        discover_domains: bool = False,
        cost_tracker=None):

        self.responses = responses
        self.var_lab = var_lab
        self.cost_tracker = cost_tracker
        self.config = config or DEFAULT_IDEA_EXTRACTION_CONFIG
        self.model_config = model_config or ModelConfig()  # kept for backward compat
        self.processing_config = processing_config or DEFAULT_PROCESSING_CONFIG
        self.warm_up_config = DEFAULT_WARM_UP_CONFIG
        # Per-stage models
        self.model_context = self.config.model_context
        self.model_taxonomy = self.config.model_taxonomy
        self.model_abstraction_ladder = self.config.model_abstraction_ladder
        self.model = self.model_abstraction_ladder  # primary model for rate limiting + backward compat
        self.language = DEFAULT_LANGUAGE
        self._results: List[models.IdeasExtractedModel] = []
        self.verbose_reporter = verbose_reporter or VerboseReporter(verbose, capture_logging=True)
        self._stats = ProcessingStats()
        self.prompt_printer = prompt_printer
        self._captured_prompt = False
        # Capture flags for each prompt type (only capture first instance)
        self._captured_context_specifier1 = False
        self._captured_context_specifier2 = False
        self._captured_taxonomy_chunk = False
        self._captured_consolidate1 = False
        self._captured_consolidate2 = False
        self._captured_taxonomy_consolidation = False
        self._captured_domain_chunk = False
        # Seeded RNG for every sample this step draws. Instance-level so a
        # second extractor in the same process starts from the same point.
        self._rng = random.Random(SAMPLING_SEED)
        self._captured_domain_consolidation = False
        self._captured_domain_orthogonalize = False
        # Initialize tokenizer for token estimation (cached)
        self.encoding = get_tiktoken_encoding(self.model)

        # Initialize OpenAI clients per stage (supports OpenAI and Azure)
        # capture_headers=True enables HeaderCaptureTransport for residual latency tracking
        unique_models = {self.model_context, self.model_taxonomy, self.model_abstraction_ladder}
        self._clients = {m: create_client(m, async_mode=True, capture_headers=True) for m in unique_models}
        self.client = self._clients[self.model_abstraction_ladder]  # backward compat for bulk path
        self._header_transport = getattr(self.client, '_header_transport', None)

        # Register model config with cost tracker
        if self.cost_tracker:
            self.cost_tracker.set_step_models("step_3_idea_extraction", {
                "context": self.model_context,
                "taxonomy": self.model_taxonomy,
                "abstraction_ladder": self.model_abstraction_ladder,
            })

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
        self.output_ratio_history = deque(maxlen=OUTPUT_RATIO_HISTORY_MAXLEN)  # Track output/input ratios
        self.estimation_errors = deque(maxlen=ERROR_WINDOW_SIZE)

        # Rolling average of actual total tokens
        self.actual_total_tokens = deque(maxlen=ERROR_WINDOW_SIZE)

        # Warm-start prediction from historical performance stats
        self._pred = perf_model.predict(self.model, "step3_idea_extraction")

        # Latency tracking (use predicted timeout as floor/default if available)
        self.latency_tracker = LatencyTracker(
            ema_alpha=self.processing_config.latency_tracker_ema_alpha,
            samples_window=self.processing_config.latency_tracker_samples_window,
            timeout_floor=self._pred.timeout_s or TIMEOUT_FLOOR_SECONDS,
            default_timeout=self._pred.timeout_s or DEFAULT_TIMEOUT_SECONDS,
        )

        # Generic specifiers (must be initialized before _calculate_avg_tokens)
        self.generic_specifiers = {}

        # Taxonomy dimension (must be initialized before _calculate_avg_tokens)
        self.primary_dimension = None
        self.primary_dimension_rationale = None
        self.primary_dimension_description = None  # Dynamic context-specific description
        self.decision_tree_stop_position = 0   # Which decision tree step triggered selection
        # Template prefix for embedding (V3: restored for normalized clustering)
        self.template_prefix = None

        # Phase 3 toggle: True = discover domains upfront; False = on-the-fly
        self.discover_domains = discover_domains

        # Calculate initial average tokens estimate
        # Prefer predicted avg_tokens (warm-up will recalibrate for this dataset)
        if self._pred.avg_tokens is not None:
            self.avg_tokens = self._pred.avg_tokens
        else:
            self.avg_tokens = self._calculate_avg_tokens()

        # Rate limiting for context extraction phases (1-3)
        # These are simple components — the full rate/concurrency control is in SmoothRequester
        self.rate_limiter = None
        self.semaphore = None
        self.tpm_bucket = None
        self.tiktoken_offset_learner = TiktokenOffsetLearner(default_offset=self._pred.tiktoken_offset)

        # Stats (populated by SmoothRequester during bulk processing)
        self.stats = {
            'tasks_processed': 0,
            'tasks_successful': 0,
            'tasks_failed': 0,
            'timeouts': 0,
            'rate_limits': 0,
            'empty_ladder_ideas': 0,
        }
        self.optimal_concurrency = None
        self.failed_task_ids: set = set()
        self.failure_log = []

    def _get_client_and_model(self, stage: str) -> tuple:
        """Return (client, model) for a given stage."""
        if stage == "context":
            m = self.model_context
        elif stage == "taxonomy":
            m = self.model_taxonomy
        elif stage == "abstraction_ladder":
            m = self.model_abstraction_ladder
        else:
            raise ValueError(f"Unknown stage: {stage}")
        return self._clients[m], m

    def _calculate_avg_tokens(self) -> int:
        """Calculate average tokens per request for rate limiting.

        V3: Uses placeholder template values for estimation.
        """
        if not self.responses:
            return DEFAULT_AVG_TOKENS

        sample_size = min(SAMPLE_SIZE_FOR_TOKEN_ESTIMATION, len(self.responses))
        sample_responses = self.responses[:sample_size]

        # Store original values to restore after estimation
        original_primary_dimension = self.primary_dimension
        original_primary_dimension_description = self.primary_dimension_description
        original_generic_specifiers = self.generic_specifiers

        # Set placeholder values for token estimation
        self.primary_dimension = "ATTRIBUTES_ASSOCIATIONS"
        self.primary_dimension_description = "general concepts and ideas"
        self.generic_specifiers = {
            "lang": "nl-NL",
            "perspective": "consumer",
            "intent": "evaluate",
            "domain": "general",
            "topic": "feedback",
            "entity": "unknown",
        }

        token_counts = []
        for response in sample_responses:
            prompt = self._build_taxonomy_enriched_prompt(response.response)
            prompt_tokens = len(self.encoding.encode(prompt))
            completion_tokens = int(prompt_tokens * 0.25)
            token_counts.append(prompt_tokens + completion_tokens)

        # Restore original values
        self.primary_dimension = original_primary_dimension
        self.primary_dimension_description = original_primary_dimension_description
        self.generic_specifiers = original_generic_specifiers

        return int(statistics.mean(token_counts)) if token_counts else DEFAULT_AVG_TOKENS

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
                    f"  - sector: {response_obj.sector}\n"
                    f"  - topic: {response_obj.topic}\n"
                    f"  - entity: {response_obj.entity}"
                )

        chunk_results_text = "\n\n".join(formatted_results)

        if group == 1:
            prompt = build_consolidate_specifiers_group1_prompt(
                survey_question=self.var_lab,
                chunk_results=chunk_results_text,
            )
            response_model = GenericSpecifierGroup1Response
            if self.prompt_printer and not self._captured_consolidate1:
                self.prompt_printer.capture_prompt(
                    step_name="idea_extraction_consolidate_specifiers",
                    utility_name="IdeaExtractor",
                    prompt_content=prompt,
                    prompt_type="consolidate_specifiers_group1",
                    metadata={"model": self._get_client_and_model("context")[1], "survey_question": self.var_lab}
                )
                self._captured_consolidate1 = True
        else:
            prompt = build_consolidate_specifiers_group2_prompt(
                survey_question=self.var_lab,
                chunk_results=chunk_results_text,
            )
            response_model = GenericSpecifierGroup2Response
            if self.prompt_printer and not self._captured_consolidate2:
                self.prompt_printer.capture_prompt(
                    step_name="idea_extraction_consolidate_specifiers",
                    utility_name="IdeaExtractor",
                    prompt_content=prompt,
                    prompt_type="consolidate_specifiers_group2",
                    metadata={"model": self._get_client_and_model("context")[1], "survey_question": self.var_lab}
                )
                self._captured_consolidate2 = True

        client, model = self._get_client_and_model("context")
        est_tokens = self._estimate_preprocessed_tokens(prompt)
        async with self.semaphore:
            await self.tpm_bucket.wait_and_acquire(est_tokens)
            await self.rate_limiter.acquire()

            response = await llm_create_async(
                client=client,
                model=model,
                response_model=response_model,
                prompt=prompt,
                temperature=0.0,
                **get_reasoning_params(model, phase="idea_extraction_context"),
            )

        if hasattr(self, 'verbose_reporter') and self.verbose_reporter.enabled:
            if group == 1:
                self.verbose_reporter.stat_line(
                    f"    Consolidated: lang={response.lang}, perspective={response.perspective}, intent={response.intent}"
                )
            else:
                self.verbose_reporter.stat_line(
                    f"    Consolidated: sector={response.sector}, topic={response.topic}, entity={response.entity}"
                )

        if group == 1:
            return {
                "lang": response.lang,
                "perspective": response.perspective,
                "intent": response.intent
            }
        else:
            return {
                "domain": response.sector,
                "topic": response.topic,
                "entity": response.entity
            }

    async def _consolidate_primary_dimension(
        self,
        chunk_results: List[Dict],
        context_specifiers: Dict,
        sample_responses: Optional[List] = None,
    ) -> PrimaryDimensionConsolidatedResponse:
        """Consolidate primary dimension selection from chunks.

        Uses majority rule when >50% of chunks agree. Falls back to LLM
        consolidation with actual response data when there is no majority.

        Args:
            chunk_results: List of dicts with 'response' containing PrimaryDimensionChunkResponse
            context_specifiers: Dict with domain, entity, topic, perspective, intent
            sample_responses: Response objects for tie-breaking (used when no majority)

        Returns:
            PrimaryDimensionConsolidatedResponse with selected dimension and description
        """
        # Try majority rule first
        chunk_response_objects = [r['response'] for r in chunk_results]
        majority_result = consolidate_primary_dimension_by_majority(chunk_response_objects)

        if majority_result is not None:
            if self.verbose_reporter.enabled:
                self.verbose_reporter.stat_line(f"  Primary dimension: majority rule -> {majority_result.primary_dimension}")
                self.verbose_reporter.stat_line(f"    {majority_result.primary_dimension_rationale}")
            return majority_result

        # No majority — run LLM consolidation with response data for grounding
        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line(f"  Primary dimension: no majority, running LLM consolidation with response sample")

        # Format chunk results for consolidation prompt
        formatted_results = []
        for idx, result in enumerate(chunk_results):
            chunk_response = result['response']
            evidence_text = "\n".join([f'    - "{e}"' for e in chunk_response.evidence])
            stop_pos = getattr(chunk_response, 'decision_tree_stop_position', 0)
            chunk_text = f"""Chunk {idx + 1}:
  Primary dimension: {chunk_response.primary_dimension} (decision tree step {stop_pos})
  Evidence:
{evidence_text}
  Clarification: {chunk_response.clarification}"""
            formatted_results.append(chunk_text)

        # Build response sample for tie-breaking
        chunk_responses_text = ""
        if sample_responses:
            grounding_sample = self._rng.sample(
                sample_responses,
                min(GENERIC_SPECIFIER_CHUNK_SIZE, len(sample_responses))
            )
            chunk_responses_text = "\n".join([f"- {r.response}" for r in grounding_sample])

        prompt = build_primary_dimension_consolidation_prompt(
            language=self.language,
            survey_question=self.var_lab,
            sector=context_specifiers['domain'],
            entity=context_specifiers['entity'],
            topic=context_specifiers['topic'],
            perspective=context_specifiers['perspective'],
            intent=context_specifiers['intent'],
            chunk_results="\n\n".join(formatted_results),
            chunk_responses=chunk_responses_text,
        )

        # Capture first taxonomy consolidation prompt
        if self.prompt_printer and not self._captured_taxonomy_consolidation:
            self.prompt_printer.capture_prompt(
                step_name="idea_extraction_taxonomy_consolidation",
                utility_name="IdeaExtractor",
                prompt_content=prompt,
                prompt_type="dimension_consolidation",
                metadata={"model": self._get_client_and_model("context")[1], "survey_question": self.var_lab, "language": self.language}
            )
            self._captured_taxonomy_consolidation = True

        client, model = self._get_client_and_model("context")
        est_tokens = self._estimate_preprocessed_tokens(prompt)
        async with self.semaphore:
            await self.tpm_bucket.wait_and_acquire(est_tokens)
            await self.rate_limiter.acquire()

            response = await llm_create_async(
                client=client,
                model=model,
                response_model=PrimaryDimensionConsolidatedResponse,
                prompt=prompt,
                temperature=0.0,
                **get_reasoning_params(model, phase="idea_extraction_context"),
            )

        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line(f"  Taxonomy consolidated (LLM):")
            self.verbose_reporter.stat_line(f"    Primary: {response.primary_dimension}")
            self.verbose_reporter.stat_line(f"    Rationale: {response.primary_dimension_rationale[:100]}...")

        return response


    @staticmethod
    def build_domain_chunks(responses: List) -> List[List]:
        """Chunk EVERY response for domain discovery, with overlap.

        Not the specifier sample: reading a dataset's properties from a fifth of the
        responses is sound, finding which themes exist is not. A theme that misses
        the draw gets no domain, and its ideas then fall through to 'Other' for the
        whole dataset. Same treatment step 4 gives facet discovery one level down.

        Static and public so `exp_consolidation_variance.py` chunks exactly the way
        production does, rather than keeping a copy that can drift.
        """
        n = len(responses)
        if n <= DOMAIN_CHUNK_SIZE_MIN:
            return [list(responses)]

        size = max(DOMAIN_CHUNK_SIZE_MIN,
                   min(max(n // DOMAIN_TARGET_CHUNKS, 1), DOMAIN_CHUNK_SIZE_MAX))
        step = max(size - int(size * DOMAIN_CHUNK_OVERLAP), 1)

        chunks, i = [], 0
        while i < n:
            chunks.append(list(responses[i:i + size]))
            i += step
            if i < n and i + size > n:
                chunks.append(list(responses[-size:]))
                break
        return chunks

    async def _consolidate_domains(self, chunk_results: List[Dict], context_specifiers: Dict, sample_responses: Optional[List] = None) -> DomainConsolidatedResponse:
        """Consolidate chunk-level domain discoveries into a single set."""
        dimension = get_dimension(self.primary_dimension)

        # Grounding sample of real responses (RC-6): judge distinctness against data, not labels
        chunk_responses_text = ""
        if sample_responses:
            grounding_sample = self._rng.sample(
                sample_responses,
                min(GENERIC_SPECIFIER_CHUNK_SIZE, len(sample_responses))
            )
            chunk_responses_text = "\n".join([f"- {r.response}" for r in grounding_sample])

        # Format chunk results for the consolidation prompt
        formatted_results = []
        for idx, result in enumerate(chunk_results):
            chunk_response = result['response']
            cats_text = "\n".join([
                f'    - "{c.label}" — {c.definition}'
                for c in chunk_response.domains
            ])
            chunk_text = f"""Chunk {idx + 1}:
  Domains:
{cats_text}"""
            formatted_results.append(chunk_text)

        prompt = build_domain_consolidation_prompt(
            language=self.language,
            survey_question=self.var_lab,
            sector=context_specifiers['domain'],
            entity=context_specifiers['entity'],
            topic=context_specifiers['topic'],
            perspective=context_specifiers['perspective'],
            intent=context_specifiers['intent'],
            primary_dimension=self.primary_dimension,
            chunk_results="\n\n".join(formatted_results),
            dimension=dimension,
            chunk_responses=chunk_responses_text,
        )

        if self.prompt_printer and not self._captured_domain_consolidation:
            self.prompt_printer.capture_prompt(
                step_name="idea_extraction_domains",
                utility_name="IdeaExtractor",
                prompt_content=prompt,
                prompt_type="domain_consolidation",
                metadata={"model": self._get_client_and_model("taxonomy")[1], "survey_question": self.var_lab, "language": self.language}
            )
            self._captured_domain_consolidation = True

        client, model = self._get_client_and_model("taxonomy")
        est_tokens = self._estimate_preprocessed_tokens(prompt)
        async with self.semaphore:
            await self.tpm_bucket.wait_and_acquire(est_tokens)
            await self.rate_limiter.acquire()

            response = await llm_create_async(
                client=client,
                model=model,
                response_model=DomainConsolidatedResponse,
                prompt=prompt,
                temperature=0.0,
                **get_reasoning_params(model, phase="idea_extraction_taxonomy"),
            )

        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line(f"  Domains consolidated:")
            for c in response.domains:
                self.verbose_reporter.stat_line(f"    {c.label}")

        return response

    async def _extract_generic_specifiers(self) -> Tuple[Dict[str, str], PrimaryDimensionConsolidatedResponse, DomainConsolidatedResponse]:
        """Extract context specifiers first, then primary dimension with context awareness, then domains.

        Two-phase extraction:
        - Phase 1: Extract context specifiers (Group 1 + Group 2) in parallel
        - Phase 2: Extract taxonomy axis scoring with context specifiers available

        Returns:
            Tuple of (context_specifiers dict, PrimaryDimensionConsolidatedResponse, DomainConsolidatedResponse)
        """
        sample_size = min(GENERIC_SPECIFIER_SAMPLE_MAX, max(int(0.2 * len(self.responses)), GENERIC_SPECIFIER_SAMPLE_MIN))
        sample = self._rng.sample(self.responses, min(sample_size, len(self.responses)))

        chunk_size = GENERIC_SPECIFIER_CHUNK_SIZE
        chunks = [sample[i:i+chunk_size] for i in range(0, len(sample), chunk_size)]
        chunk_texts = ["\n".join([f"- {r.response}" for r in chunk]) for chunk in chunks]

        self.verbose_reporter.stat_line(f"Context + Taxonomy: {len(sample)} samples, {len(chunks)} chunks")

        # === PHASE 1: Context specifiers (parallel) ===
        self.verbose_reporter.stat_line(f"  Phase 1: Extracting context specifiers...")

        context_tasks = []
        for chunk_idx, chunk in enumerate(chunks):
            # Group 1: lang/perspective/intent
            context_tasks.append({
                'task_id': f"group1_chunk{chunk_idx}",
                'group': 1,
                'chunk_idx': chunk_idx,
                'chunk_text': chunk_texts[chunk_idx],
                'chunk_size': len(chunk)
            })
            # Group 2: domain/topic/entity
            context_tasks.append({
                'task_id': f"group2_chunk{chunk_idx}",
                'group': 2,
                'chunk_idx': chunk_idx,
                'chunk_text': chunk_texts[chunk_idx],
                'chunk_size': len(chunk)
            })

        context_results = await self._process_generic_specifier_tasks(context_tasks)

        group1_results = [r for r in context_results if r['group'] == 1]
        group2_results = [r for r in context_results if r['group'] == 2]

        if self.verbose_reporter.enabled and group1_results and group2_results:
            self.verbose_reporter.stat_line(f"  Phase 1 chunk-level results:")
            for r in group1_results:
                self.verbose_reporter.stat_line(
                    f"    Chunk {r['chunk_idx']+1} (Group1): "
                    f"lang={r['response'].lang}, perspective={r['response'].perspective}, intent={r['response'].intent}"
                )
            for r in group2_results:
                self.verbose_reporter.stat_line(
                    f"    Chunk {r['chunk_idx']+1} (Group2): "
                    f"sector={r['response'].sector}, topic={r['response'].topic}, entity={r['response'].entity}"
                )

        # Hard failure if context specifier extraction produced no results
        if not group1_results or not group2_results:
            raise RuntimeError(
                f"Context specifier extraction failed: "
                f"{len(group1_results)} group1 results, {len(group2_results)} group2 results. "
                f"Cannot proceed without context specifiers. "
                f"Check LLM connectivity, rate limits, and model availability."
            )

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
                "domain": group2_results[0]['response'].sector,
                "topic": group2_results[0]['response'].topic,
                "entity": group2_results[0]['response'].entity
            }
        else:
            group2_consolidated = await self._consolidate_specifiers(2, group2_results)

        context_specifiers = {**group1_consolidated, **group2_consolidated}
        self.verbose_reporter.stat_line(f"  Phase 1 complete. Context specifiers: {context_specifiers}")

        # === PHASE 2: Taxonomy scoring (with context awareness) ===
        self.verbose_reporter.stat_line(f"  Phase 2: Scoring taxonomy axes with context (perspective={context_specifiers.get('perspective')}, intent={context_specifiers.get('intent')})...")

        taxonomy_tasks = []
        for chunk_idx, chunk in enumerate(chunks):
            taxonomy_tasks.append({
                'task_id': f"taxonomy_chunk{chunk_idx}",
                'group': 3,
                'chunk_idx': chunk_idx,
                'chunk_text': chunk_texts[chunk_idx],
                'chunk_size': len(chunk),
                'context_specifiers': context_specifiers  # Pass context to taxonomy scoring
            })

        taxonomy_results = await self._process_generic_specifier_tasks(taxonomy_tasks)

        if self.verbose_reporter.enabled and taxonomy_results:
            self.verbose_reporter.stat_line(f"  Phase 2 chunk-level results:")
            for r in taxonomy_results:
                chunk_resp = r['response']
                stop_pos = getattr(chunk_resp, 'decision_tree_stop_position', '?')
                self.verbose_reporter.stat_line(
                    f"    Chunk {r['chunk_idx']+1} (Taxonomy): Dimension={chunk_resp.primary_dimension} (tree step {stop_pos})"
                )

        # Consolidate Taxonomy — hard failure if no results
        if not taxonomy_results:
            raise RuntimeError(
                "Primary dimension selection produced no results from any chunk. "
                "Check LLM connectivity, rate limits, and model availability."
            )
        taxonomy_consolidated = await self._consolidate_primary_dimension(taxonomy_results, context_specifiers, sample_responses=sample)

        self.verbose_reporter.stat_line(f"  Context results: {context_specifiers}")
        self.verbose_reporter.stat_line(f"  Taxonomy: primary={taxonomy_consolidated.primary_dimension}")

        # Set primary dimension early so Phase 3 domain discovery can use it
        self.primary_dimension = taxonomy_consolidated.primary_dimension
        self.primary_dimension_description = taxonomy_consolidated.primary_dimension_description
        # Capture the most common decision tree stop position from chunks
        stop_positions = [
            getattr(r['response'], 'decision_tree_stop_position', 0)
            for r in taxonomy_results
        ]
        self.decision_tree_stop_position = max(set(stop_positions), key=stop_positions.count) if stop_positions else 0

        # === PHASE 3: Domain discovery (optional) ===
        if self.discover_domains:
            self.verbose_reporter.stat_line(f"  Phase 3: Discovering domains from response data...")

            domain_chunks = self.build_domain_chunks(self.responses)
            domain_chunk_texts = [
                "\n".join([f"- {r.response}" for r in ch]) for ch in domain_chunks
            ]
            self.verbose_reporter.stat_line(
                f"    Reading ALL {len(self.responses)} responses "
                f"in {len(domain_chunks)} overlapping chunks"
            )

            category_tasks = []
            for chunk_idx, chunk in enumerate(domain_chunks):
                category_tasks.append({
                    'task_id': f"topical_cat_chunk{chunk_idx}",
                    'group': 4,
                    'chunk_idx': chunk_idx,
                    'chunk_text': domain_chunk_texts[chunk_idx],
                    'chunk_size': len(chunk),
                    'context_specifiers': context_specifiers
                })

            category_results = await self._process_generic_specifier_tasks(category_tasks)

            if self.verbose_reporter.enabled and category_results:
                self.verbose_reporter.stat_line(f"  Phase 3 chunk-level results:")
                for r in category_results:
                    chunk_resp = r['response']
                    cat_labels = [c.label for c in chunk_resp.domains]
                    self.verbose_reporter.stat_line(
                        f"    Chunk {r['chunk_idx']+1}: {len(cat_labels)} domains: {cat_labels}"
                    )

            # Consolidate domains — hard failure if no results
            if not category_results:
                raise RuntimeError(
                    "Domain discovery produced no results from any chunk. "
                    "Check LLM connectivity, rate limits, and model availability."
                )

            dimension = get_dimension(self.primary_dimension)
            # Runs alongside consolidation: it needs only the dimension and the
            # language, so it costs no wall-clock.
            labels_task = asyncio.create_task(
                self._translate_standing_labels(dimension, context_specifiers))

            if len(category_results) == 1:
                # Single chunk — use directly
                categories_consolidated = DomainConsolidatedResponse(
                    domains=category_results[0]['response'].domains
                )
            else:
                categories_consolidated = await self._consolidate_domains(
                    category_results, context_specifiers, sample_responses=sample)

            # The two standing domains join the discovered ones here, so every consumer
            # downstream — the assignment menu, the domain table, the persisted
            # metadata — sees a single list and needs no special case.
            standing = self._resolve_standing_domains(await labels_task, dimension)
            categories_consolidated.domains = list(categories_consolidated.domains) + standing

            self.verbose_reporter.stat_line(
                f"  Domains: {[c.label for c in categories_consolidated.domains]}")
            self.verbose_reporter.stat_line(
                f"    (standing: {[c.label for c in standing]})")
        else:
            self.verbose_reporter.stat_line(f"  Phase 3: Skipped (on-the-fly domains)")
            categories_consolidated = DomainConsolidatedResponse(domains=[])

        return context_specifiers, taxonomy_consolidated, categories_consolidated

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

                # === Build prompt BEFORE acquiring rate limit resources ===
                # Prompts are pure string construction — no I/O, no semaphore needed.
                if task['group'] == 1:
                    # Group 1: lang/perspective/intent
                    prompt = build_context_specifier_group1_prompt(
                        language=self.language,
                        survey_question=self.var_lab,
                        chunk_responses=task['chunk_text'],
                        chunk_size=task['chunk_size'],
                    )
                    response_model = GenericSpecifierGroup1Response
                    if self.prompt_printer and not self._captured_context_specifier1:
                        self.prompt_printer.capture_prompt(
                            step_name="idea_extraction_context_specifiers",
                            utility_name="IdeaExtractor",
                            prompt_content=prompt,
                            prompt_type="context_specifier_group1",
                            metadata={"model": self._get_client_and_model("context")[1], "survey_question": self.var_lab, "language": self.language}
                        )
                        self._captured_context_specifier1 = True
                elif task['group'] == 2:
                    # Group 2: domain/topic/entity
                    prompt = build_context_specifier_group2_prompt(
                        language=self.language,
                        survey_question=self.var_lab,
                        chunk_responses=task['chunk_text'],
                        chunk_size=task['chunk_size'],
                    )
                    response_model = GenericSpecifierGroup2Response
                    if self.prompt_printer and not self._captured_context_specifier2:
                        self.prompt_printer.capture_prompt(
                            step_name="idea_extraction_context_specifiers",
                            utility_name="IdeaExtractor",
                            prompt_content=prompt,
                            prompt_type="context_specifier_group2",
                            metadata={"model": self._get_client_and_model("context")[1], "survey_question": self.var_lab, "language": self.language}
                        )
                        self._captured_context_specifier2 = True
                elif task['group'] == 3:
                    # Group 3: Taxonomy dimension selection (decision tree, context-aware)
                    ctx = task['context_specifiers']
                    prompt = build_primary_dimension_decision_tree_prompt(
                        language=self.language,
                        survey_question=self.var_lab,
                        chunk_responses=task['chunk_text'],
                        chunk_size=task['chunk_size'],
                        perspective=ctx['perspective'],
                        intent=ctx['intent'],
                        sector=ctx['domain'],
                        entity=ctx['entity'],
                        topic=ctx['topic'],
                    )
                    response_model = PrimaryDimensionChunkResponse
                    if self.prompt_printer and not self._captured_taxonomy_chunk:
                        self.prompt_printer.capture_prompt(
                            step_name="idea_extraction_taxonomy_chunk",
                            utility_name="IdeaExtractor",
                            prompt_content=prompt,
                            prompt_type="dimension_chunk_decision_tree",
                            metadata={
                                "model": self._get_client_and_model("context")[1],
                                "survey_question": self.var_lab,
                                "language": self.language,
                                "perspective": ctx['perspective'],
                                "intent": ctx['intent'],
                            }
                        )
                        self._captured_taxonomy_chunk = True
                else:  # group == 4: Domain discovery
                    ctx = task['context_specifiers']

                    dimension = get_dimension(self.primary_dimension)
                    prompt = build_domain_discovery_prompt(
                        language=self.language,
                        survey_question=self.var_lab,
                        chunk_responses=task['chunk_text'],
                        chunk_size=task['chunk_size'],
                        perspective=ctx['perspective'],
                        intent=ctx['intent'],
                        sector=ctx['domain'],
                        entity=ctx['entity'],
                        topic=ctx['topic'],
                        primary_dimension=self.primary_dimension,
                        primary_dimension_description=self.primary_dimension_description,
                        dimension=dimension,
                    )
                    response_model = DomainChunkResponse
                    if self.prompt_printer and not self._captured_domain_chunk:
                        self.prompt_printer.capture_prompt(
                            step_name="idea_extraction_domains",
                            utility_name="IdeaExtractor",
                            prompt_content=prompt,
                            prompt_type="domain_chunk",
                            metadata={"model": self._get_client_and_model("taxonomy")[1], "survey_question": self.var_lab, "language": self.language}
                        )
                        self._captured_domain_chunk = True

                # Route to correct client based on group
                if task['group'] <= 3:
                    client, model = self._get_client_and_model("context")
                else:
                    client, model = self._get_client_and_model("taxonomy")

                # === Count tokens from actual prompt, then acquire ===
                est_tokens = self._estimate_preprocessed_tokens(prompt)

                async with self.semaphore:
                    await self.tpm_bucket.wait_and_acquire(est_tokens)
                    await self.rate_limiter.acquire()

                    response = await llm_create_async(
                        client=client,
                        model=model,
                        response_model=response_model,
                        prompt=prompt,
                        temperature=0.0,
                        **get_reasoning_params(model, phase="idea_extraction_context"),
                    )

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

    @staticmethod
    def build_domain_table(domains: Optional[List]) -> str:
        """The assignment menu as the extraction prompt shows it.

        Static and public so `exp_assignment_variance.py` presents the model exactly
        the menu production presents, rather than keeping a copy that can drift.
        """
        if not domains:
            # During token estimation (_calculate_avg_tokens), domains haven't been
            # discovered yet — a placeholder is enough for sizing.
            return "(domains will be discovered during extraction)"

        return (
            "Pick the single best-fitting domain. The ✓ test and ✗ list help you CHOOSE BETWEEN "
            "domains; they are not grounds to reject a plausibly related idea.\n"
            "There are two ways to get this wrong and they are equally bad:\n"
            "  - sending an idea to a catch-all domain that one of the others does name;\n"
            "  - forcing an idea into a domain whose subject the idea never mentions.\n"
            "Each domain lists its definition, ✓ a membership test, and ✗ neighbouring domains it should not be confused with:\n"
            + "\n".join(
                f"  • {c.label} = \"{c.definition}\"\n      ✓ {c.boundary_test}\n      ✗ {', '.join(c.exclusions)}"
                for c in domains
            )
        )

    def _build_taxonomy_enriched_prompt(self, response: str) -> str:
        """Build taxonomy-enriched prompt for idea extraction."""
        assert self.primary_dimension is not None, "primary_dimension must be set before building extraction prompt"
        dimension = get_dimension(self.primary_dimension)

        domain_table = self.build_domain_table(getattr(self, 'domains', None))

        return build_taxonomy_enriched_extraction_prompt(
            language=self.language,
            var_lab=self.var_lab,
            perspective=self.generic_specifiers['perspective'],
            sector=self.generic_specifiers['domain'],
            entity=self.generic_specifiers['entity'],
            topic=self.generic_specifiers['topic'],
            intent=self.generic_specifiers['intent'],
            response=response,
            dimension=dimension,
            domain_table=domain_table,
        )

    def _estimate_preprocessed_tokens(self, prompt: str) -> int:
        """Simple token estimate for pre-processing calls (non-adaptive).

        Used for context extraction, dimension selection, consolidation, and subject
        extraction — calls that don't need adaptive estimation.
        """
        tiktoken_count = len(self.encoding.encode(prompt))
        return int((tiktoken_count + TIKTOKEN_API_OFFSET_DEFAULT) * (1 + DEFAULT_OUTPUT_RATIO))

    def create_fallback_response(self, task: Dict, reason: str = "unknown") -> models.IdeasExtractedModel:
        """Create fallback response for failed tasks"""
        return models.IdeasExtractedModel(
            respondent_id=task['respondent_id'],
            response=task['response'],
            quality_filter=task.get('quality_filter', True),
            quality_filter_code=task.get('quality_filter_code', 0),
            response_ideas=[
                models.IdeasExtractedSubmodel(
                    idea_id=f"{task['respondent_id']}_1",
                    idea=f"PROCESSING_ERROR: {reason}"
                )
            ],
            idea_count=1,
            template_prefix=self.template_prefix or ""
        )

    def _build_prepare_fn(self):
        """Build the prepare function for SmoothRequester.

        Returns prompt + LLM call parameters. The smoothRequester makes the call.
        """
        extractor = self
        config = self.config

        def prepare_fn(task: Dict) -> Dict:
            """Build prompt and call parameters for one task."""
            dimension = get_dimension(extractor.primary_dimension)

            prompt = extractor._build_taxonomy_enriched_prompt(task['response'])

            if extractor.prompt_printer and not extractor._captured_prompt:
                extractor.prompt_printer.capture_prompt(
                    step_name="idea_extraction",
                    utility_name="IdeaExtractor",
                    prompt_content=prompt,
                    prompt_type="idea_extraction",
                    metadata={
                        "model": extractor.model_abstraction_ladder,
                        "var_lab": extractor.var_lab,
                        "respondent_id": task['respondent_id'],
                        "primary_dimension": extractor.primary_dimension,
                    }
                )
                extractor._captured_prompt = True

            AxisExtractionModel = create_extraction_model(
                dimension=dimension,
                domains=getattr(extractor, 'domains', None),
            )

            return {
                'prompt': prompt,
                'response_model': List[AxisExtractionModel],
                'temperature': config.temperature,
                'max_tokens': config.max_tokens,
                'max_retries': 3,
                'extra_kwargs': get_reasoning_params(extractor.model, phase="idea_extraction_abstraction_ladder"),
            }

        return prepare_fn

    def _build_parse_fn(self):
        """Build the parse function for SmoothRequester.

        Parses raw LLM response into IdeasExtractedModel.
        """
        extractor = self

        def parse_fn(task: Dict, response) -> models.IdeasExtractedModel:
            """Parse LLM response into ideas.

            idea is derived from instance — the LLM no longer returns an idea field.
            All ladder fields are enforced non-empty by the Pydantic model.

            A response with nothing to extract yields zero ideas, and that is a
            successful extraction, not a failure — every open question produces
            answers that carry no idea. Returning None here would be indistinguishable
            from a timeout inside the requester (both surface as a None result), so
            such a response used to cost a retry pass and then land in the data as
            `PROCESSING_ERROR: timeout`. Downstream reads `response_ideas or []`
            throughout, so an empty list needs no special case.
            """
            ideas = []
            for i, idea_response in enumerate(response):
                instance = idea_response.instance
                if not instance or instance in ("NA", "N/A"):
                    continue

                ideas.append(models.IdeasExtractedSubmodel(
                    idea_id=f"{task['respondent_id']}_{i+1}",
                    idea=instance,
                    instance=instance,
                    interpretation=idea_response.interpretation,
                    abstraction=idea_response.abstraction,
                    domain=idea_response.domain,
                ))

            return models.IdeasExtractedModel(
                respondent_id=task['respondent_id'],
                response=task['response'],
                quality_filter=task.get('quality_filter', True),
                quality_filter_code=task.get('quality_filter_code', 0),
                response_ideas=ideas,
                idea_count=len(ideas),
                template_prefix=extractor.template_prefix or ""
            )

        return parse_fn

    def _build_fallback_fn(self):
        """Build the fallback function for SmoothRequester."""
        extractor = self
        def fallback_fn(task: Dict, reason: str):
            return extractor.create_fallback_response(task, reason=reason)
        return fallback_fn

    def get_failure_report(self, total_responses: int = None) -> str:
        """Return a formatted report of all PROCESSING_ERROR failures."""
        total = total_responses or self.stats.get('tasks_processed', 0)
        n_failures = len(self.failure_log)

        if n_failures == 0:
            return f"PROCESSING ERRORS: 0 of {total} responses (0%)"

        lines = [f"PROCESSING ERRORS: {n_failures} of {total} responses ({n_failures/max(total,1)*100:.1f}%)"]

        # Group by reason
        from collections import Counter
        reason_counts = Counter()
        for f in self.failure_log:
            key = f['error_type'] if f['reason'] == 'exception' else f['reason']
            reason_counts[key] += 1

        lines.append(f"  Breakdown: {', '.join(f'{count}x {reason}' for reason, count in reason_counts.most_common())}")
        lines.append("")

        for f in self.failure_log:
            reason_str = f['error_type'] if f['reason'] == 'exception' else f['reason']
            preview = f['response_preview']
            lines.append(f"  Respondent {f['respondent_id']}: {reason_str} | \"{preview}...\"")

        return "\n".join(lines)

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
        return models.ExtractionMetadata(
            # File/variable info
            filename=filename,
            var_name=var_name,
            var_lab=self.var_lab,

            # Template
            template_prefix=self.template_prefix or "",

            # Context specifiers (6 fields)
            lang=self.generic_specifiers.get('lang', ''),
            sector=self.generic_specifiers.get('domain', ''),
            topic=self.generic_specifiers.get('topic', ''),
            perspective=self.generic_specifiers.get('perspective', ''),
            entity=self.generic_specifiers.get('entity', ''),
            intent=self.generic_specifiers.get('intent', ''),

            # Taxonomy (these should always be set by the time metadata is built)
            primary_dimension=self.primary_dimension or '',
            primary_dimension_description=self.primary_dimension_description or '',
            decision_tree_stop_position=self.decision_tree_stop_position,
            # Domains — persist the full boundary (used by the prototype + step 4's
            # DomainDescription, which otherwise re-derives a weaker boundary_test).
            domains=self._domains_metadata(),
        )

    def _domains_metadata(self) -> List[Dict]:
        """Domain metadata for ExtractionMetadata.

        self.domains includes the two standing domains (appended right after
        consolidation), so they are persisted like any other. Before that they were
        assignable but absent here: step 4 then found a domain with no definition and
        substituted the placeholder "Labels related to the domain 'Other'", which
        travelled downstream as if it were real. Empty ones are dropped later by
        prune_empty_nodes().
        """
        return [
            {
                "key": getattr(c, "key", "") or c.label,
                "label": c.label,
                "definition": c.definition,
                "boundary_test": getattr(c, "boundary_test", "") or "",
                "exclusions": list(getattr(c, "exclusions", []) or []),
            }
            for c in getattr(self, 'domains', []) or []
        ]

    @staticmethod
    def _partition_standing(domains) -> Tuple[List, List]:
        """Split a domain list into (discovered, standing), order preserved in each."""
        standing_keys = (STANDING_BARE_KEY, STANDING_OTHER_KEY)
        discovered = [d for d in domains if d.key not in standing_keys]
        standing = [d for d in domains if d.key in standing_keys]
        return discovered, standing

    @staticmethod
    def _merge_orthogonalized(new_discovered, discovered, standing) -> Tuple[Optional[List], Optional[Dict]]:
        """Reassemble the domain list after a re-description of the discovered ones.

        Returns (None, None) when the model did not return exactly one entry per
        DISCOVERED domain — counting against the full list would fire on every run
        now that the standing two are no longer returned, silently skipping the
        whole phase.
        """
        if len(new_discovered) != len(discovered):
            return None, None
        rename = {}
        for old, nd in zip(discovered, new_discovered):
            nd.key = old.key          # carry identity across the rebuild...
            rename[old.label] = nd.label
        IdeaExtractor._set_domain_keys(new_discovered)   # ...then re-derive it
        return list(new_discovered) + list(standing), rename

    @staticmethod
    def _set_domain_keys(domains) -> None:
        """Derive `key` from `label`, except for the two standing domains.

        The LLM no longer returns a key, so discovered domains take theirs from the
        label. The standing domains are the exception and must survive every rebuild:
        consumers identify them by the literal key — step 4's drain-domain skip
        (`taxonomy_health.DRAIN_KEYS`) — while a label is language-dependent and can be
        re-described by the orthogonalization pass.

        Every place that (re)builds DomainItems goes through here. It is a single
        function because it was two: `_orthogonalize_domains` guarded the standing keys
        while the normalization after consolidation overwrote them unconditionally, so
        the guard protected something already destroyed and the keys never reached the
        cache (fixed 2026-08-09; the same loss was fixed once before in 6404da8e).
        """
        for d in domains or []:
            if d.key not in (STANDING_BARE_KEY, STANDING_OTHER_KEY):
                d.key = d.label

    async def _translate_standing_labels(self, dimension: DimensionDefinition,
                                         context_specifiers: Dict):
        """One small call whose only job is naming the two standing domains.

        Separate from consolidation on purpose: a call that also has to partition
        the domain space pulls the label along with it.
        """
        prompt = build_standing_labels_prompt(
            language=self.language,
            entity=context_specifiers.get("entity", ""),
            dimension=dimension,
        )
        client, model = self._get_client_and_model("taxonomy")
        try:
            async with self.semaphore:
                await self.tpm_bucket.wait_and_acquire(
                    self._estimate_preprocessed_tokens(prompt))
                await self.rate_limiter.acquire()
                return await llm_create_async(
                    client=client, model=model,
                    response_model=StandingLabelsResponse,
                    prompt=prompt, temperature=0.0,
                    **get_reasoning_params(model, phase="idea_extraction_taxonomy"),
                )
        except Exception as exc:
            self.verbose_reporter.stat_line(
                f"  Standing labels: translation failed ({exc}) — using fallback labels")
            return None

    @staticmethod
    def _resolve_standing_domains(labels, dimension: DimensionDefinition) -> List:
        """Return the two standing domains as DomainItems, built from the dimension.

        Definition, boundary_test and exclusions come from dimension_data.py and are
        never model output. A standing domain catches a failure mode of the domain
        axis, so its breadth IS its function — a phase that re-describes domains by
        their content will narrow it, and everything it used to catch then needs a
        home of its own. Only the label is translated (`labels` is a
        StandingLabelsResponse or None); a blank one falls back to the dimension's
        own wording. Never returns fewer than two — an assignment menu without them
        is what forced contentless answers into substantive domains.
        """
        bare_label = (getattr(labels, "bare_label", "") or "").strip()
        other_label = (getattr(labels, "other_label", "") or "").strip()
        return [
            DomainItem(
                key=key,
                label=label or spec.fallback_label,
                definition=spec.definition,
                boundary_test=f"Does this idea match: {spec.short}?",
                exclusions=[],
            )
            for key, spec, label in (
                (STANDING_BARE_KEY, dimension.standing_bare, bare_label),
                (STANDING_OTHER_KEY, dimension.standing_other, other_label),
            )
        ]

    def _initialize_context_rate_limiters(self, limits: 'RateLimits', num_tasks: int = 20) -> None:
        """Rate limiters for the context extraction phases (few, large calls)."""
        avg_tokens = self._pred.avg_tokens or DEFAULT_AVG_TOKENS
        headroom = self.processing_config.rate_limit_headroom
        arrival_rate = min(
            limits.requests_per_minute * headroom / 60,
            limits.tokens_per_minute * headroom / avg_tokens / 60
        )
        self.rate_limiter = AsyncLimiter(1, time_period=1.0 / max(arrival_rate, 0.1))
        concurrency = min(num_tasks, self._pred.concurrency or num_tasks)
        self.semaphore = ConcurrencyGate(concurrency)
        self.optimal_concurrency = concurrency
        self.tpm_bucket = TokenBucket(int(limits.tokens_per_minute * headroom))
        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line(
                f"  Context setup: concurrency={concurrency}, avg_tokens={avg_tokens} ({self._pred.origin_line()})")

    async def process_all_tasks_async(self, tasks: List[Dict]) -> List[models.IdeasExtractedModel]:
        """Process all tasks using queue + workers pattern with bootstrap measurement"""
        if not tasks:
            return []

        self.verbose_reporter.step_start("Idea Extraction", emoji="💡")

        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line("Fetching rate limits from API...")

        limits, _ = await fetch_rate_limits(self.model)

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

        # === PHASE 2: Initialize rate limiters for context extraction ===
        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line("Initializing rate limiters for context extraction...")
        self._initialize_context_rate_limiters(limits, num_tasks=30)

        # === PHASE 3: Extract context specifiers, primary dimension, AND domains ===
        _snap_before_context = token_tracker.snapshot() if self.cost_tracker else None
        self.verbose_reporter.stat_line("Extracting context specifiers, primary dimension, and domains...")
        self.generic_specifiers, taxonomy_result, categories_result = await self._extract_generic_specifiers()

        # Store taxonomy axis info for use in idea extraction
        self.primary_dimension = taxonomy_result.primary_dimension
        self.primary_dimension_rationale = taxonomy_result.primary_dimension_rationale
        self.primary_dimension_description = taxonomy_result.primary_dimension_description  # Dynamic context-specific description

        # Store domains for use in per-response extraction model
        # Empty list (Phase 3 skipped) → None to trigger on-the-fly mode in model factories
        self.domains = categories_result.domains or None
        # key is no longer produced by the LLM (removed from prompts) — derive it from
        # the label, standing domains excepted (see _set_domain_keys)
        self._set_domain_keys(self.domains)

        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line(f"\nTaxonomy axis selected: {self.primary_dimension}")
            if self.primary_dimension_description:
                self.verbose_reporter.stat_line(f"Description: {self.primary_dimension_description}")
            if self.domains:
                self.verbose_reporter.stat_line(f"Domains: {[c.label for c in self.domains]}")
            else:
                self.verbose_reporter.stat_line(f"Domains: on-the-fly (no pre-discovered domains)")

        # Record cost for context + discovery phases
        if self.cost_tracker and _snap_before_context is not None:
            self.cost_tracker.record_phase(
                "step_3_idea_extraction", "context_and_discovery",
                _snap_before_context, token_tracker.snapshot(), self.model_context)

        # === PHASE 4: Recalculate avg_tokens with REAL context ===
        # Predicted avg_tokens is more accurate than tiktoken — only recalculate if no prediction
        if self._pred.avg_tokens is None:
            if self.verbose_reporter.enabled:
                self.verbose_reporter.stat_line("\nRecalculating token estimates with real context...")
            old_avg = self.avg_tokens
            self.avg_tokens = self._calculate_avg_tokens()
            if self.verbose_reporter.enabled:
                self.verbose_reporter.stat_line(f"Updated avg_tokens: {old_avg} → {self.avg_tokens}")
        else:
            if self.verbose_reporter.enabled:
                self.verbose_reporter.stat_line(f"\nUsing stored avg_tokens: {self.avg_tokens} (warm-up will recalibrate)")

        # === PHASE 5+6: BULK EXTRACTION via SmoothRequester ===
        # SmoothRequester handles: rate pacing, concurrency control, workers,
        # monitoring, warm-up, retry pass, cache stats — everything.
        self._smooth_requester = SmoothRequester(
            model=self.model_abstraction_ladder,
            phase_key="step3_idea_extraction",
            num_tasks=len(tasks),
            verbose=self.verbose_reporter.enabled,
            processing_config=self.processing_config,
        )

        # Build step-specific prepare + parse functions and fallback
        prepare_fn = self._build_prepare_fn()
        parse_fn = self._build_parse_fn()
        fallback_fn = self._build_fallback_fn()

        # Run bulk extraction
        _snap_before_bulk = token_tracker.snapshot() if self.cost_tracker else None
        results = await self._smooth_requester.process_all(tasks, prepare_fn, parse_fn, fallback_fn)

        # Record cost for bulk extraction phase
        if self.cost_tracker and _snap_before_bulk is not None:
            self.cost_tracker.record_phase(
                "step_3_idea_extraction", "bulk_extraction",
                _snap_before_bulk, token_tracker.snapshot(), self.model_abstraction_ladder)

        # Transfer stats from smoothRequester
        self.stats.update(self._smooth_requester.stats)
        self.failure_log = self._smooth_requester.failure_log
        self.failed_task_ids = self._smooth_requester.failed_task_ids
        self.optimal_concurrency = self._smooth_requester.optimal_concurrency

        # === Post-extraction: orthogonalize domain descriptions (no reassignment) ===
        if ENABLE_DOMAIN_ORTHOGONALIZE:
            try:
                await self._orthogonalize_domains(results)
            except Exception as e:
                self.verbose_reporter.stat_line(f"  Domain orthogonalize skipped ({type(e).__name__}: {e})")

        return results

    async def _orthogonalize_domains(self, results: List[models.IdeasExtractedModel]) -> None:
        """One-shot reformulation: re-describe the DISCOVERED domains for maximal
        orthogonality, grounded in medoid exemplars. Sharpens
        label/definition/boundary_test/exclusions WITHOUT semantic reassignment — if a
        label changes, ideas are deterministically renamed to follow their domain slot
        (a rename, not a re-assignment). The two standing domains are shown as fixed
        reference and are not returnable: sharpening a domain means describing it by
        its content, and a catch for an axis failure has no content of its own to be
        described by. LLM decides; embeddings only select exemplars.
        """
        domains = getattr(self, 'domains', None)
        if not domains:
            return
        gs = self.generic_specifiers

        # group assigned ideas per domain (by current label) + interpretation text
        by_label: Dict[str, list] = {d.label: [] for d in domains}
        for resp in results:
            for idea in (resp.response_ideas or []):
                lab = (idea.domain or "").strip()
                txt = (idea.interpretation or idea.instance or "").strip()
                if lab in by_label and txt:
                    by_label[lab].append((idea, txt))

        discovered, standing = self._partition_standing(domains)

        # medoid → representative exemplars per discovered domain
        embedder = SharedEmbedder()
        blocks = []
        for d in discovered:
            items = by_label.get(d.label, [])
            if items:
                sub = await embedder.embed_texts([t for _, t in items])
                reps = find_representative_samples(sub, n=min(ORTHOGONALIZE_TOP_N, len(items)))
                ex_ideas = [items[r][0] for r in reps]
            else:
                ex_ideas = []
            ex = "\n".join(
                f"      • {(i.instance or '')[:40]} → {(i.interpretation or '')[:70]} → {(i.abstraction or '')[:60]}"
                for i in ex_ideas
            ) or "      (none)"
            block = f"  {d.label}: {d.definition}"
            if getattr(d, 'boundary_test', ''):
                block += f"\n    current boundary_test: {d.boundary_test}"
            if getattr(d, 'exclusions', None):
                block += f"\n    current exclusions: {', '.join(d.exclusions)}"
            block += f"\n    representative ideas:\n{ex}"
            blocks.append(block)
        domains_block = "\n\n".join(blocks)
        standing_block = "\n".join(f"  {d.label}: {d.definition}" for d in standing)

        diag = get_dimension(self.primary_dimension).prompt_rules.domain_diagnostic
        prompt = build_orthogonalize_domains_prompt(
            language=self.language, survey_question=self.var_lab,
            sector=gs["domain"], entity=gs["entity"], topic=gs["topic"],
            perspective=gs["perspective"], intent=gs["intent"],
            primary_dimension=self.primary_dimension, domain_diagnostic=diag,
            domains_block=domains_block, standing_block=standing_block,
        )
        client, model = self._get_client_and_model("taxonomy")

        if self.prompt_printer and not self._captured_domain_orthogonalize:
            self.prompt_printer.capture_prompt(
                step_name="idea_extraction_domains",
                utility_name="IdeaExtractor",
                prompt_content=prompt,
                prompt_type="domain_orthogonalize",
                metadata={"model": model, "survey_question": self.var_lab, "language": self.language}
            )
            self._captured_domain_orthogonalize = True

        est_tokens = self._estimate_preprocessed_tokens(prompt)
        async with self.semaphore:
            await self.tpm_bucket.wait_and_acquire(est_tokens)
            await self.rate_limiter.acquire()
            res = await llm_create_async(
                client=client, model=model, response_model=ReformulatedDomains,
                prompt=prompt, temperature=0.0, **get_reasoning_params(model, phase="idea_extraction_taxonomy"),
            )

        new_domains, rename = self._merge_orthogonalized(
            list(res.domains), discovered, standing)
        if new_domains is None:
            self.verbose_reporter.stat_line(
                f"  Domain orthogonalize skipped (count mismatch: "
                f"{len(res.domains)} vs {len(discovered)} discovered)")
            return

        self.domains = new_domains
        for resp in results:
            for idea in (resp.response_ideas or []):
                lab = (idea.domain or "").strip()
                if lab in rename:
                    idea.domain = rename[lab]

        relabeled = sum(1 for old, new in rename.items() if old != new)
        self.verbose_reporter.stat_line(
            f"  Domain orthogonalize: re-described {len(discovered)} domains "
            f"({relabeled} relabeled, no reassignment; {len(standing)} standing untouched)")

    # === LEGACY: Everything below this line was the old processing loop ===
    # Kept temporarily for reference — will be removed after verification.
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

        nest_asyncio.apply()
        self._results = asyncio.run(self.process_all_tasks_async(tasks))

        # Empirical stats are saved by SmoothRequester internally — no action needed here

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
                    if idea.idea and not idea.idea.startswith("PROCESSING_ERROR") and idea.idea not in ["NA", "NOT_PROCESSED"]:
                        unique_ideas.add(idea.idea)
                        idea_words = idea.idea.split()
                        total_idea_length += len(idea_words)
                        idea_count += 1
                        # Store full idea info including taxonomy
                        valid_ideas.append({
                            'idea': idea.idea,
                            'instance': idea.instance,
                            'facet': idea.facet,
                            'domain': idea.domain,
                        })

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

        self.verbose_reporter.step_complete("Idea extraction completed")

        return self._results
