"""
Codebook Generator: Code generation and consolidation pipeline (P8-P9).

Pipeline (2 stages):
  P8.  Code Generation from Attributes (per domain) — derive codebook codes
  P9.  Codebook Consolidation (cross-domain) — merge into final MECE codebook

Accepts taxonomy results from step_4_classifier as input.

Usage:
    from .codebook_generator import CodebookGenerator
    from config_steps.config_codeGenerator import CodebookConfig

    generator = CodebookGenerator(config)
    result = generator.generate(
        taxonomy_result=taxonomy_result,
        extraction_metadata=extraction_metadata,
    )
"""

import asyncio
import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Set

from pydantic import BaseModel, Field, create_model

import nest_asyncio
from aiolimiter import AsyncLimiter

from utils.llm import (
    create_client, llm_create_async, RateLimits,
    extract_rate_limits_from_response,
)
from config import (
    ProcessingConfig, DEFAULT_PROCESSING_CONFIG, OPENAI_API_KEY,
    API_PROVIDER, FALLBACK_TPM, FALLBACK_RPM, get_reasoning_params,
)
from steps.step_3_ideaExtractor.ideaExtractor import (
    ConcurrencyGate, ConcurrencyRamp,
    RealTimeTPMTracker, RealTimeRPMTracker,
    ApiLimits, compute_optimal_concurrency,
)
from steps.step_2_qualityFilter.qualityFilter import (
    TokenBucket, LatencyTracker, ConcurrencyCircuitBreaker,
)
from config_steps.config_ideaExtractor import (
    RampUpConfig,
    DEFAULT_CIRCUIT_BREAKER_CONFIG,
)
from steps.step_4_classifier.classifier import PhaseRampState
from utils.modelPerfStats import (
    load_stats, save_stats, update_phase_stats, apply_to_ramp_config,
)

from steps.step_3_ideaExtractor.dimension_data import (
    get_dimension, DimensionDefinition,
)

from steps.step_4_classifier.models_classifier import (
    DomainSet, DomainResultModel, TaxonomyResultsCache, DomainDescription,
)

from config_steps.config_codeGenerator import CodebookConfig
from .prompts_codeGenerator import (
    # P8: Code Generation from Attributes
    build_code_from_attributes_prompt,
    CodeGenerationFromAttributesResult,
    CodeFromAttributes,
    # P9: Codebook Consolidation
    build_codebook_consolidation_prompt,
    CodebookConsolidationResult,
    ConsolidatedCode,
    # Attribute types needed for P8 input formatting
    DiscoveredAttribute,
)

# Enable nested event loops (for VS Code interactive / notebook compatibility)
nest_asyncio.apply()


# =============================================================================
# SHARED DATACLASSES
# =============================================================================

@dataclass
class PromptContext:
    """Shared context passed to all prompt formatting methods."""
    survey_question: str
    language: str
    dataset_context_section: str
    dimension_name: str
    dimension_description: str
    dimension_def: Optional[DimensionDefinition] = None


@dataclass
class DomainContext:
    """Partition-specific context."""
    partition_name: str
    partition_definition: str


@dataclass
class TaxonomyResult:
    """Input from taxonomy stages P1-P7 (mirrors step_4_classifier.classifier.TaxonomyResult)."""
    partition_n_labels: Dict[str, int]
    partition_n_batches: Dict[str, int]
    partition_facets: Dict[str, list]  # domain -> [DiscoveredFacet]
    partition_assignments: Dict[str, Dict[str, str]]  # domain -> {idea_id -> facet_name}
    partition_attributes: Dict[str, Dict[str, list]]  # domain -> {facet -> [DiscoveredAttribute]}
    attribute_assignments: Dict[str, str]  # idea_id -> attribute_name


@dataclass
class DomainResult:
    """Per-domain pipeline result (v3)."""
    partition_name: str
    n_labels: int
    n_batches: int
    facets: list
    facet_assignments: Dict[str, str]  # idea_id -> facet_name
    attributes: Dict[str, list]  # facet_name -> attributes
    attribute_assignments: Dict[str, str] = field(default_factory=dict)  # idea_id -> attribute_name


@dataclass
class CodebookResult:
    """Output of codebook stages P8-P9."""
    codes: List[ConsolidatedCode]
    codebook_narrative: str


# =============================================================================
# MAIN PROCESSOR
# =============================================================================

class CodebookGenerator:
    """
    Codebook Generator: Code generation and consolidation pipeline (P8-P9).

    Pipeline (2 stages):
    P8.  CODE GENERATION:                   Per domain, derive codes from attributes
    P9.  CODEBOOK CONSOLIDATION:            Cross-domain, merge into MECE codebook
    """

    def __init__(self, config: CodebookConfig, prompt_printer=None):
        self._model_p8 = config.model_p8
        self._model_p9 = config.model_p9
        self._temperature = config.temperature
        self._max_tokens_code_from_attributes = config.max_tokens_code_from_attributes
        self._max_tokens_codebook_consolidation = config.max_tokens_codebook_consolidation

        # Prompt capture (optional)
        self._prompt_printer = prompt_printer
        self._captured_gates: Set[str] = set()

        # Concurrency ramp config
        self._ramp_config = config.ramp_config

        # Shared async resources — initialized in generate()
        self._clients = None
        self._semaphore = None
        self._rate_limiter = None
        self._fetched_limits = None
        self._perf_stats: dict = {}

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def generate(
        self,
        taxonomy_result: TaxonomyResult,
        partition_set: DomainSet,
        survey_question: str = "",
        language: str = "Dutch",
        dataset_context: Optional[Dict[str, str]] = None,
        dimension_name: str = "",
        dimension_description: str = "",
        verbose: bool = False,
        prompt_printer=None,
    ) -> CodebookResult:
        """Run codebook stages (P8-P9) from a TaxonomyResult.

        Args:
            taxonomy_result: Output from TaxonomyClassifier.process()
            partition_set: Domain partition definitions
            survey_question: The survey question being coded
            language: Language of the survey responses
            dataset_context: Optional dataset context dict
            dimension_name: Name of the dimension being analyzed
            dimension_description: Description of the dimension
            verbose: Print progress information
            prompt_printer: Optional prompt printer (overrides __init__ printer)
        """
        if prompt_printer is not None:
            self._prompt_printer = prompt_printer

        print(f"\n{'='*70}")
        print(f"CODEBOOK GENERATION (P8-P9)")
        print(f"{'='*70}")

        # Resolve dimension definition
        dimension_def = None
        if dimension_name:
            dimension_def = get_dimension(dimension_name)
            if dimension_def and verbose:
                print(f"  Dimension: {dimension_name}")
            elif not dimension_def and verbose:
                print(f"  WARNING: No DimensionDefinition found for '{dimension_name}'")

        dataset_context_section = self._build_dataset_context_section(dataset_context)

        prompt_context = PromptContext(
            survey_question=survey_question,
            language=language,
            dataset_context_section=dataset_context_section,
            dimension_name=dimension_name,
            dimension_description=dimension_description,
            dimension_def=dimension_def,
        )

        partition_contexts = self._build_all_partition_contexts(partition_set)

        async def _run():
            await self._initialize_async_resources(verbose)
            return await self._process_codebook_async(
                taxonomy_result, partition_contexts, prompt_context, verbose
            )

        return asyncio.run(_run())

    # =========================================================================
    # ASYNC ORCHESTRATION
    # =========================================================================

    async def _initialize_async_resources(self, verbose: bool):
        """Initialize clients and rate limiters for P8-P9 models."""
        self._perf_stats = load_stats()
        unique_models = {self._model_p8, self._model_p9}
        self._clients = {m: create_client(model=m, async_mode=True) for m in unique_models}

        processing_config = DEFAULT_PROCESSING_CONFIG
        headroom = processing_config.rate_limit_headroom

        if verbose:
            print("  Fetching rate limits from API...")
        limits = await self._fetch_rate_limits_from_api()

        if limits.tokens_per_minute == 0 or limits.requests_per_minute == 0:
            if verbose:
                print(f"  WARNING: Using fallback rate limits "
                      f"(TPM={FALLBACK_TPM}, RPM={FALLBACK_RPM})")
            limits = RateLimits(
                tokens_per_minute=FALLBACK_TPM,
                requests_per_minute=FALLBACK_RPM,
            )
        elif verbose:
            print(f"  Fetched from API: TPM={limits.tokens_per_minute:,}, "
                  f"RPM={limits.requests_per_minute:,}")

        self._fetched_limits = limits
        cfg = self._ramp_config
        api_limits = ApiLimits(limits.tokens_per_minute, limits.requests_per_minute)
        little_law = compute_optimal_concurrency(
            api_limits, cfg.estimated_latency_seconds, cfg.estimated_avg_tokens,
        )

        est_avg_tokens = cfg.estimated_avg_tokens
        rpm_throughput = limits.requests_per_minute * headroom / 60
        tpm_throughput = limits.tokens_per_minute * headroom / est_avg_tokens / 60
        arrival_rate = min(rpm_throughput, tpm_throughput)
        bottleneck = "RPM" if rpm_throughput < tpm_throughput else "TPM"

        default_conc = max(cfg.min_initial, int(little_law * cfg.start_fraction))
        self._semaphore = asyncio.Semaphore(default_conc)
        self._rate_limiter = AsyncLimiter(1, time_period=1.0 / max(arrival_rate, 0.01))

        if verbose:
            print(f"\n  [RATE LIMITING SETUP]")
            print(f"  Models: P8={self._model_p8}, P9={self._model_p9}")
            print(f"  RPM: {limits.requests_per_minute:,} "
                  f"({limits.requests_per_minute * headroom:,.0f} with headroom)")
            print(f"  TPM: {limits.tokens_per_minute:,} "
                  f"({limits.tokens_per_minute * headroom:,.0f} with headroom)")
            print(f"  Expected throughput: {arrival_rate:.1f}/s ({bottleneck} limited)")
            print(f"  Little's Law: {little_law} | "
                  f"Default concurrency: {default_conc} (P9 consolidation)")

    async def _process_codebook_async(
        self,
        taxonomy: TaxonomyResult,
        partition_contexts: Dict[str, DomainContext],
        prompt_context: PromptContext,
        verbose: bool,
    ) -> CodebookResult:
        """Codebook stages P8-P9: code generation + consolidation."""
        partition_facets = taxonomy.partition_facets
        partition_assignments = taxonomy.partition_assignments
        domain_facet_attributes = taxonomy.partition_attributes
        attribute_assignments = taxonomy.attribute_assignments

        start_time = time.time()

        # =================================================================
        # PHASE 8 (P8): Per-domain Code Generation
        # =================================================================
        if verbose:
            print(f"\n  Phase 8: Per-domain Code Generation...")

        t_phase8 = time.time()

        p8_tasks = {}
        for domain_name in domain_facet_attributes:
            domain_attrs = domain_facet_attributes.get(domain_name, {})
            if not domain_attrs:
                continue

            domain_facet_ids = set(partition_assignments.get(domain_name, {}).keys())
            domain_attr_assigns = {
                iid: aname for iid, aname in attribute_assignments.items()
                if iid in domain_facet_ids
            }

            excluded = [
                (other_name, partition_contexts[other_name].partition_definition)
                for other_name in partition_contexts
                if other_name != domain_name
            ]

            p8_tasks[domain_name] = self._run_code_generation_from_attributes(
                {domain_name: domain_attrs}, prompt_context,
                attribute_assignments=domain_attr_assigns,
                domain_name=domain_name,
                domain_definition=partition_contexts[domain_name].partition_definition,
                excluded_domains=excluded,
            )

        p8_state = self._create_phase_ramp("P8", len(p8_tasks), model=self._model_p8,
                                            phase_key="step5_p8_codebook_generation")
        p8_results = await self._run_with_ramp(p8_tasks.values(), p8_state)
        self._collect_phase_stats(p8_state, self._model_p8, "step5_p8_codebook_generation")

        all_codes = []
        code_provenance = {}
        codebook_narratives = []
        for key, result in zip(p8_tasks.keys(), p8_results):
            if isinstance(result, Exception):
                print(f"  P8 '{key}' FAILED: {type(result).__name__}: {result}")
            else:
                for code in result.codes:
                    code_provenance[len(all_codes)] = key
                    all_codes.append(code)
                codebook_narratives.append(f"[{key}] {result.scratchpad}")
                if verbose:
                    print(f"    {key}: {len(result.codes)} codes")

        t_phase8 = time.time() - t_phase8

        if verbose:
            print(f"\n  Phase 8 done in {t_phase8:.1f}s → {len(all_codes)} raw codes "
                  f"from {len(p8_tasks)} calls")

        attr_to_count: Dict[str, int] = {}
        for attr_name in attribute_assignments.values():
            attr_to_count[attr_name] = attr_to_count.get(attr_name, 0) + 1

        code_frequencies: Dict[int, int] = {}
        for idx, code in enumerate(all_codes):
            freq = sum(
                attr_to_count.get(attr, 0)
                for attr in (code.source_attributes or [])
            )
            code_frequencies[idx] = freq

        # =================================================================
        # PHASE 9 (P9): Cross-domain Codebook Consolidation
        # =================================================================
        if verbose:
            print(f"\n  Phase 9: Codebook Consolidation...")

        t_phase9 = time.time()

        if len(all_codes) > 0:
            consolidation_result = await self._consolidate_codebook(
                all_codes, code_provenance, prompt_context,
                code_frequencies=code_frequencies,
            )
            all_codes = consolidation_result.codes
            codebook_narratives.append(
                f"[consolidation] {consolidation_result.scratchpad}"
            )

        codebook_narrative = "\n".join(codebook_narratives)

        t_phase9 = time.time() - t_phase9

        if verbose:
            print(f"\n  Phase 9 done in {t_phase9:.1f}s → {len(all_codes)} codes "
                  f"(after consolidation)")
            for i, code in enumerate(all_codes, 1):
                print(f"    {i}. {code.code_name}: {code.definition}")

        codebook_elapsed = time.time() - start_time
        if verbose:
            print(f"\n  Codebook (P8-P9) complete in {codebook_elapsed:.1f}s")

        save_stats(self._perf_stats)

        return CodebookResult(
            codes=all_codes,
            codebook_narrative=codebook_narrative,
        )

    # =========================================================================
    # PHASE 8 (P8): CODE GENERATION FROM ATTRIBUTES
    # =========================================================================

    @staticmethod
    def _build_constrained_response_model(
        attribute_names: List[str],
    ):
        """Build a CodeGenerationFromAttributesResult with source_attributes
        constrained to an enum of valid attribute names."""
        if not attribute_names:
            return CodeGenerationFromAttributesResult

        AttrLiteral = Literal[tuple(attribute_names)]

        ConstrainedCode = create_model(
            "CodeFromAttributes",
            code_name=(str, Field(..., description="Short code name (3-5 word noun phrase)")),
            definition=(str, Field(..., description="Clear definition of what this code covers (1-2 sentences)")),
            typical_indicators=(List[str], Field(..., description="Words or phrases that signal this code")),
            source_attributes=(List[AttrLiteral], Field(
                default_factory=list,
                description="Attribute names this code is derived from (must be exact names from the inventory)",
            )),
        )

        ConstrainedResult = create_model(
            "CodeGenerationFromAttributesResult",
            scratchpad=(str, CodeGenerationFromAttributesResult.model_fields["scratchpad"]),
            codes=(List[ConstrainedCode], Field(..., description="Formal codes derived from the attribute inventory")),
        )

        return ConstrainedResult

    async def _run_code_generation_from_attributes(
        self,
        domain_facet_attributes: Dict[str, Dict[str, List[DiscoveredAttribute]]],
        prompt_context: PromptContext,
        attribute_assignments: Optional[Dict[str, str]] = None,
        domain_name: str = "",
        domain_definition: str = "",
        excluded_domains: Optional[List[tuple]] = None,
    ) -> CodeGenerationFromAttributesResult:
        """Generate codes from an attribute inventory (per-domain)."""
        prompt = build_code_from_attributes_prompt(
            survey_question=prompt_context.survey_question,
            language=prompt_context.language,
            dataset_context_section=prompt_context.dataset_context_section,
            dimension_def=prompt_context.dimension_def,
            dimension_name=prompt_context.dimension_name,
            dimension_description=prompt_context.dimension_description,
            domain_name=domain_name,
            domain_definition=domain_definition,
            domain_attributes=domain_facet_attributes,
            attribute_assignments=attribute_assignments,
            excluded_domains=excluded_domains,
        )

        all_attr_names = [
            attr.attribute_name
            for facet_attrs in domain_facet_attributes.values()
            for attrs in facet_attrs.values()
            for attr in attrs
        ]
        response_model = self._build_constrained_response_model(all_attr_names)

        domain_key = "::".join(domain_facet_attributes.keys())
        gate_key = f"qr_code_gen_{domain_key}"
        if (self._prompt_printer is not None
                and gate_key not in self._captured_gates):
            self._prompt_printer.capture_prompt(
                step_name="qualitative_researcher",
                utility_name="QualitativeResearcher",
                prompt_content=prompt,
                prompt_type="code_generation_from_attributes",
                metadata={
                    "model": self._model_p8,
                    "temperature": self._temperature,
                    "max_tokens": self._max_tokens_code_from_attributes,
                    "language": prompt_context.language,
                    "n_domains": len(domain_facet_attributes),
                    "n_total_attributes": len(all_attr_names),
                    "dimension_name": prompt_context.dimension_name,
                }
            )
            self._captured_gates.add(gate_key)

        for attempt in range(2):
            try:
                return await self._llm_call(
                    prompt, response_model,
                    self._max_tokens_code_from_attributes,
                    model=self._model_p8,
                    timeout=180.0,
                )
            except Exception as e:
                if attempt == 0:
                    print(f"    P8 CODE GENERATION failed (attempt 1), retrying: "
                          f"{type(e).__name__}: {e}")
                else:
                    print(f"    P8 CODE GENERATION failed (attempt 2), returning empty: "
                          f"{type(e).__name__}: {e}")
                    return CodeGenerationFromAttributesResult(codes=[], evaluation="PROCESSING_ERROR")

    # =========================================================================
    # PHASE 9 (P9): CODEBOOK CONSOLIDATION
    # =========================================================================

    async def _consolidate_codebook(
        self,
        raw_codes: list,
        code_provenance: dict,
        prompt_context: PromptContext,
        code_frequencies: Optional[Dict[int, int]] = None,
    ) -> CodebookConsolidationResult:
        """Consolidate per-domain codes into a final parsimonious codebook."""
        prompt = build_codebook_consolidation_prompt(
            survey_question=prompt_context.survey_question,
            language=prompt_context.language,
            dataset_context_section=prompt_context.dataset_context_section,
            dimension_name=prompt_context.dimension_name,
            dimension_description=prompt_context.dimension_description,
            dimension_def=prompt_context.dimension_def,
            raw_codes=raw_codes,
            code_provenance=code_provenance,
            code_frequencies=code_frequencies,
        )

        gate_key = "qr_codebook_consolidation"
        if (self._prompt_printer is not None
                and gate_key not in self._captured_gates):
            self._prompt_printer.capture_prompt(
                step_name="qualitative_researcher",
                utility_name="QualitativeResearcher",
                prompt_content=prompt,
                prompt_type="codebook_consolidation",
                metadata={
                    "model": self._model_p9,
                    "temperature": self._temperature,
                    "max_tokens": self._max_tokens_codebook_consolidation,
                    "language": prompt_context.language,
                    "n_raw_codes": len(raw_codes),
                    "dimension_name": prompt_context.dimension_name,
                }
            )
            self._captured_gates.add(gate_key)

        for attempt in range(2):
            try:
                return await self._llm_call(
                    prompt, CodebookConsolidationResult,
                    self._max_tokens_codebook_consolidation,
                    model=self._model_p9,
                    timeout=180.0,
                )
            except Exception as e:
                if attempt == 0:
                    print(f"    P9 CODEBOOK CONSOLIDATION failed (attempt 1), retrying: "
                          f"{type(e).__name__}: {e}")
                else:
                    print(f"    P9 CODEBOOK CONSOLIDATION failed (attempt 2), returning raw codes: "
                          f"{type(e).__name__}: {e}")
                    raise

    # =========================================================================
    # DYNAMIC RATE LIMIT DISCOVERY
    # =========================================================================

    async def _fetch_rate_limits_from_api(self) -> RateLimits:
        """Make a minimal API call to fetch rate limits from response headers."""
        from openai import AsyncOpenAI

        if API_PROVIDER == "azure":
            from config import (
                AZURE_OPENAI_ENDPOINT,
                AZURE_OPENAI_API_KEY,
                AZURE_OPENAI_DEPLOYMENT_NAME,
            )
            client = AsyncOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                base_url=f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}/",
                default_query={"api-version": "2024-10-21"},
            )
            model = AZURE_OPENAI_DEPLOYMENT_NAME
        else:
            client = AsyncOpenAI(api_key=OPENAI_API_KEY)
            model = self._model_p8

        if API_PROVIDER == "azure":
            response = await client.chat.completions.with_raw_response.create(
                model=model,
                messages=[{"role": "user", "content": "Hi"}],
                max_completion_tokens=5,
            )
        else:
            response = await client.responses.with_raw_response.create(
                model=model,
                input="Hi",
            )
        return extract_rate_limits_from_response(response)

    # =========================================================================
    # 4-LAYER RATE LIMITING (per-phase)
    # =========================================================================

    def _collect_phase_stats(
        self, state: PhaseRampState, model: str, phase_key: str
    ) -> None:
        """Extract latency/token measurements from a completed phase and update perf stats."""
        if state.latency_tracker is None or len(state.latency_tracker.values) < 5:
            return
        tokens = list(state.actual_total_tokens)
        if not tokens:
            return
        measurements = {
            "p50_latency_s": state.latency_tracker.get_p50(),
            "p95_latency_s": state.latency_tracker.get_p95(),
            "avg_tokens": sum(tokens) / len(tokens),
        }
        update_phase_stats(self._perf_stats, model, phase_key, measurements, state.completions)

    def _create_phase_ramp(self, phase_name: str, num_tasks: int,
                           model: str = None, phase_key: str = None) -> PhaseRampState:
        """Create per-phase 4-layer rate limiting stack."""
        from copy import copy
        cfg = copy(self._ramp_config)
        if phase_key and model:
            apply_to_ramp_config(self._perf_stats, model, phase_key, cfg)
        headroom = DEFAULT_PROCESSING_CONFIG.rate_limit_headroom
        api_limits = ApiLimits(
            self._fetched_limits.tokens_per_minute,
            self._fetched_limits.requests_per_minute,
        )
        little_law = compute_optimal_concurrency(
            api_limits, cfg.estimated_latency_seconds, cfg.estimated_avg_tokens,
        )
        little_law_cap = min(little_law, num_tasks)

        ramp_up_cfg = RampUpConfig(
            start_fraction=cfg.start_fraction,
            target_fraction=cfg.target_fraction,
            min_initial=cfg.min_initial,
            measurement_window_seconds=cfg.monitor_poll_interval,
            min_completions_per_step=cfg.min_completions_per_step,
        )

        # Capacity-relative starting
        half_little_law = int(little_law * cfg.start_fraction)
        initial = max(half_little_law, num_tasks)
        initial = min(initial, num_tasks)
        target = min(int(little_law_cap * cfg.target_fraction), num_tasks)

        gate = ConcurrencyGate(initial)
        ramp = ConcurrencyRamp(ramp_up_cfg, little_law_cap, num_tasks)
        is_large = num_tasks >= cfg.circuit_breaker_min_tasks

        token_bucket = None
        if is_large:
            token_bucket = TokenBucket(int(self._fetched_limits.tokens_per_minute * headroom))

        latency_tracker = None
        if is_large:
            latency_tracker = LatencyTracker(
                DEFAULT_PROCESSING_CONFIG,
                timeout_floor=cfg.timeout_floor_seconds,
                default_timeout=cfg.default_timeout_seconds,
            )

        circuit_breaker = None
        if cfg.circuit_breaker_enabled and is_large:
            circuit_breaker = ConcurrencyCircuitBreaker(
                DEFAULT_CIRCUIT_BREAKER_CONFIG, gate, initial,
            )

        mode = "4-layer" if is_large else "light"
        print(f"    [{phase_name}] Little's Law: {little_law} | "
              f"Start: {initial} → Target: {target} ({num_tasks} tasks, {mode})")

        return PhaseRampState(
            gate=gate, ramp=ramp,
            rpm_tracker=RealTimeRPMTracker(window_seconds=60.0),
            tpm_tracker=RealTimeTPMTracker(window_seconds=60.0),
            phase_name=phase_name,
            total_tasks=num_tasks,
            token_bucket=token_bucket,
            latency_tracker=latency_tracker,
            circuit_breaker=circuit_breaker,
            estimated_avg_tokens=cfg.estimated_avg_tokens,
        )

    async def _phase_monitor(self, state: PhaseRampState):
        """Background monitor: ramp + circuit breaker + progress."""
        start_time = time.monotonic()
        last_report_time = start_time
        last_reported_completions = -1

        while not state.done:
            await asyncio.sleep(self._ramp_config.monitor_poll_interval)
            now = time.monotonic()
            elapsed = now - start_time

            if state.circuit_breaker:
                state.circuit_breaker.check_and_adjust()

            if not state.ramp.is_done() and state.completions >= self._ramp_config.min_completions_per_step:
                rate = state.completions / elapsed if elapsed > 0 else 0
                state.ramp.record_measurement(
                    throughput=rate, tpm_pct=0, rpm_pct=0,
                    completions_total=state.completions,
                    timeouts_total=state.timeouts, duration=elapsed,
                )
                new_target = state.ramp.current_target()
                if new_target != state.gate.limit:
                    state.gate.set_limit(new_target)
                    if state.circuit_breaker:
                        state.circuit_breaker.baseline = new_target

            if now - last_report_time >= 2.0:
                if state.completions != last_reported_completions:
                    last_report_time = now
                    last_reported_completions = state.completions
                    rate = state.completions / elapsed if elapsed > 0 else 0
                    current_tpm = await state.tpm_tracker.get_current_tpm()
                    current_rpm = await state.rpm_tracker.get_current_rpm()
                    timeout_info = f" timeouts:{state.timeouts}" if state.timeouts > 0 else ""
                    cb_info = ""
                    if state.circuit_breaker and state.circuit_breaker.state != 'CLOSED':
                        cb_info = f" CB:{state.circuit_breaker.state}"
                    print(f"    [{state.phase_name}] {state.completions}/{state.total_tasks} "
                          f"({rate:.1f}/s) | TPM:{current_tpm:,.0f} RPM:{current_rpm:.0f} "
                          f"Conc:{state.gate.active}/{state.gate.limit}→{state.ramp._target}"
                          f"{timeout_info}{cb_info}")

    async def _run_with_ramp(self, coros, state: PhaseRampState):
        """Run coroutines via gather with a background ramp monitor."""
        async def _work():
            results = await asyncio.gather(*coros, return_exceptions=True)
            state.done = True
            return results

        results, _ = await asyncio.gather(_work(), self._phase_monitor(state))

        if state.latency_tracker and state.latency_tracker.values:
            vals = list(state.latency_tracker.values)
            p50 = float(np.percentile(vals, 50))
            p95 = float(np.percentile(vals, 95))
            avg_tok = int(np.mean(list(state.actual_total_tokens))) if state.actual_total_tokens else 0
            print(f"    [{state.phase_name}] Latency: P50={p50:.1f}s P95={p95:.1f}s | "
                  f"avg_tokens={avg_tok:,} | "
                  f"{state.completions} ok, {state.timeouts} timeouts")

        return results

    # =========================================================================
    # SHARED LLM CALL
    # =========================================================================

    async def _llm_call(self, prompt: str, response_model, max_tokens: int,
                        temperature: float | None = None, model: str | None = None,
                        timeout: float = 180.0, gate=None, phase_state: PhaseRampState = None):
        """Make a rate-limited LLM call through the 4-layer stack."""
        use_model = model or self._model_p8
        client = self._clients[use_model]
        concurrency_ctx = gate if gate is not None else self._semaphore

        est_tokens = max_tokens
        if phase_state and phase_state.estimated_avg_tokens:
            est_tokens = phase_state.estimated_avg_tokens

        async with concurrency_ctx:
            effective_timeout = timeout
            if phase_state and phase_state.latency_tracker:
                effective_timeout = phase_state.latency_tracker.get_timeout()

            if phase_state and phase_state.token_bucket:
                await phase_state.token_bucket.wait_and_acquire(est_tokens)

            async with self._rate_limiter:
                task_start = time.monotonic()
                try:
                    result = await asyncio.wait_for(
                        llm_create_async(
                            client=client,
                            model=use_model,
                            prompt=prompt,
                            response_model=response_model,
                            temperature=temperature if temperature is not None else self._temperature,
                            max_tokens=max_tokens,
                            **get_reasoning_params(use_model),
                        ),
                        timeout=effective_timeout,
                    )

                    elapsed = time.monotonic() - task_start
                    if phase_state and phase_state.latency_tracker:
                        phase_state.latency_tracker.add(elapsed)
                    if phase_state and phase_state.circuit_breaker:
                        phase_state.circuit_breaker.record_completion()

                    if phase_state is not None:
                        phase_state.completions += 1
                        await phase_state.rpm_tracker.record()
                        raw = getattr(result, '_raw_response', None)
                        usage = getattr(raw, 'usage', None) if raw else None
                        actual_tokens = None
                        if usage:
                            actual_tokens = (
                                getattr(usage, 'prompt_tokens', 0)
                                + getattr(usage, 'completion_tokens', 0)
                                + getattr(usage, 'input_tokens', 0)
                                + getattr(usage, 'output_tokens', 0)
                            )
                            await phase_state.tpm_tracker.record(actual_tokens)
                        else:
                            await phase_state.tpm_tracker.record(max_tokens)

                        if actual_tokens and phase_state.actual_total_tokens is not None:
                            phase_state.actual_total_tokens.append(actual_tokens)
                        if actual_tokens and phase_state.token_bucket:
                            delta = actual_tokens - est_tokens
                            if delta != 0:
                                await phase_state.token_bucket.reconcile(delta)

                    return result

                except asyncio.TimeoutError:
                    if phase_state and phase_state.circuit_breaker:
                        phase_state.circuit_breaker.record_timeout()
                    if phase_state is not None:
                        phase_state.timeouts += 1
                    raise

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _build_all_partition_contexts(
        self,
        partition_set: DomainSet,
    ) -> Dict[str, DomainContext]:
        """Build DomainContext for each partition."""
        contexts = {}
        for part in partition_set.partitions:
            contexts[part.partition_name] = DomainContext(
                partition_name=part.partition_name,
                partition_definition=part.inclusion_definition,
            )
        return contexts

    @staticmethod
    def _build_dataset_context_section(
        dataset_context: Optional[Dict[str, str]],
    ) -> str:
        """Build dataset context block for prompts."""
        if not dataset_context:
            return ""
        parts = []
        for key in ["domain", "entity", "topic", "perspective", "intent"]:
            value = dataset_context.get(key, "")
            if value:
                parts.append(f"{key.capitalize()}: {value}")
        if not parts:
            return ""
        return "<dataset_context>\n" + "\n".join(parts) + "\n</dataset_context>"
