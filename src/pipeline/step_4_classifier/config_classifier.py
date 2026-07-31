"""
Configuration for Taxonomy Classifier (P1-P8).

Pipeline: facet discovery → facet assignment → attribute discovery →
attribute consolidation → cross-facet dedup → cross-domain dedup.
"""

from dataclasses import dataclass, field
from typing import Optional
from config import get_step_model


@dataclass
class ClassifierRampConfig:
    """4-layer rate limiting with completion-based ramp.

    Concurrency is computed from Little's Law using estimated latency
    and real API rate limits. Ramp starts at start_fraction and
    advances toward target_fraction proportional to completions.

    Full stack (P1/P3/P4/P6): ConcurrencyGate + TokenBucket + AsyncLimiter + CircuitBreaker
    Light mode (P2/P5/P7): default semaphore + rate limiter only
    """
    # Concurrency ramp
    estimated_latency_seconds: float = 10.0    # Conservative latency estimate
    estimated_avg_tokens: int = 3000           # Conservative token estimate
    start_fraction: float = 0.50               # Start at 50% of Little's Law
    target_fraction: float = 0.90              # Ramp toward 90% of Little's Law
    min_initial: int = 5                       # Concurrency floor
    monitor_poll_interval: float = 0.5         # Monitor coroutine sleep (seconds)
    min_completions_per_step: int = 3          # Min completions before evaluating ramp

    # Warm-up calibration (recalibrate Little's Law with measured data)
    warm_up_sample_min: int = 15               # Min completions before calibration
    warm_up_sample_max: int = 30               # Max completions before forced calibration
    warm_up_min_tasks_to_enable: int = 30      # Skip warm-up for phases with fewer tasks

    # Circuit breaker (monitors timeout rate, reduces concurrency on sustained pressure)
    circuit_breaker_enabled: bool = True
    circuit_breaker_min_tasks: int = 20        # Skip CB for small phases

    # Adaptive timeout (P95 × margin, computed after gate acquisition)
    timeout_floor_seconds: float = 60.0        # Cold-start floor (chunk processing = 60s for large discovery prompts)


@dataclass
class CategoriesConfig:
    """Configuration for Taxonomy Classifier (P1-P8)."""

    # ==========================================================================
    # PARTITION SOURCE
    # ==========================================================================

    PARTITION_SOURCE = "domain"

    # ==========================================================================
    # LABEL SOURCE
    # ==========================================================================

    # Which text to collect as "observations" for facet/attribute discovery input.
    #
    # Stored fields (direct attributes on IdeasExtractedSubmodel from step 3):
    #   "instance"        — Attribute (L4): verbatim span from response
    #   "interpretation"  — Ladder rung 2: concrete meaning (survey language)
    #   "abstraction"     — Ladder rung 3: broader significance (survey language)
    #   "facet"           — Facet (L3): dimension-specific aspect
    #   "domain"          — Domain (L2): thematic domain
    #   "idea"            — full idea text incl. template prefix
    #
    # Computed composites (assembled from stored fields by format_label()):
    #   "ladder"              — instance → interpretation → abstraction
    #   "idea_interpretation" — idea → interpretation
    label_source: str = "idea"

    # Optional prefix prepended to each label string before processing.
    # "" = no prefix (default)
    # Any literal string = static prefix for all labels
    label_prefix: str = ""

    # ==========================================================================
    # TAXONOMY CLASSIFIER PIPELINE (P1-P8)
    # ==========================================================================

    # LLM settings — per-stage model selection (derived from MODEL_FAMILY toggle)
    qr_model_p1: str = get_step_model("classifier_p1")    # P1: Facet Discovery
    qr_model_p2: str = get_step_model("classifier_p2")    # P2: Facet Consolidation
    qr_model_p3: str = get_step_model("classifier_p3")    # P3: Facet Assignment
    qr_model_p4: str = get_step_model("classifier_p4")    # P4: Attribute Discovery
    qr_model_p5: str = get_step_model("classifier_p5")    # P5: Attribute Consolidation
    qr_model_p6: str = get_step_model("classifier_p6")    # P6: Attribute Assignment
    qr_model_p7: str = get_step_model("classifier_p7")    # P7: Cross-facet Attribute Consolidation
    qr_model_p8: str = get_step_model("classifier_p8")    # P8: Cross-domain Attribute Consolidation
    qr_temperature: float = 0.3

    # Output ceilings. A high ceiling is free — billing is per generated token,
    # and smoothRequester throttles on measured throughput (it estimates from the
    # prompt and corrects from actuals), not on this value. Too low is the only
    # real failure: at 4000 a 22-attribute domain truncated at P7 and lost its
    # consolidation. Upper bound is the model's own max_output — 128000 for
    # gpt-5.4, 32000 for gpt-4.1 (see OPENAI_MODEL_LIMITS in config.py).
    #
    # Discovery (P1, P4) and consolidation (P2, P5, P7) enumerate an open-ended
    # list, so their response grows with the data.
    qr_max_tokens_facet_discovery: int = 32000
    qr_max_tokens_attribute_discovery: int = 32000
    qr_max_tokens_consolidation: int = 32000

    # Assignment (P3, P6) takes one idea and returns one label: bounded by
    # construction, so it needs no headroom.
    qr_max_tokens_facet_assignment: int = 4000

    # Adaptive batching for P1 (facet discovery chunks)
    batch_size_min: int = 100      # no splitting below this (single batch)
    batch_size_max: int = 150      # ceiling: keeps prompt quality high
    target_batches: int = 6        # ideal number of chunks
    chunk_overlap: float = 0.2     # overlap fraction between adjacent chunks

    # Adaptive batching for P4 (attribute discovery chunks within a facet)
    p4_batch_size_min: int = 100   # no splitting below this (single batch)
    p4_batch_size_max: int = 150   # ceiling per chunk
    p4_target_batches: int = 5     # ideal number of chunks
    p4_chunk_overlap: float = 0.2  # overlap fraction between adjacent chunks

    # P8: Cross-domain Attribute Consolidation
    qr_max_tokens_cross_domain: int = 16000
    p8_code_source: str = "instance_interpretation"    # embedding text: instance, instance_interpretation, full_abstraction_ladder
    p8_embedding_model: str = "text-embedding-3-large"
    p8_window_size: int = 10                           # max attributes per LLM call
    p8_window_overlap: int = 2                         # overlap between adjacent windows (~20%)
    p8_similarity_threshold: float = 0.6               # noise floor for pairwise cosine similarity

    # Post-hoc over-merge correction (consolidation_corrector, runs in step 5).
    # Splits over-merged catch-all buckets back apart along provenance seams, using
    # the THRESHOLD-FREE within-bucket own>sibling decision (no magic threshold).
    correction_enabled: bool = True                    # DEFAULT ON (runs at step 5 start)
    correction_code_source: str = "instance_interpretation"  # embedding text (mirror p8)
    correction_embedding_model: str = "text-embedding-3-large"
    correction_k_min: int = 5                          # min neighbours for a measurable source
    correction_k_band: int = 2                         # own>sibling verdict checked across k ± band
    correction_min_split_sources: int = 2              # >= this many own-clusters to split a bucket
    correction_residual_dominance: float = 0.60        # share above which a source is the bucket's residual

    # Hierarchical consolidation (shared by P2 and P5)
    # When chunk count or total item count exceeds these limits,
    # consolidation becomes hierarchical: group → consolidate → recurse.
    consolidation_max_chunks_per_call: int = 6   # Rule 2: max chunks per consolidation call
    consolidation_max_items_per_call: int = 150  # Rule 3: max total items per consolidation call
    consolidation_max_rounds: int = 5            # safety cap on recursive rounds

    # ==========================================================================
    # CONCURRENCY RAMP (completion-based, no bootstrap)
    # ==========================================================================

    ramp_config: ClassifierRampConfig = field(default_factory=ClassifierRampConfig)

    # ==========================================================================
    # OUTPUT
    # ==========================================================================

    verbose: bool = True

    # Set to a phase number (1–8) to stop after that phase completes.
    # None = run full pipeline. Useful for testing specific phases without
    # waiting for the full pipeline.
    debug_stop_after_phase: Optional[int] = None


# =============================================================================
# PRESETS
# =============================================================================

DEFAULT_CONFIG = CategoriesConfig()
