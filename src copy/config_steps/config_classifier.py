"""
Configuration for Taxonomy Classifier (P1-P7).

Pipeline: facet discovery → facet assignment → attribute discovery →
attribute consolidation → cross-facet dedup.
"""

from dataclasses import dataclass, field
from config import get_model


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
    default_timeout_seconds: float = 60.0      # Cold-start timeout (matches floor; P95×3 adaptive after warm-up)


@dataclass
class CategoriesConfig:
    """Configuration for Taxonomy Classifier (P1-P7)."""

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
    #   "ladder"     — instance → interpretation → abstraction
    #   "idea_rungs" — idea → interpretation → abstraction
    label_source: str = "idea"

    # Optional prefix prepended to each label string before processing.
    # "" = no prefix (default)
    # Any literal string = static prefix for all labels
    label_prefix: str = ""

    # Prepend a valence tag ([+], [-], [0]) to each label.
    # Useful so the LLM can distinguish positive/negative observations during
    # facet discovery (P1), facet assignment (P3), and attribute discovery (P4).
    include_valence: bool = False

    # ==========================================================================
    # TAXONOMY CLASSIFIER PIPELINE (P1-P7)
    # ==========================================================================

    # LLM settings — per-stage model selection (derived from MODEL_FAMILY toggle)
    qr_model_p1: str = get_model("mini")    # P1: Facet Discovery
    qr_model_p2: str = get_model("default") # P2: Facet Consolidation
    qr_model_p3: str = get_model("nano")    # P3: Facet Assignment (classification)
    qr_model_p4: str = get_model("mini")    # P4: Attribute Discovery
    qr_model_p5: str = get_model("default") # P5: Attribute Chunk Consolidation
    qr_model_p6: str = get_model("nano")    # P6: Attribute Assignment (classification)
    qr_model_p7: str = get_model("mini")    # P7: Cross-facet Attribute Consolidation
    qr_temperature: float = 0.3

    # P1: Facet Discovery (per-domain, chunked)
    qr_max_tokens_facet_discovery: int = 4000

    # P3: Facet Assignment (per-domain, batched)
    qr_max_tokens_facet_assignment: int = 4000
    facet_assignment_batch_size: int = 10  # ideas per assignment call (nano-friendly)

    # P4: Attribute Discovery (per facet within domain)
    qr_max_tokens_attribute_discovery: int = 4000

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


# =============================================================================
# PRESETS
# =============================================================================

DEFAULT_CONFIG = CategoriesConfig()
