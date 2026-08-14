"""
Configuration for Taxonomy Classifier (P1-P9).

Pipeline, per level: discovery → assignment → consolidation.
  P1 axis discovery → P2/P3 facet discovery (with/without axes) →
  P4 facet assignment → P5 facet consolidation (in-axis) →
  P6 attribute discovery → P7 attribute assignment →
  P8 attribute consolidation (in-facet) → P9 valence-neutral merge.

Consolidation runs after assignment so it sees real idea counts and texts
rather than discovery's guesses. See dev/CLAUDE.md.
"""

from dataclasses import dataclass
from typing import Optional
from config import get_step_model


@dataclass
class CategoriesConfig:
    """Configuration for Taxonomy Classifier (P1-P9)."""

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
    # TAXONOMY CLASSIFIER PIPELINE (P1-P9)
    # P3 (facet discovery zonder assen) is dezelfde dispatch als P2 met een
    # andere prompt — geen eigen model key.
    # ==========================================================================

    axis_first_enabled: bool = True

    # LLM settings — per-stage model selection (derived from MODEL_FAMILY toggle)
    qr_model_p1: str = get_step_model("classifier_p1")    # P1: Axis Discovery
    qr_model_p2: str = get_step_model("classifier_p2")    # P2/P3: Facet Discovery
    qr_model_p4: str = get_step_model("classifier_p4")    # P4: Facet Assignment
    qr_model_p5: str = get_step_model("classifier_p5")    # P5: Facet Consolidation (in-axis)
    qr_model_p6: str = get_step_model("classifier_p6")    # P6: Attribute Discovery
    qr_model_p7: str = get_step_model("classifier_p7")    # P7: Attribute Assignment
    qr_model_p8: str = get_step_model("classifier_p8")    # P8: Attribute Consolidation (in-facet)
    qr_model_p9: str = get_step_model("classifier_p9")    # P9: Valence-neutral merge
    qr_temperature: float = 0.3

    # Output ceilings. A high ceiling is free — billing is per generated token,
    # and smoothRequester throttles on measured throughput (it estimates from the
    # prompt and corrects from actuals), not on this value. Too low is the only
    # real failure: at 4000 a 22-attribute domain truncated during consolidation
    # and lost it entirely. Upper bound is the model's own max_output — 128000 for
    # gpt-5.4, 32000 for gpt-4.1 (see OPENAI_MODEL_LIMITS in config.py).
    #
    # Discovery (P1, P5) and consolidation (P2, P6, P9) enumerate an open-ended
    # list, so their response grows with the data.
    qr_max_tokens_facet_discovery: int = 32000
    qr_max_tokens_attribute_discovery: int = 32000
    qr_max_tokens_consolidation: int = 32000

    # Assignment (P4, P8) takes one idea and returns one label: bounded by
    # construction, so it needs no headroom.
    qr_max_tokens_facet_assignment: int = 4000

    # P4 facet assignment — batch mode (gemeten 2026-08-05: beide armen PASS,
    # judge verkiest batch; armen 98% identiek). Uit-knoppen zijn byte-identiek:
    # batch uit = het oude per-idee-pad; shortlist uit = vol menu in de batch;
    # label_dedup wordt alleen in de batch-tak gelezen.
    facet_assignment_batch_enabled: bool = True
    facet_assignment_batch_k: int = 5
    facet_assignment_shortlist_enabled: bool = True
    facet_assignment_shortlist_k: int = 10
    facet_assignment_label_dedup: bool = True

    # Adaptive batching for P1 (facet discovery chunks)
    batch_size_min: int = 100      # no splitting below this (single batch)
    batch_size_max: int = 150      # ceiling: keeps prompt quality high
    target_batches: int = 6        # ideal number of chunks
    chunk_overlap: float = 0.2     # overlap fraction between adjacent chunks

    # Adaptive batching for P5 (attribute discovery chunks within a facet)
    p4_batch_size_min: int = 100   # no splitting below this (single batch)
    p4_batch_size_max: int = 150   # ceiling per chunk
    p4_target_batches: int = 5     # ideal number of chunks
    p4_chunk_overlap: float = 0.2  # overlap fraction between adjacent chunks

    # P9: how many distinct response texts to show per attribute. This is the
    # phase's whole point — it judges real contents, not the label — so the window
    # has to be wide enough to expose a foreign concept hiding inside a bucket.
    p9_contents_top_n: int = 12

    # Hierarchical consolidation (shared by P2 and P6)
    # When chunk count or total item count exceeds these limits,
    # consolidation becomes hierarchical: group → consolidate → recurse.
    consolidation_max_chunks_per_call: int = 6   # Rule 2: max chunks per consolidation call
    consolidation_max_items_per_call: int = 150  # Rule 3: max total items per consolidation call
    consolidation_max_rounds: int = 5            # safety cap on recursive rounds

    # ==========================================================================
    # OUTPUT
    # ==========================================================================

    verbose: bool = True

    # Set to a phase number (1-10) to stop after that phase completes.
    # None = run full pipeline. Useful for testing specific phases without
    # waiting for the full pipeline.
    debug_stop_after_phase: Optional[int] = None


# =============================================================================
# PRESETS
# =============================================================================

DEFAULT_CONFIG = CategoriesConfig()
