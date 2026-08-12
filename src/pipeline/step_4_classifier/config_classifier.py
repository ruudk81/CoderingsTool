"""Configuration for the Taxonomy Classifier.

Nine phases, in the order step 3 uses for the domain layer, once per level:

  facet:      discovery → consolidation → assignment → refinement
  attribute:  discovery → consolidation → assignment → refinement
  then:       cross-scope consolidation → valence-neutral merge

Cross-scope consolidation is the one phase that sees more than one scope. Every
other phase is scope-locked, so the same concept survives in several facets and
nothing else can ever see that.

Consolidation settles the inventory BEFORE a single idea is assigned, on the
observations each candidate was built from. Refinement runs AFTER assignment, on
real counts and real response texts. The two answer different questions and both
are needed; see dev/CLAUDE.md.

Phases are named by function, never by number. Renumbering was done twice and
each time it cold-started the perf model and stranded config keys.
"""

from dataclasses import dataclass
from typing import Optional
from config import get_step_model


@dataclass
class CategoriesConfig:
    """Configuration for the Taxonomy Classifier."""

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
    #   "instance"        — verbatim span from the response
    #   "interpretation"  — ladder rung 2: concrete meaning (survey language)
    #   "abstraction"     — ladder rung 3: broader significance (survey language)
    #   "facet"           — Facet (L3)
    #   "domain"          — Domain (L2)
    #   "idea"            — full idea text incl. template prefix
    #
    # Computed composites (assembled from stored fields by format_label()):
    #   "instance_interpretation" — instance → interpretation  (production)
    #   "ladder"                  — instance → interpretation → abstraction
    #   "idea_interpretation"     — idea → interpretation
    #
    # The abstraction rung is deliberately out of the production label: it
    # states an observation's broader significance, which is the facet's and the
    # domain's job. See partition_labels.py for the measurement.
    label_source: str = "idea"

    # Optional prefix prepended to each label string before processing.
    label_prefix: str = ""

    # ==========================================================================
    # MODELS — one rung per phase
    # ==========================================================================

    model_facet_discovery: str = get_step_model("classifier_facet_discovery")
    model_facet_consolidation: str = get_step_model("classifier_facet_consolidation")
    model_facet_assignment: str = get_step_model("classifier_facet_assignment")
    model_facet_refinement: str = get_step_model("classifier_facet_refinement")
    model_attribute_discovery: str = get_step_model("classifier_attribute_discovery")
    model_attribute_consolidation: str = get_step_model("classifier_attribute_consolidation")
    model_attribute_assignment: str = get_step_model("classifier_attribute_assignment")
    model_attribute_refinement: str = get_step_model("classifier_attribute_refinement")
    model_cross_scope_consolidation: str = get_step_model("classifier_cross_scope_consolidation")
    model_valence_merge: str = get_step_model("classifier_valence_merge")

    qr_temperature: float = 0.3

    # ==========================================================================
    # TOKEN CEILINGS
    # ==========================================================================
    # A high ceiling is free — billing is per generated token, and
    # smoothRequester throttles on measured throughput, not on this value. Too
    # low is the only real failure: at 4000 a 22-attribute facet truncated
    # during consolidation and lost it entirely. Upper bound is the model's own
    # max_output (see OPENAI_MODEL_LIMITS in config.py).
    #
    # Discovery, consolidation and refinement enumerate an open-ended list, so
    # their response grows with the data. Assignment returns one id per idea:
    # bounded by construction.
    qr_max_tokens_facet_discovery: int = 32000
    qr_max_tokens_attribute_discovery: int = 32000
    qr_max_tokens_consolidation: int = 32000
    qr_max_tokens_assignment: int = 4000

    # ==========================================================================
    # ASSIGNMENT — batching and menu shortlisting
    # ==========================================================================
    # Both levels use these: ideas are grouped into one rep per unique
    # normalized label, and reps are batched. One call per idea would resend the
    # whole menu every time.
    assignment_batch_k: int = 5
    assignment_shortlist_enabled: bool = True
    assignment_shortlist_k: int = 10

    # ==========================================================================
    # CHUNKING — discovery input per call
    # ==========================================================================

    # Facet discovery chunks (per domain)
    batch_size_min: int = 100      # no splitting below this (single batch)
    batch_size_max: int = 150      # ceiling: keeps prompt quality high
    target_batches: int = 6        # ideal number of chunks
    chunk_overlap: float = 0.2     # overlap fraction between adjacent chunks

    # Attribute discovery chunks (per facet)
    attribute_chunk_size_min: int = 100
    attribute_chunk_size_max: int = 150
    attribute_target_batches: int = 5
    attribute_chunk_overlap: float = 0.2

    # ==========================================================================
    # CONSOLIDATION AND REFINEMENT
    # ==========================================================================

    # How many distinct response texts to show per item during refinement. This
    # is the phase's whole point — it judges real contents, not the label — so
    # the window has to be wide enough to expose a foreign concept hiding
    # inside a bucket.
    contents_top_n: int = 12

    # Consolidation is one call per domain (facets) or per facet (attributes).
    # When a scope holds more candidates than fits one call, it consolidates in
    # rounds: group → consolidate → feed the survivors back in.
    consolidation_max_chunks_per_call: int = 6
    consolidation_max_items_per_call: int = 150
    consolidation_max_rounds: int = 5

    # ==========================================================================
    # OUTPUT
    # ==========================================================================

    verbose: bool = True

    # Stop after a named phase, for testing one phase without paying for the
    # rest. None = run everything. An unknown name raises rather than running
    # the full pipeline, which is what the old numeric version did.
    stop_after_phase: Optional[str] = None


# =============================================================================
# PRESETS
# =============================================================================

DEFAULT_CONFIG = CategoriesConfig()
