"""Configuration for the Taxonomy Classifier.

Seven phases, named by function:

  discovery → facet_consolidation → attribute_consolidation → assignment →
  refinement → cross_domain
  then: valence_merge, from the runner

Facets and attributes are FOUND together in one discovery call and SETTLED
apart, one call per level: so there is one chunking register and two
consolidation registers, each with its own cap.

Consolidation settles the inventory BEFORE a single idea is assigned, on what
the passes proposed. Refinement runs AFTER assignment, on real counts and real
response texts. The two answer different questions and both are needed.
Cross-domain is the one phase that sees more than one domain; see dev/CLAUDE.md.

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

    # Which text to collect as "observations" for discovery input, and to show
    # as an attribute's contents during refinement.
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
    #   "ladder"                  — instance → interpretation → abstraction
    #   "idea_interpretation"     — idea → interpretation
    #   "instance_interpretation" — instance → interpretation
    #
    # Production runs on "instance_interpretation": the abstraction rung is the
    # step-3 model's own generalisation, and feeding it back as the observation
    # makes step 4 name facets after that generalisation instead of after what
    # respondents said.
    label_source: str = "instance_interpretation"

    # Optional prefix prepended to each label string before processing.
    label_prefix: str = ""

    # ==========================================================================
    # MODELS — one rung per phase
    # ==========================================================================

    model_discovery: str = get_step_model("classifier_discovery")
    model_facet_consolidation: str = get_step_model("classifier_facet_consolidation")
    model_attribute_consolidation: str = get_step_model("classifier_attribute_consolidation")
    model_assignment: str = get_step_model("classifier_assignment")
    model_refinement: str = get_step_model("classifier_refinement")
    model_cross_domain: str = get_step_model("classifier_cross_domain")
    model_valence_merge: str = get_step_model("classifier_valence_merge")

    qr_temperature: float = 0.3

    # ==========================================================================
    # OUTPUT CEILINGS
    # ==========================================================================
    #
    # A high ceiling is free — billing is per generated token, and
    # smoothRequester throttles on measured throughput (it estimates from the
    # prompt and corrects from actuals), not on this value. Too low is the only
    # real failure: at 4000 a 22-attribute domain truncated during consolidation
    # and lost it entirely. Upper bound is the model's own max_output — 128000
    # for gpt-5.4, 32000 for gpt-4.1 (see OPENAI_MODEL_LIMITS in config.py).
    #
    # Discovery, consolidation and refinement enumerate an open-ended nested
    # list, so their response grows with the data.
    qr_max_tokens_discovery: int = 32000
    qr_max_tokens_consolidation: int = 32000

    # Assignment takes one label and returns one id: bounded by construction,
    # so it needs no headroom.
    qr_max_tokens_assignment: int = 4000

    # ==========================================================================
    # CHUNKING — discovery input, per domain
    # ==========================================================================

    batch_size_min: int = 100      # no splitting below this (single chunk)
    batch_size_max: int = 150      # ceiling: keeps prompt quality high
    target_batches: int = 6        # ideal number of chunks
    chunk_overlap: float = 0.2     # overlap fraction between adjacent chunks

    # ==========================================================================
    # CONSOLIDATION - two phases, two scopes, two caps
    # ==========================================================================

    # A scope over its cap is consolidated in rounds — round one on what the
    # chunks proposed, round two on the survivors of round one, and so on until
    # it fits in a single group. One budget, since both phases round the same
    # way and a phase that runs out says so in the action log.
    consolidation_max_rounds: int = 5

    # The facet call renders attribute names only, so what bounds it is the
    # number of facets being compared, not the volume hanging under them.
    facet_consolidation_max_facets_per_call: int = 40

    # The attribute call is one facet's pool. Measured 2026-08-15: the largest
    # pool in a real domain was twenty-six, so this rarely rounds.
    attribute_consolidation_max_attributes_per_call: int = 60

    # ==========================================================================
    # REFINEMENT
    # ==========================================================================

    # How many distinct response texts to show per attribute. This is the
    # phase's whole point — it judges real contents, not the label — so the
    # window has to be wide enough to expose a foreign concept hiding inside a
    # bucket.
    contents_top_n: int = 12

    # ==========================================================================
    # OUTPUT
    # ==========================================================================

    verbose: bool = True

    # A phase NAME to stop after; None runs everything. An unknown name is a
    # ValueError at construction — the numeric predecessor ran the full pipeline
    # for every value that was not a stop point, and that cost a full run to
    # discover.
    stop_after_phase: Optional[str] = None
