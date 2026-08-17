#%%
"""
View the prompts step 4 sent to the LLM.

Nine phases: discovery, facet consolidation, facet assignment, facet settle,
attribute consolidation, assignment, refinement, cross-domain, then the
valence-neutral merge. Keys are named by function, not by number, so a
reordering does not force a renaming here or in the perf model.

Usage:
    cd src && python -m pipeline.step_4_classifier.view_prompts
"""

import sys
from pathlib import Path

src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

from utils.promptViewer import render

from test_data import TEST_DATA

from pipeline.step_4_classifier.prompts_discovery import (
    DiscoveryResult,
)
from pipeline.step_4_classifier.prompts_consolidation import (
    AttributeConsolidationResult, FacetConsolidationResult,
)
from pipeline.step_4_classifier.prompts_refinement import RefinementResult
from pipeline.step_4_classifier.prompts_valence import ValenceNeutralRenameResponse
from pipeline.step_4_classifier.prompts_assignment import build_facet_assignment_model
from pipeline.step_4_classifier.prompts_facet_settle import build_facet_settle_model

SHOW_ALL = True

# Four phases build their response model at call time, because the menu ids are
# Literals in the schema. Where the capture carries those ids, the builder
# rebuilds the very model instructor was handed; where it carries only a count,
# there is nothing to rebuild and the entry renders the prompt alone.
#
# A capture taken before the ids were recorded has none, and a viewer that dies
# on an old export is worse than one that says it cannot build the schema — so
# a missing id list returns None and takes the same route as the two phases that
# never carry one.
def _facet_assignment_model(metadata: dict):
    ids = metadata.get("facet_ids")
    return build_facet_assignment_model(ids) if ids else None


def _facet_settle_model(metadata: dict):
    ids = metadata.get("facet_ids")
    return build_facet_settle_model(
        ids, metadata.get("attribute_ids") or []) if ids else None


PROMPT_MODELS = {
    "discovery": DiscoveryResult,
    "facet_consolidation": FacetConsolidationResult,
    "facet_assignment": _facet_assignment_model,
    "facet_settle": _facet_settle_model,
    "attribute_consolidation": AttributeConsolidationResult,
    "assignment": (lambda metadata: None),
    "refinement": RefinementResult,
    "cross_domain": (lambda metadata: None),
    "valence_merge": ValenceNeutralRenameResponse,
}


if __name__ == "__main__":
    render(step=4, models=PROMPT_MODELS, test_data=TEST_DATA, show_all=SHOW_ALL)
