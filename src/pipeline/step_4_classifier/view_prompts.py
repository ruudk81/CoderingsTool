#%%
"""
View the prompts step 4 sent to the LLM.

Nine phases: discovery, consolidation, assignment and refinement per level,
then the valence-neutral merge. Keys are named by function, not by number, so
a reordering does not force a renaming here or in the perf model.

Usage:
    cd src && python -m pipeline.step_4_classifier.view_prompts
"""

import sys
from pathlib import Path

src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

from utils.promptViewer import render

from test_data import TEST_DATA

from pipeline.step_4_classifier.prompts_attribute import (
    AttributeConsolidationResult,
    AttributeDiscoveryResult,
    AttributeRefinementResult,
)
from pipeline.step_4_classifier.prompts_facet import (
    FacetConsolidationResult,
    FacetDiscoveryResult,
    FacetRefinementResult,
)
from pipeline.step_4_classifier.prompts_valence import ValenceNeutralRenameResponse

SHOW_ALL = False

# Both assignment phases build their response model at call time — the menu ids
# and the idea ids are Literals in the schema — so there is no static model to
# render for them.
PROMPT_MODELS = {
    "facet_discovery": FacetDiscoveryResult,
    "facet_consolidation": FacetConsolidationResult,
    "facet_assignment": (lambda metadata: None),
    "facet_refinement": FacetRefinementResult,
    "attribute_discovery": AttributeDiscoveryResult,
    "attribute_consolidation": AttributeConsolidationResult,
    "attribute_assignment": (lambda metadata: None),
    "attribute_refinement": AttributeRefinementResult,
    "valence_merge": ValenceNeutralRenameResponse,
}


if __name__ == "__main__":
    render(step=4, models=PROMPT_MODELS, test_data=TEST_DATA, show_all=SHOW_ALL)
