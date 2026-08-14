#%%
"""
View the prompts step 4 sent to the LLM (P1-P10).

Usage:
    cd src && python -m pipeline.step_4_classifier.view_prompts
"""

import sys
from pathlib import Path

src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

from utils.promptViewer import render

from test_data import TEST_DATA

from pipeline.step_4_classifier.prompts_classifier import (
    AxisSystemResponse,
    FacetDiscoveryResult,
    FacetAssignmentResult,
    InAxisConsolidatedResponse,
    AttributeDiscoveryResult,
    AttributeAssignmentResult,
    InFacetConsolidatedResponse,
    ValenceNeutralRenameResponse,
)

SHOW_ALL = False

PROMPT_MODELS = {
    "axis_discovery": AxisSystemResponse,                          # P1
    # P2's response model is built at call time (the domain's axis names are
    # a Literal in the schema), so there is no static model to render here.
    "tagged_facet_discovery": (lambda metadata: None),             # P2
    "facet_discovery": FacetDiscoveryResult,                       # P3
    "facet_assignment": FacetAssignmentResult,                     # P4
    "in_axis_consolidation": InAxisConsolidatedResponse,           # P5
    "attribute_discovery": AttributeDiscoveryResult,               # P6
    "attribute_assignment": AttributeAssignmentResult,             # P7
    "in_facet_consolidation": InFacetConsolidatedResponse,         # P8
    "valence_neutral_rename": ValenceNeutralRenameResponse,        # P9
}


if __name__ == "__main__":
    render(step=4, models=PROMPT_MODELS, test_data=TEST_DATA, show_all=SHOW_ALL)
