#%%
"""
View the prompts step 4 sent to the LLM (P1-P8).

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
    FacetDiscoveryResult,
    FacetConsolidatedResponse,
    FacetAssignmentResult,
    AttributeDiscoveryResult,
    AttributeChunkConsolidatedResponse,
    AttributeAssignmentResult,
    InFacetConsolidatedResponse,
    ValenceNeutralRenameResponse,
)

SHOW_ALL = False

PROMPT_MODELS = {
    "facet_discovery": FacetDiscoveryResult,                       # P1
    "facet_consolidation": FacetConsolidatedResponse,              # P2
    "facet_assignment": FacetAssignmentResult,                     # P3
    "attribute_discovery": AttributeDiscoveryResult,               # P4
    "attribute_chunk_consolidation": AttributeChunkConsolidatedResponse,  # P5
    "attribute_assignment": AttributeAssignmentResult,             # P6
    "in_facet_consolidation": InFacetConsolidatedResponse,         # P7
    "valence_neutral_rename": ValenceNeutralRenameResponse,        # P8
}


if __name__ == "__main__":
    render(step=4, models=PROMPT_MODELS, test_data=TEST_DATA, show_all=SHOW_ALL)
