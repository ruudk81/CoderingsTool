#%%
"""
View the prompts step 4 sent to the LLM.

Six phases: discovery, chunk consolidation, assignment, refinement,
cross-domain, then the valence-neutral merge. Keys are named by function, not by
number, so a reordering does not force a renaming here or in the perf model.

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
    ConsolidationResult,
    DiscoveryResult,
)
from pipeline.step_4_classifier.prompts_refinement import RefinementResult
from pipeline.step_4_classifier.prompts_valence import ValenceNeutralRenameResponse

SHOW_ALL = False

# Assignment and cross-domain build their response model at call time — the menu
# ids are Literals in the schema — so there is no static model to render.
PROMPT_MODELS = {
    "discovery": DiscoveryResult,
    "chunk_consolidation": ConsolidationResult,
    "assignment": (lambda metadata: None),
    "refinement": RefinementResult,
    "cross_domain": (lambda metadata: None),
    "valence_merge": ValenceNeutralRenameResponse,
}


if __name__ == "__main__":
    render(step=4, models=PROMPT_MODELS, test_data=TEST_DATA, show_all=SHOW_ALL)
