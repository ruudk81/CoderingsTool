#%%
"""
View the prompts step 2 sent to the LLM: quality assessment.

Usage:
    cd src && python -m pipeline.step_2_qualityFilter.view_prompts
"""

import sys
from pathlib import Path

src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

from utils.promptViewer import render

from test_data import TEST_DATA

from pipeline.step_2_qualityFilter.prompts_qualityFilter import (
    QualityFilterStructuredResponse,
)

SHOW_ALL = False

PROMPT_MODELS = {
    "quality_assessment": QualityFilterStructuredResponse,
}


if __name__ == "__main__":
    render(step=2, models=PROMPT_MODELS, test_data=TEST_DATA, show_all=SHOW_ALL)
