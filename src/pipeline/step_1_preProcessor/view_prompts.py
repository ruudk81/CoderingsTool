#%%
"""
View the prompts step 1 sent to the LLM: spell correction.

Only the LLM correction stage sends a prompt; normalisation and finalisation are
deterministic. One example is captured per run, not one per respondent.

Usage:
    cd src && python -m pipeline.step_1_preProcessor.view_prompts
"""

import sys
from pathlib import Path

src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

from utils.promptViewer import render

from test_data import TEST_DATA

from pipeline.step_1_preProcessor.prompts_preProcessor import LLMCorrectionResponse

SHOW_ALL = False

PROMPT_MODELS = {
    "spell_correction": LLMCorrectionResponse,
}


if __name__ == "__main__":
    render(step=1, models=PROMPT_MODELS, test_data=TEST_DATA, show_all=SHOW_ALL)
