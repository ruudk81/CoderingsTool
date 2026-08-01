#%%
"""
View the prompts step 6 sent to the LLM: code assignment.

Usage:
    cd src && python -m pipeline.step_6_codeAssigner.view_prompts
"""

import sys
from pathlib import Path

src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

from utils.promptViewer import render

from test_data import TEST_DATA

from pipeline.step_6_codeAssigner.prompts_codeAssigner import CodeAssignmentResponse

SHOW_ALL = False

PROMPT_MODELS = {
    "code_assignment": CodeAssignmentResponse,
}


if __name__ == "__main__":
    render(step=6, models=PROMPT_MODELS, test_data=TEST_DATA, show_all=SHOW_ALL)
