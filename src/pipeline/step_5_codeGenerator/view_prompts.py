#%%
"""
View the prompts step 5 sent to the LLM: codebook generation and consolidation.

Usage:
    cd src && python -m pipeline.step_5_codeGenerator.view_prompts
"""

import sys
from pathlib import Path

src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

from utils.promptViewer import render

from test_data import TEST_DATA

from pipeline.step_5_codeGenerator.prompts_codeGenerator import (
    CodeGenerationFromAttributesResult,
    CodebookConsolidationResult,
)

SHOW_ALL = False

PROMPT_MODELS = {
    "code_generation_from_attributes": CodeGenerationFromAttributesResult,
    "codebook_consolidation": CodebookConsolidationResult,
}


if __name__ == "__main__":
    render(step=5, models=PROMPT_MODELS, test_data=TEST_DATA, show_all=SHOW_ALL)
