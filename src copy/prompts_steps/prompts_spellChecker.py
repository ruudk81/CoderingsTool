"""
Prompts for Step 1: Spell Checking

Contains the spell check instruction prompt and response models.
"""

from __future__ import annotations
from typing import Any, List
from pydantic import BaseModel, Field


# =============================================================================
# STEP 1: SPELL CHECKING
# =============================================================================

SPELLCHECK_INSTRUCTIONS = """
You are a {language} language expert specializing in correcting misspelled words in open-ended survey responses.
Your task is to process correction tasks for responses that contain placeholder tokens indicating spelling mistakes.

First, here is the survey question that the responses are answering:
<survey_question>
{var_lab}
</survey_question>

For each correction task, you will receive:
- A sentence with one or more <oov_word> placeholders
- A list of misspelled words, in the same order as the placeholders
- A list of suggested corrections, in the same order

Follow these rules when making corrections:
1. Replace each <oov_word> placeholder with the best possible correction of the corresponding misspelled word.
2. Consider the meaning and context of the survey question when choosing corrections.
3. If a better correction exists than the ones provided, use that instead.
4. You may split a misspelled word into two words only if the split preserves the intended meaning and fits grammatically.
5. If no suitable correction is possible, use "[NO RESPONSE]" as the corrected sentence for that task.

Here are the correction tasks to process:
<correction_tasks>
{tasks}
</correction_tasks>

Additional guidelines:
- Pay close attention to the context and meaning of each response when making corrections.
- Ensure that your corrections maintain the original intent of the respondent.
- If a suggested correction doesn't fit the context, consider alternative corrections that preserve the meaning.

Begin processing the correction tasks now and provide your output as valid JSON following the response schema provided.
"""


class CorrectionItem(BaseModel):
    """A single spell correction result."""
    respondent_id: Any = Field(
        description="The respondent ID from the correction task"
    )
    corrected_response: str = Field(
        description="The fully corrected response with all spelling mistakes fixed"
    )


class LLMCorrectionResponse(BaseModel):
    """Structured output for spell check corrections."""
    corrections: List[CorrectionItem] = Field(
        description="List of corrections, one for each task in the input"
    )
