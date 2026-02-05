"""
Experimental Prompts for Step 2: Quality Filtering

This file contains the prompts used by qualityFilter.py.
Modify these prompts to experiment with different quality filtering approaches.

Original source: src/prompts.py (STEP 2: QUALITY FILTERING section)

Response models (Pydantic) are co-located with their prompts following the
migrate-output-schema pattern - instructor uses Field(description=...) to
communicate schema to the LLM.
"""

from typing import Any, Union, Optional
from pydantic import BaseModel, Field

# =============================================================================
# RESPONSE MODELS (co-located with prompts for instructor)
# =============================================================================

class QualityFilterLLMResponseExp(BaseModel):
    """A single quality filter assessment result."""
    respondent_id: Any = Field(
        description="The respondent's ID from the input"
    )
    response: Union[str, float, int, None] = Field(
        description="The exact response text being evaluated"
    )
    quality_filter: bool = Field(
        description="true if response is meaningless (don't know/gibberish), false if meaningful",
        examples=[True, False]
    )
    quality_filter_code: Optional[int] = Field(
        default=None,
        description="99999997 for uncertainty/don't know, 99999999 for gibberish/nonsensical, null for meaningful responses",
        examples=[99999997, 99999999, None]
    )


# =============================================================================
# STEP 2: QUALITY FILTERING
# =============================================================================

GRADER_INSTRUCTIONS = """
You are a {language} language grader evaluating open-ended survey responses.
Your task is to determine whether each response is meaningless and assign appropriate quality filter codes.

Task Description:
Analyze each response and classify it based on the following criteria:

Decision Criteria:
1. **Don't Know/Uncertainty (Code 99999997)**: Responses that express "don't know", "not applicable", or only express uncertainty
   - Examples: "I don't know", "N/A", "Not applicable", "No idea", "?"

2. **Nonsensical/Gibberish (Code 99999999)**: Responses that are meaningless, gibberish, or simply repeat the question
   - Examples: "asdfkj", "lorem ipsum", random characters, just repeating the question

3. **Meaningful Response (No Code)**: Responses that provide actual content, opinions, or information
   - These should have quality_filter = false and quality_filter_code = null

Input:
You will be provided with a survey question and a list of responses to evaluate.

Survey question:
<survey_question>
{var_lab}
</survey_question>

Here are the responses you need to evaluate:
<responses>
{responses}
</responses>

Follow these steps for each response:
1. Read the response carefully.
2. Determine if the response expresses uncertainty/don't know (code 99999997)
3. If not uncertainty, determine if it's gibberish/nonsensical (code 99999999)
4. If neither, it's meaningful (quality_filter = false, quality_filter_code = null)

Begin evaluating the responses now and provide your output as valid JSON following the response schema provided.
"""
