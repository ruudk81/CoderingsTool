"""
Prompt for Step 2: Quality Filtering
"""

from typing import Any, Optional, Union, Literal, List
from pydantic import BaseModel, Field, model_validator

# =============================================================================
# STEP 2: QUALITY FILTERING
# =============================================================================

GRADER_INSTRUCTIONS = """
You are a {language} language grader evaluating open-ended survey responses.
Your task is to determine whether each response provides **usable, on-topic content in relation to the specific survey question**, and assign the appropriate quality_filter_code.

For each response, assign exactly one quality_filter_code:

==================================================
CODE 99999997 — Don't Know / Uncertainty
==================================================

Use this code if the respondent explicitly expresses uncertainty, lack of knowledge, or non-applicability.

This includes any response whose clear meaning in {language} is equivalent to:
- "I don't know"
- "N/A"
- "Not applicable"
- "No idea"
- "Unsure"
- "Can't say"
- "?"

==================================================
CODE 99999999 — Nonsensical/Gibberish OR completely Off-topic
==================================================

Use this code for **two different kinds of unusable responses**:

A) Pure gibberish / nonsensical
   Examples:
   - Random characters: "asdfkj", "jjjjj", "x!@#%"
   - Placeholder text: "lorem ipsum", "test test"
   - Verbatim repetition of the question with no added content
   - Completely unintelligible text

B) Intelligible but completely off-topic / totally irrelevant
   The response is understandable {language} , BUT:
   - It does NOT address the actual survey question ({var_lab}) even remotely, OR
   - It obviously avoids the question.

Illustrative examples in English (if the question is about public transport):
- "Nothing"
- "I love dogs."
- "The weather is nice today."
- "Pizza is better than pasta."
- "I work in finance."
- A personal story that has nothing to do with transportation.

These are NOT "I don't know" — they are simply irrelevant to the question.

==================================================
CODE null — Meaningful / On-topic Response
==================================================

A response is meaningful if:
- It is understandable in {language}, AND
- It engages with or relates to the survey question ({var_lab}), even if:
  - It is very short
  - It is vague
  - It is opinionated
  - It is critical
  - It is poorly written
  - It is partially incomplete

Examples (if the question is about public transport):
- "Buses are always late."
- "Too crowded."
- "Tickets are expensive."
- "The metro is unreliable."

==================================================
SURVEY QUESTION
<survey_question>
{var_lab}
</survey_question>

RESPONSES TO EVALUATE
<responses>
{responses}
</responses>

==================================================
DECISION RULE (FOLLOW EXACTLY)

For each response, apply these steps in order:

1. Does the response explicitly express uncertainty or "I don't know"?
   - If YES → quality_filter_code = 99999997
   - If NO → go to Step 2

2. Does this response provide usable content that addresses the survey question, even remotely?
   - If NO (gibberish OR off-topic) → quality_filter_code = 99999999
   - If YES → quality_filter_code = null
"""


QualityCode = Optional[Literal[99999997, 99999999]]

class QualityFilterLLMResponseExp(BaseModel):
    """A single quality filter assessment result."""

    respondent_id: Any = Field(
        description="The respondent's ID from the input (preserve exact type and format)"
    )

    response: Union[str, float, int, None] = Field(
        description="The exact response text being evaluated"
    )

    quality_filter_code: QualityCode = Field(
        default=None,
        description=(
            "99999997 = uncertainty / don't know; "
            "99999999 = gibberish OR completely off-topic; "
            "null = meaningful response"
        ),
        examples=[99999997, 99999999, None],
    )

    # Derived programmatically — not requested from LLM
    quality_filter: bool = False

    @model_validator(mode="after")
    def derive_quality_filter(self):
        """Derive quality_filter from quality_filter_code. No validation error possible."""
        self.quality_filter = self.quality_filter_code is not None
        return self
