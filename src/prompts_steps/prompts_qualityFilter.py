"""
Prompts for Step 2: Quality Filtering

Contains the quality filter grading prompt and response models.
"""

from __future__ import annotations
from typing import Any, Optional, Union, Literal, List
from pydantic import BaseModel, Field, model_validator


# =============================================================================
# STEP 2: QUALITY FILTERING
# =============================================================================

GRADER_INSTRUCTIONS = """
You are a {language} language grader evaluating open-ended survey responses.
Your task is to determine whether each response provides **usable, on-topic content in relation to the specific survey question**, and assign appropriate quality filter codes.

You will classify each response into one of three practical outcomes:

==================================================
OUTCOME A — Don't Know / Uncertainty
CODE: 99999997 | quality_filter = true
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

If a response fits this pattern → set:
quality_filter = true
quality_filter_code = 99999997

==================================================
OUTCOME B — Nonsensical/Gibberish OR completely Off-topic
CODE: 99999999 | quality_filter = true
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

Illustractive examples in English (if the question is about public transport):
- "Nothing"
- "I love dogs."
- "The weather is nice today."
- "Pizza is better than pasta."
- "I work in finance."
- A personal story that has nothing to do with transportation.

These are NOT "I don't know" — they are simply irrelevant to the question.

If a response fits **either A or B** → set:
quality_filter = true
quality_filter_code = 99999999


==================================================
OUTCOME C — Meaningful / On-topic Response
quality_filter = false | quality_filter_code = null
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

For this category → set:
quality_filter = false
quality_filter_code = null

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
   - If YES → quality_filter = true, quality_filter_code = 99999997
   - If NO → go to Step 2

2. Ask:
   "Does this response provide usable content that addresses the survey question, even remotely?"

   - If NO (because it is gibberish OR off-topic) →
     quality_filter = true, quality_filter_code = 99999999

   - If YES →
     quality_filter = false, quality_filter_code = null
"""


QualityCode = Optional[Literal[99999997, 99999999]]

class QualityFilterLLMResponse(BaseModel):
    """A single quality filter assessment result."""

    respondent_id: Any = Field(
        description="The respondent's ID from the input (preserve exact type and format)"
    )

    response: Union[str, float, int, None] = Field(
        description="The exact response text being evaluated"
    )

    quality_filter: bool = Field(
        description=(
            "true if the response is unusable (don't know OR gibberish/off-topic), "
            "false if the response is meaningful and addresses the question"
        ),
        examples=[True, False],
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

    @model_validator(mode="after")
    def check_consistency(self):
        if self.quality_filter and self.quality_filter_code is None:
            raise ValueError(
                "If quality_filter=true, quality_filter_code must be 99999997 or 99999999"
            )
        if not self.quality_filter and self.quality_filter_code is not None:
            raise ValueError(
                "If quality_filter=false, quality_filter_code must be null"
            )

        return self
