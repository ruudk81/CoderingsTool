“””Prompt for Step 2: Quality Filtering

Two prompt variants:
- GRADER_INSTRUCTIONS_NANO: Raw text output with <scratchpad>/<category> tags (Pattern A)
- GRADER_INSTRUCTIONS_STRUCTURED: Instructor-compatible with Pydantic model (Pattern B)
“””

from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field


# =============================================================================
# SHARED PROMPT BODY (categories + decision rules, used by both variants)
# =============================================================================

_CATEGORIES_BLOCK = “””A response should ONLY be classified as noise if it matches one of the following categories WITHOUT AMBIGUITY:

**Category 1: Don’t know / Not knowing the answer**
Explicit statements of not knowing, such as:
- “I don’t know”
- “Not sure”
- “No idea”
- Equivalent phrases in any language

**Category 2: Not applicable / Not having the answer**
Explicit statement of absence, such as:
- “No answer”
- “Not applicable”
- “No explanation”
- Empty placeholders: “-”, “?”, “N/A”
- Equivalent phrases in any language

**Category 3: Absence of answer / Not addressing the question**
Not addressing the question, by explicit statments such as:
- “Already mentioned it”
- “See previous question”
- “It’s written above”
- “As said before”
- “Already done”
- Equivalent phrases in any language

**Category 4: No text / Empty**
Item nonresponse, such as:
- Completely blank responses
- Only whitespace
- Single characters like “-” or “?”
- Equivalent phrases in any language

**Category 5: Invalid test / Nonsense**
Random or meaningless text, such as:
- Keyboard mashing: “asdf”, “qwerty”, “jjjjj”
- Random punctuation: “!!!”, “????”
- Placeholder text: “lorem ipsum”, “test”
“””

_CONTEXT_BLOCK = “””You are a strict quality filter for survey responses.
Your task is to evaluate whether a survey response should be flagged as low-quality or kept for analysis.

Here is the survey context:
<survey_context>
Language:
{language}

Survey question:
{var_lab}

Type of responses: Coarse, brief, informal, and low-effort statements, with occasional bursts of strong emotion or rare detailed insights.
</survey_context>

Here is the response you need to evaluate:
<response>
{response_text}
</response>

“””

_DECISION_RULE = “””First, work through your evaluation following these three steps:
1. Interpret what the response says (translate if needed)
2. Consider whether the response should be categorized as noise WITHOUT HESITATION and AMBIGUITY
3. Then provide your final categorization. Return one of:

1 → Don’t know / Not knowing the answer
2 → Not applicable  / Not having the answer
3 → Absence of answer / Not addressing the question
4 → No text / Empty
5 → Invalid test / Nonsense
null → Keep the response”””


# =============================================================================
# PATTERN A: Nano — raw text output with XML tags
# =============================================================================

GRADER_INSTRUCTIONS_NANO = _CONTEXT_BLOCK + _CATEGORIES_BLOCK + _DECISION_RULE + “””

Output

<scratchpad>
[Your analysis here following the three steps above]
</scratchpad>

<category>
[Return only the category number: 1, 2, 3, 4, 5 - or “null”]
</category>”””

# Backward compat alias
GRADER_INSTRUCTIONS = GRADER_INSTRUCTIONS_NANO


# =============================================================================
# PATTERN B: Mini/default — instructor + Pydantic response model
# =============================================================================

GRADER_INSTRUCTIONS_STRUCTURED = _CONTEXT_BLOCK + _CATEGORIES_BLOCK + _DECISION_RULE + “””

Begin processing now and provide your output as valid JSON following the response schema provided.”””


class QualityFilterStructuredResponse(BaseModel):
    “””Pydantic response model for instructor-based quality filtering (mini/default).”””
    scratchpad: str = Field(
        description=”Your evaluation reasoning: 1) interpret the response, 2) assess noise fit, 3) categorize”
    )
    category: Optional[Literal[1, 2, 3, 4, 5]] = Field(
        default=None,
        description=(
            “Quality filter category: “
            “1 = don’t know / not knowing the answer; “
            “2 = not applicable / not having the answer; “
            “3 = absence of answer / not addressing the question; “
            “4 = no text / empty; “
            “5 = invalid text / nonsense; “
            “null = keep the response (meaningful)”
        ),
        examples=[1, 5, None],
    )
