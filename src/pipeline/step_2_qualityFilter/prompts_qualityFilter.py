"""Prompt for Step 2: Quality Filtering

Two prompt variants:
- GRADER_INSTRUCTIONS_NANO: Raw text output with <scratchpad>/<category> tags (Pattern A)
- GRADER_INSTRUCTIONS_STRUCTURED: Instructor-compatible with Pydantic model (Pattern B)
"""

from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field


# =============================================================================
# SHARED PROMPT BODY (categories + decision rules, used by both variants)
# =============================================================================

_CATEGORIES_BLOCK = """A response should be flagged as noise ONLY if it clearly and unambiguously matches one of the categories below.
If a response CLEARLY carries substantive content about the question, it is meaningful: do NOT flag it.

**Category 1: Don't know / Not knowing the answer**
Explicit statements of not knowing, such as:
- "I don't know"
- "Not sure"
- "No idea"
- Equivalent phrases in any language

**Category 2: Not applicable / Not having the answer**
Explicit statement of absence, such as:
- "No answer"
- "Not applicable"
- "No explanation"
- Empty placeholders: "-", "?", "N/A"
- Equivalent phrases in any language

**Category 3: Deferral / Referring elsewhere**
The response explicitly points to an answer given elsewhere instead of answering here, such as:
- "Already mentioned it"
- "See previous question"
- "It's written above"
- "As said before"
- Equivalent phrases in any language
This category requires an EXPLICIT pointer elsewhere. A brief on-topic answer does NOT belong here — being short is not the same as not addressing the question.

**Category 4: No text / Empty**
Item nonresponse, such as:
- Completely blank responses
- Only whitespace
- Single characters like "-" or "?"
- Equivalent phrases in any language

**Category 5: Invalid test / Nonsense**
Random or meaningless text, such as:
- Keyboard mashing: "asdf", "qwerty", "jjjjj"
- Random punctuation: "!!!", "????"
- Placeholder text: "lorem ipsum", "test"
"""

_CONTEXT_BLOCK = """You are a quality filter for open-ended survey responses.
Your task is to decide whether a response genuinely carries NO substantive content, and should NOT be KEPT for analysis.

Here is the survey context:
<survey_context>
Language:
{language}

Survey question:
{var_lab}

About these responses: they are often short, informal, and briefly worded. Brevity is normal and is NOT a reason to filter — a short answer that says anything substantive about the question is a valid response.
</survey_context>

Here is the response you need to evaluate:
<response>
{response_text}
</response>

"""

_DECISION_RULE = """First, work through your evaluation following these three steps:
1. Interpret what the response says (translate if needed)
2. Consider whether the response should be categorized as noise WITHOUT HESITATION and AMBIGUITY
3. Then provide your final categorization. Return one of:

1 → Don't know / Not knowing the answer
2 → Not applicable  / Not having the answer
3 → Deferral / Referring elsewhere
4 → No text / Empty
5 → Invalid test / Nonsense
null → Keep the response"""


# =============================================================================
# PATTERN A: Nano — raw text output with XML tags
# =============================================================================

GRADER_INSTRUCTIONS_NANO = _CONTEXT_BLOCK + _CATEGORIES_BLOCK + _DECISION_RULE + """

Output

<scratchpad>
[Your analysis here following the three steps above]
</scratchpad>

<category>
[Return only the category number: 1, 2, 3, 4, 5 - or "null"]
</category>"""


# =============================================================================
# PATTERN B: Mini/default — instructor + Pydantic response model
# =============================================================================

GRADER_INSTRUCTIONS_STRUCTURED = _CONTEXT_BLOCK + _CATEGORIES_BLOCK + _DECISION_RULE + """

Begin processing now and provide your output as valid JSON following the response schema provided."""


class QualityFilterStructuredResponse(BaseModel):
    """Pydantic response model for instructor-based quality filtering (mini/default)."""
    scratchpad: str = Field(
        description="Your evaluation reasoning: 1) interpret the response, 2) assess noise fit, 3) categorize"
    )
    category: Optional[Literal[1, 2, 3, 4, 5]] = Field(
        default=None,
        description=(
            "Quality filter category: "
            "1 = don't know / not knowing the answer; "
            "2 = not applicable / not having the answer; "
            "3 = deferral / explicitly referring to an answer given elsewhere; "
            "4 = no text / empty; "
            "5 = invalid text / nonsense; "
            "null = keep the response (meaningful)"
        ),
        examples=[1, 5, None],
    )
