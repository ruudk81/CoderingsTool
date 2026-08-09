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

Three kinds of response look alike but are not the same. Separate them by what the
response reports:

- "I cannot produce an answer" — the respondent lacks the answer, the memory, or an
  opinion. Noise: category 1.
- "The answer is: none" — the respondent does answer, and what they report is that
  there is nothing. Noise: category 3.
- "I do not know the subject" — never heard of it, never used it, no experience with
  it. NOT noise: that reports a real fact about this respondent's relation to the
  subject the question asks about. Keep it.

**Category 1: Cannot give the answer**
Explicit statements of not knowing what to answer, or of having no opinion:
- "I don't know"
- "Not sure"
- "No idea"
- "No opinion"
- Equivalent phrases in any language
A response that instead states the subject itself is unknown to the respondent does
NOT belong here — that is meaningful content.

**Category 2: Not applicable**
The respondent states that the question does not apply to them:
- "Not applicable"
- "Does not apply to me"
- Abbreviations of the same, such as "N/A"
- Equivalent phrases in any language

**Category 3: No content to report**
The respondent does answer, and the answer is that there is nothing:
- "Nothing"
- "None"
- "None at all"
- "Nothing comes to mind"
- Equivalent phrases in any language
The test: the response can be rewritten as "the answer is: none". If it can only be
rewritten as "I cannot give the answer", it is category 1 instead.

**Category 4: Referring elsewhere**
The response explicitly points to an answer given elsewhere instead of answering here:
- "Already mentioned it"
- "See previous question"
- "It's written above"
- "As said before"
- Equivalent phrases in any language
This category requires an EXPLICIT pointer elsewhere. A brief on-topic answer does NOT belong here — being short is not the same as not addressing the question.

**Category 5: No text**
No answer was entered, or only a placeholder mark:
- Completely blank, or only whitespace
- Single characters such as "-", ".", "?"

**Category 6: Meaningless text**
Random or meaningless text:
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

1 → Cannot give the answer
2 → Not applicable
3 → No content to report
4 → Referring elsewhere
5 → No text
6 → Meaningless text
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
[Return only the category number: 1, 2, 3, 4, 5, 6 - or "null"]
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
    category: Optional[Literal[1, 2, 3, 4, 5, 6]] = Field(
        default=None,
        description=(
            "Quality filter category: "
            "1 = cannot give the answer: does not know, no opinion — NOT: unfamiliar with the subject, which is meaningful; "
            "2 = not applicable: the question does not apply to the respondent, incl. 'n/a'; "
            "3 = no content to report: the answer is 'none', 'nothing', 'nothing comes to mind'; "
            "4 = explicitly refers to an answer given elsewhere; "
            "5 = no text: blank, whitespace, or a bare placeholder mark such as '-' or '?'; "
            "6 = meaningless text: keyboard mashing, random punctuation, placeholder text; "
            "null = keep the response (meaningful)"
        ),
        examples=[1, 3, None],
    )
