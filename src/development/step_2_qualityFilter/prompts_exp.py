"""
Prompt for Step 2: Quality Filtering
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field

# =============================================================================
# STEP 2: QUALITY FILTERING
# =============================================================================

GRADER_INSTRUCTIONS = """
You are a research assistant evaluating an open-ended survey response.

Your task: classify the response as one of the following:
- Meaningful (null) — the response attempts to answer the question
- Uncertain (99999997) — the respondent expresses inability or unwillingness to answer
- Unusable (99999999) — the response is gibberish or completely off-topic

---

Language:
<language>
{language}
</language>

Survey question:
<survey_question>
{var_lab}
</survey_question>

Response to evaluate:
<response>
{response_text}
</response>

---

# Step-by-Step Decision Guide

## 1. Check for Uncertainty → 99999997

Classify as Uncertain if the response explicitly indicates the respondent cannot or will not provide an answer.

This includes:
- Direct statements: "I don't know", "No idea", "Not sure", "Unsure"
- Absence of answer: "No answer", "No explanation", "N/A", "Not applicable"
- Minimal uncertainty signals: "?", "-"
- Equivalent phrases in any language

RULE: If the respondent admits uncertainty or declines to answer, do NOT try to interpret further. Return 99999997.

## 2. Check for Unusable → 99999999

### A) Gibberish
- Random or meaningless input: "asdf", "qwerty", "!!!", "123123"
- Placeholder or test text: "test", "lorem ipsum"
- Copying/repeating the question without answering
- Strings with no interpretable meaning

### B) Completely Off-topic
- The response is understandable but does not attempt to answer the question at all
- No logical connection to the survey question

RULE: If a reasonable human reader would say "this doesn't answer the question at all", return 99999999.

## 3. Otherwise → Meaningful (null)

Classify as Meaningful if the response attempts to answer the question, even if:
- It is vague
- It is poorly written
- It is short or incomplete
- It contains minor irrelevance

RULE: Any attempt to answer = meaningful. Return null.

---

# Edge Cases

- Very short answers ("yes", "no", "maybe") → Meaningful if they logically relate to the question
- Partially relevant responses → Meaningful (do NOT over-penalize)
- Mixed responses (relevant + irrelevant content) → Meaningful if any part answers the question
- Sarcasm or unclear tone → Meaningful if it still answers; Unusable if not

---

# Priority Order

1. Uncertainty (99999997) → if explicitly stated
2. Unusable (99999999) → if no valid attempt to answer
3. Meaningful (null) → everything else

---

# Output

Return quality_filter_code only (99999997, 99999999, or null) following the response schema provided.
"""


QualityCode = Optional[Literal[99999997, 99999999]]

class QualityFilterLLMResponseExp(BaseModel):
    """Quality filter classification result. LLM returns only quality_filter_code."""

    quality_filter_code: QualityCode = Field(
        default=None,
        description="99999997 = don't know; 99999999 = gibberish/off-topic; null = meaningful",
    )
