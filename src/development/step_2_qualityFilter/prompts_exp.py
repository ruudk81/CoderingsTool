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
- 99999997 — Don't Know / Uncertainty 
- 99999999 — Gibberish OR Completely Off-topic
- null - no classification
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

## 1. Responses express "Don't Know / Uncertainty" anawers → 99999997

This includes:
- Direct statements: "I don't know", "No idea", "Not sure", "Unsure"
- Absence of answer: "No answer", "No explanation", "N/A", "Not applicable"
- Minimal uncertainty signals: "?", "-"
- Equivalent phrases in any language

RULE: If the respondent admits uncertainty or declines to answer, do NOT try to interpret further. Return 99999997.

## 2. Responses express "ibberish OR Completely Off-topic" anawers → 99999999

This includes:
- Gibberish:
    - Random or meaningless input: "asdf", "qwerty", "!!!", "123123"
    - Placeholder or test text: "test", "lorem ipsum"
    - Copying/repeating the question without answering
    - Strings with no interpretable meaning
- Completely Off-topic
    - The response is understandable but does not attempt to answer the question at all
    - No logical connection to the survey question


## 3. Otherwise → null

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
