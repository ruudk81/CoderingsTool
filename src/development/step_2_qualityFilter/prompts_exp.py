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

Your task: classify the response as meaningful, uncertain, or unusable.

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

# Coding Rules

## 99999997 — Don't Know / Uncertainty
The respondent clearly expresses:
- "I don't know", "No idea", "Unsure"
- "N/A", "Not applicable"
- "?", or any equivalent expression of uncertainty

## 99999999 — Gibberish OR Completely Off-topic

### A) Gibberish:
- Random characters ("asdf", "!!!")
- Placeholder text ("test", "lorem ipsum")
- Repeating the question without answering

### B) Completely Off-topic:
- Response is understandable BUT has ZERO relation to the question
- Does not attempt to answer at all

IMPORTANT:
- If there is ANY attempt to answer, even weak or vague → DO NOT use this code
- Minimal answers like "Nothing" or "None" can be valid → treat as meaningful if relevant

## null — Meaningful
The response engages with the survey question in any way, even if short, vague, or poorly written.

---

# Decision Process

Step 1: Does it express uncertainty?
→ YES: return 99999997

Step 2: Is it gibberish or completely off-topic?
→ YES: return 99999999

Else:
→ return null

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
