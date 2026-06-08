"""
Prompt builders for Code Assigner (P10).

Single idea → code assignment.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field, field_validator

from pipeline.step_5_codeGenerator.prompts_codeGenerator import CodeFromAttributes
from pipeline.step_6_codeAssigner.models_codeAssigner import CodeAssignment, CodeAssignmentBatch

# Tier-aware validation: True for mini/default (strict), False for nano (lenient).
# Set once at CodeAssigner init via configure_validation_mode().
_strict_assignment: bool = True


def configure_validation_mode(model: str) -> None:
    """Set validation strictness based on model tier. Call from CodeAssigner.__init__."""
    global _strict_assignment
    _strict_assignment = "nano" not in model.lower()


# =============================================================================
# §10 CODE ASSIGNMENT (P10) — single idea
# =============================================================================

def _build_codes_block(
    codes: List[CodeFromAttributes],
    other_label: Optional[str] = None,
) -> str:
    """Format codes for assignment prompt (code-only, no attributes)."""
    lines = []
    for i, code in enumerate(codes, 1):
        diagnostic = getattr(code, 'diagnostic_test', '') or ''
        indicators = ", ".join(code.typical_indicators[:5]) if code.typical_indicators else "(none)"
        block = (
            f"[C{i}] {code.code_name}\n"
            f"    Definition: {code.definition}\n"
        )
        if diagnostic:
            block += f"    Diagnostic: {diagnostic}\n"
        block += f"    Indicators: {indicators}"
        lines.append(block)

    if other_label:
        n = len(codes) + 1
        lines.append(
            f"[C{n}] {other_label}\n"
            f"    Definition: Ideas that do not clearly fit any of the above codes.\n"
            f"    Indicators: no matching indicators"
        )

    return "\n\n".join(lines)


def build_code_assignment_prompt(
    *,
    survey_question: str,
    language: str,
    dataset_context_section: str,
    codes: List[CodeFromAttributes],
    other_label: Optional[str],
    idea,
    facet_lookup: Optional[Dict[str, str]] = None,
) -> str:
    """Build prompt for assigning a single idea to a code."""
    codes_block = _build_codes_block(codes, other_label)

    # Format single idea (verbatim response + abstraction ladder for disambiguation)
    valence = getattr(idea, 'valence', '') or '0'
    facet = (facet_lookup or {}).get(idea.idea_id, '') or getattr(idea, 'facet', '') or ''
    domain = getattr(idea, 'domain', '') or ''
    interpretation = getattr(idea, 'interpretation', '') or ''
    abstraction = getattr(idea, 'abstraction', '') or ''

    idea_block = (
        f"response (verbatim): {idea.idea}\n"
        f"interpretation: {interpretation}\n"
        f"abstraction: {abstraction}\n"
        f"domain: {domain}\n"
        f"facet: {facet}\n"
        f"valence: {valence}"
    )

    other_label_display = other_label or "Other"

    return f"""You are a qualitative coding assistant. Assign the idea below to the best-matching code.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

<codebook>
{codes_block}
</codebook>

<idea>
{idea_block}
</idea>

<instructions>
1. Read the verbatim response together with its interpretation, abstraction, domain, facet, and valence. The verbatim response may be a bare word; use the interpretation and abstraction to disambiguate its intended meaning.
2. Find the code whose definition best matches what the respondent is expressing.
3. Return the code ID from [C#] brackets (e.g. "C1"). Do NOT return the code name.
4. Assign "{other_label_display}" only if NO code fits at all.
5. Rate confidence: 0.90+ = clear, 0.70-0.89 = good, 0.50-0.69 = approximate, <0.50 = weak.
6. Provide a brief rationale for your code choice.

All output MUST be in {language}.
Provide output as valid JSON following the response schema provided.
</instructions>
"""


class CodeAssignmentResponse(BaseModel):
    """Single idea → code assignment."""
    assigned_code_id: str = Field(
        ...,
        description="The code ID from the [C#] prefix (e.g. 'C1', 'C7'). Return ONLY the ID."
    )
    confidence: float = Field(
        ...,
        description="Confidence in the assignment (0.0 to 1.0)"
    )
    rationale: str = Field(
        ...,
        description="Brief rationale for the code choice"
    )

    @field_validator('assigned_code_id', mode='before')
    @classmethod
    def validate_code_id(cls, v):
        if not v:
            if _strict_assignment:
                raise ValueError("assigned_code_id is required")
            return ""
        return str(v).strip()

    @field_validator('confidence', mode='before')
    @classmethod
    def validate_confidence(cls, v):
        if v is None:
            if _strict_assignment:
                raise ValueError("confidence is required")
            return 0.0
        try:
            return float(v)
        except (TypeError, ValueError):
            if _strict_assignment:
                raise
            return 0.0

    @field_validator('rationale', mode='before')
    @classmethod
    def validate_rationale(cls, v):
        if not v:
            if _strict_assignment:
                raise ValueError("rationale is required")
            return ""
        return str(v).strip()
