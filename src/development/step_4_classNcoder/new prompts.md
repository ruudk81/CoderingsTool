# P4.5 Codebook Consolidation — Ruud's prompt formalized (v2)

Decisions applied:
- No provenance tags (avoid cross-domain overlap logic)
- Added back: NEIGHBOURS check, ACTIONABILITY through survey lens
- Diagnostic test uses {dimension_name} not "the brand" or "the subject"
- No target count — just "fewest needed for full coverage"
- Removed "NOT allowed: features, attributes, or sub-themes" (too aggressive)
- Added survey question grounding to workflow
- Updated main task statement per Ruud's instruction

---

## Response Model

```python
class ConsolidatedCode(BaseModel):
    """A consolidated code with label and diagnostic test."""
    code_name: str = Field(
        ..., description="Short code name (3-5 word noun phrase)"
    )
    definition: str = Field(
        ..., description=(
            "A short interpretive claim that reads like an analyst conclusion. "
            "Avoid abstract phrasing like 'perception of', 'level of', 'degree of'."
        )
    )
    diagnostic_test: str = Field(
        ..., description=(
            "Completes the sentence: 'This is about whether ...' — "
            "must be unique per code and must not overlap with other codes."
        )
    )
    valence: str = Field(
        ..., description="One of: 'positive', 'negative', 'neutral'"
    )
    typical_indicators: List[str] = Field(
        ..., description="Words or phrases that signal this code"
    )
    source_attributes: List[str] = Field(
        default_factory=list,
        description="Attribute names this code is derived from (from all merged codes)"
    )


class CodebookConsolidationResultV2(BaseModel):
    """P4.5 output: consolidated codebook."""
    evaluation: str = Field(
        ..., description="Brief analysis of what was merged/removed and why"
    )
    codes: List[ConsolidatedCode] = Field(
        ..., description="Final MECE codebook"
    )
```

---

## Prompt Function

```python
def build_codebook_consolidation_prompt_v2(
    *,
    survey_question: str,
    language: str,
    dataset_context_section: str,
    dimension_name: str,
    dimension_description: str,
    raw_codes: List[CodeFromAttributes],
    code_provenance: Dict[int, str],
) -> str:
    # Format raw codes with valence tags (no domain provenance)
    code_lines = []
    for i, code in enumerate(raw_codes):
        provenance = code_provenance.get(i, "")
        valence_tag = ""
        if "::pos" in provenance:
            valence_tag = "(+) "
        elif "::neg" in provenance:
            valence_tag = "(-) "

        attrs = ", ".join(code.source_attributes[:5]) if code.source_attributes else "—"
        indicators = "; ".join(code.typical_indicators[:3]) if code.typical_indicators else "—"
        code_lines.append(
            f"[C{i+1}] {valence_tag}{code.code_name}\n"
            f"      Definition: {code.definition}\n"
            f"      Indicators: {indicators}\n"
            f"      Source attributes: {attrs}"
        )
    codes_block = "\n\n".join(code_lines)

    return f"""You are an expert in qualitative research.

Your task is to generate a parsimonious and unambiguous codebook from {len(raw_codes)} candidate codes. The codebook must contain codes that are mutually exclusive and collectively exhaustive. A critical aspect is that there is no conceptual overlap between codes, and codes should be semantically unambiguous through the lens of the coding dimension.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

<dimension_context>
Dimension: {dimension_name} — {dimension_description}
</dimension_context>

<candidate_codes>
{codes_block}
</candidate_codes>

## CRITICAL OBJECTIVE
Create the fewest codes needed for full coverage, without conceptual overlap or semantic ambiguity.
The result must be conceptually clean, mutually exclusive, and easy for human coders to apply consistently.

<core_principles>

### 1. MAXIMAL REDUCTION
- Merge all codes that express the same underlying idea
- Ignore wording differences and examples
- Stop only when further merging would collapse clearly different dimensions
- IMPORTANT: only merge codes that share the same valence — see Principle 2

### 2. VALENCE STRUCTURE (HARD CONSTRAINT)
- Each code must have exactly ONE valence: positive, negative, or neutral
- If a dimension has both positive (+) and negative (-) candidate codes, produce TWO separate codes — one positive, one negative
- Do NOT merge positive and negative codes into a single valence-neutral code
- Example: "Duurzaamheid (+)" and "Twijfel aan duurzaamheid (-)" must remain separate codes, NOT merged into "Duurzaamheid en ethiek"
- Neutral codes are for observations without evaluative direction

### 3. LATENT DIMENSION FOCUS
- Each code must represent **ONE distinct question about {dimension_name}**
- Test: each code must complete
  **"This is about whether {dimension_name} is …"**

### 4. STRICT MECE RULE (HARD CONSTRAINT)
- Codes must be:
  - **Mutually Exclusive** → no conceptual overlap
  - **Collectively Exhaustive** → cover all meaningful variation
- If two codes of the same valence could co-occur in the same sentence → **merge them**
- If they answer different questions → **keep them separate**

### 5. NEIGHBOURS CHECK
- If a human coder would hesitate between two codes for the same response, they should be one code
- Merge codes that are too similar to be distinctively applied

### 6. APPROPRIATE ABSTRACTION LEVEL
- Codes must be at the right level of abstraction for the dimension: {dimension_description}
- Merge codes that differ only in specific examples but describe the same general phenomenon
- Do not preserve detail that would make codes too narrow to apply consistently across responses

### 7. NON-REDUNDANCY RULE
- If removing a code does not reduce explanatory power → delete it
- Avoid near-synonyms or adjacent constructs

### 8. ACTIONABILITY
- Each code must represent something meaningful and actionable given the survey question
- Remove or merge codes that are too abstract or too narrow to be useful through the lens of the survey question

</core_principles>

<code_definition_requirements>

### DUAL-LAYER CODE DEFINITION (MANDATORY)
Each code MUST include:

**code_name**
- 3–5 word noun phrase
- Short, scannable, used for coding

**definition**
- A short interpretive claim
- Must read like an analyst conclusion
- Avoid vague abstract phrasing — be concrete and specific

### CLARITY TEST (MANDATORY)
Each code must include a diagnostic_test:
"This is about whether {dimension_name} is …"
- Must be unique per code
- Must not overlap with other codes

</code_definition_requirements>

<workflow>
Follow these steps (DO NOT SKIP):
1. Cluster similar codes by topic AND valence — keep positive (+) and negative (-) clusters separate
2. Merge aggressively within the same valence — never merge across valence
3. Test for MECE overlap — for each pair of same-valence codes, ask: "would a coder hesitate between these?"
4. Remove redundancy — for each code, ask: "does removing this reduce explanatory power?"
5. Ensure one clear dimension per code
6. Assign valence label (positive, negative, neutral) to each surviving code
7. Verify each surviving code is actionable through the lens of the survey question: "{survey_question}"
</workflow>

All output MUST be in {language}.

Provide output as valid JSON following the response schema provided."""
```

---

## Bridge to downstream

`convert_codes_to_mece_categories()` would need updating to map the new fields:

```python
def convert_consolidated_codes_to_mece_categories(
    codes: List[ConsolidatedCode],
) -> List[MECECode]:
    categories = []
    for code in codes:
        categories.append(MECECode(
            category_label=code.code_name,
            inclusion_definition=code.definition,
            boundary_test=code.diagnostic_test,
            diagnostic_signals=code.typical_indicators[:5],
            key_expressions=[],
            tiebreaker_rules=[],
            subcategories=[],
        ))
    return categories
```
