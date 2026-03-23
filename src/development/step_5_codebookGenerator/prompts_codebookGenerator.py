"""
Prompt builders for Codebook Generator (P8-P10).

Organized in pipeline processing order:
  §8   Code Generation from Attributes (P8: per domain)
  §9   Codebook Consolidation (P9: cross-domain merge)
  §10  Code Assignment (P10: single idea)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from development.step_3_ideaExtractor.dimension_data import DimensionDefinition

from development.step_4_classifier.prompts_classifier import DiscoveredAttribute
from development.step_5_codebookGenerator.models_codebookGenerator import CodeAssignment, CodeAssignmentBatch


# =============================================================================
# HELPERS (duplicated from classifier prompts for self-containment)
# =============================================================================

def _extract_key_idea(instruction: str) -> str:
    """Extract the 'Key idea: ...' sentence from an instruction string."""
    marker = "Key idea: "
    idx = instruction.find(marker)
    if idx == -1:
        return instruction.strip()
    return instruction[idx + len(marker):].strip().rstrip(".")


# =============================================================================
# §8 CODE GENERATION FROM ATTRIBUTES (P8)
# =============================================================================

def build_code_from_attributes_prompt(
    *,
    survey_question: str,
    language: str,
    dataset_context_section: str,
    dimension_def: Optional['DimensionDefinition'],
    dimension_name: str,
    dimension_description: str,
    domain_name: str,
    domain_definition: str,
    domain_attributes: Dict[str, Dict[str, List[DiscoveredAttribute]]],
    attribute_assignments: Optional[Dict[str, str]] = None,
    excluded_domains: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """Generate codebook codes from a structured attribute inventory.

    Args:
        dimension_def: DimensionDefinition for taxonomy structure lines (or None for fallback)
        domain_name: Name of the domain being processed
        domain_definition: Inclusion definition of the domain
        domain_attributes: {domain_name: {facet_name: [DiscoveredAttribute, ...]}}
        attribute_assignments: idea_id -> attribute_name, for frequency display
        excluded_domains: list of (name, definition) for other domains
    """
    # Dimension-specific taxonomy structure
    if dimension_def:
        noun_phrase = dimension_def.noun_phrase_descriptor
        domain_key_idea = _extract_key_idea(dimension_def.prompt_rules.domain_instruction)
        attribute_key_idea = _extract_key_idea(dimension_def.prompt_rules.attribute_instruction)
    else:
        noun_phrase = dimension_name
        domain_key_idea = "the subject the statement refers to"
        attribute_key_idea = "the specific observable property being described"

    # Excluded domains block (names only, no definitions)
    excluded_block_light = ""
    if excluded_domains:
        excl_names = [excl_name for excl_name, _ in excluded_domains]
        excluded_block_light = "\n".join(f"- {name}" for name in excl_names)

    # Compute attribute frequencies
    attr_counts: Dict[str, int] = {}
    if attribute_assignments:
        for attr_name in attribute_assignments.values():
            attr_counts[attr_name] = attr_counts.get(attr_name, 0) + 1

    # Build inventory: Facet > Attribute (single domain)
    facet_attrs = next(iter(domain_attributes.values()), {})
    inventory_lines = []
    for facet_name, attributes in sorted(facet_attrs.items()):
        for attr in attributes:
            examples = "; ".join(attr.example_observations[:2])
            count = attr_counts.get(attr.attribute_name, 0)
            freq_tag = f" [{count} ideas]" if attr_counts else ""
            line = f"- {attr.attribute_name}{freq_tag}: {attr.attribute_description}"
            if examples:
                line += f" (e.g., {examples})"
            inventory_lines.append(line)
    inventory_block = "\n".join(inventory_lines)

    return f"""You are tasked with deriving a PARSIMONIOUS codebook with MUTUALLY EXCLUSIVE and COLLECTIVELY EXHAUSTIVE codes that represent conceptually and semantically distinct PHENOMENA from a taxonomy inventory of attributes. These attributes were derived from written responses to a survey question.

# Survey Context

Here is the survey context:

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

# Taxonomy Context

Here is the taxonomy context you are working within:

<taxonomy_context>
This is the structure:
<taxonomy_structure>
- Dimension (L1): {dimension_name}: {noun_phrase}
- Domain (L2): {domain_key_idea}
- Attribute (L3): {attribute_key_idea}
</taxonomy_structure>

You are working within this dimension:
<taxonomy_dimension>
{dimension_name} — {dimension_description}
</taxonomy_dimension>

And you are working within this domain:
<taxonomy_domain>
{domain_name} — {domain_definition}
</taxonomy_domain>
</taxonomy_context>

# Attribute Inventory

Here is the inventory of attributes for you to analyze:
<attribute_inventory>
{inventory_block}
</attribute_inventory>

# Code Derivation Rules
<code_derivation_rules>

1. Phenomenon Rule
- Codes must represent underlying PHENOMENA rather than individual attributes. Multiple attributes describing different manifestations of the same underlying phenomenon MUST be merged into a single code.

2. Dimension Rule
Only include codes that belong to this domain:
- {domain_name} — {domain_definition}

Do not include codes that belong to these excluded domains:
{excluded_block_light}

3. Specificity Rule
- Do NOT create separate codes simply because attributes differ in specificity. General statements and specific examples should be treated as indicators of the same phenomenon.
- Example: "The train was delayed by 20 minutes" and "public transport is often late" both indicate unreliable punctuality and should be coded under the same broader phenomenon.

4. Prevalence Weighting Rule
The number of ideas linked to each attribute MUST guide code construction.
* Attributes with HIGH idea counts MUST define the core structure of the codebook.
* The codebook MUST be anchored in a small number of dominant phenomena, not a long tail of low-frequency codes.
* Attributes with LOW idea counts MUST NOT become standalone codes unless they represent a clearly distinct phenomenon that cannot be abstracted further.

LOW-prevalence attributes MUST be:
* abstracted into a higher-level phenomenon aligned with dominant patterns, OR
* combined into a broader conceptual category that captures their shared meaning.

Balancing Constraint — Structured Differentiation: Do NOT collapse all attributes into a single dominant "meta-code." If multiple conceptually distinct high- or mid-prevalence patterns exist, they MUST be represented as separate codes.

5. Mutual Exclusivity Rule
Codes must represent clearly different phenomena so that responses can be coded consistently without ambiguity.

6. Valence Sensitivity Rule
- Generate separate codes for positive and negative phenomena.
- Do NOT combine praise and criticism into a single code.
- If the attributes contain both positive and negative aspects of similar phenomena, create distinct codes for each valence direction.
</code_derivation_rules>

# Required Process

Before generating your final codes, you MUST work through your analysis step-by-step in a scratchpad. In your scratchpad field:

<required_process>
1. Identify higher-prevalence attributes (those with the largest idea count share) and treat them as anchors
2. Group attributes into underlying phenomena, giving priority to higher-prevalence clusters.
* Map low-prevalence attributes onto these dominant phenomena wherever possible.
* Abstract low-prevalence attributes to a broader conceptual level rather than preserving them as standalone codes.
* Only create a separate code for a low-prevalence attribute if it represents a clearly distinct phenomenon that cannot reasonably be merged without losing essential meaning.
* Do NOT collapse conceptually distinct attributes into a single broad meta-code merely because one cluster is highly prevalent.
3. Check for domain relevance - exclude any phenomena outside the allowed domain
4. Check for valence distinctions and split positive vs negative where needed
5. Name each phenomenon using a 3-5 word noun phrase
6. Verify parsimony - ensure the codebook is dominated by high-prevalence phenomena and contains a minimal number of codes (typically 5-8)
7. Explicitly justify any code that is primarily based on low-prevalence attributes instead of merging it
</required_process>

# Output Requirements

Provide output as valid JSON following the response schema provided.

# Language Requirement

All output (code names, definitions, typical indicators, and evaluation) must be written in {language}.

Begin now by applying the required process and then return only valid JSON."""


class CodeFromAttributes(BaseModel):
    """A formal qualitative code derived from attributes."""
    code_name: str = Field(
        ..., description="Short code name (3-5 word noun phrase)"
    )
    definition: str = Field(
        ..., description="Clear definition of what this code covers (1-2 sentences)"
    )
    typical_indicators: List[str] = Field(
        ..., description="Words or phrases that signal this code"
    )
    source_attributes: List[str] = Field(
        default_factory=list,
        description="Attribute names this code is derived from"
    )


class CodeGenerationFromAttributesResult(BaseModel):
    """P8 output: codes derived from attributes."""
    scratchpad: str = Field(
        ..., description=(
            "Step-by-step reasoning before deriving codes: "
            "(1) identify higher-prevalence attributes (largest idea counts) and treat them as anchors, "
            "(2) apply Prevalence Weighting Rule with Balancing Constraint, "
            "(3) check for domain relevance - exclude any phenomena outside the allowed domain, "
            "(4) check for valence distinctions and split positive vs negative where needed, "
            "(5) name each phenomenon (3–5 word noun phrase), "
            "(6) verify parsimony - ensure the codebook is dominated by high-prevalence phenomena and contains a minimal number of codes (typically 5–8), "
            "(7) explicitly justify any code that is primarily based on low-prevalence attributes instead of merging it"
        )
    )
    codes: List[CodeFromAttributes] = Field(
        ..., description=(
            "Formal codes derived from the attribute inventory. "
            "Codes should reflect dominant, high-prevalence phenomena, with low-prevalence attributes absorbed into broader codes where possible.")
    )


# =============================================================================
# §9 CODEBOOK CONSOLIDATION (P9) — cross-domain review & merge
# =============================================================================

def build_codebook_consolidation_prompt(
    *,
    survey_question: str,
    language: str,
    dataset_context_section: str,
    dimension_name: str,
    dimension_description: str,
    dimension_def: Optional['DimensionDefinition'] = None,
    raw_codes: List[CodeFromAttributes],
    code_provenance: Dict[int, str],
    code_frequencies: Optional[Dict[int, int]] = None,
) -> str:
    """Consolidate per-domain codes into a final parsimonious, MECE codebook.

    Args:
        raw_codes: All codes from P8 (per-domain)
        code_provenance: Maps code index to domain_name
        code_frequencies: Maps code index to approximate idea count
        dimension_def: DimensionDefinition for dimension-specific diagnostics
    """
    # Dimension-specific diagnostics
    if dimension_def:
        domain_diagnostic = dimension_def.prompt_rules.domain_diagnostic
        code_diagnostic = dimension_def.prompt_rules.code_diagnostic
    else:
        domain_diagnostic = "What question is being answered?"
        code_diagnostic = "This code is about …"

    # Format raw codes with domain provenance tags and frequency
    code_lines = []
    for i, code in enumerate(raw_codes):
        provenance = code_provenance.get(i, "")
        domain_tag = f"({provenance}) " if provenance else ""
        freq = code_frequencies.get(i, 0) if code_frequencies else 0
        freq_tag = f" (~{freq} ideas)" if freq > 0 else ""

        attrs = ", ".join(code.source_attributes[:5]) if code.source_attributes else "—"
        indicators = "; ".join(code.typical_indicators[:3]) if code.typical_indicators else "—"
        code_lines.append(
            f"[C{i+1}] {code.code_name}{freq_tag}\n" #{domain_tag}
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

Before generating your final codes, you MUST work through your analysis step-by-step in a scratchpad. In your scratchpad field:

<workflow>

## STEP 1 — PRE-STRUCTURE BY VALENCE (HARD GATE)

- Generate separate codes for positive and negative phenomena.
- Do NOT combine praise and criticism into a single code.
- If the attributes contain both positive and negative aspects of similar phenomena, create distinct codes for each valence direction.

## STEP 2 — CLUSTER BY LATENT QUESTION (NOT TOPIC)

Instead of grouping by topic, group by:

**{domain_diagnostic}**

If two codes answer the same question → same cluster

## STEP 3 — AGGRESSIVE MERGING WITHIN CLUSTERS

Within each valence + question cluster:

Merge until a coder would NEVER hesitate between remaining codes.

Strict Merge Rule: If both can apply to the same sentence → merge

## STEP 4 — MECHANISM PURITY CHECK

For each code, ask: Is this describing:
* a value (e.g., fair, responsible)
* a functional property (e.g., fast, easy to use)
* a perception/judgment (e.g., reliable, outdated)
* a cause/reason (e.g., due to specific actions or policies)

If mixed → SPLIT

## STEP 5 — NEIGHBOUR STRESS TEST

For every pair of same-valence codes, ask: "Would a trained coder hesitate between these?"

If YES:
1. Try sharpening definitions
2. If still ambiguous → merge

## STEP 6 — ONE-SENTENCE COVERAGE TEST

Each code must pass: Can I explain what this covers in ONE sentence without listing multiple unrelated things?

If NO → split

## STEP 7 — NON-REDUNDANCY KILL STEP

For each code: "If I delete this, do I lose meaning?"

If NO → delete

## STEP 8 — FINAL DIAGNOSTIC UNIQUENESS CHECK

Each code must complete: "{code_diagnostic}"

If two codes produce similar completions → merge

</workflow>

<hard_rules>

### DOMAIN AWARENESS
Codes from DIFFERENT domains that share similar names may represent DIFFERENT phenomena. Do NOT merge codes across domains unless they are truly identical in meaning.

### NO DOUBLE-BARREL CODES
If a code name contains "and" joining unrelated concepts → split into separate codes.

### NO CAUSE + ATTRIBUTE MIX
Do not combine a cause/reason with a descriptive attribute in a single code. Split into separate codes for each mechanism.

</hard_rules>

<validation_checklist>
Before finalizing, verify each code passes:
- Single valence only
- Answers ONE question
- Cannot co-occur with same-valence code
- Mechanism is pure
- One-sentence coverage
- Diagnostic is unique
</validation_checklist>

<code_template>
Each code must include:

**code_name**: 3–5 word noun phrase, must reflect ONE dimension only

**definition**: clear, interpretive claim — must specify what makes this DISTINCT

**diagnostic_test**: Must follow: "{code_diagnostic}" — must NOT overlap with any other code

**valence**: positive / negative / neutral

**typical_indicators**: concrete phrases (not abstract labels)

**source_attributes**: all merged origins
</code_template>

All output MUST be in {language}.

Provide output as valid JSON following the response schema provided."""

class ConsolidatedCode(BaseModel):
    """A consolidated code with diagnostic test for MECE verification."""
    code_name: str = Field(
        ..., description="Short code name (3-5 word noun phrase)"
    )
    definition: str = Field(
        ..., description=(
            "A short interpretive claim that reads like an analyst conclusion. "
            "Avoid vague abstract phrasing — be concrete and specific."
        )
    )
    diagnostic_test: str = Field(
        ..., description=(
            "Completes the dimension-specific diagnostic stem — "
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


class CodebookConsolidationResult(BaseModel):
    """P9 output: consolidated codebook."""
    scratchpad: str = Field(
        ..., description=(
            "Step-by-step reasoning following the 8-step workflow: "
            "(1) pre-structure by valence, "
            "(2) cluster by latent question, "
            "(3) aggressive merging within clusters, "
            "(4) mechanism purity check, "
            "(5) neighbour stress test, "
            "(6) one-sentence coverage test, "
            "(7) non-redundancy kill step, "
            "(8) final diagnostic uniqueness check"
        )
    )
    codes: List[ConsolidatedCode] = Field(
        ..., description="Final MECE codebook"
    )


# =============================================================================
# §10 CODE ASSIGNMENT (P10) — single idea
# =============================================================================

# Re-export data-flow wrapper models (canonical definition in models_codebookGenerator.py)
# CodeAssignment and CodeAssignmentBatch are imported at the top of this file.


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

    # Format single idea
    valence = getattr(idea, 'valence', '') or '0'
    facet = (facet_lookup or {}).get(idea.idea_id, '') or getattr(idea, 'facet', '') or ''
    domain = getattr(idea, 'domain', '') or ''

    idea_block = (
        f"idea: {idea.idea}\n"
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
1. Read the idea text, domain, facet, and valence.
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
