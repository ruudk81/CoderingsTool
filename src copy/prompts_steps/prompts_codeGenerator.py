"""
Prompt builders for Code Generator (P8-P9).

Organized in pipeline processing order:
  §8   Code Generation from Attributes (P8: per domain)
  §9   Codebook Consolidation (P9: cross-domain merge)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from utils.dimension_data import DimensionDefinition

from prompts_steps.prompts_classifier import DiscoveredAttribute


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
## STEP 1 — VALENCE SEPARATION (MANDATORY FIRST PASS)

Before any clustering:
1. Assign each attribute a valence:
    * Positive (praise / favorable evaluation)
    * Negative (criticism / unfavorable evaluation)
    * Neutral (only if purely descriptive)
2. NEVER group or merge across valence boundaries
3. If an attribute contains both positive and negative aspects:
    * Split it into separate entries BEFORE proceeding
4. Treat positive and negative versions of the same phenomenon as distinct codes

Rule: Opposite evaluations MUST NEVER be combined.

## STEP 2 — AGGRESSIVE MERGING WITHIN CLUSTERS

Within each valence + question cluster:

Merge until a coder would NEVER hesitate between remaining codes.

Strict Merge Rule: If both can apply to the same sentence → merge

## STEP 3 — MECHANISM PURITY CHECK

For each code, ask: Is this describing:
* a value (e.g., fair, responsible)
* a functional property (e.g., fast, easy to use)
* a perception/judgment (e.g., reliable, outdated)
* a cause/reason (e.g., due to specific actions or policies)

If mixed → SPLIT

## STEP 4 — NEIGHBOR STRESS TEST

For every pair of same-valence codes, ask: "Would a trained coder hesitate between these?"

If YES:
1. Try sharpening definitions
2. If still ambiguous → merge

## STEP 5 — ONE-SENTENCE COVERAGE TEST

Each code must pass: Can I explain what this covers in ONE sentence without listing multiple unrelated things?

If NO → split

## STEP 6 — NON-REDUNDANCY KILL STEP

For each code: "If I delete this, do I lose meaning?"

If NO → DELETE it

## STEP 7 — FINAL DIAGNOSTIC UNIQUENESS CHECK

Each code must complete the sentence:
"{domain_diagnostic}"

Rules:
* The completion must be specific and distinct
* It must reflect a SINGLE valence direction (positive OR negative)

If two codes produce similar completions → MERGE

If NO → delete

## STEP 8 — PREVALENCE WEIGHTING & STRUCTURAL BALANCING

Use attribute frequency to shape the FINAL codebook.

8.1 Core Structure Rule
- High-prevalence attributes MUST define the main codes
- The codebook should be built around a small number of dominant phenomena

8.2 Low-Prevalence Constraint
Low-frequency attributes MUST NOT become standalone codes unless:
- They represent a clearly distinct phenomenon, AND
- They cannot be meaningfully merged upward

Otherwise, they must be:
- Abstracted into a higher-level code, OR
- Combined into a broader shared category

8.3 Balancing Constraint — Structured Differentiation
- DO NOT collapse everything into a single dominant code
- If multiple distinct high- or mid-prevalence patterns exist:→ they MUST remain separate

8.4 Final Check
Ask: "Does this code exist because it is conceptually necessary, or just because it appeared rarely?"

If the latter → merge or remove
</workflow>

<hard_rules>
### NO DOUBLE-BARREL CODES
If a code name contains "and" joining unrelated concepts → abstract to single phenomenon code name

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
- Prevalence weight rule with balancing constraint  
</validation_checklist>

<code_template>
Each code must include:
- **code_name**: 3–5 word noun phrase, must reflect ONE dimension only
= **definition**: clear, interpretive claim — must specify what makes this DISTINCT
- **diagnostic_test**: Must follow: "{code_diagnostic}" — must NOT overlap with any other code
- **valence**: positive / negative / neutral
- **typical_indicators**: concrete phrases (not abstract labels)
- **source_attributes**: all merged origins
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
            "(2) aggressive merging within clusters, "
            "(3) mechanism purity check, "
            "(4) neighbour stress test, "
            "(5) one-sentence coverage test, "
            "(6) non-redundancy kill step, "
            "(8) final diagnostic uniqueness check, "
            "(8) prevalence weighting and structural balancing check"
        )
    )
    codes: List[ConsolidatedCode] = Field(
        ..., description="Final MECE codebook"
    )
