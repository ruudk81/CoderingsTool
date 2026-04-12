"""
Prompt builders for Code Generator (P8-P9).

Organized in pipeline processing order:
  §8   Code Generation from Attributes (P8: per domain)
  §9   Codebook Consolidation (P9: cross-domain merge)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from pipeline.step_3_ideaExtractor.dimension_data import DimensionDefinition

from dataclasses import dataclass, field as dc_field

from pipeline.step_4_classifier.prompts_classifier import DiscoveredAttribute


# =============================================================================
# ENRICHED ATTRIBUTE (attribute + representative samples)
# =============================================================================

@dataclass
class EnrichedAttribute:
    """Attribute enriched with representative samples per valence group."""
    attribute: DiscoveredAttribute
    positive_samples: list = dc_field(default_factory=list)   # max 3 ideas
    neutral_samples: list = dc_field(default_factory=list)    # max 3 ideas
    negative_samples: list = dc_field(default_factory=list)   # max 3 ideas
    positive_count: int = 0   # total ideas with valence +
    neutral_count: int = 0    # total ideas with valence 0 or empty
    negative_count: int = 0   # total ideas with valence -


# =============================================================================
# HELPERS (duplicated from classifier prompts for self-containment)
# =============================================================================

def _format_sample(sample) -> str:
    """Format a representative sample as '- "instance → interpretation"'."""
    instance = (getattr(sample, "instance", "") or "").strip()
    interpretation = (getattr(sample, "interpretation", "") or "").strip()
    if instance and interpretation:
        return f'\n    - "{instance} → {interpretation}"'
    elif instance:
        return f'\n    - "{instance}"'
    return ""


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
    enriched_attributes: Optional[Dict[str, List[EnrichedAttribute]]] = None,
    theme_count_hint: Optional[tuple] = None,
) -> str:
    """Generate themes from a structured attribute inventory.

    Args:
        dimension_def: DimensionDefinition for taxonomy structure lines (or None for fallback)
        domain_name: Name of the domain being processed
        domain_definition: Inclusion definition of the domain
        domain_attributes: {domain_name: {facet_name: [DiscoveredAttribute, ...]}}
        attribute_assignments: idea_id -> attribute_name, for frequency display
        enriched_attributes: {facet_name: [EnrichedAttribute, ...]} for representative samples
        theme_count_hint: (low, high) theme count span from UMAP+HDBSCAN, or None
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

    # Compute attribute frequencies
    attr_counts: Dict[str, int] = {}
    if attribute_assignments:
        for attr_name in attribute_assignments.values():
            attr_counts[attr_name] = attr_counts.get(attr_name, 0) + 1

    # Build enriched lookup: attr_name -> EnrichedAttribute
    enriched_lookup: Dict[str, EnrichedAttribute] = {}
    if enriched_attributes:
        for facet_name, enriched_list in enriched_attributes.items():
            for ea in enriched_list:
                enriched_lookup[ea.attribute.attribute_name] = ea

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

            # Mixed valence: show ↑/↓ blocks with samples. Mono valence: no samples.
            ea = enriched_lookup.get(attr.attribute_name)
            if ea and ea.negative_samples:
                pos_neu_count = ea.positive_count + ea.neutral_count
                pos_neu_samples = list(ea.positive_samples) + list(ea.neutral_samples)
                if pos_neu_samples:
                    line += f"\n  ↑ [{pos_neu_count} ideas] Positive valence:"
                    for sample in pos_neu_samples:
                        line += _format_sample(sample)
                if ea.negative_samples:
                    line += f"\n  ↓ [{ea.negative_count} ideas] Negative valence:"
                    for sample in ea.negative_samples:
                        line += _format_sample(sample)

            inventory_lines.append(line)
    inventory_block = "\n\n".join(inventory_lines)

    # Theme count target — task-level framing (harmonic mean of attribute-count span)
    if theme_count_hint is not None:
        effective_n, target = theme_count_hint
        theme_target_line = f" This domain has {effective_n} attribute inputs. Aim for approximately {target} themes — deviate only if your analysis clearly justifies it."
    else:
        theme_target_line = ""

    return f"""You are a senior brand insights strategist. Your task is to analyze a brand attribute taxonomy and identify key themes that reveal how the brand is perceived.{theme_target_line}

You will be analyzing attributes within this specific context:

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
- Valence (L4): Whether that attribute is positive, negative, neutral, mixed, or another polarity scheme you choose
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

Here is the attribute inventory you need to analyze:

<attribute_inventory>
{inventory_block}
</attribute_inventory>

Your analysis must:
- Weigh both PREVALENCE (number of ideas per attribute) and VALENCE (positive/negative/neutral sentiment where available)
- Cluster attributes into higher-level themes rather than just summarizing individual attributes

# Required Process

Before generating themes, you MUST work through your analysis step-by-step in a scratchpad. In your scratchpad field:

<required_process>
1. Phenomenon Rule
- Themes must represent underlying PHENOMENA rather than individual attributes. Multiple attributes describing different manifestations of the same underlying phenomenon MUST be merged into a single theme.

2. Prevalence Weighting Rule
The number of ideas linked to each attribute MUST guide code construction.
* Attributes with HIGH idea counts MUST define the core structure of the codebook.
* The codebook MUST be anchored in a small number of dominant phenomena, not a long tail of low-frequency codes.
* Attributes with LOW idea counts MUST NOT become standalone codes unless they represent a clearly distinct phenomenon that cannot be abstracted further.

LOW-prevalence attributes MUST be:
* abstracted into a higher-level phenomenon aligned with dominant patterns, OR
* combined into a broader conceptual category that captures their shared meaning.

3. Mutual Exclusivity Rule
Themes must represent clearly different phenomena so that responses can be coded consistently without ambiguity. If a trained coder would hesitate between two themes when coding a response, they must be merged.

4. Collectively Exhaustivity Rule
Themes must cover all attributes.

5. No Generic Sentiment Themes
Every theme must describe a specific phenomenon. Avoid generic sentiment labels like "positive impression" or "negative feeling." If an attribute captures only diffuse sentiment without a specific subject, absorb it into the most relevant specific theme.

6. Valence Sensitivity Rule
- Generate separate codes for positive and negative phenomena.
- Do NOT combine praise and criticism into a single code.
- If the attributes contain both positive and negative aspects of similar phenomena, create distinct codes for each valence direction.
</required_process>

Output Requirements

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
            "Step-by-step reasoning before deriving themes: "
            "(1) Phenomenon Rule — identify underlying phenomena and merge attributes, "
            "(2) Prevalence Weighting Rule — anchor in high-prevalence phenomena, absorb low-prevalence, "
            "(3) Mutual Exclusivity — if a coder would hesitate between two themes, merge them, "
            "(4) Collective Exhaustivity — ensure all attributes are covered, "
            "(5) No Generic Sentiment — absorb diffuse sentiment into specific themes, "
            "(6) Valence Sensitivity — separate positive and negative phenomena"
        )
    )
    codes: List[CodeFromAttributes] = Field(
        ..., description=(
            "Themes derived from the attribute inventory. "
            "Themes should reflect dominant, high-prevalence phenomena, with low-prevalence attributes absorbed into broader themes where possible.")
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
    code_frequencies: Optional[Dict[int, int]] = None,
) -> str:
    """Consolidate per-domain codes into a final parsimonious, MECE codebook.

    Args:
        raw_codes: All codes from P8 (per-domain)
        code_frequencies: Maps code index to approximate idea count
        dimension_def: DimensionDefinition for dimension-specific diagnostics
    """
    # Dimension-specific diagnostic stem
    if dimension_def:
        code_diagnostic = dimension_def.prompt_rules.code_diagnostic
    else:
        code_diagnostic = "This code is about …"

    # Format candidate codes: one line per code
    code_lines = []
    for i, code in enumerate(raw_codes):
        freq = code_frequencies.get(i, 0) if code_frequencies else 0
        freq_tag = f" (~{freq} ideas)" if freq > 0 else ""
        code_lines.append(f"[C{i+1}] {code.code_name}{freq_tag}: {code.definition}")
    codes_block = "\n".join(code_lines)

    return f"""You are a senior brand insights strategist. Your task is to consolidate {len(raw_codes)} candidate themes into a parsimonious codebook where every code is mutually exclusive and collectively exhaustive.

# Survey Context

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

# Dimension Context

<dimension_context>
Dimension: {dimension_name} — {dimension_description}
</dimension_context>

# Candidate Themes

These themes were generated per domain. Your job is to consolidate across domains — merge overlaps, remove redundancy, and produce a final codebook.

<candidate_themes>
{codes_block}
</candidate_themes>

# Required Process

Before generating your final codebook, you MUST work through your analysis step-by-step in a scratchpad. In your scratchpad field:

<required_process>
1. Valence Rule
- NEVER merge across valence boundaries. Positive and negative versions of the same phenomenon are distinct codes.
- If a candidate theme mixes positive and negative aspects, split it before proceeding.

2. Consolidation Rule
- Merge aggressively within the same valence: if two themes could apply to the same response, they must be merged.
- After merging, check every pair: would a trained coder hesitate between them? If yes, merge further or sharpen definitions until the ambiguity is gone.

3. Purity Rule
- Each code must describe ONE phenomenon and ONE mechanism (value, functional property, perception, or cause — not a mix).
- Each code must be explainable in one sentence without listing unrelated things.
- Code names must not join unrelated concepts with "and" — abstract to a single phenomenon instead.

4. Prevalence Rule
- High-frequency themes define the core structure of the codebook.
- Low-frequency themes must be absorbed into broader codes unless they represent a clearly distinct phenomenon that cannot be merged upward.
- Ask: "Does this code exist because it is conceptually necessary, or just because it appeared rarely?" If the latter, merge or remove.
</required_process>

Each code must include a diagnostic_test that completes: "{code_diagnostic}"
The diagnostic must be unique per code and must not overlap with any other code.

# Output Requirements

Provide output as valid JSON following the response schema provided.

All output MUST be in {language}."""

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
            "Step-by-step reasoning: "
            "(1) Valence Rule — separate by valence, never merge across boundaries, "
            "(2) Consolidation Rule — aggressively merge within valence until no coder would hesitate, "
            "(3) Purity Rule — one phenomenon, one mechanism per code, "
            "(4) Prevalence Rule — high-frequency defines structure, absorb low-frequency upward"
        )
    )
    codes: List[ConsolidatedCode] = Field(
        ..., description="Final MECE codebook"
    )
