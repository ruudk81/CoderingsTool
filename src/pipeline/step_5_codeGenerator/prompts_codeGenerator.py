"""
Prompt builders for Code Generator (P8-P9).

Organized in pipeline processing order:
  §8   Code Generation from Attributes (P8: per domain)
  §9   Codebook Consolidation (P9: cross-domain merge)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional
from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

if TYPE_CHECKING:
    from pipeline.step_3_ideaExtractor.dimension_data import DimensionDefinition

from dataclasses import dataclass, field as dc_field

from pipeline.step_4_classifier.prompts_classifier import DiscoveredAttribute


# =============================================================================
# PROMPT TEMPLATES (text lives in prompt_templates/, separate from code)
# =============================================================================

_TEMPLATE_DIR = Path(__file__).parent / "prompt_templates"


@lru_cache(maxsize=None)
def _load_template(name: str) -> str:
    """Load a prompt template file from prompt_templates/ (cached)."""
    return (_TEMPLATE_DIR / name).read_text(encoding="utf-8")


def _render_template(name: str, **values: str) -> str:
    """Render a template by replacing {{placeholder}} markers with values.

    Plain string replace — tolerant of literal braces and '$' in the prompt text.
    """
    text = _load_template(name)
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", str(value))
    return text.rstrip()


def _valence_tag(
    source_attributes: List[str],
    attribute_valence_counts: Optional[Dict[str, Dict[str, int]]],
    code_frequencies: Optional[Dict[int, int]],
    idx: int,
) -> str:
    """Per-code count tag. '(+p / ○n / −g)' from valence counts, else '(~N ideas)'."""
    if attribute_valence_counts:
        p = n = g = 0
        for attr in (source_attributes or []):
            counts = attribute_valence_counts.get(attr, {})
            p += counts.get("positive", 0)
            n += counts.get("neutral", 0)
            g += counts.get("negative", 0)
        if p or n or g:
            return f" (+{p} / ○{n} / −{g})"
    freq = code_frequencies.get(idx, 0) if code_frequencies else 0
    return f" (~{freq} ideas)" if freq > 0 else ""


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
        theme_count_hint: (low, high) theme count span from UMAP + HDBSCAN clustering, or None
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

    # Theme count target — data-driven span from UMAP + HDBSCAN clustering
    if theme_count_hint is not None:
        low, high = theme_count_hint
        if low == high:
            theme_target_line = f" Aim for approximately {low} themes — deviate only if your analysis clearly justifies it."
        else:
            theme_target_line = f" Aim for between {low} and {high} themes — deviate only if your analysis clearly justifies it."
        theme_range = f"{low}–{high}"
    else:
        theme_target_line = ""
        theme_range = ""

    return _render_template(
        "p8_code_generation.md",
        theme_target_line=theme_target_line,
        survey_question=survey_question,
        language=language,
        dataset_context_section=dataset_context_section,
        dimension_name=dimension_name,
        noun_phrase=noun_phrase,
        domain_key_idea=domain_key_idea,
        attribute_key_idea=attribute_key_idea,
        dimension_description=dimension_description,
        domain_name=domain_name,
        domain_definition=domain_definition,
        inventory_block=inventory_block,
        theme_range=theme_range,
    )


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
    attribute_valence_counts: Optional[Dict[str, Dict[str, int]]] = None,
) -> str:
    """Consolidate per-domain codes into a final parsimonious, MECE codebook.

    Args:
        raw_codes: All codes from P8 (per-domain)
        code_frequencies: Maps code index to approximate idea count
        attribute_valence_counts: attribute_name -> {positive, neutral, negative} idea counts,
            for the prevalence-gated valence policy (per-code valence tag in the prompt)
        dimension_def: DimensionDefinition for dimension-specific diagnostics
    """
    # Dimension-specific diagnostic stem
    if dimension_def:
        code_diagnostic = dimension_def.prompt_rules.code_diagnostic
    else:
        code_diagnostic = "This code is about …"

    # Format candidate codes with per-valence idea counts
    code_lines = []
    for i, code in enumerate(raw_codes):
        tag = _valence_tag(code.source_attributes, attribute_valence_counts, code_frequencies, i)
        attrs = ", ".join(code.source_attributes[:5]) if code.source_attributes else "—"
        indicators = "; ".join(code.typical_indicators[:3]) if code.typical_indicators else "—"
        code_lines.append(
            f"[C{i+1}] {code.code_name}{tag}\n"
            f"      Definition: {code.definition}\n"
            f"      Indicators: {indicators}\n"
            f"      Source attributes: {attrs}"
        )
    codes_block = "\n\n".join(code_lines)

    return _render_template(
        "p9_consolidation.md",
        n_raw_codes=len(raw_codes),
        survey_question=survey_question,
        language=language,
        dataset_context_section=dataset_context_section,
        dimension_name=dimension_name,
        dimension_description=dimension_description,
        codes_block=codes_block,
        code_diagnostic=code_diagnostic,
    )

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
    # Stable ids (identity.py) — never part of the LLM response schema: minted at
    # cache-save (K#), or lazily at load for pre-id codebooks. source_attribute_ids
    # mirrors source_attributes as attribute ids (A#).
    code_id: SkipJsonSchema[str] = ""
    source_attribute_ids: SkipJsonSchema[List[str]] = Field(default_factory=list)


class CodebookConsolidationResult(BaseModel):
    """P9 output: consolidated codebook."""
    scratchpad: str = Field(
        ..., description=(
            "Step-by-step reasoning following the 8-step workflow: "
            "(1) valence separation, "
            "(2) aggressive merging within clusters, "
            "(3) mechanism purity check, "
            "(4) neighbor stress test, "
            "(5) one-sentence coverage test, "
            "(6) non-redundancy kill step, "
            "(7) final diagnostic uniqueness check, "
            "(8) prevalence weighting and structural balancing"
        )
    )
    codes: List[ConsolidatedCode] = Field(
        ..., description="Final MECE codebook"
    )
