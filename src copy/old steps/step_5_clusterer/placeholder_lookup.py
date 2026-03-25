"""
Placeholder value construction for CLUSTER_DESCRIPTION_PROMPT.

Centralizes all 14 placeholder values into a two-phase pipeline:
  1. build_dataset_placeholders() — once per dataset (from ExtractionMetadata)
  2. build_cluster_placeholders() — once per cluster (incorporates dataset-level values)

Design: frozen dataclasses, no .get() chains, no getattr() with defaults.
Follows the facet_data.py pattern from step_3.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ========================================================================
# Language mapping
# ========================================================================

LANG_DISPLAY_NAMES: Dict[str, str] = {
    "nl-NL": "Dutch",
    "en-GB": "English",
    "en-US": "English",
    "de-DE": "German",
    "fr-FR": "French",
    "es-ES": "Spanish",
}


def resolve_display_language(lang_code: str) -> str:
    """Map a BCP-47 lang code to a display name for prompts.

    Falls back to "Dutch" if lang_code is empty (development convenience
    for cached data that predates the lang field).
    """
    if not lang_code:
        return "Dutch"
    if lang_code in LANG_DISPLAY_NAMES:
        return LANG_DISPLAY_NAMES[lang_code]
    # Try prefix match (e.g., "nl" -> "nl-NL")
    prefix = lang_code.split("-")[0].lower()
    for code, name in LANG_DISPLAY_NAMES.items():
        if code.split("-")[0].lower() == prefix:
            return name
    raise KeyError(
        f"Unknown language code: {lang_code!r}. "
        f"Known codes: {sorted(LANG_DISPLAY_NAMES)}"
    )


# ========================================================================
# Data structures
# ========================================================================

@dataclass(frozen=True)
class FacetInfo:
    """Resolved primary facet fields from ExtractionMetadata."""
    name: str           # ExtractionMetadata.primary_facet (e.g., "EVALUATION_PRIORITIZATION")
    description: str    # ExtractionMetadata.primary_facet_description


@dataclass(frozen=True)
class DatasetPlaceholders:
    """Dataset-level placeholder values — computed once per labeling run."""
    survey_question: str
    language: str                   # Display name, e.g., "Dutch"
    dataset_context_section: str
    facet_context: str              # <primary_facet> XML block, or ""
    facet_task_guidance: str        # Parenthetical suffix, or ""
    facet_output_constraint: str    # "within the X facet", or ""
    concept_types_section: str      # Data-driven concept types from step 3
    samples_tag: str                # e.g., "response_ideas"
    sample_type: str                # e.g., "response ideas"


# ========================================================================
# Private builders — dataset-level
# ========================================================================

def _build_dataset_context_section(
    domain: str, entity: str, topic: str,
    perspective: str, intent: str,
) -> str:
    """Build the dataset context section from individual metadata fields."""
    parts = []
    if domain:
        parts.append(f"Domain: {domain}")
    if entity:
        parts.append(f"Entity: {entity}")
    if topic:
        parts.append(f"Topic: {topic}")
    if perspective:
        parts.append(f"Perspective: {perspective}")
    if intent:
        parts.append(f"Intent: {intent}")
    if not parts:
        return ""
    return "\n" + "\n".join(parts)


def _build_facet_context(info: FacetInfo) -> str:
    """Build the <primary_facet> XML block."""
    return f"""
<primary_facet>
Primary coding facet: {info.name}
Definition: {info.description or 'Not specified'}
Labels MUST describe content within this facet ONLY.
Do NOT include sentiment, evaluation, tone, or respondent intent in the label.
</primary_facet>
"""


def _build_facet_task_guidance(info: FacetInfo) -> str:
    """Build the parenthetical suffix for the task instruction."""
    return f" ({info.name})"


def _build_facet_output_constraint(info: FacetInfo) -> str:
    """Build the output constraint suffix."""
    return f" within the {info.name} facet"


def _build_concept_types_section(concept_types: List[Dict[str, str]]) -> str:
    """Build the <concept_types> section from step 3's data-driven types."""
    if not concept_types:
        return ""
    lines = []
    for ct in concept_types:
        key = ct.get("key", "")
        definition = ct.get("definition", ct.get("label", ""))
        if key and definition:
            lines.append(f"- {key}: {definition}")
        elif key:
            lines.append(f"- {key}")
    if not lines:
        return ""
    return f"""
<concept_types>
The data contains these concept types:
{chr(10).join(lines)}
</concept_types>
"""


# ========================================================================
# Private builders — cluster-level
# ========================================================================

def _format_ideas_list(sample_ideas: Tuple[str, ...]) -> str:
    """Build a numbered list of representative ideas."""
    return "\n".join(f"{i+1}. {idea}" for i, idea in enumerate(sample_ideas))


def _build_keywords_section(
    keywords: Tuple[Tuple[str, float], ...],
    sample_type: str,
) -> str:
    """Build the <statistical_keywords> XML block."""
    if not keywords:
        return ""
    kw_formatted = "\n".join(
        f"{i+1}. {kw}" for i, (kw, _score) in enumerate(keywords[:10])
    )
    return f"""
<statistical_keywords>
These terms statistically differentiate this cluster from others (c-TF-IDF).
Use to refine - but not override - the representative {sample_type}:
{kw_formatted}
</statistical_keywords>
"""


def _build_cluster_profile_section(
    distributions: Optional[Dict[str, Dict[str, float]]],
) -> str:
    """Build cluster profile section. Currently returns ''."""
    return ""


# ========================================================================
# Public API
# ========================================================================

def build_dataset_placeholders(metadata) -> DatasetPlaceholders:
    """Build all dataset-level placeholder values from ExtractionMetadata.

    Args:
        metadata: ExtractionMetadata instance (must not be None).

    Returns:
        Frozen DatasetPlaceholders with all 9 dataset-level values.
    """
    language = resolve_display_language(metadata.lang)

    # Build facet info (replaces legacy "taxonomy" concept)
    facet_info = None
    if metadata.primary_facet:
        facet_info = FacetInfo(
            name=metadata.primary_facet,
            description=metadata.primary_facet_description,
        )

    return DatasetPlaceholders(
        survey_question=metadata.var_lab,
        language=language,
        dataset_context_section=_build_dataset_context_section(
            domain=metadata.domain,
            entity=metadata.entity,
            topic=metadata.topic,
            perspective=metadata.perspective,
            intent=metadata.intent,
        ),
        facet_context=_build_facet_context(facet_info) if facet_info else "",
        facet_task_guidance=_build_facet_task_guidance(facet_info) if facet_info else "",
        facet_output_constraint=_build_facet_output_constraint(facet_info) if facet_info else "",
        concept_types_section=_build_concept_types_section(metadata.concept_types),
        samples_tag="response_ideas",
        sample_type="response ideas",
    )


def build_cluster_placeholders(
    dataset: DatasetPlaceholders,
    *,
    cluster_id: int,
    num_ideas: int,
    sample_ideas: Tuple[str, ...],
    keywords: Tuple[Tuple[str, float], ...] = (),
    distributions: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, str]:
    """Build the complete 14-placeholder dict for one cluster.

    Returns a dict ready for CLUSTER_DESCRIPTION_PROMPT.format(**result).
    """
    return {
        # Dataset-level (9)
        "survey_question": dataset.survey_question,
        "language": dataset.language,
        "dataset_context_section": dataset.dataset_context_section,
        "facet_context": dataset.facet_context,
        "facet_task_guidance": dataset.facet_task_guidance,
        "facet_output_constraint": dataset.facet_output_constraint,
        "concept_types_section": dataset.concept_types_section,
        "samples_tag": dataset.samples_tag,
        "sample_type": dataset.sample_type,
        # Cluster-level (5)
        "cluster_id": str(cluster_id),
        "num_ideas": str(num_ideas),
        "ideas_list": _format_ideas_list(sample_ideas),
        "keywords_section": _build_keywords_section(keywords, dataset.sample_type),
        "cluster_profile_section": _build_cluster_profile_section(distributions),
    }
