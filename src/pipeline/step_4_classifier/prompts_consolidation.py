"""Consolidation prompts for step 4."""

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

from pydantic import BaseModel, Field

from pipeline.step_4_classifier.prompts_discovery import DiscoveredAttribute
from pipeline.step_4_classifier.prompts_shared import (
    INSTRUCTOR_HINT, UNIVERSAL_RULES, build_context_block,
    build_facets_attributes_block, build_taxonomy_block_L3)

if TYPE_CHECKING:
    from pipeline.step_3_ideaExtractor.dimension_data import DimensionDefinition

_EXAMPLES_SHOWN = 3 #Examples shown per candidate attribute, in the attribute call

# =============================================================================
# PROMPT — FACET CONSOLIDATION
# =============================================================================

_FACET_SELF_REFERENCE = re.compile(r"\b(?:(?:deze|dit)\s+)?facet\b", re.IGNORECASE)

def _strip_facet_self_reference(text: str) -> str:
      """Haalt `deze facet` / `facet` weg, in welke casing dan ook."""
      return re.sub(r"\s{2,}", " ", _FACET_SELF_REFERENCE.sub("", text)).strip()
  
@dataclass
class FacetPool:
    facet_name: str
    facet_definition: str
    facet_question: str
    attributes: List[DiscoveredAttribute]
    # Written by facet consolidation, read in the facet-assignment menu. It has
    # a default because the round-one candidates come from discovery, which sees
    # one chunk and therefore has no sibling to draw a boundary against.
    boundary_rules: List[str] = field(default_factory=list)

def build_facet_candidate_index(pools: List[FacetPool]) -> Dict[str, FacetPool]:
    """`F1`, `F2`, … for the candidates of one facet-consolidation call."""
    return {f"F{i}": pool for i, pool in enumerate(pools, 1)}

class SettledFacet(BaseModel):
    facet_name: str = Field(
        ..., description=(
            "Short descriptive name for the facet, in the survey language "
            "(at most 5 words)"))
    facet_definition: str = Field(
        ..., description=(
            "What this facet captures — one clear underlying concept, in 1-2 "
            "sentences, in the survey language"))
    facet_question: str = Field(
        ..., description=(
            "The one question this facet answers about the responses, phrased "
            "as a question, in the survey language. No two surviving facets "
            "may state the same one"))
    source_facet_ids: List[str] = Field(
        ..., description=(
            "The bracketed ids of every candidate facet that folded into this "
            "one, e.g. ['F1', 'F7']. One that survived unchanged lists just "
            "its own id"))
    boundary_rules: List[str] = Field(
        ..., description=(
            "Decision rules that distinguish this facet from other surviving facets, "
            "in the survey language. One line per genuine boundary. "
            "Each line must name one other facet in this same answer and state the "
            "specific distinction that determines which facet ambiguous material belongs to. "
            "Do not restate the facet definition. "
            "Only include a boundary where material could plausibly fit both facets; "
            "if no such ambiguity exists, return an empty list." ))
    
class FacetConsolidationResult(BaseModel):
    decision_summary: List[str] = Field(
        ..., description=(
            "One short line per consolidation decision that took judgement, "
            "each stating what was done and why. Not a reasoning trace, and "
            "not a line for every candidate: only the calls a reader would "
            "want to check"))
    facets: List[SettledFacet] = Field(
        ..., description=(
            "The fewest mutually exclusive facets that cover the domain"))

def build_facet_candidate_block(
    pools: List[FacetPool],
    recurrence: Dict[str, int],
    n_passes: int,
) -> str:
    blocks = []
    for facet_id, pool in build_facet_candidate_index(pools).items():
        seen = recurrence.get(pool.facet_name, 1)
        lines = [f"[{facet_id}] {_strip_facet_self_reference(pool.facet_definition)}"]
        if pool.facet_question:
             lines.append(
                  f"    Question it answers: "
                  f"{_strip_facet_self_reference(pool.facet_question)}")
        names = [a.attribute_name for a in pool.attributes]
        lines.append(
            "    Attributes: "
            + (", ".join(names) if names else "(none)"))
        # Rule 4 lets prevalence set the granularity, so the number has to be
        # in front of the model. It counts how often this candidate's name was
        # proposed across the passes — but the name itself is deliberately not
        # rendered, so the line says nothing about a name the model cannot see.
        lines.append(f"    Proposed in {seen} of {n_passes} independent passes")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def build_facet_consolidation_prompt(
    *,
    language: str,
    survey_question: str,
    sector: str,
    entity: str,
    topic: str,
    perspective: str,
    intent: str,
    dimension: "DimensionDefinition",
    dimension_name: str,
    dimension_description: str,
    domain_label: str,
    domain_definition: str,
    domain_exclusions: Optional[List[str]],
    candidate_block: str,
) -> str:
    exclusion_hint = (
        "\n".join(f"- {x}" for x in domain_exclusions)
        if domain_exclusions else "- (no neighbouring domains were named)")

    return f"""You are a taxonomy consolidation specialist for surveys.
Your task is to organize the candidate attributes within one facet into the smallest possible set of meaningful attribute-containers that is MECE.
Default toward consolidation. A distinction should survive only when keeping it separate is necessary to preserve meaningful semantic differences in the context of the survey question.

# Survey context  

{build_context_block(
    language=language,
    dimension_name=dimension_name, dimension_description=dimension_description,
    survey_question=survey_question,
    sector=sector, entity=entity, topic=topic, perspective=perspective, intent=intent)}
    
# Taxonomy structure 

{build_taxonomy_block_L3(
    dimension=dimension, dimension_name=dimension_name,
    dimension_description=dimension_description)}

You are working within this domain:
<taxonomy_domain>
{domain_label} — {domain_definition}
</taxonomy_domain>

Here are the groups with attributes you need to organize into a minimal set of facets:
<attribute_groups>
{candidate_block}
</attribute_groups>

# Objective

Find the minimum number of attribute-containers required to organize all substantive candidate material belonging to this facet while remaining mutually exclusive and collectively exhaustive (MECE).

The optimization priority is:
- Correct facet membership
- MECE
- Minimum number of containers
- Interpretability
- Preservation of meaningful prevalent distinctions

Do not preserve a distinction merely because it appears in the input.

Rules
1. Minimize the number of containers. Merge candidate attributes whenever they can be represented by one broader, meaningful attribute without losing an important distinction for the survey question. When in doubt, prefer merging.
2. Keep a distinction only when it is substantively meaningful and clearly codable. Differences in wording, synonyms, closely related meanings, or broad-versus-narrow versions of the same idea normally belong in the same container.
3. The final attributes must be MECE. Each substantive idea should have one natural home, and together the attributes must cover all substantive material belonging to this facet. Avoid overlapping attributes and parent/child attributes alongside each other.
4. Use prevalence to simplify. Small or low-prevalence distinctions should normally be absorbed into the nearest broader attribute rather than becoming separate attributes, provided the resulting container remains semantically coherent.
5. Account for every candidate on its id. Every [F#] shown must appear in `source_facet_ids` of at least one surviving facet. Coverage is checked on the ids and never on names: two candidates of one domain may carry the same name. One you leave out is not removed — it stays where it was, next to the facet you meant to replace it with, so never drop a candidate.

Before returning the result, ask one final question:
"Can any two remaining attributes still be merged without losing an important, clearly codable distinction?"
If yes, merge them.

{UNIVERSAL_RULES}

{INSTRUCTOR_HINT}"""


# =============================================================================
# PROMPT — ATTRIBUTE CONSOLIDATION
# =============================================================================


class SettledAttribute(DiscoveredAttribute):
    example_observations: List[str] = Field(
        ..., description=(
            "1-3 observations carried over from the candidates that folded "
            "into this attribute, copied exactly as shown. Give what is there: "
            "an attribute that carries one example gives one. NEVER merge "
            "attributes that mean different things in order to reach a higher "
            "count — the count follows the taxonomy, never the other way round"))
    source_attribute_ids: List[str] = Field(
        ..., description=(
            "The bracketed ids of every candidate attribute that folded into "
            "this one, e.g. ['A2', 'A7']. One that survived unchanged lists "
            "just its own id"))
    boundary_rules: List[str] = Field(
        ..., description=(
            "Decision rules that distinguish this attribute from other surviving attributes, "
            "in the survey language. One line per genuine boundary. "
            "Each line must name one other attribute in this same answer and state the "
            "specific distinction that determines which attribute ambiguous material belongs to. "
            "Do not restate the attribute definition. "
            "Only include a boundary where material could plausibly fit both attributes; "
            "if no such ambiguity exists, return an empty list." ))

class AttributeConsolidationResult(BaseModel):
    decision_summary: List[str] = Field(
        ..., description=(
            "One short line per consolidation decision that took judgement, "
            "each stating what was done and why. Only the calls a reader would "
            "want to check"))
    attributes: List[SettledAttribute] = Field(
        ..., description=(
            "The fewest mutually exclusive attributes that cover what this "
            "facet holds"))


def build_attribute_candidate_index(
    attributes: List[DiscoveredAttribute],
) -> Dict[str, DiscoveredAttribute]:
    """`A1`, `A2`, … for the pool of one facet. """
    return {f"A{i}": attribute for i, attribute in enumerate(attributes, 1)}

def build_attribute_candidate_block(
    attributes: List[DiscoveredAttribute],
    recurrence: Dict[str, int],
    n_passes: int,
) -> str:
    """The pooled attributes of one facet, each with its reach and examples."""
    lines = []
    for attribute_id, attribute in build_attribute_candidate_index(attributes).items():
        times = recurrence.get(attribute.attribute_name, 1)
        lines.append(
            f"[{attribute_id}] {attribute.attribute_name} "
            f"[{times}/{n_passes} passes]: {attribute.attribute_definition}")
        for example in [e for e in attribute.example_observations
                        if e][:_EXAMPLES_SHOWN]:
            lines.append(f"    e.g. \"{example}\"")
    return "\n".join(lines)


def build_attribute_consolidation_prompt(
    *,
    language: str,
    survey_question: str,
    sector: str,
    entity: str,
    topic: str,
    perspective: str,
    intent: str,
    dimension: "DimensionDefinition",
    dimension_name: str,
    dimension_description: str,
    facet_name: str,
    facet_definition: str,
    facet_question: str,
    candidate_block: str,
) -> str:
    question_line = (f"\nThe question this facet answers: {facet_question}"
                     if facet_question else "")
    return f"""You are a taxonomy consolidation specialist for surveys.
Your task is to organize the attributes into the smallest possible set of meaningful attribute-containers within a given facet that is MECE.
Default toward consolidation. A distinction should survive only when keeping it separate is necessary to preserve meaningful semantic differences in the context of the survey question.

# Survey context 

{build_context_block(
    language=language,
    dimension_name=dimension_name, dimension_description=dimension_description,
    survey_question=survey_question,
    sector=sector, entity=entity, topic=topic, perspective=perspective, intent=intent)}
    
# Taxonomy structure 

{build_taxonomy_block_L3(
    dimension=dimension, dimension_name=dimension_name,
    dimension_description=dimension_description)}
    
You are working inside this facet:

<taxonomy_facet>
Facet: {facet_name} — {facet_definition}{question_line}
</taxonomy_facet>

Here are the attributes you need to organize into a minimal set of meaningful containers:
<candidates>
{candidate_block}
</candidates>

# Objective

Find the minimum number of attribute-containers required to organize all substantive candidate material belonging to this facet while remaining mutually exclusive and collectively exhaustive (MECE).

The optimization priority is:
- Correct facet membership
- MECE
- Minimum number of containers
- Interpretability
- Preservation of meaningful prevalent distinctions

Do not preserve a distinction merely because it appears in the input.

Rules
1. Minimize the number of containers. Merge candidate attributes whenever they can be represented by one broader, meaningful attribute without losing an important distinction for the survey question. When in doubt, prefer merging.
2. Keep a distinction only when it is substantively meaningful and clearly codable. Differences in wording, synonyms, closely related meanings, or broad-versus-narrow versions of the same idea normally belong in the same container.
3. The final attributes must be MECE. Each substantive idea should have one natural home, and together the attributes must cover all substantive material belonging to this facet. Avoid overlapping attributes and parent/child attributes alongside each other.
4. Use prevalence to simplify. Small or low-prevalence distinctions should normally be absorbed into the nearest broader attribute rather than becoming separate attributes, provided the resulting container remains semantically coherent.
5. Account for every candidate on its id. Every [A#] shown must appear in `source_attribute_ids` of at least one surviving attribute. Coverage is checked on the ids and never on names. Merging is allowed here; losing is not — one facet in view cannot judge where something else belongs, so never drop a candidate.

Before returning the result, ask one final question:
"Can any two remaining attributes still be merged without losing an important, clearly codable distinction?"
If yes, merge them.

{UNIVERSAL_RULES}

{INSTRUCTOR_HINT}"""