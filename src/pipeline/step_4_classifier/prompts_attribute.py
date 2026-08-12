"""Prompt builders and response models for the attribute layer (L4) of step 4.

The mirror of prompts_facet.py one level down, with the same four phases:

  1. discovery      build_attribute_discovery_prompt      per (facet, chunk)
  2. consolidation  build_attribute_consolidation_prompt  per facet, over chunks
  3. assignment     build_attribute_assignment_prompt     per batch of ideas
  4. refinement     build_attribute_refinement_prompt     per facet, after assignment

The facet is fixed throughout: no phase here can move an attribute or an idea to
another facet. Where the facet layer's parent is the domain, this layer's parent
is the facet — everything else about the shape is the same, deliberately.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List

from pydantic import BaseModel, Field

from .prompts_shared import (
    INSTRUCTOR_HINT,
    UNIVERSAL_RULES,
    build_context_block,
    build_taxonomy_block,
    level_diagnostic,
)

if TYPE_CHECKING:
    from pipeline.step_3_ideaExtractor.dimension_data import DimensionDefinition


# =============================================================================
# §1 DISCOVERY — per (facet, chunk)
# =============================================================================

class DiscoveredAttribute(BaseModel):
    """One attribute (L4) proposed from a chunk of observations within a facet.

    Same four boundary fields as DiscoveredFacet, one level down. There is no
    `parent_facet` field: the facet is the scope of the call, not a property of
    the item. A field the model can write is a field it can write wrongly.
    """
    attribute_name: str = Field(
        ..., description="Short descriptive name for the attribute (2-5 words)"
    )
    attribute_definition: str = Field(
        ..., description=(
            "One-sentence inclusion definition naming a single observable property. "
            "No examples, no enumerations"
        )
    )
    boundary_test: str = Field(
        ..., description=(
            "A single yes/no question a coder asks to decide whether an idea "
            "belongs to THIS attribute rather than a neighbouring one"
        )
    )
    exclusions: List[str] = Field(
        ..., description=(
            "1-3 short phrases naming what does NOT belong here — especially "
            "the neighbouring attribute it is most easily confused with"
        )
    )
    example_observations: List[str] = Field(
        ..., description="2-3 observations from the input that exemplify this attribute"
    )


class AttributeDiscoveryResult(BaseModel):
    """Discovery output for one chunk."""
    scratchpad: str = Field(
        ..., description=(
            "Reasoning before the final attribute set: "
            "(1) cluster the observations by shared descriptive meaning, "
            "(2) name the candidate attributes, "
            "(3) check each candidate is one concept and mixes no evaluation in, "
            "(4) check each pair for a clean boundary and merge where there is none, "
            "(5) check every candidate stays inside the facet"
        )
    )
    attributes: List[DiscoveredAttribute] = Field(
        ..., description="The attributes found in these observations"
    )


def build_attribute_discovery_prompt(
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
    facet_name: str,
    facet_definition: str,
    facet_boundary_test: str,
    facet_exclusions: List[str],
    observations: List[str],
) -> str:
    """Propose attributes (L4) from one chunk of observations within one facet."""
    context_block = build_context_block(
        language=language, survey_question=survey_question, sector=sector,
        entity=entity, topic=topic, perspective=perspective, intent=intent,
    )
    taxonomy_block = build_taxonomy_block(
        dimension=dimension, dimension_name=dimension_name,
        dimension_description=dimension_description,
    )
    diagnostic = level_diagnostic(dimension, "attribute")
    observations_block = "\n".join(
        f"{i}. {obs}" for i, obs in enumerate(observations, 1)
    )
    exclusions_line = "; ".join(facet_exclusions) if facet_exclusions else "(none given)"

    return f"""You are a qualitative research analyst specializing in survey response analysis.
Your task is to identify the fewest recurring attributes that fully cover a set of observations within one facet.

{context_block}

{taxonomy_block}

You are working inside ONE facet, within ONE domain. Everything you return belongs to
that facet, and nothing that falls outside it may be returned.

<parents>
Domain: {domain_label} — {domain_definition}
Facet:  {facet_name} — {facet_definition}
Facet boundary test: {facet_boundary_test}
Does NOT belong to this facet (these have their own facets): {exclusions_line}
</parents>

Here are the observations you need to account for:

<observations>
{observations_block}
</observations>

## YOUR TASK

Identify the **attributes** (level 4) present in these observations.

The question every attribute must answer for this dimension is:

<attribute_diagnostic>
{diagnostic}
</attribute_diagnostic>

An attribute is an answer to that question. It names a concrete, observable property —
not a verbatim span from a response, and not the facet restated one level up.

Aim for the smallest set of attributes that still gives every observation a clear home.
Each attribute must be:

- **Grounded in the data** — supported by a recurring pattern across observations, not
  by a single one-off phrasing.
- **Internally coherent** — one clear underlying concept. Reject or split candidates
  that combine different kinds of phenomena, or that mix descriptive content with
  evaluation.
- **Ontologically distinct** — no overlap, no subset or superset, and never two lenses
  on the same phenomenon.
- **Semantically separable** — a coder must clearly know which attribute applies. No
  "could go either way" situations.
- **Inside the facet** — strictly within the boundary stated above. A candidate that
  belongs more naturally to a neighbouring facet is left out, not stretched to fit.

Rare or one-off observations do not each earn an attribute. Group them under the
attribute whose definition honestly covers them, and only give them their own attribute
when the same distinction recurs.

{UNIVERSAL_RULES}

## OUTPUT

Work through your reasoning in the scratchpad field first.

For EACH attribute provide:
- **attribute_name** — a short descriptive name
- **attribute_definition** — one sentence naming a single observable property, with no
  examples or enumerations
- **boundary_test** — one yes/no question that decides membership
- **exclusions** — what does NOT belong, naming the neighbouring attribute it is most
  easily confused with
- **example_observations** — 2-3 observations from the input above, copied exactly

All attribute names, definitions, boundary tests and exclusions must be written in {language}.

Begin processing now and {INSTRUCTOR_HINT}"""
