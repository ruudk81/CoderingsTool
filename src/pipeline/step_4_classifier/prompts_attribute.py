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

from typing import TYPE_CHECKING, Dict, List, Literal, Tuple

from pydantic import BaseModel, Field, create_model, model_validator

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


# =============================================================================
# §2 CONSOLIDATION — per facet, across chunks, before any idea is assigned
# =============================================================================

class ConsolidatedAttribute(DiscoveredAttribute):
    """One attribute surviving consolidation, with the candidates it absorbed."""
    source_attributes: List[str] = Field(
        ..., description=(
            "The attribute_name of every candidate that goes into this one. "
            "A candidate that is kept unchanged lists its own name"
        )
    )


class AttributeConsolidationResult(BaseModel):
    """The settled attribute inventory for one facet."""
    scratchpad: str = Field(
        ..., description=(
            "Consolidation reasoning: "
            "(1) list the unique candidates across all chunks, "
            "(2) group the ones that overlap conceptually, "
            "(3) name and define each consolidated attribute, "
            "(4) for every surviving pair ask whether one response could belong "
            "to both, and merge when it could, "
            "(5) verify the survivors still cover everything the candidates covered, "
            "(6) write each survivor's boundary_test and exclusions"
        )
    )
    attributes: List[ConsolidatedAttribute] = Field(
        ..., description="The complete attribute set for this facet after consolidation"
    )


def _build_candidate_block(candidates: List[DiscoveredAttribute]) -> str:
    """Render the chunk yield as numbered candidates, each with its evidence."""
    blocks = []
    for i, candidate in enumerate(candidates, 1):
        exclusions = "; ".join(candidate.exclusions) if candidate.exclusions else "(none)"
        observations = "; ".join(candidate.example_observations)
        blocks.append(
            f"[C{i}] {candidate.attribute_name}\n"
            f"     Definition: {candidate.attribute_definition}\n"
            f"     Boundary test: {candidate.boundary_test}\n"
            f"     Does not belong: {exclusions}\n"
            f"     Observations that produced this proposal: {observations}"
        )
    return "\n\n".join(blocks)


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
    domain_label: str,
    domain_definition: str,
    facet_name: str,
    facet_definition: str,
    candidates: List[DiscoveredAttribute],
) -> str:
    """Settle one facet's attribute inventory, across all chunk proposals."""
    context_block = build_context_block(
        language=language, survey_question=survey_question, sector=sector,
        entity=entity, topic=topic, perspective=perspective, intent=intent,
    )
    taxonomy_block = build_taxonomy_block(
        dimension=dimension, dimension_name=dimension_name,
        dimension_description=dimension_description,
    )
    diagnostic = level_diagnostic(dimension, "attribute")
    candidate_block = _build_candidate_block(candidates)

    return f"""You are a taxonomy consolidation specialist for survey coding.
Your task is to merge attribute proposals from several independent passes over one facet
into a single, coherent set of attributes.

{context_block}

{taxonomy_block}

You are working inside ONE facet. Every attribute you return belongs to it.

<parents>
Domain: {domain_label} — {domain_definition}
Facet:  {facet_name} — {facet_definition}
</parents>

The question every attribute must answer for this dimension is:

<attribute_diagnostic>
{diagnostic}
</attribute_diagnostic>

Here are the candidate attributes. Each pass saw a different sample of the responses and
did not see the other passes, so the same attribute may appear several times under
different names. Each candidate carries the observations that produced it — those are
your evidence:

<candidates>
{candidate_block}
</candidates>

## YOUR TASK

Consolidate these candidates into the fewest mutually exclusive attributes needed for
full coverage.

Judge the candidates on their observations, not on their labels. Two labels that read
differently but were produced by the same kind of observation are ONE attribute. Two
labels that read alike but were produced by different observations are TWO.

Consolidation principles:

- **MERGE** candidates that overlap conceptually, are near-equivalent, or where one is
  a subset of the other.
- **MERGE** candidates that are two lenses on the same phenomenon — different wording
  for one underlying property.
- **THE BOUNDARY TEST DECIDES.** For each pair of survivors, write the boundary that
  separates them. If you cannot state a clean boundary between an attribute and its
  nearest neighbour, they are not two attributes — merge them.
- **ENSURE ontological distinctness** — no two attributes may share conceptual space,
  and none may be a subset of another.
- **ENSURE semantic separability** — a coder must not plausibly hesitate between two
  attributes.
- **MAINTAIN full coverage** — the survivors must collectively cover everything the
  candidates covered. Consolidating is not discarding.
- **MINIMIZE the count** while preserving distinctions the observations actually show.
  If the observations hold four distinct answers to the attribute question, return four
  attributes — do not collapse them because fewer is tidier.
- **STAY inside the facet.** A candidate that falls outside the facet is not an
  attribute to keep; leave it out rather than widening the facet to fit it.

Every candidate you consume must be listed in the `source_attributes` of the attribute
that consumes it. A candidate you do not list is left standing as it is, so list them.

{UNIVERSAL_RULES}

## OUTPUT

Work through the consolidation in the scratchpad field first.

For EACH consolidated attribute provide:
- **attribute_name** — a short descriptive name
- **attribute_definition** — one sentence naming a single observable property, no
  examples or enumerations
- **boundary_test** — one yes/no question that decides membership
- **exclusions** — what does NOT belong, naming the neighbouring attribute it is most
  easily confused with
- **example_observations** — 2-3 observations, copied exactly from the candidates above
- **source_attributes** — the attribute_name of every candidate consumed into this one

All attribute names, definitions, boundary tests and exclusions must be written in {language}.

Begin processing now and {INSTRUCTOR_HINT}"""


# =============================================================================
# §3 ASSIGNMENT — ideas into the settled inventory
# =============================================================================
#
# Like the facet layer's assignment, this phase carries no taxonomy block and no
# universal rules: it creates nothing, it picks an id from a menu whose entries
# already carry their definitions and boundaries, and it runs at the volume of
# the dataset.


def build_attribute_menu(attributes: List[ConsolidatedAttribute]) -> str:
    """Render the settled attributes as a numbered menu.

    The [A#] id is what the response is keyed on, so the numbering here and the
    id list handed to `build_attribute_assignment_model` must come from the same
    list in the same order.
    """
    lines = []
    for i, a in enumerate(attributes, 1):
        exclusions = "; ".join(a.exclusions) if a.exclusions else ""
        examples = "; ".join(a.example_observations[:3])
        block = (
            f"[A{i}] {a.attribute_name}\n"
            f"     Description: {a.attribute_definition}\n"
            f"     Boundary: {a.boundary_test}"
        )
        if exclusions:
            block += f"\n     Does not belong here: {exclusions}"
        if examples:
            block += f"\n     Examples: {examples}"
        lines.append(block)
    return "\n\n".join(lines)


def build_attribute_assignment_model(attribute_ids: List[str], idea_ids: List[str]):
    """Runtime response model for one assignment call.

    The attribute layer had no Literal on the assigned id until this rewrite —
    it was the weakest link in the chain, where the facet layer had one and step
    3's domain assignment had an enum. Both id spaces are Literals here, so a
    hallucinated id is a schema violation instructor retries.
    """
    attribute_id_literal = Literal[tuple(attribute_ids + ["A_NONE"])]  # type: ignore[valid-type]
    idea_id_literal = Literal[tuple(idea_ids)]  # type: ignore[valid-type]

    item_model = create_model(
        "AttributeAssignmentItem",
        idea_id=(idea_id_literal, Field(
            ..., description="The [id] tag of the idea, echoed exactly")),
        assigned_attribute_id=(attribute_id_literal, Field(
            ..., description=(
                "The attribute id from the [A#] prefix. Return ONLY the id. "
                "Use A_NONE when no attribute fits this idea"))),
        confidence=(float, Field(
            ..., ge=0.0, le=1.0, description="Assignment confidence (0.0-1.0)")),
        valence=(Literal["+", "-", "0"], Field(
            default="0",
            description=(
                "Evaluative direction relative to the attribute: "
                "+ positive, - negative, 0 neutral"))),
    )
    return create_model(
        "AttributeAssignmentResult",
        assignments=(List[item_model], Field(
            ..., description=(
                "Exactly one assignment per idea listed in the prompt, "
                "no idea skipped, no idea added"))),
    )


def build_attribute_assignment_prompt(
    *,
    language: str,
    survey_question: str,
    sector: str,
    entity: str,
    topic: str,
    perspective: str,
    intent: str,
    facet_name: str,
    facet_definition: str,
    attributes: List[ConsolidatedAttribute],
    ideas: List[Tuple[str, str]],
) -> str:
    """Assign one or more ideas to an attribute, with valence."""
    context_block = build_context_block(
        language=language, survey_question=survey_question, sector=sector,
        entity=entity, topic=topic, perspective=perspective, intent=intent,
    )
    menu = build_attribute_menu(attributes)
    ideas_block = "\n".join(f"[{idea_id}] {label}" for idea_id, label in ideas)

    return f"""You are a qualitative coding assistant. Assign each survey response idea below to the attribute it belongs to.

{context_block}

<facet>
Facet: {facet_name} — {facet_definition}
</facet>

<attributes>
{menu}

[A_NONE] None of the attributes above fits the idea.
</attributes>

<ideas>
{ideas_block}
</ideas>

### VALENCE (evaluation relative to the attribute)
- "+" Positive — the idea describes a positive instance of this attribute (present,
  sufficient, meeting expectations)
- "-" Negative — the idea describes a negative instance of this attribute (absent,
  insufficient, failing expectations)
- "0" Neutral — the idea is descriptive, ambiguous, or expresses no evaluation

Valence is not emotional sentiment. It is evaluative direction relative to the attribute.

Use each attribute's Boundary line to decide the doubtful cases; that is what it is for.

Judge every idea independently on its own text; do not let one assignment influence the
next. Return exactly one item per idea, echoing that idea's [id]. Do not skip ideas and
do not add ideas. If no attribute fits an idea, use "A_NONE" for that idea rather than
forcing it into the nearest one.

Begin processing now and {INSTRUCTOR_HINT}"""
