"""Prompt builders and response models for the facet layer (L3) of step 4.

Four phases, in the order step 3 uses for the domain layer:

  1. discovery      build_facet_discovery_prompt      per (domain, chunk)
  2. consolidation  build_facet_consolidation_prompt  per domain, over chunks
  3. assignment     build_facet_assignment_prompt     per batch of ideas
  4. refinement     build_facet_refinement_prompt     per domain, after assignment

Discovery proposes, consolidation settles the inventory before a single idea is
assigned, assignment fills it, and refinement judges the result against what the
facets actually ended up holding. The domain is fixed throughout: no phase here
can move a facet or an idea to another domain.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Literal, Tuple

from pydantic import BaseModel, Field, create_model

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
# §1 DISCOVERY — per (domain, chunk)
# =============================================================================

class DiscoveredFacet(BaseModel):
    """One facet (L3) proposed from a chunk of observations within a domain.

    Four fields carry the boundary, the same four step 3 uses per domain: what
    it is (`facet_definition`), the question that decides membership
    (`boundary_test`), and what it is not (`exclusions`). There is deliberately
    no separate inclusion/exclusion rule pair — the definition IS the inclusion
    rule, and two fields for one idea drift apart.
    """
    facet_name: str = Field(
        ..., description="Short descriptive name for the facet (2-5 words)"
    )
    facet_definition: str = Field(
        ..., description=(
            "One-sentence inclusion definition naming a single aspect. "
            "No examples, no enumerations"
        )
    )
    boundary_test: str = Field(
        ..., description=(
            "A single yes/no question a coder asks to decide whether an idea "
            "belongs to THIS facet rather than a neighbouring one"
        )
    )
    exclusions: List[str] = Field(
        ..., description=(
            "1-3 short phrases naming what does NOT belong here — especially "
            "the neighbouring facet it is most easily confused with"
        )
    )
    example_observations: List[str] = Field(
        ..., description="3-5 observations from the input that exemplify this facet"
    )


class FacetDiscoveryResult(BaseModel):
    """Discovery output for one chunk."""
    scratchpad: str = Field(
        ..., description=(
            "Reasoning before the final facet set: "
            "(1) group the observations by what they say about the domain, "
            "(2) name the candidate facets, "
            "(3) check each candidate is one aspect and not a compound, "
            "(4) check each pair for a clean boundary and merge where there is none, "
            "(5) verify every observation has a home"
        )
    )
    facets: List[DiscoveredFacet] = Field(
        ..., description="The facets found in these observations"
    )


def build_facet_discovery_prompt(
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
    domain_boundary_test: str,
    domain_exclusions: List[str],
    observations: List[str],
) -> str:
    """Propose facets (L3) from one chunk of observations within one domain."""
    context_block = build_context_block(
        language=language, survey_question=survey_question, sector=sector,
        entity=entity, topic=topic, perspective=perspective, intent=intent,
    )
    taxonomy_block = build_taxonomy_block(
        dimension=dimension, dimension_name=dimension_name,
        dimension_description=dimension_description,
    )
    diagnostic = level_diagnostic(dimension, "facet")
    observations_block = "\n".join(
        f"{i}. {obs}" for i, obs in enumerate(observations, 1)
    )
    exclusions_line = "; ".join(domain_exclusions) if domain_exclusions else "(none given)"

    return f"""You are a qualitative research methodologist specializing in taxonomy development for survey analysis.
Your task is to identify facets within one domain, based on survey response data.

{context_block}

{taxonomy_block}

You are working inside ONE domain. Everything you return belongs to it, and nothing
that falls outside it may be returned.

<domain>
Domain: {domain_label}
Definition: {domain_definition}
Boundary test: {domain_boundary_test}
Does NOT belong to this domain (these have their own domains): {exclusions_line}
</domain>

Here are the observations you need to account for:

<observations>
{observations_block}
</observations>

## YOUR TASK

Identify the **facets** (level 3) present in these observations.

The question every facet must answer for this dimension is:

<facet_diagnostic>
{diagnostic}
</facet_diagnostic>

A facet is an answer to that question. If a candidate does not answer it, it is not a
facet — it is either the domain restated (one level up) or a single concrete property
(one level down).

Aim for the smallest set of facets that still gives every observation a clear home.
Fewer is better only when full coverage and distinctness both hold. Each facet must be:

- **Ontologically distinct** — no two facets may share conceptual space. A facet must
  not be a subset of another, and two facets must not be two lenses on one phenomenon.
- **Semantically distant** — a coder assigning an observation must not plausibly
  hesitate between two facets. No "could go either way" situations.
- **One aspect** — not a compound list of several concerns joined together.
- **Inside the domain** — strictly within the boundary stated above.

NO BROAD CATCH-ALL: do not create a vague bucket that absorbs a large share of the
observations by mixing unrelated things. If a candidate would do that, split it along
the sharper distinctions the observations actually show.

Rare or one-off observations do not each earn a facet. Group them under the facet whose
definition honestly covers them, and only give them their own facet when the same
distinction recurs.

{UNIVERSAL_RULES}

## OUTPUT

Work through your reasoning in the scratchpad field first.

For EACH facet provide:
- **facet_name** — a short descriptive name
- **facet_definition** — one sentence naming a single aspect, with no examples or
  enumerations
- **boundary_test** — one yes/no question that decides membership
- **exclusions** — what does NOT belong, naming the neighbouring facet it is most
  easily confused with
- **example_observations** — 3-5 observations from the input above, copied exactly

All facet names, definitions, boundary tests and exclusions must be written in {language}.

Begin processing now and {INSTRUCTOR_HINT}"""


# =============================================================================
# §2 CONSOLIDATION — per domain, across chunks, before any idea is assigned
# =============================================================================

class ConsolidatedFacet(DiscoveredFacet):
    """One facet surviving consolidation, with the candidates it absorbed."""
    source_facets: List[str] = Field(
        ..., description=(
            "The facet_name of every candidate that goes into this one. "
            "A candidate that is kept unchanged lists its own name"
        )
    )


class FacetConsolidationResult(BaseModel):
    """The settled facet inventory for one domain."""
    scratchpad: str = Field(
        ..., description=(
            "Consolidation reasoning: "
            "(1) list the unique candidates across all chunks, "
            "(2) group the ones that overlap conceptually, "
            "(3) name and define each consolidated facet, "
            "(4) for every surviving pair ask whether one response could belong "
            "to both, and merge when it could, "
            "(5) verify the survivors still cover everything the candidates covered, "
            "(6) write each survivor's boundary_test and exclusions"
        )
    )
    facets: List[ConsolidatedFacet] = Field(
        ..., description="The complete facet set for this domain after consolidation"
    )


def _build_candidate_block(candidates: List[DiscoveredFacet]) -> str:
    """Render the chunk yield as numbered candidates, each with its evidence.

    The observations that produced a proposal travel with it. That is what makes
    this call judgeable: two candidates whose labels differ but whose observations
    are the same thing are a merge, and no amount of staring at labels alone would
    have shown it.
    """
    blocks = []
    for i, candidate in enumerate(candidates, 1):
        exclusions = "; ".join(candidate.exclusions) if candidate.exclusions else "(none)"
        observations = "; ".join(candidate.example_observations)
        blocks.append(
            f"[C{i}] {candidate.facet_name}\n"
            f"     Definition: {candidate.facet_definition}\n"
            f"     Boundary test: {candidate.boundary_test}\n"
            f"     Does not belong: {exclusions}\n"
            f"     Observations that produced this proposal: {observations}"
        )
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
    domain_boundary_test: str,
    candidates: List[DiscoveredFacet],
) -> str:
    """Settle one domain's facet inventory, across all chunk proposals.

    Runs before assignment. Each chunk proposed facets without seeing the others,
    so the same concept comes back under several names. This call decides which
    of those are one facet, using the observations each proposal was built on.
    """
    context_block = build_context_block(
        language=language, survey_question=survey_question, sector=sector,
        entity=entity, topic=topic, perspective=perspective, intent=intent,
    )
    taxonomy_block = build_taxonomy_block(
        dimension=dimension, dimension_name=dimension_name,
        dimension_description=dimension_description,
    )
    diagnostic = level_diagnostic(dimension, "facet")
    candidate_block = _build_candidate_block(candidates)

    return f"""You are a taxonomy consolidation specialist for survey coding.
Your task is to merge facet proposals from several independent passes over one domain
into a single, coherent set of facets.

{context_block}

{taxonomy_block}

You are working inside ONE domain. Every facet you return belongs to it.

<domain>
Domain: {domain_label}
Definition: {domain_definition}
Boundary test: {domain_boundary_test}
</domain>

The question every facet must answer for this dimension is:

<facet_diagnostic>
{diagnostic}
</facet_diagnostic>

Here are the candidate facets. Each pass saw a different sample of the responses and
did not see the other passes, so the same facet may appear several times under
different names. Each candidate carries the observations that produced it — those are
your evidence:

<candidates>
{candidate_block}
</candidates>

## YOUR TASK

Consolidate these candidates into the fewest mutually exclusive facets needed for full
coverage.

Judge the candidates on their observations, not on their labels. Two labels that read
differently but were produced by the same kind of observation are ONE facet. Two labels
that read alike but were produced by different observations are TWO.

Consolidation principles:

- **MERGE** candidates that overlap conceptually, are near-equivalent, or where one is
  a subset of the other.
- **MERGE** candidates that are two lenses on the same phenomenon — different wording
  for one underlying distinction.
- **THE BOUNDARY TEST DECIDES.** For each pair of survivors, write the boundary that
  separates them. If you cannot state a clean boundary between a facet and its nearest
  neighbour, they are not two facets — merge them.
- **ENSURE ontological distinctness** — no two facets may share conceptual space, and
  none may be a subset of another.
- **ENSURE semantic distance** — a coder must not plausibly hesitate between two
  facets. No "could go either way" situations.
- **MAINTAIN full coverage** — the survivors must collectively cover everything the
  candidates covered. Consolidating is not discarding.
- **MINIMIZE the count** while preserving distinctions that the observations actually
  show. If the observations hold four distinct answers to the facet question, return
  four facets — do not collapse them because fewer is tidier.
- **STAY inside the domain.** A candidate that falls outside the domain boundary is not
  a facet to keep; leave it out rather than widening the domain to fit it.

Every candidate you consume must be listed in the `source_facets` of the facet that
consumes it. A candidate you do not list is left standing as it is, so list them.

{UNIVERSAL_RULES}

## OUTPUT

Work through the consolidation in the scratchpad field first.

For EACH consolidated facet provide:
- **facet_name** — a short descriptive name
- **facet_definition** — one sentence naming a single aspect, no examples or enumerations
- **boundary_test** — one yes/no question that decides membership
- **exclusions** — what does NOT belong, naming the neighbouring facet it is most easily
  confused with
- **example_observations** — 2-5 observations, copied exactly from the candidates above
- **source_facets** — the facet_name of every candidate consumed into this one

All facet names, definitions, boundary tests and exclusions must be written in {language}.

Begin processing now and {INSTRUCTOR_HINT}"""


# =============================================================================
# §3 ASSIGNMENT — ideas into the settled inventory
# =============================================================================
#
# Two things this phase deliberately does NOT carry, unlike the other three.
# The taxonomy block and the universal rules are both about what a taxonomy may
# contain; assignment creates nothing, it picks an id from a menu whose entries
# already carry their definitions and boundaries. This is also the only phase
# that runs at the volume of the dataset, so every line here is paid thousands
# of times.


def build_facet_menu(facets: List[ConsolidatedFacet]) -> str:
    """Render the settled facets as a numbered menu.

    The [F#] id is what the response is keyed on, so the numbering here and the
    id list handed to `build_facet_assignment_model` must come from the same
    list in the same order.
    """
    lines = []
    for i, f in enumerate(facets, 1):
        exclusions = "; ".join(f.exclusions) if f.exclusions else ""
        examples = "; ".join(f.example_observations[:3])
        block = (
            f"[F{i}] {f.facet_name}\n"
            f"     Description: {f.facet_definition}\n"
            f"     Boundary: {f.boundary_test}"
        )
        if exclusions:
            block += f"\n     Does not belong here: {exclusions}"
        if examples:
            block += f"\n     Examples: {examples}"
        lines.append(block)
    return "\n\n".join(lines)


def build_facet_assignment_model(facet_ids: List[str], idea_ids: List[str]):
    """Runtime response model for one assignment call.

    Both id spaces are Literals, so a hallucinated facet id or a made-up idea id
    is a schema violation that instructor retries, not a content error that has
    to be caught downstream. "F_NONE" is the honest way out when nothing fits;
    the caller escalates those rather than forcing a wrong home.
    """
    facet_id_literal = Literal[tuple(facet_ids + ["F_NONE"])]  # type: ignore[valid-type]
    idea_id_literal = Literal[tuple(idea_ids)]  # type: ignore[valid-type]

    item_model = create_model(
        "FacetAssignmentItem",
        idea_id=(idea_id_literal, Field(
            ..., description="The [id] tag of the idea, echoed exactly")),
        assigned_facet_id=(facet_id_literal, Field(
            ..., description=(
                "The facet id from the [F#] prefix. Return ONLY the id. "
                "Use F_NONE when no facet fits this idea"))),
        confidence=(float, Field(
            ..., ge=0.0, le=1.0, description="Assignment confidence (0.0-1.0)")),
        valence=(Literal["+", "-", "0"], Field(
            default="0",
            description=(
                "Evaluative direction relative to the facet: "
                "+ positive, - negative, 0 neutral"))),
    )
    return create_model(
        "FacetAssignmentResult",
        assignments=(List[item_model], Field(
            ..., description=(
                "Exactly one assignment per idea listed in the prompt, "
                "no idea skipped, no idea added"))),
    )


def build_facet_assignment_prompt(
    *,
    language: str,
    survey_question: str,
    sector: str,
    entity: str,
    topic: str,
    perspective: str,
    intent: str,
    domain_label: str,
    domain_definition: str,
    facets: List[ConsolidatedFacet],
    ideas: List[Tuple[str, str]],
) -> str:
    """Assign one or more ideas to a facet, with valence.

    One builder for one or many ideas: a single idea is a list of length one.
    A separate single-idea variant would be a second prompt for one task, and
    two prompts for one task drift apart.
    """
    context_block = build_context_block(
        language=language, survey_question=survey_question, sector=sector,
        entity=entity, topic=topic, perspective=perspective, intent=intent,
    )
    menu = build_facet_menu(facets)
    ideas_block = "\n".join(f"[{idea_id}] {label}" for idea_id, label in ideas)

    return f"""You are a qualitative coding assistant. Assign each survey response idea below to the facet it belongs to.

{context_block}

<domain>
Domain: {domain_label} — {domain_definition}
</domain>

<facets>
{menu}

[F_NONE] None of the facets above fits the idea.
</facets>

<ideas>
{ideas_block}
</ideas>

### VALENCE (evaluation relative to the facet)
- "+" Positive — the idea describes the facet as met, present, or enhanced
- "-" Negative — the idea describes the facet as failing, absent, or detracted from
- "0" Neutral — the idea is descriptive, ambiguous, or expresses no evaluation

Valence is not emotional sentiment. It is evaluative direction relative to the facet.

Use each facet's Boundary line to decide the doubtful cases; that is what it is for.

Judge every idea independently on its own text; do not let one assignment influence the
next. Return exactly one item per idea, echoing that idea's [id]. Do not skip ideas and
do not add ideas. If no facet fits an idea, use "F_NONE" for that idea rather than
forcing it into the nearest one.

Begin processing now and {INSTRUCTOR_HINT}"""
