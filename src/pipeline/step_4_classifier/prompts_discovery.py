"""Discovery: facetten en de attributen die ze bevatten, in één beurt.

De twee lagen werden hiervoor apart uitgevraagd — eerst facetten per domein,
dan attributen per facet. Dat kostte een call per facet per chunk, en het gaf
de attribuutlaag een scope die al vastlag voordat er één idee was toegewezen.

Hier vraagt één call per (domein, chunk) allebei tegelijk: welke facetten zitten
er in deze observaties, en welke attributen zitten er in elk facet. Een model
dat beide niveaus in één blik ziet kan een attribuut niet in het verkeerde facet
hangen, want het bepaalt ze samen.

Het instructieskelet komt uit de opzet van vóór de herbouw: genummerde
scratchpad-stappen, "the fewest that provide full coverage", en een expliciete
distinctheidstoets per paar. Nieuw is stap 6, die per overgebleven facet de
attributen benoemt.

Er wordt **niet** naar dimensies of assen gevraagd. De prompt vraagt naar
facetten en attributen in die woorden; `dimension_data.py` levert wat die
niveaus betekenen voor de dimensie waaronder deze run draait.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from pydantic import BaseModel, Field

from .prompts_shared import (
    INSTRUCTOR_HINT,
    UNIVERSAL_RULES,
    _extract_definition,
    _extract_key_idea,
    build_context_block,
    build_taxonomy_block,
)

if TYPE_CHECKING:
    from pipeline.step_3_ideaExtractor.dimension_data import DimensionDefinition


# =============================================================================
# RESPONSE MODEL
# =============================================================================
#
# Elk veld hieronder wordt in de prompt bij naam genoemd, en de prompt vraagt
# geen veld dat hier niet staat. Twee registers die uit elkaar lopen leveren of
# een verwarde prompt of een antwoord dat we niet kunnen gebruiken.

class DiscoveredAttribute(BaseModel):
    """Een attribuut (L4), zoals één chunk het voorstelt."""
    attribute_name: str = Field(
        ..., description="Short descriptive name for the attribute (2-5 words)")
    attribute_definition: str = Field(
        ..., description=(
            "What this attribute captures — one concrete, observable property, "
            "in 1-2 sentences"))
    example_observations: List[str] = Field(
        ..., description=(
            "2-3 representative observations from the input, using the exact "
            "observation text"))


class DiscoveredFacet(BaseModel):
    """Een facet (L3) mét de attributen die eronder vallen."""
    facet_name: str = Field(
        ..., description="Short descriptive name for the facet (2-5 words)")
    facet_definition: str = Field(
        ..., description=(
            "What this facet captures — one clear underlying concept, in 1-2 "
            "sentences"))
    attributes: List[DiscoveredAttribute] = Field(
        ..., description=(
            "The attributes that fall under this facet. The fewest that cover "
            "the observations assigned to it"))


class DiscoveryResult(BaseModel):
    """Wat één (domein, chunk)-call teruggeeft."""
    scratchpad: str = Field(
        ..., description=(
            "Step-by-step reasoning before the final output: "
            "(1) cluster the observations by shared descriptive meaning, "
            "(2) name candidate facets and the observations supporting each, "
            "(3) check internal coherence — one clear concept per facet, "
            "(4) check distinctness — every pair ontologically distinct and "
            "semantically separable, "
            "(5) check the domain boundary and drop what belongs elsewhere, "
            "(6) for each surviving facet, name the attributes it holds, "
            "(7) prepare the final output"))
    facets: List[DiscoveredFacet] = Field(
        ..., description=(
            "The facets found in these observations, each with its attributes"))


# =============================================================================
# PROMPT
# =============================================================================

def _exclusion_lines(domain_label: str, boundary_test: str,
                     exclusions: Optional[List[str]]) -> str:
    """De domeingrens, of niets wanneer step 3 er geen heeft meegegeven."""
    lines = []
    if boundary_test:
        lines.append(f"Boundary test: {boundary_test}")
    if exclusions:
        lines.append(
            "This domain EXCLUDES the following, which belong to other domains: "
            + "; ".join(exclusions))
    if not lines:
        return ""
    return "\n" + "\n".join(lines)


def build_discovery_prompt(
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
    domain_exclusions: Optional[List[str]],
    observations: List[str],
) -> str:
    """Facetten én hun attributen, uit één chunk observaties binnen één domein."""
    rules = dimension.prompt_rules
    facet_definition = _extract_definition(rules.facet_instruction)
    attribute_definition = _extract_definition(rules.attribute_instruction)
    facet_key_idea = _extract_key_idea(rules.facet_instruction)
    attribute_key_idea = _extract_key_idea(rules.attribute_instruction)

    observations_block = "\n".join(
        f"{i}. {obs}" for i, obs in enumerate(observations, 1))
    boundary_block = _exclusion_lines(
        domain_label, domain_boundary_test, domain_exclusions)
    exclusion_hint = (
        "\n".join(f"- {x}" for x in domain_exclusions)
        if domain_exclusions else "- (no neighbouring domains were named)")

    return f"""You are a qualitative research analyst specializing in survey response analysis.
Your task is to identify the fewest recurring facets that provide full coverage of a set of
observations within one domain, and for each facet the fewest attributes that provide full
coverage of what that facet holds.

You do both levels in one pass, because they constrain each other: a facet is only a good
facet if the attributes under it are genuinely the same kind of thing.

# What a facet is

{facet_definition}
Key idea: {facet_key_idea}

A facet must be:
- Descriptive and data-grounded, never evaluative
- Internally coherent — one clear underlying concept
- Externally distinctive — ontologically distinct and semantically separable from the others
- Strictly within the domain boundary
- Supported by repeated patterns across observations, not by a single response

# What an attribute is

{attribute_definition}
Key idea: {attribute_key_idea}

An attribute must:
- Name a specific observable property, not repeat a verbatim span from a response
- Be descriptive and non-evaluative
- Stay strictly within the facet it sits under
- Be internally coherent and ontologically distinct from its siblings
- Add unique conceptual value — no attribute that restates another in different words

{build_context_block(
    language=language, survey_question=survey_question, sector=sector,
    entity=entity, topic=topic, perspective=perspective, intent=intent)}

{build_taxonomy_block(
    dimension=dimension, dimension_name=dimension_name,
    dimension_description=dimension_description)}

You are working within this domain, and only within it:

<taxonomy_domain>
{domain_label} — {domain_definition}{boundary_block}
</taxonomy_domain>

Here are the observations you need to analyze:

<observations>
{observations_block}
</observations>

# Instructions

Work through these steps in the `scratchpad` field before writing your final output.

**Step 1: Cluster the observations**
Group observations that share descriptive meaning. Identify what recurs. Focus on the kind
of quality, characteristic, practice or property being described — not on whether it is
being praised or criticised.

**Step 2: Name candidate facets**
From your clusters, name candidate facets. For each one note the name, the underlying
concept it captures, and which observations support it.
A facet names a recurring kind of meaning, not a single concrete observation.

**Step 3: Verify internal coherence**
For each candidate facet, check that it captures one clear concept. Reject or split any
candidate that combines different kinds of phenomena, mixes description with evaluation,
or is so broad that a coder could not apply it.

**Step 4: Verify distinctness**
Check every pair of candidate facets. They must be ontologically distinct — neither is a
subset of the other, and they do not overlap in conceptual space — and semantically
separable, so a coder always knows which one applies. If a pair fails, either merge them
into one broader facet or redraw the boundary between them.

**Step 5: Verify the domain boundary**
Every retained facet must fall strictly inside {domain_label}. Drop anything that belongs
more naturally to a neighbouring domain, including:
{exclusion_hint}

**Step 6: Name the attributes inside each facet**
For each facet that survived, look again at the observations you assigned to it and name
the attributes it holds — the specific properties those observations point at. Apply the
same two tests one level down: each attribute captures one property, and no two attributes
under the same facet overlap or restate each other.
If a facet turns out to hold only one attribute, that is a sign the facet and the attribute
are the same thing. Decide which level the concept really belongs to and keep it there.

**Step 7: Prepare the final output**
Keep only what passed every check. Use the fewest facets that cover the observations, and
under each the fewest attributes that cover what it holds.

# Output

Return a JSON object with these fields:
- `scratchpad`: your reasoning for steps 1-7
- `facets`: an array, one entry per facet, each with:
  - `facet_name`: a short descriptive name in {language} (2-5 words)
  - `facet_definition`: what the facet captures, in {language} (1-2 sentences)
  - `attributes`: an array, one entry per attribute inside that facet, each with:
    - `attribute_name`: a short descriptive name in {language} (2-5 words)
    - `attribute_definition`: the observable property it captures, in {language} (1-2 sentences)
    - `example_observations`: 2-3 observations from the input, using the exact observation text

All names, definitions and examples must be written in {language}.

{UNIVERSAL_RULES}

{INSTRUCTOR_HINT}"""
