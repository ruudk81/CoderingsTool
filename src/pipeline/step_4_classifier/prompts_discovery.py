"""Discovery: facets and the attributes they hold, in a single pass.

The two levels used to be asked separately — facets per domain first, then
attributes per facet. That cost one call per facet per chunk, and it gave the
attribute layer a scope that was already fixed before a single idea had been
assigned.

Here one call per (domain, chunk) asks for both at once: which facets these
observations contain, and which attributes sit inside each facet. A model that
sees both levels in one view cannot hang an attribute under the wrong facet,
because it determines them together.

The instruction skeleton comes from the pre-rebuild design: numbered scratchpad
steps, "the fewest that provide full coverage", and an explicit pairwise
distinctness test. Step 6 is new — it names the attributes inside each surviving
facet.

Dimensions and axes are **not** asked for. The prompt asks for facets and
attributes in those words; `dimension_data.py` supplies what those levels mean
for the dimension this run operates under.
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
# Every field below is named in the prompt, and the prompt asks for no field
# that is absent here. Two registers drifting apart yield either a confused
# prompt or an answer we cannot use.

class DiscoveredAttribute(BaseModel):
    """One attribute (L4), as a single chunk proposes it."""
    attribute_name: str = Field(
        ..., description=(
            "Short descriptive name for the attribute, in the survey language "
            "(at most 5 words)"))
    attribute_definition: str = Field(
        ..., description=(
            "What this attribute captures — one concrete, observable property, "
            "in 1-2 sentences, in the survey language"))
    example_observations: List[str] = Field(
        ..., description=(
            "1-3 representative observations, each copied exactly as shown, "
            "WITHOUT any leading number. Give what this attribute actually "
            "has; never invent, pad or borrow one"))


class DiscoveredFacet(BaseModel):
    """One facet (L3), together with the attributes that fall under it."""
    facet_name: str = Field(
        ..., description=(
            "Short descriptive name for the facet, in the survey language "
            "(at most 5 words)"))
    facet_definition: str = Field(
        ..., description=(
            "What this facet captures — one clear underlying concept, in 1-2 "
            "sentences, in the survey language"))
    attributes: List[DiscoveredAttribute] = Field(
        ..., description=(
            "The attributes that fall under this facet. The fewest that cover "
            "the observations assigned to it"))


class DiscoveryResult(BaseModel):
    """What one (domain, chunk) call returns."""
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
# PROMPT — DISCOVERY
# =============================================================================

def _exclusion_lines(domain_label: str, boundary_test: str,
                     exclusions: Optional[List[str]]) -> str:
    """The domain boundary, or nothing when step 3 supplied none."""
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
    """Facets and their attributes, from one chunk of observations in one domain."""
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
Your task is to identify the fewest recurring facets that provide full coverage of a set of observations within one domain, and for each facet the fewest attributes that provide full coverage of what that facet holds.

# Taxonomy Structure

{build_taxonomy_block(
    dimension=dimension, dimension_name=dimension_name,
    dimension_description=dimension_description)}

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

{UNIVERSAL_RULES}

# Conceptual Orthogonality and Coding Multiplicity

Facets and attributes must be mutually exclusive at the level of meaning, not at the level of observations.

## Requirements for Facets

Facets must be conceptually orthogonal:

- Each facet must represent a different analytical lens or type of quality
- No facet may restate, contain, specialize, or operationalize another facet
- Two facets may both apply to the same observation only when they capture genuinely different aspects of it
- A coder must be able to state a different analytical question for each facet

Two facets asking "what subject is this about?" and "through what action is it enacted?"
are distinct lenses and may coexist, even when every observation under both speaks about
the same subject matter. What separates them is the question, not the vocabulary.

## Requirements for Attributes

Attributes within a facet must be atomic and conceptually mutually exclusive:

- No attribute may be a synonym of another attribute
- No attribute may be a parent, subtype, component, combination, or concrete example of another attribute under the same facet
- All sibling attributes must describe the same kind of property and sit at the same level of abstraction
- The same atomic meaning must fit only one attribute within a facet
- An observation may receive multiple attributes when it explicitly contains multiple atomic meanings

Do not create a combined attribute when its meaning consists entirely of two existing attributes. 

If your inventory already holds attributes A and B, do not add a third meaning "A and B
together": an observation carrying both meanings receives both attributes.

## Abstraction-Level Test

For every pair of candidate facets and every pair of sibling attributes, test:

1. Is A a broader category that includes B?
2. Is A a subtype, component, manifestation, or implementation of B?
3. Is A a combination of B and another category?
4. Could the same single atomic meaning fit both A and B?
5. Do A and B answer the same analytical question?

If the answer to any of questions 1–4 is yes, the pair is not mutually exclusive and must be redrawn, merged, split, or one item must be removed.

If the answer to question 5 is no, the concepts may belong in different facets, provided each facet represents a coherent and independently analyzable lens.

## Content Versus Coding Modifiers

Only substantive content belongs in the facet and attribute taxonomy.

Do not create content facets or attributes for:
- valence
- intensity or degree
- comparative strength
- certainty or doubt
- authenticity or credibility

Treat these as separate coding modifiers unless they are explicitly part of the domain definition. Test a candidate by stripping the qualifier from it: if what remains is a subject already named elsewhere in your inventory, the qualifier was a hedge, a comparison or a doubt about that subject — not a subject of its own.

## Orientation Versus Implementation

Do not infer a concrete practice from a general adjective or orientation.

- A word naming a general disposition or quality describes an attributed orientation, not a practice, unless the observation also states a concrete action
- Create an implementation attribute only when the observation explicitly names an activity, a policy or mechanism, an operational practice, a resource commitment, a monitoring process, or a form of support

A thematic orientation and a concrete implementation may be coded together when an observation explicitly expresses both. They belong to different facets because one captures what the observation is about and the other captures how it is enacted.

## Priority of Requirements

Apply the following priority order:

1. Domain validity
2. Conceptual coherence
3. Conceptual mutual exclusivity
4. Consistent level of abstraction
5. Evidence from recurring observations
6. Full coverage of recurring, domain-relevant meanings
7. Minimization of the number of facets and attributes

"Use the fewest facets and attributes" applies only after all higher-priority requirements have been satisfied. Never merge distinct concepts or introduce an overlapping umbrella category merely to reduce the number of items or increase coverage.

Full coverage refers to recurring, domain-relevant meanings, not necessarily to every individual observation. An observation may remain uncategorized when it is outside the domain, insufficiently specific, non-recurring, or contains only a coding modifier.

# Your Task

{build_context_block(
    language=language, survey_question=survey_question, sector=sector,
    entity=entity, topic=topic, perspective=perspective, intent=intent)}

You are working within this domain, and only within it:

<taxonomy_domain>
{domain_label} — {domain_definition}{boundary_block}
</taxonomy_domain>

Here are the observations you need to analyze:

<observations>
{observations_block}
</observations>


# Process

Work through these steps in the `scratchpad` field before writing your final output.

**Step 1: Cluster the observations**
Group observations that share descriptive meaning. Identify what recurs. Focus on the kind of quality, characteristic, practice or property being described — not on whether it is being praised or criticised.

**Step 2: Name candidate facets**
From your clusters, name candidate facets. For each one note the name, the underlying concept it captures, and which observations support it. A facet names a recurring kind of meaning, not a single concrete observation.

**Step 3: Verify internal coherence**
For each candidate facet, check that it captures one clear concept. Reject or split any candidate that combines different kinds of phenomena, mixes description with evaluation, or is so broad that a coder could not apply it.

**Step 4: Verify conceptual orthogonality**
Check every pair of candidate facets and every pair of sibling attributes.

Facets and attributes must be mutually exclusive at the level of meaning:
- they must not be synonyms
- neither may contain or specialize the other
- neither may be a component, combination, manifestation, or implementation of the other
- the same atomic meaning must not fit both

Observations themselves do not need to be mutually exclusive. One observation may receive multiple codes when it contains multiple distinct meanings.

For each retained facet, state the unique analytical question it answers. For each pair of sibling attributes, verify that both answer that same facet question while naming different, atomic properties at the same level of abstraction.

If a pair fails, merge, split, relocate, redraw, or remove one of the items. Do not solve overlap by adding a broader umbrella category.

**Step 5: Verify the domain boundary**
Every retained facet must fall strictly inside {domain_label}. Drop anything that belongs
more naturally to a neighbouring domain, including:
{exclusion_hint}

**Step 6: Name the attributes inside each facet**
For each facet that survived, look again at the observations you assigned to it and name the attributes it holds — the specific properties those observations point at. Apply the same two tests one level down: each attribute captures one property, and no two attributes under the same facet overlap or restate each other.

If a facet turns out to hold only one attribute, that is a sign the facet and the attribute are the same thing. Decide which level the concept really belongs to and keep it there.

**Step 7: Prepare the final output**
Keep only what passed every check. Use the fewest facets that cover the observations, and under each the fewest attributes that cover what it holds.

Before finalizing, perform a last parent–child, whole–part, generic–specific, and orientation–implementation overlap check. Do not return the taxonomy if any such relationship remains between sibling attributes or between facets.

# Output

Return a JSON object with these fields:
- `scratchpad`: your reasoning for steps 1-7
- `facets`: an array, one entry per facet, each with:
  - `facet_name`: a short descriptive name in {language} (at most 5 words)
  - `facet_definition`: what the facet captures, in {language} (1-2 sentences)
  - `attributes`: an array, one entry per attribute inside that facet, each with:
    - `attribute_name`: a short descriptive name in {language} (at most 5 words)
    - `attribute_definition`: the observable property it captures, in {language} (1-2 sentences)
    - `example_observations`: 1-3 observations this attribute actually holds, each copied
      exactly as its line reads but WITHOUT the leading number. An attribute resting on one
      observation gives one; never pad the list, and never borrow from another attribute

All names, definitions and examples must be written in {language}.

{INSTRUCTOR_HINT}"""
