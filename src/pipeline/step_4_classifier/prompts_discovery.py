"""
Discovery: facets and the attributes they hold, in a single pass.
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
    build_taxonomy_block_L3,
    build_dimension_block
)

if TYPE_CHECKING:
    from pipeline.step_3_ideaExtractor.dimension_data import DimensionDefinition


# =============================================================================
# RESPONSE MODEL
# =============================================================================

class DiscoveredAttribute(BaseModel):
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
    facet_name: str = Field(
        ..., description=(
            "Short descriptive name for the facet, in the survey language "
            "(at most 5 words)"))
    facet_definition: str = Field(
        ..., description=(
            "What this facet captures — one clear underlying concept, in 1-2 "
            "sentences, in the survey language"))
    analytical_question: str = Field(
        ..., description=(
            "De unique analytical question this facet answers"))
    attributes: List[DiscoveredAttribute] = Field(
        ..., description=(
            "The attributes that fall under this facet. The fewest that cover "
            "the observations assigned to it"))

class DiscoveryResult(BaseModel):
    #scratchpad: str = Field(
    #    ..., description=(
    #        "Step-by-step reasoning before the final output: "
    #        "(1) cluster the observations by shared descriptive meaning, "
    #        "(2) name candidate facets and the observations supporting each, "
    #        "(3) check internal coherence — one clear concept per facet, "
    #        "(4) check distinctness — every pair ontologically distinct and "
    #        "semantically separable, "
    #       "(5) check the domain boundary and drop what belongs elsewhere, "
    #        "(6) for each surviving facet, name the attributes it holds, "
    #        "(7) prepare the final output"))
    facets: List[DiscoveredFacet] = Field(
        ..., description=(
            "The facets found in these observations, each with its attributes"))


# =============================================================================
# PROMPT — DISCOVERY
# =============================================================================

def _exclusion_lines(domain_label: str, boundary_test: str,exclusions: Optional[List[str]]) -> str:
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

    observations_block = "\n".join(f"{i}. {obs}" for i, obs in enumerate(observations, 1))
    boundary_block = _exclusion_lines(domain_label, domain_boundary_test, domain_exclusions)
    exclusion_hint = (
        "\n".join(f"- {x}" for x in domain_exclusions)
        if domain_exclusions else "- (no neighbouring domains were named)")

    return f"""You are a qualitative research analyst specializing in survey response analysis. 
Your task is to identify the fewest recurring facets that provide full coverage of a set of observations within one domain, and for each facet the fewest attributes that provide full coverage of what that facet holds.

# Taxonomy structure 

{build_taxonomy_block_L3(
    dimension=dimension, dimension_name=dimension_name,
    dimension_description=dimension_description)}

# Survey context  

{build_context_block(
    language=language,
    dimension_name=dimension_name, dimension_description=dimension_description,
    survey_question=survey_question,
    sector=sector, entity=entity, topic=topic, perspective=perspective, intent=intent)}

# The domain are working in:    

<taxonomy_domain>
{domain_label} — {domain_definition}{boundary_block}
</taxonomy_domain>

# The observations for you to analyze:

<observations>
{observations_block}
</observations>

# Task

Your task is to identify the fewest recurring facets that provide full coverage of a set of observations within one domain, and for each facet the fewest attributes that provide full coverage of what that facet holds.
You do both levels in one pass, because they constrain each other: a facet is only a good facet if the attributes under it are genuinely the same kind of thing.

## Objective

Facets must be conceptually orthogonal:

- Each facet must represent a different analytical lens or type of quality
- No facet may restate, contain, specialize, or operationalize another facet
- Two facets may both apply to the same observation only when they capture genuinely different aspects of it
- A coder must be able to state a different analytical question for each facet

Two facets asking "what subject is this about?" and "through what action is it enacted?"
are distinct lenses and may coexist, even when every observation under both speaks about
the same subject matter. What separates them is the question, not the vocabulary.

Attributes within a facet must be atomic and conceptually mutually exclusive:

- No attribute may be a synonym of another attribute
- No attribute may be a parent, subtype, component, combination, or concrete example of another attribute under the same facet
- All sibling attributes must describe the same kind of property and sit at the same level of abstraction
- The same atomic meaning must fit only one attribute within a facet
- An observation may receive multiple attributes when it explicitly contains multiple atomic meanings

Do not create a combined attribute when its meaning consists entirely of two existing attributes. 

If your inventory already holds attributes A and B, do not add a third meaning "A and B
together": an observation carrying both meanings receives both attributes.
 
# Process

Work through these steps before writing your final output.

**Step 1: Cluster the observations**
Group observations that share descriptive meaning. Identify what recurs. Focus on the kind of quality, characteristic, practice or property being described — not on whether it is being praised or criticised.

**Step 2: Name candidate facets**
From your clusters, name candidate facets. For each one note the name, the underlying concept it captures, and which observations support it. A facet names a recurring kind of meaning, not a single concrete observation.

**Step 3: Verify internal coherence**
For each candidate facet, check that it captures one clear concept. Reject or split any candidate that combines different kinds of phenomena, mixes description with evaluation, or is so broad that a coder could not apply it.

**Step 4: Verify conceptual orthogonality**
Facets and attributes must be mutually exclusive at the level of meaning, not at the level of observations.

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

All names, definitions and examples must be written in {language}.

{UNIVERSAL_RULES}

{INSTRUCTOR_HINT}"""