"""Consolidation prompts for step 4.

Discovery proposes facets and their attributes together, per chunk. What comes
back is the same concept under several names at two levels at once. Settling
that is this module's job, and it takes two calls with different scopes — see
dev/ARCHITECTURE.md.
"""
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

from pydantic import BaseModel, Field

from pipeline.step_4_classifier.prompts_discovery import DiscoveredAttribute
from pipeline.step_4_classifier.prompts_shared import (
    INSTRUCTOR_HINT, UNIVERSAL_RULES, build_context_block,
    build_facets_attributes_block, build_taxonomy_block,
)

if TYPE_CHECKING:
    from pipeline.step_3_ideaExtractor.dimension_data import DimensionDefinition


#: Examples shown per candidate attribute, in the attribute call — the facet
#: call renders no examples at all. That call is asked to carry examples over,
#: so it can only pass on what it is shown: rendering one while asking for two
#: or three left the model a choice between merging attributes it had just
#: judged distinct, and writing an example that was never in the data.
_EXAMPLES_SHOWN = 3


# =============================================================================
# PROMPT — FACET CONSOLIDATION
# =============================================================================


@dataclass
class FacetPool:
    """One facet in flight through the facet phase, with what it has absorbed.

    Not a response model: it is what the classifier carries between rounds, and
    a round-two candidate has already collected the attributes of everything
    that folded into it. The attributes ride along unconsolidated — settling
    them is the next phase's job — and they are rendered as evidence only.
    """
    facet_name: str
    facet_definition: str
    facet_question: str
    attributes: List[DiscoveredAttribute]


class SettledFacet(BaseModel):
    """A facet after consolidation. It states what folded into it, and it holds
    no attributes: settling those is a call of its own."""
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


class FacetConsolidationResult(BaseModel):
    """What one facet-consolidation call per domain returns.

    `source_facet_ids` is not bookkeeping but a safety net: without it a
    candidate that was merged looks exactly like a candidate that was
    forgotten, since neither appears in the answer. With it, whatever nobody
    claims stays instead of vanishing — and in rounds that counts double,
    because what drops out in round one never comes back.
    """
    decision_summary: List[str] = Field(
        ..., description=(
            "One short line per consolidation decision that took judgement, "
            "each stating what was done and why. Not a reasoning trace, and "
            "not a line for every candidate: only the calls a reader would "
            "want to check"))
    facets: List[SettledFacet] = Field(
        ..., description=(
            "The fewest mutually exclusive facets that cover the domain"))


def build_facet_candidate_index(pools: List[FacetPool]) -> Dict[str, FacetPool]:
    """`F1`, `F2`, … for the candidates of one facet-consolidation call.

    Positional and therefore deterministic for a given task, which is all that
    is needed: the ids exist for the length of one call and are never stored.
    Provenance runs on them because names are not unique.
    """
    return {f"F{i}": pool for i, pool in enumerate(pools, 1)}

_FACET_SELF_REFERENCE = re.compile(r"\b(?:(?:deze|dit)\s+)?facet\b", re.IGNORECASE)

def _strip_facet_self_reference(text: str) -> str:
      """Haalt `deze facet` / `facet` weg, in welke casing dan ook."""
      return re.sub(r"\s{2,}", " ", _FACET_SELF_REFERENCE.sub("", text)).strip()

def build_facet_candidate_block(
    pools: List[FacetPool],
    recurrence: Dict[str, int],
    n_passes: int,
) -> str:
    """Every pass's facets, with their reach and the attributes inside them.

    The attributes are listed by NAME ONLY. They are here so the model can
    judge whether what sits under a facet is one kind of thing — the test that
    makes a facet a good facet. Names are enough for that, and no more than that
    is wanted: it was rendering attributes in full, with their definitions and
    examples, that gave the predecessor of this call enough material to settle
    the attributes as well, and so made it do two jobs at once. Settling them is
    the next call's work, on the pool this one hands it.
    """
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
    """Every pass's facets for one domain, folded into one flat inventory.

    Only the facet inventory is decided here. The attributes come along as
    evidence of what a candidate contains, because a facet is only a good facet
    if what sits under it is one kind of thing — but they are not returned, and
    consolidating them is a call of its own.
    """
    exclusion_hint = (
        "\n".join(f"- {x}" for x in domain_exclusions)
        if domain_exclusions else "- (no neighbouring domains were named)")

    return f"""You are a taxonomy consolidation specialist for surveys.
 Your task is to organize group of attributes into a minimal set of facets within a given domain.

{build_context_block(
    language=language, survey_question=survey_question, sector=sector,
    entity=entity, topic=topic, perspective=perspective, intent=intent)}

This is the taxonomy structure you are working with:
{build_taxonomy_block(
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

# Rules

1) The set of facets need to be MECE; mutually exclusive and collectively exhaustive. This means that the facets are not allowed to overlap semantically or meaningfull in light of the survey question. And this means that the set of facets should provide full coverage for all attributes.

2) You need to find the minimal number of facets to organize the attribute groups by beging MECE.

{UNIVERSAL_RULES}

{INSTRUCTOR_HINT}"""


# =============================================================================
# PROMPT — ATTRIBUTE CONSOLIDATION
# =============================================================================


class SettledAttribute(DiscoveredAttribute):
    """An attribute after consolidation, stating what folded into it."""
    # Overridden rather than inherited: this phase has no observations in view,
    # only what the candidates brought, so its examples are carried over and
    # never chosen. The clause against merging to reach a count belongs here and
    # not in discovery, because this is the phase that may merge at all.
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


class AttributeConsolidationResult(BaseModel):
    """What one attribute-consolidation call per facet returns.

    Same safety net as one level up: without `source_attribute_ids` a merged
    candidate is indistinguishable from a forgotten one, since neither appears
    in the answer.
    """
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
    """`A1`, `A2`, … for the pool of one facet.

    Flat, because one call is one facet: there is no second level for the id to
    disambiguate. The facet-level ids of the previous phase do not survive into
    this one — what arrives here is a pool, not a nesting.
    """
    return {f"A{i}": attribute for i, attribute in enumerate(attributes, 1)}

def build_attribute_candidate_block(
    attributes: List[DiscoveredAttribute],
    recurrence: Dict[str, int],
    n_passes: int,
) -> str:
    """The pooled attributes of one facet, each with its reach and examples.

    Shown in full, unlike in the facet call: this is the material being
    consolidated, not evidence about something else. Three examples, because
    the output spec asks the model to carry examples over and it can only pass
    on what it is shown.
    """
    lines = []
    for attribute_id, attribute in build_attribute_candidate_index(attributes).items():
        times = recurrence.get(attribute.attribute_name, 1)
        lines.append(
            f"[{attribute_id}] {attribute.attribute_name} ")
            #f"[{times}/{n_passes} passes]: {attribute.attribute_definition}")
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
    facet_name: str,
    facet_definition: str,
    facet_question: str,
    candidate_block: str,
) -> str:
    """The pooled attributes of one settled facet, folded into one minimal set.

    Its own call, with its own scope. As step 6 of the combined prompt this
    judgement ran on a pool of dozens while the model's attention was on the
    facet decision above it — measured on 2026-08-15: six of eight recorded
    decisions were about facets while the attribute side had to take
    seventy-four candidates down to twenty-six.

    The facet's question is omitted when there is none: a facet settled without
    a call keeps the raw candidate, which carries no question, and a rendered
    label with nothing behind it reads as a question the facet failed to state.
    """
    question_line = (f"\nThe question this facet answers: {facet_question}"
                     if facet_question else "")
    return f"""You are a taxonomy consolidation specialist for surveys.
 Your task is to organize group of attributes into a minimal set within a given facet.
 
{build_context_block(
    language=language, survey_question=survey_question, sector=sector,
    entity=entity, topic=topic, perspective=perspective, intent=intent)}

{build_facets_attributes_block(dimension=dimension)}

You are working inside this facet:

<taxonomy_facet>
Facet: {facet_name} — {facet_definition}{question_line}
</taxonomy_facet>

Here are the attributes you need to organize into a minimal set:
<candidates>
{candidate_block}
</candidates>

# Rules

1) The set of attributes need to be MECE; mutually exclusive and collectively exhaustive. This means that the attributes are not allowed to overlap semantically or meaningfull in light of the survey question. And this means that the set of attributes should provide full coverage for all attributes.

2) You need to find the minimal number of facets to organize the attribute groups by beging MECE. The fewer attributes by achieving MECE, the better.


{UNIVERSAL_RULES}

{INSTRUCTOR_HINT}"""