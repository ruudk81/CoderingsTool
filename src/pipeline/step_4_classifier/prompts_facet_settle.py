"""Settling a domain's facets, with real placement in front of them.

Facets are first settled before any idea has been assigned — see
`prompts_consolidation.py` — so the only prevalence signal available there is
how many chunks proposed a name. That signal points the wrong way: a concept
proposed under five different wordings arrives as five one-pass candidates,
so exactly the cases that should merge look rarest. Assignment runs after
that and gives every candidate its real ideas.

This module re-judges a domain's facets once those ideas are in: how many
responses each facet actually holds, its share of the domain, and what its
attributes actually contain — not just what they claim to. A facet that
looked substantial by pass-count and turns out to hold almost nothing once
ideas are counted is exactly the case the first pass could not see.

**Two exits, both on ids handed out in the same block.** A facet can fold
into another (`source_facet_ids`), and a single attribute can move to a
better-fitting facet (`AttributeMove`) without its container being touched.
Both name their target as an id from `build_facet_settle_block`, never as a
name to be looked up afterwards — see `AttributeMove` for why that distinction
is load-bearing here.

**Attributes are evidence, not material.** Their names are rendered so the
model can judge whether a facet's question is actually being answered by what
sits under it; their definitions are not shown, because renaming or
redefining an attribute is not this phase's job — see
`prompts_consolidation.py`'s attribute-consolidation call for that.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

from pydantic import BaseModel, Field

from .drains import is_drain_item
from .prompts_shared import (
    INSTRUCTOR_HINT,
    UNIVERSAL_RULES,
    _extract_definition,
    build_context_block,
    build_taxonomy_block,
)

if TYPE_CHECKING:
    from pipeline.step_3_ideaExtractor.dimension_data import DimensionDefinition


# =============================================================================
# RESPONSE MODEL
# =============================================================================

class SettledFacetCard(BaseModel):
    """A facet after settling — folded or left standing, with real placement
    behind the decision instead of a name proposed once."""
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
            "The one question this facet answers about the responses, "
            "phrased as a question, in the survey language — one folded from "
            "several sources states the question itself, never that a merge "
            "took place. No two surviving facets may state the same one"))
    source_facet_ids: List[str] = Field(
        ..., description=(
            "The bracketed ids of every candidate facet that folds into this "
            "one, e.g. ['F1', 'F7']. One that survives unchanged lists just "
            "its own id. Name every facet id shown exactly once across all "
            "of these lists: one you leave out is not removed — it stays "
            "where it was, next to the facet you meant to replace it with"))


class AttributeMove(BaseModel):
    """One attribute relocated to a better-fitting facet, container untouched.

    Both `attribute_id` and `to_facet_id` are ids from `build_facet_settle_block`,
    never names. Refinement's own misfit exit named its destinations before the
    phase ran and resolved them against the names as they stood after it — on
    the run of 2026-08-16, 70% of routed groups landed on a name a neighbouring
    call had just consumed, and fell back silently to where they already sat.
    An id chosen from the input the model is looking at cannot go stale that
    way; resolving `to_facet_id` against the facet that finally claims it is
    the caller's job, done once the whole call is back.
    """
    attribute_id: str = Field(
        ..., description=(
            "The bracketed id, from the facets shown, of the attribute being "
            "relocated, e.g. 'A3'"))
    to_facet_id: str = Field(
        ..., description=(
            "The bracketed id of the candidate facet this attribute actually "
            "belongs under — the id as shown, in the same id space as "
            "`source_facet_ids`, not the name of the facet it ends up folded "
            "into"))


class FacetSettleResult(BaseModel):
    """What one facet-settle call per domain returns."""
    scratchpad: str = Field(
        ..., description=(
            "Work through the numbered rules of the prompt in the order they "
            "are given, before writing the output. The rules are not "
            "repeated here: two copies of them drifted apart once, and the "
            "model was handed both"))
    facets: List[SettledFacetCard] = Field(
        ..., description=(
            "The fewest mutually exclusive facets that cover this domain"))
    attribute_moves: List[AttributeMove] = Field(
        ..., description=(
            "Every attribute that answers a different candidate facet's "
            "question than the one it is shown under, redirected there. An "
            "attribute already sitting under the facet whose question it "
            "answers does not appear here"))


# =============================================================================
# BLOK
# =============================================================================

def build_facet_settle_block(
    facets: List[Dict[str, Any]],
    counts: Dict[str, int],
    shares: Dict[str, float],
    contents: Dict[str, List[str]],
    attribute_ids: Dict[str, str],
    top_n: int,
) -> str:
    """One domain's facets, each with its real size and what it actually holds.

    Attribute names are shown so the model can check them against the facet's
    own question — that is the whole test in rule 3 — but their definitions
    are not: this phase moves and folds, it does not redefine, and material to
    redefine with would only invite it to.

    `attribute_ids` is supplied rather than built here: it has to stay stable
    across every facet in the domain, since an `AttributeMove` names a
    destination facet, not the facet the attribute started under.
    """
    blocks = []
    for i, facet in enumerate(facets, 1):
        facet_id = f"F{i}"
        name = facet["facet_name"]
        tag = "  [CATCH-ALL]" if is_drain_item(facet) else ""
        lines = [f"[{facet_id}] {name} — {facet['facet_definition']}  "
                 f"{counts.get(name, 0)} responses "
                 f"({shares.get(name, 0.0):.0%} of this domain){tag}"]
        question = facet.get("facet_question")
        if question:
            lines.append(f"      Claims to answer: {question}")
        attributes = facet.get("attributes") or []
        if attributes:
            listed = ", ".join(
                f"[{attribute_ids.get(a['attribute_name'], '?')}] "
                f"{a['attribute_name']}" for a in attributes)
            lines.append(f"      Holds these attributes: {listed}")
        texts = (contents.get(name) or [])[:top_n]
        if texts:
            lines.append("      Actually holds:")
            lines.extend(f"        - {t}" for t in texts)
        else:
            lines.append("      Actually holds: (nothing was assigned to it)")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


# =============================================================================
# PROMPT
# =============================================================================

def build_facet_settle_prompt(
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
    settle_block: str,
) -> str:
    """Judgement over one domain's facets, on what they turned out to hold.

    The framing sentence states what a facet is in this dimension's own
    words — the same test rule 1 applies when it asks whether two facets
    answer the same underlying question.
    """
    rules = dimension.prompt_rules
    facet_definition = _extract_definition(rules.facet_instruction)

    return f"""You are a taxonomy consolidation specialist for surveys.
Your task is to settle this domain's facets against how the responses actually placed under them, not just how each was named. A facet, in this taxonomy, is: {facet_definition}

{build_context_block(
    language=language, survey_question=survey_question, sector=sector,
    entity=entity, topic=topic, perspective=perspective, intent=intent)}

{build_taxonomy_block(
    dimension=dimension, dimension_name=dimension_name,
    dimension_description=dimension_description)}

You are working inside this domain:
<taxonomy_domain>
{domain_label} — {domain_definition}
</taxonomy_domain>

Here are this domain's candidate facets, each with how much of the domain it
holds, the attributes currently placed under it, and what those attributes
actually contain:
<facet_placements>
{settle_block}
</facet_placements>

# Objective

Find the smallest set of facets for this domain that is mutually exclusive and
collectively exhaustive (MECE), given how the responses actually placed —
not how many candidates were originally proposed.

The optimization priority is:
- Correct facet membership
- MECE
- Minimum number of facets
- Interpretability
- Preservation of meaningful prevalent distinctions

Do not preserve a distinction merely because it appears in the input.

# Rules

1. Fold facets together that answer the same underlying question. The question is the test, not the name.
2. A facet holding a small share of its domain relative to its neighbours here belongs with the facet whose question it shares. Judge "small" against the other facets shown here, never against a fixed percentage.
3. An attribute that does not answer its own facet's question, but does answer another facet's question in this domain, moves there.
4. Do not rename or redefine attributes — that layer has not had its turn yet.
5. Facets marked [CATCH-ALL] take no part: do not fold one, do not rename one, and move nothing into one. A catch-all is an offer, not a category.
6. Every surviving facet writes its own name, definition and the question it answers — even one with a single source.

{UNIVERSAL_RULES}

{INSTRUCTOR_HINT}"""
