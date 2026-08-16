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

**Two exits, both on ids handed out in the same block, but they are not the
same kind of claim.** A facet can fold into another, via `source_facet_ids` —
a claim a coverage gate recomputes and can catch after the call. A single
attribute can move to a better-fitting facet without its container being
touched — a destination, which nothing recomputes. `build_facet_settle_model`
types the move as a `Literal` over exactly the ids this call handed out, so an
invented destination is a schema error instructor retries, not a content error
that looks identical to a facet legitimately merged away by the same call —
see that function for why the difference matters.

**Attributes are evidence, not material.** Their names are rendered so the
model can judge whether a facet's question is actually being answered by what
sits under it; their definitions are not shown, because renaming or
redefining an attribute is not this phase's job — see
`prompts_consolidation.py`'s attribute-consolidation call for that.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Literal

from pydantic import BaseModel, Field, create_model

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


def build_facet_settle_model(facet_ids: List[str], attribute_ids: List[str]):
    """Runtime model in which a move's destination is this call's own id space.

    `source_facet_ids`, on `SettledFacetCard` above, is a claim: a coverage
    gate recomputes what every candidate id was accounted for by, and an
    invented id there is caught and logged as `unknown_source_id`. A move has
    no such gate — nothing recomputes whether `to_facet_id` was ever real.
    And the wiring that resolves a move logs `move_target_gone` for the
    legitimate case where the target facet was merged away by this same call;
    an invented id would produce the identical log line, making a model error
    indistinguishable from a normal outcome. That is the exact blurring that
    let refinement's own misfit exit lose 70% of its routed groups silently
    on the run of 2026-08-16, before anyone noticed. Typing both id fields
    `Literal` over exactly the ids this call handed out turns the invented
    case into a schema error instructor retries, instead of a content error
    reaching that log line at all.

    `attribute_ids` can be empty — every facet built by facet consolidation
    carries at least one attribute in practice, but a task built against
    stubbed pools has none to hand out. `Literal` over an empty tuple is not
    itself a valid type, and there is nothing a move could legitimately name
    when no attribute is shown anyway — so rather than relax the field to
    something like `List[Any]` (which would accept any answer, including a
    real-looking move that then crashes uncaught on `move.attribute_id`
    instead of failing the schema the way an invented id does everywhere
    else), the list itself is capped at zero. A non-empty answer is then the
    same schema error instructor retries as an invented id would be — the
    protection stays in force in this degenerate case instead of being
    quietly suspended.
    """
    facet_literal = Literal[tuple(facet_ids)]  # type: ignore[valid-type]

    if attribute_ids:
        attribute_literal = Literal[tuple(attribute_ids)]  # type: ignore[valid-type]
        move = create_model(
            "AttributeMove",
            attribute_id=(attribute_literal, Field(
                ..., description=(
                    "The bracketed id, from the facets shown, of the "
                    "attribute being relocated, e.g. 'A3'"))),
            to_facet_id=(facet_literal, Field(
                ..., description=(
                    "The bracketed id of the candidate facet this attribute "
                    "actually belongs under — the id as shown, in the same "
                    "id space as `source_facet_ids`, not the name of the "
                    "facet it ends up folded into"))),
        )
        moves_field = (List[move], Field(
            ..., description=(
                "Every attribute that answers a different candidate facet's "
                "question than the one it is shown under, redirected there. "
                "An attribute already sitting under the facet whose "
                "question it answers does not appear here")))
    else:
        moves_field = (List[str], Field(
            default_factory=list, max_length=0, description=(
                "No attributes are shown in this call — this must stay "
                "empty, there is nothing to move")))

    return create_model(
        "FacetSettleResult",
        scratchpad=(str, Field(
            ..., description=(
                "Work through the numbered rules of the prompt in the order "
                "they are given, before writing the output. The rules are "
                "not repeated here: two copies of them drifted apart once, "
                "and the model was handed both"))),
        facets=(List[SettledFacetCard], Field(
            ..., description=(
                "The fewest mutually exclusive facets that cover this "
                "domain"))),
        attribute_moves=moves_field,
    )


# =============================================================================
# BLOK
# =============================================================================

def build_facet_settle_block(
    facets: List[Dict[str, Any]],
    counts: Dict[str, int],
    shares: Dict[str, float],
    contents: Dict[str, List[str]],
    top_n: int,
) -> str:
    """One domain's facets, each with its real size and what it actually holds.

    Keyed on id everywhere, never on name: `build_facet_menu` guarantees two
    facets of one domain may legally share a name, and facet consolidation
    can produce exactly that. A name-keyed `counts`/`shares`/`contents` would
    let one of them's numbers silently stand in for the other's — corrupting
    the one input this whole phase exists to get right. `counts`/`shares`/
    `contents` are keyed on the same `F#` id this function assigns below, in
    the same order `facets` is handed in.

    Each attribute dict already carries its own `attribute_id` (assigned once,
    by object identity, where the task was built) rather than being looked up
    here by name — the same reason: two attributes of one domain may share a
    name, and a name-keyed lookup would render one under the other's id.

    Attribute names are shown so the model can check them against the facet's
    own question — that is the whole test in rule 3 — but their definitions
    are not: this phase moves and folds, it does not redefine, and material to
    redefine with would only invite it to.
    """
    blocks = []
    for i, facet in enumerate(facets, 1):
        facet_id = f"F{i}"
        name = facet["facet_name"]
        lines = [f"[{facet_id}] {name} — {facet['facet_definition']}  "
                 f"{counts.get(facet_id, 0)} responses "
                 f"({shares.get(facet_id, 0.0):.0%} of this domain)"]
        question = facet.get("facet_question")
        if question:
            lines.append(f"      Claims to answer: {question}")
        attributes = facet.get("attributes") or []
        if attributes:
            listed = ", ".join(
                f"[{a['attribute_id']}] {a['attribute_name']}"
                for a in attributes)
            lines.append(f"      Holds these attributes: {listed}")
        texts = (contents.get(facet_id) or [])[:top_n]
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
5. Every surviving facet writes its own name, definition and the question it answers — even one with a single source.

{UNIVERSAL_RULES}

{INSTRUCTOR_HINT}"""
