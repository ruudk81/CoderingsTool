"""Settling a domain's facets for step 4"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Literal

from pydantic import BaseModel, Field, create_model

from .prompts_shared import (
    INSTRUCTOR_HINT,
    UNIVERSAL_RULES,
    _extract_definition,
    build_context_block,
    build_taxonomy_block_L3,
)

if TYPE_CHECKING:
    from pipeline.step_3_ideaExtractor.dimension_data import DimensionDefinition


# =============================================================================
# RESPONSE MODEL
# =============================================================================

class SettledFacetCard(BaseModel):
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
            "The one question this facet answers about the material under it, "
            "phrased as a question, in the survey language — one folded from "
            "several sources states the question itself, never that a merge "
            "took place. No two surviving facets may state the same one"))
    source_facet_ids: List[str] = Field(
        ..., description=(
            "The bracketed ids of every candidate facet that folds into this "
            "one, e.g. ['F1', 'F7']. One that survives unchanged lists just "
            "its own id"))


def build_facet_settle_model(facet_ids: List[str], attribute_ids: List[str]):
    """Runtime model in which a move's destination is this call's own id space."""

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
                "Work through the prompt's decision rules in the order they "
                "are given, before writing the output. The rules are "
                "not repeated here: two copies of them drifted apart once, "
                "and the model was handed both"))),
        decision_summary=(List[str], Field(
            ..., description=(
                "One short line per decision that took judgement, each stating "
                "what was done and why. Include here the candidate distinctions "
                "that are theoretically plausible but that the attributes do "
                "not support. Not a line for every facet: only the calls a "
                "reader would want to check"))),
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
) -> str:
    """One domain's facets, each with its real size and the attributes it holds.

    The response texts this call used to show are gone. What makes the phase
    empirical is that its numbers come from the assignment that actually ran,
    and those are still here; the attributes are the qualitative half, and they
    are still unconsolidated, which is exactly what shows whether two facets are
    being used interchangeably.

    Attributes carry their definitions, and no examples. Names alone cannot show
    that two facets use the same underlying concepts — the test this call's core
    principle actually asks for. Examples are the layer that went too far once:
    facet consolidation renders names only, because a predecessor given the full
    attribute cards started settling the attributes as well. That cannot happen
    here — `attribute_moves` can relocate an attribute and nothing else, so the
    schema forbids what a prompt rule had to forbid there.

    The size sits on the id line. This phase exists because it judges on the
    counts of a real assignment instead of on how many chunks proposed a name,
    so that number belongs where a facet is introduced, not somewhere below it.
    """
    blocks = []
    for i, facet in enumerate(facets, 1):
        facet_id = f"F{i}"
        lines = [f"[{facet_id}] {facet['facet_name']} — "
                 f"{counts.get(facet_id, 0)} responses "
                 f"({shares.get(facet_id, 0.0):.0%} of this domain)",
                 f"Definition: {facet['facet_definition']}"]
        question = facet.get("facet_question")
        if question:
            lines.append(f"Claims to answer: {question}")
        attributes = facet.get("attributes") or []
        if attributes:
            lines.append("Holds these attributes:")
            lines.extend(
                f"- [{a['attribute_id']}] {a['attribute_name']}: "
                f"{a.get('attribute_definition') or ''}".rstrip(": ")
                for a in attributes)
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
    rules = dimension.prompt_rules
    facet_definition = _extract_definition(rules.facet_instruction)

    return f"""You are a taxonomy consolidation specialist for surveys.
Your task is to settle this domain's facets based on how the attributes are actually organized under them, not just on how the candidate facets were originally named or defined.

# Taxonomy structure 

{build_taxonomy_block_L3(
    dimension=dimension, dimension_name=dimension_name,
    dimension_description=dimension_description)}

A facet, in this taxonomy, is: {facet_definition}
For this task, “independently analyzable” means independently distinguishable in the observed attributes, not merely theoretically distinguishable from the candidate definitions. 
A distinction that exists only in the wording or definitions of the candidate facets should not be preserved if their attributes represent the same underlying type of content.

# Survey context  

{build_context_block(
    language=language,
    dimension_name=dimension_name, dimension_description=dimension_description,
    survey_question=survey_question,
    sector=sector, entity=entity, topic=topic, perspective=perspective, intent=intent)}

You are working inside this domain:
<taxonomy_domain>
{domain_label} — {domain_definition}
</taxonomy_domain>

Here are this domain's candidate facets, each with how much of the domain it holds, the attributes currently placed under it, and what those attributes actually contain:
<facet_placements>
{settle_block}
</facet_placements>

# Objective

Find the smallest set of facets that organizes the attributes in this domain in a mutually exclusive and collectively exhaustive (MECE) way.

Treat the candidate facets as provisional hypotheses. The final facet structure may retain, merge or reorganize them, and attributes may be reassigned between surviving facets.

Optimize in this order:

1. Correct attribute membership
2. MECE
3. Minimum number of facets
4. Interpretability
5. Preservation of distinctions that are clearly and consistently represented by the attributes

# Core decision principle

Preserve a facet distinction only when the attributes support a clear, stable and independently recognizable boundary between different types of content.

Do not preserve a distinction merely because it:
- appears in the candidate taxonomy;
- has a theoretically distinct definition;
- contains many attributes;
- has high prevalence; or
- can be conceptually justified.

If two candidate facets are theoretically different but their attributes do not support a reliable substantive distinction, merge them.

Where possible, preserve narrower substantive differences at the attribute level rather than creating additional facets.

# Procedure

## 1. Identify attribute clusters

Temporarily set aside the candidate facet boundaries and examine the attributes themselves.

Identify the smallest number of recurring semantic groups needed to organize them coherently.

Ask:
- What type of content does each attribute represent?
- Which attributes belong to the same underlying analytical category?
- Which groups of attributes are meaningfully distinguishable from one another?
- Which apparent distinctions arise mainly from the existing candidate facet structure rather than from meaningful differences between the attributes?

## 2. Validate the facet structure

Compare these attribute groups with the candidate facets.

For each candidate distinction, decide whether to:
- retain it;
- merge it with another facet;
- move misplaced attributes to another surviving facet; or
- otherwise reorganize the structure to obtain the smallest MECE set of facets supported by the attributes.

# Decision rules

1. **Same underlying question → merge.**
   Merge facets when their attributes answer the same underlying analytical question.
   Test the question represented by the attributes, not the candidate facet names.

2. **Clear attribute boundary → keep separate.**
   Keep facets separate only when a competent coder could reliably decide which facet an attribute belongs to using a clear substantive rule.

3. **Misplaced attributes → move.**
   If an attribute does not answer its current facet's question but clearly answers another candidate facet's question in this domain, move it there. Name that candidate by its [F#] id, even when it folds into a larger facet: the surviving facets have no ids of their own.

4. **Do not consolidate attributes yet.**
   Do not rename, redefine, merge or split attributes.
   Attribute consolidation happens at a later stage.

5. **Use prevalence only as supporting evidence.**
   Prevalence may help judge whether a distinction is substantively important, but it never determines the facet structure by itself.
   A low-prevalence but clearly distinct type of attribute may warrant its own facet; a high-prevalence but overlapping type does not.

6. **Every surviving facet must stand on its own.**
   Give every surviving facet its own name, definition and facet question, including facets that survive unchanged.

7. **Every attribute must have one natural home.**
   The final facets must form a partition of the attributes in this domain.
   Each attribute must end up under exactly one surviving facet and should not require arbitrary judgment between several.

8. **Account for every candidate facet.**
   Name every [F#] id shown exactly once across all `source_facet_ids` lists. One you leave out is not removed — it stays where it was, next to the facet you meant to replace it with.

# Evidence for decisions

For every merge or retention decision, base the judgment on:

- the semantic content represented by the attributes;
- whether the attributes answer the same or different underlying analytical questions;
- the clarity of the boundary between the proposed facets; and
- whether each resulting facet represents an independently recognizable type of attribute.

Explicitly identify candidate distinctions that are theoretically plausible but are not sufficiently supported by meaningful differences between the attributes.
{UNIVERSAL_RULES}

{INSTRUCTOR_HINT}"""
