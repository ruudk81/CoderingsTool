"""Refinement inside one facet, and one round across the scope boundaries.

Consolidation judged on the observations each candidate came from; refinement is
the first phase with **real counts, shares and response texts** in front of it.
That makes it the only place where it becomes visible that a bucket holds
something other than its name promises.

**The scope is the facet.** One call is one facet, and the facet is therefore
fixed: an attribute cannot be moved out of it, and `RefinedAttribute` carries no
facet to move it to. What the narrow scope buys is the same thing attribute
consolidation bought by splitting off: the judgement gets a pool it can hold at
once, with nothing else competing for attention. The share stays measured over
the DOMAIN all the same — see `build_contents_block`.

What it costs is reach, and one exit is widened to pay for it. A group of
responses may belong under an attribute in a NEIGHBOURING facet, and with only
this facet in view the model's only remaining exit would be "out" — the verdict
reserved for text with no substance at all. `build_move_targets_block` therefore
renders the rest of the domain by name and definition only, as a destination list
and not as material to judge. Scope of judgement is the facet; scope of
destination is the domain.

**Splitting is gone.** The clause once read *"an attribute holding a large share
AND visibly diverse contents is too abstract: SPLIT it, do not widen it"*,
written when items were narrow and numerous; since discovery makes them
deliberately broad it fired constantly. Measured effect: consolidation brought
102 attributes back, refinement inflated them to 126, cross-scope removed 23
again — net standstill. It was tightened, and then dropped with the rewritten
rules. This phase only merges now, and `RefinedAttribute` carries neither an
action nor the texts a split would have had to route by.

Catch-alls take no part. They are not merged, split, moved or widened — a
catch-all is an offer, not a category, and judging it as if it were one turns it
into one after all.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional

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
# RESPONSE MODEL — REFINEMENT
# =============================================================================

class RefinedAttribute(BaseModel):
    """One attribute as refinement leaves it.

    It states no action. What happened is readable from `source_attributes`:
    more than one source is a merge, one source under another name a rename,
    one source under its own name an attribute left alone. Asking the model to
    label that as well made it classify what the code can already see, and the
    labels outlived the rules that explained when each applied.
    """
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
            "2-3 representative response texts, copied exactly from the "
            "contents shown"))
    source_attributes: List[str] = Field(
        ..., description=(
            "The exact names of the attributes shown above that this one "
            "replaces. One you leave unchanged lists just its own name. Name "
            "every attribute shown exactly once across all of these lists: one "
            "you leave out is not removed — it stays where it was, next to the "
            "attribute you meant to replace it with"))


class RefinementMisfitGroup(BaseModel):
    """A group of responses sitting in the wrong place, with where it belongs."""
    verdict: Literal["move", "out"] = Field(
        ..., description=(
            "'move' when the group belongs to another attribute in this "
            "domain, 'out' when it carries no substantive content at all"))
    target_attribute: Optional[str] = Field(
        default=None,
        description=(
            "For 'move' only: the exact name of the attribute these responses "
            "belong to, as shown in this domain"))
    instance_texts: List[str] = Field(
        ..., description=(
            "The exact response texts in this group, copied from the contents "
            "shown. Never counts, paraphrases or summaries"))


class RefinementResult(BaseModel):
    """What one refinement call per facet returns.

    The scratchpad walks the prompt's own rules in order, and ends on the
    misfits deliberately. Route a group first and merge afterwards, and the
    attribute it was routed to can disappear under the model's own hand:
    `_apply_refinement` resolves a misfit target against the attributes that
    still exist, and a target that no longer does falls back to where the
    responses already sat — no error, no log line, the move simply gone.
    """
    scratchpad: str = Field(
        ..., description=(
            "Work through this before writing the output. "
            "1. Minimize the number of containers. Merge candidate attributes whenever they can be represented by one broader, meaningful attribute without losing an important distinction for the survey question. When in doubt, prefer merging. "
            "2. Keep a distinction only when it is substantively meaningful and clearly codable. Differences in wording, synonyms, closely related meanings, or broad-versus-narrow versions of the same idea normally belong in the same container. "
            "3. The final attributes must be MECE. Each substantive idea should have one natural home, and together the attributes must cover all substantive material belonging to this facet. Avoid overlapping attributes and parent/child attributes alongside each other. "
            "4. Use prevalence to simplify. Small or low-prevalence distinctions should normally be absorbed into the nearest broader attribute rather than becoming separate attributes, provided the resulting container remains semantically coherent. "
            "5. Remove misfits. Material that does not belong in this facet should be moved to the best matching attribute in <move_targets>, or marked as carrying no substantive content. "
            "6. Before returning the result, ask one final question: Can any two remaining attributes still be merged without losing an important, clearly codable distinction? If yes, merge them. "
            "7. Check every attribute shown appears in exactly one source_attributes list"))
    attributes: List[RefinedAttribute] = Field(
        ..., description=(
            "The fewest mutually exclusive attributes that cover what this "
            "facet holds"))
    misfits: List[RefinementMisfitGroup] = Field(
        default_factory=list,
        description="Groups of responses that belong elsewhere or nowhere")


# =============================================================================
# BLOKKEN
# =============================================================================

def build_contents_block(
    attributes: List[Dict[str, Any]],
    contents: Dict[str, List[str]],
    shares: Dict[str, float],
    counts: Dict[str, int],
    top_n: int,
) -> str:
    """What each attribute of ONE facet actually holds, with its size.

    The share is the share of the DOMAIN, deliberately wider than the call. A
    share of the facet would make every facet look equally weighty — the biggest
    attribute of a facet holding a handful of responses would read the same as
    the biggest of one holding half the domain, and granularity would then be set
    on a scale that shifts per call. The domain is the one denominator every
    facet of it shares.

    No threshold appears anywhere: one would have been read off a single dataset,
    and is therefore just as use-case-bound as an example lifted from client data.

    The facet itself is no longer a header here. It is the scope of the whole
    call and is rendered once, as context, by the prompt.
    """
    blocks = []
    for attribute in attributes:
          name = attribute["attribute_name"]
          tag = "  [CATCH-ALL]" if is_drain_item(attribute) else ""
          lines = [f"{name} — {counts.get(name, 0)} responses "
                   f"({shares.get(name, 0.0):.0%} of this domain){tag}",
                   f"    Claims to capture: {attribute['attribute_definition']}"]
          texts = (contents.get(name) or [])[:top_n]
          if texts:
              lines.append("    Actually holds:")
              lines.extend(f"      - {t}" for t in texts)
          else:
              lines.append("    Actually holds: (nothing was assigned to it)")
          blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def build_move_targets_block(
    facets: List[Dict[str, Any]],
    facet_index: int,
) -> str:
    """Where a misfit group may go: the rest of the domain, names only.

    A destination list, not material to judge. Counts, shares and contents are
    deliberately absent — with them the model would start weighing these
    attributes too, which is the next facet's call and not this one.

    Excluded by POSITION, not by name: a domain may hold two facets with the
    same name, and excluding by name would hide the neighbour's attributes as
    well as this facet's own.
    """
    blocks = []
    for index, facet in enumerate(facets):
        if index == facet_index or is_drain_item(facet):
            continue
        names = [a for a in (facet.get("attributes") or [])
                 if not is_drain_item(a)]
        if not names:
            continue
        lines = [f"{facet['facet_name']}"]
        lines.extend(f"  {a['attribute_name']} — {a['attribute_definition']}"
                     for a in names)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) if blocks else "(this domain holds no other facet)"


# =============================================================================
# PROMPT — REFINEMENT
# =============================================================================

def build_refinement_prompt(
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
    facet_question: str,
    contents_block: str,
    move_targets_block: str,
) -> str:
    """Judgement over one facet, on what its buckets actually hold.

    The domain comes along as parent context — an attribute is judged against
    the question its facet answers, and that question only means something
    inside the domain it divides. The facet's own question is rendered when
    there is one; a facet settled without a call has none, and a rendered label
    with nothing behind it reads as a question the facet failed to state.
    """
    rules = dimension.prompt_rules
    attribute_definition = _extract_definition(rules.attribute_instruction)
    question_line = (f"\nThe question it answers: {facet_question}"
                     if facet_question else "")

    return f"""You are a taxonomy consolidation specialist for surveys.
Your task is to organize the candidate attributes within one facet into the smallest possible set of meaningful attribute-containers that is MECE.
Default toward consolidation. A distinction should survive only when keeping it separate is necessary to preserve meaningful semantic differences in the context of the survey question.

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

This is the facet you need to consolidate:
<facet_contents>
{facet_name} — {facet_definition}{question_line}

{contents_block}
</facet_contents>

These are the attributes of the other facets in this domain. They are a destination list,
not material to judge: a group of responses that belongs under one of them is a misfit you
send there, and you return nothing about these attributes themselves.
<move_targets>
{move_targets_block}
</move_targets>

Rules
1. Minimize the number of containers. Merge candidate attributes whenever they can be represented by one broader, meaningful attribute without losing an important distinction for the survey question. When in doubt, prefer merging.
2. Keep a distinction only when it is substantively meaningful and clearly codable. Differences in wording, synonyms, closely related meanings, or broad-versus-narrow versions of the same idea normally belong in the same container.
3. The final attributes must be MECE. Each substantive idea should have one natural home, and together the attributes must cover all substantive material belonging to this facet. Avoid overlapping attributes and parent/child attributes alongside each other.
4. Use prevalence to simplify. Small or low-prevalence distinctions should normally be absorbed into the nearest broader attribute rather than becoming separate attributes, provided the resulting container remains semantically coherent.
5. Remove misfits. Material that does not belong in this facet should be moved to the best matching attribute in <move_targets>, or marked as carrying no substantive content.

Before returning the result, ask one final question:
"Can any two remaining attributes still be merged without losing an important, clearly codable distinction?"
If yes, merge them.

{UNIVERSAL_RULES}

{INSTRUCTOR_HINT}"""


# =============================================================================
# PROMPT — CROSS-DOMAIN
# =============================================================================

def build_cross_domain_prompt(
    *,
    language: str,
    survey_question: str,
    sector: str,
    entity: str,
    topic: str,
    perspective: str,
    intent: str,
    inventory_block: str,
) -> str:
    """The only phase that sees more than one domain at a time.

    Every phase before this one is scope-bound, and that is no accident: per-idea
    (domain, facet) is a projection of where the attribute lives, so a structure
    merge across a domain boundary drags every idea in that bucket along with
    it. Which is exactly why it is allowed here and nowhere else — with dozens
    of scopes each settled on its own, the same concept survives in several
    places, and no other phase can see that.

    Works on ids, never on names: the model returns groups as `source_ids` plus a
    `home_id`, and the surviving attribute inherits both the domain and the facet
    of that home. Relocation is thereby a choice among the inputs rather than free
    text that has to be matched back, and a forgotten id is detectable instead of
    silent.
    """
    return f"""You are a taxonomy consolidation specialist for surveys.
Each domain settled its attributes on its own, without seeing the others. The same concept
can therefore survive in several domains under different names, and nothing so far has been
able to notice that. This is the one step that looks across all of them at once.

{build_context_block(
    language=language, survey_question=survey_question, sector=sector,
    entity=entity, topic=topic, perspective=perspective, intent=intent)}

Here is every attribute in the study, grouped by the domain and facet it currently sits in.
Each carries an id and the number of responses it holds:

<inventory>
{inventory_block}
</inventory>

# Your task

Find the attributes that mean the same thing across different domains or facets, and fold
each such group into one.

- Group only what genuinely refers to the same thing. Two attributes that merely sound alike,
  or that answer different questions about different subjects, stay apart.
- For each group, pick the `home_id`: the id whose domain and facet the survivor keeps.
  Choose the scope where most of these responses already sit.
- The survivor's responses are all the responses of its group. They move to the home scope,
  and that is intended — this is the only step where structure is allowed to relocate
  across domains.
- Most attributes belong to no group. An attribute that stays exactly where it is returns as
  a group of one, listing only its own id.

Leave the catch-all attributes alone. They are per-domain offers, not concepts, and folding
two of them together would merge two different domains' residuals into one meaningless bucket.
Return each of them as a group of one.

{UNIVERSAL_RULES}

{INSTRUCTOR_HINT}"""
