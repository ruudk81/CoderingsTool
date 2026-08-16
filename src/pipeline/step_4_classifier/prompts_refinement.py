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

**Routing is gone with it** (2026-08-16). The facet once carried a misfit exit:
a group of responses could be sent to an attribute in a neighbouring facet, with
the rest of the domain rendered as a destination list. It could not work as
built. The destinations were rendered from the structure as it stood *before*
the phase, while `_apply_refinement` resolved them against the names as they
stood *after* — and renaming is the main thing this phase does. Measured on the
run of 2026-08-16: 83 of 118 routed texts (70%) named a destination that its own
neighbour's call had just consumed, and fell back silently to where they already
sat. Removing the exit rather than repairing it keeps one rule per scope: this
phase merges what one facet holds, and nothing else. Placement across facets is
cross-domain's job.

**Splitting is gone.** The clause once read *"an attribute holding a large share
AND visibly diverse contents is too abstract: SPLIT it, do not widen it"*,
written when items were narrow and numerous; since discovery makes them
deliberately broad it fired constantly. Measured effect: consolidation brought
102 attributes back, refinement inflated them to 126, cross-scope removed 23
again — net standstill. It was tightened, and then dropped with the rewritten
rules. This phase only merges now, and `RefinedAttribute` carries neither an
action nor the texts a split would have had to route by.

Catch-alls take no part. They are not merged, renamed or absorbed — a catch-all
is an offer, not a category, and judging it as if it were one turns it into one
after all. That is stated as a rule *and* enforced in `_apply_refinement`: the
marker was rendered for a while with no rule left to explain it, and a
catch-all named as a SOURCE then had its ideas remapped into a content
attribute, which the drain card surviving in the structure hid completely.
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


class RefinementResult(BaseModel):
    """What one refinement call per facet returns.

    Merging is all it returns. The misfit exit that once sat beside it is gone
    — see the module docstring — so there is no ordering left to get wrong
    between routing a group and merging the attribute it was routed to.
    """
    scratchpad: str = Field(
        ..., description=(
            "Work through the numbered rules of the prompt in the order they "
            "are given, before writing the output. The rules are not repeated "
            "here: two copies of them drifted apart once, and the model was "
            "handed both"))
    attributes: List[RefinedAttribute] = Field(
        ..., description=(
            "The fewest mutually exclusive attributes that cover what this "
            "facet holds"))


# =============================================================================
# BLOKKEN
# =============================================================================

def build_contents_block(
    attributes: List[Dict[str, Any]],
    contents: Dict[str, List[str]],
    shares: Dict[str, float],
    counts: Dict[str, int],
    top_n: int,
    facet_total: int,
    domain_total: int,
) -> str:
    """What each attribute of ONE facet actually holds, with its size.

    The share is the share of the DOMAIN, deliberately wider than the call. A
    share of the facet would make every facet look equally weighty — the biggest
    attribute of a facet holding a handful of responses would read the same as
    the biggest of one holding half the domain, and granularity would then be set
    on a scale that shifts per call. The domain is the one denominator every
    facet of it shares.

    That one denominator needs a second number beside it, and the header carries
    it: how much of the domain this whole facet holds. Without it "small" has no
    calibration inside the call — a facet holding a twentieth of its domain shows
    every one of its attributes as a low percentage, and a prevalence rule then
    reads as "absorb everything". Both scales are visible; neither is a
    threshold.

    No threshold appears anywhere: one would have been read off a single dataset,
    and is therefore just as use-case-bound as an example lifted from client data.

    The facet itself is no longer a header here. It is the scope of the whole
    call and is rendered once, as context, by the prompt.
    """
    share = (facet_total / domain_total) if domain_total else 0.0
    blocks = [f"This facet holds {facet_total} of the {domain_total} responses "
              f"in its domain ({share:.0%}). The percentage on each attribute "
              f"below is its share of the DOMAIN, not of this facet."]
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

# Objective

Find the minimum number of attribute-containers required to organize all substantive candidate material belonging to this facet while remaining mutually exclusive and collectively exhaustive (MECE).

The optimization priority is:
- MECE
- Minimum number of containers
- Interpretability
- Preservation of meaningful prevalent distinctions

Do not preserve a distinction merely because it appears in the input.

# Rules

Apply these rules:

1. Minimize the number of containers. Merge candidate attributes whenever they can be represented by one broader, meaningful attribute without losing an important distinction for the survey question. When in doubt, prefer merging.
2. Keep a distinction only when it is substantively meaningful and clearly codable. Differences in wording, synonyms, closely related meanings, or broad-versus-narrow versions of the same idea normally belong in the same container.
3. The final attributes must be MECE. Each substantive idea should have one natural home, and together the attributes must cover all substantive material belonging to this facet. Avoid overlapping attributes and parent/child attributes alongside each other.
4. Use prevalence to simplify. Absorb a distinction that is far less prevalent than the others in this facet into the nearest broader attribute, provided the resulting container remains semantically coherent. Judge "far less" against the other attributes shown here, never against a fixed percentage: the share is measured over the whole domain, so in a facet that holds a small part of its domain every attribute shows a low percentage.
5. Attributes marked [CATCH-ALL] take no part. Do not merge, rename or absorb one, and never name one as a source. A catch-all is an offer, not a category, and treating it as one turns it into one.
6. Before returning the result, ask one final question: Can any two remaining attributes still be merged without losing an important, clearly codable distinction? If yes, merge them.

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
