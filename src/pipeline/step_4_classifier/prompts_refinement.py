"""Refinement inside the domain, and one round across the domain boundaries.

Consolidation judged on the observations each candidate came from; refinement is
the first phase with **real counts, shares and response texts** in front of it.
That makes it the only place where it becomes visible that a bucket holds
something other than its name promises.

Two things differ from the previous design:

**The scope is the domain, not the facet.** That leaves the facet **not** fixed:
an attribute may move to another facet within its domain. The domain stays the
same, and `facet_assignments` follows where the attribute lives, so such a move
relabels exactly the ideas it concerns. Per facet that was impossible — there
only the ideas could move, never the placement.

**The split clause has been rewritten.** It read: *"an attribute holding a large
share AND visibly diverse contents is too abstract: SPLIT it, do not widen it"*.
That was written when items were narrow and numerous; since discovery makes them
deliberately broad it fired constantly. Measured effect: consolidation brought
102 attributes back, refinement inflated them to 126, cross-scope removed 23
again. Net standstill. Splitting is now allowed only when the contents show two
distinct values on the same property that no single honest name covers.

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
    """One attribute as refinement leaves it."""
    action: Literal["keep", "merge", "widen", "split"] = Field(
        ..., description=(
            "What was done: 'keep' unchanged, 'merge' several inputs into this "
            "one, 'widen' the definition to honestly cover the contents, or "
            "'split' one input into distinct children"))
    facet_name: str = Field(
        ..., description=(
            "The facet this attribute belongs to. Must be one of the facets "
            "shown in this domain. Naming a different one than the input had "
            "moves the attribute, and its responses move with it"))
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
            "Every input attribute name that feeds this one. An attribute kept "
            "unchanged lists just itself"))
    instance_texts: List[str] = Field(
        default_factory=list,
        description=(
            "For action 'split' only: the exact response texts routed to this "
            "child. Empty for every other action"))


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
    """Wat één naslijpcall per domein teruggeeft."""
    scratchpad: str = Field(
        ..., description=(
            "Step-by-step reasoning before the final output: "
            "(1) read each attribute's contents against its label, "
            "(2) group the attributes by the question each one answers, "
            "(3) let prevalence set granularity within one question, "
            "(4) route what does not fit through one of the five exits, "
            "(5) check every label states a value a layperson can picture, "
            "(6) check every input attribute ends up in exactly one place"))
    attributes: List[RefinedAttribute] = Field(
        ..., description="Every attribute that survives in this domain")
    misfits: List[RefinementMisfitGroup] = Field(
        default_factory=list,
        description="Groups of responses that belong elsewhere or nowhere")


# =============================================================================
# BLOKKEN
# =============================================================================

def build_contents_block(
    facets: List[Dict[str, Any]],
    contents: Dict[str, List[str]],
    shares: Dict[str, float],
    counts: Dict[str, int],
    top_n: int,
) -> str:
    """What each attribute actually holds, per facet, with its size.

    Shares are shown so the model can weigh size relative to the neighbours. No
    threshold appears anywhere: one would have been read off a single dataset,
    and is therefore just as use-case-bound as an example lifted from client data.
    """
    blocks = []
    for facet in facets:
        facet_tag = "  [CATCH-ALL]" if is_drain_item(facet) else ""
        lines = [f"Facet: {facet['facet_name']} — "
                 f"{facet['facet_definition']}{facet_tag}"]
        for attribute in facet.get("attributes") or []:
            name = attribute["attribute_name"]
            tag = "  [CATCH-ALL]" if is_drain_item(attribute) else ""
            lines.append(
                f"  {name} — {counts.get(name, 0)} responses "
                f"({shares.get(name, 0.0):.0%} of this domain){tag}")
            lines.append(f"      Claims to capture: {attribute['attribute_definition']}")
            texts = (contents.get(name) or [])[:top_n]
            if texts:
                lines.append("      Actually holds:")
                lines.extend(f"        - {t}" for t in texts)
            else:
                lines.append("      Actually holds: (nothing was assigned to it)")
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
    contents_block: str,
) -> str:
    """Oordeel over één domein, op wat de bakken werkelijk bevatten."""
    rules = dimension.prompt_rules
    attribute_definition = _extract_definition(rules.attribute_instruction)

    return f"""You are a taxonomy consolidation specialist for surveys.
Every response in this domain has now been assigned. Your task is to judge the result
against what the attributes actually hold, and to correct it.

This is the first time you see real counts, real shares and real response texts. Use them.

# What an attribute is

{attribute_definition}

{build_context_block(
    language=language, survey_question=survey_question, sector=sector,
    entity=entity, topic=topic, perspective=perspective, intent=intent)}

{build_taxonomy_block(
    dimension=dimension, dimension_name=dimension_name,
    dimension_description=dimension_description)}

You are working inside this one domain:

<taxonomy_domain>
{domain_label} — {domain_definition}
</taxonomy_domain>

Here are its facets and attributes, with their real size and their real contents:

<domain_contents>
{contents_block}
</domain_contents>

# Rules

Consolidation is the goal: do NOT keep every concept separate — group. But govern grouping
by these rules, in this order.

**1. UNDERLYING QUESTION FIRST (orthogonality — the guardrail).**
For each concept, work out which underlying question it answers about the responses.
- Concepts answering DIFFERENT questions are orthogonal: never merge them into one attribute.
- Mutually exclusive ANSWERS to the SAME question are also kept apart; merging opposite
  answers creates a container that says nothing.
- Do not keep attributes separate based only on the object being discussed when the same
  underlying answer applies. An object is not a question.

**2. PREVALENCE SETS GRANULARITY (within one question only).**
Each attribute shows its share of this domain. Judge size RELATIVE to its siblings, never
against a fixed number.
- The largest attributes keep their own identity — never dissolve a well-supported concept.
- Attributes far below their siblings are GROUPED, but only with neighbours answering the
  same question, into one attribute that still names what they share in plain language.
- Variants that differ only in evaluative direction collapse to ONE attribute; the direction
  is recorded separately as valence.
Prevalence decides how finely to split WITHIN one question; it never licenses merging ACROSS
questions.

**3. WHEN TO SPLIT, AND WHEN TO WIDEN INSTEAD.**
A large attribute holding varied contents is not by itself a problem — a broad attribute
that honestly names what it holds is a good attribute.
- SPLIT only when the contents show two distinct ANSWERS to the same question, and no single
  honest name covers both. The test is that you can name each part in plain language and a
  reader would place a response in one or the other without hesitating.
- Otherwise WIDEN: restate the definition so it honestly covers what is actually in there.
If you find yourself reaching for a name like "other X" or "various X" for one of the parts,
that part is not a distinct answer and the attribute should be widened, not split.

**4. PLAIN, MEANINGFUL LABELS.**
Name every surviving attribute in everyday language. Test: reading the label alone, and
knowing the survey question, a layperson knows which distinction is meant.

**5. THE DOMAIN IS FIXED, THE FACET IS NOT.**
Every attribute you return belongs to "{domain_label}". You cannot move one to another
domain or invent one that belongs elsewhere.
Within this domain you MAY move an attribute to a different facet by naming that facet in
`facet_name`. Do so when the attribute plainly answers a different question than the facet
it currently sits under. Its responses move with it — that is intended.
If a GROUP OF RESPONSES belongs to a different attribute, that is a misfit, not a move: the
responses move and the attribute stays.

**6. FIVE EXITS FOR WHAT DOES NOT FIT.**
Read what each attribute actually contains. Where the contents do not match the label,
choose per group:
- two attributes turn out to hold the same thing
    -> action "merge": return one attribute and list both in `source_attributes`
- the contents are varied but genuinely belong together
    -> action "widen": restate the definition so it covers what is there
- the contents hold two distinct answers to the same question (rule 3)
    -> action "split": name each child and list the EXACT response texts that go to it
- a group of responses points at ANOTHER attribute in this domain
    -> `misfits`, verdict "move": name the target attribute and the EXACT response texts
- a group carries NO SUBSTANTIVE CONTENT WHATSOEVER — a bare judgment or filler with
  nothing said about the subject
    -> `misfits`, verdict "out"
"out" is not an escape hatch for "this does not fit the attributes I chose". A text that
names something real about the subject HAS substance. Only content-free text goes out.
Moves and splits must be given as EXACT response texts copied from the contents above —
never as counts, paraphrases or summaries. Every decision has to be checkable against the
data.

**7. ONE INPUT, ONE DESTINATION — unless you route by text.**
Every attribute shown above must end up in exactly one returned attribute. To divide one
input over two returned attributes, use "split" and list the exact texts for each child in
`instance_texts`. Listing one source under two returned attributes without those texts is
not interpretable, and its responses will be left where they are.

**8. KEEP THE VALUES THAT ARE ACTUALLY THERE.**
Grouping is not discarding. If the contents hold two distinct answers, return two
attributes; merging them and sending the remainder "out" loses real responses.

**9. LEAVE THE CATCH-ALLS ALONE.**
Anything marked [CATCH-ALL] above is an offer, not a category: it exists so that every
response has a home. Do not merge, split, widen, rename or move it, and do not route misfits
into it. Return it exactly as it is. Judge only what is not marked.

**Precedence when rules conflict:** 1 (orthogonality) > 5 (the domain is fixed) >
2 (prevalence grouping) > 4 (label clarity).

# Required Process

Work through these steps in the `scratchpad` field before writing your final output.

**Step 1 — Read the contents against the label**
For each attribute, compare what it HOLDS with what its name and definition CLAIM. Note
every group of contents that does not belong.

**Step 2 — Group the attributes by the question they answer**
Different questions stay separate; never collapse across them.

**Step 3 — Set granularity by prevalence, within one question**
Use the shares shown. Keep the large ones. Group the ones far below their siblings. Apply
rule 3 to decide split versus widen — and prefer widen.

**Step 4 — Check the facet placement**
For each attribute, ask whether the facet it sits under is the one whose question it
answers. If not, name the right facet in `facet_name`.

**Step 5 — Route what does not fit**
For each group from Step 1, pick one of the five exits in rule 6.

**Step 6 — Check coverage**
Every attribute shown above appears in the `source_attributes` of exactly one returned
attribute. Merging and forgetting look identical unless you list what went where.

# Output

Return a JSON object with:
- `scratchpad`: your reasoning for steps 1-6
- `attributes`: every attribute that survives, each with `action`, `facet_name`,
  `attribute_name`, `attribute_definition`, `example_observations`, `source_attributes`,
  and `instance_texts` (for "split" only)
- `misfits`: every group of responses that belongs elsewhere, each with `verdict`,
  `target_attribute` (for "move" only) and `instance_texts`

Names and definitions must be written in {language}. Response texts are copied exactly as
they appear above.

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
    merge across a boundary drags every idea in that bucket along with it. Which
    is exactly why it is allowed here and nowhere else — with dozens of scopes
    each settled on its own, the same concept survives in several places, and no
    other phase can see that.

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
  and that is intended — this is the only step where structure is allowed to relocate.
- Most attributes belong to no group. An attribute that stays exactly where it is returns as
  a group of one, listing only its own id.

Leave the catch-all attributes alone. They are per-domain offers, not concepts, and folding
two of them together would merge two different domains' residuals into one meaningless bucket.
Return each of them as a group of one.

# Output

Return a JSON object with:
- `scratchpad`: your reasoning — (1) which attributes across scopes mean the same thing,
  (2) for each group, which scope holds most of its responses, (3) a check that every id
  appears exactly once
- `items`: the merged inventory. Each entry has `name`, `definition`, `source_ids` and
  `home_id`. Every input id must appear in exactly one entry's `source_ids`.

Names and definitions must be written in {language}.

{UNIVERSAL_RULES}

{INSTRUCTOR_HINT}"""
