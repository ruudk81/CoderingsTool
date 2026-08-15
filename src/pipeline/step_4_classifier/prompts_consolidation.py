"""Consolidation prompts for step 4.

Discovery proposes facets and their attributes together, per chunk. What comes
back is the same concept under several names at two levels at once. Settling
that is this module's job, and it takes two calls with different scopes — see
dev/ARCHITECTURE.md.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from pipeline.step_4_classifier.prompts_discovery import DiscoveredAttribute
from pipeline.step_4_classifier.prompts_shared import (
    INSTRUCTOR_HINT, UNIVERSAL_RULES, build_context_block, build_taxonomy_block,
)


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
            "as a question, in the survey language"))
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
        lines = [f"[{facet_id}] {pool.facet_name} — proposed under this exact "
                 f"name in {seen} of {n_passes} independent passes",
                 f"    Definition: {pool.facet_definition}"]
        if pool.facet_question:
            lines.append(f"    Question it answers: {pool.facet_question}")
        names = [a.attribute_name for a in pool.attributes]
        lines.append(
            "    Attributes proposed inside it: "
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
Your task is to merge the facets proposed by several independent passes over one domain
into a single minimal set. The attributes each facet holds are shown as evidence of what it
contains; settling those is a separate step and not your job here.

Each pass saw only part of the domain and proposed on its own, so the same concept comes
back under different names. That is what you are resolving.

{build_context_block(
    language=language, survey_question=survey_question, sector=sector,
    entity=entity, topic=topic, perspective=perspective, intent=intent)}

{build_taxonomy_block(
    dimension=dimension, dimension_name=dimension_name,
    dimension_description=dimension_description)}

You are working within this domain, and only within it:

<taxonomy_domain>
{domain_label} — {domain_definition}
</taxonomy_domain>

Here are the candidates from all passes over this domain. Each shows how many independent
passes proposed it, and the attributes that were proposed inside it:

<candidates>
{candidate_block}
</candidates>

# Consolidation Rules

Consolidation is the goal: do NOT keep every concept separate — group. But govern grouping
by these rules, in this order.

**1. UNDERLYING QUESTION FIRST (orthogonality — the guardrail).**
For each concept, work out which underlying question it answers about the responses.
- Concepts answering DIFFERENT questions are orthogonal: never merge them into one facet.
- Distinct ANSWERS to the SAME question stay apart when merging them would erase what
  tells them apart. Merge only when what the group shares can itself be stated as an
  answer. Evaluative direction is not an answer — see the universal rules below.
- Do not create separate facets based only on the object being discussed when the same
  underlying answer applies. An object is not a question.
- A disposition, an action and an outcome are different KINDS of statement, not degrees of
  one. What something is oriented towards, what it actually does, and what follows from it
  answer three questions, so they never fold together — a group that mixes them reads as
  one item and codes as three. Do not infer one from another either: an action is only an
  action when the response names one, and an outcome only when the response states it.

**2. PREVALENCE SETS GRANULARITY (within one question only).**
Every candidate carries how many passes proposed it UNDER THAT EXACT NAME. Support for a
concept is therefore the sum over the group you form, never the number on one candidate:
five passes that each worded the same concept differently arrive as five candidates
carrying one pass each.
- A concept whose group is well supported keeps its own facet, unless it demonstrably
  draws the same analytical distinction as another surviving facet. Support is a strong
  reason to keep something, never a reason to keep a duplicate: two concepts can both be
  well supported and still be one concept said twice.
- Several thinly supported concepts answering the same question are GROUPED into one facet
  that still names what they share in plain language.
Prevalence decides how finely to split WITHIN one question; it never licenses merging
ACROSS questions.

**3. LIFT, DON'T FLATTEN.**
When grouping is needed, raise the concepts to a shared higher-level label that still
carries their meaning — not a label that merely names the question.
FORBIDDEN: a container that only names the question it sits on. The reader learns what was
asked, not what was said.
REQUIRED: a label that states the answer itself.
Test: read the label alone. If it tells you only which question was asked, it is a
container; if it tells you what the respondents expressed, it is an answer.

**4. PLAIN, MEANINGFUL LABELS.**
Name every surviving facet in everyday language. Test: reading the label
alone, and knowing the survey question, a layperson knows which distinction is meant. A
short, ordinary noun phrase is what you want — no jargon, no policy register, no long
derived constructions.

**When these conflict, decide in this order:**
1. Domain validity — the facet belongs to this domain and not to a neighbouring one.
2. Orthogonality (rule 1) — concepts answering different questions never merge.
3. Prevalence (rule 2) — how finely to split within one question.
4. Lifting (rule 3) — a group is named by what it says, never by what it asks.
5. Label clarity (rule 4).
6. Fewest facets — and only once everything above holds. Never merge distinct concepts, and
   never introduce an umbrella, merely to bring the count down. A smaller inventory that
   has lost a distinction is not a better one.

# Step-by-Step Analysis Process

Work through these steps before writing your final output. What you return is the
decisions, not the working.

**Step 1 — Scan the candidates**
Read every candidate facet from every pass. Note recurring concepts, near-duplicates, and
obvious repeats under different names.

**Step 2 — Group overlapping facets**
Group the facets that describe the same or overlapping concept across passes.

MERGE TEST — run it on any two facets before you fold them together:
1. Would the same observation be coded under both? If not, they are not duplicates.
2. Does the difference between them give an analysis anything? If it does, keep it.
3. Can you state what they share as a thing in its own right, without listing them?
4. After merging, does every attribute named under them still have one obvious place?
Merge only on four times yes. Never merge to reach a count — not of facets, not of attributes.

**Step 3 — Apply the same-question test**
For each group, WRITE DOWN the one question it answers about the responses, phrased as a
question. That sentence is what you return in `facet_question`, and it is what makes this
test checkable rather than a matter of feel: two groups that turn out to state the same
question are one facet, and two that state different questions never merge.
Same question and same meaning: group. Different questions: separate. Distinct answers to
one question: separate only when merging them would erase what tells them apart.

Test the question itself before you accept it. If it can be answered by naming a subject or
a topic, you have written a split of the domain, not a facet — every facet here shares one
subject, and what separates them is the KIND of thing said about it. A question that sorts
the material by what it is about belongs one level up, and using it here produces facets
that overlap wherever a response touches two topics at once.

**Step 4 — Let prevalence set the granularity**
Add up the passes across each group you formed; the counts are per exact name, so any
single candidate understates a concept that came back reworded. Within one question, a
well-supported group keeps its own facet; several thinly supported ones are grouped under a
single plainly named facet. Never group across questions.

**Step 5 — Verify the domain boundary**
Every surviving facet must belong to {domain_label} and not to a neighbouring domain:
{exclusion_hint}

**Step 6 — Account for every candidate**
Confirm you have the minimal set of facets that covers the domain.
Then check coverage, and do it on the bracketed ids, never on names: every candidate facet
id must appear in the `source_facet_ids` of at least one surviving facet. Two candidates can
carry the same name, so a name says nothing about which one you meant.
A candidate you deliberately dropped is not exempt — fold it into whichever survivor absorbs
its meaning. Merging and forgetting look identical in the output unless you list what went
where.

# Output

Return a JSON object with these fields:
- `decision_summary`: one short line per decision that took judgement, in {language} —
  what you did and why. Not a reasoning trace, and not a line per candidate: only the
  calls a reader would want to check.
- `facets`: an array, one entry per surviving facet, each with:
  - `facet_name`: a short descriptive name in {language} (at most 5 words)
  - `facet_definition`: what the facet captures, in {language} (1-2 sentences)
  - `facet_question`: the one question this facet answers about the responses, in
    {language}, phrased as a question. No two surviving facets may state the same one.
  - `source_facet_ids`: the bracketed ids of every candidate facet that folded into this
    one, e.g. ["F1", "F7"]. One that survived unchanged lists just its own id.

Names, definitions, questions and the decision summary are written in {language}. The
`source_facet_ids` field carries ids, not names, and is copied exactly as bracketed above.

{UNIVERSAL_RULES}

{INSTRUCTOR_HINT}"""


# =============================================================================
# PROMPT — ATTRIBUTE CONSOLIDATION
# =============================================================================


class SettledAttribute(DiscoveredAttribute):
    """An attribute after consolidation, stating what folded into it."""
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
            f"[{attribute_id}] {attribute.attribute_name} "
            f"[{times}/{n_passes} passes]: {attribute.attribute_definition}")
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
    dimension_name: str,
    dimension_description: str,
    domain_label: str,
    domain_definition: str,
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
Your task is to fold the attributes proposed for ONE facet into a single minimal set.

Several independent passes over this domain proposed these attributes, and the facets they
sat under have since been consolidated into the one below. The pool therefore holds
duplicates and near-duplicates of the same concept under different names. That is what you
are resolving.

{build_context_block(
    language=language, survey_question=survey_question, sector=sector,
    entity=entity, topic=topic, perspective=perspective, intent=intent)}

{build_taxonomy_block(
    dimension=dimension, dimension_name=dimension_name,
    dimension_description=dimension_description)}

You are working inside this facet, and only inside it:

<taxonomy_facet>
Domain: {domain_label} — {domain_definition}
Facet: {facet_name} — {facet_definition}{question_line}
</taxonomy_facet>

Here are the attributes proposed for it, each with how many independent passes proposed it
under that exact name:

<candidates>
{candidate_block}
</candidates>

# Consolidation Rules

Consolidation is the goal: do NOT keep every concept separate — group. But govern grouping
by these rules, in this order.

**1. DIFFERENT QUESTIONS STAY APART (the guardrail).**
Every attribute here answers the facet's question in its own way. Two that answer DIFFERENT
questions about the facet are orthogonal and never merge. Distinct ANSWERS to the same
question stay apart when merging them would erase what tells them apart. Evaluative
direction is not an answer — see the universal rules below.

**2. PREVALENCE SETS GRANULARITY (within one question only).**
Every candidate carries how many passes proposed it UNDER THAT EXACT NAME. Support for a
concept is therefore the sum over the group you form, never the number on one candidate:
several passes that each worded the same concept differently arrive as several candidates
carrying one pass each. A well-supported group keeps its own attribute, unless it
demonstrably draws the same distinction as another survivor. Thinly supported concepts that
share a meaning are grouped under one plainly named attribute.

**3. NO HIERARCHY UNDER ONE FACET.**
No attribute may be a broader category, a subtype, a component or a concrete instance of
another. A general item and a specific one that sits inside it are one level too many: keep
the level a coder can apply and fold the other into it. Left standing, the same response can
honestly be coded under both.

**4. LIFT, DON'T FLATTEN.**
When grouping is needed, raise the concepts to a shared label that still carries their
meaning — not a label that merely names the question. Read the label alone: if it tells you
only which question was asked, it is a container; if it tells you what the respondents
expressed, it is an answer.

**5. PLAIN, MEANINGFUL LABELS.**
Name every surviving attribute in everyday language. Reading the label alone, and knowing
the survey question, a layperson knows which distinction is meant.

**When these conflict, decide in this order:** 1 (different questions) > 2 (prevalence) >
3 (no hierarchy) > 4 (lifting) > 5 (label clarity) > fewest attributes. Never merge distinct
concepts, and never introduce an umbrella, merely to bring the count down. A smaller
inventory that has lost a distinction is not a better one.

MERGE TEST — run it on any two attributes before you fold them together:
1. Would the same observation be coded under both? If not, they are not duplicates.
2. Does the difference between them give an analysis anything? If it does, keep it.
3. Can you state what they share as a thing in its own right, without listing them?
4. After merging, does every example still have one obvious place?
Merge only on four times yes. Never merge to reach a count — not of attributes, not of examples.

NEVER DROP. You see one facet, so you cannot judge where something would belong instead.
An attribute that does not seem to fit this facet stays in your output as it is; say so in
your decision summary and leave it. A later phase sees every facet of the domain at once and
can move it.

If this facet ends up holding a single attribute, that is a signal worth recording: usually
the facet and the attribute are then the same concept stated twice. Note it in your decision
summary. Do not act on it here — the facet is not yours to change.

# Step-by-Step Analysis Process

Work through these steps before writing your final output. What you return is the decisions,
not the working.

**Step 1 — Scan the pool**
Read every candidate. Note recurring concepts, near-duplicates, and obvious repeats under
different names.

**Step 2 — Group what means the same**
Group the candidates that restate each other in different words.

**Step 3 — Apply the same-question test**
For each group, work out which question about the facet it answers. Same question and same
meaning: group. Different questions: separate. Distinct answers to one question: separate
only when merging them would erase what tells them apart.

**Step 4 — Let prevalence set the granularity**
Add up the passes across each group you formed; the counts are per exact name, so any single
candidate understates a concept that came back reworded.

**Step 5 — Check for hierarchy**
No survivor may sit inside another. Where one does, fold it into the level a coder can apply.

**Step 6 — Account for every candidate**
Check coverage on the bracketed ids, never on names: every candidate attribute id must
appear in the `source_attribute_ids` of at least one surviving attribute. Two candidates can
carry the same name, so a name says nothing about which one you meant. A candidate you
folded away is not exempt — list it under whichever survivor absorbs its meaning. Merging and
forgetting look identical in the output unless you list what went where.

# Output

Return a JSON object with these fields:
- `decision_summary`: one short line per decision that took judgement, in {language} —
  what you did and why. Not a reasoning trace, and not a line per candidate: only the
  calls a reader would want to check.
- `attributes`: an array, one entry per surviving attribute, each with:
  - `attribute_name`: a short descriptive name in {language} (at most 5 words)
  - `attribute_definition`: the observable property it captures, in {language} (1-2 sentences)
  - `example_observations`: 1-3 observations carried over from the candidates that folded
    into this attribute, copied exactly as shown. Give what is there: an attribute that
    carries one example gives one. NEVER merge attributes that mean different things in
    order to reach a higher count — the count follows the taxonomy, never the other way round
  - `source_attribute_ids`: the bracketed ids of every candidate attribute that folded into
    this one, e.g. ["A2", "A7"]. One that survived unchanged lists just its own id.

Names, definitions and the decision summary are written in {language}. The
`source_attribute_ids` field carries ids, not names, and is copied exactly as bracketed above.

{UNIVERSAL_RULES}

{INSTRUCTOR_HINT}"""
