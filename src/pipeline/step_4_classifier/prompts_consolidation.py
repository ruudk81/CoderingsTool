"""Consolidation prompts for step 4.

Discovery proposes facets and their attributes together, per chunk. What comes
back is the same concept under several names at two levels at once. Settling
that is this module's job, and it takes two calls with different scopes — see
dev/ARCHITECTURE.md.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from pipeline.step_4_classifier.prompts_discovery import (
    DiscoveredAttribute, DiscoveredFacet,
)
from pipeline.step_4_classifier.prompts_shared import (
    INSTRUCTOR_HINT, UNIVERSAL_RULES, build_context_block, build_taxonomy_block,
)


class ConsolidatedAttribute(DiscoveredAttribute):
    """An attribute after consolidation, stating what folded into it."""
    source_attribute_ids: List[str] = Field(
        ..., description=(
            "The bracketed ids of every candidate attribute that folded into "
            "this one, e.g. ['F1-A2', 'F4-A1']. One that survived unchanged "
            "lists just its own id"))


class ConsolidatedFacet(DiscoveredFacet):
    """A facet after consolidation, stating what folded into it."""
    facet_question: str = Field(
        ..., description=(
            "The one question this facet answers about the responses, phrased "
            "as a question, in the survey language"))
    source_facet_ids: List[str] = Field(
        ..., description=(
            "The bracketed ids of every candidate facet that folded into this "
            "one, e.g. ['F1', 'F7']. One that survived unchanged lists just "
            "its own id"))
    attributes: List[ConsolidatedAttribute] = Field(
        ..., description=(
            "The consolidated attributes of this facet, pooled from every "
            "candidate that folded into it"))


class ConsolidationResult(BaseModel):
    """What one consolidation call per domain returns.

    The two `source_*` fields are not bookkeeping but a safety net. Without
    them a candidate that was merged looks exactly like a candidate that was
    forgotten: neither appears in the answer. With them, whatever nobody claims
    stays (`kept_unclaimed`) instead of vanishing silently — and when
    consolidation runs in rounds that counts double, because what drops out in
    round 1 never comes back.

    `raw_facets` also preserves the state before the merge, but that serves a
    different purpose: diagnosis afterwards, not detection during the run.

    `decision_summary` replaced a free-form `scratchpad` on 2026-08-15. The
    seven steps are still worked through — they are the process — but what comes
    back is the decisions, not the working. A field that invites a full
    reasoning trace on a phase that already reasons internally produces long,
    uneven output in which the result is the smaller part.
    """
    decision_summary: List[str] = Field(
        ..., description=(
            "One short line per consolidation decision that took judgement, "
            "each stating what was done and why — 'kept X and Y apart: they "
            "answer different questions'. Not a reasoning trace, and not a "
            "line for every candidate: only the calls a reader would want to "
            "check"))
    facets: List[ConsolidatedFacet] = Field(
        ..., description=(
            "The fewest mutually exclusive facets that cover the domain, each "
            "with its consolidated attributes"))


# =============================================================================
# PROMPT — CHUNK CONSOLIDATION
# =============================================================================

#: Examples shown per candidate attribute. Consolidation is asked to carry
#: examples over, so it can only pass on what it is shown: rendering one while
#: asking for two or three left the model a choice between merging attributes it
#: had just judged distinct, and writing an example that was never in the data.
_EXAMPLES_SHOWN = 3


def build_candidate_index(
    candidates: List[DiscoveredFacet],
) -> Tuple[Dict[str, DiscoveredFacet],
           Dict[str, Tuple[str, DiscoveredAttribute]]]:
    """Stable ids for the candidates of one consolidation call.

    `F1`, and `F1-A1` for the first attribute inside it. Positional and
    therefore deterministic for a given task, which is all that is needed: the
    ids exist for the length of one call and are never stored.

    Provenance used to come back as names, and names are not unique. The same
    attribute name can sit under two different candidate facets — the review
    that prompted this found two such pairs in one domain — so a claim on a
    name was ambiguous, and a list of two identical strings is something a JSON
    layer may quietly collapse to one. Cross-domain consolidation already works
    on `[A#]` ids for exactly this reason; this brings the phase in line.
    """
    facets: Dict[str, DiscoveredFacet] = {}
    attributes: Dict[str, Tuple[str, DiscoveredAttribute]] = {}
    for i, facet in enumerate(candidates, 1):
        facet_id = f"F{i}"
        facets[facet_id] = facet
        for j, attribute in enumerate(facet.attributes, 1):
            attributes[f"{facet_id}-A{j}"] = (facet_id, attribute)
    return facets, attributes


def build_candidate_block(
    candidates: List[DiscoveredFacet],
    recurrence: Dict[str, int],
    attribute_recurrence: Dict[str, int],
    n_passes: int,
) -> str:
    """The candidates from every chunk, each with its attributes and its reach.

    Consolidation runs before a single idea has been assigned, so there are no
    counts. What there is: how many independent chunks proposed a given item.
    A concept that returns in five passes out of five is better supported than
    one that surfaced once, and that can be made visible without any assignment.

    `dedup_exact_facets` collapses byte-identical names beforehand, so the counts
    have to be carried separately — otherwise this exact signal disappears.

    Both levels carry one, because rule 2 is applied at both: step 6 asks which
    attributes are well supported, and until 2026-08-15 that judgement had no
    data behind it at all.

    The count is per EXACT name and the wording says so. A concept five passes
    proposed under five wordings arrives as five candidates of one pass each —
    precisely the case this phase exists to resolve — so a label promising
    support for the concept would mislead on exactly the wrong candidates.
    Summing over a group is the model's job, and the prompt asks for it.
    """
    facets, attributes = build_candidate_index(candidates)
    by_facet: Dict[str, List[Tuple[str, DiscoveredAttribute]]] = {}
    for attribute_id, (facet_id, attribute) in attributes.items():
        by_facet.setdefault(facet_id, []).append((attribute_id, attribute))

    blocks = []
    for facet_id, facet in facets.items():
        seen = recurrence.get(facet.facet_name, 1)
        lines = [f"[{facet_id}] {facet.facet_name} — proposed under this exact "
                 f"name in {seen} of {n_passes} independent passes",
                 f"    Definition: {facet.facet_definition}"]
        held = by_facet.get(facet_id) or []
        if held:
            lines.append("    Attributes proposed inside it:")
            for attribute_id, attribute in held:
                times = attribute_recurrence.get(attribute.attribute_name, 1)
                lines.append(
                    f"      [{attribute_id}] {attribute.attribute_name} "
                    f"[{times}/{n_passes} passes]: "
                    f"{attribute.attribute_definition}")
                examples = [e for e in attribute.example_observations
                            if e][:_EXAMPLES_SHOWN]
                for example in examples:
                    lines.append(f"        e.g. \"{example}\"")
        else:
            lines.append("    Attributes proposed inside it: (none)")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def build_chunk_consolidation_prompt(
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
    """Every chunk's yield for one domain, folded into one nested inventory.

    This is the heaviest phase of the step. Each chunk saw only part of the
    domain and proposed on its own, so the same concept comes back under
    several names — at two levels at once. Facets that merge bring their
    attributes with them, and those must then be measured against each other by
    the same yardstick.
    """
    exclusion_hint = (
        "\n".join(f"- {x}" for x in domain_exclusions)
        if domain_exclusions else "- (no neighbouring domains were named)")

    return f"""You are a taxonomy consolidation specialist for surveys.
Your task is to merge the facets proposed by several independent passes over one domain
into a single minimal set, and to do the same for the attributes those facets hold.

Each pass saw only part of the domain and proposed on its own, so the same concept comes
back under different names — at both levels at once. That is what you are resolving.

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
ACROSS questions. The same reasoning governs the attributes in step 6, on their own counts.

**3. LIFT, DON'T FLATTEN.**
When grouping is needed, raise the concepts to a shared higher-level label that still
carries their meaning — not a label that merely names the question.
FORBIDDEN: a container that only names the question it sits on. The reader learns what was
asked, not what was said.
REQUIRED: a label that states the answer itself.
Test: read the label alone. If it tells you only which question was asked, it is a
container; if it tells you what the respondents expressed, it is an answer.

**4. PLAIN, MEANINGFUL LABELS.**
Name every surviving facet and attribute in everyday language. Test: reading the label
alone, and knowing the survey question, a layperson knows which distinction is meant. A
short, ordinary noun phrase is what you want — no jargon, no policy register, no long
derived constructions.

**When these conflict, decide in this order:**
1. Domain validity — the facet belongs to this domain and not to a neighbouring one.
2. Orthogonality (rule 1) — concepts answering different questions never merge.
3. Prevalence (rule 2) — how finely to split within one question.
4. Lifting (rule 3) — a group is named by what it says, never by what it asks.
5. Label clarity (rule 4).
6. Fewest items — and only once everything above holds. Never merge distinct concepts, and
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

**Step 6 — Consolidate the attributes inside each surviving facet**
This is the step the two levels meet. For each facet you kept, POOL the attributes of every
candidate that folded into it. That pool now holds duplicates and near-duplicates from
different passes, so put it through the same four rules one level down:
- Attributes answering different questions about the facet stay apart.
- Attributes that restate each other in different words become one.
- A well-supported attribute keeps its own place; thin ones that share a meaning group.
- No attribute may be a broader category, a subtype, a component or a concrete instance of
  another under the same facet. A general item and a specific one that sits inside it are
  one level too many: keep the level a coder can apply and fold the other into it.
  Left standing, the same response can honestly be coded under both.

MERGE TEST — run it on any two items before you fold them together:
1. Would the same observation be coded under both? If not, they are not duplicates.
2. Does the difference between them give an analysis anything? If it does, keep it.
3. Can you state what they share as a thing in its own right, without listing them?
4. After merging, does every example still have one obvious place?
Merge only on four times yes. Never merge to reach a count — not of items, not of examples.

Then check the result against its facet: every attribute must sit inside the facet it hangs
under. If one does not, move it to the facet where it belongs, or drop it if no facet fits.
A facet left holding a single attribute is a WARNING, not a verdict. Usually the facet and
the attribute are the same concept stated twice, and then you keep the level that carries
the meaning. But a real lens can hold one attribute in this material and several in the
next batch. Collapse it only when you can say plainly that the two names mean one thing.

**Step 7 — Account for every candidate**
Confirm you have the minimal set of facets that covers the domain, each holding the minimal
set of attributes that covers what it contains.
Then check coverage, and do it on the bracketed ids, never on names: every candidate facet
id must appear in the `source_facet_ids` of at least one surviving facet, and every
candidate attribute id in the `source_attribute_ids` of at least one surviving attribute.
Two candidates can carry the same name, so a name says nothing about which one you meant.
A candidate whose contents genuinely divide — its attributes belonging under different
survivors — is listed by every survivor that took part. That is the honest record, not a
breach of the rule: forcing such a candidate onto one survivor would claim an absorption
that did not happen.
A candidate you deliberately dropped is not exempt — fold it into whichever survivor
absorbs its meaning. Merging and forgetting look identical in the output unless you list
what went where.

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
  - `attributes`: an array, one entry per surviving attribute in that facet, each with:
    - `attribute_name`: a short descriptive name in {language} (at most 5 words)
    - `attribute_definition`: the observable property it captures, in {language} (1-2 sentences)
    - `example_observations`: 1-3 observations carried over from the candidates that
      folded into this attribute, copied exactly as shown. Give what is there: an
      attribute that carries one example gives one. NEVER merge attributes that mean
      different things in order to reach a higher count — the count follows the
      taxonomy, never the other way round
    - `source_attribute_ids`: the bracketed ids of every candidate attribute that folded
      into this one, e.g. ["F1-A2", "F4-A1"]. One that survived unchanged lists its own id.

Names, definitions, questions and the decision summary are written in {language}. The two
`source_*_ids` fields carry ids, not names, and are copied exactly as bracketed above.

{UNIVERSAL_RULES}

{INSTRUCTOR_HINT}"""


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
    makes a facet a good facet. Rendering them in full, with definitions and
    examples, is what turned this call into two jobs at once.
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
ACROSS questions. The same reasoning governs the attributes in step 6, on their own counts.

**3. LIFT, DON'T FLATTEN.**
When grouping is needed, raise the concepts to a shared higher-level label that still
carries their meaning — not a label that merely names the question.
FORBIDDEN: a container that only names the question it sits on. The reader learns what was
asked, not what was said.
REQUIRED: a label that states the answer itself.
Test: read the label alone. If it tells you only which question was asked, it is a
container; if it tells you what the respondents expressed, it is an answer.

**4. PLAIN, MEANINGFUL LABELS.**
Name every surviving facet and attribute in everyday language. Test: reading the label
alone, and knowing the survey question, a layperson knows which distinction is meant. A
short, ordinary noun phrase is what you want — no jargon, no policy register, no long
derived constructions.

**When these conflict, decide in this order:**
1. Domain validity — the facet belongs to this domain and not to a neighbouring one.
2. Orthogonality (rule 1) — concepts answering different questions never merge.
3. Prevalence (rule 2) — how finely to split within one question.
4. Lifting (rule 3) — a group is named by what it says, never by what it asks.
5. Label clarity (rule 4).
6. Fewest items — and only once everything above holds. Never merge distinct concepts, and
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
4. After merging, does every example still have one obvious place?
Merge only on four times yes. Never merge to reach a count — not of items, not of examples.

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
