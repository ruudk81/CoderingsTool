"""Prompt builders and response models for the attribute layer (L4) of step 4.

The mirror of prompts_facet.py one level down, with the same four phases:

  1. discovery      build_attribute_discovery_prompt      per (facet, chunk)
  2. consolidation  build_attribute_consolidation_prompt  per facet, over chunks
  3. assignment     build_attribute_assignment_prompt     per batch of ideas
  4. refinement     build_attribute_refinement_prompt     per facet, after assignment

The facet is fixed throughout: no phase here can move an attribute or an idea to
another facet. Where the facet layer's parent is the domain, this layer's parent
is the facet — everything else about the shape is the same, deliberately.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Literal, Tuple

from pydantic import BaseModel, Field, create_model, model_validator

from .prompts_shared import (
    INSTRUCTOR_HINT,
    UNIVERSAL_RULES,
    build_context_block,
    build_taxonomy_block,
    level_diagnostic,
)

if TYPE_CHECKING:
    from pipeline.step_3_ideaExtractor.dimension_data import DimensionDefinition


# =============================================================================
# §1 DISCOVERY — per (facet, chunk)
# =============================================================================

class DiscoveredAttribute(BaseModel):
    """One attribute (L4) proposed from a chunk of observations within a facet.

    Same four boundary fields as DiscoveredFacet, one level down. There is no
    `parent_facet` field: the facet is the scope of the call, not a property of
    the item. A field the model can write is a field it can write wrongly.
    """
    attribute_name: str = Field(
        ..., description="Short descriptive name for the attribute (2-5 words)"
    )
    attribute_definition: str = Field(
        ..., description=(
            "One-sentence inclusion definition naming a single observable property. "
            "No examples, no enumerations"
        )
    )
    boundary_test: str = Field(
        ..., description=(
            "A single yes/no question a coder asks to decide whether an idea "
            "belongs to THIS attribute rather than a neighbouring one"
        )
    )
    exclusions: List[str] = Field(
        ..., description=(
            "1-3 short phrases naming what does NOT belong here — especially "
            "the neighbouring attribute it is most easily confused with"
        )
    )
    example_observations: List[str] = Field(
        ..., description="2-3 observations from the input that exemplify this attribute"
    )


class AttributeDiscoveryResult(BaseModel):
    """Discovery output for one chunk."""
    scratchpad: str = Field(
        ..., description=(
            "Reasoning before the final set: "
            "(1) read the observations for the ways they differ from one another, "
            "(2) name the candidate dimensions, "
            "(3) test each pair for independence in both directions and merge the pair "
            "where independence cannot be shown, "
            "(4) check no dimension is an evaluative direction in disguise, "
            "(5) check every candidate stays inside the facet"
        )
    )
    attributes: List[DiscoveredAttribute] = Field(
        ..., description=(
            "The dimensions on which these observations differ. As few as account for "
            "every difference the observations show"
        )
    )


def build_attribute_discovery_prompt(
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
    facet_boundary_test: str,
    facet_exclusions: List[str],
    observations: List[str],
) -> str:
    """Propose attributes (L4) from one chunk of observations within one facet."""
    context_block = build_context_block(
        language=language, survey_question=survey_question, sector=sector,
        entity=entity, topic=topic, perspective=perspective, intent=intent,
    )
    taxonomy_block = build_taxonomy_block(
        dimension=dimension, dimension_name=dimension_name,
        dimension_description=dimension_description,
    )
    observations_block = "\n".join(
        f"{i}. {obs}" for i, obs in enumerate(observations, 1)
    )
    exclusions_line = "; ".join(facet_exclusions) if facet_exclusions else "(none given)"

    return f"""You are a qualitative research analyst specializing in survey response analysis.
Your task is to identify the dimensions on which responses within one facet differ from one another.

{context_block}

{taxonomy_block}

You are working inside ONE facet, within ONE domain. Everything you return belongs to
that facet, and nothing that falls outside it may be returned.

<parents>
Domain: {domain_label} — {domain_definition}
Facet:  {facet_name} — {facet_definition}
Facet boundary test: {facet_boundary_test}
Does NOT belong to this facet (these have their own facets): {exclusions_line}
</parents>

Here are the observations you need to account for:

<observations>
{observations_block}
</observations>

## YOUR TASK

Identify the **dimensions** on which the responses in this facet differ from one another.
Those dimensions are this facet's attributes (level 4).

**Orthogonal.** Two responses must be able to differ on one dimension while sitting the
same way on another, and the other way round. If you cannot show that in both directions
with the observations in front of you, the two are one dimension, not two.

**As few as possible.** Enough to account for every difference the observations show, and
not one more. If the responses differ in only one way, return exactly one dimension.

**Descriptive.** A dimension names something the responses are ABOUT, never how
positively or negatively the subject is judged. "Positive versus negative", "satisfied
versus dissatisfied" and any rephrasing of those is not a dimension — evaluative
direction is recorded per response as valence and never becomes structure.

{UNIVERSAL_RULES}

## OUTPUT

Work through your reasoning in the scratchpad field first.

For EACH dimension provide, using the attribute fields:
- **attribute_name** — a short descriptive name for the dimension
- **attribute_definition** — one sentence naming what varies along it, with no examples or
  enumerations
- **boundary_test** — one yes/no question that decides whether a response belongs to THIS
  dimension rather than a neighbouring one
- **exclusions** — what does NOT belong, naming the neighbouring dimension it is most
  easily confused with
- **example_observations** — 2-3 observations from the input above, copied exactly

All names, definitions, boundary tests and exclusions must be written in {language}.

Begin processing now and {INSTRUCTOR_HINT}"""


# =============================================================================
# §2 CONSOLIDATION — per facet, across chunks, before any idea is assigned
# =============================================================================

class ConsolidatedAttribute(DiscoveredAttribute):
    """One attribute surviving consolidation, with the candidates it absorbed."""
    source_attributes: List[str] = Field(
        ..., description=(
            "The attribute_name of every candidate that goes into this one. "
            "A candidate that is kept unchanged lists its own name"
        )
    )


class AttributeConsolidationResult(BaseModel):
    """The settled attribute inventory for one facet."""
    scratchpad: str = Field(
        ..., description=(
            "Consolidation reasoning: "
            "(1) list the unique candidates across all chunks, "
            "(2) group the ones that overlap conceptually, "
            "(3) name and define each consolidated attribute, "
            "(4) for every surviving pair ask whether one response could belong "
            "to both, and merge when it could, "
            "(5) verify the survivors still cover everything the candidates covered, "
            "(6) write each survivor's boundary_test and exclusions"
        )
    )
    attributes: List[ConsolidatedAttribute] = Field(
        ..., description="The complete attribute set for this facet after consolidation"
    )


def _build_candidate_block(candidates: List[DiscoveredAttribute]) -> str:
    """Render the chunk yield as numbered candidates, each with its evidence."""
    blocks = []
    for i, candidate in enumerate(candidates, 1):
        exclusions = "; ".join(candidate.exclusions) if candidate.exclusions else "(none)"
        observations = "; ".join(candidate.example_observations)
        blocks.append(
            f"[C{i}] {candidate.attribute_name}\n"
            f"     Definition: {candidate.attribute_definition}\n"
            f"     Boundary test: {candidate.boundary_test}\n"
            f"     Does not belong: {exclusions}\n"
            f"     Observations that produced this proposal: {observations}"
        )
    return "\n\n".join(blocks)


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
    candidates: List[DiscoveredAttribute],
) -> str:
    """Settle one facet's attribute inventory, across all chunk proposals."""
    context_block = build_context_block(
        language=language, survey_question=survey_question, sector=sector,
        entity=entity, topic=topic, perspective=perspective, intent=intent,
    )
    taxonomy_block = build_taxonomy_block(
        dimension=dimension, dimension_name=dimension_name,
        dimension_description=dimension_description,
    )
    diagnostic = level_diagnostic(dimension, "attribute")
    candidate_block = _build_candidate_block(candidates)

    return f"""You are a taxonomy consolidation specialist for survey coding.
Your task is to merge attribute proposals from several independent passes over one facet
into a single, coherent set of attributes.

{context_block}

{taxonomy_block}

You are working inside ONE facet. Every attribute you return belongs to it.

<parents>
Domain: {domain_label} — {domain_definition}
Facet:  {facet_name} — {facet_definition}
</parents>

The question every attribute must answer under this lens is:

<attribute_diagnostic>
{diagnostic}
</attribute_diagnostic>

Here are the candidate attributes. Each pass saw a different sample of the responses and
did not see the other passes, so the same attribute may appear several times under
different names. Each candidate carries the observations that produced it — those are
your evidence:

<candidates>
{candidate_block}
</candidates>

## YOUR TASK

Consolidate these candidates into the fewest mutually exclusive attributes needed for
full coverage.

Judge the candidates on their observations, not on their labels. Two labels that read
differently but were produced by the same kind of observation are ONE attribute. Two
labels that read alike but were produced by different observations are TWO.

Consolidation principles:

- **MERGE** candidates that overlap conceptually, are near-equivalent, or where one is
  a subset of the other.
- **MERGE** candidates that are two lenses on the same phenomenon — different wording
  for one underlying property.
- **THE BOUNDARY TEST DECIDES.** For each pair of survivors, write the boundary that
  separates them. If you cannot state a clean boundary between an attribute and its
  nearest neighbour, they are not two attributes — merge them.
- **ENSURE ontological distinctness** — no two attributes may share conceptual space,
  and none may be a subset of another.
- **ENSURE semantic separability** — a coder must not plausibly hesitate between two
  attributes.
- **MAINTAIN full coverage** — the survivors must collectively cover everything the
  candidates covered. Consolidating is not discarding.
- **MINIMIZE the count** while preserving distinctions the observations actually show.
  If the observations hold four distinct answers to the attribute question, return four
  attributes — do not collapse them because fewer is tidier.
- **STAY inside the facet.** A candidate that falls outside the facet is not an
  attribute to keep; leave it out rather than widening the facet to fit it.

Two things are FORBIDDEN in what you return:

- **FORBIDDEN: attributes that overlap conceptually, semantically or in meaning**, judged
  in terms of the reactions and answers people gave to the survey question. Not in the
  abstract — in terms of what respondents actually said.
- **FORBIDDEN: any pair that fails the researcher's test.** Picture a researcher reading
  your final set and saying: *"these two do not actually help me organise meaningfully
  different reactions to this question — they essentially mean the same thing, or they
  overlap so heavily in meaning that the split buys me nothing."* If a pair invites that
  sentence, it is one attribute. Merge it.

Every candidate you consume must be listed in the `source_attributes` of the attribute
that consumes it. A candidate you do not list is left standing as it is, so list them.

{UNIVERSAL_RULES}

## OUTPUT

Work through the consolidation in the scratchpad field first.

For EACH consolidated attribute provide:
- **attribute_name** — a short descriptive name
- **attribute_definition** — one sentence naming a single observable property, no
  examples or enumerations
- **boundary_test** — one yes/no question that decides membership
- **exclusions** — what does NOT belong, naming the neighbouring attribute it is most
  easily confused with
- **example_observations** — 2-3 observations, copied exactly from the candidates above
- **source_attributes** — the attribute_name of every candidate consumed into this one

All attribute names, definitions, boundary tests and exclusions must be written in {language}.

Begin processing now and {INSTRUCTOR_HINT}"""


# =============================================================================
# §3 ASSIGNMENT — ideas into the settled inventory
# =============================================================================
#
# Like the facet layer's assignment, this phase carries no taxonomy block and no
# universal rules: it creates nothing, it picks an id from a menu whose entries
# already carry their definitions and boundaries, and it runs at the volume of
# the dataset.


def build_attribute_menu(attributes: List[ConsolidatedAttribute]) -> str:
    """Render the settled attributes as a numbered menu.

    The [A#] id is what the response is keyed on, so the numbering here and the
    id list handed to `build_attribute_assignment_model` must come from the same
    list in the same order.
    """
    lines = []
    for i, a in enumerate(attributes, 1):
        exclusions = "; ".join(a.exclusions) if a.exclusions else ""
        examples = "; ".join(a.example_observations[:3])
        block = (
            f"[A{i}] {a.attribute_name}\n"
            f"     Description: {a.attribute_definition}\n"
            f"     Boundary: {a.boundary_test}"
        )
        if exclusions:
            block += f"\n     Does not belong here: {exclusions}"
        if examples:
            block += f"\n     Examples: {examples}"
        lines.append(block)
    return "\n\n".join(lines)


def build_attribute_assignment_model(attribute_ids: List[str], idea_ids: List[str]):
    """Runtime response model for one assignment call.

    The attribute layer had no Literal on the assigned id until this rewrite —
    it was the weakest link in the chain, where the facet layer had one and step
    3's domain assignment had an enum. Both id spaces are Literals here, so a
    hallucinated id is a schema violation instructor retries.
    """
    attribute_id_literal = Literal[tuple(attribute_ids + ["A_NONE"])]  # type: ignore[valid-type]
    idea_id_literal = Literal[tuple(idea_ids)]  # type: ignore[valid-type]

    item_model = create_model(
        "AttributeAssignmentItem",
        idea_id=(idea_id_literal, Field(
            ..., description="The [id] tag of the idea, echoed exactly")),
        assigned_attribute_id=(attribute_id_literal, Field(
            ..., description=(
                "The attribute id from the [A#] prefix. Return ONLY the id. "
                "Use A_NONE when no attribute fits this idea"))),
        confidence=(float, Field(
            ..., ge=0.0, le=1.0, description="Assignment confidence (0.0-1.0)")),
        valence=(Literal["+", "-", "0"], Field(
            default="0",
            description=(
                "Evaluative direction relative to the attribute: "
                "+ positive, - negative, 0 neutral"))),
    )
    return create_model(
        "AttributeAssignmentResult",
        assignments=(List[item_model], Field(
            ..., description=(
                "Exactly one assignment per idea listed in the prompt, "
                "no idea skipped, no idea added"))),
    )


def build_attribute_assignment_prompt(
    *,
    language: str,
    survey_question: str,
    sector: str,
    entity: str,
    topic: str,
    perspective: str,
    intent: str,
    facet_name: str,
    facet_definition: str,
    attributes: List[ConsolidatedAttribute],
    ideas: List[Tuple[str, str]],
) -> str:
    """Assign one or more ideas to an attribute, with valence."""
    context_block = build_context_block(
        language=language, survey_question=survey_question, sector=sector,
        entity=entity, topic=topic, perspective=perspective, intent=intent,
    )
    menu = build_attribute_menu(attributes)
    ideas_block = "\n".join(f"[{idea_id}] {label}" for idea_id, label in ideas)

    return f"""You are a qualitative coding assistant. Assign each survey response idea below to the attribute it belongs to.

{context_block}

<facet>
Facet: {facet_name} — {facet_definition}
</facet>

<attributes>
{menu}

[A_NONE] None of the attributes above fits the idea.
</attributes>

<ideas>
{ideas_block}
</ideas>

### VALENCE (evaluation relative to the attribute)
- "+" Positive — the idea describes a positive instance of this attribute (present,
  sufficient, meeting expectations)
- "-" Negative — the idea describes a negative instance of this attribute (absent,
  insufficient, failing expectations)
- "0" Neutral — the idea is descriptive, ambiguous, or expresses no evaluation

Valence is not emotional sentiment. It is evaluative direction relative to the attribute.

Use each attribute's Boundary line to decide the doubtful cases; that is what it is for.

Judge every idea independently on its own text; do not let one assignment influence the
next. Return exactly one item per idea, echoing that idea's [id]. Do not skip ideas and
do not add ideas. If no attribute fits an idea, use "A_NONE" for that idea rather than
forcing it into the nearest one.

Begin processing now and {INSTRUCTOR_HINT}"""


# =============================================================================
# §4 REFINEMENT — per facet, after every idea has been assigned
# =============================================================================

class AttributeMisfitGroup(BaseModel):
    """A group of ideas sitting in an attribute they do not belong to."""
    from_attribute: str = Field(
        ..., description="The attribute currently holding these ideas"
    )
    instance_texts: List[str] = Field(
        ..., description=(
            "The exact response texts that do not belong, copied verbatim from the "
            "contents shown. Never counts, paraphrases or summaries"
        )
    )
    verdict: Literal["move", "out"] = Field(
        ..., description=(
            "'move' when these ideas belong to a named existing attribute; "
            "'out' when they carry no substantive content at all"
        )
    )
    target_attribute: str = Field(
        default="",
        description=(
            "For verdict 'move': the attribute these ideas belong to, named exactly as "
            "shown in this facet or in the neighbouring facets list. Empty for 'out'"
        ),
    )
    reason: str = Field(
        ..., description="One sentence: why these texts do not belong where they are"
    )


class RefinedAttribute(ConsolidatedAttribute):
    """One attribute surviving refinement. Its facet is fixed by the task."""
    action: Literal["keep", "merge", "widen", "split"] = Field(
        ..., description=(
            "keep = unchanged; merge = several sources into this one; "
            "widen = same attribute, description restated to cover what it holds; "
            "split = one source divided into named children (instance_texts required)"
        )
    )
    instance_texts: List[str] = Field(
        default_factory=list,
        description=(
            "For action 'split' ONLY: the exact response texts routed to this child, "
            "copied verbatim. Required when a source attribute is divided over more "
            "than one returned attribute. Empty otherwise"
        ),
    )


class AttributeRefinementResult(BaseModel):
    """Final attribute inventory for one facet, plus the misfits found in it."""
    scratchpad: str = Field(
        ..., description=(
            "Step-by-step reasoning: (1) read each attribute's contents against its "
            "label and note groups that do not belong, (2) group attributes by the "
            "underlying distinction each one answers, (3) set granularity by prevalence "
            "using the shares shown, (4) for every surviving pair, ask whether their "
            "contents can be told apart without reading the labels, and merge the pair "
            "where they cannot, (5) route each non-fitting group to one of the four "
            "exits, (6) check every label states a value rather than the question, "
            "(7) assemble the final inventory"
        )
    )
    attributes: List[RefinedAttribute] = Field(
        ..., description="The complete attribute set for this facet after refinement"
    )
    misfits: List[AttributeMisfitGroup] = Field(
        default_factory=list,
        description="Groups of ideas that do not belong to the attribute holding them",
    )

    @model_validator(mode="after")
    def _routable(self):
        """Reject an inventory whose ideas cannot be routed."""
        for a in self.attributes:
            if a.action == "split" and not a.instance_texts:
                raise ValueError(
                    f'attribute "{a.attribute_name}" has action "split" but no '
                    f'instance_texts. A split must list the exact response texts '
                    f'routed to each child, or the ideas cannot be divided.'
                )

        claimed_by: Dict[str, List[str]] = {}
        for a in self.attributes:
            for src in (a.source_attributes or []):
                claimed_by.setdefault(src, []).append(a.attribute_name)

        for src, claimants in claimed_by.items():
            if len(claimants) < 2:
                continue
            without_texts = [a.attribute_name for a in self.attributes
                             if src in (a.source_attributes or []) and not a.instance_texts]
            if without_texts:
                raise ValueError(
                    f'source attribute "{src}" is claimed by {len(claimants)} returned '
                    f'attributes ({", ".join(claimants)}), but {", ".join(without_texts)} '
                    f'give no instance_texts. Either let ONE attribute take "{src}", or '
                    f'make every claimant action "split" and list the exact response '
                    f'texts each one takes.'
                )
        return self


def build_attribute_contents_block(
    rows: List[Tuple[str, int, float, List[str]]],
) -> str:
    """Render what each attribute actually holds: name, count, share, real texts.

    `rows`: (attribute_name, n_ideas, share_of_facet, sample_texts).
    """
    blocks = []
    for name, n_ideas, share, texts in rows:
        contents = "\n".join(f"       - {t}" for t in texts)
        blocks.append(
            f"{name} — {n_ideas} responses ({round(share * 100)}% of the facet)\n"
            f"     Contents:\n{contents}"
        )
    return "\n\n".join(blocks)


def build_neighbour_block(
    neighbours: List[Tuple[str, List[Tuple[str, int]]]],
) -> str:
    """Format adjacent facets as steer-clear context for refinement.

    `neighbours`: [(facet_name, [(attribute_name, n_ideas), ...]), ...]

    Shown so the model can write its boundaries against real neighbours instead of
    abstract ones, and so it can name a target when a group of ideas belongs to one
    of them. Explicitly NOT merge candidates — without that instruction the model
    starts merging across facets, which is the failure this phase exists to prevent.
    """
    if not neighbours:
        return ""
    lines = [
        "<neighbouring_facets>",
        "These facets sit beside yours in the same domain. They are shown so you can "
        "write your boundaries against real neighbours instead of abstract ones.",
        "THEY ARE NOT MERGE CANDIDATES. You may not merge your attributes into them, "
        "and you may not restate their attributes as your own. Their only two uses:",
        "  (a) sharpen your own labels, so yours states what theirs does not;",
        "  (b) name a target when a group of ideas in YOUR facet clearly belongs to one of them.",
    ]
    for facet_name, attrs in neighbours:
        if not attrs:
            continue
        listed = ", ".join(f"{n} ({c})" for n, c in attrs)
        lines.append(f'  Facet "{facet_name}" — attributes: {listed}')
    lines.append("</neighbouring_facets>")
    return "\n".join(lines)


def build_attribute_refinement_prompt(
    *,
    language: str,
    survey_question: str,
    sector: str,
    entity: str,
    topic: str,
    perspective: str,
    intent: str,
    dimension: "DimensionDefinition",
    domain_label: str,
    domain_definition: str,
    facet_name: str,
    facet_definition: str,
    attributes_block: str,
    neighbour_block: str,
) -> str:
    """Settle one facet's attributes against what they actually ended up holding.

    The eight rules are the best-tested text step 4 had, carried over from the
    old in-facet consolidation. What is gone is the precedence ordering: that
    existed because this one call also had to clean up the near-duplicates the
    chunk yield left behind. Consolidation does that now, so the goals no longer
    collide and there is nothing left to arbitrate.
    """
    context_block = build_context_block(
        language=language, survey_question=survey_question, sector=sector,
        entity=entity, topic=topic, perspective=perspective, intent=intent,
    )
    diagnostic = level_diagnostic(dimension, "attribute")
    # Step 3's per-dimension wording for "no substantive content". Not
    # `standing_not_known.short`: not knowing the subject IS substantive.
    contentless = dimension.prompt_rules.contentless_test
    neighbours = f"\n{neighbour_block}\n" if neighbour_block else ""

    return f"""You are a taxonomy consolidation specialist for surveys.
Your task is to settle the final attribute inventory of ONE facet, now that every idea has been assigned.

{context_block}

<parents>
Domain: {domain_label} — {domain_definition}
Facet:  {facet_name} — {facet_definition}
</parents>

The question every attribute answers under this lens is:

<attribute_diagnostic>
{diagnostic}
</attribute_diagnostic>

Here are this facet's attributes, each with the number of responses actually assigned to
it, its share of the facet, and a sample of the responses it really holds:

<facet_attributes>
{attributes_block}
</facet_attributes>
{neighbours}
Judge each attribute on what it actually holds, not on how its label reads. The counts
and the response texts above are the evidence; the labels were written before a single
response had been assigned.

<refinement_rules>
**1. DISTINCTION FIRST.** Attributes that answer different distinctions stay apart,
however similar their labels look. Mutually exclusive values of the SAME distinction are
also kept apart — merging opposite poles creates an empty container. Do NOT create
separate attributes based only on the object discussed when the same underlying value
applies; an object is not a distinction.

**2. PREVALENCE SETS GRANULARITY** — within one distinction only. Each attribute shows
its share of this facet. Judge size relative to its siblings, never against an absolute
number. The largest keep their own identity. Those far below their siblings are grouped,
but only with same-distinction neighbours, into one attribute that still names the shared
value in plain language. An attribute holding a large share AND visibly diverse contents
is too abstract: split it, do not widen it.

**3. LIFT, DON'T FLATTEN.** When grouping is needed, raise the concepts to a shared
higher-abstraction label that still carries their meaning — not a label that merely names
the question. Read the label alone: if it tells you only which question was asked, it is
a container; if it tells you what the answer was, it is a value.

**4. TWO LABELS, ONE THING.** Before routing anything, read the contents of each pair
against each other. Where you cannot tell which of two attributes a response belongs to
without reading the labels, they are not two attributes: return ONE, listing both in
`source_attributes`. The labels were written before a single response had been assigned;
what each one actually caught is the evidence that they turned out to name the same
thing. This is the only phase that can see that, and a pair left standing here is left
standing for good.

**5. PLAIN, MEANINGFUL LABELS.** Name every surviving attribute in everyday language. A
layperson reading the label alone, given the survey question, should know which
distinction is meant. No jargon, no nominalizations.

**6. THE FACET IS FIXED.** Every attribute you return belongs to "{facet_name}". You
cannot move an attribute to another facet, and you cannot create one that belongs to
another facet. If a GROUP OF IDEAS belongs elsewhere, report it under `misfits` — the
ideas move, the attribute stays here.

**7. FOUR EXITS FOR WHAT DOES NOT FIT.** Read what each attribute actually contains.
Where contents do not match the label, choose per group:
   - the group points at ONE existing attribute (here or in a neighbouring facet)
       -> `misfits`, verdict "move": name the target and the EXACT response texts
   - the group is one coherent concept that has no attribute yet
       -> action "split": name the children and which EXACT texts go to each
   - the group is diverse but genuinely related to this attribute
       -> action "widen": restate the description so it honestly covers what is there
   - the group carries NO SUBSTANTIVE CONTENT WHATSOEVER — filler, or {contentless}
       -> `misfits`, verdict "out"
   "Out" is not an escape hatch for "this does not fit the attributes I chose". A text
   that names something real HAS substance: if it has no home yet, create one with
   "split". Moves and splits must be expressed as EXACT response texts copied from the
   contents shown above — never as counts, paraphrases or summaries. Every decision has
   to be checkable against the data.

**8. ONE SOURCE, ONE DESTINATION** — unless you route by text. Every attribute in the
input must end up in exactly ONE returned attribute. To divide one input attribute's
contents over TWO returned attributes, use action "split" for each part and list the
exact texts belonging to it in `instance_texts`.

**9. NOTHING THE RESPONSES SAY MAY DISAPPEAR.** Grouping is not discarding: a value that
moves under a shared label is still reported, a value sent "out" is gone. Never use
"out", or a label that silently drops what it absorbed, to make the inventory tidier.
But being a real value is not on its own a reason to stand alone — where responses are
few against their siblings, their honest home is a shared label that still names what
they say, not an attribute of their own. Collapsing a facet to a SINGLE attribute removes
a whole level of the hierarchy: the facet name then says nothing the attribute does not
already say. Do that only when the contents genuinely express one value.
</refinement_rules>

{UNIVERSAL_RULES}

## OUTPUT

Work through your reasoning in the scratchpad field first.

For EACH surviving attribute provide: action, attribute_name, attribute_definition,
boundary_test, exclusions, example_observations (exact text from the contents),
source_attributes, and — for "split" only — instance_texts.

`action` is exactly one of: "keep" (unchanged), "merge" (several sources into this one),
"widen" (description restated to cover what it holds), "split" (one source divided into
named children). Every misfit group carries verdict "move" or "out".

All attribute names, definitions, boundary tests and exclusions must be written in {language}.
Copy response texts verbatim when you route them; they are matched literally.

Begin processing now and {INSTRUCTOR_HINT}"""
