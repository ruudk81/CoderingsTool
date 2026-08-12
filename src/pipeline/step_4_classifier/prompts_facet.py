"""Prompt builders and response models for the facet layer (L3) of step 4.

Four phases, in the order step 3 uses for the domain layer:

  1. discovery      build_facet_discovery_prompt      per (domain, chunk)
  2. consolidation  build_facet_consolidation_prompt  per domain, over chunks
  3. assignment     build_facet_assignment_prompt     per batch of ideas
  4. refinement     build_facet_refinement_prompt     per domain, after assignment

Discovery proposes, consolidation settles the inventory before a single idea is
assigned, assignment fills it, and refinement judges the result against what the
facets actually ended up holding. The domain is fixed throughout: no phase here
can move a facet or an idea to another domain.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple

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
# §1 DISCOVERY — per (domain, chunk)
# =============================================================================

class DiscoveredFacet(BaseModel):
    """One facet (L3) proposed from a chunk of observations within a domain.

    Four fields carry the boundary, the same four step 3 uses per domain: what
    it is (`facet_definition`), the question that decides membership
    (`boundary_test`), and what it is not (`exclusions`). There is deliberately
    no separate inclusion/exclusion rule pair — the definition IS the inclusion
    rule, and two fields for one idea drift apart.
    """
    facet_name: str = Field(
        ..., description="Short descriptive name for the facet (2-5 words)"
    )
    facet_definition: str = Field(
        ..., description=(
            "One-sentence inclusion definition naming a single aspect. "
            "No examples, no enumerations"
        )
    )
    boundary_test: str = Field(
        ..., description=(
            "A single yes/no question a coder asks to decide whether an idea "
            "belongs to THIS facet rather than a neighbouring one"
        )
    )
    exclusions: List[str] = Field(
        ..., description=(
            "1-3 short phrases naming what does NOT belong here — especially "
            "the neighbouring facet it is most easily confused with"
        )
    )
    example_observations: List[str] = Field(
        ..., description="3-5 observations from the input that exemplify this facet"
    )


class FacetDiscoveryResult(BaseModel):
    """Discovery output for one chunk."""
    scratchpad: str = Field(
        ..., description=(
            "Reasoning before the final set: "
            "(1) read the observations for the ways they differ from one another, "
            "(2) name the candidate dimensions, "
            "(3) test each pair for independence in both directions and merge the pair "
            "where independence cannot be shown, "
            "(4) check no dimension is an evaluative direction in disguise, "
            "(5) verify every observation has a home"
        )
    )
    facets: List[DiscoveredFacet] = Field(
        ..., description=(
            "The dimensions on which these observations differ. As few as account for "
            "every difference the observations show"
        )
    )


def build_facet_discovery_prompt(
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
    domain_boundary_test: str,
    domain_exclusions: List[str],
    observations: List[str],
) -> str:
    """Propose facets (L3) from one chunk of observations within one domain."""
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
    exclusions_line = "; ".join(domain_exclusions) if domain_exclusions else "(none given)"

    return f"""You are a qualitative research methodologist specializing in taxonomy development for survey analysis.
Your task is to identify the dimensions on which responses within one domain differ from one another.

{context_block}

{taxonomy_block}

You are working inside ONE domain. Everything you return belongs to it, and nothing
that falls outside it may be returned.

<domain>
Domain: {domain_label}
Definition: {domain_definition}
Boundary test: {domain_boundary_test}
Does NOT belong to this domain (these have their own domains): {exclusions_line}
</domain>

Here are the observations you need to account for:

<observations>
{observations_block}
</observations>

## YOUR TASK

Identify the **dimensions** on which the responses in this domain differ from one
another. Those dimensions are this domain's facets (level 3).

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

For EACH dimension provide, using the facet fields:
- **facet_name** — a short descriptive name for the dimension
- **facet_definition** — one sentence naming what varies along it, with no examples or
  enumerations
- **boundary_test** — one yes/no question that decides whether a response belongs to THIS
  dimension rather than a neighbouring one
- **exclusions** — what does NOT belong, naming the neighbouring dimension it is most
  easily confused with
- **example_observations** — 3-5 observations from the input above, copied exactly

All names, definitions, boundary tests and exclusions must be written in {language}.

Begin processing now and {INSTRUCTOR_HINT}"""


# =============================================================================
# §2 CONSOLIDATION — per domain, across chunks, before any idea is assigned
# =============================================================================

class ConsolidatedFacet(DiscoveredFacet):
    """One facet surviving consolidation, with the candidates it absorbed."""
    source_facets: List[str] = Field(
        ..., description=(
            "The facet_name of every candidate that goes into this one. "
            "A candidate that is kept unchanged lists its own name"
        )
    )


class FacetConsolidationResult(BaseModel):
    """The settled facet inventory for one domain."""
    scratchpad: str = Field(
        ..., description=(
            "Consolidation reasoning: "
            "(1) list the unique candidates across all chunks, "
            "(2) group the ones that overlap conceptually, "
            "(3) name and define each consolidated facet, "
            "(4) for every surviving pair ask whether one response could belong "
            "to both, and merge when it could, "
            "(5) verify the survivors still cover everything the candidates covered, "
            "(6) write each survivor's boundary_test and exclusions"
        )
    )
    facets: List[ConsolidatedFacet] = Field(
        ..., description="The complete facet set for this domain after consolidation"
    )


def _build_candidate_block(candidates: List[DiscoveredFacet], recurrence=None,
                          n_passes: int = 0) -> str:
    """Render the chunk yield as numbered candidates, each with its evidence.

    The observations that produced a proposal travel with it. That is what makes
    this call judgeable: two candidates whose labels differ but whose observations
    are the same thing are a merge, and no amount of staring at labels alone would
    have shown it.
    """
    blocks = []
    for i, candidate in enumerate(candidates, 1):
        exclusions = "; ".join(candidate.exclusions) if candidate.exclusions else "(none)"
        observations = "; ".join(candidate.example_observations)
        seen = recurrence.get(candidate.facet_name) if recurrence else None
        support = (f"     Proposed in {seen} of {n_passes} independent passes\n"
                   if seen and n_passes else "")
        blocks.append(
            f"[C{i}] {candidate.facet_name}\n"
            f"{support}"
            f"     Definition: {candidate.facet_definition}\n"
            f"     Boundary test: {candidate.boundary_test}\n"
            f"     Does not belong: {exclusions}\n"
            f"     Observations that produced this proposal: {observations}"
        )
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
    domain_boundary_test: str,
    candidates: List[DiscoveredFacet],
    recurrence: Optional[Dict[str, int]] = None,
    n_passes: int = 0,
) -> str:
    """Settle one domain's facet inventory, across all chunk proposals.

    Runs before assignment. Each chunk proposed facets without seeing the others,
    so the same concept comes back under several names. This call decides which
    of those are one facet, using the observations each proposal was built on.
    """
    context_block = build_context_block(
        language=language, survey_question=survey_question, sector=sector,
        entity=entity, topic=topic, perspective=perspective, intent=intent,
    )
    taxonomy_block = build_taxonomy_block(
        dimension=dimension, dimension_name=dimension_name,
        dimension_description=dimension_description,
    )
    diagnostic = level_diagnostic(dimension, "facet")
    candidate_block = _build_candidate_block(candidates, recurrence, n_passes)

    return f"""You are a taxonomy consolidation specialist for survey coding.
Your task is to merge facet proposals from several independent passes over one domain
into a single, coherent set of facets.

{context_block}

{taxonomy_block}

You are working inside ONE domain. Every facet you return belongs to it.

<domain>
Domain: {domain_label}
Definition: {domain_definition}
Boundary test: {domain_boundary_test}
</domain>

The question every facet must answer under this lens is:

<facet_diagnostic>
{diagnostic}
</facet_diagnostic>

Here are the candidate facets. Each pass saw a different sample of the responses and
did not see the other passes, so the same facet may appear several times under
different names. Each candidate carries the observations that produced it — those are
your evidence:

<candidates>
{candidate_block}
</candidates>

## YOUR TASK

Consolidate these candidates into the fewest mutually exclusive facets needed for full
coverage.

Judge the candidates on their observations, not on their labels. Two labels that read
differently but were produced by the same kind of observation are ONE facet. Two labels
that read alike but were produced by different observations are TWO.

Consolidation is the goal: do NOT keep every concept separate — group. Govern the
grouping by these rules, in this order of precedence.

**1. DIMENSION FIRST — orthogonality is the guardrail.**
For each candidate, determine WHICH underlying dimension it answers. Candidates on
DIFFERENT dimensions are orthogonal: never merge them into one facet. Mutually exclusive
values or poles of the SAME dimension are also kept apart — merging opposite poles creates
an empty container. Do NOT create separate facets based only on the object discussed when
the same underlying value applies; an object is not a dimension.

**2. PREVALENCE SETS GRANULARITY — within one dimension only.**
Each candidate shows in how many independent passes it was proposed. A candidate that
recurs across passes is well supported: it keeps its own facet. Several thin,
same-dimension candidates are GROUPED into one facet that still names the shared value in
plain language. Prevalence decides how finely to split WITHIN a dimension; it never
licenses merging ACROSS dimensions.

**3. LIFT, DON'T FLATTEN.**
When grouping is needed, raise the concepts to a shared higher-abstraction label that
still carries their meaning — not a label that merely names the axis they sit on.
FORBIDDEN: an empty container that only names the dimension it sorts on. REQUIRED: a
stateable value a reader can picture. Read the label alone — if it tells you only which
question was asked, it is a container; if it tells you what the answer was, it is a value.

**4. PLAIN, MEANINGFUL LABELS.**
Name every surviving facet in everyday language. Reading the label alone, and knowing the
survey question, a layperson knows which distinction is meant. No jargon, no
nominalizations, no dimension-names.

**5. STAY INSIDE THE SCOPE.** A candidate that falls outside the boundary stated above is
not a facet to keep; leave it out rather than widening the scope to fit it.

**6. MAINTAIN FULL COVERAGE.** The survivors must collectively cover everything the
candidates covered. Consolidating is not discarding.

**Precedence when rules conflict: 1 (orthogonality) > 2 (prevalence grouping) > 4 (label
clarity).**

Two things are FORBIDDEN in what you return:

- **FORBIDDEN: facets that overlap conceptually, semantically or in meaning**, judged in
  terms of the reactions and answers people gave to the survey question. Not in the
  abstract — in terms of what respondents actually said.
- **FORBIDDEN: any pair that fails the researcher's test.** Picture a researcher reading
  your final set and saying: *"these two do not actually help me organise meaningfully
  different reactions to this question — they essentially mean the same thing, or they
  overlap so heavily in meaning that the split buys me nothing."* If a pair invites that
  sentence, it is one facet. Merge it.

Every candidate you consume must be listed in the `source_facets` of the facet that
consumes it. A candidate you do not list is left standing as it is, so list them.

{UNIVERSAL_RULES}

## OUTPUT

Work through the consolidation in the scratchpad field first.

For EACH consolidated facet provide:
- **facet_name** — a short descriptive name
- **facet_definition** — one sentence naming a single aspect, no examples or enumerations
- **boundary_test** — one yes/no question that decides membership
- **exclusions** — what does NOT belong, naming the neighbouring facet it is most easily
  confused with
- **example_observations** — 2-5 observations, copied exactly from the candidates above
- **source_facets** — the facet_name of every candidate consumed into this one

All facet names, definitions, boundary tests and exclusions must be written in {language}.

Begin processing now and {INSTRUCTOR_HINT}"""


# =============================================================================
# §3 ASSIGNMENT — ideas into the settled inventory
# =============================================================================
#
# Two things this phase deliberately does NOT carry, unlike the other three.
# The taxonomy block and the universal rules are both about what a taxonomy may
# contain; assignment creates nothing, it picks an id from a menu whose entries
# already carry their definitions and boundaries. This is also the only phase
# that runs at the volume of the dataset, so every line here is paid thousands
# of times.


def build_facet_menu(facets: List[ConsolidatedFacet]) -> str:
    """Render the settled facets as a numbered menu.

    The [F#] id is what the response is keyed on, so the numbering here and the
    id list handed to `build_facet_assignment_model` must come from the same
    list in the same order.
    """
    lines = []
    for i, f in enumerate(facets, 1):
        exclusions = "; ".join(f.exclusions) if f.exclusions else ""
        examples = "; ".join(f.example_observations[:3])
        block = (
            f"[F{i}] {f.facet_name}\n"
            f"     Description: {f.facet_definition}\n"
            f"     Boundary: {f.boundary_test}"
        )
        if exclusions:
            block += f"\n     Does not belong here: {exclusions}"
        if examples:
            block += f"\n     Examples: {examples}"
        lines.append(block)
    return "\n\n".join(lines)


def build_facet_assignment_model(facet_ids: List[str], idea_ids: List[str]):
    """Runtime response model for one assignment call.

    Both id spaces are Literals, so a hallucinated facet id or a made-up idea id
    is a schema violation that instructor retries, not a content error that has
    to be caught downstream. "F_NONE" is the honest way out when nothing fits;
    the caller escalates those rather than forcing a wrong home.
    """
    facet_id_literal = Literal[tuple(facet_ids + ["F_NONE"])]  # type: ignore[valid-type]
    idea_id_literal = Literal[tuple(idea_ids)]  # type: ignore[valid-type]

    item_model = create_model(
        "FacetAssignmentItem",
        idea_id=(idea_id_literal, Field(
            ..., description="The [id] tag of the idea, echoed exactly")),
        assigned_facet_id=(facet_id_literal, Field(
            ..., description=(
                "The facet id from the [F#] prefix. Return ONLY the id. "
                "Use F_NONE when no facet fits this idea"))),
        confidence=(float, Field(
            ..., ge=0.0, le=1.0, description="Assignment confidence (0.0-1.0)")),
        valence=(Literal["+", "-", "0"], Field(
            default="0",
            description=(
                "Evaluative direction relative to the facet: "
                "+ positive, - negative, 0 neutral"))),
    )
    return create_model(
        "FacetAssignmentResult",
        assignments=(List[item_model], Field(
            ..., description=(
                "Exactly one assignment per idea listed in the prompt, "
                "no idea skipped, no idea added"))),
    )


def build_facet_assignment_prompt(
    *,
    language: str,
    survey_question: str,
    sector: str,
    entity: str,
    topic: str,
    perspective: str,
    intent: str,
    domain_label: str,
    domain_definition: str,
    facets: List[ConsolidatedFacet],
    ideas: List[Tuple[str, str]],
) -> str:
    """Assign one or more ideas to a facet, with valence.

    One builder for one or many ideas: a single idea is a list of length one.
    A separate single-idea variant would be a second prompt for one task, and
    two prompts for one task drift apart.
    """
    context_block = build_context_block(
        language=language, survey_question=survey_question, sector=sector,
        entity=entity, topic=topic, perspective=perspective, intent=intent,
    )
    menu = build_facet_menu(facets)
    ideas_block = "\n".join(f"[{idea_id}] {label}" for idea_id, label in ideas)

    return f"""You are a qualitative coding assistant. Assign each survey response idea below to the facet it belongs to.

{context_block}

<domain>
Domain: {domain_label} — {domain_definition}
</domain>

<facets>
{menu}

[F_NONE] None of the facets above fits the idea.
</facets>

<ideas>
{ideas_block}
</ideas>

### VALENCE (evaluation relative to the facet)
- "+" Positive — the idea describes the facet as met, present, or enhanced
- "-" Negative — the idea describes the facet as failing, absent, or detracted from
- "0" Neutral — the idea is descriptive, ambiguous, or expresses no evaluation

Valence is not emotional sentiment. It is evaluative direction relative to the facet.

Use each facet's Boundary line to decide the doubtful cases; that is what it is for.

Judge every idea independently on its own text; do not let one assignment influence the
next. Return exactly one item per idea, echoing that idea's [id]. Do not skip ideas and
do not add ideas. If no facet fits an idea, use "F_NONE" for that idea rather than
forcing it into the nearest one.

Begin processing now and {INSTRUCTOR_HINT}"""


# =============================================================================
# §4 REFINEMENT — per domain, after every idea has been assigned
# =============================================================================

class FacetMisfitGroup(BaseModel):
    """A group of ideas sitting in a facet they do not belong to."""
    from_facet: str = Field(..., description="The facet currently holding them")
    instance_texts: List[str] = Field(
        ..., description=(
            "The exact response texts, copied verbatim from the contents shown. "
            "Never counts, paraphrases or summaries"
        )
    )
    verdict: Literal["move", "out"] = Field(
        ..., description=(
            "'move' when these ideas belong to a named existing facet; "
            "'out' when they carry no substantive content at all"
        )
    )
    target_facet: str = Field(
        default="", description="For 'move': the facet they belong to. Empty for 'out'"
    )
    reason: str = Field(..., description="One sentence on why they do not belong")


class RefinedFacet(ConsolidatedFacet):
    """One facet surviving refinement. Its domain is fixed by the task."""
    action: Literal["keep", "merge", "widen", "split"] = Field(
        ..., description=(
            "keep = unchanged; merge = several sources into this one; "
            "widen = same facet, description restated to cover what it holds; "
            "split = one source divided into named children (instance_texts required)"
        )
    )
    instance_texts: List[str] = Field(
        default_factory=list,
        description=(
            "For action 'split' ONLY: the exact response texts routed to this child, "
            "copied verbatim. Required when a source facet is divided over more than "
            "one returned facet. Empty otherwise"
        )
    )


class FacetRefinementResult(BaseModel):
    """Final facet inventory for one domain, plus the misfits found in it."""
    scratchpad: str = Field(
        ..., description=(
            "Step-by-step reasoning: (1) read each facet's contents against its label "
            "and note groups that do not belong, (2) group facets by the underlying "
            "distinction each one answers, (3) set granularity by prevalence using the "
            "shares shown, (4) for every surviving pair, ask whether their contents can "
            "be told apart without reading the labels, and merge the pair where they "
            "cannot, (5) route each non-fitting group to one of the four exits, "
            "(6) check every label states a value rather than the question, "
            "(7) assemble the final inventory"
        )
    )
    facets: List[RefinedFacet] = Field(
        ..., description="The complete facet set for this domain after refinement"
    )
    misfits: List[FacetMisfitGroup] = Field(
        default_factory=list,
        description="Groups of ideas that do not belong where they sit",
    )

    @model_validator(mode="after")
    def _routable(self):
        """Reject an inventory whose ideas cannot be routed.

        Enforced in the schema rather than in the prompt for the same reason the
        domain is absent from this model: a rule the model can decline to follow
        is not a rule. instructor surfaces these messages and retries, so the
        model gets to correct itself instead of silently producing an answer
        whose ideas have nowhere to go.
        """
        for f in self.facets:
            if f.action == "split" and not f.instance_texts:
                raise ValueError(
                    f'facet "{f.facet_name}" has action "split" but no instance_texts. '
                    f'A split must list the exact response texts routed to each child, '
                    f'or the ideas cannot be divided.'
                )

        claimed_by: Dict[str, List[str]] = {}
        for f in self.facets:
            for src in (f.source_facets or []):
                claimed_by.setdefault(src, []).append(f.facet_name)

        for src, claimants in claimed_by.items():
            if len(claimants) < 2:
                continue
            without_texts = [f.facet_name for f in self.facets
                             if src in (f.source_facets or []) and not f.instance_texts]
            if without_texts:
                raise ValueError(
                    f'source facet "{src}" is claimed by {len(claimants)} returned '
                    f'facets ({", ".join(claimants)}), but {", ".join(without_texts)} '
                    f'give no instance_texts. Either let ONE facet take "{src}", or '
                    f'make every claimant action "split" and list the exact response '
                    f'texts each one takes.'
                )
        return self


def build_facet_contents_block(
    rows: List[Tuple[str, int, float, List[str]]],
) -> str:
    """Render what each facet actually holds: name, count, share, real texts.

    `rows`: (facet_name, n_ideas, share_of_domain, sample_texts).

    The share is what makes granularity judgeable — "thin" and "large" only mean
    something relative to the siblings, never against an absolute number.
    """
    blocks = []
    for name, n_ideas, share, texts in rows:
        contents = "\n".join(f"       - {t}" for t in texts)
        blocks.append(
            f"{name} — {n_ideas} responses ({round(share * 100)}% of the domain)\n"
            f"     Contents:\n{contents}"
        )
    return "\n\n".join(blocks)


def build_facet_refinement_prompt(
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
    facets_block: str,
) -> str:
    """Settle one domain's facets against what they actually ended up holding.

    The one phase that can only run after assignment: real counts and real texts
    do not exist before it. The domain is fixed, so nothing here can move a facet
    out of it — when a group of ideas belongs elsewhere, the IDEAS move and the
    structure stays put.
    """
    context_block = build_context_block(
        language=language, survey_question=survey_question, sector=sector,
        entity=entity, topic=topic, perspective=perspective, intent=intent,
    )
    diagnostic = level_diagnostic(dimension, "facet")
    # What "no substantive content" means depends on the dimension. Step 3 words
    # it per dimension in `contentless_test`, which is exactly this diagnostic —
    # not `standing_not_known.short`, which says something else entirely (the
    # respondent does not know the subject, which IS substantive).
    contentless = dimension.prompt_rules.contentless_test

    return f"""You are a qualitative research analyst specializing in open-ended survey coding.
Your task is to settle the final facet set of ONE domain, now that every response has been assigned to a facet.

{context_block}

<domain>
Domain: {domain_label} — {domain_definition}
</domain>

The question every facet answers under this lens is:

<facet_diagnostic>
{diagnostic}
</facet_diagnostic>

Here are this domain's facets, each with the number of responses actually assigned to
it, its share of the domain, and a sample of the responses it really holds:

<facets>
{facets_block}
</facets>

Judge each facet on what it actually holds, not on how its label reads. The counts and
the response texts above are the evidence; the labels were written before a single
response had been assigned.

<refinement_rules>
**1. DISTINCTION FIRST.** Facets that answer different distinctions stay apart, however
similar their labels look. Orthogonality is a guardrail against merging, never a reason
to merge.

**2. PREVALENCE SETS GRANULARITY** — within one distinction only. Each facet shows its
share of the domain. Judge size relative to its siblings, never against an absolute
number. The largest keep their own identity. Those far below their siblings do not
survive on their own: group them with the sibling whose distinction they share, under a
label that names the shared value. A facet holding a large share AND visibly diverse
contents is too abstract: split it, do not widen it.

**3. LIFT, DON'T FLATTEN.** When several thin facets share a distinction, name the
concept they share. Do not dissolve them into a catch-all.

**4. TWO LABELS, ONE THING.** Before routing anything, read the contents of each pair
against each other. Where you cannot tell which of two facets a response belongs to
without reading the labels, they are not two facets: return ONE, listing both in
`source_facets`. The labels were written before a single response had been assigned;
what each one actually caught is the evidence that they turned out to name the same
thing. This is the only phase that can see that, and a pair left standing here is left
standing for good.

**5. PLAIN, MEANINGFUL LABELS.** A facet name states a value, not the question it
answers. Read the label alone: if it tells you only which question was asked, it is a
container; if it tells you what the answer was, it is a value.

**6. THE DOMAIN IS FIXED.** Every facet you return belongs to this domain. You cannot
move a facet to another domain, and you cannot create a facet that belongs to another
domain. If a GROUP OF IDEAS belongs elsewhere, report it under `misfits` — the ideas
move, the structure stays here.

**7. FOUR EXITS FOR WHAT DOES NOT FIT.** For a group of responses sitting in a facet it
does not belong to:
   - the group points at ONE existing facet
       -> `misfits`, verdict "move": name the target and the EXACT response texts
   - the group is one coherent concept that has no facet yet
       -> action "split": name the children and which EXACT texts go to each
   - the group is diverse but genuinely related to this facet
       -> action "widen": restate the description so it honestly covers what is there
   - the group carries NO SUBSTANTIVE CONTENT WHATSOEVER — filler, or {contentless}
       -> `misfits`, verdict "out"
   "Out" is not an escape hatch for "this does not fit the facets I chose". A text that
   names something real HAS substance: if it has no home yet, create one with "split".
   Moves and splits must be expressed as EXACT response texts copied from the contents
   shown above — never as counts, paraphrases or summaries.

**8. ONE SOURCE, ONE DESTINATION** — unless you route by text. Every facet in the input
must end up in exactly ONE returned facet. To divide one input facet over two returned
facets, use action "split" for each part and list its exact texts in `instance_texts`.

**9. NOTHING THE RESPONSES SAY MAY DISAPPEAR.** Grouping is not discarding: a value that
moves under a shared label is still reported, a value sent "out" is gone. Never use
"out", or a label that silently drops what it absorbed, to make the inventory tidier.
But being a real value is not on its own a reason to stand alone — where responses are
few against their siblings, their honest home is a shared label that still names what
they say, not a facet of their own. Collapsing the domain to a single facet removes a
whole level of the hierarchy: do that only when the contents genuinely express one value.
</refinement_rules>

{UNIVERSAL_RULES}

## OUTPUT

Work through your reasoning in the scratchpad field first.

For EACH surviving facet provide: action, facet_name, facet_definition, boundary_test,
exclusions, example_observations (exact text from the contents), source_facets, and —
for "split" only — instance_texts.

`action` is exactly one of: "keep" (unchanged), "merge" (several sources into this one),
"widen" (description restated to cover what it holds), "split" (one source divided into
named children). Every misfit group carries verdict "move" or "out".

All facet names, definitions, boundary tests and exclusions must be written in {language}.
Copy response texts verbatim when you route them; they are matched literally.

Begin processing now and {INSTRUCTOR_HINT}"""
