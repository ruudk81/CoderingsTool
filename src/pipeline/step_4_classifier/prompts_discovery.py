"""Discovery: facetten en de attributen die ze bevatten, in één beurt.

De twee lagen werden hiervoor apart uitgevraagd — eerst facetten per domein,
dan attributen per facet. Dat kostte een call per facet per chunk, en het gaf
de attribuutlaag een scope die al vastlag voordat er één idee was toegewezen.

Hier vraagt één call per (domein, chunk) allebei tegelijk: welke facetten zitten
er in deze observaties, en welke attributen zitten er in elk facet. Een model
dat beide niveaus in één blik ziet kan een attribuut niet in het verkeerde facet
hangen, want het bepaalt ze samen.

Het instructieskelet komt uit de opzet van vóór de herbouw: genummerde
scratchpad-stappen, "the fewest that provide full coverage", en een expliciete
distinctheidstoets per paar. Nieuw is stap 6, die per overgebleven facet de
attributen benoemt.

Er wordt **niet** naar dimensies of assen gevraagd. De prompt vraagt naar
facetten en attributen in die woorden; `dimension_data.py` levert wat die
niveaus betekenen voor de dimensie waaronder deze run draait.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

from pydantic import BaseModel, Field

from .prompts_shared import (
    INSTRUCTOR_HINT,
    UNIVERSAL_RULES,
    _extract_definition,
    _extract_key_idea,
    build_context_block,
    build_taxonomy_block,
)

if TYPE_CHECKING:
    from pipeline.step_3_ideaExtractor.dimension_data import DimensionDefinition


# =============================================================================
# RESPONSE MODEL
# =============================================================================
#
# Elk veld hieronder wordt in de prompt bij naam genoemd, en de prompt vraagt
# geen veld dat hier niet staat. Twee registers die uit elkaar lopen leveren of
# een verwarde prompt of een antwoord dat we niet kunnen gebruiken.

class DiscoveredAttribute(BaseModel):
    """Een attribuut (L4), zoals één chunk het voorstelt."""
    attribute_name: str = Field(
        ..., description="Short descriptive name for the attribute (2-5 words)")
    attribute_definition: str = Field(
        ..., description=(
            "What this attribute captures — one concrete, observable property, "
            "in 1-2 sentences"))
    example_observations: List[str] = Field(
        ..., description=(
            "2-3 representative observations from the input, using the exact "
            "observation text"))


class DiscoveredFacet(BaseModel):
    """Een facet (L3) mét de attributen die eronder vallen."""
    facet_name: str = Field(
        ..., description="Short descriptive name for the facet (2-5 words)")
    facet_definition: str = Field(
        ..., description=(
            "What this facet captures — one clear underlying concept, in 1-2 "
            "sentences"))
    attributes: List[DiscoveredAttribute] = Field(
        ..., description=(
            "The attributes that fall under this facet. The fewest that cover "
            "the observations assigned to it"))


class DiscoveryResult(BaseModel):
    """Wat één (domein, chunk)-call teruggeeft."""
    scratchpad: str = Field(
        ..., description=(
            "Step-by-step reasoning before the final output: "
            "(1) cluster the observations by shared descriptive meaning, "
            "(2) name candidate facets and the observations supporting each, "
            "(3) check internal coherence — one clear concept per facet, "
            "(4) check distinctness — every pair ontologically distinct and "
            "semantically separable, "
            "(5) check the domain boundary and drop what belongs elsewhere, "
            "(6) for each surviving facet, name the attributes it holds, "
            "(7) prepare the final output"))
    facets: List[DiscoveredFacet] = Field(
        ..., description=(
            "The facets found in these observations, each with its attributes"))


class ConsolidatedAttribute(DiscoveredAttribute):
    """Een attribuut ná consolidatie, dat zegt wat erin is opgegaan."""
    source_attributes: List[str] = Field(
        ..., description=(
            "Every candidate attribute name that folded into this one, "
            "including its own if it survived unchanged"))


class ConsolidatedFacet(DiscoveredFacet):
    """Een facet ná consolidatie, dat zegt wat erin is opgegaan."""
    source_facets: List[str] = Field(
        ..., description=(
            "Every candidate facet name that folded into this one, including "
            "its own if it survived unchanged"))
    attributes: List[ConsolidatedAttribute] = Field(
        ..., description=(
            "The consolidated attributes of this facet, pooled from every "
            "candidate that folded into it"))


class ConsolidationResult(BaseModel):
    """Wat één consolidatiecall per domein teruggeeft.

    De twee `source_*`-velden zijn geen boekhouding maar een vangnet. Zonder
    hen ziet een kandidaat die is samengevoegd er precies zo uit als een
    kandidaat die is vergeten: allebei staan ze niet in het antwoord. Mét hen
    blijft wat niemand claimt staan (`kept_unclaimed`) in plaats van stil te
    verdwijnen — en bij consolidatie in rondes telt dat dubbel, want wat in
    ronde 1 wegvalt komt nooit meer terug.

    `raw_facets` bewaart daarnaast de stand vóór de merge, maar dat dient een
    ander doel: diagnose achteraf, niet detectie tijdens de run.
    """
    scratchpad: str = Field(
        ..., description=(
            "Step-by-step reasoning before the final output: "
            "(1) scan the candidate facets from all passes, "
            "(2) group the ones that mean the same thing, "
            "(3) apply the same-question test to every pair, "
            "(4) let prevalence set the granularity within one question, "
            "(5) verify the domain boundary, "
            "(6) for each surviving facet, pool and consolidate the attributes "
            "of everything that folded into it, "
            "(7) check that every candidate is accounted for, then output"))
    facets: List[ConsolidatedFacet] = Field(
        ..., description=(
            "The fewest mutually exclusive facets that cover the domain, each "
            "with its consolidated attributes"))


# =============================================================================
# PROMPT — DISCOVERY
# =============================================================================

def _exclusion_lines(domain_label: str, boundary_test: str,
                     exclusions: Optional[List[str]]) -> str:
    """De domeingrens, of niets wanneer step 3 er geen heeft meegegeven."""
    lines = []
    if boundary_test:
        lines.append(f"Boundary test: {boundary_test}")
    if exclusions:
        lines.append(
            "This domain EXCLUDES the following, which belong to other domains: "
            + "; ".join(exclusions))
    if not lines:
        return ""
    return "\n" + "\n".join(lines)


def build_discovery_prompt(
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
    domain_exclusions: Optional[List[str]],
    observations: List[str],
) -> str:
    """Facetten én hun attributen, uit één chunk observaties binnen één domein."""
    rules = dimension.prompt_rules
    facet_definition = _extract_definition(rules.facet_instruction)
    attribute_definition = _extract_definition(rules.attribute_instruction)
    facet_key_idea = _extract_key_idea(rules.facet_instruction)
    attribute_key_idea = _extract_key_idea(rules.attribute_instruction)

    observations_block = "\n".join(
        f"{i}. {obs}" for i, obs in enumerate(observations, 1))
    boundary_block = _exclusion_lines(
        domain_label, domain_boundary_test, domain_exclusions)
    exclusion_hint = (
        "\n".join(f"- {x}" for x in domain_exclusions)
        if domain_exclusions else "- (no neighbouring domains were named)")

    return f"""You are a qualitative research analyst specializing in survey response analysis.
Your task is to identify the fewest recurring facets that provide full coverage of a set of
observations within one domain, and for each facet the fewest attributes that provide full
coverage of what that facet holds.

You do both levels in one pass, because they constrain each other: a facet is only a good
facet if the attributes under it are genuinely the same kind of thing.

# What a facet is

{facet_definition}
Key idea: {facet_key_idea}

A facet must be:
- Descriptive and data-grounded, never evaluative
- Internally coherent — one clear underlying concept
- Externally distinctive — ontologically distinct and semantically separable from the others
- Strictly within the domain boundary
- Supported by repeated patterns across observations, not by a single response

# What an attribute is

{attribute_definition}
Key idea: {attribute_key_idea}

An attribute must:
- Name a specific observable property, not repeat a verbatim span from a response
- Be descriptive and non-evaluative
- Stay strictly within the facet it sits under
- Be internally coherent and ontologically distinct from its siblings
- Add unique conceptual value — no attribute that restates another in different words

{build_context_block(
    language=language, survey_question=survey_question, sector=sector,
    entity=entity, topic=topic, perspective=perspective, intent=intent)}

{build_taxonomy_block(
    dimension=dimension, dimension_name=dimension_name,
    dimension_description=dimension_description)}

You are working within this domain, and only within it:

<taxonomy_domain>
{domain_label} — {domain_definition}{boundary_block}
</taxonomy_domain>

Here are the observations you need to analyze:

<observations>
{observations_block}
</observations>

# Instructions

Work through these steps in the `scratchpad` field before writing your final output.

**Step 1: Cluster the observations**
Group observations that share descriptive meaning. Identify what recurs. Focus on the kind
of quality, characteristic, practice or property being described — not on whether it is
being praised or criticised.

**Step 2: Name candidate facets**
From your clusters, name candidate facets. For each one note the name, the underlying
concept it captures, and which observations support it.
A facet names a recurring kind of meaning, not a single concrete observation.

**Step 3: Verify internal coherence**
For each candidate facet, check that it captures one clear concept. Reject or split any
candidate that combines different kinds of phenomena, mixes description with evaluation,
or is so broad that a coder could not apply it.

**Step 4: Verify distinctness**
Check every pair of candidate facets. They must be ontologically distinct — neither is a
subset of the other, and they do not overlap in conceptual space — and semantically
separable, so a coder always knows which one applies. If a pair fails, either merge them
into one broader facet or redraw the boundary between them.

**Step 5: Verify the domain boundary**
Every retained facet must fall strictly inside {domain_label}. Drop anything that belongs
more naturally to a neighbouring domain, including:
{exclusion_hint}

**Step 6: Name the attributes inside each facet**
For each facet that survived, look again at the observations you assigned to it and name
the attributes it holds — the specific properties those observations point at. Apply the
same two tests one level down: each attribute captures one property, and no two attributes
under the same facet overlap or restate each other.
If a facet turns out to hold only one attribute, that is a sign the facet and the attribute
are the same thing. Decide which level the concept really belongs to and keep it there.

**Step 7: Prepare the final output**
Keep only what passed every check. Use the fewest facets that cover the observations, and
under each the fewest attributes that cover what it holds.

# Output

Return a JSON object with these fields:
- `scratchpad`: your reasoning for steps 1-7
- `facets`: an array, one entry per facet, each with:
  - `facet_name`: a short descriptive name in {language} (2-5 words)
  - `facet_definition`: what the facet captures, in {language} (1-2 sentences)
  - `attributes`: an array, one entry per attribute inside that facet, each with:
    - `attribute_name`: a short descriptive name in {language} (2-5 words)
    - `attribute_definition`: the observable property it captures, in {language} (1-2 sentences)
    - `example_observations`: 2-3 observations from the input, using the exact observation text

All names, definitions and examples must be written in {language}.

{UNIVERSAL_RULES}

{INSTRUCTOR_HINT}"""


# =============================================================================
# PROMPT — CHUNK CONSOLIDATION
# =============================================================================

def build_candidate_block(
    candidates: List[DiscoveredFacet],
    recurrence: Dict[str, int],
    n_passes: int,
) -> str:
    """De kandidaten van alle chunks, elk met zijn attributen en zijn dekking.

    Consolidatie draait vóór er één idee is toegewezen, dus er zijn geen
    aantallen. Wat er wél is: in hoeveel onafhankelijke chunks een facet is
    voorgesteld. Een begrip dat in vijf van vijf passes terugkomt is beter
    onderbouwd dan een dat één keer opdook, en dat is zichtbaar te maken zonder
    één toewijzing.

    `dedup_exact_facets` klapt byte-identieke namen ervóór samen, dus de telling
    moet apart worden meegedragen — anders verdwijnt precies dit signaal.
    """
    blocks = []
    for i, facet in enumerate(candidates, 1):
        seen = recurrence.get(facet.facet_name, 1)
        lines = [f"[{i}] {facet.facet_name} — Proposed in {seen} of "
                 f"{n_passes} independent passes",
                 f"    Definition: {facet.facet_definition}"]
        if facet.attributes:
            lines.append("    Attributes proposed inside it:")
            for attribute in facet.attributes:
                example = (attribute.example_observations or [""])[0]
                lines.append(
                    f"      - {attribute.attribute_name}: "
                    f"{attribute.attribute_definition}")
                if example:
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
    """Alle chunk-opbrengsten van één domein tot één geneste inventaris.

    Dit is de zwaarste fase van de stap. Elke chunk zag maar een deel van het
    domein en stelde onafhankelijk voor, dus hetzelfde begrip komt onder
    meerdere namen terug — op twee niveaus tegelijk. Facetten die samengaan
    brengen hun attributen mee, en die moeten daarna onderling opnieuw langs
    dezelfde meetlat.
    """
    rules = dimension.prompt_rules
    facet_definition = _extract_definition(rules.facet_instruction)
    attribute_definition = _extract_definition(rules.attribute_instruction)
    exclusion_hint = (
        "\n".join(f"- {x}" for x in domain_exclusions)
        if domain_exclusions else "- (no neighbouring domains were named)")

    return f"""You are a taxonomy consolidation specialist for surveys.
Your task is to merge the facets proposed by several independent passes over one domain
into a single minimal set, and to do the same for the attributes those facets hold.

Each pass saw only part of the domain and proposed on its own, so the same concept comes
back under different names — at both levels at once. That is what you are resolving.

# What a facet is

{facet_definition}

# What an attribute is

{attribute_definition}

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
- Mutually exclusive ANSWERS to the SAME question are also kept apart; merging opposite
  answers creates a container that says nothing.
- Do not create separate facets based only on the object being discussed when the same
  underlying answer applies. An object is not a question.

**2. PREVALENCE SETS GRANULARITY (within one question only).**
- A concept that many passes proposed keeps its own facet — never dissolve a
  well-supported one.
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
Name every surviving facet and attribute in everyday language. Test: reading the label
alone, and knowing the survey question, a layperson knows which distinction is meant. No
jargon, no nominalisations.

**Precedence when rules conflict:** 1 (orthogonality) > 2 (prevalence grouping) > 4 (label
clarity).

# Step-by-Step Analysis Process

Work through these steps in the `scratchpad` field before writing your final output.

**Step 1 — Scan the candidates**
Read every candidate facet from every pass. Note recurring concepts, near-duplicates, and
obvious repeats under different names.

**Step 2 — Group overlapping facets**
Group the facets that describe the same or overlapping concept across passes.

**Step 3 — Apply the same-question test**
For each pair of candidate groups, ask: do these answer the SAME underlying question, or
different ones? Different questions, or opposite answers to one question, stay separate.
Same question and same meaning: group.

**Step 4 — Let prevalence set the granularity**
Within one question, a well-supported concept keeps its own facet; several thinly supported
ones are grouped under a single plainly named facet. Never group across questions.

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
Then check the result against its facet: every attribute must sit inside the facet it hangs
under. If one does not, move it to the facet where it belongs, or drop it if no facet fits.
A facet left holding a single attribute means the facet and the attribute are the same
concept — keep it at the level where it belongs and do not state it twice.

**Step 7 — Account for every candidate**
Confirm you have the minimal set of facets that covers the domain, each holding the minimal
set of attributes that covers what it contains.
Then check coverage: every candidate facet you were given must appear in the `source_facets`
of exactly one surviving facet, and every attribute proposed inside those candidates must
appear in the `source_attributes` of exactly one surviving attribute. A candidate you
deliberately dropped is not exempt — fold it into whichever survivor absorbs its meaning.
Merging and forgetting look identical in the output unless you list what went where.

# Output

Return a JSON object with these fields:
- `scratchpad`: your reasoning for steps 1-7
- `facets`: an array, one entry per surviving facet, each with:
  - `facet_name`: a short descriptive name in {language} (2-5 words)
  - `facet_definition`: what the facet captures, in {language} (1-2 sentences)
  - `source_facets`: the names of every candidate facet that folded into this one,
    exactly as they were given to you. A facet that survived unchanged lists just itself.
  - `attributes`: an array, one entry per surviving attribute in that facet, each with:
    - `attribute_name`: a short descriptive name in {language} (2-5 words)
    - `attribute_definition`: the observable property it captures, in {language} (1-2 sentences)
    - `example_observations`: 2-3 observations carried over from the candidates, exact text
    - `source_attributes`: the names of every candidate attribute that folded into this one,
      exactly as they were given to you. One that survived unchanged lists just itself.

Names and definitions must be written in {language}. The two `source_*` fields are the
exception: they repeat the candidate names verbatim, whatever language those were in.

{UNIVERSAL_RULES}

{INSTRUCTOR_HINT}"""
