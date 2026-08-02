"""
Prompt builders for Taxonomy Classifier (P1-P10).

Pipeline order — discovery, assignment, consolidation, once per level:

  P1   Axis discovery                    build_axis_discovery_prompt
  P2   Facet discovery WITH axes         build_tagged_facet_discovery_prompt
  P3   Facet discovery WITHOUT axes      build_facet_discovery_prompt
  P4   Facet assignment                  build_facet_assignment_prompt_single
  P5   Facet consolidation (in-axis)     build_in_axis_consolidation_prompt
  P6   Attribute discovery               build_attribute_discovery_prompt
                                         build_position_attribute_discovery_prompt
  P7   Attribute assignment              build_attribute_assignment_prompt_single
  P8   Attribute consolidation (in-facet) build_in_facet_consolidation_prompt
  P9   Valence-neutral merge             build_valence_neutral_rename_prompt

P2 and P3 are the only fork: a domain with an axis system takes P2, a domain
without one takes P3. Everything after that is a single route.

Still in the file, outside the numbering — these ran in the pre-consolidation
order (consolidate before assignment, plus a review step) and are scheduled to
go once their dispatch branches are removed:

  build_facet_consolidation_prompt            build_attribute_chunk_consolidation_prompt
  build_segment_consolidation_prompt          build_position_consolidation_prompt
  build_facet_review_prompt                   build_new_position_adjudication_prompt
  build_facet_review_v2_prompt                build_attribute_review_prompt
                                              build_attribute_review_v2_prompt

"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field, model_validator
from pydantic.json_schema import SkipJsonSchema

if TYPE_CHECKING:
    from pipeline.step_3_ideaExtractor.dimension_data import DimensionDefinition


# =============================================================================
# §0 DIMENSION CONTEXT BLOCK — shared helper for all prompts
# =============================================================================

def _norm_text(text: Optional[str]) -> str:
    """Normalise a tag value for matching. Case- and padding-insensitive
    only, mirroring `TaxonomyClassifier._norm_text` (classifier.py) — kept
    as a standalone copy here rather than an import to avoid coupling this
    prompt-builder module to the classifier."""
    return (text or "").strip().lower()


def _extract_definition(instruction: str) -> str:
    """Extract the 'Definition: ...' sentence (up to first newline) from an instruction string."""
    marker = "Definition: "
    idx = instruction.find(marker)
    if idx == -1:
        return instruction.strip()
    rest = instruction[idx + len(marker):]
    newline = rest.find("\n")
    if newline != -1:
        rest = rest[:newline]
    return rest.strip()


def _extract_key_idea(instruction: str) -> str:
    """Extract the 'Key idea: ...' sentence from an instruction string."""
    marker = "Key idea: "
    idx = instruction.find(marker)
    if idx == -1:
        return instruction.strip()
    return instruction[idx + len(marker):].strip().rstrip(".")


def _build_exclusion_block(
    items: List[Tuple[str, str]],
    tag_name: str,
) -> str:
    """Build an XML-tagged exclusion block for domains or facets.

    Args:
        items: list of (name, definition) tuples to exclude.
        tag_name: XML tag name, e.g. 'excluded_domains' or 'excluded_facets'.
    """
    if not items:
        return ""
    lines = [f"- {name} -- {definition}" for name, definition in items]
    content = "\n".join(lines)
    return (
        f"\nYou must NOT include {'facets' if tag_name == 'excluded_domains' else 'attributes'} that belong to these excluded {'domains' if tag_name == 'excluded_domains' else 'facets'}:\n"
        f"<{tag_name}>\n{content}\n</{tag_name}>\n"
    )


def _build_exclusion_block_light(
    items: List[Tuple[str, str]],
) -> str:
    """Build a short name-only exclusion list for use in scratchpad steps."""
    if not items:
        return "(none)"
    return "\n".join(f"- {name}" for name, _ in items)


def _build_exclusion__light_block(
    items: List[Tuple[str, str]],
) -> str:
    """Build an exclusion block for domains or facets without XML tags

    Args:
        items: lightweight list of (name, definition) tuples to exclude.
    """
    if not items:
        return ""
    lines = [f"- {name} — {definition}" for name, definition in items]
    content = "\n".join(lines)
    return (
        f"{content}"
    )


# =============================================================================
# §1 AXIS DISCOVERY (P1) — per-domain axis system discovery
# =============================================================================

def build_axis_discovery_prompt(
    *,
    survey_question: str,
    language: str,
    primary_dimension: str,
    noun_phrase: str,
    domain_label: str,
    domain_definition: str,
    domain_boundary_test: str,
    sample_observations: List[str],
) -> str:
    """Discover the axes along which observations in a domain differ (P1a)."""
    observations_block = "\n".join(f"- {obs}" for obs in sample_observations)

    return f"""You are a taxonomy methodologist working on open-ended survey coding. Your task is to identify coordinate axes within a specific domain of survey responses.

This is the language you are working in:

<language>
{language}
</language>

Here is the survey question that was asked:

<survey_question>
"{survey_question}"

answers vary in terms of: {noun_phrase}
</survey_question>

You are analyzing responses within the following domain:

<domain_name>
{domain_label}
</domain_name>

<domain_definition>
{domain_definition}
</domain_definition>

<domain_boundary_test>
{domain_boundary_test}
</domain_boundary_test>

Here is a broad sample of observations from this domain:

<observations>
{observations_block}
</observations>

Your task

<task>
You are identifying coordinate axes, not categories or segments.
An axis must represent a dimension along which observations could vary within the domain, independently of other axes.
If you cannot demonstrate such independence, do not create another axis.
If the data support only one axis, return exactly one axis. Do not decompose one axis into multiple pseudo-axes.
Before returning more than one axis, verify that observations could differ on axis A while sharing the same value on axis B, and differ on axis B while sharing the same value on axis A. If not, merge or drop the axis.
</task>


PROCESS:

Use the scratchpad to:
- Examine the observations for patterns of variation
- Identify potential axes
- Test each potential axis for independence from others
- Provide concrete examples demonstrating independence (if proposing multiple axes)
- Decide on the final number of axes

Then provide your final answer with:
- A clear description of each axis
- The dimension of variation it represents
- If multiple axes: explicit demonstration of their independence using examples from the observations

For each axis you identify, describe:
- The axis name
- What dimension of variation it captures
- The range or types of values observations can take along this axis
- If proposing multiple axes: concrete examples showing how observations vary independently on each axis

Important requirements:
- All output (axis names and descriptions) must be in {language}

Provide your output as valid JSON following the response schema provided.
"""


class DiscoveredAxis(BaseModel):
    """An axis along which observations within a domain differ."""
    axis_name: str = Field(
        ..., description="Short name for the axis"
    )
    axis_description: str = Field(
        ..., description="What independent dimension of variation this axis captures (1-2 sentences)"
    )
    value_range: str = Field(
        ..., description="The range or types of values observations can take along this axis"
    )


class AxisSystemResponse(BaseModel):
    """P1a output: the axis system discovered for a single domain."""
    scratchpad: str = Field(
        ..., description=(
            "Step-by-step reasoning before naming the axes: "
            "(1) examine the observations for patterns of variation, "
            "(2) identify potential axes, "
            "(3) test each potential axis for independence and orthogonality from the others, "
            "(4) give concrete examples demonstrating independence when proposing "
            "more than one axis, "
            "(5) decide on the final number of axes"
        )
    )
    independence_evidence: str = Field(
        default="", description=(
            "When more than one axis is returned: concrete examples from the observations "
            "showing that observations can differ on one axis while sharing the same value "
            "on another, in both directions. Empty when a single axis is returned."
        )
    )
    axes: List[DiscoveredAxis] = Field(
        ..., description="Axes discovered for this domain"
    )


# =============================================================================
# §2  FACET DISCOVERY (P2) — per-domain facet discovery WITH axes pasted in
# =============================================================================

def _build_axis_system_block(axis_system: AxisSystemResponse) -> str:
    """Render a validated axis system as prompt text: one numbered block per
    axis — its name, what it captures, and the values observations can take
    along it. Each block ends with a blank line, so the axes stay visually
    separated when several are shown."""
    return "".join(
        f"Axis {i}: {axis.axis_name}\n"
        f"  What it captures: {axis.axis_description}\n"
        f"  Values along this axis: {axis.value_range}\n\n"
        for i, axis in enumerate(axis_system.axes, 1)
    )


def build_tagged_facet_discovery_prompt(
    *,
    survey_question: str,
    language: str,
    noun_phrase: str,
    domain_label: str,
    domain_definition: str,
    axis_system: AxisSystemResponse,
    chunk_observations: List[str],
) -> str:
    """Discover facets (L3) from a chunk of observations, each proposal tagged
    to exactly one (axis, segment) of the domain's fixed axis system (P1b)."""
    axis_system_block = _build_axis_system_block(axis_system)
    observations_block = "\n".join(f"{i}. {obs}" for i, obs in enumerate(chunk_observations, 1))

    return f"""You are a qualitative research analyst specializing in open-ended survey coding. Your task is to induce the minimal set of facets needed to classify all observations within a specific domain.

This is the language you are working in:

<language>
{language}
</language>

Here is the survey question being analyzed:

<survey_question>
"{survey_question}"

answers vary in terms of: {noun_phrase}
</survey_question>

Here is the domain you are working within:

<domain>
Domain: {domain_label} — {domain_definition}
</domain>

Here is the axis system that defines how responses vary within this domain:

<axis_system>
{axis_system_block}
</axis_system>

Here are the observations you need to classify:

<observations>
{observations_block}
</observations>

Your task is to induce the least number of facets needed to classify all observations within this domain.

Requirements:

1. **Facet only along the provided axes.** Do not introduce facets based on themes outside the axis system. The axis defines the dimension of variation you must capture.

2. **Use the fewest facets possible.** Only create distinct facets when observations differ in the core way specified by the axis. Do not over-differentiate.

3. **Facets must be:**
   - Mutually exclusive (each observation fits in only one facet)
   - Atomically distinct (each facet represents one clear variation type)
   - Meaningfully differentiated (facets capture real differences along the axis)
   - Orthogonal to other domains/facets (don't overlap with distinctions that belong in other domains)

4. **Handle rare or singleton patterns appropriately.** Put rare or singleton patterns into a general/residual facet unless they represent a clearly recurring and axis-relevant distinction that appears multiple times.

5. **Context-dependent responses.** If a response contains a substantive improvement suggestion plus a statement of no further advice, classify only the part relevant to this domain. Note it as context-dependent or general unless it forms a recurring pattern.

Before providing your final facet set, use the scratchpad to:
- Identify the different types of variation the observations show along each axis
- Group observations by similarity along the axis
- Consider whether apparent differences are meaningful enough to warrant separate facets
- Determine the minimal set that captures all meaningful variation

Now provide your final facet set. For each facet, include:

- **Facet name**: A clear, concise label
- **Definition**: A precise description of what this facet captures
- **Inclusion rule**: What types of responses belong in this facet
- **Exclusion rule**: What types of responses do NOT belong (if helpful for clarity)
- **Example observation numbers**: List 3-5 observation numbers that exemplify this facet

After listing all facets, provide:

- **Rationale for minimality**: Explain why this is the minimal facet set needed and why you did not split or merge facets further

Output requirements:
- All output (facet names and descriptions) must be in {language}

Provide your output as valid JSON following the response schema provided.
"""

class FacetProposal(BaseModel):
    """One facet proposed on one of the domain's axes (P1b output)."""
    facet_name: str = Field(
        ..., description="A clear, concise label for this facet"
    )
    facet_definition: str = Field(
        ..., description="A precise description of what this facet captures"
    )
    inclusion_rule: str = Field(
        ..., description="What types of responses belong in this facet"
    )
    exclusion_rule: str = Field(
        default="", description=(
            "What types of responses do NOT belong in this facet — "
            "only when it helps clarify the boundary, otherwise empty"
        )
    )
    example_observations: List[int] = Field(
        ..., description="3-5 observation numbers that exemplify this facet"
    )


def build_tagged_facet_discovery_model(axis_names: List[str]) -> type[BaseModel]:
    """Build the P1b response model for one domain (P1b).

    The domain's axes are already known from P1a, so they are fixed in the
    schema itself: `axis_name` is a Literal over exactly those names. The
    model does not name an axis, it picks one of ours — and adds as many
    facets under it as that axis needs.
    """
    AxisNameLiteral = Literal[tuple(axis_names)]  # type: ignore[valid-type]

    class AxisFacets(BaseModel):
        """The facets proposed on one axis."""
        axis_name: AxisNameLiteral = Field(
            ..., description="The axis these facets sit on"
        )
        facets: List[FacetProposal] = Field(
            ..., description="The minimal set of facets needed on this axis"
        )

    class TaggedFacetDiscoveryResponse(BaseModel):
        """P1b output: facets discovered in a single chunk, grouped per axis."""
        scratchpad: str = Field(
            ..., description=(
                "Reasoning before the final facet set: group the observations by "
                "similarity along each axis, consider whether apparent differences "
                "are meaningful enough to warrant separate facets, and determine "
                "the minimal set that captures all meaningful variation"
            )
        )
        axes: List[AxisFacets] = Field(
            ..., description="One entry per axis, with the facets proposed on it"
        )
        minimality_rationale: str = Field(
            ..., description=(
                "Why this is the minimal facet set needed, and why facets were "
                "not split or merged further"
            )
        )

    return TaggedFacetDiscoveryResponse


# =============================================================================
# §4  FACET DISCOVERY (P3) — per-domain facet discovery WITHOUT axes pasted in
# =============================================================================

def build_facet_discovery_prompt(
    *,
    survey_question: str,
    language: str,
    dataset_context_section: str,
    dimension_def: Optional[DimensionDefinition],
    dimension_name: str,
    dimension_description: str,
    partition_name: str,
    partition_definition: str,
    observations: List[str],
    excluded_domains: Optional[List[Tuple[str, str]]] = None,
    boundary_test: str = "",
    exclusions: Optional[List[str]] = None,
) -> str:
    """Discover facets (L3) from a chunk of observations within a domain."""
    observations_block = "\n".join(f"{i}. {obs}" for i, obs in enumerate(observations, 1))

    _boundary_lines = []
    if boundary_test:
        _boundary_lines.append(f"Boundary test: {boundary_test}")
    if exclusions:
        _boundary_lines.append(
            "This domain EXCLUDES (these belong to other domains): " + "; ".join(exclusions)
        )
    domain_boundary_block = ("\n" + "\n".join(_boundary_lines)) if _boundary_lines else ""

    # Dimension-specific guidance
    if dimension_def:
        rules = dimension_def.prompt_rules
        facet_definition = _extract_definition(rules.facet_instruction)
        facet_key_idea = _extract_key_idea(rules.facet_instruction)
        attribute_key_idea = _extract_key_idea(rules.attribute_instruction)
        noun_phrase = dimension_def.noun_phrase_descriptor
        domain_key_idea = _extract_key_idea(rules.domain_instruction)
    else:
        facet_definition = "A facet identifies the analytical lens through which the domain is being examined."
        facet_key_idea = "the analytical lens applied to the subject"
        attribute_key_idea = "the specific observable property being described"
        noun_phrase = dimension_name
        domain_key_idea = "the subject the statement refers to"

    excluded_block = _build_exclusion_block(
        excluded_domains or [], "excluded_domains"
    )

    excluded_block_light = _build_exclusion__light_block(
        excluded_domains or []
    )

    return f"""You are a qualitative research analyst specializing in open-ended survey coding. Your task is to induce the minimal set of facets needed to classify all observations within a specific domain.

This is the language you are working in:

<language>
{language}
</language>

Here is the survey question being analyzed:

<survey_question>
"{survey_question}"
answers vary in terms of: {noun_phrase}
</survey_question>

Here is the domain you are working within:

<taxonomy_domain>
{partition_name} — {partition_definition}{domain_boundary_block}
</taxonomy_domain>
{excluded_block}

Here are the observations you need to classify:

<observations>
{observations_block}
</observations>

Your task is to induce the least number of facets needed to classify all observations within this domain.

Requirements:

1. **Identify the coordinate axes within this domain.** Find the dimensions along which responses vary orthogonally to each other.

2. **Facet only along the axes you identified.** Do not introduce facets based on themes outside those axes. The axis defines the dimension of variation you must capture.

3. **Use the fewest facets possible.** Only create distinct facets when observations differ in the core way specified by the axis. Do not over-differentiate.

4. **Facets must be:**
   - Mutually exclusive (each observation fits in only one facet)
   - Atomically distinct (each facet represents one clear variation type)
   - Meaningfully differentiated (facets capture real differences along the axis)
   - Orthogonal to other domains/facets (don't overlap with distinctions that belong in other domains)

5. **Handle rare or singleton patterns appropriately.** Put rare or singleton patterns into a general/residual facet unless they represent a clearly recurring and axis-relevant distinction that appears multiple times.

6. **Context-dependent responses.** If a response contains a substantive improvement suggestion plus a statement of no further advice, classify only the part relevant to this domain. Note it as context-dependent or general unless it forms a recurring pattern.

Before providing your final facet set, use the scratchpad to:
- Identify the different types of variation the observations show along each axis
- Group observations by similarity along the axis
- Consider whether apparent differences are meaningful enough to warrant separate facets
- Determine the minimal set that captures all meaningful variation

Now provide your final facet set. For each facet, include:

- **Facet name**: A clear, concise label
- **Definition**: A precise description of what this facet captures
- **Inclusion rule**: What types of responses belong in this facet
- **Exclusion rule**: What types of responses do NOT belong (if helpful for clarity)
- **Example observation numbers**: List 3-5 observation numbers that exemplify this facet

After listing all facets, provide:

- **Rationale for minimality**: Explain why this is the minimal facet set needed and why you did not split or merge facets further

Output requirements:
- All output (facet names and descriptions) must be in {language}

Provide your output as valid JSON following the response schema provided.
"""

class DiscoveredFacet(BaseModel):
    """A facet (L3) discovered from observations within a domain."""
    facet_name: str = Field(
        ..., description="Short descriptive name for the facet (2-5 words)"
    )
    facet_description: str = Field(
        ..., description="What this facet captures — the specific viewpoint or aspect (1-2 sentences)"
    )
    inclusion_rule: str = Field(
        default="", description="What types of responses belong in this facet"
    )
    exclusion_rule: str = Field(
        default="", description=(
            "What types of responses do NOT belong in this facet — only when it "
            "helps clarify the boundary, otherwise empty"
        )
    )
    example_observations: List[str] = Field(
        ..., description="3-5 representative observations from the input"
    )
    boundary_test: SkipJsonSchema[str] = Field(
        default="", description="One routing sentence for the doubtful case, phrased against a named sibling facet"
    )
    axis: SkipJsonSchema[str] = Field(
        default="", description="P1b provenance only: the axis this facet was tagged to (empty on the untagged path)"
    )
    segment: SkipJsonSchema[str] = Field(
        default="", description="P1b provenance only: the segment this facet was tagged to (empty on the untagged path)"
    )
    refinement: SkipJsonSchema[dict] = Field(
        default_factory=dict, description=(
            "P2 axis-first provenance only: this facet's refinement axis as "
            "{name, description, positions} (empty on the untagged path)"
        )
    )


class FacetDiscoveryResult(BaseModel):
    """P1 output: facets discovered in observations."""
    scratchpad: str = Field(
        ..., description=(
            "Step-by-step reasoning before identifying facets: "
            "(1) cluster observations by shared descriptive meaning, "
            "(2) identify candidate facets and assess coherence and distinctness, "
            "(3) verify internal coherence — one clear concept per facet, "
            "(4) verify distinctness — ontologically distinct and semantically separable, "
            "(5) verify domain boundaries — exclude facets belonging to other domains, "
            "(6) prepare final output with only dominant facets that pass all checks"
        )
    )
    facets: List[DiscoveredFacet] = Field(
        ..., description="Facets identified in the observations"
    )


# =============================================================================
# §2 FACET CONSOLIDATION (P2) — merge chunk-level facets into coherent set
# =============================================================================


def build_facet_consolidation_prompt(
    *,
    survey_question: str,
    language: str,
    dataset_context_section: str,
    dimension_def: Optional[DimensionDefinition],
    dimension_name: str,
    dimension_description: str,
    domain_name: str,
    domain_definition: str,
    chunk_results: str,
    excluded_domains: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """Consolidate chunk-level facet discoveries into a single coherent set."""
    # Dimension-specific guidance
    if dimension_def:
        rules = dimension_def.prompt_rules
        facet_definition = _extract_definition(rules.facet_instruction)
        facet_key_idea = _extract_key_idea(rules.facet_instruction)
        attribute_key_idea = _extract_key_idea(rules.attribute_instruction)
        noun_phrase = dimension_def.noun_phrase_descriptor
        domain_key_idea = _extract_key_idea(rules.domain_instruction)
    else:
        facet_definition = "A facet identifies the analytical lens through which the domain is being examined."
        facet_key_idea = "the analytical lens applied to the subject"
        attribute_key_idea = "the specific observable property being described"
        noun_phrase = dimension_name
        domain_key_idea = "the subject the statement refers to"

    excluded_block = _build_exclusion_block(
        excluded_domains or [], "excluded_domains"
    )
    excluded_block_light = _build_exclusion_block_light(
        excluded_domains or []
    )

    return f"""You are a taxonomy consolidation specialist for surveys.
Your task is to merge multiple chunk-level facet analyses into a single, minimal set of mutually exclusive facets within a given domain.

# What is a facet?

{facet_definition} Facets must be:
- Descriptive and data-grounded (not evaluative)
- Internally coherent (one clear underlying concept)
- Externally distinctive (ontologically distinct and semantically separable from other facets)
- Strictly within domain boundaries
- Supported by multiple observations or repeated patterns

# Survey Context

Here is the survey context:

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

Use the survey context to:

<survey_context_usage>
- Interpret the meaning of facets relative to the survey question
- Ensure consolidated facets are directly relevant to what is being asked
- Preserve terminology and phrasing appropriate to the survey language
- Avoid introducing facets that are not grounded in the question intent
</survey_context_usage>

# Taxonomy Context

Here is the taxonomy context that defines your working framework:

<taxonomy_context>
This is the structure:
<taxonomy_structure>
- Dimension (L1): {dimension_name} — {dimension_description}
- Domain (L2): {domain_key_idea}
- Facet (L3): {facet_key_idea}
- Attribute (L4): {attribute_key_idea}
</taxonomy_structure>

And you are working within this domain:
<taxonomy_domain>
{domain_name} — {domain_definition}
</taxonomy_domain>
{excluded_block}
</taxonomy_context>

Pay careful attention to:
- The domain you are working within (this defines the scope of valid facets)
- Any excluded domain (facets belonging to these must be removed)
- The dimension (for broader context)

# Chunk-Level Analyses to Consolidate

Here are the facets discovered from analyzing different chunks of the survey data:

<chunk_level_analyses>
{chunk_results}
</chunk_level_analyses>

# Consolidation Rules

Apply these rules strictly when consolidating facets:

Consolidation is the goal: do NOT keep every concept separate — group. But govern grouping by these rules, in order.

**1. DIMENSION FIRST (orthogonality — the guardrail).**
For each concept, determine WHICH underlying dimension it answers.
- Concepts on DIFFERENT dimensions are orthogonal: NEVER merge them into one facet (e.g. socio-economic class vs political orientation vs age are different dimensions).
- Mutually-exclusive VALUES/POLES of the SAME dimension are also kept apart (e.g. "young" vs "old"); merging opposite poles creates an empty container.
- Do NOT create separate facets based only on the object discussed, when the same underlying value applies — an object is not a dimension.

**2. PREVALENCE SETS GRANULARITY (within a dimension only).**
- A high-count value keeps its own facet — never dissolve a well-supported concept.
- Several thin, same-dimension values are GROUPED into one facet that still names the shared value/contrast in plain language.
Prevalence decides how finely to split WITHIN a dimension; it NEVER licenses merging ACROSS dimensions.

**3. LIFT, DON'T FLATTEN.**
When grouping is needed, raise concepts to a shared higher-abstraction label that still carries their meaning — NOT a label that merely names the axis.
FORBIDDEN: a container that only names the axis it sits on — the reader learns what was being measured, not what was said.
REQUIRED: a label that states the value itself, so the reader knows what the respondents expressed.
Test: read the label alone. If it tells you only which question was asked, it is a container; if it tells you what the answer was, it is a value.

**4. PLAIN, MEANINGFUL LABELS.**
Name every surviving facet in everyday language. Test: reading the label alone, a layperson knows which distinction is meant, given the survey question. No jargon, no nominalizations, no dimension-names.

**Precedence when rules conflict:** 1 (orthogonality) > 2 (prevalence grouping) > 4 (label clarity).

# Step-by-Step Analysis Process

Before providing your final output, work through your analysis systematically in a scratchpad. Follow these steps:

**Step 1 -- Scan chunk-level facets**
Review all facets from all chunks. Note recurring themes, similar concepts, and obvious duplicates.

**Step 2 -- Group overlapping facets**
Identify and group facets that describe the same or overlapping concepts across different chunks.

**Step 3 -- Apply the dimension test**
For each pair of candidate facets, ask: "Do these answer the SAME underlying dimension, or DIFFERENT dimensions?" Different dimensions (or opposite poles of one dimension) → keep separate; same dimension and same meaning → group.

**Step 4 -- Group thin same-dimension facets by prevalence**
Within a dimension, keep high-count values as their own facet; group several thin same-dimension values under one meaningful, plainly-named facet. Never merge across dimensions.

**Step 5 -- Verify domain boundaries**
Ensure each retained facet belongs to the included domain and not to any excluded domain:
{excluded_block_light}

**Step 6 -- Prepare final output**
Confirm you have the minimal set of consolidated facets that pass all checks. Prepare the name, description, and representative observations for each.

# Output Requirements

For each consolidated facet, provide:
- A short descriptive name (2-5 words)
- A description of what the facet captures (1-2 sentences)
- The parent domain name: {domain_name}
- 2-3 representative observations selected from across the merged chunks (exact text)

All facet names and descriptions must be in {language}.

# Final Notes

- Facets must be descriptive, not evaluative
- Facets must be grounded in repeated patterns across observations
- Facets must be internally coherent (one clear concept each)
- Facets must be externally distinctive (no overlap, no subset/superset)
- Facets must remain strictly within the included domain
- All output must be in {language}

Begin by writing your step-by-step analysis in the scratchpad field, then provide your final consolidated facets in valid JSON format"""


class FacetConsolidatedResponse(BaseModel):
    """Consolidated facets after merging chunk-level discoveries."""
    scratchpad: str = Field(
        ..., description=(
            "Step-by-step reasoning before consolidating facets: "
            "(1) scan chunk-level facets for recurring themes and duplicates, "
            "(2) group overlapping facets across chunks, "
            "(3) apply the dimension test — keep different dimensions or opposite poles separate, "
            "(4) group thin same-dimension facets by prevalence into meaningful, plainly-named facets, "
            "(5) verify domain boundaries — exclude facets belonging to other domains, "
            "(6) prepare final minimal set of consolidated facets"
        )
    )
    facets: List[DiscoveredFacet] = Field(
        ..., description="Fewest mutually exclusive facets needed for full coverage, consolidated from all chunks"
    )


# =============================================================================
# §2a SEGMENT FACET CONSOLIDATION (P2, axis-first path) — one consolidation
# task per (axis, segment), grouped from the domain's tagged P1b proposals in
# CODE (not by the model). Produces one facet per segment plus that facet's
# refinement axis. Used only for domains that got a validated axis system
# from P1a; domains without one keep the §2 path above untouched.
# =============================================================================

def _build_segment_proposals_block(proposals: List[DiscoveredFacet]) -> str:
    """Render the facet proposals tagged to one segment as prompt text: one
    '- name: description' line per proposal, followed by its examples."""
    lines = []
    for p in proposals:
        lines.append(f"- {p.facet_name}: {p.facet_description}")
        lines.append(f"  Examples: {'; '.join(p.example_observations[:5])}")
    return "\n".join(lines)


def build_segment_consolidation_prompt(
    *,
    survey_question: str,
    domain_label: str,
    domain_definition: str,
    axis_name: str,
    axis_description: str,
    segment_name: str,
    segment_boundary: str,
    proposals: List[DiscoveredFacet],
) -> str:
    """Consolidate all chunk-level facet proposals tagged to one (axis,
    segment) into a single facet, plus that facet's refinement axis (P2,
    axis-first path)."""
    segment_proposals_block = _build_segment_proposals_block(proposals)

    return f"""You are a taxonomy consolidation specialist for open-ended survey coding.

The survey question:
"{survey_question}"

Domain: {domain_label} — {domain_definition}
Axis: {axis_name} — {axis_description}
Segment under consolidation: {segment_name}
Segment boundary: {segment_boundary}

Below are all chunk-level facet proposals tagged to this segment, with their
examples:

<proposals>
{segment_proposals_block}
</proposals>

Your task:
1. Consolidate these proposals into ONE facet for this segment: one name, one
   description faithful to what the proposals jointly cover. List every
   proposal you consumed under source_proposals (one-for-one bookkeeping).
2. Define this facet's REFINEMENT AXIS: the sub-question along which the
   observations INSIDE this facet differ from each other. Name it, describe
   it, and divide it into 2-6 positions with one-sentence boundaries and 2-5
   verbatim examples each, plus exactly one residual position for
   observations that do not specify a value on it.

Rules:
- Work strictly inside this segment; other segments are consolidated
  separately and are none of your concern.
- Positions are conceptual values, not specificity levels; the residual
  position is where unspecific observations live.
- Descriptive wording only.

Provide your output as valid JSON following the response schema provided.
"""


class RefinementPosition(BaseModel):
    """A position (conceptual value) on a facet's refinement axis (P2 output)."""
    position_name: str = Field(
        ..., description="Short descriptive name for the position — a value on the refinement axis"
    )
    position_description: str = Field(
        ..., description="What this position captures on the refinement axis (1-2 sentences)"
    )
    boundary: str = Field(
        ..., description="One-sentence boundary distinguishing this position from its neighbours"
    )
    example_observations: List[str] = Field(
        ..., description="2-5 example observations for this position, verbatim from the proposals"
    )
    is_residual: bool = Field(
        default=False, description=(
            "True for exactly one position per refinement axis: the residual position for "
            "observations that do not specify a value on it"
        )
    )


class ConsolidatedFacet(BaseModel):
    """P2 output (axis-first path): one consolidated facet for a single
    (axis, segment), plus its refinement axis."""
    facet_name: str = Field(
        ..., description="Short descriptive name for the consolidated facet (2-5 words)"
    )
    facet_description: str = Field(
        ..., description="One description faithful to what the consumed proposals jointly cover"
    )
    source_proposals: List[str] = Field(
        ..., description="facet_name of every proposal consumed into this facet, one-for-one"
    )
    refinement_axis_name: str = Field(
        ..., description="Name of this facet's refinement axis — the sub-question the observations inside it differ along"
    )
    refinement_axis_description: str = Field(
        ..., description="One or two sentences describing the difference in the data the refinement axis captures"
    )
    positions: List[RefinementPosition] = Field(
        ..., description="2-6 substantive positions plus exactly one residual position"
    )


# =============================================================================
# §3 FACET REVIEW (P3) — per-domain quality gate on the consolidated facet set
# =============================================================================


def build_facet_review_prompt(
    *,
    survey_question: str,
    primary_dimension: str,
    domain_label: str,
    domain_definition: str,
    domain_boundary_test: str,
    facets: List[DiscoveredFacet],
) -> str:
    """Review a domain's consolidated facet set for definitional MECE-ness.

    Structure change is impossible by construction: the response schema echoes
    the input set 1-on-1 (matched by original_name) and only carries rewritten
    names/descriptions plus a boundary test and overlap flags.
    """
    facets_block = "\n".join(
        f"[F{i}] {facet.facet_name}\n    Description: {facet.facet_description}"
        for i, facet in enumerate(facets, start=1)
    )

    return f"""You are a taxonomy quality reviewer for open-ended survey coding.

The survey question:
"{survey_question}"

Primary dimension of the taxonomy: {primary_dimension}

Domain under review: {domain_label}
Domain definition: {domain_definition}
Domain boundary test: {domain_boundary_test}

Below are this domain's consolidated facets — the perspectives along which its
ideas will be organised. Assignment has not happened yet: rewrites are free, but
the facet SET is fixed.

<facets>
{facets_block}
</facets>

Your task, for every facet, in the same order:
1. Judge the definition from the viewpoint of the survey question: does it delimit
   exactly one concept, and can a coder tell it apart from every sibling facet by
   reading the definitions alone?
2. Rewrite the name and/or description where needed so that the distinction is
   explicit in the text itself. Descriptive wording only — no evaluative language
   (evaluation is captured per idea as valence, elsewhere).
3. Write one boundary test per facet: a single routing sentence for the doubtful
   case, phrased against a named sibling facet ("mentions X -> this facet; is
   about Y -> <sibling>").
4. If two facets appear to capture the same concept even after rewriting, keep
   both unchanged in your output and flag the pair with a one-sentence reason.
   Flagging is desired behaviour, not failure: flagged pairs are resolved later
   with assignment data.

Rules:
- Return exactly the facets you were given, one for one, matched by original_name.
  Do not add, merge, split or drop facets.
- Sharpen, do not re-scope: a rewrite must stay faithful to what the facet
  already covers.
- Keep names short and descriptive; keep descriptions one to three sentences.

Provide your output as valid JSON following the response schema provided."""


class ReviewedFacet(BaseModel):
    """A single facet after P3 review — rewrites are free, the set is fixed."""
    original_name: str = Field(
        ..., description="Exact name of the existing facet — the match key back to the input"
    )
    facet_name: str = Field(
        ..., description="Facet name, sharpened where needed"
    )
    facet_description: str = Field(
        ..., description="Facet description, reformulated for orthogonality"
    )
    boundary_test: str = Field(
        ..., description="One routing sentence for the doubtful case, phrased against a named sibling facet"
    )


class FacetOverlapFlag(BaseModel):
    """A pair of facets that still appear to capture the same concept after rewriting."""
    facet_a: str = Field(
        ..., description="Name of the first facet in the overlapping pair"
    )
    facet_b: str = Field(
        ..., description="Name of the second facet in the overlapping pair"
    )
    reason: str = Field(
        ..., description="One sentence explaining why the two facets overlap"
    )


class FacetReviewResponse(BaseModel):
    """P3 output: reviewed facets and any overlap flags for a single domain."""
    facets: List[ReviewedFacet] = Field(
        ..., description="Reviewed facets, exactly 1-on-1 with the input set"
    )
    overlap_flags: List[FacetOverlapFlag] = Field(
        ..., description="Pairs of facets that still appear to capture the same concept after rewriting"
    )


# =============================================================================
# §3a FACET REVIEW V2 (P3-review, axis-first path) — widened mandate: rewrite
# AND restructure (merge/split) a domain's facets, but only inside its fixed
# axis system. Used only for domains that got a validated axis system from
# P1a (axis_first_enabled); domains without one keep the §3 path above
# untouched.
# =============================================================================

def _build_domain_structure_block(facets: List[DiscoveredFacet]) -> str:
    """Render a domain's full consolidated structure for P3-review (V2): one
    'Axis: name' group per axis, each holding its segments, and per segment
    the facet that occupies it (name, description, boundary) plus that
    facet's refinement axis and positions. Built entirely from the facets'
    own axis/segment/boundary_test/refinement fields — P2 already
    denormalized the segment boundary onto boundary_test, so no separate
    axis-system object is needed here. Axes and segments are grouped in
    first-seen (facet list) order, not sorted."""
    by_axis: Dict[str, List[DiscoveredFacet]] = {}
    for f in facets:
        by_axis.setdefault(f.axis, []).append(f)

    axis_blocks = []
    for axis_name, axis_facets in by_axis.items():
        lines = [f"Axis: {axis_name}"]
        for f in axis_facets:
            lines.append(f"  Segment: {f.segment}")
            lines.append(f"    Boundary: {f.boundary_test}")
            lines.append(f"    Facet: {f.facet_name} — {f.facet_description}")
            refinement = f.refinement or {}
            positions = refinement.get("positions", [])
            if refinement:
                lines.append(
                    f"    Refinement axis: {refinement.get('name', '')} — "
                    f"{refinement.get('description', '')}"
                )
                non_residual = [p for p in positions if not p.get("is_residual")]
                residual = [p for p in positions if p.get("is_residual")]
                for p in non_residual + residual:
                    suffix = " (residual)" if p.get("is_residual") else ""
                    lines.append(
                        f"      - {p.get('position_name', '')}{suffix}: "
                        f"{p.get('position_description', '')}"
                    )
                    lines.append(f"        Boundary: {p.get('boundary', '')}")
        axis_blocks.append("\n".join(lines))
    return "\n\n".join(axis_blocks)


def build_facet_review_v2_prompt(
    *,
    survey_question: str,
    domain_label: str,
    domain_definition: str,
    facets: List[DiscoveredFacet],
) -> str:
    """Review and, where needed, restructure a domain's consolidated facet
    set inside its fixed axis system (P3-review, V2 / axis-first path)."""
    domain_structure_block = _build_domain_structure_block(facets)

    return f"""You are a taxonomy quality reviewer for open-ended survey coding.

The survey question:
"{survey_question}"

Domain under review: {domain_label} — {domain_definition}

Below is the domain's full consolidated structure: the axis system, and per
segment its facet with refinement axis and positions. Assignment has not
happened yet: restructuring is free, but everything must stay inside the axis
system.

<structure>
{domain_structure_block}
</structure>

Your task, judging from the survey question:
1. Verify that every facet occupies exactly one segment and that no two
   facets capture the same concept. Where they do: merge them (list both
   under source_facets). Where one facet straddles two segments: split it
   (same source facet in two outputs, each on its own segment).
2. Sharpen names, descriptions and segment boundaries so the distinction
   between every pair of facets is explicit in the text itself.
3. Return the complete revised facet set. Every input facet must appear in
   source_facets of exactly the outputs that absorb it — unaccounted or
   double-counted facets invalidate the review.

Rules:
- The axis system itself is fixed: you may not add, remove or rename axes or
  segments (flag structural doubts in prose via the reason fields instead).
- Descriptive wording only.

Provide your output as valid JSON following the response schema provided.
"""


class ReviewedFacetV2(BaseModel):
    """A single facet in the reviewed, possibly restructured, output set
    (P3-review V2 / axis-first path). Merge = several sources, one output.
    Split = the same source facet appearing across several outputs, each on
    its own valid (axis, segment). Rename/redescribe = one source, one
    output."""
    facet_name: str = Field(
        ..., description="Facet name, sharpened where needed"
    )
    facet_description: str = Field(
        ..., description="Facet description, reformulated for orthogonality"
    )
    axis_name: str = Field(
        ..., description="Name of the axis this facet occupies — must exist in the domain's fixed axis system"
    )
    segment_name: str = Field(
        ..., description="Name of the segment on that axis this facet occupies — must exist on that axis"
    )
    boundary: str = Field(
        ..., description="One routing sentence for the doubtful case, phrased against a named sibling facet"
    )
    source_facets: List[str] = Field(
        ..., description=(
            "Every input facet that goes into this output, by facet_name "
            "(merge: several distinct sources; split: the same source facet "
            "listed under several outputs, each on its own segment). Every "
            "input facet must appear in source_facets of exactly the "
            "outputs that absorb it"
        )
    )


class FacetReviewV2Response(BaseModel):
    """P3-review output (V2 / axis-first path): the domain's complete
    revised facet set — rewrites, merges and splits, all inside the fixed
    axis system."""
    facets: List[ReviewedFacetV2] = Field(
        ..., description="The domain's complete revised facet set"
    )


# =============================================================================
# §4 FACET ASSIGNMENT (P4) — per-domain batched assignment
# =============================================================================


def _build_facet_codebook_block(
    facets: List[DiscoveredFacet],
    other_label: Optional[str] = None,
    axis_descriptions: Optional[Dict[str, str]] = None,
) -> str:
    """Format discovered facets as a numbered codebook for assignment.

    When any facet carries an axis tag (P1b/P2 axis-first path), items are
    grouped under one 'Axis: {name} — {description}' header per axis, axes
    in first-seen order (description omitted when not supplied — cache-loaded
    contexts have no axis-system object to pull it from). F# numbering still
    reflects each facet's position in `facets`, unchanged from the untagged
    path, so it stays consistent with the facet_id_to_name mapping built
    alongside this same list. A facet list with no axis tags renders exactly
    as before this grouping existed."""
    def _render(i: int, facet: DiscoveredFacet) -> str:
        examples = "; ".join(facet.example_observations[:3])
        return (
            f"[F{i}] {facet.facet_name}\n"
            f"    Description: {facet.facet_description}\n"
            + (f"    Belongs here: {facet.inclusion_rule}\n" if facet.inclusion_rule else "")
            + (f"    Does not belong here: {facet.exclusion_rule}\n" if facet.exclusion_rule else "")
            + (f"    Boundary: {facet.boundary_test}\n" if facet.boundary_test else "")
            + f"    Examples: {examples}"
        )

    numbered = list(enumerate(facets, 1))
    if any(facet.axis for facet in facets):
        by_axis: Dict[str, List[Tuple[int, DiscoveredFacet]]] = {}
        for i, facet in numbered:
            by_axis.setdefault(facet.axis, []).append((i, facet))
        axis_descriptions = axis_descriptions or {}
        lines = []
        for axis_name, items in by_axis.items():
            desc = axis_descriptions.get(axis_name, "")
            header = f"Axis: {axis_name} — {desc}" if desc else f"Axis: {axis_name}"
            body = "\n\n".join(_render(i, facet) for i, facet in items)
            lines.append(f"{header}\n\n{body}")
    else:
        lines = [_render(i, facet) for i, facet in numbered]

    if other_label:
        n = len(facets) + 1
        lines.append(
            f"[F{n}] {other_label}\n"
            f"    Description: Observations that do not clearly fit any of the above facets.\n"
            f"    Examples: (none)"
        )
    return "\n\n".join(lines)


def build_facet_assignment_prompt_single(
    *,
    survey_question: str,
    language: str,
    dataset_context_section: str,
    domain_name: str,
    domain_definition: str,
    facets: List[DiscoveredFacet],
    idea_label: str,
    axis_descriptions: Optional[Dict[str, str]] = None,
) -> str:
    """Build prompt for assigning a single idea to a facet (L3).

    `axis_descriptions`: {axis_name: axis_description}, available only when
    the caller still holds the in-memory AxisSystemResponse for this domain
    (during a live run, via TaxonomyClassifier.axis_systems). Cache-loaded
    contexts pass None and the menu falls back to bare 'Axis: {name}'
    headers.
    """
    facet_codebook = _build_facet_codebook_block(facets, axis_descriptions=axis_descriptions)

    return f"""You are a qualitative coding assistant. Assign the survey response idea below to the facet that best captures the type of quality being described.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

<domain_context>
Domain: {domain_name} -- {domain_definition}
</domain_context>

<facets>
{facet_codebook}
</facets>

<idea>
{idea_label}
</idea>

### VALENCE (evaluation relative to facet)
- "+" Positive — The attribute is described as meeting or enhancing the facet
- "-" Negative — The attribute is described as failing to meet or detracting from the facet
- "0" Neutral — The response is descriptive, ambiguous, or does not express evaluation
- Valence is not emotional sentiment, but evaluative direction relative to the facet

Assign this idea to the single best-fitting facet. Return the facet ID (e.g. "F1", "F2"), your confidence (0.0-1.0), and the valence (+, -, or 0).

Provide your response as valid JSON following the response schema provided."""


class FacetAssignmentResult(BaseModel):
    """Single idea-to-facet assignment result."""
    assigned_facet_id: str = Field(
        ..., description=(
            "The facet ID from the [F#] prefix (e.g. 'F1', 'F3'). "
            "Return ONLY the ID, not the facet name."
        )
    )
    confidence: float = Field(
        ..., description="Confidence in the assignment (0.0 to 1.0)"
    )
    valence: Literal["+", "-", "0"] = Field(
        default="0",
        description="Evaluative direction relative to the facet: + positive, - negative, 0 neutral"
    )



# =============================================================================
# §5 ATTRIBUTE DISCOVERY (P5) — per facet within domain
# =============================================================================

def build_attribute_discovery_prompt(
    *,
    survey_question: str,
    language: str,
    dataset_context_section: str,
    dimension_def: Optional[DimensionDefinition],
    dimension_name: str,
    dimension_description: str,
    domain_name: str,
    domain_definition: str,
    facet_name: str,
    facet_description: str,
    observations: List[str],
    excluded_facets: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """Discover concrete attributes (L4) within a facet."""
    observations_block = "\n".join(
        f"{i}. {obs}" for i, obs in enumerate(observations, 1)
    )

    # Dimension-specific guidance
    if dimension_def:
        rules = dimension_def.prompt_rules
        #attribute_guidance = rules.attribute_instruction
        attribute_definition = _extract_definition(rules.attribute_instruction)
        attribute_key_idea = _extract_key_idea(rules.attribute_instruction)
        facet_key_idea = _extract_key_idea(rules.facet_instruction)
        noun_phrase = dimension_def.noun_phrase_descriptor
        domain_key_idea = _extract_key_idea(rules.domain_instruction)
    else:
        attribute_guidance = (
            "An attribute identifies the specific observable property or feature being described. "
            "It is a named property — not a verbatim span from the response."
        )
        attribute_definition = attribute_guidance
        attribute_key_idea = "the specific observable property being described"
        facet_key_idea = "the analytical lens applied to the subject"
        noun_phrase = dimension_name
        domain_key_idea = "the subject the statement refers to"

    excluded_block = _build_exclusion_block(
        excluded_facets or [], "excluded_facets"
    )

    excluded_block_light = _build_exclusion__light_block(
        excluded_facets or []
    )

    return f"""You are a qualitative research analyst specializing in survey response analysis. 
Your task is to identify the fewest recurring attributes that provide full coverage of a set of observations within a specific facet.

{attribute_definition} An attribute must:
- Be a descriptive, data-grounded category based on shared meaning across multiple observations
- Be non-evaluative (no judgment, sentiment, or valence)
- Stay strictly within the facet boundaries
- Be internally coherent (one clear underlying concept)
- Be externally distinctive:
  * Ontologically distinct (no overlap, no subset/superset, no reframing of same phenomenon)
  * Semantically separable (no ambiguity in coding; no "could go either way")
- Be non-redundant (adds unique conceptual value; no duplicate concepts)
- Be grounded in the data (supported by multiple observations or repeated patterns)

Here is the survey context you are working with:

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

Here is the taxonomy context that defines your working framework:

<taxonomy_context>
This is the structure:
<taxonomy_structure>
- Dimension (L1): {dimension_name} — {dimension_description}
- Domain (L2): {domain_key_idea}
- Facet (L3): {facet_key_idea}
- Attribute (L4): {attribute_key_idea}
</taxonomy_structure>

You are working within this domain:
<taxonomy_domain>
{domain_name} — {domain_definition}
</taxonomy_domain>

And you are working within this facet:
<taxonomy_facet>
{facet_name} — {facet_description}
</taxonomy_facet>
{excluded_block}
</taxonomy_context>

Here are the observations you need to analyze:

<observations>
{observations_block}
</observations>

# Instructions

Before writing your final output, think through your analysis in the scratchpad field:

**Step 1: Cluster observations**
Group similar observations together based on shared descriptive meaning. Identify recurring patterns in what is being said within {facet_name}.

Focus on the specific quality, property, or feature being described.

**Step 2: Identify candidate attributes**
Based on these clusters, identify candidate attributes.

For each candidate attribute, assess:
- the attribute name
- the specific observable property it captures
- which observations support it
- whether it is internally coherent
- whether it is ontologically distinct from other candidate attributes

Remember: an attribute names a specific quality or trait — a concrete, observable property, not a verbatim span from the response.

**Step 3: Verify internal coherence**
Check whether each candidate attribute captures one clear underlying concept.

Reject or split candidate attributes that:
- combine multiple different kinds of phenomena
- mix descriptive content with evaluation
- are too broad to support clear coding

**Step 4: Verify distinctness**
Check each pair of candidate attributes to ensure they are:
- ontologically distinct (not overlapping in conceptual space; one is not a subset of another)
- semantically separable (someone coding a response would clearly know which attribute applies, with no "could go either way" situations)
- not two different lenses on the same phenomenon

If two attributes fail this test, consolidate them into one broader attribute or redefine the boundaries more clearly.

**Step 5: Verify facet boundaries**
Check that each retained attribute falls strictly within the included facet of {facet_name}.

Exclude attributes that belong more naturally to other facets, including:
{excluded_block_light}

**Step 6: Prepare final output**
Return only the dominant attributes that pass all checks above.

For each attribute, provide:
- a short descriptive name in {language} (2-5 words)
- a description in {language} of what the attribute captures — a concrete, observable property (1-2 sentences)
- the parent facet name: {facet_name}
- 2-3 representative observations from the input, using the exact observation text

# Output Requirements

Provide output as valid JSON following the response schema provided.

# Language Requirement

All output (attribute names, descriptions, and example observations) must be written in {language}.

# Final Notes

- Attributes must be descriptive, not evaluative
- Never create attributes that differ only in evaluative direction (e.g. a positive and a negative version of the same concept). Capture the concept as ONE attribute; positive/negative is recorded separately as valence. A response that is only an overall judgment with no descriptive content ("good", "fine", "not great") belongs to a single residual overall-judgment attribute, never to positive/negative variants.
- Attributes must be grounded in repeated patterns across observations
- Attributes must be internally coherent
- Attributes must be externally distinctive
- Attributes must remain strictly within the included facet
- Each attribute must capture one specific quality, not multiple
- All output must be in {language}
- Use exact observation text in the examples, not observation numbers

Use your scratchpad field for Steps 1-6 to show your analytical thinking. Then provide your final output as valid JSON."""


class DiscoveredAttribute(BaseModel):
    """A concrete attribute (L4) discovered within a facet."""
    attribute_name: str = Field(
        ..., description="Short descriptive name for the attribute (2-5 words)"
    )
    attribute_description: str = Field(
        ..., description="What this attribute captures — a concrete, observable property (1-2 sentences)"
    )
    parent_facet: str = Field(
        ..., description="The facet this attribute belongs to"
    )
    example_observations: List[str] = Field(
        ..., description="2-3 representative observations from the input"
    )
    position: SkipJsonSchema[str] = Field(
        default="", description=(
            "P6 axis-first provenance only: the position on the facet's refinement "
            "axis this attribute was tagged to (empty on the untagged path)"
        )
    )
    is_residual_attr: SkipJsonSchema[bool] = Field(
        default=False, description=(
            "P6 axis-first provenance only: true when this attribute is the residual "
            "attribute for its facet's residual position (empty/false on the untagged path)"
        )
    )


class AttributeDiscoveryResult(BaseModel):
    """P5 output: attributes discovered within a facet."""
    scratchpad: str = Field(
        ..., description=(
            "Step-by-step reasoning before identifying attributes: "
            "(1) cluster observations by shared descriptive meaning, "
            "(2) identify candidate attributes and assess coherence and distinctness, "
            "(3) verify internal coherence — one clear concept per attribute, "
            "(4) verify distinctness — ontologically distinct and semantically separable, "
            "(5) verify facet boundaries — exclude attributes belonging to other facets, "
            "(6) prepare final output with only dominant attributes that pass all checks"
        )
    )
    attributes: List[DiscoveredAttribute] = Field(
        ..., description="Concrete attributes identified within the facet"
    )


# =============================================================================
# §5a POSITION-TAGGED ATTRIBUTE DISCOVERY (P5, axis-first path) — per-facet
# chunked discovery inside a pre-established, fixed refinement axis. Used only
# for facets that carry a refinement axis from P2 segment consolidation
# (axis_first_enabled); facets without one keep the untagged §5 path above
# untouched.
# =============================================================================

def _build_refinement_axis_block(refinement: dict) -> str:
    """Render a facet's refinement axis (positions) as prompt text, mirroring
    `_build_axis_system_block` one level down: one 'Axis: name — description'
    line, followed by its positions (residual last, name suffixed
    ' (residual)')."""
    positions = refinement.get("positions", [])
    non_residual = [p for p in positions if not p.get("is_residual")]
    residual = [p for p in positions if p.get("is_residual")]
    lines = [f"Axis: {refinement.get('name', '')} — {refinement.get('description', '')}"]
    for p in non_residual + residual:
        suffix = " (residual)" if p.get("is_residual") else ""
        lines.append(f"  - {p.get('position_name', '')}{suffix}: {p.get('position_description', '')}")
        lines.append(f"    Boundary: {p.get('boundary', '')}")
    return "\n".join(lines)


def _build_neighbour_facets_block(neighbours: List[DiscoveredFacet]) -> str:
    """Render a domain's other facets as context-only reference for P5
    (axis-first path): one '- name (segment: segment of axis)' line each."""
    if not neighbours:
        return "(none)"
    return "\n".join(
        f"- {f.facet_name} (segment: {f.segment} of {f.axis})" for f in neighbours
    )


def build_position_attribute_discovery_prompt(
    *,
    survey_question: str,
    domain_label: str,
    facet_name: str,
    facet_description: str,
    segment_name: str,
    axis_name: str,
    refinement: dict,
    neighbour_facets: List[DiscoveredFacet],
    chunk_observations: List[str],
) -> str:
    """Discover attributes (L4) from a chunk of observations, each proposal
    tagged to exactly one position of the facet's fixed refinement axis, or
    proposing a new position explicitly (P5, axis-first path)."""
    refinement_axis_block = _build_refinement_axis_block(refinement)
    neighbour_facets_block = _build_neighbour_facets_block(neighbour_facets)
    observations_block = "\n".join(f"{i}. {obs}" for i, obs in enumerate(chunk_observations, 1))

    return f"""You are a qualitative research analyst for open-ended survey coding.

The survey question:
"{survey_question}"

Domain: {domain_label}
Facet: {facet_name} — {facet_description}
This facet occupies segment "{segment_name}" of axis "{axis_name}".

The facet's refinement axis was established beforehand and is your frame:

<refinement_axis>
{refinement_axis_block}
</refinement_axis>

Neighbouring facets of this domain (context only — never targets for your
proposals):

<neighbours>
{neighbour_facets_block}
</neighbours>

Below are this chunk's observations for this facet:

<observations>
{observations_block}
</observations>

Your task: propose attributes, where every attribute is one atomic concept at
EXACTLY ONE position of the refinement axis. Tag each proposal with its
position_name. If this chunk's observations genuinely require a position the
axis does not have, propose it explicitly (is_new_position = true, with a
one-sentence boundary) — consolidation will adjudicate it.

Rules:
- One position can receive multiple proposals; a proposal can never span two
  positions — split it.
- Observations that name no recognisable value on the refinement axis belong
  to the residual position; never invent a general catch-all attribute.
- If an observation seems to belong to a neighbouring facet, do not propose
  for it here — it is not yours to place.
- Ground every attribute in 2-5 verbatim examples. Descriptive wording only.

Provide your output as valid JSON following the response schema provided.
"""


class TaggedAttributeProposal(BaseModel):
    """An attribute proposal from a chunk, tagged to a position of the
    facet's fixed refinement axis, or explicitly proposing a new position
    (P5 output, axis-first path)."""
    attribute_name: str = Field(
        ..., description="Short descriptive name for the attribute (2-5 words)"
    )
    attribute_description: str = Field(
        ..., description="What this attribute captures — a concrete, observable property (1-2 sentences)"
    )
    position_name: str = Field(
        ..., description=(
            "Name of the position on the refinement axis this attribute occupies — "
            "must exist on the axis above, unless is_new_position is true"
        )
    )
    is_new_position: bool = Field(
        default=False, description=(
            "True when this chunk's observations genuinely require a position the "
            "refinement axis does not have"
        )
    )
    new_position_boundary: str = Field(
        default="", description="One-sentence boundary for the proposed new position — required when is_new_position is true"
    )
    example_observations: List[str] = Field(
        ..., description="2-5 example observations grounding this attribute, verbatim from the chunk"
    )


class TaggedAttributeDiscoveryResponse(BaseModel):
    """P5 output (axis-first path): tagged attribute proposals discovered in
    a single chunk."""
    proposals: List[TaggedAttributeProposal] = Field(
        ..., description="Attribute proposals discovered in this chunk, each tagged to a position (existing or newly proposed)"
    )


# =============================================================================
# §6 ATTRIBUTE CHUNK CONSOLIDATION (P6) — merge chunk-level attributes within facet
# =============================================================================


def build_attribute_chunk_consolidation_prompt(
    *,
    survey_question: str,
    language: str,
    dataset_context_section: str,
    dimension_def: Optional[DimensionDefinition],
    dimension_name: str,
    dimension_description: str,
    domain_name: str,
    facet_name: str,
    facet_description: str,
    chunk_results: str,
    excluded_facets: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """Consolidate chunk-level attribute discoveries into a single coherent set within a facet."""
    # Dimension-specific guidance
    if dimension_def:
        rules = dimension_def.prompt_rules
        attribute_guidance = rules.attribute_instruction
        attribute_definition = _extract_definition(rules.attribute_instruction)
        attribute_key_idea = _extract_key_idea(rules.attribute_instruction)
        facet_key_idea = _extract_key_idea(rules.facet_instruction)
        noun_phrase = dimension_def.noun_phrase_descriptor
        domain_key_idea = _extract_key_idea(rules.domain_instruction)
    else:
        attribute_guidance = (
            "An attribute identifies the specific observable property or feature being described. "
            "It is a named property -- not a verbatim span from the response."
        )
        attribute_definition = attribute_guidance
        attribute_key_idea = "the specific observable property being described"
        facet_key_idea = "the analytical lens applied to the subject"
        noun_phrase = dimension_name
        domain_key_idea = "the subject the statement refers to"

    excluded_block = _build_exclusion_block(
        excluded_facets or [], "excluded_facets"
    )
    excluded_block_light = _build_exclusion_block_light(
        excluded_facets or []
    )

    return f"""You are a taxonomy consolidation specialist for surveys.
Your task is to merge multiple chunk-level attribute analyses into a single, minimal set of mutually exclusive attributes within a given facet.

# What is an Attribute?

{attribute_definition} An attribute must satisfy these requirements:
- **Stay strictly within the facet boundaries**: It must belong to the included facet and not overlap with excluded facets
- **Be internally coherent**: One clear underlying concept, not a mixture of different ideas
- **Be externally distinctive**: 
  * Ontologically distinct (no overlap, no subset/superset relationships, no reframing of the same phenomenon)
  * Semantically separable (no ambiguity in coding; an observation should not "could go either way" between two attributes)

# Survey Context

Here is the survey context:

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

Use the survey context to:

<survey_context_usage>
- Interpret the meaning of attributes relative to the survey question
- Ensure consolidated attributes are directly relevant to what is being asked
- Preserve terminology and phrasing appropriate to the survey language
- Avoid introducing attributes that are not grounded in the question intent
</survey_context_usage>

# Taxonomy Context

Here is the taxonomy context that defines your working framework:

<taxonomy_context>
This is the structure:
<taxonomy_structure>
- Dimension (L1): {dimension_name} -- {dimension_description}
- Domain (L2): {domain_key_idea}
- Facet (L3): {facet_key_idea}
- Attribute (L4): {attribute_key_idea}
</taxonomy_structure>

You are working within this domain:

<taxonomy_domain>
{domain_name}
</taxonomy_domain>

You are working within this facet:

<taxonomy_facet>
{facet_name} -- {facet_description}
</taxonomy_facet>
{excluded_block}
</taxonomy_context>

Pay careful attention to:
- The facet you are working within (this defines the scope of valid attributes)
- Any excluded facets (attributes belonging to these must be removed)
- The domain and dimension (for broader context)

# Chunk-Level Analyses to Consolidate

Here are the attributes discovered from analyzing different chunks of the survey data:

<chunk_level_analyses>
{chunk_results}
</chunk_level_analyses>

# Consolidation Rules

Apply these rules strictly when consolidating attributes:

Consolidation is the goal: do NOT keep every concept separate — group. But govern grouping by these rules, in order.

**1. DIMENSION FIRST (orthogonality — the guardrail).**
For each concept, determine WHICH underlying dimension it answers.
- Concepts on DIFFERENT dimensions are orthogonal: NEVER merge them into one attribute (e.g. socio-economic class vs political orientation vs age are different dimensions).
- Mutually-exclusive VALUES/POLES of the SAME dimension are also kept apart (e.g. "young" vs "old"); merging opposite poles creates an empty container.
- Do NOT create separate attributes based only on the object discussed, when the same underlying value applies — an object is not a dimension.

**2. PREVALENCE SETS GRANULARITY (within a dimension only).**
- A high-count value keeps its own attribute — never dissolve a well-supported concept.
- Several thin, same-dimension values are GROUPED into one attribute that still names the shared value/contrast in plain language.
- Variants that differ only in evaluative direction ("positive X" and "negative X") collapse to ONE attribute "X"; the direction is recorded separately as valence, not as separate attributes.
Prevalence decides how finely to split WITHIN a dimension; it NEVER licenses merging ACROSS dimensions.

**3. LIFT, DON'T FLATTEN.**
When grouping is needed, raise concepts to a shared higher-abstraction label that still carries their meaning — NOT a label that merely names the axis.
FORBIDDEN: a container that only names the axis it sits on — the reader learns what was being measured, not what was said.
REQUIRED: a label that states the value itself, so the reader knows what the respondents expressed.
Test: read the label alone. If it tells you only which question was asked, it is a container; if it tells you what the answer was, it is a value.

**4. PLAIN, MEANINGFUL LABELS.**
Name every surviving attribute in everyday language. Test: reading the label alone, a layperson knows which distinction is meant, given the survey question. No jargon, no nominalizations, no dimension-names.

**Precedence when rules conflict:** 1 (orthogonality) > 2 (prevalence grouping) > 4 (label clarity).

# Step-by-Step Analysis Process

Before providing your final output, work through your analysis systematically in a scratchpad. Follow these steps:

**Step 1 -- Scan chunk-level attributes**
Review all attributes from all chunks. Note recurring themes, similar concepts, and obvious duplicates.

**Step 2 -- Group overlapping attributes**
Identify and group attributes that describe the same or overlapping concepts across different chunks.

**Step 3 -- Apply the dimension test**
For each pair of candidate attributes, ask: "Do these answer the SAME underlying dimension, or DIFFERENT dimensions?" Different dimensions (or opposite poles of one dimension) → keep separate; same dimension and same meaning → group.

**Step 4 -- Group thin same-dimension attributes by prevalence**
Within a dimension, keep high-count values as their own attribute; group several thin same-dimension values under one meaningful, plainly-named attribute. Never merge across dimensions.

**Step 5 -- Verify facet boundaries**
Ensure each retained attribute belongs to the included facet and not to any excluded facet:
{excluded_block_light}

**Step 6 -- Prepare final output**
Confirm you have the minimal set of consolidated attributes that pass all checks. Prepare the name, description, and representative observations for each.

# Output Requirements

For each consolidated attribute, provide:
- A short descriptive name (2-5 words)
- A description of what the attribute captures -- a concrete, observable property (1-2 sentences)
- The parent facet name: {facet_name}
- 2-3 representative observations selected from across the merged chunks (exact text)

Provide output as valid JSON following the response schema provided.

# Language Requirement

All attribute names and descriptions must be in {language}.

# Final Reminders

- Attributes must be **descriptive, not evaluative**
- Attributes must be **internally coherent** (one clear concept each)
- Attributes must be **externally distinctive** (no overlap, no subset/superset relationships)
- Attributes must remain **strictly within the included facet**
- Group thin same-dimension values, but never merge across dimensions or opposite poles
- **When in doubt, check the dimension** before grouping
- All output must be in {language}

Begin by writing your step-by-step analysis in the scratchpad field, then provide your final consolidated attributes in valid JSON format."""


class AttributeChunkConsolidatedResponse(BaseModel):
    """Consolidated attributes after merging chunk-level discoveries within a facet."""
    scratchpad: str = Field(
        ..., description=(
            "Step-by-step reasoning before consolidating attributes: "
            "(1) scan chunk-level attributes for recurring themes and duplicates, "
            "(2) group overlapping attributes across chunks, "
            "(3) apply the dimension test -- keep different dimensions or opposite poles separate, "
            "(4) group thin same-dimension attributes by prevalence into meaningful, plainly-named attributes, "
            "(5) verify facet boundaries -- exclude attributes belonging to other facets, "
            "(6) prepare final minimal set of consolidated attributes"
        )
    )
    attributes: List[DiscoveredAttribute] = Field(
        ..., description="Fewest mutually exclusive attributes needed for full coverage, consolidated from all chunks"
    )


# =============================================================================
# §6a PER-POSITION ATTRIBUTE CONSOLIDATION (P6, axis-first path) — one
# consolidation task per populated position, grouped from the facet's tagged
# P5 proposals IN CODE (not by the model), plus adjudication of any
# new-position proposals raised during discovery. Used only for facets that
# carry a refinement axis from P2 segment consolidation; facets without one
# keep the §6 path above untouched.
# =============================================================================

def _build_position_proposals_block(proposals: List[DiscoveredAttribute]) -> str:
    """Render the attribute proposals tagged to one position as prompt text,
    mirroring `_build_segment_proposals_block` one level down."""
    lines = []
    for p in proposals:
        lines.append(f"- {p.attribute_name}: {p.attribute_description}")
        lines.append(f"  Examples: {'; '.join(p.example_observations[:5])}")
    return "\n".join(lines)


def build_position_consolidation_prompt(
    *,
    survey_question: str,
    domain_label: str,
    facet_name: str,
    facet_description: str,
    refinement_axis_name: str,
    refinement_axis_description: str,
    position_name: str,
    position_boundary: str,
    proposals: List[DiscoveredAttribute],
) -> str:
    """Consolidate all chunk-level attribute proposals tagged to one position
    into a single attribute (P6, axis-first path)."""
    position_proposals_block = _build_position_proposals_block(proposals)

    return f"""You are a taxonomy consolidation specialist for open-ended survey coding.

The survey question:
"{survey_question}"

Domain: {domain_label}
Facet: {facet_name} — {facet_description}
Refinement axis: {refinement_axis_name} — {refinement_axis_description}
Position under consolidation: {position_name}
Position boundary: {position_boundary}

Below are all chunk-level attribute proposals tagged to this position, with
their examples:

<proposals>
{position_proposals_block}
</proposals>

Your task: consolidate these proposals into ONE attribute for this position:
one name, one description faithful to what the proposals jointly cover. List
every proposal you consumed under source_proposals (one-for-one bookkeeping).

Rules:
- Work strictly inside this position; other positions are consolidated
  separately and are none of your concern.
- Descriptive wording only.

Provide your output as valid JSON following the response schema provided.
"""


class ConsolidatedAttribute(BaseModel):
    """P6 output (axis-first path): one consolidated attribute for a single
    position, plus its source bookkeeping."""
    position_name: str = Field(
        ..., description="Name of the position under consolidation — echoed back for bookkeeping"
    )
    attribute_name: str = Field(
        ..., description="Short descriptive name for the consolidated attribute (2-5 words)"
    )
    attribute_description: str = Field(
        ..., description="One description faithful to what the consumed proposals jointly cover"
    )
    source_proposals: List[str] = Field(
        ..., description="attribute_name of every proposal consumed into this attribute, one-for-one"
    )


def _build_positions_block(refinement: dict) -> str:
    """Render a facet's existing refinement positions for the adjudication
    prompt: one '- name: description' line each (residual last)."""
    positions = refinement.get("positions", [])
    non_residual = [p for p in positions if not p.get("is_residual")]
    residual = [p for p in positions if p.get("is_residual")]
    lines = []
    for p in non_residual + residual:
        suffix = " (residual)" if p.get("is_residual") else ""
        lines.append(f"- {p.get('position_name', '')}{suffix}: {p.get('position_description', '')}")
    return "\n".join(lines)


def _build_new_position_proposals_block(new_positions: List[Dict]) -> str:
    """Render a facet's proposed new positions for the adjudication prompt:
    one '- name: boundary' line each, plus its pooled examples."""
    lines = []
    for np_ in new_positions:
        lines.append(f"- {np_['position_name']}: {np_['boundary']}")
        lines.append(f"  Examples: {'; '.join(np_['examples'][:5])}")
    return "\n".join(lines)


def build_new_position_adjudication_prompt(
    *,
    survey_question: str,
    facet_name: str,
    facet_description: str,
    refinement_axis_name: str,
    refinement_axis_description: str,
    refinement: dict,
    new_positions: List[Dict],
) -> str:
    """Adjudicate a facet's new-position proposals raised during P5
    discovery: accept (the position joins the refinement axis) or fold into
    an existing position (P6 adjudication, axis-first path)."""
    positions_block = _build_positions_block(refinement)
    new_position_proposals_block = _build_new_position_proposals_block(new_positions)

    return f"""You are a taxonomy consolidation specialist for open-ended survey coding.

The survey question:
"{survey_question}"

Facet: {facet_name} — {facet_description}
Refinement axis: {refinement_axis_name} — {refinement_axis_description}
Existing positions:

<positions>
{positions_block}
</positions>

During discovery the following NEW positions were proposed, with boundaries
and examples:

<new_positions>
{new_position_proposals_block}
</new_positions>

For every proposed new position, give a verdict:
- "accept" when it captures a value on the refinement axis that no existing
  position covers — give its final name and keep its boundary;
- "fold_into" when an existing position already covers it — name that
  position and the reason.

Provide your output as valid JSON following the response schema provided.
"""


class NewPositionAdjudication(BaseModel):
    """A verdict on one proposed new position (P6 adjudication output,
    axis-first path)."""
    position_name: str = Field(
        ..., description="Name of the proposed new position being adjudicated — echoed back for bookkeeping"
    )
    verdict: Literal["accept", "fold_into"] = Field(
        ..., description=(
            "'accept' when this position captures a value on the refinement axis that "
            "no existing position covers; 'fold_into' when an existing position already covers it"
        )
    )
    fold_into_position: str = Field(
        default="", description="Name of the existing position this proposal folds into — required when verdict is 'fold_into'"
    )
    reason: str = Field(
        ..., description="One sentence explaining the verdict"
    )


class NewPositionAdjudicationResponse(BaseModel):
    """P6 adjudication output (axis-first path): verdicts on every new
    position proposed for one facet."""
    verdicts: List[NewPositionAdjudication] = Field(
        ..., description="One verdict per proposed new position for this facet"
    )


# =============================================================================
# §7 ATTRIBUTE REVIEW (P7) — per-domain quality gate on the consolidated attribute set
# =============================================================================


def build_attribute_review_prompt(
    *,
    survey_question: str,
    domain_label: str,
    domain_definition: str,
    facets: List[DiscoveredFacet],
    facet_attributes: Dict[str, List[DiscoveredAttribute]],
) -> str:
    """Review a domain's consolidated attribute set for definitional MECE-ness.

    One call per domain. Structure change is impossible by construction: the
    response schema echoes the input attributes 1-on-1 (matched by
    original_name within their facet) and only carries rewritten
    names/descriptions plus overlap flags. Facets are read-only context.
    """
    tree_lines: List[str] = []
    for facet in facets:
        tree_lines.append(f"Facet: {facet.facet_name} — {facet.facet_description}")
        if facet.boundary_test:
            tree_lines.append(f"    Boundary: {facet.boundary_test}")
        for attribute in facet_attributes.get(facet.facet_name, []):
            tree_lines.append(f"    - {attribute.attribute_name}: {attribute.attribute_description}")
    tree_block = "\n".join(tree_lines)

    return f"""You are a taxonomy quality reviewer for open-ended survey coding.

The survey question:
"{survey_question}"

Domain under review: {domain_label} — {domain_definition}

Below is this domain's full tree after attribute consolidation: every facet (with
its boundary test) and its attributes. Assignment has not happened yet: rewrites
are free, but the attribute SET is fixed and facets are read-only context.

<tree>
{tree_block}
</tree>

Your task:
1. For every attribute, judge name and description from the viewpoint of the
   survey question: does it capture one atomic concept, distinguishable from
   every other attribute in this domain — inside its facet and across sibling
   facets — by the text alone?
2. Rewrite names/descriptions where needed so each attribute reads as exactly one
   concept. Descriptive wording only — no evaluative language.
3. Flag every pair of attributes (same facet or different facets of this domain)
   that appear to capture the same concept even after rewriting. For each flagged
   pair, write one decision_rule: a single routing sentence
   ("names X -> a; is about Y -> b").

Rules:
- Return exactly the attributes you were given, one for one, matched by
  original_name within their facet. Do not add, merge, split, move or drop
  attributes.
- Sharpen, do not re-scope.
- Flagging is desired behaviour, not failure: flagged pairs are resolved later
  with assignment data.

Provide your output as valid JSON following the response schema provided."""


class ReviewedAttribute(BaseModel):
    """A single attribute after P7 review — rewrites are free, the set is fixed."""
    original_name: str = Field(
        ..., description="Exact name of the existing attribute — the match key within its facet"
    )
    facet_name: str = Field(
        ..., description="Read-only reference to the facet this attribute belongs to — not a move field"
    )
    attribute_name: str = Field(
        ..., description="Attribute name, sharpened where needed"
    )
    attribute_description: str = Field(
        ..., description="Attribute description, reformulated for orthogonality"
    )


class AttributeOverlapFlag(BaseModel):
    """A pair of attributes that still appear to capture the same concept after rewriting."""
    attr_a: str = Field(
        ..., description="Name of the first attribute in the overlapping pair"
    )
    facet_a: str = Field(
        ..., description="Facet of the first attribute in the overlapping pair"
    )
    attr_b: str = Field(
        ..., description="Name of the second attribute in the overlapping pair"
    )
    facet_b: str = Field(
        ..., description="Facet of the second attribute in the overlapping pair — may differ from facet_a: cross-facet within the domain"
    )
    reason: str = Field(
        ..., description="One sentence explaining why the two attributes overlap"
    )
    decision_rule: str = Field(
        ..., description='Single routing sentence for the doubtful case, e.g. "names X -> a; is about Y -> b"'
    )


class AttributeReviewResponse(BaseModel):
    """P7 output: reviewed attributes and any overlap flags for a single domain."""
    attributes: List[ReviewedAttribute] = Field(
        ..., description="Reviewed attributes, exactly 1-on-1 with the input set"
    )
    overlap_flags: List[AttributeOverlapFlag] = Field(
        ..., description="Pairs of attributes that still appear to capture the same concept after rewriting"
    )


# =============================================================================
# §7a ATTRIBUTE REVIEW V2 (P7-review, axis-first path) — widened mandate:
# rewrite AND restructure (merge/split) a domain's attributes, but only
# within their own facet's fixed refinement axis. Used only for domains with
# a validated P1a axis system; the §7 path above stays untouched for every
# other domain.
# =============================================================================

def _build_domain_attribute_structure_block(
    facets: List[DiscoveredFacet],
    facet_attributes: Dict[str, List[DiscoveredAttribute]],
) -> str:
    """Render a domain's full facet+attribute structure for P7-review (V2):
    per facet its segment/axis, its refinement axis, and per position (non-
    residual first, residual last — the same ordering convention as
    `_build_refinement_axis_block`, though the line format differs: one line
    carrying both the position's own description and its boundary, to keep
    each position and its attributes visually together) the attributes
    tagged to it. Built from the facets' own axis/segment/refinement fields
    (as `_build_domain_structure_block` does for P3-review) plus each
    attribute's `position` field set by P6."""
    blocks = []
    for f in facets:
        lines = [f"Facet: {f.facet_name} — {f.facet_description} (segment: {f.segment} of {f.axis})"]
        refinement = f.refinement or {}
        lines.append(
            f"  Refinement axis: {refinement.get('name', '')} — {refinement.get('description', '')}"
        )
        positions = refinement.get("positions", [])
        non_residual = [p for p in positions if not p.get("is_residual")]
        residual = [p for p in positions if p.get("is_residual")]
        attrs = facet_attributes.get(f.facet_name, [])
        for p in non_residual + residual:
            suffix = " (residual)" if p.get("is_residual") else ""
            lines.append(
                f"    [{p.get('position_name', '')}]{suffix}: "
                f"{p.get('position_description', '')} — Boundary: {p.get('boundary', '')}"
            )
            for attr in attrs:
                if attr.position == p.get("position_name", ""):
                    lines.append(f"      - {attr.attribute_name}: {attr.attribute_description}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def build_attribute_review_v2_prompt(
    *,
    survey_question: str,
    domain_label: str,
    domain_definition: str,
    facets: List[DiscoveredFacet],
    facet_attributes: Dict[str, List[DiscoveredAttribute]],
) -> str:
    """Review and, where needed, restructure a domain's consolidated
    attribute set inside each facet's fixed refinement axis (P7-review, V2 /
    axis-first path)."""
    domain_attribute_structure_block = _build_domain_attribute_structure_block(
        facets, facet_attributes
    )

    return f"""You are a taxonomy quality reviewer for open-ended survey coding.

The survey question:
"{survey_question}"

Domain under review: {domain_label} — {domain_definition}

Below is the domain's full consolidated structure: per facet its segment, its
refinement axis and its attributes on their positions. Assignment has not
happened yet: restructuring is free, but everything must stay inside the
refinement axes.

<structure>
{domain_attribute_structure_block}
</structure>

Your task, judging from the survey question:
1. Verify that every attribute occupies exactly one position of its facet's
   refinement axis and that no two attributes inside a facet capture the same
   concept. Where they do: merge them (list both under source_attributes).
   Where one attribute straddles two positions: split it.
2. Sharpen names, descriptions and position boundaries so every pair of
   attributes reads as two different concepts.
3. Where two attributes in DIFFERENT facets of this domain appear to capture
   the same concept, keep both and flag the pair with a reason and a
   decision_rule — those are resolved later with assignment data.
4. Return the complete revised attribute set with full source_attributes
   bookkeeping; unaccounted or double-counted attributes invalidate the
   review.

Rules:
- Facets, segments and refinement axes are fixed context; you restructure
  attributes only, and only within their own facet.
- Descriptive wording only.

Provide your output as valid JSON following the response schema provided.
"""


class ReviewedAttributeV2(BaseModel):
    """A single attribute in the reviewed, possibly restructured, output set
    (P7-review V2 / axis-first path). Merge = several sources, one output,
    same facet. Split = the same source attribute appearing across several
    outputs, each on its own valid position, same facet. Rename/redescribe =
    one source, one output. Restructuring across facets is forbidden — a
    source may only be claimed by outputs of its own facet."""
    attribute_name: str = Field(
        ..., description="Attribute name, sharpened where needed"
    )
    attribute_description: str = Field(
        ..., description="Attribute description, reformulated for orthogonality"
    )
    facet_name: str = Field(
        ..., description="Name of the facet this attribute belongs to — must exist in this domain; sources may only come from this same facet"
    )
    position_name: str = Field(
        ..., description="Name of the position on that facet's refinement axis this attribute occupies — must exist on that facet's refinement axis"
    )
    source_attributes: List[str] = Field(
        ..., description=(
            "Every input attribute that goes into this output, by attribute_name, "
            "all from this same facet (merge: several distinct sources; split: "
            "the same source attribute listed under several outputs, each on its "
            "own position of this facet). Every input attribute must appear in "
            "source_attributes of exactly the outputs that absorb it"
        )
    )


class AttributeReviewV2Response(BaseModel):
    """P7-review output (V2 / axis-first path): the domain's complete
    revised attribute set — rewrites, merges and splits, each staying inside
    its facet's fixed refinement axis — plus any cross-facet overlap flags."""
    attributes: List[ReviewedAttributeV2] = Field(
        ..., description="The domain's complete revised attribute set"
    )
    overlap_flags: List[AttributeOverlapFlag] = Field(
        ..., description="Pairs of attributes in different facets of this domain that appear to capture the same concept"
    )


# =============================================================================
# §8 ATTRIBUTE ASSIGNMENT (P8) — per facet
# =============================================================================


def _build_attribute_codebook_block(
    attributes: List['DiscoveredAttribute'],
    refinement: Optional[dict] = None,
) -> str:
    """Format discovered attributes as a numbered list for assignment.

    When the parent facet carries a refinement axis (P6 axis-first path,
    `refinement` = {name, description, positions}), a 'Refinement axis:
    {name} — {description}' header precedes the list, and each tagged
    attribute shows its position and that position's boundary alongside the
    existing description/examples. The [A#] id is unchanged either way —
    it is what `attr_id_to_name` keys the response parse on. A facet with
    no refinement dict (untagged path) renders exactly as before."""
    refinement = refinement or {}
    # Keyed case-/padding-insensitively (mirrors `TaxonomyClassifier._norm_text`,
    # classifier.py): P7-review validates an echoed position_name loosely
    # (_norm_text) but stores it verbatim, so a case- or whitespace-variant
    # echo must still find its boundary here instead of silently losing the
    # `Boundary:` line.
    positions_by_name = {
        _norm_text(p.get("position_name", "")): p for p in refinement.get("positions", [])
    }

    lines = []
    for i, attr in enumerate(attributes, 1):
        examples = "; ".join(attr.example_observations[:3])
        item = f"[A{i}] {attr.attribute_name}\n"
        if refinement and attr.position:
            item += f"    Position: {attr.position}\n"
        item += f"    Description: {attr.attribute_description}\n"
        if refinement and attr.position:
            boundary = positions_by_name.get(_norm_text(attr.position), {}).get("boundary", "")
            if boundary:
                item += f"    Boundary: {boundary}\n"
        item += f"    Examples: {examples}"
        lines.append(item)

    block = "\n\n".join(lines)
    if refinement:
        header = f"Refinement axis: {refinement.get('name', '')} — {refinement.get('description', '')}"
        return f"{header}\n\n{block}"
    return block


def _build_decision_rules_block(decision_rules: Optional[List[str]]) -> str:
    """Format P7 overlap decision rules for the facet being dispatched.

    Empty/None renders as an empty string, so a facet with no flagged pairs
    produces a byte-identical prompt to before this block existed.
    """
    if not decision_rules:
        return ""
    lines = ["", "Decision rules for closely related attributes:"]
    lines.extend(f"- {rule}" for rule in decision_rules)
    lines.append("")
    return "\n".join(lines)


def build_attribute_assignment_prompt_single(
    *,
    survey_question: str,
    language: str,
    dataset_context_section: str,
    facet_name: str,
    facet_description: str,
    attributes: List['DiscoveredAttribute'],
    idea_label: str,
    decision_rules: Optional[List[str]] = None,
    refinement: Optional[dict] = None,
) -> str:
    """Build prompt for assigning a single idea to an attribute (L4) within a facet.

    `decision_rules`: P7 overlap decision_rule strings for pairs flagged
    WITHIN this facet (facet_a == facet_b == this facet). None/empty omits
    the block entirely.

    `refinement`: this facet's refinement axis dict (P6 axis-first path),
    {name, description, positions}. None/empty omits the axis header and
    per-attribute position/boundary lines.
    """
    attribute_codebook = _build_attribute_codebook_block(attributes, refinement=refinement)
    decision_rules_block = _build_decision_rules_block(decision_rules)

    return f"""You are a qualitative coding assistant. Assign the survey response idea below to the attribute that best captures the specific quality being described.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

<facet_context>
Facet: {facet_name} -- {facet_description}
</facet_context>

<attributes>
{attribute_codebook}
</attributes>
{decision_rules_block}
<idea>
{idea_label}
</idea>

### VALENCE (evaluation relative to attribute)
- "+" Positive — The response describes a positive instance of this attribute (meeting expectations, present, sufficient)
- "-" Negative — The response describes a negative instance of this attribute (failing expectations, absent, insufficient)
- "0" Neutral — The response is descriptive, ambiguous, or does not express evaluation
- Valence is not emotional sentiment, but evaluative direction relative to the attribute

Assign this idea to the single best-fitting attribute. Return the attribute ID (e.g. "A1", "A2"), your confidence (0.0-1.0), and the valence (+, -, or 0).

Provide your response as valid JSON following the response schema provided."""


class AttributeAssignmentResult(BaseModel):
    """Single idea-to-attribute assignment result."""
    assigned_attribute_id: str = Field(
        ..., description=(
            "The attribute ID from the [A#] prefix (e.g. 'A1', 'A3'). "
            "Return ONLY the ID, not the attribute name."
        )
    )
    confidence: float = Field(
        ..., description="Confidence in the assignment (0.0 to 1.0)"
    )
    valence: Literal["+", "-", "0"] = Field(
        default="0",
        description="Evaluative direction relative to the attribute: + positive, - negative, 0 neutral"
    )



# =============================================================================
# §9 IN-FACET ATTRIBUTE CONSOLIDATION — post-assignment, one facet at a time
# =============================================================================

def _build_suspected_overlap_block(suspected_overlap: Optional[List[Dict]]) -> str:
    """Format P7 cross-facet overlap flags touching this facet.

    Empty/None renders as an empty string, so a facet with no cross-facet
    flags produces a byte-identical prompt to before this block existed.
    """
    if not suspected_overlap:
        return ""
    lines = [
        "", "<suspected_overlap>",
        "Pairs flagged upstream as possibly one concept — verify against the actual",
        "contents below and resolve with your normal actions:",
    ]
    for flag in suspected_overlap:
        lines.append(
            f"- {flag['attr_a']} ({flag['facet_a']}) vs {flag['attr_b']} ({flag['facet_b']}): "
            f"{flag['reason']}. Rule: {flag['decision_rule']}"
        )
    lines.append("</suspected_overlap>")
    lines.append("")
    return "\n".join(lines)


def build_in_facet_consolidation_prompt(
    *,
    survey_question: str,
    language: str,
    dataset_context_section: str,
    dimension_def: Optional[DimensionDefinition],
    dimension_name: str,
    dimension_description: str,
    domain_name: str,
    domain_definition: str,
    facet_name: str,
    facet_description: str,
    attributes_block: str,
    neighbour_block: str,
    suspected_overlap: Optional[List[Dict]] = None,
) -> str:
    """Finalise the attribute inventory of ONE facet, after every idea is assigned.

    Runs after P8, so each attribute is shown with its real size and its real
    contents instead of the examples discovery guessed at. The facet is fixed:
    nothing in this call can move an attribute to another facet. When a group of
    ideas belongs elsewhere, the IDEAS move (`misfits`) and the structure stays put.

    `suspected_overlap`: P7 CROSS-facet overlap flags (facet_a != facet_b) where
    either side is this facet, as dicts with attr_a/facet_a/attr_b/facet_b/
    reason/decision_rule. None/empty omits the block entirely.
    """
    suspected_overlap_block = _build_suspected_overlap_block(suspected_overlap)

    if dimension_def:
        rules = dimension_def.prompt_rules
        attribute_guidance = rules.attribute_instruction
        attribute_key_idea = _extract_key_idea(rules.attribute_instruction)
        facet_key_idea = _extract_key_idea(rules.facet_instruction)
        domain_key_idea = _extract_key_idea(rules.domain_instruction)
    else:
        attribute_guidance = (
            "An attribute identifies the specific observable property or feature being described. "
            "It is a named property -- not a verbatim span from the response."
        )
        attribute_key_idea = "the specific observable property being described"
        facet_key_idea = "the analytical lens applied to the subject"
        domain_key_idea = "the subject the statement refers to"

    return f"""You are a taxonomy consolidation specialist for surveys.
Your task is to finalise the attribute inventory of ONE facet: "{facet_name}", inside domain "{domain_name}".

Every idea has already been assigned, so you see what each attribute ACTUALLY holds -- not what its label promised. Judge the contents, not the name.

Here is the survey context:

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

Use the survey context to:

<survey_context_usage>
- Interpret the meaning of attributes relative to the survey question
- Ensure consolidated attributes are directly relevant to what is being asked
- Preserve terminology and phrasing appropriate to the survey language
- Avoid introducing attributes that are not grounded in the question intent
</survey_context_usage>

Here is the taxonomy context you are working within:

<taxonomy_context>
This is the structure:
<taxonomy_structure>
- Dimension (L1): {dimension_name} — {dimension_description}
- Domain (L2): {domain_key_idea}
- Facet (L3): {facet_key_idea}
- Attribute (L4): {attribute_key_idea}
</taxonomy_structure>

You are working inside this one facet:
<taxonomy_facet>
Domain: {domain_name} -- {domain_definition}
Facet:  {facet_name} -- {facet_description}
</taxonomy_facet>
</taxonomy_context>

Here are this facet's attributes, with their real size and their real contents:
<facet_attributes>
{attributes_block}
</facet_attributes>

{neighbour_block}
{suspected_overlap_block}
# Understanding Attributes

Conceptualization:
{attribute_guidance}

# Consolidation Rules

<strict_consolidation_rule>
Consolidation is the goal: do NOT keep every concept separate — group. But govern grouping by these rules, in order.

1. DIMENSION FIRST (orthogonality — the guardrail).
   For each concept, determine WHICH underlying dimension it answers.
   - Concepts on DIFFERENT dimensions are orthogonal: NEVER merge them into one attribute (e.g. socio-economic class vs political orientation vs age are different dimensions).
   - Mutually-exclusive VALUES/POLES of the SAME dimension are also kept apart (e.g. "young" vs "old"); merging opposite poles creates an empty container.
   - Do NOT create separate attributes based only on the object discussed, when the same underlying value applies — an object is not a dimension.

2. PREVALENCE SETS GRANULARITY (within a dimension only).
   Each attribute shows its share of this facet. Judge size RELATIVE to its siblings, never against an absolute number.
   - The largest attributes keep their own identity — never dissolve a well-supported concept.
   - Attributes far below their siblings are GROUPED, but only with same-dimension neighbours, into one attribute that still names the shared value in plain language.
   - An attribute holding a large share AND visibly diverse contents is too abstract: SPLIT it (rule 6), do not widen it.
   - Variants that differ only in evaluative direction ("positive X" and "negative X") collapse to ONE attribute "X"; the direction is recorded separately as valence, not as separate attributes.
   Prevalence decides how finely to split WITHIN a dimension; it NEVER licenses merging ACROSS dimensions.

3. LIFT, DON'T FLATTEN.
   When grouping is needed, raise concepts to a shared higher-abstraction label that still carries their meaning — NOT a label that merely names the axis.
   FORBIDDEN: a container that only names the axis it sits on — the reader learns what was being measured, not what was said.
   REQUIRED: a label that states the value itself, so the reader knows what the respondents expressed.
   Test: read the label alone. If it tells you only which question was asked, it is a container; if it tells you what the answer was, it is a value.

4. PLAIN, MEANINGFUL LABELS.
   Name every surviving attribute in everyday language. Test: reading the label alone, a layperson knows which distinction is meant, given the survey question. No jargon, no nominalizations, no dimension-names.

5. THE FACET IS FIXED.
   Every attribute you return belongs to "{facet_name}". You cannot move an attribute to another facet, and you cannot create an attribute that belongs to another facet.
   If a GROUP OF IDEAS belongs elsewhere, report it under `misfits` — the ideas move, the attribute stays here.

6. FOUR EXITS FOR WHAT DOES NOT FIT.
   Read what each attribute actually contains. Where contents do not match the label, choose per group:
   - the group points at ONE existing attribute (in this facet or a neighbouring one)
       -> `misfits`, verdict "move": name the target attribute and the EXACT response texts
   - the group is one coherent concept that has no attribute yet
       -> action "split": name the child attributes and which EXACT response texts go to each
   - the group is diverse but genuinely related to this attribute
       -> action "widen": restate the description so it honestly covers what is there
   - the group carries NO SUBSTANTIVE CONTENT WHATSOEVER — a bare evaluation or filler with nothing said about the subject
       -> `misfits`, verdict "out"
   "out" is not an escape hatch for "this does not fit the attributes I chose". A text that names something real about the subject HAS substance: if it has no home yet, create one with "split". Only content-free text goes out.
   Moves and splits must be expressed as EXACT response texts copied from the contents shown above — never as counts, paraphrases or summaries. Every decision has to be checkable against the data.

7. ONE SOURCE, ONE DESTINATION — unless you route by text.
   Every attribute in the input must end up in exactly ONE returned attribute.
   If you want to divide one input attribute's contents over TWO returned attributes, that is a SPLIT: use action "split" for each part and list the exact response texts belonging to it in `instance_texts`.
   Listing the same source attribute under two returned attributes WITHOUT instance_texts is not interpretable — the ideas cannot be routed and will be left where they are.

8. KEEP THE VALUES THAT ARE ACTUALLY THERE.
   Grouping is not the same as discarding. If the contents hold two distinct values, return two attributes — merging them into one and sending the remainder "out" loses real answers.
   Collapsing a facet to a SINGLE attribute removes a whole level of the hierarchy: the facet name then says nothing the attribute does not already say. Do that only when the contents genuinely express one value.

Precedence when rules conflict: 1 (orthogonality) > 5 (facet is fixed) > 2 (prevalence grouping) > 4 (label clarity).
</strict_consolidation_rule>

# Required Process

Before writing your final output, think through your analysis in the scratchpad field:

**Step 1 -- Read the contents against the label**
For each attribute, compare what it HOLDS with what its name and description CLAIM. Note every group of contents that does not belong.

**Step 2 -- Identify the dimensions present**
Group the attributes by the underlying dimension each one answers. Different dimensions stay separate; never collapse across them.

**Step 3 -- Set granularity by prevalence, within a dimension**
Use the shares shown. Keep the large ones. Group the ones far below their siblings. Split the large-and-diverse ones.

**Step 4 -- Route what does not fit**
For each group from Step 1, pick one of the four exits in rule 6. When the target is in a neighbouring facet, name it exactly as it appears in the neighbour list.

**Step 5 -- Check every label is a plain, stateable value**
No dimension-name containers; each label names a value a layperson can picture.

**Step 6 -- Prepare final output**
Return the attributes that survive for THIS facet, plus every misfit group you found.

For each surviving attribute, provide:
- action: "keep", "merge", "widen" or "split"
- A short descriptive name (2-5 words)
- A description of what it captures -- a concrete, observable property (1-2 sentences)
- 2-3 representative example observations (exact text)
- source_attributes: the original attribute names that feed this one
- instance_texts: for "split" ONLY, the exact response texts routed to this child

# Output Requirements

Provide output as valid JSON following the response schema provided.

# Language Requirement

All attribute names and descriptions must be in {language}.

# Final Notes

- Attributes must be descriptive, not evaluative
- Attributes must be internally coherent (one clear concept each)
- Attributes must be externally distinctive (no overlap, no subset/superset)
- Every returned attribute belongs to "{facet_name}" — there is no other option
- All output must be in {language}

Use your scratchpad field for Steps 1-6 to show your analytical thinking. Then provide your final output as valid JSON."""


def build_neighbour_block(
    neighbours: List[Tuple[str, List[Tuple[str, int]]]],
) -> str:
    """Format adjacent facets as steer-clear context for in-facet consolidation.

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


class InFacetAttribute(BaseModel):
    """One attribute surviving in-facet consolidation. Its facet is fixed by the task."""
    action: Literal["keep", "merge", "widen", "split"] = Field(
        ..., description=(
            "What was done: 'keep' unchanged, 'merge' several sources into one, "
            "'widen' the description to cover the real contents, "
            "'split' one bucket into children (then instance_texts is required)"
        )
    )
    attribute_name: str = Field(
        ..., description="Short descriptive name for the attribute (2-5 words)"
    )
    attribute_description: str = Field(
        ..., description="What this attribute captures (1-2 sentences)"
    )
    example_observations: List[str] = Field(
        ..., description="2-3 representative observations, exact text from the contents shown"
    )
    source_attributes: List[str] = Field(
        default_factory=list,
        description=(
            "Original attribute names feeding this one (for 'keep', its own name). "
            "A source may appear under only ONE returned attribute, unless you are "
            "splitting it — then use action 'split' and fill instance_texts."
        )
    )
    instance_texts: List[str] = Field(
        default_factory=list,
        description=(
            "For action 'split' ONLY: the exact response texts routed to this child, "
            "copied verbatim from the contents shown. Required when a source attribute "
            "is divided over more than one returned attribute. Empty otherwise."
        )
    )


class MisfitGroup(BaseModel):
    """A group of ideas sitting in this facet that does not belong to the attribute holding it."""
    from_attribute: str = Field(
        ..., description="The attribute currently holding these ideas"
    )
    instance_texts: List[str] = Field(
        ..., description=(
            "The exact response texts that do not belong, copied verbatim from the "
            "contents shown. Never counts, paraphrases or summaries."
        )
    )
    verdict: Literal["move", "out"] = Field(
        ..., description=(
            "'move' when these ideas belong to a named existing attribute; "
            "'out' when they carry no substantive content at all"
        )
    )
    target_attribute: Optional[str] = Field(
        default=None,
        description=(
            "For verdict 'move': the attribute these ideas belong to, named exactly as "
            "shown in this facet or in the neighbouring facets list. Null for 'out'."
        )
    )
    reason: str = Field(
        ..., description="One sentence: why these texts do not belong where they are"
    )


class InFacetConsolidatedResponse(BaseModel):
    """Final attribute inventory for ONE facet, plus the misfits found in it."""
    scratchpad: str = Field(
        ..., description=(
            "Step-by-step reasoning: (1) read each attribute's contents against its label "
            "and note groups that do not belong, (2) group attributes by underlying dimension, "
            "(3) set granularity by prevalence using the shares shown -- keep the large, group "
            "the thin, split the large-and-diverse, (4) route each non-fitting group to one of "
            "the four exits, (5) check every label states a value rather than an axis, "
            "(6) assemble the final inventory."
        )
    )
    attributes: List[InFacetAttribute] = Field(
        ..., description="The attributes surviving for this facet, all belonging to it"
    )
    misfits: List[MisfitGroup] = Field(
        default_factory=list,
        description="Groups of ideas that do not belong to the attribute holding them"
    )

    @model_validator(mode="after")
    def _routable(self):
        """Reject an inventory whose ideas cannot be routed.

        Enforced here rather than in the prompt for the same reason `parent_facet`
        was removed from the schema: a rule the model can decline to follow is not a
        rule. instructor surfaces these messages and retries, so the model gets to
        correct itself instead of silently producing an unroutable answer.
        """
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


# =============================================================================
# IN-AXIS FACET CONSOLIDATION — post-assignment, one axis at a time
# =============================================================================

def build_in_axis_consolidation_prompt(
    *,
    survey_question: str,
    language: str,
    noun_phrase: str,
    domain_name: str,
    domain_definition: str,
    axis_name: str,
    axis_description: str,
    facets_block: str,
    neighbour_axes_block: str = "",
) -> str:
    """Consolidate the facets on ONE axis, after every idea has been assigned.

    The mirror of `build_in_facet_consolidation_prompt` one level up: where
    that one judges attributes inside a fixed facet, this judges facets inside
    a fixed axis. The axis is not part of the response schema, so a merge can
    never move a facet to another axis — when a group of ideas belongs
    elsewhere, the IDEAS move and the structure stays put.
    """
    neighbours = f"""
Here are the other axes in this domain, for reference only. They are NOT merge
candidates — they are shown so you can name a real destination when a group of
ideas belongs on another axis, and so you can write boundaries against what
actually exists next door.

<neighbour_axes>
{neighbour_axes_block}
</neighbour_axes>
""" if neighbour_axes_block else ""

    return f"""You are a qualitative research analyst specializing in open-ended survey coding. Your task is to settle the final facet set on one axis, now that every response has been assigned to a facet.

This is the language you are working in:

<language>
{language}
</language>

Here is the survey question being analyzed:

<survey_question>
"{survey_question}"

answers vary in terms of: {noun_phrase}
</survey_question>

Here is the domain and the axis you are working within:

<domain>
{domain_name} — {domain_definition}
</domain>

<axis>
{axis_name} — {axis_description}
</axis>

Here are the facets on this axis, each with the number of responses actually
assigned to it, its share of the axis, and a sample of the responses it really
holds:

<axis_facets>
{facets_block}
</axis_facets>
{neighbours}
Judge each facet on what it actually holds, not on how its label reads. The
counts and the response texts above are the evidence; the labels were written
before a single response had been assigned.

<consolidation_rules>
**1. DIMENSION FIRST.** Facets that describe different dimensions stay apart,
however similar their labels look. Orthogonality is a guardrail against merging,
never a reason to merge.

**2. PREVALENCE SETS GRANULARITY** — within one dimension only. Use the shares
shown: keep what is large, group what is thin, split what is large and diverse.

**3. LIFT, DON'T FLATTEN.** When several thin facets share a dimension, name the
concept they share. Do not dissolve them into a catch-all.

**4. PLAIN, MEANINGFUL LABELS.** A facet name states a value, not the axis it
sits on. Descriptive only — evaluation is captured per response as valence,
elsewhere.

**5. THE AXIS IS FIXED.** Every facet you return belongs to this axis. You
cannot move a facet to another axis, and you cannot add or rename axes.

**6. FOUR EXITS FOR WHAT DOES NOT FIT.** For a group of responses sitting in a
facet it does not belong to: move it to a facet that already exists (here or on
a neighbouring axis), widen the holding facet's description so it honestly
covers them, split the facet into named children, or — only when the responses
carry no substantive content at all — send them out. "Out" is not an escape
hatch for "does not fit what I chose".

**7. ONE SOURCE, ONE DESTINATION.** A source facet may be claimed by only one
returned facet, unless you route explicitly by response text.

**8. KEEP THE VALUES THAT ARE ACTUALLY THERE.** Do not collapse the axis to a
single facet because that is tidier. If the responses show four values, return
four facets.
</consolidation_rules>

Output requirements:
- All output (facet names, descriptions and rules) must be in {language}
- Copy response texts verbatim when you route them; they are matched literally

Provide your output as valid JSON following the response schema provided.
"""


class InAxisFacet(BaseModel):
    """One facet surviving consolidation on this axis."""
    action: Literal["keep", "merge", "widen", "split"] = Field(
        ..., description=(
            "keep = unchanged; merge = several source facets into this one; "
            "widen = same facet, description restated to cover what it holds; "
            "split = one source facet divided into named children"
        )
    )
    facet_name: str = Field(..., description="Short descriptive name (2-5 words)")
    facet_description: str = Field(
        ..., description="What this facet captures, faithful to the responses it holds"
    )
    inclusion_rule: str = Field(
        ..., description="What types of responses belong in this facet"
    )
    exclusion_rule: str = Field(
        default="", description="What does NOT belong, when it clarifies the boundary"
    )
    example_observations: List[str] = Field(
        ..., description="2-5 responses this facet holds, verbatim"
    )
    source_facets: List[str] = Field(
        ..., description="facet_name of every source facet consumed into this one"
    )
    instance_texts: List[str] = Field(
        default_factory=list, description=(
            "Only for a split: the exact response texts routed to this child, verbatim"
        )
    )


class FacetMisfitGroup(BaseModel):
    """A group of responses sitting in a facet they do not belong to."""
    from_facet: str = Field(..., description="The facet currently holding them")
    instance_texts: List[str] = Field(
        ..., description="The exact response texts, verbatim"
    )
    verdict: Literal["move", "out"] = Field(
        ..., description="move = to a named existing facet; out = no substantive content"
    )
    target_facet: str = Field(
        default="", description="For 'move': the facet they belong to. Empty for 'out'."
    )
    reason: str = Field(..., description="One sentence on why they do not belong")


class InAxisConsolidatedResponse(BaseModel):
    """Final facet inventory for ONE axis, plus the misfits found on it."""
    scratchpad: str = Field(
        ..., description=(
            "Step-by-step reasoning: (1) read each facet's contents against its label "
            "and note groups that do not belong, (2) group facets by underlying "
            "dimension, (3) set granularity by prevalence using the shares shown, "
            "(4) route each non-fitting group to one of the four exits, (5) check "
            "every label states a value rather than the axis, (6) assemble the "
            "final inventory."
        )
    )
    facets: List[InAxisFacet] = Field(
        ..., description="The complete facet set for this axis after consolidation"
    )
    misfits: List[FacetMisfitGroup] = Field(
        default_factory=list, description="Response groups that do not belong where they sit"
    )


# =============================================================================
# VALENCE-NEUTRAL RENAME (collapse valence-split attribute pairs)
# =============================================================================

class ValenceNeutralAttribute(BaseModel):
    """One descriptive, valence-neutral attribute replacing a valence-split pair."""
    pair_id: int = Field(..., description="The id of the attribute pair this replaces")
    attribute_name: str = Field(
        ..., description="One descriptive, valence-neutral attribute name (2-5 words)"
    )
    attribute_description: str = Field(
        ..., description="A 1-2 sentence valence-neutral description"
    )


class ValenceNeutralRenameResponse(BaseModel):
    """Neutral replacements for the supplied valence-split attribute pairs."""
    attributes: List[ValenceNeutralAttribute] = Field(
        ..., description="Exactly one neutral attribute per input pair_id"
    )


def build_valence_neutral_rename_prompt(pairs: list, language: str = "Dutch") -> str:
    """Collapse valence-split attribute pairs into one descriptive, valence-neutral
    attribute each. `pairs`: list of dicts with pair_id, name_a, desc_a, name_b,
    desc_b, samples.
    """
    blocks = []
    for p in pairs:
        samples = ", ".join(f'"{s}"' for s in p.get("samples", []))
        blocks.append(
            f"[{p['pair_id']}]\n"
            f'  A: "{p["name_a"]}" — {p.get("desc_a", "")}\n'
            f'  B: "{p["name_b"]}" — {p.get("desc_b", "")}\n'
            f"  example mentions: {samples}"
        )
    pairs_block = "\n\n".join(blocks)

    return f"""You are cleaning up a taxonomy. Each numbered pair below wrongly split ONE concept by evaluative direction (valence): the two attributes mean the same thing, but one captures the positive side and the other the negative/neutral side. Valence has been baked into the attribute, which is wrong — valence is recorded separately per response.

For each pair, produce ONE descriptive, valence-neutral attribute that covers both sides:
- The name (2-5 words, in {language}) and description (1-2 sentences, in {language}) must be purely descriptive.
- Do NOT encode positive/negative/good/bad — that direction is captured separately as valence.
- Name the underlying subject the two share (e.g. a "positive impression" + "negative impression" pair becomes "overall impression").

Pairs:
{pairs_block}

Return exactly one entry per pair_id. Begin now and provide your output as valid JSON following the response schema provided."""
