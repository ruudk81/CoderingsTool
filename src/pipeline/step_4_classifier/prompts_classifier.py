"""
Prompt builders for Taxonomy Classifier (P1-P10).

Organized in pipeline processing order:
  §0   Dimension Context Block (shared helper)
  §1a  Axis Discovery (P1a: per-domain axis system discovery)
  §1b  Tagged Facet Discovery (P1b: per-domain, chunked, axis-tagged)
  §1   Facet Discovery (P1: per-domain, chunked)
  §2   Facet Consolidation (P2: merge chunk-level facets)
  §2a  Segment Facet Consolidation (P2, axis-first: per (axis, segment),
       plus each facet's refinement axis)
  §3   Facet Review (P3: per-domain quality gate)
  §3a  Facet Review V2 (P3-review, axis-first: widened mandate — rewrite,
       merge and split inside the fixed axis system)
  §4   Facet Assignment (P4: per-domain, batched)
  §5   Attribute Discovery (P5: per facet within domain)
  §5a  Position-Tagged Attribute Discovery (P5, axis-first: per facet,
       chunked, tagged to the facet's refinement axis)
  §6   Attribute Chunk Consolidation (P6: merge chunk-level attributes)
  §6a  Per-Position Attribute Consolidation (P6, axis-first: per position,
       plus adjudication of new-position proposals)
  §7   Attribute Review (P7: per-domain quality gate on the consolidated attribute set)
  §8   Attribute Assignment (P8: per facet)
  §9   In-Facet Attribute Consolidation (P9: post-assignment, one facet at a time)
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
# §1a AXIS DISCOVERY (P1a) — per-domain axis system discovery
# =============================================================================

def build_axis_discovery_prompt(
    *,
    survey_question: str,
    primary_dimension: str,
    domain_label: str,
    domain_definition: str,
    domain_boundary_test: str,
    sample_observations: List[str],
) -> str:
    """Discover the axes along which observations in a domain differ (P1a)."""
    observations_block = "\n".join(f"- {obs}" for obs in sample_observations)

    return f"""You are a taxonomy methodologist for open-ended survey coding.

The survey question:
"{survey_question}"

Primary dimension of the taxonomy: {primary_dimension}

Domain under analysis: {domain_label}
Domain definition: {domain_definition}
Domain boundary test: {domain_boundary_test}

Below is a broad sample of observations from this domain (drawn across the
whole domain, not one slice):

<observations>
{observations_block}
</observations>

Your task: identify the AXES along which these observations differ — the
underlying dimensions that explain why two observations in this domain are
about different things. Think of the domain as a space; you are naming its
coordinate axes so that categories can later be built as non-overlapping
segments on those axes.

For every axis:
1. Name it and describe, in one or two sentences, the difference in the data
   it captures.
2. Divide it into 2-6 segments that are mutually exclusive by their boundary
   statements: each boundary is one routing sentence phrased against the
   neighbouring segments ("is about X -> this segment; is about Y -> <other>").
3. Quote 2-5 example observations per segment, verbatim from the sample.
4. Add exactly one residual segment (is_residual = true) for observations that
   belong to this domain but do not specify a value on this axis. Do not
   invent content for it; its boundary is "names no recognisable value on
   this axis".

Rules:
- Axes must come from the data in front of you — never from general knowledge
  of the topic. If the sample only supports one axis, return one axis.
- Prefer few axes that carry many observations over many thin axes.
- Segments are conceptual values on the axis, not levels of specificity:
  "general/unspecified" is what the residual segment is for.
- Descriptive wording only; evaluation is captured per idea as valence,
  elsewhere.

Provide your output as valid JSON following the response schema provided.
"""


class AxisSegment(BaseModel):
    """A single segment (conceptual value) on a discovered axis."""
    segment_name: str = Field(
        ..., description="Short descriptive name for the segment — a value on the axis"
    )
    segment_description: str = Field(
        ..., description="What this segment captures on the axis (1-2 sentences)"
    )
    boundary: str = Field(
        ..., description=(
            "One routing sentence phrased against the neighbouring segments "
            "(\"is about X -> this segment; is about Y -> <other>\")"
        )
    )
    example_observations: List[str] = Field(
        ..., description="2-5 example observations for this segment, verbatim from the sample"
    )
    is_residual: bool = Field(
        default=False, description=(
            "True for exactly one segment per axis: the residual segment for observations "
            "that belong to this domain but name no recognisable value on this axis"
        )
    )


class DiscoveredAxis(BaseModel):
    """An axis along which observations within a domain differ."""
    axis_name: str = Field(
        ..., description="Short name for the axis — the underlying dimension the observations differ along"
    )
    axis_description: str = Field(
        ..., description="One or two sentences describing the difference in the data this axis captures"
    )
    segments: List[AxisSegment] = Field(
        ..., description="2-6 substantive segments plus exactly one residual segment"
    )


class AxisSystemResponse(BaseModel):
    """P1a output: the axis system discovered for a single domain."""
    axes: List[DiscoveredAxis] = Field(
        ..., description="1-4 axes discovered for this domain"
    )


# =============================================================================
# §1b TAGGED FACET DISCOVERY (P1b) — per-domain chunked discovery inside a
# pre-established, fixed axis system. Used only for domains that got a
# validated axis system from P1a (axis_first_enabled); domains without one
# keep the untagged §1 path below untouched.
# =============================================================================

def _build_axis_system_block(axis_system: AxisSystemResponse) -> str:
    """Render a validated axis system as prompt text: one 'Axis: name —
    description' line per axis, followed by its segments (residual segment
    last, name suffixed ' (residual)')."""
    axis_blocks = []
    for axis in axis_system.axes:
        lines = [f"Axis: {axis.axis_name} — {axis.axis_description}"]
        non_residual = [seg for seg in axis.segments if not seg.is_residual]
        residual = [seg for seg in axis.segments if seg.is_residual]
        for seg in non_residual + residual:
            suffix = " (residual)" if seg.is_residual else ""
            lines.append(f"  - {seg.segment_name}{suffix}: {seg.segment_description}")
            lines.append(f"    Boundary: {seg.boundary}")
        axis_blocks.append("\n".join(lines))
    return "\n\n".join(axis_blocks)


def build_tagged_facet_discovery_prompt(
    *,
    survey_question: str,
    domain_label: str,
    domain_definition: str,
    axis_system: AxisSystemResponse,
    chunk_observations: List[str],
) -> str:
    """Discover facets (L3) from a chunk of observations, each proposal tagged
    to exactly one (axis, segment) of the domain's fixed axis system (P1b)."""
    axis_system_block = _build_axis_system_block(axis_system)
    observations_block = "\n".join(f"{i}. {obs}" for i, obs in enumerate(chunk_observations, 1))

    return f"""You are a qualitative research analyst for open-ended survey coding.

The survey question:
"{survey_question}"

Domain: {domain_label} — {domain_definition}

This domain's axis system was established beforehand and is FIXED — you work
inside it, you do not change it:

<axis_system>
{axis_system_block}
</axis_system>

Below are the observations of your chunk:

<observations>
{observations_block}
</observations>

Your task: propose facets for this chunk, where every facet is a coherent
recurring theme that occupies EXACTLY ONE segment of ONE axis above. Tag every
proposal with that (axis_name, segment_name) — proposals with tags outside the
system are rejected unseen.

Rules:
- One segment can receive multiple proposals (consolidation merges later);
  a proposal can never span two segments — split it.
- An observation pattern that names no recognisable value on an axis belongs
  to that axis's residual segment; never invent a general catch-all facet.
- Ground every facet in this chunk's observations (quote 2-5 as examples).
- Descriptive wording only.

Provide your output as valid JSON following the response schema provided.
"""


class TaggedFacetProposal(BaseModel):
    """A facet proposal from a chunk, tagged to a segment of the domain's
    fixed axis system (P1b output)."""
    facet_name: str = Field(
        ..., description="Short descriptive name for the facet (2-5 words)"
    )
    facet_description: str = Field(
        ..., description="What this facet captures — the specific viewpoint or aspect (1-2 sentences)"
    )
    axis_name: str = Field(
        ..., description="Name of the axis this facet's segment belongs to — must exist in the axis system above"
    )
    segment_name: str = Field(
        ..., description="Name of the segment on that axis this facet occupies — must exist on that axis above"
    )
    example_observations: List[str] = Field(
        ..., description="2-5 example observations grounding this facet, verbatim from the chunk"
    )


class TaggedFacetDiscoveryResponse(BaseModel):
    """P1b output: tagged facet proposals discovered in a single chunk."""
    proposals: List[TaggedFacetProposal] = Field(
        ..., description="Facet proposals discovered in this chunk, each tagged to one (axis, segment)"
    )


# =============================================================================
# §1 FACET DISCOVERY (P1) — per-domain chunked pattern extraction
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
        #facet_guidance = rules.facet_instruction
        facet_definition = _extract_definition(rules.facet_instruction)
        facet_key_idea = _extract_key_idea(rules.facet_instruction)
        attribute_key_idea = _extract_key_idea(rules.attribute_instruction)
        noun_phrase = dimension_def.noun_phrase_descriptor
        domain_key_idea = _extract_key_idea(rules.domain_instruction)
    else:
        facet_guidance = "Identify the specific viewpoint or characteristic within the domain."
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

    return f"""You are a qualitative research analyst specializing in survey response analysis. 
Your task is to identify the fewest recurring facets that provide full coverage of a set of observations from a survey.

{facet_definition} Facets must be:
- Descriptive and data-grounded (not evaluative)
- Internally coherent (one clear underlying concept)
- Externally distinctive (ontologically distinct and semantically separable from other facets)
- Strictly within domain boundaries
- Supported by multiple observations or repeated patterns

Here is the survey context you are working with:

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
{dimension_description}
</survey_context>

Here is the taxonomy context that defines your working framework:

<taxonomy_context>
This is the structure:
<taxonomy_structure>
- Dimension (L1): {noun_phrase}
- Domain (L2): {domain_key_idea}
- Facet (L3): {facet_key_idea}
- Attribute (L4): {attribute_key_idea}
</taxonomy_structure>

You are working within this domain:

<taxonomy_domain>
{partition_name} — {partition_definition}{domain_boundary_block}
</taxonomy_domain>
{excluded_block}
</taxonomy_context>

Here are the observations you need to analyze:

<observations>
{observations_block}
</observations>

Before providing your final output, you must work through your analysis systematically in a scratchpad section. Follow these steps:

**Step 1: Cluster observations**
Group similar observations together based on shared descriptive meaning. Identify recurring patterns in what is being described. Focus on the type of quality, characteristic, principle, or practice being described, not on evaluation or sentiment.

**Step 2: Identify candidate facets**
Based on your clusters, identify candidate facets. For each candidate facet, assess:
- The facet name (2-5 words in {language})
- The underlying type of quality or attribute it captures
- Which observations support it
- Whether it is internally coherent (captures one clear concept)
- Whether it is ontologically distinct from other candidate facets

Remember: a facet identifies an analytical lens, not a single concrete observation. It captures a type of meaning that recurs across multiple observations.

**Step 3: Verify internal coherence**
Check whether each candidate facet captures one clear underlying concept. Reject or split candidate facets that:
- Combine multiple different kinds of phenomena
- Mix descriptive content with evaluation
- Are too broad to support clear coding

**Step 4: Verify distinctness**
Check each pair of candidate facets to ensure they are:
- Ontologically distinct (not overlapping in conceptual space; one is not a subset of another)
- Semantically separable (a coder would clearly know which facet applies, with no ambiguity)
- Not two different lenses on the same phenomenon

If two facets fail this test, consolidate them into one broader facet or redefine the boundaries more clearly.

**Step 5: Verify domain boundaries**
Check that each retained facet falls strictly within the included domain of {partition_name}.

Exclude facets that belong more naturally to other domains, including:
{excluded_block_light}

**Step 6: Prepare final output**
Retain only the dominant facets that pass all checks above. For each facet, prepare:
- A short descriptive name in {language} (2-5 words)
- A description in {language} of what the facet captures (1-2 sentences)
- 3-5 representative observations from the input, using the exact observation text (not observation numbers)

Your response must be structured as valid JSON with two fields:
1. "scratchpad": containing your step-by-step analytical reasoning (Steps 1-6)
2. "facets": an array of discovered facets, each with "facet_name", "facet_description", and "example_observations"

Important requirements:
- All output (facet names, descriptions, and example observations) must be in {language}
- Facets must be descriptive, not evaluative
- Facets must be grounded in repeated patterns across observations
- Each facet must capture one type of quality, not multiple
- Use exact observation text in the examples
- Only include facets that strictly belong to the included domain
- Aim for the fewest facets that provide full coverage of the observations

Provide your complete analysis in the scratchpad field, then provide your final facets as a JSON array.
"""

class DiscoveredFacet(BaseModel):
    """A facet (L3) discovered from observations within a domain."""
    facet_name: str = Field(
        ..., description="Short descriptive name for the facet (2-5 words)"
    )
    facet_description: str = Field(
        ..., description="What this facet captures — the specific viewpoint or aspect (1-2 sentences)"
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
        facet_guidance = rules.facet_instruction
        facet_definition = _extract_definition(rules.facet_instruction)
        facet_key_idea = _extract_key_idea(rules.facet_instruction)
        attribute_key_idea = _extract_key_idea(rules.attribute_instruction)
        noun_phrase = dimension_def.noun_phrase_descriptor
        domain_key_idea = _extract_key_idea(rules.domain_instruction)
    else:
        facet_guidance = "Identify the specific viewpoint or characteristic within the domain."
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
) -> str:
    """Format discovered facets as a numbered codebook for assignment."""
    lines = []
    for i, facet in enumerate(facets, 1):
        examples = "; ".join(facet.example_observations[:3])
        lines.append(
            f"[F{i}] {facet.facet_name}\n"
            f"    Description: {facet.facet_description}\n"
            + (f"    Boundary: {facet.boundary_test}\n" if facet.boundary_test else "")
            + f"    Examples: {examples}"
        )
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
) -> str:
    """Build prompt for assigning a single idea to a facet (L3)."""
    facet_codebook = _build_facet_codebook_block(facets)

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
# §8 ATTRIBUTE ASSIGNMENT (P8) — per facet
# =============================================================================


def _build_attribute_codebook_block(
    attributes: List['DiscoveredAttribute'],
) -> str:
    """Format discovered attributes as a numbered list for assignment."""
    lines = []
    for i, attr in enumerate(attributes, 1):
        examples = "; ".join(attr.example_observations[:3])
        lines.append(
            f"[A{i}] {attr.attribute_name}\n"
            f"    Description: {attr.attribute_description}\n"
            f"    Examples: {examples}"
        )
    return "\n\n".join(lines)


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
) -> str:
    """Build prompt for assigning a single idea to an attribute (L4) within a facet.

    `decision_rules`: P7 overlap decision_rule strings for pairs flagged
    WITHIN this facet (facet_a == facet_b == this facet). None/empty omits
    the block entirely.
    """
    attribute_codebook = _build_attribute_codebook_block(attributes)
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
