"""
Prompt builders for Taxonomy Classifier (P1-P7).

Organized in pipeline processing order:
  §0   Dimension Context Block (shared helper)
  §1   Facet Discovery (P1: per-domain, chunked)
  §2   Facet Consolidation (P2: merge chunk-level facets)
  §3   Facet Assignment (P3: per-domain, batched)
  §4   Attribute Discovery (P4: per facet within domain)
  §5   Attribute Chunk Consolidation (P5: merge chunk-level attributes)
  §6   Attribute Assignment (P6: per facet)
  §7   Attribute Consolidation (P7: cross-facet dedup within domain)
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field

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


def build_dimension_context_block(
    *,
    dimension_def: Optional[DimensionDefinition],
    dimension_name: str,
    dimension_description: str,
    domain_name: str,
    domain_definition: str,
) -> str:
    """Build a dimension-specific taxonomy context block for prompts. """
    if dimension_def is None:
        # Fallback: generic taxonomy block (no dimension-specific semantics)
        return f"""<taxonomy_context>
Dimension: {dimension_name} — {dimension_description}
Domain: {domain_name} — {domain_definition}

Taxonomy levels:
- Dimension (L1): the type of information expressed in the response
- Domain (L2): the subject the statement refers to
- Facet (L3): the analytical lens applied to the subject
- Attribute (L4): the specific observable property being described
</taxonomy_context>"""

    rules = dimension_def.prompt_rules

    # Extract "Key idea:" summaries from instructions
    domain_key_idea = _extract_key_idea(rules.domain_instruction)
    facet_key_idea = _extract_key_idea(rules.facet_instruction)
    attribute_key_idea = _extract_key_idea(rules.attribute_instruction)

    # Build worked example from dimension's examples
    example_block = ""
    if dimension_def.examples:
        ex = dimension_def.examples[0]
        example_block = f"""
Example (from a different survey):
  Survey: {ex.survey_context}
  Response: "{ex.response}"
  Domain: {ex.domain}
  Facet: {ex.facet}
  Instance: {ex.instance}"""

    return f"""<taxonomy_context>
Dimension: {dimension_name} — {dimension_description}
Domain: {domain_name} — {domain_definition}

Taxonomy levels for this dimension:
- Dimension (L1): {dimension_def.noun_phrase_descriptor}
- Domain (L2): {domain_key_idea}
- Facet (L3): {facet_key_idea}
- Attribute (L4): {attribute_key_idea}
{example_block}
</taxonomy_context>"""


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
) -> str:
    """Discover facets (L3) from a chunk of observations within a domain."""
    observations_block = "\n".join(f"{i}. {obs}" for i, obs in enumerate(observations, 1))

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
{partition_name} — {partition_definition}
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

**Rule 1: MERGE OVERLAP (MANDATORY)**
All facets that conceptually overlap or are variants of the same idea MUST be merged into a single facet.

**Rule 2: ORTHOGONALITY (MAIN RULE)**
For each pair of facets, ask: "Can a single observation plausibly fall under both?"
- If YES → merge them
- If DOUBTFUL → merge them
- Only if CLEARLY NO → keep them separate

**Rule 3: NO HIERARCHY**
Facets must not have a general-to-specific or principle-to-application relationship.
If one facet is a more general or specific version of another → merge them

**Rule 4: NO OBJECT SPLITTING**
Do not create separate facets based solely on the object being discussed (e.g., "humans" vs. "animals").
If the same underlying principle or concept applies → merge them

**Rule 5: MINIMALITY (MANDATORY)**
Use the smallest number of facets that provides full coverage of the observations.
If a facet is not strictly necessary to capture a distinct concept → remove it or merge it

**Disambiguation Test:**
For any pair of facets, ask: "Can a clear rule assign every observation to exactly one facet?"
- If NO → merge them

**Precedence When Rules Conflict:**
1. Non-overlap (orthogonality) is most important
2. Minimality (merge unless clearly distinct)
3. Clarity for annotation

**When in doubt → merge facets**

# Step-by-Step Analysis Process

Before providing your final output, work through your analysis systematically in a scratchpad. Follow these steps:

**Step 1 -- Scan chunk-level facets**
Review all facets from all chunks. Note recurring themes, similar concepts, and obvious duplicates.

**Step 2 -- Group overlapping facets**
Identify and group facets that describe the same or overlapping concepts across different chunks.

**Step 3 -- Apply orthogonality test**
For each pair of candidate consolidated facets, ask: "Can a single observation plausibly fall under both?" If yes or doubtful → merge them.

**Step 4 -- Apply disambiguation test**
For each pair of remaining facets, ask: "Can a clear rule assign every observation to exactly one facet?" If no → merge them.

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
            "(3) apply orthogonality test — merge if observation could fall under both, "
            "(4) apply disambiguation test — merge if no clear assignment rule, "
            "(5) verify domain boundaries — exclude facets belonging to other domains, "
            "(6) prepare final minimal set of consolidated facets"
        )
    )
    facets: List[DiscoveredFacet] = Field(
        ..., description="Fewest mutually exclusive facets needed for full coverage, consolidated from all chunks"
    )


# =============================================================================
# §3 FACET ASSIGNMENT (P3) — per-domain batched assignment
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
            f"    Examples: {examples}"
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
# §4 ATTRIBUTE DISCOVERY (P4) — per facet within domain
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


class AttributeDiscoveryResult(BaseModel):
    """P4 output: attributes discovered within a facet."""
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
# §5 ATTRIBUTE CHUNK CONSOLIDATION (P5) — merge chunk-level attributes within facet
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

**Rule 1: MERGE OVERLAP (MANDATORY)**
All attributes that conceptually overlap or are variants of the same idea MUST be merged into a single attribute.

**Rule 2: ORTHOGONALITY (MAIN RULE)**
For each pair of attributes, ask: "Can a single observation plausibly fall under both?"
- If YES → merge them
- If DOUBTFUL → merge them
- Only if CLEARLY NO → keep them separate

**Rule 3: NO HIERARCHY**
Attributes must not have a general-to-specific or principle-to-application relationship.
If one attribute is a more general or specific version of another → merge them

**Rule 4: NO OBJECT SPLITTING**
Do not create separate attributes based solely on the object being discussed (e.g., "humans" vs. "animals").
If the same underlying principle or concept applies → merge them

**Rule 5: MINIMALITY (MANDATORY)**
Use the smallest number of attributes that provides full coverage of the observations.
If an attribute is not strictly necessary to capture a distinct concept → remove it or merge it

**Disambiguation Test:**
For any pair of attributes, ask: "Can a clear rule assign every observation to exactly one attribute?"
- If NO → merge them

**Precedence When Rules Conflict:**
1. Non-overlap (orthogonality) is most important
2. Minimality (merge unless clearly distinct)
3. Clarity for annotation

**When in doubt → merge attributes**

# Step-by-Step Analysis Process

Before providing your final output, work through your analysis systematically in a scratchpad. Follow these steps:

**Step 1 -- Scan chunk-level attributes**
Review all attributes from all chunks. Note recurring themes, similar concepts, and obvious duplicates.

**Step 2 -- Group overlapping attributes**
Identify and group attributes that describe the same or overlapping concepts across different chunks.

**Step 3 -- Apply orthogonality test**
For each pair of candidate consolidated attributes, ask: "Can a single observation plausibly fall under both?" If yes or doubtful → merge them.

**Step 4 -- Apply disambiguation test**
For each pair of remaining attributes, ask: "Can a clear rule assign every observation to exactly one attribute?" If no → merge them.

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
- Prioritize **minimality**: fewer attributes are better if they provide full coverage
- **When in doubt, merge** rather than keep separate
- All output must be in {language}

Begin by writing your step-by-step analysis in the scratchpad field, then provide your final consolidated attributes in valid JSON format."""


class AttributeChunkConsolidatedResponse(BaseModel):
    """Consolidated attributes after merging chunk-level discoveries within a facet."""
    scratchpad: str = Field(
        ..., description=(
            "Step-by-step reasoning before consolidating attributes: "
            "(1) scan chunk-level attributes for recurring themes and duplicates, "
            "(2) group overlapping attributes across chunks, "
            "(3) apply orthogonality test -- merge if observation could fall under both, "
            "(4) apply disambiguation test -- merge if no clear assignment rule, "
            "(5) verify facet boundaries -- exclude attributes belonging to other facets, "
            "(6) prepare final minimal set of consolidated attributes"
        )
    )
    attributes: List[DiscoveredAttribute] = Field(
        ..., description="Fewest mutually exclusive attributes needed for full coverage, consolidated from all chunks"
    )

# =============================================================================
# §6 ATTRIBUTE ASSIGNMENT (P6) — per facet
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


def build_attribute_assignment_prompt_single(
    *,
    survey_question: str,
    language: str,
    dataset_context_section: str,
    facet_name: str,
    facet_description: str,
    attributes: List['DiscoveredAttribute'],
    idea_label: str,
) -> str:
    """Build prompt for assigning a single idea to an attribute (L4) within a facet."""
    attribute_codebook = _build_attribute_codebook_block(attributes)

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
# §7 ATTRIBUTE CONSOLIDATION (P7) — cross-facet dedup within domain
# =============================================================================

def build_attribute_consolidation_prompt(
    *,
    survey_question: str,
    language: str,
    dataset_context_section: str,
    dimension_def: Optional[DimensionDefinition],
    dimension_name: str,
    dimension_description: str,
    domain_name: str,
    domain_definition: str,
    facet_attributes_block: str,
    excluded_domains: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """Consolidate attributes across facets within a domain into a MECE set.

    P7: after P4 discovers attributes per facet independently, this step
    deduplicates overlapping attributes across facets and assigns each
    surviving attribute to its best-fitting facet.
    """
    # Dimension-specific guidance
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

    excluded_block = _build_exclusion_block(
        excluded_domains or [], "excluded_domains"
    )
    excluded_block_light = _build_exclusion_block_light(
        excluded_domains or []
    )

    return f"""You are a taxonomy consolidation specialist for surveys.
Your task is to deduplicate attributes across facets within the domain "{domain_name}", producing a single MECE attribute inventory for the entire domain.

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

You are working within this domain:
<taxonomy_domain>
{domain_name} -- {domain_definition}
</taxonomy_domain>
{excluded_block}
</taxonomy_context>

Here are all facets and their discovered attributes:
<facet_attributes>
{facet_attributes_block}
</facet_attributes>

# Understanding Attributes

Conceptualization:
{attribute_guidance}

# Attribute Consolidation Rules

<strict_consolidation_rule>
1. PREVALENCE WEIGHTING
Attributes MUST be primarily driven by the **number of ideas linked to attributes**.

- Attributes with HIGH idea counts MUST form the **core structure of the codebook**.
- Attributes with LOW idea counts MUST NOT become standalone attributes unless absolutely necessary.
- LOW-prevalence attributes SHOULD be:
  - merged into the closest HIGH-prevalence phenomenon, OR
  - grouped into a broader combined phenomenon.

If forced to choose between:
- conceptual nuance
- prevalence dominance

--> ALWAYS prioritize prevalence dominance.

2. MERGE BIAS
When in doubt:
- MERGE rather than split
- Especially when an attribute has relatively few ideas

Attributes with low prevalence (e.g., <10-15 ideas) should almost never result in standalone attributes.

3. MERGE OVERLAP (MANDATORY)
All attributes that conceptually overlap or are variants of the same idea must be merged, even if they were discovered under different facets.

4. ORTHOGONALITY (MAIN RULE)
For each pair of attributes:
"Can a single observation plausibly fall under both?"

- Yes -> merge
- Doubt -> merge
- Only if clearly no -> keep separate

5. NO HIERARCHY
Attributes must not be:
- general vs. specific
- principle vs. application
If this occurs -> merge

6. NO OBJECT SPLITTING
Do not split based on object (e.g., humans vs. animals)
If the same underlying principle applies -> merge

7. MINIMALITY (MANDATORY)
Use the smallest number of attributes that provides full coverage.
If an attribute is not strictly necessary -> remove it

8. FACET ASSIGNMENT
Assign each surviving attribute to the ONE facet where it fits best.
Do NOT restructure or rename facets -- only deduplicate attributes.
</strict_consolidation_rule>

<disambiguation_test>
For any pair of attributes:
"Can a clear rule assign every observation to exactly one attribute?"
- No -> merge
</disambiguation_test>

<precedence_rule>
When rules conflict, prioritize:
1. Non-overlap (orthogonality)
2. Minimality (merge unless clearly distinct)
3. Clarity for annotation

When in doubt -> merge attributes
</precedence_rule>

# Required Process

Before writing your final output, think through your analysis in the scratchpad field:

**Step 1 -- Identify High-Prevalence Anchors**
- Identify attributes with the highest number of ideas.
- Treat these as the PRIMARY building blocks of the consolidated inventory.

**Step 2 -- Map Lower-Prevalence Attributes**
- Map lower-prevalence attributes onto these high-prevalence anchors wherever possible.
- Only keep an attribute separate if it:
  - is conceptually distinct AND
  - cannot reasonably be merged.

**Step 3 -- Apply orthogonality and disambiguation tests**
For each pair of candidate attributes, apply the orthogonality test and disambiguation test. Merge if either test fails.

**Step 4 -- Verify domain boundaries**
Ensure each retained attribute belongs to this domain and not to any excluded domain:
{excluded_block_light}

**Step 5 -- Justify Low-Prevalence Attributes (MANDATORY)**
- If any attribute is primarily based on low idea counts:
- Explicitly justify why it was NOT merged into a higher-prevalence phenomenon.

**Step 6 -- Prepare final output**
Return only the minimal set of consolidated attributes that pass all checks.

For each consolidated attribute, provide:
- A short descriptive name (2-5 words)
- A description of what the attribute captures -- a concrete, observable property (1-2 sentences)
- The parent facet this attribute best belongs to
- 2-3 representative example observations (exact text)
- source_attributes: list of original attribute names that were merged into this one

# Output Requirements

Provide output as valid JSON following the response schema provided.

# Language Requirement

All attribute names and descriptions must be in {language}.

# Final Notes

- Attributes must be descriptive, not evaluative
- Attributes must be grounded in repeated patterns across observations
- Attributes must be internally coherent (one clear concept each)
- Attributes must be externally distinctive (no overlap, no subset/superset)
- Each attribute must be assigned to exactly ONE parent facet (best fit)
- All output must be in {language}

Use your scratchpad field for Steps 1-6 to show your analytical thinking. Then provide your final output as valid JSON."""


class ConsolidatedAttribute(BaseModel):
    """An attribute assigned to its best-fitting facet after cross-facet consolidation."""
    attribute_name: str = Field(
        ..., description="Short descriptive name for the attribute (2-5 words)"
    )
    attribute_description: str = Field(
        ..., description="What this attribute captures (1-2 sentences)"
    )
    parent_facet: str = Field(
        ..., description="The facet name this attribute best belongs to"
    )
    example_observations: List[str] = Field(
        ..., description="2-3 representative observations from the input"
    )
    source_attributes: List[str] = Field(
        default_factory=list,
        description="Original attribute names that were merged into this consolidated attribute"
    )


class AttributeConsolidatedResponse(BaseModel):
    """Consolidated attributes after cross-facet deduplication within a domain."""
    scratchpad: str = Field(
        ..., description=(
            "Step-by-step reasoning before consolidating attributes: "
            "(1) identify high-prevalence anchors from idea counts, "
            "(2) map lower-prevalence attributes onto anchors, "
            "(3) apply orthogonality and disambiguation tests, "
            "(4) verify domain boundaries -- exclude attributes belonging to other domains, "
            "(5) justify any low-prevalence attributes kept separate, "
            "(6) prepare final minimal set of consolidated attributes"
        )
    )
    attributes: List[ConsolidatedAttribute] = Field(
        ..., description="Deduplicated attributes, each assigned to its best-fitting facet"
    )
