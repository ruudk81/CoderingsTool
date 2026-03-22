"""
Prompts and Pydantic response models for Category Discovery v3.

Organized in pipeline processing order:
  §0   Dimension Context Block (shared helper)
  §1   Facet Discovery (P1: per-domain, chunked)
  §2   Facet Consolidation (P2: merge chunk-level facets)
  §3   Facet Assignment (P3: per-domain, batched)
  §4   Attribute Discovery (P4: per facet within domain)
  §5   Attribute Chunk Consolidation (P5: merge chunk-level attributes)
  §6   Attribute Assignment (P6: per facet)
  §7   Attribute Consolidation (P7: cross-facet dedup within domain)
  §8   Code Generation from Attributes (P8: per domain)
  §9   Codebook Consolidation (P9: cross-domain merge)
  §10  Code Assignment (P10: single idea)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from development.step_3_ideaExtractor.dimension_data import DimensionDefinition


# =============================================================================
# §0 DIMENSION CONTEXT BLOCK — shared helper for all prompts
# =============================================================================

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
        f"\nYou must NOT include categories that belong to these excluded areas:\n"
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
        facet_guidance = rules.facet_instruction
        facet_key_idea = _extract_key_idea(rules.facet_instruction)
        attribute_key_idea = _extract_key_idea(rules.attribute_instruction)
        noun_phrase = dimension_def.noun_phrase_descriptor
        domain_key_idea = _extract_key_idea(rules.domain_instruction)
    else:
        facet_guidance = "Identify the specific viewpoint or characteristic within the domain."
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

Here is the survey context:

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

Here is the taxonomy context you are working within:

<taxonomy_context>
This is the structure:
<taxonomy_structure>
- Dimension (L1): {noun_phrase}
- Domain (L2): {domain_key_idea}
- Facet (L3): {facet_key_idea}
- Attribute (L4): {attribute_key_idea}
</taxonomy_structure>

You are working within this dimension:
<taxonomy_dimension>
{dimension_name} — {dimension_description}
</taxonomy_dimension>

And you are working within this domain:
<taxonomy_domain>
{partition_name} — {partition_definition}
</taxonomy_domain>
{excluded_block}
Here is guidance on what facets are and how they should be defined:

<facet_definition_guidance>
Target abstraction level: FACET (L3)
{facet_guidance}

Each facet must:
- Be a descriptive, data-grounded category based on shared meaning across multiple attributes
- Be non-evaluative (no judgment, sentiment, or valence)
- Stay strictly within the domain boundaries
- Be internally coherent (one clear underlying concept)
- Be externally distinctive:
  * Ontologically distinct (no overlap, no subset/superset, no reframing of same phenomenon)
  * Semantically separable (no ambiguity in coding; no "could go either way")
- Be non-redundant (adds unique conceptual value; no duplicate concepts)
- Be grounded in the data (supported by multiple attributes or repeated patterns)
</facet_definition_guidance>
</taxonomy_context>

Here are the observations you need to analyze:

<observations>
{observations_block}
</observations>

# Instructions

Before writing your final output, think through your analysis in the scratchpad field:

**Step 1: Cluster observations**
Group similar observations together based on shared descriptive meaning. Identify recurring patterns in what is being said about {partition_name}.

Focus on the type of quality, characteristic, principle, or practice being described.

**Step 2: Identify candidate facets**
Based on these clusters, identify candidate facets.

For each candidate facet, assess:
- the facet name
- the underlying type of quality or attribute it captures
- which observations support it
- whether it is internally coherent
- whether it is ontologically distinct from other candidate facets

Remember: a facet identifies the analytical lens through which descriptive qualities are grouped. A facet captures a type of meaning, not a single concrete observation.

**Step 3: Verify internal coherence**
Check whether each candidate facet captures one clear underlying concept.

Reject or split candidate facets that:
- combine multiple different kinds of phenomena
- mix descriptive content with evaluation
- are too broad to support clear coding

**Step 4: Verify distinctness**
Check each pair of candidate facets to ensure they are:
- ontologically distinct (not overlapping in conceptual space; one is not a subset of another)
- semantically separable (someone coding a response would clearly know which facet applies, with no "could go either way" situations)
- not two different lenses on the same phenomenon

If two facets fail this test, consolidate them into one broader facet or redefine the boundaries more clearly.

**Step 5: Verify domain boundaries**
Check that each retained facet falls strictly within the included domain of {partition_name}.

Exclude facets that belong more naturally to other domains, including:
{excluded_block_light}

**Step 6: Prepare final output**
Return only the dominant facets that pass all checks above.

For each facet, provide:
- a short descriptive name in {language} (2-5 words)
- a description in {language} of what the facet captures (1-2 sentences)
- 3-5 representative observations from the input, using the exact observation text

# Output Requirements

Provide output as valid JSON following the response schema provided.

# Language Requirement

All output (facet names, descriptions, and example observations) must be written in {language}.

# Final Notes

- Facets must be descriptive, not evaluative
- Facets must be grounded in repeated patterns across observations
- Facets must be internally coherent
- Facets must be externally distinctive
- Facets must remain strictly within the included domain
- Each facet must capture one type of quality, not multiple
- All output must be in {language}
- Use exact observation text in the examples, not observation numbers

Use your scratchpad field for Steps 1-5 to show your analytical thinking. Then provide your final output as valid JSON."""

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
        facet_key_idea = _extract_key_idea(rules.facet_instruction)
        attribute_key_idea = _extract_key_idea(rules.attribute_instruction)
        noun_phrase = dimension_def.noun_phrase_descriptor
        domain_key_idea = _extract_key_idea(rules.domain_instruction)
    else:
        facet_guidance = "Identify the specific viewpoint or characteristic within the domain."
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

Here is the taxonomy context you are working within:

<taxonomy_context>
This is the structure:
<taxonomy_structure>
- Dimension (L1): {dimension_name}: {noun_phrase}
- Domain (L2): {domain_key_idea}
- Facet (L3): {facet_key_idea}
- Attribute (L4): {attribute_key_idea}
</taxonomy_structure>

You are working within this dimension:
<taxonomy_dimension>
{dimension_name} — {dimension_description}
</taxonomy_dimension>

And you are working within this domain:
<taxonomy_domain>
{domain_name} — {domain_definition}
</taxonomy_domain>
{excluded_block}
</taxonomy_context>

Here are the facets you need to consolidate:
<chunk_level_analyses>
{chunk_results}
</chunk_level_analyses>

# Understanding Facets

Conceptualization:
{facet_guidance}

# Facet Consolidation Rules

<strict_consolidation_rule>
1. MERGE OVERLAP (MANDATORY)
All facets that conceptually overlap or are variants of the same idea must be merged.

2. ORTHOGONALITY (MAIN RULE)
For each pair of facets:
"Can a single observation plausibly fall under both?"

- Yes → merge
- Doubt → merge
- Only if clearly no → keep separate

3. NO HIERARCHY
Facets must not be:
- general vs. specific
- principle vs. application
If this occurs → merge

4. NO OBJECT SPLITTING
Do not split based on object (e.g., humans vs. animals)
If the same underlying principle applies → merge

5. MINIMALITY (MANDATORY)
Use the smallest number of facets that provides full coverage.
If a facet is not strictly necessary → remove it
</strict_consolidation_rule>

<disambiguation_test>
For any pair of facets:
"Can a clear rule assign every observation to exactly one facet?"
- No → merge
</disambiguation_test>

<precedence_rule>
When rules conflict, prioritize:
1. Non-overlap (orthogonality)
2. Minimality (merge unless clearly distinct)
3. Clarity for annotation

When in doubt → merge facets
</precedence_rule>

# Required Process

Before writing your final output, think through your analysis in the scratchpad field:

**Step 1 — Scan chunk-level facets**
Review all facets from all chunks. Note recurring themes and obvious duplicates.

**Step 2 — Group overlapping facets**
Group facets that describe the same or overlapping concepts across chunks.

**Step 3 — Apply orthogonality test**
For each pair of candidate consolidated facets, ask: "Can a single observation plausibly fall under both?" If yes or doubtful → merge.

**Step 4 — Apply disambiguation test**
For each pair: "Can a clear rule assign every observation to exactly one facet?" If no → merge.

**Step 5 — Verify domain boundaries**
Ensure each retained facet belongs to the included domain and not to any excluded domain:
{excluded_block_light}

**Step 6 — Prepare final output**
Return only the minimal set of consolidated facets that pass all checks.

For each consolidated facet, provide:
- A short descriptive name (2-5 words) 
- A description of what the facet captures (1-2 sentences)
- 3-5 representative observations selected from across the merged chunks (exact text)

# Output Requirements

Provide output as valid JSON following the response schema provided.

# Language Requirement

All facet names and descriptions must be in {language}.

# Final Notes

- Facets must be descriptive, not evaluative
- Facets must be grounded in repeated patterns across observations
- Facets must be internally coherent (one clear concept each)
- Facets must be externally distinctive (no overlap, no subset/superset)
- Facets must remain strictly within the included domain
- All output must be in {language}

Use your scratchpad field for Steps 1-5 to show your analytical thinking. Then provide your final output as valid JSON."""


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


def _valence_display(idea) -> str:
    """Map idea.valence to a readable tag for prompts."""
    val = str(getattr(idea, 'valence', '') or '0').strip()
    if val in ('+', '1', '+1', 'positive'):
        return '[+]'
    if val in ('-', '-1', 'negative'):
        return '[-]'
    return '[0]'


def _build_ideas_block_for_facet_assignment(ideas: List) -> str:
    """Format ideas for assignment prompts — idea text only, no ladder."""
    lines = []
    for idea in ideas:
        idea_text = getattr(idea, 'idea', '') or getattr(idea, 'instance', '') or ''
        valence = _valence_display(idea)
        lines.append(
            f"- idea_id: {idea.idea_id}\n"
            f"  idea: {idea_text}\n"
            f"  valence: {valence}"
        )
    return "\n".join(lines)


def build_facet_assignment_prompt(
    *,
    survey_question: str,
    language: str,
    dataset_context_section: str,
    domain_name: str,
    domain_definition: str,
    facets: List[DiscoveredFacet],
    other_label: Optional[str],
    ideas: List,
) -> str:
    """Build prompt for assigning ideas to discovered facets (L3)."""
    facet_codebook = _build_facet_codebook_block(facets, other_label)
    ideas_block = _build_ideas_block_for_facet_assignment(ideas)
    other_label_display = other_label or "Other"

    return f"""You are a qualitative coding assistant. Your task is to assign survey response ideas to specific facets within a domain. Each idea represents a distinct concept extracted from a survey response, and you must determine which facet best captures the type of quality being described.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

<domain_context>
Domain: {domain_name} -- {domain_definition}
</domain_context>

Here are the facets available for assignment. Each idea must be assigned to exactly ONE of these facets:

<facets>
{facet_codebook}
</facets>

Here are the ideas you need to assign to facets:

<ideas_to_assign>
{ideas_block}
</ideas_to_assign>

For each idea in the list, follow these steps:

1. Read the idea text carefully, noting the valence tag ([+] positive, [-] negative, [0] neutral) and what type of quality is being expressed.

2. Compare the idea against each available facet. Ask yourself: "Which facet best captures the type of quality being described in this idea?" Consider:
   - The core meaning of the idea text
   - The descriptions provided for each facet
   - The examples given for each facet
   - Semantic similarity between the idea and facet descriptions

3. Assign the idea to exactly ONE facet. You must return only the facet ID (the code in [F#] brackets, such as "F1" or "F2"). Do NOT return the facet name or description. Assign "{other_label_display}" ONLY if no facet fits at all.

4. Rate your confidence in this assignment on a scale from 0.0 to 1.0, where:
   - 1.0 = completely certain this is the correct facet
   - 0.7-0.9 = confident but some ambiguity exists
   - 0.5-0.6 = moderate confidence, could reasonably fit multiple facets
   - Below 0.5 = low confidence, significant ambiguity

Important requirements:
- Assign each idea to exactly ONE facet
- Return only the facet ID (e.g., "F1"), not the facet name
- Echo back the exact idea_id and idea text from the input without modification
- All output must be in {language}

Provide your response as valid JSON matching the schema provided."""

class FacetAssignment(BaseModel):
    """Single idea-to-facet assignment."""
    idea_id: str = Field(
        ..., description="The EXACT idea_id from the input. Do not modify."
    )
    idea: str = Field(
        ..., description="Echo back the EXACT idea text from the input for this idea_id."
    )
    assigned_facet_id: str = Field(
        ..., description=(
            "The facet ID from the [F#] prefix (e.g. 'F1', 'F3'). "
            "Return ONLY the ID, not the facet name."
        )
    )
    confidence: float = Field(
        ..., description="Confidence in the assignment (0.0 to 1.0)"
    )


class FacetAssignmentBatch(BaseModel):
    """Batch of facet assignments for multiple ideas."""
    assignments: List[FacetAssignment] = Field(
        ..., description="One assignment per idea in the input batch"
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
        attribute_guidance = rules.attribute_instruction
        attribute_key_idea = _extract_key_idea(rules.attribute_instruction)
        facet_key_idea = _extract_key_idea(rules.facet_instruction)
        noun_phrase = dimension_def.noun_phrase_descriptor
        domain_key_idea = _extract_key_idea(rules.domain_instruction)
    else:
        attribute_guidance = (
            "An attribute identifies the specific observable property or feature being described. "
            "It is a named property — not a verbatim span from the response."
        )
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

    return f"""You are a qualitative research analyst specializing in survey response analysis. Your task is to identify the fewest recurring attributes that provide full coverage of a set of observations within a specific facet.

Here is the survey context:

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

Here is the taxonomy context you are working within:

<taxonomy_context>
This is the structure:
<taxonomy_structure>
- Dimension (L1): {noun_phrase}
- Domain (L2): {domain_key_idea}
- Facet (L3): {facet_key_idea}
- Attribute (L4): {attribute_key_idea}
</taxonomy_structure>

You are working within this dimension:
<taxonomy_dimension>
{dimension_name} — {dimension_description}
</taxonomy_dimension>

And you are working within this domain and facet:
<taxonomy_domain>
{domain_name} — {domain_definition}
</taxonomy_domain>
<taxonomy_facet>
{facet_name} — {facet_description}
</taxonomy_facet>
{excluded_block}

Here is guidance on what attributes are and how they should be defined:

<attribute_definition_guidance>
Target abstraction level: ATTRIBUTE (L4)
{attribute_guidance}

Each attribute must:
- Be a descriptive, data-grounded category based on shared meaning across multiple observations
- Be non-evaluative (no judgment, sentiment, or valence)
- Stay strictly within the facet boundaries
- Be internally coherent (one clear underlying concept)
- Be externally distinctive:
  * Ontologically distinct (no overlap, no subset/superset, no reframing of same phenomenon)
  * Semantically separable (no ambiguity in coding; no "could go either way")
- Be non-redundant (adds unique conceptual value; no duplicate concepts)
- Be grounded in the data (supported by multiple observations or repeated patterns)
</attribute_definition_guidance>
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

Use your scratchpad field for Steps 1-5 to show your analytical thinking. Then provide your final output as valid JSON."""


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
        attribute_key_idea = _extract_key_idea(rules.attribute_instruction)
        facet_key_idea = _extract_key_idea(rules.facet_instruction)
        noun_phrase = dimension_def.noun_phrase_descriptor
        domain_key_idea = _extract_key_idea(rules.domain_instruction)
    else:
        attribute_guidance = (
            "An attribute identifies the specific observable property or feature being described. "
            "It is a named property -- not a verbatim span from the response."
        )
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
- Dimension (L1): {dimension_name}: {noun_phrase}
- Domain (L2): {domain_key_idea}
- Facet (L3): {facet_key_idea}
- Attribute (L4): {attribute_key_idea}
</taxonomy_structure>

You are working within this dimension:
<taxonomy_dimension>
{dimension_name} -- {dimension_description}
</taxonomy_dimension>

And you are working within this domain and facet:
<taxonomy_domain>
{domain_name}
</taxonomy_domain>
<taxonomy_facet>
{facet_name} -- {facet_description}
</taxonomy_facet>
{excluded_block}
</taxonomy_context>

Here are the attributes you need to consolidate:
<chunk_level_analyses>
{chunk_results}
</chunk_level_analyses>

# Understanding Attributes

Conceptualization:
{attribute_guidance}

# Attribute Consolidation Rules

<strict_consolidation_rule>
1. MERGE OVERLAP (MANDATORY)
All attributes that conceptually overlap or are variants of the same idea must be merged.

2. ORTHOGONALITY (MAIN RULE)
For each pair of attributes:
"Can a single observation plausibly fall under both?"

- Yes -> merge
- Doubt -> merge
- Only if clearly no -> keep separate

3. NO HIERARCHY
Attributes must not be:
- general vs. specific
- principle vs. application
If this occurs -> merge

4. NO OBJECT SPLITTING
Do not split based on object (e.g., humans vs. animals)
If the same underlying principle applies -> merge

5. MINIMALITY (MANDATORY)
Use the smallest number of attributes that provides full coverage.
If an attribute is not strictly necessary -> remove it
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

**Step 1 -- Scan chunk-level attributes**
Review all attributes from all chunks. Note recurring themes and obvious duplicates.

**Step 2 -- Group overlapping attributes**
Group attributes that describe the same or overlapping concepts across chunks.

**Step 3 -- Apply orthogonality test**
For each pair of candidate consolidated attributes, ask: "Can a single observation plausibly fall under both?" If yes or doubtful -> merge.

**Step 4 -- Apply disambiguation test**
For each pair: "Can a clear rule assign every observation to exactly one attribute?" If no -> merge.

**Step 5 -- Verify facet boundaries**
Ensure each retained attribute belongs to the included facet and not to any excluded facet:
{excluded_block_light}

**Step 6 -- Prepare final output**
Return only the minimal set of consolidated attributes that pass all checks.

For each consolidated attribute, provide:
- A short descriptive name (2-5 words)
- A description of what the attribute captures -- a concrete, observable property (1-2 sentences)
- The parent facet name: {facet_name}
- 2-3 representative observations selected from across the merged chunks (exact text)

# Output Requirements

Provide output as valid JSON following the response schema provided.

# Language Requirement

All attribute names and descriptions must be in {language}.

# Final Notes

- Attributes must be descriptive, not evaluative
- Attributes must be grounded in repeated patterns across observations
- Attributes must be internally coherent (one clear concept each)
- Attributes must be externally distinctive (no overlap, no subset/superset)
- Attributes must remain strictly within the included facet
- All output must be in {language}

Use your scratchpad field for Steps 1-5 to show your analytical thinking. Then provide your final output as valid JSON."""


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


def build_attribute_assignment_prompt(
    *,
    survey_question: str,
    language: str,
    dataset_context_section: str,
    facet_name: str,
    facet_description: str,
    attributes: List['DiscoveredAttribute'],
    ideas: List,
) -> str:
    """Build prompt for assigning ideas to discovered attributes (L4) within a facet."""
    attribute_codebook = _build_attribute_codebook_block(attributes)
    ideas_block = _build_ideas_block_for_facet_assignment(ideas)

    return f"""You are a qualitative coding assistant. Your task is to assign survey response ideas to specific attributes within a facet. Each idea represents a distinct concept extracted from a survey response, and you must determine which attribute best captures the specific quality being described.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

<facet_context>
Facet: {facet_name} -- {facet_description}
</facet_context>

Here are the attributes available for assignment. Each idea must be assigned to exactly ONE of these attributes:

<attributes>
{attribute_codebook}
</attributes>

Here are the ideas you need to assign to attributes:

<ideas_to_assign>
{ideas_block}
</ideas_to_assign>

For each idea in the list, follow these steps:

1. Read the idea text carefully, noting what specific quality is being expressed.

2. Compare the idea against each available attribute. Ask yourself: "Which attribute best captures the specific quality being described in this idea?" Consider:
   - The core meaning of the idea text
   - The descriptions provided for each attribute
   - The examples given for each attribute
   - Semantic similarity between the idea and attribute descriptions

3. Assign the idea to exactly ONE attribute. You must return only the attribute ID (the code in [A#] brackets, such as "A1" or "A2"). Do NOT return the attribute name or description.

4. Rate your confidence in this assignment on a scale from 0.0 to 1.0, where:
   - 1.0 = completely certain this is the correct attribute
   - 0.7-0.9 = confident but some ambiguity exists
   - 0.5-0.6 = moderate confidence, could reasonably fit multiple attributes
   - Below 0.5 = low confidence, significant ambiguity

Important requirements:
- Assign each idea to exactly ONE attribute
- Return only the attribute ID (e.g., "A1"), not the attribute name
- Echo back the exact idea_id and idea text from the input without modification
- All output must be in {language}

Provide your response as valid JSON matching the schema provided."""

class AttributeAssignment(BaseModel):
    """Single idea-to-attribute assignment."""
    idea_id: str = Field(
        ..., description="The EXACT idea_id from the input. Do not modify."
    )
    idea: str = Field(
        ..., description="Echo back the EXACT idea text from the input for this idea_id."
    )
    assigned_attribute_id: str = Field(
        ..., description=(
            "The attribute ID from the [A#] prefix (e.g. 'A1', 'A3'). "
            "Return ONLY the ID, not the attribute name."
        )
    )
    confidence: float = Field(
        ..., description="Confidence in the assignment (0.0 to 1.0)"
    )


class AttributeAssignmentBatch(BaseModel):
    """Batch of attribute assignments for multiple ideas."""
    assignments: List[AttributeAssignment] = Field(
        ..., description="One assignment per idea in the input batch"
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
        noun_phrase = dimension_def.noun_phrase_descriptor
        domain_key_idea = _extract_key_idea(rules.domain_instruction)
    else:
        attribute_guidance = (
            "An attribute identifies the specific observable property or feature being described. "
            "It is a named property -- not a verbatim span from the response."
        )
        attribute_key_idea = "the specific observable property being described"
        facet_key_idea = "the analytical lens applied to the subject"
        noun_phrase = dimension_name
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
- Dimension (L1): {dimension_name}: {noun_phrase}
- Domain (L2): {domain_key_idea}
- Facet (L3): {facet_key_idea}
- Attribute (L4): {attribute_key_idea}
</taxonomy_structure>

You are working within this dimension:
<taxonomy_dimension>
{dimension_name} -- {dimension_description}
</taxonomy_dimension>

And you are working within this domain:
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
Codes MUST be primarily driven by the **number of ideas linked to attributes**.

- Attributes with HIGH idea counts MUST form the **core structure of the codebook**.
- Attributes with LOW idea counts MUST NOT become standalone codes unless absolutely necessary.
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

Attributes with low prevalence (e.g., <10-15 ideas) should almost never result in standalone codes.

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

**Step 5 -- Justify Low-Prevalence Codes (MANDATORY)**
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

Use your scratchpad field for Steps 1-5 to show your analytical thinking. Then provide your final output as valid JSON."""


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

# =============================================================================
# §8 CODE GENERATION FROM ATTRIBUTES (P8)
# =============================================================================

def build_code_from_attributes_prompt(
    *,
    survey_question: str,
    language: str,
    dataset_context_section: str,
    dimension_def: Optional['DimensionDefinition'],
    dimension_name: str,
    dimension_description: str,
    domain_name: str,
    domain_definition: str,
    domain_attributes: Dict[str, Dict[str, List[DiscoveredAttribute]]],
    attribute_assignments: Optional[Dict[str, str]] = None,
    excluded_domains: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """Generate codebook codes from a structured attribute inventory.

    Args:
        dimension_def: DimensionDefinition for taxonomy structure lines (or None for fallback)
        domain_name: Name of the domain being processed
        domain_definition: Inclusion definition of the domain
        domain_attributes: {domain_name: {facet_name: [DiscoveredAttribute, ...]}}
        attribute_assignments: idea_id -> attribute_name, for frequency display
        excluded_domains: list of (name, definition) for other domains
    """
    # Dimension-specific taxonomy structure
    if dimension_def:
        noun_phrase = dimension_def.noun_phrase_descriptor
        domain_key_idea = _extract_key_idea(dimension_def.prompt_rules.domain_instruction)
        facet_key_idea = _extract_key_idea(dimension_def.prompt_rules.facet_instruction)
        attribute_key_idea = _extract_key_idea(dimension_def.prompt_rules.attribute_instruction)
    else:
        noun_phrase = dimension_name
        domain_key_idea = "the subject the statement refers to"
        facet_key_idea = "the analytical lens applied to the subject"
        attribute_key_idea = "the specific observable property being described"

    # Excluded domains block
    excluded_block = ""
    if excluded_domains:
        excl_lines = [
            f"- {excl_name} — {excl_def}"
            for excl_name, excl_def in excluded_domains
        ]
        excluded_block = (
            "\nYou must NOT include codes that belong to these excluded domains:\n"
            "<excluded_domains>\n"
            + "\n".join(excl_lines)
            + "\n</excluded_domains>"
        )
    
    # Excluded domains block - "light" (names only, no definitions)
    excluded_block_light = ""
    if excluded_domains:
        excl_names = [excl_name for excl_name, _ in excluded_domains]
        excluded_block_light = "\n".join(f"- {name}" for name in excl_names)

    # Compute attribute frequencies
    attr_counts: Dict[str, int] = {}
    if attribute_assignments:
        for attr_name in attribute_assignments.values():
            attr_counts[attr_name] = attr_counts.get(attr_name, 0) + 1

    # Build inventory: Facet > Attribute (single domain)
    facet_attrs = next(iter(domain_attributes.values()), {})
    inventory_lines = []
    for facet_name, attributes in sorted(facet_attrs.items()):
        for attr in attributes:
            examples = "; ".join(attr.example_observations[:2])
            count = attr_counts.get(attr.attribute_name, 0)
            freq_tag = f" [{count} ideas]" if attr_counts else ""
            line = f"- {attr.attribute_name}{freq_tag}: {attr.attribute_description}"
            if examples:
                line += f" (e.g., {examples})"
            inventory_lines.append(line)
    inventory_block = "\n".join(inventory_lines)

    return f"""You are tasked with deriving a PARSIMONIOUS codebook with MUTUALLY EXCLUSIVE and COLLECTIVELY EXHAUSTIVE codes that represent conceptually and semantically distinct PHENOMENA from a taxonomy inventory of attributes. These attributes were derived from written responses to a survey question.

Here is the survey context:

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

Here is the taxonomy context you are working within:

<taxonomy_context>
This is the structure:
<taxonomy_structure>
- Dimension (L1): {dimension_name}: {noun_phrase}
- Domain (L2): {domain_key_idea}
- Attribute (L3): {attribute_key_idea}
</taxonomy_structure>

You are working within this dimension:
<taxonomy_dimension>
{dimension_name} — {dimension_description}
</taxonomy_dimension>

And you are working within this domain:
<taxonomy_domain>
{domain_name} — {domain_definition}
</taxonomy_domain>
{excluded_block}
</taxonomy_context>

Here is the inventory of attributes for you to analyze:
<attribute_inventory>
{inventory_block}
</attribute_inventory>

# Understanding Phenomena vs Attributes

**Attributes** are specific observations or qualities mentioned in responses. They represent individual data points.

**Phenomena** are underlying conceptual patterns that multiple attributes may indicate. A phenomenon is an underlying pattern that can manifest through multiple specific attributes.

Your task is to identify phenomena, NOT to create one code per attribute.

# Code Derivation Rules

## 1. Phenomenon Rule
Codes must represent underlying PHENOMENA rather than individual attributes. Multiple attributes describing different manifestations of the same underlying phenomenon MUST be merged into a single code.

## 2. Dimension Rule
Only include codes that belong to this domain:
{domain_name} — {domain_definition}

Do not include codes that belong to these excluded domains:
{excluded_block_light}

## 3. Specificity Rule
Do NOT create separate codes simply because attributes differ in specificity. General statements and specific examples should be treated as indicators of the same phenomenon.

Example: "The train was delayed by 20 minutes" and "public transport is often late" both indicate unreliable punctuality and should be coded under the same broader phenomenon.

## 4. Prevalence Weighting Rule (CRITICAL)
Codes MUST be primarily driven by the **number of ideas linked to attributes**.

- Attributes with HIGH idea counts MUST form the **core structure of the codebook**.
- Attributes with LOW idea counts MUST NOT become standalone codes unless absolutely necessary.
- LOW-prevalence attributes SHOULD be:
  - merged into the closest HIGH-prevalence phenomenon, OR
  - grouped into a broader combined phenomenon.

If forced to choose between:
- conceptual nuance  
- prevalence dominance  

➡️ ALWAYS prioritize prevalence dominance.

## 5. Merge Bias Rule
When in doubt:
- MERGE rather than split
- Especially when an attribute has relatively few ideas

Attributes with low prevalence (e.g., <10–15 ideas) should almost never result in standalone codes.

## 6. Parsimony Rule
Use the smallest number of codes that still capture all distinct phenomena present in the inventory.

## 7. Mutual Exclusivity Rule
Codes must represent clearly different phenomena so that responses can be coded consistently without ambiguity.

## 8. Valence Sensitivity Rule
Generate separate codes for positive and negative phenomena. Do NOT combine praise and criticism into a single code. If the attributes contain both positive and negative aspects of similar phenomena, create distinct codes for each valence direction.

## 9. Hierarchy Rule
Derive codes from attribute content rather than copying domain or facet labels directly. Domain context should be used only to determine relevance and scope.

# Required Process

Before writing your final output, think through your analysis in the scratchpad field:

**Step 1 — Identify High-Prevalence Anchors**
- Identify attributes with the highest number of ideas.
- Treat these as the PRIMARY building blocks of the codebook.

**Step 2 — Map Lower-Prevalence Attributes**
- Map lower-prevalence attributes onto these high-prevalence anchors wherever possible.
- Only create a new code if the attribute:
  - is conceptually distinct AND
  - cannot reasonably be merged.

**Step 3 — Ensure Domain Relevance**  
Ensure that each phenomenon group belongs to the included domain and not to any excluded domain.

**Step 4 — Check for Valence Distinctions**
Split positive and negative variants into separate codes where relevant.

**Step 5 — Name Each Phenomenon**
Assign a descriptive name (3-5 word noun phrase in {language}) to each distinct phenomenon.

**Step 6 — Validate Parsimony and Dominance**
- Ensure the codebook is dominated by high-prevalence phenomena
- Ensure low-prevalence attributes are absorbed rather than overrepresented
- Keep total number of codes minimal 

**Step 7 — Justify Low-Prevalence Codes (MANDATORY)**
If any code is primarily based on attributes with low idea counts:
- Explicitly justify why it was NOT merged into a higher-prevalence phenomenon

# Output Requirements

Provide output as valid JSON following the response schema provided.

# Language Requirement

All output (code names, definitions, typical indicators, and evaluation) must be written in {language}.

# Final Notes

Remember:
- This is a **frequency-weighted abstraction task**, not a conceptual listing task.
- Dominant patterns should shape the codebook.
- Rare attributes should be absorbed unless absolutely necessary.

Your goal is a **lean, high-signal codebook** that reflects the strongest patterns in the data.

Begin now by applying the required process and then return only valid JSON."""


class CodeFromAttributes(BaseModel):
    """A formal qualitative code derived from attributes."""
    code_name: str = Field(
        ..., description="Short code name (3-5 word noun phrase)"
    )
    definition: str = Field(
        ..., description="Clear definition of what this code covers (1-2 sentences)"
    )
    typical_indicators: List[str] = Field(
        ..., description="Words or phrases that signal this code"
    )
    source_attributes: List[str] = Field(
        default_factory=list,
        description="Attribute names this code is derived from"
    )


class CodeGenerationFromAttributesResult(BaseModel):
    """P8 output: codes derived from attributes."""
    scratchpad: str = Field(
        ..., description=(
            "Step-by-step reasoning before deriving codes: "
            "(1) identify high-prevalence attributes (largest idea counts) and treat them as anchors, "
            "(2) group attributes into underlying phenomena with priority given to high-prevalence clusters, "
            "    - map low-prevalence attributes onto these dominant phenomena wherever possible, "
            "    - only create a separate code for low-prevalence attributes if they are conceptually distinct and cannot be merged, "
            "(3) check for domain relevance - exclude any phenomena outside the allowed domain, "
            "(4) check for valence distinctions and split positive vs negative where needed, "
            "(5) name each phenomenon (3–5 word noun phrase), "
            "(5) verify parsimony - ensure the codebook is dominated by high-prevalence phenomena and contains a minimal number of codes (typically 5–8), "
            "(7) explicitly justify any code that is primarily based on low-prevalence attributes instead of merging it"
        )
    )
    codes: List[CodeFromAttributes] = Field(
        ..., description=(
            "Formal codes derived from the attribute inventory. "
            "Codes should reflect dominant, high-prevalence phenomena, with low-prevalence attributes absorbed into broader codes where possible.")
    )


# =============================================================================
# §9 CODEBOOK CONSOLIDATION (P9) — cross-domain review & merge
# =============================================================================

def build_codebook_consolidation_prompt(
    *,
    survey_question: str,
    language: str,
    dataset_context_section: str,
    dimension_name: str,
    dimension_description: str,
    raw_codes: List[CodeFromAttributes],
    code_provenance: Dict[int, str],
    code_frequencies: Optional[Dict[int, int]] = None,
) -> str:
    """Consolidate per-domain codes into a final parsimonious, MECE codebook.

    Args:
        raw_codes: All codes from P8 (per-domain)
        code_provenance: Maps code index to domain_name
        code_frequencies: Maps code index to approximate idea count
    """
    # Format raw codes with domain provenance tags and frequency
    code_lines = []
    for i, code in enumerate(raw_codes):
        provenance = code_provenance.get(i, "")
        domain_tag = f"({provenance}) " if provenance else ""
        freq = code_frequencies.get(i, 0) if code_frequencies else 0
        freq_tag = f" (~{freq} ideas)" if freq > 0 else ""

        attrs = ", ".join(code.source_attributes[:5]) if code.source_attributes else "—"
        indicators = "; ".join(code.typical_indicators[:3]) if code.typical_indicators else "—"
        code_lines.append(
            f"[C{i+1}] {code.code_name}{freq_tag}\n" #{domain_tag}
            f"      Definition: {code.definition}\n"
            f"      Indicators: {indicators}\n"
            f"      Source attributes: {attrs}"
        )
    codes_block = "\n\n".join(code_lines)

    return f"""You are an expert in qualitative research.

Your task is to generate a parsimonious and unambiguous codebook from {len(raw_codes)} candidate codes. The codebook must contain codes that are mutually exclusive and collectively exhaustive. A critical aspect is that there is no conceptual overlap between codes, and codes should be semantically unambiguous through the lens of the coding dimension.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

<dimension_context>
Dimension: {dimension_name} — {dimension_description}
</dimension_context>

<candidate_codes>
{codes_block}
</candidate_codes>

## CRITICAL OBJECTIVE
Create a parsimonious codebook that preserves all distinct phenomena, without conceptual overlap or semantic ambiguity.
The result must be conceptually clean, mutually exclusive, and easy for human coders to apply consistently.

<core_principles>

### 1. BALANCED PARSIMONY
- Merge codes that describe the SAME underlying phenomenon — not merely related phenomena
- Two codes are duplicates only if a human coder could not reliably distinguish them
- Stop merging when each surviving code represents a clearly distinct aspect of the dimension
- Preserve codes that cover distinct topics, even if they seem thematically related
- IMPORTANT: only merge codes that share the same valence — see Principle 2

### 2. VALENCE STRUCTURE (HARD CONSTRAINT)
- Each code must have exactly ONE valence: positive, negative, or neutral
- If a dimension has both positive (+) and negative (-) candidate codes, produce TWO separate codes — one positive, one negative
- Do NOT merge positive and negative codes into a single valence-neutral code
- Example: "Duurzaamheid (+)" and "Twijfel aan duurzaamheid (-)" must remain separate codes, NOT merged into "Duurzaamheid en ethiek"
- Neutral codes are for observations without evaluative direction

### 3. LATENT DIMENSION FOCUS
- Each code must represent **ONE distinct question about {dimension_name}**
- Test: each code must complete
  **"This is about whether {dimension_name} is …"**

### 4. STRICT MECE RULE (HARD CONSTRAINT)
- Codes must be:
  - **Mutually Exclusive** → no conceptual overlap
  - **Collectively Exhaustive** → cover all meaningful variation
- If two codes of the same valence could co-occur in the same sentence → **merge them**
- If they answer different questions → **keep them separate**

### 5. NEIGHBOURS CHECK
- If a human coder would hesitate between two codes, first try to sharpen the definitions to make them distinguishable
- Only merge if the distinction cannot be made conceptually — not just because the codes seem related
- Prefer refining boundaries over merging

### 6. APPROPRIATE ABSTRACTION LEVEL
- Codes must be at the right level of abstraction for the dimension: {dimension_description}
- Merge codes that differ only in specific examples but describe the same general phenomenon
- Do not preserve detail that would make codes too narrow to apply consistently across responses

### 7. NON-REDUNDANCY RULE
- If removing a code does not reduce explanatory power → delete it
- Avoid near-synonyms or adjacent constructs

### 8. ACTIONABILITY
- Each code must represent something meaningful and actionable given the survey question
- Remove or merge codes that are too abstract or too narrow to be useful through the lens of the survey question

### 9. DOMAIN AWARENESS
- Each candidate code is tagged with its source domain in parentheses
- Codes from DIFFERENT domains that share similar names represent DIFFERENT phenomena
- Do NOT merge codes across domains unless they are truly identical in meaning
- Example: "Reliability" from a customer service domain ≠ "Reliability" from a brand identity domain

### 10. ATTRIBUTE TYPE SEPARATION
- Codes that differ in underlying mechanism must remain separate, even if they co-occur in responses
- If two codes describe different types of phenomena (e.g., values vs functional properties vs perceptions), they represent different mechanisms and must not be merged
- Do NOT merge across attribute types into a single code

### 11. COVERAGE GUARD
- Each code must be specific enough that a human coder can confidently apply it
- If a code's definition requires listing many unrelated phenomena, it is too broad — split it
- Test: can you describe what this code covers in ONE sentence without using "and/or" more than once? If not, split it
- Codes backed by many ideas carry more analytical value — do not merge them away lightly

### 12. PRESERVE DISTINCT MECHANISMS
- Codes that differ in underlying mechanism must remain separate, even if they seem thematically related
- Examples of distinctions that must be preserved:
  - Evaluative judgments vs factual descriptions
  - Causes vs consequences
  - General impressions vs specific experiences
- When in doubt whether two codes differ: they probably do — keep them separate

</core_principles>

<code_definition_requirements>

### DUAL-LAYER CODE DEFINITION (MANDATORY)
Each code MUST include:

**code_name**
- 3–5 word noun phrase
- Short, scannable, used for coding

**definition**
- A short interpretive claim
- Must read like an analyst conclusion
- Avoid vague abstract phrasing — be concrete and specific

### CLARITY TEST (MANDATORY)
Each code must include a diagnostic_test:
"This is about whether {dimension_name} is …"
- Must be unique per code
- Must not overlap with other codes

### ADDITIONAL REQUIRED FIELDS
Each code must also include:
- **valence**: one of "positive", "negative", or "neutral"
- **typical_indicators**: words or phrases that signal this code
- **source_attributes**: list of attribute names this code is derived from (from all merged codes)

</code_definition_requirements>

<workflow>
Follow these steps (DO NOT SKIP):
1. Cluster similar codes by topic AND valence — keep positive (+) and negative (-) clusters separate
2. Merge aggressively within the same valence — never merge across valence
3. Test for MECE overlap — for each pair of same-valence codes, ask: "would a coder hesitate between these?"
4. Remove redundancy — for each code, ask: "does removing this reduce explanatory power?"
5. Ensure one clear dimension per code
6. Assign valence label (positive, negative, neutral) to each surviving code
7. Verify each surviving code is actionable through the lens of the survey question: "{survey_question}"
</workflow>

All output MUST be in {language}.

Include a brief evaluation of what was merged, removed, or preserved and why.

Provide output as valid JSON following the response schema provided."""

class ConsolidatedCode(BaseModel):
    """A consolidated code with diagnostic test for MECE verification."""
    code_name: str = Field(
        ..., description="Short code name (3-5 word noun phrase)"
    )
    definition: str = Field(
        ..., description=(
            "A short interpretive claim that reads like an analyst conclusion. "
            "Avoid vague abstract phrasing — be concrete and specific."
        )
    )
    diagnostic_test: str = Field(
        ..., description=(
            "Completes the sentence: 'This is about whether ...' — "
            "must be unique per code and must not overlap with other codes."
        )
    )
    valence: str = Field(
        ..., description="One of: 'positive', 'negative', 'neutral'"
    )
    typical_indicators: List[str] = Field(
        ..., description="Words or phrases that signal this code"
    )
    source_attributes: List[str] = Field(
        default_factory=list,
        description="Attribute names this code is derived from (from all merged codes)"
    )


class CodebookConsolidationResult(BaseModel):
    """P9 output: consolidated codebook."""
    evaluation: str = Field(
        ..., description="Brief analysis of what was merged/removed and why"
    )
    codes: List[ConsolidatedCode] = Field(
        ..., description="Final MECE codebook"
    )


# =============================================================================
# §10 CODE ASSIGNMENT (P10) — single idea
# =============================================================================

# Re-export data-flow wrapper models (canonical definition in models_exp.py)
from .models_exp import CodeAssignment, CodeAssignmentBatch


def _build_codes_block(
    codes: List[CodeFromAttributes],
    other_label: Optional[str] = None,
) -> str:
    """Format codes for assignment prompt (code-only, no attributes)."""
    lines = []
    for i, code in enumerate(codes, 1):
        diagnostic = getattr(code, 'diagnostic_test', '') or ''
        indicators = ", ".join(code.typical_indicators[:5]) if code.typical_indicators else "(none)"
        block = (
            f"[C{i}] {code.code_name}\n"
            f"    Definition: {code.definition}\n"
        )
        if diagnostic:
            block += f"    Diagnostic: {diagnostic}\n"
        block += f"    Indicators: {indicators}"
        lines.append(block)

    if other_label:
        n = len(codes) + 1
        lines.append(
            f"[C{n}] {other_label}\n"
            f"    Definition: Ideas that do not clearly fit any of the above codes.\n"
            f"    Indicators: no matching indicators"
        )

    return "\n\n".join(lines)


def build_code_assignment_prompt(
    *,
    survey_question: str,
    language: str,
    dataset_context_section: str,
    codes: List[CodeFromAttributes],
    other_label: Optional[str],
    idea,
    facet_lookup: Optional[Dict[str, str]] = None,
) -> str:
    """Build prompt for assigning a single idea to a code."""
    codes_block = _build_codes_block(codes, other_label)

    # Format single idea
    valence = getattr(idea, 'valence', '') or '0'
    facet = (facet_lookup or {}).get(idea.idea_id, '') or getattr(idea, 'facet', '') or ''
    domain = getattr(idea, 'domain', '') or ''

    idea_block = (
        f"idea: {idea.idea}\n"
        f"domain: {domain}\n"
        f"facet: {facet}\n"
        f"valence: {valence}"
    )

    other_label_display = other_label or "Other"

    return f"""You are a qualitative coding assistant. Assign the idea below to the best-matching code.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

<codebook>
{codes_block}
</codebook>

<idea>
{idea_block}
</idea>

<instructions>
1. Read the idea text, domain, facet, and valence.
2. Find the code whose definition best matches what the respondent is expressing.
3. Return the code ID from [C#] brackets (e.g. "C1"). Do NOT return the code name.
4. Assign "{other_label_display}" only if NO code fits at all.
5. Rate confidence: 0.90+ = clear, 0.70-0.89 = good, 0.50-0.69 = approximate, <0.50 = weak.
6. Provide a brief rationale for your code choice.

All output MUST be in {language}.
Provide output as valid JSON following the response schema provided.
</instructions>
"""


class CodeAssignmentResponse(BaseModel):
    """Single idea → code assignment."""
    assigned_code_id: str = Field(
        ...,
        description="The code ID from the [C#] prefix (e.g. 'C1', 'C7'). Return ONLY the ID."
    )
    confidence: float = Field(
        ...,
        description="Confidence in the assignment (0.0 to 1.0)"
    )
    rationale: str = Field(
        ...,
        description="Brief rationale for the code choice"
    )

