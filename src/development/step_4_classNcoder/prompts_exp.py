"""
Prompts and Pydantic response models for Category Discovery v3.

Organized in pipeline processing order:
  §1  Dimension Context Block (shared helper)
  §2  Facet Discovery (P1: per-domain, chunked)
  §3  Facet Assignment (P2: per-domain, batched)
  §4  Attribute Discovery (P3: per facet within domain)
  §5  Code Generation from Attributes (P4: cross-domain)
  §6  Bridge — codes → MECECode
  §7  Shared Data Models (MECECode, MECEVerification)
  §8  Code Assignment — batch (P5)
  §9  Code Assignment — single idea (P5)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from development.step_3_ideaExtractor.dimension_data import DimensionDefinition


# =============================================================================
# §1 DIMENSION CONTEXT BLOCK — shared helper for all prompts
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
    lines = [f"- {name} — {definition}" for name, definition in items]
    content = "\n".join(lines)
    return (
        f"\nYou must NOT include categories that belong to these excluded areas:\n"
        f"<{tag_name}>\n{content}\n</{tag_name}>\n"
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
# §2 FACET DISCOVERY (P1) — per-domain chunked pattern extraction
# =============================================================================

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
    facets: List[DiscoveredFacet] = Field(
        ..., description="Facets identified in the observations"
    )


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
    observations_block = "\n".join(
        f"{i}. {obs}" for i, obs in enumerate(observations, 1)
    )

    # Dimension-specific guidance
    if dimension_def:
        rules = dimension_def.prompt_rules
        facet_guidance = rules.facet_instruction
        facet_key_idea = _extract_key_idea(rules.facet_instruction)
        facet_question_stem = rules.facet_diagnostic.rstrip("?")
        attribute_key_idea = _extract_key_idea(rules.attribute_instruction)
        noun_phrase = dimension_def.noun_phrase_descriptor
        domain_key_idea = _extract_key_idea(rules.domain_instruction)

        # Build worked example
        example_block = ""
        if dimension_def.examples:
            ex = dimension_def.examples[0]
            example_block = f"""
Example (from a different survey):
  Survey: {ex.survey_context}
  Response: "{ex.response}"
  Domain: {ex.domain}
  Facet: {ex.facet}
  Instance: {ex.instance}
"""
    else:
        facet_guidance = "Identify the specific viewpoint or characteristic within the domain."
        facet_key_idea = "the analytical lens applied to the subject"
        facet_question_stem = "What specific aspect or viewpoint does this represent"
        attribute_key_idea = "the specific observable property being described"
        noun_phrase = dimension_name
        domain_key_idea = "the subject the statement refers to"
        example_block = ""

    excluded_block = _build_exclusion_block(
        excluded_domains or [], "excluded_domains"
    )

    return f"""You are assisting with qualitative analysis of survey responses. Your task is to identify the fewest recurring facets that provide full coverage of a set of observations.

All of your output must be in this language:
<language>
{language}
</language>

You are working within this domain:
<domain>
{partition_name}  
</domain>
{excluded_block}
Here are the observations you need to analyze:
<observations>
{observations_block}
</observations>

<facet_definition_guidance>
Taxonomy levels for this dimension:
- Dimension (L1): {noun_phrase}
- Domain (L2): {domain_key_idea}
- Facet (L3): {facet_key_idea}
- Attribute (L4): {attribute_key_idea}
{example_block}
Target abstraction level: FACET (L3)

{facet_guidance}

Each facet must be:
- **Ontologically distinct** — no two facets may share conceptual space. A facet must not be a subset of another facet, and two facets must not be two different lenses on the same phenomenon.
- **Semantically distant** — someone coding a response should clearly know which facet applies, with no "could go either way" situations.
- Focused on ONE specific aspect (not a compound list of multiple concerns)
- A natural grouping of related phenomena within the domain
- Strictly within the boundaries of the included domain described above
</facet_definition_guidance>

<task_instructions>
Follow these steps to complete your analysis:

**Step 1: Cluster observations**
Mentally group similar observations together. Look for recurring patterns and themes. Note which observations share the same {facet_key_idea}.

**Step 2: Identify candidate facets**
Based on your clustering, identify potential facets. For each candidate facet, write:
- The facet name
- {facet_question_stem} for this facet
- Which observation numbers support it
- Whether it is ontologically distinct from other candidates

**Step 3: Distinguish dominant from minor facets**
- Dominant facets = supported by multiple observations (3+) and represent recurring patterns
- Minor facets = supported by only 1-2 observations

Only dominant facets should be included in your final output.

**Step 4: Verify distinctness**
Check each pair of dominant facets to ensure they are:
- Ontologically distinct (not overlapping in conceptual space)
- Semantically distant (a coder would clearly know which to choose)
- Not two lenses on the same phenomenon

If two facets fail this test, consolidate them into one.

**Step 5: Provide final output**
After your analysis, provide output as valid JSON following the response schema provided.
</task_instructions>

<key_reminders>
- Return ONLY dominant facets (3+ observations)
- Ensure facets are ontologically distinct and semantically distant
- Each facet should capture ONE {facet_key_idea}, not multiple
- All facets must fall within the included domain, not the excluded domains
- All output must be in {language}
</key_reminders>"""


# =============================================================================
# §2.5 FACET CONSOLIDATION — merge chunk-level facets into coherent set
# =============================================================================

class FacetConsolidatedResponse(BaseModel):
    """Consolidated facets after merging chunk-level discoveries."""
    facets: List[DiscoveredFacet] = Field(
        ..., description="Fewest mutually exclusive facets needed for full coverage, consolidated from all chunks"
    )


def build_facet_consolidation_prompt(
    *,
    survey_question: str,
    language: str,
    dataset_context_section: str,
    dimension_def: Optional[DimensionDefinition],
    dimension_name: str,
    dimension_description: str,
    partition_name: str,
    partition_definition: str,
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

        example_block = ""
        if dimension_def.examples:
            ex = dimension_def.examples[0]
            example_block = f"""
Example (from a different survey):
  Survey: {ex.survey_context}
  Response: "{ex.response}"
  Domain: {ex.domain}
  Facet: {ex.facet}
  Instance: {ex.instance}
"""
    else:
        facet_guidance = "Identify the specific viewpoint or characteristic within the domain."
        facet_key_idea = "the analytical lens applied to the subject"
        attribute_key_idea = "the specific observable property being described"
        noun_phrase = dimension_name
        domain_key_idea = "the subject the statement refers to"
        example_block = ""

    excluded_block = _build_exclusion_block(
        excluded_domains or [], "excluded_domains"
    )

    return f"""You are a taxonomy consolidation specialist.
Your task is to merge multiple chunk-level facet analyses into a single, coherent set of facets for the domain "{partition_name}".

All of your output must be in this language:
<language>
{language}
</language>

You are working within this domain:
<domain>
{partition_name} — {partition_definition}
</domain>
{excluded_block}
Here are the facets you need to consolidate:
<chunk_level_analyses>
{chunk_results}
</chunk_level_analyses>

<facet_definition_guidance>
Taxonomy levels for this dimension:
- Dimension (L1): {noun_phrase}
- Domain (L2): {domain_key_idea}
- Facet (L3): {facet_key_idea}
- Attribute (L4): {attribute_key_idea}
{example_block}
Target abstraction level: FACET (L3)

{facet_guidance}

Each facet must be:
- **Ontologically distinct** — no two facets may share conceptual space. A facet must not be a subset of another facet, and two facets must not be two different lenses on the same phenomenon.
- **Semantically distant** — someone coding a response should clearly know which facet applies, with no "could go either way" situations.
- Focused on ONE specific aspect (not a compound list of multiple concerns)
- A natural grouping of related phenomena within the domain
- Strictly within the boundaries of the included domain described above
</facet_definition_guidance>

## YOUR TASK
Consolidate these chunk-level facet lists into the fewest mutually exclusive facets needed for full coverage within the domain "{partition_name}".

Important consolidation principles:
- MERGE facets that have conceptual overlap, near-equivalence, or represent subcategories of a broader facet
- ENSURE mutual exclusivity: no two facets in your final list should overlap in meaning
- MAINTAIN full coverage: the consolidated facets must collectively cover all concepts present in the chunk-level analyses
- MINIMIZE the total number of facets while preserving meaningful distinctions
- When merging facets, pick the most representative example observations from across the merged set (3-5 examples)
- All facet names and descriptions must be in {language}

<scratchpad>
Follow these steps to complete your analysis:
1. List all unique facets that appear across the chunk-level analyses
2. Identify groups of facets that have conceptual overlap or proximity
3. For each group, determine an appropriate consolidated facet name and description
4. Check that your consolidated facets are mutually exclusive — for each pair ask: "Could an observation plausibly belong to both?" If yes, merge them.
5. Verify that your consolidated facets provide complete coverage of the original set
</scratchpad>

All output MUST be in {language}.

Provide output as valid JSON following the response schema provided."""


# =============================================================================
# §3 FACET ASSIGNMENT (P2) — per-domain batched assignment
# =============================================================================

class FacetAssignment(BaseModel):
    """Single idea-to-facet assignment."""
    idea_id: str = Field(
        ..., description="The idea_id from the input"
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
    if val in ('1', '+1', 'positive'):
        return '[+]'
    if val in ('-1', 'negative'):
        return '[-]'
    return '[0]'


def _build_ideas_block_for_facet_assignment(ideas: List) -> str:
    """Format ideas for facet assignment prompt."""
    lines = []
    for idea in ideas:
        interpretation = getattr(idea, 'interpretation', '') or ''
        abstraction = getattr(idea, 'abstraction', '') or ''
        instance = getattr(idea, 'instance', '') or ''
        valence = _valence_display(idea)
        lines.append(
            f"- idea_id: {idea.idea_id}\n"
            f"  valence: {valence}\n"
            f"  instance: {instance}\n"
            f"  interpretation: {interpretation}\n"
            f"  abstraction: {abstraction}"
        )
    return "\n".join(lines)


def build_facet_assignment_prompt(
    *,
    survey_question: str,
    language: str,
    dataset_context_section: str,
    dimension_def: Optional[DimensionDefinition],
    dimension_name: str,
    dimension_description: str,
    domain_name: str,
    domain_definition: str,
    facets: List[DiscoveredFacet],
    other_label: Optional[str],
    ideas: List,
) -> str:
    """Build prompt for assigning ideas to discovered facets (L3)."""
    taxonomy_block = build_dimension_context_block(
        dimension_def=dimension_def,
        dimension_name=dimension_name,
        dimension_description=dimension_description,
        domain_name=domain_name,
        domain_definition=domain_definition,
    )

    facet_codebook = _build_facet_codebook_block(facets, other_label)
    ideas_block = _build_ideas_block_for_facet_assignment(ideas)

    # Dimension-specific facet question
    if dimension_def:
        facet_question = dimension_def.prompt_rules.facet_diagnostic
    else:
        facet_question = "What specific aspect or viewpoint does this represent?"

    other_label_display = other_label or "Other"

    return f"""You are a qualitative coding assistant assigning survey response ideas to discovered facets.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

{taxonomy_block}

<facets>
Assign each idea to exactly ONE of these facets:

{facet_codebook}
</facets>

<ideas_to_assign>
{ideas_block}
</ideas_to_assign>

<instructions>
For each idea:
1. Read the idea's instance, interpretation, and abstraction.
2. Determine which facet best answers the question: {facet_question}
3. Assign exactly ONE facet per idea. Return the facet ID from [F#] brackets (e.g. "F1", "F3"). Do NOT return the facet name.
4. Assign "{other_label_display}" ONLY if no facet fits at all.
5. Rate your confidence (0.0 to 1.0).

All output MUST be in {language}.

Provide output as valid JSON following the response schema provided.
</instructions>
"""


# =============================================================================
# §4 ATTRIBUTE DISCOVERY (P3) — per facet within domain
# =============================================================================

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
    """P3 output: attributes discovered within a facet."""
    attributes: List[DiscoveredAttribute] = Field(
        ..., description="Concrete attributes identified within the facet"
    )


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
        attribute_question_stem = rules.attribute_diagnostic.rstrip("?")
        facet_key_idea = _extract_key_idea(rules.facet_instruction)
        noun_phrase = dimension_def.noun_phrase_descriptor
        domain_key_idea = _extract_key_idea(rules.domain_instruction)

        example_block = ""
        if dimension_def.examples:
            ex = dimension_def.examples[0]
            example_block = f"""
Example (from a different survey):
  Survey: {ex.survey_context}
  Response: "{ex.response}"
  Domain: {ex.domain}
  Facet: {ex.facet}
  Instance: {ex.instance}
"""
    else:
        attribute_guidance = (
            "An attribute identifies the specific observable property or feature being described. "
            "It is a named property — not a verbatim span from the response."
        )
        attribute_key_idea = "the specific observable property being described"
        attribute_question_stem = "What specific feature or property is described"
        facet_key_idea = "the analytical lens applied to the subject"
        noun_phrase = dimension_name
        domain_key_idea = "the subject the statement refers to"
        example_block = ""

    excluded_block = _build_exclusion_block(
        excluded_facets or [], "excluded_facets"
    )

    return f"""You are assisting with qualitative analysis.

The observations below all belong to a specific facet within a domain. Your task is to identify the concrete attributes (L4) within this facet.

All of your output must be in this language:
<language>
{language}
</language>

You are working within this facet:
<facet>
{facet_name} — {facet_description}
</facet>
{excluded_block}
Here are the observations you need to analyze:
<observations>
{observations_block}
</observations>

<attribute_definition_guidance>
Taxonomy levels for this dimension:
- Dimension (L1): {noun_phrase}
- Domain (L2): {domain_key_idea}
- Facet (L3): {facet_key_idea}
- Attribute (L4): {attribute_key_idea}
{example_block}
Target abstraction level: ATTRIBUTE (L4)

{attribute_guidance}

Each attribute must be:
- **Ontologically distinct** — no two attributes may share conceptual space. An attribute must not be a subset of another attribute, and two attributes must not be two different lenses on the same phenomenon.
- **Semantically distant** — someone coding a response should clearly know which attribute applies, with no "could go either way" situations.
- Focused on ONE specific aspect (not a compound list of multiple concerns)
- A natural grouping of related phenomena within the facet
- Strictly within the boundaries of the included facet described above
</attribute_definition_guidance>

<task_instructions>
Follow these steps to complete your analysis:

**Step 1: Cluster observations**
Mentally group similar observations together. Look for recurring patterns and themes. Note which observations share the same {attribute_key_idea}.

**Step 2: Identify candidate attributes**
Based on your clustering, identify potential attributes. For each candidate attribute, write:
- The attribute name
- {attribute_question_stem} for this attribute
- Which observation numbers support it
- Whether it is ontologically distinct from other candidates

**Step 3: Verify distinctness**
Ensure that each attribute is:
- Ontologically distinct (not overlapping in conceptual space)
- Semantically distant (a coder would clearly know which to choose)
- Not two lenses on the same phenomenon

If two attributes fail this test, consolidate them into one.

**Step 4: Provide final output**
After your analysis, provide output as valid JSON following the response schema provided.
</task_instructions>

<key_reminders>
- Ensure attributes are ontologically distinct and semantically distant
- Each attribute should capture ONE {attribute_key_idea}, not multiple
- All attributes must fall within the included facet, not the excluded facets
- All output must be in {language}
</key_reminders>"""


# =============================================================================
# §4.25 ATTRIBUTE CHUNK CONSOLIDATION — merge chunk-level attributes within facet
# =============================================================================

class AttributeChunkConsolidatedResponse(BaseModel):
    """Consolidated attributes after merging chunk-level discoveries within a facet."""
    attributes: List[DiscoveredAttribute] = Field(
        ..., description="Fewest mutually exclusive attributes needed for full coverage, consolidated from all chunks"
    )


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

        example_block = ""
        if dimension_def.examples:
            ex = dimension_def.examples[0]
            example_block = f"""
Example (from a different survey):
  Survey: {ex.survey_context}
  Response: \"{ex.response}\"
  Domain: {ex.domain}
  Facet: {ex.facet}
  Instance: {ex.instance}
"""
    else:
        attribute_guidance = (
            "An attribute identifies the specific observable property or feature being described. "
            "It is a named property — not a verbatim span from the response."
        )
        attribute_key_idea = "the specific observable property being described"
        facet_key_idea = "the analytical lens applied to the subject"
        noun_phrase = dimension_name
        domain_key_idea = "the subject the statement refers to"
        example_block = ""

    excluded_block = _build_exclusion_block(
        excluded_facets or [], "excluded_facets"
    )

    return f"""You are a taxonomy consolidation specialist.
Your task is to merge multiple chunk-level attribute analyses into a single, coherent set of attributes for the facet \"{facet_name}\" within domain \"{domain_name}\".

All of your output must be in this language:
<language>
{language}
</language>

You are working within this facet:
<facet>
{facet_name} — {facet_description}
</facet>
{excluded_block}
Here are the attributes you need to consolidate:
<chunk_level_analyses>
{chunk_results}
</chunk_level_analyses>

<attribute_definition_guidance>
Taxonomy levels for this dimension:
- Dimension (L1): {noun_phrase}
- Domain (L2): {domain_key_idea}
- Facet (L3): {facet_key_idea}
- Attribute (L4): {attribute_key_idea}
{example_block}
Target abstraction level: ATTRIBUTE (L4)

{attribute_guidance}

Each attribute must be:
- **Ontologically distinct** — no two attributes may share conceptual space. An attribute must not be a subset of another attribute, and two attributes must not be two different lenses on the same phenomenon.
- **Semantically distant** — someone coding a response should clearly know which attribute applies, with no \"could go either way\" situations.
- Focused on ONE specific aspect (not a compound list of multiple concerns)
- A natural grouping of related phenomena within the facet
- Strictly within the boundaries of the included facet described above
</attribute_definition_guidance>

## YOUR TASK
Consolidate these chunk-level attribute lists into the fewest mutually exclusive attributes needed for full coverage within the facet \"{facet_name}\".

Important consolidation principles:
- MERGE attributes that have conceptual overlap, near-equivalence, or represent subcategories of a broader attribute
- ENSURE mutual exclusivity: no two attributes in your final list should overlap in meaning
- MAINTAIN full coverage: the consolidated attributes must collectively cover all concepts present in the chunk-level analyses
- MINIMIZE the total number of attributes while preserving meaningful distinctions
- When merging attributes, pick the most representative example observations from across the merged set (2-3 examples)
- All attribute names and descriptions must be in {language}

<scratchpad>
Follow these steps to complete your analysis:
1. List all unique attributes that appear across the chunk-level analyses
2. Identify groups of attributes that have conceptual overlap or proximity
3. For each group, determine an appropriate consolidated attribute name and description
4. Check that your consolidated attributes are mutually exclusive — for each pair ask: \"Could an observation plausibly belong to both?\" If yes, merge them.
5. Verify that your consolidated attributes provide complete coverage of the original set
</scratchpad>

All output MUST be in {language}.

Provide output as valid JSON following the response schema provided."""


# =============================================================================
# §4.5 ATTRIBUTE CONSOLIDATION (P3.5) — cross-facet dedup within domain
# =============================================================================

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


class AttributeConsolidatedResponse(BaseModel):
    """Consolidated attributes after cross-facet deduplication within a domain."""
    attributes: List[ConsolidatedAttribute] = Field(
        ..., description="Deduplicated attributes, each assigned to its best-fitting facet"
    )


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
) -> str:
    """Consolidate attributes across facets within a domain into a MECE set.

    P3.5: after P3 discovers attributes per facet independently, this step
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

        example_block = ""
        if dimension_def.examples:
            ex = dimension_def.examples[0]
            example_block = f"""
Example (from a different survey):
  Survey: {ex.survey_context}
  Response: "{ex.response}"
  Domain: {ex.domain}
  Facet: {ex.facet}
  Instance: {ex.instance}
"""
    else:
        attribute_guidance = (
            "An attribute identifies the specific observable property or feature being described. "
            "It is a named property — not a verbatim span from the response."
        )
        attribute_key_idea = "the specific observable property being described"
        facet_key_idea = "the analytical lens applied to the subject"
        noun_phrase = dimension_name
        domain_key_idea = "the subject the statement refers to"
        example_block = ""

    return f"""You are a taxonomy consolidation specialist.
Your task is to deduplicate attributes across facets within the domain "{domain_name}", producing a single MECE attribute inventory for the entire domain.

All of your output must be in this language:
<language>
{language}
</language>

You are working within this domain:
<domain>
{domain_name} — {domain_definition}
</domain>

Here are all facets and their discovered attributes:
<facet_attributes>
{facet_attributes_block}
</facet_attributes>

<attribute_definition_guidance>
Taxonomy levels for this dimension:
- Dimension (L1): {noun_phrase}
- Domain (L2): {domain_key_idea}
- Facet (L3): {facet_key_idea}
- Attribute (L4): {attribute_key_idea}
{example_block}
Target abstraction level: ATTRIBUTE (L4)

{attribute_guidance}

Each attribute must be:
- **Ontologically distinct** — no two attributes may share conceptual space, even across different facets. An attribute must not be a subset of another attribute, and two attributes must not be two different lenses on the same phenomenon.
- **Semantically distant** — someone coding a response should clearly know which attribute applies, with no "could go either way" situations.
- Assigned to exactly ONE parent facet (the best fit)
</attribute_definition_guidance>

## YOUR TASK
Deduplicate the attributes listed above. The same concept may have been discovered independently under multiple facets (e.g., "sparen" appearing under both "Financiële producten" and "Financiële functionaliteit"). Your job is to produce the fewest mutually exclusive attributes that cover all concepts, with each attribute assigned to its single best-fitting parent facet.

Important principles:
- MERGE attributes that overlap in meaning, even if they were discovered under different facets
- ASSIGN each surviving attribute to the ONE facet where it fits best
- Do NOT restructure or rename facets — only deduplicate attributes
- ENSURE mutual exclusivity: no two attributes in your final list should overlap in meaning
- MAINTAIN full coverage: every concept from the input must be represented
- When merging, pick the most representative example observations (2-3 examples)
- All attribute names and descriptions must be in {language}

<scratchpad>
Follow these steps:
1. List all attributes across all facets, noting which facets each appears in
2. Identify groups of attributes that overlap in meaning (same concept, different facet or wording)
3. For each group, choose one consolidated attribute name and assign it to its best-fitting facet
4. For each pair of surviving attributes, ask: "Could an observation plausibly belong to both?" If yes, merge them.
5. Verify complete coverage of the original attribute set
</scratchpad>

All output MUST be in {language}.

Provide output as valid JSON following the response schema provided."""


# =============================================================================
# §5 CODE GENERATION FROM ATTRIBUTES (P4) — cross-domain
# =============================================================================

class CodeFromAttributes(BaseModel):
    """A formal qualitative code derived from attributes."""
    code_name: str = Field(
        ..., description="Short code name (2-5 words)"
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


# Keep FormalCode as alias for backward compatibility with assignment infrastructure
FormalCode = CodeFromAttributes


class CodeGenerationFromAttributesResult(BaseModel):
    """P4 output: codes derived from attributes."""
    evaluation: str = Field(
        ..., description="Brief evaluation of how codes were derived from attributes"
    )
    codes: List[CodeFromAttributes] = Field(
        ..., description="Formal codes derived from the attribute inventory"
    )


def build_code_from_attributes_prompt(
    *,
    survey_question: str,
    language: str,
    dataset_context_section: str,
    dimension_name: str,
    dimension_description: str,
    domain_attributes: Dict[str, Dict[str, List[DiscoveredAttribute]]],
    valence_label: str = "",
) -> str:
    """Generate codebook codes from a (possibly valence-filtered) attribute inventory.

    Args:
        domain_attributes: {domain_name: {facet_name: [DiscoveredAttribute, ...]}}
        valence_label: "positive" or "negative" — scopes code generation by valence
    """
    # Build structured attribute inventory
    inventory_lines = []
    for domain_name, facet_attrs in sorted(domain_attributes.items()):
        inventory_lines.append(f"Domain: {domain_name}")
        for facet_name, attributes in sorted(facet_attrs.items()):
            inventory_lines.append(f"  Facet: {facet_name}")
            for attr in attributes:
                examples = "; ".join(attr.example_observations[:2])
                inventory_lines.append(
                    f"    - {attr.attribute_name}: {attr.attribute_description}"
                    + (f" (e.g., {examples})" if examples else "")
                )
        inventory_lines.append("")
    inventory_block = "\n".join(inventory_lines)

    # Valence context section
    valence_section = ""
    if valence_label == "positive":
        valence_section = """
<valence_context>
You are generating codes for POSITIVE and NEUTRAL responses only.
Focus on codes that capture what people appreciate, value, or neutrally observe.
</valence_context>
"""
    elif valence_label == "negative":
        valence_section = """
<valence_context>
You are generating codes for NEGATIVE responses only.
Focus on codes that capture complaints, criticisms, and suggestions for improvement.
</valence_context>
"""

    return f"""You are creating a qualitative codebook from a structured attribute inventory.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

<dimension_context>
Dimension: {dimension_name} — {dimension_description}
</dimension_context>
{valence_section}
The attribute inventory below is organized by Domain > Facet > Attribute.
Each attribute represents a concrete, observable phenomenon found in survey responses.

<attribute_inventory>
{inventory_block}
</attribute_inventory>

## TASK
Derive a PARSIMONIOUS codebook where each code represents a distinct, observable concept suitable for consistent coding.

<instructions>
Goal
Derive a PARSIMONIOUS codebook representing the smallest possible set of distinct {dimension_name} phenomena.

Definition of a Code
A code represents an underlying observable {dimension_name} phenomenon described in responses, not a single attribute.

Phenomenon Rule
Codes must represent underlying {dimension_name} phenomena rather than individual attributes.
Multiple attributes describing different manifestations of the same experience MUST be merged into a single code.

Specificity Rule
Do NOT create separate codes simply because attributes differ in specificity.
General statements and specific examples should be treated as indicators of the same phenomenon.

Example
"The train was delayed by 20 minutes" and "public transport is often late" both indicate unreliable punctuality and should be coded under the same broader phenomenon.

Example-Level Rule
Do NOT create codes that represent specific items or examples
These should be treated as indicators of broader phenomena.

Attribute Mapping Rule
Do NOT create a separate code for each attribute.
Attributes are observations that may belong to the same {dimension_name} phenomenon.

Minimum Coverage Rule
Each code should ideally cover multiple attributes.
Only create a single-attribute code if the phenomenon is clearly distinct.

Parsimony Rule
Prefer broader {dimension_name} codes over narrow attribute-based codes.
Use the smallest number of codes that still capture all distinct phenomena.

Expected Code Range
The final codebook should normally contain 3–5 codes unless the attributes clearly describe more distinct phenomena.

Mutual Exclusivity Rule
Codes must represent clearly different {dimension_name} phenomena so that responses can be coded consistently.

Hierarchy Rule
Only use attribute content to derive codes.
Do NOT create codes directly from domain or facet labels.

Process Requirement
Step 1 — Group attributes that describe the same underlying {dimension_name} phenomenon.
Step 2 — Assign a descriptive name to each phenomenon.
Step 3 — Convert each phenomenon into a formal qualitative code.

Output Requirements
Each code must include:
- Short name (3–5 word noun phrase)
- Clear definition
- Typical indicators
- source_attributes listing the attributes covered.

Language
All output must be written in {language}.
</instructions>

Provide output as valid JSON following the response schema provided."""


# =============================================================================
# §5.5 CODEBOOK CONSOLIDATION (P4.5) — cross-domain review & merge
# =============================================================================

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
    """P4.5 output: consolidated codebook."""
    evaluation: str = Field(
        ..., description="Brief analysis of what was merged/removed and why"
    )
    codes: List[ConsolidatedCode] = Field(
        ..., description="Final MECE codebook"
    )


def build_codebook_consolidation_prompt(
    *,
    survey_question: str,
    language: str,
    dataset_context_section: str,
    dimension_name: str,
    dimension_description: str,
    raw_codes: List[CodeFromAttributes],
    code_provenance: Dict[int, str],  # code index -> "domain::valence"
) -> str:
    """Consolidate per-domain codes into a final parsimonious, MECE codebook.

    Args:
        raw_codes: All codes from P4 (per-domain, valence-split)
        code_provenance: Maps code index to "domain_name::pos" or "domain_name::neg"
    """
    # Format raw codes with valence tags (no domain provenance)
    code_lines = []
    for i, code in enumerate(raw_codes):
        provenance = code_provenance.get(i, "")
        valence_tag = ""
        if "::pos" in provenance:
            valence_tag = "(+) "
        elif "::neg" in provenance:
            valence_tag = "(-) "

        attrs = ", ".join(code.source_attributes[:5]) if code.source_attributes else "—"
        indicators = "; ".join(code.typical_indicators[:3]) if code.typical_indicators else "—"
        code_lines.append(
            f"[C{i+1}] {valence_tag}{code.code_name}\n"
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
Create the fewest codes needed for full coverage, without conceptual overlap or semantic ambiguity.
The result must be conceptually clean, mutually exclusive, and easy for human coders to apply consistently.

<core_principles>

### 1. MAXIMAL REDUCTION
- Merge all codes that express the same underlying idea
- Ignore wording differences and examples
- Stop only when further merging would collapse clearly different dimensions
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
- If a human coder would hesitate between two codes for the same response, they should be one code
- Merge codes that are too similar to be distinctively applied

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

Provide output as valid JSON following the response schema provided."""


# =============================================================================
# §6 BRIDGE — codes → MECECode
# =============================================================================

def convert_codes_to_mece_categories(
    codes: list,
) -> List[MECECode]:
    """Convert code list to MECECode list for downstream assignment.

    Accepts both CodeFromAttributes (from P4) and ConsolidatedCode (from P4.5).
    """
    categories = []
    for code in codes:
        # ConsolidatedCode has diagnostic_test; CodeFromAttributes does not
        boundary = getattr(code, 'diagnostic_test', None)
        if not boundary:
            boundary = f"Does this idea express: {code.definition}?"

        categories.append(MECECode(
            category_label=code.code_name,
            inclusion_definition=code.definition,
            boundary_test=boundary,
            diagnostic_signals=code.typical_indicators[:5],
            key_expressions=[],
            tiebreaker_rules=[],
            subcategories=[],
        ))
    return categories


# Backward compatibility alias
convert_formal_codes_to_mece_categories = convert_codes_to_mece_categories


# =============================================================================
# §7 SHARED DATA MODELS — Used by code assignment, caching, step 6+
# =============================================================================

class MECECode(BaseModel):
    """A MECE category with independent boundary criteria."""
    hierarchy_level: Optional[int] = Field(
        default=None,
        description=(
            "This category's position in the hierarchy "
            "(1 = broadest top level, higher numbers = more specific). "
            "Leaf categories have the highest level number."
        )
    )
    interpretive_claim: Optional[str] = Field(
        default=None,
        description=(
            "Complete this sentence: 'Respondents construct [subject] as ...' "
            "For themes: the core interpretive insight. "
            "For subthemes: the specific meaning pattern this subtheme captures. "
            "Only used in thematic analysis — omit for assignment categories."
        )
    )
    category_label: str = Field(
        ...,
        description="Short noun phrase (1-4 words) naming this category"
    )
    inclusion_definition: str = Field(
        ...,
        description=(
            "What kinds of labels belong to this category. "
            "Must use observable criteria, not vague semantic descriptions."
        )
    )
    boundary_test: str = Field(
        ...,
        description=(
            "A yes/no question a human coder asks to determine if a label belongs here. "
            "Must be self-contained — no references to other categories."
        )
    )
    diagnostic_signals: List[str] = Field(
        ...,
        description=(
            "3-5 concrete words, phrases, or framings that, if present, "
            "indicate this category"
        )
    )
    key_expressions: List[str] = Field(
        ...,
        description="3-5 representative labels from the partition that exemplify this category"
    )
    tiebreaker_rules: List[str] = Field(
        ...,
        description=(
            "For each similar/adjacent category, a rule: "
            "'If ambiguous with [category X], assign here when [observable condition]'"
        )
    )
    subcategories: List[MECECode] = Field(
        default_factory=list,
        description="Child categories at the next hierarchy level."
    )


# Pydantic v2: rebuild model to resolve forward reference for recursive subcategories
MECECode.model_rebuild()


class MECEVerification(BaseModel):
    """Self-verification test for one pair of adjacent MECE categories."""
    category_a: str = Field(
        ...,
        description="First category in the pair — MUST be an exact category_label from your categories list"
    )
    category_b: str = Field(
        ...,
        description="Second category in the pair — MUST be an exact category_label from your categories list"
    )
    ambiguous_example: str = Field(
        ...,
        description="A constructed label that could plausibly fit either category"
    )
    assigned_to: str = Field(
        ...,
        description="Which category the ambiguous example is assigned to — MUST be either category_a or category_b"
    )
    reasoning: str = Field(
        ...,
        description="Why this assignment is correct, using only boundary_test and diagnostic_signals"
    )


# =============================================================================
# §8 CODE + ATTRIBUTE ASSIGNMENT (P5) — single idea, dual output
# =============================================================================

# ---- Internal wrapper models (used by code_assignment.py for downstream) ----

class CodeAssignment(BaseModel):
    """Single idea-to-code assignment (internal wrapper)."""
    idea_id: str = Field(..., description="The idea_id from the input")
    assigned_category_id: str = Field(
        ..., description="The code ID from [C#] prefix (e.g. 'C1', 'C7'). ONLY the ID."
    )
    confidence: float = Field(..., description="Confidence (0.0 to 1.0)")
    rationale: str = Field(..., description="Brief rationale")


class CodeAssignmentBatch(BaseModel):
    """Batch wrapper for uniform downstream handling."""
    assignments: List[CodeAssignment] = Field(
        ..., description="One assignment per idea"
    )


class CodeAttributeAssignment(BaseModel):
    """Single idea → code + attribute assignment."""
    assigned_code_id: str = Field(
        ...,
        description="The code ID from the [C#] prefix (e.g. 'C1', 'C7'). Return ONLY the ID."
    )
    assigned_attribute: str = Field(
        ...,
        description="The best-matching attribute name from the assigned code's attribute list."
    )
    confidence: float = Field(
        ...,
        description="Confidence in the assignment (0.0 to 1.0)"
    )
    rationale: str = Field(
        ...,
        description="Brief rationale for the code and attribute choice"
    )


def _build_codes_with_attributes_block(
    codes: List[CodeFromAttributes],
    other_label: Optional[str] = None,
) -> str:
    """Format codes with their source attributes for assignment prompt."""
    lines = []
    for i, code in enumerate(codes, 1):
        indicators = ", ".join(code.typical_indicators[:5]) if code.typical_indicators else "(none)"
        block = (
            f"[C{i}] {code.code_name}\n"
            f"    Definition: {code.definition}\n"
            f"    Indicators: {indicators}"
        )
        if code.source_attributes:
            attrs = "\n".join(f"      - {a}" for a in code.source_attributes)
            block += f"\n    Attributes:\n{attrs}"
        lines.append(block)

    if other_label:
        n = len(codes) + 1
        lines.append(
            f"[C{n}] {other_label}\n"
            f"    Definition: Ideas that do not clearly fit any of the above codes.\n"
            f"    Indicators: no matching indicators\n"
            f"    Attributes:\n      - (none)"
        )

    return "\n\n".join(lines)


def build_single_dual_assignment_prompt(
    *,
    survey_question: str,
    language: str,
    dataset_context_section: str,
    codes: List[CodeFromAttributes],
    other_label: Optional[str],
    idea,
    facet_lookup: Optional[Dict[str, str]] = None,
) -> str:
    """Build prompt for assigning a single idea to a code AND attribute."""
    codes_block = _build_codes_with_attributes_block(codes, other_label)

    # Format single idea
    valence = getattr(idea, 'valence', '') or '0'
    interpretation = getattr(idea, 'interpretation', '') or ''
    abstraction = getattr(idea, 'abstraction', '') or ''
    facet = (facet_lookup or {}).get(idea.idea_id, '') or getattr(idea, 'facet', '') or ''
    domain = getattr(idea, 'domain', '') or ''

    idea_block = (
        f"idea: {idea.idea}\n"
        f"interpretation: {interpretation}\n"
        f"abstraction: {abstraction}\n"
        f"domain: {domain}\n"
        f"facet: {facet}\n"
        f"valence: {valence}"
    )

    other_label_display = other_label or "Other"

    return f"""You are a qualitative coding assistant. Assign the idea below to the best-matching code AND attribute.

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
1. Read the idea's text, interpretation, facet, and valence.
2. Find the code whose definition best matches what the respondent is expressing.
3. Within that code, pick the attribute that most closely describes the specific phenomenon in this idea.
4. Return the code ID from [C#] brackets (e.g. "C1"). Do NOT return the code name.
5. Return the attribute name exactly as listed under the code.
6. Assign "{other_label_display}" only if NO code fits at all.
7. Rate confidence: 0.90+ = clear, 0.70-0.89 = good, 0.50-0.69 = approximate, <0.50 = weak.

All output MUST be in {language}.
Provide output as valid JSON following the response schema provided.
</instructions>
"""
