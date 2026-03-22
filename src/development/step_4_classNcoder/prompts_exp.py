"""
Prompts and Pydantic response models for Category Discovery v3.

Organized in pipeline processing order:
  §1   Dimension Context Block (shared helper)
  §2   Facet Discovery (P1: per-domain, chunked)
  §3   Facet Consolidation (P1.5: merge chunk-level facets)
  §4   Facet Assignment (P2: per-domain, batched)
  §5   Attribute Discovery (P3: per facet within domain)
  §6   Attribute Chunk Consolidation (P3.25: merge chunk-level attributes)
  §7   Attribute Assignment (P4a: per facet)
  §8   Attribute Consolidation (P3.5: cross-facet dedup within domain)
  §9   Code Generation from Attributes (P4: cross-domain)
  §10  Codebook Consolidation (P4.5: cross-domain merge)
  §11  Code Assignment (P5: single idea)
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
        tag_name: XML tag name, e.g. 'excluded_domains' or 'excluded_facets'.grea
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

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

You are working within this domain:
<domain>
{partition_name} — {partition_definition}
</domain>
{excluded_block}
Here are the observations you need to analyze:
<observations>
{observations_block}
</observations>

<facet_definition_guidance>
Dimension: {dimension_name} — {dimension_description}

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
For each dominant facet, provide:
- A short descriptive name (2-5 words)
- A description of what the facet captures (1-2 sentences)
- 3-5 representative observations from the input (exact text, not numbers)

Provide output as valid JSON following the response schema provided.
</task_instructions>

<key_reminders>
- Return ONLY dominant facets (3+ observations)
- Ensure facets are ontologically distinct and semantically distant
- Each facet should capture ONE {facet_key_idea}, not multiple
- All facets must fall within the included domain, not the excluded domains
- All output must be in {language}
</key_reminders>"""


# =============================================================================
# §3 FACET CONSOLIDATION (P1.5) — merge chunk-level facets into coherent set
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
Your task is to merge multiple chunk-level facet analyses into a single, minimal set of mutually exclusive facets within a given domain.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

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
Dimension: {dimension_name} — {dimension_description}

Target abstraction level: FACET (L3)
{facet_guidance}

Each facet must be:
- **Ontologically distinct** — each facet represents a fundamentally different underlying property, not a different framing of the same principle. Facets must not be subsets of one another and must not overlap in a way that creates ambiguity in classification.
- **Minimally ambiguous in application** — a coder should be able to assign an observation to exactly one facet in most cases. Edge cases are acceptable but must be resolvable through clear decision rules.
- **Focused on ONE specific aspect** — a facet captures a single distinguishable property, not a bundle of loosely related concerns.
- **A natural grouping** — the facet may group closely related phenomena, but only when they clearly stem from the same underlying principle.
- **Strictly within domain scope** — the facet must stay fully within the conceptual boundaries of the defined domain and not leak into adjacent domains.

Within this taxonomy:
- Dimension (L1): {noun_phrase}
- Domain (L2): {domain_key_idea}
- Facet (L3): {facet_key_idea}
- Attribute (L4): {attribute_key_idea}
</facet_definition_guidance>

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
“Can a clear rule assign every observation to exactly one facet?”
- No → merge
</disambiguation_test>

<precedence_rule>
When rules conflict, prioritize:
1. Non-overlap (orthogonality)
2. Minimality (merge unless clearly distinct)
3. Clarity for annotation

When in doubt → merge facets
</precedence_rule>

For each consolidated facet, provide:
- A short descriptive name (2-5 words)
- A description of what the facet captures (1-2 sentences)
- 3-5 representative observations selected from across the merged chunks (exact text)

All facet names and descriptions must be in {language}.

Provide output as valid JSON following the response schema provided."""

# =============================================================================
# §4 FACET ASSIGNMENT (P2) — per-domain batched assignment
# =============================================================================

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
1. Read the idea text and valence ([+] positive, [-] negative, [0] neutral).
2. Determine which facet best answers the question: {facet_question}
3. Assign exactly ONE facet per idea. Return the facet ID from [F#] brackets (e.g. "F1", "F3"). Do NOT return the facet name.
4. Assign "{other_label_display}" ONLY if no facet fits at all.
5. Rate your confidence (0.0 to 1.0).

All output MUST be in {language}.

Provide output as valid JSON following the response schema provided.
</instructions>
"""


# =============================================================================
# §5 ATTRIBUTE DISCOVERY (P3) — per facet within domain
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

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

You are working within this domain and facet:
<domain>
{domain_name} — {domain_definition}
</domain>
<facet>
{facet_name} — {facet_description}
</facet>
{excluded_block}
Here are the observations you need to analyze:
<observations>
{observations_block}
</observations>

<attribute_definition_guidance>
Dimension: {dimension_name} — {dimension_description}

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
For each attribute, provide:
- A short descriptive name (2-5 words)
- A description of what the attribute captures — a concrete, observable property (1-2 sentences)
- 2-3 representative observations from the input (exact text, not numbers)

Provide output as valid JSON following the response schema provided.
</task_instructions>

<key_reminders>
- Ensure attributes are ontologically distinct and semantically distant
- Each attribute should capture ONE {attribute_key_idea}, not multiple
- All attributes must fall within the included facet, not the excluded facets
- All output must be in {language}
</key_reminders>"""


# =============================================================================
# §6 ATTRIBUTE CHUNK CONSOLIDATION (P3.25) — merge chunk-level attributes within facet
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
Your task is to merge multiple chunk-level attribute analyses into a single, minimal set of mutually exclusive attributes within a given facet.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

You are working within this domain and facet:
<domain>
{domain_name}
</domain>
<facet>
{facet_name} — {facet_description}
</facet>
{excluded_block}

Here are the attributes you need to consolidate:
<chunk_level_analyses>
{chunk_results}
</chunk_level_analyses>

<attribute_definition_guidance>
Dimension: {dimension_name} — {dimension_description}

Target abstraction level: ATTRIBUTE (L4)
{attribute_guidance}

Each attribute must be:
- **Ontologically distinct** — each attribute represents a fundamentally different underlying property, not a different framing of the same principle. Attributes must not be subsets of one another and must not overlap in a way that creates ambiguity in classification.
- **Minimally ambiguous in application** — a coder should be able to assign an observation to exactly one attribute in most cases. Edge cases are acceptable but must be resolvable through clear decision rules.
- **Focused on ONE specific aspect** — an attribute captures a single distinguishable property, not a bundle of loosely related concerns.
- **A natural grouping** — the attribute may group closely related phenomena, but only when they clearly stem from the same underlying principle.
- **Strictly within facet scope** — the attribute must stay fully within the conceptual boundaries of the defined facet and not leak into adjacent facets.

Within this taxonomy:
- Dimension (L1): {noun_phrase}
- Domain (L2): {domain_key_idea}
- Facet (L3): {facet_key_idea}
- Attribute (L4): {attribute_key_idea}
</attribute_definition_guidance>

<strict_consolidation_rule>
1. MERGE OVERLAP (MANDATORY)
All attributes that conceptually overlap or are variants of the same idea must be merged.

2. ORTHOGONALITY (MAIN RULE)
For each pair of attributes:
"Can a single observation plausibly fall under both?"

- Yes → merge
- Doubt → merge
- Only if clearly no → keep separate

3. NO HIERARCHY
Attributes must not be:
- general vs. specific
- principle vs. application
If this occurs → merge

4. NO OBJECT SPLITTING
Do not split based on object (e.g., humans vs. animals)
If the same underlying principle applies → merge

5. MINIMALITY (MANDATORY)
Use the smallest number of attributes that provides full coverage.
If an attribute is not strictly necessary → remove it
</strict_consolidation_rule>

<disambiguation_test>
For any pair of attributes:
“Can a clear rule assign every observation to exactly one attribute?”
- No → merge
</disambiguation_test>

<precedence_rule>
When rules conflict, prioritize:
1. Non-overlap (orthogonality)
2. Minimality (merge unless clearly distinct)
3. Clarity for annotation

When in doubt → merge attributes
</precedence_rule>

For each consolidated attribute, provide:
- A short descriptive name (2-5 words)
- A description of what the attribute captures — a concrete, observable property (1-2 sentences)
- 2-3 representative observations selected from across the merged chunks (exact text)

All attribute names and descriptions must be in {language}.

Provide output as valid JSON following the response schema provided."""


# =============================================================================
# §7 ATTRIBUTE ASSIGNMENT (P4a) — per facet
# =============================================================================

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
    dimension_def: Optional[DimensionDefinition],
    dimension_name: str,
    dimension_description: str,
    domain_name: str,
    domain_definition: str,
    facet_name: str,
    facet_description: str,
    attributes: List['DiscoveredAttribute'],
    ideas: List,
) -> str:
    """Build prompt for assigning ideas to discovered attributes (L4) within a facet."""
    taxonomy_block = build_dimension_context_block(
        dimension_def=dimension_def,
        dimension_name=dimension_name,
        dimension_description=dimension_description,
        domain_name=domain_name,
        domain_definition=domain_definition,
    )

    attribute_codebook = _build_attribute_codebook_block(attributes)
    ideas_block = _build_ideas_block_for_facet_assignment(ideas)

    # Dimension-specific attribute question
    if dimension_def:
        attr_question = dimension_def.prompt_rules.attribute_diagnostic
    else:
        attr_question = "What specific quality or trait is being described?"

    return f"""You are a qualitative coding assistant assigning survey response ideas to discovered attributes within a facet.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

{taxonomy_block}

<facet_context>
Facet: {facet_name} — {facet_description}
</facet_context>

<attributes>
Assign each idea to exactly ONE of these attributes within the facet above:

{attribute_codebook}
</attributes>

<ideas_to_assign>
{ideas_block}
</ideas_to_assign>

<instructions>
For each idea:
1. Read the idea text and valence ([+] positive, [-] negative, [0] neutral).
2. Determine which attribute best answers the question: {attr_question}
3. Assign exactly ONE attribute per idea. Return the attribute ID from [A#] brackets (e.g. "A1", "A3"). Do NOT return the attribute name.
4. Rate your confidence (0.0 to 1.0).

All output MUST be in {language}.

Provide output as valid JSON following the response schema provided.
</instructions>
"""


# =============================================================================
# §8 ATTRIBUTE CONSOLIDATION (P3.5) — cross-facet dedup within domain
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
    source_attributes: List[str] = Field(
        default_factory=list,
        description="Original attribute names that were merged into this consolidated attribute"
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

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

You are working within this domain:
<domain>
{domain_name} — {domain_definition}
</domain>

Here are all facets and their discovered attributes:
<facet_attributes>
{facet_attributes_block}
</facet_attributes>

<attribute_definition_guidance>
Dimension: {dimension_name} — {dimension_description}

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
Deduplicate the attributes listed above. The same concept may have been discovered independently under multiple facets. Your job is to produce the fewest mutually exclusive attributes that cover all concepts, with each attribute assigned to its single best-fitting parent facet.

Do NOT restructure or rename facets — only deduplicate attributes.

<strict_consolidation_rule>
1. MERGE OVERLAP (MANDATORY)
All attributes that conceptually overlap or are variants of the same idea must be merged, even if they were discovered under different facets.

2. ORTHOGONALITY (MAIN RULE)
For each pair of attributes:
"Can a single observation plausibly fall under both?"

- Yes → merge
- Doubt → merge
- Only if clearly no → keep separate

3. NO HIERARCHY
Attributes must not be:
- general vs. specific
- principle vs. application
If this occurs → merge

4. NO OBJECT SPLITTING
Do not split based on object (e.g., humans vs. animals)
If the same underlying principle applies → merge

5. MINIMALITY (MANDATORY)
Use the smallest number of attributes that provides full coverage.
If an attribute is not strictly necessary → remove it

6. FACET ASSIGNMENT
Assign each surviving attribute to the ONE facet where it fits best.
</strict_consolidation_rule>

<disambiguation_test>
For any pair of attributes:
"Can a clear rule assign every observation to exactly one attribute?"
- No → merge
</disambiguation_test>

<precedence_rule>
When rules conflict, prioritize:
1. Non-overlap (orthogonality)
2. Minimality (merge unless clearly distinct)
3. Clarity for annotation

When in doubt → merge attributes
</precedence_rule>

For each consolidated attribute, provide:
- A short descriptive name (2-5 words)
- A description of what the attribute captures (1-2 sentences)
- The parent facet this attribute best belongs to
- 2-3 representative example observations (exact text)
- source_attributes: list of original attribute names that were merged into this one

All output MUST be in {language}.

Provide output as valid JSON following the response schema provided."""


# =============================================================================
# §9 CODE GENERATION FROM ATTRIBUTES (P4) — cross-domain
# =============================================================================

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
    """P4 output: codes derived from attributes."""
    scratchpad: str = Field(
        ..., description=(
            "Step-by-step reasoning before deriving codes: "
            "(1) identify underlying phenomena by grouping attributes, "
            "(2) check for valence distinctions, "
            "(3) name each phenomenon, "
            "(4) verify parsimony and coverage"
        )
    )
    evaluation: str = Field(
        ..., description="Brief evaluation of how codes were derived from attributes — what was merged and why"
    )
    codes: List[CodeFromAttributes] = Field(
        ..., description="Formal codes derived from the attribute inventory"
    )


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
            "\nYou must NOT include categories that belong to these excluded domains:\n"
            "<excluded_domains>\n"
            + "\n\n".join(excl_lines)
            + "\n</excluded_domains>"
        )

    # Compute attribute frequencies
    attr_counts: Dict[str, int] = {}
    if attribute_assignments:
        for attr_name in attribute_assignments.values():
            attr_counts[attr_name] = attr_counts.get(attr_name, 0) + 1

    # Build inventory: Facet > Attribute (single domain)
    facet_attrs = next(iter(domain_attributes.values()), {})
    inventory_lines = []
    for facet_name, attributes in sorted(facet_attrs.items()):
        inventory_lines.append(f"\n{facet_name}")
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

Here is the taxonomy inventory of attributes organized by facet:

<taxonomy_inventory>
The inventory below is organized by Facet > Attribute
{inventory_block}
</taxonomy_inventory>

# Understanding Phenomena vs Attributes

**Attributes** are specific observations or qualities mentioned in responses. They represent individual data points.

**Phenomena** are underlying conceptual patterns that multiple attributes may indicate. A phenomenon is the broader experience, perception, or association that manifests through various specific attributes.

Your task is to identify phenomena, NOT to create one code per attribute.

# Code Derivation Rules

## 1. Phenomenon Rule
Codes must represent underlying PHENOMENA rather than individual attributes. Multiple attributes describing different manifestations of the same underlying experience, perception, or association MUST be merged into a single code.

## 2. Specificity Rule
Do NOT create separate codes simply because attributes differ in specificity. General statements and specific examples should be treated as indicators of the same phenomenon.

Example: "The train was delayed by 20 minutes" and "public transport is often late" both indicate unreliable punctuality and should be coded under the same broader phenomenon.

## 3. Example-Level Rule
Do NOT create codes that represent specific items or examples. These should be treated as indicators of broader phenomena.

## 4. Attribute Mapping Rule
Do NOT create a separate code for each attribute. Attributes are observations that may belong to the same phenomenon.

## 5. Minimum Coverage Rule
Each code should ideally cover multiple attributes. Only create a single-attribute code if the phenomenon is clearly distinct and cannot be meaningfully merged with others.

## 6. Parsimony Rule
Prefer broader phenomenon-based codes over narrow attribute-based codes. Use the smallest number of codes that still capture all distinct phenomena present in the inventory.

## 7. Expected Code Range
The final codebook should normally contain 3–5 codes unless the attributes clearly describe more distinct phenomena.

## 8. Mutual Exclusivity Rule
Codes must represent clearly different phenomena so that responses can be coded consistently without ambiguity.

## 9. Valence Sensitivity Rule
Generate separate codes for positive and negative phenomena. Do NOT combine praise and criticism into a single code. If the attributes contain both positive and negative aspects of similar phenomena, create distinct codes for each valence direction.

## 10. Hierarchy Rule
Only use attribute content to derive codes. Do NOT create codes directly from domain or facet labels. Facets are organizational structures; your codes should emerge from the actual attribute patterns.

# Required Process

Before writing your final output, think through your analysis in the scratchpad field:

**Step 1 — Identify Underlying Phenomena**
Review all attributes across all facets. Look for patterns where multiple attributes describe different manifestations of the same underlying phenomenon. Group attributes that share the same conceptual core.

**Step 2 — Check for Valence Distinctions**
Within each phenomenon group, check whether positive and negative valences are present. If so, split into separate codes.

**Step 3 — Name Each Phenomenon**
Assign a descriptive name (3-5 word noun phrase in {language}) to each distinct phenomenon.

**Step 4 — Verify Parsimony and Coverage**
Ensure you have the minimum number of codes needed while covering all attributes. Aim for 3-5 codes unless the data clearly requires more.

# Output Requirements

Provide output as valid JSON following the response schema provided.

# Language Requirement

All output (code names, definitions, typical indicators, and evaluation) must be written in {language}.

# Final Notes

Remember: You are creating a PARSIMONIOUS codebook. Resist the temptation to create one code per attribute or per facet. Look for the deeper phenomena that connect multiple attributes together. Your goal is conceptual clarity with minimal redundancy.

Begin your analysis now."""


# =============================================================================
# §10 CODEBOOK CONSOLIDATION (P4.5) — cross-domain review & merge
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
    code_provenance: Dict[int, str],
    code_frequencies: Optional[Dict[int, int]] = None,
) -> str:
    """Consolidate per-domain codes into a final parsimonious, MECE codebook.

    Args:
        raw_codes: All codes from P4 (per-domain)
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



# =============================================================================
# §11 CODE ASSIGNMENT (P5) — single idea
# =============================================================================

# Re-export data-flow wrapper models (canonical definition in models_exp.py)
from .models_exp import CodeAssignment, CodeAssignmentBatch


class CodeAttributeAssignment(BaseModel):
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
