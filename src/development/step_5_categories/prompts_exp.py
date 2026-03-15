"""
Prompts and Pydantic response models for Category Discovery v3.

Organized in pipeline processing order:
  §1  Dimension Context Block (shared helper)
  §2  Facet Discovery (P1: per-domain, chunked)
  §3  Facet Assignment (P2: per-domain, batched)
  §4  Attribute Discovery (P3: per facet within domain)
  §5  Code Generation from Attributes (P4: cross-domain)
  §6  Bridge — codes → MECECategory
  §7  Shared Data Models (MECECategory, MECEVerification)
  §8  Code Assignment — batch (P5)
  §9  Code Assignment — single idea (P5)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from development.step_3_ideaExtractor.dimension_data import DimensionDefinition


# =============================================================================
# §1 DIMENSION CONTEXT BLOCK — shared helper for all prompts
# =============================================================================

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

This dimension analyzes: {dimension_def.criterion}

Taxonomy levels for this dimension:
- Domain (L2): {rules.domain_diagnostic}
- Facet (L3): {rules.facet_diagnostic}
  {rules.facet_instruction}
- Attribute (L4): {rules.attribute_diagnostic}
  {rules.attribute_instruction}
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
) -> str:
    """Discover facets (L3) from a chunk of observations within a domain."""
    observations_block = "\n".join(
        f"{i}. {obs}" for i, obs in enumerate(observations, 1)
    )

    taxonomy_block = build_dimension_context_block(
        dimension_def=dimension_def,
        dimension_name=dimension_name,
        dimension_description=dimension_description,
        domain_name=partition_name,
        domain_definition=partition_definition,
    )

    # Dimension-specific facet guidance
    if dimension_def:
        rules = dimension_def.prompt_rules
        facet_question = rules.facet_diagnostic
        facet_guidance = rules.facet_instruction
    else:
        facet_question = "What specific aspect or viewpoint does this represent?"
        facet_guidance = "Identify the specific viewpoint or characteristic within the domain."

    return f"""You are assisting with qualitative analysis.

The observations below are derived from responses to a survey question, grouped within a specified domain.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

{taxonomy_block}

<facet_level_guidance>
Target abstraction level: FACET (L3)

{facet_guidance}
A facet answers the question: {facet_question}

Your goal is to identify the **fewest meaningful facets** that explain the observations within the domain.

Guidelines:
- Prefer FEWER and BROADER facets rather than many narrow ones
- Merge facets whenever they could belong to the same broader viewpoint
- If two facets differ only by specific examples, they should be merged
- Do NOT create facets for observations outside the domain: {partition_name}
</facet_level_guidance>

## TASK
Identify recurring facets (L3) within the domain "{partition_name}" by answering: {facet_question}

<instructions>
Step 1 — Identify the main facets through which the entity is described within this domain.
Step 2 — Group observations under these facets.
Step 3 — Organize into dominant facets (capture the majority) and minor facets (infrequent or singletons).

Only return dominant facets.
</instructions>

<observations>
{observations_block}
</observations>

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


def _build_ideas_block_for_facet_assignment(ideas: List) -> str:
    """Format ideas for facet assignment prompt."""
    lines = []
    for idea in ideas:
        interpretation = getattr(idea, 'interpretation', '') or ''
        abstraction = getattr(idea, 'abstraction', '') or ''
        instance = getattr(idea, 'instance', '') or ''
        lines.append(
            f"- idea_id: {idea.idea_id}\n"
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
) -> str:
    """Discover concrete attributes (L4) within a facet."""
    observations_block = "\n".join(
        f"{i}. {obs}" for i, obs in enumerate(observations, 1)
    )

    taxonomy_block = build_dimension_context_block(
        dimension_def=dimension_def,
        dimension_name=dimension_name,
        dimension_description=dimension_description,
        domain_name=domain_name,
        domain_definition=domain_definition,
    )

    # Dimension-specific attribute guidance
    if dimension_def:
        attribute_guidance = dimension_def.prompt_rules.attribute_instruction
    else:
        attribute_guidance = (
            "An attribute identifies the specific observable property or feature being described. "
            "It is a named property — not a verbatim span from the response."
        )

    return f"""You are assisting with qualitative analysis.

The observations below all belong to a specific facet within a domain. Your task is to identify the concrete attributes (L4) within this facet.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

{taxonomy_block}

<facet_context>
Facet: {facet_name} — {facet_description}
</facet_context>

<attribute_level_guidance>
Target abstraction level: ATTRIBUTE (L4)

An attribute is a concrete, observable property within the facet "{facet_name}".
{attribute_guidance}

Guidelines:
- Attributes should be MORE SPECIFIC than the facet — they name distinct observable phenomena within it
- Each attribute should represent a clearly different concrete signal
- Merge attributes that express the same phenomenon in different words
- Aim for 3-10 attributes per facet (depending on the variety of observations)
</attribute_level_guidance>

## TASK
Within the facet "{facet_name}" ({facet_description}), identify the distinct concrete attributes present in the observations.

<observations>
{observations_block}
</observations>

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
) -> str:
    """Generate codebook codes from the complete attribute inventory.

    Args:
        domain_attributes: {domain_name: {facet_name: [DiscoveredAttribute, ...]}}
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

    return f"""You are creating a qualitative codebook from a structured attribute inventory.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

<dimension_context>
Dimension: {dimension_name} — {dimension_description}
</dimension_context>

The attribute inventory below is organized by Domain > Facet > Attribute.
Each attribute represents a concrete, observable phenomenon found in survey responses.

<attribute_inventory>
{inventory_block}
</attribute_inventory>

## TASK
Derive a codebook where each code represents a distinct, observable concept suitable for consistent coding.

<instructions>
- Each code should be grounded in one or more attributes from the inventory.
- If multiple attributes across different facets or domains express the same phenomenon, merge them into a single code.
- Codes must be mutually exclusive — a response should clearly belong to one code.
- Prefer broader codes over narrow ones, but ensure each code remains concrete and actionable.
- Each code needs: a short name, a clear definition, typical indicators (words/phrases that signal it), and which source attributes it covers.
- Include the source_attributes field listing which attribute names each code is derived from.
</instructions>

All output MUST be in {language}.

Provide output as valid JSON following the response schema provided."""


# =============================================================================
# §6 BRIDGE — codes → MECECategory
# =============================================================================

def convert_codes_to_mece_categories(
    codes: List[CodeFromAttributes],
) -> List[MECECategory]:
    """Convert CodeFromAttributes list to MECECategory list for downstream assignment."""
    categories = []
    for code in codes:
        categories.append(MECECategory(
            category_label=code.code_name,
            inclusion_definition=code.definition,
            boundary_test=f"Does this idea express: {code.definition}?",
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

class MECECategory(BaseModel):
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
    subcategories: List[MECECategory] = Field(
        default_factory=list,
        description="Child categories at the next hierarchy level."
    )


# Pydantic v2: rebuild model to resolve forward reference for recursive subcategories
MECECategory.model_rebuild()


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
# §8 CODE ASSIGNMENT — batch (P5)
# =============================================================================

# ---- Prompt helpers ----

def _build_ideas_block(*, ideas: List, facet_lookup: Optional[Dict[str, str]] = None) -> str:
    """Format ideas for the assignment prompt."""
    lines = []
    for idea in ideas:
        valence = getattr(idea, 'valence', '') or '0'
        interpretation = getattr(idea, 'interpretation', '') or ''
        abstraction = getattr(idea, 'abstraction', '') or ''
        # Use P2 facet assignment if available, fall back to idea.facet
        facet = (facet_lookup or {}).get(idea.idea_id, '') or getattr(idea, 'facet', '') or ''
        lines.append(
            f"- idea_id: {idea.idea_id}\n"
            f"  idea: {idea.idea}\n"
            f"  interpretation: {interpretation}\n"
            f"  abstraction: {abstraction}\n"
            f"  facet: {facet}\n"
            f"  valence: {valence}"
        )
    return "\n".join(lines)


def _format_leaf_category(
    *,
    cat: MECECategory,
    number: int,
    indent: str = "",
) -> str:
    """Format a single leaf category block for the assignment prompt."""
    indicators = (
        ", ".join(cat.diagnostic_signals)
        if cat.diagnostic_signals else "(none)"
    )
    block = (
        f"{indent}[C{number}] {cat.category_label}\n"
        f"{indent}    Definition: {cat.inclusion_definition}\n"
        f"{indent}    Indicators: {indicators}"
    )
    return block


def _build_categories_block(
    *,
    categories: List[MECECategory],
    other_label: Optional[str] = None,
    hierarchical_categories: Optional[List[MECECategory]] = None,
) -> str:
    """Format MECE categories for the assignment prompt.

    If hierarchical_categories is provided and contains subcategories,
    renders parent scope headers with numbered leaf items inside.
    Otherwise falls back to the original flat numbered list.
    """
    has_hierarchy = (
        hierarchical_categories is not None
        and any(cat.subcategories for cat in hierarchical_categories)
    )
    if has_hierarchy:
        return _build_hierarchical_categories_block(
            categories=hierarchical_categories, other_label=other_label
        )
    return _build_flat_categories_block(
        categories=categories, other_label=other_label
    )


def _build_flat_categories_block(
    *,
    categories: List[MECECategory],
    other_label: Optional[str] = None,
) -> str:
    """Format flat MECE categories for the prompt (original behavior)."""
    lines = []
    for i, cat in enumerate(categories, 1):
        lines.append(
            _format_leaf_category(cat=cat, number=i)
        )

    if other_label:
        n = len(categories) + 1
        lines.append(
            f"[C{n}] {other_label}\n"
            f"    Definition: Ideas that do not clearly fit any of the "
            f"above categories.\n"
            f"    Indicators: no matching indicators from any category"
        )

    return "\n\n".join(lines)


def _build_hierarchical_categories_block(
    *,
    categories: List[MECECategory],
    other_label: Optional[str] = None,
) -> str:
    """Format hierarchical MECE categories for the prompt.

    Parent categories shown as [Parent: ...] scope headers.
    Leaf categories get sequential numbers [1], [2]... across all
    parents — these are the assignment targets.
    """
    lines: List[str] = []
    counter = [0]  # mutable for closure access

    def _render(cat: MECECategory, depth: int):
        indent = "  " * depth
        if cat.subcategories:
            # Parent: show as scope header
            header = f"{indent}[Parent: {cat.category_label}]"
            if cat.inclusion_definition:
                header += f"\n{indent}  Definition: {cat.inclusion_definition}"
            lines.append(header)
            for child in cat.subcategories:
                _render(child, depth + 1)
        else:
            # Leaf: numbered assignment target
            counter[0] += 1
            lines.append(
                _format_leaf_category(
                    cat=cat, number=counter[0], indent=indent
                )
            )

    for cat in categories:
        _render(cat, depth=0)

    if other_label:
        counter[0] += 1
        lines.append(
            f"[C{counter[0]}] {other_label}\n"
            f"    Definition: Ideas that do not clearly fit any of the "
            f"above categories.\n"
            f"    Indicators: no matching indicators from any category"
        )

    return "\n\n".join(lines)


# ---- Prompt builder ----

def build_category_assignment_prompt(
    *,
    survey_question: str,
    language: str,
    dataset_context_section: str,
    partition_name: str,
    partition_inclusion: str,
    categories: List[MECECategory],
    other_label: Optional[str],
    hierarchical_categories: Optional[List[MECECategory]],
    ideas: List,
    facet_lookup: Optional[Dict[str, str]] = None,
) -> str:
    """Build prompt for assigning ideas to MECE categories."""
    categories_block = _build_categories_block(
        categories=categories,
        other_label=other_label,
        hierarchical_categories=hierarchical_categories,
    )
    ideas_block = _build_ideas_block(ideas=ideas, facet_lookup=facet_lookup)

    # Derive hierarchy instruction from hierarchical categories
    hierarchy_instruction = ""
    if hierarchical_categories and any(c.subcategories for c in hierarchical_categories):
        # Compute actual depth of the rendered hierarchy
        def _max_depth(cats: List[MECECategory], d: int = 1) -> int:
            if not cats:
                return d
            return max(
                (_max_depth(c.subcategories, d + 1) if c.subcategories else d)
                for c in cats
            )
        actual_depth = _max_depth(hierarchical_categories)

        depth_explanation = ""
        if actual_depth >= 3:
            depth_explanation = (
                f" The hierarchy has {actual_depth} levels: parent groups "
                f"can themselves be nested within higher-level groups. "
                f"Each [Parent: ...] header shows one level of the tree."
            )

        hierarchy_instruction = (
            f"Codes are organized in a {actual_depth}-level hierarchy.{depth_explanation}\n\n"
            f"Assignment rules for hierarchical codes:\n"
            f"- You MUST assign ideas to numbered leaf codes [C1], [C2], ... only.\n"
            f"- Use the parent's Definition to narrow your search: first identify "
            f"which parent group the idea belongs to, then select the best leaf code within that group.\n"
            f"- When a leaf code could fit under multiple parents, the parent's Definition "
            f"acts as a tiebreaker: choose the leaf whose parent best matches "
            f"the idea's meaning.\n\n"
        )

    # Use other_label for prompt references (fallback for display)
    other_label_display = other_label or "Other"

    return f"""You are a coding assistant assigning survey response ideas to pre-defined MECE coding categories.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

<partition_context>
Concept type: "{partition_name}"
Partition scope: {partition_inclusion}
</partition_context>

<categories>
You MUST assign each idea to exactly ONE of these categories:

{categories_block}
</categories>

<ideas_to_assign>
{ideas_block}
</ideas_to_assign>

<instructions>
{hierarchy_instruction}For each idea:
1. Read the idea text, facet, and valence.
2. Compare the idea against each code's definition. Which code best describes what the respondent is expressing?
3. Use indicators as supporting evidence — if the idea contains words or phrases matching a code's indicators, that strengthens the match.
4. If ambiguous between two codes, re-read their definitions and pick the one whose definition more precisely covers the idea's core meaning.
5. Assign "{other_label_display}" ONLY as a last resort — when no code even approximately fits.
6. Assign exactly ONE code per idea. For assigned_category_id, return the code ID shown in [C#] brackets (e.g. "C1", "C7"). Do NOT return the code name — return ONLY the ID.
7. Rate your confidence (0.0 to 1.0):
   - 0.90-1.00: code clearly fits, definition and indicators confirm
   - 0.70-0.89: code fits well, definition covers the idea
   - 0.50-0.69: approximate fit — code covers the gist but not perfectly
   - below 0.50: weak fit — strongly consider "{other_label_display}" instead
8. Provide a brief rationale explaining why this code fits the idea.

All output (rationale) MUST be in {language}.

Provide output as valid JSON following the response schema provided.
</instructions>
"""


# ---- Response models ----

class CategoryAssignment(BaseModel):
    """Single idea-to-category assignment."""
    idea_id: str = Field(
        ...,
        description="The idea_id from the input"
    )
    assigned_category_id: str = Field(
        ...,
        description=(
            "The category ID from the [C#] prefix (e.g. 'C1', 'C7', 'C12'). "
            "Return ONLY the ID, not the category label text."
        )
    )
    confidence: float = Field(
        ...,
        description="Confidence in the assignment (0.0 to 1.0)"
    )
    rationale: str = Field(
        ...,
        description="Brief rationale referencing boundary_test or diagnostic_signals"
    )


class CategoryAssignmentBatch(BaseModel):
    """Batch of category assignments for multiple ideas."""
    assignments: List[CategoryAssignment] = Field(
        ...,
        description="One assignment per idea in the input batch"
    )


# =============================================================================
# §9 CODE ASSIGNMENT — single idea (P5)
# =============================================================================

def build_single_idea_assignment_prompt(
    *,
    survey_question: str,
    language: str,
    dataset_context_section: str,
    partition_name: str,
    partition_inclusion: str,
    categories: List[MECECategory],
    other_label: Optional[str],
    idea,
    facet_lookup: Optional[Dict[str, str]] = None,
) -> str:
    """Build prompt for assigning a SINGLE idea to one MECE category.

    Uses a flat category list (no hierarchy) for maximum clarity with
    smaller models. Each category shown with inclusion criteria and
    diagnostic signals.
    """
    # Build flat category block (no hierarchy, just leaves)
    categories_block = _build_flat_categories_block(
        categories=categories,
        other_label=other_label,
    )

    # Format single idea
    valence = getattr(idea, 'valence', '') or '0'
    interpretation = getattr(idea, 'interpretation', '') or ''
    abstraction = getattr(idea, 'abstraction', '') or ''
    # Use P2 facet assignment if available, fall back to idea.facet
    facet = (facet_lookup or {}).get(idea.idea_id, '') or getattr(idea, 'facet', '') or ''
    idea_block = (
        f"idea_id: {idea.idea_id}\n"
        f"idea: {idea.idea}\n"
        f"interpretation: {interpretation}\n"
        f"abstraction: {abstraction}\n"
        f"facet: {facet}\n"
        f"valence: {valence}"
    )

    other_label_display = other_label or "Other"

    return f"""You are a qualitative coding assistant. Your task is to assign a single survey response idea to the most appropriate category from a predefined coding scheme.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

<partition_context>
Concept type: "{partition_name}"
Partition scope: {partition_inclusion}
</partition_context>

<categories>
Assign the idea to exactly ONE of these categories:

{categories_block}
</categories>

<idea>
{idea_block}
</idea>

<instructions>
1. Read the idea text, facet, and valence.
2. Compare against each code's definition. Which code best describes what the respondent is expressing?
3. Use indicators as supporting evidence.
4. If ambiguous between codes, re-read their definitions and pick the one that more precisely covers the idea.
5. Assign "{other_label_display}" only if NO code fits at all.
6. For assigned_category_id, return the code ID shown in [C#] brackets (e.g. "C1", "C7"). Do NOT return the code name — return ONLY the ID.
7. Rate confidence: 0.90-1.00 = clear match, 0.70-0.89 = good fit, 0.50-0.69 = approximate, <0.50 = weak.

All output MUST be in {language}.
Provide output as valid JSON following the response schema provided.
</instructions>
"""


class SingleCategoryAssignment(BaseModel):
    """Single idea-to-category assignment (one idea per call)."""
    assigned_category_id: str = Field(
        ...,
        description=(
            "The category ID from the [C#] prefix (e.g. 'C1', 'C7', 'C12'). "
            "Return ONLY the ID, not the category label text."
        )
    )
    confidence: float = Field(
        ...,
        description="Confidence in the assignment (0.0 to 1.0)"
    )
    rationale: str = Field(
        ...,
        description="Brief rationale referencing boundary_test or diagnostic_signals"
    )
