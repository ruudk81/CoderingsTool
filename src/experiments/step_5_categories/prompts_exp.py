"""
Prompts and Pydantic response models for Category Discovery.

Organized in pipeline processing order:
  §1  Theme Discovery (per-partition, chunked)
  §2  Theme Consolidation (per-partition)
  §3  Concept Discovery (per-partition)
  §4  Cross-Partition COC Consolidation
  §5  Hierarchical Codebook Construction
  §6  Shared Data Models (MECECategory, MECEVerification, ThematicAnalysisResult)
  §7  Bridge — HierarchicalCodebookResult → List[MECECategory]
  §8  Category Assignment (batch)
  §9  Category Assignment (single idea)
"""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# =============================================================================
# §1 THEME DISCOVERY — per-partition chunked theme extraction
# =============================================================================

def build_theme_discovery_prompt(
    *,
    survey_question: str,
    language: str,
    dataset_context_section: str,
    dimension_name: str,
    dimension_description: str,
    partition_name: str,
    partition_definition: str,
    labels: List[str],
    label_domains: Optional[List[Optional[str]]] = None,
) -> str:
    """Discover themes/insights from a chunk of labels."""
    _domains = label_domains or []
    labels_block = "\n".join(
        f"{i}. {label}" + (
            f"  [domain: {_domains[i - 1]}]"
            if i - 1 < len(_domains) and _domains[i - 1] else ""
        )
        for i, label in enumerate(labels, 1)
    )

    return f"""You are a qualitative research analyst.

Your task is to identify atomic themes that capture the shared patterns of meaning across coded responses.

A theme is a pattern of shared meaning across the dataset that answers the research question.
Themes are not just topics — they should capture something meaningful about the data.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
</survey_context>

<example>
Codes:
- loneliness
- homesickness
- missing family

Theme:
Emotional challenges of transition to university

Braun & Clarke emphasize that themes are constructed by the researcher, not simply "found" in the data.
</example>

All output MUST be in {language}.

Provide output as valid JSON following the response schema provided.

<coded_responses>
{labels_block}
</coded_responses>"""


class ThemeDiscoveryResult(BaseModel):
    """Prompt 1 output: concise theme phrases from a chunk of labels."""
    themes: List[str] = Field(
        ...,
        description="Concise theme phrases (one short sentence or phrase each)"
    )


# =============================================================================
# §2 THEME CONSOLIDATION — deduplicate themes per partition
# =============================================================================

def build_theme_consolidation_prompt(
    *,
    survey_question: str,
    language: str,
    dataset_context_section: str,
    dimension_name: str,
    dimension_description: str,
    partition_name: str,
    partition_definition: str,
    themes: List[str],
) -> str:
    """Consolidate raw themes per partition."""
    themes_block = "\n".join(
        f"{i}. {theme}" for i, theme in enumerate(themes, 1)
    )

    return f"""You are a qualitative research analyst performing theme consolidation.

You will receive a list of {len(themes)} themes/insights that were independently discovered from overlapping chunks of the same dataset. Because chunks overlap and theme discovery runs independently per chunk, the list contains SUBSTANTIAL redundancy: many themes are paraphrases, near-duplicates, or minor variations of the same underlying insight.

These themes were discovered along the taxonomy dimension "{dimension_name}": {dimension_description}
Within this taxonomy, we are looking at the section "{partition_name}": {partition_definition}

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

<raw_themes>
{themes_block}
</raw_themes>

<instructions>
Your task is to AGGRESSIVELY consolidate these {len(themes)} themes into a clean, deduplicated list. The input is expected to contain 50-80% redundancy — your output should be dramatically shorter than the input.

Consolidation rules:
1. Two themes describe the SAME insight if they refer to the same perception, evaluation, or experience — even when they use different words, emphasize different facets, or vary in specificity. For example:
   - "Strong focus on sustainability" and "Commitment to green energy and environment" → SAME insight (both about environmental commitment)
   - "Good customer service experience" and "Helpful and responsive staff" → SAME insight (both about positive service interactions)
   - "High prices compared to competitors" and "Products are expensive" → SAME insight (both about price perception)

2. For each group of overlapping themes, write ONE consolidated theme that captures the richest, most specific version of the insight. Do not water it down into a vague umbrella term.

3. Do NOT add new themes that aren't present in the input.

4. After your initial pass, do a SELF-CHECK: review your consolidated list and ask "Could any two of these remaining themes still be merged?" If yes, merge them. Repeat until no further merges are possible.

A typical result has 10-25 themes. If your output is longer than 30 themes, you are almost certainly under-merging — revisit and merge more aggressively.
</instructions>

All output MUST be in {language}.

Provide output as valid JSON following the response schema provided."""


class ConsolidatedThemesResult(BaseModel):
    """Prompt 1.5 output: consolidated themes for a single partition."""
    themes: List[str] = Field(
        ...,
        description="Consolidated list of distinct themes (each a concise, information-rich phrase)"
    )


# =============================================================================
# §3 CONCEPT DISCOVERY — themes → organizing concepts (per-partition)
# =============================================================================

def build_concept_discovery_prompt(
    *,
    survey_question: str,
    language: str,
    dataset_context_section: str,
    dimension_name: str,
    dimension_description: str,
    partition_name: str,
    partition_definition: str,
    themes: List[str],
) -> str:
    """Build Prompt 2a: Concept Discovery (per partition).

    Takes consolidated descriptive codes from a SINGLE partition and identifies
    central organizing concepts — interpretive claims about shared meaning
    patterns, not topic buckets.
    """
    themes_list = "\n".join(
        f"  {i}. {theme}" for i, theme in enumerate(themes, 1)
    )

    return f"""You are assisting with qualitative data analysis.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

<context>
Below is a set of descriptive codes derived from open-ended survey responses.
These codes originate from thematic analysis of the "{partition_name}" partition ({partition_definition}).
They belong to the taxonomy dimension "{dimension_name}" ({dimension_description}).

Your task is to identify the CENTRAL ORGANIZING CONCEPTS that explain the patterns across these codes.
</context>

<important_principles>
- Do not create topic buckets.
- Concepts must express an interpretive meaning about how respondents construct or understand the topic.
- Each concept should capture a shared meaning pattern across multiple descriptive codes.
- Concepts may reflect dimensions such as identity, practices, values, symbolism, audience alignment, time orientation, or legitimacy.
</important_principles>

<style_instruction>
Write in framework style, not narrative style.
Concepts should resemble analytical labels used in conceptual frameworks.
</style_instruction>

<descriptive_codes>
{len(themes)} descriptive codes from partition "{partition_name}":

{themes_list}
</descriptive_codes>

Follow the steps below.

STEP 1 — Normalize descriptive codes

Clean the list of descriptive codes so each expresses one clear meaning.

- Remove redundancy
- Split codes containing multiple ideas
- Rewrite as short analytic statements if needed

Output: List of normalized descriptive codes.

STEP 2 — Identify central organizing concepts

Group the descriptive codes into meaningful clusters.

For each concept provide:
- Concept name
- Short explanation (1-2 sentences explaining the interpretive idea)
- Descriptive codes included

Concepts must represent shared meaning patterns rather than topical groupings.

STEP 3 — Identify underlying meaning dimensions

Identify the conceptual dimensions that organize the concepts.

Possible examples include:
- Identity
- Practices or behaviors
- Values or principles
- Symbolism
- Audience or identity alignment
- Time orientation
- Legitimacy or credibility

Output: List of dimensions and the concepts associated with them.

STEP 4 — Concept compression (MECE-ready)

Convert the organizing concepts into a concise list of concept statements.

Requirements:
- Produce 4-7 concepts
- Each concept must be a short interpretive claim
- Maximum 12 words per concept
- No explanations or paragraphs
- Use the structure: [Subject] is constructed as [interpretive meaning]

Examples of acceptable structure:
- Sustainability is constructed as the moral core of the brand
- Banking is constructed as a mechanism for directing money toward impact
- Sustainability positioning invites scrutiny of authenticity

All output MUST be in {language}.

Provide output as valid JSON following the response schema provided."""


class OrganizingConcept(BaseModel):
    """A central organizing concept identified from descriptive codes."""
    concept_name: str = Field(
        ...,
        description=(
            "Short interpretive claim, max 12 words. "
            "Structure: '[Subject] is constructed as [interpretive meaning]'"
        )
    )
    explanation: str = Field(
        ...,
        description="1-2 sentence explanation of the interpretive idea"
    )
    codes_covered: List[str] = Field(
        ...,
        description="Descriptive codes from the input that this concept subsumes"
    )


class MeaningDimension(BaseModel):
    """An underlying meaning dimension that organizes concepts."""
    dimension_name: str = Field(
        ...,
        description="Name of the meaning dimension (e.g. identity, values, legitimacy)"
    )
    concepts_associated: List[str] = Field(
        ...,
        description="concept_name values that fall under this dimension"
    )


class ConceptDiscoveryResult(BaseModel):
    """Prompt 2a output: organizing concepts from descriptive codes."""
    normalized_codes: List[str] = Field(
        ...,
        description="Cleaned, deduplicated descriptive codes (one clear meaning each)"
    )
    organizing_concepts: List[OrganizingConcept] = Field(
        ...,
        description="4-7 central organizing concepts as short interpretive claims"
    )
    meaning_dimensions: List[MeaningDimension] = Field(
        ...,
        description="Conceptual dimensions that organize the concepts"
    )
    compressed_concepts: List[str] = Field(
        ...,
        description=(
            "Final 4-7 concept statements in the form "
            "'[Subject] is constructed as [interpretive meaning]'. "
            "Max 12 words each."
        )
    )


# =============================================================================
# §4 CROSS-PARTITION COC CONSOLIDATION
# =============================================================================

def build_coc_consolidation_prompt(
    *,
    survey_question: str,
    language: str,
    dataset_context_section: str,
    partition_concepts: Dict[str, ConceptDiscoveryResult],
    partition_definitions: Dict[str, str],
) -> str:
    """Build Prompt 3: Cross-Partition COC Consolidation.

    Takes organizing concepts from all partitions and produces the minimum
    set of unified concepts for full coverage — MECE, parsimonious,
    non-overlapping, non-redundant.
    """
    # Format per-partition concepts block
    partition_blocks = []
    for name in sorted(partition_concepts.keys()):
        concept_result = partition_concepts[name]
        definition = partition_definitions.get(name, "")
        concepts_list = "\n".join(
            f"  {i}. {c.concept_name}: {c.explanation}"
            for i, c in enumerate(concept_result.organizing_concepts, 1)
        )
        partition_blocks.append(
            f'Partition: "{name}" ({definition})\n{concepts_list}'
        )

    all_partitions_block = "\n\n".join(partition_blocks)

    return f"""You are assisting with qualitative research codebook development.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

<context>
Below are central organizing concepts discovered independently within each data partition.
Many concepts overlap across partitions because they capture the same underlying meaning pattern.

Your task is to consolidate these into the MINIMUM set of organizing concepts that covers ALL partitions.
</context>

<per_partition_concepts>
{all_partitions_block}
</per_partition_concepts>

<consolidation_criteria>
- MECE: each consolidated concept covers a distinct meaning dimension
- Parsimonious: use as few concepts as possible while maintaining full coverage
- Non-overlapping: no two consolidated concepts should capture the same meaning
- Non-redundant: every concept adds a distinct analytical insight
- Full coverage: every original per-partition concept must map to exactly one consolidated concept
- Traceability: record which partitions and original concepts feed into each consolidated concept
</consolidation_criteria>

<interpretive_requirement>
Concepts must represent interpretive claims about how respondents understand the topic.
Do NOT create topic buckets. A concept should answer: "Respondents construct [subject] as..."
</interpretive_requirement>

Follow the steps below.

STEP 1 — Identify overlaps

Group concepts across partitions that share the same interpretive claim or meaning dimension.
List each group with the original concept names and their source partitions.

STEP 2 — Merge redundant concepts

For each group of overlapping concepts, produce a single consolidated concept that:
- Captures the shared interpretive claim in a single statement
- Preserves the strongest/most precise explanation
- Records all source partitions and original concept names

STEP 3 — Preserve unique concepts

Identify concepts that are genuinely unique to a single partition or a small subset.
Keep these as separate consolidated concepts — do not force-merge distinct meanings.

STEP 4 — Final consolidated set

Present the final set of consolidated concepts.
Verify: every original per-partition concept must appear in exactly one consolidated concept's source_concepts list.

All output MUST be in {language}.

Provide output as valid JSON following the response schema provided."""


class ConsolidatedCOC(BaseModel):
    """A single consolidated organizing concept."""
    concept_name: str = Field(
        ..., description="Short interpretive claim (max 12 words)"
    )
    explanation: str = Field(
        ..., description="1-2 sentence explanation of this concept"
    )
    source_partitions: List[str] = Field(
        ..., description="Which partitions contributed to this concept"
    )
    source_concepts: List[str] = Field(
        ..., description="Original per-partition concept names merged into this one"
    )


class COCConsolidationResult(BaseModel):
    """Prompt 3 output: consolidated COCs across all partitions."""
    consolidation_rationale: str = Field(
        ..., description=(
            "Brief explanation (~150 words) of merge decisions: "
            "which concepts were combined, which kept distinct, and why."
        )
    )
    consolidated_concepts: List[ConsolidatedCOC] = Field(
        ..., description="The minimum set of organizing concepts for full coverage"
    )


# =============================================================================
# §5 HIERARCHICAL CODEBOOK CONSTRUCTION
# =============================================================================

def build_hierarchical_codebook_prompt(
    *,
    survey_question: str,
    language: str,
    dataset_context_section: str,
    consolidated_concepts: List[ConsolidatedCOC],
) -> str:
    """Build Prompt 4: Hierarchical Codebook Construction.

    Takes consolidated COCs from Prompt 3 and produces a 2-3 level
    hierarchical codebook (theme → subtheme/topic → optional sentiment/valence).
    """
    concepts_block = "\n".join(
        f"{i}. {c.concept_name}: {c.explanation} "
        f"[from: {', '.join(c.source_partitions)}]"
        for i, c in enumerate(consolidated_concepts, 1)
    )

    return f"""You are assisting with qualitative research codebook development.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

<context>
Below are consolidated organizing concepts derived from cross-partition thematic analysis.
These represent the unified set of meaning patterns across all data partitions.

Your task is to convert these concepts into a hierarchical MECE codebook.
</context>

<style_instruction>
Write in codebook style, not narrative or academic prose.
The output should resemble a coding manual used by qualitative researchers.
</style_instruction>

<hierarchy_structure>
The codebook uses a 2-3 level hierarchy:
- Level 1: THEMES — broad interpretive dimensions that organize the codebook
- Level 2: SUBTHEMES/TOPICS — specific codes under each theme, the primary coding targets
- Level 3: SENTIMENT/VALENCE — optional, only where valence (positive/negative/neutral framing) is analytically meaningful for a subtheme

Level 3 is NOT required for every subtheme. Only add it where the distinction between positive and negative framing adds analytical value.
</hierarchy_structure>

<mece_requirements>
- Mutually exclusive at each level: themes are distinct, subthemes within a theme are distinct
- Collectively exhaustive: all consolidated concepts must be covered
- Parsimonious: use as few themes and codes as possible
- Non-redundant: each code adds a distinct analytical insight
</mece_requirements>

<interpretive_requirement>
Codes must represent interpretive claims about how respondents understand the topic.
Do NOT create topic buckets.
</interpretive_requirement>

<constraint>
Use the provided consolidated concepts as the analytical foundation.
Every consolidated concept must be covered by at least one code.
Do NOT introduce new concepts.
</constraint>

<consolidated_concepts>
{concepts_block}
</consolidated_concepts>

Follow the steps below.

STEP 1 — Identify themes

Group the consolidated concepts into the smallest set of broad interpretive dimensions (themes).
Each theme should represent a distinct way respondents construct meaning about the topic.

STEP 2 — Construct subthemes

For each theme, create level 2 codes (subthemes/topics).

For each code provide:
- Code label: 2-4 word label
- Definition: Max 25 words describing the interpretive claim
- Include when: 2-3 short bullet rules
- Exclude when: 2 short bullet rules clarifying boundaries
- Diagnostic signals: 3-5 concrete words/phrases from respondent language
- Concepts covered: which consolidated concepts this code covers

STEP 3 — Add valence where meaningful

For subthemes where sentiment/valence adds analytical value, add level 3 codes.
These capture whether respondents frame the subtheme positively, negatively, or neutrally.

Only add level 3 where the distinction is analytically meaningful — do NOT add it mechanically to every subtheme.

STEP 4 — MECE validation

Provide a brief validation (~120 words) assessing:
- Mutual exclusivity at each level
- Collective exhaustiveness (all concepts covered)
- Parsimony

If problems are detected: revise and present the improved structure.

All output MUST be in {language}.

Provide output as valid JSON following the response schema provided."""


class CodebookCode(BaseModel):
    """Level 2 or 3 code in the hierarchical codebook."""
    code_label: str = Field(
        ..., description="Short label (2-4 words)"
    )
    definition: str = Field(
        ..., description="Interpretive claim, max 25 words"
    )
    level: int = Field(
        ..., description="Hierarchy level: 2 = subtheme/topic, 3 = sentiment/valence"
    )
    include_when: List[str] = Field(
        ..., description="2-3 short bullet rules for when this code applies"
    )
    exclude_when: List[str] = Field(
        ..., description="2 short bullet rules clarifying boundaries with other codes"
    )
    diagnostic_signals: List[str] = Field(
        ..., description="3-5 concrete words/phrases from respondent language"
    )
    concepts_covered: List[str] = Field(
        ..., description="Which consolidated COCs this code covers"
    )
    subcodes: List[CodebookCode] = Field(
        default_factory=list,
        description=(
            "Optional level 3 codes (sentiment/valence). "
            "Only include when valence is analytically meaningful for this subtheme."
        )
    )


# Resolve forward reference for recursive subcodes
CodebookCode.model_rebuild()


class CodebookTheme(BaseModel):
    """Level 1 theme in the hierarchical codebook."""
    theme_label: str = Field(
        ..., description="Broad theme label (2-5 words)"
    )
    theme_definition: str = Field(
        ..., description="Interpretive claim for this theme, max 30 words"
    )
    codes: List[CodebookCode] = Field(
        ..., description="Level 2 subtheme/topic codes under this theme"
    )


class HierarchicalCodebookResult(BaseModel):
    """Prompt 4 output: hierarchical codebook from consolidated COCs."""
    themes: List[CodebookTheme] = Field(
        ..., description="Level 1 themes, each containing level 2 (and optionally level 3) codes"
    )
    mece_validation: str = Field(
        ..., description=(
            "Brief validation (~120 words) assessing mutual exclusivity, "
            "collective exhaustiveness, and parsimony at each hierarchy level."
        )
    )


# =============================================================================
# §6 SHARED DATA MODELS — Used by category assignment, caching, step 6+
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


class ThematicAnalysisResult(BaseModel):
    """Wrapper for backward compatibility with qualitative_researcher.thematic_analysis property."""
    themes: List[MECECategory] = Field(
        ...,
        description=(
            "Analytical themes — each with an interpretive_claim ('Respondents construct [subject] as ...'). "
            "Each theme uses subcategories for its subthemes. "
            "Leaf subthemes are the coding targets. All themes and subthemes must have interpretive_claim filled."
        )
    )
    thematic_map: str = Field(
        ...,
        description=(
            "Brief narrative explaining how themes relate to each other, "
            "which are core vs contextual/evaluative, and the overall meaning pattern."
        )
    )


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
# §7 BRIDGE — HierarchicalCodebookResult → List[MECECategory]
# =============================================================================

def convert_hierarchical_to_mece_categories(
    result: HierarchicalCodebookResult,
) -> List[MECECategory]:
    """Convert Prompt 4 output to List[MECECategory] for downstream compatibility.

    Produces a hierarchical list: each CodebookTheme becomes a level 1
    MECECategory, with CodebookCodes as level 2 subcategories (and
    optional level 3 subcodes as level 3 subcategories).
    """
    categories = []
    for theme in result.themes:
        # Build level 2 subcategories
        subcats = []
        for code in theme.codes:
            # Build level 3 sub-subcategories (if any)
            level3_subcats = []
            for subcode in code.subcodes:
                level3_subcats.append(MECECategory(
                    hierarchy_level=3,
                    interpretive_claim=subcode.definition,
                    category_label=subcode.code_label,
                    inclusion_definition=subcode.definition,
                    boundary_test=_synthesize_boundary_test(subcode.include_when),
                    diagnostic_signals=subcode.diagnostic_signals,
                    key_expressions=subcode.concepts_covered[:5],
                    tiebreaker_rules=[
                        f"Exclude when: {rule}" for rule in subcode.exclude_when
                    ],
                    subcategories=[],
                ))

            subcats.append(MECECategory(
                hierarchy_level=2,
                interpretive_claim=code.definition,
                category_label=code.code_label,
                inclusion_definition=code.definition,
                boundary_test=_synthesize_boundary_test(code.include_when),
                diagnostic_signals=code.diagnostic_signals,
                key_expressions=code.concepts_covered[:5],
                tiebreaker_rules=[
                    f"Exclude when: {rule}" for rule in code.exclude_when
                ],
                subcategories=level3_subcats,
            ))

        # Build level 1 theme
        categories.append(MECECategory(
            hierarchy_level=1,
            interpretive_claim=theme.theme_definition,
            category_label=theme.theme_label,
            inclusion_definition=theme.theme_definition,
            boundary_test=f"Does this idea relate to: {theme.theme_label}?",
            diagnostic_signals=[],  # themes are organizational, not directly coded
            key_expressions=[c.code_label for c in theme.codes],
            tiebreaker_rules=[],
            subcategories=subcats,
        ))

    return categories


def _synthesize_boundary_test(include_when: List[str]) -> str:
    """Synthesize a yes/no boundary test question from include_when criteria."""
    if not include_when:
        return "Does this idea fit this code?"
    # Use the first (most important) inclusion criterion as the boundary test
    criterion = include_when[0].rstrip(".")
    return f"Does this idea express: {criterion}?"


# =============================================================================
# §8 CATEGORY ASSIGNMENT — batch
# =============================================================================

# ---- Prompt helpers ----

def _build_ideas_block(*, ideas: List) -> str:
    """Format ideas for the assignment prompt."""
    lines = []
    for idea in ideas:
        valence = getattr(idea, 'valence', '') or '0'
        lines.append(
            f"- idea_id: {idea.idea_id}\n"
            f"  idea: {idea.idea}\n"
            f"  interpretation: {idea.interpretation}\n"
            f"  abstraction: {idea.abstraction}\n"
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
    signals = (
        ", ".join(cat.diagnostic_signals)
        if cat.diagnostic_signals else "(none)"
    )
    block = f'{indent}[C{number}] {cat.category_label}'
    if cat.interpretive_claim:
        block += f"\n{indent}    Claim: {cat.interpretive_claim}"
    block += (
        f"\n{indent}    Inclusion: {cat.inclusion_definition}\n"
        f"{indent}    Boundary test: {cat.boundary_test}\n"
        f"{indent}    Diagnostic signals: {signals}"
    )
    if cat.tiebreaker_rules:
        tb_lines = "\n".join(
            f"{indent}      - {r}" for r in cat.tiebreaker_rules
        )
        block += f"\n{indent}    Tiebreaker rules:\n{tb_lines}"
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
            f"    Inclusion: Ideas that do not clearly fit any of the "
            f"above categories after applying all boundary tests.\n"
            f"    Boundary test: Do all other categories' boundary tests "
            f"fail for this idea?\n"
            f"    Diagnostic signals: no matching signals from any category"
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
            # Parent: show as scope header with claim + boundary test
            header = f"{indent}[Parent: {cat.category_label}]"
            if cat.interpretive_claim:
                header += f"\n{indent}  Claim: {cat.interpretive_claim}"
            if cat.inclusion_definition:
                header += f"\n{indent}  Scope: {cat.inclusion_definition}"
            if cat.boundary_test:
                header += f"\n{indent}  Boundary test: {cat.boundary_test}"
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
            f"    Inclusion: Ideas that do not clearly fit any of the "
            f"above categories after applying all boundary tests.\n"
            f"    Boundary test: Do all other categories' boundary tests "
            f"fail for this idea?\n"
            f"    Diagnostic signals: no matching signals from any category"
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
) -> str:
    """Build prompt for assigning ideas to MECE categories."""
    categories_block = _build_categories_block(
        categories=categories,
        other_label=other_label,
        hierarchical_categories=hierarchical_categories,
    )
    ideas_block = _build_ideas_block(ideas=ideas)

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
            f"Categories are organized in a {actual_depth}-level hierarchy.{depth_explanation}\n\n"
            f"Assignment rules for hierarchical categories:\n"
            f"- You MUST assign ideas to numbered leaf categories [C1], [C2], ... only.\n"
            f"- Use the parent's Scope and Boundary test to narrow your search: first identify "
            f"which parent group the idea belongs to, then select the best leaf within that group.\n"
            f"- When a leaf category could fit under multiple parents, the parent's Scope "
            f"acts as a tiebreaker: choose the leaf whose parent's Scope best matches "
            f"the idea's broader context.\n"
            f"- If two leaf categories across different parents seem equally fitting, "
            f"prefer the one whose parent Scope aligns more closely with the idea's "
            f"abstraction (broader significance).\n\n"
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
1. Read the idea text, interpretation (concrete interpretation), abstraction (broader significance), and valence (+/-/0). Understand the meaning pattern the respondent is expressing.
2. Theme-level screening: read each parent theme's Claim ("Respondents construct [subject] as ..."). Which theme captures the meaning pattern this idea participates in? The Claim tells you what interpretive lens to apply.
3. Subtheme matching: within the matching parent theme, read each subtheme's Claim and inclusion_definition. Which subtheme best captures the specific meaning pattern of this idea? Use diagnostic_signals as supporting evidence (one matching signal is sufficient).
4. Confirm with the boundary_test to verify the match.
5. If ambiguous between two subthemes, apply the tiebreaker_rules. If ambiguous between two parent themes, compare their Claims — which interpretive lens better fits the idea's abstraction (broader significance)?
6. Assign "{other_label_display}" ONLY as a last resort — when no subtheme even approximately captures the meaning pattern the respondent is expressing. Do not assign "{other_label_display}" simply because boundary criteria are not perfectly met; if a subtheme reasonably captures the idea's meaning, assign it.
7. Assign exactly ONE category per idea. For assigned_category_id, return the category ID shown in [C#] brackets (e.g. "C1", "C7"). Do NOT return the category name — return ONLY the ID.
8. Rate your confidence (0.0 to 1.0):
   - 0.90-1.00: subtheme clearly captures the meaning pattern, boundary_test and signals confirm
   - 0.70-0.89: subtheme captures the meaning pattern well, boundary_test mostly passes
   - 0.50-0.69: approximate fit — subtheme captures the gist but not perfectly
   - below 0.50: weak fit — strongly consider "{other_label_display}" instead
9. Provide a brief rationale explaining which meaning pattern you identified and why this subtheme captures it.

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
# §9 CATEGORY ASSIGNMENT — single idea
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
    idea_block = (
        f"idea_id: {idea.idea_id}\n"
        f"idea: {idea.idea}\n"
        f"interpretation: {idea.interpretation}\n"
        f"abstraction: {idea.abstraction}\n"
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
1. Read the idea text, interpretation, abstraction, and valence.
2. Compare against each category's inclusion_definition and boundary_test.
3. Use diagnostic_signals as supporting evidence.
4. If ambiguous between categories, apply tiebreaker_rules.
5. Assign "{other_label_display}" only if NO category fits at all.
6. For assigned_category_id, return the category ID shown in [C#] brackets (e.g. "C1", "C7"). Do NOT return the category name — return ONLY the ID.
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
