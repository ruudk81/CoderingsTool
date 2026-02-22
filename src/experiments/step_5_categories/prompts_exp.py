"""
Prompts and Pydantic response models for Theme Discovery.

Three-step MAP/REDUCE/MECE pipeline operating on descriptive codes
within each concept_type partition.

Terminology:
  - partition = concept_type group (data-driven from step 3)
  - theme     = overarching pattern of shared meaning discovered by MAP/REDUCE
  - category  = operationalized theme with MECE boundaries (from MECE step)
  - label     = the text string being analyzed (default: concept_type_definition field)
"""

from typing import List
from pydantic import BaseModel, Field


# =============================================================================
# PARTITION MODELS (data-driven concept_type groups)
# =============================================================================

class PartitionDescription(BaseModel):
    """Description of a concept_type partition."""
    partition_name: str = Field(
        ...,
        description="Concept type name (data-driven, e.g., 'recommendation', 'product_feature')"
    )
    inclusion_definition: str = Field(
        ...,
        description=(
            "What kinds of statements belong to this partition. "
            "Uses observable criteria."
        )
    )
    boundary_test: str = Field(
        ...,
        description=(
            "A yes/no question a coder asks to determine if a statement "
            "belongs to this partition."
        )
    )
    diagnostic_signals: List[str] = Field(
        ...,
        description="3-5 concrete words or phrases that indicate this partition"
    )


class PartitionSet(BaseModel):
    """Complete set of concept_type partitions."""
    partitions: List[PartitionDescription] = Field(
        ...,
        description="List of populated concept_type partitions"
    )


# =============================================================================
# MAP — Candidate theme extraction per batch
# =============================================================================

MAP_CATEGORIES_PROMPT = """You are a codebook designer specialized in thematic analysis as professed by Braun and Clarcke (2006)
Your goal is identify themes within a cluster of descriptive codes derived from responses to a survey question that share semantic meaning.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

<cluster_context>
Current cluster: 
{partition_name}

Cluster description: 
{partition_inclusion}

CRITICAL: Only identify categories that fall WITHIN "{partition_name}".
</cluster_context>

<cluster_with_descriptive_codes>
{labels_list}
<\cluster_with_descriptive_codes>

<task>
Identify the overarching theme or themes that capture patterns of shared meaning across the cluster with descriptive codes
</task>

<instruction>
- Work bottom-up from the existing codes and categories.
- Do not impose external theories, frameworks, or pre-defined analytic concepts.
- Do not simply restate or rename the existing categories.
- Identify the minimum number of themes needed to account for the patterned meanings in the data.
- If the data supports more than one overarching theme, explicitly justify why multiple themes are needed.

- For each theme:
    • Describe the central organizing idea that gives the theme coherence
    • Explain how the theme integrates multiple codes or categories
    • Provide a clear analytic theme label
    • Provide a more descriptive or stakeholder-friendly label, if different
    • Make explicit how the themes relate to one another (e.g., hierarchy, tension, conditionality, complementarity).

- Output goal: Produce a small set of inductively derived themes that represent meaningful patterns in the dataset and can be clearly communicated to non-academic audiences.
</instruction>

Provide output as valid JSON following the response schema.
All output (labels, descriptions, rationales) MUST be in {language}.
"""


class CandidateTheme(BaseModel):
    """A candidate theme identified in a batch of descriptive codes."""
    theme_label: str = Field(
        ...,
        description="Short noun phrase (1-4 words) naming this theme"
    )
    central_organizing_idea: str = Field(
        ...,
        description=(
            "The core idea that gives this theme coherence — "
            "what pattern of shared meaning unites the codes"
        )
    )
    description: str = Field(
        ...,
        description="How this theme integrates multiple codes: what they share, why they cluster"
    )
    example_codes: List[str] = Field(
        ...,
        description="2-4 descriptive codes from the batch that exemplify this theme (exact quotes)"
    )
    rationale: str = Field(
        ...,
        description="Why these codes belong together under this theme"
    )


class MapBatchThemes(BaseModel):
    """Candidate themes identified in a single batch of descriptive codes."""
    themes: List[CandidateTheme] = Field(
        ...,
        description="Candidate themes identified in this batch (typically 3-8)"
    )


# =============================================================================
# REDUCE — Cross-batch thematic synthesis
# =============================================================================

REDUCE_THEMES_PROMPT = """You are a thematic analyst synthesizing candidate themes discovered across multiple batches of descriptive codes from survey responses, following the approach of Braun and Clarke (2006).

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

{taxonomy_context}

<partition_context>
You are analyzing descriptive codes within the concept type: "{partition_name}"
Partition scope: {partition_inclusion}
Partition boundary test: {partition_boundary_test}

Peer partitions (do NOT identify themes that belong to these):
{peer_partitions_list}

CRITICAL: Only retain themes that fall WITHIN "{partition_name}".
</partition_context>

{cluster_hints}

<instruction>
You analyzed {n_batches} batches of descriptive codes from the partition "{partition_name}" and identified {n_total_themes} candidate themes across all batches. Because batches were processed independently, many themes overlap or capture different facets of the same underlying pattern of meaning. Your task is to synthesize them into a coherent set of overarching themes.
{grouping_instruction}
</instruction>

<principles>
1. INTEGRATE candidate themes that capture the same underlying pattern of shared meaning,
   even if they were labelled differently across batches.
2. For each synthesized theme, articulate its CENTRAL ORGANIZING IDEA: the core pattern
   of meaning that gives the theme coherence and explains why the codes cluster together.
3. AIM for the minimum number of themes needed to account for the patterned meanings
   in this partition. Fewer well-defined themes are better than many fragmented ones.
4. When integrating, pick the BEST theme label from the input candidates
   (do NOT invent a new label unless none of the existing labels are adequate).
5. Every input candidate theme must appear in exactly one synthesized theme's
   integrated_from list. Do not drop any.
6. Preserve the most informative central organizing idea and description when integrating.
7. DISCARD any candidate themes that actually belong to peer partitions, not to "{partition_name}".
</principles>

<candidate_themes_from_batches>
{batch_categories_list}
</candidate_themes_from_batches>

<task>
1. Read all {n_total_themes} candidate themes from {n_batches} batches.
2. Identify themes that capture the same or overlapping patterns of shared meaning.
3. Synthesize them into a coherent set of 3-12 overarching themes for this partition.
   If you have significantly more, you are likely preserving unnecessary distinctions.
4. For each synthesized theme, articulate its central organizing idea — the core pattern
   that unites the underlying codes.
5. CRITICAL: every input candidate theme label must appear in exactly one synthesized
   theme's integrated_from list. Do not drop any.

All output (labels, descriptions, central organizing ideas) MUST be in {language}.

Provide output as valid JSON following the response schema provided.
</task>
"""


class SynthesizedTheme(BaseModel):
    """A theme synthesized from candidate themes across MAP batches."""
    theme_label: str = Field(
        ...,
        description="Short noun phrase (1-4 words) naming this theme"
    )
    central_organizing_idea: str = Field(
        ...,
        description=(
            "The core pattern of shared meaning that gives this theme coherence — "
            "why the underlying codes cluster together"
        )
    )
    description: str = Field(
        ...,
        description="How this theme integrates the underlying codes — what they share and why they cluster"
    )
    integrated_from: List[str] = Field(
        ...,
        description=(
            "All candidate theme labels from the MAP batches that were integrated "
            "into this theme. If standalone, list contains only the original label."
        )
    )


class SynthesizedThemeList(BaseModel):
    """Synthesized theme list from the reduce step."""
    themes: List[SynthesizedTheme] = Field(
        ...,
        description="Overarching themes after cross-batch synthesis (typically 3-12)"
    )


# =============================================================================
# MECE — Apply boundaries with self-verification
# =============================================================================

MECE_BOUNDARIES_PROMPT = """You are a codebook designer applying MECE (Mutually Exclusive, Collectively Exhaustive) constraints to a list of coding categories within a single partition of survey response labels.

Your output must be a DECISION SYSTEM — not a semantic description. Each category must be defined by criteria a human coder can independently apply.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

{taxonomy_context}

<partition_context>
You are analyzing labels within the concept type: "{partition_name}"
Partition scope: {partition_inclusion}
Partition boundary test: {partition_boundary_test}

Peer partitions (do NOT identify categories that belong to these):
{peer_partitions_list}

CRITICAL: Only retain categories that fall WITHIN "{partition_name}".
</partition_context>

{cluster_hints}

<instruction>
You are given {n_categories} consolidated coding categories from the partition "{partition_name}" containing {n_labels} unique category labels. Your task is to ensure this list is MECE:
- Mutually Exclusive: each label should clearly belong to exactly one category
- Collectively Exhaustive: every label in this partition should fit one category

You may merge categories that still overlap, split categories that are too broad, or adjust boundaries.
{grouping_instruction}
CRITICAL DESIGN RULE: Define each category using POSITIVE, INDEPENDENT criteria.
- DO NOT define a category by what it is NOT or by referencing other categories.
- Each category's boundary_test must work WITHOUT knowing what other categories exist.
- Use observable characteristics of the label, not abstract semantic descriptions.
</instruction>

<principles>
1. The result must be a clean, non-overlapping set of categories WITHIN "{partition_name}".
2. Each category needs:
   - A BOUNDARY TEST: a yes/no question a coder can ask independently
   - DIAGNOSTIC SIGNALS: concrete words/phrases that trigger assignment
   - TIEBREAKER RULES: for each neighboring category, a concrete rule for ambiguous cases
3. Categories are VALENCE-NEUTRAL: name what people talk about, not how they feel.
4. Actively merge related categories into broader containers.
   Fewer well-defined categories are far better than many overlapping ones.
5. AIM for 3-10 final categories per partition. If you have significantly more,
   merge categories that a coder would confuse.
6. Every category from the input must be accounted for in the output (merged, kept, or split).
7. If a category actually belongs to a peer partition, exclude it from the final set.
</principles>

<label_constraints>
Category labels MUST:
- Be in {language}
Category labels MUST NOT:
- contain "en/and/und/et",
- contain slashes,
- stack multiple adjectives,
- list multiple attributes.
Each label must express ONE core subject only.

All output (labels, definitions, key_expressions) MUST be in {language}.
</label_constraints>

<consolidated_categories>
Partition: {partition_name}
Total unique labels in partition: {n_labels}

{categories_list}
</consolidated_categories>

<task>
1. Review the {n_categories} categories and their descriptions.
2. Check for overlaps: merge categories where a coder would hesitate between them.
3. Check for gaps: is there any common label type within "{partition_name}" not covered?
4. For each final category, define:
   - category_label: short name
   - inclusion_definition: what labels belong (observable criteria)
   - boundary_test: a yes/no question a coder asks to determine membership
     (MUST be self-contained — no references to other categories)
   - diagnostic_signals: 3-5 concrete words/phrases/framings that trigger assignment
   - key_expressions: 3-5 representative labels from the partition
   - tiebreaker_rules: for each similar/adjacent category, a rule:
     "If ambiguous with [category X], assign here when [observable condition]"
5. VERIFY your categories are MECE by testing each pair of adjacent/similar categories:
   - Construct one AMBIGUOUS example (a label that could plausibly fit either category)
   - Show which category gets it and WHY, using only your boundary_test and diagnostic_signals
   - If you cannot decide using your criteria alone, your categories are NOT MECE — merge or redefine them before finalizing.

Provide output as valid JSON following the response schema provided.
</task>
"""


class MECECategory(BaseModel):
    """A MECE category with independent boundary criteria."""
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


class MECEVerification(BaseModel):
    """Self-verification test for one pair of adjacent MECE categories."""
    category_a: str = Field(
        ...,
        description="First category in the pair"
    )
    category_b: str = Field(
        ...,
        description="Second category in the pair"
    )
    ambiguous_example: str = Field(
        ...,
        description="A constructed label that could plausibly fit either category"
    )
    assigned_to: str = Field(
        ...,
        description="Which category the ambiguous example is assigned to"
    )
    reasoning: str = Field(
        ...,
        description="Why this assignment is correct, using only boundary_test and diagnostic_signals"
    )


class MECECategorySet(BaseModel):
    """Complete MECE category set for a single partition, with self-verification."""
    categories: List[MECECategory] = Field(
        ...,
        description="MECE categories for this partition"
    )
    mece_verifications: List[MECEVerification] = Field(
        ...,
        description=(
            "Self-verification tests: one per pair of adjacent/similar categories. "
            "Proves the boundary criteria actually work."
        )
    )


# =============================================================================
# CATEGORY ASSIGNMENT — Prompt + Response Models
# =============================================================================

CATEGORY_ASSIGNMENT_PROMPT = """You are a coding assistant assigning survey response ideas to pre-defined MECE coding categories.

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
For each idea:
1. Read the idea text, concept, and concept_type_definition.
2. Apply each category's boundary_test to determine which category fits.
3. Use diagnostic_signals to confirm your choice.
4. If ambiguous between two categories, apply the tiebreaker_rules.
5. If NO category's boundary_test passes and no diagnostic_signals match,
   assign "{other_category_label}" — do NOT force-fit into an ill-matching category.
6. Assign exactly ONE category per idea — use the exact category_label from the list above.
7. Rate your confidence (0.0 to 1.0):
   - 0.90-1.00: boundary_test clearly matches, diagnostic_signals confirm
   - 0.70-0.89: boundary_test matches, signals partially confirm
   - 0.50-0.69: plausible fit, decided by tiebreaker
   - below 0.50: weak fit — strongly consider "{other_category_label}" instead
8. Provide a brief rationale referencing the boundary_test or diagnostic_signals that decided it.

All output (rationale) MUST be in {language}.

Provide output as valid JSON following the response schema provided.
</instructions>
"""


class CategoryAssignment(BaseModel):
    """Single idea-to-category assignment."""
    idea_id: str = Field(
        ...,
        description="The idea_id from the input"
    )
    assigned_category: str = Field(
        ...,
        description="The exact category_label of the assigned MECE category"
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
