"""
Prompts and Pydantic response models for Category Discovery V5.

Three-step MAP/REDUCE/MECE pipeline operating on category labels
within each semantic_category partition.

Terminology (V5):
  - partition = semantic_category group (6 fixed)
  - category  = coding category discovered by MAP/REDUCE/MECE
  - label     = the text string being analyzed (default: category_label field)
"""

from typing import List
from pydantic import BaseModel, Field


# =============================================================================
# PARTITION MODELS (the 6 fixed semantic_category groups)
# =============================================================================

class PartitionDescription(BaseModel):
    """Description of a semantic_category partition."""
    partition_name: str = Field(
        ...,
        description="Semantic category name (e.g., 'attribute', 'function')"
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
    """Complete set of semantic_category partitions."""
    partitions: List[PartitionDescription] = Field(
        ...,
        description="List of populated semantic_category partitions"
    )


# =============================================================================
# MAP — Candidate category extraction per batch
# =============================================================================

MAP_CATEGORIES_PROMPT = """You are a codebook designer identifying coding categories from a batch of category labels extracted from survey responses. Your goal is to produce categories that a human coder could reliably and consistently apply.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

{taxonomy_context}

<partition_context>
You are analyzing labels within the semantic category: "{partition_name}"
Partition scope: {partition_inclusion}
Partition boundary test: {partition_boundary_test}

Peer partitions (do NOT identify categories that belong to these):
{peer_partitions_list}

CRITICAL: Only identify categories that fall WITHIN "{partition_name}".
</partition_context>

{cluster_hints}

<instruction>
You are given {n_labels} category labels that belong to the partition "{partition_name}".

Your task is to identify the smallest set of CODING CATEGORIES that organize the labels WITHIN the scope of "{partition_name}".
{grouping_instruction}
A coding category is:
- A classification bucket a human coder could reliably apply
- Defined by what statements look like, not by abstract meaning
- Independent from evaluative stance (positive/negative about same subject = one category)
- Broader than wording variations, narrower than the partition itself
- MUST fall within the scope of "{partition_name}" (not peer partitions)

Each coding category should capture multiple variations such as:
- positive vs negative evaluation of the same subject
- doubt vs confirmation about the same subject
- different intensities of the same subject
</instruction>

<principles>

1. ABSTRACT ONE LEVEL UP
   If multiple labels differ only by evaluation, framing, or intensity, group them under ONE higher-level category.

2. SEPARATE SUBJECT FROM STANCE
   If labels express doubt, pride, criticism, or enthusiasm about the same subject, that is ONE category.

3. MAXIMIZE CODER AGREEMENT
   A valid category should produce the same assignment regardless of which trained coder applies it.

4. MINIMIZE FRAGMENTATION
   Prefer fewer, clearly distinct categories over many micro-categories.

5. ENSURE MECE AT CATEGORY LEVEL
   Categories should represent clearly distinct types of statements a coder can distinguish.

6. RESPECT PARTITION BOUNDARIES
   If a label seems to belong to a peer partition, do NOT create a category for it.

</principles>

<label_constraints>
Category labels MUST:
- Be in {language}
- Be 1–3 words
- Be noun phrases
- Express ONE core subject
- Avoid evaluative language (no doubt, good, bad, real, fake)
- Avoid stacking adjectives

All output (labels, descriptions, rationales) MUST be in {language}.
</label_constraints>

<batch>
Partition: {partition_name}
Batch {batch_number} of {total_batches}

Labels in this batch ({n_labels}):

{labels_list}
</batch>

<task>

1. Identify the distinct types of labels present WITHIN "{partition_name}".
2. Collapse stance variations into shared categories.
3. Return 3–8 coding categories (rarely more).
4. For each category provide:
   - category_label: short name for this coding category
   - description: what labels this category captures and how a coder would recognize them
   - recognition_cue: complete the sentence "A coder should assign a label here when it..."
   - example_labels: 2–3 labels from the batch showing variation within this category
   - rationale: 1 sentence explaining why these variations belong in one category

It is better to slightly under-split than to over-fragment.

Provide output as valid JSON following the response schema.
</task>
"""


class CandidateCategory(BaseModel):
    """A coding category identified in a batch of labels."""
    category_label: str = Field(
        ...,
        description="Short noun phrase (1-3 words) naming this coding category"
    )
    description: str = Field(
        ...,
        description="What labels this category captures and how a coder would recognize them"
    )
    recognition_cue: str = Field(
        ...,
        description=(
            "Completes the sentence 'A coder should assign a label here when it...'. "
            "Must describe observable characteristics."
        )
    )
    example_labels: List[str] = Field(
        ...,
        description="2-3 labels from the batch that exemplify this category (exact quotes)"
    )
    rationale: str = Field(
        ...,
        description="1 sentence explaining why variations belong together under this category"
    )


class MapBatchCategories(BaseModel):
    """Coding categories identified in a single batch of labels."""
    categories: List[CandidateCategory] = Field(
        ...,
        description="Coding categories identified in this batch (typically 3-8)"
    )


# =============================================================================
# REDUCE — Cross-batch consolidation
# =============================================================================

REDUCE_CATEGORIES_PROMPT = """You are a codebook designer consolidating coding categories discovered across multiple batches of category labels from survey responses.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

{taxonomy_context}

<partition_context>
You are analyzing labels within the semantic category: "{partition_name}"
Partition scope: {partition_inclusion}
Partition boundary test: {partition_boundary_test}

Peer partitions (do NOT identify categories that belong to these):
{peer_partitions_list}

CRITICAL: Only retain categories that fall WITHIN "{partition_name}".
</partition_context>

{cluster_hints}

<instruction>
You analyzed {n_batches} batches of labels from the partition "{partition_name}" and found {n_total_categories} coding categories across all batches. Because batches were processed independently, many categories are duplicates or near-duplicates. Your task is to consolidate them into a unified, non-redundant list.
{grouping_instruction}
</instruction>

<principles>
1. MERGE categories that describe the same type of label (even with slightly different wording).
2. MERGE categories that a coder would confuse — if a trained coder would hesitate
   between two categories for the same label, they must be merged.
3. AIM for the same abstraction level as the MAP step: coding categories
   a human coder can reliably distinguish, not atomic micro-categories.
4. VERIFY MUTUAL EXCLUSIVITY: after consolidation, review the remaining list.
   Could a coder confidently assign any label to exactly one category
   without hesitation? If not, merge further.
5. When merging, pick the BEST label from the input categories (do NOT invent a new label unless none of the existing labels are adequate).
6. Every input category must appear in exactly one consolidated category's merged_from list.
7. Preserve the most informative description and recognition cue when merging.
8. DISCARD any categories that actually belong to peer partitions, not to "{partition_name}".
</principles>

<categories_from_batches>
{batch_categories_list}
</categories_from_batches>

<task>
1. Read all {n_total_categories} coding categories from {n_batches} batches.
2. Identify duplicates, near-duplicates, and categories that would confuse a coder.
3. Merge them into a unified list of 3-12 coding categories for this partition.
   If you have significantly more, you are likely preserving unnecessary distinctions.
4. CRITICAL: every input category label must appear in exactly one consolidated category's merged_from list. Do not drop any.

All output (labels, descriptions) MUST be in {language}.

Provide output as valid JSON following the response schema provided.
</task>
"""


class ConsolidatedCategory(BaseModel):
    """A coding category from the reduce step (merged from multiple map batches)."""
    category_label: str = Field(
        ...,
        description="Short noun phrase (1-4 words) naming this category"
    )
    description: str = Field(
        ...,
        description="What labels this category captures and how a coder would recognize them"
    )
    merged_from: List[str] = Field(
        ...,
        description=(
            "All category labels from the map step that were merged "
            "into this category. If standalone, list contains only the original label."
        )
    )


class ReducedCategoryList(BaseModel):
    """Consolidated coding category list from the reduce step."""
    categories: List[ConsolidatedCategory] = Field(
        ...,
        description="Unified list of coding categories after cross-batch consolidation (typically 3-12)"
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
You are analyzing labels within the semantic category: "{partition_name}"
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
