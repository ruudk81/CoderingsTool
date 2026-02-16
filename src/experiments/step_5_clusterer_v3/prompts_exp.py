"""
Map-Reduce MECE prompts — V3

Three-step per-cluster pipeline:
  MAP:    Find ALL atomic themes in a batch of ideas
  REDUCE: Consolidate atomic themes across batches into unified list
  MECE:   Apply mutually exclusive / collectively exhaustive boundaries

This is an isolated copy for experimentation in step_5_clusterer_v3.
Changes here do NOT affect the production pipeline.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# =============================================================================
# V2 COMPATIBILITY STUBS
# =============================================================================
# These are needed because clusterer_helpers_exp.py (copied from V2 unchanged)
# imports them at module level for the ThemeGenerator class.
# V3 does not use ThemeGenerator, but the import must succeed.

CLUSTER_THEME_PROMPT = ""  # Unused in V3


class ClusterThemeDescription(BaseModel):
    """Unused in V3 — stub for clusterer_helpers_exp.py import compatibility."""
    theme: str = ""
    inclusion_definition: str = ""
    key_concepts: List[str] = []


# =============================================================================
# MAP STEP: Atomic theme extraction per batch
# =============================================================================

MAP_THEMES_PROMPT = """You are a senior qualitative researcher identifying reusable Central Organizing Concepts (COCs) from a batch of survey response ideas.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

{taxonomy_context}

<instruction>
You are given {n_ideas} ideas from ONE cluster of survey responses.

Your task is NOT to extract every semantic nuance.
Your task is to identify the smallest set of reusable Central Organizing Concepts (COCs) that organize the ideas at a structural level.

A COC is:
- A stable conceptual container
- Reusable across different datasets
- Independent from evaluative stance
- Broader than wording variations
- Narrower than generic meta-themes like “Perception” or “Brand”

Each COC should be able to contain multiple variations such as:
- positive vs negative evaluation
- doubt vs confirmation
- positioning vs reputation
- striving vs achievement
</instruction>

<principles>

1. ABSTRACT ONE LEVEL UP  
   If multiple ideas differ only by evaluation, framing, or intensity, group them under ONE higher-level concept.

2. SEPARATE SUBJECT FROM STANCE  
   If people express doubt, pride, criticism, or enthusiasm about the same subject, that is ONE COC.

3. MAXIMIZE REUSABILITY  
   A valid COC should make sense in a completely different dataset about another brand or organization.

4. MINIMIZE FRAGMENTATION  
   Prefer fewer, structurally distinct concepts over many micro-themes.

5. ENSURE MECE AT CONCEPT LEVEL  
   Concepts should represent clearly distinct domains of meaning.

</principles>

<label_constraints>
Theme labels MUST:
- Be in {language}
- Be 1–3 words
- Be noun phrases
- Express ONE core subject
- Avoid evaluative language (no doubt, good, bad, real, fake)
- Avoid stacking adjectives

All output (labels, descriptions, rationales) MUST be in {language}.
</label_constraints>

<batch>
Cluster ID: {cluster_id}
Batch {batch_number} of {total_batches}

Ideas in this batch ({n_ideas}):

{ideas_list}
</batch>

<task>

1. Identify the underlying conceptual domains represented.
2. Collapse stance variations into shared containers.
3. Return 3–8 reusable COCs (rarely more).
4. For each COC provide:
   - theme_label
   - description (what conceptual territory it covers)
   - example_ideas (2–3 quotes showing variation inside the container)
   - rationale (1 sentence explaining why variations belong together)

It is better to slightly under-split than to over-fragment.

Provide output as valid JSON following the response schema.
</task>
"""

MAP_THEMES_PROMPT_OLD = """You are a qualitative researcher identifying ALL distinct atomic themes expressed in a batch of survey response ideas.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

{taxonomy_context}

<instruction>
You are given a batch of {n_ideas} ideas from ONE cluster of survey responses. 
Your task is to identify ALL unique Central Organizing Concepts (COCs) in this batch.
A COC is an atomic meta-theme that captures a shared pattern of meaning accross multiple responses in light of the taxonomy and survey question.
</instruction>

<principles>
1. Be EXHAUSTIVE: identify every distinct theme, even if only 1-2 ideas express it.
2. Be ATOMIC: each theme should describe ONE subject, not a combination.
3. Be SPECIFIC: name the concrete subject, not a meta-category.
4. Themes are VALENCE-NEUTRAL: name what people talk about, not how they feel.
5. Do not invent themes not supported by the ideas in this batch.
</principles>

<label_constraints>
Theme labels MUST NOT:
- contain "en/and/und/et",
- contain slashes (e.g. duurzaam/groen),
- stack multiple adjectives,
- list multiple attributes.
Each label must express ONE core subject only.
</label_constraints>

<batch>
Cluster ID: {cluster_id}
Batch {batch_number} of {total_batches}
Ideas in this batch ({n_ideas}):

{ideas_list}
</batch>

<task>
1. Read all {n_ideas} ideas carefully.
2. Identify every distinct atomic theme expressed.
3. For each theme, provide a label, description, and 2-3 example ideas (exact quotes).
4. It is better to find too many themes than to miss one.

Provide output as valid JSON following the response schema provided.
</task>
"""


class COCTheme(BaseModel):
    """A Central Organizing Concept identified in a batch of ideas."""
    theme_label: str = Field(
        ...,
        description="Short noun phrase (1-3 words) naming this COC"
    )
    description: str = Field(
        ...,
        description="1-2 sentences describing what conceptual territory this COC covers"
    )
    example_ideas: List[str] = Field(
        ...,
        description="2-3 ideas from the batch that exemplify this COC (exact quotes)"
    )
    rationale: str = Field(
        ...,
        description="1 sentence explaining why variations belong together under this COC"
    )


class MapBatchCOCs(BaseModel):
    """Central Organizing Concepts identified in a single batch of ideas."""
    themes: List[COCTheme] = Field(
        ...,
        description="Central Organizing Concepts identified in this batch (typically 3-8)"
    )


# =============================================================================
# REDUCE STEP: Cross-batch theme consolidation
# =============================================================================

REDUCE_THEMES_PROMPT = """You are a qualitative researcher consolidating Central Organizing Concepts (COCs) discovered across multiple batches of survey response ideas.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

{taxonomy_context}

<instruction>
You analyzed {n_batches} batches of ideas from the same cluster and found {n_total_themes} COCs across all batches. Because batches were processed independently, many COCs are duplicates or near-duplicates. Your task is to consolidate them into a unified, non-redundant list.
</instruction>

<principles>
1. MERGE themes that describe the same subject (even with slightly different wording).
2. MERGE themes that cover the same conceptual territory from different angles
   (e.g., "innovativeness" and "future orientation" → keep separate only if
   they represent truly independent domains of meaning).
3. AIM for the same abstraction level as the MAP step: reusable Central
   Organizing Concepts, not atomic micro-themes.
4. VERIFY MUTUAL EXCLUSIVITY: after consolidation, review the remaining list.
   If two COCs could plausibly claim the same idea, they overlap and must be
   merged. The test: could a coder confidently assign any idea to exactly one
   COC without hesitation? If not, merge further.
5. When merging, pick the BEST label from the input themes (do NOT invent a new label unless none of the existing labels are adequate).
6. Every input theme must appear in exactly one consolidated theme's merged_from list.
7. Preserve the most informative description when merging.
</principles>

<cocs_from_batches>
{batch_themes_list}
</cocs_from_batches>

<task>
1. Read all {n_total_themes} COCs from {n_batches} batches.
2. Identify duplicates, near-duplicates, and conceptually overlapping COCs across batches.
3. Merge them into a unified list of 3-12 COCs for the cluster.
   If you have significantly more, you are likely preserving unnecessary distinctions.
4. CRITICAL: every input theme label must appear in exactly one consolidated theme's merged_from list. Do not drop any.

All output (labels, descriptions) MUST be in {language}.

Provide output as valid JSON following the response schema provided.
</task>
"""


class ConsolidatedTheme(BaseModel):
    """A COC from the reduce step (merged from multiple map batches)."""
    theme_label: str = Field(
        ...,
        description="Short noun phrase (1-4 words) naming this theme"
    )
    description: str = Field(
        ...,
        description="1-2 sentences describing what this theme covers"
    )
    merged_from: List[str] = Field(
        ...,
        description=(
            "All atomic theme labels from the map step that were merged "
            "into this theme. If standalone, list contains only the original label."
        )
    )


class ReducedThemeList(BaseModel):
    """Consolidated COC list from the reduce step."""
    themes: List[ConsolidatedTheme] = Field(
        ...,
        description="Unified list of COCs after cross-batch consolidation (typically 3-12)"
    )


# =============================================================================
# MECE STEP: Apply mutual exclusivity and exhaustiveness boundaries
# =============================================================================

MECE_BOUNDARIES_PROMPT = """You are a qualitative researcher applying MECE (Mutually Exclusive, Collectively Exhaustive) constraints to a list of themes within a single cluster of survey responses.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

{taxonomy_context}

<instruction>
You are given {n_themes} consolidated themes from a single cluster of {n_ideas} survey response ideas. Your task is to ensure this list is MECE:
- Mutually Exclusive: each idea should clearly belong to exactly one topic
- Collectively Exhaustive: every idea in the cluster should fit one topic

You may merge themes that still overlap, split themes that are too broad, or adjust boundaries.
</instruction>

<principles>
1. The result must be a clean, non-overlapping set of topics.
2. Each topic needs clear INCLUSION boundaries (what belongs) and EXCLUSION boundaries (what does NOT belong, referencing peer topics).
3. Topics are VALENCE-NEUTRAL: name what people talk about, not how they feel.
4. Actively merge related topics into broader containers.
   Fewer well-defined topics are far better than many overlapping ones.
5. AIM for 3-10 final topics per cluster. If you have significantly more,
   merge topics that share the same conceptual domain.
6. Every theme from the input must be accounted for in the output (merged, kept, or split).
</principles>

<label_constraints>
Topic labels MUST:
- Be in {language}
Topic labels MUST NOT:
- contain "en/and/und/et",
- contain slashes,
- stack multiple adjectives,
- list multiple attributes.
Each label must express ONE core subject only.

All output (labels, definitions, key_expressions) MUST be in {language}.
</label_constraints>

<consolidated_themes>
Cluster ID: {cluster_id}
Total ideas in cluster: {n_ideas}

{themes_list}
</consolidated_themes>

<task>
1. Review the {n_themes} themes and their descriptions.
2. Check for overlaps: merge themes with unclear boundaries.
3. Check for gaps: is there any common idea type not covered?
4. For each final topic, write clear inclusion and exclusion definitions.
5. Include 3-5 representative expressions (key_expressions) from the cluster for each topic.

Provide output as valid JSON following the response schema provided.
</task>
"""


class MECETopic(BaseModel):
    """A MECE topic with inclusion/exclusion boundaries."""
    topic_label: str = Field(
        ...,
        description="Short noun phrase (1-4 words) naming this topic"
    )
    inclusion_definition: str = Field(
        ...,
        description="1-2 sentences defining what kinds of statements belong to this topic"
    )
    exclusion_definition: str = Field(
        ...,
        description=(
            "What does NOT belong to this topic, referencing specific "
            "peer topics by name"
        )
    )
    key_expressions: List[str] = Field(
        ...,
        description="3-5 representative expressions from the cluster that exemplify this topic"
    )


class ClusterMECETopicSet(BaseModel):
    """Complete MECE topic set for a single cluster."""
    topics: List[MECETopic] = Field(
        ...,
        description="MECE topics for this cluster"
    )
