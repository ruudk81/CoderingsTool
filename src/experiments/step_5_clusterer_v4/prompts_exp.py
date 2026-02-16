"""
V4 Prompts — Object-Aware Map-Reduce MECE

Three stages:
  Stage 1: Object Discovery
    - CLUSTER_OBJECT_PROMPT: Per-cluster object theme generation (from V2)
    - FRAMING_QUESTION_PROMPT: Select analytic framing question for consistent object level
    - MECE_OBJECT_CONSOLIDATION_PROMPT: Cross-cluster MECE consolidation (with framing constraint)
  Stage 3: Object-Aware Map-Reduce MECE (per object)
    - OBJECT_AWARE_MAP_THEMES_PROMPT: Coding category extraction with object context
    - OBJECT_AWARE_REDUCE_THEMES_PROMPT: Cross-batch consolidation with object context
    - OBJECT_AWARE_MECE_BOUNDARIES_PROMPT: MECE boundaries with self-verification

Design philosophy: Decision-system design, not semantic consolidation.
All prompts force the LLM to think as a codebook designer who must produce
criteria a human coder can reliably apply — not as a thesaurus builder.

This is an isolated copy for experimentation in step_5_clusterer_v4.
Changes here do NOT affect the production pipeline.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# =============================================================================
# STAGE 1 — PER-CLUSTER OBJECT THEME GENERATION (from V2)
# =============================================================================

CLUSTER_OBJECT_PROMPT = """You are a qualitative researcher identifying the shared object that a cluster of .... represents.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

<instruction>
You are given a cluster of semeantically similiar ..... . Identify the ONE shared object that unifies this cluster.
</instruction>

{taxonomy_context}

<label_constraints>
The object label MUST NOT:
- contain "en/and/und/et",
- contain slashes,
- stack multiple adjectives,
- list multiple attributes.
The label must express ONE core concept only.
</label_constraints>

<cluster_evidence>
Cluster ID: {cluster_id}
{keywords_section}
{cluster_profile_section}
</cluster_evidence>

<task>
1. Read all objects in this cluster.
2. Identify the single shared concept that unifies them.
3. Name it as a short noun phrase (1-3 words).
4. Define what kinds of nodes belong to this concept.
5. Do not introduce concepts not supported by the data.
</task>

All output MUST be in {language}.
Provide output as valid JSON following the response schema provided.
"""

# Alias for clusterer_helpers_exp.py compatibility (imports CLUSTER_THEME_PROMPT)
CLUSTER_THEME_PROMPT = CLUSTER_OBJECT_PROMPT


class ClusterThemeDescription(BaseModel):
    """LLM-generated cluster theme description (Stage 1 per-cluster output)."""
    theme: str = Field(
        ...,
        description="Short noun phrase (1-3 words) naming the shared concept"
    )
    inclusion_definition: str = Field(
        ...,
        description="1-2 sentences defining what kinds of nodes belong to this concept"
    )
    key_concepts: List[str] = Field(
        ...,
        description="3-5 representative nodes from the cluster"
    )


# =============================================================================
# STAGE 1 — FRAMING QUESTION SELECTION
# NOTE: This prompt is a FALLBACK — only used when taxonomy info from Step 3
# (ExtractionMetadata + template_lookup.py) is unavailable. The primary path
# constructs the FramingQuestionResult deterministically from the cached
# taxonomy axis data. See object_discovery.py:_framing_from_taxonomy().
# =============================================================================

FRAMING_QUESTION_PROMPT = """You are a codebook designer preparing a MECE (Mutually Exclusive, Collectively Exhaustive) classification system for survey responses.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

{taxonomy_context}

<instruction>
You are given {n_candidate_themes} candidate object themes, each derived from a cluster of survey response concepts. Before consolidating these into a MECE object set, you must determine the FRAMING QUESTION that all objects must answer.

A framing question ensures all objects sit at the same level. Without it, objects about what an entity IS get mixed with objects about what it DOES or how it is PERCEIVED — violating MECE.
</instruction>

<candidate_themes>
{candidate_themes_list}
</candidate_themes>

<task>
1. Examine the survey question and all candidate themes.
2. Determine what analytic question respondents are fundamentally answering.
3. Define a FRAMING QUESTION that ALL objects must answer. This framing question:
   - Must be a single question that partitions the response space
   - Must be at a level where each candidate theme is a plausible answer
   - Must be specific enough to enforce a consistent ontological level

Examples of framing questions:
- "What domain of [entity] is the respondent describing?"
- "What type of characteristic of [entity] is being mentioned?"
- "What aspect of [entity]'s role is being addressed?"

4. Describe how a coder should interpret this framing question.

All output MUST be in {language}.
Provide output as valid JSON following the response schema provided.
</task>
"""


class FramingQuestionResult(BaseModel):
    """Result of the framing question selection step."""
    analytic_question: str = Field(
        ...,
        description="What analytic question respondents are fundamentally answering"
    )
    framing_question: str = Field(
        ...,
        description=(
            "A single question that ALL objects must answer. "
            "Partitions the response space at a consistent level."
        )
    )
    level_description: str = Field(
        ...,
        description="How a coder should interpret this framing question when classifying"
    )


# =============================================================================
# STAGE 1 — MECE OBJECT CONSOLIDATION (revised: decision-system design)
# =============================================================================

MECE_OBJECT_CONSOLIDATION_PROMPT = """You are a codebook designer consolidating candidate objects into a MECE (Mutually Exclusive, Collectively Exhaustive) object set that a human coder can reliably apply.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

<instruction>
You are given {n_candidate_themes} candidate objects, each derived from a cluster of ontology nodes (canonical concepts from survey responses). Many of these objects overlap or describe the same domain. Merge overlapping candidates into a clean, MECE set of objects.

CRITICAL: You are designing a DECISION SYSTEM, not a thesaurus. Each object must be defined by positive, independent criteria — not by referencing what other objects cover.
</instruction>

{taxonomy_context}

<framing_constraint>
The framing question for this codebook is: "{framing_question}"
Interpretation: {level_description}

ALL objects must be valid answers to this framing question.
If a candidate object answers a DIFFERENT question, it must be either:
- Reframed to fit the framing question, OR
- Absorbed into an existing object that does fit
</framing_constraint>

<principles>
1. MERGE liberally: if two candidate objects describe the same concept domain, merge them.
2. Each final object must be a single, coherent concept — not a catch-all.
3. Every cluster must be assigned to exactly one object.
4. ALL objects must answer the framing question at the same level — do NOT mix levels.
5. Aim for parsimony: fewer well-defined objects are better than many overlapping ones.
6. Define each object using POSITIVE, INDEPENDENT criteria:
   - What it IS (not what it is NOT)
   - A boundary test a coder can apply without knowing other objects
   - Observable signals in the response text
</principles>

<candidate_topics>
{candidate_themes_list}
</candidate_topics>

<task>
1. Read all candidate objects and their definitions.
2. Verify each candidate answers the framing question: "{framing_question}"
   - If not, reframe or absorb it.
3. Identify groups of candidates that describe the same or overlapping concepts.
4. For each group, create ONE merged object with a clear label.
5. For candidates that are already distinct, keep them as standalone objects.
6. For each final object, provide:
   - inclusion_definition: What kinds of statements belong (observable criteria, not vague descriptions)
   - boundary_test: A yes/no question a human coder could ask to determine if a statement belongs to this object
     (e.g., "Does this statement describe a financial product or service offered by the entity?")
   - diagnostic_signals: 3-5 concrete words, phrases, or framings that, if present in a response, indicate this object
7. Ensure the final set is MECE: every cluster accounted for, no overlaps between objects.

All output MUST be in {language}.
Provide output as valid JSON following the response schema provided.
</task>
"""


class MECEObjectDescription(BaseModel):
    """A single MECE object from Stage 1 consolidation."""
    topic_label: str = Field(
        ...,
        description="Short noun phrase (1-3 words) naming this MECE object"
    )
    inclusion_definition: str = Field(
        ...,
        description=(
            "What kinds of statements belong to this object. "
            "Must use observable criteria, not vague descriptions."
        )
    )
    boundary_test: str = Field(
        ...,
        description=(
            "A yes/no question a human coder could ask to determine if a statement "
            "belongs to this object. Must be self-contained — no references to other objects."
        )
    )
    diagnostic_signals: List[str] = Field(
        ...,
        description=(
            "3-5 concrete words, phrases, or framings that, if present in a response, "
            "indicate this object"
        )
    )
    source_cluster_ids: List[int] = Field(
        ...,
        description="List of cluster IDs that were merged into this object"
    )
    merge_rationale: str = Field(
        ...,
        description="Why these clusters belong together, or 'standalone' if single cluster"
    )


class MECEObjectSet(BaseModel):
    """Complete MECE object set from Stage 1 consolidation."""
    topics: List[MECEObjectDescription] = Field(
        ...,
        description="List of MECE objects covering all clusters"
    )


# =============================================================================
# STAGE 3 — OBJECT-AWARE MAP: Coding category extraction per batch
# =============================================================================

OBJECT_AWARE_MAP_THEMES_PROMPT = """You are a codebook designer identifying coding categories from a batch of survey response ideas. Your goal is to produce categories that a human coder could reliably and consistently apply.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

{taxonomy_context}

<object_context>
You are analyzing ideas within the MECE object: "{object_label}"
Object scope: {object_inclusion}
Object boundary test: {object_boundary_test}

Peer objects (do NOT identify categories that belong to these):
{peer_objects_list}

CRITICAL: Only identify categories that fall WITHIN "{object_label}".
</object_context>

<instruction>
You are given {n_ideas} ideas that belong to the object "{object_label}".

Your task is to identify the smallest set of CODING CATEGORIES that organize the ideas WITHIN the scope of "{object_label}".
{grouping_instruction}
A coding category is:
- A classification bucket a human coder could reliably apply
- Defined by what statements look like, not by abstract meaning
- Independent from evaluative stance (positive/negative about same subject = one category)
- Broader than wording variations, narrower than the object itself
- MUST fall within the scope of "{object_label}" (not peer objects)

Each coding category should capture multiple variations such as:
- positive vs negative evaluation of the same subject
- doubt vs confirmation about the same subject
- different intensities of the same subject
</instruction>

<principles>

1. ABSTRACT ONE LEVEL UP
   If multiple ideas differ only by evaluation, framing, or intensity, group them under ONE higher-level category.

2. SEPARATE SUBJECT FROM STANCE
   If people express doubt, pride, criticism, or enthusiasm about the same subject, that is ONE category.

3. MAXIMIZE CODER AGREEMENT
   A valid category should produce the same assignment regardless of which trained coder applies it.

4. MINIMIZE FRAGMENTATION
   Prefer fewer, clearly distinct categories over many micro-categories.

5. ENSURE MECE AT CATEGORY LEVEL
   Categories should represent clearly distinct types of statements a coder can distinguish.

6. RESPECT OBJECT BOUNDARIES
   If an idea seems to belong to a peer object, do NOT create a category for it.

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
Object: {object_label}
Batch {batch_number} of {total_batches}

Ideas in this batch ({n_ideas}):

{ideas_list}
</batch>

<task>

1. Identify the distinct types of statements present WITHIN "{object_label}".
2. Collapse stance variations into shared categories.
3. Return 3–8 coding categories (rarely more).
4. For each category provide:
   - theme_label: short name for this coding category
   - description: what statements this category captures and how a coder would recognize them
   - recognition_cue: complete the sentence "A coder should assign a statement here when it..."
   - example_ideas: 2–3 quotes from the batch showing variation within this category
   - rationale: 1 sentence explaining why these variations belong in one category

It is better to slightly under-split than to over-fragment.

Provide output as valid JSON following the response schema.
</task>
"""


class COCTheme(BaseModel):
    """A coding category identified in a batch of ideas."""
    theme_label: str = Field(
        ...,
        description="Short noun phrase (1-3 words) naming this coding category"
    )
    description: str = Field(
        ...,
        description=(
            "What statements this category captures and how a coder would recognize them"
        )
    )
    recognition_cue: str = Field(
        ...,
        description=(
            "Completes the sentence 'A coder should assign a statement here when it...'. "
            "Must describe observable characteristics of the statement."
        )
    )
    example_ideas: List[str] = Field(
        ...,
        description="2-3 ideas from the batch that exemplify this category (exact quotes)"
    )
    rationale: str = Field(
        ...,
        description="1 sentence explaining why variations belong together under this category"
    )


class MapBatchCOCs(BaseModel):
    """Coding categories identified in a single batch of ideas."""
    themes: List[COCTheme] = Field(
        ...,
        description="Coding categories identified in this batch (typically 3-8)"
    )


# =============================================================================
# STAGE 3 — OBJECT-AWARE REDUCE: Cross-batch consolidation
# =============================================================================

OBJECT_AWARE_REDUCE_THEMES_PROMPT = """You are a codebook designer consolidating coding categories discovered across multiple batches of survey response ideas.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

{taxonomy_context}

<object_context>
You are analyzing ideas within the MECE object: "{object_label}"
Object scope: {object_inclusion}
Object boundary test: {object_boundary_test}

Peer objects (do NOT identify categories that belong to these):
{peer_objects_list}

CRITICAL: Only retain categories that fall WITHIN "{object_label}".
</object_context>

<instruction>
You analyzed {n_batches} batches of ideas from the object "{object_label}" and found {n_total_themes} coding categories across all batches. Because batches were processed independently, many categories are duplicates or near-duplicates. Your task is to consolidate them into a unified, non-redundant list.
{grouping_instruction}
</instruction>

<principles>
1. MERGE categories that describe the same type of statement (even with slightly different wording).
2. MERGE categories that a coder would confuse — if a trained coder would hesitate
   between two categories for the same statement, they must be merged.
3. AIM for the same abstraction level as the MAP step: coding categories
   a human coder can reliably distinguish, not atomic micro-categories.
4. VERIFY MUTUAL EXCLUSIVITY: after consolidation, review the remaining list.
   Could a coder confidently assign any statement to exactly one category
   without hesitation? If not, merge further.
5. When merging, pick the BEST label from the input categories (do NOT invent a new label unless none of the existing labels are adequate).
6. Every input category must appear in exactly one consolidated category's merged_from list.
7. Preserve the most informative description and recognition cue when merging.
8. DISCARD any categories that actually belong to peer objects, not to "{object_label}".
</principles>

<categories_from_batches>
{batch_themes_list}
</categories_from_batches>

<task>
1. Read all {n_total_themes} coding categories from {n_batches} batches.
2. Identify duplicates, near-duplicates, and categories that would confuse a coder.
3. Merge them into a unified list of 3-12 coding categories for this object.
   If you have significantly more, you are likely preserving unnecessary distinctions.
4. CRITICAL: every input category label must appear in exactly one consolidated category's merged_from list. Do not drop any.

All output (labels, descriptions) MUST be in {language}.

Provide output as valid JSON following the response schema provided.
</task>
"""


class ConsolidatedTheme(BaseModel):
    """A coding category from the reduce step (merged from multiple map batches)."""
    theme_label: str = Field(
        ...,
        description="Short noun phrase (1-4 words) naming this category"
    )
    description: str = Field(
        ...,
        description="What statements this category captures and how a coder would recognize them"
    )
    merged_from: List[str] = Field(
        ...,
        description=(
            "All category labels from the map step that were merged "
            "into this category. If standalone, list contains only the original label."
        )
    )


class ReducedThemeList(BaseModel):
    """Consolidated coding category list from the reduce step."""
    themes: List[ConsolidatedTheme] = Field(
        ...,
        description="Unified list of coding categories after cross-batch consolidation (typically 3-12)"
    )


# =============================================================================
# STAGE 3 — OBJECT-AWARE MECE: Apply boundaries with self-verification
# =============================================================================

OBJECT_AWARE_MECE_BOUNDARIES_PROMPT = """You are a codebook designer applying MECE (Mutually Exclusive, Collectively Exhaustive) constraints to a list of coding categories within a single object of survey responses.

Your output must be a DECISION SYSTEM — not a semantic description. Each category must be defined by criteria a human coder can independently apply.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

{taxonomy_context}

<object_context>
You are analyzing ideas within the MECE object: "{object_label}"
Object scope: {object_inclusion}
Object boundary test: {object_boundary_test}

Peer objects (do NOT identify categories that belong to these):
{peer_objects_list}

CRITICAL: Only retain topics that fall WITHIN "{object_label}".
</object_context>

<instruction>
You are given {n_themes} consolidated coding categories from the object "{object_label}" containing {n_ideas} survey response ideas. Your task is to ensure this list is MECE:
- Mutually Exclusive: each statement should clearly belong to exactly one topic
- Collectively Exhaustive: every statement in this object should fit one topic

You may merge categories that still overlap, split categories that are too broad, or adjust boundaries.
{grouping_instruction}
CRITICAL DESIGN RULE: Define each topic using POSITIVE, INDEPENDENT criteria.
- DO NOT define a topic by what it is NOT or by referencing other topics.
- Each topic's boundary_test must work WITHOUT knowing what other topics exist.
- Use observable characteristics of the statement, not abstract semantic labels.
</instruction>

<principles>
1. The result must be a clean, non-overlapping set of topics WITHIN "{object_label}".
2. Each topic needs:
   - A BOUNDARY TEST: a yes/no question a coder can ask independently
   - DIAGNOSTIC SIGNALS: concrete words/phrases that trigger assignment
   - TIEBREAKER RULES: for each neighboring topic, a concrete rule for ambiguous cases
3. Topics are VALENCE-NEUTRAL: name what people talk about, not how they feel.
4. Actively merge related topics into broader containers.
   Fewer well-defined topics are far better than many overlapping ones.
5. AIM for 3-10 final topics per object. If you have significantly more,
   merge topics that a coder would confuse.
6. Every category from the input must be accounted for in the output (merged, kept, or split).
7. If a category actually belongs to a peer object, exclude it from the final set.
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

<consolidated_categories>
Object: {object_label}
Total ideas in object: {n_ideas}

{themes_list}
</consolidated_categories>

<task>
1. Review the {n_themes} categories and their descriptions.
2. Check for overlaps: merge categories where a coder would hesitate between them.
3. Check for gaps: is there any common statement type within "{object_label}" not covered?
4. For each final topic, define:
   - topic_label: short name
   - inclusion_definition: what statements belong (observable criteria)
   - boundary_test: a yes/no question a coder asks to determine membership
     (MUST be self-contained — no references to other topics)
   - diagnostic_signals: 3-5 concrete words/phrases/framings that trigger assignment
   - key_expressions: 3-5 representative expressions from the object
   - tiebreaker_rules: for each similar/adjacent topic, a rule:
     "If ambiguous with [topic X], assign here when [observable condition]"
5. VERIFY your topics are MECE by testing each pair of adjacent/similar topics:
   - Construct one AMBIGUOUS example (a statement that could plausibly fit either topic)
   - Show which topic gets it and WHY, using only your boundary_test and diagnostic_signals
   - If you cannot decide using your criteria alone, your topics are NOT MECE — merge or redefine them before finalizing.

Provide output as valid JSON following the response schema provided.
</task>
"""


class MECETopic(BaseModel):
    """A MECE topic with independent boundary criteria (Stage 3 output)."""
    topic_label: str = Field(
        ...,
        description="Short noun phrase (1-4 words) naming this topic"
    )
    inclusion_definition: str = Field(
        ...,
        description=(
            "What kinds of statements belong to this topic. "
            "Must use observable criteria, not vague semantic descriptions."
        )
    )
    boundary_test: str = Field(
        ...,
        description=(
            "A yes/no question a human coder asks to determine if a statement belongs here. "
            "Must be self-contained — no references to other topics."
        )
    )
    diagnostic_signals: List[str] = Field(
        ...,
        description=(
            "3-5 concrete words, phrases, or framings that, if present, "
            "indicate this topic"
        )
    )
    key_expressions: List[str] = Field(
        ...,
        description="3-5 representative expressions from the object that exemplify this topic"
    )
    tiebreaker_rules: List[str] = Field(
        ...,
        description=(
            "For each similar/adjacent topic, a rule: "
            "'If ambiguous with [topic X], assign here when [observable condition]'"
        )
    )


class MECEVerification(BaseModel):
    """Self-verification test for one pair of adjacent MECE topics."""
    topic_a: str = Field(
        ...,
        description="First topic in the pair"
    )
    topic_b: str = Field(
        ...,
        description="Second topic in the pair"
    )
    ambiguous_example: str = Field(
        ...,
        description="A constructed statement that could plausibly fit either topic"
    )
    assigned_to: str = Field(
        ...,
        description="Which topic the ambiguous example is assigned to"
    )
    reasoning: str = Field(
        ...,
        description="Why this assignment is correct, using only boundary_test and diagnostic_signals"
    )


class ClusterMECETopicSet(BaseModel):
    """Complete MECE topic set for a single object, with self-verification."""
    topics: List[MECETopic] = Field(
        ...,
        description="MECE topics for this object"
    )
    mece_verifications: List[MECEVerification] = Field(
        ...,
        description=(
            "Self-verification tests: one per pair of adjacent/similar topics. "
            "Proves the boundary criteria actually work."
        )
    )
