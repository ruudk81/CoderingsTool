"""
Theme generation and MECE consolidation prompts - V2

Phase B: Per-cluster theme/object generation with inclusion definitions
Phase C: MECE consolidation across all candidate themes/objects

Supports two discovery modes:
- Topics: from clustering ideas (survey responses)
- Objects: from clustering nodes (canonical ontology concepts)

This is an isolated copy for experimentation in step_5_clusterer_v2.
Changes here do NOT affect the production pipeline.
"""

from typing import List
from pydantic import BaseModel, Field


# =============================================================================
# PHASE B: PER-CLUSTER THEME GENERATION
# =============================================================================

CLUSTER_THEME_PROMPT = """You are a qualitative researcher identifying the topic that a cluster of survey responses represents.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

<instruction>
Identify the TOPIC this cluster represents. A topic is the subject matter respondents are talking about — not a code, not a checkbox answer, not an evaluation.
</instruction>

{taxonomy_context}

<theory_of_topics>
A valid topic must represent the Central Organizing Concept (COC):

The COC is the SINGLE underlying subject or domain that best explains why these responses cluster together.
It is NOT a summary, list, span, or combination of ideas.

If multiple related ideas appear in the cluster, abstract to ONE higher-order subject that unifies them.

A topic is VALENCE-NEUTRAL: it names what people talk about, not how they feel about it.
</theory_of_topics>

<label_constraints>
The topic label MUST NOT:
- contain "en/and/und/et",
- contain slashes (e.g. duurzaam/groen),
- stack multiple adjectives,
- list multiple attributes.
The label must express ONE core subject only.
</label_constraints>

<cluster_evidence>
Cluster ID: {cluster_id}
Number of {sample_type}: {num_ideas}

<representative_{samples_tag}>
These {sample_type} are representative of the cluster:
{ideas_list}
</representative_{samples_tag}>
{keywords_section}
{cluster_profile_section}
</cluster_evidence>

<task>
1. Review the representative {sample_type} to identify what subject they share.
2. Use the statistical keywords to sharpen what makes this cluster distinct.
3. Name the ONE core subject (the topic).
4. Define what belongs to this topic: what kinds of statements fall under it?
5. Do not introduce concepts not supported by the data.
</task>

<output_format>
Provide your analysis in {language}:
- theme: Short noun phrase (1-4 words) naming the topic
- inclusion_definition: 1-2 sentences defining what kinds of statements belong to this topic
- key_concepts: 3-5 concrete concepts grounded in data
</output_format>
"""


class ClusterThemeDescription(BaseModel):
    """LLM-generated cluster theme description (Phase B structured output)."""
    theme: str = Field(
        ...,
        description="Short noun phrase (1-4 words) naming the topic this cluster represents"
    )
    inclusion_definition: str = Field(
        ...,
        description="1-2 sentences defining what kinds of statements belong to this topic"
    )
    key_concepts: List[str] = Field(
        ...,
        description="3-5 concrete concepts grounded in data (from keywords or samples)"
    )


# =============================================================================
# PHASE C: MECE TOPIC CONSOLIDATION
# =============================================================================

MECE_CONSOLIDATION_PROMPT = """You are a qualitative researcher consolidating candidate topics into a MECE (Mutually Exclusive, Collectively Exhaustive) topic set.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

<instruction>
You are given {n_candidate_themes} candidate topics, each derived from a separate cluster of survey responses. Many of these topics overlap or describe the same domain from different angles. Your task is to merge overlapping candidates into a clean, MECE set of topics.
</instruction>

{taxonomy_context}

<principles>
1. MERGE liberally: if two candidate topics describe the same domain, merge them into one.
2. The clusters are your evidence — they represent what respondents actually talk about. Trust them.
3. Each final topic must be a single, coherent domain — not a catch-all.
4. Every cluster must be assigned to exactly one topic.
5. Topic labels should be at a consistent level of abstraction.
6. Topics must be VALENCE-NEUTRAL: they name what people talk about, not how they feel.
7. Aim for parsimony: fewer well-defined topics are better than many overlapping ones.
</principles>

<candidate_topics>
{candidate_themes_list}
</candidate_topics>

<task>
1. Read all candidate topics and their definitions.
2. Identify groups of candidates that describe the same or overlapping domains.
3. For each group, create ONE merged topic with a clear label.
4. For candidates that are already distinct, keep them as standalone topics.
5. Write an inclusion definition (what belongs) and exclusion definition (what does NOT belong, referencing other topics) for each final topic.
6. Ensure the final set is MECE: every cluster accounted for, no overlaps between topics.
</task>

<output_format>
Provide your analysis in {language}.
For each MECE topic, provide:
- topic_label: Short noun phrase (1-4 words)
- inclusion_definition: What kinds of statements belong to this topic
- exclusion_definition: What does NOT belong (reference specific peer topics)
- source_cluster_ids: List of cluster IDs that were merged into this topic
- merge_rationale: Brief explanation of why these clusters belong together (or "standalone" if not merged)
</output_format>
"""


class MECETopic(BaseModel):
    """A single MECE topic derived from one or more clusters."""
    topic_label: str = Field(
        ...,
        description="Short noun phrase (1-4 words) naming this MECE topic"
    )
    inclusion_definition: str = Field(
        ...,
        description="1-2 sentences defining what kinds of statements belong to this topic"
    )
    exclusion_definition: str = Field(
        ...,
        description="What does NOT belong to this topic, referencing specific peer topics"
    )
    source_cluster_ids: List[int] = Field(
        ...,
        description="List of cluster IDs that were merged into this topic"
    )
    merge_rationale: str = Field(
        ...,
        description="Why these clusters belong together, or 'standalone' if single cluster"
    )


class MECETopicSet(BaseModel):
    """Complete MECE topic set from consolidation (Phase C structured output)."""
    topics: List[MECETopic] = Field(
        ...,
        description="List of MECE topics covering all clusters"
    )


# =============================================================================
# PHASE B (OBJECTS): PER-CLUSTER OBJECT IDENTIFICATION
# =============================================================================

CLUSTER_OBJECT_PROMPT = """You are a qualitative researcher identifying the shared class that a cluster of semantically similar classes represents.


<instruction>
You are given a cluster with classes that share semtantic similarity. Classes are structural groupings of responses to a survey question. Identify the ONE shared class that unifies this cluster.
</instruction>


<label_constraints>
The class label MUST NOT:
- contain "en/and/und/et",
- contain slashes,
- stack multiple adjectives,
- list multiple attributes.
The label must express ONE core concept only.
</label_constraints>

<cluster_evidence>
Cluster ID: {cluster_id}
These classes are in this cluster:
{ideas_list}
</cluster_evidence>

<task>
1. Read all the classes in this cluster.
2. Identify the single shared class that unifies them.
3. Name it as a short noun phrase (1-3 words).
5. Do not introduce a new class not supported by the data.
</task>

<output_format>
Provide your analysis in {language}:
- Class: Short noun phrase (1-3 words) naming the shared class
- inclusion_definition: 1-2 sentences defining what kinds of nodes belong
- key_concepts: 3-5 representative nodes from the cluster
</output_format>
"""


# =============================================================================
# PHASE C (OBJECTS): MECE OBJECT CONSOLIDATION
# =============================================================================

MECE_OBJECT_CONSOLIDATION_PROMPT = """You are a qualitative researcher consolidating candidate objects into a MECE (Mutually Exclusive, Collectively Exhaustive) object set.


<instruction>
You are given {n_candidate_themes} candidate objects, each derived from a cluster of ontology nodes (canonical concepts from survey responses). Many of these objects overlap or describe the same domain. Merge overlapping candidates into a clean, MECE set of objects.
</instruction>

{taxonomy_context}

<principles>
1. MERGE liberally: if two candidate objects describe the same concept domain, merge them.
2. Each final object must be a single, coherent concept — not a catch-all.
3. Every cluster must be assigned to exactly one object.
4. Object labels should be at a consistent level of abstraction.
5. Aim for parsimony: fewer well-defined objects are better than many overlapping ones.
</principles>

<candidate_topics>
{candidate_themes_list}
</candidate_topics>

<task>
1. Read all candidate objects and their definitions.
2. Identify groups of candidates that describe the same or overlapping concepts.
3. For each group, create ONE merged object with a clear label.
4. For candidates that are already distinct, keep them as standalone objects.
5. Write an inclusion definition (what belongs) and exclusion definition (what does NOT belong, referencing other objects) for each final object.
6. Ensure the final set is MECE: every cluster accounted for, no overlaps between objects.
</task>

<output_format>
Provide your analysis in {language}.
For each MECE object, provide:
- topic_label: Short noun phrase (1-3 words)
- inclusion_definition: What kinds of nodes belong to this object
- exclusion_definition: What does NOT belong (reference specific peer objects)
- source_cluster_ids: List of cluster IDs merged into this object
- merge_rationale: Brief explanation of why these clusters belong together (or "standalone")
</output_format>
"""


# =============================================================================
# DIRECT MECE TOPIC EXTRACTION (per-cluster, all-probability sampling)
# =============================================================================

CLUSTER_MECE_TOPIC_EXTRACTION_PROMPT = """You are a qualitative researcher extracting the distinct MECE topics expressed within a single cluster of survey responses.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

<instruction>
You are given a stratified sample of ideas from one cluster, drawn from ALL probability levels (core members through peripheral members). Your task is to identify ALL distinct, mutually exclusive topics expressed across this cluster.

First, identify the SINGLE semantic theme that explains why these ideas cluster together. Then decompose that theme into 1-5 specific MECE topics.
</instruction>

{taxonomy_context}

<theory_of_topics>
The semantic theme is the Central Organizing Concept (COC) — the single underlying domain that explains why these responses cluster together.

Within that domain, MECE topics are the distinct sub-themes respondents express:
- Mutually Exclusive: each idea should map to exactly one topic
- Collectively Exhaustive: every idea in the cluster (not just the sample) should fit one topic
- Topics should be as META as possible — name the abstract semantic category, not specific instances
- Topics are VALENCE-NEUTRAL: they name what people talk about, not how they feel
</theory_of_topics>

<label_constraints>
Topic labels MUST NOT:
- contain "en/and/und/et",
- contain slashes (e.g. duurzaam/groen),
- stack multiple adjectives,
- list multiple attributes.
Each label must express ONE core subject only.
</label_constraints>

<cluster_evidence>
Cluster ID: {cluster_id}
Total ideas in cluster: {total_cluster_size}
Sample size: {sample_size} (stratified across probability bands)
{keywords_section}
<stratified_sample>
{stratified_sample_text}
</stratified_sample>
</cluster_evidence>

<task>
1. Read ALL sampled ideas across all probability bands to understand the full breadth of this cluster.
2. Identify the single semantic theme that unifies all members.
3. Within that theme, identify 1-5 distinct MECE topics.
4. For peripheral/low-probability members: consider whether they represent a separate topic or simply a weaker expression of an existing one.
5. Each topic must be grounded in the data — do not invent topics not supported by the sample.
6. Name topics as abstract semantic categories, not specific instances.
</task>

<output_format>
Provide your analysis in {language}:
- semantic_theme: The overarching theme shared by ALL cluster members (1-3 word noun phrase)
- topics: 1-5 MECE topics, each with:
  - topic_label: Short noun phrase (1-4 words)
  - inclusion_definition: 1-2 sentences defining what belongs
  - key_expressions: 3-5 representative expressions from the sample

Provide output as valid JSON following the response schema provided.
</output_format>
"""


class ExtractedMECETopic(BaseModel):
    """A single MECE topic discovered within a cluster."""
    topic_label: str = Field(
        ...,
        description="Short noun phrase (1-4 words) naming this topic"
    )
    inclusion_definition: str = Field(
        ...,
        description="1-2 sentences defining what kinds of statements belong to this topic"
    )
    key_expressions: List[str] = Field(
        ...,
        description="3-5 representative expressions from the cluster that exemplify this topic"
    )


class ClusterMECETopics(BaseModel):
    """Complete MECE topic set extracted from a single cluster."""
    semantic_theme: str = Field(
        ...,
        description="The overarching semantic theme shared by ALL members of this cluster (1-3 word noun phrase)"
    )
    topics: List[ExtractedMECETopic] = Field(
        ...,
        description="1-5 MECE topics discovered within this cluster"
    )


# =============================================================================
# OBJECT DISCOVERY: group topics into MECE objects (aspects of the entity)
# =============================================================================

TOPIC_TO_OBJECT_PROMPT = """You are a qualitative researcher discovering the MECE objects (aspects/attributes) of {entity} that respondents address.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

<concept_distinction>
There are two levels of analysis:
- OBJECT = what aspect or attribute of {entity} is being addressed (e.g., "duurzaamheidsbeleid", "dienstverlening", "imago")
- TOPIC = what specific thing people mention about that object (e.g., "groen beleggen", "groene uitstraling", "klantenservice")

You are given {n_topics} topics. Your task: discover {min_objects}-{max_objects} MECE objects by answering, for each group of related topics: "What aspect of {entity} are these topics about?"
</concept_distinction>

<principles>
1. An object names the ASPECT of {entity} that the topics address.
   Ask yourself: "If I had to label the column in a table where these topics would be rows — what would the column header be?"
   - GOOD: "duurzaamheidsbeleid", "visuele identiteit", "financiële producten", "imago"
   - BAD: "associaties", "percepties", "kenmerken", "diverse onderwerpen" (too abstract / meta)
2. The object label must be the most specific common denominator — the narrowest term that covers all its topics.
3. Every topic must be assigned to exactly one object.
4. Objects must be at a consistent level of abstraction (all about aspects of {entity}).
5. Prefer fewer, well-defined objects over many small ones.
6. Topics from different clusters that address the same aspect of {entity} MUST be grouped into the same object.
</principles>

<label_constraints>
Object labels MUST NOT:
- contain "en/and/und/et",
- contain slashes,
- be generic meta-terms like "overig", "divers", "associaties", "kenmerken".
Each label must name a specific aspect of {entity}.
</label_constraints>

<topics>
{topics_list}
</topics>

<task>
1. Read all topics and their definitions.
2. For each topic, ask: "What aspect of {entity} is this about?"
3. Group topics that address the same aspect into one object.
4. Name each object as the specific aspect of {entity} it represents.
5. List all member topics under their object.
6. CRITICAL: every single topic from the input list must appear in exactly one object. Do not drop any.
</task>

<output_format>
Provide your analysis in {language}.
For each object:
- object_label: Short noun phrase (1-4 words) naming the aspect of {entity}
- object_definition: 1-2 sentences defining what aspect of {entity} this covers
- member_topics: List of ALL topic labels that belong to this object (exact match from input)
- source_cluster_ids: List of all cluster IDs represented in this object

Provide output as valid JSON following the response schema provided.
</output_format>
"""


class MECEObject(BaseModel):
    """An object = an aspect/attribute of the entity that topics address."""
    object_label: str = Field(
        ...,
        description="Short noun phrase (1-4 words) naming the aspect of the entity"
    )
    object_definition: str = Field(
        ...,
        description="1-2 sentences defining what aspect of the entity this covers"
    )
    member_topics: List[str] = Field(
        ...,
        description="List of topic labels that belong to this object (exact match from input)"
    )
    source_cluster_ids: List[int] = Field(
        ...,
        description="List of all cluster IDs represented in this object"
    )


class MECEObjectSet(BaseModel):
    """Complete set of MECE objects discovered from topics."""
    objects: List[MECEObject] = Field(
        ...,
        description="MECE objects covering all topics"
    )


# =============================================================================
# MAP-REDUCE TOPIC CONSOLIDATION
# =============================================================================

MAP_CONSOLIDATION_PROMPT = """You are a qualitative researcher consolidating overlapping topics into a clean, non-redundant set.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

<instruction>
You are given {n_input_topics} candidate topics from survey response analysis. Many of these topics are duplicates or describe the same domain from slightly different angles. Your task is to MERGE overlapping topics into a consolidated set.
</instruction>

{taxonomy_context}

<critical_constraint>
**TOPIC LABEL SELECTION RULE:**
When you merge multiple topics, you MUST pick one of the INPUT topic labels as the consolidated label.
DO NOT generate a new label. DO NOT abstract to a higher level.
DO NOT create compound labels like "X and Y" or "X/Y".

Pick the BEST REPRESENTATIVE label from the merged topics — the most clear, specific, and canonical phrasing.

WRONG: Creating "duurzaamheid en milieu" when merging "duurzaamheid" + "milieubewustzijn"
RIGHT: Pick "duurzaamheid" (most canonical) or "milieubewustzijn" (most specific)
</critical_constraint>

<principles>
1. MERGE liberally: if two topics describe the same domain or are clear duplicates, merge them.
2. Preserve granularity: if topics address genuinely distinct sub-domains, keep them separate.
3. Use the definitions to judge overlap — topics with overlapping definitions should merge.
4. Every input topic must appear in exactly one consolidated topic's merged_from list.
5. Aim for roughly 30-50% reduction (e.g., 20 inputs → 10-14 outputs).
</principles>

<input_topics>
{topics_list}
</input_topics>

<task>
1. Read all {n_input_topics} topics and their definitions.
2. Identify groups of topics that are duplicates or describe overlapping domains.
3. For each group: pick the BEST input label as the consolidated label (do NOT create a new label).
4. For topics that are already distinct, keep them standalone.
5. CRITICAL: every single input topic must appear in exactly one consolidated topic's merged_from list. Do not drop any.

Provide your output as valid JSON following the response schema provided.
</task>
"""


REDUCE_CONSOLIDATION_PROMPT = """You are a qualitative researcher performing final consolidation of topics discovered across multiple analysis batches.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

<instruction>
You are given {n_map_outputs} consolidated topics from previous analysis rounds. These topics were discovered independently across different batches, so there may be duplicates or overlaps between batches. Your task is to produce the FINAL consolidated topic set.
</instruction>

{taxonomy_context}

<critical_constraint>
**TOPIC LABEL SELECTION RULE:**
When you merge multiple topics, you MUST pick one of the INPUT topic labels as the final label.
DO NOT generate a new label. DO NOT abstract to a higher level.
DO NOT create compound labels like "X and Y" or "X/Y".

Pick the BEST REPRESENTATIVE label — the most clear, specific, and canonical phrasing.
</critical_constraint>

<principles>
1. MERGE aggressively: topics from different batches describing the same domain MUST be merged.
2. Target 15-20 final topics.
3. Every input topic must appear in exactly one final topic's merged_from list.
4. When merging, the merged_from list should contain ALL original topic labels (flatten any nested merges).
5. Preserve the most informative definitions when merging.
</principles>

<consolidated_topics_from_map_phase>
{map_outputs_list}
</consolidated_topics_from_map_phase>

<task>
1. Read all {n_map_outputs} topics from the map phase.
2. Identify cross-batch duplicates and overlaps.
3. Merge aggressively to reach 15-20 final topics.
4. For each final topic: pick the BEST input label (do NOT create a new label).
5. CRITICAL: every single input topic must appear in exactly one final topic's merged_from list. Do not drop any.

Provide your output as valid JSON following the response schema provided.
</task>
"""


class ConsolidatedTopic(BaseModel):
    """A consolidated topic formed by merging 1+ input topics."""
    topic_label: str = Field(
        ...,
        description=(
            "MUST be EXACTLY one of the input topic labels (copy-paste). "
            "Do NOT generate a new or abstract label."
        )
    )
    merged_from: List[str] = Field(
        ...,
        description=(
            "ALL input topic labels merged into this topic, including "
            "the topic_label itself. If standalone, list contains only one label."
        )
    )
    inclusion_definition: str = Field(
        ...,
        description="Consolidated definition of what belongs to this topic (1-2 sentences)"
    )
    merge_rationale: str = Field(
        ...,
        description="Why these topics were merged, or 'standalone' if not merged"
    )
    source_cluster_ids: List[int] = Field(
        ...,
        description="Combined list of all source cluster IDs from merged topics"
    )


class MapBatchResult(BaseModel):
    """Result of consolidating topics within a single map batch."""
    consolidated_topics: List[ConsolidatedTopic]


class ReduceResult(BaseModel):
    """Final result from reduce phase."""
    consolidated_topics: List[ConsolidatedTopic]
