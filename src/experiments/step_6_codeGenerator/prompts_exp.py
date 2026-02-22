"""
Experimental Prompts for Step 6: Codebook Generation

This file contains the prompts used by codeGenerator.py for the 4-prompt chain.
Modify these prompts to experiment with different codebook generation approaches.

Original source: src/prompts.py (STEP 6: CODEBOOK GENERATION section)

Response models (Pydantic) are co-located with their prompts following the
migrate-output-schema pattern - instructor uses Field(description=...) to
communicate schema to the LLM.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator

# =============================================================================
# STEP 6: CODEBOOK GENERATION - 4 PROMPT CHAIN
# =============================================================================

# -----------------------------------------------------------------------------
# 1. CLUSTER_SUMMARY_PROMPT
# -----------------------------------------------------------------------------

CLUSTER_SUMMARY_PROMPT = """
You are a qualitative researcher tasked with refining pre-extracted MECE (Mutually Exclusive, Collectively Exhaustive) topics from survey responses. 
Your goal is to produce two outputs:

1. **Cluster-level Central Organizing Concepts (COCs)** — analytic syntheses that explain what unifies the cluster as a whole
2. **Atomic, grounded themes** — operational coding categories that function as stand-alone codes within a specified taxonomy axis

Your work is interpretive but strictly data-bound: you may organize and name patterns, but you must not introduce concepts absent from the provided topics and key expressions.

## Survey Context

First, here is the context for the survey data you'll be analyzing:

<survey_context>
- Survey question asked: "{survey_question}"
- Language of responses: {language}
- Domain (subject area): {domain}
- Topic (specific focus): {topic}
- Perspective (whose viewpoint): {perspective}
- Intent (purpose of responses): {intent}
- Entity (what/who is being discussed): {entity}
</survey_context>

## Taxonomy 

All themes you identify must adhere to these taxonomy rules

<taxonomy_rules>
- Themes must describe only the dimension defined by {facet_name}.
- Labels naming {facet_valid_labels} are VALID.
- Labels describing {facet_invalid_labels} are INVALID.
</taxonomy_rules>


## Key Definitions

**Central Organizing Concept (COC):** A short analytic statement (not a code label) that captures what ties the cluster together as a whole. COCs are interpretive and describe shared patterns across multiple topics. They are NOT used as coding labels.

**Atomic theme:** A single, separable idea that could function as a stand-alone code in a codebook. It must have exactly one semantic nucleus and cannot bundle multiple distinct ideas together.

**MECE:** Mutually Exclusive, Collectively Exhaustive — themes should not overlap in meaning, and together should cover all relevant concepts in the data.

## Cluster Data to Analyze

Here is the pre-extracted cluster data containing topics and key expressions from survey responses:

<cluster_data>
<cluster_id>
{cluster_id}
</cluster_id>

<cluster_text>
{cluster_text}
</cluster_text>
</cluster_data>

Each topic in the cluster includes:
- A topic label
- An inclusion definition
- Key expressions from original responses

Use these as your ONLY evidence base. Do not introduce concepts not present in this data.

## Cluster Analysis (COCs)

Identify 1–2 Central Organizing Concepts that explain what unifies this cluster of responses.

COCs should:
- Be analytic and interpretive, not literal labels from the data
- Describe a shared pattern across multiple topics in the cluster
- NOT be used as coding labels themselves
- Capture the conceptual thread that ties the cluster together

## Theme Identification Rules

Each theme you identify must satisfy ALL of these criteria:

1. **Atomicity:** Express exactly ONE COC with one semantic nucleus
2. **Boundary compliance:** Fall strictly within the specified taxonomy axis
3. **Grounding:** Be directly supported by the provided key expressions
4. **Specificity:** Avoid repeating the perspective, domain, topic, or entity from the survey context in the label
5. **Operationality:** Be precise enough to function as a real code in a codebook

## Theme Label Constraints

Theme labels must:
- Be a noun phrase of 1–4 words (maximum 10 words if absolutely necessary)
- Contain one semantic head (modifiers are allowed)
- Avoid "and/or," slashes, commas, or lists
- Name only the relevant concept within the taxonomy axis
- Be precise enough to distinguish from other themes

## Theme Definition Constraints

Each theme definition must:
- Be 30 words or fewer
- Describe observable assignment cues (what respondents say or describe)
- Avoid discussing causes, motives, conditions, or outcomes
- Avoid repeating the perspective, domain, topic, or entity from the survey context
- Be concrete enough to guide consistent coding decisions

## Assignment Examples Requirements

For each theme, you must provide:

**Inclusion examples (2–3):** Observable cues for what counts as this theme
- Start with action verbs (e.g., "Describes…," "Mentions…," "Reports…")
- Be concrete and traceable to the provided key expressions
- Show what should be coded to this theme

**Exclusion examples (1–2):** Boundary cases that should NOT be coded here
- Start with action verbs
- Clarify what might be confused with this theme but doesn't belong
- Help distinguish from similar themes

**Near neighbor:** The closest potentially confusable theme
- Identify the label of the most similar theme
- Provide one sentence explaining how to tell them apart
- If no meaningful neighbor exists, write "Unknown"

## Required Analysis Steps

Before providing your final output, document your reasoning in the analysis field:

1. State the 1–2 cluster-level COCs you identified
2. Explain whether you kept, split, or merged topics — and why
3. Note any themes you discarded due to weak grounding in the data
4. Justify why you have one theme versus multiple themes

## Output Requirements

1. Write all output in the same language as the survey responses
2. Follow the response schema exactly
3. For each valid atomic theme, include:
   - theme_id (sequential starting at 1)
   - theme_label (1-4 word noun phrase)
   - theme_clarification (≤30 words)
   - abstraction_level ("L1-topic" or "L2-action-mechanism")
   - assignment_examples (inclusion, exclusion, near_neighbor)
4. In the analysis field, document your COCs and refinement decisions
5. Output valid JSON that matches the response schema structure

Begin your analysis now. Think carefully through the cluster data, identify the unifying COCs, refine the topics into atomic themes, and provide your output in valid JSON format.
"""

# -----------------------------------------------------------------------------
# CLUSTER_SUMMARY_PROMPT_MECE: variant that takes pre-extracted MECE topics
# instead of raw sampled ideas. Same atomicity/taxonomy/label/output rules.
# -----------------------------------------------------------------------------

CLUSTER_SUMMARY_PROMPT_MECE = """
You are a qualitative researcher tasked with refining pre-extracted MECE (Mutually Exclusive, Collectively Exhaustive) topics from survey responses. 
Your goal is to produce two outputs:

1. **Cluster-level Central Organizing Concepts (COCs)** — analytic syntheses that explain what unifies the cluster as a whole
2. **Atomic, grounded themes** — operational coding categories that function as stand-alone codes within a specified taxonomy axis

Your work is interpretive but strictly data-bound: you may organize and name patterns, but you must not introduce concepts absent from the provided topics and key expressions.

## Survey Context

First, here is the context for the survey data you'll be analyzing:

<survey_context>
- Survey question asked: "{survey_question}"
- Language of responses: {language}
- Domain (subject area): {domain}
- Topic (specific focus): {topic}
- Perspective (whose viewpoint): {perspective}
- Intent (purpose of responses): {intent}
- Entity (what/who is being discussed): {entity}
</survey_context>

## Taxonomy 

All themes you identify must adhere to these taxonomy rules

<taxonomy_rules>
- Themes must describe only the dimension defined by {facet_name}.
- Labels naming {facet_valid_labels} are VALID.
- Labels describing {facet_invalid_labels} are INVALID.
</taxonomy_rules>


## Key Definitions

**Central Organizing Concept (COC):** A short analytic statement (not a code label) that captures what ties the cluster together as a whole. COCs are interpretive and describe shared patterns across multiple topics. They are NOT used as coding labels.

**Atomic theme:** A single, separable idea that could function as a stand-alone code in a codebook. It must have exactly one semantic nucleus and cannot bundle multiple distinct ideas together.

**MECE:** Mutually Exclusive, Collectively Exhaustive — themes should not overlap in meaning, and together should cover all relevant concepts in the data.

## Cluster Data to Analyze

Here is the pre-extracted cluster data containing topics and key expressions from survey responses:

<cluster_data>
<cluster_id>
{cluster_id}
</cluster_id>

<cluster_text>
{cluster_text}
</cluster_text>
</cluster_data>

Each topic in the cluster includes:
- A topic label
- An inclusion definition
- Key expressions from original responses

Use these as your ONLY evidence base. Do not introduce concepts not present in this data.

## Cluster Analysis (COCs)

Identify 1–2 Central Organizing Concepts that explain what unifies this cluster of responses.

COCs should:
- Be analytic and interpretive, not literal labels from the data
- Describe a shared pattern across multiple topics in the cluster
- NOT be used as coding labels themselves
- Capture the conceptual thread that ties the cluster together

## Theme Identification Rules

Each theme you identify must satisfy ALL of these criteria:

1. **Atomicity:** Express exactly ONE COC with one semantic nucleus
2. **Boundary compliance:** Fall strictly within the specified taxonomy axis
3. **Grounding:** Be directly supported by the provided key expressions
4. **Specificity:** Avoid repeating the perspective, domain, topic, or entity from the survey context in the label
5. **Operationality:** Be precise enough to function as a real code in a codebook

## Theme Label Constraints

Theme labels must:
- Be a noun phrase of 1–4 words (maximum 10 words if absolutely necessary)
- Contain one semantic head (modifiers are allowed)
- Avoid "and/or," slashes, commas, or lists
- Name only the relevant concept within the taxonomy axis
- Be precise enough to distinguish from other themes

## Theme Definition Constraints

Each theme definition must:
- Be 30 words or fewer
- Describe observable assignment cues (what respondents say or describe)
- Avoid discussing causes, motives, conditions, or outcomes
- Avoid repeating the perspective, domain, topic, or entity from the survey context
- Be concrete enough to guide consistent coding decisions


## Assignment Examples Requirements

For each theme, you must provide:

**Inclusion examples (2–3):** Observable cues for what counts as this theme
- Start with action verbs (e.g., "Describes…," "Mentions…," "Reports…")
- Be concrete and traceable to the provided key expressions
- Show what should be coded to this theme

**Exclusion examples (1–2):** Boundary cases that should NOT be coded here
- Start with action verbs
- Clarify what might be confused with this theme but doesn't belong
- Help distinguish from similar themes

**Near neighbor:** The closest potentially confusable theme
- Identify the label of the most similar theme
- Provide one sentence explaining how to tell them apart
- If no meaningful neighbor exists, write "Unknown"

## Required Analysis Steps

Before providing your final output, document your reasoning in the analysis field:

1. State the 1–2 cluster-level COCs you identified
2. Explain whether you kept, split, or merged topics — and why
3. Note any themes you discarded due to weak grounding in the data
4. Justify why you have one theme versus multiple themes

## Output Requirements

1. Write all output in the same language as the survey responses
2. Follow the response schema exactly
3. For each valid atomic theme, include:
   - theme_id (sequential starting at 1)
   - theme_label (1-4 word noun phrase)
   - theme_clarification (≤30 words)
   - abstraction_level ("L1-topic" or "L2-action-mechanism")
   - assignment_examples (inclusion, exclusion, near_neighbor)
4. In the analysis field, document your COCs and refinement decisions
5. Output valid JSON that matches the response schema structure

Begin your analysis now. Think carefully through the cluster data, identify the unifying COCs, refine the topics into atomic themes, and provide your output in valid JSON format.
"""

# -----------------------------------------------------------------------------
# 1c. CATEGORY_SUMMARY_PROMPT: variant for step_5_categories MECE categories
#     with full category metadata + confidence-band sampled ideas
# -----------------------------------------------------------------------------

CATEGORY_SUMMARY_PROMPT = """
You are a qualitative researcher tasked with refining a pre-defined MECE (Mutually Exclusive, Collectively Exhaustive) coding category into atomic, grounded themes for a codebook.

You are given:
1. **A structured MECE category definition** — with inclusion criteria, boundary test, diagnostic signals, key expressions, and tiebreaker rules
2. **Representative survey response ideas** — actual ideas assigned to this category, sampled across confidence levels

Your goal is to produce:
1. **Category-level Central Organizing Concepts (COCs)** — analytic syntheses that explain what unifies the ideas within this category
2. **Atomic, grounded themes** — operational coding categories that function as stand-alone codes within a specified taxonomy axis

Your work is interpretive but strictly data-bound: the category definition anchors the scope, and the assigned ideas provide evidence. You must not introduce concepts absent from either source.

## Survey Context

First, here is the context for the survey data you'll be analyzing:

<survey_context>
- Survey question asked: "{survey_question}"
- Language of responses: {language}
- Domain (subject area): {domain}
- Topic (specific focus): {topic}
- Perspective (whose viewpoint): {perspective}
- Intent (purpose of responses): {intent}
- Entity (what/who is being discussed): {entity}
</survey_context>

## Taxonomy

All themes you identify must adhere to these taxonomy rules

<taxonomy_rules>
- Themes must describe only the dimension defined by {facet_name}.
- Labels naming {facet_valid_labels} are VALID.
- Labels describing {facet_invalid_labels} are INVALID.
</taxonomy_rules>


## Key Definitions

**Central Organizing Concept (COC):** A short analytic statement (not a code label) that captures what ties the category's ideas together. COCs are interpretive and describe shared patterns. They are NOT used as coding labels.

**Atomic theme:** A single, separable idea that could function as a stand-alone code in a codebook. It must have exactly one semantic nucleus and cannot bundle multiple distinct ideas together.

**MECE:** Mutually Exclusive, Collectively Exhaustive — themes should not overlap in meaning, and together should cover all relevant concepts in the data.

## Category Data to Analyze

<category_data>
<category_id>
{cluster_id}
</category_id>

<category_definition>
{cluster_text}
</category_definition>
</category_data>

The category data above contains two sections:

**Category metadata** — the structured MECE category definition including:
- Category label and inclusion definition (what belongs here)
- Boundary test (a yes/no question for membership)
- Diagnostic signals (trigger words/phrases)
- Key expressions (representative labels from the data)
- Tiebreaker rules (how to resolve ambiguous cases with other categories)

**Assigned ideas** — actual survey response ideas assigned to this category, grouped by assignment confidence:
- Inner members: high confidence assignments (0.6-0.8)
- Border members: medium confidence (0.4-0.6)
- Fringe members: low confidence (0.0-0.4)

Use the category metadata to understand the intended scope and boundaries. Use the assigned ideas as your evidence base for theme identification.

## Category Analysis (COCs)

Identify 1-2 Central Organizing Concepts that explain what unifies the ideas within this category.

COCs should:
- Be analytic and interpretive, not literal labels from the data
- Describe a shared pattern across the assigned ideas
- NOT be used as coding labels themselves
- Capture the conceptual thread that ties the category together
- Be consistent with the category's inclusion definition and boundary test

## Theme Identification Rules

Each theme you identify must satisfy ALL of these criteria:

1. **Atomicity:** Express exactly ONE concept with one semantic nucleus
2. **Boundary compliance:** Fall strictly within the specified taxonomy axis
3. **Grounding:** Be directly supported by the assigned ideas
4. **Specificity:** Avoid repeating the perspective, domain, topic, or entity from the survey context in the label
5. **Operationality:** Be precise enough to function as a real code in a codebook
6. **Category coherence:** Be consistent with the category's inclusion definition and boundary test

## Theme Label Constraints

Theme labels must:
- Be a noun phrase of 1-4 words (maximum 10 words if absolutely necessary)
- Contain one semantic head (modifiers are allowed)
- Avoid "and/or," slashes, commas, or lists
- Name only the relevant concept within the taxonomy axis
- Be precise enough to distinguish from other themes

## Theme Definition Constraints

Each theme definition must:
- Be 30 words or fewer
- Describe observable assignment cues (what respondents say or describe)
- Avoid discussing causes, motives, conditions, or outcomes
- Avoid repeating the perspective, domain, topic, or entity from the survey context
- Be concrete enough to guide consistent coding decisions

## Assignment Examples Requirements

For each theme, you must provide:

**Inclusion examples (2-3):** Observable cues for what counts as this theme
- Start with action verbs (e.g., "Describes...," "Mentions...," "Reports...")
- Be concrete and traceable to the assigned ideas
- Show what should be coded to this theme

**Exclusion examples (1-2):** Boundary cases that should NOT be coded here
- Start with action verbs
- Clarify what might be confused with this theme but doesn't belong
- Help distinguish from similar themes

**Near neighbor:** The closest potentially confusable theme
- Identify the label of the most similar theme
- Provide one sentence explaining how to tell them apart
- If no meaningful neighbor exists, write "Unknown"

## Required Analysis Steps

Before providing your final output, document your reasoning in the analysis field:

1. State the 1-2 category-level COCs you identified
2. Explain how the category's boundary test and diagnostic signals informed your theme identification
3. Note any sub-themes you identified, split, or merged — and why
4. Justify why you have one theme versus multiple themes

## Output Requirements

1. Write all output in the same language as the survey responses
2. Follow the response schema exactly
3. For each valid atomic theme, include:
   - theme_id (sequential starting at 1)
   - theme_label (1-4 word noun phrase)
   - theme_clarification (<=30 words)
   - abstraction_level ("L1-topic" or "L2-action-mechanism")
   - assignment_examples (inclusion, exclusion, near_neighbor)
4. In the analysis field, document your COCs and refinement decisions
5. Output valid JSON that matches the response schema structure

Begin your analysis now. Think carefully through the category definition and assigned ideas, identify the unifying COCs, derive atomic themes, and provide your output in valid JSON format.
"""

class NearNeighbor(BaseModel):
    label: str = Field(
        ...,
        min_length=1,
        description="Label of closest potentially-confusable theme, or 'Unknown' if none exists",
        examples=["Product Quality", "Unknown"]
    )
    tell_apart_rule: str = Field(
        default="",
        min_length=0,
        description="One sentence distinguishing this theme from the neighbor",
        examples=["This theme focuses on speed, not accuracy"]
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)


class AssignmentExamples(BaseModel):
    inclusion: List[str] = Field(
        ...,
        min_length=1,
        description="2-3 observable cues starting with verbs for what to include",
        examples=[["Mentions waiting time explicitly", "Describes delay in service"]]
    )
    exclusion: List[str] = Field(
        ...,
        min_length=1,
        description="1-2 boundary cases for what must NOT be included",
        examples=[["General complaints without time reference", "Mentions speed positively"]]
    )
    near_neighbor: NearNeighbor = Field(
        ...,
        description="Closest confusable theme and how to distinguish",
        examples=[{"label": "Product Quality", "tell_apart_rule": "This theme focuses on speed, not accuracy"}]
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ClusterThemeItem(BaseModel):
    theme_id: int = Field(
        ...,
        description="Sequential theme identifier starting at 1",
        examples=[1, 2]
    )
    theme_label: str = Field(
        ...,
        max_length=100,
        description="1-3 word atomic noun phrase theme label",
        examples=["Waiting Time", "Product Quality", "Staff Friendliness"]
    )
    theme_clarification: str = Field(
        ...,
        max_length=300,
        description="<=30-word grounded definition describing what belongs in this theme",
        examples=["Responses mentioning the duration of waiting for service or products"]
    )
    abstraction_level: str = Field(
        ...,
        description="Abstraction level indicator",
        examples=["L2 -action-mechanism theme", "L1 -topic theme"]
    )
    assignment_examples: AssignmentExamples = Field(
        ...,
        description="Concrete inclusion/exclusion examples for coding",
        examples=[{
            "inclusion": ["Mentions waiting time explicitly", "Describes delay in service"],
            "exclusion": ["General complaints without time reference"],
            "near_neighbor": {"label": "Service Quality", "tell_apart_rule": "This theme focuses on duration, not quality"}
        }]
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator('theme_label')
    @classmethod
    def validate_label_length(cls, v):
        word_count = len(v.split())
        if word_count > 10:
            raise ValueError(f"theme_label must be ≤10 words, got {word_count}")
        return v


class ClusterSummaryOutput(BaseModel):
    cluster_id: str = Field(
        ...,
        description="The cluster identifier exactly as provided",
        examples=["3", "5", "12"]
    )
    analysis: str = Field(
        ...,
        description="Document analysis: state COCs identified/retained, justify single vs multiple themes",
        examples=["Identified 2 COCs: 'speed' and 'accuracy'. Retained both as distinct atomic concepts."]
    )
    extracted_themes: List[ClusterThemeItem] = Field(
        ...,
        description="Final theme entries for valid atomic themes"
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)


# -----------------------------------------------------------------------------
# 2. CCODING_DECISION_PROMPT
# -----------------------------------------------------------------------------

CODING_DECISION_PROMPT = """
You are a qualitative research assistant responsible for maintaining a parsimonious and structured codebook for thematic analysis following Braun & Clarke (2006) methodology.
Your task is to classify a newly identified theme within the {facet_name} facet and decide whether to USE an existing code, MODIFY an existing code, or CREATE a new code.
You must ensure the codebook remains MECE (Mutually Exclusive, Collectively Exhaustive) by strictly adhering to the specified taxonomy structure.

---

CODEBOOK PARAMETERS

<language>
{language}
</language>

<context>
- Domain: {domain}
- Topic: {topic}
- Perspective: {perspective}
- Entity: {entity}
Survey Question: "{survey_question}"
</context>

<taxonomy_parameters>
Facet: {facet_name}
Facet description: {facet_description}
</taxonomy_parameters>

<new_theme>
New Theme to Classify:
- name: "{theme_name}"
- description: "{theme_description}"
- what's included:
    {inclusion}
</new_theme>

<existing_codes>
Existing Codes:
{code_text}
</existing_codes>

---

FACET RULES

Theme labels must describe only the dimension defined by {facet_name}.
- Labels naming {facet_valid_labels} are VALID.
- Labels describing {facet_invalid_labels} are INVALID.

---

DECISION OPTIONS

You must choose one of the following actions:

- **USE** - An existing code fully captures the new theme's meaning; use it as-is without modification
- **MODIFY_HORIZONTAL** - An existing code needs broader definition and inclusion rules to cover the new theme, but remains at the same abstraction level within the {facet_name} facet
- **MODIFY_VERTICAL** - The existing code and new theme belong to the same conceptual family but differ in abstraction level; create or reference a parent code for both
- **CREATE** - Add a new code because the theme represents a distinct concept not covered by existing codes

---

ANALYSIS FRAMEWORK

Follow these steps systematically:

**STEP 0: Initial Matching**
- Review the new theme and all existing codes
- Identify the best matching existing code(s) based on core meaning and practical relevance in light of the research question, taxonomy axis, and primary coding dimension

**STEP 1: Conceptual Family Test**
Ask: Do the new theme and the best matching existing code belong to the same conceptual family, given the research question and the {facet_name} facet?
- If the new theme and best matching existing code share the same core concept and have the same practical relevance -> SAME FAMILY
- Otherwise -> DIFFERENT FAMILY

**STEP 2: Abstraction Level Test**
Ask: Are the new theme and the best matching existing code at the same abstraction level on the taxonomy axis/coding dimension?
- If the level of generality/specificity is similar -> SAME ABSTRACTION LEVEL
- Otherwise -> DIFFERENT ABSTRACTION LEVEL

**STEP 3: Decision Logic**
Apply the following decision rules:

- If the new theme is fully covered in meaning and scope by an existing code -> USE existing code.
- If the new theme is not fully covered by an existing code:
  - If it belongs to the same code family and is at the same abstraction level -> MODIFY_HORIZONTAL
      - Broaden the existing code's definition and inclusion rules to incorporate the new expression, ensuring the original core meaning remains intact.
  - If it belongs to the same code family but at a different abstraction level -> MODIFY_VERTICAL
      - Introduce or reference a higher-level parent code, treating the existing code and new theme as related sub-codes.
  - If it belongs to a different code family -> CREATE a new code for the distinct concept.

**STEP 4: Multi-Concept Theme Check**
If the new theme contains multiple distinct concepts (e.g., "salt reduction AND mild spices"):
- Identify which concept(s) semantically match the existing code
- If only ONE concept matches and MODIFY would require changing the existing code's core meaning to accommodate the other: Decision = **CREATE**
- A MODIFY should never replace an existing code's central meaning with a different concept
- Preserve the existing code unchanged and create a new code for the theme

---

ASSIGNMENT EXAMPLE UPDATE RULES

Based on your decision, update the assignment examples as follows:

If decision is **USE**:
- Preserve all original assignment_examples unchanged

If decision is **MODIFY_VERTICAL** or **MODIFY_HORIZONTAL**:
- inclusion: Combine original inclusion examples + new expressions from the theme
- exclusion: Combine original exclusion examples + new boundary clarifications if needed
- near_neighbor: Update the label if boundaries shifted due to modification
- tell_apart_rule: Update if the distinction from neighbor changed

If decision is **CREATE**:
- Use assignment_examples from the new theme as-is

---

LABEL CONSTRAINTS

When creating theme labels, follow these strict rules:
- Use a noun phrase of 1-10 words.
- Exactly one semantic nucleus; modifiers allowed.
- No coordination (and/or), no lists, no multi-concept bundles.
- Name only the core concept present.
- Labels naming {facet_valid_labels} are VALID.
- Labels describing {facet_invalid_labels} are INVALID.
- DO NOT repeat {perspective}, {domain}, {topic}, or {entity} in the label.

---

DEFINITION CONSTRAINTS

When creating theme definitions, follow these strict rules:
- Use 30 words or fewer.
- Describe what belongs in this code (not why it happens).
- Use observable assignment cues (behaviors, expressions, practices).
- Avoid causes, conditions, interpretations, or outcomes.
- DO NOT repeat {perspective}, {domain}, {topic}, or {entity} in the definition.

---

REQUIRED ANALYSIS STEPS

Before providing your final answer, use <scratchpad> tags to work through your analysis systematically:

1. Identify the top candidate code(s) based on semantic similarity to the theme's core concept.
2. Note any cosine similarity scores for top candidates (if provided).
3. Apply the Conceptual Family Test (STEP 1): Do they share the same core concept?
4. Apply the Abstraction Level Test (STEP 2): Same specificity level within the {facet_name} facet?
5. Apply the Decision Logic (STEP 3): USE, MODIFY_HORIZONTAL, MODIFY_VERTICAL, or CREATE.
6. Check for multi-concept themes (STEP 4): Does the theme contain multiple distinct concepts?
7. Verify label compliance: Ensure the facet rules are satisfied (VALID/INVALID criteria).
8. Determine your final decision with justification referencing conceptual family and abstraction level analysis.
9. Plan what updates are needed to assignment examples based on your decision.

After completing your analysis in the scratchpad, provide your final answer as valid JSON following the response schema provided.

**Output Requirements:**
- Keep field names in English; write values in {language}
- Include conceptual family and abstraction level comparison explicitly in justification
- Ensure all updates maintain MECE principles and code atomicity
- Reference any cosine similarity scores (if provided) in your justification
"""


class MatchedCandidate(BaseModel):
    code: str = Field(
        ...,
        description="Exact candidate code name from existing codebook",
        examples=["Product Quality", "Waiting Time"]
    )
    definition: Optional[str] = Field(
        default=None,
        description="Code definition in light of the survey question",
        examples=["References to product quality concerns", "Mentions of waiting duration"]
    )
    definition_source: Literal["provided", "inferred"] = Field(
        default="inferred",
        description="Whether definition was provided in codebook or inferred"
    )
    assignment_examples: Optional[AssignmentExamples] = Field(
        default=None,
        description="Assignment examples from existing code",
        examples=[{
            "inclusion": ["References to product quality", "Mentions satisfaction with quality"],
            "exclusion": ["Price complaints without quality mention"],
            "near_neighbor": {"label": "Product Value", "tell_apart_rule": "Quality focuses on product attributes, value on price-quality ratio"}
        }]
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ModifyParameters(BaseModel):
    modify_instruction: Literal["vertical_broaden_same_level", "hierarchical_parent_diff_level", "none"] = Field(
        ...,
        description="Type of modification to apply",
        examples=["vertical_broaden_same_level", "hierarchical_parent_diff_level", "none"]
    )
    conceptual_family: Literal["same", "different", "none"] = Field(
        ...,
        description="Whether new theme belongs to same conceptual family as matched code",
        examples=["same", "different"]
    )
    abstraction_level: Literal["same", "different", "none"] = Field(
        ...,
        description="Whether theme is at same abstraction level as matched code",
        examples=["same", "different"]
    )
    abstraction_level_action: Literal["keep", "broaden_to_parent", "none"] = Field(
        ...,
        description="Action to take regarding abstraction level",
        examples=["keep", "broaden_to_parent", "none"]
    )
    inclusion_update: Optional[str] = Field(
        default=None,
        description="Concrete additions to inclusion rules if modifying",
        examples=["Add expressions about delivery speed", None]
    )
    exclusion_update: Optional[str] = Field(
        default=None,
        description="Concrete boundary clarifications for exclusion rules",
        examples=["Exclude mentions of product defects", None]
    )
    parent_theme_label: Optional[str] = Field(
        default=None,
        description="Suggested parent label for vertical modification",
        examples=["Service Experience", None]
    )
    near_neighbor_label_update: Optional[str] = Field(
        default=None,
        description="Updated neighbor label if boundaries changed",
        examples=["Response Time", None]
    )
    tell_apart_rule_update: Optional[str] = Field(
        default=None,
        description="Updated tell-apart rule if distinction changed",
        examples=["This theme focuses on duration, neighbor on frequency", None]
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)


class CodingDecision(BaseModel):
    theme_number: int = Field(
        ...,
        description="Sequential theme identifier as provided",
        examples=[1, 2, 3]
    )
    theme_name: str = Field(
        ...,
        description="Theme name as provided",
        examples=["Waiting Time", "Product Quality"]
    )
    matched_candidates: List[MatchedCandidate] = Field(
        ...,
        description="Best matching existing codes from codebook",
        examples=[[{"code": "Service Speed", "definition": "References to speed of service"}]]
    )
    decision: Literal["USE", "MODIFY_VERTICAL", "MODIFY_HORIZONTAL", "CREATE"] = Field(
        ...,
        description="Coding decision: USE existing, MODIFY existing, or CREATE new",
        examples=["USE", "MODIFY_HORIZONTAL", "CREATE"]
    )
    source_code: Optional[str] = Field(
        default=None,
        description="Exact candidate code name if USE/MODIFY, null if CREATE",
        examples=["Service Speed", None]
    )
    modify_parameters: ModifyParameters = Field(
        ...,
        description="Parameters for modification (populate even if not modifying)"
    )
    justification: str = Field(
        ...,
        description="Decision explanation referencing conceptual family and abstraction level comparison",
        examples=["Theme belongs to same family (service timing) at same abstraction level - MODIFY_HORIZONTAL to broaden scope"]
    )
    updated_assignment_examples: Optional[AssignmentExamples] = Field(
        default=None,
        description="Updated assignment examples reflecting the decision",
        examples=[{
            "inclusion": ["Updated inclusion example 1", "Updated inclusion example 2"],
            "exclusion": ["Updated boundary case to exclude"],
            "near_neighbor": {"label": "Related Code", "tell_apart_rule": "This code focuses on X, neighbor focuses on Y"}
        }]
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)


class CodingDecisionOutput(BaseModel):
    coding_decision: CodingDecision = Field(
        ...,
        validation_alias="coding_devision",
        description="The coding decision result"
    )
    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)


# -----------------------------------------------------------------------------
# 3a. CCODE_CREATION_PROMPT
# -----------------------------------------------------------------------------


CODE_CREATION_PROMPT = """
You are a {language} qualitative research assistant.
Your task is to CREATE a new code that captures the meaning of a newly identified atomic theme within the {facet_name} facet from survey responses, using the specified taxonomy framework.

---

ATOMICITY RULES (must all be satisfied)

A code must be:
- ATOMIC: It expresses exactly ONE semantic nucleus - one indivisible concept.
- SINGLE-HEADED: The code label contains exactly ONE head noun.
- NO COORDINATION: The label must NOT contain "and", "or", "/", commas, or multiple content nouns.
- UNSPLITTABLE: If a label could be split into two meaningful labels, it is NOT atomic.
- ACTIONABLE: Can be clearly identified and address the survey question directly and explicitly.

---

FACET RULES

Codes must describe only the dimension defined by {facet_name}.
- Labels naming {facet_valid_labels} are VALID.
- Labels describing {facet_invalid_labels} are INVALID.

---

CODEBOOK PARAMETERS

<language>
{language}
</language>

<context>
- Domain: {domain}
- Topic: {topic}
- Perspective: {perspective}
- Entity: {entity}
Survey Question: "{survey_question}"
</context>

<new_theme>
New theme:
- name: "{theme_name}"
- description: "{theme_description}"
- Included expressions (these SHOULD be covered by the code):
  {inclusion}
</new_theme>

<taxonomy_parameters>
Facet: {facet_name}
Facet description: {facet_description}
</taxonomy_parameters>

---

LABEL CONSTRAINTS

- Use a noun phrase of 1-10 words.
- Exactly one semantic nucleus; modifiers allowed.
- No coordination (and/or), no lists, no multi-concept bundles.
- Name only the core concept present.
- Labels naming {facet_valid_labels} are VALID.
- Labels describing {facet_invalid_labels} are INVALID.
- DO NOT repeat {perspective}, {domain}, {topic}, or {entity} in the label.

---

DEFINITION CONSTRAINTS

- Use 30 words or fewer.
- Describe what belongs in this code (not why it happens).
- Use observable assignment cues (behaviors, expressions, practices).
- Avoid causes, conditions, interpretations, or outcomes.
- DO NOT repeat {perspective}, {domain}, {topic}, or {entity} in the definition.

GOOD DEFINITION PATTERNS:
- "References to..."
- "Mentions of..."
- "Expressions of..."
- "Concerns about..."

AVOID:
- Broad summaries (e.g., "general dissatisfaction").
- Multi-part or layered meaning.
- Psychological interpretation not grounded in wording.

---

ASSIGNMENT EXAMPLES

- Provide concrete, actionable assignment examples to guide future code assignment.
- inclusion: 2-3 short examples of expressions that SHOULD be coded here.
- exclusion: 1-2 short examples of what should NOT be included.
- near_neighbor: Identify closest confusable concept and how to tell them apart.

---

FINAL OUTPUT

Provide valid JSON following the response schema. Use theme_number and theme_name exactly as provided. Set source_code to null for new codes. Write all values in {language}.
"""

# -----------------------------------------------------------------------------
# 3b. CODING_MODIFICATION_PROMPT
# -----------------------------------------------------------------------------

HORIZONTAL_INSTRUCTIONS = """
   - Keep the abstraction level of the original code.
   - Create a **single atomic shared concept** that:
        (a) captures the meaning of both original code and new theme,
        (b) is grounded in the shared conceptual family and abstraction level,
        (c) remains expressible as **one idea** in the label.
   - The modified label must:
        * reflect the broadened meaning,
        * NOT introduce multiple aspects or abstraction levels,
        * NOT be more abstract than necessary.
   - The modified definition must:
        * describe the **shared meaning space**,
        * reflect: original inclusions + inclusion_update,
        * exclude: original exclusions + exclusion_update.
   - Do **not** modify assignment rules here."""

VERTICAL_INSTRUCTIONS = """
   - Shared conceptual family but different abstraction levels -> create hierarchical structure.
   - Original code and new theme remain **atomic child codes**.
   - Parent code represents the shared **conceptual family**.

   Parent label:
        - parent theme = {parent_theme_label}
        - If parent theme is not None or Null -> use it as-is.
        - If null -> generate a label at a higher abstraction level (Driver/Why level).
        - Must:
            * express shared conceptual family,
            * NOT describe behaviors/outcomes,
            * NOT blend child labels,
            * be broader, not vaguer.

   Structure:
       - Parent = conceptual anchor (higher abstraction level),
       - Children = distinct manifestations (different abstraction levels),
       - Child meanings **do not change**."""


CODING_MODIFICATION_PROMPT = """
You are a {language} qualitative research assistant updating a codebook.
Your task is to MODIFY an existing code so that it fully and correctly includes a new theme within the {facet_name} facet, while preserving **atomic meaning** and **clear conceptual boundaries**.

---

ATOMICITY RULES (must all be satisfied post-modification)

The modified code must remain:
- ATOMIC: It expresses exactly ONE semantic nucleus - one indivisible concept.
- SINGLE-HEADED: The code label contains exactly ONE head noun.
- NO COORDINATION: The label must NOT contain "and", "or", "/", commas, or multiple content nouns.
- UNSPLITTABLE: If a label could be split into two meaningful labels, the modification is INVALID.

---

FACET RULES

Modified codes must describe only the dimension defined by {facet_name}.
- Labels naming {facet_valid_labels} are VALID.
- Labels describing {facet_invalid_labels} are INVALID.

---

CODEBOOK PARAMETERS

<language>
{language}
</language>

<context>
- Domain: {domain}
- Topic: {topic}
- Perspective: {perspective}
- Entity: {entity}
Survey Question: "{survey_question}"
</context>

<new_theme>
New theme to integrate:
- name: "{theme_name}"
- description: "{theme_description}"
- Included expressions (these SHOULD be covered by the code):
  {inclusion}
</new_theme>

<original_code>
Original code (to be modified):
- code_label: {source_code}
- code_definition: {source_definition}
</original_code>

<current_assignment_examples>
Current inclusion examples:
  {current_inclusion}
</current_assignment_examples>

<required_modifications>
- inclusion_update (new expressions that must now be included in-scope):
  {inclusion_update}
- exclusion_update (boundaries to clarify so scope does not overextend):
  {exclusion_update}
</required_modifications>

<taxonomy_parameters>
Facet: {facet_name}
Facet description: {facet_description}
</taxonomy_parameters>

---

MODIFICATION INSTRUCTIONS

Follow these instructions exactly and in order. Do not skip or reorder any instruction.

{modification_instructions}

---

LABEL CONSTRAINTS

- Use a noun phrase of 1-10 words.
- Exactly one semantic nucleus; modifiers allowed.
- No coordination (and/or), no lists, no multi-concept bundles.
- Name only the core concept present.
- Labels naming {facet_valid_labels} are VALID.
- Labels describing {facet_invalid_labels} are INVALID.
- DO NOT repeat {perspective}, {domain}, {topic}, or {entity} in the label.

---

DEFINITION CONSTRAINTS

- Use 30 words or fewer.
- Describe what belongs in this code (not why it happens).
- Use observable assignment cues (behaviors, expressions, practices).
- Avoid causes, conditions, interpretations, or outcomes.
- DO NOT repeat {perspective}, {domain}, {topic}, or {entity} in the definition.

GOOD DEFINITION PATTERNS:
- "References to..."
- "Mentions of..."
- "Expressions of..."
- "Concerns about..."

AVOID:
- Broad summaries (e.g., "general dissatisfaction").
- Multi-part or layered meaning.
- Psychological interpretation not grounded in wording.

---

ASSIGNMENT EXAMPLES

- Provide concrete, actionable assignment examples to guide future code assignment.
- inclusion: Combine original + new expressions from inclusion_update.
- exclusion: Combine original + new boundaries from exclusion_update.
- near_neighbor: Update label/rule if boundaries changed due to modification. Identify closest confusable concept.

FINAL OUTPUT

Provide valid JSON following the response schema. Use theme_number, theme_name, and source_code exactly as provided. Write all values in {language}. If hierarchical_parent_diff_level, ensure parent label is conceptual, not descriptive.
"""


class GeneratedCode(BaseModel):
    theme_number: int = Field(
        ...,
        description="Sequential theme identifier as provided",
        examples=[1, 2, 3]
    )
    theme_name: str = Field(
        ...,
        description="Theme name (cluster_summary) as provided",
        examples=["Waiting Time", "Product Quality"]
    )
    source_code: Optional[str] = Field(
        default=None,
        description="Existing code being modified (null for CREATE)",
        examples=["Service Speed", None]
    )
    code_label: str = Field(
        ...,
        max_length=100,
        description="New or modified code label (1-10 word noun phrase)",
        examples=["Waiting Time", "Service Response"]
    )
    code_definition: str = Field(
        ...,
        max_length=200,
        description="<=25-word operational definition describing what belongs in this code",
        examples=["References to duration of waiting for service or products"]
    )
    assignment_examples: Optional[AssignmentExamples] = Field(
        default=None,
        description="Concrete inclusion/exclusion examples for coding",
        examples=[{
            "inclusion": ["Expresses concern about waiting", "Mentions delay as negative"],
            "exclusion": ["Positive comments about speed"],
            "near_neighbor": {"label": "Service Speed", "tell_apart_rule": "Waiting focuses on negative delay, speed on positive efficiency"}
        }]
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)


class CodeGenerationOutput(BaseModel):
    generated_code: GeneratedCode = Field(
        ...,
        description="The generated or modified code"
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)


# -----------------------------------------------------------------------------
# 4. VALIDATION_PROMPT
# -----------------------------------------------------------------------------

USE_VALIDATION_INSTRUCTIONS = """
**Scenario: USE existing code**

Your task is to validate the proposal that an existing code already captures this theme's meaning.
You must APPROVE or REJECT this proposal. If rejected, provide a corrected final decision.

Scenario-specific validation questions:
- Does the existing code's definition fully cover the expressions in the new theme?
- Would assigning this theme to the existing code lose any meaningful distinctions?
- Are there any expressions in the new theme that the existing code would NOT capture?

General coding validation question:
- Does the existing code align with these critical rules? A code must be:
  * TAXONOMIC: Aligned with the specified taxonomy axis and coding dimension
  * ATOMIC: Express one indivisible concept; cannot be split into separate concepts that are practically meaningful for explaining responses to the survey question
  * SINGLE-VALUED: Represent one clear concept without blending distinct concepts from different conceptual families
  * ACTIONABLE: Be clearly identifiable and address the survey question directly and explicitly
- No code label may contain conjunctions ("and", "or", "&"), slashes, or compound constructions. If present, split into separate atomic codes unless one part has no independent analytic meaning.

If validation FAILS (the existing code does not fully capture the theme):
-> Recommend MODIFY (horizontal or vertical refinement) or CREATE (if substantially different)
"""


MODIFY_HORIZONTAL_VALIDATION_INSTRUCTIONS = """
**Scenario: MODIFY_HORIZONTAL (broaden at same abstraction level)**
Your task is to validate the coding proposal that the modification BROADENS the code while PRESERVING its semantic core.
You need to APPROVE or REJECT this proposal, and if rejected, provide a corrected final decision.

Scenario-specific validation questions:
- Does the new label preserve the original code's central meaning?
- Is the modification genuinely broadening scope, not replacing the concept?
- Do BOTH the original expressions AND new expressions fit under the unified meaning?
- Would a coder still recognize this as the same code with expanded coverage?

General coding validation question:
- Does the existing code align with these critical rules? A code must be:
  * TAXONOMIC: Aligned with the specified taxonomy axis and coding dimension
  * ATOMIC: Express one indivisible concept; cannot be split into separate concepts that are practically meaningful for explaining responses to the survey question
  * SINGLE-VALUED: Represent one clear concept without blending distinct concepts from different conceptual families
  * ACTIONABLE: Be clearly identifiable and address the survey question directly and explicitly
- No code label may contain conjunctions ("and", "or", "&"), slashes, or compound constructions. If present, split into separate atomic codes unless one part has no independent analytic meaning.

CRITICAL: If the new label shifts or replaces the core concept rather than extending it:
-> REJECT and recommend CREATE instead (preserve original code, create new one)
"""

MODIFY_VERTICAL_VALIDATION_INSTRUCTIONS = """
**Scenario:  MODIFY_VERTICAL (create parent at higher abstraction level)**
Your task is to validate the coding proposal that the modification propely forms hierarchical structure.
You need to APPROVE or REJECT this proposal, and if rejected, provide a corrected final decision.

Scenario-specific validation questions:
- Is the parent code abstract enough to encompass both child codes?
- Does the parent represent the shared conceptual family, not just a blend of labels?
- Do the child codes remain atomic and distinct at their abstraction levels?
- Is there a genuine abstraction-level difference (not just wording variation)?

General coding validation question:
- Does the existing code align with these critical rules? A code must be:
  * TAXONOMIC: Aligned with the specified taxonomy axis and coding dimension
  * ATOMIC: Express one indivisible concept; cannot be split into separate concepts that are practically meaningful for explaining responses to the survey question
  * SINGLE-VALUED: Represent one clear concept without blending distinct concepts from different conceptual families
  * ACTIONABLE: Be clearly identifiable and address the survey question directly and explicitly
- No code label may contain conjunctions ("and", "or", "&"), slashes, or compound constructions. If present, split into separate atomic codes unless one part has no independent analytic meaning.

If validation FAILS:
-> Recommend MODIFY_VERTICAL (if same level) or CREATE (if unrelated)
"""

CREATE_VALIDATION_INSTRUCTIONS = """
**Scenario: CREATE new code**
Your task is to validate the coding propoal that this theme represents a genuinely novel concept, requiring a new code.
You need to APPROVE or REJECT this proposal, and if rejected, provide a corrected final decision.

Scenario-specific validation questions:
- Is there truly NO existing code that partially or fully covers this theme?
- Does this fill a real gap in the codebook (not just a wording preference)?
- Would adding this code improve the codebook's ability to capture distinct meanings?
- Is the new code sufficiently different from ALL existing codes?

General coding validation question:
- Does the existing code align with these critical rules? A code must be:
  * TAXONOMIC: Aligned with the specified taxonomy axis and coding dimension
  * ATOMIC: Express one indivisible concept; cannot be split into separate concepts that are practically meaningful for explaining responses to the survey question
  * SINGLE-VALUED: Represent one clear concept without blending distinct concepts from different conceptual families
  * ACTIONABLE: Be clearly identifiable and address the survey question directly and explicitly
- No code label may contain conjunctions ("and", "or", "&"), slashes, or compound constructions. If present, split into separate atomic codes unless one part has no independent analytic meaning.

If validation FAILS (an existing code could cover this):
-> Recommend USE (if fully covered) or MODIFY (if partial overlap)
"""

VALIDATION_PROMPT = """
You are a codebook curator for thematic analysis following Braun & Clarke (2006) methodology.
Your role is to maintain parsimonious codebooks with non-overlapping and non-redundant codes by reviewing and making final decisions on coding proposals.

Here is the codebook context you will be working with:

<codebook_context>
- Domain: {domain}
- Topic: {topic}

Existing codes in codebook:
{code_text}
</codebook_context>

Here is the coding proposal you need to evaluate:

<coding_proposal>
A new theme emerged from analyzing responses to this survey question:
"{survey_question}"

This is a new theme to be evaluated:
- name: "{theme_name}"
- description: "{theme_description}"

The proposal to review:
{step3_recommendation}

Further information about the new code:
- The new code should cover these expressions:
  {inclusion_examples}
</coding_proposal>

Here is the taxonomy framework guiding your analysis:
- Facet: {facet_name}
- Facet description: {facet_description}

Here are the scenario-specific validation instructions for this decision type:

<scenario_instructions>
{validation_instructions}
</scenario_instructions>

<scratchpad>
Work through your evaluation systematically:

**Apply Scenario-Specific Validation**
- Review the scenario instructions in <scenario_instructions> above
- Apply the scenario-specific validation questions for this decision type (USE / MODIFY_HORIZONTAL / MODIFY_VERTICAL / CREATE)
- Document whether the proposal passes or fails each criterion
- If the scenario instructions recommend a different action than the proposal, the proposal fails validation

**Provide a correct final decision for the codebook, if proposal is rejected**
- If the proposal is APPROVED -> final decision = original recommendation
- If the proposal is REJECTED -> final decision = USE, MODIFY_HORIZONTAL, MODIFY_VERTICAL, or CREATE based on your analysis

**Determine final decision components**
- validated_decision: Final decision (USE, MODIFY_HORIZONTAL, MODIFY_VERTICAL, or CREATE)
- source_code:
   - If USE -> exact code from proposal
   - If MODIFY_HORIZONTAL or MODIFY_VERTICAL -> exact existing code being modified
   - If CREATE -> null
- validated_code: Final compliant label, definition, and assignment examples
- decision_rationale: Brief explanation of why the proposal was approved or rejected

**Generate final decision codes, labels and descriptions, if proposal is rejected****

CODING RULES:
A code must be:
- TAXONOMIC: Aligned with the specified taxonomy axis and coding dimension
- ATOMIC: Expresses one indivisible concept; cannot be split into separate concepts that are practically meaningful for explaining responses to the survey question
- SINGLE-VALUED: Represents one clear concept without blending distinct concepts from different conceptual families
- ACTIONABLE: Can be clearly identified and address the survey question directly and explicity

LABEL RULES:
- Use a short noun phrase of 10 words or fewer
- Make the semantic core of the theme the head of the noun phrase
- The label must describe an ATOMIC theme in light of the research question, taxonomy axis, and coding dimension
- All naming and labeling of ATOMIC THEMES must be single-valued
- No code label may contain conjunctions ("and", "or", "&"), slashes, or compound constructions. If present, split into separate atomic codes unless one part has no independent analytic meaning.
- DO NOT repeat the actor, domain, topic, or entity in the label (do not repeat: {perspective}, {domain}, {topic} and {entity})

DEFINITION RULES:
- Use 30 words or fewer
- Ground the definition in the cluster data
- Describe **what belongs in this code**, not why it happens
- Align directly with the survey question, taxonomy axis, and coding dimension
- Use a clear, observable assignment cue (e.g., behaviors, expressions, judgments)
- Do NOT explain causes, conditions, or interpretations
- DO NOT repeat the actor, domain, topic, or entity in the description (do not repeat: {perspective}, {domain}, {topic} and {entity})

GOOD DEFINITION PATTERNS FOR FINAL DECISION::
- "References to..."
- "Mentions of..."
- "Expressions of..."
- "Concerns about..."
</scratchpad>

Now provide your final evaluation as valid JSON following the response schema.

**Output Requirements:**
- Use theme_number and theme_name exactly as provided in the coding proposal
- For source_code: If USE, use exact code from proposal; If MODIFY, use exact existing code being modified; If CREATE, use null
- Write all values in {language}
- Ensure all labels and definitions strictly follow the rules above
"""


class ValidatedCode(BaseModel):
    code: str = Field(
        ...,
        max_length=100,
        description="Final validated code label (<=10 words, rule-compliant)",
        examples=["Waiting Time", "Service Response"]
    )
    definition: str = Field(
        ...,
        max_length=200,
        description="Final validated definition (<=25 words, operational, grounded)",
        examples=["References to duration of waiting for service or products"]
    )
    assignment_examples: Optional[AssignmentExamples] = Field(
        default=None,
        description="Validated/refined assignment examples for coding",
        examples=[{
            "inclusion": ["Validated inclusion example 1", "Validated inclusion example 2"],
            "exclusion": ["Validated boundary case to exclude"],
            "near_neighbor": {"label": "Nearby Theme", "tell_apart_rule": "This theme focuses on X, nearby theme focuses on Y"}
        }]
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)


class OriginalRecommendation(BaseModel):
    code: str = Field(
        ...,
        description="Exact recommended code label from proposal",
        examples=["Waiting Time", "Product Quality"]
    )
    definition: str = Field(
        ...,
        description="Exact recommended definition from proposal",
        examples=["References to time spent waiting"]
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)


class CodeValidation(BaseModel):
    theme_number: int = Field(
        ...,
        description="Sequential theme identifier as provided in proposal",
        examples=[1, 2, 3]
    )
    theme_name: str = Field(
        ...,
        description="Theme name as provided in proposal",
        examples=["Waiting Time", "Product Quality"]
    )
    original_recommendation: OriginalRecommendation = Field(
        ...,
        description="The original coding recommendation being validated"
    )
    verdict: Literal["APPROVE", "REJECT"] = Field(
        ...,
        description="Binary accept/reject of original proposal. APPROVE = proposal is acceptable as-is; REJECT = proposal needs correction. NOT a coding decision - use validated_decision for USE/MODIFY/CREATE.",
        examples=["APPROVE", "REJECT"]
    )
    decision_rationale: str = Field(
        ...,
        description="Brief explanation of why proposal was approved or rejected",
        examples=["Proposal maintains atomicity and aligns with taxonomy axis"]
    )
    validated_decision: Literal["USE", "MODIFY_HORIZONTAL", "MODIFY_VERTICAL", "CREATE"] = Field(
        ...,
        description="Final decision after validation",
        examples=["USE", "MODIFY_HORIZONTAL", "CREATE"]
    )
    source_code: Optional[str] = Field(
        default=None,
        description="If USE: exact code from proposal; If MODIFY: exact existing code being modified; If CREATE: null",
        examples=["Service Speed", None]
    )
    validated_code: ValidatedCode = Field(
        ...,
        description="Final validated code with label, definition, and assignment examples"
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ValidationResult(BaseModel):
    code_validation: CodeValidation = Field(
        ...,
        description="The validation result for the coding proposal"
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)
