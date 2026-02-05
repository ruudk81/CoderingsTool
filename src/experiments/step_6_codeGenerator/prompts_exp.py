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

AXIS_LABEL_CONTRACT = {
  "WHAT": {
    "theme_head": "topic",  # or "object"
    "must_be": "topic/object/attribute (a thing being referenced)",
    "must_not_be": "action/method, intent/outcome, actor, evaluation, time, location"
  },
  "WHY": {
    "theme_head": "motive", # or "intent"
     "must_be": "intent/purpose/reason (a goal or desired outcome)",
    "must_not_be": "action/method, topic/object, actor, evaluation, time, location"
  },
  "HOW": {
    "theme_head": "mechanism", # or "practice"
    "must_be": "action/method/mechanism (a practice or way of doing)",
    "must_not_be": "intent/outcome, topic/object, actor, evaluation, time, location"
  },
  "WHO": {
    "theme_head": "actor", # or "stakeholder"
    "must_be": "actor/target group (a person/group/stakeholder)",
    "must_not_be": "action/method, intent/outcome, topic/object, evaluation, time, location"
  },
  "WHEN": {
    "theme_head": "timing", # or "time-reference"
    "must_be": "time/urgency/sequence reference",
    "must_not_be": "action/method, intent/outcome, topic/object, actor, evaluation, location"
  },
  "WHERE": {
    "theme_head": "context",  #or "setting"
    "must_be": "location/context/channel/setting",
    "must_not_be": "action/method, intent/outcome, topic/object, actor, evaluation, time"
  }
}

# -----------------------------------------------------------------------------
# 1. CLUSTER_SUMMARY_PROMPT
# -----------------------------------------------------------------------------

CLUSTER_SUMMARY_PROMPT = """
You are a qualitative researcher responsible for extracting ATOMIC {taxonomy_actionable_type}-{theme_head} THEMES from descriptive codes representing survey responses to a survey question.
An ATOMIC {taxonomy_actionable_type}-{theme_head} THEME is a single, indivisible {taxonomy_actionable_type} or {theme_head} present in the data.

Atomicity rules (must all be satisfied):
- The theme expresses exactly ONE semantic nucleus.
- The theme label contains exactly ONE head noun.
- The label must NOT contain "and", "or", "/", commas, or multiple content nouns.
- If a label could be split into two meaningful labels, it is NOT atomic.
- If multiple aspects appear in the cluster, you MUST split them into separate atomic themes. Do NOT invent meta-parent or umbrella concepts.

{taxonomy_actionable_type}-{theme_head} rules:
- Themes must describe only the dimension defined by {taxonomy_axis} ({taxonomy_actionable_type}s or {theme_head}s).
- Labels naming are {must_be} VALID.
- Labels naming are {must_not_be} INVALID.

Grounding rules:
- Themes must be directly supported by the descriptive codes.
- Do not introduce themes not present in the cluster.

Dimension constraint:
All themes MUST remain strictly within the specified taxonomy axis: {taxonomy_axis}: {taxonomy_axis_description}.

---

SURVEY CONTEXT

<survey_context>
Survey question: "{survey_question}"
Language: {language}
Domain: {domain}
Topic: {topic}
Perspective: {perspective}
Intent: {intent}
Entity: {entity}
</survey_context>

---

CLUSTER DATA

<cluster_id>
{cluster_id}
</cluster_id>

<cluster_text>
{cluster_text}
</cluster_text>

---

LABEL CONSTRAINTS

Theme labels must:
- Be a noun phrase of 1-3 words.
- Exactly one semantic head (one core concept); modifiers allowed
- no coordination (and/or), no lists, no multi-concept bundles
- Name only the {taxonomy_actionable_type}-{theme_head} present.
- Avoid repeating {perspective}, {domain}, {topic}, or {entity}.

---

DEFINITION CONSTRAINTS

Theme definitions must:
- Use 30 words or fewer.
- Describe what belongs in this theme.
- Use observable assignment cues (behaviors, expressions, practices).
- Avoid causes, conditions, interpretations, or outcomes.
- Avoid repeating {perspective}, {domain}, {topic}, or {entity}.

---

REQUIRED ANALYSIS STEPS

1. Identify distinct {taxonomy_actionable_type}-{theme_head} in light of the survey quesition that is present in the cluster.
2. List 1-5 candidate {taxonomy_actionable_type}-{theme_head}s. Each must be a 1-3 word noun phrase satisfying atomic label rules.
3. If more than one {taxonomy_actionable_type}-{theme_head}s is found, treat them as separate potential atomic themes.
4. For each {taxonomy_actionable_type}-{theme_head}, verify grounding in cluster codes.
5. Do not merge {taxonomy_actionable_type}-{theme_head}s into umbrella or meta-parent concepts.
6. Select only {taxonomy_actionable_type}-{theme_head}s that are clearly supported by the data.
7. Produce final theme entries only for valid atomic {taxonomy_actionable_type}-{theme_head}s.

---

FINAL OUTPUT FORMAT

After analysis, output valid JSON as instruced by the response schema provided.

---

OUTPUT REQUIREMENTS

- All field values must be written in {language}
- The cluster_id must be exactly "{cluster_id}" as provided
- Conduct your entire analysis in {language}
- If multiple themes are identified, include each as a separate object with sequential theme_id values
- Provide 2-3 inclusion examples and 1-2 exclusion examples for each theme
- Assignment examples should be short, concrete, and start with verbs
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
Your task is to classify a newly identified {taxonomy_actionable_type}-{theme_head} theme and decide whether to USE an existing code, MODIFY an existing code, or CREATE a new code.
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
Taxonomy Axis: {taxonomy_axis}: {taxonomy_axis_description}
Primary Coding Dimension: {taxonomy_actionable_type}
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

{taxonomy_actionable_type}-{theme_head} RULES

Theme labels must describe only the dimension defined by {taxonomy_axis} ({taxonomy_actionable_type}s or {theme_head}s).
- Labels naming {must_be} are VALID.
- Labels naming {must_not_be} are INVALID.

---

DECISION OPTIONS

You must choose one of the following actions:

- **USE** - An existing code fully captures the new theme's meaning; use it as-is without modification
- **MODIFY_HORIZONTAL** - An existing code needs broader definition and inclusion rules to cover the new theme, but remains at the same abstraction level on the coding dimension ("{taxonomy_axis}:{taxonomy_actionable_type}")
- **MODIFY_VERTICAL** - The existing code and new theme belong to the same conceptual family but differ in abstraction level; create or reference a parent code for both
- **CREATE** - Add a new code because the theme represents a distinct {taxonomy_actionable_type}-{theme_head} not covered by existing codes

---

ANALYSIS FRAMEWORK

Follow these steps systematically:

**STEP 0: Initial Matching**
- Review the new theme and all existing codes
- Identify the best matching existing code(s) based on core meaning and practical relevance in light of the research question, taxonomy axis, and primary coding dimension

**STEP 1: Conceptual Family Test**
Ask: Do the new theme and the best matching existing code belong to the same conceptual family, given the research question, taxonomy axis ({taxonomy_axis}), and primary coding dimension ({taxonomy_actionable_type})?
- If the new theme and best matching existing code share the same core {theme_head} and have the same practical relevance -> SAME FAMILY
- Otherwise -> DIFFERENT FAMILY

**STEP 2: Abstraction Level Test**
Ask: Are the new theme and the best matching existing code at the same abstraction level on the taxonomy axis/coding dimension?
- If the height of generality/specificity is similar -> SAME ABSTRACTION LEVEL
- Otherwise -> DIFFERENT ABSTRACTION LEVEL

**STEP 3: Decision Logic**
Apply the following decision rules:

- If the new theme is fully covered in meaning and scope by an existing code -> USE existing code.
- If the new theme is not fully covered by an existing code:
  - If it belongs to the same code family and is at the same abstraction level -> MODIFY_HORIZONTAL
      - Broaden the existing code's definition and inclusion rules to incorporate the new expression, ensuring the original core meaning remains intact.
  - If it belongs to the same code family but at a different abstraction level -> MODIFY_VERTICAL
      - Introduce or reference a higher-level parent code, treating the existing code and new theme as related sub-codes.
  - If it belongs to a different code family -> CREATE a new code for the distinct {taxonomy_actionable_type}-{theme_head}.

**STEP 4: Multi-Concept Theme Check**
If the new theme contains multiple distinct {taxonomy_actionable_type}-{theme_head}s (e.g., "salt reduction AND mild spices"):
- Identify which {taxonomy_actionable_type}-{theme_head}(s) semantically match the existing code
- If only ONE {taxonomy_actionable_type}-{theme_head} matches and MODIFY would require changing the existing code's core meaning to accommodate the other: Decision = **CREATE**
- A MODIFY should never replace an existing code's central meaning with a different {theme_head}
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
- Exactly one semantic head (one core {theme_head}); modifiers allowed.
- No coordination (and/or), no lists, no multi-concept bundles.
- Name only the {taxonomy_actionable_type}-{theme_head} present.
- Labels naming {must_be} are VALID.
- Labels naming {must_not_be} are INVALID.
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

1. Identify the top candidate code(s) based on semantic similarity to the {taxonomy_actionable_type}-{theme_head}.
2. Note any cosine similarity scores for top candidates (if provided).
3. Apply the Conceptual Family Test (STEP 1): Do they share the same core {theme_head}?
4. Apply the Abstraction Level Test (STEP 2): Same specificity level on {taxonomy_axis}?
5. Apply the Decision Logic (STEP 3): USE, MODIFY_HORIZONTAL, MODIFY_VERTICAL, or CREATE.
6. Check for multi-concept themes (STEP 4): Does the theme contain multiple distinct {taxonomy_actionable_type}-{theme_head}s?
7. Verify label compliance: Ensure the {taxonomy_actionable_type}-{theme_head} rules are satisfied (VALID/INVALID criteria).
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
Your task is to CREATE a new code that captures the meaning of a newly identified atomic {taxonomy_actionable_type}-{theme_head} theme from survey responses, using the specified taxonomy framework.

---

ATOMICITY RULES (must all be satisfied)

A code must be:
- ATOMIC: It expresses exactly ONE semantic nucleus - one indivisible {taxonomy_actionable_type} or {theme_head}.
- SINGLE-HEADED: The code label contains exactly ONE head noun.
- NO COORDINATION: The label must NOT contain "and", "or", "/", commas, or multiple content nouns.
- UNSPLITTABLE: If a label could be split into two meaningful labels, it is NOT atomic.
- ACTIONABLE: Can be clearly identified and address the survey question directly and explicitly.

---

{taxonomy_actionable_type}-{theme_head} RULES

Codes must describe only the dimension defined by {taxonomy_axis} ({taxonomy_actionable_type}s or {theme_head}s).
- Labels naming {must_be} are VALID.
- Labels naming {must_not_be} are INVALID.

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
Taxonomy Axis: {taxonomy_axis}: {taxonomy_axis_description}
Primary Coding Dimension: {taxonomy_actionable_type}
</taxonomy_parameters>

---

LABEL CONSTRAINTS

- Use a noun phrase of 1-10 words.
- Exactly one semantic head (one core {theme_head}); modifiers allowed.
- No coordination (and/or), no lists, no multi-concept bundles.
- Name only the {taxonomy_actionable_type}-{theme_head} present.
- Labels naming {must_be} are VALID.
- Labels naming {must_not_be} are INVALID.
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
- near_neighbor: Identify closest confusable {taxonomy_actionable_type}-{theme_head} and how to tell them apart.

---

FINAL OUTPUT

Provide valid JSON following the response schema. Use theme_number and theme_name exactly as provided. Set source_code to null for new codes. Write all values in {language}.
"""

# Placeholders for CODING_MODIFICATION_PROMPT
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


# -----------------------------------------------------------------------------
# 3b. CODING_MODIFICATION_PROMPT
# -----------------------------------------------------------------------------


CODING_MODIFICATION_PROMPT = """
You are a {language} qualitative research assistant updating a codebook.
Your task is to MODIFY an existing code so that it fully and correctly includes a new {taxonomy_actionable_type}-{theme_head} theme, while preserving **atomic meaning** and **clear conceptual boundaries**.

---

ATOMICITY RULES (must all be satisfied post-modification)

The modified code must remain:
- ATOMIC: It expresses exactly ONE semantic nucleus - one indivisible {taxonomy_actionable_type} or {theme_head}.
- SINGLE-HEADED: The code label contains exactly ONE head noun.
- NO COORDINATION: The label must NOT contain "and", "or", "/", commas, or multiple content nouns.
- UNSPLITTABLE: If a label could be split into two meaningful labels, the modification is INVALID.

---

{taxonomy_actionable_type}-{theme_head} RULES

Modified codes must describe only the dimension defined by {taxonomy_axis} ({taxonomy_actionable_type}s or {theme_head}s).
- Labels naming {must_be} are VALID.
- Labels naming {must_not_be} are INVALID.

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
Taxonomy Axis: {taxonomy_axis}: {taxonomy_axis_description}
Primary Coding Dimension: {taxonomy_actionable_type}
</taxonomy_parameters>

---

MODIFICATION INSTRUCTIONS

Follow these instructions exactly and in order. Do not skip or reorder any instruction.

{modification_instructions}

---

LABEL CONSTRAINTS

- Use a noun phrase of 1-10 words.
- Exactly one semantic head (one core {theme_head}); modifiers allowed.
- No coordination (and/or), no lists, no multi-concept bundles.
- Name only the {taxonomy_actionable_type}-{theme_head} present.
- Labels naming {must_be} are VALID.
- Labels naming {must_not_be} are INVALID.
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
- near_neighbor: Update label/rule if boundaries changed due to modification. Identify closest confusable {taxonomy_actionable_type}-{theme_head}.

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
  * ATOMIC: Express one indivisible {taxonomy_actionable_type}; cannot be split into separate {taxonomy_actionable_type} that are practically meaningful for explaining responses to the survey question
  * SINGLE-VALUED: Represent one clear concept without blending distinct {taxonomy_actionable_type} from different conceptual families
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
  * ATOMIC: Express one indivisible {taxonomy_actionable_type}; cannot be split into separate {taxonomy_actionable_type} that are practically meaningful for explaining responses to the survey question
  * SINGLE-VALUED: Represent one clear concept without blending distinct {taxonomy_actionable_type} from different conceptual families
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
  * ATOMIC: Express one indivisible {taxonomy_actionable_type}; cannot be split into separate {taxonomy_actionable_type} that are practically meaningful for explaining responses to the survey question
  * SINGLE-VALUED: Represent one clear concept without blending distinct {taxonomy_actionable_type} from different conceptual families
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
  * ATOMIC: Express one indivisible {taxonomy_actionable_type}; cannot be split into separate {taxonomy_actionable_type} that are practically meaningful for explaining responses to the survey question
  * SINGLE-VALUED: Represent one clear concept without blending distinct {taxonomy_actionable_type} from different conceptual families
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
- Taxonmy Axis:  {taxonomy_axis}: {taxonomy_axis_description}
- primary Coding Dimension: {taxonomy_actionable_type}

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
- ATOMIC: Expresses one indivisible  {taxonomy_actionable_type};  cannot be split into separate {taxonomy_actionable_type} that are practically meaningful for explaining responses to the survey question
- SINGLE-VALUED: Represents one clear concept without blending distinct {taxonomy_actionable_type} from different conceptual families
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
