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
# 1. CLUSTER_SUMMARY_PROMPT  (unified -- works for all Stage 1 input routes)
#
# Dynamic placeholders filled by the caller:
#   {data_unit}        -- "cluster" or "category"
#   {evidence_source}  -- "the provided key expressions" or "the assigned ideas"
#   {data_description} -- paragraph(s) explaining the data block structure
#   {analysis_step_2}  -- route-specific reasoning instruction
#   {analysis_step_3}  -- route-specific reasoning instruction
# -----------------------------------------------------------------------------

CLUSTER_SUMMARY_PROMPT = """
You are a qualitative researcher identifying meaning-based codes from survey response data for a codebook. Your work is interpretive but strictly data-bound: you may organize and name patterns, but you must not introduce concepts absent from the provided data.

THEME PRINCIPLE (Braun & Clarke, 2006)
A code is a meaning-based organizing concept, not a topic label. It captures WHAT UNIFIES a set of responses -- the shared interpretive thread that explains WHY these expressions belong together. If a candidate can be expressed as a single everyday word (e.g., "price"), it is likely a topic, not a theme. A well-formed code names the specific meaning pattern respondents express.

## Research Context

<survey_context>
Respondents were asked: "{survey_question}"
Reframe as: What meanings do {perspective}s actually express about {entity} regarding {topic}?
Your codes must capture these expressed meanings, not just the surface topics mentioned.

- Language: {language}
- Domain: {domain}
- Perspective: {perspective}
- Intent: {intent}
- Entity: {entity}
</survey_context>

## Taxonomy

<taxonomy_rules>
This codebook covers the {facet_name} dimension.
Scope check: codes must capture concepts of type {facet_valid_labels}.
Out of scope: {facet_invalid_labels}.
Note: these concept types define what BELONGS here, not how to WORD labels.
</taxonomy_rules>

## {data_unit} Data to Analyze

<{data_unit}_data>
<{data_unit}_id>
{cluster_id}
</{data_unit}_id>

<{data_unit}_content>
{cluster_text}
</{data_unit}_content>
</{data_unit}_data>

{data_description}

## Analysis Instructions

First, identify 1-2 Central Organizing Concepts (COCs) -- analytic statements that capture what ties this {data_unit}'s data together. COCs describe shared patterns across {evidence_source}. They are NOT code labels.

Then derive atomic themes from the data. Each theme must satisfy ALL criteria:

THEME CRITERIA
1. Atomic: one semantic nucleus, unsplittable into independently meaningful parts.
2. Within-facet: falls within the {facet_name} scope.
3. Grounded: directly supported by {evidence_source}.
4. Operational: precise enough to serve as a codebook entry.
5. Meaning-based: captures a shared meaning pattern, not just a topic.
6. Metadata-coherent: consistent with any structured metadata provided (inclusion definitions, boundary tests).

LABEL RULES
- Concise noun phrase, 1-5 words (maximum 8 if necessary).
- One semantic head noun; modifiers allowed.
- No conjunctions (and/or), slashes, commas, or multi-concept bundles.
- Do NOT repeat {perspective}, {domain}, {topic}, or {entity}.
- Name the SPECIFIC content, not the facet dimension. The taxonomy already establishes that codes are about {facet_name}. Labels name WHAT was experienced/perceived/suggested — not THAT it was experienced/perceived/suggested.
- WRONG: "Experience of musical enjoyment", "Perception of atmosphere", "Feeling of togetherness"
- RIGHT: "Musical enjoyment", "Festival atmosphere", "Togetherness"

DEFINITION RULES
- 30 words or fewer.
- Describe observable assignment cues (what respondents say/describe).
- No causes, motives, conditions, or outcomes.
- Do NOT repeat {perspective}, {domain}, {topic}, or {entity}.
- Patterns: "References to...", "Mentions of...", "Expressions of..."

ASSIGNMENT EXAMPLES
For each theme provide:
- inclusion (2-3): Observable cues starting with verbs ("Describes...", "Mentions..."), traceable to {evidence_source}.
- exclusion (1-2): Boundary cases that should NOT be coded here.
- near_neighbor: Closest confusable theme label + one sentence distinguishing them. Use "Unknown" if none.

## Required Analysis

Document your reasoning in the analysis field:
1. State 1-2 {data_unit}-level COCs
2. {analysis_step_2}
3. {analysis_step_3}
4. Justify single vs multiple themes

Write all output in {language}. Provide your output as valid JSON following the response schema provided.
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
        max_length=60,
        description="1-5 word meaning-based noun phrase theme label",
        examples=["Waiting Time", "Product Quality", "Staff Friendliness"]
    )
    theme_clarification: str = Field(
        ...,
        max_length=250,
        description="<=30-word grounded definition describing what belongs in this theme",
        examples=["Responses mentioning the duration of waiting for service or products"]
    )
    abstraction_level: Literal["concrete-experiential", "interpretive-pattern"] = Field(
        ...,
        description="'concrete-experiential' = directly observable in responses; 'interpretive-pattern' = inferred shared meaning pattern",
        examples=["concrete-experiential", "interpretive-pattern"]
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
            raise ValueError(f"theme_label must be <=10 words, got {word_count}")
        return v

    @field_validator('theme_clarification')
    @classmethod
    def validate_clarification_length(cls, v):
        word_count = len(v.split())
        if word_count > 36:
            raise ValueError(f"theme_clarification must be <=36 words, got {word_count}")
        return v


class ClusterSummaryOutput(BaseModel):
    cluster_id: str = Field(
        ...,
        description="The cluster identifier exactly as provided",
        examples=["3", "5", "12"]
    )
    analysis: str = Field(
        ...,
        description="1-3 sentence analysis: COCs identified, splits/merges made, rationale for single vs multiple themes",
        examples=["Identified 2 COCs: 'speed' and 'accuracy'. Retained both as distinct atomic concepts."]
    )
    extracted_themes: List[ClusterThemeItem] = Field(
        ...,
        description="Final theme entries for valid atomic themes"
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)


# -----------------------------------------------------------------------------
# 2. CODING_DECISION_PROMPT
# -----------------------------------------------------------------------------

CODING_DECISION_PROMPT = """
You are a qualitative research assistant maintaining a parsimonious codebook following Braun & Clarke (2006) methodology. Your task: classify a newly identified theme within the {facet_name} facet and decide whether to USE, MODIFY, or CREATE a code.

THEME PRINCIPLE (Braun & Clarke, 2006)
A code is a meaning-based organizing concept, not a topic label. It captures WHAT UNIFIES responses -- the shared interpretive thread that explains WHY expressions belong together. The codebook must remain MECE (Mutually Exclusive, Collectively Exhaustive).

---

PARAMETERS

<context>
Respondents were asked: "{survey_question}"
Reframe as: What meanings do {perspective}s actually express about {entity} regarding {topic}?
Language: {language}
Domain: {domain} | Entity: {entity}
</context>

<taxonomy>
This codebook covers the {facet_name} dimension.
Scope check: codes must capture concepts of type {facet_valid_labels}.
Out of scope: {facet_invalid_labels}.
Note: these concept types define what BELONGS here, not how to WORD labels.
Facet instruction: {facet_description}
</taxonomy>

<new_theme>
- name: "{theme_name}"
- description: "{theme_description}"
- included expressions:
    {inclusion}
</new_theme>

<existing_codes>
{code_text}
</existing_codes>

---

DECISION OPTIONS

- **USE** -- Existing code fully captures the new theme's meaning; use as-is.
- **MODIFY_HORIZONTAL** -- Existing code needs broader scope at the same abstraction level.
- **MODIFY_VERTICAL** -- Same conceptual family but different abstraction level; create/reference a parent code.
- **CREATE** -- Theme represents a distinct concept not covered by existing codes.

If the codebook is empty, CREATE is the only valid decision.

---

ANALYSIS FRAMEWORK

1. **MATCH**: Identify best-matching existing code(s) by core meaning and practical relevance.
2. **FAMILY TEST**: Do the theme and best match share the same conceptual family (same core concept, same practical relevance given the research question)?
3. **LEVEL TEST**: Are they at the same abstraction level on the {facet_name} axis?
4. **DECIDE**:
   - Fully covered in meaning and scope -> USE
   - Same family + same level + not fully covered -> MODIFY_HORIZONTAL
   - Same family + different level -> MODIFY_VERTICAL
   - Different family -> CREATE
   - Multi-concept theme where MODIFY would replace core meaning -> CREATE

---

LABEL RULES
- Concise noun phrase, 1-5 words (maximum 8 if necessary).
- One semantic head noun; modifiers allowed.
- No conjunctions (and/or), slashes, commas, or multi-concept bundles.
- Do NOT repeat {perspective}, {domain}, {topic}, or {entity}.
- Name the SPECIFIC content, not the facet dimension. The taxonomy already establishes that codes are about {facet_name}. Labels name WHAT was experienced/perceived/suggested — not THAT it was experienced/perceived/suggested.
- WRONG: "Experience of musical enjoyment", "Perception of atmosphere", "Feeling of togetherness"
- RIGHT: "Musical enjoyment", "Festival atmosphere", "Togetherness"

DEFINITION RULES
- 30 words or fewer.
- Describe observable assignment cues (what respondents say/describe).
- No causes, motives, conditions, or outcomes.
- Do NOT repeat {perspective}, {domain}, {topic}, or {entity}.
- Patterns: "References to...", "Mentions of...", "Expressions of..."

---

OUTPUT INSTRUCTIONS

Reason through the analysis framework before producing your answer. In your justification, reference: (a) matched code(s), (b) family test result, (c) level test result. Reference cosine similarity scores if provided.

Write field names in English, values in {language}. Provide your output as valid JSON following the response schema provided.
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


# NOTE on modify_instruction values: Due to legacy naming, the Literal values
# are confusingly inverted relative to the decision names:
#   "vertical_broaden_same_level"       -> used for MODIFY_HORIZONTAL (broaden at same level)
#   "hierarchical_parent_diff_level"    -> used for MODIFY_VERTICAL (create parent)
# These values are checked in codeGenerator_exp.py lines ~5744-5748.
# Do NOT rename without updating the caller.
class ModifyParameters(BaseModel):
    modify_instruction: Literal["vertical_broaden_same_level", "hierarchical_parent_diff_level", "none"] = Field(
        ...,
        description="Type of modification: 'vertical_broaden_same_level' = broaden at same level (HORIZONTAL), 'hierarchical_parent_diff_level' = create parent (VERTICAL), 'none' = no modification",
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
        description="Best matching existing codes from codebook (for traceability)",
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
        description="Parameters for modification. Set all fields to 'none'/null for USE and CREATE decisions."
    )
    justification: str = Field(
        ...,
        description="Decision explanation referencing conceptual family and abstraction level comparison",
        examples=["Theme belongs to same family (service timing) at same abstraction level - MODIFY_HORIZONTAL to broaden scope"]
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)


# NOTE: validation_alias="coding_devision" is a workaround for LLM outputs
# that occasionally misspell "decision" as "devision". populate_by_name=True
# allows both the canonical field name and the alias to be accepted.
class CodingDecisionOutput(BaseModel):
    coding_decision: CodingDecision = Field(
        ...,
        validation_alias="coding_devision",
        description="The coding decision result"
    )
    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)


# -----------------------------------------------------------------------------
# 3a. CODE_CREATION_PROMPT
# -----------------------------------------------------------------------------


CODE_CREATION_PROMPT = """
You are a {language} qualitative research assistant. Your task: CREATE a new code for a newly identified theme within the {facet_name} facet.

THEME PRINCIPLE (Braun & Clarke, 2006)
A code is a meaning-based organizing concept, not a topic label. It captures WHAT UNIFIES responses -- the shared interpretive thread. The code must be atomic (one semantic nucleus) and within the {facet_name} dimension.

---

PARAMETERS

<context>
Respondents were asked: "{survey_question}"
Reframe as: What meanings do {perspective}s actually express about {entity} regarding {topic}?
Language: {language}
Domain: {domain} | Entity: {entity}
</context>

<new_theme>
- name: "{theme_name}"
- description: "{theme_description}"
- Included expressions (must be covered):
  {inclusion}
</new_theme>

<taxonomy>
This codebook covers the {facet_name} dimension.
Scope check: codes must capture concepts of type {facet_valid_labels}.
Out of scope: {facet_invalid_labels}.
Note: these concept types define what BELONGS here, not how to WORD labels.
Facet instruction: {facet_description}
</taxonomy>

---

LABEL RULES
- Concise noun phrase, 1-5 words (maximum 8 if necessary).
- One semantic head noun; modifiers allowed.
- No conjunctions (and/or), slashes, commas, or multi-concept bundles.
- Do NOT repeat {perspective}, {domain}, {topic}, or {entity}.
- Name the SPECIFIC content, not the facet dimension. The taxonomy already establishes that codes are about {facet_name}. Labels name WHAT was experienced/perceived/suggested — not THAT it was experienced/perceived/suggested.
- WRONG: "Experience of musical enjoyment", "Perception of atmosphere", "Feeling of togetherness"
- RIGHT: "Musical enjoyment", "Festival atmosphere", "Togetherness"

DEFINITION RULES
- 30 words or fewer.
- Describe observable assignment cues (what respondents say/describe).
- No causes, motives, conditions, or outcomes.
- Do NOT repeat {perspective}, {domain}, {topic}, or {entity}.
- Patterns: "References to...", "Mentions of...", "Expressions of..."

ASSIGNMENT EXAMPLES
- inclusion: 2-3 expressions that SHOULD be coded here.
- exclusion: 1-2 boundary cases that should NOT.
- near_neighbor: closest confusable theme + one distinguishing sentence.

---

Use theme_number and theme_name exactly as provided. Set source_code to null. Write all values in {language}. Provide your output as valid JSON following the response schema provided.
"""

# -----------------------------------------------------------------------------
# 3b. CODING_MODIFICATION_PROMPT
# -----------------------------------------------------------------------------

HORIZONTAL_INSTRUCTIONS = """
- Keep the original code's abstraction level.
- Find the single atomic shared concept that covers both the original code
  and the new theme within their shared conceptual family.
- The modified label must reflect the broadened meaning without introducing
  multiple aspects or becoming more abstract than necessary.
- The modified definition must describe the shared meaning space,
  incorporating original inclusions + inclusion_update and
  original exclusions + exclusion_update."""

VERTICAL_INSTRUCTIONS = """
- Same conceptual family but different abstraction levels: create hierarchical structure.
- Original code and new theme remain atomic child codes.
- Parent code represents the shared conceptual family.

Parent label:
  - If {parent_theme_label} is provided, use it as-is.
  - Otherwise, generate a label at a higher abstraction level.
  - Must express shared conceptual family, not blend child labels.
  - Must be broader, not vaguer.

Structure: Parent = conceptual anchor (broader), Children = distinct manifestations (narrower).
Child meanings do not change."""


CODING_MODIFICATION_PROMPT = """
You are a {language} qualitative research assistant. Your task: MODIFY an existing code to include a new theme within the {facet_name} facet, preserving atomic meaning and clear boundaries.

THEME PRINCIPLE (Braun & Clarke, 2006)
A code is a meaning-based organizing concept, not a topic label. The modified code must remain atomic (one semantic nucleus) and within the {facet_name} dimension.

---

PARAMETERS

<context>
Respondents were asked: "{survey_question}"
Reframe as: What meanings do {perspective}s actually express about {entity} regarding {topic}?
Language: {language}
Domain: {domain} | Entity: {entity}
</context>

<new_theme>
- name: "{theme_name}"
- description: "{theme_description}"
- Included expressions (must be covered):
  {inclusion}
</new_theme>

<original_code>
- code_label: {source_code}
- code_definition: {source_definition}
</original_code>

<current_assignment_examples>
Current inclusion examples:
  {current_inclusion}
</current_assignment_examples>

<required_modifications>
- inclusion_update: {inclusion_update}
- exclusion_update: {exclusion_update}
</required_modifications>

<taxonomy>
This codebook covers the {facet_name} dimension.
Scope check: codes must capture concepts of type {facet_valid_labels}.
Out of scope: {facet_invalid_labels}.
Note: these concept types define what BELONGS here, not how to WORD labels.
Facet instruction: {facet_description}
</taxonomy>

---

MODIFICATION INSTRUCTIONS
{modification_instructions}

---

LABEL RULES
- Concise noun phrase, 1-5 words (maximum 8 if necessary).
- One semantic head noun; modifiers allowed.
- No conjunctions (and/or), slashes, commas, or multi-concept bundles.
- Do NOT repeat {perspective}, {domain}, {topic}, or {entity}.
- Name the SPECIFIC content, not the facet dimension. The taxonomy already establishes that codes are about {facet_name}. Labels name WHAT was experienced/perceived/suggested — not THAT it was experienced/perceived/suggested.
- WRONG: "Experience of musical enjoyment", "Perception of atmosphere", "Feeling of togetherness"
- RIGHT: "Musical enjoyment", "Festival atmosphere", "Togetherness"

DEFINITION RULES
- 30 words or fewer.
- Describe observable assignment cues (what respondents say/describe).
- No causes, motives, conditions, or outcomes.
- Do NOT repeat {perspective}, {domain}, {topic}, or {entity}.
- Patterns: "References to...", "Mentions of...", "Expressions of..."

ASSIGNMENT EXAMPLES
- inclusion: Combine original + new expressions from inclusion_update.
- exclusion: Combine original + new boundaries from exclusion_update.
- near_neighbor: Update if boundaries changed. Identify closest confusable concept.

---

Use theme_number, theme_name, and source_code exactly as provided. Write all values in {language}. Provide your output as valid JSON following the response schema provided.
"""


class GeneratedCode(BaseModel):
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
    source_code: Optional[str] = Field(
        default=None,
        description="Existing code being modified (null for CREATE)",
        examples=["Service Speed", None]
    )
    code_label: str = Field(
        ...,
        max_length=60,
        description="New or modified code label (1-5 word meaning-based noun phrase)",
        examples=["Waiting Time", "Service Response"]
    )
    code_definition: str = Field(
        ...,
        max_length=250,
        description="<=30-word operational definition describing what belongs in this code",
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

    @field_validator('code_label')
    @classmethod
    def validate_label_length(cls, v):
        word_count = len(v.split())
        if word_count > 10:
            raise ValueError(f"code_label must be <=10 words, got {word_count}")
        return v

    @field_validator('code_definition')
    @classmethod
    def validate_definition_length(cls, v):
        word_count = len(v.split())
        if word_count > 36:
            raise ValueError(f"code_definition must be <=36 words, got {word_count}")
        return v


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

Validate that the existing code fully captures this theme's meaning.
- Does the code's definition cover all expressions in the new theme?
- Would assigning this theme lose any meaningful distinctions?
- Are there uncovered expressions?

If FAILS: recommend MODIFY (horizontal or vertical) or CREATE.
"""


MODIFY_HORIZONTAL_VALIDATION_INSTRUCTIONS = """
**Scenario: MODIFY_HORIZONTAL (broaden at same abstraction level)**

Validate that the modification broadens scope while preserving the semantic core.
- Does the new label preserve the original code's central meaning?
- Do BOTH original and new expressions fit under the unified meaning?
- Is this genuinely broadening, not replacing the concept?

CRITICAL: If the label shifts the core concept -> REJECT, recommend CREATE.
"""

MODIFY_VERTICAL_VALIDATION_INSTRUCTIONS = """
**Scenario: MODIFY_VERTICAL (create parent at higher abstraction level)**

Validate that the hierarchical structure is well-formed.
- Is the parent abstract enough to encompass both children?
- Does the parent represent the shared family, not a label blend?
- Do children remain atomic and distinct?
- Is the abstraction difference genuine, not just wording variation?

If FAILS: recommend MODIFY_HORIZONTAL or CREATE.
"""

CREATE_VALIDATION_INSTRUCTIONS = """
**Scenario: CREATE new code**

Validate that this theme represents a genuinely novel concept.
- Is there truly no existing code that covers this theme?
- Does this fill a real gap, not just a wording preference?
- Is the new code sufficiently different from ALL existing codes?

If FAILS: recommend USE or MODIFY.
"""

VALIDATION_PROMPT = """
You are a codebook curator for thematic analysis following Braun & Clarke (2006) methodology. Your role: maintain a parsimonious codebook with non-overlapping, non-redundant codes by reviewing coding proposals.

THEME PRINCIPLE (Braun & Clarke, 2006)
A code is a meaning-based organizing concept, not a topic label. It captures WHAT UNIFIES responses -- the shared interpretive thread that explains WHY expressions belong together.

---

CONTEXT

<codebook_context>
Respondents were asked: "{survey_question}"
Reframe as: What meanings do {perspective}s actually express about {entity} regarding {topic}?
Domain: {domain} | Topic: {topic}

Existing codes in codebook:
{code_text}
</codebook_context>

<coding_proposal>
Theme to evaluate:
- name: "{theme_name}"
- description: "{theme_description}"

Proposal to review:
{step3_recommendation}

Expressions the code should cover:
  {inclusion_examples}
</coding_proposal>

<taxonomy>
This codebook covers the {facet_name} dimension.
Scope check: codes must capture concepts of type {facet_valid_labels}.
Out of scope: {facet_invalid_labels}.
Note: these concept types define what BELONGS here, not how to WORD labels.
Facet instruction: {facet_description}
</taxonomy>

---

SCENARIO-SPECIFIC VALIDATION

{validation_instructions}

---

CODE QUALITY RULES (apply to all scenarios)
A code must be:
- TAXONOMIC: falls within the {facet_name} scope.
- ATOMIC: one indivisible concept; unsplittable into independently meaningful parts.
- SINGLE-VALUED: one clear concept, no blending from different families.
- ACTIONABLE: directly identifiable and relevant to the research intent.
No conjunctions, slashes, or compound constructions in labels.

---

VALIDATION STEPS

1. Apply the scenario-specific validation questions above.
2. If the proposal fails any criterion, determine the correct decision (USE / MODIFY_HORIZONTAL / MODIFY_VERTICAL / CREATE).
3. Produce a final code (label + definition + assignment examples) per the rules below.

LABEL RULES
- Concise noun phrase, 1-5 words (maximum 8 if necessary).
- One semantic head noun; modifiers allowed.
- No conjunctions (and/or), slashes, commas, or multi-concept bundles.
- Do NOT repeat {perspective}, {domain}, {topic}, or {entity}.
- Name the SPECIFIC content, not the facet dimension. The taxonomy already establishes that codes are about {facet_name}. Labels name WHAT was experienced/perceived/suggested — not THAT it was experienced/perceived/suggested.
- WRONG: "Experience of musical enjoyment", "Perception of atmosphere", "Feeling of togetherness"
- RIGHT: "Musical enjoyment", "Festival atmosphere", "Togetherness"

DEFINITION RULES
- 30 words or fewer.
- Describe observable assignment cues (what respondents say/describe).
- No causes, motives, conditions, or outcomes.
- Do NOT repeat {perspective}, {domain}, {topic}, or {entity}.
- Patterns: "References to...", "Mentions of...", "Expressions of..."

---

OUTPUT INSTRUCTIONS

- Use theme_number and theme_name exactly as provided in the proposal.
- source_code: USE = exact code from proposal; MODIFY = exact existing code; CREATE = null.
- Write all values in {language}.
- Provide your output as valid JSON following the response schema provided.
"""


class ValidatedCode(BaseModel):
    code: str = Field(
        ...,
        max_length=60,
        description="Final validated code label (<=8 words, rule-compliant)",
        examples=["Waiting Time", "Service Response"]
    )
    definition: str = Field(
        ...,
        max_length=250,
        description="Final validated definition (<=30 words, operational, grounded)",
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

    @field_validator('code')
    @classmethod
    def validate_label_length(cls, v):
        word_count = len(v.split())
        if word_count > 10:
            raise ValueError(f"code label must be <=10 words, got {word_count}")
        return v

    @field_validator('definition')
    @classmethod
    def validate_definition_length(cls, v):
        word_count = len(v.split())
        if word_count > 36:
            raise ValueError(f"definition must be <=36 words, got {word_count}")
        return v


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
