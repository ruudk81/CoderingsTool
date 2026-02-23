"""
Prompts module - Contains all LLM prompt templates for the pipeline.
"""

from typing import Any, List, Optional, Union, Literal
from pydantic import BaseModel, Field, model_validator


# =============================================================================
# STEP 1: SPELL CHECKING
# =============================================================================

SPELLCHECK_INSTRUCTIONS = """
You are a {language} language expert specializing in correcting misspelled words in open-ended survey responses.
Your task is to process correction tasks for responses that contain placeholder tokens indicating spelling mistakes.

First, here is the survey question that the responses are answering:
<survey_question>
{var_lab}
</survey_question>

For each correction task, you will receive:
- A sentence with one or more <oov_word> placeholders
- A list of misspelled words, in the same order as the placeholders
- A list of suggested corrections, in the same order

Follow these rules when making corrections:
1. Replace each <oov_word> placeholder with the best possible correction of the corresponding misspelled word.
2. Consider the meaning and context of the survey question when choosing corrections.
3. If a better correction exists than the ones provided, use that instead.
4. You may split a misspelled word into two words only if the split preserves the intended meaning and fits grammatically.
5. If no suitable correction is possible, use "[NO RESPONSE]" as the corrected sentence for that task.

Here are the correction tasks to process:
<correction_tasks>
{tasks}
</correction_tasks>

Additional guidelines:
- Pay close attention to the context and meaning of each response when making corrections.
- Ensure that your corrections maintain the original intent of the respondent.
- If a suggested correction doesn't fit the context, consider alternative corrections that preserve the meaning.

Begin processing the correction tasks now and provide your output as valid JSON following the response schema provided.
"""


class CorrectionItem(BaseModel):
    """A single spell correction result."""
    respondent_id: Any = Field(
        description="The respondent ID from the correction task"
    )
    corrected_response: str = Field(
        description="The fully corrected response with all spelling mistakes fixed"
    )


class LLMCorrectionResponse(BaseModel):
    """Structured output for spell check corrections."""
    corrections: List[CorrectionItem] = Field(
        description="List of corrections, one for each task in the input"
    )

# =============================================================================
# STEP 2: QUALITY FILTERING 
# =============================================================================

GRADER_INSTRUCTIONS = """
You are a {language} language grader evaluating open-ended survey responses.
Your task is to determine whether each response provides **usable, on-topic content in relation to the specific survey question**, and assign appropriate quality filter codes.

You will classify each response into one of three practical outcomes:

==================================================
OUTCOME A — Don't Know / Uncertainty
CODE: 99999997 | quality_filter = true
==================================================

Use this code if the respondent explicitly expresses uncertainty, lack of knowledge, or non-applicability.

This includes any response whose clear meaning in {language} is equivalent to:
- "I don't know"
- "N/A"
- "Not applicable"
- "No idea"
- "Unsure"
- "Can't say"
- "?"

If a response fits this pattern → set:
quality_filter = true
quality_filter_code = 99999997

==================================================
OUTCOME B — Nonsensical/Gibberish OR completely Off-topic
CODE: 99999999 | quality_filter = true
==================================================

Use this code for **two different kinds of unusable responses**:

A) Pure gibberish / nonsensical
   Examples:
   - Random characters: "asdfkj", "jjjjj", "x!@#%"
   - Placeholder text: "lorem ipsum", "test test"
   - Verbatim repetition of the question with no added content
   - Completely unintelligible text

B) Intelligible but completely off-topic / totally irrelevant
   The response is understandable {language} , BUT:
   - It does NOT address the actual survey question ({var_lab}) even remotely, OR
   - It obviously avoids the question.

Illustractive examples in English (if the question is about public transport):
- "Nothing"
- "I love dogs."
- "The weather is nice today."
- "Pizza is better than pasta."
- "I work in finance."
- A personal story that has nothing to do with transportation.

These are NOT "I don't know" — they are simply irrelevant to the question.

If a response fits **either A or B** → set:
quality_filter = true
quality_filter_code = 99999999


==================================================
OUTCOME C — Meaningful / On-topic Response
quality_filter = false | quality_filter_code = null
==================================================

A response is meaningful if:
- It is understandable in {language}, AND
- It engages with or relates to the survey question ({var_lab}), even if:
  - It is very short
  - It is vague
  - It is opinionated
  - It is critical
  - It is poorly written
  - It is partially incomplete

Examples (if the question is about public transport):
- "Buses are always late."
- "Too crowded."
- "Tickets are expensive."
- "The metro is unreliable."

For this category → set:
quality_filter = false
quality_filter_code = null

==================================================
SURVEY QUESTION
<survey_question>
{var_lab}
</survey_question>

RESPONSES TO EVALUATE
<responses>
{responses}
</responses>

==================================================
DECISION RULE (FOLLOW EXACTLY)

For each response, apply these steps in order:

1. Does the response explicitly express uncertainty or "I don't know"?
   - If YES → quality_filter = true, quality_filter_code = 99999997
   - If NO → go to Step 2

2. Ask:
   "Does this response provide usable content that addresses the survey question, even remotely?"

   - If NO (because it is gibberish OR off-topic) →
     quality_filter = true, quality_filter_code = 99999999

   - If YES →
     quality_filter = false, quality_filter_code = null
"""


QualityCode = Optional[Literal[99999997, 99999999]]

class QualityFilterLLMResponse(BaseModel):
    """A single quality filter assessment result."""

    respondent_id: Any = Field(
        description="The respondent's ID from the input (preserve exact type and format)"
    )

    response: Union[str, float, int, None] = Field(
        description="The exact response text being evaluated"
    )

    quality_filter: bool = Field(
        description=(
            "true if the response is unusable (don't know OR gibberish/off-topic), "
            "false if the response is meaningful and addresses the question"
        ),
        examples=[True, False],
    )

    quality_filter_code: QualityCode = Field(
        default=None,
        description=(
            "99999997 = uncertainty / don't know; "
            "99999999 = gibberish OR completely off-topic; "
            "null = meaningful response"
        ),
        examples=[99999997, 99999999, None],
    )

    @model_validator(mode="after")
    def check_consistency(self):
        if self.quality_filter and self.quality_filter_code is None:
            raise ValueError(
                "If quality_filter=true, quality_filter_code must be 99999997 or 99999999"
            )
        if not self.quality_filter and self.quality_filter_code is not None:
            raise ValueError(
                "If quality_filter=false, quality_filter_code must be null"
            )

        return self


# =============================================================================
# STEP 3: IDEA EXTRACTION  
# =============================================================================

CONTEXT_SPECIFIER_PROMPT1 = """
You are analyzing survey responses to extract contextual metadata.

Survey question: {survey_question}

Sample responses ({chunk_size} examples):
{chunk_responses}

Extract these GROUP 1 specifiers (speaker characteristics):

1. **lang**: Language/dialect code
   - Identify the primary language and any dialect/regional variations
   - Format: ISO code 
   - Examples: "nl-NL" (Dutch Netherlands), "en-GB" (British English)

2. **perspective**: Stakeholder viewpoint
   - From whose perspective are these responses given?
   - Common values: "consumer", "employee", "partner", "expert", "general_public"
   - Examples: "consumer" (customer feedback), "employee" (internal survey)

3. **intent**: Purpose/communicative function
   - What are respondents trying to do with their responses?
   - Common values: "evaluate", "describe", "suggest", "complain", "praise", "question"
   - Examples: "evaluate" (assessing brand), "suggest" (recommendations)

Provide concise answers (2-5 words each) in {language}.""" 


CONTEXT_SPECIFIER_PROMPT2 = """
You are analyzing survey responses to extract contextual metadata.

Survey question: {survey_question}

Sample responses ({chunk_size} examples):
{chunk_responses}

Extract these GROUP 2 specifiers (subject matter):

1. **domain**: Industry/sector domain
   - What industry or sector does this survey concern?
   - Examples: "finance" (banking survey), "healthcare" (hospital satisfaction)

2. **topic**: Specific subject matter
   - What is the specific topic being discussed?
   - Examples: "brand_association" (brand perception), "customer_service" (support experience)

3. **entity**: Main entity/subject
   - What specific organization, product, or brand is the primary focus?
   - Use lowercase with underscores for multi-word names
   - Examples: "asn_bank", "tesla_model_3", "albert_heijn", "ns_trains"

Provide concise answers (2-5 words each) in {language}."""  


CONSOLIDATE_SPECIFIERS_GROUP1 = """
You are consolidating contextual metadata extracted from multiple chunks of survey responses.

Survey question: {survey_question}

Different chunks produced these GROUP 1 specifiers (speaker characteristics):

{chunk_results}

Your task: Consolidate these into ONE canonical set of specifiers.

Guidelines:
- Resolve semantic variations (e.g., "evaluative" vs "assessment viewpoint" → choose most accurate)
- For **lang**: Standardize to ISO format (e.g., "Dutch" → "nl-NL", "English" → "en-US")
- For **perspective**: Choose the most representative viewpoint across all chunks
- For **intent**: Choose the most common communicative goal

If chunks agree: use the consensus value
If chunks disagree: choose the most frequently occurring concept (semantic similarity, not lexical match)

Return ONE consolidated set of GROUP 1 specifiers."""


CONSOLIDATE_SPECIFIERS_GROUP2 = """
You are consolidating contextual metadata extracted from multiple chunks of survey responses.

Survey question: {survey_question}

Different chunks produced these GROUP 2 specifiers (subject matter):

{chunk_results}

Your task: Consolidate these into ONE canonical set of specifiers.

Guidelines:
- Resolve semantic variations (e.g., "financial services" vs "banking sector" → choose most accurate)
- For **domain**: Standardize to lowercase, single/hyphenated word
- For **topic**: Choose the most representative subject matter across all chunks
- For **entity**: Standardize format (lowercase_with_underscores)

If chunks agree: use the consensus value
If chunks disagree: choose the most frequently occurring concept (semantic similarity, not lexical match)

Return ONE consolidated set of GROUP 2 specifiers."""

TAXONOMY_CHUNK_SCORING_PROMPT = """
You are analyzing a chunk of survey responses to determine the SINGLE PRIMARY TAXONOMY AXIS that should be used to generate MECE (Mutually Exclusive and Collectively Exhaustive) descriptive codes.

Your output will be used downstream to consolidate multiple chunks into a single global code spine, so precision and consistency matter.

---

## INPUTS

Survey question (in {language}):

<survey_question>
{survey_question}
</survey_question>

Survey responses to analyze (in {language}):

<chunk_responses>
{chunk_responses}
</chunk_responses>

This chunk contains {chunk_size} responses.

---

## YOUR TASK

Identify ONE dominant organizing principle (taxonomy axis) that best differentiates these responses such that a code set built on it will be:

- Mutually exclusive (no overlap between codes)
- Collectively exhaustive (all responses can be coded)

You must also assess how strongly each possible coding dimension applies.

---

## CORE PRINCIPLE

Base your decision on COMMUNICATIVE MEANING — what respondents are primarily trying to convey — not on grammatical form, phrasing, or surface-level keywords.

Ask: *What kind of difference between responses actually matters for understanding what people are saying?*

---

## SIX POSSIBLE CODING DIMENSIONS

Evaluate all six, then select the ONE that produces the cleanest MECE partition for this specific question and response set:

1) **WHAT** — attributes, features, characteristics, properties, or aspects of something  
2) **WHY** — motivations, goals, reasons, desired outcomes, or value sought  
3) **HOW** — actions, behaviors, methods, processes, or implementation approaches  
4) **WHO** — people, groups, roles, stakeholders, or beneficiaries  
5) **WHEN** — timing, urgency, sequence, frequency, or temporal aspects  
6) **WHERE** — location, channel, setting, or situational context  

---

## DEFINING THE PRIMARY TAXONOMY AXIS

Follow these steps:

1. Understand what the survey question is asking
2. Review the responses and identify how they meaningfully differ
3. Evaluate each of the six dimensions as a potential organizing principle
4. Select the dimension that yields the cleanest MECE partition
5. Define a specific taxonomy axis within that dimension

**Important constraints:**
- You must select ONLY ONE primary dimension
- The taxonomy axis you define must clearly fall within the selected dimension
- The axis should be phrased so that code labels can be directly created from it

---

## OUTPUT FORMAT

After reasoning in the scratchpad, return the following structured output.
All content must be written in {language}.

<analysis>

<primary_dimension>
WHAT | WHY | HOW | WHO | WHEN | WHERE
</primary_dimension>

<taxonomy_axis>
In 1–2 sentences, define the specific organizing axis within the chosen dimension.
Phrase it so downstream code labels can be directly derived from it.
</taxonomy_axis>

<dimension_scores>
Provide a confidence score from 0.0 to 1.0 for EACH dimension, reflecting how well it could organize these responses.
Scores do not need to sum to 1.

WHAT: float
WHY: float
HOW: float
WHO: float
WHEN: float
WHERE: float
</dimension_scores>

<evidence>
Provide 2–3 brief verbatim snippets from the responses that demonstrate why the selected axis is dominant.
</evidence>

<mece_check>
In 1–2 sentences, explain why codes built on this axis will minimize overlap and fully cover the response set.
</mece_check>

</analysis>
"""

TAXONOMY_CONSOLIDATION_PROMPT = """
You are consolidating multiple chunk-level taxonomy analyses into a SINGLE GLOBAL TAXONOMY AXIS.
Each chunk analysis evaluated the same survey question and produced:
- a primary coding dimension
- a specific taxonomy axis
- confidence scores for all six dimensions
- supporting evidence

Your job is to determine the best overall organizing principle for building a MECE
(Mutually Exclusive, Collectively Exhaustive) descriptive codebook across ALL responses.

---

## INPUTS

Survey question:
{survey_question}

Chunk-level analyses:
{chunk_results}

---

## YOUR TASK

### 1. Aggregate dimension evidence
- Compute evidence-weighted average scores for each of the six dimensions
- Use each chunk’s dimension_scores as inputs
- Weight scores by the number of responses in each chunk (e.g., chunk_size or evidence_count)

### 2. Select the PRIMARY taxonomy dimension
Choose the ONE dimension that:
- Has strong and consistent weighted scores across chunks
- Captures the dominant communicative meaning in the full response set
- Produces the cleanest MECE partition when used as a code spine

Important:
- Do NOT select a dimension solely because it appears most frequently
- Favor partition clarity and interpretability over raw score dominance

### 3. Define the GLOBAL taxonomy axis (code spine)
Write a context-specific taxonomy axis description that:
- Is specific to THIS survey question and response domain
- Clearly falls within the selected primary dimension
- Operates at a mid-level of abstraction (not too narrow, not too broad)
- Can directly seed downstream code labels
- Indicates what coders should extract from each response

### 4. Evaluate the need for a SECONDARY dimension (optional)
Only select a secondary dimension if ALL of the following are true:
- It is orthogonal to the primary dimension
- It captures meaningful variation not represented by the primary axis
- A single response would not typically require multiple secondary codes
- It does not introduce overlap with primary-axis codes

If no such dimension exists, set secondary_dimension to null.

---

## DECISION RULES

- If chunk analyses converge, follow the consensus
- If chunk analyses diverge, rely on evidence-weighted dimension scores
- Optimize for MECE quality and downstream coding usability
- Prefer clarity and stability over cleverness

---

## OUTPUT FORMAT

Return JSON ONLY. Do not include explanations outside the JSON object.

{{
  "primary_dimension": "WHAT | WHY | HOW | WHO | WHEN | WHERE",
  "primary_dimension_rationale": "Brief explanation of why this dimension is dominant",
  "primary_dimension_description": "Definition of the taxonomy axis (code spine) at proper abstraction level",
  "primary_dimension_score": float,
  "secondary_dimension": "WHAT | WHY | HOW | WHO | WHEN | WHERE" or null,
  "secondary_dimension_rationale": string or null,
  "all_dimension_scores": {{
     "WHAT": float,
     "WHY": float,
     "HOW": float,
     "WHO": float,
     "WHEN": float,
     "WHERE": float
  }}
}}
"""


TAXONOMY_CONSOLIDATION_PROMPT = """
You are consolidating chunk-level taxonomy analyses.
Each chunk summary identified which coding dimension best organizes its responses.
Your job is to determine the SINGLE PRIMARY TAXONOMY DIMENSION overall — the organizing axis that will be used to build a MECE (Mutually Exclusive, Collectively Exhaustive) descriptive codebook.

<inputs>
Survey question: {survey_question}

Chunk results:
{chunk_results}
</inputs>

## YOUR TASK

1. Aggregate dimension scores across all chunks
   - Compute evidence-weighted average scores using evidence_count as weights

2. Select the PRIMARY dimension
   - Choose the dimension that best serves as the global organizing principle
   - Prioritize the dimension that will yield the cleanest MECE partition across all responses
   - Do not select a dimension merely because it appears frequently; it must meaningfully differentiate responses

3. Write a CONTEXT-SPECIFIC TAXONOMY AXIS description
   - Define the code-spine that downstream coding should follow
   - Must be at a mid-level of abstraction: not too narrow, not too broad

4. Optionally select a SECONDARY dimension
   - Only if it is truly orthogonal to the primary dimension
   - Must represent an independent organizing principle, not a sub-type of the primary
   - Otherwise set to null

## PRIMARY DIMENSION SELECTION CRITERIA

- High and consistent weighted scores across chunks
- Captures the dominant communicative meaning in responses
- Produces non-overlapping, actionable coding categories
- Enables full coverage of response variation

## TAXONOMY AXIS DESCRIPTION GUIDELINES

Write 1–2 sentences that:
- Are specific to THIS survey question and response domain
- State the category-type by which responses differ
- Use wording that can directly seed code labels
- Indicate what downstream coders should extract

## SECONDARY DIMENSION RULES

- Must add an independent perspective not already captured by the primary
- Must not cause code overlap with primary-axis codes
- If no such dimension exists, return null

## CONSOLIDATION RULES

- If chunk results converge: follow the consensus
- If chunk results diverge: rely on evidence-weighted averages
- Favor MECE partition quality over raw score dominance

## RETURN JSON ONLY

{{
  "primary_dimension": "WHAT | WHY | HOW | WHO | WHEN | WHERE",
  "primary_dimension_rationale": "Brief explanation of why this dimension is dominant",
  "primary_dimension_description": "Definition of the taxonomy axis (code spine) at proper abstraction level",
  "primary_dimension_score": float,
  "secondary_dimension": "WHAT | WHY | HOW | WHO | WHEN | WHERE" or null,
  "secondary_dimension_rationale": string or null,
  "all_dimension_scores": {{
     "WHAT": float,
     "WHY": float,
     "HOW": float,
     "WHO": float,
     "WHEN": float,
     "WHERE": float
  }}
}}
"""


TAXONOMY_AWARE_SUBJECT_PROMPT = """
You are a {language} language expert generating a phrasing template for survey response analysis.

<input>
Language: {language}
Survey question: {survey_question}
Primary taxonomy axis: {primary_axis} ({primary_axis_description})

Context (from prior analysis):
- Domain: {domain}
- Topic: {topic}
- Entity: {entity}
- Type of respondent: {perspective}
- Intent by response: {intent}
</input>

<definitions>
- **Canonical subject**: The main product, brand, service, actor, organization, location, topic, issue, or other subject the question is essentially about.
- **Actionable taxonomy dimension**: A specific dimension type within the primary taxonomy axis that guides code generation (e.g., for WHAT: "attributes" or "features").
- **Phrasing template**: A flexible sentence structure that can be used to extract ideas aligned with the actionable taxonomy dimension.
</definitions>

<actionable_taxonomy_dimensions>
"WHAT": "topic_object - concepts, things, topics, features, attributes"
"WHY": "intent_purpose - goals, desired outcomes, improvements, reasons"
"HOW": "action_method - actions, steps, processes, methods, ways"
"WHO": "actor_target - people, groups, stakeholders, beneficiaries"
"WHEN": "time_urgency - time references, urgency, sequence, timing"
"WHERE": "location_context - place, context, channel, location"
</actionable_taxonomy_dimensions>

## Task 1: Identify the canonical subject

- Find the main product, brand, service, actor, organization, location, topic, issue, or other subject the question is essentially about.
- Return a concise, normalized noun phrase in {language}.
- Preserve capitalization for proper nouns; otherwise use lowercase.
- If the survey question does not reference a concrete entity, select the most stable abstract noun phrase that responses consistently refer to (e.g., "the experience", "the process", "the service").

## Task 2: Choose the actionable taxonomy dimension

From <actionable_taxonomy_dimensions>, select the SINGLE most relevant dimension type for the {primary_axis} axis.
- Look at the options for {primary_axis}.
- Choose ONE specific term that best fits this survey's responses.
- Choose the term that best supports mutually exclusive, non-overlapping code labels.

## Task 3: Create a phrasing template

Use this flexible structure:
**"[CANONICAL_TERM] [VERB/STATE] [SCAFFOLDING_WORDS] [ACTIONABLE_TAXONOMY_DIMENSION]"**

Where:
- CANONICAL_TERM: the focus entity from Task 1
- VERB/STATE: appropriate verb in {language}
- SCAFFOLDING_WORDS: grammatical words needed for completeness (may be empty)
- [ACTIONABLE_TAXONOMY_DIMENSION]: placeholder for response content aligned with the selected dimension

### Axis-aware verb/state guidance

- **WHAT**: "has/heeft", "is characterized by/kenmerkt zich door"
- **WHY**: "should achieve/moet bereiken", "needs to provide/moet bieden"
- **HOW**: "should/moet", "can/kan", "needs to/moet"
- **WHO**: focus on the actor, e.g., "needs/heeft nodig", "should receive/moet krijgen"
- **WHEN**: include timing context in scaffolding
- **WHERE**: include location context in scaffolding

Choose the verb/state that sounds MOST NATURAL in {language}.

### Grammatical completeness constraint

The template MUST produce a natural language response that directly addresses the survey question when [ACTIONABLE_TAXONOMY_DIMENSION] is replaced with a single adjective or noun phrase.

Test your template with a one-word or short noun-phrase example.

Output format (return only this JSON object):
{
  "canonical_term": "main subject/entity from the survey question in {language}",
  "taxonomy_axis": "{primary_axis}",
  "taxonomy_actionable_type": "chosen single dimension (e.g., 'attributes' or 'features')",
  "canonical_phrasing": "natural template ending with [ACTIONABLE_TAXONOMY_DIMENSION] in {language}"
}
"""


TAXONOMY_ENRICHED_EXTRACTION_PROMPT = """
You are an expert in extracting structured ideas from survey responses using taxonomy-aware analysis. Your task is to identify all ideas expressed in a survey response, classify them according to a given taxonomy axis, and format them according to a specific template structure.

Here is the survey context:

<survey_context>
language of responses: {language}
Survey question: {var_lab}

Domain: {domain}
Topic: {topic}

Type of respondent: {perspective}
Dominant response frame: {intent}
Object of intent: {entity}
</survey_context>

Here is the taxonomy configuration:

<taxonomy_config>
Taxonomy dimension to be used: {taxonomy_axis}: {taxonomy_actionable_type}
</taxonomy_config>

Here is the response you need to process:

<response>
Respondent ID: {respondent_id}
Response: {response}
</response>

For each idea you identify, you must extract the following information:

**1. respondent_id**: Use the respondent ID provided in the response context.

**2. idea_id**: Assign a sequential number as a string (e.g., "1", "2", "3") for each idea extracted from this response.

**3. idea**: The complete formatted idea statement. This MUST follow these rules:
   - Begin with EXACTLY this template provided: {template_prefix}
   - Then replace the placeholder ("[{taxonomy_actionable_type}]") with idea identified in the response.
   - The replacement text must be a complete, grammatical sentence fragment (5-20 words total)
   - Must directly address the survey question when combined with the template prefix
   - Must be concise and specific
   - Must NOT contain pronouns or references to the respondent
   - Must NOT contain filler words or unnecessary phrases
   - Must be written in the {language} specified in the survey context  

**4. ontology**
For each idea, identify its position in a conceptual hierarchy for the given taxonomy dimension.

You must extract:

- **instance**: the literal action / object / concept expressed in the idea (verbatim, no paraphrasing)
- **node**: the canonical, reusable ontology concept instantiated by the instance (noun phrase)
- **category**: the immediate parent grouping of the node
- **root**: the top-level domain framing implied by the research question and taxonomy dimension

Rules:
- The instance MUST be a contiguous span from the idea text (no rewording or abstraction).
- The node MUST be reusable across multiple responses.
- Category and root MUST be consistent with the taxonomy axis ({taxonomy_axis}) and actionable type ({taxonomy_actionable_type}).
- Do NOT repeat the idea text verbatim at all levels.
- Prefer stable, domain-relevant concepts over stylistic paraphrases.
- If multiple interpretations exist, choose the primary one implied by the survey context.

Write all ontology fields in {language}.

**5. taxonomy_phrase**: - A concise noun-phrase that abstracts the idea into a reusable {taxonomy_actionable_type}-category on the taxonomy axis ({taxonomy_axis}: {taxonomy_axis_description})
   - Make the semantic core of the taxonomy_phrase the HEAD of the noun phrase
   - DO NOT repeat entities already mentioned in the domain context: {domain}, {topic} and {entity}
   - Prefer single-word attribute nouns over compound action-nouns
     * Example: Instead of "price reduction" use "price level"
     * Example: Instead of "assortment expansion" use "assortment variety"
   - Avoid meta-language about perception, opinion, or thought
   - Avoid verbs or verb-noun compounds
   -  Written in {language}.

**6. sentiment**: Choose exactly one of: positive | negative | neutral
   - positive = praise, approval, satisfaction
   - negative = complaint, dissatisfaction, criticism
   - neutral = suggestion without judgment, factual mention

**7. sense**: Choose exactly one of: factual | evaluative | aspirational | experiential
   - factual = objective statement of fact
   - evaluative = judgment or assessment
   - aspirational = desire or wish for something
   - experiential = description of personal experience

**IDEA SPLITTING RULES:**

When a response contains multiple conceptually distinct aspects, split them into separate ideas. Each atomic concept should be extracted as its own idea with its own idea_id.

Example: If a response mentions both "sustainable investment options" and "lower fees", extract these as two separate ideas because sustainability and fees are conceptually distinct aspects.


**EXAMPLE:**

Template: {canonical_phrasing}
Template prefix: "{template_prefix}"

Response: "I'd love more shaded seating areas and better evening lighting for safety."

Extracted ideas:
[
  {{
    "respondent_id": "{respondent_id}",
    "idea_id": "1",
    "idea": "{template_prefix} more shaded seating areas",
    "taxonomy_phrase": "shaded seating",
    "ontology": {{
      "instance": "more shaded seating areas",
      "node": "shade provision",
      "category": "amenity design changes",
      "root": "environmental intervention"
    }},
    "sentiment": "positive",
    "sense": "aspirational"
  }},
  {{
    "respondent_id": "{respondent_id}",
    "idea_id": "2",
    "idea": "{template_prefix} better evening lighting",
    "taxonomy_phrase": "evening lighting",
    "ontology": {{
      "instance": "better evening lighting",
      "node": "lighting improvement",
      "category": "amenity design changes",
      "root": "environmental intervention"
    }},
    "sentiment": "positive",
    "sense": "aspirational"
  }}
]

OUTPUT FORMAT

Return a JSON array. Each item:

{{
  "respondent_id": "{respondent_id}",
  "idea_id": "sequential number as string",
  "idea": "{template_prefix} [specific {taxonomy_actionable_type}]",
  "taxonomy_phrase": string,
  "ontology": {{
      "instance": string,
      "node": string,
      "category": string,
      "root": string 
      }}
  "sentiment": "positive|negative|neutral",
  "sense": "factual|evaluative|aspirational|experiential"
}}

Edge cases:
- Empty or irrelevant response: return []
- Single idea: return one item
- Multiple ideas: return multiple items with sequential idea_id

CRITICAL: Make the semantic core of the taxonomy_phrase the head of the noun phrase.

Return ONLY the JSON array. Field names in English; text values in {language}.
"""

# Helper dict for taxonomy axis descriptions
TAXONOMY_AXIS_DESCRIPTIONS = {
    "WHAT": "topic_object - concepts, things, topics, features, attributes",
    "WHY": "intent_purpose - goals, desired outcomes, improvements, reasons",
    "HOW": "action_method - actions, steps, processes, methods, ways",
    "WHO": "actor_target - people, groups, stakeholders, beneficiaries",
    "SENTIMENT": "evaluation - judgment, opinion, positive/negative evaluation",
    "WHEN": "time_urgency - time references, urgency, sequence, timing",
    "WHERE": "location_context - place, context, channel, location"
}

# ============================================================================
# STEP 5: CLUSTER LABEL GENERATION
# =============================================================================

CLUSTER_DESCRIPTION_PROMPT = """You are a qualitative researcher labeling survey-response clusters.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>
<instruction>The theme label must read as a natural-language answer category to the survey question.</instruction>
{taxonomy_context}
<cluster_evidence>
Cluster ID: {cluster_id}
Number of {sample_type}: {num_ideas}

<representative_{samples_tag}>
These {sample_type} are representative of the cluster:
{ideas_list}
</representative_{samples_tag}>
{keywords_section}{cluster_profile_section}
</cluster_evidence>

<task>
1. Review the representative {sample_type} to identify common meaning.
2. Use the statistical keywords to sharpen what makes this cluster distinct.
3. Identify the common atomic theme expressed directly in the data.
4. Do not introduce concepts not supported by the {sample_type} or keywords.
5. Ensure the theme stays strictly within the taxonomy dimension{taxonomy_task_guidance}.
6. Ensure the theme reads as a short, noun-phrased natural-language answer to the survey question. Use the essence as the head noun, avoid generic language, clutter and verbs.
</task>

<output_format>
Provide your analysis in {language}:
- theme: Short noun-phrased label{taxonomy_output_constraint} (3-10 words)
- description: 1-2 sentence explanation of what respondents associate with the entity
- key_concepts: 3-5 concrete concepts grounded in data (from keywords or representative samples)
</output_format>"""


class ClusterDescription(BaseModel):
    """LLM-generated cluster description (structured output model)."""
    theme: str = Field(..., description="Short noun-phrased thematic label (3-10 words), reads as answer to survey question")
    description: str = Field(..., description="1-2 sentence explanation of what respondents associate with the entity")
    key_concepts: List[str] = Field(..., description="3-5 concrete concepts grounded in data (from keywords or samples)")


# =============================================================================
#  Speculative codes
# =============================================================================

INITIAL_CODEBOOK_CREATION_PROMPT = """
You are an {language} expert qualitative data analyst specializing in rigorous thematic analysis and code creation. 
Your task is to generate hypothetical codes that might be encountered when analyzing written answers to a specific survey question.

Here are the critical coding principles you must follow:
- ATOMIC: Each code must capture ONE concept only - no compound ideas with "and", "including", "with"
- PRECISE: Clear boundaries that enable reliable coding decisions
- CONCISE: Code names must be 2-5 words maximum
- OPERATIONAL: Definitions must use observable criteria, not interpretations
- MUTUALLY EXCLUSIVE: Minimal overlap between codes

You will be working with the following inputs:
- Language to use: <language> {language} </language>
- Number of codes to generate: <n_codes> {n_codes} </n_codes>
- Survey question to analyze: <survey_question> {survey_question} </survey_question>

Your task is to generate {n_codes} diverse, hypothetical codes that might emerge from analyzing responses to the given survey question. Create codes that could apply to ANY survey topic. Do not assume the survey is about education, healthcare, or any specific domain. Let the survey question guide your code generation.

Consider different code types when generating your codes:
- Attribute codes: Qualities or characteristics mentioned
- Process codes: Actions, procedures, or methods described
- Relational codes: Interactions or connections between elements
- State codes: Conditions, situations, or circumstances
- Evaluative codes: Assessments, judgments, or opinions expressed

Provide your response in {language} as a JSON array of objects, where each object has "code" and "definition" fields. 
Here's an example of the structure to follow (using generic placeholders):
<example>
[
  {{"code": "Quality assessment", "definition": "References to evaluating the quality/characteristic of topic-specific element."}},
  {{"code": "Process difficulties", "definition": "Mentions of challenges in topic-specific process."}},
  {{"code": "Actor perspectives", "definition": "Expessions of viewpoints of relevant actors/participants."}}
]
</example>

Examples of well-structured code definitions:
- "References to [specific limitation or constraint] affecting [process or outcome]."
- "Mentions of [positive or negative] changes in [behavior or practice]."
- "Expressions of [emotion or attitude] regarding [situation or process]."

Avoid these weak definitions:
- Compound: "References to [issue A] including [aspect 1], [aspect 2], and [aspect 3]"
- Vague: "Mentions of various [things] related to [topic]"
- Interpretive: "Underlying [abstract concept] manifesting in different ways"


Return ONLY the JSON array in {language}. Do not include any additional text or explanations outside of the JSON array.
"""

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


CLUSTER_SUMMARY_PROMPT = """
You are a qualitative researcher responsible for extracting ATOMIC {taxonomy_actionable_type}-{theme_head} THEMES from descriptive codes representing survey responses to a survey question.
An ATOMIC {taxonomy_actionable_type}-{theme_head} THEME is a single, indivisible {taxonomy_actionable_type} or {theme_head} present in the data. 

Atomicity rules (must all be satisfied): 
- The theme expresses exactly ONE semantic nucleus. 
- The theme label contains exactly ONE head noun. 
- The label must NOT contain “and”, “or”, “/”, commas, or multiple content nouns. 
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
- Be a noun phrase of 1–3 words.
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
2. List 1–5 candidate {taxonomy_actionable_type}-{theme_head}s. Each must be a 1–3 word noun phrase satisfying atomic label rules.
3. If more than one {taxonomy_actionable_type}-{theme_head}s is found, treat them as separate potential atomic themes.
4. For each {taxonomy_actionable_type}-{theme_head}, verify grounding in cluster codes.
5. Do not merge {taxonomy_actionable_type}-{theme_head}s into umbrella or meta-parent concepts.
6. Select only {taxonomy_actionable_type}-{theme_head}s that are clearly supported by the data.
7. Produce final theme entries only for valid atomic {taxonomy_actionable_type}-{theme_head}s.

---

FINAL OUTPUT FORMAT

After analysis, output valid JSON in the following structure.
Field names must be in English. Values must be written in {language}.

{{
  "cluster_id": "{cluster_id}",
  "analysis": "Document your analysis here. State how many COCs were identified and retained. If only one COC: explain why it is sufficient. If multiple COCs: justify why a single COC would violate atomicity or clarity.",
  "extracted_themes": [
    {{
      "theme_id": 1,
      "theme_label": "1–3 word atomic {taxonomy_actionable_type}-{theme_head} label",
      "theme_clarification": "≤30-word grounded definition describing what belongs in this theme",
      "abstraction_level": "L2 —{taxonomy_actionable_type}-{theme_head} theme",
      "assignment_examples": {{
        "inclusion": [
          "Example 1: Observable cue starting with a verb",
          "Example 2: Observable cue starting with a verb"
        ],
        "exclusion": [
          "Boundary case 1: What must NOT be included",
          "Boundary case 2: What must NOT be included"
        ],
        "near_neighbor": {{
          "label": "Label of closest potentially-confusable theme, or 'Unknown' if none exists",
          "tell_apart_rule": "One sentence distinguishing this theme from the neighbor"
        }}
      }}
    }}
  ]
}}

Critical requirements:
- Output must be valid JSON only — no extra commentary or explanation before or after the JSON
- Keep field names in English; write all values in {language}
- The cluster_id value must be exactly "{cluster_id}" as provided
- Conduct your entire analysis in {language}
- If multiple themes are identified, include each as a separate object in the extracted_themes array with sequential theme_id values
- Provide 2-3 inclusion examples and 1-2 exclusion examples for each theme
- Assignment examples should be short, concrete, and start with verbs (for inclusion/exclusion)
"""

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

- **USE** — An existing code fully captures the new theme's meaning; use it as-is without modification
- **MODIFY_HORIZONTAL** — An existing code needs broader definition and inclusion rules to cover the new theme, but remains at the same abstraction level on the coding dimension ("{taxonomy_axis}:{taxonomy_actionable_type}")
- **MODIFY_VERTICAL** — The existing code and new theme belong to the same conceptual family but differ in abstraction level; create or reference a parent code for both
- **CREATE** — Add a new code because the theme represents a distinct {taxonomy_actionable_type}-{theme_head} not covered by existing codes

---

ANALYSIS FRAMEWORK

Follow these steps systematically:

**STEP 0: Initial Matching**
- Review the new theme and all existing codes
- Identify the best matching existing code(s) based on core meaning and practical relevance in light of the research question, taxonomy axis, and primary coding dimension

**STEP 1: Conceptual Family Test**
Ask: Do the new theme and the best matching existing code belong to the same conceptual family, given the research question, taxonomy axis ({taxonomy_axis}), and primary coding dimension ({taxonomy_actionable_type})?
- If the new theme and best matching existing code share the same core {theme_head} and have the same practical relevance → SAME FAMILY
- Otherwise → DIFFERENT FAMILY

**STEP 2: Abstraction Level Test**
Ask: Are the new theme and the best matching existing code at the same abstraction level on the taxonomy axis/coding dimension?
- If the height of generality/specificity is similar → SAME ABSTRACTION LEVEL
- Otherwise → DIFFERENT ABSTRACTION LEVEL

**STEP 3: Decision Logic**
Apply the following decision rules:

- If the new theme is fully covered in meaning and scope by an existing code → USE existing code.
- If the new theme is not fully covered by an existing code:
  - If it belongs to the same code family and is at the same abstraction level → MODIFY_HORIZONTAL
      - Broaden the existing code's definition and inclusion rules to incorporate the new expression, ensuring the original core meaning remains intact.
  - If it belongs to the same code family but at a different abstraction level → MODIFY_VERTICAL
      - Introduce or reference a higher-level parent code, treating the existing code and new theme as related sub-codes.
  - If it belongs to a different code family → CREATE a new code for the distinct {taxonomy_actionable_type}-{theme_head}.

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
- Use a noun phrase of 1–10 words.
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

After completing your analysis in the scratchpad, provide your final answer as valid JSON only inside <json_output> tags.

The JSON must follow this exact structure:

{{
  "coding_decision": {{
    "theme_number": {theme_id},
    "theme_name": {theme_name},
    "matched_candidates": [
        {{"code": "Exact candidate code A", "definition": "Definition in light of the survey question"}},
        // Add additional candidaties,if there are any
        ]
    "decision": "USE | MODIFY_VERTICAL | MODIFY_HORIZONTAL | CREATE",
    "source_code": "Exact candidate code name if use/modify, or null if create",
    "modify_parameters":{{
       "modify_instruction": "vertical_broaden_same_level | hierarchical_parent_diff_level | none",
       "conceptual_family": "same | different",
       "abstraction_level": "same | different",
       "abstraction_level_action": "keep | broaden_to_parent | none",
       "inclusion_update": "null or concrete additions to inclusion rules",
       "exclusion_update": "null or concrete boundary clarifications",
       "parent_theme_label": "null or suggested parent label",
       "near_neighbor_label_update": "null or updated neighbor label if boundaries changed",
       "tell_apart_rule_update": "null or updated tell-apart rule if distinction changed"}},
    "justification": "Explain decision by referencing conceptual family and abstraction level comparison, or null if use/create",
    "updated_assignment_examples": {{
      "inclusion": ["[updated or original inclusion examples in {language}]"],
      "exclusion": ["[updated or original exclusion examples in {language}]"],
      "near_neighbor": {{
        "label": "[updated or original neighbor label in {language}]",
        "tell_apart_rule": "[updated or original tell-apart rule in {language}]"
      }} | null
    }}.
  }}
}}


**Requirements:**
- Output must be valid JSON only inside json_output tags (no additional commentary outside these tags)
- Keep field names in English; write values in the language specified in codebook_parameters
- Include conceptual family and abstraction level comparison explicitly in justification
- Ensure all updates maintain MECE principles and code atomicity
- Reference any cosine similarity scores (if provided) in your justification
"""

CODE_CREATION_PROMPT = """
You are a {language} qualitative research assistant.
Your task is to CREATE a new code that captures the meaning of a newly identified atomic {taxonomy_actionable_type}-{theme_head} theme from survey responses, using the specified taxonomy framework.

---

ATOMICITY RULES (must all be satisfied)

A code must be:
- ATOMIC: It expresses exactly ONE semantic nucleus — one indivisible {taxonomy_actionable_type} or {theme_head}.
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

- Use a noun phrase of 1–10 words.
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
- "References to…"
- "Mentions of…"
- "Expressions of…"
- "Concerns about…"

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

FINAL OUTPUT FORMAT

Output the result in this strict JSON schema (no commentary or explanation):
{{
  "generated_code": {{
    "theme_number": {theme_id},
    "theme_name": "{cluster_summary}",
    "source_code": "null",
    "code_label": "new or modified code label in {language}",
    "code_definition": "≤25-word operational definition in {language}",
    "assignment_examples": {{
      "inclusion": ["[2-3 concrete examples of what to include in {language}]"],
      "exclusion": ["[1-2 concrete examples of what to exclude in {language}]"],
      "near_neighbor": {{
        "label": "[closest confusable concept in {language} or 'Unknown']",
        "tell_apart_rule": "[1-sentence distinction in {language}]"
      }}
    }}
  }}
}}

Critical remarks:
- Use theme_id provided.
- Use theme_name provided.
- Use source_code provided
"""

# Placeholders for CODING_MODIFICATION_PROMPT
HORIZONTAL_INSTRUCTIONS = """
   - Keep the abstraction level of the original code.
   - Create a **single atomic shared concept** that:
        (a) captures the meaning of both original code and new theme,
        (b) is grounded in the shared conceptual family and abstraction level,
        (c) remains expressible as **one idea** in the label.
   - The modified label must:
        • reflect the broadened meaning,
        • NOT introduce multiple aspects or abstraction levels,
        • NOT be more abstract than necessary.
   - The modified definition must:
        • describe the **shared meaning space**,
        • reflect: original inclusions + inclusion_update,
        • exclude: original exclusions + exclusion_update.
   - Do **not** modify assignment rules here."""

VERTICAL_INSTRUCTIONS = """
   - Shared conceptual family but different abstraction levels → create hierarchical structure.
   - Original code and new theme remain **atomic child codes**.
   - Parent code represents the shared **conceptual family**.

   Parent label:
        - parent theme = {parent_theme_label}
        - If parent theme is not None or Null → use it as-is.
        - If null → generate a label at a higher abstraction level (Driver/Why level).
        - Must:
            • express shared conceptual family,
            • NOT describe behaviors/outcomes,
            • NOT blend child labels,
            • be broader, not vaguer.

   Structure:
       - Parent = conceptual anchor (higher abstraction level),
       - Children = distinct manifestations (different abstraction levels),
       - Child meanings **do not change**."""

CODING_MODIFICATION_PROMPT = """
You are a {language} qualitative research assistant updating a codebook.
Your task is to MODIFY an existing code so that it fully and correctly includes a new {taxonomy_actionable_type}-{theme_head} theme, while preserving **atomic meaning** and **clear conceptual boundaries**.

---

ATOMICITY RULES (must all be satisfied post-modification)

The modified code must remain:
- ATOMIC: It expresses exactly ONE semantic nucleus — one indivisible {taxonomy_actionable_type} or {theme_head}.
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

- Use a noun phrase of 1–10 words.
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
- "References to…"
- "Mentions of…"
- "Expressions of…"
- "Concerns about…"

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

OUTPUT FORMAT (valid JSON only, no commentary, in {language}):

{{
  "generated_code": {{
    "theme_number": {theme_id},
    "theme_name": "{cluster_summary}",
    "source_code": {source_code},
    "code_label": "yur new/modified code label in {language}",
    "code_definition": "your definition in {language}",
    "assignment_examples": {{
      "inclusion": ["[updated inclusion examples combining original + new in {language}]"],
      "exclusion": ["[updated exclusion examples combining original + new in {language}]"],
      "near_neighbor": {{
        "label": "[updated or original neighbor label in {language}]",
        "tell_apart_rule": "[updated or original tell-apart rule in {language}]"
      }}
    }}
  }}
}}

REQUIREMENTS:
- Output must be valid JSON only.
- No commentary outside JSON.
- If hierarchical_parent_diff_level → ensure parent label is conceptual, not descriptive or repetitive.
"""

# =============================================================================
# VALIDATION INSTRUCTION VARIANTS (for scenario-specific validation)
# =============================================================================

USE_VALIDATION_INSTRUCTIONS = """
**Scenario: USE existing code**

Your task is to validate the proposal that an existing code already captures this theme’s meaning.
You must APPROVE or REJECT this proposal. If rejected, provide a corrected final decision.

Scenario-specific validation questions:
- Does the existing code’s definition fully cover the expressions in the new theme?
- Would assigning this theme to the existing code lose any meaningful distinctions?
- Are there any expressions in the new theme that the existing code would NOT capture?

General coding validation question:
- Does the existing code align with these critical rules? A code must be:
  • TAXONOMIC: Aligned with the specified taxonomy axis and coding dimension  
  • ATOMIC: Express one indivisible {taxonomy_actionable_type}; cannot be split into separate {taxonomy_actionable_type} that are practically meaningful for explaining responses to the survey question  
  • SINGLE-VALUED: Represent one clear concept without blending distinct {taxonomy_actionable_type} from different conceptual families  
  • ACTIONABLE: Be clearly identifiable and address the survey question directly and explicitly  
- No code label may contain conjunctions (“and”, “or”, “&”), slashes, or compound constructions. If present, split into separate atomic codes unless one part has no independent analytic meaning.

If validation FAILS (the existing code does not fully capture the theme):
→ Recommend MODIFY (horizontal or vertical refinement) or CREATE (if substantially different)
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
  • TAXONOMIC: Aligned with the specified taxonomy axis and coding dimension  
  • ATOMIC: Express one indivisible {taxonomy_actionable_type}; cannot be split into separate {taxonomy_actionable_type} that are practically meaningful for explaining responses to the survey question  
  • SINGLE-VALUED: Represent one clear concept without blending distinct {taxonomy_actionable_type} from different conceptual families  
  • ACTIONABLE: Be clearly identifiable and address the survey question directly and explicitly  
- No code label may contain conjunctions (“and”, “or”, “&”), slashes, or compound constructions. If present, split into separate atomic codes unless one part has no independent analytic meaning.

CRITICAL: If the new label shifts or replaces the core concept rather than extending it:
→ REJECT and recommend CREATE instead (preserve original code, create new one)
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
  • TAXONOMIC: Aligned with the specified taxonomy axis and coding dimension  
  • ATOMIC: Express one indivisible {taxonomy_actionable_type}; cannot be split into separate {taxonomy_actionable_type} that are practically meaningful for explaining responses to the survey question  
  • SINGLE-VALUED: Represent one clear concept without blending distinct {taxonomy_actionable_type} from different conceptual families  
  • ACTIONABLE: Be clearly identifiable and address the survey question directly and explicitly  
- No code label may contain conjunctions (“and”, “or”, “&”), slashes, or compound constructions. If present, split into separate atomic codes unless one part has no independent analytic meaning.

If validation FAILS:
→ Recommend MODIFY_VERTICAL (if same level) or CREATE (if unrelated)
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
  • TAXONOMIC: Aligned with the specified taxonomy axis and coding dimension  
  • ATOMIC: Express one indivisible {taxonomy_actionable_type}; cannot be split into separate {taxonomy_actionable_type} that are practically meaningful for explaining responses to the survey question  
  • SINGLE-VALUED: Represent one clear concept without blending distinct {taxonomy_actionable_type} from different conceptual families  
  • ACTIONABLE: Be clearly identifiable and address the survey question directly and explicitly  
- No code label may contain conjunctions (“and”, “or”, “&”), slashes, or compound constructions. If present, split into separate atomic codes unless one part has no independent analytic meaning.

If validation FAILS (an existing code could cover this):
→ Recommend USE (if fully covered) or MODIFY (if partial overlap)
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
- If the proposal is APPROVED → final decision = original recommendation
- If the proposal is REJECTED → final decision = USE, MODIFY_HORIZONTAL, MODIFY_VERTICAL, or CREATE based on your analysis

**Determine final decision components**
- validated_decision: Final decision (USE, MODIFY_HORIZONTAL, MODIFY_VERTICAL, or CREATE)
- source_code:
   - If USE → exact code from proposal
   - If MODIFY_HORIZONTAL or MODIFY_VERTICAL → exact existing code being modified
   - If CREATE → null
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
- No code label may contain conjunctions (“and”, “or”, “&”), slashes, or compound constructions. If present, split into separate atomic codes unless one part has no independent analytic meaning.
- DO NOT repeat the actor, domain, topic, or entity in the label (do not repeat: {perspective}, {domain}, {topic} and {entity})

DEFINITION RULES:
- Use 30 words or fewer
- Ground the definition in the cluster data
- Describe **what belongs in this code**, not why it happens
- Align directly with the survey question, taxonomy axis, and coding dimension
- Use a clear, observable assignment cue (e.g., behaviors, expressions, judgments)
- Do NOT explain causes, conditions, or interpretations
- DO NOT repeat the actor, domai§, topic, or entity in the description (do not repeat: {perspective}, {domain}, {topic} and {entity})

GOOD DEFINITION PATTERNS FOR FINAL DECISION::
- "References to…"
- "Mentions of…"
- "Expressions of…"
- "Concerns about…"
</scratchpad>

Now provide your final evaluation as valid JSON in the language specified below. Return ONLY the JSON response with no additional text, comments, or extra fields.

Output schema:
{{
  "code_validation": {{
    "theme_number": {theme_id},
    "theme_name": {cluster_summary},
    "original_recommendation": {{
        "code": "Exact recommended label",
        "definition": "Exact recommended definition"
      }},
    "verdict": "APPROVE" | "REJECT",
    "decision_rationale": "Brief explanation as to why the recommendation was approved or rejected",
    "validated_decision" : "USE or MODIFY_HORIZONTAL or MODIFY_VERTICAL or CREATE"
    "source_code": "If USE, this exact code: {source_code}; If MODIFY_HORIZONTAL or MODIFY_VERTICAL, the exact code from the existing codebook you seek to modify - or null, if CREATE",
    "validated_code": {{
      "code": "Final validated label (≤10 words, rule-compliant)",
      "definition": "Final validated definition (≤25 words, operational, grounded)",
      "assignment_examples": {{
        "inclusion": ["[validated/refined inclusion examples in {language}]"],
        "exclusion": ["[validated/refined exclusion examples in {language}]"],
        "near_neighbor": {{
          "label": "[validated neighbor label in {language}]",
          "tell_apart_rule": "[validated tell-apart rule in {language}]"
        }}
      }}
    }}
  }}
}}

**Critical remarks:**
- Use theme_number and theme_name exactly as provided in the coding proposal
- For source_code: IIf USE, this exact code: {source_code}; If  MODIFY_HORIZONTAL or MODIFY_VERTICAL, the exact code from the existing codebook you seek to modify - or null, if CREATE
- All text in assignment_examples, near_neighbor label, and tell_apart_rule must be in the specified output language
- Return only valid JSON with no additional commentary
- Ensure all labels and definitions strictly follow the rules above
"""

#


# =============================================================================
# STEP 7 THEME ORGANIZATION WITH REASONING MODELS
# =============================================================================

CODEBOOK_REFINEMENT_PROMPT = """
You are a qualitative research methodologist organizing codes into a hierarchical codebook.
Your PRIMARY task is to ORGANIZE codes into themes - NOT to reduce their number.

Here is the survey question:
<survey_question>
{survey_question}
</survey_question>

Here are the codes to organize:
<raw_codes>
{raw_codes}
</raw_codes>

Output your response in this language:
<language>
{language}
</language>

# CRITICAL: Preservation Over Reduction
Your goal is to ORGANIZE codes into a clear hierarchy, NOT to reduce the number of codes.
- PRESERVE all codes that represent distinct concepts
- Only MERGE codes that are TRUE DUPLICATES (identical meaning, just different wording)
- When in doubt, KEEP codes separate

# When to Merge (STRICT criteria - all must apply)
ONLY merge two codes if ALL of the following are true:
1. They describe the EXACT same concept (not just related concepts)
2. A researcher would report them as ONE finding (not as variations)
3. Their inclusion examples overlap completely
4. No exclusion examples or tell-apart rules distinguish them

If ANY doubt exists → KEEP SEPARATE

# Structure and Hierarchy
Organize codes into a 2-level or 3-level hierarchy:

**2-level (Theme → Code)**: Use when themes are simple and codes don't need sub-grouping
**3-level (Theme → Category → Code)**: Use when a theme contains multiple sub-concepts that benefit from grouping

Guidelines:
- Every code must belong to exactly one theme
- Themes should be conceptually coherent (related codes grouped together)
- Use 3-level hierarchy when ≥3 codes share a clear sub-concept within a theme
- Aim for 5-15 themes depending on codebook size

# Theme and Code Naming

**Theme Labels**
- ≤ 10 words, noun phrases preferred
- Describe the conceptual domain (e.g., "Duurzaamheid", "Klantenservice", "Prijsperceptie")
- No conjunctions or slashes

**Code Labels**
- Keep original code labels unless they violate naming rules
- ≤ 10 words, specific and atomic

**Code Descriptions**
- ≤ 20 words
- Define what belongs in this code
- Use patterns like: "Mentions of…", "References to…"

# Required Output Format

Think through the organization, then provide JSON:

{{
  "analysis": "In {language}: (1) How codes were organized into themes, (2) Any codes merged (with justification - should be very few), (3) Hierarchy structure chosen (2-level or 3-level), (4) Final count: X codes organized into Y themes.",
  "refined_codebook": [
    {{
      "theme": "Theme label",
      "codes": [
        {{
          "id": "original code_id (or comma-separated IDs if merged)",
          "code": "Code label",
          "description": "≤ 20 words explanation",
          "category": ""  // Empty for 2-level, category name for 3-level
        }}
      ]
    }}
  ]
}}

Notes:
- The number of codes in output should be close to the number of input codes (merging should be rare)
- No commentary before or after JSON
- All text in the specified output language

Begin organizing the codebook.
"""


CODEBOOK_MERGE_PROMPT = """
You are a qualitative research methodologist consolidating multiple codebooks into one unified structure.
Your PRIMARY task is to UNIFY the organization - NOT to reduce the number of codes.

Here is the survey question:
<survey_question>
{survey_question}
</survey_question>

Here are the codebooks to consolidate:
<codebooks>
{codebooks_summary}
</codebooks>

All output must be in this language:
<language>
{language}
</language>

# CRITICAL: Preservation Over Reduction
You are consolidating {n_codebooks} codebooks from different batches. Your goal is to:
1. PRESERVE all unique codes from all codebooks
2. Only MERGE codes that are TRUE DUPLICATES (identical meaning appearing in multiple codebooks)
3. Create a unified theme structure that organizes ALL codes

# When to Merge (STRICT criteria)
ONLY merge codes if they are TRUE DUPLICATES:
- EXACT same concept appearing in multiple codebooks (due to batch overlap)
- A researcher would consider them identical findings

Do NOT merge codes that are:
- Related but distinct concepts
- Different aspects of the same topic
- Similar but with different nuances

When in doubt → KEEP SEPARATE

# Consolidation Steps
1. Identify TRUE duplicates across codebooks (codes with identical meaning)
2. Keep all unique codes
3. Organize all codes into a unified theme structure
4. Use 2-level or 3-level hierarchy as appropriate

# Theme Structure
**2-level (Theme → Code)**: Simple organization
**3-level (Theme → Category → Code)**: Use when themes have clear sub-groupings

Guidelines:
- Merge similar THEMES across codebooks (organizational labels), but preserve the CODES within them
- Every code must appear exactly once in the final codebook
- Aim for 5-15 themes depending on total code count

# Label Rules
- Theme labels: ≤10 words, noun phrases, no conjunctions/slashes
- Code labels: Keep original labels, ≤10 words
- Descriptions: ≤30 words, define when to use the code

# Output Format

{{
  "analysis": "In {language}: (1) How codebooks were unified, (2) Any duplicate codes merged (should be few - only true duplicates from batch overlaps), (3) Theme structure chosen, (4) Final count: X codes from Y input codebooks organized into Z themes.",
  "refined_codebook": [
    {{
      "theme": "Theme label",
      "codes": [
        {{
          "id": "original code ID(s)",
          "code": "Code label",
          "description": "Code definition (≤30 words)",
          "category": ""  // Empty for 2-level, category name for 3-level
        }}
      ]
    }}
  ]
}}

IMPORTANT: The total number of unique codes in your output should be close to the total unique codes across all input codebooks. Significant reduction indicates over-merging.

Begin consolidating the codebooks.
"""


# =============================================================================
# STEP 8: CODE ASSIGNMENT
# =============================================================================

# Stage 1: Evaluate default code from cluster
DEFAULT_CODE_EVALUATION_PROMPT = """
You are a {language} qualitative coding specialist who assigns codes from a codebook to survey responses. 
Your task is to determine if there is explicit or clearly paraphrased evidence that a specific code appears in a given response text.

The language you will be working in: {language}

Here is the survey question for context:
<survey_question>
{var_lab}
</survey_question>

Here is the response you need to analyze:
<response>
Idea ID: {idea_id}
Idea Text: {idea_text}
</response>

Here is the code you need to evaluate:
<code_details>
Code: {default_code}
Definition: {default_definition}

Inclusion Examples (valid references for this code):
    {inclusion_examples}

Exclusion Examples (invalid references for this code):
    {exclusion_examples}

Boundary: This code covers "{default_code}", which differs from "{near_neighbor_label}"
How to tell them apart: {tell_apart_rule}
</code_details>

Follow these DECISION RULES strictly:

1) Evidence types
   • Explicit: the response uses terms that directly express the target concept.
   • Unambiguous paraphrase: different wording that clearly conveys the target concept without reasonable alternative readings.
   • Do NOT infer intent beyond the text. Do not rely on general world knowledge.

2) Include vs Exclude
   • Include if the target concept is explicit or an unambiguous paraphrase appears anywhere in the response.
   • Exclude if the response:
       – Only expresses the near neighbor concept (per {tell_apart_rule});
       – Matches any Exclusion Example pattern;
       – Mentions the concept only in a negated or hypothetical/conditional way (e.g., “would/if/might” without an asserted claim);
       – Is too generic or off-topic.
   • If both Inclusion-like and Exclusion-like signals appear, Exclusion takes precedence unless the Inclusion is explicit and clearly satisfies the Definition.

3) Minimal supporting span
   • If Including, identify the shortest verbatim span in the response that demonstrates the concept.
   • If Excluding, no supporting span is needed.
   • Preserve original casing and spelling; do not correct typos.

4) Multiple claims / long answers
   • Evaluate the entire Idea Text. If any part contains qualifying evidence, Include.
   • If the answer only restates the survey question or is empty/“N/A”, Exclude.

5) Confidence (0.00–1.00)
• 0.90–1.00 (A: Explicit Evidence): The meaning of the code is explicitly and directly stated in the response; no interpretation needed. Another trained coder would definitely agree.
• 0.70–0.89 (B: Unambiguous Paraphrase): The meaning of the code is explicitly present, but phrased differently. The link is clear without inference. Another trained coder would likely agree.
• 0.50–0.69 (C: Related / Weakly Implied): The code is related but requires interpretive judgment to justify. Reasonable coder disagreement is likely; discussion may be required.
• 0.00–0.49 (D: No Fit): The code is not present, is only tangentially related, or contradicts the response. Another trained coder would not assign this code.


6) **Confidence Threshold Rule (Critical)**
   • If the confidence score would be **below 0.70**, the decision **must be EXCLUDE**.
   • Borderline or partially implied concepts should **not** be coded as present.


IMPORTANT — RATIONALE STRUCTURE:
   • The rationale MUST begin with either "INCLUDE:" or "EXCLUDE:"
   • If INCLUDE: follow with the minimal supporting span in quotes, then a short explanation referencing the definition.
     Example: INCLUDE: "we krijgen geen begeleiding" → explicitly expresses lack of support.
   • If EXCLUDE: briefly state the rule-based reason for exclusion.
     Example: EXCLUDE: No text expresses the target concept; content is generic.

   
Provide your response in this exact JSON format:
{{
  "idea_id": "{idea_id}",
  "confidence": CONFIDENCE_SCORE,
  "rationale":  "INCLUDE: \"...\" → explanation in {language}" OR "EXCLUDE: brief explanation in {language}"
}}

Critical requirements:
- The confidence score must be a number between 0.00 and 1.00
- If the confidence score is below 0.70, the rationale MUST begin with "EXCLUDE:"
- The rationale must follow the INCLUDE:/EXCLUDE: format exactly
- Focus only on the specific concept defined by the code
- Return ONLY the JSON object, no additional commentary

Begin your evaluation now.
"""

# Stage 1b: Evaluate multiple codes from cluster family
FAMILY_CODE_EVALUATION_PROMPT = """
You are a {language} qualitative coding specialist who assigns codes from a codebook to survey responses.
Your task is to evaluate MULTIPLE candidate codes from the same cluster family and determine which one (if any) best fits the response.

The language you will be working in: {language}

Here is the survey question for context:
<survey_question>
{var_lab}
</survey_question>

Here is the response you need to analyze:
<response>
Idea ID: {idea_id}
Idea Text: {idea_text}
</response>

Here are the candidate codes from this cluster family. Evaluate EACH code against the response:
<candidate_codes>
{candidate_codes_formatted}
</candidate_codes>

**Evaluation Process:**
1. Evaluate EACH candidate code against the response independently
2. For each code, determine if there is explicit or clearly paraphrased evidence
3. If multiple codes match with confidence >= 0.70, choose the MOST SPECIFIC one that fits the evidence
4. If no code matches with confidence >= 0.70, set best_match.code to "NONE"

**Decision Rules (apply to EACH code):**

1) Evidence types
   • Explicit: the response uses terms that directly express the target concept.
   • Unambiguous paraphrase: different wording that clearly conveys the target concept without reasonable alternative readings.
   • Do NOT infer intent beyond the text. Do not rely on general world knowledge.

2) Include vs Exclude
   • Include if the target concept is explicit or an unambiguous paraphrase appears anywhere in the response.
   • Exclude if the response:
       – Only expresses a different code's concept;
       – Matches any Exclusion Example pattern;
       – Mentions the concept only in a negated or hypothetical/conditional way;
       – Is too generic or off-topic.

3) Minimal supporting span
   • If Including, identify the shortest verbatim span in the response that demonstrates the concept.
   • Preserve original casing and spelling; do not correct typos.

**Confidence Anchors (0.00–1.00):**
• 0.90–1.00 (A: Explicit Evidence): The meaning of the code is explicitly and directly stated in the response; no interpretation needed.
• 0.70–0.89 (B: Unambiguous Paraphrase): The meaning of the code is explicitly present, but phrased differently. The link is clear without inference.
• 0.50–0.69 (C: Related / Weakly Implied): The code is related but requires interpretive judgment. EXCLUDE threshold.
• 0.00–0.49 (D: No Fit): The code is not present, is only tangentially related, or contradicts the response.

**Confidence Threshold Rule (Critical):**
• Only codes with confidence >= 0.70 qualify as matches.
• If the best-fitting code has confidence < 0.70, set best_match.code to "NONE".

**Tie-Breaking (when multiple codes have confidence >= 0.70):**
1. Choose the code with the highest confidence score.
2. If still tied, prefer the more specific code (narrower definition).
3. If still tied, prefer the code whose definition most closely matches the supporting span.

Provide your response in this exact JSON format:
{{
  "idea_id": "{idea_id}",
  "evaluations": [
    {{"code": "CODE_NAME_1", "confidence": SCORE, "rationale": "INCLUDE: \"span\" → explanation" or "EXCLUDE: reason"}},
    {{"code": "CODE_NAME_2", "confidence": SCORE, "rationale": "INCLUDE: \"span\" → explanation" or "EXCLUDE: reason"}}
  ],
  "best_match": {{
    "code": "BEST_CODE_NAME or NONE",
    "confidence": SCORE,
    "rationale": "INCLUDE: \"span\" → explanation in {language}" or "EXCLUDE: brief explanation in {language}"
  }}
}}

Critical requirements:
- Evaluate ALL candidate codes in the evaluations array
- The best_match.code must be one of the evaluated codes OR "NONE" if no code reaches 0.70 confidence
- The best_match.confidence must match the confidence of the selected code (or 0.0 if NONE)
- All rationales must follow the INCLUDE:/EXCLUDE: format
- Return ONLY the JSON object, no additional commentary

Begin your evaluation now.
"""

# Stage 2: Fallback assignment from all codes
FALLBACK_CODE_ASSIGNMENT_PROMPT = """
You are a {language} qualitative coding specialist who assigns codes from a codebook to survey responses. 
Your task is to assign exactly one existing code from the provided codebook to a response, but only if there is explicit or clearly paraphrased evidence that the specific code concept appears in the response text.

Here is the survey question context:
<survey_question>
{var_lab}
</survey_question>

Here is the response you need to analyze:
<response>
Idea ID: {idea_id}
Idea Text: {idea_text}
</response>

Here are the available codes in the codebook:
<codebook>
{all_codes}
</codebook>

**Decision Rules:**
- Assign EXACTLY ONE code from the codebook if — and only if — the response explicitly states or unambiguously paraphrases the specific concept in that code’s definition.
- If the response is broader/more generic than a code’s definition, that code does NOT fit.
- Prefer codes whose definitions are most specific to the quoted evidence (not merely thematically related).
- Do not infer meaning beyond the text. Negated or hypothetical/conditional mentions (e.g., “not X”, “would/if/might”) do NOT qualify as evidence.
- If no code has clear evidence, assign "{unknown_label}" with low confidence.

**Confidence Level Anchors:**
• 0.90–1.00 (A: Explicit Evidence): The meaning of the code is explicitly and directly stated in the response; no interpretation needed. Another trained coder would definitely agree.
• 0.70–0.89 (B: Unambiguous Paraphrase): The meaning of the code is explicitly present, but phrased differently. The link is clear without inference. Another trained coder would likely agree.
• 0.50–0.69 (C: Related / Weakly Implied): The code is related but requires interpretive judgment to justify. Reasonable coder disagreement is likely; discussion may be required.
• 0.00–0.49 (D: No Fit): The code is not present, is only tangentially related, or contradicts the response. Another trained coder would not assign this code.

Tie-breaking (when multiple candidates look plausible):
1) Choose the code supported by the strongest minimal verbatim span most closely matching its definition.
2) If still tied, choose the code with the more specific definition.
3) If still tied or evidence remains ambiguous, assign "{unknown_label}".

**Confidence Threshold Rule:**
- If the best-fitting interpretation would result in a confidence score below 0.70, assign "{unknown_label}".

**IMPORTANT — RATIONALE FORMAT:**
- The assignment_rationale MUST begin with either:
     "Match:" if assigning a code (confidence ≥ 0.70)
     "{unknown_label}:" if assigning "{unknown_label}" (confidence < 0.70 or no clear concept match)
- If MATCH: include the minimal supporting span in quotes, then explain why it fits the selected code.
- If {unknown_label}: briefly explain that no code was clearly supported.

**Analysis Process:**
1) Evidence Identification: Scan the response for candidate spans that might support specific code concepts.
2) Supporting Span Extraction: For the best-fitting code, identify the shortest verbatim span that demonstrates the concept (preserve casing/spelling).
3) Conceptual Matching: Confirm the span satisfies the chosen code’s definition (not just a related theme).
4) Confidence Assessment: Apply the anchors above.
5) Final Assignment: Output a single code, or "{unknown_label}" if none fit well.

Provide your analysis and assignment in this exact JSON format:
{{
  "idea_id": "{idea_id}",
  "assigned_codes": ["SINGLE_CODE_NAME"],
  "assignment_confidence": CONFIDENCE_SCORE,
  "assignment_rationale": "Match: \"...\" → explanation" OR "{unknown_label}: explanation in {language}"
}}

"""


