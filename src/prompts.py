"""
Prompts module - Contains all LLM prompt templates for the pipeline.
"""

from typing import List
from pydantic import BaseModel, Field


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

After processing all tasks, provide your output in the following JSON format:
{{
  "corrections": [
    {{
      "respondent_id": "ID_FROM_TASK",
      "corrected_response": "The fully corrected response"
    }},
    ...
  ]
}}

Ensure that your output is a valid JSON object with a single key "corrections", whose value is an array of objects. Each object in the "corrections" array must have exactly these fields:
- "respondent_id": "ID_FROM_TASK"
- "corrected_response": "The fully corrected response"

Additional guidelines:
- Pay close attention to the context and meaning of each response when making corrections.
- Ensure that your corrections maintain the original intent of the respondent.
- If a suggested correction doesn't fit the context, consider alternative corrections that preserve the meaning.
- Double-check that your JSON output is properly formatted and includes all corrected responses.

Begin processing the correction tasks now, and provide your output in the specified JSON format.
"""

# =============================================================================
# STEP 2: QUALITY FILTERING 
# =============================================================================

GRADER_INSTRUCTIONS = """
You are a {language} language grader evaluating open-ended survey responses. 
Your task is to determine whether each response is meaningless and assign appropriate quality filter codes.

Task Description:
Analyze each response and classify it based on the following criteria:

Decision Criteria:
1. **Don't Know/Uncertainty (Code 99999997)**: Responses that express "don't know", "not applicable", or only express uncertainty
   - Examples: "I don't know", "N/A", "Not applicable", "No idea", "?"

2. **Nonsensical/Gibberish (Code 99999999)**: Responses that are meaningless, gibberish, or simply repeat the question
   - Examples: "asdfkj", "lorem ipsum", random characters, just repeating the question

3. **Meaningful Response (No Code)**: Responses that provide actual content, opinions, or information
   - These should have quality_filter = false and quality_filter_code = null

Input:
You will be provided with a survey question and a list of responses to evaluate.

Survey question:
<survey_question>
{var_lab}
</survey_question>

Here are the responses you need to evaluate:
<responses>
{responses}
</responses>

Your output should be a JSON array. Each object in the array must contain exactly:
- "respondent_id": (string or number) The respondent's ID
- "response": (string) The exact response text
- "quality_filter": (boolean) true if meaningless, false if meaningful
- "quality_filter_code": (number or null) 99999997 for uncertainty, 99999999 for gibberish, null for meaningful

Follow these steps for each response:
1. Read the response carefully.
2. Determine if the response expresses uncertainty/don't know (code 99999997)
3. If not uncertainty, determine if it's gibberish/nonsensical (code 99999999)
4. If neither, it's meaningful (quality_filter = false, quality_filter_code = null)
5. Create a JSON object with all required fields

After processing all responses, return the complete JSON array.

Remember to use the exact format specified. Here's an example of how entries in your output should look:
[
  {{
    "respondent_id": "1",
    "response": "I don't know",
    "quality_filter": true,
    "quality_filter_code": 99999997
  }},
  {{
    "respondent_id": "2",
    "response": "The product is easy to use and has great features.",
    "quality_filter": false,
    "quality_filter_code": null
  }},
  {{
    "respondent_id": "3",
    "response": "asdfghjkl",
    "quality_filter": true,
    "quality_filter_code": 99999999
  }}
]

Ensure that your entire output is a valid JSON array containing all evaluated responses.
"""

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

Provide concise answers (2-5 words each) in {language}.""" #structure output given to instructor = pydantic model


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
   - Examples: "merk_x", "tesla_model_3", "albert_heijn", "ns_trains"

Provide concise answers (2-5 words each) in {language}.""" #structure output given to instructor = pydantic model


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
You are analyzing survey responses to determine the SINGLE PRIMARY TAXONOMY AXIS that should be used to generate MECE (Mutually Exclusive and Collectively Exhaustive) descriptive codes for a chunk of responses.

Here is the survey question that was asked in {language}:

<survey_question>
{survey_question}
</survey_question>

Here are the responses in {language} you need to analyze:

<chunk_responses>
{chunk_size}
</chunk_responses>

This chunk contains {chunk_size} sample responses.

## YOUR TASK

Idebtifty ONE dominant organizing principle as the primary taxonomy axis that best differentiates the responses such that a code set built on it will be:
- Mutually exclusive (no overlap between codes)
- Collectively exhaustive (all responses can be coded)

## CORE PRINCIPLE

Base your decision on COMMUNICATIVE MEANING — what respondents are primarily trying to convey in their responses — not on grammatical form or surface-level word choice.

## SIX POSSIBLE CODING DIMENSIONS

You must choose the dimension that produces the CLEANEST MECE PARTITION for this specific question + response set:

1) **WHAT** — attributes, features, characteristics, properties, or aspects of something
   - Use when responses primarily describe different features, qualities, or attributes

2) **WHY** — goals, motivations, desired outcomes, reasons, purposes, or value sought
   - Use when responses primarily express different motivations, goals, or reasons

3) **HOW** — actions, steps, processes, methods, behaviors, or implementation approaches
   - Use when responses primarily describe different actions, methods, or behaviors

4) **WHO** — people, groups, stakeholders, roles, or beneficiaries
   - Use when responses primarily differ by the person, group, or role involved

5) **WHEN** — timing, urgency, sequence, frequency, or temporal aspects
   - Use when responses primarily differ by time, timing, or frequency

6) **WHERE** — place, location, channel, setting, or situational context
   - Use when responses primarily differ by location, channel, or context

## DEFINING THE PRIMARY AXIS

Before providing your final answer, use the scratchpad to:
1. Review the survey question and understand what is being asked
2. Read through the responses and identify the main themes or patterns
3. Consider which of the six dimensions best captures how responses differ from each other
4. Test your choice: Would codes based on this dimension be mutually exclusive and collectively exhaustive?
5. Define the specific taxonomy axis within your chosen dimension

Use <scratchpad> tags for your thinking process.

## OUTPUT FORMAT After your analysis in the scratchpad, provide your final answer in the following structure:

<analysis>
<taxonomy_axis>
In 1–2 sentences, define the specific organizing axis within the chosen dimension that will become the MECE code spine.
It must be phrased so that code labels can be created directly from it.
</taxonomy_axis>

<evidence>
Provide 2–3 brief verbatim snippets from the responses that show why this axis is dominant.
</evidence>

<mece_check>
In 1–2 sentences, explain why codes built on this axis will minimize overlap and cover all responses.
</mece_check>
</analysis>

All output in your scratchpad and analysis must be written in {language}.
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
Secondary axis (if any): {secondary_axis}
</input>

<taxonomy_axis_dimensions>
"WHAT": "topic_object - concepts, things, topics, features, attributes"
"WHY": "intent_purpose - goals, desired outcomes, improvements, reasons"
"HOW": "action_method - actions, steps, processes, methods, ways"
"WHO": "actor_target - people, groups, stakeholders, beneficiaries"
"WHEN": "time_urgency - time references, urgency, sequence, timing"
"WHERE": "location_context - place, context, channel, location"
</taxonomy_axis_dimensions>

## Task 1: Identify the canonical subject
- Find the main product, brand, service, actor, or topic the question is about
- Return a concise, normalized noun phrase in {language}
- Preserve capitalization for proper nouns; otherwise use lowercase

## Task 2: Create a phrasing template

Use this flexible structure:
**"[CANONICAL_TERM] [VERB/STATE] [SCAFFOLDING_WORDS] [ATTRIBUTE_OR_ACTION]"**

Where:
- CANONICAL_TERM: the focus entity from Task 1
- VERB/STATE: appropriate verb in {language} (e.g., "is", "has", "should", "needs", "zijn", "heeft", "moet")
- SCAFFOLDING_WORDS: grammatical words needed for completeness (may be empty if verb alone works)
- [ATTRIBUTE_OR_ACTION]: placeholder for actual content

### Axis-aware verb/state guidance

The **{primary_axis}** axis suggests certain verb patterns work best:

- **WHAT** (features, properties): verbs like "has/heeft", "is characterized by/kenmerkt zich door"
- **WHY** (goals, improvements): verbs like "should achieve/moet bereiken", "needs to provide/moet bieden"
- **HOW** (actions, methods): verbs like "should/moet", "can/kan", "needs to/moet"
- **WHO** (stakeholders): focus on the actor, verbs like "needs/heeft nodig", "should receive/moet krijgen"
- **SENTIMENT** (evaluations): verbs like "is/is", "performs/presteert"
- **WHEN** (timing): include timing context in scaffolding
- **WHERE** (location): include location context in scaffolding

Choose the verb/state that sounds MOST NATURAL in {language} for this specific question.

### Grammatical completeness constraint

The template MUST produce a grammatically complete sentence when [ATTRIBUTE_OR_ACTION] is replaced with a simple word.

Test your template by filling [ATTRIBUTE_OR_ACTION] with a one-word example (e.g., "better", "quality", "beter", "kwaliteit").

Common scaffolding patterns:
* For "has/heeft" → often needs: "has the [quality/feature] [ATTRIBUTE_OR_ACTION]" / "heeft de [eigenschap] [ATTRIBUTE_OR_ACTION]"
* For "should/moet" → often works directly: "should [ATTRIBUTE_OR_ACTION]" / "moet [ATTRIBUTE_OR_ACTION]"
* For "is/is" → often works directly: "is [ATTRIBUTE_OR_ACTION]"

## Task 3: Choose the actionable taxonomy dimension

From <taxonomy_axis_dimensions>, select the SINGLE most relevant dimension type for the {primary_axis} axis.
- Look at the options for {primary_axis} (e.g., for WHAT: "concepts, things, topics, features, attributes")
- Choose ONE specific term that best fits this survey's responses
- This narrows the taxonomy focus for MECE code generation

Output format (return **only** this JSON object):
{{
  "canonical_term": "main subject/entity from the survey question in {language}",
  "canonical_phrasing": "natural template ending with [ATTRIBUTE_OR_ACTION] in {language}",
  "taxonomy_axis": "{primary_axis}",
  "taxonomy_actionable_type": "chosen single dimension (e.g., 'attributes' or 'features')"
}}
"""


TAXONOMY_ENRICHED_EXTRACTION_PROMPT = """
You are a {language} language expert extracting structured ideas through the lens of a specified taxonomy axis,

<inputs>
Survey question: {var_lab}

Primary taxonomy axis: {taxonomy_axis}
Primary taxonomy axis description: {taxonomy_axis_description}
Taxonomy actionable type: {taxonomy_actionable_type}

Domain context: {domain} / {topic} / {entity}

Respondent ID: {respondent_id}
Response: {response}
</inputs>

<template_rule>
REQUIRED FORMAT for the "idea" field:
- Start with EXACTLY: "{template_prefix}"
- Then add the specific {taxonomy_actionable_type}

Template structure: {canonical_phrasing}

CORRECT examples:
- "{template_prefix} duurzaamheid"
- "{template_prefix} goede service"

INCORRECT examples:
- Starting with pronouns: "Ze hebben goede service"
- Missing prefix: "duurzaamheid"
- Rephrased prefix: "De bank staat voor duurzaamheid"
</template_rule>

<instructions>
Extract ALL distinct ideas expressed in this response.

For each idea, produce:

1. **idea** (5–20 words)
   REQUIRED FORMAT:
   - Start with EXACTLY: "{template_prefix}"
   - Then add the specific {taxonomy_actionable_type}
   - Written in {language}

   The idea MUST begin with the template prefix verbatim.

2. **taxonomy_phrase** (1–3 words)
- A concise noun-phrase that abstracts the idea into a reusable {taxonomy_actionable_type} aligned with the taxonomy axis ({taxonomy_axis}: {taxonomy_axis_description})
- Phrasing rules:
   * Make the semantic core of the taxonomy_phrase the head of the noun phrase.
   * DO NOT repeat the domain entity or product name (e.g., "{entity}", "maaltijden", "product").
     - BAD: "zoutgehalte maaltijden" → GOOD: "zoutgehalte"
     - BAD: "voedingswaarde maaltijden" → GOOD: "voedingswaarde"
   * Prefer single-word attribute nouns over compound action-nouns.
     - BAD: "assortiment uitbreiding" → GOOD: "assortimentsvariatie"
     - BAD: "prijs verlaging" → GOOD: "prijsniveau"
   * Avoid meta-language about perception, opinion, or thought.
   * Avoid verbs or verb-noun compounds.
-  Written in {language}.

3. **parent_category** (1–2 words)
- A high-level abstract grouping theme for MECE clustering.
- Must be a single abstract noun or short noun-phrase (e.g., "samenstelling", "aanbod", "kwaliteit", "prijs").
- DO NOT add qualifiers like "van ingrediënten" or "van product" — keep it minimal.
- Written in {language}.

4. **sentiment** (choose one): positive | negative | neutral

- positive = praise / approval, etc.
- negative = complaint / dissatisfaction, etc.
- neutral = suggestion without judgment / factual mention, etc.

5. **sense** (choose one): factual | evaluative | aspirational | experiential

IDEA SPLITTING RULES

- Split multi-aspect ideas into separate ideas when conceptually distinct.
- Example: "Sustaainble investment" can be seperated into two atomic concepts "sustainability" and "investment".

EXAMPLE

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
    "parent_category": "facilities",
    "sentiment": "positive",
    "sense": "aspirational"
  }},
  {{
    "respondent_id": "{respondent_id}",
    "idea_id": "2",
    "idea": "{template_prefix} better evening lighting",
    "taxonomy_phrase": "evening lighting",
    "parent_category": "facilities",
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
  "parent_category": string,
  "sentiment": "positive|negative|neutral",
  "sense": "factual|evaluative|aspirational|experiential"
}}

Edge cases:
- Empty or irrelevant response: return []
- Single idea: return one item
- Multiple ideas: return multiple items with sequential idea_id

CRITICAL: Make the semantic core of the taxonomy_phrase the head of the noun phrase.

Return ONLY the JSON array. Field names in English; text values in {language}.
</instructions>
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

# V2 prompt (legacy) - kept for reference
CLUSTER_DESCRIPTION_PROMPT_V2 = """You are a qualitative researcher analyzing survey response clusters.

<inputs>
Language: {language}
Cluster ID: {cluster_id}
Research question: "{survey_question}"
Number of response ideas: {num_ideas}

Statistical keywords (cluster-distinguishing terms):
{keywords_section}

Representative response ideas from this cluster:
{ideas_list}
</inputs>

<task>
Analyze this cluster and provide a thematic summary:
1. Read all response ideas carefully
2. Consider statistical keywords as hints about distinguishing terms
3. Identify the common atomic theme that unifies these responses
4. Extract 3-5 key concepts present in this cluster
</task>

<output_format>
Provide your analysis in the following structured format:
- theme: Short atomic thematic label (≤10 words)
- description: Clear description (1-2 sentences)
- key_concepts: List of 3-5 key concepts/themes
</output_format>"""

# V4 prompt (MECE-grounded) - active default
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

Your task is to generate {{n_codes}} diverse, hypothetical codes that might emerge from analyzing responses to the given survey question. Create codes that could apply to ANY survey topic. Do not assume the survey is about education, healthcare, or any specific domain. Let the survey question guide your code generation.

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

CLUSTER_SUMMARY_PROMPT = """
You are a qualitative researcher applying Braun & Clarke's (2006) thematic analysis method.
Your task is to analyze a cluster of descriptive codes and construct one or more ATOMIC themes (central organizing concepts) along a given taxonomy axis, then draft initial codebook entries for each theme.

Before beginning your analysis, understand these key definitions:
<key_definitions>
- TAXONOMY AXIS: This is the conceptual dimension along which all themes will be organized. It defines what kind of phenomenon we are theming. It sets the "coordinate system" for the analysis.
- CODING DIMENSION: This is the specific analytic focus within the taxonomy axis that determines where themes should be actionable. It narrows the axis into a practical working slice.
- ATOMIC THEME: This is a central organizing concept that groups descriptive codes under one coherent meta-idea that is not divisible without losing meaning. It is the smallest meaningful thematic unit at that level of abstraction.
- ABSTRACTION LEVEL: This indicates how conceptual vs. concrete the atomic theme is, relative to the taxonomy axis and coding dimension.
- NEAR-NEIGHBOR: This is a conceptually adjacent theme that could be easily confused with the atomic theme and therefore needs to be distinguished in the codebook.
</key_definitions>

You will be working in the following language:
<language>
{language}
</language>

All your analytical work, theme labels, definitions, and examples must be written in this language. Only JSON field names should remain in English.

Here is the cluster identifier you are analyzing:
<cluster_id>
{cluster_id}
</cluster_id>

<context>
- Domain: {domain}
- Topic: {topic}
</context>

<research_parameters>
Research question: "{survey_question}"

Response interpretation:
- Perspective: {perspective}
- Intent: {intent}
</research_parameters>

<cluster_data>
{cluster_text}
</cluster_data>

Here is the taxonomy framework guiding your analysis:
<taxonomy>
Taxonomy axis: {taxonomy_axis}
Axis description: {taxonomy_axis_description}
primary Coding Dimension: {taxonomy_actionable_type}
</taxonomy>

Some constraints and strict rules you need to know about:
<label_constraints>
Label constraints (strict):
- Short noun phrase (≤ 10 words) of an ATOMIC theme in light of the research question, taxonomy axis and coding dimension.
- Make the semantic core of the theme the head of the noun phrase.
- Adhere to the atomicity rules below.
- DO NOT repeat the actor, domain, topic or entity in the label ({perspective}, {domain}, {topic} and {entity}).
</label_constraints>

<atomicity_rules>
- Atomicity rules (strict):
  • All naming and labelling of ATOMIC THEMES must be single-valued.
  • Do not output multi-labels, spans, or hybrids.
  • When uncertainty exists, commit to one value and record uncertainty only in supporting text (definition / near-neighbor / notes).
  • NEVER use punctuation: "/", "&", ",", "–", ":" (unless lexicalized).
- If a code mentions multiple aspects or mixed sentiment → split into separate atomic concepts.
</atomicity_rules>

<definition_constraints>
Definition constraints (strict):
- ≤ 30 words
- grounded in cluster data
- Must describe **what belongs in this code**, not why it happens.
- Must align directly with the survey question, taxonomy axis, and coding dimension.
- Use a **clear, observable assignment cue** (e.g., behaviors, expressions, judgments).
- Do not explain causes, conditions, or interpretations.
- DO NOT repeat the actor, domain, topic or entity in the description ({perspective}, {domain}, {topic} and {entity}).
</definition_constraints>

Now follow these analysis steps exactly and in order. Do not skip or reorder any step:

<analysis_steps>
Step 1. Interpret descriptive codes in light of the taxonomy focus and research question:
    - Review each code in the cluster data
    - Ask yourself: How does each code address the research question, taxonomy axis, and coding dimension?
    - Identify what patterns are meaningful for analyzing concrete, actionable answers

Step 2. Remove outliers.
    - Eliminate codes that do not connect to any broader pattern across multiple codes.
    - Eliminate codes that do not represent a meaningful segment (too rare, irrelevant, or idiosyncratic).

Step 3. Identify COC(s):
    - Practical test: Can all codes be summarized in one sentence that:
     (a) aligns with <guidance>,
     (b) captures exactly ONE ATOMIC theme,
     (c) uses no coordinating conjunctions or list punctuation,
     (d) preserves unity, consistency, and contrast?
   - If yes → single COC. If no → multiple COCs.
   - Proceed with the resulting COC(s).

Step 4. Refine COCs:
    - Remove singletons (COCs based on only one code or a neglectable number of codes in full sumple).

Step 5: Create codebook entries
For each retained COC, produce a codebook-ready draft with:
  - theme_label: Follow all label constraints above
  - theme_clarification: Follow all definition constraints above; include illustrative descriptive codes from the cluster data that clarify and support the label
  - abstraction_level: Select exactly one of: "Driver/Motive/Why" | "Attribute/What" | "Action/How"
  - assignment_examples (these are EXAMPLES, not exhaustive rules):
    • inclusion: Provide 2–3 short, positive example assignment expressions (each starts with a verb; use observable cues)
    • exclusion: Provide 1–2 short example boundary cases to prevent overreach (what must NOT be included)
    • near_neighbor:
      — label: The closest potentially-confusable theme, or "Unknown" if none exists
      — tell_apart_rule: One sentence explaining how to distinguish this theme from the neighbor (e.g., "This theme focuses on X (driver/what/how), whereas the neighbor focuses on Y.")

Step 6) Document the analysis:
   - State how many COCs were identified and retained.
   - If only one COC: explain why it is sufficient.
   - If multiple COCs: justify why a single COC would violate atomicity or clarity.
</analysis_steps>

Output strictly as valid JSON using this exact structure (values in {language}, field names in English):
{{
  "cluster_id": "{cluster_id}",
  "analysis": "Provide your analysis here in {language}.",
  "extracted_themes": [
    {{
      "theme_id": 1,
      "theme_label": "[≤ 10 words | active/actionable formulation of ONE ATOMIC theme in relation to the research question]",
      "theme_clarification": "[≤ 30 words | illustrative descriptive codes from <inputs> that clarify and support the label — tight, grounded, evidence-based]",
      "abstraction_level": "Driver/Motive/Why | Attribute/What | Action/How",
      "assignment_examples": {{
        "inclusion": [
          "[examples inclusion in {language}]"
        ],
        "exclusion": [
          "[examples exclusion in {language}]"
        ],
        "near_neighbor": {{
          "label": "[neighbor label in {language} or 'Unknown']",
          "tell_apart_rule": "[1-sentence distinction in {language}]"
        }}
      }}
    }}
  ]
}}


Critical requirements:
- Output must be valid JSON only — no extra commentary or explanation before or after.
- Keep field names in English; write values in {language}.
- The cluster_id value must be exactly "{cluster_id}" as provided in the inputs.
- Conduct your analysis in the specified language.
"""

CODING_DECISION_PROMPT = """
You are a qualitative research assistant responsible for maintaining a structured codebook for thematic analysis following Braun & Clarke (2006) methodology.
Your task is to classify a newly identified theme by comparing it against existing codes in the codebook.
You must ensure the codebook remains MECE (Mutually Exclusive, Collectively Exhaustive) by strictly adhering to the specified taxonomy structure.

You will be working with the following parameters:

<language>
 {language}
</language>

<taxonomy_parameters>
Taxonmy Axis:  {taxonomy_axis}
Axis description: {taxonomy_axis_description}
primary Coding Dimension: {taxonomy_actionable_type}
</taxonomy_parameters>

<context>
- Domain: {domain}
- Topic: {topic}
Survey Question: "{survey_question}"
</context>

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

Your task is to classify the new theme by deciding which of the following actions to take:

**Decision Options:**
- **USE** — The theme is semantically identical to an existing code (no changes needed)
- **MODIFY_VERTICAL** — Broaden an existing code at the same abstraction level to include this theme as an additional expression within the primary coding dimension
- **MODIFY_HIERARCHICAL** — Create or reference a parent code that groups multiple related codes across different abstraction levels within the same conceptual family
- **CREATE** — Add a new code because the theme represents a distinct concept not covered by existing codes

**Analysis Framework:**

Follow these steps in order:

**STEP 1: Apply Cosine Similarity Rules**
- If cosine similarity ≥ 0.95 → Same meaning → Decision: USE
- If cosine similarity < 0.55 → Different meanings → Decision: CREATE
- If 0.55 ≤ cosine similarity < 0.95 → Proceed to STEP 2

**STEP 2: Semantic Test (Only if cosine similarity is 0.55–0.95)**

Evaluate the new theme against the single highest-similarity code by answering:

Q1 (Conceptual Family): Do the new theme and the highest-similarity code belong to the same conceptual family, given the research question, taxonomy axis, and primary coding dimension?

Q2 (Abstraction Level): Are the new theme and the highest-similarity code at the same abstraction level (Driver/Attribute/Action) on the taxonomy axis/coding dimension?

Apply these decision rules:
- If Same Family AND Same Abstraction Level → Decision: MODIFY_VERTICAL
  - Broaden the existing code's inclusion rules to cover the new expression
  - Keep the abstraction level unchanged
  - Maintain atomicity (one core concept per code)

- If Same Family AND Different Abstraction Level → Decision: MODIFY_HIERARCHICAL
  - Create or reference a parent code at a higher abstraction level
  - Treat the existing code and new theme as sub-codes within the same family

- If Not Same Family → Decision: CREATE

**STEP 3: Enforce Structural Constraints**
- Codes must remain atomic: one idea, one motive, one sentiment
- Inclusion rules describe when to assign the theme
- Exclusion rules describe common misfits to maintain clear boundaries
- The codebook must remain MECE

**STEP 3.5: Multi-Concept Theme Check**
If the new theme contains multiple distinct concepts (e.g., "salt reduction AND mild spices"):
- Identify which concept(s) semantically match the existing code
- If only ONE concept matches and MODIFY would require changing the existing code's core meaning to accommodate the other concept(s): Decision = CREATE
- Preserve the existing code unchanged and create a new code for the theme
- A MODIFY should never replace an existing code's central meaning with a different concept

**STEP 4: Determine Assignment Example Updates**

If decision is **USE**:
- Preserve original assignment_examples unchanged

If decision is **MODIFY_VERTICAL** or **MODIFY_HIERARCHICAL**:
- inclusion: Combine original + new expressions from the theme
- exclusion: Combine original + new boundary clarifications if needed
- near_neighbor: Update label if boundaries shifted due to modification
- tell_apart_rule: Update if the distinction from neighbor changed

If decision is **CREATE**:
- Use assignment_examples from the new theme as-is

Before providing your final answer, use <scratchpad> tags to work through your analysis systematically:

1. Identify the top candidate codes based on semantic similarity
2. Note the cosine similarity scores for top candidates
3. Apply the cosine similarity rules from STEP 1
4. If needed (cosine 0.55-0.95), perform the semantic test from STEP 2
5. Determine your decision (USE/MODIFY/CREATE) and provide justification referencing the cosine score and conceptual family/abstraction level analysis
6. Plan what updates are needed to assignment examples based on your decision

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
    "decision": "USE | MODIFY_VERTICAL | MODIFY_HIERARCHICAL | CREATE",
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
    "justification": "Explain modification decision by referencing cosine score and conceptual family/abstraction level comparison, or null if use/create",
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

**Interpretation Examples:**
1. **MODIFY_VERTICAL Example**: cosine = 0.81, same conceptual family, same abstraction level → "This theme expresses the same underlying concept as Code A but in a new form. MODIFY_VERTICAL to broaden inclusion rules."
2. **MODIFY_HIERARCHICAL Example**: cosine = 0.80, same conceptual family, different abstraction level → "This theme shares the conceptual domain of Code A but at a different abstraction level. MODIFY_HIERARCHICAL to create parent code."
3. **CREATE Example**: cosine = 0.64, different conceptual family → "The meaning and conceptual family differ significantly. CREATE new code."

**Requirements:**
- Output must be valid JSON only (no additional commentary outside the json_output tags)
- Keep field names in English; write values in the language specified in codebook_parameters
- Include cosine score in justification
- Identify conceptual family and abstraction level comparison explicitly in justification
- Ensure all updates maintain MECE principles and code atomicity
"""

CODE_CREATION_PROMPT = """
You are a {language} qualitative research assistant.
Your task is to CREATE a new code that captures the meaning of a newly identified atomic theme from survey responses.

Before beginning your analysis, understand these key definitions:
<key_definitions>
- TAXONOMY AXIS: This is the conceptual dimension along which all themes will be organized. It defines what kind of phenomenon we are theming. It sets the "coordinate system" for the analysis.
- CODING DIMENSION: This is the specific analytic focus within the taxonomy axis that determines where themes should be actionable. It narrows the axis into a practical working slice.
- ATOMIC THEME: This is a central organizing concept that groups descriptive codes under one coherent meta-idea that is not divisible without losing meaning. It is the smallest meaningful thematic unit at that level of abstraction.
- ABSTRACTION LEVEL: This indicates how conceptual vs. concrete the atomic theme is, relative to the taxonomy axis and coding dimension.
</key_definitions>

You will be working with the following parameters:

<language>
 {language}
</language>

<context>
- Domain: {domain}
- Topic: {topic}
Survey Question: "{survey_question}"
</context>

<new_theme>
New theme:
- name: "{theme_name}"
- description: "{theme_description}"
- Included expressions (these SHOULD be covered by the code):
  {inclusion}
</new_theme>

Here is the taxonomy framework guiding your analysis:
<taxonomy_parameters>
Taxonmy Axis:  {taxonomy_axis}
Axis description: {taxonomy_axis_description}
primary Coding Dimension: {taxonomy_actionable_type}
</taxonomy_parameters>

LABEL RULES (strict):
- Short noun phrase (≤ 10 words) of an ATOMIC theme in light of the research question, taxonomy axis and coding dimension.
- Make the semantic core of the theme the head of the noun phrase.
- Adhere to the atomicity rules below.
- DO NOT repeat the actor, domain, topic or entity in the label ({perspective}, {domain}, {topic} and {entity}).

ATOMICITY  RULES (strict):
- All categorical codes, labels and names  must be single-valued.
- Do not output multi-labels, spans, or hybrids.
- When uncertainty exists, commit to one value and record uncertainty only in supporting text (definition / near-neighbor / notes).
- NEVER include reasons (no "to", "so that", "because").
- NEVER use punctuation: "/", "&", ",", "–", ":" (unless lexicalized).

DEFINITION RULES:
- ≤ 30 words
- grounded in cluster data
- Must describe **what belongs in this code**, not why it happens.
- Must align directly with the survey question, taxonomy axis, and coding dimension.
- Use a **clear, observable assignment cue** (e.g., behaviors, expressions, judgments).
- Do not explain causes, conditions, or interpretations.
- DO NOT repeat the actor, domain, topic or entity in the description ({perspective}, {domain}, {topic} and {entity}).

GOOD DEFINITION PATTERNS:
- "References to…"
- "Mentions of…"
- "Expressions of…"
- "Concerns about…"

AVOID:
- Broad summaries (e.g., "general dissatisfaction").
- Multi-part or layered meaning.
- Psychological interpretation not grounded in wording.

ASSIGNMENT EXAMPLES:
- Provide concrete, actionable assignment examples to guide future code assignment
- inclusion: 2-3 short examples of expressions that SHOULD be coded here
- exclusion: 1-2 short examples of what should NOT be included
- near_neighbor: Identify closest confusable concept and how to tell them apart

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
VERTICAL_INSTRUCTIONS = """
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

HIERARCHICAL_INSTRUCTIONS = """
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
Your task is to MODIFY an existing code so that it fully and correctly includes a new theme, while preserving **atomic meaning** and **clear conceptual boundaries**.

Before beginning your analysis, understand these key definitions:

<key_definitions>
- TAXONOMY AXIS: This is the conceptual dimension along which all themes will be organized. It defines what kind of phenomenon we are theming. It sets the "coordinate system" for the analysis.
- CODING DIMENSION: This is the specific analytic focus within the taxonomy axis that determines where themes should be actionable. It narrows the axis into a practical working slice.
- ATOMIC THEME: This is a central organizing concept that groups descriptive codes under one coherent meta-idea that is not divisible without losing meaning. It is the smallest meaningful thematic unit at that level of abstraction.
- ABSTRACTION LEVEL: This indicates how conceptual vs. concrete the atomic theme is, relative to the taxonomy axis and coding dimension.
</key_definitions>

You will be working with the following parameters:

<language>
 {language}
</language>

<taxonomy_parameters>
Taxonmy Axis:  {taxonomy_axis}
Axis description: {taxonomy_axis_description}
primary Coding Dimension: {taxonomy_actionable_type}
</taxonomy_parameters>

<context>
- Domain: {domain}
- Topic: {topic}
Survey Question: "{survey_question}"
</context>

New theme to integrate:
- name: "{theme_name}"
- description: "{theme_description}"
- Included expressions (these SHOULD be covered by the code):
  {inclusion}

Original code (to be modified):
- code_label: {source_code}
- code_definition: {source_definition}

Current assignment examples (before modification):
- Current inclusion examples:
  {current_inclusion}
- Current exclusion examples:
  {current_exclusion}
- Current near neighbor boundary:
  {current_near_neighbor}

Required modifications:
- inclusion_update (new expressions that must now be included in-scope):
  {inclusion_update}
- exclusion_update (boundaries to clarify so scope does not overextend):
  {exclusion_update}
</inputs>

Follow these instruction exactly and in order. Do not skip or reorder any instruction.

<coding_instructions>
MODIFICATION INSTRUCTIONS:
{modification_instructions}

Here is the taxonomy framework guiding your analysis:
<taxonomy_parameters>
Taxonmy Axis:  {taxonomy_axis}
Axis description: {taxonomy_axis_description}
primary Coding Dimension: {taxonomy_actionable_type}
</taxonomy_parameters>

LABEL RULES (strict):
- Short noun phrase (≤ 10 words) of an ATOMIC theme in light of the research question, taxonomy axis and coding dimension.
- Make the semantic core of the theme the head of the noun phrase.
- Adhere to the atomicity rules below.
- DO NOT repeat the actor, domain, topic or entity in the label ({perspective}, {domain}, {topic} and {entity}).
- SEMANTIC PRESERVATION: The modified label must preserve the semantic core of the original code. You are BROADENING the code's coverage, not REPLACING its meaning. If the theme would require changing what the original code fundamentally represents, this indicates CREATE should have been chosen instead.

ATOMICITY  RULES (strict):
- All categorical codes, labels and names  must be single-valued.
- Do not output multi-labels, spans, or hybrids.
- When uncertainty exists, commit to one value and record uncertainty only in supporting text (definition / near-neighbor / notes).
- NEVER include reasons (no "to", "so that", "because").
- NEVER use punctuation: "/", "&", ",", "–", ":" (unless lexicalized).

DEFINITION RULES:
- ≤ 30 words
- grounded in cluster data
- Must describe **what belongs in this code**, not why it happens.
- Must align directly with the survey question, taxonomy axis, and coding dimension.
- Use a **clear, observable assignment cue** (e.g., behaviors, expressions, judgments).
- Do not explain causes, conditions, or interpretations.
- DO NOT repeat the actor, domain, topic or entity in the description ({perspective}, {domain}, {topic} and {entity}).

GOOD DEFINITION PATTERNS:
- "References to…"
- "Mentions of…"
- "Expressions of…"
- "Concerns about…"

AVOID:
- Broad summaries (e.g., "general dissatisfaction").
- Multi-part or layered meaning.
- Psychological interpretation not grounded in wording.

ASSIGNMENT EXAMPLES:
- Provide concrete, actionable assignment examples to guide future code assignment
- inclusion: 2-3 short examples of expressions that SHOULD be coded here
- exclusion: 1-2 short examples of what should NOT be included
- near_neighbor: Identify closest confusable concept and how to tell them apart

ASSIGNMENT EXAMPLES INSTRUCTIONS:
    - Update assignment examples to reflect the modified code:
      • inclusion: Combine original + new expressions from inclusion_update
      • exclusion: Combine original + new boundaries from exclusion_update
      • near_neighbor: Update label/rule if boundaries changed due to modification
</coding_instructions>

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

VALIDATION_PROMPT = """
You are a codebook curator for thematic analysis following Braun & Clarke (2006) methodology.
Your role is to maintain parsimonious codebooks with non-overlapping and non-redundant codes by reviewing and making final decisions on coding proposals.
MECE is your guiding principle - codes must be Mutually Exclusive and Collectively Exhaustive.

Before beginning your task, understand these key definitions:

<key_definitions>
- TAXONOMY AXIS: The conceptual dimension along which all themes are organized. It defines what kind of phenomenon we are theming and sets the "coordinate system" for the analysis.
- CODING DIMENSION: The specific analytic focus within the taxonomy axis that determines where themes should be actionable. It narrows the axis into a practical working slice.
- ATOMIC THEME: A central organizing concept that groups descriptive codes under one coherent meta-idea that is not divisible without losing meaning. It is the smallest meaningful thematic unit at that level of abstraction.
- ABSTRACTION LEVEL: How conceptual vs. concrete the atomic theme is, relative to the taxonomy axis and coding dimension.
</key_definitions>

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
- The new code should NOT cover these expressions:
  {exclusion_examples}
- near_neighbor: {near_neighbor_label} (Tell apart: {tell_apart_rule})
</coding_proposal>

<validation_criteria>
a. Parsimony: Has sufficient effort been made to use existing codes or combinations before proposing new/modified codes?
b. Abstraction Level: Is any proposed new code at an appropriate abstraction level, consistent with existing codes?
c. Non-Redundancy: Does the proposal avoid creating codes that significantly overlap with existing ones?
d. Atomicity: do labels adhere to these atomicity rules:
- All categorical codes, labels and names  must be single-valued.
- Do not output multi-labels, spans, or hybrids.
- When uncertainty exists, commit to one value and record uncertainty only in supporting text (definition / near-neighbor / notes).
- NEVER include reasons (no "to", "so that", "because").
- NEVER use punctuation: "/", "&", ",", "–", ":" (unless lexicalized).
e. DO NOT repeat the actor, domain, topic or entity in the label or names ({perspective}, {domain}, {topic} and {entity}).
f. Semantic Preservation (MODIFY only): If the decision is MODIFY, does the new label preserve the semantic core of the original code being modified? A MODIFY should BROADEN a code's coverage while keeping its original meaning intact. If the new label shifts or replaces the core concept rather than extending it, reject the MODIFY and recommend CREATE instead.
</validation_criteria>

Your task is to systematically evaluate the coding proposal against the validation criteria and make a final coding decision. Work through your evaluation step-by-step in a scratchpad before providing your final response.

<scratchpad>
Work through your evaluation systematically following these steps:

**Step 1: Evaluate the CREATE/MODIFY/USE decision**
- Parsimony: Has sufficient effort been made to use existing codes or combinations before proposing new/modified codes?
- Abstraction Level: Is any proposed new code at an appropriate abstraction level, consistent with existing codes?
- Non-Redundancy: Does the proposal avoid creating codes that significantly overlap with existing ones?

**Step 2: Evaluate code label and definition quality**
- Atomicity:
  • Does it express only one idea (no merged/compound themes)?
  • Does it avoid forbidden punctuation: "/", "&", "+", ",", ";", ":", "-", "–" (unless lexicalized within a compound noun)?
  • Does it contain at most ONE main action (verb)?
  • Does it avoid conjunctions ("and/or") unless lexicalized?
  • Does it avoid reasons (no "to", "so that", "because")?

- Form & Length:
  • Is the label ≤10 words with no canonical subject from survey question and no implied actor?
  • Does it follow allowed forms: noun phrase (<adjective(s)> <noun>), imperative verb + object (<verb> <object>), or infinitive (<to/infinitive verb> <object>)?
  • Is the definition ≤25 words, operational/observable, grounded in responses, and non-vague?

- Alignment:
  • Does the label and definition align directly with the survey question, taxonomy axis, and coding dimension?
  • Does the definition describe **what belongs in this code**, not why it happens?
  • Does the definition use a **clear, observable assignment cue** (e.g., behaviors, expressions, judgments)?
  • Does it avoid repeating the actor, domain, topic or entity in the label or definition ({perspective}, {domain}, {topic} and {entity})?

**Step 3: Determine APPROVE or REJECT**
- If all criteria PASS → APPROVE (you may make minor refinements for full compliance)
- If any criterion FAILS → REJECT (identify issues and rewrite to comply)

**Step 4: If rejected, make final decision**
- **USE**: An existing code already captures the new theme's central meaning sufficiently
- **MODIFY**: An existing code is close but needs refinement for clarity, scope, or better alignment
- **CREATE**: No existing code sufficiently captures the new theme

**Step 5: Validate assignment examples**
- Ensure inclusion examples align with the refined code/definition
- Ensure exclusion examples maintain clear boundaries
- Verify near_neighbor and tell_apart_rule are still accurate
- Refine if needed to match validated code

**Step 6: Determine final components**
- validated_decision: USE, MODIFY, or CREATE
- source_code: If USE, the exact code from the proposal; if MODIFY, the exact code from the existing codebook you seek to modify; if CREATE, write "null"
- validated_code: final compliant label, definition, and assignment_examples
- decision_rationale: brief explanation of approval/rejection
</scratchpad>


Now provide your final evaluation as valid JSON in the specified language. Return ONLY the JSON response with no additional text, comments, or extra fields. Use this exact schema:

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
    "validated_decision" : "USE | MODIFY | CREATE" ,
    "source_code": "If USE, this exact code: {source_code}; If MODIFY, the exact code from the existing codebook you seek to modify - or null, if CREATE",
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

If USE, this exact code: {source_code}; If MODIFY, the exact code from the existing codebook you seek to modify - or null, if CREATE

Critical remarks:
- Use theme_id provided.
- Use theme_name provided.
- Use source_code provided
"""

#


# =============================================================================
# STEP 7 THEME ORGANIZATION WITH REASONING MODELS
# =============================================================================

CODEBOOK_REFINEMENT_PROMPT = """
You are a qualitative research methodologist and codebook architect.
Your task is to transform a raw list of descriptive codes into a MECE (Mutually Exclusive, Collectively Exhaustive) codebook.

Here is the survey question:
<survey_question>
{survey_question}
</survey_question>

Here are the raw codes to refine:
<raw_codes>
{raw_codes}
</raw_codes>

Output your response in this language:
<language>
{language}
</language>

# Core Definitions (read carefully)
- Atomic label = a single, irreducible evaluative idea (no conjunctions like "en/and/&", "of/or", no slashes "/", no hypen "-", no bundled meanings). Example: “Waarde”, “Gezondheid”.
- Meta-theme = a higher-order organizational container for multiple distinct atomic themes within the same analytic domain. A meta-theme is NOT atomic, but its label must still be a single, clean concept word/phrase (e.g., “Productverwachtingen”).
- Theme = either (a) an atomic evaluative construct, or (b) a meta-theme that groups multiple atomic subthemes (use only when it improves coder clarity or reporting).

# Core Rules

## 1) Enforce MECE (Mutually Exclusive, Collectively Exhaustive)
Preserve all distinct conceptual meanings. If two codes differ only in wording, phrasing, tone, or surface form—and a trained coder would treat them as the same action or recommendation—MERGE them into a single code.

**Key Merge Criteria**
- Operational Action: If the recommended intervention would be the same → MERGE
- Reporting Test: If results would be reported as one insight → MERGE

**Decision Tests**
- Action Test (ACT): “Is the practical implication the same given the survey question?” If yes → merge
- Reporting Test (RT): “Would a researcher distinguish them in the results?” If no → merge
- Examples Test (XT): If inclusion examples point to the same expression types AND exclusion examples do not reveal boundaries → merge

## 2) Use Assignment Examples (if present)
Raw codes may include:
- inclusion examples
- exclusion examples
- near_neighbor (with tell-apart rules)

Apply them as follows:
a) Inclusion overlap → candidate merge
b) Exclusion conflicts → keep separate (boundaries exist)
c) Near neighbors with tell-apart rules → respect separation unless the difference is purely linguistic (then merge)

If examples are missing, infer cautiously from code labels and context; prefer precision over over-merging.

## 3) Structure and Hierarchy
- Every code must belong to exactly one parent theme.
- Themes must be conceptually non-overlapping.
- Use a **3-level hierarchy (Theme → Subtheme/Category → Code)** whenever it improves coder decisions or reporting clarity.
  - Use a subtheme/category only when ≥2 codes clearly share a specific sub-idea that aids assignment.
- A **2-level hierarchy (Theme → Code)** is acceptable if all themes are atomic and subthemes would not add clarity.

## 4) Theme Naming and Descriptions

**Theme Labels**
- ≤ 10 words
- Describe what is being evaluated in terms specific to the survey question
- Atomic (single idea; one polarity: either increase/strengthen OR reduce/avoid)
- Prefer noun phrases; avoid generic labels (e.g., “Quality”)
- No reasons or purposes (avoid “to…”, “so that…”, “because…”)
- Avoid punctuation: “/”, “&”, “,”, “–”, “:” (unless lexicalized)
- Avoid synonym duplication across themes; choose one canonical term

**Code Descriptions**
- ≤ 20 words
- Define what belongs in this code (observable assignment cues); not causes or interpretations
- Align directly with the survey question
- Use patterns like: “Mentions of…”, “References to…”, “Expressions of…”, “Concerns about…”

## 5) Atomicity & Label Hygiene Checks
Before finalizing:
- Atomicity Test (AT): Can each label be expressed as one evaluative lens? If not, split or reassign under a meta-theme.
- Boundary Test (BT): Are differences between themes/subthemes clear and stable?
- Wording Test (WT): Labels and codes must avoid conjunctions/slashes/comma lists; keep concise and specific.

# Required Output Format

First, think through the structure step-by-step in <analysis_thinking> tags. Consider:
- Which codes should be merged and why (reference assignment_examples where available)
- Which similar codes should be kept separate and why
- How to structure the hierarchy (2-level vs. 3-level; where categories help)
- Total codes preserved vs. merged count

Then provide your response as valid JSON only, structured exactly as follows: 
    
{{
  "analysis": "Provide detailed analysis in {language}: (1) Which codes were merged and why (include IDs and reference assignment_examples to justify), (2) Which similar codes were kept separate and why (reference inclusion/exclusion examples or near_neighbor boundaries), (3) How hierarchy was structured, (4) Total codes preserved vs. merged count.",
  "refined_codebook": [
    {{
      "theme": "Main theme label",
      "codes": [
        {{
          "id": "original code_id (or comma-separated IDs if merged)",
          "code": "Code label",
          "description": "≤ 20 words explanation",
          "category": ""  // Empty string for 2-level, or category name for 3-level
        }}
      ]
    }}
  ]
}}

Notes:
- Use empty string for "category" in 2-level hierarchy; otherwise use a category name for 3-level (subtheme).
- No commentary before or after JSON.
- No markdown formatting outside of code blocks.
- All text must be in the specified output language.

Begin your analysis and provide the refined codebook.
"""


CODEBOOK_MERGE_PROMPT = """
You are a qualitative research methodologist performing final codebook consolidation.
You will be given multiple independent codebooks from different subsets of survey responses. Your task is to produce one unified, MECE (Mutually Exclusive, Collectively Exhaustive) codebook.

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

# Core Definitions (read carefully)
- Atomic label = a single, irreducible evaluative idea (no conjunctions like "en/and/&", "of/or", no slashes "/", no hypen "-", no bundled meanings). Example: “Waarde”, “Gezondheid”.
- Meta-theme = a higher-order organizational container for multiple distinct atomic themes that belong to the same analytic domain. A meta-theme is NOT atomic, but its LABEL must still be a single, clean concept word/phrase (e.g., “Productverwachtingen”).
- Theme = either (a) an atomic evaluative construct, or (b) a meta-theme that groups multiple atomic subthemes (use only when it truly improves clarity).

# Your Task
You have multiple codebooks. Some themes will:
1) Appear in multiple codebooks (duplicates) → MERGE
2) Be unique (distinct) → KEEP
3) Overlap partially → Decide merge vs keep

# Consolidation Principles
- Parsimony: Keep as simple as possible while preserving conceptual clarity.
- Non-redundancy: No two themes/codes state the same meaning.
- Mutual Exclusivity (ME): Each code should logically belong to only one theme.
- Collective Exhaustiveness (CE): Themes jointly cover all meaningful responses.
- Conceptual Coherence: Codes under a theme are variations of the same underlying idea.
- Atomicity: Labels at the atomic level express one clear idea (no multi-concepts).
- Action/Reporting Alignment: Merge if they would be reported together or lead to the same recommendation.
- Boundary Clarity: Differences between themes are obvious and justifiable to another researcher.

# Decision Tests (apply explicitly)
- Atomicity Test (AT): Can the theme be expressed as ONE evaluative lens? If any subparts require different “whys”, it’s NOT atomic.
- Action Test (ACT): Would two candidate themes lead to the same action/recommendation? If yes, merge.
- Boundary Test (BT): Are differences between two themes clear and stable? If yes, keep separate.
- Wording Test (WT): Labels must be ≤10 words, contain no “and/&/ /” or comma lists, and preferably be noun phrases.
  (Note: WT applies to both themes and codes; use concise, atomic wording.)

# Handling Composite Labels
If an input theme label is composite (fails AT or WT), do one of:
- Preferred option A (Group): Create a META-THEME with a single, clean label that expresses their shared domain (not a conjunction), and place the atomic subthemes beneath it; or
- Option B (Split): Convert it into multiple atomic themes.
Schema note: When you use a meta-theme in the final JSON, put the meta label in "theme" and use the "category" field on each code to name the atomic subtheme/category beneath that meta-theme.

Choose the option that best improves MECE and reporting clarity.

# Handling Duplicate Codes
If the same/very similar code appears in multiple themes:
1) Choose the BEST semantic fit (use ACT, BT, and AT).
2) Assign it to ONLY that theme (ensuring mutual exclusivity).
3) Note the reassignment in the analysis log.

# Hierarchy Preference
- Use a **3-level structure (Meta-theme → Atomic Theme/Category → Code)** whenever it improves conceptual or reporting clarity.
  - The meta-theme groups multiple distinct but related atomic themes under one analytic domain.
  - In the JSON output, represent the meta-theme as the "theme" and use the "category" field on each code to indicate the atomic subtheme/category.
- A **2-level structure (Theme → Code)** may be used only if all themes are atomic and no meta-themes are needed.

# Label Rules (strict)
- No conjunctions (“en/of/and/or/&/-”), no slashes (“/”), no comma-joined lists.
- Prefer ≤6-word noun phrases where possible; never exceed WT’s ≤10-word limit.
- Avoid synonyms across labels; pick one canonical term per concept.
- Each code needs a crisp ≤30-word definition focused on when to use the code.

# Scratchpad (required, hidden in final)
<scratchpad>
Think through, step by step, in {language}:
1) Which themes are duplicates across codebooks? State merges and the reason (ACT + AT).
2) Which themes are distinct and should remain separate? Justify (BT).
3) Which input labels are composite? For each, choose Option A (Split to atomic themes) or Option B (Create meta-theme) and justify.
4) How were duplicate or cross-assigned codes resolved (final placements)?
5) Final theme count vs. input theme count; why this is more parsimonious and still CE.
6) Whether 2-level or 3-level hierarchy is used and why.
7) Run the Wording Test on ALL final labels; rewrite any violations.
</scratchpad>

# Output (JSON only; no commentary before or after)
Provide your final answer as valid JSON only, with no commentary before or after:

{{
  "analysis":  "In [language]: (1) Which themes were merged across codebooks and why (list specific theme names), (2) Which themes were kept separate and why, (3) How duplicate codes were resolved, (4) Final theme count vs input theme count, (5) Rationale for hierarchy structure.",
  "refined_codebook": [
    {{
      "theme": "Final theme label (≤10 words)",
      "codes": [
        {{
          "id": "original code ID(s) from input codebooks",
          "code": "Code label",
          "description": "Code definition (≤30 words)",
          "category": ""  // Empty for 2-level, category name for 3-level (atomic subtheme when using a meta-theme)
        }}
      ]
    }}
  ]
}}
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


