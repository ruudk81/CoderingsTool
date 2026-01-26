"""
Experimental prompts for ideaExtractor_v2

This file contains COPIES of the production prompts that can be modified
for experimentation without affecting production code.

To use experimental prompts:
    - Set USE_EXPERIMENTAL_EXTRACTOR = True (uses local extractor + these prompts)
    - Or set USE_EXPERIMENTAL_PROMPTS = True (uses production extractor + these prompts)

COPIED FROM: src/prompts.py (lines 134-459)
"""

# =============================================================================
# CONTEXT SPECIFIER PROMPTS
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

3. **intent**: Purpose/communicative function
   - What are respondents trying to do with their responses?
   - Common values: "evaluate", "describe", "suggest", "complain", "praise", "question"

Provide concise answers (2-5 words each) in {language}."""


CONTEXT_SPECIFIER_PROMPT2 = """
You are analyzing survey responses to extract contextual metadata.

Survey question: {survey_question}

Sample responses ({chunk_size} examples):
{chunk_responses}

Extract these GROUP 2 specifiers (subject matter):

1. **domain**: Industry/sector domain
   - What industry or sector does this survey concern?

2. **topic**: Specific subject matter
   - What is the specific topic being discussed?

3. **entity**: Main entity/subject
   - What specific organization, product, or brand is the primary focus?
   - Use lowercase with underscores for multi-word names

Provide concise answers (2-5 words each) in {language}."""


CONSOLIDATE_SPECIFIERS_GROUP1 = """
You are consolidating contextual metadata extracted from multiple chunks of survey responses.

Survey question: {survey_question}

Different chunks produced these GROUP 1 specifiers (speaker characteristics):

{chunk_results}

Your task: Consolidate these into ONE canonical set of specifiers.

Guidelines:
- Resolve semantic variations 
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
- Resolve semantic variations 
- For **domain**: Standardize to lowercase, single/hyphenated word
- For **topic**: Choose the most representative subject matter across all chunks
- For **entity**: Standardize format (lowercase_with_underscores)

If chunks agree: use the consensus value
If chunks disagree: choose the most frequently occurring concept (semantic similarity, not lexical match)

Return ONE consolidated set of GROUP 2 specifiers."""


# =============================================================================
# TAXONOMY AXIS PROMPTS (for taxonomy-aware extraction)
# =============================================================================

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


# =============================================================================
# TAXONOMY-ENRICHED IDEA EXTRACTION PROMPT
# =============================================================================     

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
-  A concise noun-phrase that abstracts the idea into a reusable {taxonomy_actionable_type} aligned with the taxonomy axis ({taxonomy_axis}: {taxonomy_axis_description})
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


# =============================================================================
# AXIS DESCRIPTIONS (helper for prompts)
# =============================================================================

TAXONOMY_AXIS_DESCRIPTIONS = {
    "WHAT": "topic_object - concepts, things, topics, features, attributes",
    "WHY": "intent_purpose - goals, desired outcomes, improvements, reasons",
    "HOW": "action_method - actions, steps, processes, methods, ways",
    "WHO": "actor_target - people, groups, stakeholders, beneficiaries",
    "SENTIMENT": "evaluation - judgment, opinion, positive/negative evaluation",
    "WHEN": "time_urgency - time references, urgency, sequence, timing",
    "WHERE": "location_context - place, context, channel, location"
}
