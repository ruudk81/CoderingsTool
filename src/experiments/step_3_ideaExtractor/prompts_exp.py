"""
Experimental Prompts for Step 3: Idea Extraction

This file contains the prompts used by ideaExtractor.py.
Modify these prompts to experiment with different idea extraction approaches.

Original source: src/prompts.py (STEP 3: IDEA EXTRACTION section)
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

3. **entity**: Main entity of interest
   - What entity (group, person or thing) is the primary focus?
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

Return ONE consolidated set of GROUP 2 specifiers.
"""

TAXONOMY_CHUNK_SCORING_PROMPT = """
You are selecting the SINGLE best taxonomy axis for organizing a set of survey responses.

Your task is NOT to summarize responses, judge quality, or assign labels to each response.
Your ONLY goal is to decide which ONE axis — WHAT, WHY, HOW, WHO, WHEN, or WHERE — best explains the MAIN way the responses DIFFER from one another.

Here is the language you will be working in:
<language>
{language}
</language>

Here is contextual information about the survey question:
<context>
- Domain: {domain}: {entity}
- Topic: {topic}
</context>

Here is the type of respondent who answered the question:
<respondent_type>
{perspective}
</respondent_type>

Here is the survey question that was asked:
<survey_question>
{survey_question}
</survey_question>

Here is the intent behind the responses:
<intent>
{intent}
</intent>

Here is a sample of SHORT, COARSE responses for you to analyze:
<sample_responses>
{chunk_responses}
</sample_responses>

------------------------------
HOW TO THINK ABOUT THE TASK
------------------------------
Ask yourself:
“If I had to cluster these responses into groups, which axis would create the most meaningful separation, given the intent ("{intent}") in answering the survey question?”

Choose the axis that explains the LARGEST share of meaningful variation across MOST responses (not just edge cases).
Prefer the axis that yields the cleanest MECE coding scheme downstream (mutually exclusive and collectively exhaustive).

If multiple axes seem plausible:
- Choose the axis that would be used as the *top-level folder* to organize these responses.
- If still tied, choose the axis that applies to a larger fraction of responses.

------------------------------
TAXONOMY AXES (choose exactly one)
------------------------------

1) WHY (reason_driver)
- Differences are motivations, goals, values, concerns, or trade-offs.
- Excludes: attributes (WHAT), methods (HOW), actors (WHO), timing (WHEN), context/channel (WHERE).

2) HOW (outcome_enablers)
- Differences are about how an outcome would be achieved or carried out, including:
   A) Change-enabling mechanisms: actions, changes, interventions, tools, or mechanisms that make the outcome possible  
   B) Execution pathways: steps, processes, workflows, procedures, or ways of carrying something out  
- Includes: recommendations, tactics, methods, implementation approaches, processes, or preferred ways of “getting from here to there.”
- Excludes: what something *is or has* (WHAT), why someone wants something (WHY), who is involved (WHO), timing (WHEN), context/channel (WHERE).

3) WHO (actor_target)
- Differences are who is involved, affected, targeted, or responsible.
- Excludes: methods (HOW), motivations (WHY), attributes (WHAT), timing (WHEN), context/channel (WHERE).

4) WHEN (time_urgency)
- Differences are timing, urgency, frequency, sequence, or lifecycle stage.
- Excludes: methods (HOW), motivations (WHY), actors (WHO), attributes (WHAT), context/channel (WHERE).

5) WHERE (location_context)
- Differences are environment, setting, channel, platform, touchpoint, or situation.
- Excludes: methods (HOW), motivations (WHY), actors (WHO), timing (WHEN), attributes (WHAT).

6) WHAT (entity_descriptor)
- Differences are properties, attributes, features, or constraints of the entity as it currently exists or has existed.
- This is descriptive, not prescriptive.
- Excludes: 
  - desired changes or improvements (these belong to HOW),
  - motivations (WHY),
  - actors (WHO),
  - timing (WHEN),
  - context/channel (WHERE)

------------------------------
ANALYSIS PROCESS (internal)
------------------------------
Do NOT output your step-by-step reasoning.
You MUST still follow this process internally:
1) Identify the dominant pattern of variation across the sample, in light of the intent ("{intent}") in answering the survey question.
2) Score each axis for explanatory power over the variation: 0 = absent, 1 = present but secondary, 2 = primary.
3) Choose the single axis with the highest score (break ties using the rules above).
4) Extract 2–3 verbatim snippets from <sample_responses> that support the chosen axis.
5) Write a 1–2 sentence axis description that enables MECE coding downstream.

------------------------------
OUTPUT INSTRUCTIONS
------------------------------
- Return JSON format with keys in English and values in {language}.
- All string values (including evidence snippets) must be in {language}.
- Evidence snippets must be copied verbatim from <sample_responses>.
- If fewer than 3 distinct snippets exist, include as many as possible without inventing any.
- Clarification must explicitly contrast the chosen dimension with at least one plausible alternative.
"""

TAXONOMY_CONSOLIDATION_PROMPT = """
You are a taxonomy consolidation specialist. 
Your task is to analyze multiple chunk-level taxonomy analyses and consolidate them into a single, coherent global taxonomy axis for a survey question.


Here is the language the survey responses are written in:
<language>
{language}
</language>

Here is the survey question that was asked:
<survey_question>
{survey_question}
</survey_question>

Here is contextual information from prior analysis:
<context>
- Domain: {domain}: {entity}
- Topic: {topic}
- Type of respondent: {perspective}
- Intent by response: {intent}
</context>

There are six possible taxonomy axes (organizing dimensions) that can structure how responses are coded:

<taxonomy_axes>
1) **WHAT** — attributes, features, properties, or aspects used to describe someone or something
2) **WHY** — underlying reason, motivation, rationale, goal, values sought or concern that explain preference, behaviour or response
3) **HOW** — concrete recommendations, suggestions, actions or interventions to carry out an activity or enable a desired outcome
4) **WHO** — people, groups, roles, stakeholders, or beneficiaries who are involved, affected, responsible, or addressed
5) **WHEN** — timing, urgency, sequence, or frequency associated with when something occurs or is expected
6) **WHERE** — physical or digital location, channel, setting, or situational context in which something occurs or is encountered
</taxonomy_axes>

Here are the chunk-level analyses you need to consolidate:
<chunk_level_analyses>
{chunk_results}
</chunk_level_analyses> 

## YOUR TASK

You must consolidate these chunk-level analyses into a single global taxonomy axis. Each chunk analysis evaluated the same survey question and produced a primary coding dimension, a specific taxonomy axis, and supporting evidence. Your job is to synthesize these into one coherent framework.

## ANALYSIS STEPS

Follow these steps in order:

**Step 1: Review and consolidate chunk-level analyses**
Examine all chunk-level analyses carefully. Note areas of convergence and divergence. Identify which dimensions appear across multiple chunks and assess the quality of evidence supporting each.

**Step 2: Select the PRIMARY taxonomy dimension**
Choose the ONE dimension (WHAT, WHY, HOW, WHO, WHEN, or WHERE) that:
- Shows strong and consistent support across chunks
- Provides the clearest partition boundaries for coding responses
- Offers the best interpretability and stability for downstream use

Important: Do NOT select a dimension solely because it appears most frequently. Favor partition clarity, boundary stability, and interpretability over raw frequency counts.

**Step 3: Define the GLOBAL taxonomy axis**
Write a taxonomy axis description that:
- Is specific to THIS survey question and response domain
- Clearly falls within the selected primary dimension
- Reconciles and generalizes the chunk-level axes without introducing new organizing principles
- Operates at a mid-level of abstraction (not too narrow, not too broad)
- Can directly seed downstream descriptive code labels
- Clearly indicates what coders should extract from each response

## DECISION RULES

When consolidating:
- If chunk analyses converge on the same dimension, follow the consensus
- If chunk analyses diverge, rely on MECE quality (mutually exclusive, collectively exhaustive) to determine which dimension provides the clearest boundaries
- Optimize for downstream coding usability and cross-coder consistency
- Prefer clarity and stability over cleverness or novelty

## OUTPUT INSTRUCTIONS
- Return JSON format with keys in English and values in {language}.
"""


TAXONOMY_AWARE_SUBJECT_PROMPT = """
You are a language expert tasked with generating a precise phrasing template for survey response analysis.  
You will be given context about a survey question and must produce a structured JSON output that follows all constraints exactly.

Here is the language you will be working in:
<language>
{language}
</language>

Here is the survey question you are analyzing:
<survey_question>
{survey_question}
</survey_question>

Here is the context from prior analysis:
<context>
- Domain: {domain}: {entity}
- Topic: {topic}
- Type of respondent: {perspective}
- Intent by response: {intent}
</context>

Here is the taxonomy guidance you must follow:
<taxonomy_guidance>
Taxonomy axis: {primary_dimension}
Taxonomy axis description: {primary_dimension_description}
</taxonomy_guidance>

Here is the template guidance you must use:
<template_guidance>
Template pattern: "{{SUBJECT}} {{VERB_STATE}} {{SCAFFOLD}} [ACTIONABLE_TAXONOMY_DIMENSION]."

Required form (json format): 
{required_form_json}

Slot guidance in (json format): 
{slot_guidance_json}
</template_guidance>

Your task has three parts:

--------------------------------------------------
**TASK 1 — Identify the canonical subject (entity-of-interest)**  
Goal: choose the noun phrase that the template sentence should be about — the central entity being evaluated or described.

Rules:
- It must be a noun or noun phrase (not a full clause).
- Avoid placeholder subjects (e.g., “it,” “there”).
- Avoid pronouns (“I,” “you,” “we”) unless the survey perspective explicitly requires them.
- If the question is framed around the respondent (e.g., “How satisfied are you with X?”), choose **X**, not “you.”
- If multiple entities appear, select the one most aligned with:
  (a) Primary focus: {entity}  
  (b) Primary taxonomy axis: {primary_dimension}  
- Return a concise, normalized noun phrase in {language}.  
- Preserve capitalization for proper nouns; otherwise use lowercase.

--------------------------------------------------
**TASK 2 — Choose the actionable taxonomy dimension**  
Select **exactly one** actionable taxonomy type on the primary axis that best captures what is being evaluated or varied in the survey question.

Use this as your guide:
{primary_dimension_description}

--------------------------------------------------
**TASK 3 — Create a phrasing template**  

Your output field **canonical_phrasing** MUST:

1) Follow this pattern exactly:
   "{{SUBJECT}} {{VERB_STATE}} {{SCAFFOLD}} [ACTIONABLE_TAXONOMY_DIMENSION]."

2) Use:
   - SUBJECT from Task 1  
   - an appropriate minimal VERB_STATE (e.g., “is,” “are,” “has,” etc.). Do not use modal verbs. 
   - a SCAFFOLD that naturally connects the subject to the marker  

3) **CRITICAL: Do NOT replace the marker.**  
   The literal token **[ACTIONABLE_TAXONOMY_DIMENSION]** must appear exactly as written, as the final bracketed term in the sentence (a period may follow).

4) The full sentence must read like a **natural {language} answer** to the survey question.

Mental check:
If the marker were replaced with a short phrase (e.g., “battery life,” “price,” “speed”), the sentence should sound like a direct, grammatical answer to the question.

OUTPUT INSTRUCTIONS
- Return JSON format with keys in English and values in {language}.
"""

TAXONOMY_ENRICHED_EXTRACTION_PROMPT = """
You are an expert in extracting structured ideas from survey responses using taxonomy-aware analysis.

Your task is to:
1) Identify all distinct ideas in the response,
2) Reformulate each idea using a provided template prefix,
3) Classify each idea according to the specified taxonomy axis, and
4) Position each idea within a lightweight conceptual ontology aligned to that axis.

================================================
SURVEY CONTEXT
================================================

<survey_context>
Language of responses: {language}
Survey question: {var_lab}

Domain: {domain}
Topic: {topic}
Main entity of interest: {entity}

Type of respondent: {perspective}
Dominant response frame: {intent}
</survey_context>

================================================
TAXONOMY CONFIGURATION (AUTHORITATIVE)
================================================

<taxonomy_config>
Taxonomy dimension (axis): {taxonomy_axis}
Actionable type: {taxonomy_actionable_type}
Primary description: {primary_dimension_description}

Lookup-table axis definition:
- Dimension description: {axis_dimension_description}
- Required form for slot content: {axis_required_form}
- Canonical template pattern (reference): {axis_template_pattern}

Slot guidance (reference):
{axis_slot_guidance}

Axis-specific rules (from lookup table):
- Node instruction: {axis_node_instruction}
- Category instruction: {axis_category_instruction}
- Taxonomy phrase instruction: {axis_taxonomy_phrase_instruction}

Additional focus rules:
{axis_focus_rules}
</taxonomy_config>

================================================
RESPONSE TO PROCESS
================================================

<response>
Respondent ID: {respondent_id}
Response: {response}
</response>

================================================
IDEA SPLITTING RULES
================================================

- Identify all conceptually distinct ideas.
- If multiple independent aspects are mentioned, split them into separate ideas.
- Each atomic concept becomes its own idea with its own idea_id.
- If the response is empty, nonsensical, or irrelevant, return [].

================================================
OUTPUT FIELDS (FOR EACH IDEA)
================================================

Return a JSON array of objects. Each object must include:

1) respondent_id
- Use exactly the respondent ID provided above.

2) idea_id
- Assign a sequential number as a string ("1", "2", "3", ...).

3) idea (template-based reformulation)
Construct the complete idea statement. It MUST:
- Begin EXACTLY with this provided template prefix:
  "{canonical_phrasing}"
- Replace [ACTIONABLE_TAXONOMY_DIMENSION] with content consistent with:
  - Axis dimension description: {axis_dimension_description}
  - Required form: {axis_required_form}
- The replacement text must be 5–20 words.
- Be concise, specific, and grammatical.
- Directly answer the survey question when combined with the template prefix.
- Contain NO pronouns or references to the respondent.
- Contain NO filler phrases.
- Be written in {language}.

4) ontology
Provide a lightweight hierarchy aligned with the active axis.

ontology fields:
- instance:
  The SHORTEST contiguous verbatim span from the ORIGINAL response that captures the core idea.
  Rules:
  - MUST be copied exactly from the response (no paraphrasing).
  - Prefer minimal span; exclude asides/opinions/explanations unless essential.

- node:
  A canonical, reusable noun phrase that represents the PRIMARY AXIS CONCEPT.
  Apply: {axis_node_instruction}
  Additional rules:
  - Must be reusable across multiple responses.
  - Must be written in {language}.
  - Must not simply repeat the instance verbatim.

- category:
  The immediate parent grouping of the node suitable for clustering many responses.
  Apply: {axis_category_instruction}
  Additional rules:
  - Must be broader than node and stable across many responses.
  - Must be written in {language}.
  - Must not repeat the instance verbatim.

5) taxonomy_phrase
A concise reusable noun phrase abstracting the idea as a {taxonomy_actionable_type} concept on axis {taxonomy_axis}.
Apply: {axis_taxonomy_phrase_instruction}
General rules:
- Prefer 1–3 words.
- Do NOT repeat entities already mentioned in: {domain}, {topic}, {entity}.
- Avoid meta-language about opinions/perceptions/thoughts.
- Written in {language}.

6) sense
Choose exactly one of:
- factual | evaluative | aspirational | experiential

7) sentiment
Choose exactly one of:
- positive | negative | neutral

================================================
OUTPUT INSTRUCTIONS
================================================

Return valid JSON only.
- Keys must be in English.
- Values must be in {language}.

Edge cases:
- Empty or irrelevant response: return []
- Single idea: return one item
- Multiple ideas: return multiple items with sequential idea_id
"""



TAXONOMY_ENRICHED_EXTRACTION_PROMPT_old = """
You are an expert in extracting structured ideas from survey responses using taxonomy-aware analysis. 
Your task is to identify all ideas expressed in a survey response, classify them according to a given taxonomy axis, and format them according to a specific template structure.

Here is the survey context:

<survey_context>
language of responses: {language}
Survey question: {var_lab}

Domain: {domain}
Topic: {topic}
Main entity of intererst: {entity}

Type of respondent: {perspective}
Dominant response frame: {intent}
</survey_context>

Here is the taxonomy configuration:

<taxonomy_config>
Taxonomy dimension to be used: {taxonomy_axis}: {taxonomy_actionable_type}
Descriptopm: {primary_dimension_description}
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
   - Begin with EXACTLY this template provided: "{canonical_phrasing}"
   - Then replace the placeholder ACTIONABLE_TAXONOMY_DIMENSION] with the idea identified in the response.
   - The replacement text must be a complete, grammatical sentence fragment (5-20 words total)
   - Must directly address the survey question when combined with the template prefix
   - Must be concise and specific
   - Must NOT contain pronouns or references to the respondent
   - Must NOT contain filler words or unnecessary phrases
   - Must be written in the {language} specified in the survey context  

**4. ontology**
For each idea, identify its position in a conceptual hierarchy for the given taxonomy dimension: {taxonomy_axis} - {taxonomy_actionable_type}

You must extract:

- **instance**: the literal idea expressed in the response (verbatim, no paraphrasing)
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

**5. taxonomy_phrase**: - A concise noun-phrase that abstracts the idea into a reusable {taxonomy_actionable_type}-concept on the taxonomy axis ({taxonomy_axis}: {taxonomy_actionable_type})
   - Make the semantic core of the taxonomy_phrase the HEAD of the noun phrase
   - DO NOT repeat entities already mentioned in the domain context: {domain}, {topic} and {entity}
   - Prefer single-word attribute nouns over compound action-nouns
   - Avoid meta-language about perception, opinion, or thought
   - Avoid verbs or verb-noun compounds
   - Written in {language}.

**6. sense**: Choose exactly one of: factual | evaluative | aspirational | experiential
   - **factual**: objective observation, factual mention, neutral statement
   - **evaluative**: praise, approval, satisfaction, complaint, dissatisfaction, criticism
   - **aspirational**: suggestion, wish, desire, recommendation for improvement
   - **experiential**: personal experience, anecdote, lived encounter

**7. sentiment**: Choose exactly one of: positive | negative | neutral

**IDEA SPLITTING RULES:**

When a response contains multiple conceptually distinct aspects, split them into separate ideas. Each atomic concept should be extracted as its own idea with its own idea_id.

Example: If a response mentions both "sustainable investment options" and "lower fees", extract these as two separate ideas because sustainability and fees are conceptually distinct aspects.


**EXAMPLE:**

EXAMPLE

Survey question: "What improvements would you like to see in our public parks?"
Template: "The public park should have {{VERB_STATE}} {{SCAFFOLD}} [ACTIONABLE_TAXONOMY_DIMENSION]"

Response: "I'd love more shaded seating areas and better evening lighting for safety."

Extracted ideas:

[
  {{
    "respondent_id": "{{respondent_id}}",
    "idea_id": "1",
    "idea": "The public park should have more shaded seating areas.",
    "taxonomy_phrase": "shaded seating",
    "ontology": {{
      "instance": "more shaded seating areas",
      "node": "shade provision",
      "category": "amenity design changes",
      "root": "environmental intervention"
    }},
    "sentiment": "neutral",
    "sense": "aspirational"
  }},
  {{
    "respondent_id": "{{respondent_id}}",
    "idea_id": "2",
    "idea": "The public park should have better evening lighting for safety.",
    "taxonomy_phrase": "evening lighting",
    "ontology": {{
      "instance": "better evening lighting for safety",
      "node": "lighting improvement",
      "category": "amenity design changes",
      "root": "environmental intervention"
    }},
    "sentiment": "neutral",
    "sense": "aspirational"
  }}
]

OUTPUT INSTRUCTIONS
Return JSON format with keys in English and values in {language}.

Edge cases:
- Empty or irrelevant response: return []
- Single idea: return one item
- Multiple ideas: return multiple items with sequential idea_id

"""

# Helper dict for taxonomy axis descriptions
TAXONOMY_AXIS_DESCRIPTIONS = {
    "WHAT": "attributes, features, properties, or aspects being described or evaluated of someone or something ",
    "WHY": "underlying reason, motivation, rationale, goal, values sougt or concern that explain preference, behaviour or response",
    "HOW": "concrete actions, methods, behaviors, or implementation approaches that are proposed or implied to carry out an activity or enable a desired outcome",
    "WHO": "peoople, groups, roles, stakeholders, or beneficiaries who are involved, affected, responsible, or addressed ",
    "WHEN": "timing, urgency, sequence, or frequency associated with when something occurs or is expected",
    "WHERE": "physical or digital location, channel, setting, or situational context in which something occurs or is encountered"
}
