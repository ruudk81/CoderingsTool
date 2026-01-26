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


# =============================================================================
# TAXONOMY AXIS PROMPTS (for taxonomy-aware extraction)
# =============================================================================

TAXONOMY_CHUNK_SCORING_PROMPT = """
You are analyzing a chunk of survey responses to score taxonomy axis relevance.

Survey question: {survey_question}

Sample responses ({chunk_size} examples):
{chunk_responses}

## CORE PRINCIPLE

Score axes based on **COMMUNICATIVE MEANING**, not grammatical surface form.

Survey responses are often elliptical — respondents omit words that are implied by the question context. Before scoring, mentally expand each answer into a minimal complete proposition.

**Ellipsis expansion examples:**
- Question: "What could the producer improve?" Answer: "Lower prices"
  → Expanded: "The producer should offer lower prices" → WHY (desired improvement)
- Question: "What associations do you have?" Answer: "Sustainable"
  → Expanded: "The brand is sustainable" → WHAT (property/attribute)
- Question: "What do you like?" Answer: "Fast delivery"
  → Expanded: "I like that delivery is fast" → WHAT (appreciated feature)

## INTERPRETATION GUIDANCE

Use the **survey question** to guide interpretation:
- "What could X improve/do better?" → Responses are likely WHY (improvement suggestions)
- "What associations/characteristics?" → Responses are likely WHAT (attributes)
- "How should X do this?" → Responses are likely HOW (methods)
- "What do you like/dislike?" → Responses are likely WHAT (features being evaluated)

## TAXONOMY AXES

Score each axis (0.0-1.0) based on how well it captures the PRIMARY dimension:

1. **WHAT** (topic_object): Properties, characteristics, attributes of something
   - PRIMARY when respondents describe what something IS or HAS
   - "Fast delivery" (describing a feature), "Sustainable brand" (attribute)
   - NOT primary for desired changes or improvements (that's WHY)

2. **WHY** (intent_purpose): Goals, desired outcomes, improvements, reasons
   - PRIMARY when respondents express what SHOULD change or be achieved
   - "Lower prices" (desired change), "Better service" (improvement goal)
   - Includes elliptical improvement suggestions even if phrased as nouns

3. **HOW** (action_method): Actions, steps, processes, methods
   - PRIMARY when respondents describe procedures or ways of doing
   - "By automating checkout", "Through faster processing"

4. **WHO** (actor_target): People, groups, stakeholders, beneficiaries
   - PRIMARY when respondents focus on specific actors or affected parties
   - "Staff should...", "Customers need..."

5. **SENTIMENT** (evaluation): Judgment, opinion, positive/negative evaluation
   - PRIMARY when responses are primarily evaluative stance
   - Often SECONDARY (orthogonal to WHAT/WHY/HOW)

6. **WHEN** (time_urgency): Time, urgency, sequence, timing
   - PRIMARY when temporal aspects are the main dimension
   - "Faster", "More frequent", "Immediately"

7. **WHERE** (location_context): Place, context, channel, location
   - PRIMARY when location/context is the main organizing dimension
   - "In the app", "At checkout", "Online vs in-store"

## SCORING CRITERIA

- High (0.7-1.0): Axis captures the PRIMARY communicative function of most responses
- Medium (0.4-0.7): Axis is present but secondary
- Low (0.0-0.4): Axis rarely applies

For each axis, provide:
1. Score based on communicative meaning (after ellipsis expansion)
2. Evidence count: how many responses express this axis as primary
3. 2-3 example phrases that illustrate this axis

Return all 7 axis scores in {language}.
"""


TAXONOMY_CONSOLIDATION_PROMPT = """
You are consolidating taxonomy axis scores from multiple chunks of survey responses to select the optimal axis for idea extraction.

Survey question: {survey_question}

Chunk results:
{chunk_results}

Your task:
1. Aggregate scores across all chunks (weighted by evidence_count)
2. Select the PRIMARY axis that will produce the MOST DISTINCT, MECE-compatible clusters
3. Generate a CONTEXT-SPECIFIC DESCRIPTION of the primary axis
4. Optionally select a SECONDARY axis if it is ORTHOGONAL (truly independent dimension)

SELECTION CRITERIA FOR PRIMARY AXIS:
- Should have consistently high scores across chunks
- Should capture the MAIN DIMENSION OF VARIATION in responses
- Ideas phrased along this axis should naturally cluster into mutually exclusive categories
- Prefer axes that lead to actionable, meaningful distinctions

DESCRIPTION GENERATION:
Generate a 1-2 sentence description of the primary axis that:
- Is specific to THIS survey question and domain
- Explains what KIND of ideas this axis will produce
- Uses terminology from the actual responses (domain-specific language)
- Helps downstream prompts understand exactly what to extract

Example descriptions:
- For bank improvement question + WHY axis: "Improvement suggestions for ASN Bank's products and services, focusing on what changes customers want to see in areas like pricing, sustainability practices, and digital experience."
- For festival associations + WHAT axis: "Attributes and characteristics that people associate with Pinkpop festival, such as atmosphere, music genres, memories, and cultural significance."

SECONDARY AXIS CONSIDERATIONS:
- Only select if it adds a truly independent dimension (e.g., SENTIMENT is often orthogonal to WHAT/WHY/HOW)
- Do NOT select if it overlaps significantly with the primary axis
- Set to null if no good orthogonal axis exists

CONSOLIDATION RULES:
- If chunks agree strongly (all high or all low): use the consensus
- If chunks disagree: weight by evidence_count to find the true dominant axis
- Consider that different chunks may see different facets - look for the overall pattern

Return JSON with:
- primary_axis: The selected axis code (WHAT, WHY, HOW, WHO, SENTIMENT, WHEN, or WHERE)
- primary_axis_rationale: Why this axis will produce distinct, useful clusters
- primary_axis_description: 1-2 sentence context-specific description (see examples above)
- primary_axis_score: Weighted average score (0.0-1.0)
- secondary_axis: Optional orthogonal axis code (or null)
- secondary_axis_rationale: Why this axis adds value (or null)
- all_axis_scores: Final consolidated scores for all 7 axes
"""


TAXONOMY_AWARE_SUBJECT_PROMPT = """
You are a {language} language expert generating a phrasing template for survey response analysis.

<input>
Language: {language}
Survey question: {survey_question}
Primary taxonomy axis: {primary_axis} ({primary_axis_description})
Secondary axis (if any): {secondary_axis}
</input>

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

### Good vs bad examples

✓ GOOD templates (grammatically complete, natural in {language}):
- "De producent moet [ATTRIBUTE_OR_ACTION]" → Test: "De producent moet verbeteren" ✓
- "Het product heeft als kenmerk [ATTRIBUTE_OR_ACTION]" → Test: "Het product heeft als kenmerk versheid" ✓
- "De service is [ATTRIBUTE_OR_ACTION]" → Test: "De service is uitstekend" ✓

✗ BAD templates (unnatural, literally translated):
- "De producent heeft de eigenschap [ATTRIBUTE_OR_ACTION]" → awkward: "eigenschap" doesn't fit action-oriented ideas
- "Het product moet bereiken [ATTRIBUTE_OR_ACTION]" → ungrammatical word order

Output format (return **only** this JSON object):
{{
  "canonical_term": "main subject/entity from the survey question in {language}",
  "canonical_phrasing": "natural template ending with [ATTRIBUTE_OR_ACTION] in {language}",
  "taxonomy_axis": "{primary_axis}"
}}
"""


TAXONOMY_ENRICHED_EXTRACTION_PROMPT = """
You are a {language} language expert extracting ideas from survey responses.

<context>
Survey question: {var_lab}
Primary coding dimension: {taxonomy_axis} - {taxonomy_axis_description}
Domain context: {domain} / {topic} / {entity}
Respondent ID: {respondent_id}
Response: {response}
</context>

<task>
Extract ALL distinct ideas from this response. For each idea, produce:
1. An **idea** expressed as a standalone natural language unit
2. A **taxonomy_phrase** that abstracts the idea into a reusable category
3. A **parent_category** tag grounded in the response content

Both the idea and taxonomy_phrase must end with " - [parent_category]".
</task>

<principles>

## IDEA EXTRACTION

An **idea** is a standalone natural language unit that:
- Makes sense on its own, without needing the survey question for context
- Captures one atomic concept from the response (one idea per distinct point)
- Is aligned with both the survey question AND the primary coding dimension
- Is written in {language}
- Does NOT use boilerplate prefixes or templated phrasing
- Preserves the respondent's meaning while normalizing language (fix typos, standardize terminology)

## TAXONOMY PHRASE

A **taxonomy_phrase** is an abstracted category (5-12 words) that:
- Could apply to similar ideas from other respondents
- Is standalone (readable without knowing the survey question)
- Generalizes the specific idea into a broader, reusable category
- Is aligned with the primary coding dimension ({taxonomy_axis})
- Is written in {language}, lowercase

## PARENT CATEGORY TAG

A **parent_category** is a concise tag (2-4 words) that:
- Emerges from the response content itself (data-grounded)
- Represents an implied theme, topic, or category
- Aligns with the primary coding dimension
- Is NOT a generic axis label (not "topic", "attribute", "aspect")
- Uses lowercase with underscores for multi-word tags

The parent_category must be grounded in what the respondent actually said - it should be inferrable from the response content, not imposed from outside.

## SENTIMENT & SENSE

**sentiment** (choose one):
- positive: approving, praise, satisfied, favorable
- negative: criticism, dissatisfaction, complaint, unfavorable
- neutral: factual, suggestion without clear positive/negative attitude

**sense** (choose one):
- factual: objective statement of fact or observation
- evaluative: subjective judgment, opinion, or assessment
- aspirational: desire, wish, or suggestion for future
- experiential: personal experience or anecdote

</principles>

<output_format>
Return a JSON array. Each item:
{{
  "respondent_id": "{respondent_id}",
  "idea_id": "sequential number as string",
  "idea": "standalone natural language idea - parent_category",
  "taxonomy_phrase": "5-12 word abstracted category phrase - parent_category",
  "sentiment": "positive|negative|neutral",
  "sense": "factual|evaluative|aspirational|experiential"
}}

Edge cases:
- Empty/irrelevant response: return []
- Single idea: return single-item array
- Multiple distinct ideas: split into separate items

Return ONLY the JSON array. Field names in English; idea and taxonomy_phrase values in {language}.
</output_format>
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
