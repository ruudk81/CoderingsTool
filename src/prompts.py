
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

EXTRACT_SUBJECT = """
You are a {language} language expert in interpreting survey questions.
Your task is to extract the canonical focus entity — the single main product, brand, service, actor, or topic the question is about — and produce a reusable phrasing template.

<input>
Language: {language}
Survey question: {survey_question}
</input>

Definitions
- Canonical focus entity: the main noun phrase that answers "what or who is this question about?"
  It can be:
  - a product/brand/service being evaluated
  - a person/group/institution expected to act
  - a topic/concept/event being discussed

Constraints
- Choose exactly one canonical focus.
- Return a **concise, normalized noun phrase** in {language}.
- Do **not** include determiners (e.g., "the", "de", "het"), quotes, brackets, or trailing punctuation.
- Preserve capitalization for proper nouns/brands; otherwise use lowercase.
- If the question uses pronouns/deictics (e.g., "this brand", "our app"), resolve to the most specific **named** entity present; if none, use the most specific generic noun ("app", "klantenservice", "bezorgproces").
- If multiple entities appear, pick the one **most central** to what is being evaluated, judged, or requested.
- If the question refers to a sub-part/feature (e.g., "payment process" of a store), pick the **sub-part** if that is clearly the evaluation target.

Template requirement
1. Create a phrasing template that downstream steps can use to express evaluations or actions.
2. Use this structure:
   **"[CANONICAL_TERM] [VERB/STATE] [SCAFFOLDING_WORDS] [ATTRIBUTE_OR_ACTION]"**

   Where:
   - CANONICAL_TERM: the focus entity (e.g., "electric vehicles", "klantenservice")
   - VERB/STATE: appropriate verb in {language} (e.g., "is", "has", "should", "needs", "zijn", "heeft", "moet")
   - SCAFFOLDING_WORDS: grammatical words needed for completeness (may be empty if verb alone is sufficient)
   - ATTRIBUTE_OR_ACTION: placeholder for the actual content

3. Insert:
   - The canonical term (noun phrase)
   - The most natural verb/state in {language}
   - Any necessary scaffolding words (articles, prepositions, auxiliary verbs) to ensure grammatical completeness
   - Leave `[ATTRIBUTE_OR_ACTION]` as a placeholder

4. The result should sound natural and complete up to the placeholder.

Grammatical completeness constraint
- The template MUST produce a grammatically complete sentence when [ATTRIBUTE_OR_ACTION] is replaced with a simple adjective or noun.
- Test your template by filling [ATTRIBUTE_OR_ACTION] with a one-word example (e.g., "expensive", "better", "quality").
- If the result is grammatically incomplete, add necessary scaffolding words BEFORE the placeholder.
- Common scaffolding patterns:
  * For "has/heeft": → "has the [quality/feature/association/characteristic] [ATTRIBUTE_OR_ACTION]"
  * For "needs/moet": → "needs to [action verb] [ATTRIBUTE_OR_ACTION]"
  * For "should/zou moeten": → "should [action verb] [ATTRIBUTE_OR_ACTION]"
  * For "is/zijn": → "is [ATTRIBUTE_OR_ACTION]" (often already complete)

Examples of template construction

✓ GOOD templates (grammatically complete):
- Survey: "Which associations do you have with Brand X?"
  → "Brand X has the association [ATTRIBUTE_OR_ACTION]"
  → Test: "Brand X has the association expensive" ✓

- Survey: "What could the manufacturer improve?"
  → "The manufacturer should improve [ATTRIBUTE_OR_ACTION]"
  → Test: "The manufacturer should improve quality" ✓

- Survey: "How do you rate the service?"
  → "The service is [ATTRIBUTE_OR_ACTION]"
  → Test: "The service is excellent" ✓

✗ BAD templates (grammatically incomplete):
- "Brand X has [ATTRIBUTE_OR_ACTION]"
  → Test: "Brand X has expensive" ✗ (incomplete - missing noun after "has")

- "The manufacturer should [ATTRIBUTE_OR_ACTION]"
  → Test: "The manufacturer should quality" ✗ (missing verb)

- "The service needs [ATTRIBUTE_OR_ACTION]"
  → Test: "The service needs better" ✗ (incomplete - "needs" requires "to be" or noun)

Output format (return **only** this JSON object)
{{
  "canonical_term": "canonical noun entity in {language}",
  "canonical_phrasing": "template with canonical term and verb/state inserted in {language}, e.g. 'Elektrische voertuigen zijn [EIGENSCHAP_OF_ACTIE]'"
}}

Validation checklist before returning:
- Exactly two keys: `canonical_term`, `canonical_phrasing`
- Values are in {language}
- `canonical_term` is a noun phrase with no article or punctuation
- `canonical_phrasing` includes the correct verb/state (no placeholders for it) and ends with `[ATTRIBUTE_OR_ACTION]`
- Test grammatical completeness: replace [ATTRIBUTE_OR_ACTION] with a simple word - result must be a complete sentence
- If incomplete, add necessary scaffolding words (articles, prepositions, auxiliary verbs) before the placeholder

"""

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

IDEA_EXTRACTION_PROMPT = """
You are a {language} language expert in analyzing written responses to open-ended questions in surveys. 
Your task is to extract ALL distinct ideas expressed in a respondent's written answer.  

<inputs>
Survey question: {var_lab}
Canonical subject focus: {subject}
Respondent ID: {respondent_id}
Written response: {response}
</inputs>

<instructions>
1. Understand the context by reading the survey question and response carefully. 
        
2. Idea Identification
    - Extract all distinct ideas that directly answer or relate to the survey question. 
    - An “idea” is:
        - A single, complete thought or opinion
        - A specific action, behavior, or experience mentioned
        - A reason, cause, or explanation given
        - An emotion, attitude, or evaluation expressed

3. Atomicity
    - Keep each idea atomic — only one concept per idea.
    - Avoid merging ideas even if they are related.
    - Split compound statements connected by “and”, “but”, or similar connectors into separate ideas.
    - The idea you will return must not contain coordinating conjunctions or list markers in {language}.
    - Forbid list/coordination punctuation: "/", "&", ",", ";", ":", "-", "–" (hyphens allowed only inside a single lexicalized word, not to join ideas).

4. MANDATORY Phrasing template
    - **CRITICAL REQUIREMENT**: EVERY idea you extract MUST follow this EXACT structure:
      {phrasing_template}

    - This is NOT optional. The template structure is MANDATORY for EVERY single idea.

    - Replace [ATTRIBUTE_OR_ACTION] with the actual content, but keep everything before it EXACTLY as shown.

    - WRONG examples (DO NOT do this):
      ✗ Starting with different words than the template
      ✗ Omitting the template prefix entirely
      ✗ Reordering or rephrasing the template structure

    - CORRECT approach:
      ✓ Take the exact prefix from the template (everything before [ATTRIBUTE_OR_ACTION])
      ✓ Add your specific content after that prefix
      ✓ Verify the final idea starts with the template prefix character-for-character

    - Validation: Before outputting each idea, verify it starts with the exact prefix shown in the template.

    - Do not change sentiment or tone during normalization.

5. Include Implicit Ideas
    - Capture both explicit statements and ideas that are clearly implied by the response.

6. Edge Cases
    - If the response is empty, irrelevant, or "N/A": return an empty array [].
    - If the response is off-topic: extract ideas anyway but note they may be off-topic.
    - If only one idea is present: return it in a single-item array.

7. Sentiment and Sense Extraction

   For EACH idea you extract, you must also classify:

   **sentiment**: The emotional/evaluative tone of the idea
   - **positive**: Favorable, satisfied, praising, appreciative
   - **negative**: Critical, dissatisfied, complaining, unfavorable
   - **neutral**: Factual, neither positive nor negative, descriptive
   - **mixed**: Contains both positive and negative elements

   Examples:
   - "excellent service" → positive
   - "poor quality" → negative
   - "located in Amsterdam" → neutral
   - "good product but expensive" → mixed

   **sense**: The modality or nature of the statement
   - **factual**: Objective statement of fact or observation
   - **evaluative**: Subjective judgment, opinion, or assessment
   - **aspirational**: Desire, wish, or suggestion for future
   - **experiential**: Personal experience or anecdote

   Examples:
   - "Merk X is sustainable" → evaluative (judgment)
   - "I received a loan from Merk X" → experiential (personal experience)
   - "Merk X should improve service" → aspirational (suggestion)
   - "Merk X has 500,000 customers" → factual (objective fact)
</instructions>

<output_format>
Return the extracted ideas as a JSON array. Each item should include:
- "respondent_id": exactly as provided
- "idea_id": a string number (numbering always starts at "1" and increments sequentially -e.g. "1", "2", etc.).
- "idea": the descriptive phrase in {language}, normalized and phrased using the provided phrasing template.
- "sentiment": one of ["positive", "negative", "neutral", "mixed"]
- "sense": one of ["factual", "evaluative", "aspirational", "experiential"]

**TEMPLATE COMPLIANCE CHECK:**
Before returning, verify that EVERY "idea" field starts with the exact prefix from the phrasing template.
If an idea doesn't match the template, reformulate it to match.

Always output in {language}.
</output_format>

Here's an example of the desired output example based on the input example:

<input_example>
Survey question: "What could the manufacturer of electric vehicles do better in your opinion?"
Respondent ID: 987654321
Response: "They should make the cars charge faster and improve battery life."
</input_example>

<output_example>
[
  {{
    "respondent_id": "987654321",
    "idea_id": "1", 
    "idea": "Electric vehicles should charge faster"
  }},
  {{
    "respondent_id": "987654321",
    "idea_id": "2",
    "idea": "Electric vehicles should have improved battery life"
  }}
]
</output_example>

Notice how:
- The original term “cars” was replaced with the canonical term “electric vehicles” from the survey question.
- All ideas use SUBJECT phrasing consistently.
- Each idea is separate and atomic.
- The meaning and sentiment of the original statement are preserved.
- Terms are consistent across all ideas, even if the respondent used different or less specific words.    
    
Return ONLY the JSON array. Keep field names in English; write values in {language}.
"""


# =============================================================================
# Speculative codes
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
# STEP 6:  CODEBOOK GENERATION : 4 promt chain
# =============================================================================

CLUSTER_SUMMARY_PROMPT = """
You are a qualitative researcher applying Braun & Clarke’s (2006) thematic analysis method.

Your task is to:
a) analyze a cluster of descriptive codes and construct one or more ATOMIC themes (central organizing concepts) 
b) draft first-version codebook entries for each theme, including inclusion/exclusion examples and abstraction level. 

Please realize that the drafts are the initial parameters that future updates will refine.

<inputs>
Language to use: {language}

Cluster ID: {cluster_id}

Research question:
"{survey_question}"

Cluster of descriptive codes to analyze:
{cluster_text}
</inputs>

<definitions>
- ATOMIC THEME = one single idea, action, expectation, or motive relevant to the research question (no mixing).
- ABSTRACTION LEVEL = the conceptual “height” of the theme. Use one of:
  • "Driver/Motive/Why" (highest) 
  • "Attribute/What" (mid)
  • "Action/How" (concrete)
- NEAR-NEIGHBOR = an adjacent concept likely to be confused with the theme.
</definitions>

<guidance>
- Atomic means:
  • Single idea only (no “and/or” combinations).
  • One aspect/domain/category/attribute only.
  • One consistent sentiment/intention (no polarity mixing).
  • Concrete and directly actionable.
- If a code mentions multiple aspects or mixed sentiment → split into separate atomic concepts.

Label constraints (strict):
- ≤ 10 words 
- Active/actionable formulation of ONE ATOMIC theme in relation to the research question
- Atomic means "One domain or aspect only"
- If verb is used → one main verb (present tense).
- **Never** include reasons (no "to", "so that", "because").
- Avoid punctuation: "/", "&", ",", "–", ":" (unless lexicalized).
- Maintain **one polarity** (either increase/strengthen OR reduce/avoid).

Definition constraints (strict):
- ≤ 30 words
- grounded in theme description
- Must describe **what belongs in this code**, not why it happens.
- Must align directly with the survey question.
- Use a **clear, observable assignment cue** (e.g., behaviors, expressions, judgments).
- Do not explain causes, conditions, or interpretations.
</guidance>

Follow these steps exactly and in order. Do not skip or reorder any step. Use your analytical judgment and reflexivity throughout—remember that themes are not discovered in data but actively constructed.

<analysis_steps>
1. Interpret descriptive codes in light of the research question:
    - How does each code address the research question?
    - What patterns are meaningful for analyzing concrete, actionable answers?

2. Remove outliers. 
    - Eliminate codes that do not connect to any broader pattern across multiple codes.
    - Eliminate codes that do not represent a meaningful segment (too rare, irrelevant, or idiosyncratic).

3. Identify COC(s):
    - Practical test: Can all codes be summarized in one sentence that:
     (a) aligns with <guidance>,
     (b) captures exactly ONE ATOMIC theme,
     (c) uses no coordinating conjunctions or list punctuation,
     (d) preserves unity, consistency, and contrast?
   - If yes → single COC. If no → multiple COCs.
   - Proceed with the resulting COC(s).
      
4. Refine COCs:
    - Remove singletons (COCs based on only one code).
    - Remove COCs without a dominant shared pattern or conceptual overlap.
    - Remove vague or overly broad COCs (e.g., “positivity,” “challenges,” “general satisfaction”).

5. For each retained COC, produce a codebook-ready draft with:
   - theme_label: "[≤ 10 words | active/actionable formulation of ONE ATOMIC theme in relation to the research question]"
   - theme_clarification: "[≤ 30 words | illustrative descriptive codes from <inputs> that clarify and support the label — tight, grounded, evidence-based]"
   - abstraction_level: Select one of: "Driver/Motive/Why" | "Attribute/What" | "Action/How"
   - assignment_examples (EXAMPLES, not rules):
       • inclusion: Provide 2–3 short, positive EXAMPLE assignment expressions (each starts with a verb; use observable cues).
       • exclusion: Provide 1–2 short EXAMPLE boundary examples to prevent overreach (what must NOT be included).
       • near_neighbor:
         — label: closest potentially-confusable theme or "Unknown"
         — tell_apart_rule: one sentence explaining how to distinguish the two (e.g., “This theme focuses on X (driver/what/how), whereas the neighbor focuses on Y.”)
  
6) Document the analysis:
   - State how many COCs were identified and retained.
   - If only one COC: explain why it is sufficient.
   - If multiple COCs: justify why a single COC would violate atomicity or clarity.
</analysis_steps>

Output strictly as valid JSON using this exact structure (values in {language}, field names in English):
{{
  "{cluster_id}": {{
    "analysis": "Provide your analysis here in {language}.",
    "extracted_themes": [
      {{
        "theme_id": 1,
        "theme_label": "[≤ 10 words | active/actionable formulation of ONE ATOMIC theme in relation to the research question]",
        "theme_clarification": "[≤ 30 words | illustrative descriptive codes from <inputs> that clarify and support the label — tight, grounded, evidence-based]",
        "abstraction_level": "Driver/Motive/Why | Attribute/What | Action/How",
        "assignment_examples": {{
          "inclusion": [
            "[examples inclusion in {language}]",
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
      // Add additional theme objects here if more than one COC was found
    ]
  }}
}}


Critical requirements:
- Output must be valid JSON only — no extra commentary or explanation before or after.
- Keep field names in English; write values in {language}.
- Replace "cluster_id"  with the actual values provided.
- Conduct your analysis in the specified language.
"""

CODING_DECISION_PROMPT = """
You are a {language} qualitative research assistant responsible for maintaining a structured codebook.

Your task is to classify a new theme by deciding whether it should:
- be assigned to an existing code (**USE**),
- extend an existing code (**MODIFY – Vertical / same motive**),
- consolidate into a parent code that groups multiple codes (**MODIFY – Hierarchical / different motive but same conceptual family**),
- or be added as a new code (**CREATE**).

You must base your reasoning on:
1) Cosine similarity (semantic proximity),
2) The underlying motive/intent (the "why" behind the theme),
3) The conceptual structure of the codebook (to maintain clear boundaries).

<inputs>
Survey Question:
"{survey_question}"

New Theme to Classify:
- name: "{theme_name}"
- description: "{theme_description}"

Further information about the theme:
- abstraction level: "{abstraction_level}"
- what's included: 
    {inclusion}
- what's excluded: 
    {exclusion}
- boundary: {near_neighbor}

Existing Codes:
{code_text}
</inputs>

Follow these steps exactly and in order. Do not skip or reorder any step. 

<analysis_steps>
1. Cosine Similarity Rules (EXTREMELY STRICT - no deviation allowed):

- **≥ 0.90** → Same meaning → **USE**
- **< 0.70** → Different meaning → **CREATE**
- **0.70–0.90** → Perform Motive Test

2. MOTIVE TEST:
Ask: *Does the new theme share the same underlying motive/driver as the highest-similarity code?*

- Same motive → **MODIFY (Vertical Expansion)**  
  • Add new expressions to **inclusion rules**  
  • Keep same abstraction level  
  • Maintain atomicity

- Different motive but same conceptual family → **MODIFY (Hierarchical Expansion)**  
  • Create or reference a **parent** theme at a higher abstraction level  
  • Existing code and new theme become **sub-themes**

- Different motive and not in same family → **CREATE** new code

3. Structural Constraints (Always Enforce)
- Codes must remain **atomic**: one idea, one motive, one sentiment.
- Inclusion rules describe when to assign the theme.
- Exclusion rules describe common misfits to keep boundaries clear.

4. Update Assignment Examples
- If decision is **USE** → preserve original assignment_examples unchanged
- If decision is **MODIFY**:
  • inclusion: combine original + new expressions from the theme
  • exclusion: combine original + new boundary clarifications if needed
  • near_neighbor: update label if boundaries shifted due to modification
  • tell_apart_rule: update if the distinction from neighbor changed
- If decision is **CREATE** → use assignment_examples from the new theme as-is
</analysis_steps>

Respond with **valid JSON only** in the following structure:

{{
  "coding_decision": {{
    "theme_number": {theme_id},
    "theme_name": "Exact name of the theme from <theme_to_code>",
    "matched_candidates": [
        {{"code": "Exact candidate code A", "definition": "Definition in light of the survey question"}},
        // Add additional candidaties,if there are any 
        ]
    "decision": "USE | MODIFY | CREATE",
    "source_code": "Exact candidate code name if use/modify, or null if create",
    "modify_parameters":{{
       "modify_instruction": "vertical_broaden_same_motive | hierarchical_parent_diff_motive_same_family | none",
       "motive_comparison": "same | different_same_family | different_not_related",
       "abstraction_level_action": "keep | broaden_to_parent | none",
       "inclusion_update": "null or concrete additions to inclusion rules",
       "exclusion_update": "null or concrete boundary clarifications",
       "parent_theme_label": "null or suggested parent label",
       "near_neighbor_label_update": "null or updated neighbor label if boundaries changed",
       "tell_apart_rule_update": "null or updated tell-apart rule if distinction changed"}},
    "justification": "Explain modification decision by referencing cosine score and whether motive was same or different, or null if use/create",
    "updated_assignment_examples": {{
      "inclusion": ["[updated or original inclusion examples in {language}]"],
      "exclusion": ["[updated or original exclusion examples in {language}]"],
      "near_neighbor": {{
        "label": "[updated or original neighbor label in {language}]",
        "tell_apart_rule": "[updated or original tell-apart rule in {language}]"
      }}
    }}.
  }}
}}

Examples (for Interpretation)

1. **Vertical MODIFY Example**
cosine = 0.81, same motive
→ "This theme expresses the same underlying reason as Code A but in a new form. MODIFY (vertical)."

2. **Hierarchical MODIFY Example**
cosine = 0.80, different motive but same conceptual family
→ "This theme shares the domain of Code A but with a different driving motive. MODIFY (hierarchical)."

3. **CREATE Example**
cosine = 0.64
→ "The meaning and motive differ significantly. CREATE new code."

REQUIREMENTS
- Output **must be valid JSON only** (no commentary).
- Keep field names in English; write values in {language}.
- Include **cosine score** in justification.
- Identify motive comparison explicitly.

"""

CODE_CREATION_PROMPT = """
You are a {language} qualitative research assistant. 
Your task is to CREATE a new **atomic** code that captures the meaning of a newly identified theme from survey responses.

<inputs>
Survey question:
"{survey_question}"

New theme:
- name: "{theme_name}"
- description: "{theme_description}"
</inputs>

DEFINITION OF AN ATOMIC CODE:
- Expresses exactly **one** idea, action, attitude, or expectation.
- Cannot be split into two meaningful codes without losing clarity.
- Contains **no conjunctions** (and/or), **no lists**, and **no dual motives**.

LABEL RULES (strict):
- ≤ 10 words 
- Active/actionable formulation of ONE ATOMIC theme in relation to the research question
- Atomic means "One domain or aspect only"
- If verb is used → one main verb (present tense).
- **Never** include reasons (no "to", "so that", "because").
- Avoid punctuation: "/", "&", ",", "–", ":" (unless lexicalized).
- Maintain **one polarity** (either increase/strengthen OR reduce/avoid).

DEFINITION RULES:
- ≤ 30 words
- grounded in theme description
- Must describe **what belongs in this code**, not why it happens.
- Must align directly with the survey question.
- Use a **clear, observable assignment cue** (e.g., behaviors, expressions, judgments).
- Do not explain causes, conditions, or interpretations.

GOOD DEFINITION PATTERNS:
- "References to…"
- "Mentions of…"
- "Expressions of…"
- "Concerns about…"

AVOID:
- Broad summaries (e.g., “general dissatisfaction”).
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

#Placeholders CODING_MODIFICATION_PROMPT 
VERTICAL_INSTRUCTIONS = """
   - Keep the abstraction level of the original code.
   - Create a **single atomic shared concept** that:
        (a) captures the meaning of both original code and new theme,
        (b) is grounded in the shared intent (same motive),
        (c) remains expressible as **one idea** in the label.
   - The modified label must:
        • reflect the broadened meaning,
        • NOT introduce multiple aspects or motives,
        • NOT be more abstract than necessary.
   - The modified definition must:
        • describe the **shared meaning space**,
        • reflect: original inclusions + inclusion_update,
        • exclude: original exclusions + exclusion_update.
   - Do **not** modify assignment rules here."""

HIERARCHICAL_INSTRUCTIONS = """
   - Shared conceptual domain but different motives → create hierarchical structure.
   - Original code and new theme remain **atomic child codes**.
   - Parent code represents the shared **purpose/motive domain**.

   Parent label:
        - parent theme = {parent_theme_label}  
        - If parent theme is not None or Null → use it as-is.
        - If null → generate a label at **Driver/Motive/Why** level.
        - Must:
            • express shared purpose/orientation,
            • NOT describe behaviors/outcomes,
            • NOT blend child labels,
            • be broader, not vaguer.

   Structure:
       - Parent = conceptual anchor (why level),
       - Children = distinct manifestations (how/what),
       - Child meanings **do not change**."""

CODING_MODIFICATION_PROMPT = """
You are a {language} qualitative research assistant updating a codebook.
Your task is to MODIFY an existing code so that it fully and correctly includes a new theme, while preserving **atomic meaning** and **clear conceptual boundaries**.

<inputs>
Survey question:
"{survey_question}"

New theme to integrate:
- name: "{theme_name}"
- description: "{theme_description}"
- Included expressions (these SHOULD be covered by the code):
  {inclusion}
- Excluded expressions (these should NOT be covered by the code):
  {exclusion}

Original code (to be modified):
- code_label: {source_code}
- code_definition: {source_definition}

Current assignment examples (to be updated):
- Current inclusion examples:
  {current_inclusion}
- Current exclusion examples:
  {current_exclusion}
- Current boundary (near neighbor):
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

LABEL INSTRUCTIONS:
    - ≤ 10 words 
    - Active/actionable formulation of ONE ATOMIC theme in relation to the research question
    - Atomic means "One domain or aspect only"
    - If verb is used → one main verb (present tense).
    - **Never** include reasons (no "to", "so that", "because").
    - Avoid punctuation: "/", "&", ",", "–", ":" (unless lexicalized).
    - Maintain **one polarity** (either increase/strengthen OR reduce/avoid).

DEFINITION INSTRUCTIONS:
    - ≤ 30 words
    - grounded in theme description
    - Must describe **what belongs in this code**, not why it happens.
    - Must align directly with the survey question.
    - Use a **clear, observable assignment cue** (e.g., behaviors, expressions, judgments).
    - Do not explain causes, conditions, or interpretations.
    - GOOD DEFINITION PATTERNS:
        • "References to…"
        • "Mentions of…"
        • "Expressions of…"
        • "Concerns about…"

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
- If hierarchical_parent_diff_motive_same_family → ensure parent label is conceptual, not descriptive or repetitive.
"""


VALIDATION_PROMPT = """
You are a {language} curator of codebooks for thematic analysis following Braun & Clarke (2006) methodology. 
Your role is to maintain parsimonious codebooks with non-overlapping and non-redundant codes by reviewing and making final decisions on coding proposals.

<inputs>
Language to use: {language}

Existing codes in codebook:
{code_text}
    
Proposal background:
    
A new theme emerged from analyzing responses to this survey question: 
"{survey_question}" 

This is the new theme:
- name: "{theme_name}"
- description: "{theme_description}"

Assignment examples to validate:
- inclusion: {inclusion_examples}
- exclusion: {exclusion_examples}
- near_neighbor: {near_neighbor_label} (Tell apart: {tell_apart_rule})

The proposal to review:
{step3_recommendation}
</inputs>

Your task is to systematically evaluate this proposal and make a final coding decision. Use the scratchpad below to work through your evaluation before providing your final JSON response.

<scratchpad>
Work through your evaluation systematically:

Step 1: Evaluate the CREATE/MODIFY decision against these criteria:
a. Parsimony: Has sufficient effort been made to use existing codes or combinations before proposing new/modified codes?
b. Abstraction Level: Is any proposed new code at an appropriate abstraction level, consistent with existing codes?
c. Non-Redundancy: Does the proposal avoid creating codes that significantly overlap with existing ones?

Step 2: Evaluate code label and definition:
d. Atomicity:
   • Does it express only one idea (no merged/compound themes)?
   • Does it avoid forbidden punctuation: "/", "&", "+", ",", ";", ":", "-", "–" (unless the punctuation is lexicalized within a compound noun (e.g., ‘gebruiksklaar-product’ is allowed).")?
   • Does it contain at most ONE main action (verb)?
   • Does it avoid conjunctions ("and/or") unless lexicalized?

e. Form & Length:
   • Is the label ≤10 words with no canonical subject from survey question and no implied actor?
   • Does it follow allowed forms:
     – Noun phrase: <adjective(s)> <noun>
     – Imperative verb + object: <verb> <object>
     – Infinitive: <to/infinitive verb> <object>
   • Is the definition ≤25 words, operational/observable, grounded in responses, and non-vague?
   
Step 3: APPROVE/REJECT proposal
- If all criteria PASS → APPROVE (you may make minor refinements for full compliance)
- If any criterion FAILS → REJECT (identify issues and rewrite to comply)

Step 4: If rejected on the grounds of parsimony, non-redunancy or overlap, make a final decision about:  
- **USE**: An existing code already captures the new theme's central meaning sufficiently
- **MODIFY**: An existing code is close but needs refinement for clarity, scope, or better alignment  
- **CREATE**: No existing code sufficiently captures the new theme

Step 5: Validate Assignment Examples:
- Ensure inclusion examples align with the refined code/definition
- Ensure exclusion examples maintain clear boundaries
- Verify near_neighbor and tell_apart_rule are still accurate
- Refine if needed to match validated code

Step 6: Determine your final components:
- validated_decision: USE, MODIFY, or CREATE code
- source_code:
    - If USE, this exact code: {source_code}
    - If MODIFY, the exact code from the existing codebook you seek to modify
    - If CREATE, write "null"
- validated_code and validated_decision: final compliant label, definition, and assignment_examples
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

# =============================================================================
# STEP 7 THEME ORGANIZATION WITH REASONING MODELS
# =============================================================================

CODEBOOK_REFINEMENT_PROMPT = """
You are a qualitative research methodologist and codebook architect. 
Your task is to transform a raw list of descriptive codes into a MECE (Mutually Exclusive, Collectively Exhaustive) codebook.

Here is the survey question you are working with:
<survey_question>
{survey_question}
</survey_question>

Here are the raw descriptive codes to organize:
<raw_codes>
{raw_codes}
</raw_codes>

You must conduct your analysis and provide all output in this language:
<language>
{language}
</language>

Now, internalize these strict guidelines:
<guidelines>
1. MECE Principle and Code Preservation

Your goal is to organize codes into a MECE structure while preserving all distinct conceptual content.

- Mutually Exclusive: Sibling codes must represent clearly different ideas. No code should logically belong under multiple siblings.
- Collectively Exhaustive: Every original code must be placed somewhere (no omissions).

Preservation Rules:
- DO NOT remove any code or idea.
- DO NOT collapse distinct concepts into broader labels.
- You MAY merge codes only when they express the exact same concept in different wording.
- If two codes differ in evaluation, focus, actor, intensity, or practical use → they are distinct and must remain separate.

**Key Decision Test for Merging:**
If a trained human coder could reliably treat these two codes as interchangeable without losing meaning → merge.
Otherwise → keep separate.

**Key Decision Test for Separating:**
If a researcher would want to report these codes separately → keep separate.

2. Hierarchy Structure

Use the smallest structure that preserves clarity:
- Prefer 2-level hierarchy: Theme → Code.
- Use 3-level hierarchy: Theme → Concept → Code **only** if:
  - At least 2 codes share a highly specific sub-idea,
  - AND grouping them improves coder accuracy.

Each code has exactly one parent.

3. Theme Naming

To prevent abstraction drift:
- Themes must be named using the *interpretive lens of the survey question*, not generic language.
- Themes must describe *what is being talked about*, not just the domain label.
- Avoid vague theme names like "Quality", "Experience", "General Feedback".
- Prefer concrete semantic anchors, e.g., "Smaak & Textuur", "Portiegrootte & Verzadiging", "Prijsperceptie".

</guidelines>

Follow these codebook refinement steps exactly and in order. Do not skip or reorder any instruction. 

1. **Deduplicate (MERGE-ONLY):** Identify semantic duplicates and justify each merge with a one-sentence "identity test" rationale
2. **Propose Themes:** Create themes that reflect the survey question lens (not generic buckets)
3. **Place Codes:** Organize codes under themes, adding Concept level if it improves the structure
4. **MECE Audit:**
   - **Sibling overlap check:** For each sibling set, identify potential confusion pairs and explain how your decision rules separate them
   - **Coverage check:** List any raw codes that remain out-of-scope and explain why
   - **Count summary:** Provide X original → Y after merge (Z merged, Y−Z preserved)


You must output your response as valid JSON only, with no commentary before or after. Use this exact structure:
{{
  "analysis": "Provide detailed analysis in {language}: (1) Which codes were merged and why (include IDs), (2) Which similar codes were kept separate and why, (3) How hierarchy was structured, (4) Total codes preserved vs. merged count.",
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

Critical Requirements:
- **Output must be valid JSON only** — no commentary before or after
- **Conduct analysis in the specified language**
- **Naming:** Avoid abstract nouns alone; prefer "object + facet" or "action + object" patterns
- **Granularity:** If two codes frequently co-occur in one segment, consider one as a Concept with both as child Codes rather than merging
- **Polarity:** Opposite valences are separate codes unless your analysis never reports polarity
- **Category field:** Use empty string "" for 2-level hierarchy (preferred), or category name for 3-level hierarchy

Begin your analysis and provide the JSON output.
"""


# CODEBOOK_REFINEMENT_PROMPT = """
# You are a qualitative researcher and codebook methodologist.
# Your task is to take a raw list of descriptive codes and transform it into a refined and structured codebook.
# The descriptive codes are derived from survey responses.

# <inputs>
# Language to use: {language}

# survey_question: {survey_question}

# Raw descriptive codes to refine:
# {raw_codes}
# </inputs>

# <critical_requirement>
# Preserve the *conceptual content* of ALL codes.
# - Do NOT remove or lose any unique ideas.
# - Do NOT collapse distinct concepts into a single code.
# - You MAY merge true duplicates (semantically identical codes).
# - You MUST preserve ALL meaningful distinctions between different ideas.

# IMPORTANT DISTINCTION:
# • MERGING = Combining semantic duplicates (reduces redundancy) ✓
# • COLLAPSING = Combining distinct concepts (loses information) ✗

# Example of MERGING (correct):
#   "Price transparency" + "Clear pricing information" → "Price transparency" (same concept, different words)

# Example of COLLAPSING (incorrect):
#   "Salt concerns" + "Bitterness" + "Sweetness preferences" → "Taste preferences" (3 DIFFERENT concepts lost!)

# NEVER collapse codes just because they belong to the same category or theme.
# </critical_requirement>

# <do_not_collapse_rules>
# Keep codes SEPARATE when they represent:
# 1. Different specific concepts (even if related)
#    ✗ WRONG: "Fast delivery" + "Careful packaging" → "Delivery quality"
#    ✓ RIGHT: Keep both as separate codes under "Delivery" category

# 2. Different aspects of the same topic
#    ✗ WRONG: "High price concern" + "Value for money" → "Pricing"
#    ✓ RIGHT: Keep both (different evaluative perspectives on price)

# 3. Different levels of specificity with practical utility
#    ✗ WRONG: "Salt level" + "Sugar amount" + "Spiciness" → "Seasoning"
#    ✓ RIGHT: Keep all three (researchers may want to report these separately)

# 4. Contrasting or opposing viewpoints
#    ✗ WRONG: "Too expensive" + "Good value" → "Price perception"
#    ✓ RIGHT: Keep both (opposite evaluations matter for analysis)

# The ONLY time to merge codes:
# • They express the EXACT SAME concept using different words (synonyms/paraphrases)
# • Examples: "Reliable service" + "Trustworthy provider" → "Reliability"
# • Test: Would a researcher ever want to distinguish between these? If yes → keep separate.
# </do_not_collapse_rules>

# <guidance>
# A high-quality codebook must be:
# - Non-redundant: No semantic duplicates (same idea in different words)
# - Preserves distinctions: ALL meaningfully different codes retained
# - Well-structured: Organize related codes in clear hierarchy (prefer 2-level)
# - Each code has exactly one parent (no multi-parenting)
# - Consistently labeled: Use short, uniform, action-oriented labels meaningful for: "{survey_question}"

# Hierarchy guidelines:
# • **PREFER 2-Level** (Theme → Codes) when possible
#   Example:
#   - Theme: "Pricing"
#     - Code: "Price transparency"
#     - Code: "Value for money perception"
#     - Code: "Competitive pricing"

# • **Use 3-Level** (Theme → Category → Codes) ONLY when:
#   - Multiple codes naturally group under a clear intermediate concept
#   - The category adds organizational clarity (not just abstraction)
#   - Example:
#     - Theme: "Product Quality"
#       - Category: "Taste Attributes"
#         - Code: "Saltiness level" (keep separate)
#         - Code: "Bitterness perception" (keep separate)
#         - Code: "Sweetness preferences" (keep separate)
#       - Code: "Freshness" (directly under theme)

# Remember: Categories GROUP codes, they don't REPLACE them!

# Labeling:
# - Prefer active, specific labels (e.g., "Seeks clearer instructions" over "Clarity")
# - Include a short definition/decision rule (≤ 20 words)
# </guidance>

# <merge_decision_criteria>
# For EVERY potential merge, ask:
# 1. Are these the EXACT SAME concept? (semantic identity test)
# 2. Would researchers want to distinguish these in analysis? (practical utility test)

# If answer to #1 is NO → DO NOT MERGE (keep as separate codes)
# If answer to #2 is YES → DO NOT MERGE (keep as separate codes)

# Only merge when #1 is YES AND #2 is NO.

# Examples:
# • "Merk X is reliable" + "Experiences reliable service"
#   → #1: YES (same concept), #2: NO (no utility in separating)
#   → ACTION: MERGE to "Merk X reliability"

# • "Salt concerns" + "Sweetness preferences"
#   → #1: NO (different concepts), #2: YES (researchers want to distinguish)
#   → ACTION: KEEP SEPARATE (group under "Additives" category if helpful)

# • "High fees" + "Expensive pricing"
#   → #1: YES (same concept), #2: NO (no utility in separating)
#   → ACTION: MERGE to "High pricing concerns"

# • "High fees" + "Good value for money"
#   → #1: NO (contrasting perspectives), #2: YES (contrast is meaningful)
#   → ACTION: KEEP SEPARATE
# </merge_decision_criteria>

# <analysis_steps>
# 1. Review all raw codes and identify TRUE duplicates (semantic identity test)
# 2. Merge ONLY semantically equivalent items (apply decision criteria)
# 3. Construct main themes (2–5 per dataset, based on data)
# 4. Assign ALL remaining codes to themes (prefer 2-level structure)
# 5. Create categories (3-level) ONLY if they add clear organizational value
# 6. Use concise, active phrasing for code labels
# 7. Provide detailed analysis:
#    - Which codes were merged (with IDs) and why
#    - Which similar codes were kept separate and why
#    - How hierarchy was structured
#    - Total codes preserved vs. merged
# </analysis_steps>

# <examples>
# Example 1 - Correct approach (keeping distinctions):
# Raw codes: ["Salt content too high", "Bitter aftertaste", "Sweetness level perfect", "Fresh taste"]
# Result:
# - Theme: "Taste"
#   - Code: "Salt content concerns" (id: 1)
#   - Code: "Bitter aftertaste" (id: 2)
#   - Code: "Sweetness satisfaction" (id: 3)
#   - Code: "Freshness perception" (id: 4)
# Analysis: "All four codes represent distinct taste perceptions. Kept separate despite shared theme."

# Example 2 - Correct merging (semantic duplicates):
# Raw codes: ["Price transparency", "Clear pricing info", "See prices clearly", "Value for money"]
# Result:
# - Theme: "Pricing"
#   - Code: "Price transparency" (merged ids: 1,2,3)
#   - Code: "Value for money perception" (id: 4)
# Analysis: "Merged codes 1-3 as semantic duplicates. Kept code 4 separate (different concept: value vs. clarity)."
# </examples>

# Output strictly as JSON:
# {{
#   "analysis": "Provide detailed analysis in {language}: (1) Which codes were merged and why (include IDs), (2) Which similar codes were kept separate and why, (3) How hierarchy was structured, (4) Total codes preserved vs. merged count.",
#   "refined_codebook": [
#     {{
#       "theme": "Main theme label",
#       "codes": [
#         {{
#           "id": "original code_id (or comma-separated IDs if merged)",
#           "code": "Code label",
#           "description": "≤ 20 words explanation",
#           "category": ""  // Empty string for 2-level (prefer this), or category name for 3-level
#         }}
#       ]
#     }}
#   ]
# }}

# Critical requirements:
# - Output must be valid JSON only — no commentary before or after
# - Default to "category": "" (2-level structure preferred)
# - Use "category": "Name" ONLY when it adds clear organizational value
# - In analysis, justify EVERY merge with semantic identity reasoning
# - In analysis, explain why related codes were kept separate
# - Report total: X original codes → Y refined codes (Z merged, Y-Z preserved)
# - Conduct analysis in {language}
# """

# =============================================================================
# STEP 8: CODE ASSIGNMENT
# =============================================================================

# CODE_ASSIGNMENT_PROMPT = """
# You are a {language} language expert in qualitative data analysis, specializing in applying codebooks to open-ended survey responses. Your task is to assign the single most appropriate code from a focused list of 6 candidate codes to a specific response segment.

# First, review the original survey question:
# <survey_question>
# {var_lab}
# </survey_question>

# Next, examine the response segment you need to analyze:
# <idea_to_analyze>
# Idea ID: {idea_id}
# Idea Text: {idea_text}
# </idea_to_analyze>

# Now, review the 6 candidate codes and their descriptions:
# <candidate_codes>
# {candidate_codes}
# </candidate_codes>

# Your goal is to select the single best fitting code for the response segment. Follow these steps:
# 1. Carefully read and understand each candidate code's definition.
# 2. Analyze the semantic meaning of the response segment, considering the context of the survey question.
# 3. Identify which code best captures the core concept expressed in the response.
# 4. Assign exactly one code, even if the fit isn't perfect. Choose the best available option based on semantic meaning.

# When selecting the best fitting code:
# - Prioritize exact conceptual matches based on meaning.
# - Do not rely solely on surface keywords. Base your choice on semantic alignment with the code's definition.


# After selecting the code, rate the strength of the fit using this scale:
# - Excellent (0.90–1.00): Exact match — the idea uses the same language or concepts as the code definition, with no ambiguity or need for interpretation.
# - Very Good (0.80–0.89): Very strong fit — conceptually aligns and clearly supports the code with minimal nuance or deviation.
# - Good (0.60–0.79): Clear but partial fit — the idea relates directly to the code, though some nuance, context, or wording differs from the definition.
# - Moderate (0.50–0.59): Partial or uncertain fit — the idea touches on similar concepts but lacks clarity, depth, or consistent alignment with the code.
# - Poor (0.30–0.49): Barely relevant — the connection to the code is weak or indirect, applied mainly due to lack of a better alternative.
# - Very Poor (0.00–0.29): Not relevant — the idea does not reflect the intent, meaning, or scope of the code.

# Provide your response in the following JSON format:
# <output_format>
# {{
#   "idea_id": "{idea_id}",
#   "idea": "{idea_text}",
#   "assigned_codes": ["SINGLE_CODE_NAME"],
#   "assignment_confidence": CONFIDENCE_SCORE,
#   "assignment_rationale": "Brief explanation of the conceptual match (in {language})"
# }}
# </output_format>

# Critical requirements:
# - Use exact code names as provided in the candidate codes list.
# - Assign one and only one code per response.
# - The confidence score must reflect conceptual fit, not how likely you feel about the assignment.
# - The rationale must explain the semantic connection to the code definition.
# - Return ONLY the JSON object in {language}.

# Begin the code assignment now.
# """

# # Stage 1: Evaluate default code from cluster
# DEFAULT_CODE_EVALUATION_PROMPT = """
# You are a {language} language expert in qualitative data analysis. Your task is to evaluate how well a default code fits a specific response segment.

# First, review the survey question:
# <survey_question>
# {var_lab}
# </survey_question>

# Next, examine the response segment to analyze:
# <idea_to_analyze>
# Idea ID: {idea_id}
# Idea Text: {idea_text}
# </idea_to_analyze>

# This response segment came from a cluster of similar responses. Here is the default code generated from that cluster:
# <default_code>
# Code: {default_code}
# Definition: {default_definition}
# </default_code>

# Your task is to evaluate how well this default code captures the meaning of the response segment.

# Consider:
# 1. Does the code definition accurately describe the idea expressed?
# 2. Is there semantic alignment between the idea and the code?
# 3. Would this code be appropriate for categorizing this response?

# Provide a confidence score using this scale:
# • 0.90–1.00: Extreme Confidence — Essentially identical meaning; no meaningful differences.
# • 0.70–0.89: High Confidence — Strong semantic overlap; only minor nuance differences.
# • 0.60–0.69: Moderate Confidence — Related meaning, but not consistently aligned.
# • 0.50–0.59: Low Confidence — Loosely related topic; weak semantic alignment.
# • 0.30–0.49: Very Low Confidence — Barely related; mostly mismatched meaning.
# • 0.00–0.29: No Confidence — No meaningful similarity at all.


# Provide your response in the following JSON format:
# {{
#   "idea_id": "{idea_id}",
#   "confidence": CONFIDENCE_SCORE,
#   "rationale": "Brief explanation of why this code does or does not fit (in {language})"
# }}

# Critical requirements:
# - The confidence score must reflect semantic fit
# - The rationale must explain the conceptual match or mismatch
# - Return ONLY the JSON object

# Begin the evaluation now.
# """

# # Stage 2: Fallback assignment from all codes
# FALLBACK_CODE_ASSIGNMENT_PROMPT = """
# You are a {language} language expert in qualitative data analysis. Your task is to assign the best code from all available codes to a response segment.

# First, review the survey question:
# <survey_question>
# {var_lab}
# </survey_question>

# Next, examine the response segment to analyze:
# <idea_to_analyze>
# Idea ID: {idea_id}
# Idea Text: {idea_text}
# </idea_to_analyze>

# The default code from this idea's cluster did not fit well (confidence: {default_confidence:.2f}).

# Now, review ALL available codes from the complete codebook:
# <all_codes>
# {all_codes}
# </all_codes>

# Your task is to select the single best code that captures the meaning of this response segment.

# Follow these steps:
# 1. Carefully read and understand each code's definition
# 2. Analyze the semantic meaning of the response segment
# 3. Identify which code best captures the core concept expressed
# 4. Assign exactly one code based on semantic alignment

# Provide a confidence score using this scale:
# • 0.90–1.00: Extreme Confidence — Essentially identical meaning; no meaningful differences.
# • 0.70–0.89: High Confidence — Strong semantic overlap; only minor nuance differences.
# • 0.60–0.69: Moderate Confidence — Related meaning, but not consistently aligned.
# • 0.50–0.59: Low Confidence — Loosely related topic; weak semantic alignment.
# • 0.30–0.49: Very Low Confidence — Barely related; mostly mismatched meaning.
# • 0.00–0.29: No Confidence — No meaningful similarity at all.


# Provide your response in the following JSON format:
# {{
#   "idea_id": "{idea_id}",
#   "assigned_codes": ["SINGLE_CODE_NAME"],
#   "assignment_confidence": CONFIDENCE_SCORE,
#   "assignment_rationale": "Brief explanation of the conceptual match (in {language})"
# }}

# Critical requirements:
# - Use exact code names as provided in the codes list
# - Assign one and only one code
# - The confidence score must reflect semantic fit
# - The rationale must explain the conceptual connection
# - Return ONLY the JSON object

# Begin the code assignment now.
# """

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

Follow these steps to complete your analysis:

1. First, carefully read the response text and identify any content that might relate to the code concept.
2. Extract the minimal supporting span from the response that provides evidence for the code. If no supporting evidence exists, note this.
3. Determine your confidence level using these anchors:
   - 0.90–1.00: Specific concept is explicitly present (or unambiguous paraphrase); clear supporting span exists; another trained coder would almost certainly agree
   - 0.70–0.89: Strong overlap with minor nuance gaps; supporting span exists
   - 0.50–0.69: Related theme but concept is incomplete, vague, or partly implied; supporting span is weak or generic
   - 0.00–0.49: Too general, tangential, no evidence, opposite meaning, or out of scope
4. Apply these decision rules:
   - If the response is broader/more generic than the code → the code does NOT fit (confidence ≤ 0.49)
   - If the response only touches the same theme but not the specific concept → does NOT fit (confidence ≤ 0.49)
   - If no explicit or clearly paraphrased evidence of the code's concept appears in the text → does NOT fit (confidence ≤ 0.39)


Provide your response in this exact JSON format:
{{
  "idea_id": "{idea_id}",
  "confidence": CONFIDENCE_SCORE,
  "rationale": "Brief explanation of why this code does or does not fit (in {language})"
}}

Critical requirements:
- The confidence score must be a number between 0.00 and 1.00 that reflects semantic fit
- The rationale must explain the conceptual match or mismatch between the code and response
- Focus on whether the response contains the specific concept defined by the code, not just related themes
- Return ONLY the JSON object, no additional text

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
- You must assign EXACTLY ONE code from the codebook
- A code only fits if the specific concept in its definition is explicitly stated or clearly paraphrased in the response
- If the response is more general than a code's definition, that code does NOT fit
- If no code has clear evidence in the response, assign "ONBEKEND" (UNMATCHED) with low confidence
- Do not infer meaning that is not directly stated or clearly paraphrased

**Confidence Level Anchors:**
- 0.90–1.00: Specific concept is explicitly present or unambiguously paraphrased; clear supporting evidence exists; another trained coder would almost certainly agree
- 0.70–0.89: Strong conceptual overlap with only minor nuance gaps; solid supporting evidence exists
- 0.50–0.69: Related theme but concept is incomplete, vague, or partly implied; supporting evidence is weak or generic
- 0.00–0.49: Too general, tangential, no evidence, opposite meaning, or out of scope

**Analysis Process:**
Follow these steps in your analysis:

1. **Evidence Identification**: Carefully read the response text and identify any content that might relate to specific code concepts from the codebook.
2. **Supporting Span Extraction**: Extract the minimal supporting span from the response that provides evidence for potential codes. If no supporting evidence exists for any code, note this clearly.
3. **Conceptual Matching**: For each potential code match, determine if the response contains the specific concept described in the code definition, not just a related theme.
4. **Confidence Assessment**: Using the confidence anchors above, assess your confidence level. Remember:
   - If the response is broader/more generic than the code → the code does NOT fit (confidence ≤ 0.49)
   - If the response only touches the same theme but not the specific concept → does NOT fit (confidence ≤ 0.49)
   - If no explicit or clearly paraphrased evidence appears → does NOT fit (confidence ≤ 0.39)
5. **Final Assignment**: Select the single best-fitting code, or "ONBEKEND" if no code fits well.

Provide your analysis and assignment in this exact JSON format:
{{
  "idea_id": "{idea_id}",
  "assigned_codes": ["SINGLE_CODE_NAME"],
  "assignment_confidence": CONFIDENCE_SCORE,
  "assignment_rationale": "Brief explanation of the conceptual match (in {language})"
}}

"""
