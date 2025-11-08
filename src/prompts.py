
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
b) draft first-version codebook entries for each theme, including inclusion/exclusion rules and abstraction level. 

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
    - Can all codes be grouped around one central organizing ?
    - Practical test: 
        - Can I summarize all codes into one sentence that:
            a) alligns with <guidance>,
            b) captures exactly ONE ATOMIC theme,
            c) contains no coordinating conjunctions (e.g., "and," "or") or list punctuation (commas, slashes),
            d) preserves unity, consistency, and contrast?
        - If yes → single COC. If no → multiple COCs.
    - If one COC exists, continue with it; otherwise, work with multiple COCs.

4. Refine COCs:
    - Remove singletons (COCs based on only one code).
    - Remove COCs without a dominant shared pattern or conceptual overlap.
    - Remove vague or overly broad COCs (e.g., “positivity,” “challenges,” “general satisfaction”).

5. For each retained COC, produce a codebook-ready draft with:
   - theme_label: "[≤ 10 words | active/actionable formulation of ONE ATOMIC theme in relation to the research question]"
   - theme_clarification: "[≤ 30 words | illustrative descriptive codes from <inputs> that clarify and support the label — tight, grounded, evidence-based]"
   - abstraction_level: Select one of: "Driver/Motive/Why" | "Attribute/What" | "Action/How"
   - assignment_rules:
       inclusion: Write precise, positive rules that describe when to assign the theme (start each rule with a verb; prefer observable cues).
       exclusion: Define boundaries that prevent overreach (what must NOT be included).
       near_neighbor:
        • label: closest potentially-confusable theme or "Unknown"
        • tell_apart_rule: one sentence explaining how to distinguish the two.
       
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
        "assignment_rules": {{
          "inclusion": [
            "[inclusion rule 1 in {language}]",
            "[inclusion rule 2 in {language}]"
          ],
          "exclusion": [
            "[exclusion rule 1 in {language}]",
            "[exclusion rule 2 in {language}]"
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
1. Cosine Similarity Rules:

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
       "parent_theme_label": "null or suggested parent label"}},
    "justification": "Explain modification decision by referencing cosine score and whether motive was same or different, or null if use/create".
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

Output the result in this strict JSON schema (no commentary or explanation):
{{
  "generated_code": {{
    "theme_number": {theme_id},
    "theme_name": "{cluster_summary}",
    "source_code": "null",
    "code_label": "new or modified code label in {language}",
    "code_definition": "≤25-word operational definition in {language}"
  }}
}}

Critical remarks:
- Use theme_id provided.
- Use theme_name provided.
- Use source_code provided
"""

CODING_MODIFICATION_PROMPT = """
You are a {language} qualitative research assistant updating a codebook.
Your task is to MODIFY an existing code so that it fully and correctly includes a new theme,
while preserving **atomic meaning** and **clear conceptual boundaries**.

<inputs>
Survey question:
"{survey_question}"

New theme:
- name: "{theme_name}"
- description: "{theme_description}"

- inclusion (what this theme *does* refer to): 
    {inclusion}
- exclusion (what this theme *does NOT* refer to): 
    {exclusion}

Original code to modify:
{source_code}

Modification parameters (from previous decision stage):
- modify_instruction: {modify_instruction}
- motive_comparison: {motive_comparison}
- abstraction_level_action: {abstraction_level_action}
- inclusion_update: {inclusion_update}
- exclusion_update: {exclusion_update}
- parent_theme_label: {parent_theme_label}
</inputs>

------------------------------------------------------------
MODIFICATION INSTRUCTIONS:
{modification_instructions}

------------------------------------------------------------
LABEL RULES:
    - ≤ 10 words 
    - Active/actionable formulation of ONE ATOMIC theme in relation to the research question
    - Atomic means "One domain or aspect only"
    - If verb is used → one main verb (present tense).
    - **Never** include reasons (no "to", "so that", "because").
    - Avoid punctuation: "/", "&", ",", "–", ":" (unless lexicalized).
    - Maintain **one polarity** (either increase/strengthen OR reduce/avoid).

4) DEFINITION RULES:
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

------------------------------------------------------------
OUTPUT FORMAT (valid JSON only, no commentary, in {language}):

{{
  "generated_code": {{
    "theme_number": {theme_id},
    "theme_name": "{cluster_summary}",
    "source_code": {source_code},
    "code_label": "yur new/modified code label in {language}",
    "code_definition": "your definition in {language}"
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

Step 5: Determine your final components:
- validated_decision: USE, MODIFY, or CREATE code
- source_code: 
    - If USE, this exact code: {source_code}
    - If MODIFY, the exact code from the existing codebook you seek to modify
    - If CREATE, write "null"
- validated_code and validated_decision: final compliant label and definition
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
      "definition": "Final validated definition (≤25 words, operational, grounded)"
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

# BACKUP (2025-10-31): Original version before anti-collapse fix
# Issue: Was collapsing meaningfully different codes based on abstraction level
# Example: "Salt", "Bitter", "Sweet" → collapsed to "Taste preferences"
# Fix: Distinguish semantic merging from hierarchical grouping

CODEBOOK_REFINEMENT_PROMPT = """
You are a qualitative researcher and codebook methodologist.
Your task is to take a raw list of descriptive codes and transform it into a refined and structured codebook.
The descriptive codes are derived from survey responses.

<inputs>
Language to use: {language}

survey_question: {survey_question}

Raw descriptive codes to refine:
{raw_codes}
</inputs>

<critical_requirement>
Preserve the *conceptual content* of ALL codes.
- Do NOT remove or lose any unique ideas.
- Do NOT collapse distinct concepts into a single code.
- You MAY merge true duplicates (semantically identical codes).
- You MUST preserve ALL meaningful distinctions between different ideas.

IMPORTANT DISTINCTION:
• MERGING = Combining semantic duplicates (reduces redundancy) ✓
• COLLAPSING = Combining distinct concepts (loses information) ✗

Example of MERGING (correct):
  "Price transparency" + "Clear pricing information" → "Price transparency" (same concept, different words)

Example of COLLAPSING (incorrect):
  "Salt concerns" + "Bitterness" + "Sweetness preferences" → "Taste preferences" (3 DIFFERENT concepts lost!)

NEVER collapse codes just because they belong to the same category or theme.
</critical_requirement>

<do_not_collapse_rules>
Keep codes SEPARATE when they represent:
1. Different specific concepts (even if related)
   ✗ WRONG: "Fast delivery" + "Careful packaging" → "Delivery quality"
   ✓ RIGHT: Keep both as separate codes under "Delivery" category

2. Different aspects of the same topic
   ✗ WRONG: "High price concern" + "Value for money" → "Pricing"
   ✓ RIGHT: Keep both (different evaluative perspectives on price)

3. Different levels of specificity with practical utility
   ✗ WRONG: "Salt level" + "Sugar amount" + "Spiciness" → "Seasoning"
   ✓ RIGHT: Keep all three (researchers may want to report these separately)

4. Contrasting or opposing viewpoints
   ✗ WRONG: "Too expensive" + "Good value" → "Price perception"
   ✓ RIGHT: Keep both (opposite evaluations matter for analysis)

The ONLY time to merge codes:
• They express the EXACT SAME concept using different words (synonyms/paraphrases)
• Examples: "Reliable service" + "Trustworthy provider" → "Reliability"
• Test: Would a researcher ever want to distinguish between these? If yes → keep separate.
</do_not_collapse_rules>

<guidance>
A high-quality codebook must be:
- Non-redundant: No semantic duplicates (same idea in different words)
- Preserves distinctions: ALL meaningfully different codes retained
- Well-structured: Organize related codes in clear hierarchy (prefer 2-level)
- Each code has exactly one parent (no multi-parenting)
- Consistently labeled: Use short, uniform, action-oriented labels meaningful for: "{survey_question}"

Hierarchy guidelines:
• **PREFER 2-Level** (Theme → Codes) when possible
  Example:
  - Theme: "Pricing"
    - Code: "Price transparency"
    - Code: "Value for money perception"
    - Code: "Competitive pricing"

• **Use 3-Level** (Theme → Category → Codes) ONLY when:
  - Multiple codes naturally group under a clear intermediate concept
  - The category adds organizational clarity (not just abstraction)
  - Example:
    - Theme: "Product Quality"
      - Category: "Taste Attributes"
        - Code: "Saltiness level" (keep separate)
        - Code: "Bitterness perception" (keep separate)
        - Code: "Sweetness preferences" (keep separate)
      - Code: "Freshness" (directly under theme)

Remember: Categories GROUP codes, they don't REPLACE them!

Labeling:
- Prefer active, specific labels (e.g., "Seeks clearer instructions" over "Clarity")
- Include a short definition/decision rule (≤ 20 words)
</guidance>

<merge_decision_criteria>
For EVERY potential merge, ask:
1. Are these the EXACT SAME concept? (semantic identity test)
2. Would researchers want to distinguish these in analysis? (practical utility test)

If answer to #1 is NO → DO NOT MERGE (keep as separate codes)
If answer to #2 is YES → DO NOT MERGE (keep as separate codes)

Only merge when #1 is YES AND #2 is NO.

Examples:
• "Merk X is reliable" + "Experiences reliable service"
  → #1: YES (same concept), #2: NO (no utility in separating)
  → ACTION: MERGE to "Merk X reliability"

• "Salt concerns" + "Sweetness preferences"
  → #1: NO (different concepts), #2: YES (researchers want to distinguish)
  → ACTION: KEEP SEPARATE (group under "Additives" category if helpful)

• "High fees" + "Expensive pricing"
  → #1: YES (same concept), #2: NO (no utility in separating)
  → ACTION: MERGE to "High pricing concerns"

• "High fees" + "Good value for money"
  → #1: NO (contrasting perspectives), #2: YES (contrast is meaningful)
  → ACTION: KEEP SEPARATE
</merge_decision_criteria>

<analysis_steps>
1. Review all raw codes and identify TRUE duplicates (semantic identity test)
2. Merge ONLY semantically equivalent items (apply decision criteria)
3. Construct main themes (2–5 per dataset, based on data)
4. Assign ALL remaining codes to themes (prefer 2-level structure)
5. Create categories (3-level) ONLY if they add clear organizational value
6. Use concise, active phrasing for code labels
7. Provide detailed analysis:
   - Which codes were merged (with IDs) and why
   - Which similar codes were kept separate and why
   - How hierarchy was structured
   - Total codes preserved vs. merged
</analysis_steps>

<examples>
Example 1 - Correct approach (keeping distinctions):
Raw codes: ["Salt content too high", "Bitter aftertaste", "Sweetness level perfect", "Fresh taste"]
Result:
- Theme: "Taste"
  - Code: "Salt content concerns" (id: 1)
  - Code: "Bitter aftertaste" (id: 2)
  - Code: "Sweetness satisfaction" (id: 3)
  - Code: "Freshness perception" (id: 4)
Analysis: "All four codes represent distinct taste perceptions. Kept separate despite shared theme."

Example 2 - Correct merging (semantic duplicates):
Raw codes: ["Price transparency", "Clear pricing info", "See prices clearly", "Value for money"]
Result:
- Theme: "Pricing"
  - Code: "Price transparency" (merged ids: 1,2,3)
  - Code: "Value for money perception" (id: 4)
Analysis: "Merged codes 1-3 as semantic duplicates. Kept code 4 separate (different concept: value vs. clarity)."
</examples>

Output strictly as JSON:
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
          "category": ""  // Empty string for 2-level (prefer this), or category name for 3-level
        }}
      ]
    }}
  ]
}}

Critical requirements:
- Output must be valid JSON only — no commentary before or after
- Default to "category": "" (2-level structure preferred)
- Use "category": "Name" ONLY when it adds clear organizational value
- In analysis, justify EVERY merge with semantic identity reasoning
- In analysis, explain why related codes were kept separate
- Report total: X original codes → Y refined codes (Z merged, Y-Z preserved)
- Conduct analysis in {language}
"""

THEME_ORGANIZATION_REASONING_PROMPT = """
You are a qualitative research specialist with expertise in thematic analysis. Your task is to organize a set of codes into atomic themes based on semantic similarity in the context of a research question.

Here is the research question you should consider:
<research_question>
{research_question}
</research_question>

Here are the codes with their definitions that you need to organize:
<codes_and_definitions>
{codebook}
</codes_and_definitions>

You should provide all labels and descriptions in this language:
<language>
{language}
</language>

## CORE TASK
Organize the provided codes into **atomic themes** based on semantic similarity in light of the research question. An atomic theme label refers to one single idea that cannot be split into multiple, distinct concepts without losing clarity.

## KEY REQUIREMENTS

**Atomic Theme Names:**
- Must express only ONE clear concept
- Do NOT use compound labels, slashes (/), conjunctions (like "and"/"en"), or lists of terms
- ✅ Good: "Price", "Health", "Convenience" 
- ❌ Bad: "Price and Quality", "Health/Nutrition", "Time and Convenience"

**Grouping Rules:**
- Each theme must contain 2 or more codes that express related ideas
- Base groupings on meaning in the context of the research question
- Every unique idea should be represented once. If two or more codes express the same or highly similar concept, you may merge them or remove the duplicates to maintain atomicity and avoid redundancy. Each remaining code should appear only once.
- Group codes that address similar aspects or concerns related to the research question

## INSTRUCTIONS

<scratchpad>
Before providing your final answer, think through the following:

1. **Analyze the codes:** What are the main conceptual areas represented in the codes?
2. **Identify natural groupings:** Which codes seem to address similar aspects of the research question?
3. **Create atomic theme names:** For each grouping, what single concept best captures what those codes are about?
4. **Verify completeness:** Have you included every code exactly once?
5. **Check atomicity:** Are all your theme names truly atomic (single concepts)?
6. **Handle redundancy:** 
   - If you encounter codes that are semantically identical (e.g., different phrasings of the same idea) or highly overlapping (e.g., small wording differences but no substantial conceptual distinction), you may:
     - Keep only the clearest or most comprehensive version.
     - Remove or merge the others.
   
</scratchpad>

Now provide your analysis in the following JSON format. Return only valid JSON with no additional text:

```json
{{
  "themes": [
    {{
      "theme_name": "[Atomic theme name in {language}]",
      "theme_description": "[Detailed description of what this theme represents in {language}]",
      "codes": [
        {{
          "code": "[Original code name]",
          "definition": "[Original code definition]",
        }}
      ]
    }}
  ],
  "methodology": "Single-prompt hierarchical theme organization using reasoning model",
  "total_codes_organized": {codes_count},
  "language": "{language}"
}}    
```

## FINAL CHECKLIST
Before submitting, verify that:
- All theme names are atomic (no conjunctions, slashes, or multiple concepts)
- Each code appears exactly once
- Every theme contains at least 2 codes
- All labels and descriptions are in the specified language
- The JSON format is valid and complete
"""

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

# Stage 1: Evaluate default code from cluster
DEFAULT_CODE_EVALUATION_PROMPT = """
You are a {language} language expert in qualitative data analysis. Your task is to evaluate how well a default code fits a specific response segment.

First, review the survey question:
<survey_question>
{var_lab}
</survey_question>

Next, examine the response segment to analyze:
<idea_to_analyze>
Idea ID: {idea_id}
Idea Text: {idea_text}
</idea_to_analyze>

This response segment came from a cluster of similar responses. Here is the default code generated from that cluster:
<default_code>
Code: {default_code}
Definition: {default_definition}
</default_code>

Your task is to evaluate how well this default code captures the meaning of the response segment.

Consider:
1. Does the code definition accurately describe the idea expressed?
2. Is there semantic alignment between the idea and the code?
3. Would this code be appropriate for categorizing this response?

Provide a confidence score using this scale:
• 0.90–1.00: Extreme Confidence — Essentially identical meaning; no meaningful differences.
• 0.70–0.89: High Confidence — Strong semantic overlap; only minor nuance differences.
• 0.60–0.69: Moderate Confidence — Related meaning, but not consistently aligned.
• 0.50–0.59: Low Confidence — Loosely related topic; weak semantic alignment.
• 0.30–0.49: Very Low Confidence — Barely related; mostly mismatched meaning.
• 0.00–0.29: No Confidence — No meaningful similarity at all.


Provide your response in the following JSON format:
{{
  "idea_id": "{idea_id}",
  "confidence": CONFIDENCE_SCORE,
  "rationale": "Brief explanation of why this code does or does not fit (in {language})"
}}

Critical requirements:
- The confidence score must reflect semantic fit
- The rationale must explain the conceptual match or mismatch
- Return ONLY the JSON object

Begin the evaluation now.
"""

# Stage 2: Fallback assignment from all codes
FALLBACK_CODE_ASSIGNMENT_PROMPT = """
You are a {language} language expert in qualitative data analysis. Your task is to assign the best code from all available codes to a response segment.

First, review the survey question:
<survey_question>
{var_lab}
</survey_question>

Next, examine the response segment to analyze:
<idea_to_analyze>
Idea ID: {idea_id}
Idea Text: {idea_text}
</idea_to_analyze>

The default code from this idea's cluster did not fit well (confidence: {default_confidence:.2f}).

Now, review ALL available codes from the complete codebook:
<all_codes>
{all_codes}
</all_codes>

Your task is to select the single best code that captures the meaning of this response segment.

Follow these steps:
1. Carefully read and understand each code's definition
2. Analyze the semantic meaning of the response segment
3. Identify which code best captures the core concept expressed
4. Assign exactly one code based on semantic alignment

Provide a confidence score using this scale:
• 0.90–1.00: Extreme Confidence — Essentially identical meaning; no meaningful differences.
• 0.70–0.89: High Confidence — Strong semantic overlap; only minor nuance differences.
• 0.60–0.69: Moderate Confidence — Related meaning, but not consistently aligned.
• 0.50–0.59: Low Confidence — Loosely related topic; weak semantic alignment.
• 0.30–0.49: Very Low Confidence — Barely related; mostly mismatched meaning.
• 0.00–0.29: No Confidence — No meaningful similarity at all.


Provide your response in the following JSON format:
{{
  "idea_id": "{idea_id}",
  "assigned_codes": ["SINGLE_CODE_NAME"],
  "assignment_confidence": CONFIDENCE_SCORE,
  "assignment_rationale": "Brief explanation of the conceptual match (in {language})"
}}

Critical requirements:
- Use exact code names as provided in the codes list
- Assign one and only one code
- The confidence score must reflect semantic fit
- The rationale must explain the conceptual connection
- Return ONLY the JSON object

Begin the code assignment now.
"""
