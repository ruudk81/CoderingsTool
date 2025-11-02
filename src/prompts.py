
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
    - If the response is empty, irrelevant, or “N/A”: return an empty array [].
    - If the response is off-topic: extract ideas anyway but note they may be off-topic.
    - If only one idea is present: return it in a single-item array.
</instructions>

<output_format>
Return the extracted ideas as a JSON array. Each item should include:
- "respondent_id": exactly as provided
- "idea_id": a string number (numbering always starts at "1" and increments sequentially -e.g. "1", "2", etc.).
- "idea": the descriptive phrase in {language}, normalized and phrased using the provided phrasing template.

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
Your task is to analyze a cluster of descriptive codes and construct one or more themes by identifying dominant patterns of shared meaning. 
The descriptive codes are derived from responses to the research question.

<inputs>
Language to use: {language}

Cluster ID: {cluster_id}

Research question:
"{survey_question}"

Cluster of descriptive codes to analyze:
{cluster_text}
</inputs>

<guidance>
- A central organizing concept (COC) is a unifying pattern of shared meaning, sentiment, or intention that brings together a group of codes into an **ATOMIC** theme in light of the research question.

- Atomic means:
  • One single idea, action, or expectation (not multiple at once).  
  • Belongs to one aspect, domain, category, or product attribute only.  
  • Expresses one consistent sentiment or intention (no mixing of positive/negative, or “keep” vs. “change”).  
  • Is concrete and directly actionable — cannot be meaningfully split further.  
  • If a code mentions multiple aspects (joined by “and,” “or,” commas, lists), crosses domains, or contains contradictory stances → split into multiple atomic concepts.  
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

5. Construct theme(s):
    - Each theme must represent ONE ATOMIC concept only — see <guidance>.
    - If multiple atomic concepts exist, create multiple theme objects (one per concept).
    - Do not combine multiple aspects, domains, or contradictory stances into a single theme.
   
6. Document the analysis:
    - State how many COCs were identified.
    - If only one COC: explain why it is sufficient.
    - If multiple COCs: justify why not a single COC.

7. Write a theme label and clarification for that label:
    - Theme Label Template: "[≤ 10 words |  active/actionable formulation of ONE ATOMIC theme in relation to the research question - short, active, atomic]"
    - Clarification Template: "[≤ 30 words | illustrative descriptive codes from <inputs> that clarify and support the label - tight, grounded, evidence-based]"

</analysis_steps>

Provide your response as a valid JSON dictionary using this exact structure:
{{
  "{cluster_id}": {{
    "analysis": "Provide your analysis here in {language}. Use the following format:\n\n1. [First analytical point]\n2. [Second analytical point]\n3. [Continue numbering each point sequentially].\n\nEach new point must start on a new line and be clearly numbered.",
    "extracted_themes": [
      {{
        "theme_id": 1,
        "theme_label": "[≤10-word label in {language}]",
        "theme_clarification": "[≤30-word clarification in {language}, with representative codes]"
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
You are a {language} qualitative research assistant helping to maintain a codebook for survey data analysis.
Your task is to decide whether existing codes in a codebook are sufficient to describe a new theme, or whether modifications or new codes are needed.

<inputs>
Here is the survey question being analyzed:
"{survey_question}"

Here is the new theme:
- name: "{theme_name}"
- description: "{theme_description}"

Here is the list with existing codes in the codebook:
{code_text}
</inputs>

You have three possible decisions:
- **USE**: An existing code already captures the new theme's central meaning sufficiently
- **MODIFY**: An existing code is close but needs refinement for clarity, scope, or better alignment
- **CREATE**: No existing code sufficiently captures the new theme

---

A. **Understanding Cosine Similarity**
<cosine_similarity>
Each code is provided with a cosine similarity score (0.0-1.0) comparing it to the new theme:

**Cosine Similarity** measures semantic similarity - do the theme and code MEAN the same thing?
- Uses AI embeddings to capture meaning beyond exact wording
- Handles synonyms and paraphrasing (e.g., "fast delivery" ≈ "quick shipping")
- High cosine (≥0.88) = Semantically equivalent concepts
- Medium cosine (0.75-0.88) = Similar concepts needing refinement
- Low cosine (<0.75) = Different concepts

Cosine captures what matters in qualitative coding: whether two descriptions refer to the same underlying concept, regardless of the specific words used.
</cosine_similarity>

---
B. **Decision Guidelines (STRICT)**
<decision_guidelines>
Make your decision strictly based on cosine similarity — no exceptions:

   1. Pick the single best match = candidate with the highest cosine.
   2. Apply guidelines:
       - USE if cosine ≥ 0.88
       - MODIFY if 0.75 ≤ cosine < 0.88
       - CREATE if cosine < 0.75
   3. Final consistency check before output:
       - If decision=MODIFY and cosine<0.75 ⇒ set to CREATE
       - If decision=USE and cosine<0.88 ⇒ set to MODIFY
</decision_guidelines>

---

C. **Decision Examples**
<examples>
Example 1 - Clear USE:
  Theme: "Delivery speed problems"
  Code:  "Delivery speed issues" (cosine: 0.92)
  → Decision: USE (semantically equivalent - nearly identical meaning)

Example 2 - Clear USE (synonyms):
  Theme: "Fast shipping concerns"
  Code:  "Quick delivery issues" (cosine: 0.91)
  → Decision: USE (different words, same meaning - cosine captures this)

Example 3 - Clear MODIFY:
  Theme: "Fast delivery and tracking concerns"
  Code:  "Delivery issues" (cosine: 0.82)
  → Decision: MODIFY (related concepts, but theme includes tracking aspect - broaden code scope)

Example 4 - Clear CREATE:
  Theme: "Packaging quality defects"
  Code:  "Delivery speed" (cosine: 0.35)
  → Decision: CREATE (completely different concepts - insufficient overlap)
</examples>

---

D. **Edge Cases**
<edge_cases>
- For borderline cases (cosine 0.85-0.90): Use qualitative judgment - if meaning is fully captured despite minor wording differences, choose USE; if refinement would improve clarity, choose MODIFY.
- If the theme combines multiple distinct ideas (e.g., "X and Y"), choose CREATE (themes should be atomic).
- When modifying, ensure changes don't broaden codes beyond their intended scope or make them too general.
- Always cite the cosine score in your justification (e.g., "cosine 0.83 indicates semantic similarity; MODIFY to broaden scope").
- Trust the cosine score - it captures semantic meaning that word-matching cannot.
</edge_cases>

---

Your response must be valid JSON only, following this exact format:

Output schema:
{{
  "coding_decision": {{
    "theme_number": {theme_id},
    "theme_name": "Exact name of the theme from <theme_to_code>",
    "matched_candidates": [
      {{"code": "Exact candidate code A", "definition": "Definition in light of the survey question"}},
      {{"code": "Exact candidate code B", "definition": "Definition in light of the survey question"}}
    ],
    "decision": "USE | MODIFY | CREATE",
    "source_code": "Exact candidate code name if use/modify, or null if create",
    "justification": "Justify decision by referencing "best_cosine": <number> and "rule_applied": "USE(≥0.88)" | "MODIFY(0.75–0.88)" | "CREATE(<0.75)".
  }}
}}


Critical requirements:
- Output must be valid JSON only — no extra commentary or explanation.
- Use theme_id provided.
- Keep field names in English; write values in {language}.
- Justification must clearly state the estimated overlap percentage and reference the cosine score and decision rule applied.
"""

CODE_CREATION_PROMPT = """
You are a qualitative research assistant helping to maintain a codebook for survey data analysis. 
Your task is to CREATE a new code that captures a distinct theme emerging from responses to a specific survey question.

<inputs>
1) Language to use: {language}

2) Survey question:
"{survey_question}"

3) New theme that emerged:
- name: "{theme_name}"
- description: "{theme_description}"
</inputs>

Follow these critical coding principles:

- **ATOMIC**: The code must express ONE clear idea only.
    - Forbidden punctuation (unless lexicalized): "/", "&", "+", ",", ";", ":", "-", "–"
    - Use only ONE main verb if present.
- **PRECISE**: Code boundaries must be specific enough to enable consistent coding decisions.
- **CONCISE**: Code labels should be 5–10 words maximum. If longer, shorten while preserving meaning.
- **OPERATIONAL**: Code label and definition must align with the survey question.
- **GROUNDED**: Base your label and definition on the provided theme.    

Consider these possible **code types**:
- Attribute codes: qualities or characteristics
- Process codes: actions, procedures, or methods  
- Relational codes: interactions or relationships
- State codes: existing conditions or statuses
- Evaluative codes: assessments or judgments
- Brand/company names: named entities

Use these **strong patterns** for code definitions (≤25 words):
- "References to [specific limitation or constraint] affecting [process or outcome]."
- "Mentions of [positive or negative] changes in [behavior or practice]."
- "Expressions of [emotion or attitude] regarding [situation or process]."

Avoid these **weak patterns**:
- Compound: "References to [issue A] including [aspect 1], [aspect 2], [aspect 3]."
- Vague: "Mentions of various [things] related to [topic]."
- Interpretive: "Underlying [abstract concept] manifesting in different ways."

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
You are a qualitative research assistant helping to maintain a codebook for survey data analysis. 
Your task is to MODIFY an existing code so that it fits a new theme, while preserving the **core meaning** of the original code.

<inputs>
1) Language to use: {language}

2) Survey question:
"{survey_question}"

3) New theme that emerged:
- name: "{theme_name}"
- description: "{theme_description}"

4) Original code to modify:
{source_code}
</inputs>

Follow these critical coding principles:

- **ATOMIC**: The code must express ONE clear idea only.
    - Forbidden punctuation (unless lexicalized): "/", "&", "+", ",", ";", ":", "-", "–"
    - Use only ONE main verb if present.
- **PRECISE**: Code boundaries must be specific enough to enable consistent coding decisions.
- **CONCISE**: Code labels should be 5–10 words maximum. If longer, shorten while preserving meaning.
- **OPERATIONAL**: Code label and definition must align with the survey question.
- **GROUNDED**: Base your label and definition on the provided theme.    

Consider these possible **code types**:
- Attribute codes: qualities or characteristics
- Process codes: actions, procedures, or methods  
- Relational codes: interactions or relationships
- State codes: existing conditions or statuses
- Evaluative codes: assessments or judgments
- Brand/company names: named entities

Use these **strong patterns** for code definitions (≤25 words):
- "References to [specific limitation or constraint] affecting [process or outcome]."
- "Mentions of [positive or negative] changes in [behavior or practice]."
- "Expressions of [emotion or attitude] regarding [situation or process]."

Avoid these **weak patterns**:
- Compound: "References to [issue A] including [aspect 1], [aspect 2], [aspect 3]."
- Vague: "Mentions of various [things] related to [topic]."
- Interpretive: "Underlying [abstract concept] manifesting in different ways.

Output the result in this strict JSON schema (no commentary or explanation):
{{
  "generated_code": {{
    "theme_number": {theme_id},
    "theme_name": "{cluster_summary}",
    "source_code": {source_code},
    "code_label": "new or modified code label in {language}",
    "code_definition": "≤25-word operational definition in {language}"
  }}
}}

Critical remarks:
- Use theme_id provided.
- Use theme_name provided.
- Use source_code provided
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
• "ASN is reliable" + "Experiences reliable service"
  → #1: YES (same concept), #2: NO (no utility in separating)
  → ACTION: MERGE to "ASN reliability"

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

CODE_ASSIGNMENT_PROMPT = """
You are a {language} language expert in qualitative data analysis, specializing in applying codebooks to open-ended survey responses. Your task is to assign the single most appropriate code from a focused list of 6 candidate codes to a specific response segment.

First, review the original survey question:
<survey_question>
{var_lab}
</survey_question>

Next, examine the response segment you need to analyze:
<idea_to_analyze>
Idea ID: {idea_id}
Idea Text: {idea_text}
</idea_to_analyze>

Now, review the 6 candidate codes and their descriptions:
<candidate_codes>
{candidate_codes}
</candidate_codes>

Your goal is to select the single best fitting code for the response segment. Follow these steps:
1. Carefully read and understand each candidate code's definition.
2. Analyze the semantic meaning of the response segment, considering the context of the survey question.
3. Identify which code best captures the core concept expressed in the response.
4. Assign exactly one code, even if the fit isn't perfect. Choose the best available option based on semantic meaning.

When selecting the best fitting code:
- Prioritize exact conceptual matches based on meaning.
- Do not rely solely on surface keywords. Base your choice on semantic alignment with the code's definition.


After selecting the code, rate the strength of the fit using this scale:
- Excellent (0.90–1.00): Exact match — the idea uses the same language or concepts as the code definition, with no ambiguity or need for interpretation.
- Very Good (0.80–0.89): Very strong fit — conceptually aligns and clearly supports the code with minimal nuance or deviation.
- Good (0.60–0.79): Clear but partial fit — the idea relates directly to the code, though some nuance, context, or wording differs from the definition.
- Moderate (0.50–0.59): Partial or uncertain fit — the idea touches on similar concepts but lacks clarity, depth, or consistent alignment with the code.
- Poor (0.30–0.49): Barely relevant — the connection to the code is weak or indirect, applied mainly due to lack of a better alternative.
- Very Poor (0.00–0.29): Not relevant — the idea does not reflect the intent, meaning, or scope of the code.

Provide your response in the following JSON format:
<output_format>
{{
  "idea_id": "{idea_id}",
  "idea": "{idea_text}",
  "assigned_codes": ["SINGLE_CODE_NAME"],
  "assignment_confidence": CONFIDENCE_SCORE,
  "assignment_rationale": "Brief explanation of the conceptual match (in {language})"
}}
</output_format>

Critical requirements:
- Use exact code names as provided in the candidate codes list.
- Assign one and only one code per response.
- The confidence score must reflect conceptual fit, not how likely you feel about the assignment.
- The rationale must explain the semantic connection to the code definition.
- Return ONLY the JSON object in {language}.

Begin the code assignment now.
"""
