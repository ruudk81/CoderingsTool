
# =============================================================================
# STEP 2: SPELL CHECKING
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
# STEP 3: QUALITY FILTERING 
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
# STEP 4: IDEA EXTRACTION  
# =============================================================================

IDEA_EXTRACTION_PROMPT = """
You are a {language} language expert in analyzing written responses to open-ended questions in {language} collected in surveys. 
Your task is to extract ALL distinct ideas expressed in a respondent's written answer.  

<inputs>
survey question: {var_lab}
Respondent ID: {respondent_id}
Written response: {response}
</inputs>

<instructions>
1. Understand the Context
    - Read the survey question and response carefully.
    - Identify the primary subject(s) of the question (e.g., a product, service, experience, or event).
    - Determine the CANONICAL SUBJECT (the main product/service/topic named or implied by the survey question).
    - Determine the CANONICAL ACTOR (who is expected to act: e.g., manufacturer, retailer, teacher; derive from the question or response).
    - Decide whether to use SUBJECT or ACTOR phrasing for the entire response:
        - Prefer SUBJECT phrasing unless the question explicitly focuses on the actor's actions.
        - Use the same template type consistently for all ideas in the response.

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


4. Canonical Phrasing Templates
    - SUBJECT template: "[CANONICAL_SUBJECT] [should/needs to/must/is/are] [property or outcome]"
    - ACTOR template: "[CANONICAL_ACTOR] [should/needs to/must] [action] [on/for/to] [CANONICAL_SUBJECT if applicable]"
    - Normalize synonyms/abbreviations/omissions to the canonical forms. Do not add extra qualifiers beyond the canonical forms.

5. Preserve Meaning, Normalize Terms
    - Preserve the respondent’s intended meaning.
    - Use their own words where possible but normalize key terms to a consistent canonical form derived from the survey question’s primary subject(s).
        - Replace synonyms, abbreviations, or omitted references with the canonical form.
        - Apply this uniformly to all extracted ideas from that response. 
        - Example: If the question is about “electric vehicles” and the respondent says “cars” or “EVs,” standardize to “electric vehicles.”
    - Do not change sentiment or tone during normalization.

6. Include Implicit Ideas
    - Capture both explicit statements and ideas that are clearly implied by the response.

7. Deidentification
    - Replace personal names with [PERSON].
    - Use gender-neutral pronouns (they/them/their) for individuals.
    - Keep role descriptors (manager, teacher, etc.) when relevant.

8. Edge Cases
    - If the response is empty, irrelevant, or “N/A”: return an empty array [].
    - If the response is off-topic: extract ideas anyway but note they may be off-topic.
    - If only one idea is present: return it in a single-item array.
</instructions>

<output_format>
Return the extracted ideas as a JSON array. Each item should include:
- "respondent_id": exactly as provided
- "idea_id": a string number ("1", "2", etc.)
- "idea": the descriptive phrase in {language}, normalized and phrased using the chosen template consistently for the entire response.
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
    
Begin your analysis now and return ONLY the JSON array in {language}.
"""


# =============================================================================
# STEP 7:  CODEBOOK GENERATION : speculative codes
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
# STEP 7:  CODEBOOK GENERATION : 4 promt chain
# =============================================================================

CLUSTER_SUMMARY_PROMPT = """
You are a {language} qualitative researcher trained in thematic analysis by Braun & Clarke.
Your task is to interpretively construct a theme or themes from descriptive codes that have been derived from written responses to an open-ended survey question.

<inputs>
Cluster ID: {cluster_id}

Research question:
"{survey_question}"

Cluster of descriptive codes to analyze:
{cluster_text}
</inputs>

Conceptual guidance:
<conceptual_guidance>
- Each theme must be built around exactly one Central Organizing Concept (COC), underpinned by a shared pattern of meaning across multiple codes.
- A shared pattern of meaning reflects a common viewpoint, rationale, or concern that emerges across different codes and helps to make sense of the responses in relation to the research question.
- The COC is the core analytic idea that holds the pattern together. It should capture what the theme is really about — not just what the descriptive codes say on the surface, but how and why they matter in the context of the research question.
- All codes within a theme should relate clearly and meaningfully to the COC, and the theme must not include codes that do not fit this conceptual unity.
- If a single coherent COC cannot be identified, the cluster should be split into multiple COCs, each with its own shared pattern of meaning.
- If you cannot summarize the theme in a single sentence around a COC, you probably have more than one theme.
</conceptual_guidance>


Follow these steps exactly and in order. Do not skip or reorder any step.
- Always use your analytical judgment and reflexivity
- Themes are not discovered in the data — they are actively constructed.

1. Understand the context:
   - Read the research question carefully. 
   - Review all descriptive codes thoroughly.
   - Consider the broader meaning of codes in light of the resarch question

2. Identify number of COCs
   - Assess whether all descriptive codes can be organized around one coherent COC, following <conceptual_guidance>.
   - If you can identify one clear COC that unifies the cluster, proceed with that COC.
   - If the cluster contains multiple distinct rationales, viewpoints, or concerns that cannot be summarized in a single sentence, identify multiple COCs instead.
   - Avoid vague or overly general terms. COCs must be interpretive, specific, and explanatory.

3. Filter COCs:
   - Exclude all potential COCs that rest on a single code (singletons).
   - Exclude all potential COCs that lack conceptual overlap between multiple codes.
   - Only retain COCs that reflect a shared pattern of meaning across at least two codes.

4. Document your analysis:
  - Clearly state how many COCs you identified.
  - If only one COC is retained, explain why it is sufficient to account for the shared meaning across the cluster.
  - If multiple COCs are retained, justify why it is not possible to work with a single COC.
  - For each COC, explain how it helps to answer the research question.
  - Support your explanation by referencing the descriptive codes that exemplify each COC.

6. Construct themes:
   - Create exactly one theme per COC.
   - A valid theme MUST:
     - Express a clear analytic idea, not just a topic.
     - Explain how and why the responses hang together around a shared meaning.
     - Be grounded in the data but go beyond mere description.
   - A theme MUST NOT be:
     - A raw list of responses.
     - A purely descriptive label.
     - A speculation beyond the data.
     - A mix of unrelated or conflicting ideas.
 
7. Create theme labels:
   - Each label must:
 	- contain ≤ 10 words.
 	- Express exactly one unifying analytic idea.
   	- State the central organizing concept.
   - Labels must go beyond surface-level categories (e.g., not just "lack of attention").
   
8. Write theme descriptions:
   - each description must 
	- contain ≤ 30 words words.
   	- State the strongest shared pattern of meaning.
   	- Explain how the pattern relates to the research question.

Checklist (before finalizing)
[ ] Invalid patterns with single codes or codes without conceptual overlap are excluded
[ ] Label is a noun phrase
[ ] Label expresses only one concept (no combined ideas)
[ ] Description does not repeat the canonical subjects already present in the question
[ ] Description does not mention canonical actors implied by the research setting (e.g. “respondents” or “target group”)
[ ] Each theme can be summarized in one coherent sentence around a COC

Output instructions:
- Return your output as a valid JSON dictionary with the following structure:

{{
  "exact cluster_id as string here": {{
    "analysis": "provide your analysis here in {language}",
    "extracted_themes": [
      {{
        "theme_id": [number],
        "theme_label": "[theme label in ≤10 words, in {language}]",
        "theme_description": "[label description in ≤30 words, in {language}]"
      }}
      // add more theme objects here if needed
    ]
  }}
}}

Critical requirements:
- Output must ONLY be valid JSON — no extra text before or after.
"""


CANDIDATE_CODE_SELECTION_PROMPT = """
You are a {language} qualitative analyst specializing in matching descriptive codes from a codebook to theme(s) identified in response patterns to an open-ended survey question. 
Your task is to return all existing codes from the codebook that meaningfully correspond to the presented themes.

First, carefully review the following survey question:

<survey_question>
{survey_question}
</survey_question>

Next, examine the theme name(s) and theme descriptions(s):

<themes>
{cluster_summary}
</themes>

Now, review the existing codebook:

<existing_codebook>
{code_text}
</existing_codebook>

When matching codes to theme(s), follow these guidelines:

1. Review EACH theme in <themes> carefully in relation to codes in <existing_codebook>.
2. Match based primarily on the theme NAME; use the DESCRIPTION as supporting context only.
3. Match on semantic meaning, not word overlap. Focus on meaning, scope, and level of abstraction in the context of the survey question. Ignore superficial matches.
4. Include only codes with clear conceptual overlap. Do not force matches.
5. Preserve codebook integrity. Copy code names and definitions exactly as provided, including spelling, capitalization, and punctuation. Do not add, remove, or alter any fields or wording.
6. If a code matches multiple themes, include it only once in the final output.


Your output should be in the following JSON format, strictly in {language}:

[
  {{
    "code": "exact name of existing code A",
    "definition": "exact definition of existing code A"
  }},
  {{
    "code": "exact name of existing code B",
    "definition": "exact definition of existing code B"
  }},
  ...
  {{
    "code": "exact name of existing code N",
    "definition": "exact definition of existing code N"
  }}
]

Critical requirements:
- You may return ZERO, ONE, or MULTIPLE codes, depending on theme relevance.
- If no codes match, return an empty JSON array: [].
- Output must be a SINGLE JSON array combining matches across ALL theme statements (no per-theme lists).
- Do not create, remove, or modify codes.
- Do not duplicate codes if they apply to multiple themes.
- Output ONLY the JSON array — no explanation, headers, or extra text.
- Each object must include ONLY the fields: "code" and "definition".
- No comments, no trailing commas, no additional fields.
- All objects must be unique by code name.
"""

CODE_GENERATION_PROMPT = """
You are a {language} codebook curator who is responsible for coding themes expressed in survey responses.
Your task is to analyze these themes and decide whether to use existing codes, modify them, or create new ones. 
Your goal is to integrate new insights into the codebook while avoiding redundancy and ensuring conceptual clarity.

Here are the inputs you will be working with:
<input>
1) Survey question
<survey_question>
{survey_question}
</survey_question>

2) Themes expressed survey responses
<themes>
{cluster_summary}
</themes>

3) Existing codes in the codebook
<existing_codes>
{candidate_codes}
</existing_codes>

Note:
- Prioritize semantic alignment between the THEME NAME and the CODE LABEL.
- Use the THEME DESCRIPTION and CODE DESCRIPTION only as supporting context to confirm scope and clarify meaning.
</input>

DECISION RULES:

Your job is to decide for each theme whether to:
- USE an existing code as is,
- MODIFY an existing code with a minimal and justified change,
- or CREATE a new code.

Use the following thresholds:

A) If the theme NAME and a candidate code are at the same abstraction level (e.g., both are specific behaviors or both are mid-level categories):
- USE if the existing code covers ≥90% of the theme's meaning.
- MODIFY if the existing code covers ≥70% and <90%, and a minimal change to the definition resolves the gap.
- CREATE if the existing code covers <70%, or if the gap cannot be resolved with a minimal change.

B) If the theme NAME and all candidate codes are at different abstraction levels (e.g., one is broader or narrower than the other):
- CREATE by default.
- Only USE if a code covers 95% of the theme's meaning and using it does not distort the structure.

Tie-breakers:
- If multiple codes qualify at the same threshold, choose the most general, well-scoped code that avoids semantic overlap or confusion.

Failsafes:
- Never broaden a code beyond its natural meaning.
- If in doubt between MODIFY and CREATE, choose CREATE.

LABELLING RULES:
    
When creating or modifying codes:

- Labels: ≤10 words; atomic; no compound structures; avoid "and," "or," "with," "including," "/," "&," ",:;–-".
- Do not repeat the actor or subject from the survey question.
- Allowed forms:
  • Noun phrase: <adjective(s)> <noun>
  • Imperative verb + object
  • Infinitive verb + object
- Definitions: ≤25 words; operational, observable, aligned with the label; avoid vague or interpretive language.

OUTPUT:
    
Return ONLY valid JSON, in {language}. No extra text. No trailing commas. Each theme maps to exactly one decision. The justification must explain both the abstraction relationship and the coverage rule applied.

Required schema:
<output_format>
{{
  "coding_decisions": [
    {{
      "theme_number": 1,
      "theme_name": "Exact name of the theme from <themes>",
      "decision": "Must be exactly one of: 'use', 'modify', or 'create'",
      "final_code_label": "label of the code to be used/modified/created in {language}",
      "final_code_definition": "≤25 words, operational definition in {language}",
      "source_code": "exact name of reused/modified existing code, or null if new",
      "justification": "Single sentence stating abstraction relationship and coverage rule applied, in {language}"
    }}
    /* Repeat this object structure for each subsequent theme, incrementing theme_number by 1 */
  ]
}}
</output_format>

Begin your analysis now and provide the final JSON in the required format. Ensure that your output is valid JSON and follows the exact schema provided above.
"""

VALIDATION_PROMPT = """
You are a {language} qualitative data analyst specializing in codebook validation. Your task is to review coding recommendations for theme statements and finalize consistent, atomic, parsimonious codes for the codebook. All outputs must comply with the validation criteria provided.

Here are the inputs you will be working with:
<input>
1. Survey question:
<survey_question>
{survey_question}
</survey_question>

2. Themes to code:
<themes_to_code>
{cluster_summary}

Note:
- Prioritize the THEME NAME; Use the THEME DESCRIPTION only as supporting context to confirm scope and clarify meaning.
</themes_to_code>

3. Coding recommendation:
<coding_recommendation>
{step3_recommendation}
</coding_recommendation>
</input>

Follow these step-by-step instructions:

1. Carefully read the <survey_question>, <themes_to_code>, and <coding_recommendation>.
2. For each theme in the coding_recommendation:
   a. Review the proposed code and definition against the following validation criteria.
    - Semantic fit: Code must capture the theme's meaning and scope.
    - Atomicity:
        • Code label and definition must express one idea only — no compounds or merged themes.
        • Forbidden punctuation: "/", "&", "+", ",", ";", ":", "-", "–" (except in lexicalized words).
        • At most ONE main action (verb).
    - Naming rules:
        • ≤10 words, no canonical subject from the survey question, no actor expected to act.
        • Allowed syntactic forms:
            • Noun phrase: <adjective(s)> <noun>
            • Imperative verb + object: <verb> <object>
            • Infinitive form: <infinitive verb> + <object>
    - Definition rules:
        • ≤25 words, operational, observable, grounded in responses.
        • No vagueness, compound structures, or interpretive abstractions.
   b. Decide whether to APPROVE or REJECT the recommendation based on your review.
   c. If APPROVE:
      - Ensure the code fully complies with all criteria.
      - If any adjustments are needed, make minor refinements to ensure full compliance.
   d. If REJECT:
      - Identify the specific issues that led to rejection.
      - Rewrite the code and/or definition to address these issues and ensure full compliance with all criteria.
   e. Draft a validated_code object with the final code and definition.
   f. Perform a self-audit using the Atomicity Enforcement Checklist:
      - Conjunctions: label contains no coordinating conjunctions (e.g., "and", "or", "en", "of", "met", "en/of").
      - List markers & punctuation: label contains none of "/", "&", "+", ",", ";", ":", " - ", " – " (hyphens allowed only in lexicalized words).
      - Single idea: label and definition each express one idea (no compounds or merged themes).
      - Single action: definition contains at most one main action (verb).
      - Length: label ≤10 words; definition ≤25 words.
   g. If any item in the self-audit FAILS, rewrite the label/definition until all items PASS.
   h. Write a decision_rationale explaining your decision and including the self-audit summary.
   i. Add the theme_number, original_recommendation, decision, decision_rationale, and validated_code to the code_validations array.
3. After processing all themes, review the entire output to ensure consistency and compliance with all requirements.
4. Format the output as valid JSON in the specified structure.

Provide your output in strict JSON format, in {language} , ensuring all validation criteria are applied. Use the following structure:

{{
  "code_validations": [
    {{
      "theme_number": [exact theme number in <coding_recommendation>],
      "theme_name": "Exact name theme in <themes_to_code> ",
      "original_recommendation": {{
        "code": "Exact code label recommended",
        "definition": "Exact label definition recommended"
      }},
      "decision": "APPROVE | REJECT",
      "decision_rationale": "Include a brief explanation AND the self-audit summary in this format: Conjunctions=PASS/FAIL; Punctuation=PASS/FAIL; SingleIdea=PASS/FAIL; SingleAction=PASS/FAIL; NameLen=PASS/FAIL; DefLen=PASS/FAIL.",
      "validated_code": {{
        "code": "final validated label (atomic, rule-compliant, ≤10 words)",
        "definition": "final validated definition (≤25 words, operational, grounded)"
      }}
    }}
    /* Repeat this object structure for each subsequent theme, incrementing theme_number by 1 */
  ]
}}

Critical requirements:
- Always return a validated_code object for every theme_number, regardless of decision.
- Output ONLY valid JSON — no other text.
- All fields must be in {language}.
"""

# =============================================================================
# STEP 8: THEME IDENTIFICATION  
# =============================================================================

THEME_IDENTIFICATION_PROMPT = """
You are a {language} language expert and qualitative researcher specializing in thematic analysis using Braun & Clarke (2006).

Your task is to evaluate a cluster of descriptive codes and decide whether they should:
1. Be grouped under an existing theme,
2. Be assigned to a revised version of an existing theme,
3. Form a new, single theme, or
4. Be split into multiple distinct themes, or
5. Be rejected as too mixed or incoherent to group meaningfully.

You may only group codes if they conceptually share a single, ATOMIC idea and express a consistent sentiment (all positive, all negative, or all neutral).

Follow these steps carefully:

### STEP 1: Review the data

<survey_question>
{survey_question}
</survey_question>

<existing_themes>
{existing_themes_text}
</existing_themes>

<clustered_codes>
{codes_text}
</clustered_codes>

This cluster contains {codes_count} codes.

### STEP 2: Analyze the cluster

Ask yourself:

- **Conceptual Coherence**: Do all codes refer to the same core idea or experience?
- **Atomicity**: Is that idea *one single concept* (not a combination like “price and usability”)?
- **Sentiment Consistency**: Do the codes express a similar tone (all positive, all negative, or all neutral)?
- **Thematic Sentence Test**: Can every code in the cluster complete the phrase:  
  **“This is about…”** with the same concept **and** sentiment?

### STEP 3: Make a decision

Follow these guidelines:
- **Use Existing Theme**  
  If all codes fit well within a single existing theme in both concept and sentiment.

- **Revise Existing Theme**  
  If the cluster mostly fits but requires a more specific or clearer version of an existing theme.

- **Create New Theme**  
  If there’s strong conceptual and sentiment unity, but no suitable theme currently exists.

- **Split into Multiple Themes**  
  If the cluster contains:
    - Two or more distinct conceptual groups (e.g. “price concerns” vs. “design preferences”)
    - The same concept expressed with different sentiments (e.g. “trust in staff” vs. “distrust in staff”)

- **Reject Mixed Cluster**  
  If codes are too diverse or vague to meaningfully group.


### STEP 4: Ensure quality of themes

Each proposed theme must be:

- **Atomic**: One idea only — no compound or vague themes like "quality and service"
- **Concise**: 2–5 words for the name
- **Narrative**: Describes a meaningful pattern or concept
- **Distinct**: No overlap with other themes
- **Language**: Use {language} for all theme names and descriptions


### STEP 5: Return output in this exact JSON format:

{{
  "decision": "create_single_theme | use_existing_theme | split_into_multiple_themes | reject_mixed_cluster",
  "themes": [
    {{
      "theme_name": "[Theme name in {language}]",
      "theme_description": "[Brief conceptual description in {language}]",
      "assigned_codes": [1, 3, 5],
      "confidence": "high | medium | low",
      "is_existing": false
    }}
  ],
  "existing_theme_used": "[Exact name from above, or null]",
  "rationale": "[Detailed explanation of decision, including conceptual grouping logic if splitting]"
}}

Final Reminders:
- DO NOT invent vague or overly broad themes.
- NEVER group together different sentiments or unrelated ideas.
- ALWAYS return a valid JSON object as shown above — nothing else.
- Theme name and description must be in {language}.
"""

THEME_IDENTIFICATION_PROMPT = """
You are a {language} language expert and qualitative researcher specializing in thematic analysis using Braun & Clarke (2006) methodology. Your task is to analyze a cluster of codes and recommend whether to use an existing theme, revise one, or create a new theme — but only if the codes conceptually share an overarching, unifying and ATOMIC theme.
Follow these steps carefully:

1. Review the survey question that generated the codes:
<survey_question>
{survey_question}
</survey_question>


2. Examine the existing themes in the codebook:
<existing_themes>
{existing_themes_text}
</existing_themes>

3. Analyze the following cluster of {{CODES_COUNT}} codes that may require a new or revised theme:
<clustered_codes>
{codes_text}
</clustered_codes>


4. Check whether the codes in this cluster can be grouped together. Ask yourself:
   - Do all the codes in the cluster conceptually share an overarching, unifying and ATOMIC idea, concept or theme?
   - Do all the codes express similar sentiment (all positive, all negative, or all neutral)?
   - Can they all complete the sentence: "This is about…" with the same concept AND sentiment?

5. After this analysis, decide whether to create a new theme, modify one, or use an existing theme.

Decision Guidelines:
- Use existing theme if the cluster fits well with an existing theme (concept + sentiment).
- Create a new single theme if existing themes don't suffice AND the cluster has a shared, coherent theme.
- Split into multiple themes if either conceptual unity OR sentiment consistency is missing.

Critical:   
- Never combine unrelated subthemes (e.g. "Concerns about price and product design") into one theme.
- Never combine different sentiments about the same concept (e.g. "positive communication experiences" + "negative communication problems") into one theme.

When Splitting Clusters:
1. Identify distinct conceptual groups within the cluster 
2. Identify sentiment groups - codes clearly expressing different sentiment (positive vs negative) 
3. Prioritize conceptual splits first, then sentiment splits within concepts if needed
4. Assign each code to its most appropriate group by code number (concept + sentiment)
5. Create separate themes for each group, ensuring each is atomic
6. Verify each group has consistent concept AND sentiment

New Themes Must Be:
- ATOMIC: One idea only — no compound concepts with "and", "with", or "including"
- CONCISE: Theme labels must be 2–5 words
- NARRATIVE: Each theme should tell a meaningful story about the data
- DISTINCT: No major conceptual overlap between themes (mutual exclusivity)

Output your analysis and decision in the following JSON format:
{{
  "decision": "create_single_theme | use_existing_theme | split_into_multiple_themes | reject_mixed_cluster",
  "themes": [
    {{
      "theme_name": "[Theme name in {language}]",
      "theme_description": "[Brief conceptual description in {language}]",
      "assigned_codes": [1, 3, 5],
      "confidence": "high | medium | low",
      "is_existing": false
    }}
  ],
  "existing_theme_used": "[Exact name from above, or null]",
  "rationale": "[Detailed explanation of decision, including conceptual grouping logic if splitting]"
}}

Important reminders:
- Theme name and description must be in {{LANGUAGE}}
- If using an existing theme, copy the exact name from the list above
- Focus on conceptual fit, not surface similarity or keyword overlap
- Return ONLY the JSON object as your final output
"""

ASSIGN_MISCELLANEOUS_PROMPT = """
You are {language} language expert and a qualitative researcher specializing in thematic analysis following Braun & Clarke (2006) methodology.
Your task is to find the BEST placement for a single code that was classified as noise/outlier by the clustering algorithm.

SURVEY QUESTION:
{survey_question}

EXISTING THEMES:
{existing_themes_text}

CODE TO ANALYZE:
Code {code_number}: {code_name}
Definition: {definition}

INSTRUCTIONS:
- Examine this single code carefully in the context of the survey question
- Consider which existing theme this code fits BEST conceptually
- Focus on the underlying meaning and purpose of the code
- If the code truly doesn't fit any existing theme, you may recommend keeping it miscellaneous
- Prioritize conceptual fit over surface-level keyword matching

DECISION CRITERIA:
1. Does this code share the same underlying concept as an existing theme?
2. Would including this code strengthen or weaken the theme's coherence?
3. Does the code address the same aspect of the survey question as an existing theme?

OUTPUT FORMAT (JSON):
{{
  "decision": "assign|miscellaneous",
  "target_theme": "[exact theme name from list above, or null if miscellaneous]",
  "confidence": "high|medium|low",
  "rationale": "[detailed explanation of why this code belongs with this theme or remains miscellaneous]"
}}

CRITICAL: Focus ONLY on this single code. Give it your complete analytical attention.

Return ONLY the JSON object with all content in {language}."""

# =============================================================================
# STEP 9: CODE ASSIGNMENT
# =============================================================================

CODE_ASSIGNMENT_PROMPT = """
You are a {language} language expert in qualitative data analysis, specializing in applying codebooks to open-ended survey responses. Your task is to assign the single most appropriate code from a focused list of 5 candidate codes to a specific response segment.

First, review the original survey question:
<survey_question>
{var_lab}
</survey_question>

Next, examine the response segment you need to analyze:
<idea_to_analyze>
Idea ID: {idea_id}
Idea Text: {idea_text}
</idea_to_analyze>

Now, review the 5 candidate codes and their descriptions:
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
- Excellent (0.9–1.0): The idea directly expresses the code definition.
- Good (0.7–0.8): Strong match, minor interpretation needed.
- Moderate (0.5–0.6): Somewhat related, but requires reasonable interpretation.
- Poor (0.3–0.4): Weak fit, conceptually strained but closest available.
- Very Poor (0.0–0.2): Barely applicable, forced fit due to lack of better options.

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

# # =============================================================================
# # DEDUPLICATION PROMPT
# # =============================================================================

# DEDUPLICATION_PROMPT = """
# You are a {language} qualitative data analyst creating the MOST EFFICIENT, MINIMAL codebook for analyzing this research question.

# Your mission: Create a codebook where every code is ESSENTIAL and IRREPLACEABLE for analysis.
# Think like a researcher analyzing 1000+ responses - codes that are too similar create confusion and inconsistent coding.

# ────────────────────────────────────────
# SURVEY CONTEXT
# ────────────────────────────────────────
# Survey Question: {survey_question}

# Language: {language}

# ────────────────────────────────────────
# CODES TO ANALYZE
# ────────────────────────────────────────
# {codes_batch}

# ────────────────────────────────────────
# EFFICIENCY-FOCUSED PRINCIPLES
# ────────────────────────────────────────
# Create a codebook optimized for:

# 🎯 RESEARCH EFFICIENCY: Minimal codes that capture maximum analytical insight
# 🧠 CODER CLARITY: Codes so distinct that human coders never hesitate between options
# 📊 ANALYTICAL POWER: Each code must justify its separate existence for this research question
# ⚡ PRACTICAL USE: Designed for coding hundreds of real survey responses

# ────────────────────────────────────────
# AGGRESSIVE MERGING MANDATE
# ────────────────────────────────────────
# You are tasked with creating the MOST CONDENSED possible codebook.

# MERGE codes that express the same core concept, even if they differ in:
# - Specific wording
# - Minor details  
# - Slight emphasis differences

# Default to MERGING. Only keep codes separate if they represent fundamentally different concepts that cannot be combined.

# ────────────────────────────────────────
# MERGE DECISION PROCESS
# ────────────────────────────────────────
# For ANY two codes that seem related:

# 1. Do they address the same basic respondent concern? → MERGE
# 2. Would survey responses fit under both codes? → MERGE  
# 3. Is the difference mainly in wording, not meaning? → MERGE

# ONLY keep separate if codes represent completely different themes that cannot be logically combined.

# ────────────────────────────────────────
# OUTPUT FORMAT (JSON only, no other text)
# ────────────────────────────────────────
# {{
#   "merge_decisions": [
#     {{
#       "codes_to_merge": ["exact code name 1", "exact code name 2"],
#       "final_code_name": "best merged code name",
#       "final_definition": "clear combined definition in 1-2 sentences",
#       "justification": "why these codes are semantically identical for this survey"
#     }}
#   ],
#   "codes_to_keep_unchanged": ["exact code name 3", "exact code name 4", ...]
# }}

# Rules:
# - All field values must be in {language}
# - Use exact code names as they appear above
# - Only merge codes that are truly duplicates
# - If no duplicates found, return empty merge_decisions array
# - Output ONLY valid JSON, no other text
# """



