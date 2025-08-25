
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
You are a {language} qualitative analyst. Convert a cluster of responses into atomic, code-ready proposals.

────────────────────────────────────────
INPUTS  (XML blocks will be interpolated)
────────────────────────────────────────
<survey_question>
{survey_question}
</survey_question>

<cluster_text>
{cluster_text}
</cluster_text>

────────────────────────────────────────
GOAL
────────────────────────────────────────
Ideally produce ONE thematic code proposal that best captures the shared response pattern in the cluster, ready for a codebook.

────────────────────────────────────────
QUALITY CRITERIA
────────────────────────────────────────
- ATOMIC: one idea per code (no compounding).
- CONCISE NAME: short phrase (max 10 words) that directly address the survey question.
- THEME NAME RULE:
    - Must NOT contain the canonical subject (main product/service/topic from survey question) or actor (who is expected to act, if applicable).
    - Must be one of these allowed syntactic patterns:
      • Noun phrase: <adjective(s)> <noun>  
      • Imperative verb + object: <verb> <object>  
      • Infinitive form: <infinitive  verb> + <object>  
- OPERATIONAL SUMMARY: ≤25 words, observable, precise, no interpretation.
- PARSIMONIOUS: prefer one unified proposal; split only if truly necessary.
- GROUNDED: use wording present in responses; minimal generalisation by combining/shortening existing phrases only (no new synonyms/jargon).

Atomicity enforcement (language-agnostic):
- The theme_name must not contain coordinating conjunctions or list markers in {language}.
- Forbid list/coordination punctuation: "/", "&", ",", ";", ":", "-", "–" (hyphens allowed only inside a single lexicalized word, not to join ideas).
- Max ONE main action (verb). If two actions appear, keep only the single core action.
- Allowed forms for theme_name (choose ONE):
  • Noun phrase (e.g., "<concept>"),
  • Imperative verb + object (e.g., "<verb> <object>"),
  • Single-action infinitive/gerund phrase.

────────────────────────────────────────
PROCESS
────────────────────────────────────────
1) Coherence gate: first attempt to identify a dominant single unifying theme covering the responses.
2) If a single theme is not honest/possible, output at most TWO atomic proposals.
   - Only split when there are clearly distinct ideas.
   - Ignore singletons/outliers; avoid splitting for one-off mentions.
3) Each proposal must be directly traceable to the literal wording in the responses.
4) Atomicity self-check for theme_name:
   - If theme_name contains any coordinator/list marker (in {language}) or any of "/", "&", ",", ";", ":", "-", "–", or more than one main verb, REWRITE to a single permitted form.
   - Focus on the shared theme, not on subthemes.
5) Deduplicate near-duplicates (keep the shortest literal phrase). theme_id must be sequential starting at 1.

────────────────────────────────────────
OUTPUT FORMAT (strict JSON, in {language})
────────────────────────────────────────
[
  {{
    "theme_id": 1,
    "theme_name": "<short phrase in {language} clarifying the theme in light of the survey question>",
    "summary": "<≤25 words describing the contents of the cluster in {language}, directly derivable from the cluster text and fitting the theme>"
  }}
  // Continue for each (sub)theme in order
]

────────────────────────────────────────
CRITICAL REQUIREMENTS
────────────────────────────────────────
- Output ONLY the JSON array — no extra text before or after.
- No comments, no trailing commas.
- All text fields must be in {language}.
- Valid JSON syntax only.
- Start your response immediately with '[' - no explanation text.
- End your response with ']' - no additional text.
- Do not include any commentary or explanation outside the JSON.
"""


CANDIDATE_CODE_SELECTION_PROMPT = """
You are a {language} qualitative analyst.
For each proposed code, find matches in the existing codebook.

────────────────────────────────────────
INPUTS  (XML blocks will be interpolated)
────────────────────────────────────────
<survey_question>
{survey_question}
</survey_question>

<code_proposals>
{cluster_summary}
</code_proposals>

<existing_codebook>
{code_text}
</existing_codebook>

────────────────────────────────────────
MATCHING GUIDANCE
────────────────────────────────────────
1) If more than one proposal is provided, review each separately but return a single, deduplicated list of matching codes.
2) Match on meaning, scope, and abstraction level in light of the survey question (do not rely on superficial word overlap).
3) Select only codes that cover the proposal substantially (≈≥60% of its meaning); prefer near-equivalents (≈≥90%) when available.
4) Be selective: exclude codes unrelated to the proposal; avoid near-duplicates and do not include both a parent and a redundant child code.
5) Be comprehensive: if a strong match exists, include it. If none exist, return an empty list.
6) Copy names/definitions EXACTLY as in the codebook (no edits).

────────────────────────────────────────
OUTPUT  (raw JSON, in {language})
────────────────────────────────────────

[
  {{
    "code": "exact same name of existing code 1",
    "definition": "exact same definition of existing code 1"
  }},
  {{
    "code": "exact same name of existing code 2",
    "definition": "exact same definition of existing code 2"
  }}
]

Repeat the above format for every code that matches the proposal.

IMPORTANT:
- You may select NONE, ONE, or MULTIPLE candidate codes per proposal 
- Do not create new codes or modify existing ones
- Output ONLY the JSON array - no other text

"""

CODE_GENERATION_PROMPT = """
You are a {language} qualitative codebook curator. Decide the final action for each proposed code using codebook matches.

────────────────────────────────────────
INPUTS  (XML blocks will be interpolated)
────────────────────────────────────────
<survey_question>
{survey_question}
</survey_question>

<code_proposals>
{cluster_summary}
</code_proposals>

<existing_codes>
{candidate_codes}
</existing_codes>

────────────────────────────────────────
CONCEPTUAL GUIDANCE
────────────────────────────────────────
Each code—existing or proposed—represents a theme expressed in responses to an open-ended survey question.
A theme is atomic: one coherent idea that cannot be meaningfully reduced.
First identify the theme each proposal addresses (use the proposal’s wording/level).

────────────────────────────────────────
DECISION RULES
────────────────────────────────────────
- use → some existing code covers ≥90% of the proposal (meaning & scope).
- modify → best existing covers 60–89%; refine name/definition to fit both the proposal and existing scope.
- create → no existing code ≥60%.

────────────────────────────────────────
CONSTRAINTS for any final code (modified or new)
────────────────────────────────────────
- Name: short phrase (max 10 words), directly addressing the survey question; atomic (no "and/with/including").
- Definition: ≤30 words, operational, precise, grounded in observed wording/behavior (no interpretation).
- Atomicity — Single idea only; no "and", "with", "including".
- Abstraction: match the proposal’s level; do not over-generalise.
- Parsimony & Exclusivity: prefer one code per theme; avoid overlap and redundancy with existing codes.
- Grounding: prefer minimal generalisation by combining/shortening phrases from the data; avoid new jargon/synonyms.

CODE NAME RULE:
- Must NOT contain the canonical subject (main product/service/topic from survey question) or actor (who is expected to act, if applicable).
- Must be one of these allowed syntactic patterns:
    • Noun phrase: <adjective(s)> <noun>  
    • Imperative verb + object: <verb> <object>  
    • Infinitive form: <infinitive  verb> + <object>  

ATOMICITY ENFORCEMENT (language-agnostic):
- The theme_name must not contain coordinating conjunctions or list markers in {language}.
- Forbid list/coordination punctuation: "/", "&", ",", ";", ":", "-", "–" (hyphens allowed only inside a single lexicalized word, not to join ideas).
- Max ONE main action (verb). If two actions appear, keep only the single core action.
- Allowed forms for theme_name (choose ONE):
  • Noun phrase (e.g., "<concept>"),
  • Imperative verb + object (e.g., "<verb> <object>"),
  • Single-action infinitive/gerund phrase.
    
───────────────────────────────────────
EDGE CASES
────────────────────────────────────────
- If multiple existing codes are ≥0.90, choose the most general, well-scoped one (parsimony).
- If no suitable existing codes, create exactly one new atomic code per theme.
- Only use multiple existing codes for one theme when they are non-overlapping and both are strictly necessary (justify).

────────────────────────────────────────
OUTPUT  (Strict JSON, in {language})
────────────────────────────────────────
{{
  "cluster_analysis": {{
    "number_of_themes": integer,
    "theme_descriptions": [
      "Brief description of theme 1",
      "Brief description of theme 2 (if present)"
    ]
  }},
  "coding_decisions": [
    {{
      "theme_number": 1,
      "theme_description": "short description of this theme",
      "decision": "use | modify | create",
      "action_details": {{
        "codes_to_use": ["exact existing code names"],
        "codes_to_modify": "exact existing code name or null",
        "modified_code_name": "new name if modifying, else null",
        "modified_code_definition": "max 30 words if modifying, else null",
        "new_code_name": "name if creating new, else null",
        "new_code_definition": "max 30 words if creating new, else null"
      }},
      "justification": "brief reason for this decision"
    }},
    {{
      "theme_number": 2,
      "theme_description": "...",
      "decision": "...",
      "action_details": {{ ... }},
      "justification": "..."
    }}
  ],
  "overall_justification": "why these codes preserve atomicity and improve codebook quality"
}}

────────────────────────────────────────
GOOD DEFINITION EXAMPLES
────────────────────────────────────────
- "References to [specific limitation or constraint] affecting [process or outcome]."
- "Mentions of [positive or negative] changes in [behavior or practice]."
- "Expressions of [emotion or attitude] regarding [situation or process]."

WEAK DEFINITION EXAMPLES
- Compound: "References to [issue A] including [aspect 1], [aspect 2], and [aspect 3]"
- Vague: "Mentions of various [things] related to [topic]"
- Interpretive: "Underlying [abstract concept] manifesting in different ways"

IMPORTANT:
- All output fields must be in {language}.
- Output ONLY valid JSON (no extra text, no trailing commas, use straight quotes).
"""


VALIDATION_PROMPT = """
You are a {language} qualitative data analyst specializing in codebook development and evaluation.
You will assess your colleague's coding recommendations for a cluster that contains 1 or 2 themes summarizing ideas expressed in written survey responses.
Your task is to maintain a parsimonious codebook with clear, atomic code names (no compounding of subthemes).

────────────────────────────────────────
INPUTS  (XML blocks will be interpolated)
────────────────────────────────────────
<survey_question>
{survey_question}
</survey_question>

<themes_to_code>
{cluster_summary}
</themes_to_code>

<existing_codes_in_codebook>
{candidate_codes}
</existing_codes_in_codebook>

<coding_recommendation>
{step3_recommendation}
</coding_recommendation>

────────────────────────────────────────
EVALUATION GUIDANCE
────────────────────────────────────────
- Expect ONE themes in the cluster summary, but be ready for more themes.
- Assess EACH recommendation using these criteria:
  a) Semantic fit — Does the code accurately capture the theme’s meaning and scope?
  b) Atomicity — Single idea only; no "and", "with", "including".
  c) Parsimony — Prefer reuse of existing codes when ≥90% aligned; avoid redundant additions.
  d) Non-redundancy — Avoid overlap with existing codes and between recommended codes.
  e) Abstraction level — Match the theme’s level; avoid over-/under-generalisation.
  f) Grounding — Worded to be operational and observable; ≤30-word definition.

CODE NAME RULE:
- Must NOT contain the canonical subject (main product/service/topic from survey question) or actor (who is expected to act, if applicable).
- Must be one of these allowed syntactic patterns:
    • Noun phrase: <adjective(s)> <noun>  
    • Imperative verb + object: <verb> <object>  
    • Infinitive form: <infinitive  verb> + <object>      

ATOMICITY ENFORCEMENT (language-agnostic):
- The theme_name must not contain coordinating conjunctions or list markers in {language}.
- Forbid list/coordination punctuation: "/", "&", ",", ";", ":", "-", "–" (hyphens allowed only inside a single lexicalized word, not to join ideas).
- Max ONE main action (verb). If two actions appear, keep only the single core action.
- Allowed forms for theme_name (choose ONE):
  • Noun phrase (e.g., "<concept>"),
  • Imperative verb + object (e.g., "<verb> <object>"),
  • Single-action infinitive/gerund phrase.
    
────────────────────────────────────────
VALIDATION DECISION RULES
────────────────────────────────────────
For EACH theme/code recommendation:
- APPROVE — Necessary, atomic, name rule applied, correct abstraction level, well-justified, non-overlapping.
- REVISE — Generally valid but needs refinement (atomic name, rule name enforced, clearer ≤30-word operational definition, or tightened scope).
- REJECT — Unnecessary or redundant; replace with a single best existing code if available.

When REVISE or REJECT:
- Always provide a final validated code (applicatoin of name rule + ≤30-word operational definition).
- If REJECT due to a better existing code, set validated_code to that existing code’s exact name and definition.

Edge cases:
- If multiple codes were recommended for one theme, approve at most those that are strictly non-overlapping and necessary; otherwise consolidate to one.
- Use exact existing names/definitions when approving or substituting existing codes.

────────────────────────────────────────
OUTPUT  (raw JSON, no extra text, in {language})
────────────────────────────────────────
{{
  "theme_assessment": {{
    "number_of_themes_identified": integer,
    "theme_separation_valid": true/false,
    "theme_separation_reasoning": "are the themes truly distinct or should they be merged/split?"
  }},
  "code_validations": [
    {{
      "theme_number": 1,
      "theme_description": "what theme is being coded",
      "original_recommendation": "what was proposed",
      "evaluation": {{
        "semantic_fit": "assessment of how well the code captures the theme's meaning",
        "atomicity": "assessment of whether code represents single concept",
        "parsimony": "assessment of whether existing codes were properly considered",
        "redundancy": "assessment of whether code overlaps with existing codes"
      }},
      "decision": "APPROVE | REVISE | REJECT",
      "decision_rationale": "explanation",
      "validated_code": {{
        "code": "final code name with atomicity enforced and name rule applied",
        "definition": "final definition"
      }}
    }},
    {{
      "theme_number": 2,
      "theme_description": "...",
      "original_recommendation": "...",
      "evaluation": {{
        "semantic_fit": "...",
        "atomicity": "...",
        "parsimony": "...",
        "redundancy": "..."
      }},
      "decision": "...",
      "decision_rationale": "...",
      "validated_code": {{
        "code": "...",
        "definition": "..."
      }}
    }}
  ],
  "overall_validation": {{
    "all_themes_coded": true/false,
    "final_code_count": integer,
    "summary": "brief summary of the validation outcome"
  }}
}}

────────────────────────────────────────
CRITICAL FORMAT REQUIREMENTS
────────────────────────────────────────
- `validated_code` is always a single object with "code" and "definition".
- Validate EACH theme/code pair separately.
- Names must be short phrases of max words; definitions ≤30 words; no compounding.
- Use exact existing names/definitions when relevant.
- Output ONLY valid JSON, no other text.
- All output fields must be in {language}.
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
You are a {language} language expert in qualitative data analysis, specializing in applying codebooks to open-ended survey responses.

Your task is to assign the **single most appropriate code** from a focused list of 5 candidate codes to a specific response segment.

---

You are working with a **pre-selected set of 5 candidate codes** that are most likely to fit this response. This reduces ambiguity and improves accuracy. **You must assign exactly ONE code**, even if fit is low — choose the best available option based on semantic meaning.

---

Step 1: Review the original survey question:
<survey_question>
{var_lab}
</survey_question>

Step 2: Examine the response segment:
<idea_to_analyze>
Idea ID: {idea_id}
Idea Text: {idea_text}
</idea_to_analyze>

Step 3: Review the 5 candidate codes and their descriptions:
<candidate_codes>
{candidate_codes}
</candidate_codes>

---

**Assignment Instructions**:

1. **Select the best fitting code**:
   - Prioritize **exact conceptual matches** based on meaning.
   - Do **not** rely on surface keywords — base your choice on **semantic alignment** with the code's definition.

2. **Rate the strength of the fit** using this scale:
   - **Excellent (0.9–1.0)** – The idea directly expresses the code definition.
   - **Good (0.7–0.8)** – Strong match, minor interpretation needed.
   - **Moderate (0.5–0.6)** – Somewhat related, but requires reasonable interpretation.
   - **Poor (0.3–0.4)** – Weak fit, conceptually strained but closest available.
   - **Very Poor (0.0–0.2)** – Barely applicable, forced fit due to lack of better options.

---

**Return your response in the following JSON format:**
<output_format>
{{
  "idea_id": "{idea_id}",
  "idea": "{idea_text}",
  "assigned_codes": ["SINGLE_CODE_NAME"],
  "assignment_confidence": CONFIDENCE_SCORE,
  "assignment_rationale": "Brief explanation of the conceptual match (in {language})"
}}
</output_format>

---

**Critical Requirements**:
- Use **exact code names** as provided
- Assign **one and only one** code per response
- The **confidence score must reflect conceptual fit**, not how likely the model feels
- The **rationale must explain the semantic connection** to the code definition
- Return **ONLY the JSON object** in {language}

Begin the code assignment now.
"""

# =============================================================================
# DEDUPLICATION PROMPT
# =============================================================================

DEDUPLICATION_PROMPT = """
You are a {language} qualitative data analyst creating the MOST EFFICIENT, MINIMAL codebook for analyzing this research question.

Your mission: Create a codebook where every code is ESSENTIAL and IRREPLACEABLE for analysis.
Think like a researcher analyzing 1000+ responses - codes that are too similar create confusion and inconsistent coding.

────────────────────────────────────────
SURVEY CONTEXT
────────────────────────────────────────
Survey Question: {survey_question}

Language: {language}

────────────────────────────────────────
CODES TO ANALYZE
────────────────────────────────────────
{codes_batch}

────────────────────────────────────────
EFFICIENCY-FOCUSED PRINCIPLES
────────────────────────────────────────
Create a codebook optimized for:

🎯 RESEARCH EFFICIENCY: Minimal codes that capture maximum analytical insight
🧠 CODER CLARITY: Codes so distinct that human coders never hesitate between options
📊 ANALYTICAL POWER: Each code must justify its separate existence for this research question
⚡ PRACTICAL USE: Designed for coding hundreds of real survey responses

────────────────────────────────────────
AGGRESSIVE MERGING MANDATE
────────────────────────────────────────
You are tasked with creating the MOST CONDENSED possible codebook.

MERGE codes that express the same core concept, even if they differ in:
- Specific wording
- Minor details  
- Slight emphasis differences

Default to MERGING. Only keep codes separate if they represent fundamentally different concepts that cannot be combined.

────────────────────────────────────────
MERGE DECISION PROCESS
────────────────────────────────────────
For ANY two codes that seem related:

1. Do they address the same basic respondent concern? → MERGE
2. Would survey responses fit under both codes? → MERGE  
3. Is the difference mainly in wording, not meaning? → MERGE

ONLY keep separate if codes represent completely different themes that cannot be logically combined.

────────────────────────────────────────
OUTPUT FORMAT (JSON only, no other text)
────────────────────────────────────────
{{
  "merge_decisions": [
    {{
      "codes_to_merge": ["exact code name 1", "exact code name 2"],
      "final_code_name": "best merged code name",
      "final_definition": "clear combined definition in 1-2 sentences",
      "justification": "why these codes are semantically identical for this survey"
    }}
  ],
  "codes_to_keep_unchanged": ["exact code name 3", "exact code name 4", ...]
}}

Rules:
- All field values must be in {language}
- Use exact code names as they appear above
- Only merge codes that are truly duplicates
- If no duplicates found, return empty merge_decisions array
- Output ONLY valid JSON, no other text
"""



