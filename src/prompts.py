
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
You are a {language} language expert in analyzing written responses to open-ended questions collected in surveys. 
Your task is to extract ALL distinct ideas expressed in response to the following survey question: 
        
<survey_question>
{var_lab}
</survey_question>

Here is the respondent information and their response:
<respondent_info>
Respondent ID: {respondent_id}
Written response: {response}
</respondent_info>

Please follow these instructions carefully:

1. **Thorough Analysis**: Read the response multiple times to ensure no ideas are missed.

2. **Idea Identification**: Extract ALL distinct ideas that directly answer or relate to the survey question. An "idea" is:
   - A single, complete thought or opinion
   - A specific action, behavior, or experience mentioned
   - A reason, cause, or explanation given
   - An emotion, attitude, or evaluation expressed

3. **Extraction Guidelines**:
   - Keep each idea atomic - each idea must capture ONE concept only - no compound ideas with "and", "including", "with"
   - Preserve the respondent's intended meaning
   - Use the respondent's own words where possible, but clarify if ambiguous
   - Include both explicit statements and clearly implied ideas
   - If the response contains "and" or "but", check if these connect separate ideas

4. **Descriptive Phrases**: For each idea, create a short phrase that:
   - Captures the essence in context of the survey question
   - Is self-contained and understandable without the full response
   - Maintains the sentiment/tone of the original (positive, negative, neutral)
   - Uses {language} language

5. **Deidentification Requirements**:
   - Replace all personal names with [PERSON]
   - Replace organization/company names with [ORGANIZATION] 
   - Replace specific locations with [LOCATION] if identifying
   - Use gender-neutral pronouns (they/them/their) for all individuals
   - Preserve role descriptors (manager, colleague, teacher) as they may be analytically relevant

6. **Edge Cases**:
   - If response is empty, irrelevant, or "N/A": return empty array []
   - If response doesn't answer the question: extract ideas anyway but note they may be off-topic
   - If response contains only one idea: still return as array with one item


Return the extracted ideas ideas as a JSON array. Each item should include:
- `"respondent_id"`: exactly as provided
- `"idea_id"`: a string number ("1", "2", etc.)
- `"idea"`: the descriptive phrase capturing the essence of the idea in {language}

Here's an example of the input and desired output format:
    
<example>
Survey question: "What aspects of your supervisor's performance stood out to you?"
Respondent ID: 123456789
Response: "Jared did a great job responding quickly to emails and turning in good work. However, he sometimes seemed overwhelmed when multiple projects came up at once."

Example output:
[
  {{
    "respondent_id": "123456789",
    "idea_id": "1", 
    "idea": "[PERSON] responded quickly to emails""
  }},
  {{
    "respondent_id": "123456789",
    "idea_id": "2",
    "idea": "[PERSON] turned in good work"
  }},
  {{
    "respondent_id": "123456789",
    "idea_id": "3",
    "idea": "[PERSON] seemed overwhelmed with multiple simultaneous projects"
  }}
]
</example>

Notice how:
- Names are replaced with [PERSON]
- Each idea is separate and atomic
- The contrasting sentiment ("However") creates a new idea
- Ideas preserve the original meaning and context

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
You are a {language} qualitative analyst who treats survey data as interconnected narratives. 

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
DEFINITIONS
────────────────────────────────────────
- "“Semantically coherent”: the responses can truthfully be summarised by one short phrase that is either explicitly present in the responses or can be formed by combining more general wording from the responses with any included specifics.
- "Mixed cluster": the responses contain multiple, distinct ideas that cannot honestly be captured under one phrase without adding new words or interpretations.

────────────────────────────────────────
ANALYTIC GUIDANCE (Scenario 1 → Scenario 2)
────────────────────────────────────────
1) Read every response as if it were a line in a short story.

2) Coherence check (gatekeeper):
   - ALWAYS first attempt to produce one single, unified, atomic theme that covers all/allmost all responses.
   - When some responses mention a specific example of a broader concept present in other responses, treat the specific as part of that broader concept if doing so produces a truthful unifying theme.
   - Allowed generalisation = only by:
     • combining/reordering/shortening existing words/phrases from the responses,
     • choosing the most general formulation that literally appears in the data.
   - NOT allowed: synonyms or terms not literally in the responses, implicit assumptions, or background knowledge.

3) If one unified theme is honestly not possible (mixed cluster):
   a) Identify sub-themes that are explicitly supported by at least 2 responses each, and output them as separate atomic themes.
   b) Any remaining single responses (1-off ideas) become separate atomic themes.

4) Summaries:
   - Maximum 25 words.
   - Must reflect both the survey question and the cluster content.
   - Must be strictly grounded in the literal wording of the responses (no new terms).

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
CRITICAL RULES
────────────────────────────────────────
- All output must be in {language}.
- Output only valid JSON (no extra text).
- Always attempt one unified theme first (Scenario 1). Only use multiple themes if this is genuinely impossible (Scenario 2).
- Sub-themes only when ≥2 explicit, similar responses; ignore singletons in your output.
""" 

CANDIDATE_CODE_SELECTION_PROMPT = """
You are a {language} qualitative analyst mapping themes to existing codes in a codebook.
A codebook in this setting is a set of code names and descriptions that can label and describe written survey responses.

────────────────────────────────────────
INPUTS  (XML blocks will be interpolated)
────────────────────────────────────────
<survey_question>
{survey_question}
</survey_question>

<cluster_summary>
{cluster_summary}
</cluster_summary>

<existing_codebook>
{code_text}
</existing_codebook>

────────────────────────────────────────
CONCEPTUAL FOUNDATION
────────────────────────────────────────
Existing codes represent our current understanding.
Your task: Find best matches.
Multiple themes require multiple codes.

────────────────────────────────────────
ANALYTIC GUIDANCE
────────────────────────────────────────

1. For **EACH theme** numbered in the summary, attempt to find matching existing codes
3. Select the codes, if any, that are relevant in capturing the themes
4. Be selective: only select suitable codes for our codebook
5. Be comprehensive: ensure no theme is left uncoded
6. Present your selection in the JSON array format described below

────────────────────────────────────────
OUTPUT  (raw JSON, no extra text in {language})
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
  // repeat for every theme_id
]

IMPORTANT:
- You may select NONE, ONE, or MULTIPLE candidate codes per theme
- Do not create new codes or modify existing ones
- If a concept has no suitable existing codes, that's acceptable - return what you can find
- Multi-theme clusters are normal and expected - code each theme appropriately
- Output ONLY the JSON array - no other text

"""

CODE_GENERATION_PROMPT = """
You are a {language} qualitative data analyst specializing in thematic coding and codebook development.
You will be given 1 or 2 theme(s) that summarize ideas expressed in written survey responses.
Your task is to maintain the codebook by deciding whether to use, modify, or create codes that accurately and precisely describe each theme.
    
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

────────────────────────────────────────
PRINCIPLES FOR HIGH QUALITY CODING 
────────────────────────────────────────
- PARSIMONIOUS: only as many codes as needed, no redundancy
- ATOMIC: each code captures ONE concept only (no “and”, “including”, “with”)
- PRECISE: clear boundaries enabling reliable coding
- CONCISE: code names 2–5 words
- OPERATIONAL: definitions use observable criteria, not interpretations
- CONSISTENT ABSTRACTION LEVEL: match the theme’s level of abstraction
- MUTUALLY EXCLUSIVE: minimal overlap between codes

────────────────────────────────────────
ANALYTIC GUIDANCE
────────────────────────────────────────
Step 1: Review
- Expect exactly 1 or 2 themes in the provided summary
- A theme is a single, coherent idea stated in ≤20 words

Step 2: Choose the Appropriate Coding Action for Each Theme
- `use_existing` → existing code(s) fully capture the theme (≥90% match)
- `modify_existing` → existing code captures most but not all of the theme (≥60% and <90% match)
- `create_new` → no existing code captures ≥60% of the theme

When modifying or creating:
- Name must be 2–5 words
- Definition must be ≤30 words, precise, and operational
- Match the theme’s abstraction level
- Never combine subthemes in names using "and", "with", etc.

────────────────────────────────────────
OUTPUT  (raw JSON, no extra text, in {language})
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
      "decision": "use_existing | modify_existing | create_new",
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
- All output fields must be in {language}
- Output ONLY valid JSON, no other text
"""


VALIDATION_PROMPT = """
You are a {language} qualitative data analyst specializing in codebook development and evaluation. 
You will assess your colleague's coding recommendations for a cluster that contains 1 or 2 themes summarizing ideas expressed in written survey responses.
Your task is to maintain a parsimonious codebook with clear code names (no compounding of subthemes!).

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
- Expect exactly 1 or 2 themes in the provided cluster summary.
- Assess EACH recommendation using these criteria:
  a) **Semantic fit** – Does the code accurately capture the theme’s meaning?
  b) **Atomicity** – Does the code name represent a single theme? No compounding with "and", "with", "including".
  c) **Parsimony** – Were existing codes considered before creating/modifying?
  d) **Non-redundancy** – Would this code overlap with others in the codebook?
  e) **Abstraction level** – Does the code match the theme’s level of abstraction?

────────────────────────────────────────
VALIDATION DECISION RULES
────────────────────────────────────────
For EACH theme/code pair:
- **APPROVE** – Code is necessary, atomic, correct abstraction level, and well-justified.
- **REVISE** – Code is generally valid but needs refinement (e.g., shorter name, clearer definition).
- **REJECT** – Code is unnecessary or redundant; use an existing code instead.

When REVISE or REJECT:
- Always propose a final validated code name (2–5 words, atomic) and definition (≤30 words, operational, precise) to replace or substitute the original.

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
        "code": "final code name",
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
- Ensure atomic codes (one theme per code) — no compounding.
- Output ONLY valid JSON, no other text.
- All output fields must be in {language}.
"""



# CLUSTER_SUMMARY_PROMPT = """
# You are a {language} qualitative analyst who treats survey data as stories waiting to be told.

# ────────────────────────────────────────
# INPUTS  (XML blocks will be interpolated)
# ────────────────────────────────────────
# <survey_question>
# {survey_question}
# </survey_question>

# <cluster_text>
# {cluster_text}
# </cluster_text>

# ────────────────────────────────────────
# CONCEPTUAL FOUNDATION
# ────────────────────────────────────────
# A "theme" is an atomic, shared concept that:
# • Represents ONE indivisible idea (cannot be split further)
# • Meaningfully answers the survey question
# • Has explanatory power for why these responses matter
# • Appears in a significant portion of responses

# Mathematical similarity (via embeddings) ≠ conceptual unity.
# Clusters may contain: unified theme | multiple themes | noise

# ────────────────────────────────────────
# ANALYTIC GUIDANCE
# ────────────────────────────────────────
# 1. **Immerse yourself** in the cluster
#    Read every response completely before identifying patterns.

# 2. **Active theme construction using decision tree:**
#    ```
#    IF one concept appears in ≥75% of responses:
#        → Single unified theme
#    ELIF multiple distinct concepts each appear in ≥30%:
#        → Multiple separate themes  
#    ELSE:
#        → Noise (no coherent theme)
#    ```

# 3. **Seek unity before fragmentation**
#    ▸ Ask: "What overarching narrative connects these voices?"
#    ▸ Look for latent meaning beneath surface wording
#    ▸ Connect dots between seemingly unrelated responses
#    ▸ Only fragment when distinct ideas truly cannot unite

# 4. **Reflection checkpoint**
#    - Before finalising, internally decide whether the cluster is unified or fragmented, based on the data's coherence.

# 5. **Verify atomicity**
#    ✗ Compound concepts: "cost and accessibility"
#    ✗ Nested ideas: "quality including customer service"  
#    ✓ Single concept: "affordability" or "product quality"

# 6. **Calculate coverage precisely**
#    Count responses expressing theme ÷ total responses × 100
#    Round to nearest 5% for clarity

# ────────────────────────────────────────
# OUTPUT FORMAT (strict JSON, in {language})
# ────────────────────────────────────────

# **Single theme:**
# {{
#   "themes": [
#     "This theme covers [≤25-word synopsis capturing the cluster's gist]\\nThe responses mention [brief list of explicit mentions, using exact words from the data]."
#   ]
# }}

# **Multiple themes:**
# {{
#   "themes": [
#     "Theme 1: This theme covers [≤25-word synopsis capturing the cluster's gist]\\nThe responses mention [brief list of explicit mentions, using exact words from the data].",
#     "Theme 2: This theme covers [≤25-word synopsis capturing the cluster's gist]\\nThe responses mention [brief list of explicit mentions, using exact words from the data]."
#     // Continue for each theme
#   ]
# }}

# **If classified as noise:**
# {{
#   "themes": []
# }}

# ────────────────────────────────────────
# CRITICAL OUTPUT RULES:
# ────────────────────────────────────────
# - All text must be in {language}.
# - Output ONLY valid JSON (no markdown, no extra text).
# - Keep summaries faithful to the provided cluster_text; do not paraphrase beyond clarity.
# - Use only explicit mentions found in the data for the "responses mention" section.

# """

# CANDIDATE_CODE_SELECTION_PROMPT = """
# You are a {language} qualitative analyst mapping themes to existing codes.

# ────────────────────────────────────────
# INPUTS  (XML blocks will be interpolated)
# ────────────────────────────────────────
# <survey_question>
# {survey_question}
# </survey_question>

# <cluster_summary>
# {cluster_summary}
# </cluster_summary>

# <existing_codebook>
# {code_text}
# </existing_codebook>

# ────────────────────────────────────────
# CONCEPTUAL FOUNDATION
# ────────────────────────────────────────
# Existing codes represent our current understanding.
# Your task: Find best matches while identifying gaps.
# Remember: Multiple themes require multiple codes (atomicity).

# ────────────────────────────────────────
# ANALYTIC GUIDANCE
# ────────────────────────────────────────
# 1. **Work theme-by-theme**
#    For each theme_id, grasp its core atomic concept.

# 2. **Score semantic overlap systematically**
   
#    | Score Range | Classification | Action |
#    |-------------|---------------|---------|
#    | 90-100% | Strong match | Definitely use |
#    | 70-89% | Good match | Use if gap is trivial |
#    | 40-69% | Partial match | Only if combines with others |
#    | 0-39% | Weak match | Ignore |

# 3. **Apply selection logic**
#    ```
#    IF any single code scores ≥90%:
#        → Select that code alone
#    ELIF single code scores 70-89% AND gap is minor:
#        → Select with documented gap
#    ELIF 2 codes together score ≥90%:
#        → Select both (max 2 per theme)
#    ELSE:
#        → Mark for new code creation
#    ```

# 4. **Document uncovered aspects**
#    List what remains uncoded in 3-7 words.

# 5. **Maintain abstraction consistency**
#    Selected codes should match the specificity level
#    of the existing codebook.

# ────────────────────────────────────────
# OUTPUT  (raw JSON, no extra text in {language})
# ────────────────────────────────────────

# [
#   {{
#     "code": "exact same name of existing code 1",
#     "definition": "exact same definition of existing code 1"
#   }},
#   {{
#     "code": "exact same name of existing code 2",
#     "definition": "exact same definition of existing code 2"
#   }}
#   // repeat for every theme_id
# ]

# IMPORTANT:
# - You may select NONE, ONE, or MULTIPLE candidate codes per theme
# - Do not create new codes or modify existing ones
# - If a concept has no suitable existing codes, that's acceptable - return what you can find
# - Multi-theme clusters are normal and expected - code each theme appropriately
# - Output ONLY the JSON array - no other text

# """

# CODE_GENERATION_PROMPT = """
# You are a {language} coding-scheme designer ensuring every theme gets exactly 
# one atomic, non-redundant code.

# ────────────────────────────────────────
# INPUTS  (XML blocks will be interpolated)
# ────────────────────────────────────────
# <survey_question>
# {survey_question}
# </survey_question>

# <cluster_summary>
# {cluster_summary}
# </cluster_summary>

# <code_selection>
# {candidate_codes}
# </code_selection>

# ────────────────────────────────────────
# CONCEPTUAL FOUNDATION
# ────────────────────────────────────────
# Parsimony: Minimum codes for maximum explanatory power
# Atomicity: Each code = one indivisible concept
# Consistency: Same abstraction level across codebook
# Non-redundancy: No overlapping or duplicate codes

# ────────────────────────────────────────
# DECISION FRAMEWORK
# ────────────────────────────────────────
# For each theme, choose exactly ONE operation:

# ┌─────────────────┬────────────────────────────────────┬──────────────────┐
# │ Operation       │ When to Choose                     │ Threshold        │
# ├─────────────────┼────────────────────────────────────┼──────────────────┤
# │ use_existing    │ Single code fully captures theme   │ ≥95% overlap     │
# │ modify_existing │ Code needs minor adjustment        │ 80-94% overlap   │
# │ create_new      │ No adequate existing code          │ <80% overlap     │
# └─────────────────┴────────────────────────────────────┴──────────────────┘

# CRITICAL PARSIMONY CHECKS:
# ✗ Never create if existing code has ≥95% overlap
# ✗ Never modify to become broader (violates atomicity)
# ✗ Never use compound concepts ("and", "with", "plus")
# ✓ Always prefer reuse over creation
# ✓ Keep modifications minimal and specific
# ✓ Ensure new codes fill genuine gaps

# ────────────────────────────────────────
# ANALYTIC GUIDANCE
# ────────────────────────────────────────
# 1. **Review theme-by-theme**
#    Examine matches, overlap scores, and uncovered concepts.

# 2. **Check for redundancy first**
#    Before creating: Could ANY existing code work with 
#    minor modification? Check entire codebook, not just matches.
#    Remember: Only use existing codes if they achieve ≥95% semantic overlap.

# 3. **Maintain abstraction level**
#    New/modified codes must match existing codebook's:
#    - Specificity (broad vs narrow)
#    - Scope (single behavior vs category)
#    - Phrasing style (noun vs verb phrases)

# 4. **Write atomic definitions**
#    Definition = 1-2 sentences, one testable concept
#    No examples in definition (keep it conceptual)

# 5. **Document decision rationale**
#    Explain why this preserves parsimony and atomicity.

# ────────────────────────────────────────
# OUTPUT  (raw JSON, no extra text, in {language})
# ────────────────────────────────────────
# {{
#   "cluster_analysis": {{
#     "number_of_themes": "integer (1, 2, or more)",
#     "theme_descriptions": [
#       "Brief description of theme 1",
#       "Brief description of theme 2 (if applicable)"
#     ]
#   }},
#   "coding_decisions": [
#     {{
#       "theme_number": 1,
#       "theme_description": "what this theme is about",
#       "decision": "use_existing | modify_existing | create_new",
#       "action_details": {{
#         "codes_to_use": ["exact code names"],
#         "codes_to_modify": "single exact code name or null",
#         "modified_code_name": "new name if modifying, else null",
#         "modified_code_definition": "1-2 sentence definition if modifying, else null",
#         "new_code_name": "name if creating new, else null",
#         "new_code_definition": "1-2 sentence definition if creating new, else null"
#       }},
#       "justification": "why this action is appropriate for this specific theme"
#     }},
#     {{
#       "theme_number": 2,
#       "theme_description": "what this second theme is about",
#       "decision": "use_existing | modify_existing | create_new",
#       "action_details": {{...}},
#       "justification": "..."
#     }}
#     // repeat for every theme_id
#   ],
#   "overall_justification": "why treating these as separate themes preserves atomicity and improves codebook quality"
# }}

# IMPORTANT:
# - Process EACH distinct theme separately in the coding_decisions array
# - Never merge distinct themes into a single code
# - All output fields must be in {language}
# - Output ONLY valid JSON, no other text
# """

# VALIDATION_PROMPT = """
# You are a {language} QA auditor ensuring codebook rigor and parsimony.

# ────────────────────────────────────────
# INPUTS  (XML blocks will be interpolated)
# ────────────────────────────────────────
# <survey_question>
# {survey_question}
# </survey_question>

# <cluster_summary>
# {cluster_summary}
# </cluster_summary>

# <code_selection>
# {candidate_codes}
# </code_selection>

# <coding_decisions>
# {step3_recommendation}
# </coding_decisions>

# ────────────────────────────────────────
# EVALUATION RUBRIC
# ────────────────────────────────────────
# Assess each decision on five dimensions (provide brief text assessment):

# a) **Theme Separation** 
#    Are themes truly independent concepts?
#    Could any be merged without loss of meaning?

# b) **Semantic Fit**  
#    Does the code capture the theme's full meaning?
#    Any important aspects missing?

# c) **Atomicity** 
#    Is this truly ONE concept?
#    Any hidden "and" or "with" relationships?

# d) **Parsimony** 
#    Is this the minimum intervention needed?
#    Could existing codes work instead?

# e) **Redundancy** 
#    Does this duplicate any existing codes?
#    Clear boundaries with other codes?

# ────────────────────────────────────────
# VALIDATION DECISION TREE
# ────────────────────────────────────────
# For each coding decision:
# ```
# IF all dimensions are satisfactory:
#     → APPROVE as proposed
# ELIF major atomicity or redundancy issues:
#     → REJECT or SPLIT (fundamental issue)
# ELIF semantic fit or parsimony issues:
#     → REVISE (adjust name/definition)
# ELIF better existing code available:
#     → REPLACE with existing
# ELSE:
#     → APPROVE with noted concerns
# ```

# ────────────────────────────────────────
# VALIDATION ACTIONS
# ────────────────────────────────────────
# | Verdict | When | Result |
# |---------|------|--------|
# | APPROVE | Meets all criteria | Implement as is |
# | REVISE | Minor adjustments needed | Provide improved version |
# | REPLACE | Better option exists | Suggest alternative |
# | REJECT | Redundant/unnecessary | Use existing code |
# | SPLIT | Violates atomicity | Divide into multiple codes |
# | MERGE | Themes not distinct | Combine into single code |

# ────────────────────────────────────────
# OUTPUT  (raw JSON, no extra text, in {language})
# ────────────────────────────────────────
# {{
#   "theme_assessment": {{
#     "number_of_themes_identified": "integer",
#     "theme_separation_valid": "true/false",
#     "theme_separation_reasoning": "are the themes truly distinct or should they be merged/split?"
#   }},
#   "code_validations": [
#     {{
#       "theme_number": 1,
#       "theme_description": "what theme is being coded",
#       "original_recommendation": "what was proposed",
#       "evaluation": {{
#         "semantic_fit": "assessment",
#         "atomicity": "assessment",
#         "parsimony": "assessment",
#         "redundancy": "assessment"
#       }},
#       "decision": "APPROVE | REVISE | REJECT | MERGE | SPLIT",
#       "decision_rationale": "explanation",
#       "validated_code": {{
#         "code": "final code name",
#         "definition": "final definition"
#       }}
#     }},
#     {{
#       "theme_number": 2,
#       "theme_description": "...",
#       "original_recommendation": "...",
#       "evaluation": {{...}},
#       "decision": "...",
#       "decision_rationale": "...",
#       "validated_code": {{...}}
#     }}
#   ],
#   "overall_validation": {{
#     "all_themes_coded": "true/false",
#     "final_code_count": "integer",
#     "summary": "brief summary of the validation outcome"
#   }}
# }}

# IMPORTANT FORMAT NOTES:
# - For SPLIT decisions: "validated_code" should be an ARRAY of code objects: [{{\"code\": \"name1\", \"definition\": \"def1\"}}, {{\"code\": \"name2\", \"definition\": \"def2\"}}]
# - For all other decisions (APPROVE/REVISE/REJECT/MERGE): "validated_code" should be a SINGLE object: {{\"code\": \"name\", \"definition\": \"definition\"}}

# Strict rules:
# - Validate EACH theme/code pair separately
# - Ensure atomic codes (one theme per code)
# - Don't force unrelated themes into single codes
# - Allow multiple codes when multiple distinct themes exist
# - Output ONLY valid JSON, no other text
# - All output fields must be in {language}
# """


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


# HIERARCHY_MAP_PROMPT = """
# You are a {language} language exeprt and a qualitative researcher  specializing in thematic analysis following Braun & Clarke methodology.
# Your task is to analyze EXACTLY 10 codes and organize them into a 3-level hierarchy: codes → domains → themes.

# <survey_question>
# {survey_question}
# </survey_question>

# <codes>
# {codes_batch}
# </codes>

# <instructions>
# Step 1. Review the codes - these capture shared ideas from survey responses.
# Step 2. Look for patterns and shared meanings among the codes. Consider how different codes might be combined based on underlying concepts or features of the data.
# Step 3. Group related codes with shared meaning into 2 or more practical DOMAINS
# Step 4. Group related domains that share an overarching narrative into 1 or more broad THEMES.
# Step 5. Actively construe relationships - themes don't simply "emerge" from data.
# Step 6. Consider salience over frequency - meaningful patterns matter more than code counts.
# Step 7. Aim for distinctive yet coherent groupings that may even be contradictory.
# Step 8. Ensure ALL codes are included - none can be left out.
# Step 9. Create balanced groupings - avoid unwieldy structures.
# step 10. Consider creating a ”miscellaneous” category for codes that don’t fit elsewhere.

# CRITICAL:
# 1. You MUST include ALL 10 codes in your output - check if these code numbers are included: {codes_to_include}
# 2. The codes are numbered - you must use these EXACT numbers
# 3. Each code can appear ONLY ONCE in the hierarchy
# </instructions>

# OUTPUT FORMAT (JSON):
# {{
#   "batch_id": {batch_number},
#   "themes": [
#     {{
#       "theme_name": "[Theme name in {language}]",
#       "domains": [
#         {{
#           "domain_name": "[Domain name in {language}]",
#           "codes": [
#             {{
#               "code_number": [exact number from input],
#               "code_name": "[exact code text from input]"
#             }}
#           ]
#         }}
#       ]
#     }}
#   ]
# }}

# BEFORE SUBMITTING:
# Verify: Is each code included only exactly once?

# Return ONLY the JSON object with all content in {language}.
# """

# HIERARCHY_REDUCE_PROMPT = """
# You are a {language} language expert and qualitative researcher specializing in thematic analysis following Braun & Clarke methodology. 
# Your task is to create a well-structured codebook based on a first draft of the codebook. 
# This codebook will be used to categorize responses to an open-ended survey question, and consists of 3 levels: specific codes > concrete domains > broad themes.

# Here is the survey question:
# <survey_question>
# {survey_question}
# </survey_question>

# Here is the first draft of the codebook:
# <first_draft_of_codebook>
# {batch_hierarchies}
# </first_draft_of_codebook>>

# CRITICAL: There are {total_codes} codes total that MUST ALL appear in your final codebook.

# <instructions>
# YOUR TASK: Create a refined, consolidated codebook by:

# 1. IMPROVING LABELS:
#    - All labels should be concise and make sense as stand-alone terms in light of the survey question
#    - Theme names should capture overarching concepts (not just list topics)
#    - Domain names should be clear and distinctive
#    - Avoid overlapping or vague labels

# 2. MERGING STRATEGICALLY:
#    - Combine themes that address the same overarching concept
#    - Merge domains that group similar types of codes
#    - Track which original themes/domains you're combining

# 3. ENSURING QUALITY:
#    - Each theme should represent a coherent narrative
#    - Domains within a theme should be clearly distinguished
#    - Balance the structure (avoid one huge theme with many tiny ones)

# 4. PRESERVING ALL CODES:
#    - Every single code must appear exactly once
#    - If codes don't fit well after merging, use "Overige" theme
#    - Never duplicate or omit codes
# </instructions>

# Your output should follow this structure:

# <output>
# {{
#   "themes": [
#     {{
#       "theme_name": "[Refined theme name in {language}]",
#       "theme_concept": "[What unifying concept this theme represents]",
#       "domains": [
#         {{
#           "domain_name": "[Clear domain name in {language}]",
#           "domain_description": "[What distinguishes this domain]",
#           "codes": [
#             {{
#               "code_number": [exact number],
#               "code_name": "[exact original code name]",
#               "fit_rationale": "[Brief reason why this code belongs here]"
#             }}
#           ]
#         }}
#       ]
#     }}
#   ],
#   "transformation_notes": {{
#     "themes_merged": [
#       {{
#         "original": ["Theme A from Codebook 1", "Theme B from Codebook 2"],
#         "final": "New Combined Theme",
#         "reason": "Both addressed X concept"
#       }}
#     ],
#     "domains_merged": [
#       {{
#         "original": ["Domain 1", "Domain 2"],
#         "final": "Merged Domain",
#         "within_theme": "Theme Name",
#         "reason": "Both grouped Y type of codes"
#       }}
#     ]
#   }}
# }}
# </output>

# Remember, it is CRITICAL that all {total_codes} codes from the original codebook appear exactly once in your final codebook. Do not omit or duplicate any codes.
# Before providing your final output, use a <scratchpad> to think through your process of refining and consolidating the codebook. Consider how you will improve labels, merge themes and domains strategically, ensure quality, and preserve all codes.
# After your thought process, provide your final refined codebook in the specified JSON format within <output> tags.
# Return ONLY the JSON object with all text in {language}.
# """

# =============================================================================
# STEP 7: CODEBOOK GENERATION - NEW ARCHITECTURE
# =============================================================================

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
