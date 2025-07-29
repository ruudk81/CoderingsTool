
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
# STEP 7:  CODEBOOK GENERATION
# =============================================================================

INITIAL_CODEBOOK_CREATION_PROMPT = """
You are an expert qualitative data analyst specializing in rigorous thematic analysis and code creation.
Your task is to generate hypothetical codes that might be encountered when analyzing written answers to a specific survey question.

CRITICAL CODING PRINCIPLES:
- **ATOMIC**: Each code must capture ONE concept only - no compound ideas with "and", "including", "with"
- **PRECISE**: Clear boundaries that enable reliable coding decisions
- **CONCISE**: Code names must be 2-5 words maximum
- **OPERATIONAL**: Definitions must use observable criteria, not interpretations
- **MUTUALLY EXCLUSIVE**: Minimal overlap between codes

Here are the details for your task:
- Language to use: <language>{language}</language>
- Number of codes to generate: <n_codes>{n_codes}</n_codes>
- Survey question to analyze: <survey_question>{survey_question}</survey_question>

REQUIRED DEFINITION FORMAT:
All definitions must follow this structure:
"References to [specific concept/aspect]."

Generate {n_codes} diverse, hypothetical codes that might emerge from analyzing responses to this survey question. 
Create codes that could apply to ANY survey topic. Do not assume the survey is about education, healthcare, or any specific domain. Let the survey question guide your code generation.

Think about different code types:
- **Attribute codes**: Qualities or characteristics mentioned
- **Process codes**: Actions, procedures, or methods described
- **Relational codes**: Interactions or connections between elements
- **State codes**: Conditions, situations, or circumstances
- **Evaluative codes**: Assessments, judgments, or opinions expressed

Please provide your response in {language} as a JSON array of objects, where each object has "code" and "definition" fields.  

Here's an example of the STRUCTURE to follow (using generic placeholders):
<example>
[
  {{"code": "Quality assessment", "definition": "References to evaluating the quality/characteristic of topic-specific element."}},
  {{"code": "Process difficulties", "definition": "Mentions of challenges in topic-specific process."}},
  {{"code": "Actor perspectives", "definition": "References to viewpoints of relevant actors/participants."}},
  {{"code": "Outcome experiences", "definition": "Mentions of positive/negative outcomes experienced."}}
]
</example>

Return ONLY the JSON array in {language}.
"""

SYSTEM_MESSAGE = """
Act as a {language} qualitative data analyst who is a helpful assistent in the development of a codebook.
A codebook in this setting is a collection of codes and definitions that can be used to describe pieces of data collected from an open-ended question in a survey. 
"""

CODEBOOK_ANALYSIS_PROMPT = """
{system_message}
This time we will focus on written responses to the following survey question: "{survey_question}".

Given these codes from the codebook:
<existing_codebook>
{code_text}
</existing_codebook>

Analyze what specific aspects or dimensions of the survey question do these codes address?

Output a concise summary in {language} following this structure:
"Coverage: These codes address [short description of aspects and dimensions covered]"

IMPORTANT: Return ONLY the analysis text following this exact format, no JSON or additional explanation.
"""

RESPONSE_SUMMARY_PROMPT = """
{system_message}

Analyze this cluster of semantically related ideas expressed in response to this survey question: "{survey_question}"

<clustered_ideas>
{cluster_text}
</clustered_ideas>

Extract the cluster's pattern to enable code matching:
1. **Core theme**: What is the central concept unifying these responses? Be specific and use the language of the respondents where appropriate.
2. **Key components**: What are the 2-3 essential elements that ALL responses in this cluster share?

Output a concise analysis in {language} following this structure:
"This cluster's core theme is: [core theme description]. The essential shared components are [element 1], [element 2], and [element 3]."

IMPORTANT: Return ONLY the analysis text, no JSON formatting or additional explanation.
"""

MATCH_AND_RECOMMEND_PROMPT = """
You are a {language} qualitative data analyst specializing in codebook development for survey responses.
Your task is to analyze clustered ideas from open-ended survey responses and recommend whether to use existing codes, modify them, or create new ones.

Your goal is to maintain a codebook that is:
- ***Parsimonious***: Use the fewest number of codes necessary to capture essential meanings. Avoid overcoding or overly specific codes.
- ***Non-redundant***: Ensure each code represents a distinct, non-overlapping concept.
- ***Clear***: Each code should have a precise, easy-to-understand definition, ideally with examples, so it can be applied consistently.

---

First, review the survey question that generated the responses:
<survey_question>
{survey_question}
</survey_question>

Next, examine the existing codes in the codebook:
<existing_codes>
{existing_codes}

{codebook_analysis}
</existing_codes>

Now, review a cluster of survey responses that may require a new code:
<clustered_survey_responses>
{clustered_ideas}
</clustered_survey_responses>

Also take note of this summary about the cluster's core theme:
<summary>
{summaries}
</summary>    

---

**Follow this evaluation process:**
1. Compare the cluster’s core theme to each existing code.
2. Assess thematic coverage: How well do existing codes capture the cluster’s meaning?
3. Determine fit: Can one or more existing codes (with or without revision) represent the cluster adequately?
4. Only create a new code if no existing code meaningfully captures the cluster’s concept.
5. Always favor parsimony. Reuse or revise existing codes when possible; treat new codes as a last resort.

If you need to create or revise codes, ensure they meet these criteria:
1. **Conceptual unity (ATOMIC)**: The code should represent ONE clear concept only—avoid compound ideas joined by "and", "including", or "with".
2. **Operational clarity**: Coders must be able to reliably identify when the code applies vs. not.
3. **Parsimony**: Use the simplest expression possible without losing essential meaning.

---

Output ONE recommendation as valid JSON:
{{
  "cluster_core_theme": "description of the core theme here",
  "decision": "use_existing | modify_existing | create_new",
  "action_details": {{
    "codes_to_use": ["list of code names"] or null,
    "codes_to_modify": "code name to modify" or null,
    "modified_code_name": "re-name if modify_existing" or null,
    "modified_code_definition": "re-define if modify_existing" or null,
    "new_code_name": "name if create_new" or null,
    "new_code_definition": "definition if create_new" or null
  }},
  "justification": "explain why this decision best balances parsimony with conceptual accuracy"
}}

---

Examples of well-structured code definitions:
- "References to [specific limitation or constraint] affecting [process or outcome]."
- "Mentions of [positive or negative] changes in [behavior or practice]."
- "Expressions of [emotion or attitude] regarding [situation or process]."

Avoid these weak definitions:
- ❌ Compound: "References to [issue A] including [aspect 1], [aspect 2], and [aspect 3]"
- ❌ Vague: "Mentions of various [things] related to [topic]"
- ❌ Interpretive: "Underlying [abstract concept] manifesting in different ways"

---

Return ONLY the JSON object in {language}.
Fill only relevant fields in `action_details` based on your decision.
"""


VALIDATION_PROMPT = """
You are a {language} language expert and qualitative researcher tasked with validating recommendations for adding new codes to a codebook used for analyzing survey responses. 
Your goal is to ensure that the codebook remains **parsimonious**, **non-redundant**, and **clear**.

---

You will be evaluating a recommendation to add a new code to a codebook created to analyze responses to this survey question:
<survey_question>
{survey_question}
</survey_question>

Next, examine the existing codes and analysis notes:
<existing_codes>
{existing_codes}
</existing_codes>

Now, examine the extracted ideas from survey responses:
<extracted_ideas>
{clustered_ideas}
</extracted_ideas>

Now, consider the recommendation for adding a new code:
<recommendation>
{step3_recommendation}
</recommendation>

---

Your task is to **APPROVE**, **REVISE**, or **REJECT** the recommendation based on the following evaluation criteria:

1. **Parsimony**: Were existing code options properly exhausted? Would using or modifying an existing code result in a meaningful loss of nuance?
2. **Non-redundancy**: Does the proposed code avoid conceptual overlap with existing codes?
3. **Justification alignment**: Is the reasoning provided consistent and logically supportive of the proposed action?

---

Use these decision guidelines:
- **APPROVE**: All criteria are met. The code is necessary, atomic, well-formed, and clear.
- **REVISE**: The core concept is valid, but the label or definition needs refinement (e.g., too vague, compound, or imprecise).
- **REJECT**: The recommendation fails parsimony (existing codes suffice), shows substantial overlap with existing codes, or includes multiple unrelated concepts.

When approving or revising a code, ensure that it adheres to these principles:
- **ATOMIC**: One idea per code — no compound concepts using "and", "with", or "including".
- **CONCISE LABELS**: 2–5 words capturing the essence of the code without being vague.
- **MUTUALLY EXCLUSIVE**: Avoid overlap that would cause ambiguity when coding.

Use this definition structure:
**"References to [specific concept/aspect]."**

---

Provide your validation output in {language} as a valid JSON object in this format:
{{
  "evaluation": {{
    "parsimony_reasoning": "assessment of whether existing options were exhausted",
    "redundancy_reasoning": "assessment of conceptual overlap with existing codes",
    "justification_reasoning": "assessment of logic consistency in the recommendation"
  }},
  "decision": "APPROVE | REVISE | REJECT",
  "decision_rationale": "synthesize the evaluation into a clear decision explanation",
  "validated_code": {{
    "code": "final code name (approved/revised) or null if rejected",
    "definition": "final definition (approved/revised) or null if rejected"
  }}
}}

---

**Important:**
- Return **ONLY** the JSON object in {language}
- Base your assessment strictly on observable content
- Do not speculate or overinterpret
- The ultimate goal is to ensure the codebook is parsimonious, non-redundant, and clear.
"""

# =============================================================================
# STEP 8: THEME IDENTIFICATION  
# =============================================================================

THEME_IDENTIFICATION_PROMPT = """
You are a {language} language expert and qualitative researcher specializing in thematic analysis using the Braun & Clarke (2006) methodology.
Your task is to analyze a cluster of codes and recommend whether to use existing themes, modify them, or create new ones.

---

First, review the survey question that generated the codes:
<survey_question>
{survey_question}
</survey_question>

Next, examine the themes that currently exist in the codebook:
<existing_themes>
{existing_themes_text}
</existing_themes>

Now, analyze the following cluster of {codes_count} codes that may require a new or revised theme:
<clustered_codes>
{codes_text}
</clustered_codes>

---

Follow this evaluation process:
1. Compare the conceptual focus of the clustered codes to each existing theme.
2. Assess thematic coverage: To what extent do existing themes capture the unifying idea(s) in the cluster?
3. Determine fit: Can one or more existing themes (as-is or with revision) adequately represent this cluster?
4. Assess sentiment alignment: Do the codes consistently express a similar sentiment (e.g., all positive, all negative)? Or do they differ in evaluative tone?

Decision guidelines:
4. **Create a new theme** only if no existing theme meaningfully captures the cluster's unifying concept.
5. If the codes reflect both positive and negative sentiment—and this is clearly signaled in their names and definitions—consider creating **both a positive and a negative variant** of the theme.
6. **Reuse or revise existing themes** when feasible. Favor parsimony and conceptual clarity.

When creating or revising a theme, ensure it meets these standards:
- **Coherent pattern**: The theme must capture a consistent pattern of meaning across the cluster.
- **Conceptual importance > frequency**: Prioritize what the theme tells us, not how often it appears.
- **Narrative value**: The theme should help tell a meaningful story about the data.
- **ATOMIC**: One clear idea per theme. Avoid compound constructions like "and", "including", or "with".
- **CONCISE LABEL**: Theme name should be 2–5 words, capturing its essence without vagueness.
- **MUTUALLY EXCLUSIVE**: Themes should have minimal overlap in scope to avoid ambiguity.

---

Return your assessment as a valid JSON object in this format:
{{
  "decision": "create_new | use_existing",
  "theme_name": "[Theme name in {language}]",
  "theme_description": "[Brief description of what conceptually unites the codes]",
  "existing_theme_used": "[Exact theme name from the list above, or null]",
  "confidence": "high | medium | low",
  "rationale": "[Detailed explanation of your decision based on fit, clarity, and distinctiveness]"
}}

---

**IMPORTANT:**
- Theme names and descriptions must be written in {language}.
- If using an existing theme, use the **EXACT** theme name provided above.
- Focus on **conceptual fit**, not surface similarity or keyword overlap.
- Return **ONLY** the JSON object.
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
You are a {language} language expert in qualitative data analysis specializing in applying codebooks to survey responses.
Your task is to assign the most appropriate code(s) from a set of candidate codes to a single extracted idea.

SURVEY QUESTION:
<survey_question>
{var_lab}
</survey_question>

IDEA TO CODE:
<idea_to_analyze>
Idea ID: {idea_id}
Idea Text: {idea_text}
</idea_to_analyze>

CANDIDATE CODES (5 most semantically similar):
<candidate_codes>
{candidate_codes}
</candidate_codes>
Note: These are the 5 codes from the codebook that are most semantically similar to this idea based on embedding similarity.

CODING INSTRUCTIONS:
1. **Analyze the idea** carefully in the context of the survey question
2. **Review the 5 candidate codes** and their definitions
3. **Select the best fitting code(s)**:
   - Prioritize exact conceptual matches
   - An idea can receive multiple codes if it clearly expresses multiple concepts
   - You must assign at least one code - select the best available option
   - Focus on the semantic meaning, not just keyword matching

4. **Assignment criteria**:
   - **Excellent fit (0.9-1.0)**: Idea directly expresses what the code definition describes
   - **Good fit (0.7-0.8)**: Idea relates closely to code definition with reasonable interpretation
   - **Moderate fit (0.5-0.6)**: Best available option but requires some interpretation
   - **Poor fit (0.3-0.4)**: Significant stretching of code definition required
   - **Very poor fit (0.0-0.2)**: Forced assignment due to lack of better options

IMPORTANT: You are working with a focused set of 5 pre-selected codes that are most likely to fit this idea. This reduces cognitive load and improves assignment accuracy.

Return your analysis as a JSON object with this exact structure:
{{
  "idea_id": "{idea_id}",
  "idea": "{idea_text}",
  "assigned_codes": ["code_name1", "code_name2"],
  "assignment_confidence": 0.85,
  "assignment_rationale": "Brief explanation of why these codes were chosen and how they relate to the idea's meaning"
}}

CRITICAL REQUIREMENTS:
- Must assign at least one code from the candidate list
- Use exact code names as provided
- Confidence score must reflect the actual conceptual fit
- Rationale must explain the semantic connection
- Return ONLY the JSON object in {language}

Begin the code assignment now."""


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