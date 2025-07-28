
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
   - Keep each idea atomic (one concept per idea)
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

Please provide your response as a JSON array of objects, where each object has "code" and "definition" fields.  
Here's an example of the STRUCTURE to follow (using generic placeholders):
<example>
[
  {{"code": "[Quality] assessment", "definition": "References to evaluating the [quality/characteristic] of [topic-specific element]."}},
  {{"code": "[Process] difficulties", "definition": "Mentions of challenges in [topic-specific process]."}},
  {{"code": "[Actor] perspectives", "definition": "References to viewpoints of [relevant actors/participants]."}},
  {{"code": "[Outcome] experiences", "definition": "Mentions of [positive/negative] outcomes experienced."}}
]
</example>

BAD examples to AVOID:
- Too compound: "[Aspect A] and [Aspect B]" (combines two concepts)
- Too vague: "Various [things]" (no clear boundaries)  
- Too interpretive: "Underlying [patterns]" (not directly observable)

Ensure each code is atomic, has clear boundaries, and follows the exact definition format.
Return ONLY the JSON array in {language}.
"""

SYSTEM_MESSAGE = """
Act as a {language} qualitative data analyst specializing in rigorous thematic analysis for the development of a codebooks.
A codebook in this setting is a collection of codes and definitions for those codes that can be used to describe pieces of data collected in a survey with open-ended questions. 
You create precise, mutually exclusive codes following these principles:
- Codes must be ATOMIC: Each code captures ONE coherent concept only
- Definitions must be PRECISE: Clear boundaries that enable reliable inter-coder agreement  
- Labels must be CONCISE: 2-5 words maximum, avoiding compound concepts
- Definitions must be OPERATIONAL: State explicit inclusion/exclusion criteria
- Structure must be CONSISTENT: All codes follow the same definition format
"""

CODEBOOK_ANALYSIS_PROMPT = """
{system_message}
This time we will focus on written responses to the following survey question: "{survey_question}".

Given these codes from the codebook:
<existing_codebook>
{code_text}
</existing_codebook>

Analyze these codes to help future matching decisions:
1. What specific aspects or dimensions of the survey question do these codes address?
2. What potential gaps exist - what types of responses or themes might NOT fit these codes?
3. Are there any conceptual overlaps or boundary issues between codes?    

Output a concise analysis in {language} following this structure:
"Coverage: These codes collectively address [aspects of the survey question].
Gaps: Responses about [gap 1], [gap 2], and [gap 3] might not fit existing codes.
Boundaries: [Note any overlap or ambiguity between existing codes, if relevant]."

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
2. **Abstraction level**: Is this cluster about specific instances, general patterns, or abstract concepts?
3. **Key components**: What are the 2-3 essential elements that ALL responses in this cluster share?
4. **Distinguishing features**: What makes this cluster semantically distinct from other possible themes?

Output a concise analysis in {language} following this structure:
"This cluster represents [core theme description using respondent language] at a [specific/general/abstract] level. The essential shared components are [element 1], [element 2], and [element 3]. What distinguishes this cluster is [unique aspect that separates it from other themes]."

IMPORTANT: Return ONLY the analysis text, no JSON formatting or additional explanation.
"""

MATCH_AND_RECOMMEND_PROMPT = """
{system_message}

This time we will focus on written responses to the following survey question: "{survey_question}".
You are making a codebook recommendation for a cluster of semantically similar survey responses.
Specifically, you need to recommend whether or not a new code needs to be created.

INPUT DATA:
<existing_codes>
{existing_codes}

Analysis notes: {codebook_analysis}
</existing_codes>
Note: These are the 5 codes nearest to this cluster's centroid embedding.

<clustered_ideas>
{clustered_ideas}

Cluster analysis: {summaries}
</clustered_ideas>
Note: These responses were grouped by HDBSCAN based on semantic similarity.

EVALUATION PROCESS:
1. Compare the cluster's core theme against each existing code
2. Assess thematic coverage: how well do existing codes capture the cluster's essential meaning?
3. Always favor parsimony: use existing codes when they adequately represent the cluster
4. Create new codes only when existing codes fundamentally cannot capture the cluster's core concept
5. Modification should be rare and only for minor extensions, not major broadening

CREATION CRITERIA:   
1. **Conceptual unity (ATOMIC)**: Does the new code represent ONE clear concept only? No compound ideas joined by "and", "including", or "with"
2. **Mutual exclusivity**: Is there clear differentiation from existing codes with minimal conceptual overlap?
3. **Appropriate scope**: Neither too narrow nor too broad - captures a meaningful, distinct concept at the right granularity?
4. **Abstraction consistency**: Matches the abstraction level of existing codes?
5. **Operational clarity**: Can coders reliably identify when this code applies vs. when it doesn't?
6. **Parsimony**: Is this the simplest way to capture this concept without losing essential meaning?

DEFINITION FORMAT GUIDELINES:
- Start with active language: "References to...", "Mentions of...", "Expressions of..."
- Focus on what IS included (positive criteria)
- Use observable language - what respondents explicitly state, not interpretations
- Keep definitions concise and specific
- Avoid lists of examples; focus on the core concept

Output ONE recommendation as valid JSON:
{{
  "cluster_core_theme": "identify from cluster analysis notes",
  "best_matching_codes": ["code1", "code2"],
  "coverage_assessment": {{
    "percentage": 80,
    "rationale": "explain specifically what aspects are/aren't covered by existing codes"
  }},
  "decision": "use_existing|modify_existing|create_new",
  "action_details": {{
    "codes_to_use": ["list if use_existing"] or null,
    "code_to_modify": "name if modify_existing" or null,
    "modification_suggestion": "specific wording to broaden if modify_existing" or null,
    "new_code_name": "name if create_new" or null,
    "new_code_definition": "definition if create_new" or null
  }},
  "justification": "explain why this decision best balances parsimony with conceptual accuracy"
}}

IMPORTANT:
- Return ONLY the JSON object in {language}
- Fill only relevant fields in action_details based on your decision
- Coverage percentage: estimate how well existing codes capture the cluster's core concept (0-100%)
  - 90-100%: Existing codes fully capture the cluster's meaning
  - 70-89%: Existing codes capture most but miss some nuances
  - 50-69%: Partial coverage, significant aspects missing
  - Below 50%: Poor coverage, new code likely needed
- One cluster = one recommendation
"""

VALIDATION_PROMPT = """
{system_message}

This time we will focus on written responses to the following survey question: "{survey_question}".
You are reviewing a code recommendation for clustered ideas extracted from survey responses.
Your job is to APPROVE, REVISE or REJECT the recommendation, and to provide a final validated code name and definition.

This is the extracted ideas: 
<clustered_ideas>
{clustered_ideas}
</clustered_ideas>
Note: These are ideas extracted from the original survey responses that prompted this recommendation.

This is the recommendation:
<recommendation>
{step3_recommendation}
</recommendation>
Note: This is the complete recommendation from the matching analysis.

These are existing codes in the code book:
<existing_codebook>
{existing_codes}
</existing_codebook>
Note: These are the 5 codes most similar to the recommended definition by semantic similarity.

EVALUATION CRITERIA:
1. **Parsimony**: Were existing code options properly exhausted? Would using/modifying existing codes sacrifice important nuance?
2. **Non-redundancy**: Is there minimal conceptual overlap with any existing code?
3. **Clarity**: Is the code name intuitive and the definition unambiguous?
4. **Scope appropriateness**: Does the code capture a single concept without being too narrow or broad?
5. **Justification alignment**: Does the recommendation's reasoning support its conclusion?

CODING PRINCIPLES FOR HIGH-QUALITY CODES:
- **ATOMIC**: One concept per code - no compound ideas with "and", "including", "with"
- **PRECISE BOUNDARIES**: Clear inclusion/exclusion criteria that enable consistent application
- **CONCISE LABELS**: 2-5 words maximum, capturing the essence without being vague
- **OPERATIONAL DEFINITIONS**: Observable criteria, not interpretive assumptions
- **MUTUALLY EXCLUSIVE**: Minimal overlap between codes to avoid coding ambiguity

REQUIRED DEFINITION STRUCTURE:
"References to [specific concept/aspect]."

Examples of GOOD definition structures:
- "References to [specific limitation/constraint] affecting [process/outcome]."
- "Mentions of [positive/negative] changes in [behavior/practice]."
- "Expressions of [emotion/attitude] regarding [situation/process]."

Examples of POOR definitions (avoid these):
- Too compound: "References to [issue A] including [aspect 1], [aspect 2], and [aspect 3]"
- Too vague: "Mentions of various [things] related to [topic]"
- Too interpretive: "Underlying [abstract concept] manifesting in different ways"

DECISION GUIDELINES:
- APPROVE: All criteria met, code is well-formed and truly necessary
- REVISE: Core concept is valid but needs refinement (unclear name, imprecise definition, minor scope issues)
- REJECT: Fails parsimony (existing codes suffice), significant redundancy, or covers multiple unrelated concepts

Output a validation assessment in {language}:
{{
  "evaluation": {{
    "parsimony_reasoning": "assessment of whether existing options were exhausted",
    "redundancy_reasoning": "assessment of conceptual overlap with existing codes",
    "justification_reasoning": "assessment of logic consistency in the recommendation"
  }},
  "decision": "APPROVE/REVISE/REJECT",
  "decision_rationale": "synthesize the evaluation into a clear decision explanation",
  "validated_code": {{
    "code": "final code name (approved/revised) or null if rejected",
    "definition": "final definition (approved/revised) or null if rejected"
  }}
}}

IMPORTANT: 
- Return ONLY the JSON object in {language}
- For REVISE decisions, provide an improved version in validated_code
- Code names MUST be 2-5 words maximum (no compound concepts)
- Definitions MUST follow the structure: "References to [specific concept]."
- Keep definitions concise and specific
- Focus on observable content, not interpretations
- Ensure the code is truly atomic (one concept only)
"""

# =============================================================================
# STEP 8: THEME IDENTIFICATION  
# =============================================================================

THEME_IDENTIFICATION_PROMPT = """
You are {language} language expert and a qualitative researcher specializing in thematic analysis following Braun & Clarke (2006) methodology.
You have been given a cluster of semantically related codes that were grouped using HDBSCAN clustering based on their embedding similarity.

SURVEY QUESTION:
{survey_question}

CLUSTER CODES ({codes_count} codes):
{codes_text}

EXISTING THEMES:
{existing_themes_text}

YOUR TASK:
Analyze this cluster of codes and decide whether to:
1. CREATE A NEW THEME that captures the shared conceptual pattern of these codes, OR
2. USE AN EXISTING THEME if one of the existing themes above adequately represents this cluster

DECISION CRITERIA:
- For CREATE NEW: The codes share a coherent conceptual pattern that is distinct from existing themes
- For USE EXISTING: An existing theme adequately captures the essence of these codes  
- Focus on conceptual meaning rather than surface-level keyword matching
- Consider the survey question context

BRAUN & CLARKE PRINCIPLES:
- Themes should capture coherent patterns of meaning
- Focus on conceptual importance, not just frequency
- Each theme should tell a meaningful story about the data
- Themes should be distinct yet coherent

LABELS:
- Theme names should capture overarching concepts (not just list topics)
- Theme names should also make semantic sense as stand-a-lone concepts in light of the survey question
- Theme descriptions should consists of concise phrases or sentences that clarify the theme name in light of the survey question.
- Avoid overlapping or vague labels

OUTPUT FORMAT (JSON):
{{
  "decision": "create_new|use_existing",
  "theme_name": "[Theme name in {language}]",
  "theme_description": "[Brief description of what unites these codes conceptually]",
  "existing_theme_used": "[Name of existing theme if used, or null]",
  "confidence": "high|medium|low",
  "rationale": "[Detailed explanation of your decision]"
}}

IMPORTANT:
- Theme names and descriptions must be in {language}
- If using existing theme, use the EXACT theme name from the list above
- Base your decision on conceptual fit, not just similarity scores

Return ONLY the JSON object."""

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