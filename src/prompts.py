
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

CANDIDATE_CODE_SELECTION_PROMPT = """
You are a {language} qualitative data analyst specializing in generating qualitative codebooks for thematic analysis. 
Your task is to select appropriate codes for describing and labelling the theme or themes discussed in a summary of clustered survey responses. 

IMPORTANT: Clusters may contain **multiple distinct themes** that require different codes. Your job is to identify ALL relevant codes for ALL themes present.

First, review the survey question that generated the responses:
<survey_question>
{survey_question}
</survey_question>

Now, examine a summary of a cluster of semantically similar survey responses with 1 or more themes:
<cluster_summary>
{cluster_summary}
</cluster_summary>

Finally, examine the existing codes in the codebook:
<existing_codebook>
{code_text}
</existing_codebook>

To select appropriate candidate codes, follow these steps:
1. Identify **ALL distinct themes** present in the cluster summary (there may be 1 or more)
2. For **EACH theme**, attempt to find matching existing codes
3. Select the codes, if any, that are relevant in capturing the themes
4. Be selective: only select suitable codes for our codebook
5. Be comprehensive: ensure no theme is left uncoded
6. Present your selection in the JSON array format described below

<json_output_format>
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
</json_output_format>

IMPORTANT:
- You may select NONE, ONE, or MULTIPLE candidate codes per theme
- Do not create new codes or modify existing ones
- If a concept has no suitable existing codes, that's acceptable - return what you can find
- Multi-theme clusters are normal and expected - code each theme appropriately
- Output ONLY the JSON array - no other text
"""

CLUSTER_SUMMARY_PROMPT = """
You are a {language} data analyst specializing in thematic analysis of open-ended survey responses.
Your task is to analyze a cluster of responses grouped by HDBSCAN based on embedding similarity, and identify the theme(s) present.

IMPORTANT: Mathematical similarity (via embeddings) does not guarantee conceptual unity. Clusters may legitimately contain:
- One unified theme
- Multiple distinct themes (2 or more)
- No coherent theme (just superficial similarity)

A "theme" is an atomic, shared concept that meaningfully answers the survey question. A valid theme:
- Appears in at least 30% of the cluster's responses
- Represents a distinct, separable idea
- Cannot be split into smaller independent concepts
- Has explanatory power for why these responses matter in light of the survey question

Follow these steps:

1. Review the survey question:
<survey_question>
{survey_question}
</survey_question>

2. Look for patterns and shared meanings across responses in this cluster :
<cluster_text>
{cluster_text}
</cluster_text>

3. Active Theme Construction Process:
   a) **Look for an overarching narrative** that the response ideas might communicate
      - What story are these responses collectively telling?
      - What deeper message connects seemingly different surface expressions?
   
   b) **Realize that themes don't simply "emerge"** — actively construe relationships and meaning
      - Don't just list what you see; interpret what it means
      - Connect dots between responses that might not be obviously related
      - Consider latent meanings, not just manifest content
   
   c) **First, attempt to identify ONE common theme** from the overarching narrative
      - Can you find a single conceptual thread that weaves through most responses?
      - Is there a unifying perspective or concern being expressed in different ways?
      - Push yourself to find unity before accepting fragmentation
   
   d) **Only conclude multiple themes if** they are clearly distinct and independently meaningful
      - Multiple themes should represent fundamentally different ideas
      - Each theme must stand alone and appear in ≥30% of responses
      - Themes should not just be different aspects of the same underlying concept

4. Make your determination after active construction:
   - ONE THEME: You've found an overarching narrative that unifies ≥70% of responses
   - MULTIPLE THEMES: Despite seeking unity, 2+ distinct narratives each appear in ≥30% of responses
   - NO THEME: No narrative emerges that connects ≥30% of responses meaningfully

5. Report your result in {language} using ONLY this JSON structure:

{{
  "cluster_summary": "Brief summary of what this cluster contains",
  "themes": [
    "Theme 1: [concise theme statement]. This is meaningful because [core rationale].",
    "Theme 2: [concise theme statement]. This is meaningful because [core rationale]."
  ]
}}

For NO_THEME cases, return:
{{
  "cluster_summary": "This cluster lacks coherent themes. [Brief explanation]",
  "themes": []
}}

CRITICAL OUTPUT RULES:
- Output ONLY valid JSON - no text before or after
- For NO_THEME: set themes array to []
- For ONE_THEME: include one theme in the array
- For MULTIPLE_THEMES: include all distinct themes in the array
- All text fields must be in {language}
- theme_statement should be 5-15 words, optimized for embedding
- Keep theme_statement focused on the core concept for better embedding/search
"""

CODE_GENERATION_PROMPT = """
You are a {language} data analyst specializing in qualitative research and thematic coding. 
Your expertise lies in developing parsimonious, clearly defined, non-redundant codebooks at a consistent level of abstraction.

CRITICAL: Clusters may contain **multiple distinct themes**. Each theme should be evaluated and coded separately. Do not force multiple themes into a single code.

Your task is to analyze a cluster summary and determine the appropriate coding strategy for EACH distinct theme present.

<context>
Survey Question:
{survey_question}

Summary about a cluster of semantically related responses:
{cluster_summary}

Available Codes:
{candidate_codes}
</context>

Follow this process:

Step 1: Identify Distinct Themes
- Carefully examine the cluster summary
- Identify how many distinct themes are present (1, 2, or more)
- Each theme should be atomic and independently meaningful

Step 2: For EACH Theme, Evaluate Coding Options
For each identified theme, determine whether to:
a) Use existing code(s) that adequately capture this theme
b) Modify an existing code to better fit this theme
c) Create a new code for this theme

Decision Guidelines Per Theme:
- Use `use_existing` if current codes fully describe the theme
- Use `modify_existing` when a code captures 60-90% of the theme's meaning
- Use `create_new` if no code captures >60% of the theme

Atomicity Rules (Critical):
- Each code must describe ONE single, indivisible theme
- Multiple themes require multiple separate codes
- Never combine distinct themes into one code using "and", "with", etc.

Output instructions:
Return only valid raw JSON. Structure your response to handle multiple themes:

{{
  "cluster_analysis": {{
    "number_of_themes": "integer (1, 2, or more)",
    "theme_descriptions": [
      "Brief description of theme 1",
      "Brief description of theme 2 (if applicable)"
    ]
  }},
  "coding_decisions": [
    {{
      "theme_number": 1,
      "theme_description": "what this theme is about",
      "decision": "use_existing | modify_existing | create_new",
      "action_details": {{
        "codes_to_use": ["exact code names"],
        "codes_to_modify": "single exact code name or null",
        "modified_code_name": "new name if modifying, else null",
        "modified_code_definition": "1-2 sentence definition if modifying, else null",
        "new_code_name": "name if creating new, else null",
        "new_code_definition": "1-2 sentence definition if creating new, else null"
      }},
      "justification": "why this action is appropriate for this specific theme"
    }},
    {{
      "theme_number": 2,
      "theme_description": "what this second theme is about",
      "decision": "use_existing | modify_existing | create_new",
      "action_details": {{...}},
      "justification": "..."
    }}
  ],
  "overall_justification": "why treating these as separate themes preserves atomicity and improves codebook quality"
}}

IMPORTANT:
- Process EACH distinct theme separately in the coding_decisions array
- Never merge distinct themes into a single code
- All text must be in {language}
- Output ONLY valid JSON, no other text
"""

VALIDATION_PROMPT = """
You are a {language} qualitative data analyst specializing in codebook development and evaluation. 

CRITICAL: You are validating recommendations that may involve **multiple distinct themes and multiple codes**. Each theme/code pair must be evaluated separately.

Your task is to assess your colleague's coding recommendations for a cluster that may contain multiple themes.

Step 1: Review the survey question:
<survey_question>
{survey_question}
</survey_question>

Step 2: Examine the candidate codes:
<candidate_codes>
{candidate_codes}
</candidate_codes>

Step 3: Review the cluster summary:
<cluster_summary>
{cluster_summary}
</cluster_summary>

Step 4: Evaluate your colleague's recommendations:
<recommendation>
{step3_recommendation}
</recommendation>

Step 5: Assess EACH coding decision using these criteria:
a) **Theme Separation** – Are distinct themes appropriately identified and separated?
b) **Semantic Fit** – Does each proposed code capture its target theme?
c) **Atomicity** – Is each code truly atomic (single, indivisible theme)?
d) **Parsimony** – Were existing options properly considered?
e) **Non-redundancy** – Do the codes avoid overlap?

Decision Rules for EACH code:
- **APPROVE** – The code is necessary, atomic, and well-justified
- **REVISE** – The theme is valid but the code needs refinement
- **REJECT** – Use an existing code instead
- **MERGE** – Multiple codes should be combined (only if they represent the same theme)
- **SPLIT** – A proposed code tries to cover multiple themes and should be split

Validation Output:
Return your validation as raw JSON:

{{
  "theme_assessment": {{
    "number_of_themes_identified": "integer",
    "theme_separation_valid": "true/false",
    "theme_separation_reasoning": "are the themes truly distinct or should they be merged/split?"
  }},
  "code_validations": [
    {{
      "theme_number": 1,
      "theme_description": "what theme is being coded",
      "original_recommendation": "what was proposed",
      "evaluation": {{
        "semantic_fit": "assessment",
        "atomicity": "assessment",
        "parsimony": "assessment",
        "redundancy": "assessment"
      }},
      "decision": "APPROVE | REVISE | REJECT | MERGE | SPLIT",
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
      "evaluation": {{...}},
      "decision": "...",
      "decision_rationale": "...",
      "validated_code": {{...}}
    }}
  ],
  "overall_validation": {{
    "all_themes_coded": "true/false",
    "final_code_count": "integer",
    "summary": "brief summary of the validation outcome"
  }}
}}

Strict rules:
- Validate EACH theme/code pair separately
- Ensure atomic codes (one theme per code)
- Don't force unrelated themes into single codes
- Allow multiple codes when multiple distinct themes exist
- Output ONLY valid JSON, no other text
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