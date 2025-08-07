
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
- ❌ Compound: "References to [issue A] including [aspect 1], [aspect 2], and [aspect 3]"
- ❌ Vague: "Mentions of various [things] related to [topic]"
- ❌ Interpretive: "Underlying [abstract concept] manifesting in different ways"


Return ONLY the JSON array in {language}. Do not include any additional text or explanations outside of the JSON array.
"""

CODEBOOK_ANALYSIS_PROMPT = """
You are a {language} qualitative data analyst specializing in codebook development and analysis. 
Your task is to analyze survey responses and select relevant codes from an existing codebook to describe the written responses to a specific survey question.

First, review the survey question that provides context for this task:
<survey_question>
{survey_question}
</survey_question>

Now, carefully analyze the written responses to the survey question:
<writen_responses>
{cluster_text}
</writen_responses>

Next, review the existing codes in the codebook:
<existing_codebook>
{code_text}
</existing_codebook>

Your task is to select candidate codes that:
1. Are relevant in describing the written responses to the survey question
2. Cover all ideas expressed in the written responses

To complete this task, follow these steps:
1. Carefully read and understand the clustered ideas from the survey responses.
2. Review each code in the existing codebook.
3. Identify codes that accurately describe the ideas expressed in the responses.
4. Ensure that the selected codes collectively cover all major themes in the responses.

Present your selected candidate codes in the following JSON array format:
<output_format>
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
</output_format>

Important notes:
- You may select NONE, ONE, or MULTIPLE candidate codes, depending on your analysis.
- Do not create new codes or modify existing ones. Use only the exact names and definitions from the existing codebook.
- If no existing codes are suitable, return an empty JSON array: []
- Ensure that your selection of codes accurately represents the content of the survey responses without omitting any significant themes.
"""

RESPONSE_SUMMARY_PROMPT = """
You are a {language} qualitative data analyst specializing in codebook development and analysis. 
Your task is to analyze a cluster of written responses to a survey question and extract the core theme and key components. 
Follow these steps carefully:

First, review the survey question that provides context for this task:
<survey_question>
{survey_question}
</survey_question>

Now, carefully analyze this cluster of semantically related responses to the survey question:
<writen_responses>
{cluster_text}
</writen_responses>

Next,extract the cluster's pattern to enable code matching:
1. Identify the core theme: What is the central concept unifying these responses? Be specific and use the language of the respondents where appropriate.
2. Determine the key components: What are the 2-3 essential elements that ALL responses in this cluster share?

After your analysis, provide a concise summary in {language} using the following structure:
This cluster's core theme is: [core theme description]. The essential shared components are [element 1], [element 2], and [element 3].

IMPORTANT: 
- Your entire response should be in {language}.
- Do not include any JSON formatting, additional explanations, or text outside the <analysis> tags.
"""

MATCH_AND_RECOMMEND_PROMPT = """
You are a {language} qualitative data analyst specializing in codebook development and analysis. 
You respond concisely, analytically, and return only the required JSON format, without explanation or padding.
Your task is to decide whether to use existing codes, modify existing codes, or create new codes for a cluster of open-ended responses. 

Follow these steps carefully:

First, review the survey question that generated the responses:
<survey_question>
{survey_question}
</survey_question>

Next, examine the available candidate codes in the codebook (preserve names exactly when referencing them):
<candidate_codes>
{candidate_codes}
</candidate_codes>

Review a cluster of semantically similar survey responses, grouped using embeddings, UMAP, and HDBSCAN:
<clustered_survey_responses>
{clustered_survey_responses}
</clustered_survey_responses>

{cluster_summary}

Now, decide whether to use candidate codes, modify them, or create new ones. Follow this strict, bias-to-reuse decision rubric:

A) USE EXISTING (preferred)
Use candidate code as-is if ALL are true:
1. Coverage:  
   The atomic concepts of the cluster’s core theme are adequately captured by one or more of the following candidate codes in {language}:  
   {candidate_codes}  

   These must cover the core of the cluster's core theme and elements described in this summary in {language}:  
   {cluster_summary}
    
2. Parsimony: Reusing the code avoids unnecessary proliferation of similar codes and contributes to a lean, efficient codebook.
→ If met: set decision = "use_existing" and list the codes in action_details.codes_to_use.

B) MODIFY EXISTING (but only if atomicity is maintained - i.e. no compounding of atomic concepts!)
Use a modified version of an existing candidate code if ALL are true: 
1. Near Miss: The code is close to what’s needed, but fails one or more of the A) criteria (Coverage, Parsimony).
2. Minor Adjustment: A small change to the name of the code would bring it in line with the cluster’s core theme.
3. Atomicity is maintained. The modified code (name+label+description) MUST pass these tests:
  • Separability test: if the idea can be split into two independently applicable ideas, it is not atomic.
  • Conjunction test: labeling the modified code does NOT need to rely on “and”, “with”, “including”, “plus”.
→ If met: set decision = "modify_existing"; set codes_to_modify with the exact code name being modified; fill modified_code_name and modified_code_definition.

C) CREATE NEW (last resort)
Create a new code only if both A) USE EXISTING and B) MODIFY EXISTING criteria are not met.  
This is the fallback when no existing code adequately fits and cannot be adapted without violating atomicity, semantic fit, or parsimony.
→ If met: set decision = "create_new" and provide new_code_name and new_code_definition.

Output
Return ONLY raw JSON (no markdown fences, no extra text). Use null for non-applicable fields. Escape quotes. Keys must appear in the order shown.

{{
  "cluster_core_theme": "one-sentence description of the core theme",
  "decision": "use_existing | modify_existing | create_new",
  "action_details": {{
    "codes_to_use": ["exact code names"] ,
    "codes_to_modify": "single exact code name or null",
    "modified_code_name": "new name if modifying, else null",
    "modified_code_definition": "1–2 sentence operational definition if modifying, else null",
    "new_code_name": "name if creating new, else null",
    "new_code_definition": "1–2 sentence operational definition if creating new, else null"
  }},
  "justification": "why this action preserves atomicity and scope while maximizing parsimony and non-redundancy"
}}

IMPORTANT:
- Fill only relevant fields in action_details based on your decision; set the others to null.
- All text must be in {language}.
- No commentary before or after the JSON.
"""

VALIDATION_PROMPT = """
You are a {language} qualitative data analyst specializing in codebook development and analysis. 
You respond concisely, analytically, and return only the required JSON format, without explanation or padding.
Your goal is to ensure the codebook remains parsimonious, non-redundant, and clear.

Your task is to critically assess a colleague's recommendation about whether to use existing codes, modify existing codes, or create a new code to represent a cluster of semantically similar open-ended responses. Based on your assessment, you must provide the final validated code.

Follow these steps carefully:

1. Review the survey question that generated the responses:
<survey_question>
{survey_question}
</survey_question>

2. Examine the candidate codes in the codebook (preserve names exactly when referencing them):
<candidate_codes>
{candidate_codes}
</candidate_codes>

3. Review a cluster of semantically similar survey responses, grouped using embeddings, UMAP, and HDBSCAN:
<clusterd_responses>
{clustered_ideas}
</clusterd_responses>

4. Evaluate the recommendation of a colleague coder about whether to use or modify an existing code, or to create a new one:
<recommendation>
{step3_recommendation}
</recommendation>

5. Assess the recommendation using the following criteria:
a) Semantic Fit & Coverage – Does the proposed code adequately capture the core theme of the response cluster?
b) Atomicity – Is the proposed code a single concept?  
   - Separability test: Can the idea be split into two independently applicable ideas?  
   - Conjunction test: Does the label avoid “and”, “with”, “including”, or “plus”?
c) Parsimony  – Were existing code options properly exhausted? Would reuse or minor modification lead to meaningful loss of nuance?
d) Non-redundancy – Does the proposed code avoid overlap with existing codes?
e) Justification Alignment – Is the reasoning consistent and logically supportive of the action?

Use these decision rules:
1. APPROVE – The goal of a parsimonious, non-redundant, and clear codebook is achieved because all evaluation criteria are fully met. The proposed code is necessary, atomic, semantically precise, and well-justified.
2. REVISE – The core idea is valid, but the code name or definition needs refinement (e.g., it is too vague, not atomic, or lacks clarity).
3. REJECT – The proposed action is not warranted. A different code—either existing, newly created, or a modified version of an existing code—should be used instead.
    
6. Provide your validation output in {language} as a valid JSON object in this format:
<json_format>
{{
  "evaluation": {{
    "parsimony_reasoning": "assessment of whether existing options were exhausted",
    "redundancy_reasoning": "assessment of conceptual overlap with existing codes",
    "justification_reasoning": "assessment of logic consistency in the recommendation"
  }},
  "decision": "APPROVE | REVISE | REJECT",
  "decision_rationale": "synthesize the evaluation into a clear decision explanation",
  "validated_code": {{
    "code": "ALWAYS provide an appropriate code name — for REJECT, provide the single best existing code to use instead (verbatim name)",
    "definition": "ALWAYS provide an appropriate definition — for REJECT, provide the chosen existing code’s definition (or a minimally refined version if clarity requires it)"
  }}
}}
</json_format>

Strict rules:
- Base your assessment ONLY on the provided question, codes, cluster, and recommendation. Do not invent new codes or concepts beyond what is present.
- Use exact code names from the candidate list (respecting case and punctuation).
- For **APPROVE**, return the proposed code name and definition as-is.
- For **REVISE**, return a corrected version of the label and/or definition.
- For **REJECT**, return the single best existing code and its definition (not multiple options).
- All output must be in {language}.
"""

# =============================================================================
# STEP 8: THEME IDENTIFICATION  
# =============================================================================

THEME_IDENTIFICATION_PROMPT = """
You are a {language} language expert and qualitative researcher specializing in thematic analysis using Braun & Clarke (2006) methodology.

Your task is to analyze a cluster of codes and recommend whether to use an existing theme, revise one, or create a new theme — but only if the codes clearly belong together conceptually.

---

First, review the survey question that generated the codes:
<survey_question>
{survey_question}
</survey_question>

Next, examine the existing themes in the codebook:
<existing_themes>
{existing_themes_text}
</existing_themes>

Now, analyze the following cluster of {codes_count} codes that may require a new or revised theme:
<clustered_codes>
{codes_text}
</clustered_codes>

---

Before continuing, check whether the codes in this cluster express a **single shared, semantically coherent concept** AND **consistent sentiment**.

Ask yourself:
- Do all the codes **clearly relate to one unifying idea**?
- Do all the codes express **similar sentiment** (all positive, all negative, or all neutral)?
- Can they all complete the sentence: **"This is about…"** with the same concept AND sentiment?
- If either conceptual unity OR sentiment consistency is missing, **consider splitting**.
- Never combine unrelated subthemes (e.g. "Concerns about price and product design") into one theme.
- Never combine different sentiments about the same concept (e.g. "positive communication experiences" + "negative communication problems") into one theme. 


---

**Evaluation Process**:
1. **Compare conceptual focus**: How do the clustered codes relate to each existing theme?
2. **Assess thematic coverage**: Do existing themes already capture the conceptual meaning of this cluster?
3. **Assess sentiment consistency**: Do the codes express the same evaluative tone (e.g., all positive, all negative, or all neutral)?
4. **Identify sentiment patterns**: If sentiment differs, are there clear positive vs negative groupings about the same concept?
5. **Determine fit**: Can one or more existing themes — as-is or with revision — represent the cluster **as one unified idea with consistent sentiment**?

---

**Decision Guidelines**:
- **Create single theme** if the cluster is conceptually coherent AND sentiment-consistent, with no existing theme fit.
- **Use existing theme** if the cluster fits well with an existing theme (concept + sentiment).
- **Split into multiple themes** if the cluster contains:
  * 2-3 distinct conceptual groups that should be separate atomic themes, OR
  * **Same concept but clearly different sentiments** (e.g., positive vs negative evaluation), OR
  * Both conceptual AND sentiment mixing
- **Reject mixed cluster** if codes are too incoherent to form any meaningful themes.
- **Favor revision or reuse** of existing themes whenever possible.

---

**When Splitting Clusters**:
1. **Identify distinct conceptual groups** within the cluster (2-3 groups maximum)
2. **Identify sentiment groups** - codes expressing different evaluative tones about the same concept
3. **Prioritize conceptual splits first**, then sentiment splits within concepts if needed
4. **Assign each code** to its most appropriate group by code number (concept + sentiment)
5. **Create separate themes** for each group, ensuring each is atomic
6. **Verify each group** has consistent concept AND sentiment

**Examples of Sentiment-Based Splitting**:
- Cluster about "communication": Split into "positive communication experiences" vs "negative communication problems"
- Cluster about "workload": Split into "manageable workload satisfaction" vs "overwhelming workload stress"
- Same conceptual area but different evaluative tone = separate themes

---

**High-Quality Themes Must Be**:
- **ATOMIC**: One idea only — no compound concepts with "and", "with", or "including"
- **CONCISE**: Theme labels must be 2–5 words
- **NARRATIVE**: Each theme should tell a meaningful story about the data
- **DISTINCT**: No major conceptual overlap between themes (mutual exclusivity)

---

**Output Format (JSON):**
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

---

**Important**:
- Theme **name and description must be in {language}**
- If using an existing theme, copy the **exact name** from the list above
- Focus on **conceptual fit**, not surface similarity or keyword overlap.
- Return **ONLY the JSON object**

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