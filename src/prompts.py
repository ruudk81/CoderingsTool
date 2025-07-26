
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
You are an {language} language expert in analyzing written responses to open-ended question collected in surveys. 
Your task is to extract ALL ideas expressed in answering the following question: 
    
<survey_question>
{var_lab}
</survey_question>

Here is the respondent information and their response:

<respondent_info>
Respondent ID: {respondent_id}
Written response: {response}
</respondent_info>

Please follow these instructions carefully:
1. Read and analyze the provided text thoroughly.
2. Identify ALL ideas expressed in answering the question.
3. For each idea, provide a short descriptive phrase that captures the essence of the idea expressed in light of the survey question.
4. For deidentification purposes:
   - Remove all names of individuals mentioned in the text.
   - Use gender-neutral pronouns (they/them/their) when referring to any individuals.

Return the descriptive phrease the extracted ideas as a JSON array. Each item should include:
- `"respondent_id"`: exactly as provided
- `"idea_id"`: a string number ("1", "2", etc.)
- `"idea"`: the descriptive phrase capturing the essence of the idea in {language}

Here's an example of the input and desired output format:
    
<example>
Example input: 
Respondent ID: 123456789
Response: "Jared did a great job responding quickly to emails and turning in good work."

Example output:
[
  {{
    "respondent_id": "123456789",
    "idea_id": "1", 
    "idea": "Responded quickly to emails"
  }},
  {{
    "respondent_id": "123456789",
    "idea_id": "2",
    "idea": "Turned in good work"
  }}
]
</example>

Notice how the main ideas are summarized without including names or gendered pronouns.
You may include as many items in your list as necessary to capture all the ideas present in the write response.
"""

# =============================================================================
# STEP 6:  CODEBOOK GENERATION
# =============================================================================

INITIAL_CODEBOOK_CREATION_PROMPT = """
You are an expert qualitative data analyst specializing in applying codes to analyze qualitative data. 
Your task is to generate hypothetical codes that might be encountered when analyzing written answers to a specific survey question.

Here are the details for your task:
- Language to use: <language>{language}</language>
- Number of codes to generate: <n_codes>{n_codes}</n_codes>
- Survey question to analyze: <survey_question>{survey_question}</survey_question>

Generate {n_codes} hypothetical codes based on the given survey question. Each code should be a short phrase in plain {language} without examples. Do not include any additional text or explanations beyond the codes and their definitions.

Begin your list now using the following template:
{code_template}

Please provide your response as a JSON array of objects, where each object has "code" and "definition" fields.  
Here's an example of the input and desired output format:
<example>
[
  {{"code": "Technical difficulties", "definition": "Issues related to technology or system failures"}},
  {{"code": "Communication problems", "definition": "Challenges in exchanging information or understanding"}}
]
</example>

Ensure that your output adheres to this format exactly, with no additional text before or after the JSON array.
Remember to provide your output in the specified language: {language}.
Begin generating the codes now and present them in the required JSON format.
"""

INDUCTIVE_CODEBOOK_GENERATION_PROMPT = """
You are an expert {language} qualitative data analyst specializing in generating parsimonious codebooks for thematic analysis. 
Your task is to analyze a summary of {data_type}s collected in the context of {data_collection_context} and determine if the existing codebook is sufficient or if new codes are needed.

Here is the existing codebook:
<existing_codebook>
{codes}
</existing_codebook>

Now, here is the summary of {data_type}s to analyze:
<text_to_analyze>
{text}
</text_to_analyze>

Follow these steps in your analysis:

1. Examine the existing codebook carefully.
2. Read and understand the summary of {data_type}s.
3. Attempt to describe the main theme using existing codes.
4. If existing codes are insufficient, consider creating a new code.
5. Evaluate your suggestion using the criteria below.
6. Provide your final recommendation.

When considering whether to create a new code, use these evaluation criteria:

1. Parsimony: Have you made every effort to use existing codes or combinations before proposing a new one?
2. Abstraction Level: Is any proposed new code at an appropriate level of abstraction, consistent with existing codes?
3. Non-Redundancy: Have you avoided creating codes that significantly overlap with existing ones?

To illustrate non-redundancy, here is an example of redundant codes:
{redundancy_example}

Format your response as follows in {lagnuage}:

My expert analysis:
[Provide numbered, step-by-step reasoning for each of the six steps above]

My logical recommendation:
[State "No new codes needed" or provide new code(s) in this format:]
Code: [code]
Definition: [definition]

Remember, there is a significant penalty for creating redundant or unnecessary codes. 
Only create a new code if you are absolutely certain the existing ones are insufficient, even when combined or broadened. 
Your goal is to use the least number of new codes possible while still accurately representing the data.
"""

# =============================================================================
# STEP 7: THEME IDENTIFICATION  
# =============================================================================

THEME_IDENTIFICATION_PROMPT = """
You are a qualitative researcher expert specializing in thematic analysis in the {language} language. 
Your task is to analyze a list of codes and identify potential themes following the guidance of Braun and Clarke. 
The goal is to identify themes that help to answer the following research question:
    
<research_question>
{research_question}
</research_question>

You will be analyzing codes from writen answers in response to this question. 
Here is the list of codes to analyze:

<codes>
{codes}
</codes>

Follow these steps to identify themes:

1. Review the list of codes provided above.
2. Look for patterns and shared meanings among the codes. Consider how different codes might be combined based on underlying concepts or features of the data.
3. Identify overarching narratives that might represent broader themes or sub-themes.
4. Actively construe relationships among the codes and examine how these relationships inform potential themes.
5. Consider the importance and salience of potential themes. Remember, the number of codes supporting a theme is less important than whether the pattern communicates something meaningful that helps answer the research question.
6. Aim for themes that are distinctive yet coherent with the overall analysis. Themes may even be contradictory to each other.
7. Be willing to let go of codes or potential themes that don't fit the overall analysis. Consider creating a "miscellaneous" category for codes that don't fit elsewhere.
8. Strive for a balance in the number of themes - not so many that the analysis becomes unwieldy, but enough to fully explore the depth and breadth of the data.
9. For each theme, prepare a structured description including the theme name, its underlying concept, associated codes, and how these codes relate to each other and the overall theme.
10. Reflect on your analysis considering: themes that seem too broad or narrow, contradictions or unexpected patterns, need for subthemes, and codes that don't fit well into the current themes.

Present your analysis in JSON format in {language} with the following structure:
    
<json_structure>
{{
  "initial_observations": [
    "observation1"
  ],
  "suggested_themes": [
    {{
      "theme_name": "Theme 1",
      "concept": "Brief description of the underlying concept or narrative",
      "codes": [
        "Code 1"
      ],
      "relationship": "Brief explanation of how these codes relate to each other and the overall theme"
    }}
  ],
  "reflection": {{
    "broad_or_narrow_themes": "Discussion of any themes that seem too broad or too narrow",
    "contradictions_or_unexpected_patterns": "Description of any contradictions or unexpected patterns", 
    "potential_subthemes": "Discussion of any need for subthemes within the main themes",
    "unclassified_codes": "List of any codes that were not included in the proposed themes"
  }}
}}
</json_structure>

Use this JSON structure as a template. 
Expand on the template by adding as many observations, themes, and codes as necessary based on your analysis. 
Ensure that your response remains a valid JSON object. 
Do not include any text outside of this JSON structure.

Before you begin your analysis, take a moment to gather your expert thoughts. 
When you are ready, proceed with your analysis and present your findings in the specified JSON format.
"""

# =============================================================================
# LANGCHAIN v1
# =============================================================================

SYSTEM_MESSAGE_CODEBOOK = """
You are the world’s leading expert in qualitative data analysis, with a specialization in developing high-quality codebooks for thematic analysis of open-ended survey responses.
Your primary goal is to create a parsimonious codebook containing distinct, non-overlapping, and non-redundant codes. A codebook in this context is a structured list of code labels and their definitions, used to categorize meaningful segments of qualitative data.
I will provide you with task instructions, evaluation criteria, and formatting requirements—follow them precisely. After that, I will share the data to analyze. Use your expertise and the information provided to generate a clear, rigorous, and practical codebook that aligns with qualitative research standards.
You understand English prompts but will return your output in {language}.
"""

INITIAL_CODEBOOK_GENERATION = """
<existing codebook>
{code_text}
</existing codebook>

<text to analyze>
survey question : {survey_question}
responses: 
{cluster_text}
</text to analyze>

We are trying to determine whether or not an exsiting codebook is sufficient for analyzing one survey responses that you have been given in the <text to analyze> tag. 

Your important task is to analyze a summary of survey responses and determine if the theme discussed in the summary is already covered by the
codes in an existing codebook that will be given to you in the <existing codebook> tag or if instead the codebook needs one or more new code to cover the theme in the text to analyze.

You should complete your task by following these steps:

Step 1: Read existing codebook.
Examine the existing codebook given to you in the <existing codebook> tag. Describe what these codes are discussing.

Step 2: Read the summary of the survey responses.
Read the new summary of the survry responses given to you in the <text to analyze> tag and identify the main theme discussed in the summary.

Step 3: Try to use existing codebook.
Attempt to describe the main theme of the survey responses using one or more of the existing codes in the existing codebook. 
Think at a high level of abstraction and consider if any new themes could be subcategories of existing codes. 
If you determine that there is no need to create a new code, say ”No new codes needed”.

Step 4: Create new code if needed.
If in step 3 you discover that you are unable to use the current codes to describe the main theme in the summary of the survey response that you are analyzing, determine whether the existing codebook needs new labels to describe the summary in the <text to analyze> tag.

You should complete this determination by reasoning step−by−step. 
If you determine that a new code is necessary, explicitly justify why existing codes or combinations thereof are insufficient. 

Finally, generate a new code (or codes, if multiple ones are absolutely necessary) that captures the main concepts or themes discussed in the survey responses that you review.

Remember, you specialize in creating parsimonious codebooks and avoid creating redundant codes. 
Your goal is to use the least number of new codes possible while still accurately representing the data.
There is a VERY significant penalty for creating redundant or unnecessary codes, so you should only create a new code if you are ∗∗absolutely∗∗ certain the existing ones are insufficient, even when combined or broadened. 

If you decide to generate a new code, please provide:
− The code (a short phrase).
− A brief definition of what the label represents.

Remember, you output needs to in {language}"""

REVIEW_CODEBOOK_GENERATION = """
<code suggestion to review>
{code_text}
</code suggestionto review>

<text to analyze>
survey question : {survey_question}
responses: 
{cluster_text}
</text to analyze>

Evaluate the <code suggestion to review> tag for the summary of open responses in the <text to analyze> tag.

To guide your work, you must consider the following three evaluation criteria. 
These three evaluation criteria will be used by other famous expert qualitative data analysts to evaluate the quality of your work. 
In the reflection step, you must check whether you have satisfied each of these three criteria:

<evaluation criteria>
Evaluation Criteria 1. Parsimony: Have you made every effort to use existing codes or combinations of existing codes before proposing a new one?
Evaluation Criteria 2. Abstraction Level: Is any proposed new code at an appropriate level of abstraction, consistent with existing codes?
Evaluation Criteria 3. Non−Redundancy: Have you avoided creating codes that significantly overlap with existing ones?
To help illustrate what I mean by non−redundancy, here is an example of redundant codes and an explanation of their redundancy: 'Product quality' and 'Quality of product' are redundant because they refer to the same concept"
Use the evaluation criteria and these task instructions to help you in your step−by−step
reasoning for each of the preparation, analysis, and reflection steps given to you in these
instructions.
It is CRUCIAL TO REMEMBER that if you do not think a new code should be created, you must say ”No new codes needed”.
</evaluation criteria>

Present your final logical recommendation on a new line about any codes to create or whether none are needed on a new line.

<formatting instructions>
You should state ”My logical recommendation:” followed by your recommendation on yet another new line. 
Your recommendations can either be ”No new codes needed” if no new codes are needed or the actual codes you suggest adding to the codebook.
If you do think one or more new codes should be created, your response should start ’Code: ’followed by your code, then on a new line ’Definition: ’ followed by your definition for that code. 
For example:
Code: <code 1>
Definition: <definition 1>
</formatting instructions>

Remember that your output needs be in {language}
"""

# =============================================================================
# LANGCHAIN v2
# =============================================================================

SYSTEM_MESSAGE = """
Act as if you are the world's best language {language} qualitative data analyst with expertise in generating qualitative codebooks for thematic analysis. 
You specialize in creating parsimonious codebooks with non-overlapping and non-redundant codes."""

# PROMPT 1: Codebook Analysis  
CODEBOOK_ANALYSIS_PROMPT = """
{system_message}

Analyze this existing codebook for responses to the following survey question: "{survey_question}":

<existing_codebook>
{code_text}
</existing_codebook>

Provide a structured analysis:
1. The codes: [list of codes]
2. Description of overall idea expressed: [short description of the main idea captured by the codes in light of the survey question]
2. Main thematic categories: [list major themes]

Output in JSON format:
{{
  "codes" : ["code1", "code2"],
  "idea": [descrption here],
  "themes": ["theme1", "theme2"]
}}

Remember: the output needs to be in {language}
"""

# PROMPT 2: Response Summarization  
RESPONSE_SUMMARY_PROMPT = """
{system_message}

Summarize the responses presented below to this survey question: "{survey_question}"
You need to focus on main themes.

<responses>
{cluster_text}
</responses>

For each response, extract:
1. Primary theme/concept
2. Emotional tone
3. Key phrases or terminology
4. Unique aspects

Output format:
<summaries>
Response 1: {{theme: "...", tone: "...", key_phrases: [...], unique: "..."}}
Response 2: ...
</summaries>

Remember: the output needs to be in {language}
"""

# PROMPT 3: Integrated Matching and Recommendation
MATCH_AND_RECOMMEND_PROMPT = """
{system_message}

Given this codebook analysis:
<codebook_analysis>
{codebook_analysis}
</codebook_analysis>

And these response summaries:
<summaries>
{summaries}
</summaries>

Your task:
1. For each theme in the summaries, identify if existing codes can describe it
2. If existing codes are insufficient, explain specifically why
3. Only recommend new codes if absolutely necessary

Remember: You have a reputation for parsimony. Consider:
- Can existing codes be combined?
- Can existing codes be slightly broadened?
- Is the new concept truly distinct?

Output format:
<analysis>
Theme: [theme]
Existing code matches: [list codes that apply]
Coverage: [full/partial/none]
Gap analysis: [what's missing if partial/none]
Recommendation: [use existing/create new]
If new code needed:
  - Code: "..."
  - Definition: "..."
  - Justification: "..."
</analysis>

Remember: the output needs to be in {language}
"""

# PROMPT 4: Final Validation (with criteria)
VALIDATION_PROMPT = """
{system_message}

Review these code recommendations against strict criteria:

<recommendations>
{recommendations}
</recommendations>

Evaluation criteria:
1. Parsimony: Did we exhaust existing code options?
2. Non-redundancy: No overlap with existing codes?
3. Abstraction consistency: Same level as existing codes?

Example of redundancy to avoid:
{redundancy_example}

For each proposed new code:
- Score parsimony (0-10): [score] [reasoning]
- Score non-redundancy (0-10): [score] [reasoning]  
- Score abstraction (0-10): [score] [reasoning]
- Final decision: KEEP/REJECT

Final output:
<validated_codes>
[Only codes scoring 8+ on all criteria]
</validated_codes>

Remember: the output needs to be in {language}
"""

