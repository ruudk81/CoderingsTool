
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
# STEP 7:  CODEBOOK GENERATION
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

SYSTEM_MESSAGE = """
Act as a {language} qualitative data analyst specializing in thematic analysis.
You specialize in creating codebooks.
A codebook in this setting is a collection of labels and definitions for those labels that can be used to describe pieces of data collected in a survey with open-ended questions. 
"""

CODEBOOK_ANALYSIS_PROMPT = """
{system_message}
This time we will focus on written responses to the following survey question: "{survey_question}".

Given these codes from the codebook:
<existing_codebook>
{code_text}
</existing_codebook>

Analyze these codes to help future matching decisions:
1. What specific aspects of the survey question do these codes address?
2. What potential gaps exist - what types of responses might NOT fit these codes?

Output a concise analysis in {language} following this structure:
"Coverage: These codes collectively address [aspects of the survey question].

Gaps: Responses about [gap 1], [gap 2], and [gap 3] might not fit existing codes."

IMPORTANT: Return ONLY the analysis text following this exact format, no JSON or additional explanation.
"""

RESPONSE_SUMMARY_PROMPT = """
{system_message}

Analyze this cluster of semantically related ideas expressed in response to this survey question: "{survey_question}"

<clustered_ideas>
{cluster_text}
</clustered_ideas>

Extract the cluster's pattern to enable code matching:
1. **Core theme**: What is the central concept? Be specific.
2. **Abstraction level**: Is this cluster about a specific instance or a general pattern?
3. **Key components**: What are the 2-3 essential elements that define this cluster?
4. **Distinguishing features**: What makes this cluster different from other possible themes?

Output a concise analysis in {language} following this structure:
"[cluster description] at a [specific/general] level. The key components are [element 1], [element 2], and [element 3]. What distinguishes this cluster is [unique aspect]."

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
Notes: {codebook_analysis}
</existing_codes>
Note: These are the 5 codes nearest to this cluster's centroid embedding.

<clustered_ideas>
{clustered_ideas}
Notes: {summaries}
</clustered_ideas>
Note: These responses were grouped by HDBSCAN based on semantic similarity.

EVALUATION PROCESS:
1. Compare the cluster's core theme against each existing code
2. Assess coverage of themes in the clustered ideas by the existing codes
3. Decide that creating a new code is appropriate when current codes don't cover the clustered ideas enough
4. Always favor parsimony: use existing when in doubt 

CREATION CRITERIA:   
1. **Conceptual unity**: Does the new code represent ONE clear concept?
2. **Mutual exclusivity**: Would a coder be confused about when to use this vs other codes?
3. **Appropriate scope**: Is this trying to cover too much ground?
4. **Abstraction consistency**: Same level as existing codes?

Output ONE recommendation as valid JSON:
{{
  "cluster_core_theme": "identify from cluster analysis notes",
  "best_matching_codes": ["code1", "code2"],
  "coverage_assessment": {{
    "percentage": 0-100,
    "rationale": "explain what aspects are/aren't covered"
  }},
  "decision": "use_existing|modify_existing|create_new",
  "action_details": {{
    "codes_to_use": ["list if use_existing"] or null,
    "code_to_modify": "name if modify_existing" or null,
    "modification_suggestion": "how to broaden if modify_existing" or null,
    "new_code_name": "name if create_new" or null,
    "new_code_definition": "definition if create_new" or null
  }},
  "justification": "explain why this is the most parsimonious choice"
}}

IMPORTANT:
- Return ONLY the JSON object in {language}
- Fill only relevant fields in action_details based on your decision
- One cluster = one recommendation
"""

VALIDATION_PROMPT = """
{system_message}
This time we will focus on written responses to the following survey question: "{survey_question}".

You are reviewing a code recommendation for clustered ideas extracted from survey responses.

This is the extracted ideas: 
<clustered_ideas>
{clustered_ideas}
</clustered_ideas>
Note: These are the original survey responses that prompted this recommendation.

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
1. **Parsimony**: Were existing code options properly exhausted?
2. **Non-redundancy**: No overlap with existing codes?
3. **Justification alignment**: Does the recommendation match its reasoning?

Output a validation assessment in {language}:
{{
  "evaluation": {{
    "parsimony_reasoning": "assessment of whether existing options were exhausted",
    "redundancy_reasoning": "assessment of overlap with existing codes",
    "justification_reasoning": "assessment of decision alignment with reasoning"
  }},
  "decision": "APPROVE/REVISE/REJECT",
  "decision_rationale": "explanation for the overall decision",
  "validated_code": {{
    "code": "final code name (approved/revised) or null if rejected",
    "definition": "final definition (approved/revised) or null if rejected"
  }}
}}

IMPORTANT: 
- Return ONLY the JSON object in {language}
- APPROVE only if recommendation represents ONE clear, focused concept
- REVISE if concept needs refinement to improve focus or clarity (provide revised code in validated_code)
- REJECT if recommendation covers multiple unrelated concepts or creates confusion
- Always populate validated_code for APPROVE/REVISE decisions
- Ensure codes represent single, mutually exclusive themes
"""

# =============================================================================
# STEP 8: THEME IDENTIFICATION  
# =============================================================================

THEME_IDENTIFICATION_PROMPT = """
You are a qualitative researcher expert specializing in thematic analysis in the {language} language. 
Your task is to analyze a list of codes and identify potential themes following the guidance of Braun and Clarke. 
The goal is to identify themes that help to answer the following survey question:
    
<survey_question>
{survey_question}
</survey_question>

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

HIERARCHY_MAP_PROMPT = """
{system_message}

Create a complete hierarchy for these codes from survey responses:

<survey_question>
{survey_question}
</survey_question>

<codes>
{codes_batch}
</codes>

TASK: Create a 3-level hierarchy
1. Group codes into 2-4 DOMAINS (practical groupings)
2. Group domains into 1-3 THEMES (conceptual groupings)  
3. Include ALL codes - none can be left out

Output format (JSON):
{{
  "batch_id": {batch_number},
  "themes": [
    {{
      "theme_name": "Theme name in Dutch",
      "domains": [
        {{
          "domain_name": "Domain name in Dutch", 
          "codes": [
            {{
              "code_number": 1,
              "code_name": "Original code name"
            }}
          ]
        }}
      ]
    }}
  ]
}}

Return ONLY the JSON object in {language}.
"""

HIERARCHY_REDUCE_PROMPT = """
{system_message}

Merge these small hierarchies into one consolidated hierarchy:

<survey_question>
{survey_question}
</survey_question>

<batch_hierarchies>
{batch_hierarchies}
</batch_hierarchies>

TASK: Consolidate all hierarchies
1. Merge similar themes across batches
2. Merge similar domains within themes
3. Keep ALL codes - none can be lost
4. Create 4-7 final themes with logical domains

Simple rules:
- If themes have similar concepts, merge them
- If domains have similar purposes, merge them  
- Preserve every single code in the final structure

Output format (JSON):
{{
  "themes": [
    {{
      "theme_name": "Final theme name in Dutch",
      "theme_concept": "What this theme covers",
      "domains": [
        {{
          "domain_name": "Final domain name in Dutch",
          "domain_description": "What this domain covers", 
          "codes": [
            {{
              "code_number": 1,
              "code_name": "Original code name"
            }}
          ]
        }}
      ]
    }}
  ],
  "coverage_statistics": {{
    "total_codes": {total_codes},
    "classified_codes": {total_codes},
    "coverage_percentage": 100.0,
    "themes_count": 0,
    "domains_count": 0,
    "avg_codes_per_domain": 0.0
  }}
}}

Return ONLY the JSON object in {language}.
"""

