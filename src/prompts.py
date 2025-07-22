
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
# STEP 4: ATOMIC IDEA EXTRACTION  
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
# STEP 6: CODEBOOK GENERATION  
# =============================================================================

PHASE1_DESCRIPTIVE_CODING_PROMPT = """
You are an expert in thematic analysis working in {language}.
Your task is to perform thematic coding on a cluster of similar segments from open-ended survey responses.
You will provide both a thematic label and a natural description that captures the common theme.

Here are the coding principles you must follow:
1. Understand the survey responses in light of the survey question
2. Stay close to the data: Use respondents' own concepts
3. Be descriptive: Capture what is said, not why
4. Be specific: Focus on the distinct pattern in these segments
5. Create single-themed labels: Each label should reflect ONE idea, topic, or theme

Survey question: 
<survey_question>
{survey_question}
</survey_question>

Cluster ID: 
<cluster_id>
{cluster_id}
</cluster_id>

Representative segments:
<representatives>
{representatives}
</representatives>

To complete this task:
1. Carefully read through all the representative segments
2. Identify the common theme or pattern expressed across these segments from the point of view of the survey question ("{survey_question}")
3. Create both a label and description that capture this common theme or pattern

Your output should be in the following format:
{{
  "cluster_id": "{cluster_id}",
  "segment_label": "YOUR_THEMATIC_LABEL_HERE",
  "segment_description": "Your natural-sounding description that captures what respondents in this cluster are expressing"
}}

Requirements:
- **segment_label**: ALL_CAPS_WITH_UNDERSCORES, up to 5 words (adjectives/nouns only)
- **segment_description**: Natural first-person or imperative statement in {language}

Remember to provide all output in {language}
"""

PHASE2_EXTRACT_ATOMIC_CONCEPTS_PROMPT = """
You are an expert in thematic analysis working in {language}.
Your task is to identify ALL atomic concepts present across descriptive codes derived from survey responses.

Survey question:
<survey_question>
{survey_question}
</survey_question>

Descriptive codes of response segments in sample:
<Descriptive codes>
{codes}
</Descriptive codes>

Instructions:
1. Review all descriptive codes carefully
2. Identify EVERY ATOMIC CONCEPT - the irreducible, single ideas that appear across responses
3. Focus on WHAT respondents are talking about (not WHY)
4. Be COMPLETELY EXHAUSTIVE - capture every meaningful concept, even if it appears only once
5. Do NOT group or merge similar concepts - keep them separate
6. Include specific subconcepts (e.g., Don’t just say “technology” if respondents specifically mention “smartphone”, “WiFi”, or “apps”.)

An atomic concept is:
- A single, indivisible idea (e.g., "battery life”, “screen size”, “weight”, “camera quality”)
- Cannot be meaningfully broken down further
- Clear and specific

IMPORTANT: 
- If respondents mention specific aspects (like “ticket price”, “seat comfort”, “departure time”, “onboard WiFi”), list these as separate concepts
- Don't combine concepts like "packaging" and "sustainable packaging" - keep them separate
- Include concepts even if they appear in only one or two codes

Begin with analytical notes:
<analytical_notepad>
Work through your analysis here:
- List EVERY distinct concept you find, no matter how specific
- Note which descriptive codes contain each concept
- Do not summarize or group - be comprehensive
[Your analysis]
</analytical_notepad>

Output JSON:
{{
  "analytical_notes": "Your complete working notes from above",
  "atomic_concepts": [
    {{
      "concept": "Concept name",
      "description": "What this concept represents", 
      "evidence": ["0", "2", "3", "15", "27"]  // List EVERY source ID where this concept appears - do NOT provide just examples
    }}
  ]
}}

CRITICAL: For the "evidence" field, you MUST list ALL source IDs where each concept appears, not just a few examples. Go through each descriptive code systematically and include every single occurrence.

Remember: 
- Keep concepts truly atomic
- List EVERY concept separately (e.g., “loading time”, “navigation menu”, “font size”, “error messages” as separate concepts if they appear)
- Use respondents' exact frame of reference
- When in doubt, include it as a separate concept
- The evidence list must be COMPLETE - every source ID where the concept appears

Return output in {language}.
"""

PHASE2_5_EVIDENCE_SCORING_PROMPT = """
You are an expert in qualitative analysis working in {language}.
Your task is to evaluate confidence scores for clusters that appear in concept evidence.

Survey question: {survey_question}

Atomic concepts with evidence:
{atomic_concepts}

Descriptive codes to evaluate:
{descriptive_codes}

INSTRUCTIONS:
Evaluate ONLY the cluster-concept pairs where the cluster ID appears in the concept's evidence list.

CONFIDENCE SCORING:
- 0.0-0.3: No match or very weak connection
- 0.4-0.6: Partial match or some elements relate  
- 0.7-0.8: Good match with most elements aligning
- 0.9-1.0: Excellent match, clearly belongs to this concept

Output JSON format:
{{
  "analytical_notes": "Your analysis process",
  "confidence_scores": [
    {{
      "cluster_id": 0,
      "concept": "Concept name",
      "confidence": 0.85,
      "reasoning": "Brief explanation"
    }}
  ]
}}

Return exactly {expected_scores} confidence scores in {language}.
"""

PHASE2_5_UNASSIGNED_SCORING_PROMPT = """
You are an expert in qualitative analysis working in {language}.
Your task is to evaluate unassigned clusters against ALL atomic concepts.

Survey question: {survey_question}

Atomic concepts:
{atomic_concepts}

Unassigned descriptive codes (not in any evidence):
{unassigned_codes}

INSTRUCTIONS:
Evaluate EVERY unassigned cluster against EVERY atomic concept.

CONFIDENCE SCORING:
- 0.0-0.3: No match or very weak connection
- 0.4-0.6: Partial match or some elements relate  
- 0.7-0.8: Good match with most elements aligning
- 0.9-1.0: Excellent match, clearly belongs to this concept

CRITICAL: You must evaluate ALL {num_unassigned} clusters against ALL {num_concepts} concepts.
Expected output: {expected_scores} confidence scores.

Output JSON format:
{{
  "analytical_notes": "Your analysis process",
  "confidence_scores": [
    {{
      "cluster_id": 10,
      "concept": "Concept name",
      "confidence": 0.25,
      "reasoning": "Brief explanation"
    }}
  ]
}}

Return exactly {expected_scores} confidence scores in {language}.
"""

PHASE2_5_CONCEPT_FOCUSED_SCORING_PROMPT = """
You are an expert in qualitative analysis working in {language}.
Your task is to evaluate ALL clusters against a SINGLE atomic concept for focused analysis.

Survey question: {survey_question}

TARGET CONCEPT:
Name: {concept_name}
Description: {concept_description}

ALL DESCRIPTIVE CODES TO EVALUATE:
{all_cluster_codes}

INSTRUCTIONS:
Evaluate EVERY cluster against ONLY the target concept "{concept_name}".

CONFIDENCE SCORING:
- 0.0-0.3: No match or very weak connection to {concept_name}
- 0.4-0.6: Partial match or some elements relate to {concept_name}
- 0.7-0.8: Good match with most elements aligning with {concept_name}
- 0.9-1.0: Excellent match, clearly belongs to {concept_name}

CRITICAL: You must provide exactly {expected_scores} confidence scores - one for each cluster listed above.

Output JSON format:
{{
  "analytical_notes": "Your analysis process for evaluating all clusters against {concept_name}",
  "confidence_scores": [
    {{
      "cluster_id": 0,
      "concept": "{concept_name}",
      "confidence": 0.85,
      "reasoning": "Brief explanation for this cluster's match to {concept_name}"
    }}
  ]
}}

Return exactly {expected_scores} confidence scores for concept "{concept_name}" in {language}.
"""

PHASE3_GROUP_CONCEPTS_INTO_THEMES_PROMPT = """
You are an expert in qualitative analysis working in {language}.
Your task is to group atomic concepts into meaningful themes.

Survey question:
<survey_question>
{survey_question}
</survey_question>

Atomic concepts identified:
<atomic_concepts>
{atomic_concepts}
</atomic_concepts>

Instructions:
1. Group atomic concepts that share a common theme, dimension, or aspect
2. Create clear, meaningful theme labels
3. Ensure each theme has a coherent focus
4. CRITICAL: Every single atomic concept from the input list must appear in your output

Guidelines for themes:
- Should represent a broad area of response
- Aim for 3-7 themes typically, but let the data guide you. Use more themes if the concepts are truly distinct.
- Each theme should contain related atomic concepts
- Theme names should be clear and descriptive

IMPORTANT REQUIREMENTS:
- You must account for ALL {total_concepts} atomic concepts provided
- If a concept doesn't fit well with others, put it in "unassigned_concepts"
- Do NOT merge, combine, or drop any concepts
- Use the exact concept names from the input list

Output JSON:
{{
  "themes": [
    {{
      "theme_id": "1",
      "label": "Theme Name",
      "description": "What this theme encompasses",
      "atomic_concepts": [
        {{
          "concept_id": "1.1",
          "label": "EXACT_CONCEPT_NAME_FROM_INPUT",
          "description": "What this concept covers"
        }}
      ]
    }}
  ],
  "unassigned_concepts": ["EXACT_CONCEPT_NAME_IF_UNASSIGNED"]
}}

VERIFY: Your output must contain exactly {total_concepts} concepts total (in themes + unassigned).
Return output in {language}.
"""

PHASE4_LABEL_REFINEMENT_PROMPT = """
You are an expert in creating clear, professional codebooks. 
Your task is to refine all labels and descriptions for maximum clarity and usability. 

You will be working in the following language:
<language>
{language}
</language>

Here is the survey question you will be working with:
<survey_question>
{survey_question}
</survey_question>

Now, here is the current codebook with cluster assignments:
<codebook_with_assignments>
{codebook_with_cluster_counts}
</codebook_with_assignments>

CRITICAL REQUIREMENT:
You MUST preserve the "Stable IDs" for each concept (like concept_1, concept_2, concept_other). Include the "stable_id" field in EVERY atomic concept in your response.

Your refinement goals are:
1. Create relevant labels: Clear (2-4 words) and relevant in light of the survey question
2. Provide clear descriptions: Explain how the label addresses the survey question
3. Ensure consistency: Use parallel structure and maintain a professional tone

Follow these guidelines:
- Use clear, non-technical language
- Ensure labels are distinct from each other
- Descriptions should help coders understand boundaries between concepts

DO NOT:
- Change the structure or assignments
- Merge or split any items
- Add new themes or concepts
- Move concepts between themes
- Change or omit the stable_id values

Provide your output in the following JSON format:
{{
  "refined_codebook": {{
    "themes": [
      {{
        "theme_id": "1",
        "label": "Refined Theme Label",
        "description": "Clear description of what this theme encompasses",
        "atomic_concepts": [
          {{
            "concept_id": "1.1",
            "label": "Refined Concept Label",
            "description": "Precise description of this atomic concept",
            "stable_id": "concept_1"
          }},
          {{
            "concept_id": "1.2",
            "label": "Another Refined Concept",
            "description": "Description of this concept",
            "stable_id": "concept_3"
          }}
        ]
      }}
    ],
    }}
  }}
}}
"""

# =============================================================================
# STEP 6: GATOS CODEBOOK GENERATION
# =============================================================================

INITIAL_CODEBOOK_CREATION_PROMPT = """Act as if you are the world's best qualitative data analysis. You specialize in applying codes to analyze qualitative data. I need your help. Your important task is to generate {k_to_start} hypothetical codes that one might encounter when analyzing {data_type}s from {data_collection_context}. 

You should format your response by filling in the template I give you at the end of these instructions, which is an enumerated list of {k_to_start} codes. The list should contain {k_to_start} short phrases with regular spacing between words written in plain English without examples. After the final code, you should stop writing so that it is easy for your response to be parsed for downstream tasks. 

Begin your list now using the following template:
{code_template}"""

INDUCTIVE_CODEBOOK_GENERATION_PROMPT = """Act as if you are the world's best qualitative data analyst with expertise in generating qualitative codebooks for thematic analysis. You specialize in creating parsimonious codebooks with non-overlapping and non-redundant codes. A codebook in this setting is a collection of labels and definitions for those labels that can be used to describe pieces of data in a qualitative research study. 

I need your help to create a qualitative codebook to analyze {data_type}s from {data_collection_context}. To aid you in this process, I am going to send you instructions in the <instructions> XML tag. Use the instructions to analyze the data in the <data to analyze> tag. You must follow these instructions using your expertise and data to analyze in the <data to analyze> XML tag. I will provide you the instructions first and then the data to analyze afterward. Be aware that your instructions contain task instructions, evaluation criteria, and formatting instructions, each in their respective XML tags.

<instructions>
<task instructions>
We are trying to determine whether or not an existing codebook is sufficient for analyzing one {data_type} that you have been given in the <text to analyze> tag. Your important task is to analyze one summary of {data_type}s collected in the context of {data_collection_context} and determine if the theme discussed in the {data_type} summary is already covered by the codes in an existing codebook that will be given to you in the <existing codebook> tag or if instead the codebook needs one or more new code to cover the theme in the text to analyze.

You should complete your task by following these steps:

Step 1: Codebook examination.
Read and understand the existing codebook. Study each code and its definition carefully to understand what themes are already covered.

Step 2: Current data examination.
Read and understand the summaries of the {data_type} in the <text to analyze> tag and identify the main theme discussed in the summary.

Step 3: Try to use existing codebook.
Attempt to describe the main theme of the {data_type} using one or more of the existing codes in the existing codebook. Think at a high level of abstraction and consider if any new themes could be subcategories of existing codes. If you determine that there is no need to create a new code, say "No new codes needed".

Step 4: Create new code if needed.
If in step 3 you discover that you are unable to use the current codes to describe the main theme in the summary of the {data_type} that you are analyzing, determine whether the existing codebook needs new labels to describe the summary in the <text to analyze> tag. You should complete this determination by reasoning step-by-step. If you determine that a new code is necessary, explicitly justify why existing codes or combinations thereof are insufficient. Finally, generate a new code (or codes, if multiple ones are absolutely necessary) that captures the main concepts or themes discussed in the {data_type}s that you review. Remember, you specialize in creating parsimonious codebooks and avoid creating redundant codes. Your goal is to use the least number of new codes possible while still accurately representing the data.

There is a VERY significant penalty for creating redundant or unnecessary codes, so you should only create a new code if you are **absolutely** certain the existing ones are insufficient, even when combined or broadened. If you decide to generate a new code, please provide:
- The code (a short phrase).
- A brief definition of what the label represents.

Step 5: Evaluate your suggestion.
To guide your work, you must consider the following three evaluation criteria. These three evaluation criteria will be used by other famous expert qualitative data analysts to evaluate the quality of your work. In the reflection step, you must check whether you have satisfied each of these three criteria:

<evaluation criteria>
Evaluation Criteria 1. Parsimony: Have you made every effort to use existing codes or combinations of existing codes before proposing a new one?
Evaluation Criteria 2. Abstraction Level: Is any proposed new code at an appropriate level of abstraction, consistent with existing codes?
Evaluation Criteria 3. Non-Redundancy: Have you avoided creating codes that significantly overlap with existing ones?

To help illustrate what I mean by non-redundancy, here is an example of redundant codes and an explanation of their redundancy:
{redundancy_example}

Use the evaluation criteria and these task instructions to help you in your step-by-step reasoning for each of the preparation, analysis, and reflection steps given to you in these instructions.

It is CRUCIAL TO REMEMBER that if you do not think a new code should be created, you must say "No new codes needed".
</evaluation criteria>

Step 6: Final recommendation.
Present your final logical recommendation on a new line about any codes to create or whether none are needed on a new line.
</task instructions>

<formatting instructions>
I will give you a template to use for your response. The main parts of the template are the following. First, your response should start with "My expert analysis:". Then, on a new line, you should write your logical step-by-step reasoning about the existing codes and the {data_type}s. This will include the two orientation steps, the two analysis steps, the reflection step, and the recommendation step. Your analysis notes should be succinct and formatted in a numbered list rather than long prose. This means that each step in your step-by-step reasoning should get its own line as if it were a premise in a proof. These notes should be logical, adhere perfectly to your task instructions, be concise, and be in a numbered list. Then, on another new line, you should state "My logical recommendation:" followed by your recommendation on yet another new line. Your recommendations can either be "No new codes needed" if no new codes are needed or the actual codes you suggest adding to the codebook.

If you do think one or more new codes should be created, your response should start 'Code: ' followed by your code, then on a new line 'Definition: ' followed by your definition for that code.

For example:
Code: <code 1>
Definition: <definition 1>
</formatting instructions>

This concludes your task and formatting instructions.
</instructions>

Now I will give you the data to analyze:

<data to analyze>
<existing codebook>
{codes}
</existing codebook>

And here is a summary of one {data_type} for you to analyze.
<text to analyze>
{text}
</text to analyze>
</data to analyze>

Now that you have meticulously studied the data to analyze using your task instructions, formatting instructions, and evaluation criteria, take a moment to gather your expert thoughts and observations. When you are ready, begin your flawless and logical step-by-step analysis using the instructions and evaluation criteria outlined above. Be sure to display your expertise in creating parsimonious codebooks and minimizing redundancy and use the full analysis template, provided below. Be sure to use spaces in any codes you write rather than concatenating words together (e.g., say "example code" rather than "examplecode"). Here is the template to use for your analysis. Begin your expert analysis when you are ready.

FULL ANALYSIS TEMPLATE:
My expert analysis:
Step 1 (codebook examination)
[your step 1 notes describing the existing code go here]

Step 2 (current data examination)
[your step 2 notes go here to identify the main theme in the {data_type}]

Step 3 (analysis part 1)
[your step 3 notes to describe main theme in the {data_type}s with existing codes here]

Step 4 (analysis part 2)
[your step 4 notes considering whether to create new code here, favoring parsimony and avoiding unnecessary code creation]

Step 5 (reflection on planned suggestions)
[your evaluation reflection notes here reviewing the evaluation criteria]

My logical recommendation:
[logical recommendation based on expert step-by-step reasoning about whether or not to create zero, one, or more than one new codes. These notes will reflect your reputation for only creating essential codes]"""

# =============================================================================
# STEP 7: THEME IDENTIFICATION  
# =============================================================================

THEME_IDENTIFICATION_PROMPT = """You are an expert qualitative researcher specializing in thematic analysis. Your task is to analyze a list of codes that will be given to you below in the <codes> tag and identify potential themes following the guidance of Braun and Clarke. The goal is to identify themes that help to answer the research question '{research_question}'.

Please follow these steps outlined in the <instructions> tag carefully.

<instructions>
Step 1. Review the list of codes provided below in the <codes> tag below. These codes are being used to analyze {data_type}s from {data_collection_context}.

Step 2. Look for patterns and shared meanings among the codes. Consider how different codes might be combined based on underlying concepts or features of the data.

Step 3. Identify overarching narratives that might represent broader themes or sub-themes.

Step 4. Remember that themes don't simply "emerge" from the data. Actively construe relationships among the codes and examine how these relationships inform potential themes.

Step 5. Consider the importance and salience of potential themes. Remember, the number of codes supporting a theme is less important than whether the pattern communicates something meaningful that helps answer the research question(s). On that note, remember that the research question for this research is {research_question}.

Step 6. Aim for themes that are distinctive yet coherent with the overall analysis. Themes may even be contradictory to each other.

Step 7. Be willing to let go of codes or potential themes that don't fit the overall analysis. Consider creating a "miscellaneous" category for codes that don't fit elsewhere.

Step 8. Strive for a balance in the number of themes - not so many that the analysis becomes unwieldy, but enough to fully explore the depth and breadth of the data.

Step 9. For each theme, prepare a structured description including the theme name, its underlying concept, associated codes, and how these codes relate to each other and the overall theme.

Step 10. Reflect on your analysis considering: themes that seem too broad or narrow, contradictions or unexpected patterns, need for subthemes, and codes that don't fit well into the current themes.

Step 11. Organize your analysis into a structured format with initial observations, an array of suggested themes (each as an object with name, concept, codes, and relationship), and your reflection.
</instructions>

Now that you have studied your instructions carefully, here is the list of codes to analyze to identify themes related to the research question "{research_question}":

<codes>
{codes}
</codes>

Proceed with your expert analysis, explaining your reasoning at each step. Present your analysis in JSON format with the following structure:

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

Use this JSON structure I have given you as a template. Expand on the template by adding as many observations, themes, and codes as necessary based on your analysis. Ensure that your response remains a valid JSON object. Do not include any text outside of this JSON structure.

Now that you have thoroughly read your task instructions, formatting instructions, and the codes to analyze, take a moment to gather your expert thoughts. Begin your analysis when you are ready."""
