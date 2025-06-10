
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
# STEP 4: SEGMENTATION AND DESCRIPTION
# =========================================

SEGMENTATION_PROMPT = """
You are a helpful {language} language expert in analyzing survey responses. 
Your task is to segment free-text survey responses into distinct, standalone segments with the highest quality to eliminate the need for further refinement.

First, here is the survey question:
<survey_question>
{var_lab}
</survey_question>

Now, here is the response you need to segment:
<respondent_info>
Respondent ID: {respondent_id}
Response: {response}
</respondent_info>

Your task is to break the response into the smallest meaningful standalone units, where each segment represents EXACTLY ONE:
- Opinion
- Preference  
- Issue
- Topic
- Idea
- Response pattern

Follow these ENHANCED segmentation rules:
1. Split at conjunctions (and, or, but, also) when they connect DIFFERENT ideas or topics
2. Split listed items into separate segments (e.g., "milk and sugar" → "milk", "sugar")
3. When items share context, preserve that context in each segment:
   Example: "I like milk and sugar in my coffee" →
   - "I like milk in my coffee"
   - "I like sugar in my coffee"
4. Use the respondent's exact words - do not paraphrase or correct
5. Keep meaningless responses (e.g., "Don't know", "?") as a single segment
6. CRITICAL: Each segment must be completely standalone and meaningful
7. CRITICAL: Avoid segments that would need further splitting - be thorough now
8. Handle complex compound statements carefully to ensure clean separation

Your output must be a JSON array with these fields for each segment:
- "respondent_id": The exact respondent ID provided
- "segment_id": A sequential number as string ("1", "2", etc.)  
- "segment_response": The exact segmented text with necessary context preserved

Example output format:
[
  {{
    "respondent_id": "{respondent_id}",
    "segment_id": "1", 
    "segment_response": "Betere interactie met de docent."
  }},
  {{
    "respondent_id": "{respondent_id}",
    "segment_id": "2",
    "segment_response": "Ouders moeten betrokken zijn."
  }}
]

Before providing your final output, think through the segmentation process carefully. 
Consider how many segments the response should be divided into and apply the enhanced segmentation rules thoroughly.

After your analysis, provide the final segmented output formatted as a JSON array as specified above.
"""

CODING_PROMPT = """
You are a {language} language expert in thematic analysis of survey responses.
Your task is to generate ONLY thematic labels for segments from responses to the survey question: "{var_lab}"

Here are the coded segments you need to label:
<segments>
{coded_segments}
</segments>

For each segment, you will:
1. Keep the original respondent_id, segment_id, and segment_response
2. Add ONLY a segment_label (thematic label)

Requirements for segment labels:
- Create a concise label of up to 5 words total, using ONLY ADJECTIVES AND NOUNS in {language}
- Capture the CENTRAL MEANING of the segment in relation to the survey question
- ONLY return labels that reflect ONE idea, topic, concern, issue, or theme
- NEVER return multi-headed labels or combinations of multiple ideas
- Format: ALL_CAPS_WITH_UNDERSCORES
- Examples: "ONBETROUWBARE_DIENSTREGELING", "BETERE_GEBRUIKSERVARING", "DOCENTCONTACT"
- Language: {language}

Your output must be a JSON array with these fields for each segment:
- "respondent_id": The original respondent ID
- "segment_id": The original segment ID  
- "segment_response": The original segment text
- "segment_label": Your thematic label in ALL_CAPS_WITH_UNDERSCORES

Example output:
[
  {{
    "respondent_id": "12345",
    "segment_id": "1",
    "segment_response": "Betere interactie met de docent.",
    "segment_label": "DOCENTCONTACT"
  }}
]

Ensure all labels are written in {language} and focus solely on creating high-quality thematic labels.
"""

DESCRIPTION_PROMPT = """
You are a {language} language expert in creating natural-sounding descriptions for thematic codes.
Your task is to generate descriptions for labeled segments from responses to the survey question: "{var_lab}"

Here are the labeled segments you need to describe:
<labeled_segments>
{labeled_segments}
</labeled_segments>

For each segment, you will:
1. Keep the original respondent_id, segment_id, segment_response, and segment_label
2. Add ONLY a segment_description (natural-sounding description)

Requirements for segment descriptions:
- Rewrite the segment as a natural-sounding first-person response to the survey question
- Use the segment_label as context to understand the thematic meaning
- Make sure it sounds like something a person would actually say when answering the question
- Use a direct, conversational or instructional tone:
  - If the segment is a suggestion: use an imperative tone (e.g., "Maak...", "Laat...")
  - If the segment expresses a wish or opinion: use first-person (e.g., "Ik wil...", "Ik vind...")
- NEVER rephrase the segment as a third-person summary

CRITICAL CONSTRAINT - FOLLOW EXACTLY:
- NEVER add information not explicitly stated in the original segment

BEFORE writing each description, ask yourself:
- "Is every word in my description based on something explicitly stated in the segment?"
- "Am I adding any explanations, reasons, or context not in the original?"
- "Would someone reading just the segment come to the exact same description?"

Language: {language}

Your output must be a JSON array with these fields for each segment:
- "respondent_id": The original respondent ID
- "segment_id": The original segment ID
- "segment_response": The original segment text  
- "segment_label": The original segment label
- "segment_description": Your natural-sounding description

Example output demonstrating CRITICAL CONSTRAINT:

CORRECT examples (stay within segment content):
[
  {{
    "respondent_id": "12345",
    "segment_id": "1", 
    "segment_response": "Betere interactie met de docent.",
    "segment_label": "DOCENTCONTACT",
    "segment_description": "Ik wil betere interactie met de docent."
  }},
  {{
    "respondent_id": "12346",
    "segment_id": "1",
    "segment_response": "De app is soms lastig te navigeren.",
    "segment_label": "GEBRUIKSVRIENDELIJKHEID_APP",
    "segment_description": "Ik vind de app niet altijd gebruiksvriendelijk."
  }}
]

"""

# =============================================================================
# STEP 6: HIERARCHICAL LABELING - 6 PHASES
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

PHASE2_LABEL_MERGER_PROMPT = """
You are an expert in qualitative research working in {language}. 
Your task is to evaluate and merge descriptive labels from survey response clusters that are semantically identical or meaningfully equivalent in the context of the survey question.

Here is the survey question:
<survey_question>
{survey_question}
</survey_question>

Here are the current labels to evaluate for merging:
<labels>
{labels}
</labels>

Merge labels (YES) ONLY IF:
- Labels are semantically identical or meaningfully equivalent in light of the survey question ("{survey_question}")

Important guidelines:
- Be conservative - when in doubt, keep clusters separate.
- Consider the context of the survey question when evaluating semantic similarity.
- Pay attention to nuances in meaning that might be important to preserve.

For merged labels:
1. Choose a label that represents the merged labels as the new merged label
2. Assign a new sequential cluster ID starting from 0
3. List all original cluster IDs that are being merged

{{
  "merged_groups": [
    {{
      "new_cluster_id": 0,
      "merged_label": "Best Representative Label",
      "original_cluster_ids": [1, 5, 12]
    }},
    {{
      "new_cluster_id": 1,
      "merged_label": "Another Representative Label", 
      "original_cluster_ids": [3, 8]
    }}
  ],
  "unchanged_labels": [
    {{
      "new_cluster_id": 2,
      "label": "Unique Label",
      "original_cluster_id": 7
    }}
  ]
}}

Provide your decisions in the required JSON format.
"""


PHASE3_EXTRACT_THEMES_PROMPT = """
You are an expert in thematic analysis working in {language}.
Your goal is to discover overarching themes from initial codes of survey responses to an open-ended question. 
Follow these instructions carefully:

1. Review the survey question:
<survey_question>
{survey_question}
</survey_question>

2. Examine the initial codes:
<initial_codes>
{codes}
</initial_codes>

3. Begin your analysis by using the analytical notepad. This is your space for exploratory thinking:

<analytical_notepad>
Use this space to work through your analysis. Consider the following:

- What atomic concepts do you see in each code?
- What patterns are you noticing across codes?
- What might be the deeper story here?
- Which codes seem to cluster around similar underlying ideas?
- What abstractions would explain these clusters?
- Are there more connections between seemingly different codes?
- What other themes would reveal insights?

[Write your analytical notes here]
</analytical_notepad>

4. After your exploratory analysis, extract themes following these steps:

STEP 1: Identify atomic concepts within each code
STEP 2: Discover themes that are TRUE ABSTRACTIONS

5. Adhere to these CRITICAL REQUIREMENTS:
- Themes must be conceptual abstractions, NOT collections of examples
- Each theme should capture a pattern that requires interpretation to see
- Be exhaustive - alle codes need to be represented by the themes

6. AVOID these pitfalls:
- Descriptive groupings that just summarize visible features
- Surface-level categories without deeper insight
- Themes that are obvious from just reading code titles

7. TEST each theme by asking:
- "Does this reveal an insight that wasn't immediately apparent?"
- "Does this explain an underlying dynamic?"
- "Is this a genuine conceptual discovery?"

8. Provide your final output in JSON format:
{{
  "analytical_notes": "Your working notes from the notepad above",
  "themes": ["Theme Name 1", "Theme Name 2", "Theme Name 3", ...],
  "conceptual_insights": {{
    "Theme Name 1": "The non-obvious pattern this theme captures",
    "Theme Name 2": "The non-obvious pattern this theme captures",
    ...
  }}
}}

Remember:
- Theme names should be 1-4 words.
- Conduct all analysis in {language}.
- Ensure your themes are true abstractions that reveal non-obvious insights.
"""

PHASE4_CREATE_CODEBOOK_PROMPT = """
You are an expert in creating hierarchical codebooks for qualitative analysis working in {language}.
Your task is to organize descriptive codes into a structured codebook based on the themes identified.

First, you will be given a list of themes. These are broad and coherent patterns in the data that address the research question. They form Level 1 of the codebook structure.

Themes:
<themes>
{themes}
</themes>

Next, you will be presented with a list of merged clusters. 
These are initial codes derived from the data and will form Level 3 of the codebook structure.

Merged Clusters:
<merged_clusters>
{merged_clusters}
</merged_clusters>

Your task is to organize these initial codes into topics under the provided themes. Topics are specific facets or dimensions of a theme and will form Level 2 of the codebook structure.

Follow these steps:

1. Review the list of themes and merged clusters carefully.
2. For each theme, identify relevant merged clusters that fit within it.
3. Group related merged clusters into topics under each theme.
4. Assign a descriptive name to each topic in Dutch.
5. Ensure that each merged cluster is placed under only one topic and theme.

Present your results in the following format:
Provide the codebook as JSON:
{{
  "themes": [
    {{
      "label": "Theme Name",
      "description": "What this theme encompasses",
      "topics": [
        {{
          "label": "Topic Name", 
          "description": "What this topic covers",
          "codes": [
            {{
              "label": "Original descriptive code label",
              "description": "Expanded description",
              "source_codes": [0, 5]  // Original cluster IDs
            }}
          ]
        }}
      ]
    }}
  ]
}}
    

Additional guidelines:
- Aim for 2-5 topics per theme, depending on the breadth of the theme and the number of relevant merged clusters.
- It's okay if some themes have more topics than others.
- If a merged cluster doesn't fit well under any theme, you may create a "Miscellaneous" theme at the end of your codebook.
- Use your expertise to make informed decisions about grouping and categorization.
- Maintain consistency in the level of detail and granularity across topics and themes.

Begin your analysis and present the codebook structure as instructed.
"""

PHASE5_LABEL_REFINEMENT_PROMPT = """
You are an expert in refining hierarchical codebooks working in {language}.
Your task is to polish and standardize all labels and descriptions for clarity and consistency.

Survey question:
<survey_question>
{survey_question}
</survey_question>

Current codebook structure:
<codebook>
{codebook}
</codebook>

Refinement guidelines:
1. **Labels**: Make concise (2-4 words), clear, and parallel in structure
2. **Descriptions**: Ensure they clearly explain what each level encompasses
3. **Consistency**: Use similar grammatical structures at each level
4. **Clarity**: Remove ambiguity and ensure distinctions are clear
5. **Language**: Ensure all text is properly in {language}

DO NOT:
- Change the hierarchical structure
- Move items between categories
- Add or remove any themes, topics, or codes
- Change the meaning of any item

Provide refinements as JSON:
{{
  "refined_themes": {{
    "1": {{
      "label": "Refined Theme Label",
      "description": "Clear theme description"
    }}
  }},
  "refined_topics": {{
    "1.1": {{
      "label": "Refined Topic Label",
      "description": "Clear topic description"
    }}
  }},
  "refined_codes": {{
    "1.1.1": {{
      "label": "Refined Code Label",
      "description": "Clear code description"
    }}
  }}
}}

Only include items where you're making refinements.
"""

PHASE6_ASSIGNMENT_PROMPT = """
You are an expert in qualitative coding working in {language}.
Your task is to assign a cluster to the most appropriate place in the hierarchical codebook.

Survey question:
<survey_question>
{survey_question}
</survey_question>

Cluster to assign:
- ID: {cluster_id}
- Label: {cluster_label}
- Representative examples:
{cluster_representatives}

Hierarchical codebook:
<codebook>
{codebook}
</codebook>

Instructions:
1. Find the most semantically appropriate code for this cluster
2. Consider the full hierarchy: theme → topic → code
3. Base your decision on meaning alignment, not just keyword matching
4. If no good match exists, assign to code 99.1.1 (Other/Unclassified)

Provide your assignment as JSON:
{{
  "primary_assignment": {{
    "theme_id": "1",
    "topic_id": "1.2", 
    "code_id": "1.2.3"
  }},
  "confidence": 0.85,
  "alternatives": [
    {{
      "code_id": "2.1.1",
      "confidence": 0.15
    }}
  ]
}}

Remember: Focus on semantic meaning in the context of the survey question.
"""
