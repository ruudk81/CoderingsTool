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

ENHANCED_SEGMENTATION_PROMPT = """
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

FOCUSED_CODING_PROMPT = """
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

DESCRIPTION_GENERATION_PROMPT = """
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

MERGE_PROMPT = """
You are an AI assistant tasked with evaluating whether clusters of survey responses represent meaningfully different answers to a specific survey question. 
Your goal is to determine whether each pair of response clusters should be merged or remain separate.

The survey question you will be working with is:
<survey_question>
"{var_lab}"
</survey_question>

You will be provided with pairs of clusters, each containing representative responses.
Your task is to analyze these pairs and decide whether they should be merged or kept separate based on the following criteria:

Merge clusters (YES) ONLY IF:
- Both clusters express essentially the same response pattern in light of the survey question.
- The differences between them are minimal or irrelevant to the survey question.
- A survey analyst would reasonably group these responses together as the same type of answer.

Keep clusters separate (NO) IF:
- The clusters represent distinct response patterns, viewpoints, suggestions, ideas, or concerns.
- They focus on different aspects in addressing the survey question.
- They provide unique or complementary feedback or information.
- They represent different topics even within the same broad theme.
- There is ANY meaningful differentiation relevant to understanding survey responses.

Important guidelines:
- Focus SPECIFICALLY on the survey question context.
- Base decisions on the MOST REPRESENTATIVE responses in each cluster (shown by cosine similarity to centroid).
- Be conservative - when in doubt, keep clusters separate.
- Consider semantic meaning, not just surface-level wording.
- The responses are in Dutch, so make sure to understand the meaning in that language.

Your output should be a JSON object with a single key "decisions", containing an array of objects with these fields:
- "cluster_id_1": First cluster ID
- "cluster_id_2": Second cluster ID  
- "should_merge": Boolean (true ONLY if clusters are not meaningfully differentiated)
- "reason": Brief explanation of your decision (1-2 sentences maximum)

Follow these steps for each cluster pair:
1. Read the representative responses for both clusters carefully.
2. Consider how these responses relate to the survey question.
3. Determine if the responses in both clusters are essentially saying the same thing or if they represent meaningfully different answers.
4. Make a decision on whether to merge the clusters or keep them separate.
5. Provide a brief reason for your decision.
6. Format your decision as a JSON object as specified above.

Return a JSON object with a single key "decisions", containing an array of objects with these fields:
- "cluster_id_1": First cluster ID
- "cluster_id_2": Second cluster ID  
- "should_merge": Boolean (true ONLY if clusters are not meaningfully differentiated)
- "reason": Brief explanation of your decision (1-2 sentences maximum)

Here's an example of how your output should be structured:
{
  "decisions": [
    {
      "cluster_id_1": "13",
      "cluster_id_2": "2",
      "should_merge": true,
      "reason": "Both clusters consistently express the desire for less salt in meals. There is no meaningful differentiation between the responses."
    },
    {
      "cluster_id_1": "17",
      "cluster_id_2": "2",
      "should_merge": true,
      "reason": "All responses in both clusters uniformly request less salt in meals. The clusters are semantically identical."
    }
  ]
}

Now, analyze the following cluster pairs:
{cluster_pairs}

Provide your decisions in the required JSON format.
"""


LABEL_MERGER_PROMPT = """
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

Your objective is to merge labels that are semantically identical or address the same response pattern for most researchers analyzing this survey question. Labels should be merged when they:

1. Express essentially the same meaning or concept
2. Address identical response patterns to the survey question
3. Would be grouped together by researchers seeking to understand response patterns
4. Differ only in wording but capture the same underlying idea

Do NOT merge labels when they:
- Address different aspects of the survey question
- Represent distinct response patterns or viewpoints
- Provide unique insights or information
- Have meaningful differences that would matter to researchers

For each group of labels that should be merged:
1. Choose the most representative label from the group as the new merged label
2. Assign a new sequential cluster ID starting from 0
3. List all original cluster IDs that are being merged

Output your merging decisions in the following JSON format:
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

Remember: Be conservative - when in doubt, keep labels separate. Only merge when labels are truly semantically equivalent in the context of this specific survey question.
"""

INITIAL_THEMES_PROMPT = """
You are an expert in thematic analysis working in {language}.
Your task is to identify broad, organizing themes that provide a preliminary framework for understanding responses to the survey question. These initial themes will guide more detailed thematic analysis in subsequent steps.

Here is the survey question:
<survey_question>
{survey_question}
</survey_question>

Here are the merged labels from the initial analysis:
<merged_labels>
{merged_labels}
</merged_labels>

Your goal is to identify initial themes that:
1. Capture the major conceptual areas represented in the labels
2. Provide a broad organizing framework for understanding response patterns
3. Are comprehensive enough to encompass all labels
4. Are distinct enough to represent meaningfully different aspects of responses
5. Give context for subsequent detailed thematic analysis

Guidelines for initial themes:
- Each theme should represent a central organizing concept
- Themes should be broad enough to accommodate detailed sub-analysis
- Themes should be semantically meaningful in relation to the survey question
- Use 2-4 words maximum per theme name
- Aim for 3-7 themes total (adjust based on data complexity)
- Ensure all labels can naturally fit under one of the themes

Analyze the labels and identify the initial themes that best organize the response patterns.

Output your analysis in the following JSON format:
{{
  "initial_themes": [
    {{
      "theme_name": "Theme Name 1",
      "description": "Brief description of what this theme encompasses",
      "related_labels": ["Label 1", "Label 2", "Label 3"]
    }},
    {{
      "theme_name": "Theme Name 2", 
      "description": "Brief description of what this theme encompasses",
      "related_labels": ["Label 4", "Label 5"]
    }}
  ],
  "rationale": "Brief explanation of the thematic framework and how it organizes the data"
}}

Remember: These are initial, broad themes that will be refined in subsequent analysis. Focus on creating a useful organizing framework rather than detailed categorization.
"""

PHASE1_DESCRIPTIVE_CODING_PROMPT = """
You are an expert in descriptive coding working in {language}.
Your task is to perform descriptive coding on segments from open-ended survey responses. 
You will provide one concise label that captures what respondents are expressing in these segments.

Here are the coding principles you must follow:
1. Stay close to the data: Use respondents' own concepts
2. Be descriptive: Capture what is said, not why
3. Be specific: Focus on the distinct pattern in these segments
4. Be concise: Maximum 5 words
5. Use Title Case for labels

Rules to adhere to:
- Describe only what is explicitly stated
- No interpretation or inference beyond the text
- No evaluation or judgment
- Focus on a single coherent idea
- If segments express multiple ideas, identify the primary pattern

The output format should be:
{{
  "label": "Descriptive Label Here"
}}

You must output the label in {language}.

Now, here is the input you will work with:

Survey question: 
<survey_question>
{survey_question}
</survey_question>

Segment ID: 
<cluster_id>
{cluster_id}
</cluster_id>

Representative segments:
<representatives>
{representatives}
</representatives>

Follow these steps to complete the task:

1. Carefully read the survey question and all representative segments.
2. Identify the main theme or idea expressed across the segments.
3. Create a concise label (maximum 5 words) that captures this main idea.
4. Ensure your label adheres to all coding principles and rules mentioned above.
5. Double-check that your label is in Title Case and in the correct language ({language}).
6. Format your output according to the specified JSON format.

Remember, your goal is to provide a single, concise label that accurately represents the content of the segments without interpretation or judgment. Output your final answer within the specified JSON format tags.
"""

PHASE2_EXTRACT_THEMES_PROMPT = """
You are an expert in thematic analysis working in {language}. 
Your task is to extract themes from initial codes of survey responses to an open-ended survey question. 
These themes will be used for a codebook, but you only need to discover the themes themselves.

Here is the survey question:
<survey_question>
{survey_question}
</survey_question>

Previous analysis has identified these initial organizing themes:
<initial_themes>
{initial_themes}
</initial_themes>

Now, carefully review these initial codes:
<initial_codes>
{codes}
</initial_codes>

Extract themes from these initial codes following these guidelines:

IMPORTANT: Use the initial themes above as a starting framework, but refine, modify, or expand them as needed based on the detailed analysis of the codes. The initial themes provide context, but your analysis of the codes should take precedence.

1. Each theme must be underpinned by a central organizing concept or a singular latent idea that makes semantic sense in light of the survey question.
2. Ensure each theme is:
   - Semantically meaningful: The theme is meaningful in light of the survey question
   - Thematically coherent: The label reflects a central organizing concept or singular latent idea
   - Conceptually clear: The meaning of the theme is clear and unambiguous
3. The number of themes should be:
   - Large enough to ensure collective exhaustion of all initial codes (every code should naturally fit under a theme)
   - Small enough to maintain meaningful distinctions and allow for topic-level organization
   - Balanced to avoid themes that are too broad (becoming meaningless) or too narrow (preventing proper topic grouping)

After careful consideration, provide your output in the following JSON format:
{{
  "themes": ["Theme Name 1", "Theme Name 2", "Theme Name 3", ...]
}}

Each theme name should be 1-4 words, capturing the essence of a major pattern.
Remember to conduct this analysis and provide your response in {language}.
"""

PHASE2_GROUP_TOPICS_PROMPT = """
You are an expert in thematic analysis working in {language}. 
Your task is to organize initial codes into topics under provided themes. 

These will be used for a codebook with the following structure:
- Themes (Level 1): Broad and coherent patterns in the data that address the research question.
- Topics (Level 2): Specific facets or dimensions of a theme.
- Codes (Level 3): Short descriptive labels grounded in data capturing the essence of responses for a segment of the data.

Your primary focus is to create and assign topics under each theme.

Here is the context for your analysis:

Survey question:
<survey_question>
{survey_question}
</survey_question>

Themes identified:
<themes>
{themes}
</themes>

Initial codes to organize into topics:
<codes>
{codes}
</codes>

Instructions for creating topics:
1. Group codes with shared meaning to help organize nuance and complexity within each theme.
2. Ensure each topic is:
   a) Semantically meaningful: The topic is meaningful in light of the theme it is positioned under.
   b) Topically coherent: A topic must reflect a unified concept.
   c) Conceptually clear: The meaning of the topic is clear and unambiguous.
3. The number of topics per theme should be:
   a) Sufficient to capture the distinct dimensions and facets within each theme.
   b) Parsimonious to avoid artificial subdivisions that fragment related codes.
   c) Comprehensive to ensure every code has a natural home within the topic structure.
   d) Balanced such that topics represent meaningful distinctions without becoming too granular or too broad.

When creating your topics, consider the following guidelines:
1. Analyze the codes carefully to identify common threads or patterns.
2. Think about how the codes relate to the overarching theme and to each other.
3. Create topic names that are concise (1-4 words) yet descriptive of the grouped codes.
4. Ensure that all codes are accounted for in your topic structure.
5. Avoid creating too many or too few topics; aim for a balanced and meaningful organization.

Your output should be a JSON object with the following structure:
{{
  "structure": [
    {{
      "theme": "Theme Name 1",
      "topics": ["Topic Name A", "Topic Name B", "Topic Name C"]
    }},
    {{
      "theme": "Theme Name 2", 
      "topics": ["Topic Name D", "Topic Name E"]
    }}
  ]
}}

Remember:
- Topic names should be 1-4 words each.
- Ensure all themes are included in the structure.
- All codes should be accounted for in your topic organization, even if not explicitly mentioned in the output.
- The analysis and output should be in {language}.

Provide your response in the specified JSON format.
"""

PHASE2_CREATE_CODEBOOK_PROMPT = """
You are an expert in thematic analysis working in {LANGUAGE}. 
Your task is to create a codebook that organizes initial codes representing segments of survey responses to an open-ended survey question.

Here are the input variables you will be working with:

<survey_question>
{survey_question}
</survey_question>

<structure>
{structure}
</structure>

<codes>
{codes}
</codes>

The codebook has the following structure:
- Themes (Level 1): A broad and coherent pattern in the data that addresses the research question.
- Topics (Level 2): A specific facet or dimension of a theme.
- Codes (Level 3): A short descriptive label grounded in data capturing the essence of responses for a segment of the data.

To create the codebook, follow these steps:
1. Use the provided theme-topic structure as the foundation.
2. Assign EVERY initial code to the most appropriate topic.
3. Include the numeric ID from [source ID: X] in the source_codes array for each code.
4. Create meaningful descriptions for all levels (themes, topics, and codes).

Ensure that the codebook meets these criteria:
- Semantically meaningful: The labels of the Themes, Topics, and Codes should be semantically meaningful in light of the survey question.
- Thematically coherent: All codes and topics within a theme should reflect a unified idea.
- Internally consistent: There should be no contradictory or unrelated codes in a topic or theme.
- Conceptually clear: The meaning of the labels for each theme, topic, and code should be clear and unambiguous.

Present your codebook in the following JSON format:
{{
  "themes": [
    {{
      "id": "1",
      "label": "Theme Label (max 4 words)",
      "description": "What unifies this theme (max 20 words)",
      "topics": [
        {{
          "id": "1.1",
          "label": "Topic Label (max 4 words)", 
          "description": "What this topic represents (max 20 words)",
          "codes": [
            {{
              "id": "1.1.1",
              "label": "Code Label (max 4 words)",
              "description": "What kind of feedback this captures (max 20 words)",
              "source_codes": [1, 5, 12]  // Numeric IDs from [source ID: X]
            }}
          ]
        }}
      ]
    }}
  ]
}}

Ensure that your response is in {language}.
"""

PHASE3_THEME_JUDGER_PROMPT = """
You are an expert in qualitative research and thematic analysis. 
Your task is to audit a codebook for open-ended survey responses. 

The codebook is structured as follows:
- Themes (Level 1): Underpinned by a central organizing concept, providing meaningful insight beyond surface-level observations.
- Topics (Level 2): Group codes with shared meaning, organizing nuance and complexity within a theme.
- Codes (Level 3): Reflect response segments with similar meaning, representing the smallest unit of meaning in this hierarchy.
  - Each code contains source_codes (numeric IDs from initial clustering)

Here is the input you will be working with:

Survey question:
<survey_question>
"{survey_question}"
</survey_question>

Current codebook:
<codebook>
{codebook}
</codebook>

Codes not yet in the codebook:
<unused_codes>
{unused_codes}
</unused_codes>


Your task is to evaluate if the codebook is:
1. Semantically coherent: Each code must semantically fit the full path it is placed under (Theme -> Topic -> Code).
2. Internally consistent: No codes that semantically contradict the meaning of a topic, and no topics that semantically contradict the meaning of a theme.
3. Collectively exhaustive: No unused source codes.

CRITICAL: Focus on the MEANING of labels, not their IDs or position in the hierarchy. Read the label of a Theme, Topic or Code and ask "Does this semantically belong here (Theme -> Topic -> Code)?"

If all criteria are met, set needs_revision to false. Otherwise, set it to true.
If revision is needed, follow these steps:

1. Incorporate unused codes WHERE THEY SEMANTICALLY BELONG:
   - Read the LABEL of the unused code (ignore the ID number)
   - Ask: "What is this code actually about based on its label?"
   - Match semantic meaning of the code's label with Themes and Topics in the following order: Themes -> Topics
   - If SAME MEANING as existing code:
     ADD: Add source_codes from unused code to an existing code that shares its meaning
   - If genuinely DIFFERENT meaning but fits existing topic semantically:
     ADD: Add new code with a label and source_codes to a Theme and Topic WHERE THE LABEL MEANINGS MATCH
   - If no semantically appropriate topic exists based on label meanings:
     CREATE: Create new topic under appropriate theme, or create new theme if needed

2. Fix semantic misplacements and structural issues:
   - If codes are in wrong semantic locations:
     RESTRUCTURE: Move codes to themes and topics where they semantically belong in the following order: Themes -> Topics
   - If not Thematically coherent:
     CREATE: Create new topic with a label under a SEMANTICALLY APPROPRIATE theme - or create a new Theme, if needed
   - If not Internally consistent:
     CLARIFY: Split a topic containing semantically diverse codes into distinct topics or themes -> Topics

FORBIDDEN: 
- Placing codes in themes where they don't semantically belong or placing codes in Topics they don't semantically belong
- Providing instructions that don't consider semantic meaning

IMPORTANT: Every action MUST specify the complete path and ALL affected codes with their source_codes.
EXAMPLE OF A PATH: Theme 1, Topic 1.1, Code 1.1.1 with source_codes [ID 1, ID 2] (replace placeholders of source_code ID's with actual ID's)

Provide your output in the following format:
{{
  "needs_revision": true/false,
  "summary": "One sentence describing the main issue or confirming quality",
  "issues": [
    "List specific problems found, especially semantic misplacements",
   ],
  "actions": [
    {{
      "type": "RESTRUCTURE/CREATE/CONSOLIDATE/CLARIFY/ADD",
      "details": "Specific action  with semantic justification and all affected IDs in the complete path (Theme -> Topic -> Codes -> Source Code)"
    }}
  ]
}}

Remember to focus on semantic meaning and coherence throughout your analysis and recommendations.
"""

PHASE4_THEME_REVIEW_PROMPT = """
You are an expert in qualitative research and thematic analysis working in {language}. 
Your task is to revise a hierarchical codebook based on feedback from a quality review.

First, carefully review the current codebook:

<current_codebook>
{current_codebook}
</current_codebook>

Now, review the feedback for revision:

<feedback>
- Summary: 
{summary}

- Issues identified:
{issues}

- Required actions:
{actions}
</feedback>

To revise the codebook, follow these instructions:

1. Carefully analyze the feedback and identify all required changes.
2. Make the specified revisions to the codebook structure, including:
   - Creating new themes or topics as instructed
   - Moving codes to semantically appropriate locations
   - Adding new codes to existing or new topics
3. Ensure all changes align with the feedback and maintain the overall semantic structure of the codebook.

Rules for maintaining codebook structure:
- Preserve the hierarchical structure: Themes > Topics > Codes
- Maintain consistent numbering (themes: 1,2,3... topics: X.1,X.2... codes: X.Y.1,X.Y.2...)
- Keep all existing source_codes arrays intact
- Ensure all labels and descriptions are in Dutch

Output the revised codebook in the following JSON format:

{{
  "themes": [
    {{
      "id": "1",
      "label": "Theme label (max 4 words)",
      "description": "What unifies this theme",
      "topics": [
        {{
          "id": "1.1",
          "label": "Topic label (max 4 words)",
          "description": "What this topic represents",
          "codes": [
            {{
              "id": "1.1.1",
              "label": "Code label (max 4 words)",
              "description": "What specific response pattern this captures",
              "source_codes": [1, 5, 12]  // MUST preserve from original
            }}
          ]
        }}
      ]
    }}
  ]
}}
"""

PHASE5_LABEL_REFINEMENT_PROMPT = """
You are an expert in qualitative coding and codebook design working in {language}. 
Your goal is to refine labels and descriptions in a hierarchical codebook to ensure they are clear, precise, and conceptually meaningful.

Here's an overview of the codebook structure:
- Themes (Level 1): Broad patterns with central organizing concepts
- Topics (Level 2): Specific facets that group codes with shared meaning
- Codes (Level 3): Short descriptive labels for distinct response patterns

When refining labels, follow these principles:
- Maximum 4 words
- Capture the underlying concept, not surface features
- Use precise, meaningful terms
- Avoid compound labels (X and Y and Z)
- Express a single coherent idea

When refining descriptions, adhere to these guidelines:
- Maximum 20 words
- Add insight beyond what's in the label
- Explain the significance or meaning
- Use clear, active language
- Focus on why this grouping matters

You will be provided with a survey question and the current codebook. 
Your task is to review and refine the codebook entries that need improvement. 
You should only output the refined entries, not the entire codebook.

Here's the survey question:
<survey_question>
"{survey_question}"
</survey_question>

And here's the current codebook:
<codebook>
{codebook}
</codebook>

Follow these steps to refine the codebook:

1. Carefully review each theme, topic, and code in the codebook.
2. For each entry, evaluate if the label and description meet the principles outlined above.
3. If an entry needs improvement, refine the label and/or description according to the principles.
4. Ensure all refinements are in {language}.
5. Only include entries that you have refined in your output.

Format your output in JSON as follows:
{{
  "refined_themes": {{
    "1": {{
      "label": "Improved label",
      "description": "Clearer description"
    }}
  }},
  "refined_topics": {{
    "1.1": {{
      "label": "Improved label",
      "description": "Clearer description"
    }}
  }},
  "refined_codes": {{
    "1.1.1": {{
      "label": "Improved label",
      "description": "Clearer description"
    }}
  }}
}}

Remember to only include entries that you have refined. If no refinements are necessary, output an empty JSON object. 
Ensure all labels and descriptions are in {language}.
"""

PHASE6_ASSIGNMENT_PROMPT = """
You are a {language} language coder tasked with assigning a cluster to an existing codebook. 
Your goal is to choose the most appropriate path (Theme → Topic → Code) for the given cluster based on the provided codebook.

First, review the survey question:
<survey_question>
{survey_question}
</survey_question>

Next, examine the cluster to be assigned:
<cluster_to_assign>
- ID          : {cluster_id}
- Label       : {cluster_label}
- Examples    :
{cluster_representatives}
</cluster_to_assign>

Now, carefully review the codebook:
<codebook>
Codebook
{codebook}
</codebook>

To assign the cluster, follow these steps:
1. Analyze the cluster label and examples.
2. Compare them with the themes, topics, and codes in the codebook.
3. Choose exactly ONE path (Theme → Topic → Code) that best matches the cluster.
4. Determine your confidence in this assignment on a scale of 0 to 1.

Provide your assignment in the following JSON format:
{{
  "primary_assignment": {{
    "theme_id": "1",
    "topic_id": "1.2",
    "code_id": "1.2.3"
  }},
  "confidence": 0.95,
  "alternatives": [
    {{
      "theme_id": "2",
      "topic_id": "2.1",
      "code_id": "2.1.1",
      "confidence": 0.6
    }}
  ]
}}

Important: If your confidence is less than 0.60, assign the cluster to Theme 99 → Topic 99.1 → Code 99.1.1 ("Other").
Remember to focus on the meanings of the labels rather than the ID numbers when making your assignment.
"""





