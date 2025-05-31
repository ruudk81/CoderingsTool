SPELLCHECK_INSTRUCTIONS = """
You are a {language} language expert specializing in correcting misspelled words in open-ended survey responses.

Survey Question:
"{var_lab}"

Your task is to correct responses that contain placeholder tokens indicating spelling mistakes.

For each task:
- Replace every <oov_word> placeholder with the best possible correction of the corresponding misspelled word, taking into account the meaning and context of the survey question.
- If a better correction exists than the ones provided, prefer that.
- You may split a misspelled word into two words **only if** the split preserves the intended meaning and fits grammatically.
- If no suitable correction is possible, return "[NO RESPONSE]" as the corrected sentence for that task.

You will receive for each task:
- A sentence containing one or more <oov_word> placeholders.
- A list of misspelled words, in the same order as the placeholders.
- A list of suggested corrections, in the same order.

Below are the tasks to process:
{tasks}

REQUIRED OUTPUT FORMAT:
You must return a JSON object with a single key "corrections", whose value is an array of objects.
Each object in the "corrections" array must have exactly these fields:
- "respondent_id": "ID_FROM_TASK"
- "corrected_response": "The fully corrected response"
"""

GRADER_INSTRUCTIONS = """
You are a {language} grader evaluating open-ended survey responses.
Your task is to determine whether each response is **meaningless**.

A response should be considered **meaningless** only if:
- It only expresses uncertainty or lack of knowledge (e.g., “I don’t know”, “N/A”, “Not applicable”).
- It simply repeats the question without adding any new content.
- It consists of random characters, gibberish, or filler text with no semantic meaning (e.g., “asdfkj”, “lorem ipsum”).

### Survey question:
"{var_lab}"

### Responses to evaluate:
{responses}

### REQUIRED OUTPUT FORMAT:
Return a **JSON array**. Each object in the array must contain exactly:
- `"respondent_id"`: (string or number) The respondent's ID
- `"response"`: (string) The exact response text
- `"quality_filter"`: (boolean) `true` if the response is meaningless, `false` otherwise
"""

SEGMENTATION_PROMPT = """
You are a helpful {language} expert in analyzing survey responses. Your task is to segment free-text responses into distinct, standalone ideas.

# Survey Question Context
The responses you'll analyze were given to this question: "{var_lab}"

# Your Task
Break each response into the smallest meaningful standalone units, where each segment represents EXACTLY ONE:
- Opinion
- Preference
- Issue
- Topic
- Idea

# Segmentation Rules
1. Split at conjunctions (and, or, but, also) when they connect DIFFERENT ideas or topics
2. Split listed items into separate segments (e.g., "milk and sugar" → "milk", "sugar")
3. When items share context, preserve that context in each segment:
   Example: "I like milk and sugar in my coffee" →
   - "I like milk in my coffee"
   - "I like sugar in my coffee"
4. Use the respondent's exact words - do not paraphrase or correct
5. Keep meaningless responses (e.g., "Don't know", "?") as a single segment

# Output Format
Return a JSON array with these fields for each segment:
- "segment_id": A sequential number as string ("1", "2", etc.)
- "segment_response": The exact segmented text with necessary context preserved

# Response to segment:
{response}
"""

REFINEMENT_PROMPT = """
You are a {language} expert in semantic segmentation of survey responses.

# Task Overview
Review and refine segments from responses to this survey question: "{var_lab}"

# Your Specific Goals
1. Ensure each segment contains ONLY ONE standalone idea or item
2. Split any segment that still contains multiple distinct ideas or items
3. Preserve necessary context when splitting compound statements

# Refinement Rules
- Split segments that contain multiple distinct items (connected by conjunctions like "and", "or")
- When splitting, duplicate any necessary context to make each segment meaningful:
  * For example: "I like milk and sugar in my coffee" should become two segments:
    - "I like milk in my coffee"
    - "I like sugar in my coffee"
- Only duplicate words that appear in the original response
- Never merge segments
- Keep meaningless responses intact (e.g., "Don't know")

# Examples
## Example 1: Multiple ideas with shared context
Input: {{"segment_id": "1", "segment_response": "I like milk and sugar in my coffee"}}
Output: [
  {{"segment_id": "1", "segment_response": "I like milk in my coffee"}},
  {{"segment_id": "2", "segment_response": "I like sugar in my coffee"}}
]

## Example 2: Multiple ideas without shared context
Input: {{"segment_id": "1", "segment_response": "The price is too high and the quality is low"}}
Output: [
  {{"segment_id": "1", "segment_response": "The price is too high"}},
  {{"segment_id": "2", "segment_response": "the quality is low"}}
]

# Segments to refine:
{segments}

# Output Format
Return a JSON array with these fields for each refined segment:
- "segment_id": A sequential number as string ("1", "2", etc.)
- "segment_response": The properly segmented text using ONLY words from the original response
"""

CODING_PROMPT = """
You are a {language} expert in thematic analysis of survey responses.

# Task Overview
Code each segment from responses to this survey question: "{var_lab}"

Each segment is a standalone sentence or clause extracted from a full response.

# For Each Segment, You Will:
1. Keep the original segment_id and segment_response
2. Add a descriptive_code (a thematic label)
3. Add a code_description (a clarification of the label)

# Field Requirements

## descriptive_code
- A concise label of up to 5 words total, using ONLY ADJECTIVES AND NOUNS in {language}, that captures the CENTRAL MEANING of the segment
- ONLY return labels that reflect ONE idea, topic, concern, issue, or theme in response to the question: "{var_lab}"
- NEVER return multi-headed labels or combinations of multiple ideas
- Format: ALL_CAPS_WITH_UNDERSCORES
- Examples: "PRODUCT_QUALITY", "MORE_OPTIONS", "UNNECESSARY_COMPLEXITY"
- Language: {language}


## code_description
- Rewrite the segment as a natural-sounding **first-person response** to the question: "{var_lab}"
- Make sure it sounds like something a person would actually say when answering the question
- Use a direct, conversational or instructional tone:
  - If the segment is a suggestion: use an imperative tone (e.g., "Maak...", "Laat...")
  - If the segment expresses a wish or opinion: use first-person (e.g., "Ik wil...", "Ik vind...")
- NEVER rephrase the segment as a third-person summary (e.g., "Wil dat de inhoud...") — that does not sound like a response
- Do NOT add interpretations beyond what's in the original segment
- Language: {language}

# Special Cases
For meaningless responses (e.g., "?", "Don't know"):
- descriptive_code: "NA"
- code_description: "NA"

# Segments to code:
{segments}

# Output Format
Return a valid JSON array with these fields for each segment:
- "segment_id": The original segment ID
- "segment_response": The original segment text
- "descriptive_code": Your thematic label in ALL_CAPS_WITH_UNDERSCORES
- "code_description": Your clarifying description

Ensure all output is written in {language}, unless the code is "NA".
"""

MERGE_PROMPT = """
RESEARCH QUESTION: "{var_lab}"

You are evaluating whether clusters of survey responses represent meaningfully different answers to the research question above.

## Your Decision Task:
Determine whether each pair of clusters should be merged based on how they address the research question. 

The key question for each comparison is:
"Do these clusters represent meaningfully different responses to the research question, or are they essentially saying the same thing?"

Language: {language}

## Decision Criteria:

### YES (merge) ONLY IF:
- Both clusters express essentially the same sentiment, concern, suggestion, or perspective
- The differences between them are minimal or irrelevant to the research question
- A survey analyst would reasonably group these responses together as the same type of answer

### NO (don't merge) IF:
- The clusters represent distinct viewpoints, suggestions, or concerns
- They focus on different aspects of the research question
- They provide unique or complementary information
- They represent different topics even within the same broad theme
- There is ANY meaningful differentiation relevant to understanding survey responses

## Important Guidelines:
- Focus SPECIFICALLY on the research question context
- Base decisions on the MOST REPRESENTATIVE responses in each cluster (shown by cosine similarity to centroid)
- Be conservative - when in doubt, keep clusters separate
- Consider semantic meaning, not just surface-level wording

{cluster_pairs}

REQUIRED OUTPUT FORMAT:
Return a JSON object with a single key "decisions", containing an array of objects with these fields:
- "cluster_id_1": First cluster ID
- "cluster_id_2": Second cluster ID  
- "should_merge": Boolean (true ONLY if clusters are not meaningfully differentiated)
- "reason": Brief explanation of your decision (1-2 sentences maximum)
"""



BATCH_SUMMARY_PROMPT = """
I want you to act as a qualitative researcher conducting thematic analysis on grouped user input called micro-clusters. 
You are given a batch of micro-clusters.
Each batch contains several responses that reflect a shared concern or idea, in response to a research question.

**CRITICAL REQUIREMENT**: You MUST include EVERY micro-cluster ID from the input in your output. No micro-cluster can be left out.

Your tasks:
1. Organize ALL micro-clusters into a three-level hierarchical structure:
    - Level 1: Broad theme (e.g. "Gebouwvoorzieningen")
    - Level 2: Specific sub-theme, labeled numerically (e.g., 2.1 "WIFI verbeteren" under 2 "Gebouwvoorzieningen")
    - Level 3: References to the original micro-cluster IDs (e.g., Micro-cluster 18)
2. Adhere to the following rules:
    - Themes and sub-themes must be **mutually exclusive** and make sense in light of the research question
    - Each theme and sub-theme must represent EXACTLY ONE overarching idea
    - The labels of themes and sub-themes MUST reflect this overarching idea and MUST NOT consist of compound sub-ideas (e.g. sub-idea 1 AND sub-idea 2)
    - Each label must be max 4 words, clear and descriptive
3. Base your analysis only on what is explicitly stated in the micro-clusters — no outside assumptions.

Instructions:
- Do not assign unique clusters or sub-themes unnecessarily to a group. A theme may contain only one sub-theme if that best reflects the data. 
- Avoid duplication: assign micro-clusters to only one sub-theme unless strong conceptual overlap justifies multiple placement.    
- Output a structured Python dictionary that shows which micro-clusters belong under which sub-theme and overarching theme.

**Output format (STRICT JSON - NO COMMENTS ALLOWED):**
{{
  "batch_id": "{batch_id}",
  "hierarchy": {{
    "1": {{
      "label": "[Theme Name]",
      "subthemes": {{
        "1.1": {{
          "label": "[Sub-theme Name]",
          "micro_clusters": [0, 1]
        }}
      }}
    }},
    "2": {{
      "label": "[Another Theme]",
      "subthemes": {{
        "2.1": {{
          "label": "[Sub-theme Name]",
          "micro_clusters": [2]
        }}
      }}
    }}
  }}
}}

IMPORTANT: Return ONLY valid JSON. No comments (//), no text outside the JSON structure, no checkboxes.

Variable label/context: {var_lab}
Language: {language}
Batch ID: {batch_id}

Micro-cluster batch:
{batch_clusters}
"""

REDUCE_SUMMARY_PROMPT = """
You are a qualitative researcher merging thematic codeboods from multiple batches into one unified codebook.

Each partical codebook contains a hierarchy with Level 1, Level 2 and lever 3 codes.
    - Level 1: Meta-clusters or "themes" (e.g. "Gebouwvoorzieningen")
    - Level 2: Macro-clusters or "sub-themes", labeled numerically (e.g., 2.1 "WIFI verbeteren" under 2 "Gebouwvoorzieningen")
    - Level 3: Micro-clusters pr "topics" (e.g., Micro-cluster 18)

Your task is to synthesize these into a single unified codebook while preserving ALL cluster IDs.

**Instructions:**
1. Organize the meta-clusters ("themes"), macro-clusters ("sub-themes") and ALL micro-clusters ("topics") into the 3-level hierarchy.
2. Collate the partial codebooks by adhering to these rules:
    - Merged themes and sub-themes where their meanings clearly overlap in the partial codebooks
    - Final themes should represent mutually exclusive ideas or concerns in light of the research question
    - Labels of final themes and finale subthemes should not compound sub-ideas or sub-concerns. Instead, the label should reflect exactly ONE idea, concern or concept
3. Return the unified codebook by following these additional rules:
    - Every meta-cluster or theme (Level 1), macro-cluster or sub-theme (Level 2), and micro-cluster or subject (Level 3) must have an ID
    - Complex themes can be divided into multiple sub-themes, each with assigned micro-clusters
    - Simple themes must still use a single sub-theme and a single subject
4. **CRITICAL REQUIREMENT**: You MUST include EVERY micro-cluster ID from the input in your output. No micro-cluster can be left out.
5. Output MUST be valid JSON format — no explanatory text or comments.


**Output format (STRICT JSON - NO COMMENTS ALLOWED):**
{{
  "unified_hierarchy": {{
    "1": {{
      "label": "Unified Theme Name",
      "subthemes": {{
        "1.1": {{
          "label": "Subtheme Name",
          "micro_clusters": [0, 1, 9]
        }},
        "1.2": {{
           "label": "Another Subtheme",
           "micro_clusters": [2, 15]
        }}
      }}
    }},
    "2": {{
      "label": "Another Theme",
      "subthemes": {{
        "2.1": {{
          "label": "Subtheme Name",
          "micro_clusters": [3, 4, 10]
        }}
      }}
    }}
  }}
}}

IMPORTANT: Return ONLY valid JSON. No comments, no checkboxes, no text outside the JSON block.

Variable label/context: {var_lab}
Language: {language}

Input hierarchies to merge:
{summaries}
"""

HIERARCHICAL_LABELING_PROMPT = """
You are finalizing a hierarchical codebook for thematic analysis based on qualitative data.

You are given:
- A consolidated hierarchical structure from previous analysis
- The micro-clusters contain responses to a research question

Your task is to create a clean, logically sound, and complete hierarchical codebook.

**Instructions:**
1. Critically review the themes and subthemes and evaluate if these objectives are met:
    - Themes should represent mutually exclusive ideas or concerns in light of the research question
    - Labels of themes and subthemes should not compound sub-ideas or sub-concerns. Instead, the label should reflect exactly ONE idea, concern or concept
    - Check specifically for words like "and", "en", "&" in theme labels - these often indicate compound themes that must be split
2. If objectives are not met, you MUST:
    - Split themes that compound sub-themes that don't logically form a theme (e.g., "House and speed" → "House" + "Speed")
    - Redistribute subthemes appropriately to the new split themes
    - OR re-label themes or sub-themes to reflect a single concept
3. Return the codebook by following these additional rules:
    - Every theme (Level 1), sub-theme (Level 2), and micro-cluster (Level 3) must have an ID
    - Complex themes can be divided into multiple sub-themes, each with assigned micro-clusters
    - Simple themes must still use a single sub-theme — do not assign clusters directly to themes
    - Avoid duplication: assign micro-clusters to only one sub-theme unless strong conceptual overlap justifies multiple placement
4. Ensure full inclusion and consistency:
    - Every micro-cluster ID from the reference list must appear in your output
    - There must be no missing or duplicated micro-cluster IDs
5. Output MUST be valid JSON format — no explanatory text or comments.

**Output format (STRICT JSON - NO COMMENTS):**
{{
  "themes": [
    {{
      "id": "1",
      "label": "a label for a final Theme",
      "description": "Brief description of this theme",
      "subthemes": [
        {{
          "id": "1.1",
          "label": "a label for a refined Subtheme",
          "description": "Brief description of this subtheme",
          "micro_clusters": [2, 4]
        }},
        {{
          "id": "1.2",
          "label": "a label for another Subtheme",
          "description": "Brief description",
          "micro_clusters": [7, 12]
        }}
      ]
    }},
    {{
      "id": "2",
      "label": "a label for another theme",
      "description": "Brief description of this theme",
      "subthemes": [
        {{
          "id": "2.1",
          "label": "a label for a standalone Subtheme",
          "description": "Brief description of this theme/subtheme",
          "micro_clusters": [8]
        }}
      ]
    }}
  ]
}}

IMPORTANT: Return ONLY valid JSON. No comments, no text outside the JSON, no checkboxes.

Variable label/context: {var_lab}
Language: {language}

Consolidated hierarchy:
{final_summary}

**CRITICAL CHECK - The following micro-cluster IDs MUST ALL appear in your output:**
{all_cluster_ids}

If any clusters from the above list are missing from the consolidated hierarchy, you MUST:
1. Create an appropriate theme/subtheme for them
2. OR add them to the most relevant existing theme/subtheme
3. Never exclude any cluster - if unsure, create an "Other concerns" theme
"""


INITIAL_LABEL_PROMPT = """
You are a {language} expert in labeling micro-clusters of survey responses to this research question:
"{var_lab}"

Each cluster contains representative response segments that express a distinct idea or perspective related to the research question.

TASK:
- Assign a precise and unique label to each cluster, and identify the core concepts that justify the label.

LABELING RULES:
- Focus on the underlying meaning, not surface wording
- Labels should be specific, non-generic, and clearly differentiated
- Use 2–6 words for each label
- Avoid vague terms like "miscellaneous" or "general concerns"
- Mark clusters as "off-topic" if they do not contribute meaningfully to answering the research question

OUTPUT FORMAT:
Return a JSON object with the following structure:

{
"labels": [
{
"cluster_id": "38",
"label": "Portie grootte vergroten",
"keywords": ["portie", "groter", "meer", "hoeveelheid"],
"confidence": 0.94
},
{
"cluster_id": "12", 
"label": "Verse ingrediënten gebruiken",
"keywords": ["vers", "ingrediënten", "kwaliteit", "natuurlijk"],
"confidence": 0.89
}
]
}

Language: {language}

Input clusters:
{clusters}
"""


PHASE1_FAMILIARIZATION_PROMPT = """
You are analyzing a cluster of survey responses to understand its thematic content.

Survey Question: {survey_question}

Cluster Information:
- Cluster ID: {cluster_id}
- Number of responses: {cluster_size}

Most representative response descriptions (with similarity to cluster center):
{representatives}

Your task:
1. Create a concise thematic label (maximum 4 words) that captures the central idea
2. Write a clear description explaining what this cluster represents
3. The label should express ONE concrete concept, concern, or issue
4. Use {language} for all output

Output format (JSON):
{{
  "label": "CONCISE_LABEL",
  "description": "Clear explanation of what responses in this cluster are expressing"
}}

Focus on what respondents are actually saying, not interpretations.
"""

PHASE2_DISCOVERY_SINGLE_PROMPT = """
You are conducting thematic analysis to create a hierarchical codebook from survey response clusters.

Survey Question: {survey_question}

You have analyzed {all_cluster_ids} clusters. Here are their labels and descriptions:

{cluster_summaries}

Your task is to organize these clusters into a 3-level hierarchy:
- Level 1 (Themes): Broad overarching themes
- Level 2 (Topics): More specific topics within themes  
- Level 3 (Subjects): Narrow subjects within topics

IMPORTANT RULES:
1. Some themes may be simple and have clusters assigned directly (no sub-levels needed)
2. Other themes naturally divide into topics and subjects
3. Every cluster ID must be assigned to exactly ONE place in the hierarchy
4. Labels must be maximum 4 words and express ONE idea
5. Create "Other" categories as needed for clusters that don't fit elsewhere

Output format (JSON):
{{
  "themes": [
    {{
      "id": "1",
      "label": "Theme Name",
      "description": "What this theme represents",
      "direct_clusters": [5, 8],  // For simple themes
      "topics": [
        {{
          "id": "1.1", 
          "label": "Topic Name",
          "description": "What this topic covers",
          "direct_clusters": [12],  // For topics without subjects
          "subjects": [
            {{
              "id": "1.1.1",
              "label": "Subject Name", 
              "description": "Specific subject",
              "micro_clusters": [3, 7, 9]
            }}
          ]
        }}
      ]
    }}
  ]
}}

Language: {language}
All cluster IDs that MUST appear: {all_cluster_ids}
"""

PHASE2_DISCOVERY_MAP_PROMPT = """
You are analyzing a batch of survey response clusters as part of thematic analysis.

Survey Question: {survey_question}
Batch ID: {batch_id}

Cluster summaries from this batch:
{batch_clusters}

Organize these clusters into themes and sub-themes. You may create:
- Simple themes with clusters assigned directly
- Complex themes with topics and subjects

Output format (JSON):
{{
  "themes": [
    {{
      "id": "temp_1",
      "label": "Theme Name",
      "description": "Description",
      "direct_clusters": [1, 2]
    }}
  ],
  "topics": [
    {{
      "id": "temp_1.1",
      "label": "Topic Name", 
      "description": "Description",
      "parent_id": "temp_1",
      "direct_clusters": [3]
    }}
  ],
  "subjects": [
    {{
      "id": "temp_1.1.1",
      "label": "Subject Name",
      "description": "Description", 
      "parent_id": "temp_1.1",
      "direct_clusters": [4, 5]
    }}
  ]
}}

Language: {language}
"""

PHASE2_DISCOVERY_REDUCE_PROMPT = """
You are merging multiple partial hierarchies into a unified codebook.

Survey Question: {survey_question}

Partial hierarchies to merge:
{hierarchies}

Your task:
1. Identify similar themes/topics/subjects across hierarchies and merge them
2. Preserve ALL cluster assignments - every cluster ID must appear in the output
3. Create a coherent unified structure
4. Renumber themes/topics/subjects for consistency (1, 1.1, 1.1.1, etc.)

Output the same JSON structure as the input hierarchies, but unified and renumbered.

Language: {language}
"""

PHASE3_ASSIGNMENT_PROMPT = """
You are assigning a survey response cluster to the appropriate place in a hierarchical codebook.

Survey Question: {survey_question}

Cluster to assign:
- ID: {cluster_id}
- Label: {cluster_label}
- Description: {cluster_description}
- Example responses:
{representatives}

Hierarchical Codebook:
{codebook}

Assign probability scores (0.0 to 1.0) for how well this cluster fits each theme, topic, and subject.
Probabilities at each level should sum to approximately 1.0.

Consider:
- Semantic similarity between cluster and codebook entries
- Hierarchical consistency (if assigned to subject 1.1.1, should also score high for topic 1.1 and theme 1)
- If cluster doesn't fit well anywhere, assign high probability to "other"

Output format (JSON):
{{
  "theme_assignments": {{
    "1": 0.8,
    "2": 0.1,
    "other": 0.1
  }},
  "topic_assignments": {{
    "1.1": 0.7,
    "1.2": 0.2,
    "other": 0.1
  }},
  "subject_assignments": {{
    "1.1.1": 0.9,
    "other": 0.1
  }}
}}

Language: {language}
"""

PHASE4_REFINEMENT_PROMPT = """
You are refining the final labels in a hierarchical codebook to ensure clarity and consistency.

Survey Question: {survey_question}

Current hierarchy with assignments:
{hierarchy_with_assignments}

Review and refine all labels to ensure:
1. Labels are maximum 4 words
2. Each label expresses exactly ONE idea
3. Labels are mutually exclusive within their level
4. Labels clearly communicate the concept in context of the survey question
5. Consistent terminology across the hierarchy

Output format (JSON):
{{
  "refined_labels": {{
    "themes": {{
      "1": "Refined Theme Name",
      "2": "Another Theme"
    }},
    "topics": {{
      "1.1": "Refined Topic",
      "1.2": "Another Topic"
    }},
    "subjects": {{
      "1.1.1": "Refined Subject"
    }}
  }}
}}

Language: {language}
"""