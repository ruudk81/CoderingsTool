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

# VALIDATION_CHECK_PROMPT = """
# Review this hierarchy and verify:

# 1. Total clusters in input: {total_expected}
# 2. Total clusters assigned: {total_assigned}
# 3. Duplicate assignments: {duplicates}
# 4. Missing clusters: {missing}

# Is this hierarchy complete and valid? If not, what needs to be fixed?
# """

# HISTORY:

# BATCH_SUMMARY_PROMPT = """
# You are a qualitative researcher performing thematic analysis.

# You are given a batch of micro-clusters. Your task is to organize them into a hierarchical structure.

# **CRITICAL RULES**:
# 1. Use EVERY micro-cluster ID exactly ONCE
# 2. Never assign the same cluster to multiple places
# 3. Output MUST be valid JSON - no comments, no emojis, no checkboxes

# **STRUCTURAL FLEXIBILITY**:
# - Create sub-themes only when they add meaningful organization
# - For simple/unified themes, assign clusters directly to the theme
# - Avoid creating artificial subdivisions

# **Output format (STRICT JSON - NO COMMENTS ALLOWED):**
# {{
#   "batch_id": "{batch_id}",
#   "hierarchy": {{
#     "1": {{
#       "node": "Theme Name",
#       "subthemes": {{
#         "1.1": {{
#           "node": "Subtheme Name", 
#           "micro_clusters": [0, 1]
#         }}
#       }},
#       "direct_clusters": []
#     }},
#     "2": {{
#       "node": "Simple Theme",
#       "subthemes": {{}},
#       "direct_clusters": [3, 4]
#     }}
#   }}
# }}

# IMPORTANT: Return ONLY valid JSON. No comments (//), no text outside the JSON structure, no checkboxes.

# Variable label/context: {var_lab}
# Language: {language}
# Batch ID: {batch_id}

# Micro-cluster batch:
# {batch_clusters}
# """

# REDUCE_SUMMARY_PROMPT = """
# You are a qualitative researcher merging thematic structures from multiple batches into one unified codebook.

# Each structure contains a partial thematic hierarchy with Level 1 and Level 2 codes. Your task is to synthesize these into a single consistent hierarchy.

# **Instructions:**
# - Merge similar themes and subthemes where meanings clearly overlap
# - Each micro-cluster ID must appear exactly once in the output
# - Output MUST be valid JSON format - no comments allowed

# **Output format (STRICT JSON):**
# {{
#   "unified_hierarchy": {{
#     "1": {{
#       "node": "Unified Theme Name",
#       "subthemes": {{
#         "1.1": {{
#           "node": "Subtheme Name",
#           "micro_clusters": [0, 1, 9]
#         }},
#         "1.2": {{
#           "node": "Another Subtheme",
#           "micro_clusters": [2, 15]
#         }}
#       }},
#       "direct_clusters": []
#     }},
#     "2": {{
#       "node": "Another Theme",
#       "subthemes": {{
#         "2.1": {{
#           "node": "Subtheme Name",
#           "micro_clusters": [3, 4, 10]
#         }}
#       }},
#       "direct_clusters": []
#     }}
#   }}
# }}

# Variable label/context: {var_lab}
# Language: {language}

# Input hierarchies to merge:
# {summaries}
# """


# BATCH_SUMMARY_PROMPT = """
# You are a qualitative researcher performing thematic analysis.

# You are given a batch of micro-clusters (each cluster contains ~3-5 short statements with similar meaning). Your task is to analyze this batch and produce a hierarchical code structure.

# **IMPORTANT STRUCTURAL FLEXIBILITY**:
# - Some themes naturally have sub-themes (e.g., "Health" might have "Salt Reduction", "Fat Reduction", etc.)
# - Other themes are standalone concepts that don't need sub-themes (e.g., "Satisfaction with Current Product")
# - You should create sub-themes ONLY when they add meaningful organization
# - If a theme is simple/unified, assign micro-clusters directly to the theme level

# **Instructions:**
# - Identify Level 1 nodes (broad themes)
# - For complex themes: create Level 2 nodes (specific subthemes) and assign micro-clusters there
# - For simple themes: assign micro-clusters directly to the Level 1 theme
# - Each node name must be no more than 4 words and semantically clear
# - Each micro-cluster ID must be assigned exactly once

# **Output format:**
# Return a JSON object with this exact structure:
# {{
#   "batch_id": "{batch_id}",
#   "hierarchy": {{
#     "1": {{
#       "node": "Complex Theme Name",
#       "subthemes": {{
#         "1.1": {{
#           "node": "Subtheme Name",
#           "micro_clusters": [0, 1]
#         }},
#         "1.2": {{
#           "node": "Another Subtheme",
#           "micro_clusters": [2]
#         }}
#       }},
#       "direct_clusters": []  // Empty for themes with subthemes
#     }},
#     "2": {{
#       "node": "Simple Theme Name",
#       "subthemes": {{}},  // Empty for simple themes
#       "direct_clusters": [3, 4]  // Clusters assigned directly to theme
#     }}
#   }}
# }}

# Variable label/context: {var_lab}
# Language: {language}
# Batch ID: {batch_id}

# Micro-cluster batch:
# {batch_clusters}
# """

# REDUCE_SUMMARY_PROMPT = """
# You are a qualitative researcher merging thematic structures from multiple batches into one unified codebook.

# Each structure contains a partial thematic hierarchy with Level 1 and Level 2 codes. Your task is to synthesize these into a single consistent hierarchy.

# **Instructions:**
# - Merge similar themes and subthemes where meanings clearly overlap.
# - Ensure Level 1 and Level 2 themes are **mutually exclusive**, **concise**, and **clearly distinguishable**.
# - Each node name must be no more than 4 words.
# - Adjust theme labels if needed to clarify differences.
# - Reassign Level 1 and Level 2 numbers to maintain proper hierarchy.
# - Preserve all micro-cluster assignments.
# - **CRITICAL**: When merging, ensure each micro-cluster ID appears in exactly ONE location. If you see the same cluster ID in multiple places, keep it in only the most appropriate location.

# **Duplicate Prevention Rules:**
# 1. Before assigning a cluster ID, check if it already exists elsewhere
# 2. If a cluster ID appears in multiple input hierarchies, assign it to the most semantically appropriate theme/subtheme
# 3. Never duplicate cluster IDs in your output

# **Output format:**
# Return a JSON object with this structure:
# {{
#   "unified_hierarchy": {{
#     "1": {{
#       "node": "Unified Theme Name",
#       "subthemes": {{
#         "1.1": {{
#           "node": "Subtheme Name",
#           "micro_clusters": [0, 1, 9]  // No duplicates across entire structure
#         }},
#         "1.2": {{
#           "node": "Another Subtheme",
#           "micro_clusters": [2, 15]  // Each ID unique
#         }}
#       }}
#     }},
#     "2": {{
#       "node": "Another Theme",
#       "subthemes": {{
#         "2.1": {{
#           "node": "Subtheme Name",
#           "micro_clusters": [3, 4, 10]  // Never repeat IDs from above
#         }}
#       }}
#     }}
#   }}
# }}

# Variable label/context: {var_lab}
# Language: {language}

# Input hierarchies to merge:
# {summaries}
# """


# HIERARCHICAL_LABELING_PROMPT = """
# You are finalizing a hierarchical codebook for thematic analysis based on qualitative data.

# You are given:
# - A consolidated hierarchical structure from previous analysis
# - A list of all original micro-clusters for reference

# **STRUCTURAL GUIDELINES**:
# 1. Some themes naturally require topics (sub-themes) for organization
# 2. Other themes are standalone and should NOT have artificial sub-divisions
# 3. Create topics ONLY when they represent meaningfully different aspects of a theme
# 4. For simple/unified themes, assign clusters directly to the theme level

# **Examples of when to use topics**:
# - "Health & Nutrition" → "Salt Reduction", "Fat Reduction", "More Vegetables" (distinct aspects)
# - "Product Range" → "Vegetarian Options", "Vegan Options", "Portion Sizes" (different categories)

# **Examples of when NOT to use topics**:
# - "Satisfaction" → Just assign satisfied responses directly (don't create "Types of Satisfaction")
# - "Price" → If all clusters are about "lower price", don't subdivide

# **Your task**:
# - Review the hierarchy and remove unnecessary topic levels
# - For simple themes, move micro-clusters to direct assignment
# - Ensure each micro-cluster appears exactly once
# - All micro-clusters must be assigned

# **Output format:**
# {{
#   "themes": [
#     {{
#       "id": "1",
#       "name": "Complex Theme",
#       "description": "Theme requiring sub-organization",
#       "topics": [
#         {{
#           "id": "1.1",
#           "name": "Specific Aspect",
#           "description": "One aspect of the theme",
#           "micro_clusters": [2, 4]
#         }}
#       ],
#       "direct_clusters": []  // Empty when using topics
#     }},
#     {{
#       "id": "2",
#       "name": "Simple Theme",
#       "description": "Standalone theme",
#       "topics": [],  // No topics needed
#       "direct_clusters": [0, 3, 8]  // Clusters assigned directly
#     }}
#   ]
# }}

# Language: {language}

# Consolidated hierarchy:
# {final_summary}

# All micro-clusters for reference (each must be assigned exactly once):
# {micro_cluster_list}
# """