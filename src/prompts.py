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

CLUSTER_LABELING_PROMPT = """
You are a {language} expert in thematic analysis of survey responses.

# Survey Question Context
These responses were given to: "{var_lab}"

# Task Overview
Create a descriptive label for a {cluster_type} cluster containing {n_items} similar responses.

# Cluster Type Definition
This is a {cluster_type} cluster, which represents {cluster_type_description}.

# Cluster Content
## Representative responses:
{responses}

## Associated descriptive codes:
{codes}

## Code descriptions:
{descriptions}

# Label Requirements
1. Length: 2-5 words maximum
2. Language: {language}
3. Format: Natural {language} phrase (not ALL_CAPS)
4. Content: Use ADJECTIVES and NOUNS that capture the CENTRAL MEANING
5. Specificity: 
   - theme: Broad enough to encompass multiple topics
   - topic: Specific enough to distinguish from other topics
   - code: Detailed enough to capture specific issues

# Instructions
Based on the survey question context, create labels that:
- Directly relate to what's being asked in the survey
- Use domain-appropriate terminology based on the question topic
- Capture the essence of the clustered responses

# Output Format
You must return a JSON object with exactly these fields:
- "label": A concise, descriptive label in {language}
- "confidence": Your confidence score (0.0 to 1.0)
- "reasoning": Brief explanation in {language} of why this label fits
"""

RESPONSE_SUMMARY_PROMPT = """
You are a {language} expert summarizing survey responses.

# Survey Question
"{var_lab}"

# Original Response
{original_response}

# Assigned Labels
Themes: {themes}
Topics: {topics}
Codes: {codes}

# Task
Create a 1-2 sentence summary in {language} that:
1. Captures the main points from the original response
2. Uses the assigned labels as context
3. Sounds natural as a summary of survey feedback
4. Maintains the respondent's perspective
5. Relates directly to the survey question

# Output
Write a concise summary that helps understand what the respondent is saying about the topic asked in the survey question.
"""

THEME_SUMMARY_PROMPT = """
You are a {language} expert creating a summary for a theme in survey response analysis.

# Theme
{theme_label}

# Survey Question
"{var_lab}"

# Most Representative Codes and Examples
{representative_items}

# Task
Based on these representative examples and the survey question context, create a comprehensive summary that:
1. Captures the main concerns or topics within this theme
2. Explains how the different codes relate to each other
3. Provides insight into what respondents are saying overall about this theme
4. Relates directly to what was asked in the survey question

# Requirements
- Write in {language}
- Be specific about respondent views and concerns
- Use 2-3 sentences maximum
- Focus on the common patterns across the examples
- Ensure the summary makes sense in the context of the survey question

# Output
Write a clear, insightful summary that helps understand what this theme represents in relation to the survey question.
"""

# New Labeller prompts for hierarchical labeling system

INITIAL_LABEL_PROMPT = """
You are analyzing survey responses to the question: "{var_lab}"

Please label the following clusters based on their content. Each cluster shows the MOST REPRESENTATIVE items,
selected using cosine similarity to the cluster centroid. These are the items that best capture the essence
of each cluster.

For each cluster, provide:
1. A concise, descriptive label that captures the main theme
2. 3-5 keywords that represent the cluster
3. A confidence score (0.0-1.0)

Focus on creating labels that directly answer or relate to the survey question. Base your labels primarily
on the representative items shown, as they are the most characteristic of each cluster.

Language: {language}

{clusters}

REQUIRED OUTPUT FORMAT:
Return a JSON object with a single key "labels", containing an array of objects with these fields:
- "cluster_id": The cluster ID
- "label": A concise descriptive label
- "keywords": An array of 3-5 keywords
- "confidence": A confidence score between 0.0 and 1.0
"""

SIMILARITY_SCORING_PROMPT = """
You are tasked with comparing clusters based on their labels and most representative 
descriptive codes and code descriptions. Please give a score from 0 to 1 for how 
similar they are from the point of view of addressing the research question.

Research question: "{var_lab}"
Language: {language}

Scoring scale:
- 0 = maximally differentiated (completely different themes)
- 0.5 = pretty similar, probably sharing an overarching theme or response pattern
- 1 = not positively differentiated at all, there is no difference or the difference 
      does not help in any way to explain how respondents answered the research 
      question differently

For each pair of clusters, also indicate whether they should be merged (score >= 0.7).

{cluster_pairs}

REQUIRED OUTPUT FORMAT:
Return a JSON object with a single key "scores", containing an array of objects with these fields:
- "cluster_id_1": First cluster ID
- "cluster_id_2": Second cluster ID  
- "score": Similarity score between 0.0 and 1.0
- "merge_suggested": Boolean indicating if merge is recommended (score >= 0.7)
- "reason": Brief explanation of the similarity/difference
"""

HIERARCHY_CREATION_PROMPT = """
You are organizing survey response clusters for the question: "{var_lab}"

Your task is to group these clusters into {level}-level categories that represent major themes.
Each {level} should:
1. Represent a distinct, broad theme related to the survey question
2. Contain clusters that share conceptual similarity
3. Be meaningful and interpretable in the context of the survey

Language: {language}

Here are the clusters to organize:
{clusters}

Group these clusters into 3-7 {level} categories. Each cluster should belong to exactly one category.

REQUIRED OUTPUT FORMAT:
Return a JSON object with a single key "{level}s", containing an array of objects with these fields:
- "label": A descriptive name for the {level}
- "cluster_ids": An array of cluster IDs belonging to this {level}
- "explanation": Brief explanation of what unifies these clusters
"""

HIERARCHICAL_THEME_SUMMARY_PROMPT = """
You are summarizing a theme from survey responses to the question: "{var_lab}"

Theme: {theme_label}

This theme contains the following structure:
{theme_structure}

Your task is to:
1. Write a comprehensive summary of this theme (2-3 paragraphs)
2. Explain how this theme addresses the research question

Focus on:
- What respondents in this theme are expressing
- Common patterns or perspectives
- How these responses relate to the research question
- Key insights or takeaways

Language: {language}

REQUIRED OUTPUT FORMAT:
Return a JSON object with these fields:
- "summary": A comprehensive 2-3 paragraph summary of the theme
- "relevance": A 1-2 sentence explanation of how this theme addresses the research question
"""


