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


TOPIC_CREATION_PROMPT = """
You are a {language} expert in organizing micro-cluster labels into **TOPICS** (Level 2 nodes).
Each topic contains 1 or more micro-clusters that represent ideas that express different ideas, issues or concerns in light of this research question:
"{var_lab}"

TASK:
Group the provided cluster labels into topic categories that:
- Are mutually exclusive
- Represent a clear, specific area of concern or meaning
- Can later be grouped under broader themes

OUTPUT FORMAT:
Return a JSON object with the following structure:

{
"topics": [
{
"label": "Portie grootte aanpassingen",
"cluster_ids": ["38", "15", "22"],
"explanation": "These clusters deal with requests to modify portion sizes and offer portion variety options."
},
{
"label": "Ingrediënt kwaliteit verbetering",
"cluster_ids": ["12", "29"],
"explanation": "These responses focus on using fresher, higher quality ingredients in meal preparation."
}
]
}

Language: {language}

Input cluster labels:
{clusters}
"""

THEME_CREATION_PROMPT = """
You are a {language} expert in grouping topics into THEMES (Level 1 nodes).
Themes represent broad conceptual areas that organize topics into coherent, high-level insights related to the research question:
"{var_lab}"

TASK:
Group the topics into themes that:
- Each contain 2 or more topics
- Reflect a distinct major dimension of the issue
- Are clearly different from each other
- Help interpret how respondents experience or approach the research question

OUTPUT FORMAT:
Return a JSON object with the following structure:

{
"themes": [
{
"label": "Product samenstelling en kwaliteit",
"topic_labels": ["Portie grootte aanpassingen", "Ingrediënt kwaliteit verbetering"],
"explanation": "This theme groups topics reflecting how respondents want producers to modify the physical composition and quality of ready-made meals."
},
{
"label": "Voedingswaarde en gezondheid",
"topic_labels": ["Zout en suiker vermindering", "Gezonde ingrediënten toevoegen"],
"explanation": "This theme centers on health-focused improvements that respondents want to see in nutritional content."
}
]
}

Language: {language}

Input topic labels:
{topics}
"""

HIERARCHICAL_THEME_SUMMARY_PROMPT = """
You are a {language} expert in analyzing a theme derived from survey responses to the question:
"{var_lab}"

Theme: {theme_label}

Structure:
{theme_structure}

TASK:
- Write an analytical summary of this theme. Your goal is to explain what the theme reveals about how respondents interpret or experience the issue.

FOCUS:
- What dimension of the research question does this theme address?
- What are the key ideas, concerns, or viewpoints expressed?
- How do the topics within this theme offer different perspectives or nuances?
- What are the implications for interpreting the overall dataset?

OUTPUT FORMAT:
Return a JSON object with the following structure:

{
"summary": "This theme highlights respondents' shared focus on improving the physical composition of ready-made meals, particularly around portion sizes and ingredient quality. Topics within this theme show nuanced differences, from requests for larger portions to demands for fresher ingredients. Collectively, they reveal that consumers want more control over meal composition and higher quality standards.",
"relevance": "This theme explains how respondents want producers to fundamentally change what goes into meals and how much is provided."
}

Language: {language}
"""


