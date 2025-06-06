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
2. Add a segment_label (a thematic label)
3. Add a segment_description (a clarification of the label)

# Field Requirements

## segment_label
- A concise label of up to 5 words total, using ONLY ADJECTIVES AND NOUNS in {language}, that captures the CENTRAL MEANING of the segment
- ONLY return labels that reflect ONE idea, topic, concern, issue, or theme in response to the question: "{var_lab}"
- NEVER return multi-headed labels or combinations of multiple ideas
- Format: ALL_CAPS_WITH_UNDERSCORES
- Examples: "PRODUCT_QUALITY", "MORE_OPTIONS", "UNNECESSARY_COMPLEXITY"
- Language: {language}


## segment_description
- Rewrite the segment as a natural-sounding **first-person response** to the question: "{var_lab}"
- Make sure it sounds like something a person would actually say when answering the question
- Use a direct, conversational or instructional tone:
  - If the segment is a suggestion: use an imperative tone (e.g., "Maak...", "Laat...")
  - If the segment expresses a wish or opinion: use first-person (e.g., "Ik wil...", "Ik vind...")
- NEVER rephrase the segment as a third-person summary (e.g., "Wil dat de inhoud...") — that does not sound like a response
- CRITICAL: Do NOT add interpretations beyond what's in the original segment
- Language: {language}

# Special Cases
For meaningless responses (e.g., "?", "Don't know"):
- segment_label: "NA"
- segment_description: "NA"

# Segments to code:
{segments}

# Output Format
Return a valid JSON array with these fields for each segment:
- "segment_id": The original segment ID
- "segment_response": The original segment text
- "segment_label": Your thematic label in ALL_CAPS_WITH_UNDERSCORES
- "segment_description": Your clarifying description

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

PHASE1_DESCRIPTIVE_CODING_PROMPT = """
You are a {language} expert performing **descriptive coding** of short survey-response segments.

=========================
TASK
=========================
Give **one label** (Title Case noun phrase, ≤ 6 words).

=========================
RULES
=========================
• Describe only what is explicitly said.  
• No interpretation, sentiment, or judgment.  
• Do not combine multiple ideas.

=========================
INPUT
=========================
Question    : {survey_question}
Segment ID  : {cluster_id}
Segment text:
{representatives}

=========================
Output format (JSON):
=========================
{{
  "label": "Descriptive label (Title Case noun phrase, ≤ 6 words)",
}}

=========================
Language: {language}
=========================
"""

PHASE2_DISCOVERY_PROMPT = """
You are an expert in thematic analysis working in {language}.

Your task is to generate a structured codebook with three hierarchical levels based on descriptive codes from an open-ended survey.

RESEARCH QUESTION:
"{survey_question}"

INPUT DATA:
Below is a list of descriptive codes based on individual open-text survey responses. Each code captures one distinct idea expressed by a respondent.

=========================
 YOUR TASK:
=========================
Organize these descriptive codes into a three-level thematic hierarchy:

• Code (Level 3) – A concrete, specific consumer feedback point.  
• Topic (Level 2) – A focused subcategory grouping several related codes.  
• Theme (Level 1) – A broad, abstract domain that unites related topics.

=========================
GUIDING PRINCIPLES:
=========================
• Use **all provided codes** — do not omit or merge them.
• Ensure that every Code belongs to **exactly one Topic** and every Topic belongs to **exactly one Theme**.
• Group codes by **semantic similarity** and logical content.
• Labels must be **maximum 4 words**, in Title Case.
• Provide a clear **English description** for each theme, topic, and code.
• Use

=========================
OUTPUT FORMAT:
=========================
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
              "description": "What kind of feedback this captures"
            }}
          ]
        }}
      ]
    }}
  ]
}}

=========================
INPUT CODES:
{cluster_summaries}
=========================

All output must be in **{language}**.
"""

# PHASE3_THEME_JUDGER_PROMPT = """
# You are a {language} expert in qualitative research and thematic analysis.

# =========================
# YOUR TASK:
# =========================
# Critically evaluate how codes, topics, and themes are ORGANIZED within a fixed 3-level hierarchy. 
# Provide concrete, specific recommendations to improve their grouping based on semantic logic.

# =========================
# WHAT YOU'RE EVALUATING:
# =========================
# - Whether each code is placed under the most logical topic
# - Whether each topic is placed under the most logical theme  
# - Whether any groups should be added, split, moved, or renamed
# - Whether the current organization maximizes clarity for analysis

# =========================
# EVALUATION CRITERIA:
# =========================
# 1. LOGICAL COHERENCE - Do grouped items share clear semantic relationships?
# 2. MUTUAL EXCLUSIVITY - Can each code belong to only ONE topic without ambiguity?
# 3. COLLECTIVE EXHAUSTIVENESS - Does every code have an appropriate home?

# =========================
# THE HIERARCHY STRUCTURE (this is fixed - do not change):
# =========================
# - Theme (Level 1): Broad conceptual domain
# - Topic (Level 2): Focused subcategory
# - Code (Level 3): Specific feedback point

# =========================
# ALLOWED ACTIONS:
# =========================
# - ADD - Create new themes or topics (but NOT codes!) if existing ones don't fit
# - MOVE - Relocate a code to a different topic (use exact IDs and labels)
# - SPLIT - Divide a theme or topic into multiple parts
# - RENAME - Change a label for clarity

# CRITICAL: Keep all codes separate - do not merge or delete codes.

# ============================
# INPUT
# ============================
# Survey question:  
# "{survey_question}"

# Codebook to review:  
# {codebook}

# ============================
# OUTPUT
# ============================
# Return ONLY this JSON structure (no explanatory text):
# {{
#   "needs_revision": true/false,
#   "summary": "One sentence assessment",
#   "issues": [
#     "List each structural issue found"
#   ],
#   "actions": [
#     {{
#       "type": "ADD/MOVE/SPLIT/RENAME",
#       "details": "Specific instruction with IDs and labels"
#     }}
#   ]
# }}

# All output must be in **English**.
# """

PHASE3_THEME_JUDGER_PROMPT = """
You are a {language} expert in qualitative research evaluating a hierarchical codebook.

Survey question: "{survey_question}"

Codebook to review:
{codebook}

Evaluate the semantic coherence of this codebook - each grouping should unite semantically related elements.
The goal: ensure the codebook supports clear, meaningful, and non-redundant analysis.

============================
CHECK FOR:
============================
- Duplicate or overlapping concepts in different places
- Codes misaligned with their topic's core meaning
- Topics lacking coherence (mixing unrelated concepts)
- Themes lacking coherence (grouping unrelated topics)
- Any organization that defies common sense

A coherent structure means each element has ONE logical home, and all groupings are semantically unified.

============================
ALLOWED ACTIONS:
============================
- MOVE: Relocate misplaced codes (use exact IDs)
- ADD: Create new themes/topics for orphan codes
- SPLIT: Separate incoherent groupings
- RENAME: Clarify ambiguous labels

Keep all codes - do not merge or delete.

============================
OUTPUT:
============================
Return this JSON:
{{
  "needs_revision": true/false,
  "summary": "Main coherence issue in one sentence",
  "issues": [
    "Specific problems with IDs"
  ],
  "actions": [
    {{
      "type": "MOVE/ADD/SPLIT/RENAME",
      "details": "Specific fix with exact IDs and labels"
    }}
  ]
}}

Language: **English**
"""

PHASE4_THEME_REVIEW_PROMPT = """
You are a {language} expert in qualitative research and thematic analysis.

Your task is to **reorganize and improve** a hierarchical codebook based on instructions.

The codebook has the following structure:
• **Theme (Level 1)** – Broad conceptual domains  
• **Topic (Level 2)** – Mid-level subcategories  
• **Code (Level 3)** – Specific, discrete feedback ideas

============================
INPUT
============================
• Survey question:
"{survey_question}"

• Current codebook:
{current_codebook}

• Summary of feedback:
{summary}

• Specific issues to address:
{issues}

• Actions to take:
{actions}

============================
OUTPUT FORMAT
============================
Return the improved codebook structure in the same format as Phase 2:

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
              "description": "What kind of feedback this captures"
            }}
          ]
        }}
      ]
    }}
  ]
}}

============================
IMPORTANT
============================
- Return the COMPLETE restructured codebook
- Ensure all codes from the original are present  
- Apply all changes systematically
- Maintain logical ID numbering
- All output must be in **{language}**
"""

PHASE5_LABEL_REFINEMENT_PROMPT = """
You are a {language} expert in qualitative coding and codebook design.

Your task is to create meaningful **labels** and **descriptions** that capture the essence of each hierarchical level.

HIERARCHY:
• Theme (Level 1): Overarching conceptual domain
• Topic (Level 2): Unifying subcategory
• Code (Level 3): Specific feedback point

=======================
LABEL PRINCIPLES
=======================
• Maximum 4 words
• Capture the UNDERLYING CONCEPT, not surface features
• Themes/Topics: Find the deeper unifying principle
  ❌ BAD: Compound labels that just list components
  ✅ GOOD: Single concept that captures the essence
• Be conceptually precise - each word should add meaning

=======================
DESCRIPTION PRINCIPLES
=======================
• Maximum 20 words
• Add INSIGHT beyond the label
• Explain the WHY or HOW, not just repeat WHAT
• Use active voice when possible

EXAMPLES:
❌ BAD: Compound labels like "A and B and C"
✅ GOOD: Find the underlying concept that unifies A, B, and C

❌ BAD Description: "Request for [repeating the label]"
✅ GOOD Description: Explain the benefit, impact, or deeper meaning

=======================
YOUR TASK
=======================
Review each label and description. Ask yourself:
1. Does the label capture the ESSENCE, not just list components?
2. Does the description ADD VALUE beyond restating the label?
3. For Themes/Topics: Is there a deeper unifying concept?
4. For Codes: Is the consumer benefit or concern clear?

=======================
INPUT
=======================
Survey context: {survey_question}

Current codebook:
{codebook}

=======================
OUTPUT FORMAT
=======================
Only return entries that need improvement:

{{
  "refined_themes": {{
    "1": {{
      "label": "Improved conceptual label",
      "description": "Insight into what unifies this theme"
    }}
  }},
  "refined_topics": {{
    "1.1": {{
      "label": "Focused concept",
      "description": "How this topic addresses consumer needs"
    }}
  }},
  "refined_codes": {{
    "1.1.1": {{
      "label": "Precise feedback point",
      "description": "What improvement this represents for consumers"
    }}
  }}
}}

Remember: Find the deeper meaning. What do consumers REALLY care about?
Language: {language}
"""


PHASE6_ASSIGNMENT_PROMPT = """
You are a {language} coder assigning a cluster to the existing codebook.

=========================
INSTRUCTIONS
=========================
1. Choose exactly ONE path (Theme → Topic → Code).  
2. Provide a confidence score 0–1.  
3. If confidence < 0.60, assign to Theme 99 → Topic 99.1 → Code 99.1.1 (“Other”).  
4. Add a one-sentence note explaining your choice.

=========================
INPUT
=========================
Survey Question: {survey_question}

Cluster to assign:
- ID          : {cluster_id}
- Label       : {cluster_label}
- Examples    :
{cluster_representatives}

Codebook
{codebook}

=========================
Output format (JSON):
=========================
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

Note: Ensure all IDs exist in the provided codebook structure.
Language: {language}
"""





