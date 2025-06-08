SPELLCHECK_INSTRUCTIONS = """
You are a {language} language expert specializing in correcting misspelled words in open-ended survey responses.

=========================
INSTRUCTIONS
=========================
Your need to correct responses that contain placeholder tokens indicating spelling mistakes.

You will receive input. This input contains correction tasks:
- A sentence with one or more <oov_word> placeholders.
- A list of misspelled words, in the same order as the placeholders.
- A list of suggested corrections, in the same order.

For each correction task:
- Replace every <oov_word> placeholder with the best possible correction of the corresponding misspelled word, taking into account the meaning and context of the survey question.
- If a better correction exists than the ones provided, prefer that.
- You may split a misspelled word into two words **only if** the split preserves the intended meaning and fits grammatically.
- If no suitable correction is possible, return "[NO RESPONSE]" as the corrected sentence for that task.

=========================
INPUT
=========================
Survey Question:
"{var_lab}"

Below are the tasks to process:
{tasks}

=========================
REQUIRED OUTPUT FORMAT:
=========================
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

Ensure that your output is a valid JSON object with a single key "corrections", whose value is an array of objects. 
Each object in the "corrections" array must have exactly these fields:
- "respondent_id": "ID_FROM_TASK"
- "corrected_response": "The fully corrected response"
=========================
"""


GRADER_INSTRUCTIONS = """
You are a {language} grader evaluating open-ended survey responses.

=========================
TASK
=========================
Your task is to determine whether each response is **meaningless**.

=========================
DECISION CRITERIA
=========================
A response should be considered **meaningless** only if:
- It only expresses uncertainty or lack of knowledge (e.g., “I don’t know”, “N/A”, “Not applicable”).
- It simply repeats the question without adding any new content.
- It consists of random characters, gibberish, or filler text with no semantic meaning (e.g., “asdfkj”, “lorem ipsum”).

=========================
INPUT
=========================
Survey question:
"{var_lab}"

Responses to evaluate:
{responses}

=========================
REQUIRED OUTPUT FORMAT:
=========================
Return a **JSON array**. Each object in the array must contain exactly:
- `"respondent_id"`: (string or number) The respondent's ID
- `"response"`: (string) The exact response text
- `"quality_filter"`: (boolean) `true` if the response is meaningless, `false` otherwise
=========================
"""

SEGMENTATION_PROMPT = """
You are a helpful {language} expert in analyzing survey responses. 
You are taked with segmenting free-text responses into distinct, standalone ideas.

=========================
INSTRUCTION
=========================
Break each response into the smallest meaningful standalone units, where each segment represents EXACTLY ONE:
- Opinion
- Preference
- Issue
- Topic
- Idea
- Response pattern

=========================
SEGMENTATION RULES
=========================
1. Split at conjunctions (and, or, but, also) when they connect DIFFERENT ideas or topics
2. Split listed items into separate segments (e.g., "milk and sugar" → "milk", "sugar")
3. When items share context, preserve that context in each segment:
   Example: "I like milk and sugar in my coffee" →
   - "I like milk in my coffee"
   - "I like sugar in my coffee"
4. Use the respondent's exact words - do not paraphrase or correct
5. Keep meaningless responses (e.g., "Don't know", "?") as a single segment

=========================
INPUT
=========================
Survey question:
"{var_lab}"

Response to segment:
{response}

=========================
OUTPUT
=========================
Return a JSON array with these fields for each segment:
- "segment_id": A sequential number as string ("1", "2", etc.)
- "segment_response": The exact segmented text with necessary context preserved
=========================
"""

REFINEMENT_PROMPT = """
You are a {language} expert in semantic segmentation of survey responses.

=========================
TASK
=========================
Review and refine response segments derived from responses to this survey question: "{var_lab}"

=========================
INSTRUCTIONS
=========================
1. Ensure tha each segment contains ONLY ONE standalone idea or response segment
2. Split any segment that still contains multiple distinct ideas or response segments
3. Preserve necessary context when splitting compound statements

=========================
REFINEMENT RULES
=========================
- Split segments that contain multiple distinct ideas or response segments (connected by conjunctions like "and", "or")
- When splitting, duplicate any necessary context to make each segment meaningful:
  * For example: "I like milk and sugar in my coffee" should become two segments:
    - "I like milk in my coffee"
    - "I like sugar in my coffee"
- Only duplicate words that appear in the original response
- Never merge segments
- Keep meaningless responses intact (e.g., "Don't know")

=========================
INPUT
=========================
Segments to refine:
{segments}

=========================
OUTPUT FORMAT
=========================
Return a JSON array with these fields for each refined segment:
- "segment_id": A sequential number as string ("1", "2", etc.)
- "segment_response": The properly segmented text using ONLY words from the original response

Example 1: Multiple ideas with shared context
Input: {{"segment_id": "1", "segment_response": "I like milk and sugar in my coffee"}}
Output: [
  {{"segment_id": "1", "segment_response": "I like milk in my coffee"}},
  {{"segment_id": "2", "segment_response": "I like sugar in my coffee"}}
]

Example 2: Multiple ideas without shared context
Input: {{"segment_id": "1", "segment_response": "The price is too high and the quality is low"}}
Output: [
  {{"segment_id": "1", "segment_response": "The price is too high"}},
  {{"segment_id": "2", "segment_response": "the quality is low"}}
]
=========================
"""

CODING_PROMPT = """
You are a {language} expert in thematic analysis of survey responses.

=========================
TASK
=========================
Code each segment from responses to this survey question: "{var_lab}"
Each segment is a standalone sentence or clause extracted from a full response.

=========================
INSTRUCTIONS
=========================
For Each Segment, You Will:
1. Keep the original segment_id and segment_response
2. Add a segment_label (a thematic label)
3. Add a segment_description (a clarification of the label)

=========================
REQUIREMENTS
=========================
For segment Labels:
- A concise label of up to 5 words total, using ONLY ADJECTIVES AND NOUNS in {language}, that captures the CENTRAL MEANING of the segment
- ONLY return labels that reflect ONE idea, topic, concern, issue, or theme in response to the question: "{var_lab}"
- NEVER return multi-headed labels or combinations of multiple ideas
- Format: ALL_CAPS_WITH_UNDERSCORES
- Examples: "PRODUCT_QUALITY", "MORE_OPTIONS", "UNNECESSARY_COMPLEXITY"
- Language: {language}

For segment descriptions:
- Rewrite the segment as a natural-sounding **first-person response** to the question: "{var_lab}"
- Make sure it sounds like something a person would actually say when answering the question
- Use a direct, conversational or instructional tone:
  - If the segment is a suggestion: use an imperative tone (e.g., "Maak...", "Laat...")
  - If the segment expresses a wish or opinion: use first-person (e.g., "Ik wil...", "Ik vind...")
- NEVER rephrase the segment as a third-person summary (e.g., "Wil dat de inhoud...") — that does not sound like a response
- CRITICAL: Do NOT add interpretations beyond what's in the original segment
- Language: {language}

**Special Cases**
For meaningless responses (e.g., "?", "Don't know"):
- segment_label: "NA"
- segment_description: "NA"

=========================
INPUT
=========================
Segments to code:
{segments}

=========================
OUTPUT FORMAT
=========================
Return a valid JSON array with these fields for each segment:
- "segment_id": The original segment ID
- "segment_response": The original segment text
- "segment_label": Your thematic label in ALL_CAPS_WITH_UNDERSCORES
- "segment_description": Your clarifying description

Ensure all output is written in {language}, unless the code is "NA".
=========================
"""

MERGE_PROMPT = """
You are evaluating whether clusters of survey responses represent meaningfully different answers to this survey question:
"var_lab}"

=========================
INSTRUCTIONS
=========================
You need to determine whether each pair of response clusters should be merged or remain separate.
The key question for each comparison is: "Do these clusters represent meaningfully different responses to the research question, or are they essentially saying the same thing?"

Language: {language}

=========================
DECISION CRITERIA
=========================
YES (merge) ONLY IF:
- Both clusters express essentially the same response pattern in light of the survey question.
- The differences between them are minimal or irrelevant to the survey question
- A survey analyst would reasonably group these responses together as the same type of answer

NO (don't merge) IF:
- The clusters represent distinct response pattersn, viewpoints, suggestions, ideas or concerns
- They focus on different aspects in addressing the survey question
- They provide unique or complementary feedback or information
- They represent different topics even within the same broad theme
- There is ANY meaningful differentiation relevant to understanding survey responses

Important Guidelines:
- Focus SPECIFICALLY on the survey question context
- Base decisions on the MOST REPRESENTATIVE responses in each cluster (shown by cosine similarity to centroid)
- Be conservative - when in doubt, keep clusters separate
- Consider semantic meaning, not just surface-level wording

=========================
INPUT
=========================
{cluster_pairs}

=========================
REQUIRED OUTPUT FORMAT:
=========================
Return a JSON object with a single key "decisions", containing an array of objects with these fields:
- "cluster_id_1": First cluster ID
- "cluster_id_2": Second cluster ID  
- "should_merge": Boolean (true ONLY if clusters are not meaningfully differentiated)
- "reason": Brief explanation of your decision (1-2 sentences maximum)
=========================
"""


PHASE1_DESCRIPTIVE_CODING_PROMPT = """
You are an expert in descriptive coding working in {language}.

=========================
TASK
=========================
You are performing descriptive coding on segments from open-ended survey responses.
Provide one concise label that captures what respondents are expressing in these segments.

=========================
CODING PRINCIPLES
=========================
- Stay close to the data: Use respondents' own concepts
- Be descriptive: Capture what is said, not why
- Be specific: Focus on the distinct pattern in these segments
- Be concise: Maximum 5 words
- Use Title Case for labels

=========================
RULES
=========================
- Describe only what is explicitly stated
- No interpretation or inference beyond the text
- No evaluation or judgment
- Focus on a single coherent idea
- If segments express multiple ideas, identify the primary pattern

=========================
OUTPUT FORMAT
=========================
{{
  "label": "Descriptive Label Here"
}}

Output the label in **{language}**.

=========================
INPUT
=========================
Survey question: {survey_question}

Segment ID: {cluster_id}

Representative segments:
{representatives}
=========================
"""

PHASE2_EXTRACT_THEMES_PROMPT = """
You are an expert in thematic analysis working in {language}.

=========================
TASK
=========================
You are tasked with extracting themes from initial codes of survey responses to an open-ended survey question.
These themes will be used for a codebook with the following structure:
- **Themes (Level 1)** – A broad and coherent pattern in the data that addresses the research question. 
- **Topics (Level 2)** – A specific facet or dimension of a theme. 
- **Codes (Level 3)** – A short descriptive label grounded in data capturing the essence of responses for a segment of the data.

You ONLY need to discover the themes.

=========================
INSTRUCTIONS
=========================
Each theme must be underpinned by a central organizing concept or a singular latent idea that makes semantic sense in light of the survey question.

Additionally, each theme must be:
- **Semantically meaningful**: The theme is meaningful in light of the survey question
- **Thematically coherent**: The label reflects a central organizing concept or singular latent idea
- **Conceptually clear**: The meaning of the theme is clear and unambiguous

The number of themes should be:
- **Large enough** to ensure collective exhaustion of all initial codes (every code should naturally fit under a theme)
- **Small enough** to maintain meaningful distinctions and allow for topic-level organization
- **Balanced** to avoid themes that are too broad (becoming meaningless) or too narrow (preventing proper topic grouping)

=========================
OUTPUT
=========================
Return a JSON object with a single key "themes" containing an array of theme names.
Each theme name should be 1-4 words, capturing the essence of a major pattern.

{{"themes": ["Theme Name 1", "Theme Name 2", "Theme Name 3", ...]}}

Language: {language}

=========================
INPUT
=========================
Survey question: {survey_question}

Extract themes from these initial codes:
{codes}
=========================
"""

PHASE2_GROUP_TOPICS_PROMPT = """
You are an expert in thematic analysis working in {language}.

=========================
TASK
=========================
You are tasked with organizing initial codes into topics under the provided themes.
These will be used for a codebook with the following structure:
- **Themes (Level 1)** – A broad and coherent pattern in the data that addresses the research question. 
- **Topics (Level 2)** – A specific facet or dimension of a theme. 
- **Codes (Level 3)** – A short descriptive label grounded in data capturing the essence of responses for a segment of the data.

You ONLY need to create and assign topics under each theme.

=========================
INSTRUCTIONS
=========================
Each topic must group codes with shared meaning and help organize nuance and complexity within a theme.

Additionally, each topic must be:
- **Semantically meaningful**: The topic is meaningful in light of the theme it is positioned under
- **Topically coherent**: A topic must reflect a unified concept
- **Conceptually clear**: The meaning of the topic is clear and unambiguous

The number of topics per theme should be:
- **Sufficient** to capture the distinct dimensions and facets within each theme
- **Parsimonious** to avoid artificial subdivisions that fragment related codes
- **Comprehensive** to ensure every code has a natural home within the topic structure
- **Balanced** such that topics represent meaningful distinctions without becoming too granular or too broad

=========================
OUTPUT
=========================
Return a JSON object with theme-topic structure.
Topic names should be 1-4 words each.

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

Language: {language}
=========================
INPUT
=========================
Survey question: {survey_question}

Themes identified:
{themes}

Organize these initial codes into topics:
{codes}
=========================
"""

PHASE2_CREATE_CODEBOOK_PROMPT = """
You are an expert in thematic analysis working in {language}.

=========================
TASK
=========================
You are tasked with creating a codebook that organizes initial codes representing segments of survey responses to an open-ended survey question.
The codebook has the following structure:
- **Themes (Level 1)** – A broad and coherent pattern in the data that addresses the research question. 
- **Topics (Level 2)** – A specific facet or dimension of a theme. 
- **Codes (Level 3)** – A short descriptive label grounded in data capturing the essence of responses for a segment of the data.

=========================
INSTRUCTIONS
=========================
The codebook needs to consist of Themes, Topics and Codes that are:
- **Semantically meaningful**: The labels of the Themes, Topics and Codes need to be semantically meaningful in light of the survey question
- **Thematically coherent**: All codes and topics within a theme need to reflect a unified idea
- **Internally consistent**: There are no contradictory or unrelated codes in a topic or theme
- **Conceptually clear**: The meaning of the labels for each theme, topic and code is clear and unambiguous

Create a complete hierarchical codebook by:
1. Using the provided theme-topic structure
2. Assigning EVERY initial code to the most appropriate topic
3. Including the numeric ID from [source ID: X] in the source_codes array
4. Creating meaningful descriptions for all levels

=========================
OUTPUT
=========================
Return as JSON with this exact structure:
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

Language: {language}

=========================
INPUT
=========================
Survey question: {survey_question}

Theme-topic structure:
{structure}

Assign these initial codes to the structure:
{codes}
=========================
"""

PHASE3_THEME_JUDGER_PROMPT = """
You are an expert in qualitative research and thematic analysis working in {language}.

You are tasked with auditing a codebook.
This codebook has the following structure for organizing open-ended survey responses:
- **Themes (Level 1)** – A theme is underpinned by a central organizing concept and provides meaningful insight beyond surface-level observations.
- **Topics (Level 2)** – Topics group codes with shared meaning and help organize nuance and complexity within a theme.
- **Codes (Level 3)** – Codes reflect response segments with similar meaning and represent the smallest unit of meaning in this hierarchy.
  - Each code contains source_codes (numeric IDs from initial clustering)

=========================
INSTRUCTIONS
=========================
CRITICAL: Focus on the MEANING of labels, not their IDs or position in the hierarchy. 
Read the label of a Theme, Topic or Code and ask "Does this semantically belong here (Theme -> Topic -> Code)?"

Evaluate if the codebook is:
- **Semantically coherent**: each code needs to semantically fit the full path it is placed under: Theme -> Topic -> Code.
- **Internally consistent**: No codes that semantically contradict the meaning of a topic, and no topics that semantically contradict the meaning of a theme.
- **Collectively exhaustive**: No unused source codes.

=========================
REVISION RECOMMENDATIONS
=========================
If all criteria are met:
    - set needs_revision: false 

Otherwise:
    - set needs_revision: true
   
=========================
REVISION ACTIONS (needs_revision: true)
=========================
First, incorporate unused codes WHERE THEY SEMANTICALLY BELONG:
- Read the LABEL of the unused code (ignore the ID number)
- Ask: "What is this code actually about based on its label?"
- Match semantic meaning of the code's label with Themes and Topics in the following order: Themes -> Topics
- If SAME MEANING as existing code:
  **ADD**: Add source_codes from unused code to an existing code that shares its meaning
- If genuinely DIFFERENT meaning but fits existing topic semantically:
  **ADD**: Add new code with a label and source_codes to a Theme and Topic WHERE THE LABEL MEANINGS MATCH
- If no semantically appropriate topic exists based on label meanings:
  **CREATE**: Create new topic under appropriate theme, or create new theme if needed

Second, fix semantic misplacements and structural issues:
- If codes are in wrong semantic locations:
  **RESTRUCTURE**: Move codes to themes and topics where they semantically belong in the following order: Themes -> Topics
- If not Thematically coherent:
  **CREATE**: Create new topic with a label under a SEMANTICALLY APPROPRIATE theme - or create a new Theme, if needed
- If not Internally consistent:
  **CLARIFY**: Split a topic containing semantically diverse codes into distinct topics or themes -> Topics

FORBIDDEN: 
- Placing codes in themes where they don't semantically belong or placing codes in Topics they don't semantically belong
- Providing instructions that don't consider semantic meaning

IMPORTANT: Every action MUST specify the complete path and ALL affected codes with their source_codes.
EXAMPLE OF A PATH: Theme 1, Topic 1.1, Code 1.1.1 with source_codes [ID 1, ID 2] (replace placeholders of source_code ID's with actual ID's)

=========================
OUTPUT FORMAT
=========================
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

=========================
INPUT
=========================
Survey question: "{survey_question}"

Current codebook:
{codebook}

Codes not yet in the codebook:
{unused_codes}
=========================
"""

PHASE4_THEME_REVIEW_PROMPT = """
You are an expert in qualitative research and thematic analysis working in {language}.

=========================
TASK
=========================
You are revising a hierarchical codebook based on feedback from a quality review.

=========================
CODEBOOK STRUCTURE
=========================
- **Themes (Level 1)** – Broad patterns with central organizing concepts
- **Topics (Level 2)** – Specific facets that group codes with shared meaning
- **Codes (Level 3)** – Short descriptive labels for distinct response patterns
  - Each code references the "source_codes" of the original data (the initial codes)

=========================
REVISION INSTRUCTIONS
=========================
You must stricly follow the instructions specified in the feedback.
Instructions include:
- **RESTRUCTURE**: Move codes to where their LABEL meaning fits in the following order: Theme -> Topic
- **CREATE**: Add new themes/topics when no existing LABELS match the semantic meaning of orphaned codes
- **CONSOLIDATE**: Merge ONLY when topic LABELS represent the same concept
- **CLARIFY**: Split topics whose LABEL doesn't accurately represent the diverse codes within
- **ADD**: Add codes only to topics whose LABEL is semantically compatible

=========================
RULES
=========================
- Return a complete codebook with ALL codes in semantically appropriate places
- Maintain numbering (themes: 1,2,3... topics: X.1,X.2... codes: X.Y.1,X.Y.2...)
- PRESERVE all source_codes arrays from the input
- Output all labels and descriptions in **{language}**

=========================
OUTPUT FORMAT
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
              "description": "What specific response pattern this captures",
              "source_codes": [1, 5, 12]  // MUST preserve from original
            }}
          ]
        }}
      ]
    }}
  ]
}}

=========================
INPUT
=========================
Survey question: "{survey_question}"

Initial codes for reference:
{cluster_summaries}

Current codebook to improve:
{current_codebook}

Feedback for revision:
- Summary: 
{summary}

- Issues identified:
{issues}

- Required actions:
{actions}
=========================
"""

PHASE5_LABEL_REFINEMENT_PROMPT = """
You are an expert in qualitative coding and codebook design working in {language}.

=========================
TASK
=========================
You are refining labels and descriptions in a hierarchical codebook to ensure they are clear, precise, and conceptually meaningful.

=========================
CODEBOOK STRUCTURE
=========================
- **Themes (Level 1)** – Broad patterns with central organizing concepts
- **Topics (Level 2)** – Specific facets that group codes with shared meaning
- **Codes (Level 3)** – Short descriptive labels for distinct response patterns

=========================
LABEL PRINCIPLES
=========================
- Maximum 4 words
- Capture the underlying concept, not surface features
- Use precise, meaningful terms
- Avoid compound labels (X and Y and Z)
- Express a single coherent idea

=========================
DESCRIPTION PRINCIPLES
=========================
- Maximum 20 words
- Add insight beyond what's in the label
- Explain the significance or meaning
- Use clear, active language
- Focus on why this grouping matters

=========================
OUTPUT FORMAT
=========================
Return ONLY entries that need improvement:
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

Output all refined labels and descriptions in **{language}**.

=========================
INPUT
=========================
Survey question: "{survey_question}"

Current codebook:
{codebook}
=========================
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





