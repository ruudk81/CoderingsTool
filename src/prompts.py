"""
Centralized prompt management for CoderingsTool
This module contains all prompts used in the pipeline, organized by phase.
Simplified 6-phase workflow for hierarchical labeling.
"""

# =============================================================================
# STEP 2: SPELL CHECKING
# =============================================================================

LLM_SPELLCHECK_PROMPT = """
Your task is to review potential spelling errors identified by a spell checker.
Only correct the word if it's a common spelling error. 
If it's a colloquialism, proper noun, or slang, return the original word.

Context: This is a response to the survey question "{var_label}"

Sentence: {sentence}
Word flagged: "{word}"
Spell checker suggestions: {suggestions}

Return JSON:
{{
    "corrected_word": "corrected word or original if no correction needed",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}
"""

# =============================================================================
# STEP 3: QUALITY FILTERING 
# =============================================================================

CATEGORIZE_RESPONSES_PROMPT = """
You are an expert at categorizing survey responses. 
Your task is to analyze qualitative responses to determine if they are meaningful contributions.

Survey question:
<survey_question>
{survey_question}
</survey_question>

Responses to categorize:
<responses>
{responses}
</responses>

For each response:
1. Determine if it contains meaningful content related to the survey question
2. Categorize any non-meaningful response appropriately
3. Be conservative - when in doubt, classify as meaningful

Categories:
- meaningful: Response provides substantive content
- uncertainty: Only expresses uncertainty ("I don't know", "not sure", etc.)
- nonsense: Random characters, gibberish, or meaningless content
- meta_commentary: Comments about the survey itself rather than answering

Provide results as a JSON array:
[
  {{
    "respondent_id": 12345,
    "category": "meaningful"
  }},
  {{
    "respondent_id": 12346,
    "category": "uncertainty",
    "reasoning": "Response only says 'geen idee' (no idea)"
  }}
]

Focus on accuracy. Do not include reasoning for meaningful responses.
"""

# =============================================================================
# STEP 4: SEGMENTATION AND DESCRIPTION
# =============================================================================

SEGMENT_AND_DESCRIBE_INSTRUCTION = [
    {
        "role": "system",
        "content": "You are an AI trained to perform qualitative analysis on open-ended survey responses."
    },
    {
        "role": "user",
        "content": """
## Task: Qualitative Analysis of Survey Response

Survey question: "{var_label}"

Response to analyze:
"{response}"

### Your analysis should:
1. **Segment the response** if it contains multiple distinct ideas (most responses have 1-2 segments)
2. **Create a descriptive label** (3-6 words) that captures the essence of each segment
3. **Write a clear description** (10-20 words) expanding on what the respondent is expressing

### Format each segment as:
{{
  "segment_label": "Brief descriptive label",
  "segment_description": "Clear description of what respondent is expressing",
  "segment_response": "The exact text from the response that forms this segment"
}}

### Important guidelines:
- Keep segments focused on single ideas or closely related points
- Use the respondent's own language and concepts where possible
- Be descriptive, not interpretive
- Ensure descriptions are clear and specific

Provide your analysis as a JSON array of segments.
"""
    }
]

# =============================================================================
# STEP 6: HIERARCHICAL LABELING - 6 PHASES
# =============================================================================

# -----------------------------------------------------------------------------
# PHASE 1: DESCRIPTIVE CODING
# -----------------------------------------------------------------------------

PHASE1_DESCRIPTIVE_CODING_PROMPT = """
You are an expert in descriptive coding working in {language}.
Your task is to perform descriptive coding on segments from open-ended survey responses. 
You will provide one description that captures what respondents are expressing in these segments.

Here are the coding principles you must follow:
1. Stay close to the data: Use respondents' own concepts
2. Be descriptive: Capture what is said, not why
3. Be specific: Focus on the distinct pattern in these segments

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

To complete this task:
1. Carefully read through all the representative segments.
2. Identify the common theme or pattern expressed across these segments.
3. Create a concise label that accurately captures this theme or pattern.
4. Ensure your label adheres to the coding principles mentioned above.
5. Double-check that your label is specific to this cluster and not too general.

Your output should be in the following format:
{{
  "label": "Your Descriptive Label Here"
}}

Remember to provide the label in {language}
"""

# -----------------------------------------------------------------------------
# PHASE 2: LABEL MERGER
# -----------------------------------------------------------------------------

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
- Both clusters are semantically identical or meaningfully equivalent

Important guidelines:
- Be conservative - when in doubt, keep clusters separate.
- Consider the context of the survey question when evaluating semantic similarity.
- Pay attention to nuances in meaning that might be important to preserve.


For merged labels :
1. Choose the most representative label as the new merged label
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

# -----------------------------------------------------------------------------
# PHASE 3: EXTRACT THEMES (uses gpt-4o for better quality)
# -----------------------------------------------------------------------------

PHASE3_EXTRACT_THEMES_PROMPT = """
You are an expert in thematic analysis working in {language}.
Your task is to extract main themes from descriptive codes developed from survey responses.

Survey question:
<survey_question>
{survey_question}
</survey_question>

Descriptive codes from Phase 1 (after merging):
<codes>
{codes}
</codes>

Instructions:
1. Analyze all the descriptive codes carefully
2. Identify the main overarching themes that emerge from these codes
3. Themes should be broad enough to group related codes but specific enough to be meaningful
4. Aim for 3-8 main themes depending on the diversity of responses
5. Each theme should represent a distinct aspect of how respondents answered the survey question

Provide a list of main themes as a JSON response:
{{
  "themes": [
    "Theme 1 name",
    "Theme 2 name", 
    "Theme 3 name"
  ]
}}

Remember: Use {language} for all theme names.
"""

# -----------------------------------------------------------------------------
# PHASE 4: CREATE CODEBOOK
# -----------------------------------------------------------------------------

PHASE4_CREATE_CODEBOOK_PROMPT = """
You are an expert in creating hierarchical codebooks for qualitative analysis working in {language}.
Your task is to organize descriptive codes into a structured codebook based on the themes identified.

Survey question:
<survey_question>
{survey_question}
</survey_question>

Main themes identified:
<themes>
{themes}
</themes>

Descriptive codes to organize:
<codes>
{codes}
</codes>

Create a hierarchical codebook with this EXACT structure:
1. THEMES (top level) - The main themes provided
2. TOPICS (sub-themes) - Group related codes under themes
3. CODES (bottom level) - The specific descriptive codes

Rules:
1. Use ALL themes provided
2. Create 2-4 topics under each theme
3. Assign EVERY code to the most appropriate topic
4. Each code must have source_codes listing its original cluster IDs
5. Maintain the original meaning of codes - do not modify them

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

Important: Every code from the input must appear in the codebook with its source_codes preserved.
"""

# -----------------------------------------------------------------------------
# PHASE 5: LABEL REFINEMENT
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# PHASE 6: ASSIGNMENT
# -----------------------------------------------------------------------------

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

# =============================================================================
# REMOVED/DEPRECATED PROMPTS
# =============================================================================
# The following prompts have been removed in the simplified 6-phase workflow:
# - INITIAL_THEMES_PROMPT (was Phase 1c)
# - PHASE2_GROUP_TOPICS_PROMPT (middle step of old Phase 2)
# - PHASE3_THEME_JUDGER_PROMPT (old Phase 3)
# - PHASE4_THEME_REVIEW_PROMPT (old Phase 4)