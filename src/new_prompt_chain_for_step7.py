#prompt 1: current RESPONSE_SUMMARY_PROMPT 
CLUSTER_SUMMARY_PROMPT= """
You are a {language} data analyst specializing in thematic analysis of open-ended survey responses.
Your task is to analyze a cluster of responses grouped by HDBSCAN based on embedding similarity, and identify the theme(s) present.

IMPORTANT: Mathematical similarity (via embeddings) does not guarantee conceptual unity. Clusters may legitimately contain:
- One unified theme
- Multiple distinct themes (2 or more)
- No coherent theme (just superficial similarity)

A "theme" is an atomic, shared concept that meaningfully answers the survey question. A valid theme:
- Appears in at least 30% of the cluster's responses
- Represents a distinct, separable idea
- Cannot be split into smaller independent concepts
- Has explanatory power for why these responses matter in light of the survey question

Follow these steps:

1. Review the survey question:
<survey_question>
{survey_question}
</survey_question>

2. Look for patterns and shared meanings across responses in this cluster :
<cluster_text>
{cluster_text}
</cluster_text>

3. Active Theme Construction Process:
   a) **Look for an overarching narrative** that the response ideas might communicate
      - What story are these responses collectively telling?
      - What deeper message connects seemingly different surface expressions?
   
   b) **Realize that themes don't simply "emerge"** — actively construe relationships and meaning
      - Don't just list what you see; interpret what it means
      - Connect dots between responses that might not be obviously related
      - Consider latent meanings, not just manifest content
   
   c) **First, attempt to identify ONE common theme** from the overarching narrative
      - Can you find a single conceptual thread that weaves through most responses?
      - Is there a unifying perspective or concern being expressed in different ways?
      - Push yourself to find unity before accepting fragmentation
   
   d) **Only conclude multiple themes if** they are clearly distinct and independently meaningful
      - Multiple themes should represent fundamentally different ideas
      - Each theme must stand alone and appear in ≥30% of responses
      - Themes should not just be different aspects of the same underlying concept

4. Make your determination after active construction:
   - ONE THEME: You've found an overarching narrative that unifies ≥70% of responses
   - MULTIPLE THEMES: Despite seeking unity, 2+ distinct narratives each appear in ≥30% of responses
   - NO THEME: No narrative emerges that connects ≥30% of responses meaningfully

5. Report your result in {language} using ONLY this JSON structure:

{{
  "cluster_summary": "Brief summary of what this cluster contains",
  "themes": [
    "Theme 1: [concise theme statement]. This is meaningful because [core rationale].",
    "Theme 2: [concise theme statement]. This is meaningful because [core rationale]."
  ]
}}

For NO_THEME cases, return:
{{
  "cluster_summary": "This cluster lacks coherent themes. [Brief explanation]",
  "themes": []
}}

RITICAL OUTPUT RULES:
- Output ONLY valid JSON - no text before or after
- For NO_THEME: set theme_count to 0 and themes array to []
- For ONE_THEME: set theme_count to 1 with one theme object
- For MULTIPLE_THEMES: include all distinct themes in the array
- All text fields must be in {language}
- theme_statement should be 5-15 words, optimized for embedding
- Keep theme_statement focused on the core concept for better embedding/search
"""

#prompt 2: current CODEBOOK_ANALYSIS_PROMP
CANDIDATE_CODE_SELECTION_PROMPT = """
You are a {language} qualitative data analyst specializing in generating qualitative codebooks for thematic analysis. 
Your task is to select appropriate codes for describing a cluster of responses grouped by HDBSCAN based on embedding similarity.

IMPORTANT: Clusters may contain **multiple distinct themes** that require different codes. Your job is to identify ALL relevant codes for ALL themes present.

First, review the survey question that generated the responses:
<survey_question>
{survey_question}
</survey_question>

Now, examine the summary of a cluster of semantically similar survey responses with 1 or more themes:
<cluster_summary>
{cluster_summary}
</cluster_summary>

Finally, examine the existing codes in the codebook:
<existing_codebook>
{code_text}
</existing_codebook>

To select appropriate candidate codes, follow these steps:
1. Review **ALL distinct themes** present in the cluster summary
2. For **EACH theme**, search for matching existing codes that capture its meaning
3. Select codes based on semantic fit (codes should meaningfully capture the theme)
4. Be selective but comprehensive: only include suitable codes, but ensure every theme has candidates if possible

Output your selection as a JSON array. Include ALL relevant codes that could apply to ANY theme in the cluster:

[
  {{
    "code": "exact same name of existing code 1",
    "definition": "exact same definition of existing code 1"
  }},
  {{
    "code": "exact same name of existing code 2",
    "definition": "exact same definition of existing code 2"
  }}
]

IMPORTANT:
- Output ONLY the JSON array - no other text
- You may select NONE (empty array []), ONE, or MULTIPLE codes
- Use EXACT code names and definitions from the existing codebook
- If no existing codes fit any themes, return an empty array: []
- Do NOT create new codes or modify existing ones
- Multi-theme clusters may need codes for each theme
"""

CANDIDATE_CODE_SELECTION_PROMPT = """
You are a {language} qualitative data analyst specializing in generating qualitative codebooks for thematic analysis. 
Your task is to select appropriate codes for a cluster of semantically similar responses to be used in constructing a codebook. 
A codebook in this context is a structured collection of code names and definitions used to label and interpret open-ended survey responses.

IMPORTANT: Clusters may contain **multiple distinct themes** that require different codes. Your job is to identify ALL relevant codes for ALL themes present.

First, review the survey question that generated the responses:
<survey_question>
{survey_question}
</survey_question>

Now, examine a summary the a cluster of semantically similar survey responses with 1 or more hare themes:
<cluster_summary>
{cluster_summary}
</cluster_summary>

Finally, examine the existing codes in the codebook:
<existing_codebook>
{code_text}
</existing_codebook>

To select appropriate candidate codes, follow these steps:
1. Identify **ALL distinct themes** present in the cluster summary (there may be 1 or more)
2. For **EACH theme**, attempt to find matching existing codes
2. Select the codes, if any, that are relevant in capturing the themes
3. Be selective: only select suitable codes for our codebook
5. Be comprehensive: ensure no theme is left uncoded
6. Present your selection in the JSON array format described below

<json_output_format>
[
  {{
    "code": "exact same name of existing code 1",
    "definition": "exact same definition of existing code 1"
  }},
  {{
    "code": "exact same name of existing code 2",
    "definition": "exact same definition of existing code 2"
  }}
]
</json_output_format>

IMPORTANT:
- You may select NONE, ONE, or MULTIPLE candidate codes per theme
- Do not create new codes or modify existing ones
- If a concept has no suitable existing codes, list it in "uncoded_concepts"
- Multi-theme clusters are normal and expected - code each theme appropriately
"""

#prompt 3: current MATCH_AND_RECOMMEND_PROMPT
CODE_GENERATION_PROMPT = """
You are a {language} data analyst specializing in qualitative research and thematic coding. 
Your expertise lies in developing parsimonious, clearly defined, non-redundant codebooks at a consistent level of abstraction.

CRITICAL: Clusters may contain **multiple distinct themes**. Each theme should be evaluated and coded separately. Do not force multiple themes into a single code.

Your task is to analyze a cluster summary and determine the appropriate coding strategy for EACH distinct theme present.

<context>
Survey Question:
{survey_question}

Summary about a cluster of semantically related responses:
{cluster_summary}

Available Codes:
{candidate_codes}
</context>

Follow this process:

Step 1: Identify Distinct Themes
- Carefully examine the cluster summary
- Identify how many distinct themes are present (1, 2, or more)
- Each theme should be atomic and independently meaningful

Step 2: For EACH Theme, Evaluate Coding Options
For each identified theme, determine whether to:
a) Use existing code(s) that adequately capture this theme
b) Modify an existing code to better fit this theme
c) Create a new code for this theme

Decision Guidelines Per Theme:
- Use `use_existing` if current codes fully describe the theme
- Use `modify_existing` when a code captures 60-90% of the theme's meaning
- Use `create_new` if no code captures >60% of the theme

Atomicity Rules (Critical):
- Each code must describe ONE single, indivisible theme
- Multiple themes require multiple separate codes
- Never combine distinct themes into one code using "and", "with", etc.

Output instructions:
Return only valid raw JSON. Structure your response to handle multiple themes:

{{
  "cluster_analysis": {{
    "number_of_themes": "integer (1, 2, or more)",
    "theme_descriptions": [
      "Brief description of theme 1",
      "Brief description of theme 2 (if applicable)"
    ]
  }},
  "coding_decisions": [
    {{
      "theme_number": 1,
      "theme_description": "what this theme is about",
      "decision": "use_existing | modify_existing | create_new",
      "action_details": {{
        "codes_to_use": ["exact code names"],
        "codes_to_modify": "single exact code name or null",
        "modified_code_name": "new name if modifying, else null",
        "modified_code_definition": "1-2 sentence definition if modifying, else null",
        "new_code_name": "name if creating new, else null",
        "new_code_definition": "1-2 sentence definition if creating new, else null"
      }},
      "justification": "why this action is appropriate for this specific theme"
    }},
    {{
      "theme_number": 2,
      "theme_description": "what this second theme is about",
      "decision": "use_existing | modify_existing | create_new",
      "action_details": {{...}},
      "justification": "..."
    }}
  ],
  "overall_justification": "why treating these as separate themes preserves atomicity and improves codebook quality"
}}

IMPORTANT:
- Process EACH distinct theme separately in the coding_decisions array
- Never merge distinct themes into a single code
- All text must be in {language}
- Output ONLY valid JSON, no other text
"""


# VALIDATION_PROMPT with Consistent Theme Terminology
VALIDATION_PROMPT = """
You are a {language} qualitative data analyst specializing in codebook development and evaluation. 

CRITICAL: You are validating recommendations that may involve **multiple distinct themes and multiple codes**. Each theme/code pair must be evaluated separately.

Your task is to assess your colleague's coding recommendations for a cluster that may contain multiple themes.

Step 1: Review the survey question:
<survey_question>
{survey_question}
</survey_question>

Step 2: Examine the candidate codes:
<candidate_codes>
{candidate_codes}
</candidate_codes>

Step 3: Review the cluster summary:
<cluster_summary>
{cluster_summary}
</cluster_summary>

Step 4: Evaluate your colleague's recommendations:
<recommendation>
{step3_recommendation}
</recommendation>

Step 5: Assess EACH coding decision using these criteria:
a) **Theme Separation** – Are distinct themes appropriately identified and separated?
b) **Semantic Fit** – Does each proposed code capture its target theme?
c) **Atomicity** – Is each code truly atomic (single, indivisible theme)?
d) **Parsimony** – Were existing options properly considered?
e) **Non-redundancy** – Do the codes avoid overlap?

Decision Rules for EACH code:
- **APPROVE** – The code is necessary, atomic, and well-justified
- **REVISE** – The theme is valid but the code needs refinement
- **REJECT** – Use an existing code instead
- **MERGE** – Multiple codes should be combined (only if they represent the same theme)
- **SPLIT** – A proposed code tries to cover multiple themes and should be split

Validation Output:
Return your validation as raw JSON:

{{
  "theme_assessment": {{
    "number_of_themes_identified": "integer",
    "theme_separation_valid": "true/false",
    "theme_separation_reasoning": "are the themes truly distinct or should they be merged/split?"
  }},
  "code_validations": [
    {{
      "theme_number": 1,
      "theme_description": "what theme is being coded",
      "original_recommendation": "what was proposed",
      "evaluation": {{
        "semantic_fit": "assessment",
        "atomicity": "assessment",
        "parsimony": "assessment",
        "redundancy": "assessment"
      }},
      "decision": "APPROVE | REVISE | REJECT | MERGE | SPLIT",
      "decision_rationale": "explanation",
      "validated_code": {{
        "code": "final code name",
        "definition": "final definition"
      }}
    }},
    {{
      "theme_number": 2,
      "theme_description": "...",
      "original_recommendation": "...",
      "evaluation": {{...}},
      "decision": "...",
      "decision_rationale": "...",
      "validated_code": {{...}}
    }}
  ],
  "overall_validation": {{
    "all_themes_coded": "true/false",
    "final_code_count": "integer",
    "summary": "brief summary of the validation outcome"
  }}
}}

Strict rules:
- Validate EACH theme/code pair separately
- Ensure atomic codes (one theme per code)
- Don't force unrelated themes into single codes
- Allow multiple codes when multiple distinct themes exist
- Output ONLY valid JSON, no other text
"""
