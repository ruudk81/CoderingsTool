
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

PHASE2_EXTRACT_ATOMIC_CONCEPTS_PROMPT = """
You are an expert in thematic analysis working in {language}.
Your task is to identify ALL atomic concepts present across descriptive codes derived from survey responses.

Survey question:
<survey_question>
{survey_question}
</survey_question>

Descriptive codes of response segments in sample:
<Descriptive codes>
{codes}
</Descriptive codes>

Instructions:
1. Review all descriptive codes carefully
2. Identify EVERY ATOMIC CONCEPT - the irreducible, single ideas that appear across responses
3. Focus on WHAT respondents are talking about (not WHY)
4. Be COMPLETELY EXHAUSTIVE - capture every meaningful concept, even if it appears only once
5. Do NOT group or merge similar concepts - keep them separate
6. Include specific subconcepts (e.g., Don’t just say “technology” if respondents specifically mention “smartphone”, “WiFi”, or “apps”.
)

An atomic concept is:
- A single, indivisible idea (e.g., "battery life”, “screen size”, “weight”, “camera quality”)
- Cannot be meaningfully broken down further
- Clear and specific

IMPORTANT: 
- If respondents mention specific aspects (like “ticket price”, “seat comfort”, “departure time”, “onboard WiFi”), list these as separate concepts
- Don't combine concepts like "packaging" and "sustainable packaging" - keep them separate
- Include concepts even if they appear in only one or two codes

Begin with analytical notes:
<analytical_notepad>
Work through your analysis here:
- List EVERY distinct concept you find, no matter how specific
- Note which descriptive codes contain each concept
- Do not summarize or group - be comprehensive
[Your analysis]
</analytical_notepad>

Output JSON:
{{
  "analytical_notes": "Your complete working notes from above",
  "atomic_concepts": [
    {{
      "concept": "Concept name",
      "description": "What this concept represents", 
      "evidence": ["0", "2", "3", "15", "27"]  // List EVERY source ID where this concept appears - do NOT provide just examples
    }}
  ]
}}

CRITICAL: For the "evidence" field, you MUST list ALL source IDs where each concept appears, not just a few examples. Go through each descriptive code systematically and include every single occurrence.

Remember: 
- Keep concepts truly atomic
- List EVERY concept separately (e.g., “loading time”, “navigation menu”, “font size”, “error messages” as separate concepts if they appear)
- Use respondents' exact frame of reference
- When in doubt, include it as a separate concept
- The evidence list must be COMPLETE - every source ID where the concept appears

Return output in {language}.
"""

PHASE2_5_EVIDENCE_SCORING_PROMPT = """
You are an expert in qualitative analysis working in {language}.
Your task is to evaluate confidence scores for clusters that appear in concept evidence.

Survey question: {survey_question}

Atomic concepts with evidence:
{atomic_concepts}

Descriptive codes to evaluate:
{descriptive_codes}

INSTRUCTIONS:
Evaluate ONLY the cluster-concept pairs where the cluster ID appears in the concept's evidence list.

CONFIDENCE SCORING:
- 0.0-0.3: No match or very weak connection
- 0.4-0.6: Partial match or some elements relate  
- 0.7-0.8: Good match with most elements aligning
- 0.9-1.0: Excellent match, clearly belongs to this concept

Output JSON format:
{{
  "analytical_notes": "Your analysis process",
  "confidence_scores": [
    {{
      "cluster_id": 0,
      "concept": "Concept name",
      "confidence": 0.85,
      "reasoning": "Brief explanation"
    }}
  ]
}}

Return exactly {expected_scores} confidence scores in {language}.
"""

PHASE2_5_UNASSIGNED_SCORING_PROMPT = """
You are an expert in qualitative analysis working in {language}.
Your task is to evaluate unassigned clusters against ALL atomic concepts.

Survey question: {survey_question}

Atomic concepts:
{atomic_concepts}

Unassigned descriptive codes (not in any evidence):
{unassigned_codes}

INSTRUCTIONS:
Evaluate EVERY unassigned cluster against EVERY atomic concept.

CONFIDENCE SCORING:
- 0.0-0.3: No match or very weak connection
- 0.4-0.6: Partial match or some elements relate  
- 0.7-0.8: Good match with most elements aligning
- 0.9-1.0: Excellent match, clearly belongs to this concept

CRITICAL: You must evaluate ALL {num_unassigned} clusters against ALL {num_concepts} concepts.
Expected output: {expected_scores} confidence scores.

Output JSON format:
{{
  "analytical_notes": "Your analysis process",
  "confidence_scores": [
    {{
      "cluster_id": 10,
      "concept": "Concept name",
      "confidence": 0.25,
      "reasoning": "Brief explanation"
    }}
  ]
}}

Return exactly {expected_scores} confidence scores in {language}.
"""

PHASE2_5_CONCEPT_FOCUSED_SCORING_PROMPT = """
You are an expert in qualitative analysis working in {language}.
Your task is to evaluate ALL clusters against a SINGLE atomic concept for focused analysis.

Survey question: {survey_question}

TARGET CONCEPT:
Name: {concept_name}
Description: {concept_description}

ALL DESCRIPTIVE CODES TO EVALUATE:
{all_cluster_codes}

INSTRUCTIONS:
Evaluate EVERY cluster against ONLY the target concept "{concept_name}".

CONFIDENCE SCORING:
- 0.0-0.3: No match or very weak connection to {concept_name}
- 0.4-0.6: Partial match or some elements relate to {concept_name}
- 0.7-0.8: Good match with most elements aligning with {concept_name}
- 0.9-1.0: Excellent match, clearly belongs to {concept_name}

CRITICAL: You must provide exactly {expected_scores} confidence scores - one for each cluster listed above.

Output JSON format:
{{
  "analytical_notes": "Your analysis process for evaluating all clusters against {concept_name}",
  "confidence_scores": [
    {{
      "cluster_id": 0,
      "concept": "{concept_name}",
      "confidence": 0.85,
      "reasoning": "Brief explanation for this cluster's match to {concept_name}"
    }}
  ]
}}

Return exactly {expected_scores} confidence scores for concept "{concept_name}" in {language}.
"""

PHASE3_GROUP_CONCEPTS_INTO_THEMES_PROMPT = """
You are an expert in qualitative analysis working in {language}.
Your task is to group atomic concepts into meaningful themes.

Survey question:
<survey_question>
{survey_question}
</survey_question>

Atomic concepts identified:
<atomic_concepts>
{atomic_concepts}
</atomic_concepts>

Instructions:
1. Group atomic concepts that share a common theme, dimension, or aspect
2. Create clear, meaningful theme labels
3. Ensure each theme has a coherent focus
4. CRITICAL: Every single atomic concept from the input list must appear in your output

Guidelines for themes:
- Should represent a broad area of response
- Aim for 3-7 themes typically, but let the data guide you. Use more themes if the concepts are truly distinct.
- Each theme should contain related atomic concepts
- Theme names should be clear and descriptive

IMPORTANT REQUIREMENTS:
- You must account for ALL {total_concepts} atomic concepts provided
- If a concept doesn't fit well with others, put it in "unassigned_concepts"
- Do NOT merge, combine, or drop any concepts
- Use the exact concept names from the input list

Output JSON:
{{
  "themes": [
    {{
      "theme_id": "1",
      "label": "Theme Name",
      "description": "What this theme encompasses",
      "atomic_concepts": [
        {{
          "concept_id": "1.1",
          "label": "EXACT_CONCEPT_NAME_FROM_INPUT",
          "description": "What this concept covers"
        }}
      ]
    }}
  ],
  "unassigned_concepts": ["EXACT_CONCEPT_NAME_IF_UNASSIGNED"]
}}

VERIFY: Your output must contain exactly {total_concepts} concepts total (in themes + unassigned).
Return output in {language}.
"""

PHASE4_LABEL_REFINEMENT_PROMPT = """
You are an expert in creating clear, professional codebooks working in {language}.
Your task is to refine all labels and descriptions for maximum clarity and usability.

Survey question:
<survey_question>
{survey_question}
</survey_question>

Current codebook with cluster assignments:
<codebook_with_assignments>
{codebook_with_cluster_counts}
</codebook_with_assignments>

CRITICAL REQUIREMENT:
**PRESERVE STABLE IDs**: Each concept has a "Stable ID" (like concept_1, concept_2, concept_other). 
You MUST include the "stable_id" field in EVERY atomic concept in your response.

Refinement goals:
1. **Theme Labels**: Clear, broad areas (2-4 words)
2. **Concept Labels**: Precise, specific ideas (2-4 words)
3. **Descriptions**: Clear, distinguishable, explain what's included
4. **Examples**: Add 1-2 representative quotes per concept
5. **Consistency**: Parallel structure and professional tone

Guidelines:
- Use clear, non-technical language
- Ensure labels are distinct from each other
- Descriptions should help coders understand boundaries

DO NOT:
- Change the structure or assignments
- Merge or split any items
- Add new themes or concepts
- Move concepts between themes
- Change or omit the stable_id values

Output JSON:
{{
  "refined_codebook": {{
    "themes": [
      {{
        "theme_id": "1",
        "label": "Refined Theme Label",
        "description": "Clear description of what this theme encompasses",
        "atomic_concepts": [
          {{
            "concept_id": "1.1",
            "label": "Refined Concept Label",
            "description": "Precise description of this atomic concept",
            "stable_id": "concept_1"
          }},
          {{
            "concept_id": "1.2",
            "label": "Another Refined Concept",
            "description": "Description of this concept",
            "stable_id": "concept_3"
          }}
        ]
      }}
    ],
    }}
  }}
}}

Return output in {language}.
"""
