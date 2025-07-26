
SYSTEM_MESSAGE = """
Act as a {language} qualitative data analyst specializing in thematic analysis.
You specialize in creating codebooks.
A codebook in this setting is a collection of labels and definitions for those labels that can be used to describe pieces of data collected in a survey with open-ended questions. 
"""

CODEBOOK_ANALYSIS_PROMPT = """
{system_message}
This time we will focus on written responses to the following survey question: "{survey_question}".

Given these codes from the codebook:
<existing_codebook>
{code_text}
</existing_codebook>

Analyze these codes to help future matching decisions:
1. What specific aspects of the survey question do these codes address?
2. What potential gaps exist - what types of responses might NOT fit these codes?

Output a concise analysis in {language} following this structure:
"Coverage: These codes collectively address [aspects of the survey question].

Gaps: Responses about [gap 1], [gap 2], and [gap 3] might not fit existing codes."

IMPORTANT: Return ONLY the analysis text following this exact format, no JSON or additional explanation.
"""

RESPONSE_SUMMARY_PROMPT = """
{system_message}

Analyze this cluster of semantically related ideas expressed in response to this survey question: "{survey_question}"

<clustered_ideas>
{cluster_text}
</clustered_ideas>

Extract the cluster's pattern to enable code matching:
1. **Core theme**: What is the central concept? Be specific.
2. **Abstraction level**: Is this cluster about a specific instance or a general pattern?
3. **Key components**: What are the 2-3 essential elements that define this cluster?
4. **Distinguishing features**: What makes this cluster different from other possible themes?

Output a concise analysis in {language} following this structure:
"[cluster description] at a [specific/general] level. The key components are [element 1], [element 2], and [element 3]. What distinguishes this cluster is [unique aspect]."

IMPORTANT: Return ONLY the analysis text, no JSON formatting or additional explanation.
"""

MATCH_AND_RECOMMEND_PROMPT = """
{system_message}
This time we will focus on written responses to the following survey question: "{survey_question}".
You are making a codebook recommendation for a cluster of semantically similar survey responses.
Specifically, you need to recommend whether or not a new code needs to be created.

INPUT DATA:
<existing_codes>
{existing_codes}
Notes: {codebook_analysis}
</existing_codes>
Note: These are the 5 codes nearest to this cluster's centroid embedding.

<clustered_ideas>
{clustered_ideas}
Notes: {summaries}
</clustered_ideas>
Note: These responses were grouped by HDBSCAN based on semantic similarity.

EVALUATION PROCESS:
1. Compare the cluster's core theme against each existing code
2. Assess coverage of themes in the clustered ideas by the existing codes
3. Decide that creating a new code is appropriate when current codes don't cover the clustered ideas enough
4. Always favor parsimony: use existing when in doubt 

CREATION CRITERIA:   
1. **Conceptual unity**: Does the new code represent ONE clear concept?
2. **Mutual exclusivity**: Would a coder be confused about when to use this vs other codes?
3. **Appropriate scope**: Is this trying to cover too much ground?
4. **Abstraction consistency**: Same level as existing codes?

Output ONE recommendation as valid JSON:
{{
  "cluster_core_theme": "identify from cluster analysis notes",
  "best_matching_codes": ["code1", "code2"],
  "coverage_assessment": {{
    "percentage": 0-100,
    "rationale": "explain what aspects are/aren't covered"
  }},
  "decision": "use_existing|modify_existing|create_new",
  "action_details": {{
    "codes_to_use": ["list if use_existing"] or null,
    "code_to_modify": "name if modify_existing" or null,
    "modification_suggestion": "how to broaden if modify_existing" or null,
    "new_code_name": "name if create_new" or null,
    "new_code_definition": "definition if create_new" or null
  }},
  "justification": "explain why this is the most parsimonious choice"
}}

IMPORTANT:
- Return ONLY the JSON object in {language}
- Fill only relevant fields in action_details based on your decision
- One cluster = one recommendation
"""

VALIDATION_PROMPT = """
{system_message}
This time we will focus on written responses to the following survey question: "{survey_question}".

You are reviewing a code recommendation for clustered ideas extracted from survey responses.

This is the extracted ideas: 
<clustered_ideas>
{clustered_ideas}
</clustered_ideas>
Note: These are the original survey responses that prompted this recommendation.

This is the recommendation:
<recommendation>
{step3_recommendation}
</recommendation>
Note: This is the complete recommendation from the matching analysis.

These are existing codes in the code book:
<existing_codebook>
{existing_codes}
</existing_codebook>
Note: These are the 5 codes most similar to the recommended code definition.

EVALUATION CRITERIA:
1. **Parsimony**: Were existing code options properly exhausted?
2. **Non-redundancy**: No overlap with existing codes?
3. **Justification alignment**: Does the recommendation match its reasoning?

Output a validation assessment in {language}:
{{
  "evaluation": {{
    "parsimony_reasoning": "assessment of whether existing options were exhausted",
    "redundancy_reasoning": "assessment of overlap with existing codes",
    "justification_reasoning": "assessment of decision alignment with reasoning"
  }},
  "decision": "APPROVE/REVISE/REJECT",
  "decision_rationale": "explanation for the overall decision",
  "validated_code": {{
    "code": "final code name (approved/revised) or null if rejected",
    "definition": "final definition (approved/revised) or null if rejected"
  }}
}}

IMPORTANT: 
- Return ONLY the JSON object in {language}
- APPROVE only if recommendation represents ONE clear, focused concept
- REVISE if concept needs refinement to improve focus or clarity (provide revised code in validated_code)
- REJECT if recommendation covers multiple unrelated concepts or creates confusion
- Always populate validated_code for APPROVE/REVISE decisions
- Ensure codes represent single, mutually exclusive themes
"""

