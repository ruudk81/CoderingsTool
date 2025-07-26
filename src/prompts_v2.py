
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

DECISION RULES:
1. **USE EXISTING CODE(S)** when:
   - One or more existing codes capture ≥80% of the cluster's core theme
   - The cluster represents a specific instance of an existing broader code
   - Combining 2-3 existing codes fully describes the cluster

2. **MODIFY EXISTING CODE** when:
   - An existing code captures 60-79% of the theme but needs slight broadening
   - The cluster reveals a systematic gap in an existing code's definition
   - Small adjustments would make the code applicable to many similar clusters

3. **CREATE NEW CODE** when:
   - No existing codes capture >60% of the cluster's core theme
   - The cluster represents a fundamentally distinct concept
   - The theme appears frequently enough to warrant its own code (not a one-off)

EVALUATION PROCESS:
1. Compare the cluster's core theme against each existing code
2. Assess coverage percentage (subjective but justified)
3. Consider if modification would be more parsimonious than creation
4. Ensure new codes maintain similar abstraction level as existing codes

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

# PROMPT 4: Final Validation (with criteria)
VALIDATION_PROMPT = """
{system_message}
This time we will focus on written responses to the following survey question: "{survey_question}".

You are reviewing a code recommendation to ensure quality and consistency.

CONTEXT DATA:
<existing_codebook>
{existing_codes}
</existing_codebook>
Note: These are the 5 codes most similar to the recommended code definition.

<clustered_ideas>
{clustered_ideas}
</clustered_ideas>
Note: These are the original survey responses that prompted this recommendation.

<step3_recommendation>
{step3_recommendation}
</step3_recommendation>
Note: This is the complete recommendation from the matching analysis.

EVALUATION CRITERIA:
1. **Parsimony** (0-10): Were existing code options properly exhausted?
   - Did Step 3 correctly assess coverage percentages?
   - Could existing codes be combined/broadened instead?

2. **Non-redundancy** (0-10): No overlap with existing codes?
   - Compare recommended definition against the 5 similar codes
   - Check for semantic overlap or near-duplicate concepts

3. **Abstraction consistency** (0-10): Same level as existing codes?
   - Does the new code match the abstraction level of existing codes?
   - Is it neither too specific nor too broad compared to others?

4. **Justification alignment** (0-10): Does the recommendation match its reasoning?
   - Is the decision (use_existing/modify_existing/create_new) well-justified?
   - Do the coverage assessment and reasoning align?

Output a validation assessment in {language}:
{{
  "evaluation": {{
    "parsimony_score": 0-10,
    "parsimony_reasoning": "assessment of whether existing options were exhausted",
    "redundancy_score": 0-10,
    "redundancy_reasoning": "assessment of overlap with existing codes",
    "abstraction_score": 0-10,
    "abstraction_reasoning": "assessment of abstraction level consistency",
    "justification_score": 0-10,
    "justification_reasoning": "assessment of decision alignment with reasoning"
  }},
  "decision": "APPROVE/REVISE/REJECT",
  "decision_rationale": "explanation for the overall decision",
  "validated_code": {{
    "code": "approved code name or null if rejected",
    "definition": "approved definition or null if rejected"
  }}
}}

IMPORTANT: 
- Return ONLY the JSON object in {language}
- APPROVE only if all scores ≥8
- REVISE if scores 6-7 (provide specific suggestions)
- REJECT if any score <6
"""

