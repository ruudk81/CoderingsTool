
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
1. What is the scope and abstraction level of each code?
2. What specific aspects of the survey question do these codes address?
3. What potential gaps exist - what types of responses might NOT fit these codes?

Output a structured analysis in {language}:
"Code Analysis:
[For each code: - CodeName: what this code covers at [specific/general] abstraction level]

Coverage: These codes collectively address [aspects of the survey question].

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

Output a concise analysis in {language}:
"This cluster focuses on [core theme] at a [specific/general] level. The key components are [element 1], [element 2], and [element 3]. What distinguishes this cluster is [unique aspect]."

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

Review these code recommendations against strict criteria:

<recommendations>
{recommendations}
</recommendations>

Evaluation criteria:
1. Parsimony: Did we exhaust existing code options?
2. Non-redundancy: No overlap with existing codes?
3. Abstraction consistency: Same level as existing codes?

Example of redundancy to avoid:
{redundancy_example}

Output ONLY valid JSON with no additional text:
{{
  "evaluations": [
    {{
      "code": "code name",
      "parsimony_score": 0,
      "parsimony_reasoning": "...",
      "redundancy_score": 0,
      "redundancy_reasoning": "...",
      "abstraction_score": 0,
      "abstraction_reasoning": "...",
      "decision": "KEEP/REJECT"
    }}
  ],
  "validated_codes": [
    {{
      "code": "code name",
      "definition": "definition"
    }}
  ]
}}

IMPORTANT: Return ONLY the JSON object. Include in validated_codes only those scoring 8+ on all criteria.
Remember: the output needs to be in {language}
"""

