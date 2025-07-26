
SYSTEM_MESSAGE = """
Act as a {language} qualitative data analyst specializing in thematic analysis.
You specialize in creating codebooks.
A codebook in this setting is a collection of labels and definitions for those labels that can be used to describe pieces of data collected in a survey with open-ended questions. 
"""

CODEBOOK_ANALYSIS_PROMPT = """
{system_message}
This time we will focus on writen responses to the following survey question: "{survey_question}".

Given these codes from the codebook:
<existing_codebook>
{code_text}
</existing_codebook>

Analyze the thematic landscape of these codes:
1. What main thematic areas do these codes cover in light of the survey question?
2. How do these codes relate to each other?    

Return your analysis as valid JSON:
{{
   "thematic_coverage": "description of the main thematic areas these codes address",
   "code_relationships": "how these codes connect and relate to each other"
}}

Output in {language}. Return ONLY the JSON object.
"""

RESPONSE_SUMMARY_PROMPT = """
{system_message}

Analyze this cluster of semantically related ideas expressed in response to this survey question: "{survey_question}"

These ideas were grouped together because they share conceptual similarity. Your task is to identify the unified response pattern this cluster represents.

<clustered_ideas>
{cluster_text}
</clustered_ideas>

Extract the cluster's coherent pattern:
1. **Core theme**: What specific aspect of the approach are these ideas discussing? (e.g., implementation, results, organization, support)
2. **Sentiment pattern**: Do these ideas express predominantly positive satisfaction, negative dissatisfaction, or mixed feelings?
3. **Reasoning focus**: What main reasons or justifications do they provide for their satisfaction/dissatisfaction?
4. **Shared terminology**: What consistent concepts, phrases, or language patterns appear across these ideas?

This cluster analysis will help determine if existing codes adequately capture this response pattern or if new codes are needed.

Output ONLY valid JSON with no additional text:
{{
  "core_theme": "specific aspect being discussed",
  "sentiment_pattern": "positive/negative/mixed",
  "reasoning_focus": "main justification provided", 
  "shared_terminology": ["key term 1", "key term 2", "key term 3"],
  "cluster_coherence": "explanation of what unites these ideas conceptually"
}}

IMPORTANT: Return ONLY the JSON object, no explanations or additional text. All property names must be in double quotes.
Remember: the output needs to be in {language}
"""

MATCH_AND_RECOMMEND_PROMPT = """
{system_message}

You are analyzing a cluster of ideas to determine if existing codes adequately capture the concepts expressed.

<existing_codes>
{existing_codes}
</existing_codes>

<clustered_ideas>
{clustered_ideas}
</clustered_ideas>

<codebook_analysis>
{codebook_analysis}
</codebook_analysis>

<cluster_analysis>
{summaries}
</cluster_analysis>

Your task:
1. Examine the actual clustered ideas and determine if existing codes can adequately describe them
2. If existing codes are insufficient, explain specifically why
3. Only recommend new codes if absolutely necessary

Remember: You have a reputation for parsimony. Consider:
- Can existing codes be combined?
- Can existing codes be slightly broadened?
- Is the new concept truly distinct?

Analyze this cluster as a single thematic unit. Output ONLY valid JSON array with no additional text:
[
  {{
    "cluster_theme": "the core theme identified in cluster analysis",
    "existing_code_matches": ["code1", "code2"],
    "coverage": "full/partial/none",
    "gap_analysis": "what's missing if partial/none",
    "recommendation": "use existing/create new",
    "new_code": "code name or null",
    "new_definition": "definition or null",
    "justification": "justification for recommendation or null"
  }}
]

IMPORTANT: Return ONLY the JSON array, no explanations or additional text. Use null for fields that don't apply.
Remember: the output needs to be in {language}
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

