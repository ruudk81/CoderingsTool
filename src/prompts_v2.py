
SYSTEM_MESSAGE = """
Act as if you are the world's best language {language} qualitative data analyst with expertise in generating qualitative codebooks for thematic analysis. 
You specialize in creating parsimonious codebooks with non-overlapping and non-redundant codes."""

# PROMPT 1: Codebook Analysis  
CODEBOOK_ANALYSIS_PROMPT = """
{system_message}

Analyze this existing codebook for responses to the following survey question: "{survey_question}":

<existing_codebook>
{code_text}
</existing_codebook>

Provide a structured analysis:
1. The codes: [list of codes]
2. Description of overall idea expressed: [short description of the main idea captured by the codes in light of the survey question]
2. Main thematic categories: [list major themes]

Output ONLY valid JSON with no additional text:
{{
  "codes": ["code1", "code2"],
  "idea": "description here",
  "themes": ["theme1", "theme2"]
}}

IMPORTANT: Return ONLY the JSON object, no explanations or additional text.
Remember: the output needs to be in {language}
"""

# PROMPT 2: Response Summarization  
RESPONSE_SUMMARY_PROMPT = """
{system_message}

Summarize the responses presented below to this survey question: "{survey_question}"
You need to focus on main themes.

<responses>
{cluster_text}
</responses>

For each response, extract:
1. Primary theme/concept
2. Emotional tone
3. Key phrases or terminology
4. Unique aspects

Output ONLY valid JSON array with no additional text:
[
  {{"theme": "...", "tone": "...", "key_phrases": ["...", "..."], "unique": "..."}},
  {{"theme": "...", "tone": "...", "key_phrases": ["...", "..."], "unique": "..."}}
]

IMPORTANT: Return ONLY the JSON array, no explanations or additional text. All property names must be in double quotes.
Remember: the output needs to be in {language}
"""

# PROMPT 3: Integrated Matching and Recommendation
MATCH_AND_RECOMMEND_PROMPT = """
{system_message}

Given this codebook analysis:
<codebook_analysis>
{codebook_analysis}
</codebook_analysis>

And these response summaries:
<summaries>
{summaries}
</summaries>

Your task:
1. For each theme in the summaries, identify if existing codes can describe it
2. If existing codes are insufficient, explain specifically why
3. Only recommend new codes if absolutely necessary

Remember: You have a reputation for parsimony. Consider:
- Can existing codes be combined?
- Can existing codes be slightly broadened?
- Is the new concept truly distinct?

Output ONLY valid JSON array with no additional text:
[
  {{
    "theme": "theme name",
    "existing_code_matches": ["code1", "code2"],
    "coverage": "full/partial/none",
    "gap_analysis": "what's missing if partial/none",
    "recommendation": "use existing/create new",
    "new_code": "code name or null",
    "new_definition": "definition or null",
    "justification": "justification or null"
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

