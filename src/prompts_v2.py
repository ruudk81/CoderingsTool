
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

Output in JSON format:
{{
  "codes" : ["code1", "code2"],
  "idea": [descrption here],
  "themes": ["theme1", "theme2"]
}}

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

Output format:
<summaries>
Response 1: {{theme: "...", tone: "...", key_phrases: [...], unique: "..."}}
Response 2: ...
</summaries>

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

Output format:
<analysis>
Theme: [theme]
Existing code matches: [list codes that apply]
Coverage: [full/partial/none]
Gap analysis: [what's missing if partial/none]
Recommendation: [use existing/create new]
If new code needed:
  - Code: "..."
  - Definition: "..."
  - Justification: "..."
</analysis>

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

For each proposed new code:
- Score parsimony (0-10): [score] [reasoning]
- Score non-redundancy (0-10): [score] [reasoning]  
- Score abstraction (0-10): [score] [reasoning]
- Final decision: KEEP/REJECT

Final output:
<validated_codes>
[Only codes scoring 8+ on all criteria]
</validated_codes>

Remember: the output needs to be in {language}
"""

