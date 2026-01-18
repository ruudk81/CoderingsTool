"""
Experimental prompts for cluster_analysis.py

This file contains prompt templates used in experimental cluster analysis.
These are separate from production prompts in src/prompts.py.
"""

# Cluster description prompt (adapted from Step 6 CLUSTER_SUMMARY_PROMPT)
CLUSTER_DESCRIPTION_PROMPT = """You are a qualitative researcher applying Braun & Clarke's (2006) thematic analysis method to analyze survey response clusters.

Remember: themes are not discovered in data but actively constructed through your analytical judgment and reflexivity.

<inputs>
Language: {language}
Cluster ID: {cluster_id}
Research question: "{survey_question}"
Number of response ideas: {num_ideas}

Statistical keywords (cluster-distinguishing terms):
{keywords_section}

Representative response ideas from this cluster:
{ideas_list}
</inputs>

<definitions>
- ATOMIC THEME: One single idea, action, expectation, or sentiment relevant to the research question (no mixing)
- Statistical keywords: Terms that distinguish this cluster from others (use as hints, not constraints)
</definitions>

<guidance>
Atomic means:
• Single idea only (no "and/or" combinations)
• One aspect/domain/category only
• One consistent sentiment/polarity (no mixing positive and negative)
• Concrete and directly actionable

Label constraints (strict):
• ≤ 10 words
• Active/actionable formulation of ONE ATOMIC theme
• If verb is used → one main verb (present tense)
• Never include reasons (no "to", "so that", "because")
• Avoid punctuation: "/", "&", ",", "–", ":" (unless lexicalized)
• Maintain one polarity (either increase/strengthen OR reduce/avoid)

Description constraints:
• 1-2 sentences, clear and informative
• Describe what belongs in this theme, not why it happens
• Grounded in the actual response ideas
• Use observable cues (behaviors, expressions, judgments)
</guidance>

<task>
Analyze this cluster and provide a thematic summary:

1. Read all response ideas carefully
2. Consider statistical keywords as hints about distinguishing terms
3. Identify the common atomic theme that unifies these responses
4. Extract 3-5 key concepts present in this cluster

Focus on:
• What makes this cluster distinct from other clusters?
• What is the underlying sentiment, concern, or suggestion?
• What single atomic theme captures the essence?
</task>

<output_format>
Provide your analysis in the following structured format:
- theme: Short atomic thematic label (≤10 words, following label constraints above)
- description: Clear description (1-2 sentences, following description constraints above)
- key_concepts: List of 3-5 key concepts/themes
</output_format>"""

# Additional experimental prompts can be added here
