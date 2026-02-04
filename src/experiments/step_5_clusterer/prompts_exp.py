"""
Cluster label generation prompts - EXPERIMENTAL VERSION

This is an isolated copy for experimentation in step_5_clusterer.
Changes here do NOT affect the production pipeline.

Original: src/prompts.py (CLUSTER_DESCRIPTION_PROMPT, ClusterDescription)
"""

from typing import List
from pydantic import BaseModel, Field


# =============================================================================
# STEP 5: CLUSTER LABEL GENERATION
# =============================================================================

CLUSTER_DESCRIPTION_PROMPT = """You are a qualitative researcher labeling survey-response clusters.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

<instruction>
The theme label must read like ONE possible checkbox answer to the survey question.
</instruction>

{taxonomy_context}

<theory_of_coding>
A valid theme must represent the Central Organizing Concept (COC):

The COC is the SINGLE underlying property or characteristic that best explains why these responses cluster together.
It is NOT a summary, list, span, or combination of ideas.

If multiple related ideas appear in the cluster, abstract to ONE higher-order property that unifies them.
</theory_of_coding>

<label_constraints>
The theme MUST NOT:
- contain "en/and/und/et",
- contain slashes (bijv. duurzaam/groen),
- stack multiple adjectives,
- list multiple attributes.
The label must express ONE core property only.
</label_constraints>

<cluster_evidence>
Cluster ID: {cluster_id}
Number of {sample_type}: {num_ideas}

<representative_{samples_tag}>
These {sample_type} are representative of the cluster:
{ideas_list}
</representative_{samples_tag}>
{keywords_section}
{cluster_profile_section}
</cluster_evidence>

<task>
1. Review the representative {sample_type} to identify common meaning.
2. Use the statistical keywords to sharpen what makes this cluster distinct.
3. Identify the ONE-WORD core essence (the pure COC).
4. Convert that essence into a short, natural noun phrase answer category.
5. Do not introduce concepts not supported by the data.
{taxonomy_task_constraint}</task>
{taxonomy_alignment_section}
<output_format>
Provide your analysis in {language}:
- theme: Short noun phrase (1–4 words) expressing ONE core property
- description: 1–2 sentences explaining what respondents associate with the entity
- key_concepts: 3–5 concrete concepts grounded in data
</output_format>
"""

class ClusterDescription(BaseModel):
    """LLM-generated cluster description (structured output model)."""
    theme: str = Field(..., description="Short noun-phrased thematic label (3-10 words), reads as answer to survey question")
    description: str = Field(..., description="1-2 sentence explanation of what respondents associate with the entity")
    key_concepts: List[str] = Field(..., description="3-5 concrete concepts grounded in data (from keywords or samples)")
