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
<instruction>The theme label must read as a natural-language answer category to the survey question.</instruction>
{facet_context}
{concept_types_section}
<cluster_evidence>
Cluster ID: {cluster_id}
Number of {sample_type}: {num_ideas}

<representative_{samples_tag}>
These {sample_type} are representative of the cluster:
{ideas_list}
</representative_{samples_tag}>
{keywords_section}{cluster_profile_section}
</cluster_evidence>

<task>
1. Review the representative {sample_type} to identify common meaning.
2. Use the statistical keywords to sharpen what makes this cluster distinct.
3. Identify the common atomic theme expressed directly in the data.
4. Do not introduce concepts not supported by the {sample_type} or keywords.
5. Ensure the theme stays strictly within the primary coding facet{facet_task_guidance}.
6. Ensure the theme reads as a short, noun-phrased natural-language answer to the survey question. Use the essence as the head noun, avoid generic language, clutter and verbs.
</task>

<output_format>
Provide your analysis in {language}:
- theme: Short noun-phrased label{facet_output_constraint} (3-10 words)
- description: 1-2 sentence explanation of what respondents associate with the entity
- key_concepts: 3-5 concrete concepts grounded in data (from keywords or representative samples)
</output_format>"""


class ClusterDescription(BaseModel):
    """LLM-generated cluster description (structured output model)."""
    theme: str = Field(..., description="Short noun-phrased thematic label (3-10 words), reads as answer to survey question")
    description: str = Field(..., description="1-2 sentence explanation of what respondents associate with the entity")
    key_concepts: List[str] = Field(..., description="3-5 concrete concepts grounded in data (from keywords or samples)")
