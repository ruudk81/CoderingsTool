"""
Experimental prompts for cluster labeling.

This file contains experimental prompt variants that can be modified
freely without affecting the main step_5_clusterer experiment.

Key experiments:
- Including low-probability members in prompts
- Different prompt structures (verbose vs minimal)
- Confidence-aware responses
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# =============================================================================
# RESPONSE MODELS
# =============================================================================

class ClusterDescriptionExp(BaseModel):
    """Extended cluster description with confidence rating."""
    theme: str = Field(
        ...,
        description="Short noun phrase (1-4 words) capturing the cluster's essence"
    )
    description: str = Field(
        ...,
        description="1-2 sentence explanation of what unifies this cluster"
    )
    key_concepts: List[str] = Field(
        ...,
        description="3-5 key concepts found in this cluster"
    )
    confidence: str = Field(
        default="medium",
        description="high/medium/low - confidence in the label given the evidence"
    )
    notes: Optional[str] = Field(
        default=None,
        description="Optional notes about ambiguity or alternative interpretations"
    )


class ClusterDescriptionMinimal(BaseModel):
    """Minimal response model for quick experiments."""
    theme: str = Field(..., description="2-4 word theme")
    description: str = Field(..., description="One sentence description")
    key_concepts: List[str] = Field(..., description="3-5 key concepts")


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

# V2: Includes separate sections for high and low probability members
CLUSTER_DESCRIPTION_PROMPT_V2 = """You are analyzing a cluster of survey responses.

<survey_context>
Survey question: "{survey_question}"
Language: {language}
{dataset_context_section}
</survey_context>

{taxonomy_context}

<cluster_evidence>
Cluster ID: {cluster_id}

## High-Confidence Members (probability >= {high_threshold})
These responses are strongly associated with this cluster:
{high_prob_samples}

## Lower-Confidence Members (probability {low_threshold} - {high_threshold})
These responses have weaker association but may still be relevant:
{low_prob_samples}

## Keywords (extracted via {keyword_method})
{keywords}
</cluster_evidence>

<task>
1. Identify the common theme unifying the HIGH-confidence members
2. Consider whether LOW-confidence members suggest a broader interpretation
3. Provide a label that captures the cluster's essence
4. Rate your confidence: high (clear theme), medium (some ambiguity), low (very mixed)
</task>

Provide your response as valid JSON following the schema provided.
"""


# Minimal prompt for A/B testing - stripped down to essentials
CLUSTER_DESCRIPTION_PROMPT_MINIMAL = """Survey question: {survey_question}

Representative responses from this cluster:
{samples}

Keywords: {keywords}

What theme unifies these responses? Provide theme, description, and key_concepts.
"""


# V3: Focus on low-probability members only (for understanding fuzzy boundaries)
CLUSTER_DESCRIPTION_PROMPT_LOW_PROB_FOCUS = """You are analyzing the BOUNDARY members of a cluster.

<context>
Survey question: "{survey_question}"
This cluster's core theme (based on high-confidence members): {existing_theme}
</context>

<boundary_members>
These responses are assigned to the cluster but with lower confidence (probability < {threshold}):
{low_prob_samples}
</boundary_members>

<task>
Analyze these boundary members:
1. Do they actually belong to the theme "{existing_theme}"?
2. Are they misclassified, or do they represent valid edge cases?
3. Would the theme benefit from broadening to include them?
4. What alternative clusters might be better fits?
</task>

<output>
- boundary_analysis: Brief analysis of boundary members
- theme_adjustment: null (keep theme) OR suggested broader theme
- confidence: high/medium/low
</output>
"""


class BoundaryAnalysis(BaseModel):
    """Response model for boundary member analysis."""
    boundary_analysis: str = Field(
        ...,
        description="Analysis of whether boundary members fit the cluster"
    )
    theme_adjustment: Optional[str] = Field(
        default=None,
        description="Suggested theme adjustment if needed, or null to keep current"
    )
    misfit_count: int = Field(
        default=0,
        description="Estimated number of members that don't fit the theme"
    )
    confidence: str = Field(
        default="medium",
        description="Confidence in the analysis"
    )


# =============================================================================
# PROMPT BUILDER FUNCTIONS
# =============================================================================

def format_samples_with_probability(
    samples: List[tuple],
    max_samples: int = 10
) -> str:
    """
    Format samples with their probability scores.

    Args:
        samples: List of (text, probability) tuples
        max_samples: Maximum number to include

    Returns:
        Formatted string with numbered samples and probabilities
    """
    lines = []
    for i, (text, prob) in enumerate(samples[:max_samples], 1):
        lines.append(f"{i}. [{prob:.2f}] {text}")
    return "\n".join(lines)


def format_keywords(
    keywords: List[tuple],
    n_keywords: int = 10
) -> str:
    """
    Format keywords with scores.

    Args:
        keywords: List of (keyword, score) tuples
        n_keywords: Number to include

    Returns:
        Formatted keyword string
    """
    kw_strings = [f"{kw} ({score:.3f})" for kw, score in keywords[:n_keywords]]
    return ", ".join(kw_strings)


def build_prompt_v2(
    survey_question: str,
    cluster_id: int,
    high_prob_samples: List[tuple],
    low_prob_samples: List[tuple],
    keywords: List[tuple],
    high_threshold: float = 0.8,
    low_threshold: float = 0.5,
    keyword_method: str = "mmr",
    language: str = "Dutch",
    dataset_context: str = "",
    taxonomy_context: str = "",
) -> str:
    """
    Build the V2 prompt with high and low probability sections.

    Args:
        survey_question: The survey question text
        cluster_id: Cluster identifier
        high_prob_samples: List of (text, prob) for high-confidence members
        low_prob_samples: List of (text, prob) for lower-confidence members
        keywords: List of (keyword, score) tuples
        high_threshold: Threshold for high probability
        low_threshold: Lower bound for low probability section
        keyword_method: Method used for keyword extraction
        language: Output language
        dataset_context: Optional dataset context section
        taxonomy_context: Optional taxonomy context section

    Returns:
        Formatted prompt string
    """
    return CLUSTER_DESCRIPTION_PROMPT_V2.format(
        survey_question=survey_question,
        language=language,
        dataset_context_section=dataset_context,
        taxonomy_context=taxonomy_context,
        cluster_id=cluster_id,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
        high_prob_samples=format_samples_with_probability(high_prob_samples),
        low_prob_samples=format_samples_with_probability(low_prob_samples) if low_prob_samples else "(none)",
        keyword_method=keyword_method,
        keywords=format_keywords(keywords),
    )
