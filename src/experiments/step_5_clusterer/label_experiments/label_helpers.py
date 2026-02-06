"""
Experimental Label Generator for cluster labeling experiments.

This is a simplified, modifiable version of LabelGenerator from
clusterer_helpers_exp.py, designed for experimenting with:
- Different probability thresholds
- Including low-probability members in prompts
- A/B testing different prompt templates
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
import sys
from pathlib import Path

# Ensure imports work
project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

from utils.llm import llm_create_sync, create_client
import models

# Local imports
try:
    from .config_label_exp import LabelExperimentConfig
    from .prompts_label_exp import (
        ClusterDescriptionExp,
        ClusterDescriptionMinimal,
        BoundaryAnalysis,
        CLUSTER_DESCRIPTION_PROMPT_V2,
        CLUSTER_DESCRIPTION_PROMPT_MINIMAL,
        CLUSTER_DESCRIPTION_PROMPT_LOW_PROB_FOCUS,
        format_samples_with_probability,
        format_keywords,
        build_prompt_v2,
    )
except ImportError:
    from config_label_exp import LabelExperimentConfig
    from prompts_label_exp import (
        ClusterDescriptionExp,
        ClusterDescriptionMinimal,
        BoundaryAnalysis,
        CLUSTER_DESCRIPTION_PROMPT_V2,
        CLUSTER_DESCRIPTION_PROMPT_MINIMAL,
        CLUSTER_DESCRIPTION_PROMPT_LOW_PROB_FOCUS,
        format_samples_with_probability,
        format_keywords,
        build_prompt_v2,
    )


@dataclass
class LabelResultExp:
    """Extended label result with metadata about generation."""
    cluster_id: int
    theme: str
    description: str
    key_concepts: List[str]
    confidence: str
    notes: Optional[str]

    # Metadata about generation
    prompt_template: str
    high_prob_count: int
    low_prob_count: int
    high_threshold: float
    low_threshold: float


class LabelGeneratorExp:
    """
    Experimental label generator for A/B testing different approaches.

    Key differences from production LabelGenerator:
    - Supports separate high/low probability member sections
    - Can use different prompt templates
    - Returns extended results with generation metadata
    - Works directly with cached data structures
    """

    def __init__(self, config: Optional[LabelExperimentConfig] = None):
        """Initialize experimental label generator."""
        self.config = config or LabelExperimentConfig()

    def generate_label_v2(
        self,
        cluster_id: int,
        high_prob_samples: List[Tuple[str, float]],
        low_prob_samples: List[Tuple[str, float]],
        keywords: List[Tuple[str, float]],
        survey_question: str = "",
        language: str = "Dutch",
        dataset_context: str = "",
        taxonomy_context: str = "",
        verbose: bool = True
    ) -> Tuple[LabelResultExp, str]:
        """
        Generate label using V2 prompt with high/low probability sections.

        Args:
            cluster_id: Cluster identifier
            high_prob_samples: List of (text, probability) for core members
            low_prob_samples: List of (text, probability) for boundary members
            keywords: List of (keyword, score) tuples
            survey_question: The survey question text
            language: Output language
            dataset_context: Optional dataset context
            taxonomy_context: Optional taxonomy context
            verbose: Print debug info

        Returns:
            Tuple of (LabelResultExp, prompt_string)
        """
        # Limit samples
        high_samples = high_prob_samples[:self.config.max_samples_per_cluster]
        low_samples = low_prob_samples[:self.config.max_low_prob_samples]

        # Build prompt
        prompt = build_prompt_v2(
            survey_question=survey_question,
            cluster_id=cluster_id,
            high_prob_samples=high_samples,
            low_prob_samples=low_samples if self.config.include_low_prob_section else [],
            keywords=keywords,
            high_threshold=self.config.high_prob_threshold,
            low_threshold=self.config.low_prob_threshold,
            keyword_method=self.config.keyword_method,
            language=language,
            dataset_context=dataset_context,
            taxonomy_context=taxonomy_context,
        )

        if verbose:
            print(f"\n[Generating label for cluster {cluster_id}]")
            print(f"  High-prob samples: {len(high_samples)}")
            print(f"  Low-prob samples: {len(low_samples)}")
            print(f"  Keywords: {len(keywords)}")

        try:
            client = create_client(model=self.config.model, async_mode=False)
            response = llm_create_sync(
                client=client,
                model=self.config.model,
                prompt=prompt,
                response_model=ClusterDescriptionExp,
                temperature=self.config.temperature,
                max_tokens=1000
            )

            result = LabelResultExp(
                cluster_id=cluster_id,
                theme=response.theme,
                description=response.description,
                key_concepts=response.key_concepts,
                confidence=response.confidence,
                notes=response.notes,
                prompt_template="v2",
                high_prob_count=len(high_samples),
                low_prob_count=len(low_samples),
                high_threshold=self.config.high_prob_threshold,
                low_threshold=self.config.low_prob_threshold,
            )

            if verbose:
                print(f"  Theme: {result.theme}")
                print(f"  Confidence: {result.confidence}")

            return result, prompt

        except Exception as e:
            if verbose:
                print(f"  ERROR: {type(e).__name__}: {e}")

            result = LabelResultExp(
                cluster_id=cluster_id,
                theme=f"Error: {type(e).__name__}",
                description=str(e),
                key_concepts=[],
                confidence="low",
                notes=f"LLM call failed: {e}",
                prompt_template="v2",
                high_prob_count=len(high_samples),
                low_prob_count=len(low_samples),
                high_threshold=self.config.high_prob_threshold,
                low_threshold=self.config.low_prob_threshold,
            )
            return result, prompt

    def generate_label_minimal(
        self,
        cluster_id: int,
        samples: List[Tuple[str, float]],
        keywords: List[Tuple[str, float]],
        survey_question: str = "",
        verbose: bool = True
    ) -> Tuple[ClusterDescriptionMinimal, str]:
        """
        Generate label using minimal prompt (for A/B testing).

        Args:
            cluster_id: Cluster identifier
            samples: List of (text, score) tuples
            keywords: List of (keyword, score) tuples
            survey_question: The survey question

        Returns:
            Tuple of (ClusterDescriptionMinimal, prompt_string)
        """
        samples_text = format_samples_with_probability(
            samples[:self.config.max_samples_per_cluster]
        )
        keywords_text = format_keywords(keywords[:self.config.n_keywords])

        prompt = CLUSTER_DESCRIPTION_PROMPT_MINIMAL.format(
            survey_question=survey_question,
            samples=samples_text,
            keywords=keywords_text,
        )

        if verbose:
            print(f"\n[Generating minimal label for cluster {cluster_id}]")

        try:
            client = create_client(model=self.config.model, async_mode=False)
            response = llm_create_sync(
                client=client,
                model=self.config.model,
                prompt=prompt,
                response_model=ClusterDescriptionMinimal,
                temperature=self.config.temperature,
                max_tokens=500
            )

            if verbose:
                print(f"  Theme: {response.theme}")

            return response, prompt

        except Exception as e:
            if verbose:
                print(f"  ERROR: {e}")
            return ClusterDescriptionMinimal(
                theme=f"Error: {type(e).__name__}",
                description=str(e),
                key_concepts=[]
            ), prompt

    def analyze_boundary_members(
        self,
        cluster_id: int,
        existing_theme: str,
        low_prob_samples: List[Tuple[str, float]],
        survey_question: str = "",
        threshold: float = 0.8,
        verbose: bool = True
    ) -> Tuple[BoundaryAnalysis, str]:
        """
        Analyze boundary (low-probability) members for a cluster.

        This helps understand:
        - Do low-prob members actually fit the theme?
        - Should the theme be broadened?
        - Are some members misclassified?

        Args:
            cluster_id: Cluster identifier
            existing_theme: Current theme from high-prob members
            low_prob_samples: List of (text, probability) for boundary members
            survey_question: The survey question
            threshold: Probability threshold used
            verbose: Print debug info

        Returns:
            Tuple of (BoundaryAnalysis, prompt_string)
        """
        samples_text = format_samples_with_probability(low_prob_samples)

        prompt = CLUSTER_DESCRIPTION_PROMPT_LOW_PROB_FOCUS.format(
            survey_question=survey_question,
            existing_theme=existing_theme,
            threshold=threshold,
            low_prob_samples=samples_text,
        )

        if verbose:
            print(f"\n[Analyzing boundary members for cluster {cluster_id}]")
            print(f"  Existing theme: {existing_theme}")
            print(f"  Boundary members: {len(low_prob_samples)}")

        try:
            client = create_client(model=self.config.model, async_mode=False)
            response = llm_create_sync(
                client=client,
                model=self.config.model,
                prompt=prompt,
                response_model=BoundaryAnalysis,
                temperature=self.config.temperature,
                max_tokens=800
            )

            if verbose:
                print(f"  Theme adjustment: {response.theme_adjustment or '(keep current)'}")
                print(f"  Misfit count: {response.misfit_count}")
                print(f"  Confidence: {response.confidence}")

            return response, prompt

        except Exception as e:
            if verbose:
                print(f"  ERROR: {e}")
            return BoundaryAnalysis(
                boundary_analysis=f"Error: {e}",
                theme_adjustment=None,
                misfit_count=0,
                confidence="low"
            ), prompt

    def compare_thresholds(
        self,
        cluster_id: int,
        all_samples: List[Tuple[str, float]],
        keywords: List[Tuple[str, float]],
        thresholds: List[float],
        survey_question: str = "",
        verbose: bool = True
    ) -> Dict[float, LabelResultExp]:
        """
        Compare labels generated at different probability thresholds.

        Args:
            cluster_id: Cluster identifier
            all_samples: All samples with probabilities
            keywords: Keywords for the cluster
            thresholds: List of thresholds to test
            survey_question: Survey question
            verbose: Print debug info

        Returns:
            Dict mapping threshold to LabelResultExp
        """
        results = {}

        for threshold in thresholds:
            if verbose:
                print(f"\n{'='*50}")
                print(f"Testing threshold: {threshold}")

            # Split samples by this threshold
            high_prob = [(t, p) for t, p in all_samples if p >= threshold]
            low_prob = [(t, p) for t, p in all_samples if p < threshold and p >= 0.3]

            if not high_prob:
                if verbose:
                    print(f"  No samples above threshold {threshold}")
                continue

            # Temporarily update config
            original_high = self.config.high_prob_threshold
            self.config.high_prob_threshold = threshold

            result, _ = self.generate_label_v2(
                cluster_id=cluster_id,
                high_prob_samples=high_prob,
                low_prob_samples=low_prob,
                keywords=keywords,
                survey_question=survey_question,
                verbose=verbose
            )

            # Restore config
            self.config.high_prob_threshold = original_high

            results[threshold] = result

        return results


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def extract_samples_from_cluster_models(
    cluster_models: List[models.ClusterModel],
    cluster_id: int
) -> List[Tuple[str, float]]:
    """
    Extract (text, probability) samples for a cluster from ClusterModel list.

    Args:
        cluster_models: List of ClusterModel from cache
        cluster_id: Cluster to extract

    Returns:
        List of (idea_text, probability) tuples, sorted by probability desc
    """
    samples = []

    for model in cluster_models:
        if model.response_ideas:
            for idea in model.response_ideas:
                if idea.initial_cluster == cluster_id:
                    prob = idea.cluster_probability or 0.0
                    samples.append((idea.idea, prob))

    # Sort by probability descending
    samples.sort(key=lambda x: x[1], reverse=True)

    return samples


def extract_keywords_from_metadata(
    metadata: models.ClusteringMetadataModel,
    cluster_id: int,
    method: str = "mmr"
) -> List[Tuple[str, float]]:
    """
    Extract keywords for a cluster from cached metadata.

    Args:
        metadata: ClusteringMetadataModel from cache
        cluster_id: Cluster ID
        method: Keyword method ("mmr", "ctfidf", "tfidf")

    Returns:
        List of (keyword, score) tuples
    """
    if cluster_id not in metadata.clusters:
        return []

    cluster = metadata.clusters[cluster_id]

    if method == "mmr":
        return cluster.keywords_mmr or []
    elif method == "ctfidf":
        return cluster.keywords_ctfidf or []
    elif method == "tfidf":
        return cluster.keywords_tfidf or []
    else:
        return cluster.keywords_mmr or cluster.keywords_ctfidf or []


def get_survey_question_from_metadata(
    metadata: Optional[models.ClusteringMetadataModel]
) -> str:
    """Extract survey question from metadata."""
    if metadata and metadata.llm_context:
        return metadata.llm_context.survey_question or ""
    return ""
