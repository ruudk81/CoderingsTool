"""
Configuration for label generation experiments.

This is a separate configuration from the main clusterer config,
focused specifically on label generation tuning.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LabelExperimentConfig:
    """Configuration for label generation experiments."""

    # ==========================================================================
    # PROBABILITY THRESHOLDS
    # ==========================================================================

    # High-probability threshold (ideas considered "core" cluster members)
    high_prob_threshold: float = 0.8

    # Low-probability threshold (ideas to include in secondary section)
    low_prob_threshold: float = 0.5

    # Thresholds to experiment with in batch runs
    probability_thresholds: List[float] = field(
        default_factory=lambda: [0.5, 0.6, 0.7, 0.8, 0.9]
    )

    # ==========================================================================
    # REPRESENTATIVE SELECTION
    # ==========================================================================

    # Method for selecting representative ideas: "dense_region", "centroid", "all_members"
    selection_method: str = "dense_region"

    # Max samples to include per cluster in LLM prompt
    max_samples_per_cluster: int = 10

    # Max low-prob samples to include (separate section)
    max_low_prob_samples: int = 5

    # ==========================================================================
    # KEYWORD CONFIGURATION
    # ==========================================================================

    # Keyword method: "mmr", "ctfidf", "tfidf", "combined"
    keyword_method: str = "mmr"

    # Number of keywords to show
    n_keywords: int = 10

    # ==========================================================================
    # LLM SETTINGS
    # ==========================================================================

    # Model for label generation
    model: str = "gpt-4.1"

    # Temperature (lower = more deterministic)
    temperature: float = 0.3

    # ==========================================================================
    # EXPERIMENT SCOPE
    # ==========================================================================

    # Which clusters to process (None = all, or list of specific IDs)
    cluster_ids: Optional[List[int]] = None

    # Include low-probability members in separate prompt section
    include_low_prob_section: bool = True

    # Verbose output
    verbose: bool = True
