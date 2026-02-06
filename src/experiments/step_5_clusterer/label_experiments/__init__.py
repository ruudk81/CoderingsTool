"""
Label Experiments - Extension layer for cluster label generation tuning.

This module provides tools for experimenting with:
- Low-probability cluster members (< 0.8)
- HDBSCAN tree structures (hierarchy analysis)
- Prompt construction and formatting variations
- A/B testing different labeling approaches

Usage:
    cd src && python -m experiments.step_5_clusterer.label_experiments.run_label_experiments

Or open run_label_experiments.py in VS Code and run cells interactively.
"""

from .config_label_exp import LabelExperimentConfig

__all__ = ["LabelExperimentConfig"]
