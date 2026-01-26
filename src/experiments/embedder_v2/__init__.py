"""
Embedder V2 Experiment Module

This module provides experimental embedding generation for Step 4 of the pipeline.
It allows testing different embedding configurations, models, and text formats
without affecting production code.

Usage:
    cd src && python -m experiments.embedder_v2.run_experiment
"""

from .config import EmbedderExperimentConfig

__all__ = ['EmbedderExperimentConfig']
