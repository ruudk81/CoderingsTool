"""
IdeaExtractor V2 Experiment Package

Provides an isolated experimentation environment for ideaExtractor development.
Allows testing prompt changes and logic modifications without affecting the production pipeline.

Usage:
    cd src && python -m experiments.ideaExtractor_v2.run_experiment

Toggle modes (in run_experiment.py):
    USE_EXPERIMENTAL_EXTRACTOR = True  -> Uses local extractor + local prompts
    USE_EXPERIMENTAL_PROMPTS = True    -> Uses production extractor + local prompts
"""

from .run_experiment import run_experiment, ExperimentConfig, EXPERIMENT_CONFIG

__all__ = ['run_experiment', 'ExperimentConfig', 'EXPERIMENT_CONFIG']
