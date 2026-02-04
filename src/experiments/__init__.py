"""
Experiments Module

This module contains experiment runners for each pipeline step.
Each step folder provides:
- run_experiment.py: Runs the step in isolation with production/experimental toggle
- debug_*.py: Debug scripts for inspecting cached data and prompts
- *_exp.py: Experimental util versions (copy from utils/ when experimenting)
- config_exp.py: Experimental config (copy from config_*.py when experimenting)
- prompts_exp.py: Experimental prompts (copy from prompts.py when experimenting)

Usage:
    cd src && python -m experiments.step_3_ideaExtractor.run_experiment
    cd src && python -m experiments.step_5_clusterer.debug_clusters
"""
