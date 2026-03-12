"""Temporary script to run step_8 experiment with OpenAI provider. Delete after use."""
import config
config.API_PROVIDER = "openai"

from experiments.step_8_codeAssigner.run_experiment import run_experiment
run_experiment()
