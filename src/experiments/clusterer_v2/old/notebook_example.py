"""
Jupyter/VS Code Notebook Example for Cluster Analysis Experiments

This file shows how to run experiments in a notebook environment.
Copy the cells below into your notebook or run this file as cells.
"""

# %%
# Cell 1: Imports and Setup
import sys
import os

# Ensure parent directory is in path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from experiments.cluster_analysis import ExperimentConfig, EXPERIMENTS, run_experiment

# %%
# Cell 2: Configure and Run Experiment

config = ExperimentConfig(
    # Data source (must match cached Step 5 data)
    filename="M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav",
    var_name="Q20",
    sample_size=50,

    # Choose experiment configuration
    tfidf_config=EXPERIMENTS["bigrams"],  # Options: "baseline", "bigrams", "strict_filtering"

    # LLM settings
    description_model="gpt-4.1",
    use_keywords_in_prompt=True,  # Toggle: True = with keywords, False = without

    # Display settings
    n_sample_clusters=5,  # Number of sample clusters to show
    show_comparisons=True,  # Compare with original Step 6 codes
    verbose=True
)

run_experiment(config)

# %%
# Cell 3: Compare Different TF-IDF Configurations

# Run multiple experiments to compare approaches
for exp_name in ["baseline", "bigrams", "strict_filtering"]:
    print(f"\n{'='*80}")
    print(f"RUNNING EXPERIMENT: {exp_name.upper()}")
    print(f"{'='*80}\n")

    config.tfidf_config = EXPERIMENTS[exp_name]
    run_experiment(config)

    print("\n\n")

# %%
# Cell 4: Compare With vs Without Keyword Enhancement

print("="*80)
print("EXPERIMENT 1: WITHOUT KEYWORD ENHANCEMENT")
print("="*80)
config.use_keywords_in_prompt = False
config.tfidf_config = EXPERIMENTS["bigrams"]
run_experiment(config)

print("\n\n")
print("="*80)
print("EXPERIMENT 2: WITH KEYWORD ENHANCEMENT")
print("="*80)
config.use_keywords_in_prompt = True
run_experiment(config)

# %%
# Cell 5: Custom TF-IDF Configuration

from experiments.tfidf_analyzer import TfidfConfig

# Create custom configuration
custom_config = TfidfConfig(
    max_features=1500,
    ngram_range=(1, 3),  # Include trigrams
    min_df=2,
    max_df=0.7,
    top_k_keywords=12,
    language="nl",
    custom_stopwords=["producent", "kant", "klare"]  # Add domain-specific stopwords
)

config.tfidf_config = custom_config
run_experiment(config)
