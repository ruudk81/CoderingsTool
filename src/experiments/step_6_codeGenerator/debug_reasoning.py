#%%

"""
Debug script for Step 6: Code Generator - Reasoning Display
Loads codebook reasoning from cache and displays cluster analysis.

Usage:
    cd src && python -m experiments.step_6_codeGenerator.debug_reasoning
"""

import sys
from pathlib import Path
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

from config import CacheConfig
from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from experiments.step_6_codeGenerator.codeGenerator_exp import CodeGeneratorReasoningResults
from utils.codegenResults import display_cluster_analysis
from experiments.test_data import TEST_DATA

# Configuration — uses centralized test_data.py (same dataset as run_experiment.py)
FILENAME = TEST_DATA.filename
VAR_NAME = TEST_DATA.var_name
SAMPLE_SIZE = TEST_DATA.sample_size

# Optional: specify cluster ID to focus on, or None for all
CLUSTER_ID = None


def main():
    variable_key = generate_enhanced_variable_key([VAR_NAME], False, SAMPLE_SIZE)
    cache_manager = CacheManager(CacheConfig())

    # Load codebook reasoning
    reasoning_models = cache_manager.load_from_cache(
        FILENAME, "codebook_generation_reasoning", variable_key, CodeGeneratorReasoningResults
    )

    if not reasoning_models:
        print("No codebook reasoning found in cache")
        return

    codebook_reasoning = reasoning_models[0]

    print(f"Loaded codebook reasoning")
    if codebook_reasoning.codebook:
        print(f"Codes generated: {len(codebook_reasoning.codebook)}")

    print("\n" + "=" * 70)
    print("CLUSTER ANALYSIS")
    print("=" * 70)

    if CLUSTER_ID:
        display_cluster_analysis(codebook_reasoning, cluster_id=CLUSTER_ID)
    else:
        display_cluster_analysis(codebook_reasoning)


if __name__ == "__main__":
    main()
