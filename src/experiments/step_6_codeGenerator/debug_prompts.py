#%%

"""
Debug script for Step 6: Code Generator - Prompt Testing
Tests prompt generation for specific clusters.

Usage:
    cd src && python -m experiments.step_6_codeGenerator.debug_prompts
"""

import sys
from pathlib import Path
src_dir = Path(__file__).parent.parent.parent
project_root = src_dir.parent
data_dir = project_root / "data"
sys.path.insert(0, str(src_dir))

import random
from config import CacheConfig
from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from utils.codeGenerator import CodeGeneratorReasoningResults
from utils.codegenPromptTester import SimplePromptTester
from utils.codegenResults import display_cluster_analysis
from utils import dataLoader

# Configuration
FILENAME = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
VAR_NAME = "Q20"
SAMPLE_SIZE = 500

# Optional: specify cluster ID, or None for random
CLUSTER_ID = None


def main():
    variable_key = generate_enhanced_variable_key([VAR_NAME], False, SAMPLE_SIZE)
    cache_manager = CacheManager(CacheConfig())

    # Get variable label
    loader = dataLoader.DataLoader(data_dir=str(data_dir), verbose=False)
    var_lab = loader.get_varlab(filename=FILENAME, var_name=VAR_NAME)

    # Load codebook reasoning
    reasoning_models = cache_manager.load_from_cache(
        FILENAME, "codebook_generation_reasoning", variable_key, CodeGeneratorReasoningResults
    )

    if not reasoning_models:
        print("No codebook reasoning found in cache")
        return

    codebook_reasoning = reasoning_models[0]

    # Get available cluster IDs
    step3_recommendations = getattr(codebook_reasoning, 'step3_recommendations', {})
    if not step3_recommendations:
        print("No step3_recommendations found in reasoning")
        return

    available_ids = list(step3_recommendations.keys())
    print(f"Available cluster IDs: {len(available_ids)}")

    # Select cluster
    if CLUSTER_ID and CLUSTER_ID in available_ids:
        cluster_id = CLUSTER_ID
    else:
        cluster_id = random.choice(available_ids)

    print(f"\nTesting prompts for cluster: {cluster_id}")
    print("=" * 70)

    # Test all prompts
    tester = SimplePromptTester(cluster_id=cluster_id, var_lab=var_lab)

    print("\n--- PROMPT 1 ---")
    tester.test_prompt_1()

    print("\n--- PROMPT 2 ---")
    tester.test_prompt_2()

    print("\n--- PROMPT 3 ---")
    tester.test_prompt_3()

    print("\n--- PROMPT 4 ---")
    tester.test_prompt_4()

    # Display cluster analysis
    print("\n" + "=" * 70)
    print("CLUSTER ANALYSIS")
    print("=" * 70)
    display_cluster_analysis(codebook_reasoning, cluster_id=cluster_id)


if __name__ == "__main__":
    main()
