"""
Debug script for Step 8: Code Assigner
Loads code assignments from cache and displays random samples with prompts.

Usage:
    cd src && python -m experiments.step_8_codeAssigner.debug_samples
"""

import sys
from pathlib import Path
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

import random
from experiments import models_exp as models
from config import CacheConfig
from utils.cacheManager import CacheManager, generate_enhanced_variable_key

# Configuration
FILENAME = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
VAR_NAME = "Q20"
SAMPLE_SIZE = 500
N_SAMPLES = 5

# Optional: filter by specific code
FILTER_CODE = None  # e.g., "ONBEKEND"


def main():
    variable_key = generate_enhanced_variable_key([VAR_NAME], False, SAMPLE_SIZE)
    cache_manager = CacheManager(CacheConfig())

    # Load code assigned results
    code_assigned_results = cache_manager.load_from_cache(
        FILENAME, "code_assignment_direct", variable_key, models.CodeAssignedModel
    )

    print(f"Loaded {len(code_assigned_results)} code-assigned responses")

    # Collect all ideas with assignments
    all_ideas = []
    for result in code_assigned_results:
        if result.response_ideas:
            for idea in result.response_ideas:
                if idea.assigned_codes:
                    all_ideas.append({
                        'respondent_id': result.respondent_id,
                        'response': result.response,
                        'idea': idea.idea,
                        'codes': idea.assigned_codes,
                        'themes': idea.assigned_themes or []
                    })

    print(f"Total ideas with codes: {len(all_ideas)}")

    # Filter if specified
    if FILTER_CODE:
        all_ideas = [i for i in all_ideas if FILTER_CODE in i['codes']]
        print(f"Filtered to '{FILTER_CODE}': {len(all_ideas)} ideas")

    if not all_ideas:
        print("No ideas found matching criteria")
        return

    # Sample
    samples = random.sample(all_ideas, min(N_SAMPLES, len(all_ideas)))

    print("\n" + "=" * 70)
    print("SAMPLE CODE ASSIGNMENTS")
    print("=" * 70)

    for i, item in enumerate(samples, 1):
        print(f"\n--- Sample {i} ---")
        print(f"Response: {item['response'][:100]}..." if len(item['response']) > 100 else f"Response: {item['response']}")
        print(f"\nIdea: {item['idea']}")
        print(f"\nAssigned codes: {item['codes']}")
        print(f"Assigned themes: {item['themes']}")
        print("-" * 70)


if __name__ == "__main__":
    main()
