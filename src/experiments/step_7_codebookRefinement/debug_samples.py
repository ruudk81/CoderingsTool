"""
Debug script for Step 7: Codebook Refinement
Loads refined codebook from cache and displays structure.

Usage:
    cd src && python -m experiments.step_7_codebookRefinement.debug_samples
"""

import sys
from pathlib import Path
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

from experiments import models_exp as models
from config import CacheConfig
from utils.cacheManager import CacheManager, generate_enhanced_variable_key

# Configuration
FILENAME = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
VAR_NAME = "Q20"
SAMPLE_SIZE = 500


def main():
    variable_key = generate_enhanced_variable_key([VAR_NAME], False, SAMPLE_SIZE)
    cache_manager = CacheManager(CacheConfig())

    # Load refinement results
    refinement_results = cache_manager.load_from_cache(
        FILENAME, "codebook_refinement", variable_key, models.CodeRefinementResults
    )

    if not refinement_results:
        print("No refinement results found in cache")
        return

    refinement = refinement_results[0]

    print("=" * 70)
    print("REFINED CODEBOOK STRUCTURE")
    print("=" * 70)

    final_codebook = refinement.refined_codebook

    for entry in final_codebook.refined_codebook:
        print(f"\n[THEME] {entry.category}")
        print("-" * 50)
        for subcode in entry.subcodes:
            print(f"  - {subcode.code}")
            if subcode.description:
                desc = subcode.description[:60] + "..." if len(subcode.description) > 60 else subcode.description
                print(f"    {desc}")
        print()

    # Summary
    total_themes = len(final_codebook.refined_codebook)
    total_codes = sum(len(e.subcodes) for e in final_codebook.refined_codebook)
    print("=" * 70)
    print(f"Total themes: {total_themes}")
    print(f"Total codes: {total_codes}")
    print("=" * 70)


if __name__ == "__main__":
    main()
