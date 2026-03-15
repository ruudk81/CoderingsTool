"""
Debug script for Step 2: Quality Filter
Loads cached results and prints samples of filtered responses.

Usage:
    cd src && python -m development.step_2_qualityFilter.debug_samples
"""

import sys
from pathlib import Path
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

import random
import models
from config import CacheConfig
from utils.cacheManager import CacheManager, generate_enhanced_variable_key

# Configuration
FILENAME = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
VAR_NAME = "Q20"
SAMPLE_SIZE = 500
N_SAMPLES = 5


def main():
    variable_key = generate_enhanced_variable_key([VAR_NAME], False, SAMPLE_SIZE)
    cache_manager = CacheManager(CacheConfig())

    # Load quality filtered data
    quality_filtered_text = cache_manager.load_from_cache(
        FILENAME, "quality_filter", variable_key, models.QualityFilteredModel
    )

    print(f"Loaded {len(quality_filtered_text)} quality-filtered responses")

    # Separate filtered vs passed
    filtered = [item for item in quality_filtered_text if item.quality_filter]
    passed = [item for item in quality_filtered_text if not item.quality_filter]

    print(f"Filtered out: {len(filtered)}")
    print(f"Passed: {len(passed)}")

    # Code meanings
    code_meanings = {
        99999997: "Don't know (uncertainty)",
        99999998: "No response (empty/NA)",
        99999999: "Meaningless (gibberish)"
    }

    # Show filtered samples
    print("\n" + "=" * 70)
    print("FILTERED RESPONSES (samples)")
    print("=" * 70)

    if filtered:
        samples = random.sample(filtered, min(N_SAMPLES, len(filtered)))
        for item in samples:
            code = item.quality_filter_code
            meaning = code_meanings.get(code, "Unknown")
            print(f"\n[{code}: {meaning}]")
            print(f"Response: {item.response}")
            print("-" * 70)
    else:
        print("No filtered responses")

    # Show passed samples
    print("\n" + "=" * 70)
    print("PASSED RESPONSES (samples)")
    print("=" * 70)

    if passed:
        samples = random.sample(passed, min(N_SAMPLES, len(passed)))
        for item in samples:
            print(f"\nResponse: {item.response}")
            print("-" * 70)


if __name__ == "__main__":
    main()
