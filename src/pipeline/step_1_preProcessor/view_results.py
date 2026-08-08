#%%

"""
Debug script for Step 1: Preprocess
Loads cached results and prints sample comparisons (raw vs preprocessed).

Usage:
    cd src && python -m pipeline.step_1_preProcessor.debug_samples
"""

import sys
from pathlib import Path
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

import random
import models
from config import CacheConfig
from test_data import TEST_DATA
from utils.cacheManager import CacheManager, generate_enhanced_variable_key

# Dataset comes from test_data.py — never hardcode a filename here.
FILENAME = TEST_DATA.filename
VAR_NAME = TEST_DATA.var_name
SAMPLE_SIZE = TEST_DATA.sample_size
N_SAMPLES = 5


def main():
    variable_key = generate_enhanced_variable_key([VAR_NAME], False, SAMPLE_SIZE)
    cache_manager = CacheManager(CacheConfig())

    # Load raw data (step 0)
    raw_text_list = cache_manager.load_from_cache(
        FILENAME, "data", variable_key, models.ResponseModel
    )

    # Load preprocessed data (step 1)
    preprocessed_text = cache_manager.load_from_cache(
        FILENAME, "preprocessed", variable_key, models.PreprocessedModel
    )

    print(f"Loaded {len(raw_text_list)} raw responses")
    print(f"Loaded {len(preprocessed_text)} preprocessed responses")

    # Create lookup for comparison
    preprocessed_map = {item.respondent_id: item for item in preprocessed_text}

    # Sample and compare
    indices = random.sample(range(len(raw_text_list)), min(N_SAMPLES, len(raw_text_list)))

    print("\n" + "=" * 70)
    print("SAMPLE COMPARISONS: Raw vs Preprocessed")
    print("=" * 70)

    for i in indices:
        raw = raw_text_list[i]
        preprocessed = preprocessed_map.get(raw.respondent_id)

        print(f"\n--- Respondent {raw.respondent_id} ---")
        print(f"Raw:          {raw.response}")
        if preprocessed:
            print(f"Preprocessed: {preprocessed.response}")
            if preprocessed.quality_filter_code:
                print(f"Filter code:  {preprocessed.quality_filter_code}")
        else:
            print("Preprocessed: (not found)")
        print("-" * 70)


if __name__ == "__main__":
    main()
