#%%

"""
View quality filter results: inspect what the LLM classified.

Shows:
- Summary: counts by quality_filter_code
- Filtered responses: all responses marked as don't know or gibberish
- Sample meaningful responses: random sample of passed responses

Usage:
    cd src && python -m development.step_2_qualityFilter.view_quality_results
"""

import sys
from collections import Counter
from pathlib import Path
from typing import Optional, List

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import models
from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from config import CacheConfig
from utils import dataLoader

try:
    from development.test_data import TEST_DATA
except ImportError:
    exp_root = Path(__file__).parent.parent
    if str(exp_root) not in sys.path:
        sys.path.insert(0, str(exp_root))
    from test_data import TEST_DATA


# =============================================================================
# CONFIGURATION
# =============================================================================

FILENAME = TEST_DATA.filename
VARIABLE = TEST_DATA.var_name
SAMPLE_SIZE = TEST_DATA.sample_size

SHOW_ALL = True  # Show all responses ordered by code (97, 99, meaningful)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_results(
    filename: str = FILENAME,
    variable: str = VARIABLE,
    sample_size: Optional[int] = SAMPLE_SIZE,
) -> List[models.QualityFilteredModel]:
    """Load quality filter results from cache."""
    variable_key = generate_enhanced_variable_key([variable], False, sample_size)
    cache_manager = CacheManager(CacheConfig())

    if not cache_manager.is_cache_valid(filename, "quality_filter", variable_key):
        print(f"Cache not found: quality_filter/{variable_key}")
        print("Run step 2 first.")
        return []

    return cache_manager.load_from_cache(
        filename, "quality_filter", variable_key, models.QualityFilteredModel
    )


# =============================================================================
# DISPLAY
# =============================================================================

def main():
    data_dir = project_root / "data"
    loader = dataLoader.DataLoader(data_dir=str(data_dir), verbose=False)
    var_lab = loader.get_varlab(filename=FILENAME, var_name=VARIABLE)

    results = load_results()
    if not results:
        return

    # Summary
    code_counts = Counter(r.quality_filter_code for r in results)
    meaningful = [r for r in results if not r.quality_filter]
    filtered = [r for r in results if r.quality_filter]
    dont_know = [r for r in results if r.quality_filter_code == 99999997]
    no_response = [r for r in results if r.quality_filter_code == 99999998]
    gibberish = [r for r in results if r.quality_filter_code == 99999999]
    errors = [r for r in results if r.quality_filter_code == -1]

    print("=" * 100)
    print("QUALITY FILTER RESULTS")
    print("=" * 100)
    print(f"Variable:    {VARIABLE} - {var_lab}")
    print(f"Sample size: {SAMPLE_SIZE}")
    print(f"Total:       {len(results)}")
    print()
    print(f"  Meaningful (null):     {len(meaningful):>5}  ({len(meaningful)/len(results)*100:.1f}%)")
    print(f"  Don't know (99999997): {len(dont_know):>5}  ({len(dont_know)/len(results)*100:.1f}%)")
    print(f"  No response(99999998): {len(no_response):>5}  ({len(no_response)/len(results)*100:.1f}%)")
    print(f"  Gibberish  (99999999): {len(gibberish):>5}  ({len(gibberish)/len(results)*100:.1f}%)")
    if errors:
        print(f"  Errors     (-1):       {len(errors):>5}  ({len(errors)/len(results)*100:.1f}%)")
    print()

    # All responses ordered by code: 97, 99, errors, meaningful
    idx = 0

    if dont_know:
        print(f"\n{'='*100}")
        print(f"DON'T KNOW RESPONSES — code 99999997 ({len(dont_know)})")
        print(f"{'='*100}")
        for r in dont_know:
            idx += 1
            text = str(r.response)[:300] if r.response else "(empty)"
            print(f"  {idx:>4}. [{r.respondent_id}] \"{text}\"")

    if no_response:
        print(f"\n{'='*100}")
        print(f"NO RESPONSE (empty/NA) — code 99999998 ({len(no_response)})")
        print(f"{'='*100}")
        for r in no_response:
            idx += 1
            text = str(r.response)[:300] if r.response else "(empty)"
            print(f"  {idx:>4}. [{r.respondent_id}] \"{text}\"")

    if gibberish:
        print(f"\n{'='*100}")
        print(f"GIBBERISH / OFF-TOPIC RESPONSES — code 99999999 ({len(gibberish)})")
        print(f"{'='*100}")
        for r in gibberish:
            idx += 1
            text = str(r.response)[:300] if r.response else "(empty)"
            print(f"  {idx:>4}. [{r.respondent_id}] \"{text}\"")

    if errors:
        print(f"\n{'='*100}")
        print(f"ERROR RESPONSES — code -1 / fallback ({len(errors)})")
        print(f"{'='*100}")
        for r in errors:
            idx += 1
            text = str(r.response)[:300] if r.response else "(empty)"
            print(f"  {idx:>4}. [{r.respondent_id}] \"{text}\"")

    print(f"\n{'='*100}")
    print(f"MEANINGFUL RESPONSES — code null ({len(meaningful)})")
    print(f"{'='*100}")
    for r in meaningful:
        idx += 1
        text = str(r.response)[:300] if r.response else "(empty)"
        print(f"  {idx:>4}. [{r.respondent_id}] \"{text}\"")

    print(f"\n{'='*100}")
    print(f"Total: {idx} responses")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
