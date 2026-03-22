#%%
"""
View codebook results (P8-P9): codes with definitions, indicators, source attributes.

Loads from cached MECE results (step "mece_categories").

Usage:
    cd src && python -m development.step_4_classNcoder.view_codebook
"""

import sys
from pathlib import Path

src_dir = Path(__file__).parent.parent.parent
project_root = src_dir.parent
sys.path.insert(0, str(src_dir))

from utils.cacheManager import CacheManager, generate_enhanced_variable_key

try:
    from development.test_data import TEST_DATA
except ImportError:
    exp_root = Path(__file__).parent.parent
    if str(exp_root) not in sys.path:
        sys.path.insert(0, str(exp_root))
    from test_data import TEST_DATA

from development.step_4_classNcoder.models_exp import CodingResultsCache
from development.step_4_classNcoder.prompts_exp import ConsolidatedCode

FILENAME = TEST_DATA.filename
VAR_NAME = TEST_DATA.var_name
SAMPLE_SIZE = TEST_DATA.sample_size


def main():
    print("=" * 80)
    print("CODEBOOK VIEWER (P8-P9)")
    print("=" * 80)
    print(f"Variable:     {VAR_NAME}")
    print(f"Sample size:  {SAMPLE_SIZE}")

    variable_key = generate_enhanced_variable_key([VAR_NAME], False, SAMPLE_SIZE)
    cache_manager = CacheManager()
    mece_cache = cache_manager.load_metadata_from_cache(
        filename=FILENAME,
        step="mece_categories",
        variable_key=variable_key,
        model_cls=CodingResultsCache,
    )

    if mece_cache is None:
        print("\nNo cached codebook results found.")
        print("Run codebook generation first: RUN_MODE = 'codebook' or 'all'")
        return

    # Reconstruct ConsolidatedCode objects
    codes = [ConsolidatedCode(**d) for d in mece_cache.raw_codes] if mece_cache.raw_codes else []

    print(f"\n{len(codes)} codes")
    print("=" * 80)

    for j, code in enumerate(codes, 1):
        indicators = ", ".join(code.typical_indicators[:5]) if code.typical_indicators else "(none)"
        sources = ", ".join(code.source_attributes[:5]) if code.source_attributes else "(none)"
        valence = getattr(code, 'valence', '') or ''
        diagnostic = getattr(code, 'diagnostic_test', '') or ''

        print(f"\n  [{j}] {code.code_name}")
        print(f"      Definition:    {code.definition}")
        if diagnostic:
            print(f"      Diagnostic:    {diagnostic}")
        if valence:
            print(f"      Valence:       {valence}")
        print(f"      Indicators:    {indicators}")
        print(f"      Source attrs:  {sources}")

    # Summary by domain (from partition_results)
    print(f"\n{'='*80}")
    print(f"DOMAIN SUMMARY")
    print(f"{'='*80}")
    for name, result in mece_cache.partition_results.items():
        n_facets = len(result.facets)
        n_attrs = sum(len(a) for a in result.attributes.values())
        print(f"  {name}: {n_facets} facets, {n_attrs} attributes")

    print(f"\n{'='*80}")
    print(f"Total codes: {len(codes)}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

# %%
