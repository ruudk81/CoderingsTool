#%%
"""
View taxonomy results (P1-P7): domains, facets, attributes, assignments.

Loads from cached taxonomy results (step "taxonomy").

Usage:
    cd src && python -m steps.step_4_classifier.view_taxonomy
"""

import sys
from pathlib import Path

src_dir = Path(__file__).parent.parent.parent
project_root = src_dir.parent
sys.path.insert(0, str(src_dir))

from utils.cacheManager import CacheManager, generate_enhanced_variable_key

from test_data import TEST_DATA

from pipeline.step_4_classifier.models_classifier import TaxonomyResultsCache

FILENAME = TEST_DATA.filename
VAR_NAME = TEST_DATA.var_name
SAMPLE_SIZE = TEST_DATA.sample_size


def main():
    print("=" * 80)
    print("TAXONOMY RESULTS VIEWER (P1-P7)")
    print("=" * 80)
    print(f"Variable:     {VAR_NAME}")
    print(f"Sample size:  {SAMPLE_SIZE}")

    variable_key = generate_enhanced_variable_key([VAR_NAME], False, SAMPLE_SIZE)
    cache_manager = CacheManager()
    taxonomy_cache = cache_manager.load_metadata_from_cache(
        filename=FILENAME,
        step="taxonomy",
        variable_key=variable_key,
        model_cls=TaxonomyResultsCache,
    )

    if taxonomy_cache is None:
        print("\nNo cached taxonomy results found.")
        print("Run taxonomy first: RUN_MODE = 'taxonomy' or 'all'")
        return

    partition_set = taxonomy_cache.partition_set
    results = taxonomy_cache.partition_results

    print(f"\n{len(partition_set.partitions)} domains")
    print("=" * 80)

    for i, part in enumerate(partition_set.partitions, 1):
        name = part.partition_name
        result = results.get(name)

        print(f"\n{'─'*80}")
        n_labels = result.n_labels if result else 0
        n_facets = len(result.facets) if result else 0
        n_attrs = sum(
            len(attrs) for attrs in result.attributes.values()
        ) if result else 0
        n_attr_assigned = len(result.attribute_assignments) if result else 0
        print(f"DOMAIN {i}: {name}")
        print(f"  {n_labels} observations, {n_facets} facets, "
              f"{n_attrs} attributes, {n_attr_assigned} ideas with attributes")
        print(f"{'─'*80}")
        print(f"  Definition: {part.inclusion_definition}")

        if result and result.facets:
            print(f"\n  Facets ({len(result.facets)}):")
            for j, facet_dict in enumerate(result.facets, 1):
                facet_name = facet_dict.get("facet_name", "?")
                facet_desc = facet_dict.get("facet_description", "")
                print(f"    {j}. {facet_name}: {facet_desc}")

        if result and result.attributes:
            print(f"\n  Attributes per facet:")
            for facet_name, attrs in sorted(result.attributes.items()):
                # Count ideas assigned to each attribute
                attr_counts = {}
                for attr_name in result.attribute_assignments.values():
                    attr_counts[attr_name] = attr_counts.get(attr_name, 0) + 1

                print(f"    {facet_name} ({len(attrs)} attributes):")
                for attr_dict in attrs:
                    attr_name = attr_dict.get("attribute_name", "?")
                    attr_desc = attr_dict.get("attribute_description", "")
                    count = attr_counts.get(attr_name, 0)
                    print(f"      - {attr_name}: {count}")

    # Summary
    total_facets = sum(len(r.facets) for r in results.values())
    total_attrs = sum(
        len(attrs)
        for r in results.values()
        for attrs in r.attributes.values()
    )
    total_assigned = sum(
        len(r.attribute_assignments) for r in results.values()
    )

    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"  Domains:              {len(partition_set.partitions)}")
    print(f"  Facets:               {total_facets}")
    print(f"  Attributes:           {total_attrs}")
    print(f"  Ideas with attributes:{total_assigned}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

# %%
