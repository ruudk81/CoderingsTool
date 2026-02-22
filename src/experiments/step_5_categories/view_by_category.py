#%%
#
"""
View Step 5 category assignment results organized by partition and assigned category.
Displays all ideas grouped by partition_name → assigned_category, showing:
idea, concept, concept_type_definition, valence, confidence, rationale.

Usage:
    cd src && python -m experiments.step_5_categories.view_by_category
"""

import sys
from pathlib import Path
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

import re
from collections import defaultdict

from experiments import models_exp as models
from config import CacheConfig
from utils.cacheManager import CacheManager, generate_enhanced_variable_key

# Import centralized test data config
try:
    from experiments.test_data import TEST_DATA
except ImportError:
    exp_root = Path(__file__).parent.parent
    if str(exp_root) not in sys.path:
        sys.path.insert(0, str(exp_root))
    from test_data import TEST_DATA

# Configuration (from centralized test_data.py)
FILENAME = TEST_DATA.filename
VAR_NAME = TEST_DATA.var_name
SAMPLE_SIZE = TEST_DATA.sample_size


def clean_idea(idea: str) -> str:
    """Remove brackets and normalize whitespace."""
    cleaned = re.sub(r"\[.*?\]", "", idea)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def main():
    variable_key = generate_enhanced_variable_key([VAR_NAME], False, SAMPLE_SIZE)
    cache_manager = CacheManager(CacheConfig())

    # Load category assignment results
    assigned_data = cache_manager.load_from_cache(
        FILENAME, "category_assignment", variable_key, models.CategoryAssignedModel
    )
    if not assigned_data:
        print("No category assignment results found in cache.")
        print("Run category assignment first via run_experiment.py")
        return

    print(f"Loaded {len(assigned_data)} responses")

    # Optionally load MECE cache for category definitions
    mece_cache = cache_manager.load_metadata_from_cache(
        filename=FILENAME,
        step="mece_categories",
        variable_key=variable_key,
        model_cls=models.MECEResultsCache,
    )
    category_defs = {}
    if mece_cache and mece_cache.partition_results:
        for name, result in mece_cache.partition_results.items():
            for cat in result.categories:
                category_defs[(name, cat.category_label)] = cat.inclusion_definition

    # Group ideas: partition_name → assigned_category → [ideas]
    partitions = defaultdict(lambda: defaultdict(list))
    unassigned = defaultdict(list)
    total_ideas = 0

    for item in assigned_data:
        if not item.response_ideas:
            continue
        for idea in item.response_ideas:
            total_ideas += 1
            partition = (idea.partition_name or "(unknown)").strip().lower()
            if idea.assigned_category:
                partitions[partition][idea.assigned_category].append(idea)
            else:
                unassigned[partition].append(idea)

    assigned_count = sum(
        len(ideas)
        for cats in partitions.values()
        for ideas in cats.values()
    )
    unassigned_count = sum(len(ideas) for ideas in unassigned.values())
    print(f"Total ideas: {total_ideas} ({assigned_count} assigned, "
          f"{unassigned_count} unassigned)")

    # Display each partition
    all_partitions = sorted(set(list(partitions.keys()) + list(unassigned.keys())))

    for partition in all_partitions:
        categories = partitions.get(partition, {})
        partition_unassigned = unassigned.get(partition, [])
        partition_total = (
            sum(len(ideas) for ideas in categories.values())
            + len(partition_unassigned)
        )

        print("\n" + "=" * 70)
        print(f"PARTITION: {partition.upper()} ({partition_total} ideas)")
        print("=" * 70)

        # Sort categories by count descending
        for cat_name in sorted(categories.keys(),
                               key=lambda c: -len(categories[c])):
            ideas = categories[cat_name]
            confidences = [
                i.category_confidence for i in ideas
                if i.category_confidence is not None
            ]
            avg_conf = (
                sum(confidences) / len(confidences) if confidences else 0.0
            )

            print(f"\n  CATEGORY: {cat_name} "
                  f"({len(ideas)} ideas, avg confidence: {avg_conf:.2f})")

            # Show category definition if available
            cat_def = category_defs.get((partition, cat_name))
            if cat_def:
                print(f"  Inclusion: {cat_def}")

            print(f"  {'-' * 60}")

            ideas.sort(key=lambda i: (i.concept or "", i.valence or ""))
            for idea in ideas:
                valence_str = f" [{idea.valence}]" if idea.valence else ""
                ctd_str = (f" ({idea.concept_type_definition})"
                           if idea.concept_type_definition else "")
                conf_str = (f" conf={idea.category_confidence:.2f}"
                            if idea.category_confidence is not None else "")
                print(f"  - {clean_idea(idea.idea)} | "
                      f"{idea.concept}{ctd_str}{valence_str}{conf_str}")

        # Show unassigned ideas
        if partition_unassigned:
            print(f"\n  (UNASSIGNED) ({len(partition_unassigned)} ideas)")
            print(f"  {'-' * 60}")
            partition_unassigned.sort(
                key=lambda i: (i.concept or "", i.valence or "")
            )
            for idea in partition_unassigned:
                valence_str = f" [{idea.valence}]" if idea.valence else ""
                ctd_str = (f" ({idea.concept_type_definition})"
                           if idea.concept_type_definition else "")
                print(f"  - {clean_idea(idea.idea)} | "
                      f"{idea.concept}{ctd_str}{valence_str}")

    # Grand summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Partitions: {len(all_partitions)}")
    print(f"  Total ideas: {total_ideas}")
    print(f"  Assigned: {assigned_count}")
    print(f"  Unassigned: {unassigned_count}")
    total_cats = sum(len(cats) for cats in partitions.values())
    print(f"  Total categories used: {total_cats}")


if __name__ == "__main__":
    main()
