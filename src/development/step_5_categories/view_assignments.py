#%%

"""
View category assignments: inspect which code each idea/ladder was assigned to.

Modes:
  - "idea":   compact view — idea text → assigned category + confidence
  - "ladder": detailed view — instance → rung_1 → rung_2 → category + rationale
"""

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "src" / "development"))

from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from development.step_5_categories.models_exp import CategoryAssignedModel

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

VIEW_MODE = "idea"          # "idea" or "ladder"
GROUP_BY = "category"       # "category" (by code across partitions) or "partition" (by partition → code)
SORT_BY = "category"        # "category" (grouped by code) or "confidence" (desc)
MAX_PER_CATEGORY = None       # None for all, or N to limit per category

FILENAME = TEST_DATA.filename
VARIABLE = TEST_DATA.var_name
SAMPLE_SIZE = TEST_DATA.sample_size


# =============================================================================
# DATA LOADING
# =============================================================================

def load_assignments(
    filename: str = FILENAME,
    variable: str = VARIABLE,
    sample_size: Optional[int] = SAMPLE_SIZE,
) -> List[CategoryAssignedModel]:
    """Load category assignment results from cache."""
    variable_key = generate_enhanced_variable_key(
        selected_variables=[variable],
        is_merged=False,
        sample_size=sample_size,
    )

    cache_manager = CacheManager()
    data = cache_manager.load_from_cache(
        filename, "category_assignment", variable_key, CategoryAssignedModel
    )

    if not data:
        raise FileNotFoundError(
            f"No cached assignment results for step 'category_assignment' / "
            f"variable_key '{variable_key}'.\n"
            f"Run the assignment first."
        )

    total_ideas = sum(
        len(r.response_ideas) for r in data if r.response_ideas
    )
    print(f"Loaded {len(data)} responses, {total_ideas} ideas")
    return data


# =============================================================================
# GROUPING
# =============================================================================

def group_by_partition(
    data: List[CategoryAssignedModel],
) -> Dict[str, Dict[str, List]]:
    """Group ideas by partition → (parent_category, assigned_category) → [ideas].

    Returns:
        Dict[partition_name, Dict[assigned_category, List[idea]]]
        Each idea is a CategoryAssignedSubmodel.
    """
    grouped: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))

    for response in data:
        if not response.response_ideas:
            continue
        for idea in response.response_ideas:
            partition = (idea.partition_name or idea.domain or "(unknown)").strip()
            category = idea.assigned_category or "(unassigned)"
            grouped[partition][category].append(idea)

    return dict(grouped)


def group_by_category(
    data: List[CategoryAssignedModel],
) -> Dict[str, List]:
    """Group ideas by (parent_category, assigned_category) across all partitions.

    Returns:
        Dict[assigned_category, List[idea]]
    """
    grouped: Dict[str, List] = defaultdict(list)

    for response in data:
        if not response.response_ideas:
            continue
        for idea in response.response_ideas:
            category = idea.assigned_category or "(unassigned)"
            grouped[category].append(idea)

    return dict(grouped)


# =============================================================================
# DISPLAY
# =============================================================================

def _category_sort_key(item):
    """Sort categories: parent-grouped, then by count descending."""
    category, ideas = item
    parent = ideas[0].parent_category or ""
    return (parent, -len(ideas))


def print_idea_mode(grouped: Dict[str, Dict[str, List]]):
    """Compact view: idea → code + confidence."""
    for partition in sorted(grouped.keys()):
        categories = grouped[partition]
        total = sum(len(ideas) for ideas in categories.values())

        print(f"\n{'═' * 80}")
        print(f"PARTITION: {partition} ({total} ideas)")
        print(f"{'═' * 80}")

        sorted_cats = sorted(categories.items(), key=_category_sort_key)

        for category, ideas in sorted_cats:
            parent = ideas[0].parent_category or ""
            avg_conf = (
                sum(i.category_confidence or 0 for i in ideas) / len(ideas)
                if ideas else 0
            )

            header = f"[{parent}] " if parent else ""
            print(f"\n  {header}{category} ({len(ideas)} ideas, avg conf: {avg_conf:.2f})")
            print(f"  {'─' * 70}")

            # Sort by confidence descending
            sorted_ideas = sorted(
                ideas, key=lambda i: i.category_confidence or 0, reverse=True
            )

            limit = MAX_PER_CATEGORY or len(sorted_ideas)
            for idx, idea in enumerate(sorted_ideas[:limit], 1):
                conf = idea.category_confidence or 0
                text = idea.idea or ""
                if len(text) > 80:
                    text = text[:77] + "..."
                valence = idea.valence or "0"
                print(f"    {idx:3d}. [{conf:.2f}] ({valence}) \"{text}\"  ({idea.idea_id})")

            if len(sorted_ideas) > limit:
                print(f"    ... +{len(sorted_ideas) - limit} more")


def print_ladder_mode(grouped: Dict[str, Dict[str, List]]):
    """Detailed view: full ladder + rationale per idea."""
    for partition in sorted(grouped.keys()):
        categories = grouped[partition]
        total = sum(len(ideas) for ideas in categories.values())

        print(f"\n{'═' * 80}")
        print(f"PARTITION: {partition} ({total} ideas)")
        print(f"{'═' * 80}")

        sorted_cats = sorted(categories.items(), key=_category_sort_key)

        for category, ideas in sorted_cats:
            parent = ideas[0].parent_category or ""
            avg_conf = (
                sum(i.category_confidence or 0 for i in ideas) / len(ideas)
                if ideas else 0
            )

            header = f"[{parent}] " if parent else ""
            print(f"\n  {header}{category} ({len(ideas)} ideas, avg conf: {avg_conf:.2f})")
            print(f"  {'─' * 70}")

            sorted_ideas = sorted(
                ideas, key=lambda i: i.category_confidence or 0, reverse=True
            )

            limit = MAX_PER_CATEGORY or len(sorted_ideas)
            for idx, idea in enumerate(sorted_ideas[:limit], 1):
                conf = idea.category_confidence or 0
                valence = idea.valence or "0"
                print(f"    {idx:3d}. [{conf:.2f}] ({idea.idea_id}) valence={valence}")
                print(f"         instance: \"{idea.instance or ''}\"")
                print(f"         rung_1:   {idea.rung_1 or ''}")
                print(f"         rung_2:   {idea.rung_2 or ''}")
                if idea.category_rationale:
                    rationale = idea.category_rationale
                    if len(rationale) > 120:
                        rationale = rationale[:117] + "..."
                    print(f"         rationale: {rationale}")

            if len(sorted_ideas) > limit:
                print(f"    ... +{len(sorted_ideas) - limit} more")


def print_summary(grouped: Dict[str, Dict[str, List]]):
    """Print a summary table of partitions and category counts."""
    print(f"\n{'═' * 80}")
    print(f"SUMMARY")
    print(f"{'═' * 80}")

    total_ideas = 0
    total_assigned = 0

    for partition in sorted(grouped.keys()):
        categories = grouped[partition]
        n_ideas = sum(len(ideas) for ideas in categories.values())
        n_assigned = sum(
            len([i for i in ideas if i.assigned_category])
            for ideas in categories.values()
        )
        n_cats = len(categories)
        avg_conf = (
            sum(
                i.category_confidence or 0
                for ideas in categories.values()
                for i in ideas
                if i.category_confidence
            ) / max(n_assigned, 1)
        )

        total_ideas += n_ideas
        total_assigned += n_assigned

        print(f"  {partition:45s}  {n_ideas:4d} ideas  {n_cats:3d} categories  avg conf: {avg_conf:.2f}")

    print(f"  {'─' * 70}")
    print(f"  {'TOTAL':45s}  {total_ideas:4d} ideas  assigned: {total_assigned}/{total_ideas}")


# ---------------------------------------------------------------------------
# Category-grouped display (GROUP_BY = "category")
# ---------------------------------------------------------------------------

def _cat_group_sort_key(item):
    """Sort by parent theme, then by count descending."""
    category, ideas = item
    parent = ideas[0].parent_category or ""
    return (parent, -len(ideas))


def print_category_idea_mode(cat_grouped: Dict[str, List]):
    """Compact view grouped by category across all partitions."""
    sorted_cats = sorted(cat_grouped.items(), key=_cat_group_sort_key)

    current_parent = None
    for category, ideas in sorted_cats:
        parent = ideas[0].parent_category or ""
        avg_conf = sum(i.category_confidence or 0 for i in ideas) / len(ideas)

        # Print parent header when it changes
        if parent != current_parent:
            current_parent = parent
            if parent:
                print(f"\n{'═' * 80}")
                print(f"THEME: {parent}")
                print(f"{'═' * 80}")
            else:
                print(f"\n{'═' * 80}")
                print(f"(no parent theme)")
                print(f"{'═' * 80}")

        # Partition distribution
        partition_counts = defaultdict(int)
        for idea in ideas:
            p = (idea.partition_name or idea.domain or "?").strip()
            partition_counts[p] += 1
        partition_str = ", ".join(
            f"{p}: {c}" for p, c in sorted(partition_counts.items(), key=lambda x: -x[1])
        )

        print(f"\n  {category} ({len(ideas)} ideas, avg conf: {avg_conf:.2f})")
        print(f"  partitions: {partition_str}")
        print(f"  {'─' * 70}")

        sorted_ideas = sorted(ideas, key=lambda i: i.category_confidence or 0, reverse=True)
        limit = MAX_PER_CATEGORY or len(sorted_ideas)
        for idx, idea in enumerate(sorted_ideas[:limit], 1):
            conf = idea.category_confidence or 0
            text = idea.idea or ""
            if len(text) > 70:
                text = text[:67] + "..."
            valence = idea.valence or "0"
            part = (idea.partition_name or "?")[:20]
            print(f"    {idx:3d}. [{conf:.2f}] ({valence}) \"{text}\"  [{part}]")

        if len(sorted_ideas) > limit:
            print(f"    ... +{len(sorted_ideas) - limit} more")


def print_category_ladder_mode(cat_grouped: Dict[str, List]):
    """Detailed ladder view grouped by category across all partitions."""
    sorted_cats = sorted(cat_grouped.items(), key=_cat_group_sort_key)

    current_parent = None
    for category, ideas in sorted_cats:
        parent = ideas[0].parent_category or ""
        avg_conf = sum(i.category_confidence or 0 for i in ideas) / len(ideas)

        if parent != current_parent:
            current_parent = parent
            if parent:
                print(f"\n{'═' * 80}")
                print(f"THEME: {parent}")
                print(f"{'═' * 80}")
            else:
                print(f"\n{'═' * 80}")
                print(f"(no parent theme)")
                print(f"{'═' * 80}")

        partition_counts = defaultdict(int)
        for idea in ideas:
            p = (idea.partition_name or idea.domain or "?").strip()
            partition_counts[p] += 1
        partition_str = ", ".join(
            f"{p}: {c}" for p, c in sorted(partition_counts.items(), key=lambda x: -x[1])
        )

        print(f"\n  {category} ({len(ideas)} ideas, avg conf: {avg_conf:.2f})")
        print(f"  partitions: {partition_str}")
        print(f"  {'─' * 70}")

        sorted_ideas = sorted(ideas, key=lambda i: i.category_confidence or 0, reverse=True)
        limit = MAX_PER_CATEGORY or len(sorted_ideas)
        for idx, idea in enumerate(sorted_ideas[:limit], 1):
            conf = idea.category_confidence or 0
            valence = idea.valence or "0"
            part = (idea.partition_name or "?")[:25]
            print(f"    {idx:3d}. [{conf:.2f}] ({idea.idea_id}) valence={valence}  [{part}]")
            print(f"         instance: \"{idea.instance or ''}\"")
            print(f"         rung_1:   {idea.rung_1 or ''}")
            print(f"         rung_2:   {idea.rung_2 or ''}")
            if idea.category_rationale:
                rationale = idea.category_rationale
                if len(rationale) > 120:
                    rationale = rationale[:117] + "..."
                print(f"         rationale: {rationale}")

        if len(sorted_ideas) > limit:
            print(f"    ... +{len(sorted_ideas) - limit} more")


def print_category_summary(cat_grouped: Dict[str, List]):
    """Summary table for category-grouped view."""
    print(f"\n{'═' * 80}")
    print(f"SUMMARY BY CATEGORY")
    print(f"{'═' * 80}")

    total = 0
    sorted_cats = sorted(cat_grouped.items(), key=_cat_group_sort_key)

    current_parent = None
    for category, ideas in sorted_cats:
        parent = ideas[0].parent_category or ""
        if parent != current_parent:
            current_parent = parent
            if parent:
                print(f"\n  [{parent}]")

        avg_conf = sum(i.category_confidence or 0 for i in ideas) / len(ideas)
        print(f"    {category:50s}  {len(ideas):4d} ideas  avg conf: {avg_conf:.2f}")
        total += len(ideas)

    print(f"\n  {'─' * 70}")
    print(f"  {'TOTAL':54s}  {total:4d} ideas")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print(f"VIEW ASSIGNMENTS  (mode: {VIEW_MODE}, group by: {GROUP_BY})")
    print("=" * 80)
    print(f"Dataset:     {FILENAME}")
    print(f"Variable:    {VARIABLE}")
    print(f"Sample size: {SAMPLE_SIZE}")
    if MAX_PER_CATEGORY:
        print(f"Max display: {MAX_PER_CATEGORY} per category")
    print()

    data = load_assignments()

    if GROUP_BY == "category":
        cat_grouped = group_by_category(data)
        print_category_summary(cat_grouped)
        if VIEW_MODE == "ladder":
            print_category_ladder_mode(cat_grouped)
        else:
            print_category_idea_mode(cat_grouped)
    else:
        grouped = group_by_partition(data)
        print_summary(grouped)
        if VIEW_MODE == "ladder":
            print_ladder_mode(grouped)
        else:
            print_idea_mode(grouped)


if __name__ == "__main__":
    main()

# %%
