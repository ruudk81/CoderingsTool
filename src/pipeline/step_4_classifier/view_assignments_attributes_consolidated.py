#%%

"""
View attribute assignments: inspect which attribute each idea was assigned to.

Groups ideas by domain → facet → attribute, showing the abstraction ladder for each.
"""

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "src" / "steps"))

from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from pipeline.step_4_classifier.models_classifier import (
    TaxonomyClassifiedModel, TaxonomyClassifiedSubmodel,
)

from test_data import TEST_DATA


# =============================================================================
# CONFIGURATION
# =============================================================================

FILENAME = TEST_DATA.filename
VARIABLE = TEST_DATA.var_name
SAMPLE_SIZE = TEST_DATA.sample_size

MAX_PER_ATTRIBUTE = None  # None for all, or N to limit ideas per attribute
GROUP_BY = "domain"       # "attribute" (flat) or "domain" (domain → facet → attribute)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_ideas(
    filename: str = FILENAME,
    variable: str = VARIABLE,
    sample_size: Optional[int] = SAMPLE_SIZE,
) -> List[TaxonomyClassifiedSubmodel]:
    """Load ideas from step 4 growing model (taxonomy_classified cache).

    The growing model contains per-idea facet, attribute, and partition_name
    populated by P3/P6/P7.
    """
    variable_key = generate_enhanced_variable_key(
        selected_variables=[variable],
        is_merged=False,
        sample_size=sample_size,
    )
    cache_manager = CacheManager()

    data = cache_manager.load_from_cache(
        filename, "taxonomy_classified", variable_key, TaxonomyClassifiedModel
    )

    if not data:
        raise FileNotFoundError(
            f"No cached results found for variable_key '{variable_key}'.\n"
            f"Run at least taxonomy (P1-P7) first."
        )

    ideas = []
    for resp in data:
        if resp.response_ideas:
            ideas.extend(resp.response_ideas)

    total = len(ideas)
    with_attr = sum(1 for i in ideas if i.attribute and i.attribute.strip())
    print(f"Loaded {total} ideas ({with_attr} with attribute assignments)")
    return ideas


# =============================================================================
# DISPLAY
# =============================================================================

def print_by_attribute(ideas: List[TaxonomyClassifiedSubmodel], max_per_attribute: Optional[int] = None):
    """Print ideas grouped by attribute (flat list, sorted by count)."""
    attr_groups: Dict[str, List[TaxonomyClassifiedSubmodel]] = defaultdict(list)
    for idea in ideas:
        attr = idea.attribute or "(no attribute)"
        attr_groups[attr].append(idea)

    sorted_attrs = sorted(attr_groups.items(), key=lambda x: -len(x[1]))

    total = len(ideas)
    print(f"\n{'='*80}")
    print(f"ATTRIBUTE ASSIGNMENTS ({total} ideas, {len(sorted_attrs)} attributes)")
    print(f"{'='*80}")

    for attr_name, attr_ideas in sorted_attrs:
        print(f"\n{'─'*80}")
        print(f"ATTRIBUTE: {attr_name} — {len(attr_ideas)} ideas")
        print(f"{'─'*80}")

        display = attr_ideas[:max_per_attribute] if max_per_attribute else attr_ideas
        for idea in display:
            instance = idea.instance or ""
            interpretation = idea.interpretation or ""
            abstraction = idea.abstraction or ""
            valence = idea.valence or "0"
            domain = idea.domain or ""
            facet = idea.facet or ""

            print(f"\n  • Idea: {idea.idea_id} — \"{instance}\"")
            print(f"    Ladder: {instance} → {interpretation} → {abstraction} [{valence}]")
            print(f"    Taxonomy: {domain} > {facet} > {attr_name}")

        if max_per_attribute and len(attr_ideas) > max_per_attribute:
            print(f"\n    ... ({len(attr_ideas) - max_per_attribute} more ideas)")


def print_by_domain(ideas: List[TaxonomyClassifiedSubmodel], max_per_attribute: Optional[int] = None):
    """Print ideas grouped by domain → facet → attribute."""
    # Build hierarchy
    hierarchy: Dict[str, Dict[str, Dict[str, List[TaxonomyClassifiedSubmodel]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for idea in ideas:
        domain = idea.domain or "(unknown)"
        facet = idea.facet or "(unknown)"
        attr = idea.attribute or "(no attribute)"
        hierarchy[domain][facet][attr].append(idea)

    total = len(ideas)
    print(f"\n{'='*80}")
    print(f"ATTRIBUTE ASSIGNMENTS BY DOMAIN ({total} ideas)")
    print(f"{'='*80}")

    for domain in sorted(hierarchy.keys()):
        facets = hierarchy[domain]
        domain_count = sum(len(ids) for f in facets.values() for ids in f.values())

        print(f"\n{'='*80}")
        print(f"DOMAIN: {domain} — {domain_count} ideas")
        print(f"{'='*80}")

        for facet in sorted(facets.keys()):
            attrs = facets[facet]
            facet_count = sum(len(ids) for ids in attrs.values())

            print(f"\n  FACET: {facet} — {facet_count} ideas")

            for attr_name in sorted(attrs.keys(), key=lambda a: -len(attrs[a])):
                attr_ideas = attrs[attr_name]
                print(f"\n    ATTRIBUTE: {attr_name} — {len(attr_ideas)} ideas")

                display = attr_ideas[:max_per_attribute] if max_per_attribute else attr_ideas
                for idea in display:
                    instance = idea.instance or ""
                    interpretation = idea.interpretation or ""
                    valence = idea.valence or "0"
                    print(f"      • {idea.idea_id}: \"{instance}\" → {interpretation} [{valence}]")

                if max_per_attribute and len(attr_ideas) > max_per_attribute:
                    print(f"      ... ({len(attr_ideas) - max_per_attribute} more)")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    ideas = load_ideas()

    if GROUP_BY == "domain":
        print_by_domain(ideas, max_per_attribute=MAX_PER_ATTRIBUTE)
    else:
        print_by_attribute(ideas, max_per_attribute=MAX_PER_ATTRIBUTE)
