#%%

"""
View facet assignments: inspect which facet each idea was assigned to.

Groups ideas by domain → facet, showing the abstraction ladder and attribute for each.
"""

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "src" / "steps"))

from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from models import (
    TaxonomyClassifiedModel, TaxonomyClassifiedSubmodel,
)

from test_data import TEST_DATA


# =============================================================================
# CONFIGURATION
# =============================================================================

FILENAME = TEST_DATA.filename
VARIABLE = TEST_DATA.var_name
SAMPLE_SIZE = TEST_DATA.sample_size

MAX_PER_FACET = None  # None for all, or N to limit ideas per facet
GROUP_BY = "domain"   # "facet" (flat) or "domain" (domain → facet)


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
    populated by assignment, refinement and cross-domain.
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
            f"Run step 4 (taxonomy) first."
        )

    ideas = []
    for resp in data:
        if resp.response_ideas:
            ideas.extend(resp.response_ideas)

    total = len(ideas)
    with_facet = sum(1 for i in ideas if i.facet and i.facet.strip())
    print(f"Loaded {total} ideas ({with_facet} with facet assignments)")
    return ideas


# =============================================================================
# DISPLAY
# =============================================================================

def print_by_facet(ideas: List[TaxonomyClassifiedSubmodel], max_per_facet: Optional[int] = None):
    """Print ideas grouped by facet (flat list, sorted by count)."""
    facet_groups: Dict[str, List[TaxonomyClassifiedSubmodel]] = defaultdict(list)
    for idea in ideas:
        facet = idea.facet or "(no facet)"
        facet_groups[facet].append(idea)

    sorted_facets = sorted(facet_groups.items(), key=lambda x: -len(x[1]))

    total = len(ideas)
    print(f"\n{'='*80}")
    print(f"FACET ASSIGNMENTS ({total} ideas, {len(sorted_facets)} facets)")
    print(f"{'='*80}")

    for facet_name, facet_ideas in sorted_facets:
        print(f"\n{'─'*80}")
        print(f"FACET: {facet_name} — {len(facet_ideas)} ideas")
        print(f"{'─'*80}")

        display = facet_ideas[:max_per_facet] if max_per_facet else facet_ideas
        for idea in display:
            instance = idea.instance or ""
            interpretation = idea.interpretation or ""
            valence = idea.valence or "0"
            domain = idea.domain or ""
            attribute = idea.attribute or ""

            conf_str = f"  conf: {idea.facet_confidence:.2f}" if idea.facet_confidence is not None else ""
            print(f"\n  • {idea.idea_id}: \"{instance}\" → {interpretation} [{valence}]{conf_str}")
            print(f"    Taxonomy: {domain} > {facet_name} > {attribute}")

        if max_per_facet and len(facet_ideas) > max_per_facet:
            print(f"\n    ... ({len(facet_ideas) - max_per_facet} more ideas)")


def print_by_domain(ideas: List[TaxonomyClassifiedSubmodel], max_per_facet: Optional[int] = None):
    """Print ideas grouped by domain → facet."""
    hierarchy: Dict[str, Dict[str, List[TaxonomyClassifiedSubmodel]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for idea in ideas:
        domain = idea.domain or "(unknown)"
        facet = idea.facet or "(no facet)"
        hierarchy[domain][facet].append(idea)

    total = len(ideas)
    print(f"\n{'='*80}")
    print(f"FACET ASSIGNMENTS BY DOMAIN ({total} ideas)")
    print(f"{'='*80}")

    for domain in sorted(hierarchy.keys()):
        facets = hierarchy[domain]
        domain_count = sum(len(ids) for ids in facets.values())

        print(f"\n{'='*80}")
        print(f"DOMAIN: {domain} — {domain_count} ideas")
        print(f"{'='*80}")

        for facet_name in sorted(facets.keys(), key=lambda f: -len(facets[f])):
            facet_ideas = facets[facet_name]
            print(f"\n  FACET: {facet_name} — {len(facet_ideas)} ideas")

            display = facet_ideas[:max_per_facet] if max_per_facet else facet_ideas
            for idea in display:
                instance = idea.instance or ""
                interpretation = idea.interpretation or ""
                valence = idea.valence or "0"
                attribute = idea.attribute or ""
                conf_str = f"  conf: {idea.facet_confidence:.2f}" if idea.facet_confidence is not None else ""
                print(f"    • {idea.idea_id}: \"{instance}\" → {interpretation} [{valence}]  attr: {attribute}{conf_str}")

            if max_per_facet and len(facet_ideas) > max_per_facet:
                print(f"    ... ({len(facet_ideas) - max_per_facet} more)")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    ideas = load_ideas()

    if GROUP_BY == "domain":
        print_by_domain(ideas, max_per_facet=MAX_PER_FACET)
    else:
        print_by_facet(ideas, max_per_facet=MAX_PER_FACET)
