#%%

"""
View raw (pre-P9) attribute assignments: inspect which attribute each idea was
assigned to BEFORE cross-facet consolidation.

Groups ideas by domain → facet → raw attribute, showing the abstraction ladder for each.
Compare with view_assignments_attributes_consolidated.py to see P9 remap effects.
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
    TaxonomyResultsCache,
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

def load_ideas_with_raw_attributes(
    filename: str = FILENAME,
    variable: str = VARIABLE,
    sample_size: Optional[int] = SAMPLE_SIZE,
) -> List[TaxonomyClassifiedSubmodel]:
    """Load ideas and override their attribute with raw (pre-P9) assignments.

    Reads raw_attribute_assignments from the metadata cache and applies them
    to the growing model ideas (which normally carry post-P9 attributes).
    """
    variable_key = generate_enhanced_variable_key(
        selected_variables=[variable],
        is_merged=False,
        sample_size=sample_size,
    )
    cache_manager = CacheManager()

    # Load metadata cache for raw attribute assignments
    taxonomy_cache = cache_manager.load_metadata_from_cache(
        filename, "taxonomy", variable_key, TaxonomyResultsCache
    )
    if not taxonomy_cache:
        raise FileNotFoundError(
            f"No taxonomy metadata found for variable_key '{variable_key}'.\n"
            f"Run the taxonomy pipeline first."
        )

    # Build raw attribute lookup and confidence lookup from metadata
    raw_attr_lookup: Dict[str, str] = {}
    attr_conf_lookup: Dict[str, float] = {}
    has_raw_data = False
    for domain_result in taxonomy_cache.partition_results.values():
        if domain_result.raw_attribute_assignments:
            has_raw_data = True
        raw_attr_lookup.update(domain_result.raw_attribute_assignments)
        attr_conf_lookup.update(domain_result.attribute_confidence)

    if not has_raw_data:
        print("WARNING: No raw P8 data found in cache — re-run the pipeline to populate.")
        print("         Falling back to consolidated (post-P9) attributes.\n")

    # Load growing model for idea text
    data = cache_manager.load_from_cache(
        filename, "taxonomy_classified", variable_key, TaxonomyClassifiedModel
    )
    if not data:
        raise FileNotFoundError(
            f"No cached results found for variable_key '{variable_key}'.\n"
            f"Run at least taxonomy (P1-P9) first."
        )

    ideas = []
    for resp in data:
        if resp.response_ideas:
            ideas.extend(resp.response_ideas)

    # Override attribute with raw (pre-P9) value where available, and apply confidence
    if has_raw_data:
        for idea in ideas:
            raw_attr = raw_attr_lookup.get(idea.idea_id)
            if raw_attr is not None:
                idea.attribute = raw_attr
            conf = attr_conf_lookup.get(idea.idea_id)
            if conf is not None:
                idea.attribute_confidence = conf

    total = len(ideas)
    with_attr = sum(1 for i in ideas if i.attribute and i.attribute.strip())
    print(f"Loaded {total} ideas ({with_attr} with raw attribute assignments)")
    return ideas


# =============================================================================
# DISPLAY
# =============================================================================

def print_by_attribute(ideas: List[TaxonomyClassifiedSubmodel], max_per_attribute: Optional[int] = None):
    """Print ideas grouped by raw attribute (flat list, sorted by count)."""
    attr_groups: Dict[str, List[TaxonomyClassifiedSubmodel]] = defaultdict(list)
    for idea in ideas:
        attr = idea.attribute or "(no attribute)"
        attr_groups[attr].append(idea)

    sorted_attrs = sorted(attr_groups.items(), key=lambda x: -len(x[1]))

    total = len(ideas)
    print(f"\n{'='*80}")
    print(f"RAW ATTRIBUTE ASSIGNMENTS — pre-P9 ({total} ideas, {len(sorted_attrs)} attributes)")
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

            conf_str = f"  conf: {idea.attribute_confidence:.2f}" if idea.attribute_confidence is not None else ""
            print(f"\n  • Idea: {idea.idea_id} — \"{instance}\"{conf_str}")
            print(f"    Ladder: {instance} → {interpretation} → {abstraction} [{valence}]")
            print(f"    Taxonomy: {domain} > {facet} > {attr_name}")

        if max_per_attribute and len(attr_ideas) > max_per_attribute:
            print(f"\n    ... ({len(attr_ideas) - max_per_attribute} more ideas)")


def print_by_domain(ideas: List[TaxonomyClassifiedSubmodel], max_per_attribute: Optional[int] = None):
    """Print ideas grouped by domain → facet → raw attribute."""
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
    print(f"RAW ATTRIBUTE ASSIGNMENTS BY DOMAIN — pre-P9 ({total} ideas)")
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
                    conf_str = f"  conf: {idea.attribute_confidence:.2f}" if idea.attribute_confidence is not None else ""
                    print(f"      • {idea.idea_id}: \"{instance}\" → {interpretation} [{valence}]{conf_str}")

                if max_per_attribute and len(attr_ideas) > max_per_attribute:
                    print(f"      ... ({len(attr_ideas) - max_per_attribute} more)")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    ideas = load_ideas_with_raw_attributes()

    if GROUP_BY == "domain":
        print_by_domain(ideas, max_per_attribute=MAX_PER_ATTRIBUTE)
    else:
        print_by_attribute(ideas, max_per_attribute=MAX_PER_ATTRIBUTE)
