#%%

"""
View domain assignments: inspect which domain (L2) each idea was assigned to.

Groups ideas by domain, showing the full abstraction ladder
(instance → interpretation → abstraction) and valence for each idea, so the
content fit of the domain partitioning can be judged by eye.

Modeled on step_4_classifier/view_assignments_facets.py.
"""

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from utils.cacheManager import CacheManager, generate_enhanced_variable_key
import models

from test_data import TEST_DATA


# =============================================================================
# CONFIGURATION
# =============================================================================

FILENAME = TEST_DATA.filename
VARIABLE = TEST_DATA.var_name
SAMPLE_SIZE = TEST_DATA.sample_size

MAX_PER_DOMAIN = None  # None for all, or N to limit ideas per domain


# =============================================================================
# DATA LOADING
# =============================================================================

def load_ideas(
    filename: str = FILENAME,
    variable: str = VARIABLE,
    sample_size: Optional[int] = SAMPLE_SIZE,
) -> List[models.IdeasExtractedSubmodel]:
    """Load ideas from step 3 growing model (extracted_ideas cache).

    The growing model contains per-idea domain (L2) and the abstraction ladder
    (instance, interpretation, abstraction) populated by step 3 extraction.
    """
    variable_key = generate_enhanced_variable_key(
        selected_variables=[variable],
        is_merged=False,
        sample_size=sample_size,
    )
    cache_manager = CacheManager()

    data = cache_manager.load_from_cache(
        filename, "extracted_ideas", variable_key, models.IdeasExtractedModel
    )

    if not data:
        raise FileNotFoundError(
            f"No cached results found for variable_key '{variable_key}'.\n"
            f"Run step 3 (idea extraction) first."
        )

    ideas = []
    for resp in data:
        if resp.response_ideas:
            ideas.extend(resp.response_ideas)

    total = len(ideas)
    with_domain = sum(1 for i in ideas if i.domain and i.domain.strip())
    print(f"Loaded {total} ideas ({with_domain} with domain assignments)")
    return ideas


def load_domain_definitions(
    filename: str = FILENAME,
    variable: str = VARIABLE,
    sample_size: Optional[int] = SAMPLE_SIZE,
) -> Dict[str, str]:
    """Load {domain_label: definition} from step 3 extraction metadata (best effort)."""
    variable_key = generate_enhanced_variable_key(
        selected_variables=[variable],
        is_merged=False,
        sample_size=sample_size,
    )
    cache_manager = CacheManager()
    try:
        meta = cache_manager.load_metadata_from_cache(
            filename, "extracted_ideas", variable_key, models.ExtractionMetadata
        )
    except Exception:
        return {}
    if not meta or not getattr(meta, "domains", None):
        return {}
    return {d.get("label", ""): d.get("definition", "") for d in meta.domains}


# =============================================================================
# DISPLAY
# =============================================================================

def print_by_domain(
    ideas: List[models.IdeasExtractedSubmodel],
    domain_defs: Optional[Dict[str, str]] = None,
    max_per_domain: Optional[int] = None,
):
    """Print ideas grouped by domain, showing the full abstraction ladder."""
    domain_defs = domain_defs or {}
    domain_groups: Dict[str, List[models.IdeasExtractedSubmodel]] = defaultdict(list)
    for idea in ideas:
        domain = idea.domain or "(no domain)"
        domain_groups[domain].append(idea)

    sorted_domains = sorted(domain_groups.items(), key=lambda x: -len(x[1]))

    total = len(ideas)
    print(f"\n{'='*80}")
    print(f"DOMAIN ASSIGNMENTS ({total} ideas, {len(sorted_domains)} domains)")
    print(f"{'='*80}")

    for domain_name, domain_ideas in sorted_domains:
        pct = 100 * len(domain_ideas) / total if total else 0
        print(f"\n{'='*80}")
        print(f"DOMAIN: {domain_name} — {len(domain_ideas)} ideas ({pct:.1f}%)")
        definition = domain_defs.get(domain_name)
        if definition:
            print(f"  ↳ {definition}")
        print(f"{'='*80}")

        display = domain_ideas[:max_per_domain] if max_per_domain else domain_ideas
        for idea in display:
            instance = idea.instance or ""
            interpretation = idea.interpretation or ""
            abstraction = idea.abstraction or ""
            valence = idea.valence or "0"
            print(f"\n  • {idea.idea_id} [{valence}]")
            print(f"      instance:       {instance}")
            print(f"      interpretation: {interpretation}")
            print(f"      abstraction:    {abstraction}")

        if max_per_domain and len(domain_ideas) > max_per_domain:
            print(f"\n    ... ({len(domain_ideas) - max_per_domain} more ideas)")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    ideas = load_ideas()
    domain_defs = load_domain_definitions()
    print_by_domain(ideas, domain_defs=domain_defs, max_per_domain=MAX_PER_DOMAIN)
