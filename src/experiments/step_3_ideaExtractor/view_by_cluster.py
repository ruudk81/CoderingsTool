#%%
#
"""
View Step 3 results organized by concept type.
Displays all ideas grouped by concept_type, showing: idea, instance, concept, concept_type_definition, valence.

Usage:
    cd src && python -m "experiments.step_3_ideaExtractor v4.view_by_cluster"
"""

import sys
from pathlib import Path
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

import re
from collections import defaultdict
# v4 uses local models with primary_facet/concept_type fields
try:
    from experiments.step_3_ideaExtractor_v4 import models_exp_v3 as models
except ImportError:
    models_v4_dir = Path(__file__).parent
    if str(models_v4_dir) not in sys.path:
        sys.path.insert(0, str(models_v4_dir))
    import models_exp_v3 as models
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

    # Load extracted ideas
    encoded_text = cache_manager.load_from_cache(
        FILENAME, "extracted_ideas", variable_key, models.IdeasExtractedModel
    )

    print(f"Loaded {len(encoded_text)} responses")

    # Collect all ideas and group by concept_type
    groups = defaultdict(list)
    for item in encoded_text:
        if item.response_ideas:
            for idea in item.response_ideas:
                groups[idea.concept_type].append(idea)

    total_ideas = sum(len(v) for v in groups.values())
    print(f"Total ideas: {total_ideas}")

    # Display each group
    for ct in sorted(groups.keys()):
        ideas = groups[ct]
        ideas.sort(key=lambda i: (i.concept or "", i.valence or ""))

        print("\n" + "=" * 60)
        print(f"{ct.upper()} ({len(ideas)} ideas)")
        print("=" * 60)
        for idea in ideas:
            valence_str = f" [{idea.valence}]" if idea.valence else ""
            ftd_str = f" ({idea.concept_type_definition})" if idea.concept_type_definition else ""
            print(f"- {clean_idea(idea.idea)} | {idea.concept}{ftd_str}{valence_str}")


if __name__ == "__main__":
    main()
