#%%
#
"""
View Step 3 results organized by domain.
Displays all ideas grouped by domain, showing: idea, facet, valence.

Usage:
    cd src && python -m pipeline.step_3_ideaExtractor.view_by_cluster
"""

import sys
from pathlib import Path
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

import re
from collections import defaultdict
try:
    from pipeline.step_3_ideaExtractor import models
except ImportError:
    models_dir = Path(__file__).parent
    if str(models_dir) not in sys.path:
        sys.path.insert(0, str(models_dir))
    import models
from config import CacheConfig
from utils.cacheManager import CacheManager, generate_enhanced_variable_key

# Import centralized test data config
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

    # Collect all ideas and group by domain
    groups = defaultdict(list)
    for item in encoded_text:
        if item.response_ideas:
            for idea in item.response_ideas:
                groups[idea.domain].append(idea)

    total_ideas = sum(len(v) for v in groups.values())
    print(f"Total ideas: {total_ideas}")

    # Display each group
    for d in sorted(groups.keys()):
        ideas = groups[d]
        ideas.sort(key=lambda i: (i.facet or "", i.valence or ""))

        print("\n" + "=" * 60)
        print(f"{d.upper()} ({len(ideas)} ideas)")
        print("=" * 60)
        for idea in ideas:
            valence_str = f" [{idea.valence}]" if idea.valence else ""
            taxonomy = " → ".join(v for v in (idea.instance, idea.facet) if v)
            print(f"- {taxonomy}{valence_str}")


if __name__ == "__main__":
    main()
