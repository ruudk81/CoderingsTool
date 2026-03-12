#%%
#
"""
Debug script for Step 3: Idea Extractor
Loads cached results and prints sample responses with extracted ideas.

Usage:
    cd src && python -m experiments.step_3_ideaExtractor.debug_samples
"""

import sys
from pathlib import Path
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

import random
import re
try:
    from experiments.step_3_ideaExtractor import models_exp as models
except ImportError:
    models_dir = Path(__file__).parent
    if str(models_dir) not in sys.path:
        sys.path.insert(0, str(models_dir))
    import models_exp as models
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
N_SAMPLES = 5


def clean_idea(idea: str) -> str:
    """Remove brackets and normalize whitespace."""
    cleaned = re.sub(r"\[.*?\]", "", idea)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def print_extraction_metadata(cache_manager, filename, variable_key):
    """Load and print extraction metadata (dimension, domains, context specifiers)."""
    metadata = cache_manager.load_metadata_from_cache(
        filename, "extracted_ideas", variable_key, models.ExtractionMetadata
    )

    if metadata is None:
        print("No extraction metadata found in cache.")
        return

    print("=" * 70)
    print("EXTRACTION METADATA")
    print("=" * 70)

    # Context specifiers
    print("\n[Context Specifiers]")
    print(f"  Language:    {metadata.lang or '(not set)'}")
    print(f"  Sector:      {metadata.sector or '(not set)'}")
    print(f"  Topic:       {metadata.topic or '(not set)'}")
    print(f"  Perspective: {metadata.perspective or '(not set)'}")
    print(f"  Entity:      {metadata.entity or '(not set)'}")
    print(f"  Intent:      {metadata.intent or '(not set)'}")

    # Template
    if metadata.template_prefix:
        print(f"\n[Template Prefix]")
        print(f"  \"{metadata.template_prefix}\"")

    # Primary Dimension
    print("\n[Primary Dimension]")
    print(f"  Dimension:             {metadata.primary_dimension or '(not set)'}")
    print(f"  Dimension description: {metadata.primary_dimension_description or '(not set)'}")

    # Domains
    if metadata.domains:
        print("\n[Domains]")
        for d in metadata.domains:
            print(f"  {d['key']}: {d['label']} — {d['definition']}")

    print()


def print_idea_details(idea: models.IdeasExtractedSubmodel, indent: str = "  "):
    """Print all details for a single idea."""
    cleaned = clean_idea(idea.idea)
    print(f"{indent}Idea: {cleaned}")

    # Abstraction ladder (bottom-up: instance → interpretation → abstraction)
    ladder_parts = [v for v in (idea.instance, idea.interpretation, idea.abstraction) if v]
    if ladder_parts:
        print(f"{indent}  ladder: {' → '.join(ladder_parts)}")
    if idea.domain:
        print(f"{indent}  domain: {idea.domain}")
    if idea.valence:
        print(f"{indent}  valence: {idea.valence}")


def main():
    variable_key = generate_enhanced_variable_key([VAR_NAME], False, SAMPLE_SIZE)
    cache_manager = CacheManager(CacheConfig())

    # Load and print extraction metadata first
    print_extraction_metadata(cache_manager, FILENAME, variable_key)

    # Load extracted ideas
    encoded_text = cache_manager.load_from_cache(
        FILENAME, "extracted_ideas", variable_key, models.IdeasExtractedModel
    )

    print(f"Loaded {len(encoded_text)} responses with ideas")

    # Count total ideas
    total_ideas = sum(item.idea_count for item in encoded_text)
    print(f"Total ideas: {total_ideas}")
    print(f"Average ideas per response: {total_ideas / len(encoded_text):.2f}")

    # Sample and display
    print("\n" + "=" * 70)
    print("SAMPLE RESPONSES WITH EXTRACTED IDEAS")
    print("=" * 70)

    samples = random.sample(encoded_text, min(N_SAMPLES, len(encoded_text)))

    for item in samples:
        print(f"\n--- Response (ID: {item.respondent_id}) ---")
        print(f"Original: {item.response}")
        if item.template_prefix:
            print(f"Template: \"{item.template_prefix}\"")
        print(f"\nIdeas ({item.idea_count}):")

        if item.response_ideas:
            for idea in item.response_ideas:
                print_idea_details(idea)
                print()

        print("-" * 70)


if __name__ == "__main__":
    main()
