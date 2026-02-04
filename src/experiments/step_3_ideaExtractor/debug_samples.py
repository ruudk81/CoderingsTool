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
import models
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
    """Load and print extraction metadata (taxonomy, context specifiers)."""
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
    print(f"  Domain:      {metadata.domain or '(not set)'}")
    print(f"  Topic:       {metadata.topic or '(not set)'}")
    print(f"  Perspective: {metadata.perspective or '(not set)'}")
    print(f"  Entity:      {metadata.entity or '(not set)'}")
    print(f"  Intent:      {metadata.intent or '(not set)'}")

    # Template
    if metadata.template_prefix:
        print(f"\n[Template Prefix]")
        print(f"  \"{metadata.template_prefix}\"")

    # Taxonomy
    print("\n[Taxonomy]")
    print(f"  Primary axis:      {metadata.taxonomy_primary_axis or '(not set)'}")
    if metadata.taxonomy_secondary_axis:
        print(f"  Secondary axis:    {metadata.taxonomy_secondary_axis}")
    print(f"  Actionable type:   {metadata.taxonomy_actionable_type or '(not set)'}")
    print(f"  Axis description:  {metadata.taxonomy_axis_description or '(not set)'}")
    if metadata.taxonomy_rationale:
        print(f"  Rationale:         {metadata.taxonomy_rationale}")
    if metadata.taxonomy_sample_phrases:
        print(f"  Sample phrases:    {', '.join(metadata.taxonomy_sample_phrases)}")

    print()


def print_idea_details(idea: models.IdeasExtractedSubmodel, indent: str = "  "):
    """Print all details for a single idea."""
    cleaned = clean_idea(idea.idea)
    print(f"{indent}Idea: {cleaned}")

    # Taxonomy phrase
    if idea.taxonomy_phrase:
        print(f"{indent}  taxonomy_phrase: \"{idea.taxonomy_phrase}\"")

    # Sentiment and sense
    print(f"{indent}  sentiment: {idea.sentiment}  |  sense: {idea.sense}")

    # Ontology (if present)
    if idea.ontology:
        ont = idea.ontology
        ontology_str = " → ".join(filter(None, [ont.instance, ont.node, ont.category, ont.root]))
        if ontology_str:
            print(f"{indent}  ontology: {ontology_str}")


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

    # Sentiment/sense distribution
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
    sense_counts = {"factual": 0, "evaluative": 0, "aspirational": 0, "experiential": 0}

    for item in encoded_text:
        if item.response_ideas:
            for idea in item.response_ideas:
                sentiment_counts[idea.sentiment] = sentiment_counts.get(idea.sentiment, 0) + 1
                sense_counts[idea.sense] = sense_counts.get(idea.sense, 0) + 1

    print(f"\nSentiment distribution: {sentiment_counts}")
    print(f"Sense distribution: {sense_counts}")

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
