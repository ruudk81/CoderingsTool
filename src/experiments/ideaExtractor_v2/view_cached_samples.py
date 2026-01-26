#%%

"""
View Cached Samples - Load and display samples from cached extraction data.
"""

import os
import sys
import random
import re

# Ensure src directory is in path
src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import models
from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from config import CacheConfig

# Configuration matching run_experiment.py
#FILENAME = "M000000 Associatiemonitor Merk X net databestand.sav"
#VAR_NAME = "Qd1_combined"
#SAMPLE_SIZE = 2000
#N_SAMPLES = 20  # Number of samples to display

FILENAME="M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
VAR_NAME="Q20"
#sample_size=50
SAMPLE_SIZE=500
N_SAMPLES = 20 




def load_cached_ideas():
    """Load extracted ideas from cache."""
    variable_key = generate_enhanced_variable_key(
        selected_variables=[VAR_NAME],
        is_merged=False,
        sample_size=SAMPLE_SIZE
    )

    cache_manager = CacheManager(CacheConfig())
    step = "extracted_ideas"

    if not cache_manager.is_cache_valid(FILENAME, step, variable_key):
        print(f"❌ Cache not found for step '{step}'")
        print(f"   Filename: {FILENAME}")
        print(f"   Variable key: {variable_key}")
        return None

    extracted_ideas = cache_manager.load_from_cache(
        filename=FILENAME,
        step=step,
        variable_key=variable_key,
        model_cls=models.IdeasExtractedModel
    )

    return extracted_ideas


def print_sample_ideas(results, n_samples=20):
    """Print sample outputs from the extraction results."""
    # Filter to responses that have ideas
    responses_with_ideas = [r for r in results if r.response_ideas]

    if not responses_with_ideas:
        print("\n⚠️ No responses with extracted ideas found")
        return

    n_samples = min(n_samples, len(responses_with_ideas))
    samples = random.sample(responses_with_ideas, n_samples)

    print("\n" + "=" * 80)
    print(f"SAMPLE OUTPUTS ({n_samples} responses)")
    print("=" * 80)

    for i, item in enumerate(samples, 1):
        print(f"\n[{i}] Respondent: {item.respondent_id}")
        print(f"    Response: {item.response[:200]}{'...' if len(item.response) > 200 else ''}")
        print(f"    Template prefix: {item.template_prefix or 'N/A'}")
        print(f"    Ideas ({len(item.response_ideas)}):")

        for segment in item.response_ideas:
            idea_text = segment.idea

            # Extract taxonomy metadata from idea text
            axis_match = re.search(r"\[axis=([^\]]+)\]", idea_text)
            taxonomy_phrase_match = re.search(r"\[taxonomy_phrase=([^\]]+)\]", idea_text)
            sentiment_match = re.search(r"\[sentiment=([^\]]+)\]", idea_text)
            sense_match = re.search(r"\[sense=([^\]]+)\]", idea_text)

            # Clean the idea text (remove all metadata tags)
            cleaned = re.sub(r"\[.*?\]", "", idea_text)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()

            # Build metadata display
            metadata_parts = []
            if axis_match:
                metadata_parts.append(f"axis={axis_match.group(1)}")
            if taxonomy_phrase_match:
                metadata_parts.append(f"phrase=\"{taxonomy_phrase_match.group(1)}\"")
            if sentiment_match:
                metadata_parts.append(f"sentiment={sentiment_match.group(1)}")
            if sense_match:
                metadata_parts.append(f"sense={sense_match.group(1)}")

            # Fallback: check object attributes if not in text
            if not sentiment_match and hasattr(segment, 'sentiment') and segment.sentiment:
                metadata_parts.append(f"sentiment={segment.sentiment}")
            if not sense_match and hasattr(segment, 'sense') and segment.sense:
                metadata_parts.append(f"sense={segment.sense}")

            metadata_str = f" ({', '.join(metadata_parts)})" if metadata_parts else ""
            print(f"      [{segment.idea_id}]: {cleaned}{metadata_str}")

    print("\n" + "-" * 80)


if __name__ == "__main__":
    print("Loading cached extraction data...")
    ideas = load_cached_ideas()

    if ideas:
        total_ideas = sum(item.idea_count for item in ideas)
        print(f"\n✅ Loaded {len(ideas)} responses with {total_ideas} total ideas")
        print_sample_ideas(ideas, N_SAMPLES)
    else:
        print("\n❌ Failed to load cached data")
        sys.exit(1)

# %%
