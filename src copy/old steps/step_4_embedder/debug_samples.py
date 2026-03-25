"""
Debug script for Step 4: Embedder
Loads cached results and prints sample responses with embedding info.

Usage:
    cd src && python -m development.step_4_embedder.debug_samples
"""

import sys
from pathlib import Path
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

import random
import re
from development import models_exp as models
from config import CacheConfig
from utils.cacheManager import CacheManager, generate_enhanced_variable_key

# Configuration
FILENAME = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
VAR_NAME = "Q20"
SAMPLE_SIZE = 500
N_SAMPLES = 3


def clean_idea(idea: str) -> str:
    """Remove brackets and normalize whitespace."""
    cleaned = re.sub(r"\[.*?\]", "", idea)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def main():
    variable_key = generate_enhanced_variable_key([VAR_NAME], False, SAMPLE_SIZE)
    cache_manager = CacheManager(CacheConfig())

    # Load embedded data
    embedded_text = cache_manager.load_from_cache(
        FILENAME, "embeddings", variable_key, models.EmbeddingsModel
    )

    print(f"Loaded {len(embedded_text)} responses with embeddings")

    # Count embeddings
    total_embeddings = sum(
        1 for resp in embedded_text
        if resp.response_ideas
        for idea in resp.response_ideas
        if idea.idea_embedding is not None
    )
    print(f"Total embeddings: {total_embeddings}")

    # Sample and display
    print("\n" + "=" * 70)
    print("SAMPLE RESPONSES WITH EMBEDDINGS")
    print("=" * 70)

    samples = random.sample(embedded_text, min(N_SAMPLES, len(embedded_text)))

    for item in samples:
        print(f"\n--- Response (ID: {item.respondent_id}) ---")
        print(f"Original: {item.response}")
        print(f"\nIdeas with embeddings:")
        for segment in item.response_ideas:
            cleaned = clean_idea(segment.idea)
            has_embedding = segment.idea_embedding is not None
            embedding_dim = len(segment.idea_embedding) if has_embedding else 0
            print(f"  - {cleaned}")
            print(f"    Embedding: {'Yes' if has_embedding else 'No'} (dim: {embedding_dim})")
        print("-" * 70)


if __name__ == "__main__":
    main()
