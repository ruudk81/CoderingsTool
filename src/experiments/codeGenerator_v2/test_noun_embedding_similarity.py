"""
Test script to measure cosine similarity between Dutch noun/phrase embeddings.

This helps determine if lemmatized noun matching is viable for
idea → sub-theme assignment.

Usage:
    cd src && python -m experiments.codeGenerator_v2.test_noun_embedding_similarity
"""

import os
import sys

# Ensure src directory is in path
src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

from config import OPENAI_API_KEY, DEFAULT_CODEDESIGNER_CONFIG


def get_embeddings(texts: list[str], model: str = None) -> list[np.ndarray]:
    """Get embeddings for a list of texts."""
    if model is None:
        model = DEFAULT_CODEDESIGNER_CONFIG.embedding_model

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(input=texts, model=model)
    return [np.array(item.embedding) for item in response.data]


def test_similarity_pairs():
    """Test cosine similarity for various Dutch word/phrase pairs."""

    # Define test pairs with expected relationship
    test_pairs = [
        # Category: Subset/Type relationships (should be high)
        ("verpakking", "plastic verpakking", "subset"),
        ("verpakking", "kartonnen verpakking", "subset"),
        ("smaak", "goede smaak", "subset"),
        ("prijs", "lage prijs", "subset"),

        # Category: Synonyms/Near-synonyms (should be moderate-high)
        ("verpakking", "doos", "synonym"),
        ("verminderen", "minder", "synonym"),
        ("verbeteren", "beter maken", "synonym"),
        ("gezond", "gezondheid", "word-form"),

        # Category: Same domain, different concepts (should be moderate)
        ("verpakking", "smaak", "same-domain"),
        ("prijs", "kwaliteit", "same-domain"),
        ("gezondheid", "smaak", "same-domain"),
        ("plastic", "karton", "same-domain"),

        # Category: Related but indirect (should be low-moderate)
        ("gezondheid", "minder zout", "indirect"),
        ("milieu", "plastic verminderen", "indirect"),
        ("versheid", "houdbaarheidsdatum", "indirect"),

        # Category: Unrelated (should be low)
        ("verpakking", "televisie", "unrelated"),
        ("smaak", "politiek", "unrelated"),
        ("prijs", "weer", "unrelated"),

        # Category: Theme labels vs idea nouns (realistic test cases)
        ("verpakking verminderen", "plastic verpakking", "theme-vs-idea"),
        ("smaak verbeteren", "smaak kruiden", "theme-vs-idea"),
        ("gezondere samenstelling", "zout suiker", "theme-vs-idea"),
        ("prijzen verlagen", "prijs kosten", "theme-vs-idea"),
        ("variatie vergroten", "keuze assortiment", "theme-vs-idea"),
    ]

    # Extract all unique texts for batch embedding
    all_texts = list(set([t[0] for t in test_pairs] + [t[1] for t in test_pairs]))

    print(f"Embedding {len(all_texts)} unique texts...")
    print(f"Model: {DEFAULT_CODEDESIGNER_CONFIG.embedding_model}\n")

    embeddings = get_embeddings(all_texts)
    text_to_embedding = {text: emb for text, emb in zip(all_texts, embeddings)}

    # Calculate and display results by category
    results_by_category = {}

    for text1, text2, category in test_pairs:
        emb1 = text_to_embedding[text1].reshape(1, -1)
        emb2 = text_to_embedding[text2].reshape(1, -1)
        sim = cosine_similarity(emb1, emb2)[0, 0]

        if category not in results_by_category:
            results_by_category[category] = []
        results_by_category[category].append((text1, text2, sim))

    # Print results grouped by category
    print("=" * 80)
    print("COSINE SIMILARITY TEST RESULTS")
    print("=" * 80)

    category_order = ["subset", "synonym", "word-form", "same-domain",
                      "indirect", "unrelated", "theme-vs-idea"]

    for category in category_order:
        if category not in results_by_category:
            continue

        pairs = results_by_category[category]
        avg_sim = np.mean([p[2] for p in pairs])

        print(f"\n{category.upper()} (avg: {avg_sim:.3f})")
        print("-" * 60)

        for text1, text2, sim in sorted(pairs, key=lambda x: -x[2]):
            # Color coding based on similarity
            if sim >= 0.8:
                indicator = "+++"
            elif sim >= 0.6:
                indicator = "++"
            elif sim >= 0.4:
                indicator = "+"
            elif sim >= 0.2:
                indicator = "~"
            else:
                indicator = "-"

            print(f"  {sim:.3f} {indicator}  '{text1}' <-> '{text2}'")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\nCategory averages:")
    for category in category_order:
        if category in results_by_category:
            avg = np.mean([p[2] for p in results_by_category[category]])
            print(f"  {category:20s}: {avg:.3f}")

    # Discrimination analysis
    print("\nDiscrimination analysis:")
    subset_avg = np.mean([p[2] for p in results_by_category.get("subset", [(0,0,0)])])
    unrelated_avg = np.mean([p[2] for p in results_by_category.get("unrelated", [(0,0,0)])])
    theme_avg = np.mean([p[2] for p in results_by_category.get("theme-vs-idea", [(0,0,0)])])

    print(f"  Subset avg - Unrelated avg = {subset_avg - unrelated_avg:.3f} (discrimination gap)")
    print(f"  Theme-vs-idea avg = {theme_avg:.3f}")

    if subset_avg - unrelated_avg > 0.3:
        print("\n  -> Good discrimination: subset vs unrelated differ by >0.3")
    else:
        print("\n  -> Weak discrimination: subset vs unrelated differ by <0.3")


if __name__ == "__main__":
    test_similarity_pairs()
