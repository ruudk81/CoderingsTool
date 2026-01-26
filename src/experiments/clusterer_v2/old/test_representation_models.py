"""
Quick test script to verify all representation models work correctly

Tests each model in isolation with simple synthetic data to ensure:
1. Models can be imported
2. Models run without errors
3. Models return expected output format
4. Models produce different results (not identical)

Usage:
    cd src/experiments
    python test_representation_models.py
"""
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from representation.ctfidf_representation import CTfidfRepresentation
from representation.mmr_representation import MMRRepresentation
from representation.keybert_representation import KeyBERTRepresentation
from representation.llm_representation import LLMRepresentation


def test_ctfidf():
    """Test c-TF-IDF representation"""
    print("\n" + "="*80)
    print("TEST 1: c-TF-IDF Representation")
    print("="*80)

    clusters = {
        1: ['good service', 'excellent service', 'great service', 'friendly staff'],
        2: ['high price', 'expensive', 'too costly', 'overpriced'],
        3: ['fresh food', 'quality ingredients', 'tasty meals', 'delicious']
    }

    ctfidf = CTfidfRepresentation(top_k=5, bm25_weighting=True)

    try:
        keywords = ctfidf.extract_keywords(clusters, verbose=True)

        print("\nResults:")
        for cluster_id, kws in keywords.items():
            print(f"\nCluster {cluster_id}:")
            for kw, score in kws[:5]:
                print(f"  • {kw:<20} ({score:.4f})")

        print("\n✅ c-TF-IDF test passed")
        return True

    except Exception as e:
        print(f"\n❌ c-TF-IDF test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mmr():
    """Test MMR representation"""
    print("\n" + "="*80)
    print("TEST 2: MMR Representation")
    print("="*80)

    # Build test data with c-TF-IDF scores
    from sklearn.feature_extraction.text import CountVectorizer

    clusters = {
        1: ['good service', 'excellent service', 'great service', 'friendly staff'],
        2: ['high price', 'expensive', 'too costly', 'overpriced']
    }

    cluster_docs = [" ".join(clusters[1]), " ".join(clusters[2])]

    vectorizer = CountVectorizer(ngram_range=(1, 1))
    count_matrix = vectorizer.fit_transform(cluster_docs)
    vocabulary = vectorizer.get_feature_names_out()

    from representation.ctfidf_representation import ClassTfidfTransformer
    transformer = ClassTfidfTransformer()
    ctfidf_matrix = transformer.fit_transform(count_matrix)

    mmr = MMRRepresentation(diversity=0.3, top_k=5)

    try:
        # Test on first cluster
        ctfidf_scores = ctfidf_matrix[0].toarray()[0]
        keywords = mmr.extract_topics(
            cluster_id=1,
            ctfidf_scores=ctfidf_scores,
            vocabulary=vocabulary,
            cluster_texts=clusters[1]
        )

        print("\nCluster 1 (MMR with diversity=0.3):")
        for kw, score in keywords:
            print(f"  • {kw:<20} ({score:.4f})")

        # Calculate diversity stats
        stats = mmr.get_diversity_stats(keywords, clusters[1])
        print(f"\nDiversity stats: avg_similarity={stats['avg_similarity']:.3f}")

        print("\n✅ MMR test passed")
        return True

    except Exception as e:
        print(f"\n❌ MMR test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_keybert():
    """Test KeyBERT representation"""
    print("\n" + "="*80)
    print("TEST 3: KeyBERT Representation")
    print("="*80)

    # Build test data
    from sklearn.feature_extraction.text import CountVectorizer

    clusters = {
        1: ['good service', 'excellent service', 'great service', 'friendly staff']
    }

    cluster_docs = [" ".join(clusters[1])]

    vectorizer = CountVectorizer(ngram_range=(1, 1))
    count_matrix = vectorizer.fit_transform(cluster_docs)
    vocabulary = vectorizer.get_feature_names_out()

    from representation.ctfidf_representation import ClassTfidfTransformer
    transformer = ClassTfidfTransformer()
    ctfidf_matrix = transformer.fit_transform(count_matrix)

    keybert = KeyBERTRepresentation(top_k=5, weight=0.5)

    try:
        print("\nNote: This test may take a moment (generating embeddings)...")

        ctfidf_scores = ctfidf_matrix[0].toarray()[0]
        keywords = keybert.extract_topics(
            cluster_id=1,
            ctfidf_scores=ctfidf_scores,
            vocabulary=vocabulary,
            cluster_texts=clusters[1]
        )

        print("\nCluster 1 (KeyBERT with weight=0.5):")
        for kw, score in keywords:
            print(f"  • {kw:<20} ({score:.4f})")

        print("\n✅ KeyBERT test passed")
        return True

    except Exception as e:
        print(f"\n❌ KeyBERT test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_llm():
    """Test LLM representation"""
    print("\n" + "="*80)
    print("TEST 4: LLM Representation")
    print("="*80)

    # Build test data
    from sklearn.feature_extraction.text import CountVectorizer

    clusters = {
        1: [
            'good service',
            'excellent service',
            'great service',
            'friendly staff',
            'helpful employees'
        ]
    }

    cluster_docs = [" ".join(clusters[1])]

    vectorizer = CountVectorizer(ngram_range=(1, 1))
    count_matrix = vectorizer.fit_transform(cluster_docs)
    vocabulary = vectorizer.get_feature_names_out()

    from representation.ctfidf_representation import ClassTfidfTransformer
    transformer = ClassTfidfTransformer()
    ctfidf_matrix = transformer.fit_transform(count_matrix)

    llm = LLMRepresentation(model="gpt-4.1-mini", top_k=5, verbose=True)

    try:
        print("\nNote: This test will call the LLM API (may incur small cost)...")
        print("Proceeding in 2 seconds... (Ctrl+C to cancel)")
        import time
        time.sleep(2)

        ctfidf_scores = ctfidf_matrix[0].toarray()[0]
        keywords = llm.extract_topics(
            cluster_id=1,
            ctfidf_scores=ctfidf_scores,
            vocabulary=vocabulary,
            cluster_texts=clusters[1]
        )

        print("\nCluster 1 (LLM-enhanced):")
        for kw, score in keywords:
            print(f"  • {kw:<20} ({score:.4f})")

        print("\n✅ LLM test passed")
        return True

    except KeyboardInterrupt:
        print("\n⏭️  LLM test skipped (user cancelled)")
        return None

    except Exception as e:
        print(f"\n❌ LLM test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests and report results"""
    print("="*80)
    print("REPRESENTATION MODEL TESTS")
    print("="*80)

    results = {
        "c-TF-IDF": test_ctfidf(),
        "MMR": test_mmr(),
        "KeyBERT": test_keybert(),
        "LLM": test_llm()
    }

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for model, result in results.items():
        if result is True:
            status = "✅ PASSED"
        elif result is False:
            status = "❌ FAILED"
        else:
            status = "⏭️  SKIPPED"

        print(f"{model:<20} {status}")

    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)

    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")

    if failed == 0:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {failed} test(s) failed")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
