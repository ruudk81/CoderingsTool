import numpy as np
import pytest
from pipeline.step_5_codeGenerator_experiment.phenomenon_clusterer import (
    attribute_centroids, discover_phenomena, DegenerateClusteringError, missing_attributes)

def _fixture_centroids():
    base = {0: np.eye(8)[0], 1: np.eye(8)[1], 2: np.eye(8)[2]}
    cents = {}
    for g in range(3):
        for j in range(3):
            v = base[g] + 0.05 * np.eye(8)[3 + j]
            cents[f"attr_g{g}_{j}"] = v / np.linalg.norm(v)
    return cents

def test_three_clean_clusters():
    res = discover_phenomena(_fixture_centroids())
    assert len(res.clusters) == 3
    for members in res.clusters.values():
        gs = {m.split("_")[1] for m in members}
        assert len(gs) == 1          # geen gemengde groepen

def test_deterministic():
    a = discover_phenomena(_fixture_centroids())
    b = discover_phenomena(_fixture_centroids())
    assert a.labels == b.labels and a.threshold == b.threshold

def test_degenerate_raises():
    cents = {f"a{i}": np.eye(4)[0] for i in range(5)}  # identiek → 1 cluster
    with pytest.raises(DegenerateClusteringError):
        discover_phenomena(cents)

def test_centroids_unit_norm():
    emb = {"i1": [1.0, 0.0], "i2": [0.0, 1.0]}
    cents = attribute_centroids(emb, {"i1": "A", "i2": "A"})
    assert abs(np.linalg.norm(cents["A"]) - 1.0) < 1e-9

def test_clean_fixture_no_ambiguous():
    """In perfectly separated data, ambiguous list is empty."""
    res = discover_phenomena(_fixture_centroids())
    assert res.ambiguous == [], f"Expected empty ambiguous list, got {res.ambiguous}"

def test_margins_sanity():
    """All finite margins > 0; neighbor points to different cluster."""
    res = discover_phenomena(_fixture_centroids())
    for attr, margin in res.margins.items():
        if margin != float("inf"):
            assert margin > 0, f"Attribute {attr} has non-positive margin {margin}"
        # neighbor must point to different cluster
        own_cluster = res.labels[attr]
        neighbor_cluster = res.neighbor[attr]
        assert own_cluster != neighbor_cluster, f"Attribute {attr} neighbor in same cluster"

def test_determinism_extended():
    """Determinism includes plateau_len and margins."""
    a = discover_phenomena(_fixture_centroids())
    b = discover_phenomena(_fixture_centroids())
    assert a.labels == b.labels
    assert a.threshold == b.threshold
    assert a.plateau_len == b.plateau_len
    assert a.margins == b.margins

def test_two_attribute_orthogonal():
    """Two orthogonal vectors → 2 singletons."""
    cents = {
        "attr_0": np.eye(4)[0],
        "attr_1": np.eye(4)[1]
    }
    res = discover_phenomena(cents)
    assert len(res.clusters) == 2
    assert len(res.clusters[1]) == 1
    assert len(res.clusters[2]) == 1
    assert res.margins["attr_0"] == float("inf")
    assert res.margins["attr_1"] == float("inf")
    assert res.ambiguous == []
    assert res.neighbor["attr_0"] == 2
    assert res.neighbor["attr_1"] == 1
    assert res.plateau_len == 0

def test_two_attribute_identical():
    """Two identical centroids → DegenerateClusteringError."""
    cents = {
        "attr_0": np.eye(4)[0],
        "attr_1": np.eye(4)[0]
    }
    with pytest.raises(DegenerateClusteringError):
        discover_phenomena(cents)

def test_missing_attributes():
    """missing_attributes returns attributes without centroids."""
    emb = {"i1": [1.0, 0.0], "i2": [0.0, 1.0], "i3": [1.0, 1.0]}
    assignments = {"i1": "A", "i2": "B", "i3": "C"}
    cents = attribute_centroids(emb, assignments)
    # cents has A, B, C (from the embeddings)
    # Create assignments where D, E are assigned but lack embeddings
    assignments_with_missing = {"i1": "A", "i2": "B", "i3": "D", "i4": "E"}
    missing = missing_attributes(assignments_with_missing, cents)
    # D and E are assigned but not in centroids
    assert "D" in missing
    assert "E" in missing
    assert "A" not in missing
    assert "B" not in missing
    assert len(missing) == 2

def test_missing_attributes_sorted():
    """missing_attributes returns sorted list."""
    cents = {"z_attr": np.array([1.0, 0.0])}
    assignments = {"i1": "a_missing", "i2": "z_missing", "i3": "z_attr"}
    missing = missing_attributes(assignments, cents)
    assert missing == ["a_missing", "z_missing"]
