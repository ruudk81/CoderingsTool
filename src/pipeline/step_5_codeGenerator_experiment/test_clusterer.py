import numpy as np
import pytest
from pipeline.step_5_codeGenerator_experiment.phenomenon_clusterer import (
    attribute_centroids, discover_phenomena, DegenerateClusteringError)

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
