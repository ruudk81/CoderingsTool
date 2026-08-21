"""Round-trip: wat we wegschrijven moet er identiek weer uit komen."""
import json
from pathlib import Path

from pipeline.step_5_codeGenerator.consensus.storage import RunSet, load_runset, save_runset


def test_runset_overleeft_een_rondje_json(tmp_path):
    runset = RunSet(
        model="gpt-5.6-luna",
        effort="medium",
        attribute_ids=["A1", "A2", "A3"],
        attribute_names={"A1": "Duurzaamheid", "A2": "Sparen", "A3": "Kosten"},
        n_respondents=1092,
        runs=[[("A1", "A2"), ("A3",)],
              [("A1",), ("A2", "A3")]],
    )
    path = tmp_path / "runs.json"

    save_runset(runset, path)

    assert load_runset(path) == runset


def test_clusters_komen_terug_als_tuples(tmp_path):
    """JSON kent geen tuples; zonder conversie breekt elke vergelijking erop."""
    runset = RunSet(model="m", effort="high", attribute_ids=["A1", "A2"],
                    attribute_names={"A1": "x", "A2": "y"}, n_respondents=10,
                    runs=[[("A1", "A2")]])
    path = tmp_path / "runs.json"
    save_runset(runset, path)

    geladen = load_runset(path)

    assert geladen.runs[0][0] == ("A1", "A2")
    assert isinstance(geladen.runs[0][0], tuple)


def test_oude_bestanden_zonder_n_failed_laden_nog(tmp_path):
    """De sets van 2026-08-20 kennen dit veld niet; ze moeten leesbaar blijven,
    anders is al het verzamelde materiaal weg."""
    path = tmp_path / "oud.json"
    path.write_text(json.dumps({
        "model": "gpt-5.6-luna", "effort": "medium",
        "attribute_ids": ["A1"], "attribute_names": {"A1": "x"},
        "n_respondents": 10, "runs": [[["A1"]]], "salted": True,
    }), encoding="utf-8")

    assert load_runset(path).n_failed == 0
