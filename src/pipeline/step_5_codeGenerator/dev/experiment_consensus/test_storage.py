"""Round-trip: wat we wegschrijven moet er identiek weer uit komen."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from storage import RunSet, load_runset, save_runset


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


def test_brug_telt_paren_over_runs():
    from stability_bridge import together_from_runs

    runs = [[("A1", "A2"), ("A3",)],
            [("A1", "A2"), ("A3",)],
            [("A1",), ("A2", "A3")]]

    together = together_from_runs(runs, ["A1", "A2", "A3"])

    assert together[frozenset({"A1", "A2"})] == 2
    assert together[frozenset({"A2", "A3"})] == 1
    assert together[frozenset({"A1", "A3"})] == 0


def test_verslagnaam_draagt_de_instellingen():
    """Twee varianten mogen nooit op hetzelfde bestand landen."""
    from run_experiment import report_path

    a = report_path("luna", 3, "consensus", 0.6, "two")
    b = report_path("luna", 3, "consensus", 0.7, "two")
    c = report_path("luna", 3, "consensus", 0.6, "three")

    assert a.name == "codeboek_luna_set3_consensus_tau06_twopolen.txt"
    assert len({a.name, b.name, c.name}) == 3


def test_basislijn_krijgt_geen_tau_in_de_naam():
    """Zonder consensus doet tau niets, dus hij hoort niet in de naam."""
    from run_experiment import report_path

    assert report_path("luna", 1, "baseline", 0.7, "three").name == \
        "codeboek_luna_set1_baseline_threepolen.txt"
