"""Tests voor de lezende acties — `analyse` en `vergelijk`. Ze draaien geen
LLM-calls en schrijven niets weg, dus zijn goedkoop om te toetsen: het
materiaal komt van schijf, niet van een call."""
from pipeline.step_5_codeGenerator.consensus import run_codebook as runner
from pipeline.step_5_codeGenerator.consensus.config_consensus import ConsensusConfig
from pipeline.step_5_codeGenerator.consensus.storage import RunSet, save_runset


def test_setpad_draagt_configuratie_en_nummer():
    """Twee configuraties mogen nooit op hetzelfde bestand landen."""
    assert runner.runset_path("luna", 5).name == "consensus_luna_set5.json"
    assert runner.runset_path("gpt54", 1).name == "consensus_gpt54_set1.json"


def test_analyse_meldt_mislukte_runs(tmp_path, monkeypatch, capsys):
    """`n_failed` wordt bewaard maar tot deze fix nergens gelezen — een latere
    analyse zou dan `len(runs)` voor het gevraagde aantal aanzien en elke
    drempel stil laten verschuiven. Het hoort in de kop van `analyse` te staan,
    waar deze meting hem daadwerkelijk leest."""
    monkeypatch.setattr(runner, "OUT_DIR", tmp_path)
    runset = RunSet(
        model="gpt-5.6-luna", effort="medium",
        attribute_ids=["A1", "A2"], attribute_names={"A1": "x", "A2": "y"},
        n_respondents=10, runs=[[("A1", "A2")], [("A1",), ("A2",)]],
        salted=True, n_failed=3,
    )
    save_runset(runset, tmp_path / "consensus_luna_set9.json")

    runner.analyse(ConsensusConfig(config_name="luna"), 9)

    assert "3 mislukt" in capsys.readouterr().out
