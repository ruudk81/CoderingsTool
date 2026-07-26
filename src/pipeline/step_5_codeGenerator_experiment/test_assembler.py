"""Tests for assembler.py (phase 5): assembly, decision log, cache-save, scorecard.

Synthetic fixture: two phenomena —
- cluster 0: a split pair (positive/negative) sharing source_attributes, from
  attributes "AttrPos"/"AttrNeg".
- cluster 1: a single dimensional (neutral) code, from attribute "AttrNeutral".
Plus a dangling idea-assigned attribute ("GhostAttr", not in the taxonomy)
that must land in Overig.

No real LLM calls (namings are fake CodeNaming instances), no real cache
write (CacheManager monkeypatched), no real export into the project tree
(project_root=tmp_path).
"""
import json

import pytest

from models import DomainResultModel, DomainSet
from pipeline.step_5_codeGenerator_experiment.assembler import (
    Decision,
    assemble_codebook,
    render_decision_log,
    run_scorecard_on,
    save_experiment,
)
from pipeline.step_5_codeGenerator_experiment.data_io import ExperimentInputs
from pipeline.step_5_codeGenerator_experiment.judgments import CodeNaming
from pipeline.step_5_codeGenerator_experiment.phenomenon_clusterer import ClusterResult


# =============================================================================
# Fixtures
# =============================================================================
def _partition_results():
    return {
        "domain1": DomainResultModel(
            partition_name="domain1",
            n_labels=3,
            n_batches=1,
            facets=[{"facet_name": "facet1"}],
            attributes={
                "facet1": [
                    {"attribute_name": "AttrPos"},
                    {"attribute_name": "AttrNeg"},
                    {"attribute_name": "AttrNeutral"},
                ]
            },
            attribute_assignments={
                "idea1": "AttrPos",
                "idea2": "AttrNeg",
                "idea3": "AttrNeutral",
                "idea4": "GhostAttr",  # dangling: not in taxonomy attribute list
            },
        ),
    }


def _inputs():
    pr = _partition_results()
    return ExperimentInputs(
        partition_results=pr,
        idea_assignments={
            "idea1": "AttrPos", "idea2": "AttrNeg",
            "idea3": "AttrNeutral", "idea4": "GhostAttr",
        },
        attr_valence={},
        idea_texts={
            "idea1": "great price", "idea2": "bad price",
            "idea3": "average service", "idea4": "??",
        },
        idea_embeddings={},
        language="Dutch",
        variable_key="Q1_full",
        survey_question="Wat vindt u van dit merk?",
    )


def _cluster_result():
    return ClusterResult(
        labels={"AttrPos": 0, "AttrNeg": 0, "AttrNeutral": 1},
        clusters={0: ["AttrPos", "AttrNeg"], 1: ["AttrNeutral"]},
        threshold=0.4,
        plateau_len=5,
    )


def _code_plans():
    return {
        0: [
            {"valence": "positive", "expected": 6},
            {"valence": "negative", "expected": 4},
        ],
        1: [
            {"valence": "neutral", "expected": 3},
        ],
    }


def _namings():
    return {
        (0, "positive"): CodeNaming(
            code_name="Prijs positief", definition="Positieve prijsopmerkingen",
            diagnostic_test="noemt prijs gunstig", typical_indicators=["goedkoop"],
        ),
        (0, "negative"): CodeNaming(
            code_name="Prijs negatief", definition="Negatieve prijsopmerkingen",
            diagnostic_test="noemt prijs ongunstig", typical_indicators=["duur"],
        ),
        (1, "neutral"): CodeNaming(
            code_name="Service algemeen", definition="Algemene opmerkingen over service",
            diagnostic_test="noemt service zonder oordeel", typical_indicators=["service"],
        ),
    }


def _decisions():
    return [
        Decision(phase="clustering", subject="AttrPos+AttrNeg", outcome="grouped",
                 evidence={"cosine": 0.1}),
        Decision(phase="direction", subject="cluster 0", outcome="split",
                 evidence={"pos": 6, "neg": 4}, is_borderline=True),
        Decision(phase="direction", subject="cluster 1", outcome="dimensional",
                 evidence={"pos": 1, "neg": 1}),
        Decision(phase="naming", subject="Prijs positief", outcome="named",
                 votes={"a": 1}, is_borderline=True),
    ]


def _partition_set():
    return DomainSet(partitions=[])


def _assembled():
    return assemble_codebook(
        inputs=_inputs(),
        cluster_result=_cluster_result(),
        code_plans=_code_plans(),
        namings=_namings(),
        decisions=_decisions(),
        partition_set=_partition_set(),
    )


# =============================================================================
# (a) split pair shares source_attributes
# =============================================================================
def test_split_pair_shares_source_attributes():
    cache = _assembled()
    by_name = {c["code_name"]: c for c in cache.raw_codes}
    pos = by_name["Prijs positief"]
    neg = by_name["Prijs negatief"]
    assert pos["source_attributes"] == neg["source_attributes"]
    assert set(pos["source_attributes"]) == {"AttrPos", "AttrNeg"}


# =============================================================================
# (b) Overig present, last in list
# =============================================================================
def test_overig_present_and_last():
    cache = _assembled()
    assert cache.raw_codes[-1]["code_name"] == "Overig"
    assert cache.raw_codes[-1]["valence"] == "neutral"
    # GhostAttr is idea-assigned but not in the taxonomy attribute list
    assert cache.raw_codes[-1]["source_attributes"] == ["GhostAttr"]


def test_overig_may_be_empty_when_nothing_dangling():
    inputs = _inputs()
    inputs.idea_assignments = {
        "idea1": "AttrPos", "idea2": "AttrNeg", "idea3": "AttrNeutral",
    }
    cache = assemble_codebook(
        inputs=inputs, cluster_result=_cluster_result(), code_plans=_code_plans(),
        namings=_namings(), decisions=[], partition_set=_partition_set(),
    )
    assert cache.raw_codes[-1]["code_name"] == "Overig"
    assert cache.raw_codes[-1]["source_attributes"] == []


# =============================================================================
# (c) codebook_narrative contains every Decision subject
# =============================================================================
def test_narrative_contains_every_decision_subject():
    cache = _assembled()
    for d in _decisions():
        assert d.subject in cache.codebook_narrative


def test_render_decision_log_empty_is_empty_string():
    assert render_decision_log([]) == ""


# =============================================================================
# (d) K#-ids present after ensure_codebook_ids
# =============================================================================
def test_codes_and_attributes_carry_stable_ids():
    cache = _assembled()
    for c in cache.raw_codes:
        assert c["code_id"].startswith("K")
    # attribute ids minted in place on partition_results
    attrs = cache.partition_results["domain1"].attributes["facet1"]
    for a in attrs:
        assert a["attribute_id"].startswith("A")
    # non-Overig codes resolve their source_attribute_ids
    by_name = {c["code_name"]: c for c in cache.raw_codes}
    assert by_name["Prijs positief"]["source_attribute_ids"]
    assert by_name["Prijs positief"]["source_attribute_ids"] == by_name["Prijs negatief"]["source_attribute_ids"]


def test_code_ids_are_unique_and_sequential():
    cache = _assembled()
    ids = [c["code_id"] for c in cache.raw_codes]
    assert len(ids) == len(set(ids))
    assert ids == [f"K{i}" for i in range(1, len(ids) + 1)]


# =============================================================================
# save_experiment: cache-save (monkeypatched) + exports (tmp_path)
# =============================================================================
class _FakeCacheManager:
    """Records save_metadata_to_cache calls instead of touching disk."""
    calls = []

    def __init__(self, *args, **kwargs):
        pass

    def save_metadata_to_cache(self, metadata, filename, step, variable_key):
        _FakeCacheManager.calls.append(
            {"metadata": metadata, "filename": filename, "step": step,
             "variable_key": variable_key}
        )
        return True


@pytest.fixture(autouse=True)
def _reset_fake_cache_manager():
    _FakeCacheManager.calls = []
    yield
    _FakeCacheManager.calls = []


def test_save_experiment_caches_under_mece_codes_exp_step_name(monkeypatch, tmp_path):
    import pipeline.step_5_codeGenerator_experiment.assembler as assembler_mod
    monkeypatch.setattr(assembler_mod, "CacheManager", _FakeCacheManager)

    cache = _assembled()
    save_experiment(
        cache=cache, filename="survey.sav", variable_key="Q1_full",
        decisions=_decisions(), project_root=tmp_path,
    )

    assert len(_FakeCacheManager.calls) == 1
    call = _FakeCacheManager.calls[0]
    assert call["step"] == "mece_codes_exp"
    assert call["filename"] == "survey.sav"
    assert call["variable_key"] == "Q1_full"
    assert call["metadata"] is cache


def test_save_experiment_never_uses_baseline_step_names(monkeypatch, tmp_path):
    import pipeline.step_5_codeGenerator_experiment.assembler as assembler_mod
    monkeypatch.setattr(assembler_mod, "CacheManager", _FakeCacheManager)

    save_experiment(
        cache=_assembled(), filename="survey.sav", variable_key="Q1_full",
        decisions=_decisions(), project_root=tmp_path,
    )
    step = _FakeCacheManager.calls[0]["step"]
    assert step not in {"mece_codes", "taxonomy_codes"}


def test_save_experiment_writes_decisions_json(monkeypatch, tmp_path):
    import pipeline.step_5_codeGenerator_experiment.assembler as assembler_mod
    monkeypatch.setattr(assembler_mod, "CacheManager", _FakeCacheManager)

    decisions_path, _ = save_experiment(
        cache=_assembled(), filename="survey.sav", variable_key="Q1_full",
        decisions=_decisions(), project_root=tmp_path,
    )
    assert decisions_path.exists()
    assert decisions_path.parent == tmp_path / "exports" / "codebook"
    assert decisions_path.name == "codebook_survey_Q1_full_EXP_decisions.json"
    data = json.loads(decisions_path.read_text(encoding="utf-8"))
    assert len(data) == len(_decisions())
    assert {d["subject"] for d in data} == {d.subject for d in _decisions()}


# =============================================================================
# (e) grensgevallen export contains exactly the borderline subset
# =============================================================================
def test_grensgevallen_export_is_exact_borderline_subset(monkeypatch, tmp_path):
    import pipeline.step_5_codeGenerator_experiment.assembler as assembler_mod
    monkeypatch.setattr(assembler_mod, "CacheManager", _FakeCacheManager)

    decisions = _decisions()
    borderline_subjects = {d.subject for d in decisions if d.is_borderline}
    non_borderline_subjects = {d.subject for d in decisions if not d.is_borderline}
    assert borderline_subjects and non_borderline_subjects  # fixture sanity

    _, grens_path = save_experiment(
        cache=_assembled(), filename="survey.sav", variable_key="Q1_full",
        decisions=decisions, project_root=tmp_path,
    )
    assert grens_path.name == "codebook_survey_Q1_full_EXP_grensgevallen.txt"
    content = grens_path.read_text(encoding="utf-8")
    for subj in borderline_subjects:
        assert subj in content
    for subj in non_borderline_subjects:
        assert subj not in content


# =============================================================================
# run_scorecard_on: reuses build_scorecard/format_scorecard, no copy
# =============================================================================
def test_run_scorecard_on_reuses_baseline_builder(capsys):
    cache = _assembled()
    scorecard = run_scorecard_on(cache, cache.partition_results)
    assert scorecard.overig_code_name == "Overig"
    assert scorecard.n_codes == len(cache.raw_codes)
    captured = capsys.readouterr()
    assert "PASS" in captured.out or "FAIL" in captured.out
