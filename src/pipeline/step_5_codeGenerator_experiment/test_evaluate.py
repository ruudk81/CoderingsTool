"""Tests for evaluate_experiment.py (Task 7): richting_dekking, jaccard,
compare, and dump_run_artifacts.

Synthetic fixtures throughout — no real cache, no real LLM, no real export
into the project tree (mirrors test_assembler.py's style: monkeypatched
CacheManager, tmp_path for anything that touches disk).
"""
import json

import pytest

from models import CodingResultsCache, DomainResultModel, DomainSet
from pipeline.step_5_codeGenerator_experiment.evaluate_experiment import (
    compare,
    dump_run_artifacts,
    jaccard,
    richting_dekking,
)


# =============================================================================
# richting_dekking fixtures
# =============================================================================
def _domain(attribute_assignments, attribute_valence, attribute_names=None):
    names = attribute_names or sorted(set(attribute_assignments.values()))
    return {
        "d": DomainResultModel(
            partition_name="d",
            n_labels=len(names),
            n_batches=1,
            facets=[{"facet_name": "f"}],
            attributes={"f": [{"attribute_name": n} for n in names]},
            attribute_assignments=attribute_assignments,
            attribute_valence=attribute_valence,
        )
    }


def test_richting_dekking_pure_pair_is_fully_covered():
    # AttrPos: 6 ideas "+", AttrNeg: 4 ideas "-". total=10, floor=max(2,int(log(10)))=2.
    # Both poles substantial; a positive and a negative code share the same
    # bronnenset -> both covered.
    assignments = {}
    valence = {}
    for i in range(6):
        idea = f"pos{i}"
        assignments[idea] = "AttrPos"
        valence[idea] = "+"
    for i in range(4):
        idea = f"neg{i}"
        assignments[idea] = "AttrNeg"
        valence[idea] = "-"
    pr = _domain(assignments, valence, ["AttrPos", "AttrNeg"])

    codes = [
        {"code_name": "Prijs positief", "valence": "positive",
         "source_attributes": ["AttrPos", "AttrNeg"]},
        {"code_name": "Prijs negatief", "valence": "negative",
         "source_attributes": ["AttrPos", "AttrNeg"]},
    ]
    assert richting_dekking(codes, pr) == 1.0


def test_richting_dekking_dimensional_code_with_both_poles_substantial_is_zero():
    # AttrDim: 8 ideas "+", 8 ideas "-". total=16, floor=max(2,int(log(16)))=2.
    # Both poles substantial, but the only code is neutral -> neither pole covered.
    assignments = {}
    valence = {}
    for i in range(8):
        idea = f"p{i}"
        assignments[idea] = "AttrDim"
        valence[idea] = "+"
    for i in range(8):
        idea = f"n{i}"
        assignments[idea] = "AttrDim"
        valence[idea] = "-"
    pr = _domain(assignments, valence, ["AttrDim"])

    codes = [
        {"code_name": "Merk algemeen", "valence": "neutral", "source_attributes": ["AttrDim"]},
    ]
    assert richting_dekking(codes, pr) == 0.0


def test_richting_dekking_mono_pool_phenomenon_is_fully_covered():
    # AttrMono: 5 ideas "+", nothing else. total=5, floor=max(2,int(log(5)))=2.
    # Only "positive" clears the floor -> single substantial pole, covered.
    assignments = {}
    valence = {}
    for i in range(5):
        idea = f"m{i}"
        assignments[idea] = "AttrMono"
        valence[idea] = "+"
    pr = _domain(assignments, valence, ["AttrMono"])

    codes = [
        {"code_name": "Positief mono", "valence": "positive", "source_attributes": ["AttrMono"]},
    ]
    assert richting_dekking(codes, pr) == 1.0


def test_richting_dekking_excludes_overig():
    # Overig's own bronnenset (dangling attribute) must not count as a phenomenon.
    assignments, valence = {}, {}
    for i in range(6):
        idea = f"g{i}"
        assignments[idea] = "GhostAttr"
        valence[idea] = "+"
    pr = _domain(assignments, valence, [])  # GhostAttr not in taxonomy attribute list

    codes = [
        {"code_name": "Overig", "valence": "neutral", "source_attributes": ["GhostAttr"]},
    ]
    # No non-Overig phenomenon exists -> vacuously fully covered.
    assert richting_dekking(codes, pr) == 1.0


def test_richting_dekking_no_data_is_vacuously_covered():
    pr = _domain({}, {}, [])
    assert richting_dekking([], pr) == 1.0


# =============================================================================
# jaccard
# =============================================================================
def test_jaccard_identical_sets_is_one():
    assert jaccard({"Prijs positief", "Prijs negatief"}, {"prijs positief", "PRIJS NEGATIEF"}) == 1.0


def test_jaccard_disjoint_sets_is_zero():
    assert jaccard({"A"}, {"B"}) == 0.0


def test_jaccard_partial_overlap_case_insensitive():
    a = {"Prijs positief", "Service", "Kwaliteit"}
    b = {"prijs POSITIEF", "Service", "Iets anders"}
    # intersection={prijs positief, service} (2), union has 4 distinct names
    assert jaccard(a, b) == pytest.approx(2 / 4)


def test_jaccard_both_empty_is_one():
    assert jaccard(set(), set()) == 1.0


# =============================================================================
# compare — full pipeline against synthetic runs written to tmp_path
# =============================================================================
def _baseline_cache():
    assignments = {
        "idea1": "AttrPos", "idea2": "AttrPos", "idea3": "AttrNeg",
        "idea4": "AttrNeutral",
    }
    valence = {"idea1": "+", "idea2": "+", "idea3": "-"}
    pr = _domain(assignments, valence, ["AttrPos", "AttrNeg", "AttrNeutral"])
    codes = [
        {"code_name": "Prijs positief", "valence": "positive",
         "source_attributes": ["AttrPos", "AttrNeg"]},
        {"code_name": "Prijs negatief", "valence": "negative",
         "source_attributes": ["AttrPos", "AttrNeg"]},
        {"code_name": "Service", "valence": "neutral", "source_attributes": ["AttrNeutral"]},
        {"code_name": "Overig", "valence": "neutral", "source_attributes": []},
    ]
    return CodingResultsCache(
        partition_set=DomainSet(partitions=[]),
        partition_results=pr,
        raw_codes=codes,
    ), pr


def _write_run(tmp_path, name, codes):
    run_dir = tmp_path / name
    run_dir.mkdir()
    (run_dir / "raw_codes.json").write_text(json.dumps(codes), encoding="utf-8")
    return run_dir


def test_compare_produces_all_columns_without_crashing(tmp_path):
    baseline_cache, pr = _baseline_cache()

    run1_codes = [
        {"code_name": "Prijs positief", "valence": "positive",
         "source_attributes": ["AttrPos", "AttrNeg"]},
        {"code_name": "Prijs negatief", "valence": "negative",
         "source_attributes": ["AttrPos", "AttrNeg"]},
        {"code_name": "Service algemeen", "valence": "neutral", "source_attributes": ["AttrNeutral"]},
        {"code_name": "Overig", "valence": "neutral", "source_attributes": []},
    ]
    run2_codes = [
        {"code_name": "Prijs positief", "valence": "positive",
         "source_attributes": ["AttrPos", "AttrNeg"]},
        {"code_name": "Prijs negatief", "valence": "negative",
         "source_attributes": ["AttrPos", "AttrNeg"]},
        {"code_name": "Service anders genoemd", "valence": "neutral",
         "source_attributes": ["AttrNeutral"]},
        {"code_name": "Overig", "valence": "neutral", "source_attributes": []},
    ]
    run1 = _write_run(tmp_path, "EXP_run1", run1_codes)
    run2 = _write_run(tmp_path, "EXP_run2", run2_codes)

    report = compare(baseline_cache, [run1, run2], pr)

    assert "Codes" in report
    assert "Onder-split adviezen" in report
    assert "Mini-codes" in report
    assert "Overig-share" in report
    assert "Richting-dekking" in report
    assert "baseline" in report
    assert "run1" in report and "run2" in report
    assert "Reproduceerbaarheid" in report
    assert "codeaantal spreiding" in report
    assert "Jaccard(run1, run2)" in report


def test_compare_with_no_runs_reports_baseline_only(tmp_path):
    baseline_cache, pr = _baseline_cache()
    report = compare(baseline_cache, [], pr)
    assert "baseline" in report
    assert "geen experimentruns" in report


# =============================================================================
# dump_run_artifacts — cache monkeypatched, exports under tmp_path
# =============================================================================
class _FakeCacheManager:
    instance_cache = None

    def __init__(self, *args, **kwargs):
        pass

    def load_metadata_from_cache(self, filename, step, variable_key, model_cls):
        assert step == "mece_codes_exp"
        return _FakeCacheManager.instance_cache


def test_dump_run_artifacts_writes_raw_codes_and_narrative(monkeypatch, tmp_path):
    import pipeline.step_5_codeGenerator_experiment.evaluate_experiment as evaluate_mod

    cache, pr = _baseline_cache()
    cache.codebook_narrative = "=== clustering ===\n- foo -> bar\n"
    _FakeCacheManager.instance_cache = cache
    monkeypatch.setattr(evaluate_mod, "CacheManager", _FakeCacheManager)
    monkeypatch.setattr(evaluate_mod, "_PROJECT_ROOT", tmp_path)

    run_dir = tmp_path / "EXP_run1"
    dump_run_artifacts(run_dir, filename="survey.sav", var_name="Q1", sample_size=2000)

    raw = json.loads((run_dir / "raw_codes.json").read_text(encoding="utf-8"))
    assert raw == cache.raw_codes
    assert (run_dir / "narrative.txt").read_text(encoding="utf-8") == cache.codebook_narrative


def test_dump_run_artifacts_copies_exports_when_present(monkeypatch, tmp_path):
    import pipeline.step_5_codeGenerator_experiment.evaluate_experiment as evaluate_mod

    cache, _ = _baseline_cache()
    _FakeCacheManager.instance_cache = cache
    monkeypatch.setattr(evaluate_mod, "CacheManager", _FakeCacheManager)
    monkeypatch.setattr(evaluate_mod, "_PROJECT_ROOT", tmp_path)

    from utils.cacheManager import generate_enhanced_variable_key
    vk = generate_enhanced_variable_key(selected_variables=["Q1"], is_merged=False, sample_size=2000)
    export_dir = tmp_path / "exports" / "codebook"
    export_dir.mkdir(parents=True)
    prefix = f"codebook_survey_{vk}"
    (export_dir / f"{prefix}_EXP_decisions.json").write_text("[]", encoding="utf-8")
    (export_dir / f"{prefix}_EXP_grensgevallen.txt").write_text("", encoding="utf-8")

    run_dir = tmp_path / "EXP_run1"
    dump_run_artifacts(run_dir, filename="survey.sav", var_name="Q1", sample_size=2000)

    assert (run_dir / f"{prefix}_EXP_decisions.json").exists()
    assert (run_dir / f"{prefix}_EXP_grensgevallen.txt").exists()


def test_dump_run_artifacts_missing_cache_raises(monkeypatch, tmp_path):
    import pipeline.step_5_codeGenerator_experiment.evaluate_experiment as evaluate_mod

    _FakeCacheManager.instance_cache = None
    monkeypatch.setattr(evaluate_mod, "CacheManager", _FakeCacheManager)
    monkeypatch.setattr(evaluate_mod, "_PROJECT_ROOT", tmp_path)

    with pytest.raises(RuntimeError):
        dump_run_artifacts(tmp_path / "EXP_run1", filename="survey.sav", var_name="Q1", sample_size=2000)
