"""Fake-LLM end-to-end test for the Task 6 orchestrator (run_from_inputs).

Synthetic fixture (no real cache, no real LLM/embedding calls):
- 12 attributes in 3 well-separated phenomenon clusters (4 attrs each), built
  the same way test_clusterer.py's clean fixture is (base direction + small
  per-attribute jitter in an unused dimension), one idea per attribute so its
  embedding IS the attribute's centroid.
- Cluster g0's attr_valence is set so both poles clear the floor -> triggers
  `needs_noise_check` -> the fake NoiseVote (always genuine) -> a split
  positive/negative pair.
- Clusters g1 (all-positive) and g2 (all-neutral) are `dimensional` -> a
  single neutral code each.

The fake `llm_call` dispatches on `response_model`: MembershipVote -> "A",
NoiseVote -> genuine_opposition=True, CodeNaming -> a deterministic name
derived from the first member attribute mentioned in the prompt (regex over
the rendered evidence block — the naming prompt's own per-vote evidence
shuffle decides which member is "first").

`assembler.CacheManager` is monkeypatched (as test_assembler.py does) so
`save_experiment`'s cache write never touches disk or the "mece_codes" step;
exports land under tmp_path.
"""
import re

import numpy as np
import pytest

from models import DomainResultModel, DomainSet
from pipeline.step_5_codeGenerator_experiment.data_io import ExperimentInputs
from pipeline.step_5_codeGenerator_experiment.judgments import CodeNaming, MembershipVote, NoiseVote
from pipeline.step_5_codeGenerator_experiment.run_experiment import run_from_inputs


# =============================================================================
# Synthetic fixture: 12 attributes / 3 clusters
# =============================================================================
GROUPS = {
    0: ["attr_g0_0", "attr_g0_1", "attr_g0_2", "attr_g0_3"],  # -> needs_noise_check -> split
    1: ["attr_g1_0", "attr_g1_1", "attr_g1_2", "attr_g1_3"],  # -> dimensional (all positive)
    2: ["attr_g2_0", "attr_g2_1", "attr_g2_2", "attr_g2_3"],  # -> dimensional (all neutral)
}
ALL_ATTRS = [a for members in GROUPS.values() for a in members]

# g0: pos=10/neu=2/neg=10 (both poles well above floor, no neutral-third)
G0_VALENCE = {
    "attr_g0_0": {"positive": 5, "neutral": 1, "negative": 0},
    "attr_g0_1": {"positive": 5, "neutral": 0, "negative": 0},
    "attr_g0_2": {"positive": 0, "neutral": 0, "negative": 5},
    "attr_g0_3": {"positive": 0, "neutral": 1, "negative": 5},
}
# g1: all positive -> dimensional (negative never clears the floor)
G1_VALENCE = {a: {"positive": 5, "neutral": 0, "negative": 0} for a in GROUPS[1]}
# g2: all neutral -> dimensional
G2_VALENCE = {a: {"positive": 0, "neutral": 5, "negative": 0} for a in GROUPS[2]}
ATTR_VALENCE = {**G0_VALENCE, **G1_VALENCE, **G2_VALENCE}


def _idea_id(attr: str) -> str:
    return f"idea_{attr}"


def _centroid_vectors() -> dict:
    """One well-separated unit vector per attribute: a shared group direction
    plus a small per-attribute jitter in its own unused dimension (mirrors
    test_clusterer.py's _fixture_centroids helper, extended to 4 per group)."""
    base = {0: np.eye(8)[0], 1: np.eye(8)[1], 2: np.eye(8)[2]}
    out = {}
    for g, members in GROUPS.items():
        for j, attr in enumerate(members):
            v = base[g] + 0.05 * np.eye(8)[3 + j]
            out[attr] = v / np.linalg.norm(v)
    return out


def _partition_results() -> dict:
    vectors = _centroid_vectors()
    return {
        "domain1": DomainResultModel(
            partition_name="domain1",
            n_labels=len(ALL_ATTRS),
            n_batches=1,
            facets=[{"facet_name": "facet1"}],
            attributes={"facet1": [{"attribute_name": a} for a in ALL_ATTRS]},
            attribute_assignments={_idea_id(a): a for a in ALL_ATTRS},
        ),
    }, vectors


def _inputs() -> ExperimentInputs:
    partition_results, vectors = _partition_results()
    idea_texts = {_idea_id(a): f"statement about {a}" for a in ALL_ATTRS}
    idea_embeddings = {_idea_id(a): vectors[a].tolist() for a in ALL_ATTRS}
    return ExperimentInputs(
        partition_results=partition_results,
        idea_assignments={_idea_id(a): a for a in ALL_ATTRS},
        attr_valence=ATTR_VALENCE,
        idea_texts=idea_texts,
        idea_embeddings=idea_embeddings,
        language="Dutch",
        variable_key="Q1_full",
        survey_question="Wat vindt u van dit merk?",
    )


def _partition_set() -> DomainSet:
    return DomainSet(partitions=[])


# =============================================================================
# Fake llm_call: dispatches on response_model
# =============================================================================
_FIRST_MEMBER_RE = re.compile(r"\n- ([^:\n]+):")


async def fake_llm_call(prompt: str, response_model):
    if response_model is MembershipVote:
        return MembershipVote(choice="A", reason="fake: always A")
    if response_model is NoiseVote:
        return NoiseVote(genuine_opposition=True, reason="fake: always genuine")
    if response_model is CodeNaming:
        m = _FIRST_MEMBER_RE.search(prompt)
        first = m.group(1) if m else "Code"
        return CodeNaming(
            code_name=f"Code {first}",
            definition=f"Covers statements about {first}.",
            diagnostic_test=f"Mentions {first}.",
            typical_indicators=[first],
        )
    raise AssertionError(f"fake_llm_call: unexpected response_model {response_model}")


# =============================================================================
# Fake CacheManager — records save_metadata_to_cache instead of touching disk
# =============================================================================
class _FakeCacheManager:
    calls = []

    def __init__(self, *args, **kwargs):
        pass

    def save_metadata_to_cache(self, metadata, filename, step, variable_key):
        _FakeCacheManager.calls.append(
            {"metadata": metadata, "filename": filename, "step": step, "variable_key": variable_key}
        )
        return True


@pytest.fixture(autouse=True)
def _reset_fake_cache_manager():
    _FakeCacheManager.calls = []
    yield
    _FakeCacheManager.calls = []


# =============================================================================
# The e2e test
# =============================================================================
@pytest.mark.asyncio
async def test_run_from_inputs_end_to_end_with_fake_llm(monkeypatch, tmp_path):
    import pipeline.step_5_codeGenerator_experiment.assembler as assembler_mod
    monkeypatch.setattr(assembler_mod, "CacheManager", _FakeCacheManager)

    cache = await run_from_inputs(
        inputs=_inputs(),
        partition_set=_partition_set(),
        filename="survey.sav",
        llm_call=fake_llm_call,
        project_root=tmp_path,
    )

    # cache with more than 3 codes, including Overig
    assert len(cache.raw_codes) > 3
    code_names = [c["code_name"] for c in cache.raw_codes]
    assert "Overig" in code_names
    assert cache.raw_codes[-1]["code_name"] == "Overig"

    # a split pair (positive/negative) is present, sharing source_attributes
    valences = [c["valence"] for c in cache.raw_codes]
    assert "positive" in valences and "negative" in valences
    by_valence = {c["valence"]: c for c in cache.raw_codes if c["valence"] in ("positive", "negative")}
    assert set(by_valence["positive"]["source_attributes"]) == set(GROUPS[0])
    assert set(by_valence["negative"]["source_attributes"]) == set(GROUPS[0])

    # decision log (beslislog) is populated
    assert cache.codebook_narrative.strip() != ""
    assert "=== direction ===" in cache.codebook_narrative
    assert "=== naming ===" in cache.codebook_narrative

    # cache-save never used the baseline's step name
    assert len(_FakeCacheManager.calls) == 1
    assert _FakeCacheManager.calls[0]["step"] == "mece_codes_exp"
    assert _FakeCacheManager.calls[0]["step"] not in {"mece_codes", "taxonomy_codes"}

    # exports landed under tmp_path, never the real project tree
    export_dir = tmp_path / "exports" / "codebook"
    assert (export_dir / "codebook_survey_Q1_full_EXP_decisions.json").exists()
    assert (export_dir / "codebook_survey_Q1_full_EXP_grensgevallen.txt").exists()


def test_no_write_to_mece_codes_step_name_anywhere_in_module():
    """Static guard: the orchestrator module source never spells out the
    baseline's cache step name, so there's no path — now or after an edit —
    that could accidentally write the experiment's cache under it."""
    import pathlib
    import pipeline.step_5_codeGenerator_experiment.run_experiment as run_experiment_mod
    src = pathlib.Path(run_experiment_mod.__file__).read_text(encoding="utf-8")
    assert '"mece_codes"' not in src
    assert "'mece_codes'" not in src
