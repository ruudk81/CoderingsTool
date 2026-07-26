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


# =============================================================================
# Targeted test: membership vote relocates an ambiguous attribute
# =============================================================================
# Same 3-group base/jitter construction as GROUPS above, plus one attribute
# ("attr_wobble") placed deliberately between group 0 and group 1's centroids
# so its margin falls below the ambiguity cut (0.5x median). Verified against
# phenomenon_clusterer.discover_phenomena directly: attr_wobble's own cluster
# is group 0's, its margin is 0.41 (well under the ~0.98-0.99 the clean
# members get), and its neighbor is group 1's cluster.
RELOC_GROUPS = {
    0: ["attr_g0_0", "attr_g0_1", "attr_g0_2"],
    1: ["attr_g1_0", "attr_g1_1", "attr_g1_2"],
    2: ["attr_g2_0", "attr_g2_1", "attr_g2_2"],
}
WOBBLE = "attr_wobble"
RELOC_ALL_ATTRS = [a for members in RELOC_GROUPS.values() for a in members] + [WOBBLE]
# All-positive across the board: every cluster (before and after the move)
# resolves to "dimensional" -> no noise vote needed, isolating the test to
# the membership-relocation path alone.
RELOC_VALENCE = {a: {"positive": 5, "neutral": 0, "negative": 0} for a in RELOC_ALL_ATTRS}


def _reloc_vectors() -> dict:
    base = {0: np.eye(8)[0], 1: np.eye(8)[1], 2: np.eye(8)[2]}
    out = {}
    for g, members in RELOC_GROUPS.items():
        for j, attr in enumerate(members):
            v = base[g] + 0.05 * np.eye(8)[3 + j]
            out[attr] = v / np.linalg.norm(v)
    w = 0.5 * base[0] + 0.5 * base[1]
    out[WOBBLE] = w / np.linalg.norm(w)
    return out


def _reloc_inputs() -> ExperimentInputs:
    vectors = _reloc_vectors()
    partition_results = {
        "domain1": DomainResultModel(
            partition_name="domain1",
            n_labels=len(RELOC_ALL_ATTRS),
            n_batches=1,
            facets=[{"facet_name": "facet1"}],
            attributes={"facet1": [{"attribute_name": a} for a in RELOC_ALL_ATTRS]},
            attribute_assignments={_idea_id(a): a for a in RELOC_ALL_ATTRS},
        ),
    }
    idea_texts = {_idea_id(a): f"statement about {a}" for a in RELOC_ALL_ATTRS}
    idea_embeddings = {_idea_id(a): vectors[a].tolist() for a in RELOC_ALL_ATTRS}
    return ExperimentInputs(
        partition_results=partition_results,
        idea_assignments={_idea_id(a): a for a in RELOC_ALL_ATTRS},
        attr_valence=RELOC_VALENCE,
        idea_texts=idea_texts,
        idea_embeddings=idea_embeddings,
        language="Dutch",
        variable_key="Q1_full",
        survey_question="Wat vindt u van dit merk?",
    )


async def _fake_llm_always_b_membership(prompt: str, response_model):
    """Membership -> always "B" (move); naming -> a fresh unique name per
    call (no collision logic under test here)."""
    if response_model is MembershipVote:
        return MembershipVote(choice="B", reason="fake: always B")
    if response_model is CodeNaming:
        n = next(_fake_llm_always_b_membership._counter)
        return CodeNaming(
            code_name=f"RelocCode{n}", definition="d", diagnostic_test="t", typical_indicators=["i"],
        )
    raise AssertionError(f"unexpected response_model in relocation test: {response_model}")


_fake_llm_always_b_membership._counter = iter(range(1, 100))


@pytest.mark.asyncio
async def test_membership_vote_relocates_ambiguous_attribute_to_neighbor_cluster(monkeypatch, tmp_path):
    import json

    import pipeline.step_5_codeGenerator_experiment.assembler as assembler_mod
    monkeypatch.setattr(assembler_mod, "CacheManager", _FakeCacheManager)

    cache = await run_from_inputs(
        inputs=_reloc_inputs(),
        partition_set=_partition_set(),
        filename="survey.sav",
        llm_call=_fake_llm_always_b_membership,
        project_root=tmp_path,
    )

    non_overig = [c for c in cache.raw_codes if c["code_name"] != "Overig"]
    by_sources = {frozenset(c["source_attributes"]): c for c in non_overig}

    # the wobble attribute ends up in group 1's code, not group 0's — labels
    # and clusters were mutated consistently (a mismatch here would either
    # KeyError during assembly or leave attr_wobble in both/neither code).
    assert frozenset(RELOC_GROUPS[0]) in by_sources, by_sources.keys()
    assert frozenset(RELOC_GROUPS[1] + [WOBBLE]) in by_sources, by_sources.keys()
    assert frozenset(RELOC_GROUPS[2]) in by_sources, by_sources.keys()

    # the relocated attribute's code counts it among its source_attribute_ids too
    relocated_code = by_sources[frozenset(RELOC_GROUPS[1] + [WOBBLE])]
    attrs_by_name = {a["attribute_name"]: a for a in cache.partition_results["domain1"].attributes["facet1"]}
    assert attrs_by_name[WOBBLE]["attribute_id"] in relocated_code["source_attribute_ids"]

    # a borderline "moved_to_neighbor" Decision record was logged
    decisions_path = tmp_path / "exports" / "codebook" / "codebook_survey_Q1_full_EXP_decisions.json"
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    membership_records = [d for d in decisions if d["phase"] == "membership" and d["subject"] == WOBBLE]
    assert len(membership_records) == 1
    record = membership_records[0]
    assert record["outcome"] == "moved_to_neighbor"
    assert record["is_borderline"] is True


# =============================================================================
# Targeted test: an attribute with no embeddings is routed to Overig
# =============================================================================
MISSING_GROUPS = {
    0: ["attr_g0_0", "attr_g0_1", "attr_g0_2"],
    1: ["attr_g1_0", "attr_g1_1", "attr_g1_2"],
    2: ["attr_g2_0", "attr_g2_1", "attr_g2_2"],
}
MISSING_ATTR = "attr_missing"
MISSING_CLUSTERED_ATTRS = [a for members in MISSING_GROUPS.values() for a in members]
MISSING_ALL_ATTRS = MISSING_CLUSTERED_ATTRS + [MISSING_ATTR]
# All-positive -> every cluster resolves "dimensional", no noise vote needed.
MISSING_VALENCE = {a: {"positive": 5, "neutral": 0, "negative": 0} for a in MISSING_CLUSTERED_ATTRS}


def _missing_vectors() -> dict:
    base = {0: np.eye(8)[0], 1: np.eye(8)[1], 2: np.eye(8)[2]}
    out = {}
    for g, members in MISSING_GROUPS.items():
        for j, attr in enumerate(members):
            v = base[g] + 0.05 * np.eye(8)[3 + j]
            out[attr] = v / np.linalg.norm(v)
    return out


def _missing_inputs() -> ExperimentInputs:
    vectors = _missing_vectors()
    partition_results = {
        "domain1": DomainResultModel(
            partition_name="domain1",
            n_labels=len(MISSING_ALL_ATTRS),
            n_batches=1,
            facets=[{"facet_name": "facet1"}],
            # attr_missing IS a real taxonomy attribute — the point of this
            # test is that assembler's dangling-name Overig sweep does NOT
            # catch it (it's not dangling, it's just embedding-less), so the
            # orchestrator itself must route it.
            attributes={"facet1": [{"attribute_name": a} for a in MISSING_ALL_ATTRS]},
            attribute_assignments={_idea_id(a): a for a in MISSING_ALL_ATTRS},
        ),
    }
    idea_texts = {_idea_id(a): f"statement about {a}" for a in MISSING_ALL_ATTRS}
    # attr_missing's idea is assigned (above) but deliberately has NO embedding.
    idea_embeddings = {_idea_id(a): vectors[a].tolist() for a in MISSING_CLUSTERED_ATTRS}
    return ExperimentInputs(
        partition_results=partition_results,
        idea_assignments={_idea_id(a): a for a in MISSING_ALL_ATTRS},
        attr_valence=MISSING_VALENCE,
        idea_texts=idea_texts,
        idea_embeddings=idea_embeddings,
        language="Dutch",
        variable_key="Q1_full",
        survey_question="Wat vindt u van dit merk?",
    )


async def _fake_llm_naming_only(prompt: str, response_model):
    if response_model is CodeNaming:
        n = next(_fake_llm_naming_only._counter)
        return CodeNaming(
            code_name=f"MissingCode{n}", definition="d", diagnostic_test="t", typical_indicators=["i"],
        )
    raise AssertionError(f"unexpected response_model in missing-embedding test: {response_model}")


_fake_llm_naming_only._counter = iter(range(1, 100))


@pytest.mark.asyncio
async def test_missing_embedding_attribute_is_routed_to_overig(monkeypatch, tmp_path):
    import json

    import pipeline.step_5_codeGenerator_experiment.assembler as assembler_mod
    monkeypatch.setattr(assembler_mod, "CacheManager", _FakeCacheManager)

    cache = await run_from_inputs(
        inputs=_missing_inputs(),
        partition_set=_partition_set(),
        filename="survey.sav",
        llm_call=_fake_llm_naming_only,
        project_root=tmp_path,
    )

    non_overig = [c for c in cache.raw_codes if c["code_name"] != "Overig"]
    overig = cache.raw_codes[-1]
    assert overig["code_name"] == "Overig"

    # the attribute never landed in a phenomenon cluster/code ...
    for c in non_overig:
        assert MISSING_ATTR not in c["source_attributes"]
    # ... it landed in Overig instead
    assert MISSING_ATTR in overig["source_attributes"]

    # a routed_to_overig Decision record was logged
    decisions_path = tmp_path / "exports" / "codebook" / "codebook_survey_Q1_full_EXP_decisions.json"
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    routed_records = [d for d in decisions if d["phase"] == "clustering" and d["subject"] == MISSING_ATTR]
    assert len(routed_records) == 1
    assert routed_records[0]["outcome"] == "routed_to_overig"

    # K#-ids stayed valid after the post-hoc patch + second ensure_codebook_ids:
    # unique, sequential, and Overig's source_attribute_ids actually resolved
    # attr_missing's real A#-id (not left empty by the force-recompute reset).
    ids = [c["code_id"] for c in cache.raw_codes]
    assert len(ids) == len(set(ids))
    assert ids == [f"K{i}" for i in range(1, len(ids) + 1)]

    attrs_by_name = {a["attribute_name"]: a for a in cache.partition_results["domain1"].attributes["facet1"]}
    assert attrs_by_name[MISSING_ATTR]["attribute_id"] in overig["source_attribute_ids"]


# =============================================================================
# Targeted test: a taxonomy attribute with ZERO idea assignments -> Overig
# =============================================================================
# Distinct from the missing-embedding case above: this attribute never
# appears in `attribute_assignments`/`idea_assignments` at all (e.g. a
# taxonomy attribute like "Eerlijke, integere bank" that step 4 defined but
# no idea was ever classified into). `missing_attributes()` cannot see it
# either — it only inspects attributes that DO appear in
# `idea_assignments.values()`. Only `collect_taxonomy_attributes(...)`
# (ground truth from the taxonomy structure itself) exposes the gap.
ZERO_GROUPS = {
    0: ["attr_z0_0", "attr_z0_1", "attr_z0_2"],
    1: ["attr_z1_0", "attr_z1_1", "attr_z1_2"],
    2: ["attr_z2_0", "attr_z2_1", "attr_z2_2"],
}
ZERO_ATTR = "attr_zero_assignments"
ZERO_CLUSTERED_ATTRS = [a for members in ZERO_GROUPS.values() for a in members]
# All-positive -> every cluster resolves "dimensional", no noise vote needed.
ZERO_VALENCE = {a: {"positive": 5, "neutral": 0, "negative": 0} for a in ZERO_CLUSTERED_ATTRS}


def _zero_vectors() -> dict:
    base = {0: np.eye(8)[0], 1: np.eye(8)[1], 2: np.eye(8)[2]}
    out = {}
    for g, members in ZERO_GROUPS.items():
        for j, attr in enumerate(members):
            v = base[g] + 0.05 * np.eye(8)[3 + j]
            out[attr] = v / np.linalg.norm(v)
    return out


def _zero_inputs() -> ExperimentInputs:
    vectors = _zero_vectors()
    partition_results = {
        "domain1": DomainResultModel(
            partition_name="domain1",
            n_labels=len(ZERO_CLUSTERED_ATTRS) + 1,
            n_batches=1,
            facets=[{"facet_name": "facet1"}],
            # ZERO_ATTR is a real taxonomy attribute (present in the
            # structure below) but is never referenced in
            # attribute_assignments — no idea was ever classified into it.
            attributes={"facet1": [{"attribute_name": a} for a in ZERO_CLUSTERED_ATTRS + [ZERO_ATTR]]},
            attribute_assignments={_idea_id(a): a for a in ZERO_CLUSTERED_ATTRS},
        ),
    }
    idea_texts = {_idea_id(a): f"statement about {a}" for a in ZERO_CLUSTERED_ATTRS}
    idea_embeddings = {_idea_id(a): vectors[a].tolist() for a in ZERO_CLUSTERED_ATTRS}
    return ExperimentInputs(
        partition_results=partition_results,
        idea_assignments={_idea_id(a): a for a in ZERO_CLUSTERED_ATTRS},
        attr_valence=ZERO_VALENCE,
        idea_texts=idea_texts,
        idea_embeddings=idea_embeddings,
        language="Dutch",
        variable_key="Q1_full",
        survey_question="Wat vindt u van dit merk?",
    )


@pytest.mark.asyncio
async def test_zero_assignment_attribute_is_routed_to_overig(monkeypatch, tmp_path):
    import json

    import pipeline.step_5_codeGenerator_experiment.assembler as assembler_mod
    from pipeline.step_5_codeGenerator.codebook_verifier import build_scorecard
    monkeypatch.setattr(assembler_mod, "CacheManager", _FakeCacheManager)

    cache = await run_from_inputs(
        inputs=_zero_inputs(),
        partition_set=_partition_set(),
        filename="survey.sav",
        llm_call=_fake_llm_naming_only,
        project_root=tmp_path,
    )

    non_overig = [c for c in cache.raw_codes if c["code_name"] != "Overig"]
    overig = cache.raw_codes[-1]
    assert overig["code_name"] == "Overig"

    # never landed in a phenomenon cluster/code ...
    for c in non_overig:
        assert ZERO_ATTR not in c["source_attributes"]
    # ... it landed in Overig instead
    assert ZERO_ATTR in overig["source_attributes"]

    # a routed_to_overig Decision record was logged, with the distinct reason
    decisions_path = tmp_path / "exports" / "codebook" / "codebook_survey_Q1_full_EXP_decisions.json"
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    routed_records = [d for d in decisions if d["phase"] == "clustering" and d["subject"] == ZERO_ATTR]
    assert len(routed_records) == 1
    assert routed_records[0]["outcome"] == "routed_to_overig"
    assert routed_records[0]["evidence"].get("reason") == "no assigned ideas"

    # the scorecard now reports 100% attribute coverage — the whole point of the fix
    scorecard = build_scorecard(cache.raw_codes, cache.partition_results, overig["code_name"])
    assert scorecard.attribute_coverage_pct == 100.0
    assert ZERO_ATTR not in scorecard.orphan_attributes
