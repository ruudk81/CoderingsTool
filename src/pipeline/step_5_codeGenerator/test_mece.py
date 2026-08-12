"""Tests voor stap 5 van step 5: MECE-afdwinging over de codeverzameling
(`mece.py`). Deterministische delen (samenvoegen, componenten, vereniging i.p.v.
som, alleen-zelfde-richting, iteratiestop) staan los van de dispatch-tests."""
import asyncio

from utils.smoothRequester import SmoothRequester

from pipeline.step_5_codeGenerator import mece
from pipeline.step_5_codeGenerator.config_codeGenerator import CodebookConfig
from pipeline.step_5_codeGenerator.consolidator import CodeShape
from pipeline.step_5_codeGenerator.prompts_mece import (
    CandidatePair, CodeCandidate, OverlapDetectionResult, OverlapVerdict,
    PairAdjudicationResult, PairVerdict,
)


def shape(key, valence, members, n_resp=10, origin="solo"):
    resp = frozenset(f"{key}R{i}" for i in range(n_resp))
    return CodeShape(key=key, members=tuple(members), valence=valence,
                     umbrella="u", resp_ids=resp, resp_pos=resp,
                     resp_neg=frozenset(), resp_neu=frozenset(), origin=origin)


def candidate(name, valence="positive", members=None, n_resp=10, indicators=("a",)):
    members = members or (f"A_{name}",)
    return CodeCandidate(name=name, definition=f"def {name}", indicators=tuple(indicators),
                         valence=valence, shape=shape(name, valence, members, n_resp))


# ---------------------------------------------------------------------------
# build_candidate_pairs — deterministisch
# ---------------------------------------------------------------------------

def test_build_candidate_pairs_keeps_a_same_valence_pair():
    verdicts = [OverlapVerdict(code="A", hardest_to_separate_from="B")]
    valence_by_name = {"A": "positive", "B": "positive"}
    pairs = mece.build_candidate_pairs(verdicts, valence_by_name)
    assert [(p.code_a, p.code_b) for p in pairs] == [("A", "B")]


def test_build_candidate_pairs_drops_cross_valence():
    verdicts = [OverlapVerdict(code="A", hardest_to_separate_from="B")]
    valence_by_name = {"A": "positive", "B": "negative"}
    pairs = mece.build_candidate_pairs(verdicts, valence_by_name)
    assert pairs == []


def test_build_candidate_pairs_drops_null():
    verdicts = [OverlapVerdict(code="A", hardest_to_separate_from=None)]
    valence_by_name = {"A": "positive"}
    pairs = mece.build_candidate_pairs(verdicts, valence_by_name)
    assert pairs == []


def test_build_candidate_pairs_drops_self_reference():
    verdicts = [OverlapVerdict(code="A", hardest_to_separate_from="A")]
    valence_by_name = {"A": "positive"}
    pairs = mece.build_candidate_pairs(verdicts, valence_by_name)
    assert pairs == []


def test_build_candidate_pairs_dedups_a_mutual_proposal():
    verdicts = [OverlapVerdict(code="A", hardest_to_separate_from="B"),
                OverlapVerdict(code="B", hardest_to_separate_from="A")]
    valence_by_name = {"A": "positive", "B": "positive"}
    pairs = mece.build_candidate_pairs(verdicts, valence_by_name)
    assert len(pairs) == 1


def test_build_candidate_pairs_ids_are_stable_across_calls():
    verdicts = [OverlapVerdict(code="A", hardest_to_separate_from="B"),
                OverlapVerdict(code="C", hardest_to_separate_from="D")]
    valence_by_name = {"A": "positive", "B": "positive", "C": "negative", "D": "negative"}
    first = mece.build_candidate_pairs(verdicts, valence_by_name)
    second = mece.build_candidate_pairs(verdicts, valence_by_name)
    assert [(p.pair_id, p.code_a, p.code_b) for p in first] == \
           [(p.pair_id, p.code_a, p.code_b) for p in second]


# ---------------------------------------------------------------------------
# merge_components — union-find
# ---------------------------------------------------------------------------

def test_merge_components_chain_collapses_to_one_group():
    pair_by_id = {1: CandidatePair(1, "A", "B"), 2: CandidatePair(2, "B", "C")}
    verdicts = [PairVerdict(pair_id=1, separation_rule="", one_dimension=True),
                PairVerdict(pair_id=2, separation_rule="", one_dimension=True)]
    components = mece.merge_components(pair_by_id, verdicts)
    assert components == [{"A", "B", "C"}]


def test_merge_components_ignores_a_pair_judged_separate():
    pair_by_id = {1: CandidatePair(1, "A", "B")}
    verdicts = [PairVerdict(pair_id=1, separation_rule="a real rule", one_dimension=False)]
    components = mece.merge_components(pair_by_id, verdicts)
    assert components == []


def test_merge_components_chain_order_independent():
    # Same chain, verdicts in the opposite order — the union-find result must
    # not depend on which pair was resolved first.
    pair_by_id = {1: CandidatePair(1, "A", "B"), 2: CandidatePair(2, "B", "C")}
    verdicts = [PairVerdict(pair_id=2, separation_rule="", one_dimension=True),
                PairVerdict(pair_id=1, separation_rule="", one_dimension=True)]
    components = mece.merge_components(pair_by_id, verdicts)
    assert components == [{"A", "B", "C"}]


# ---------------------------------------------------------------------------
# merge_candidates — vereniging, nooit som; valence-bewaking
# ---------------------------------------------------------------------------

def test_merge_candidates_unions_members_not_concatenates_duplicates():
    a = candidate("A", members=("M1", "M2"))
    b = candidate("B", members=("M2", "M3"))
    merged = mece.merge_candidates({"A", "B"}, {"A": a, "B": b}, key="X")
    assert sorted(merged.shape.members) == ["M1", "M2", "M3"]


def test_merge_candidates_resp_ids_are_a_union_not_a_sum():
    shared = frozenset({"R1", "R2", "R3"})
    a = CodeCandidate(name="A", definition="d", indicators=("i",), valence="positive",
                      shape=CodeShape(key="A", members=("M1",), valence="positive",
                                     umbrella="u", resp_ids=shared, resp_pos=shared,
                                     resp_neg=frozenset(), resp_neu=frozenset(), origin="solo"))
    b = CodeCandidate(name="B", definition="d", indicators=("i",), valence="positive",
                      shape=CodeShape(key="B", members=("M2",), valence="positive",
                                     umbrella="u", resp_ids=shared, resp_pos=shared,
                                     resp_neg=frozenset(), resp_neu=frozenset(), origin="solo"))
    merged = mece.merge_candidates({"A", "B"}, {"A": a, "B": b}, key="X")
    assert len(merged.shape.resp_ids) == 3  # niet 6


def test_merge_candidates_raises_on_cross_valence():
    a = candidate("A", valence="positive")
    b = candidate("B", valence="negative")
    try:
        mece.merge_candidates({"A", "B"}, {"A": a, "B": b}, key="X")
    except ValueError:
        return
    raise AssertionError("samenvoegen over richting heen had moeten falen")


def test_merge_candidates_sets_origin_mece_merge():
    a, b = candidate("A"), candidate("B")
    merged = mece.merge_candidates({"A", "B"}, {"A": a, "B": b}, key="X")
    assert merged.shape.origin == "mece_merge"


def test_merge_candidates_picks_the_most_populous_member_as_canonical_name():
    a = candidate("A", members=("M1",))
    b = candidate("B", members=("M1", "M2", "M3"))
    merged = mece.merge_candidates({"A", "B"}, {"A": a, "B": b}, key="X")
    assert merged.name == "B"


# ---------------------------------------------------------------------------
# apply_merges — leden ongemoeid laten, alleen componenten samenvoegen
# ---------------------------------------------------------------------------

def test_apply_merges_leaves_untouched_candidates_identical():
    a, b, c = candidate("A"), candidate("B"), candidate("C")
    result = mece.apply_merges([a, b, c], [{"A", "B"}])
    untouched = [x for x in result if x.name == "C"]
    assert untouched == [c]


def test_apply_merges_replaces_a_merged_group_with_one_candidate():
    a, b, c = candidate("A"), candidate("B"), candidate("C")
    result = mece.apply_merges([a, b, c], [{"A", "B"}])
    assert len(result) == 2
    merged = [x for x in result if x.shape.origin == "mece_merge"]
    assert len(merged) == 1
    assert sorted(merged[0].shape.members) == sorted(a.shape.members + b.shape.members)


# ---------------------------------------------------------------------------
# Dispatch — het SmoothRequester-contract, zoals test_relations.py
# ---------------------------------------------------------------------------

def test_resolve_overlap_detection_sends_a_one_element_list_of_dicts(monkeypatch):
    captured = {}

    async def fake_process_all(self, tasks, prepare_fn, parse_fn, fallback_fn=None):
        captured["tasks"] = tasks
        call_params = prepare_fn(tasks[0])
        assert "prompt" in call_params
        assert "response_model" in call_params
        return [OverlapDetectionResult(verdicts=[])]

    monkeypatch.setattr(SmoothRequester, "process_all", fake_process_all)

    candidates = [candidate("A")]
    result = asyncio.run(mece.resolve_overlap_detection(candidates, CodebookConfig()))

    tasks = captured["tasks"]
    assert isinstance(tasks, list) and len(tasks) == 1 and isinstance(tasks[0], dict)
    assert result.verdicts == []


def test_resolve_overlap_detection_returns_none_when_the_call_fails(monkeypatch):
    async def fake_process_all(self, tasks, prepare_fn, parse_fn, fallback_fn=None):
        return [fallback_fn(tasks[0], "boom")]

    monkeypatch.setattr(SmoothRequester, "process_all", fake_process_all)

    result = asyncio.run(mece.resolve_overlap_detection([candidate("A")], CodebookConfig()))
    assert result is None


def test_resolve_pair_adjudication_sends_a_one_element_list_of_dicts(monkeypatch):
    captured = {}

    async def fake_process_all(self, tasks, prepare_fn, parse_fn, fallback_fn=None):
        captured["tasks"] = tasks
        call_params = prepare_fn(tasks[0])
        assert "prompt" in call_params
        assert "response_model" in call_params
        return [PairAdjudicationResult(verdicts=[])]

    monkeypatch.setattr(SmoothRequester, "process_all", fake_process_all)

    pairs = [CandidatePair(1, "A", "B")]
    candidate_by_name = {"A": candidate("A"), "B": candidate("B")}
    result = asyncio.run(mece.resolve_pair_adjudication(pairs, candidate_by_name, CodebookConfig()))

    tasks = captured["tasks"]
    assert isinstance(tasks, list) and len(tasks) == 1 and isinstance(tasks[0], dict)
    assert result.verdicts == []


def test_resolve_pair_adjudication_returns_none_when_the_call_fails(monkeypatch):
    async def fake_process_all(self, tasks, prepare_fn, parse_fn, fallback_fn=None):
        return [fallback_fn(tasks[0], "boom")]

    monkeypatch.setattr(SmoothRequester, "process_all", fake_process_all)

    pairs = [CandidatePair(1, "A", "B")]
    candidate_by_name = {"A": candidate("A"), "B": candidate("B")}
    result = asyncio.run(mece.resolve_pair_adjudication(pairs, candidate_by_name, CodebookConfig()))
    assert result is None


# ---------------------------------------------------------------------------
# enforce_mece — orkestratie: rondes, stop, cap
# ---------------------------------------------------------------------------

class FakeLog:
    def __init__(self):
        self.calls = []

    def add(self, **kwargs):
        self.calls.append(kwargs)


def test_enforce_mece_merges_and_stops_when_fewer_than_two_candidates_remain(monkeypatch):
    # Only A and B exist, so merging them leaves a single candidate — no round
    # can ever find a pair after that. The loop stops without spending another
    # call on a foregone conclusion (same guard as the "fewer than two to
    # start with" case).
    calls = {"overlap": 0, "pair": 0}

    async def fake_overlap(candidates, config, *a, **kw):
        calls["overlap"] += 1
        return OverlapDetectionResult(verdicts=[
            OverlapVerdict(code="A", hardest_to_separate_from="B")
        ])

    async def fake_pair(pairs, candidate_by_name, config, *a, **kw):
        calls["pair"] += 1
        return PairAdjudicationResult(verdicts=[
            PairVerdict(pair_id=pairs[0].pair_id, separation_rule="", one_dimension=True)
        ])

    monkeypatch.setattr(mece, "resolve_overlap_detection", fake_overlap)
    monkeypatch.setattr(mece, "resolve_pair_adjudication", fake_pair)

    candidates = [candidate("A"), candidate("B")]
    result = asyncio.run(mece.enforce_mece(candidates, CodebookConfig()))

    assert len(result) == 1
    assert result[0].shape.origin == "mece_merge"
    assert calls["overlap"] == 1
    assert calls["pair"] == 1


def test_enforce_mece_stops_immediately_when_pass_a_finds_nothing(monkeypatch):
    calls = {"overlap": 0, "pair": 0}

    async def fake_overlap(candidates, config, *a, **kw):
        calls["overlap"] += 1
        return OverlapDetectionResult(verdicts=[])

    async def fake_pair(pairs, candidate_by_name, config, *a, **kw):
        calls["pair"] += 1
        return PairAdjudicationResult(verdicts=[])

    monkeypatch.setattr(mece, "resolve_overlap_detection", fake_overlap)
    monkeypatch.setattr(mece, "resolve_pair_adjudication", fake_pair)

    candidates = [candidate("A"), candidate("B")]
    result = asyncio.run(mece.enforce_mece(candidates, CodebookConfig()))

    assert [c.name for c in result] == ["A", "B"]
    assert calls["overlap"] == 1
    assert calls["pair"] == 0


def test_enforce_mece_stops_when_no_pair_is_judged_one_dimension(monkeypatch):
    async def fake_overlap(candidates, config, *a, **kw):
        return OverlapDetectionResult(verdicts=[
            OverlapVerdict(code="A", hardest_to_separate_from="B")
        ])

    async def fake_pair(pairs, candidate_by_name, config, *a, **kw):
        return PairAdjudicationResult(verdicts=[
            PairVerdict(pair_id=pairs[0].pair_id, separation_rule="a real rule", one_dimension=False)
        ])

    monkeypatch.setattr(mece, "resolve_overlap_detection", fake_overlap)
    monkeypatch.setattr(mece, "resolve_pair_adjudication", fake_pair)

    candidates = [candidate("A"), candidate("B")]
    result = asyncio.run(mece.enforce_mece(candidates, CodebookConfig()))
    assert sorted(c.name for c in result) == ["A", "B"]


def test_enforce_mece_caps_at_max_rounds(monkeypatch):
    # Five same-valence singletons; the fakes always pair up whichever two
    # names come first in the current candidate set and merge them — an
    # ever-available merge, so the cap (not "no more merges") is what stops
    # the loop.
    calls = {"overlap": 0}

    async def fake_overlap(candidates, config, *a, **kw):
        calls["overlap"] += 1
        names = sorted(c.name for c in candidates)
        return OverlapDetectionResult(verdicts=[
            OverlapVerdict(code=names[0], hardest_to_separate_from=names[1])
        ])

    async def fake_pair(pairs, candidate_by_name, config, *a, **kw):
        return PairAdjudicationResult(verdicts=[
            PairVerdict(pair_id=pairs[0].pair_id, separation_rule="", one_dimension=True)
        ])

    monkeypatch.setattr(mece, "resolve_overlap_detection", fake_overlap)
    monkeypatch.setattr(mece, "resolve_pair_adjudication", fake_pair)

    config = CodebookConfig()
    assert config.mece_max_rounds == 3
    candidates = [candidate(n) for n in ["P1", "P2", "P3", "P4", "P5"]]
    result = asyncio.run(mece.enforce_mece(candidates, config))

    assert calls["overlap"] == 3
    assert len(result) == 5 - 3  # one merge per round, capped at 3 rounds


def test_enforce_mece_never_merges_across_valence(monkeypatch):
    async def fake_overlap(candidates, config, *a, **kw):
        return OverlapDetectionResult(verdicts=[
            OverlapVerdict(code="A", hardest_to_separate_from="B")
        ])

    async def fake_pair(pairs, candidate_by_name, config, *a, **kw):
        raise AssertionError("pass B should never run: the pair is cross-valence")

    monkeypatch.setattr(mece, "resolve_overlap_detection", fake_overlap)
    monkeypatch.setattr(mece, "resolve_pair_adjudication", fake_pair)

    candidates = [candidate("A", valence="positive"), candidate("B", valence="negative")]
    result = asyncio.run(mece.enforce_mece(candidates, CodebookConfig()))
    assert sorted(c.name for c in result) == ["A", "B"]


def test_enforce_mece_logs_each_round(monkeypatch):
    async def fake_overlap(candidates, config, *a, **kw):
        return OverlapDetectionResult(verdicts=[])

    async def fake_pair(pairs, candidate_by_name, config, *a, **kw):
        raise AssertionError("no candidate pairs, pass B should not run")

    monkeypatch.setattr(mece, "resolve_overlap_detection", fake_overlap)
    monkeypatch.setattr(mece, "resolve_pair_adjudication", fake_pair)

    log = FakeLog()
    asyncio.run(mece.enforce_mece([candidate("A"), candidate("B")], CodebookConfig(), log=log))
    assert len(log.calls) == 1
    assert log.calls[0]["action"] == "MECE_ROUND"
    assert log.calls[0]["round"] == 1
    assert log.calls[0]["merges"] == 0


def test_enforce_mece_returns_candidates_unchanged_without_a_call_when_fewer_than_two(monkeypatch):
    called = False

    async def fake_overlap(candidates, config, *a, **kw):
        nonlocal called
        called = True
        return OverlapDetectionResult(verdicts=[])

    monkeypatch.setattr(mece, "resolve_overlap_detection", fake_overlap)

    result = asyncio.run(mece.enforce_mece([candidate("A")], CodebookConfig()))
    assert [c.name for c in result] == ["A"]
    assert called is False
