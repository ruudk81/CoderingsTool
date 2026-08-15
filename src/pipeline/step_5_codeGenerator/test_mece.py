"""Tests for step 5 of step 5: MECE enforcement across the code set
(`mece.py`). The deterministic parts (merging, components, union rather than
sum, same-direction-only, iteration stop, scoring, threshold decision) stand
apart from the dispatch tests."""
import asyncio

from utils.smoothRequester import SmoothRequester

from pipeline.step_5_codeGenerator import mece
from pipeline.step_5_codeGenerator.config_codeGenerator import CodebookConfig
from pipeline.step_5_codeGenerator.consolidator import CodeShape
from pipeline.step_5_codeGenerator.prompts_mece import (
    CandidatePair, CodeCandidate, IdeaAssignment, OverlapDetectionResult, OverlapVerdict,
    ProbeResult,
)
from pipeline.step_5_codeGenerator.taxonomy_input import IdeaUnit


def shape(key, valence, members, n_resp=10, origin="solo"):
    resp = frozenset(f"{key}R{i}" for i in range(n_resp))
    return CodeShape(key=key, members=tuple(members), valence=valence,
                     umbrella="u", resp_ids=resp, resp_pos=resp,
                     resp_neg=frozenset(), resp_neu=frozenset(), origin=origin)


def candidate(name, valence="positive", members=None, n_resp=10, indicators=("a",)):
    members = members or (f"A_{name}",)
    return CodeCandidate(name=name, definition=f"def {name}", indicators=tuple(indicators),
                         valence=valence, shape=shape(name, valence, members, n_resp))


def idea(idea_id, attribute_id, instance="tekst", interpretation="", respondent_id=None):
    return IdeaUnit(idea_id=idea_id, respondent_id=respondent_id or f"R{idea_id}",
                    attribute_id=attribute_id, valence="+", instance=instance,
                    interpretation=interpretation)


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
    verdicts = [mece.PairVerdict(pair_id=1, accuracy=0.3, both_rate=0.0, one_dimension=True),
                mece.PairVerdict(pair_id=2, accuracy=0.3, both_rate=0.0, one_dimension=True)]
    components = mece.merge_components(pair_by_id, verdicts)
    assert components == [{"A", "B", "C"}]


def test_merge_components_ignores_a_pair_judged_separate():
    pair_by_id = {1: CandidatePair(1, "A", "B")}
    verdicts = [mece.PairVerdict(pair_id=1, accuracy=0.95, both_rate=0.0, one_dimension=False)]
    components = mece.merge_components(pair_by_id, verdicts)
    assert components == []


def test_merge_components_chain_order_independent():
    # Same chain, verdicts in the opposite order — the union-find result must
    # not depend on which pair was resolved first.
    pair_by_id = {1: CandidatePair(1, "A", "B"), 2: CandidatePair(2, "B", "C")}
    verdicts = [mece.PairVerdict(pair_id=2, accuracy=0.3, both_rate=0.0, one_dimension=True),
                mece.PairVerdict(pair_id=1, accuracy=0.3, both_rate=0.0, one_dimension=True)]
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


# ---------------------------------------------------------------------------
# build_pair_probe — deterministisch: bemonstering, schudden, waarheidssleutel
# ---------------------------------------------------------------------------

def test_build_pair_probe_pools_ideas_from_both_sides():
    a, b = candidate("A", members=("Attr1",)), candidate("B", members=("Attr2",))
    candidate_by_name = {"A": a, "B": b}
    idea_units = {"Attr1": [idea(f"ia{i}", "Attr1") for i in range(3)],
                 "Attr2": [idea(f"ib{i}", "Attr2") for i in range(3)]}
    probe = mece.build_pair_probe(CandidatePair(1, "A", "B"), candidate_by_name, idea_units, 8)
    assert len(probe.ideas) == 6
    assert len(probe.truth) == 6
    assert list(probe.truth.values()).count("A") == 3
    assert list(probe.truth.values()).count("B") == 3


def test_build_pair_probe_caps_at_ideas_per_code():
    a, b = candidate("A", members=("Attr1",)), candidate("B", members=("Attr2",))
    candidate_by_name = {"A": a, "B": b}
    idea_units = {"Attr1": [idea(f"ia{i}", "Attr1") for i in range(10)],
                 "Attr2": [idea(f"ib{i}", "Attr2") for i in range(10)]}
    probe = mece.build_pair_probe(CandidatePair(1, "A", "B"), candidate_by_name, idea_units, 2)
    assert len(probe.ideas) == 4


def test_build_pair_probe_returns_none_when_one_side_has_no_ideas():
    a, b = candidate("A", members=("Attr1",)), candidate("B", members=("Attr2",))
    candidate_by_name = {"A": a, "B": b}
    idea_units = {"Attr1": [idea("ia0", "Attr1")]}  # Attr2 heeft niets
    probe = mece.build_pair_probe(CandidatePair(1, "A", "B"), candidate_by_name, idea_units, 8)
    assert probe is None


def test_build_pair_probe_is_deterministic_across_calls():
    a, b = candidate("A", members=("Attr1",)), candidate("B", members=("Attr2",))
    candidate_by_name = {"A": a, "B": b}
    idea_units = {"Attr1": [idea(f"ia{i}", "Attr1") for i in range(4)],
                 "Attr2": [idea(f"ib{i}", "Attr2") for i in range(4)]}
    pair = CandidatePair(1, "A", "B")
    first = mece.build_pair_probe(pair, candidate_by_name, idea_units, 8)
    second = mece.build_pair_probe(pair, candidate_by_name, idea_units, 8)
    assert [i.text for i in first.ideas] == [i.text for i in second.ideas]
    assert first.truth == second.truth


# ---------------------------------------------------------------------------
# score_probe — deterministic: never the model's own claim
# ---------------------------------------------------------------------------

def test_score_probe_all_correct_is_accuracy_one():
    truth = {1: "A", 2: "B"}
    assignments = [IdeaAssignment(idea_ref=1, assigned_to="A"),
                   IdeaAssignment(idea_ref=2, assigned_to="B")]
    score = mece.score_probe(assignments, truth)
    assert score.accuracy == 1.0
    assert score.both_rate == 0.0


def test_score_probe_all_wrong_is_accuracy_zero():
    truth = {1: "A", 2: "B"}
    assignments = [IdeaAssignment(idea_ref=1, assigned_to="B"),
                   IdeaAssignment(idea_ref=2, assigned_to="A")]
    assert mece.score_probe(assignments, truth).accuracy == 0.0


def test_score_probe_missing_assignment_is_excluded_from_accuracy():
    # idea_ref 2 has no assignment at all: it is not "put on a side", so it
    # does not enter the accuracy denominator (unlike a BOTH answer, which
    # is also excluded but does count towards both_rate below).
    truth = {1: "A", 2: "B"}
    assignments = [IdeaAssignment(idea_ref=1, assigned_to="A")]
    assert mece.score_probe(assignments, truth).accuracy == 1.0


def test_score_probe_duplicate_ref_lets_the_last_answer_win():
    truth = {1: "A"}
    assignments = [IdeaAssignment(idea_ref=1, assigned_to="B"),
                   IdeaAssignment(idea_ref=1, assigned_to="A")]
    assert mece.score_probe(assignments, truth).accuracy == 1.0


def test_score_probe_empty_truth_is_zero_not_a_crash():
    score = mece.score_probe([], {})
    assert score.accuracy == 0.0
    assert score.both_rate == 0.0


def test_score_probe_accuracy_excludes_both_answers():
    # Two ideas: one correctly sided, one answered BOTH. Accuracy is over the
    # sided idea only (1/1 = 1.0), not diluted by the BOTH answer.
    truth = {1: "A", 2: "B"}
    assignments = [IdeaAssignment(idea_ref=1, assigned_to="A"),
                   IdeaAssignment(idea_ref=2, assigned_to="BOTH")]
    score = mece.score_probe(assignments, truth)
    assert score.accuracy == 1.0


def test_score_probe_both_rate_counts_share_of_all_probed_ideas():
    truth = {1: "A", 2: "B", 3: "A", 4: "B"}
    assignments = [IdeaAssignment(idea_ref=1, assigned_to="BOTH"),
                   IdeaAssignment(idea_ref=2, assigned_to="BOTH"),
                   IdeaAssignment(idea_ref=3, assigned_to="A"),
                   IdeaAssignment(idea_ref=4, assigned_to="B")]
    score = mece.score_probe(assignments, truth)
    assert score.both_rate == 0.5
    assert score.accuracy == 1.0  # de twee gezijde ideeën zijn allebei juist


def test_score_probe_all_both_gives_accuracy_zero_and_both_rate_one():
    # No idea was ever put on a side -> the accuracy denominator is empty ->
    # 0.0, not a crash; both_rate is 1.0.
    truth = {1: "A", 2: "B"}
    assignments = [IdeaAssignment(idea_ref=1, assigned_to="BOTH"),
                   IdeaAssignment(idea_ref=2, assigned_to="BOTH")]
    score = mece.score_probe(assignments, truth)
    assert score.accuracy == 0.0
    assert score.both_rate == 1.0


# ---------------------------------------------------------------------------
# is_one_dimension — de drempelbeslissing: OF, geen EN
# ---------------------------------------------------------------------------

def test_is_one_dimension_false_when_neither_threshold_fires():
    assert mece.is_one_dimension(0.9, 0.10, 0.70, 0.30) is False


def test_is_one_dimension_true_at_accuracy_threshold():
    assert mece.is_one_dimension(0.70, 0.0, 0.70, 0.30) is True


def test_is_one_dimension_true_when_only_accuracy_threshold_fires():
    # Unseparable (low accuracy) but nobody said BOTH — still a merge.
    assert mece.is_one_dimension(0.4, 0.0, 0.70, 0.30) is True


def test_is_one_dimension_true_when_only_both_rate_threshold_fires():
    # Perfectly separable on wording (accuracy 1.0, well above threshold) but
    # a third of ideas genuinely fit either side — still a merge. This is
    # the sustainability case: lexically distinguishable, same dimension.
    assert mece.is_one_dimension(1.0, 0.30, 0.70, 0.30) is True


def test_is_one_dimension_true_when_both_thresholds_fire_together():
    assert mece.is_one_dimension(0.5, 0.5, 0.70, 0.30) is True


# ---------------------------------------------------------------------------
# Dispatch — het SmoothRequester-contract, zoals test_relations.py
# ---------------------------------------------------------------------------

def test_resolve_pair_probes_sends_one_task_per_pair(monkeypatch):
    captured = {}

    async def fake_process_all(self, tasks, prepare_fn, parse_fn, fallback_fn=None):
        captured["tasks"] = tasks
        results = []
        for task in tasks:
            call_params = prepare_fn(task)
            assert "prompt" in call_params
            assert "response_model" in call_params
            results.append(ProbeResult(assignments=[]))
        return results

    monkeypatch.setattr(SmoothRequester, "process_all", fake_process_all)

    a, b = candidate("A", members=("Attr1",)), candidate("B", members=("Attr2",))
    candidate_by_name = {"A": a, "B": b}
    idea_units = {"Attr1": [idea("ia0", "Attr1")], "Attr2": [idea("ib0", "Attr2")]}
    pairs = [CandidatePair(1, "A", "B")]

    verdicts = asyncio.run(
        mece.resolve_pair_probes(pairs, candidate_by_name, idea_units, CodebookConfig())
    )

    assert len(captured["tasks"]) == 1
    # empty assignments -> accuracy 0.0 -> op/onder de standaarddrempel (0.70)
    assert verdicts[1].accuracy == 0.0
    assert verdicts[1].one_dimension is True


def test_resolve_pair_probes_skips_pairs_without_material_and_makes_no_call(monkeypatch):
    called = False

    async def fake_process_all(self, tasks, prepare_fn, parse_fn, fallback_fn=None):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(SmoothRequester, "process_all", fake_process_all)

    a, b = candidate("A", members=("Attr1",)), candidate("B", members=("Attr2",))
    candidate_by_name = {"A": a, "B": b}
    idea_units = {"Attr1": [idea("ia0", "Attr1")]}  # Attr2 heeft niets
    pairs = [CandidatePair(1, "A", "B")]

    verdicts = asyncio.run(
        mece.resolve_pair_probes(pairs, candidate_by_name, idea_units, CodebookConfig())
    )
    assert verdicts == {}
    assert called is False


def test_resolve_pair_probes_returns_empty_dict_when_the_call_fails(monkeypatch):
    async def fake_process_all(self, tasks, prepare_fn, parse_fn, fallback_fn=None):
        return [fallback_fn(t, "boom") for t in tasks]

    monkeypatch.setattr(SmoothRequester, "process_all", fake_process_all)

    a, b = candidate("A", members=("Attr1",)), candidate("B", members=("Attr2",))
    candidate_by_name = {"A": a, "B": b}
    idea_units = {"Attr1": [idea("ia0", "Attr1")], "Attr2": [idea("ib0", "Attr2")]}
    pairs = [CandidatePair(1, "A", "B")]

    verdicts = asyncio.run(
        mece.resolve_pair_probes(pairs, candidate_by_name, idea_units, CodebookConfig())
    )
    assert verdicts == {}


def test_resolve_pair_probes_scores_a_perfect_response_as_separable(monkeypatch):
    captured = {}

    async def fake_process_all(self, tasks, prepare_fn, parse_fn, fallback_fn=None):
        captured["tasks"] = tasks
        results = []
        for task in tasks:
            probe = task["probe"]
            assignments = [IdeaAssignment(idea_ref=ref, assigned_to=code)
                           for ref, code in probe.truth.items()]
            results.append(ProbeResult(assignments=assignments))
        return results

    monkeypatch.setattr(SmoothRequester, "process_all", fake_process_all)

    a, b = candidate("A", members=("Attr1",)), candidate("B", members=("Attr2",))
    candidate_by_name = {"A": a, "B": b}
    idea_units = {"Attr1": [idea("ia0", "Attr1")], "Attr2": [idea("ib0", "Attr2")]}
    pairs = [CandidatePair(1, "A", "B")]

    verdicts = asyncio.run(
        mece.resolve_pair_probes(pairs, candidate_by_name, idea_units, CodebookConfig())
    )
    # Pins the SmoothRequester.process_all contract (List[Dict]) so a bare-
    # object task list — which crashes _execute_task on task.get(), then
    # crashes the error handler on the same thing, hiding the real cause —
    # cannot creep back in unnoticed.
    assert captured["tasks"] and all(isinstance(t, dict) for t in captured["tasks"])
    assert verdicts[1].accuracy == 1.0
    assert verdicts[1].both_rate == 0.0
    assert verdicts[1].one_dimension is False


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

    async def fake_pair(pairs, candidate_by_name, idea_units_by_attribute, config, *a, **kw):
        calls["pair"] += 1
        return {pairs[0].pair_id: mece.PairVerdict(pair_id=pairs[0].pair_id, accuracy=0.3, both_rate=0.0, one_dimension=True)}

    monkeypatch.setattr(mece, "resolve_overlap_detection", fake_overlap)
    monkeypatch.setattr(mece, "resolve_pair_probes", fake_pair)

    candidates = [candidate("A"), candidate("B")]
    result = asyncio.run(mece.enforce_mece(candidates, {}, CodebookConfig()))

    assert len(result) == 1
    assert result[0].shape.origin == "mece_merge"
    assert calls["overlap"] == 1
    assert calls["pair"] == 1


def test_enforce_mece_stops_immediately_when_pass_a_finds_nothing(monkeypatch):
    calls = {"overlap": 0, "pair": 0}

    async def fake_overlap(candidates, config, *a, **kw):
        calls["overlap"] += 1
        return OverlapDetectionResult(verdicts=[])

    async def fake_pair(pairs, candidate_by_name, idea_units_by_attribute, config, *a, **kw):
        calls["pair"] += 1
        return {}

    monkeypatch.setattr(mece, "resolve_overlap_detection", fake_overlap)
    monkeypatch.setattr(mece, "resolve_pair_probes", fake_pair)

    candidates = [candidate("A"), candidate("B")]
    result = asyncio.run(mece.enforce_mece(candidates, {}, CodebookConfig()))

    assert [c.name for c in result] == ["A", "B"]
    assert calls["overlap"] == 1
    assert calls["pair"] == 0


def test_enforce_mece_stops_when_no_pair_is_judged_one_dimension(monkeypatch):
    async def fake_overlap(candidates, config, *a, **kw):
        return OverlapDetectionResult(verdicts=[
            OverlapVerdict(code="A", hardest_to_separate_from="B")
        ])

    async def fake_pair(pairs, candidate_by_name, idea_units_by_attribute, config, *a, **kw):
        return {pairs[0].pair_id: mece.PairVerdict(pair_id=pairs[0].pair_id, accuracy=0.95, both_rate=0.0, one_dimension=False)}

    monkeypatch.setattr(mece, "resolve_overlap_detection", fake_overlap)
    monkeypatch.setattr(mece, "resolve_pair_probes", fake_pair)

    candidates = [candidate("A"), candidate("B")]
    result = asyncio.run(mece.enforce_mece(candidates, {}, CodebookConfig()))
    assert sorted(c.name for c in result) == ["A", "B"]


def test_enforce_mece_caps_at_max_rounds(monkeypatch):
    # Eight same-valence singletons; the fakes always pair up whichever two
    # names come first in the current candidate set and merge them — an
    # ever-available merge, so the cap (not "no more merges", and not running
    # out of candidates: 8 - 6 = 2 still mergeable) is what stops the loop.
    calls = {"overlap": 0}

    async def fake_overlap(candidates, config, *a, **kw):
        calls["overlap"] += 1
        names = sorted(c.name for c in candidates)
        return OverlapDetectionResult(verdicts=[
            OverlapVerdict(code=names[0], hardest_to_separate_from=names[1])
        ])

    async def fake_pair(pairs, candidate_by_name, idea_units_by_attribute, config, *a, **kw):
        return {pairs[0].pair_id: mece.PairVerdict(pair_id=pairs[0].pair_id, accuracy=0.3, both_rate=0.0, one_dimension=True)}

    monkeypatch.setattr(mece, "resolve_overlap_detection", fake_overlap)
    monkeypatch.setattr(mece, "resolve_pair_probes", fake_pair)

    config = CodebookConfig()
    assert config.mece_max_rounds == 6
    candidates = [candidate(n) for n in ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]]
    result = asyncio.run(mece.enforce_mece(candidates, {}, config))

    assert calls["overlap"] == 6
    assert len(result) == 8 - 6  # one merge per round, capped at 6 rounds


def test_enforce_mece_never_merges_across_valence(monkeypatch):
    async def fake_overlap(candidates, config, *a, **kw):
        return OverlapDetectionResult(verdicts=[
            OverlapVerdict(code="A", hardest_to_separate_from="B")
        ])

    async def fake_pair(pairs, candidate_by_name, idea_units_by_attribute, config, *a, **kw):
        raise AssertionError("pass B should never run: the pair is cross-valence")

    monkeypatch.setattr(mece, "resolve_overlap_detection", fake_overlap)
    monkeypatch.setattr(mece, "resolve_pair_probes", fake_pair)

    candidates = [candidate("A", valence="positive"), candidate("B", valence="negative")]
    result = asyncio.run(mece.enforce_mece(candidates, {}, CodebookConfig()))
    assert sorted(c.name for c in result) == ["A", "B"]


def test_enforce_mece_logs_no_pairs_reason(monkeypatch):
    async def fake_overlap(candidates, config, *a, **kw):
        return OverlapDetectionResult(verdicts=[])

    async def fake_pair(pairs, candidate_by_name, idea_units_by_attribute, config, *a, **kw):
        raise AssertionError("no candidate pairs, pass B should not run")

    monkeypatch.setattr(mece, "resolve_overlap_detection", fake_overlap)
    monkeypatch.setattr(mece, "resolve_pair_probes", fake_pair)

    log = FakeLog()
    asyncio.run(mece.enforce_mece([candidate("A"), candidate("B")], {}, CodebookConfig(), log=log))
    assert len(log.calls) == 1
    assert log.calls[0]["action"] == "MECE_ROUND"
    assert log.calls[0]["round"] == 1
    assert log.calls[0]["merges"] == 0
    assert log.calls[0]["reason"] == "no_pairs"


def test_enforce_mece_logs_detection_failed_reason_distinctly(monkeypatch):
    async def fake_overlap(candidates, config, *a, **kw):
        return None

    monkeypatch.setattr(mece, "resolve_overlap_detection", fake_overlap)

    log = FakeLog()
    asyncio.run(mece.enforce_mece([candidate("A"), candidate("B")], {}, CodebookConfig(), log=log))
    assert log.calls[0]["reason"] == "detection_failed"
    assert log.calls[0]["merges"] == 0


def test_enforce_mece_logs_probe_failed_reason_with_pairs_found(monkeypatch):
    async def fake_overlap(candidates, config, *a, **kw):
        return OverlapDetectionResult(verdicts=[
            OverlapVerdict(code="A", hardest_to_separate_from="B")
        ])

    async def fake_pair(pairs, candidate_by_name, idea_units_by_attribute, config, *a, **kw):
        return {}

    monkeypatch.setattr(mece, "resolve_overlap_detection", fake_overlap)
    monkeypatch.setattr(mece, "resolve_pair_probes", fake_pair)

    log = FakeLog()
    asyncio.run(mece.enforce_mece([candidate("A"), candidate("B")], {}, CodebookConfig(), log=log))
    assert log.calls[0]["reason"] == "probe_failed"
    assert log.calls[0]["pairs_found"] == 1
    assert log.calls[0]["merges"] == 0


def test_enforce_mece_logs_no_components_reason_with_stats(monkeypatch):
    async def fake_overlap(candidates, config, *a, **kw):
        return OverlapDetectionResult(verdicts=[
            OverlapVerdict(code="A", hardest_to_separate_from="B")
        ])

    async def fake_pair(pairs, candidate_by_name, idea_units_by_attribute, config, *a, **kw):
        return {pairs[0].pair_id: mece.PairVerdict(pair_id=pairs[0].pair_id, accuracy=0.9, both_rate=0.0, one_dimension=False)}

    monkeypatch.setattr(mece, "resolve_overlap_detection", fake_overlap)
    monkeypatch.setattr(mece, "resolve_pair_probes", fake_pair)

    log = FakeLog()
    asyncio.run(mece.enforce_mece([candidate("A"), candidate("B")], {}, CodebookConfig(), log=log))
    assert log.calls[0]["reason"] == "no_components"
    assert log.calls[0]["pairs_found"] == 1
    assert log.calls[0]["pairs_probed"] == 1
    assert log.calls[0]["mean_accuracy"] == 0.9
    assert log.calls[0]["merges"] == 0


def test_enforce_mece_logs_a_merge_round_with_stats_and_no_reason(monkeypatch):
    async def fake_overlap(candidates, config, *a, **kw):
        return OverlapDetectionResult(verdicts=[
            OverlapVerdict(code="A", hardest_to_separate_from="B")
        ])

    async def fake_pair(pairs, candidate_by_name, idea_units_by_attribute, config, *a, **kw):
        return {pairs[0].pair_id: mece.PairVerdict(pair_id=pairs[0].pair_id, accuracy=0.3, both_rate=0.0, one_dimension=True)}

    monkeypatch.setattr(mece, "resolve_overlap_detection", fake_overlap)
    monkeypatch.setattr(mece, "resolve_pair_probes", fake_pair)

    log = FakeLog()
    asyncio.run(mece.enforce_mece([candidate("A"), candidate("B")], {}, CodebookConfig(), log=log))
    assert log.calls[0]["reason"] is None
    assert log.calls[0]["merges"] == 1
    assert log.calls[0]["mean_accuracy"] == 0.3
    assert log.calls[0]["pairs_found"] == 1
    assert log.calls[0]["pairs_probed"] == 1
    assert log.calls[0]["pairs"] == [
        {"code_a": "A", "code_b": "B", "accuracy": 0.3, "both_rate": 0.0, "merged": True}
    ]


def test_enforce_mece_returns_candidates_unchanged_without_a_call_when_fewer_than_two(monkeypatch):
    called = False

    async def fake_overlap(candidates, config, *a, **kw):
        nonlocal called
        called = True
        return OverlapDetectionResult(verdicts=[])

    monkeypatch.setattr(mece, "resolve_overlap_detection", fake_overlap)

    result = asyncio.run(mece.enforce_mece([candidate("A")], {}, CodebookConfig()))
    assert [c.name for c in result] == ["A"]
    assert called is False
