"""Tests voor de post-mortem: kandidaatselectie en het toepassen van splitsingen."""
import pytest

from pipeline.step_5_codeGenerator.concept_inventory import Concept
from pipeline.step_5_codeGenerator.v2.grouping import Group
from pipeline.step_5_codeGenerator.v2.postmortem import (
    SplitVerdict, apply_splits, select_candidates,
)
from pipeline.step_5_codeGenerator.v2.stability import measure_stability


def concept(attribute_id, n_resp, prefix="r"):
    resp = frozenset(f"{prefix}{i}" for i in range(n_resp))
    return Concept(attribute_id=attribute_id, name=f"naam-{attribute_id}",
                   definition="d", domain="D", facet="F", n_iu=n_resp,
                   resp_ids=resp, resp_pos=resp, resp_neg=frozenset(),
                   resp_neu=frozenset())


def group(*ids, name="G"):
    return Group(member_ids=tuple(ids), proposed_name=name, explanation="e")


def stable_report(attribute_ids):
    """Twee identieke runs waarin niets samen zit: alles stabiel."""
    return measure_stability([[group(a) for a in attribute_ids]] * 2, attribute_ids)


# --- selectie ---------------------------------------------------------------

def test_oversized_group_is_a_candidate():
    concepts = [concept("A1", 300), concept("A2", 20, prefix="q")]
    groups = [group("A1", "A2", name="Groot"), group("A2", name="Klein")]

    picked = select_candidates([groups[0]], concepts, stable_report(["A1", "A2"]),
                               n_respondents=1000, share_threshold=0.20)

    assert [g.proposed_name for g in picked] == ["Groot"]


def test_group_below_the_share_threshold_is_left_alone():
    concepts = [concept("A1", 50)]

    picked = select_candidates([group("A1")], concepts, stable_report(["A1"]),
                               n_respondents=1000, share_threshold=0.20)

    assert picked == []


def test_group_containing_an_unstable_pair_is_a_candidate_however_small():
    concepts = [concept("A1", 10), concept("A2", 10, prefix="q")]
    runs = [[group("A1", "A2")], [group("A1"), group("A2")]]
    report = measure_stability(runs, ["A1", "A2"])

    picked = select_candidates([group("A1", "A2")], concepts, report,
                               n_respondents=1000, share_threshold=0.20)

    assert len(picked) == 1


def test_solo_group_is_never_a_candidate_however_large():
    """Een groep van één attribuut valt niet te splitsen langs attribuutgrenzen."""
    concepts = [concept("A1", 900)]

    picked = select_candidates([group("A1")], concepts, stable_report(["A1"]),
                               n_respondents=1000, share_threshold=0.20)

    assert picked == []


def test_respondents_are_unioned_across_members_not_summed():
    """Twee attributen met dezelfde respondenten zijn samen niet twee keer zo groot."""
    shared = frozenset({"r1", "r2"})
    concepts = [
        Concept(attribute_id="A1", name="a", definition="d", domain="D", facet="F",
                n_iu=2, resp_ids=shared, resp_pos=shared, resp_neg=frozenset(),
                resp_neu=frozenset()),
        Concept(attribute_id="A2", name="b", definition="d", domain="D", facet="F",
                n_iu=2, resp_ids=shared, resp_pos=shared, resp_neg=frozenset(),
                resp_neu=frozenset()),
    ]

    picked = select_candidates([group("A1", "A2")], concepts, stable_report(["A1", "A2"]),
                               n_respondents=10, share_threshold=0.30)

    assert picked == []


# --- toepassen --------------------------------------------------------------

def test_split_replaces_the_group_with_its_parts():
    groups = [group("A1", "A2", "A3", name="Breed"), group("A4", name="Rust")]
    verdicts = [SplitVerdict(group_name="Breed", parts=(("A1", "A2"), ("A3",)))]

    out, log = apply_splits(groups, verdicts)

    assert [g.member_ids for g in out] == [("A1", "A2"), ("A3",), ("A4",)]
    assert log[0]["action"] == "POSTMORTEM_SPLIT"


def test_keep_verdict_leaves_the_group_untouched():
    groups = [group("A1", "A2", name="Breed")]
    verdicts = [SplitVerdict(group_name="Breed", parts=())]

    out, log = apply_splits(groups, verdicts)

    assert out == groups
    assert log == []


def test_a_part_that_drops_an_attribute_is_rejected_whole():
    """Splitsen mag herverdelen, niet weggooien — anders lekt er een attribuut weg
    en is de partitie niet meer heel."""
    groups = [group("A1", "A2", "A3", name="Breed")]
    verdicts = [SplitVerdict(group_name="Breed", parts=(("A1",), ("A2",)))]

    out, log = apply_splits(groups, verdicts)

    assert out == groups
    assert log[0]["action"] == "POSTMORTEM_SPLIT_REJECTED"
    assert "A3" in log[0]["reason"]


def test_a_part_that_invents_an_attribute_is_rejected_whole():
    groups = [group("A1", "A2", name="Breed")]
    verdicts = [SplitVerdict(group_name="Breed", parts=(("A1",), ("A2", "A9")))]

    out, log = apply_splits(groups, verdicts)

    assert out == groups
    assert log[0]["action"] == "POSTMORTEM_SPLIT_REJECTED"


def test_a_part_repeating_an_attribute_is_rejected_whole():
    groups = [group("A1", "A2", name="Breed")]
    verdicts = [SplitVerdict(group_name="Breed", parts=(("A1", "A2"), ("A2",)))]

    out, log = apply_splits(groups, verdicts)

    assert out == groups
    assert log[0]["action"] == "POSTMORTEM_SPLIT_REJECTED"


def test_a_single_part_covering_everything_is_a_no_op_not_a_split():
    groups = [group("A1", "A2", name="Breed")]
    verdicts = [SplitVerdict(group_name="Breed", parts=(("A1", "A2"),))]

    out, log = apply_splits(groups, verdicts)

    assert out == groups
    assert log == []


def test_order_of_untouched_groups_is_preserved():
    groups = [group("A1", name="Een"), group("A2", "A3", name="Twee"),
              group("A4", name="Drie")]
    verdicts = [SplitVerdict(group_name="Twee", parts=(("A2",), ("A3",)))]

    out, _ = apply_splits(groups, verdicts)

    assert [g.proposed_name for g in out] == ["Een", "Twee", "Twee", "Drie"]


def test_verdict_for_an_unknown_group_is_ignored():
    groups = [group("A1", name="Een")]
    verdicts = [SplitVerdict(group_name="Bestaat niet", parts=(("A9",),))]

    out, log = apply_splits(groups, verdicts)

    assert out == groups
    assert log == []
