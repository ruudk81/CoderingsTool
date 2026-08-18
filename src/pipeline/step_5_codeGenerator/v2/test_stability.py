"""Tests voor de stabiliteitsmeting over herhaalde consolidatieruns."""
import pytest

from pipeline.step_5_codeGenerator.v2.grouping import Group
from pipeline.step_5_codeGenerator.v2.stability import (
    StabilityReport, measure_stability, pairs_from_groups,
)


def group(*ids, name="G"):
    return Group(member_ids=tuple(ids), proposed_name=name, explanation="e")


def test_pairs_from_groups_lists_every_co_membership():
    pairs = pairs_from_groups([group("A1", "A2", "A3"), group("A4")])

    assert pairs == {
        frozenset({"A1", "A2"}), frozenset({"A1", "A3"}), frozenset({"A2", "A3"}),
    }


def test_a_solo_group_contributes_no_pair():
    assert pairs_from_groups([group("A1"), group("A2")]) == set()


def test_pair_together_in_every_run_is_stable():
    runs = [[group("A1", "A2")], [group("A1", "A2")], [group("A1", "A2")]]

    report = measure_stability(runs, attribute_ids=["A1", "A2"])

    assert report.runs == 3
    assert report.together[frozenset({"A1", "A2"})] == 3
    assert report.unstable_pairs() == []


def test_pair_apart_in_every_run_is_also_stable():
    """Nooit samen is net zo goed een besluit als altijd samen."""
    runs = [[group("A1"), group("A2")]] * 3

    report = measure_stability(runs, attribute_ids=["A1", "A2"])

    assert report.together[frozenset({"A1", "A2"})] == 0
    assert report.unstable_pairs() == []


def test_pair_that_switches_is_unstable():
    runs = [[group("A1", "A2")], [group("A1"), group("A2")], [group("A1", "A2")]]

    report = measure_stability(runs, attribute_ids=["A1", "A2"])

    assert report.together[frozenset({"A1", "A2"})] == 2
    assert report.unstable_pairs() == [("A1", "A2")]


def test_unstable_attributes_collects_both_sides_of_every_wobbling_pair():
    runs = [[group("A1", "A2"), group("A3")], [group("A1"), group("A2", "A3")]]

    report = measure_stability(runs, attribute_ids=["A1", "A2", "A3"])

    assert report.unstable_attributes() == {"A1", "A2", "A3"}


def test_attribute_missing_from_a_run_does_not_crash_the_count():
    """Een run kan een attribuut kwijtraken; repair_partition vangt dat af, maar
    de meting mag er hoe dan ook niet op stuklopen."""
    runs = [[group("A1", "A2")], [group("A1")]]

    report = measure_stability(runs, attribute_ids=["A1", "A2"])

    assert report.together[frozenset({"A1", "A2"})] == 1
    assert report.unstable_pairs() == [("A1", "A2")]


def test_pair_stability_share_reports_how_settled_the_whole_run_set_is():
    runs = [[group("A1", "A2"), group("A3")], [group("A1", "A2"), group("A3")]]

    report = measure_stability(runs, attribute_ids=["A1", "A2", "A3"])

    assert report.stable_share() == 1.0


def test_a_single_run_cannot_measure_anything():
    with pytest.raises(ValueError, match="minstens twee"):
        measure_stability([[group("A1", "A2")]], attribute_ids=["A1", "A2"])


def test_report_orders_unstable_pairs_reproducibly():
    runs = [[group("A1", "A2", "A3")], [group("A1"), group("A2"), group("A3")]]

    report = measure_stability(runs, attribute_ids=["A3", "A1", "A2"])

    assert report.unstable_pairs() == [("A1", "A2"), ("A1", "A3"), ("A2", "A3")]
