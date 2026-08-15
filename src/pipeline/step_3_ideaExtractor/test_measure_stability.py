"""Tests for the measuring functions of measure_stability.py — no cache, no LLM.

The ARI is hand-written and carries the conclusion about partition stability. A
yardstick that is not itself calibrated yields numbers nobody can weigh.
"""

import math

from .measure_stability import (
    _not_known_pct,
    adjusted_rand_index,
    noise_floor,
    print_comparison,
)


# ── Adjusted Rand Index ────────────────────────────────────────────────────

def test_identical_partitions_score_one():
    a = {"1": "x", "2": "x", "3": "y", "4": "y"}
    assert adjusted_rand_index(a, a) == 1.0


def test_renaming_every_domain_changes_nothing():
    """De kern: labels verschillen per run, groeperingen zijn wat telt."""
    a = {"1": "Merkuitingen", "2": "Merkuitingen", "3": "Klantbediening"}
    b = {"1": "Merkcommunicatie", "2": "Merkcommunicatie", "3": "Klantinteractie"}
    assert adjusted_rand_index(a, b) == 1.0


def test_crossed_partitions_score_at_or_below_chance():
    """Geen enkel paar dat in beide runs samen zit."""
    a = {"1": "x", "2": "x", "3": "y", "4": "y"}
    b = {"1": "p", "2": "q", "3": "p", "4": "q"}
    assert adjusted_rand_index(a, b) < 0


def test_a_merge_scores_between_chance_and_one():
    """Run B voegt twee domeinen van run A samen: deels eens, niet volledig."""
    a = {"1": "x", "2": "x", "3": "y", "4": "y", "5": "z", "6": "z"}
    b = {"1": "p", "2": "p", "3": "q", "4": "q", "5": "q", "6": "q"}
    ari = adjusted_rand_index(a, b)
    assert 0 < ari < 1


def test_too_few_shared_units_is_not_a_number():
    assert math.isnan(adjusted_rand_index({"1": "x"}, {"1": "y"}))


def test_only_shared_respondents_are_compared():
    """Runs kunnen verschillen in wie precies één idee opleverde."""
    a = {"1": "x", "2": "x", "3": "y", "9": "y"}
    b = {"1": "p", "2": "p", "3": "q", "8": "q"}
    assert adjusted_rand_index(a, b) == 1.0


# ── Ruisvloer ──────────────────────────────────────────────────────────────

def _snap(assignments, texts):
    return {"assignments": assignments, "texts": texts}


def test_noise_floor_counts_only_the_minority_side():
    """Drie keer dezelfde tekst, één afwijker: één fout, geen drie."""
    snap = _snap(
        {"1": "A", "2": "A", "3": "B"},
        {"1": "bank", "2": "bank", "3": "bank"},
    )
    nf = noise_floor(snap)
    assert nf["repeated_ideas"] == 3
    assert nf["minority"] == 1
    assert nf["pct"] == round(100 / 3, 1)


def test_texts_occurring_once_are_not_measurable():
    """Without repetition there is no reference point — those do not count in
    the denominator."""
    snap = _snap(
        {"1": "A", "2": "B", "3": "A", "4": "A"},
        {"1": "uniek", "2": "ook uniek", "3": "bank", "4": "bank"},
    )
    nf = noise_floor(snap)
    assert nf["repeated_ideas"] == 2
    assert nf["minority"] == 0
    assert nf["pct"] == 0.0


def test_a_four_way_split_counts_three_as_minority():
    snap = _snap(
        {"1": "A", "2": "B", "3": "C", "4": "D"},
        {str(i): "mijn bank" for i in range(1, 5)},
    )
    nf = noise_floor(snap)
    assert nf["minority"] == 3
    assert nf["inconsistent_texts"] == 1


def test_no_repeated_texts_at_all_reports_zero_not_a_crash():
    nf = noise_floor(_snap({"1": "A"}, {"1": "uniek"}))
    assert nf["repeated_ideas"] == 0
    assert nf["pct"] == 0.0


# ── Tolerante lezing van vóór-hernoeming snapshots (bare_evaluation_pct) ────
#
# `data/step3_stability.jsonl` is append-only geschiedenis en wordt nooit
# rewritten — nine rows in it still carry the field under the old name.
# New snapshots write only `not_known_pct`; the reading has to cope with
# beide overweg kunnen.

def _full_snap(recorded_at, not_known_field, not_known_value):
    domains = [
        {"key": "Duurzaamheid", "label": "Duurzaamheid"},
        {"key": "not_known", "label": "Kale associatie"},
        {"key": "other", "label": "Overig"},
    ]
    return {
        "recorded_at": recorded_at,
        "substantive_domains": 1,
        "ideas": 10,
        "other_pct": 5.0,
        "processing_errors": 0,
        "assignments": {"1": "Duurzaamheid", "2": "Duurzaamheid"},
        "texts": {"1": "bank", "2": "bank"},
        "domains": domains,
        not_known_field: not_known_value,
    }


def test_not_known_pct_prefers_the_new_field_but_falls_back_to_the_old_one():
    assert _not_known_pct({"bare_evaluation_pct": 18.3}) == 18.3
    assert _not_known_pct({"not_known_pct": 12.5}) == 12.5
    assert _not_known_pct({"not_known_pct": 12.5, "bare_evaluation_pct": 18.3}) == 12.5


def test_print_comparison_reads_mixed_old_and_new_snapshots_without_raising(capsys):
    """The regression: print_comparison() read `snap['not_known_pct']`
    unconditionally, so a comparison containing a pre-rename row gave a KeyError
    instead of a table.
    """
    old_row = _full_snap("2026-08-01T10:00:00", "bare_evaluation_pct", 18.3)
    new_row = _full_snap("2026-08-12T10:00:00", "not_known_pct", 12.5)

    print_comparison([old_row, new_row])

    out = capsys.readouterr().out
    assert "18.3" in out
    assert "12.5" in out
