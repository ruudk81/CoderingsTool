"""Tests for `drain_domains` (finding B in the 2026-08-12 review).

A step-3 cache written under a stale key matches on whichever `DRAIN_KEYS`
member survived and returns fewer than two — non-empty, so nothing downstream
notices on its own. These pin the warning that now makes that visible.
"""

from types import SimpleNamespace

from pipeline.step_4_classifier.taxonomy_health import DRAIN_KEYS, drain_domains


def _meta(domains):
    return SimpleNamespace(domains=domains)


def test_drain_domains_finds_both_by_current_keys():
    meta = _meta([
        {"key": "not_known", "label": "Weet niet"},
        {"key": "other", "label": "Overig"},
        {"key": "", "label": "Duurzaamheid"},
    ])
    assert drain_domains(meta) == {"Weet niet", "Overig"}


def test_drain_domains_warns_on_a_partial_match(capsys):
    """A stale cache key (an old rename of a standing domain) matches only
    `other` — non-empty, so this must not pass silently."""
    meta = _meta([
        {"key": "stale_key", "label": "Kale associatie"},
        {"key": "other", "label": "Overig"},
    ])
    found = drain_domains(meta)

    assert found == {"Overig"}
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "1/2" in out
    assert "Overig" in out


def test_drain_domains_warns_when_both_keys_are_stale(capsys):
    meta = _meta([
        {"key": "stale_key", "label": "Kale associatie"},
        {"key": "unclassified", "label": "Overig"},
    ])
    found = drain_domains(meta)

    assert found == set()
    assert "WARNING" in capsys.readouterr().out


def test_drain_domains_is_silent_when_domain_discovery_never_ran(capsys):
    """No domains at all is a legitimate, silent zero — not every empty
    result is a stale-key bug."""
    assert drain_domains(_meta(None)) == set()
    assert drain_domains(_meta([])) == set()
    assert capsys.readouterr().out == ""


def test_drain_domains_is_silent_on_a_full_match(capsys):
    meta = _meta([
        {"key": "not_known", "label": "Weet niet"},
        {"key": "other", "label": "Overig"},
    ])
    drain_domains(meta)
    assert capsys.readouterr().out == ""


def test_drain_keys_has_two_members():
    """The warning threshold (`len(DRAIN_KEYS)`) is only meaningful if this
    holds — pinned so a future third drain key is a deliberate change, not a
    silent widening of what counts as a full match."""
    assert len(DRAIN_KEYS) == 2
