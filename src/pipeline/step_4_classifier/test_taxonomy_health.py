"""Tests for `drain_domains` (finding B in the 2026-08-12 review) and for the
guard in `prune_empty_nodes`.

A step-3 cache written under a stale key matches on whichever `DRAIN_KEYS`
member survived and returns fewer than two — non-empty, so nothing downstream
notices on its own. These pin the warning that now makes that visible.
"""

from types import SimpleNamespace

from models import DomainResultModel, DomainSet, TaxonomyResultsCache
from pipeline.step_4_classifier.taxonomy_health import (
    DRAIN_KEYS,
    drain_domains,
    prune_empty_nodes,
)


def _meta(domains):
    return SimpleNamespace(domains=domains)


def _tax(assignments):
    """One domain holding one facet with one attribute, with `assignments` on it."""
    return TaxonomyResultsCache(
        partition_set=DomainSet(partitions=[]),
        partition_results={
            "duurzaamheid": DomainResultModel(
                partition_name="duurzaamheid",
                n_labels=3,
                n_batches=1,
                facets=[{"facet_name": "Ecologie", "facet_definition": "…"}],
                attributes={"Ecologie": [{"attribute_name": "Milieugerichtheid"}]},
                attribute_assignments=assignments,
            )
        },
    )


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
    assert "1/3" in out
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
        {"key": "no_subject", "label": "Zonder onderwerp"},
    ])
    drain_domains(meta)
    assert capsys.readouterr().out == ""


def test_drain_keys_has_three_members():
    """The warning threshold (`len(DRAIN_KEYS)`) is only meaningful if this
    holds — pinned so a fourth drain key is a deliberate change, not a silent
    widening of what counts as a full match. Went from two to three on
    2026-08-13, when `no_subject` gave subject-less answers a home of their own
    instead of spreading them over the content domains."""
    assert len(DRAIN_KEYS) == 3
    assert DRAIN_KEYS == frozenset({"other", "not_known", "no_subject"})


def test_drain_share_counts_only_ideas_inside_a_drain():
    """The counter-metric to coarser grouping: every merge that goes too far
    pushes responses into a catch-all."""
    from models import DomainResultModel, DomainSet, TaxonomyResultsCache
    from pipeline.step_4_classifier.drains import make_drain_attribute
    from pipeline.step_4_classifier.taxonomy_health import measure

    drain = make_drain_attribute("F", "Dutch")
    tax = TaxonomyResultsCache(
        partition_set=DomainSet(partitions=[]),
        partition_results={"D": DomainResultModel(
            partition_name="D", n_labels=3, n_batches=1,
            facets=[{"facet_name": "F"}],
            attributes={"F": [{"attribute_name": "Wachttijd"}, drain]},
            attribute_assignments={
                "i1": "Wachttijd", "i2": drain["attribute_name"],
                "i3": drain["attribute_name"]},
        )},
    )
    report = measure(tax)
    assert report.n_drain_ideas == 2
    assert round(report.drain_share) == 67


# =============================================================================
# PRUNE — the guard for a run that stops before assignment
# =============================================================================

def test_prune_leaves_everything_when_nothing_was_assigned():
    """A `stop_after_phase` before `assignment` yields structure without
    assignments. "No ideas here" then does not mean the node is empty, only that
    nothing has been assigned yet — and pruning on that reading wiped the whole
    taxonomy. Measured 2026-08-15: 55 facets and 179 attributes discovered, all
    234 pruned, an empty taxonomy written over the complete one."""
    tax = _tax({})
    report = prune_empty_nodes(tax)

    dr = tax.partition_results["duurzaamheid"]
    assert len(dr.facets) == 1
    assert dr.attributes["Ecologie"]
    assert report.facets == [] and report.attributes == []


def test_prune_does_run_once_assignments_exist():
    """With assignments in the run, "no ideas here" does mean empty: the guard
    must not switch off the normal pruning."""
    tax = _tax({"idea-1": "Iets anders"})
    report = prune_empty_nodes(tax)

    dr = tax.partition_results["duurzaamheid"]
    assert dr.facets == []
    assert "Ecologie" not in dr.attributes
    assert [n for _, _, n in report.attributes] == ["Milieugerichtheid"]
