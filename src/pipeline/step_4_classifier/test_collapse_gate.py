"""Tests for the gate that rejects a collapsed consolidation answer.

The failure this pins is not a schema failure. On 2026-08-16 one attribute call
returned a single attribute named "Voorlopige consolidatie" citing 1 of its 12
candidates, alongside three decision lines describing merges it never made. That
answer validated, so it counted as a success: the net rebuilt the other eleven
candidates and the log recorded `12 -> 12`, which reads exactly like a call that
deliberately kept everything.

Two things are checked here. That such an answer now raises — which is what puts
the task back through the requester, and after that into the fallback the phase
already had. And that a good answer still passes, including the two ways a
candidate can legitimately be accounted for.
"""
import pytest

from pipeline.step_4_classifier.classifier import (
    ConsolidationCollapse, TaxonomyClassifier, _accounted_for, _norm,
)
from pipeline.step_4_classifier.config_classifier import CategoriesConfig
from pipeline.step_4_classifier.prompts_consolidation import (
    AttributeConsolidationResult, FacetConsolidationResult, FacetPool,
    SettledAttribute, SettledFacet, build_attribute_candidate_index,
)
from pipeline.step_4_classifier.drains import make_drain_attribute
from pipeline.step_4_classifier.prompts_discovery import DiscoveredAttribute
from pipeline.step_4_classifier.prompts_refinement import (
    RefinedAttribute, RefinementResult,
)


def _clf():
    return TaxonomyClassifier(CategoriesConfig())


def _attributes(*names):
    return [DiscoveredAttribute(
        attribute_name=name, attribute_definition=f"De eigenschap {name}.",
        example_observations=[f"observatie over {name}"]) for name in names]


def _pool(name, *attr_names):
    return FacetPool(
        facet_name=name, facet_definition=f"Wat {name} vastlegt.",
        facet_question=f"Wat zegt {name}?", attributes=_attributes(*attr_names))


def _facet_task(*pools):
    return {"domain_label": "duurzaamheid", "candidates": list(pools)}


def _attribute_task(*attr_names):
    return {"domain_label": "merkidentiteit", "facet_index": 0,
            "facet": _pool("Reclame en communicatie"),
            "candidates": _attributes(*attr_names)}


def _settled_facet(name, *source_ids):
    return SettledFacet(boundary_rules=[], 
        facet_name=name, facet_definition="d", facet_question="Wat?",
        source_facet_ids=list(source_ids))


def _settled_attribute(name, *source_ids):
    return SettledAttribute(boundary_rules=[], 
        attribute_name=name, attribute_definition="d",
        example_observations=["e"], source_attribute_ids=list(source_ids))


def _facet_answer(*facets):
    return FacetConsolidationResult(decision_summary=["x"], facets=list(facets))


def _attribute_answer(*attributes):
    return AttributeConsolidationResult(
        decision_summary=["x"], attributes=list(attributes))


# =============================================================================
# A scaffold is not a result
# =============================================================================

def test_an_attribute_answer_that_accounts_for_almost_nothing_is_rejected():
    """The 2026-08-16 case, to the number: 12 candidates, one returned
    attribute, one cited id."""
    parse = TaxonomyClassifier._attribute_consolidation_parse_fn()
    task = _attribute_task(*[f"kandidaat {i}" for i in range(1, 13)])
    answer = _attribute_answer(_settled_attribute("Voorlopige consolidatie", "A2"))
    with pytest.raises(ConsolidationCollapse):
        parse(task, answer)


def test_a_facet_answer_that_accounts_for_almost_nothing_is_rejected():
    """Same gate one level up. Nought occurrences so far is not immunity —
    it is one run."""
    parse = TaxonomyClassifier._facet_consolidation_parse_fn()
    task = _facet_task(*[_pool(f"facet {i}", "a") for i in range(1, 11)])
    with pytest.raises(ConsolidationCollapse):
        parse(task, _facet_answer(_settled_facet("Iets", "F1")))


def test_an_empty_answer_is_rejected_rather_than_quietly_kept():
    """Returning nothing is the extreme of the same failure, and used to reach
    the net by a different route than a near-empty one."""
    parse = TaxonomyClassifier._attribute_consolidation_parse_fn()
    task = _attribute_task("a", "b", "c", "d")
    with pytest.raises(ConsolidationCollapse):
        parse(task, _attribute_answer())


# =============================================================================
# A real merge still passes — the gate must not punish consolidation
# =============================================================================

def test_a_hard_merge_passes_when_every_candidate_is_accounted_for():
    """Twelve candidates folded into two is the phase doing its job. What the
    gate measures is coverage of the input, never the size of the output."""
    parse = TaxonomyClassifier._attribute_consolidation_parse_fn()
    task = _attribute_task(*[f"kandidaat {i}" for i in range(1, 13)])
    answer = _attribute_answer(
        _settled_attribute("Zichtbaarheid", *[f"A{i}" for i in range(1, 7)]),
        _settled_attribute("Uitwerking", *[f"A{i}" for i in range(7, 13)]))
    assert parse(task, answer) is answer


def test_a_pass_through_answer_passes():
    parse = TaxonomyClassifier._facet_consolidation_parse_fn()
    task = _facet_task(_pool("een", "a"), _pool("twee", "b"))
    answer = _facet_answer(_settled_facet("een", "F1"), _settled_facet("twee", "F2"))
    assert parse(task, answer) is answer


def test_a_survivor_that_kept_a_unique_name_counts_as_accounting_for_it():
    """The second route the nets accept: a survivor holding a candidate's name
    without citing its id. The gate has to accept it too, or it would reject an
    answer the net would have handled without a mark on it."""
    parse = TaxonomyClassifier._attribute_consolidation_parse_fn()
    task = _attribute_task("alpha", "beta", "gamma", "delta")
    answer = _attribute_answer(
        _settled_attribute("alpha"), _settled_attribute("beta"),
        _settled_attribute("gamma"), _settled_attribute("delta"))
    assert parse(task, answer) is answer


def test_a_shared_name_does_not_count_because_it_names_no_one():
    """Two candidates under one name: the name says nothing about which was
    meant, so neither is accounted for — the same rule the nets apply."""
    parse = TaxonomyClassifier._attribute_consolidation_parse_fn()
    task = _attribute_task("alpha", "alpha", "beta", "gamma")
    with pytest.raises(ConsolidationCollapse):
        parse(task, _attribute_answer(_settled_attribute("alpha")))


# =============================================================================
# The gate and the net have to mean the same thing by "forgotten"
# =============================================================================

def test_the_gate_and_the_net_agree_on_which_candidates_are_forgotten():
    """One definition, checked from both ends. If these ever drift, the net
    starts repairing exactly what the gate rejected, or the other way round —
    and both failures are silent."""
    clf = _clf()
    task = _attribute_task("alpha", "beta", "gamma", "delta")
    answer = _attribute_answer(
        _settled_attribute("Samengevoegd", "A1", "A2"),
        _settled_attribute("gamma"))

    index = build_attribute_candidate_index(task["candidates"])
    accounted = _accounted_for(
        index,
        {s for a in answer.attributes for s in a.source_attribute_ids},
        {_norm(a.attribute_name) for a in answer.attributes},
        lambda candidate: _norm(candidate.attribute_name))

    clf._attribute_consolidation_survivors(task, answer)
    kept = {e["id"] for e in clf._action_log
            if e["action"] == "attribute_kept_unclaimed"}

    assert accounted == set(index) - kept
    assert kept == {"A4"}


# =============================================================================
# The same gate on refinement — the other phase that folds
# =============================================================================
#
# Refinement claims on NAME (`source_attributes`), not on id: it shows the model
# names and never hands out ids. Coverage is therefore counted over the names of
# the facet's own attributes. Catch-alls stay out of that count — rule 5 forbids
# naming one as a source, so requiring the model to account for one would demand
# exactly what the prompt prohibits.


def _refinement_task(*names, with_drain=False):
    attributes = [{"attribute_name": name,
                   "attribute_definition": f"De eigenschap {name}."}
                  for name in names]
    if with_drain:
        attributes.append(make_drain_attribute("Reclame", "nl-NL"))
    return {
        "domain_label": "merkidentiteit", "facet_index": 0,
        "facet": {"facet_name": "Reclame", "facet_definition": "d",
                  "facet_question": "Wat?", "attributes": attributes},
        "counts": {}, "shares": {}, "facet_total": 0, "domain_total": 0,
        "contents": {}}


def _refined(name, *sources):
    return RefinedAttribute(
        attribute_name=name, attribute_definition="d",
        example_observations=["e"], source_attributes=list(sources))


def _refinement_answer(*attributes):
    return RefinementResult(scratchpad="x", attributes=list(attributes))


def test_a_refinement_answer_that_accounts_for_almost_nothing_is_rejected():
    """Naslijpen had no gate at all: a call that named 2 of its 7 attributes
    counted as a success, and the net then wrote the result instead of
    repairing it."""
    parse = TaxonomyClassifier._refinement_parse_fn()
    task = _refinement_task(*[f"attribuut {i}" for i in range(1, 9)])
    answer = _refinement_answer(_refined("Samengevoegd", "attribuut 1",
                                         "attribuut 2"))
    with pytest.raises(ConsolidationCollapse):
        parse(task, answer)


def test_a_refinement_merge_that_accounts_for_everything_passes():
    """Eight into two is the phase doing its job. The gate measures coverage of
    the input, never the size of the output."""
    parse = TaxonomyClassifier._refinement_parse_fn()
    names = [f"attribuut {i}" for i in range(1, 9)]
    task = _refinement_task(*names)
    answer = _refinement_answer(
        _refined("Eerste", *names[:4]), _refined("Tweede", *names[4:]))
    assert parse(task, answer) is answer


def test_a_survivor_that_kept_a_name_accounts_for_it_without_citing_it():
    """The second route the net accepts, here the only one that differs from a
    citation: a survivor holding a candidate's name."""
    parse = TaxonomyClassifier._refinement_parse_fn()
    task = _refinement_task("alpha", "beta")
    answer = _refinement_answer(_refined("alpha"), _refined("beta"))
    assert parse(task, answer) is answer


def test_a_catch_all_is_not_counted_against_a_refinement_answer():
    """Four real attributes plus a catch-all, two accounted for. Counting the
    catch-all would make that 2/5 and reject an answer that obeyed rule 5."""
    parse = TaxonomyClassifier._refinement_parse_fn()
    names = [f"attribuut {i}" for i in range(1, 5)]
    task = _refinement_task(*names, with_drain=True)
    answer = _refinement_answer(_refined("Samengevoegd", *names[:2]))
    assert parse(task, answer) is answer


def test_an_empty_refinement_answer_is_rejected():
    parse = TaxonomyClassifier._refinement_parse_fn()
    with pytest.raises(ConsolidationCollapse):
        parse(_refinement_task("a", "b", "c", "d"), _refinement_answer())


def test_the_refinement_log_carries_the_coverage_of_every_call():
    """A gate that does not report itself only moves the problem. The pair also
    shows the catch-all staying out of the denominator, where a pass/fail check
    can only straddle the threshold to see it."""
    clf = _clf()
    names = [f"attribuut {i}" for i in range(1, 5)]
    task = _refinement_task(*names, with_drain=True)
    clf._apply_refinement(
        tasks=[task], results=[_refinement_answer(_refined("Samen", *names[:2]))],
        structure={"merkidentiteit": [task["facet"]]}, assignments={})
    row = next(e for e in clf._action_log
               if e["action"] == "refinement_provenance")
    assert (row["accounted_for"], row["candidates"]) == (2, 4)


def test_the_log_carries_the_coverage_of_every_call():
    """Without the number, a call that judged nothing and a call that judged
    everything both read as `before -> after`."""
    clf = _clf()
    task = _attribute_task("alpha", "beta", "gamma", "delta")
    clf._attribute_consolidation_survivors(
        task, _attribute_answer(_settled_attribute("Samen", "A1", "A2", "A3")))
    row = next(e for e in clf._action_log
               if e["action"] == "attribute_consolidation")
    assert (row["accounted_for"], row["candidates"]) == (3, 4)
