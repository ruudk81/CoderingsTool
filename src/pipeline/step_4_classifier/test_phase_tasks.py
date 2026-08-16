"""Tests for the task shapes of the seven phases (step 4).

The `_build_<phase>_tasks` are pure functions of their arguments, so scope,
skipping, chunking and counts can be checked without an LLM call.
"""
import asyncio

import pytest

from pipeline.step_3_ideaExtractor.dimension_data import get_dimensions_in_decision_order
from pipeline.step_4_classifier.classifier import (
    ConsolidationCollapse, PromptContext, TaxonomyClassifier, _strip_enumeration,
    attribute_dicts, Placement, facet_dicts, flatten_placements,
)
from pipeline.step_4_classifier.config_classifier import CategoriesConfig
from pipeline.step_4_classifier.drains import is_drain_item, make_drain_attribute
from pipeline.step_4_classifier.prompts_facet_settle import (
    SettledFacetCard, build_facet_settle_model,
)
from pipeline.step_4_classifier.prompts_refinement import (
    RefinedAttribute, RefinementResult,
)
from pipeline.step_4_classifier.prompts_discovery import (
    DiscoveredAttribute, DiscoveredFacet,
)
from pipeline.step_4_classifier.prompts_consolidation import (
    AttributeConsolidationResult, FacetConsolidationResult, FacetPool,
    SettledAttribute, SettledFacet,
)

DIM = get_dimensions_in_decision_order()[0]


def _clf(**overrides):
    return TaxonomyClassifier(CategoriesConfig(**overrides))


def _ctx(domains, drains=()):
    return PromptContext(
        language="Dutch", survey_question="?",
        dimension=DIM, dimension_name=DIM.key,
        dimension_description=DIM.dimension_description,
        domains={
            name: {"label": name, "definition": "d", "boundary_test": "",
                   "exclusions": [], "observations": obs}
            for name, obs in domains.items()},
        drain_labels=set(drains),
    )


def _facet(name, *attrs):
    return DiscoveredFacet(
        facet_name=name, facet_definition="d",
        attributes=[DiscoveredAttribute(
            attribute_name=a, attribute_definition="d",
            example_observations=["e"]) for a in attrs])


def _pool(name, *attr_names, question=""):
    return FacetPool(
        facet_name=name,
        facet_definition=f"Wat {name} vastlegt.",
        facet_question=question,
        attributes=[DiscoveredAttribute(
            attribute_name=a,
            attribute_definition=f"De eigenschap {a}.",
            example_observations=[f"observatie over {a}"],
        ) for a in attr_names])


def _structure(facets_per_domain):
    return {d: [f.model_dump() for f in facets]
            for d, facets in facets_per_domain.items()}


def _actions(clf, action):
    return [e for e in clf._action_log if e["action"] == action]


# =============================================================================
# DISCOVERY
# =============================================================================

def test_discovery_skips_the_catch_all_domains():
    """Step 3 defines them as deliberately broad catch-alls; imposing structure
    on them invents distinctions the responses do not carry."""
    ctx = _ctx({"inhoud": ["a"], "Overig": ["b"]}, drains=["overig"])
    tasks = _clf()._build_discovery_tasks(ctx)
    assert {t["domain_label"] for t in tasks} == {"inhoud"}


def test_catch_all_matching_is_case_insensitive():
    """Step 3 writes the label capitalised, domain discovery lowercases the
    partition name. An exact match found neither, silently."""
    ctx = _ctx({"Overig": ["b"]}, drains=["overig"])
    assert ctx.is_drain("Overig") is True
    assert _clf()._build_discovery_tasks(ctx) == []


def test_a_small_scope_gets_one_chunk():
    tasks = _clf()._build_discovery_tasks(_ctx({"d": ["a", "b", "c"]}))
    assert len(tasks) == 1
    assert tasks[0]["total_chunks"] == 1


def test_a_large_scope_is_chunked_with_overlap():
    ctx = _ctx({"d": [f"obs{i}" for i in range(500)]})
    tasks = _clf()._build_discovery_tasks(ctx)
    assert len(tasks) > 1
    assert all(t["total_chunks"] == len(tasks) for t in tasks)
    gezien = {o for t in tasks for o in t["observations"]}
    assert gezien == set(ctx.domain("d")["observations"])


def test_the_list_number_is_taken_back_off_an_example():
    """Discovery renders its input as `f"{i}. {obs}"` so the scratchpad can point
    at lines. The model copies the whole line, and that rendering artefact then
    travelled through consolidation into the codebook."""
    assert _strip_enumeration("6. investeert in natuur → investeringen") == (
        "investeert in natuur → investeringen")
    assert _strip_enumeration("20.  duurzaam") == "duurzaam"


def test_stripping_leaves_an_ordinary_answer_alone():
    """A response may legitimately start with a figure."""
    assert _strip_enumeration("1 op de 5 keer mis") == "1 op de 5 keer mis"
    assert _strip_enumeration("geen winst → niet op winst gericht") == (
        "geen winst → niet op winst gericht")


def test_discovery_parse_cleans_the_examples():
    clf = _clf()
    parse = clf._discovery_parse_fn()

    class _Response:
        facets = [_facet("Snelheid", "Wachttijd")]

    _Response.facets[0].attributes[0].example_observations = [
        "3. lang wachten", "traag"]
    facets = parse({}, _Response())
    assert facets[0].attributes[0].example_observations == [
        "lang wachten", "traag"]


# =============================================================================
# FACET CONSOLIDATION
# =============================================================================

def test_a_facet_group_is_capped_on_facets_not_attributes():
    """The facet call renders attribute names only, so its prompt no longer
    grows with the attribute count. Facets are what it judges."""
    clf = _clf(facet_consolidation_max_facets_per_call=2)
    pools = [_pool(f"F{i}", *[f"a{j}" for j in range(40)]) for i in range(5)]
    groups = clf._facet_consolidation_groups(pools)
    assert [len(g) for g in groups] == [2, 2, 1]


def test_candidates_are_sorted_by_name_before_grouping():
    """Near-identical names end up next to each other, so they usually fall in
    the same group instead of missing each other for a whole round.

    Usually, not always: a fixed-size split can still place the group boundary
    exactly between two neighbours. That is accepted — the next round puts the
    survivors together after all.
    """
    clf = _clf(facet_consolidation_max_facets_per_call=2)
    candidates = [_pool("Snelheid van afhandeling", "a"),
                  _pool("Bejegening", "b"), _pool("Snelheid", "c")]
    order = [p.facet_name.lower()
             for g in clf._facet_consolidation_groups(candidates) for p in g]
    assert order == sorted(order)


def test_a_survivor_pools_the_attributes_of_what_it_claimed():
    """The facet phase does not consolidate attributes; it accumulates them, so
    the next phase sees the union."""
    clf = _clf()
    task = {"domain_label": "d",
            "candidates": [_pool("Snelheid", "Wachttijd"),
                           _pool("Snelheid van afhandeling", "Doorlooptijd")],
            "recurrence": {}, "n_passes": 3}
    result = FacetConsolidationResult(
        decision_summary=[],
        facets=[SettledFacet(
            facet_name="Snelheid", facet_definition="…",
            facet_question="Hoe snel?", source_facet_ids=["F1", "F2"])])
    survivors = clf._facet_consolidation_survivors(task, result)
    assert len(survivors) == 1
    names = [a.attribute_name for a in survivors[0].attributes]
    assert names == ["Wachttijd", "Doorlooptijd"]


def test_a_pooled_attribute_proposed_twice_is_collapsed_once():
    clf = _clf()
    task = {"domain_label": "d",
            "candidates": [_pool("Snelheid", "Wachttijd"),
                           _pool("Snelheid van afhandeling", "Wachttijd")],
            "recurrence": {}, "n_passes": 3}
    result = FacetConsolidationResult(
        decision_summary=[],
        facets=[SettledFacet(
            facet_name="Snelheid", facet_definition="…",
            facet_question="Hoe snel?", source_facet_ids=["F1", "F2"])])
    survivors = clf._facet_consolidation_survivors(task, result)
    assert [a.attribute_name for a in survivors[0].attributes] == ["Wachttijd"]


def test_an_unclaimed_facet_is_kept_whole():
    """Merging and forgetting look identical in the answer: both are absent."""
    clf = _clf()
    task = {"domain_label": "d",
            "candidates": [_pool("Snelheid", "Wachttijd"),
                           _pool("Bejegening", "Vriendelijkheid")],
            "recurrence": {}, "n_passes": 3}
    result = FacetConsolidationResult(
        decision_summary=[],
        facets=[SettledFacet(
            facet_name="Snelheid", facet_definition="…",
            facet_question="Hoe snel?", source_facet_ids=["F1"])])
    survivors = clf._facet_consolidation_survivors(task, result)
    assert sorted(s.facet_name for s in survivors) == ["Bejegening", "Snelheid"]
    assert any(e["action"] == "facet_kept_unclaimed"
               for e in clf._action_log)


def test_a_name_fallback_only_counts_when_the_name_is_unique():
    """Two candidates sharing a name say nothing about which was meant; letting
    the name count would undo what the ids fixed."""
    clf = _clf()
    task = {"domain_label": "d",
            "candidates": [_pool("Snelheid", "Wachttijd"),
                           _pool("Snelheid", "Doorlooptijd")],
            "recurrence": {}, "n_passes": 3}
    result = FacetConsolidationResult(
        decision_summary=[],
        facets=[SettledFacet(
            facet_name="Snelheid", facet_definition="…",
            facet_question="Hoe snel?", source_facet_ids=[])])
    survivors = clf._facet_consolidation_survivors(task, result)
    assert len(survivors) == 3


def test_two_survivors_stating_one_question_are_still_reported():
    """Rule 1 made visible. Logged, never repaired: merging here would overrule
    a judgement the model made with every candidate in view."""
    clf = _clf()
    task = {"domain_label": "d",
            "candidates": [_pool("A", "x"), _pool("B", "y")],
            "recurrence": {}, "n_passes": 3}
    result = FacetConsolidationResult(
        decision_summary=[],
        facets=[
            SettledFacet(facet_name="A", facet_definition="…",
                         facet_question="Hoe snel?", source_facet_ids=["F1"]),
            SettledFacet(facet_name="B", facet_definition="…",
                         facet_question="Hoe snel?", source_facet_ids=["F2"])])
    clf._facet_consolidation_survivors(task, result)
    assert _actions(clf, "duplicate_facet_question")


def test_distinct_questions_are_not_reported():
    """The negative twin of the test above: a breach report that fires on every
    run says nothing about the run it fired on."""
    clf = _clf()
    task = {"domain_label": "d",
            "candidates": [_pool("Snelheid", "x"), _pool("Bejegening", "y")],
            "recurrence": {}, "n_passes": 3}
    result = FacetConsolidationResult(
        decision_summary=[],
        facets=[
            SettledFacet(facet_name="Snelheid", facet_definition="…",
                         facet_question="Hoe snel gaat het?",
                         source_facet_ids=["F1"]),
            SettledFacet(facet_name="Bejegening", facet_definition="…",
                         facet_question="Hoe wordt men bejegend?",
                         source_facet_ids=["F2"])])
    clf._facet_consolidation_survivors(task, result)
    assert _actions(clf, "duplicate_facet_question") == []


def test_facet_provenance_pins_which_candidate_went_where():
    """Survivors are rebuilt as plain pools, so `source_facet_ids` dies with the
    phase unless the action log keeps it. Without the record the log says forty
    facets became twelve but not which absorbed which."""
    clf = _clf()
    task = {"domain_label": "d",
            "candidates": [_pool("Snelheid", "x"), _pool("Tempo", "y")],
            "recurrence": {}, "n_passes": 3}
    result = FacetConsolidationResult(
        decision_summary=["hield X en Y uiteen"],
        facets=[SettledFacet(
            facet_name="Snelheid", facet_definition="…",
            facet_question="Hoe snel?", source_facet_ids=["F1", "F2"])])
    clf._facet_consolidation_survivors(task, result)
    entry = _actions(clf, "facet_provenance")[0]
    assert entry["facets"][0]["source_facet_ids"] == ["F1", "F2"]
    assert entry["facets"][0]["facet_question"] == "Hoe snel?"
    assert entry["decisions"] == ["hield X en Y uiteen"]


def test_a_cited_id_that_was_never_handed_out_is_logged():
    clf = _clf()
    task = {"domain_label": "d", "candidates": [_pool("Snelheid", "Wachttijd")],
            "recurrence": {}, "n_passes": 3}
    result = FacetConsolidationResult(
        decision_summary=[],
        facets=[SettledFacet(
            facet_name="Snelheid", facet_definition="…",
            facet_question="Hoe snel?", source_facet_ids=["F1", "F9"])])
    clf._facet_consolidation_survivors(task, result)
    entry = next(e for e in clf._action_log
                 if e["action"] == "unknown_source_id")
    assert entry["facets"] == ["F9"]


def test_a_candidate_claimed_by_two_survivors_pools_into_the_first():
    """Pooling into both would let the same attribute survive under two facets.
    First claimant wins, and the split is logged."""
    clf = _clf()
    task = {"domain_label": "d", "candidates": [_pool("Snelheid", "Wachttijd")],
            "recurrence": {}, "n_passes": 3}
    result = FacetConsolidationResult(
        decision_summary=[],
        facets=[
            SettledFacet(facet_name="A", facet_definition="…",
                         facet_question="?", source_facet_ids=["F1"]),
            SettledFacet(facet_name="B", facet_definition="…",
                         facet_question="?", source_facet_ids=["F1"])])
    survivors = clf._facet_consolidation_survivors(task, result)
    by_name = {s.facet_name: s for s in survivors}
    assert [a.attribute_name for a in by_name["A"].attributes] == ["Wachttijd"]
    assert by_name["B"].attributes == []
    assert any(e["action"] == "divided_source_facet"
               for e in clf._action_log)


def test_one_survivor_citing_an_id_twice_is_not_a_split():
    """A repeat inside one citation list claims the candidate once. Counted as
    two claimants it reported a split that never happened, in the very log line
    the first run is meant to be judged on."""
    clf = _clf()
    task = {"domain_label": "d", "candidates": [_pool("Snelheid", "Wachttijd")],
            "recurrence": {}, "n_passes": 3}
    result = FacetConsolidationResult(
        decision_summary=[],
        facets=[SettledFacet(facet_name="A", facet_definition="…",
                             facet_question="?",
                             source_facet_ids=["F1", "F1"])])
    survivors = clf._facet_consolidation_survivors(task, result)
    assert [a.attribute_name for a in survivors[0].attributes] == ["Wachttijd"]
    assert not any(e["action"] == "divided_source_facet"
                   for e in clf._action_log)


def test_a_name_fallback_hands_over_the_attributes_it_covers():
    """The unique-name branch treats the candidate as absorbed, so it must move
    the pool too. `source_facet_ids` is the only channel to the next phase: what
    is not pooled here is not lost from a log, it is lost from the taxonomy."""
    clf = _clf()
    task = {"domain_label": "d",
            "candidates": [_pool("Snelheid", "Wachttijd"),
                           _pool("Bejegening", "Vriendelijkheid")],
            "recurrence": {}, "n_passes": 3}
    result = FacetConsolidationResult(
        decision_summary=[],
        facets=[SettledFacet(
            facet_name="Snelheid", facet_definition="…",
            facet_question="Hoe snel?", source_facet_ids=[])])
    survivors = clf._facet_consolidation_survivors(task, result)
    by_name = {s.facet_name: s for s in survivors}
    assert [a.attribute_name for a in by_name["Snelheid"].attributes] == [
        "Wachttijd"]
    assert [a.attribute_name for a in by_name["Bejegening"].attributes] == [
        "Vriendelijkheid"]
    assert _actions(clf, "facet_claimed_by_name")[0]["id"] == "F1"


def test_an_invented_id_does_not_cost_the_candidate_its_attributes():
    """A garbled id claims nothing, but the name still identifies one candidate.
    Reporting the id and dropping the pool would be the worse half of both."""
    clf = _clf()
    task = {"domain_label": "d", "candidates": [_pool("Snelheid", "Wachttijd")],
            "recurrence": {}, "n_passes": 3}
    result = FacetConsolidationResult(
        decision_summary=[],
        facets=[SettledFacet(
            facet_name="Snelheid", facet_definition="…",
            facet_question="Hoe snel?", source_facet_ids=["F9"])])
    survivors = clf._facet_consolidation_survivors(task, result)
    assert len(survivors) == 1
    assert [a.attribute_name for a in survivors[0].attributes] == ["Wachttijd"]
    assert _actions(clf, "unknown_source_id")[0]["facets"] == ["F9"]


def test_a_name_fallback_does_not_pool_a_repeat_twice():
    clf = _clf()
    task = {"domain_label": "d",
            "candidates": [_pool("Snelheid", "Wachttijd"),
                           _pool("Tempo", "Wachttijd")],
            "recurrence": {}, "n_passes": 3}
    result = FacetConsolidationResult(
        decision_summary=[],
        facets=[SettledFacet(
            facet_name="Tempo", facet_definition="…",
            facet_question="Hoe snel?", source_facet_ids=["F1"])])
    survivors = clf._facet_consolidation_survivors(task, result)
    assert [a.attribute_name for a in survivors[0].attributes] == ["Wachttijd"]


# =============================================================================
# FACET CONSOLIDATION — THE ROUNDS
# =============================================================================

def _stub_dispatch(clf, answer):
    """Replace the LLM round-trip with a canned answer per task.

    Returns the list of task batches it was handed, one entry per round, so a
    test can assert that a round happened — or that it did not.
    """
    rounds = []

    async def dispatch(phase, tasks, prepare_fn, parse_fn, fallback_fn,
                       verbose, **kwargs):
        rounds.append(tasks)
        return [answer(task) for task in tasks]

    clf._dispatch = dispatch
    return rounds


def _settle(name, *sources):
    return SettledFacet(facet_name=name, facet_definition="…",
                        facet_question=f"Wat zegt dit over {name}?",
                        source_facet_ids=list(sources))


def test_a_domain_that_fits_one_group_is_settled_after_one_round():
    clf = _clf()
    rounds = _stub_dispatch(clf, lambda task: FacetConsolidationResult(
        decision_summary=[], facets=[_settle("Snelheid", "F1", "F2")]))
    raw = {"d": [_facet("Snelheid", "Wachttijd"),
                 _facet("Tempo", "Doorlooptijd")]}
    settled = asyncio.run(
        clf._run_facet_consolidation(_ctx({"d": []}), raw, verbose=False))
    assert len(rounds) == 1
    assert [p.facet_name for p in settled["d"]] == ["Snelheid"]
    assert [a.attribute_name for a in settled["d"][0].attributes] == [
        "Wachttijd", "Doorlooptijd"]


def test_a_single_candidate_domain_never_reaches_the_model():
    """One candidate is nothing to merge, and a call would invite the model to
    invent a distinction to justify itself."""
    clf = _clf()
    rounds = _stub_dispatch(clf, lambda task: None)
    raw = {"d": [_facet("Snelheid", "Wachttijd")]}
    settled = asyncio.run(
        clf._run_facet_consolidation(_ctx({"d": []}), raw, verbose=False))
    assert rounds == []
    assert [p.facet_name for p in settled["d"]] == ["Snelheid"]
    assert [a.attribute_name for a in settled["d"][0].attributes] == [
        "Wachttijd"]


def test_a_domain_that_never_converges_is_reported_not_dropped():
    """Rounds are capped, and the cap is the one place where survivors could
    quietly fall off the end. They are kept, and the ceiling is logged."""
    clf = _clf(facet_consolidation_max_facets_per_call=1,
               consolidation_max_rounds=2)
    rounds = _stub_dispatch(clf, lambda task: FacetConsolidationResult(
        decision_summary=[],
        facets=[_settle(task["candidates"][0].facet_name, "F1")]))
    raw = {"d": [_facet("Snelheid", "Wachttijd"),
                 _facet("Tempo", "Doorlooptijd")]}
    settled = asyncio.run(
        clf._run_facet_consolidation(_ctx({"d": []}), raw, verbose=False))
    assert len(rounds) == 2
    assert sorted(p.facet_name for p in settled["d"]) == ["Snelheid", "Tempo"]
    assert sorted(a.attribute_name
                  for p in settled["d"] for a in p.attributes) == [
        "Doorlooptijd", "Wachttijd"]
    assert _actions(clf, "consolidation_rounds_exhausted")[0]["remaining"] == 2


# =============================================================================
# FACET ASSIGNMENT
# =============================================================================

def test_een_taak_per_uniek_label_per_domein():
    """Ideeën met hetzelfde label worden één rep: één call beslist voor
    allemaal. Dat is geen batch, het is niet twee keer dezelfde vraag stellen."""
    clf = _clf()
    settled = {"d": [_pool("f1", "a1"), _pool("f2", "a2")]}
    labels = {"d": {"i1": "traag", "i2": "traag", "i3": "duur"}}
    tasks = clf._build_facet_assignment_tasks(_ctx({"d": []}), settled, labels)
    assert len(tasks) == 2


def test_het_vangnetfacet_staat_altijd_in_het_menu():
    """Zonder vangnet heeft een idee dat geen enkele facetvraag beantwoordt geen
    geldig antwoord, en dan komt `__UNASSIGNED__` terug — precies wat de
    één-poort-toewijzing ooit moest oplossen."""
    clf = _clf()
    tasks = clf._build_facet_assignment_tasks(
        _ctx({"d": []}), {"d": [_pool("f1", "a1")]}, {"d": {"i1": "traag"}})
    assert len(tasks) == 1
    assert [v["is_drain"] for v in tasks[0]["id_map"].values()] == [False, True]


def test_een_domein_zonder_inhoudelijk_facet_krijgt_geen_taak():
    """Kiezen tussen alleen een vangnet is geen keuze."""
    clf = _clf()
    tasks = clf._build_facet_assignment_tasks(
        _ctx({"d": []}), {"d": []}, {"d": {"i1": "traag"}})
    assert tasks == []


def test_het_menu_wordt_een_keer_per_domein_gebouwd():
    """Alle taken van een domein delen één id_map, anders zou F1 per call iets
    anders kunnen betekenen en is de uitkomst niet te remappen."""
    clf = _clf()
    settled = {"d": [_pool("f1", "a1"), _pool("f2", "a2")]}
    labels = {"d": {"i1": "traag", "i2": "duur"}}
    tasks = clf._build_facet_assignment_tasks(_ctx({"d": []}), settled, labels)
    assert tasks[0]["id_map"] == tasks[1]["id_map"]


# =============================================================================
# FACET SETTLE
# =============================================================================

def _settle_result(facets, moves=(), facet_ids=("F1", "F2"), attribute_ids=()):
    """Een resultaat van één facet_settle-call, tegen het per-call model.

    `moves` is een reeks `(attribuut_id, doel_facet_id)`-paren."""
    model = build_facet_settle_model(list(facet_ids), list(attribute_ids))
    return model(
        scratchpad="s", facets=list(facets),
        attribute_moves=[{"attribute_id": a, "to_facet_id": f} for a, f in moves])


def _settled(name, *source_ids, question="v"):
    return SettledFacetCard(facet_name=name, facet_definition="d",
                            facet_question=question,
                            source_facet_ids=list(source_ids))


def _settle_task(pools, id_map, attribute_ids=None):
    """`attribute_ids` mapt een id op het attribuut-OBJECT, niet op zijn naam.

    Op naam zou de verplaatsing stukgaan waar twee facetten van één domein
    dezelfde attribuutnaam dragen — en dat mag. Objectidentiteit overleeft de
    merge, want samenvouwen zet dezelfde objecten in een nieuwe pool."""
    if attribute_ids is None:
        attribute_ids = {f"A{i}": a for i, a in enumerate(
            (a for p in pools for a in p.attributes), start=1)}
    return {"domain_label": "d", "pools": pools,
            "id_map": id_map, "attribute_ids": attribute_ids}


def _names(pool):
    return [a.attribute_name for a in pool.attributes]


def test_een_domein_met_een_facet_krijgt_geen_taak():
    """Er valt niets te vergelijken, dus geen call."""
    clf = _clf()
    settled = {"d": [_pool("f1", "a1")]}
    id_maps = {"d": {"F1": {"facet_name": "f1", "is_drain": False}}}
    tasks = clf._build_facet_settle_tasks(
        _ctx({"d": []}), settled, {}, id_maps, {"d": {}})
    assert tasks == []


def test_de_tellingen_komen_van_echte_toewijzingen_en_het_vangnet_telt_apart():
    """Dit is de reden dat de fase hier staat: het aandeel is een telling van
    ideeën, niet van hoeveel chunks een naam voorstelden."""
    clf = _clf()
    settled = {"d": [_pool("f1", "a1"), _pool("f2", "a2")]}
    id_maps = {"d": {
        "F1": {"facet_name": "f1", "is_drain": False},
        "F2": {"facet_name": "f2", "is_drain": False},
        "F3": {"facet_name": "Overig", "is_drain": True}}}
    facet_assignments = {"i1": "F1", "i2": "F1", "i3": "F2", "i4": "F3"}
    labels = {"d": {"i1": "traag", "i2": "traag", "i3": "duur", "i4": "geen idee"}}
    task = clf._build_facet_settle_tasks(
        _ctx({"d": []}), settled, facet_assignments, id_maps, labels)[0]
    assert task["counts"] == {"f1": 2, "f2": 1}
    assert task["contents"]["f1"] == ["traag"]
    assert task["domain_total"] == 4
    assert task["shares"]["f1"] == 0.5
    assert task["drain_count"] == 1


def test_attribuut_ids_lopen_in_documentvolgorde_over_alle_pools_heen():
    clf = _clf()
    settled = {"d": [_pool("f1", "a1", "a2"), _pool("f2", "a3")]}
    id_maps = {"d": {"F1": {"facet_name": "f1", "is_drain": False},
                     "F2": {"facet_name": "f2", "is_drain": False}}}
    task = clf._build_facet_settle_tasks(
        _ctx({"d": []}), settled, {}, id_maps, {"d": {}})[0]
    assert list(task["id_map"]) == ["F1", "F2"]
    assert [a.attribute_name for a in task["attribute_ids"].values()] == [
        "a1", "a2", "a3"]


def test_twee_facetten_worden_er_een_en_hun_pools_plakken_aan_elkaar():
    """Attributen reizen mee. Er hangen op dit moment nog geen ideeën aan, dus
    dit kost niets — dat is de reden dat deze fase vóór de attribuuttoewijzing
    staat en niet erna."""
    clf = _clf()
    settled = {"d": [_pool("f1", "a1"), _pool("f2", "a2")]}
    task = _settle_task(settled["d"],
                        {"F1": {"facet_name": "f1"}, "F2": {"facet_name": "f2"}})
    result = _settle_result([_settled("f", "F1", "F2")])
    out = clf._apply_facet_settle(tasks=[task], results=[result], settled=settled)
    assert [p.facet_name for p in out["d"]] == ["f"]
    assert _names(out["d"][0]) == ["a1", "a2"]


def test_een_overlevend_facet_draagt_de_vraag_die_het_model_opschreef():
    """Een vraag die van één van de bronnen is overgenomen beschrijft de merge
    niet, dus het model schrijft hem opnieuw op en die versie reist door."""
    clf = _clf()
    settled = {"d": [_pool("f1", "a1"), _pool("f2", "a2")]}
    task = _settle_task(settled["d"],
                        {"F1": {"facet_name": "f1"}, "F2": {"facet_name": "f2"}}, {})
    result = _settle_result([_settled("f", "F1", "F2", question="Wat vraagt dit?")])
    out = clf._apply_facet_settle(tasks=[task], results=[result], settled=settled)
    assert out["d"][0].facet_question == "Wat vraagt dit?"


def test_een_ongeclaimd_facet_blijft_staan():
    """Zonder bronvelden ziet een samengevouwen facet er identiek uit aan een
    vergeten facet: allebei staan ze niet in het antwoord."""
    clf = _clf()
    settled = {"d": [_pool("f1", "a1"), _pool("f2", "a2")]}
    task = _settle_task(settled["d"],
                        {"F1": {"facet_name": "f1"}, "F2": {"facet_name": "f2"}}, {})
    result = _settle_result([_settled("f", "F1")])
    out = clf._apply_facet_settle(tasks=[task], results=[result], settled=settled)
    assert [p.facet_name for p in out["d"]] == ["f", "f2"]
    assert _actions(clf, "facet_kept_unclaimed_in_settle")[0]["facet"] == "f2"


def test_een_bron_die_twee_keer_wordt_geclaimd_gaat_naar_de_eerste():
    """Aan allebei geven zou één attribuut onder twee facetten laten overleven."""
    clf = _clf()
    settled = {"d": [_pool("f1", "a1"), _pool("f2", "a2")]}
    task = _settle_task(settled["d"],
                        {"F1": {"facet_name": "f1"}, "F2": {"facet_name": "f2"}}, {})
    result = _settle_result([_settled("x", "F1", "F2"), _settled("y", "F2")])
    out = clf._apply_facet_settle(tasks=[task], results=[result], settled=settled)
    assert _names(out["d"][0]) == ["a1", "a2"]
    assert _names(out["d"][1]) == []
    assert _actions(clf, "divided_source_facet_in_settle")


def test_een_attribuut_verhuist_naar_het_facet_dat_het_doel_noemt():
    clf = _clf()
    settled = {"d": [_pool("f1", "a1", "a2"), _pool("f2", "a3")]}
    task = _settle_task(settled["d"],
                        {"F1": {"facet_name": "f1"}, "F2": {"facet_name": "f2"}})
    result = _settle_result(
        [_settled("f1", "F1"), _settled("f2", "F2")],
        moves=[("A2", "F2")], attribute_ids=("A1", "A2", "A3"))
    out = clf._apply_facet_settle(tasks=[task], results=[result], settled=settled)
    assert _names(out["d"][0]) == ["a1"]
    assert _names(out["d"][1]) == ["a3", "a2"]
    assert _actions(clf, "attribute_moved_between_facets")[0]["attribute"] == "a2"


def test_een_verplaatsing_naar_een_verdwenen_facet_laat_het_attribuut_staan():
    """Het doelfacet kan door de merge van dezelfde call verdwenen zijn. Dan
    blijft het attribuut waar het stond, met een logregel — nooit stil. Dat is
    de les van de misfit-uitgang, hier afgedwongen in plaats van gehoopt."""
    clf = _clf()
    settled = {"d": [_pool("f1", "a1"), _pool("f2", "a2")]}
    task = _settle_task(settled["d"],
                        {"F1": {"facet_name": "f1"}, "F2": {"facet_name": "f2"}})
    result = _settle_result(
        [_settled("f", "F1", "F2")],
        moves=[("A1", "F2")], attribute_ids=("A1", "A2"))
    out = clf._apply_facet_settle(tasks=[task], results=[result], settled=settled)
    assert sorted(_names(out["d"][0])) == ["a1", "a2"]
    assert _actions(clf, "move_target_gone")


def test_een_mislukte_call_laat_het_domein_zoals_consolidatie_het_zette():
    clf = _clf()
    settled = {"d": [_pool("f1", "a1"), _pool("f2", "a2")]}
    task = _settle_task(settled["d"],
                        {"F1": {"facet_name": "f1"}, "F2": {"facet_name": "f2"}}, {})
    out = clf._apply_facet_settle(tasks=[task], results=[None], settled=settled)
    assert [p.facet_name for p in out["d"]] == ["f1", "f2"]
    assert _actions(clf, "facet_settle_failed")


def test_een_antwoord_dat_bijna_niets_verantwoordt_wordt_geweigerd():
    """Dezelfde poort als bij consolidatie: onder de helft is geen oordeel maar
    een steiger, en het net zou het resultaat gaan schrijven in plaats van
    repareren. Getoetst op de parse van déze fase, niet op het gedeelde
    hulpmiddel — de poort is pas een poort als hij ook aangeroepen wordt."""
    clf = _clf()
    pools = [_pool(f"f{i}", f"a{i}") for i in range(1, 11)]
    task = _settle_task(pools,
                        {f"F{i}": {"facet_name": f"f{i}"} for i in range(1, 11)}, {})
    result = _settle_result([_settled("f", "F1")],
                            facet_ids=tuple(f"F{i}" for i in range(1, 11)))
    with pytest.raises(ConsolidationCollapse):
        clf._facet_settle_parse_fn()(task, result)


# =============================================================================
# ATTRIBUTE CONSOLIDATION
# =============================================================================

def test_one_task_per_facet_across_every_domain():
    """The scope of an attribute call is one facet: that is the split."""
    clf = _clf()
    settled = {"d1": [_pool("A", "x", "x2"), _pool("B", "y", "y2")],
               "d2": [_pool("C", "z", "z2")]}
    tasks = clf._build_attribute_consolidation_tasks(_ctx({}), settled)
    assert len(tasks) == 3
    assert {(t["domain_label"], t["facet"].facet_name) for t in tasks} == {
        ("d1", "A"), ("d1", "B"), ("d2", "C")}


def test_a_facet_with_one_attribute_needs_no_call():
    """Nothing to merge. Paired with the two-attribute case, so a helper that
    stopped producing attributes could not make the skip pass vacuously."""
    clf = _clf()
    tasks = clf._build_attribute_consolidation_tasks(
        _ctx({}), {"d": [_pool("A", "x")]})
    assert tasks == []
    tasks = clf._build_attribute_consolidation_tasks(
        _ctx({}), {"d": [_pool("A", "x", "y")]})
    assert [t["facet"].facet_name for t in tasks] == ["A"]


def test_an_unclaimed_attribute_is_kept():
    clf = _clf()
    task = {"domain_label": "d", "facet": _pool("A", "x", "y"),
            "candidates": _pool("A", "x", "y").attributes,
            "recurrence": {}, "n_passes": 3}
    result = AttributeConsolidationResult(
        decision_summary=[],
        attributes=[SettledAttribute(
            attribute_name="x", attribute_definition="…",
            example_observations=["o"], source_attribute_ids=["A1"])])
    kept = clf._attribute_consolidation_survivors(task, result)
    assert sorted(a.attribute_name for a in kept) == ["x", "y"]
    assert any(e["action"] == "attribute_kept_unclaimed"
               for e in clf._action_log)


def test_a_failed_attribute_call_keeps_the_whole_pool():
    clf = _clf()
    pool = _pool("A", "x", "y")
    task = {"domain_label": "d", "facet": pool, "candidates": pool.attributes,
            "recurrence": {}, "n_passes": 3}
    kept = clf._attribute_consolidation_survivors(task, None)
    assert len(kept) == 2


def test_a_cited_attribute_id_that_was_never_handed_out_is_logged():
    """An invented id claims nothing, so the candidate it was meant to cover
    falls through to the unclaimed net rather than disappearing."""
    clf = _clf()
    pool = _pool("A", "x", "y")
    task = {"domain_label": "d", "facet": pool, "candidates": pool.attributes,
            "recurrence": {}, "n_passes": 3}
    result = AttributeConsolidationResult(
        decision_summary=[],
        attributes=[SettledAttribute(
            attribute_name="Snelheid", attribute_definition="…",
            example_observations=["o"], source_attribute_ids=["A1", "A9"])])
    kept = clf._attribute_consolidation_survivors(task, result)
    assert _actions(clf, "unknown_source_id")[0]["attributes"] == ["A9"]
    assert sorted(a.attribute_name for a in kept) == ["Snelheid", "y"]


def test_the_same_name_twice_in_one_pool_is_two_claims():
    """Names are not unique, not even inside a single facet. The name fallback
    counting for both candidates would silently treat one as absorbed — which is
    exactly what the ids were introduced to stop."""
    clf = _clf()
    pool = _pool("A", "verantwoordelijkheid", "verantwoordelijkheid")
    task = {"domain_label": "d", "facet": pool, "candidates": pool.attributes,
            "recurrence": {}, "n_passes": 3}
    result = AttributeConsolidationResult(
        decision_summary=[],
        attributes=[SettledAttribute(
            attribute_name="verantwoordelijkheid", attribute_definition="…",
            example_observations=["o"], source_attribute_ids=[])])
    kept = clf._attribute_consolidation_survivors(task, result)
    assert len(kept) == 3
    assert [e["id"] for e in _actions(clf, "attribute_kept_unclaimed")] == [
        "A1", "A2"]


def test_attribute_provenance_pins_the_level_below_the_facet():
    """Provenance is recorded at both levels, one phase each. The facet half
    says which candidate facet went where; this half does the same for the pool
    inside one settled facet, and names the facet it ran in."""
    clf = _clf()
    pool = _pool("A", "x", "y")
    task = {"domain_label": "d", "facet": pool, "candidates": pool.attributes,
            "recurrence": {}, "n_passes": 3}
    result = AttributeConsolidationResult(
        decision_summary=["x en y vielen samen"],
        attributes=[SettledAttribute(
            attribute_name="x", attribute_definition="…",
            example_observations=["o"], source_attribute_ids=["A1", "A2"])])
    clf._attribute_consolidation_survivors(task, result)
    entry = _actions(clf, "attribute_provenance")[0]
    assert entry["facet"] == "A"
    assert entry["attributes"][0]["source_attribute_ids"] == ["A1", "A2"]
    assert entry["decisions"] == ["x en y vielen samen"]


def test_the_structure_carries_no_source_fields():
    """The response models describe what an LLM proposed; the structure is a
    different thing and must not inherit their bookkeeping.

    `facet_question` is the one field that does cross: step 5 reads it, and it
    is what makes rule 1 checkable after the run.
    """
    clf = _clf()
    structure = clf._assemble_structure(
        {"d": [_pool("A", "x", question="Hoe snel?")]},
        {("d", 0): _pool("A", "x").attributes})
    card = structure["d"][0]
    assert set(card) == {
        "facet_name", "facet_definition", "facet_question", "attributes"}
    assert set(card["attributes"][0]) == {
        "attribute_name", "attribute_definition", "example_observations"}


def test_the_structure_keeps_the_facet_question():
    """Step 5 reads it, and it is what makes rule 1 checkable afterwards."""
    clf = _clf()
    structure = clf._assemble_structure(
        {"d": [_pool("A", "x", question="Hoe snel?")]},
        {("d", 0): [_pool("A", "x").attributes[0]]})
    assert structure["d"][0]["facet_question"] == "Hoe snel?"
    assert structure["d"][0]["attributes"][0]["attribute_name"] == "x"


def test_a_facet_with_no_entry_keeps_its_own_pool():
    """The skip and the fallback both leave a facet out of the consolidated map.
    Both must come out of the assembler holding what they went in with."""
    clf = _clf()
    structure = clf._assemble_structure(
        {"d": [_pool("A", "x"), _pool("B", "p", "q")]},
        {("d", 1): _pool("B", "p").attributes})
    by_name = {card["facet_name"]: card for card in structure["d"]}
    assert [a["attribute_name"] for a in by_name["A"]["attributes"]] == ["x"]
    assert [a["attribute_name"] for a in by_name["B"]["attributes"]] == ["p"]


def test_a_result_lands_on_the_facet_it_came_from():
    """Two cards of the same name must not read each other's entry."""
    clf = _clf()
    structure = clf._assemble_structure(
        {"d": [_pool("Snelheid", "solo"), _pool("Snelheid", "x", "y")]},
        {("d", 1): _pool("Snelheid", "x").attributes})
    assert [c["facet_name"] for c in structure["d"]] == ["Snelheid", "Snelheid"]
    assert [_attribute_names(structure, index=i) for i in (0, 1)] == [
        ["solo"], ["x"]]


# =============================================================================
# ATTRIBUTE CONSOLIDATION — THE ROUNDS
# =============================================================================

def _settle_attribute(candidate, *sources):
    return SettledAttribute(
        attribute_name=candidate.attribute_name,
        attribute_definition=candidate.attribute_definition,
        example_observations=list(candidate.example_observations),
        source_attribute_ids=list(sources))


def _fold_into_first(task):
    """A stub answer that folds a whole group into its first candidate.

    Every id is cited, because an uncited candidate is kept by the survivor net
    and the pool would never shrink — which is the net doing its job, not a
    merge the model made.
    """
    ids = [f"A{i}" for i in range(1, len(task["candidates"]) + 1)]
    return AttributeConsolidationResult(
        decision_summary=[],
        attributes=[_settle_attribute(task["candidates"][0], *ids)])


def _attribute_names(structure, label="d", index=0):
    return sorted(a["attribute_name"]
                  for a in structure[label][index]["attributes"])


def test_a_facet_whose_pool_fits_is_settled_after_one_round():
    clf = _clf()
    rounds = _stub_dispatch(clf, _fold_into_first)
    structure = asyncio.run(clf._run_attribute_consolidation(
        _ctx({"d": []}), {"d": [_pool("A", "x", "y")]}, verbose=False))
    assert len(rounds) == 1
    assert _attribute_names(structure) == ["x"]


def test_a_skipped_facet_never_reaches_the_model_and_keeps_its_attribute():
    clf = _clf()
    rounds = _stub_dispatch(clf, lambda task: None)
    structure = asyncio.run(clf._run_attribute_consolidation(
        _ctx({"d": []}), {"d": [_pool("A", "x")]}, verbose=False))
    assert rounds == []
    assert _attribute_names(structure) == ["x"]


def test_a_failed_call_leaves_the_facet_pool_whole_in_the_structure():
    """The fallback returns nothing, the survivor net returns everything, and
    the assembler must carry that all the way into the structure."""
    clf = _clf()
    _stub_dispatch(clf, lambda task: None)
    structure = asyncio.run(clf._run_attribute_consolidation(
        _ctx({"d": []}), {"d": [_pool("A", "x", "y", "z")]}, verbose=False))
    assert _attribute_names(structure) == ["x", "y", "z"]
    assert _actions(clf, "attribute_consolidation_failed")[0]["facet"] == "A"


def test_a_pool_wider_than_the_cap_rounds_until_one_call_saw_everything():
    """Groups never see each other, so a facet split over several is not settled
    until its survivors have been put back together in ONE call.

    Ending on "the survivors would now fit in one group" is a round too early:
    the second round here leaves v and z, which fit the cap but were judged in
    separate calls and never compared. The last round must be a single task —
    that is the invariant, and the round counts alone do not show it.
    """
    clf = _clf(attribute_consolidation_max_attributes_per_call=2)
    rounds = _stub_dispatch(clf, _fold_into_first)
    structure = asyncio.run(clf._run_attribute_consolidation(
        _ctx({"d": []}), {"d": [_pool("A", "v", "w", "x", "y", "z")]},
        verbose=False))
    assert [len(r) for r in rounds] == [3, 2, 1]
    assert len(rounds[-1]) == 1
    assert _attribute_names(structure) == ["v"]


def test_an_exhausted_round_budget_carries_every_attribute():
    """The round cap is the one place survivors could quietly fall off the end.
    A stub that merges nothing never converges, and still loses nothing."""
    clf = _clf(attribute_consolidation_max_attributes_per_call=2,
               consolidation_max_rounds=2)
    rounds = _stub_dispatch(clf, lambda task: AttributeConsolidationResult(
        decision_summary=[],
        attributes=[_settle_attribute(c, f"A{i}")
                    for i, c in enumerate(task["candidates"], 1)]))
    structure = asyncio.run(clf._run_attribute_consolidation(
        _ctx({"d": []}), {"d": [_pool("A", "x", "y", "z")]}, verbose=False))
    assert len(rounds) == 2
    assert _attribute_names(structure) == ["x", "y", "z"]
    entry = _actions(clf, "consolidation_rounds_exhausted")[0]
    assert (entry["facet"], entry["rounds"], entry["remaining"]) == ("A", 2, 3)


def test_a_phase_that_converged_reports_no_exhausted_rounds():
    """Paired with the test above: an unconditional log over what is left in
    `pending` would fire on the skip path too, and read as a phase in trouble."""
    clf = _clf()
    _stub_dispatch(clf, _fold_into_first)
    asyncio.run(clf._run_attribute_consolidation(
        _ctx({"d": []}), {"d": [_pool("A", "x"), _pool("B", "p", "q")]},
        verbose=False))
    assert _actions(clf, "consolidation_rounds_exhausted") == []
    assert _actions(clf, "attribute_consolidation")[0]["facet"] == "B"


def test_two_facets_sharing_a_name_keep_their_own_attributes():
    """A domain can hold two facets with the same name: task 4's facet net keeps
    both candidates whole when the name is ambiguous about which was meant, so
    one round is enough to produce it.

    Keyed on the name, the skipped facet was handed the other facet's result and
    its own attribute was gone for good — the exact failure class this refactor
    must not make worse. Keyed on position, each card keeps what is its own.
    """
    clf = _clf()
    rounds = _stub_dispatch(clf, _fold_into_first)
    structure = asyncio.run(clf._run_attribute_consolidation(
        _ctx({"d": []}),
        {"d": [_pool("Snelheid", "solo"), _pool("Snelheid", "x", "y")]},
        verbose=False))
    assert [len(r) for r in rounds] == [1]
    assert [_attribute_names(structure, index=i) for i in (0, 1)] == [
        ["solo"], ["x"]]


def test_a_partial_requeue_still_writes_back_to_the_right_facet():
    """`pending` shrinks between rounds, so from round two a pool's position in
    it no longer matches its position in `settled`. Here facet A settles in
    round one and facet B does not, which shifts B from position 1 to position
    0; without the translation back, B's round-two result is written onto A.

    The only test that separates the two: where every pool requeues, the
    mapping is the identity and a naive key looks correct.
    """
    clf = _clf(attribute_consolidation_max_attributes_per_call=2)
    rounds = _stub_dispatch(clf, _fold_into_first)
    structure = asyncio.run(clf._run_attribute_consolidation(
        _ctx({"d": []}),
        {"d": [_pool("A", "p", "q"),
               _pool("B", "v", "w", "x", "y", "z")]},
        verbose=False))
    assert [len(r) for r in rounds] == [4, 2, 1]
    assert [_attribute_names(structure, index=i) for i in (0, 1)] == [
        ["p"], ["v"]]


def test_a_second_round_writes_back_to_the_right_facet():
    """The re-round carries each pool's position along. Two same-named facets
    in one domain, both oversized, must not overwrite each other's result."""
    clf = _clf(attribute_consolidation_max_attributes_per_call=2)
    rounds = _stub_dispatch(clf, _fold_into_first)
    structure = asyncio.run(clf._run_attribute_consolidation(
        _ctx({"d": []}),
        {"d": [_pool("Snelheid", "a", "b", "c", "d", "e"),
               _pool("Snelheid", "v", "w", "x", "y", "z")]},
        verbose=False))
    assert [len(r) for r in rounds] == [6, 4, 2]
    assert [_attribute_names(structure, index=i) for i in (0, 1)] == [
        ["a"], ["v"]]


# =============================================================================
# ASSIGNMENT
# =============================================================================

def test_een_taak_per_uniek_label():
    """Identical text shares one judgement; that is not a batch."""
    structure = _structure({"d": [_facet("f", "a1", "a2")]})
    tasks = _clf()._build_assignment_tasks(
        _ctx({"d": []}), structure,
        {"d": {"i1": "zelfde", "i2": "zelfde", "i3": "ander"}})
    assert len(tasks) == 2
    assert sorted(len(t["rep"].idea_ids) for t in tasks) == [1, 2]


def test_a_menu_of_one_gets_no_task():
    structure = _structure({"d": [_facet("f", "enige")]})
    tasks = _clf()._build_assignment_tasks(
        _ctx({"d": []}), structure, {"d": {"i1": "x"}})
    assert tasks == []


def test_the_menu_is_domain_wide_and_grouped_per_facet():
    structure = _structure({"d": [_facet("f1", "a1"), _facet("f2", "a2")]})
    tasks = _clf()._build_assignment_tasks(
        _ctx({"d": []}), structure, {"d": {"i1": "x"}})
    id_map = tasks[0]["id_map"]
    assert {v["facet_name"] for v in id_map.values()} == {"f1", "f2"}


# =============================================================================
# VANGNETTEN AANHAKEN
# =============================================================================

def test_every_facet_gets_an_other_and_every_domain_an_other_facet():
    clf = _clf()
    structure = clf._add_drains(
        _ctx({"d": []}), _structure({"d": [_facet("f1", "a1"), _facet("f2", "a2")]}))
    drain_facets = [f for f in structure["d"] if is_drain_item(f)]
    assert len(drain_facets) == 1
    for facet in structure["d"]:
        assert any(is_drain_item(a) for a in facet["attributes"]), facet["facet_name"]


def test_catch_alls_arrive_after_consolidation_not_before():
    """Consolidation judges what the passes proposed; a bucket that exists by
    construction is not a proposal."""
    clf = _clf()
    voor = _structure({"d": [_facet("f1", "a1")]})
    na = clf._add_drains(_ctx({"d": []}), voor)
    assert len(voor["d"][0]["attributes"]) == 1
    assert len(na["d"][0]["attributes"]) == 2


# =============================================================================
# FACET ASSIGNMENT IS DERIVED
# =============================================================================

def test_de_plaatsing_valt_uiteen_in_de_twee_cacheregisters():
    """One source. Two separately determined registers could put an idea in
    facet F and on an attribute hanging under G; a placement cannot."""
    placements = {"i1": Placement("d", "f2", "a2"),
                  "i2": Placement("d", "f1", "a1")}
    attributen, facetten = flatten_placements(placements)
    assert attributen == {"i1": "a2", "i2": "a1"}
    assert facetten == {"d": {"i1": "f2", "i2": "f1"}}


def test_dezelfde_attribuutnaam_in_twee_facetten_blijft_uit_elkaar():
    """Waar de naamopzoeking de laatste liet winnen, houdt de plaatsing beide
    ideeën in hun eigen facet."""
    placements = {"i1": Placement("d", "f1", "Merkbekendheid"),
                  "i2": Placement("d", "f2", "Merkbekendheid")}
    _, facetten = flatten_placements(placements)
    assert facetten == {"d": {"i1": "f1", "i2": "f2"}}


# =============================================================================
# NASLIJPEN
# =============================================================================

def _refined(name, *sources, definition="d"):
    return RefinedAttribute(
        attribute_name=name, attribute_definition=definition,
        example_observations=["e"], source_attributes=list(sources))


def _card(facet_name, *attribute_names, drain=None):
    attributes = [{"attribute_name": a, "attribute_definition": "d",
                   "example_observations": ["e"]} for a in attribute_names]
    if drain:
        attributes.append(make_drain_attribute(facet_name, "Dutch"))
    return {"facet_name": facet_name, "facet_definition": "d",
            "facet_question": "", "attributes": attributes}


def test_a_catch_all_named_as_a_source_keeps_its_ideas():
    """Protecting the returned name and the card was not enough: a catch-all
    named as a SOURCE was remapped like any other attribute, so its ideas
    emptied into a content attribute while the drain card stayed in place and
    hid it."""
    clf = _clf()
    card = _card("f", "a1", drain=True)
    drain_name = card["attributes"][-1]["attribute_name"]
    structure = {"d": [card]}
    assignments = {"i1": Placement("d", "f", "a1"),
                   "i2": Placement("d", "f", drain_name)}
    result = RefinementResult(
        scratchpad="s", attributes=[_refined("Breed", "a1", drain_name)])

    structure, out = clf._apply_refinement(
        tasks=[{"domain_label": "d", "facet_index": 0, "facet": card}],
        results=[result], structure=structure, assignments=assignments)

    assert out["i1"] == Placement("d", "f", "Breed")
    assert out["i2"] == Placement("d", "f", drain_name)
    assert _actions(clf, "drain_source_ignored")[0]["sources"] == [drain_name]


def test_a_catch_all_survives_the_rebuild_as_a_card():
    clf = _clf()
    card = _card("f", "a1", drain=True)
    drain_name = card["attributes"][-1]["attribute_name"]
    structure, _ = clf._apply_refinement(
        tasks=[{"domain_label": "d", "facet_index": 0, "facet": card}],
        results=[RefinementResult(
            scratchpad="s", attributes=[_refined("Breed", "a1")])],
        structure={"d": [card]},
        assignments={"i1": Placement("d", "f", "a1")})
    assert [a["attribute_name"] for a in structure["d"][0]["attributes"]] == [
        "Breed", drain_name]


def test_refinement_moves_nothing_out_of_its_facet():
    """The misfit exit named its destinations before the phase and resolved
    them after, while renaming is the phase's whole job — 70% of routed texts
    landed on a name their neighbour had just consumed. It is gone, and so is
    every field it needed."""
    assert set(RefinementResult.model_fields) == {"scratchpad", "attributes"}


# =============================================================================
# DUBBELE NAMEN BINNEN EEN DOMEIN
# =============================================================================

def test_one_name_in_two_facets_folds_into_the_bigger_one():
    """Two refinement calls do not see each other and can land on one name.
    Within a domain, one name is one attribute — no model needed, and
    cross-domain no longer spends a merge on a name identical to itself."""
    clf = _clf()
    structure = {"d": [_card("f1", "Omvang"), _card("f2", "Omvang", "Rest")]}
    assignments = {
        "i1": Placement("d", "f1", "Omvang"),
        "i2": Placement("d", "f2", "Omvang"),
        "i3": Placement("d", "f2", "Omvang"),
        "i4": Placement("d", "f2", "Rest")}

    structure, out = clf._merge_duplicate_names(structure, assignments)

    assert [a["attribute_name"] for a in structure["d"][0]["attributes"]] == []
    assert [a["attribute_name"] for a in structure["d"][1]["attributes"]] == [
        "Omvang", "Rest"]
    assert out["i1"] == Placement("d", "f2", "Omvang")
    assert out["i4"] == Placement("d", "f2", "Rest")
    logged = _actions(clf, "duplicate_name_merged")[0]
    assert logged["kept_in"] == "f2" and logged["dropped_from"] == ["f1"]
    assert logged["n_ideas"] == 3


def test_a_tie_is_broken_by_position_not_by_chance():
    clf = _clf()
    structure = {"d": [_card("f1", "Omvang"), _card("f2", "Omvang")]}
    assignments = {"i1": Placement("d", "f1", "Omvang"),
                   "i2": Placement("d", "f2", "Omvang")}
    structure, out = clf._merge_duplicate_names(structure, assignments)
    assert out["i2"] == Placement("d", "f1", "Omvang")


def test_one_name_in_two_domains_is_left_alone():
    """Two domains settled separately; folding them is cross-domain's call,
    and it needs the model because the two may not mean the same thing."""
    clf = _clf()
    structure = {"d1": [_card("f", "Omvang")], "d2": [_card("f", "Omvang")]}
    assignments = {"i1": Placement("d1", "f", "Omvang"),
                   "i2": Placement("d2", "f", "Omvang")}
    _, out = clf._merge_duplicate_names(structure, assignments)
    assert out == assignments
    assert not _actions(clf, "duplicate_name_merged")


def test_a_catch_all_takes_no_part_in_the_fold():
    clf = _clf()
    structure = {"d": [_card("f1", drain=True), _card("f2", drain=True)]}
    names = [f["attributes"][0]["attribute_name"] for f in structure["d"]]
    structure, _ = clf._merge_duplicate_names(structure, {})
    assert [f["attributes"][0]["attribute_name"] for f in structure["d"]] == names


# =============================================================================
# UITPAKKEN NAAR DE TWEE CACHEREGISTERS
# =============================================================================

def test_facet_cards_do_not_carry_their_attributes():
    """The cache holds facets and attributes in two registers; the nesting that
    carries the structure through the run is unpacked at the end."""
    nested = _structure({"d": [_facet("f", "a1", "a2")]})["d"]
    assert "attributes" not in facet_dicts(nested)[0]
    assert [a["attribute_name"] for a in attribute_dicts(nested)["f"]] == ["a1", "a2"]


# =============================================================================
# STOPPUNTEN
# =============================================================================

def test_every_phase_name_is_a_valid_stop_point():
    for phase in TaxonomyClassifier.PHASES:
        _clf(stop_after_phase=phase)


# =============================================================================
# RATE LIMITS
# =============================================================================

def test_rate_limits_are_unpacked_not_repacked(monkeypatch):
    """`fetch_rate_limits` already returns (RateLimits, has_headers). Wrapping
    that in a tuple once more gave an AttributeError on the first print line —
    after setup, and therefore in the middle of a paid run.
    """
    import asyncio

    import pipeline.step_4_classifier.classifier as mod
    from utils.llm import RateLimits

    async def fake_fetch(model):
        return RateLimits(tokens_per_minute=1000, requests_per_minute=10), True

    monkeypatch.setattr(mod, "llm_fetch_rate_limits", fake_fetch)
    clf = _clf()
    asyncio.run(clf._initialize_async_resources(verbose=True))

    for phase in TaxonomyClassifier.PHASES:
        if phase == "valence_merge":
            continue
        limits = clf._limits_by_model[clf._model[phase]]
        assert limits.tokens_per_minute == 1000
        assert clf._has_headers_by_model[clf._model[phase]] is True


def test_nul_limieten_vallen_terug_op_de_fallback(monkeypatch):
    """A deployment that returns no limits must not run on zero."""
    import asyncio

    import pipeline.step_4_classifier.classifier as mod
    from config import FALLBACK_TPM
    from utils.llm import RateLimits

    async def fake_fetch(model):
        return RateLimits(tokens_per_minute=0, requests_per_minute=0), False

    monkeypatch.setattr(mod, "llm_fetch_rate_limits", fake_fetch)
    clf = _clf()
    asyncio.run(clf._initialize_async_resources(verbose=False))

    limits = clf._limits_by_model[clf._model["discovery"]]
    assert limits.tokens_per_minute == FALLBACK_TPM


# =============================================================================
# TELLEN
# =============================================================================

def test_the_count_separates_catch_alls_from_real_items():
    """A phase that counts the catch-alls next to a phase that does not reads as
    growth that is not there."""
    from pipeline.step_4_classifier.classifier import count_structure, format_counts
    clf = _clf()
    structure = clf._add_drains(
        _ctx({"d": []}), _structure({"d": [_facet("f1", "a1", "a2")]}))
    c = count_structure(structure)
    assert c == {"facets": 1, "drain_facets": 1,
                 "attributes": 2, "drain_attributes": 2}
    assert format_counts(structure) == (
        "1 facets, 2 attributes (+1 catch-all facets, 2 catch-all attributes)")


def test_a_count_without_catch_alls_does_not_mention_them():
    from pipeline.step_4_classifier.classifier import format_counts
    assert format_counts(_structure({"d": [_facet("f", "a")]})) == "1 facets, 1 attributes"
