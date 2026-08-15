"""Tests for the task shapes of the six phases (step 4).

The `_build_<phase>_tasks` are pure functions of their arguments, so scope,
skipping, chunking and counts can be checked without an LLM call.
"""
from pipeline.step_3_ideaExtractor.dimension_data import get_dimensions_in_decision_order
from pipeline.step_4_classifier.classifier import (
    PromptContext, TaxonomyClassifier, _strip_enumeration, attribute_dicts,
    derive_facet_assignments, facet_dicts,
)
from pipeline.step_4_classifier.config_classifier import CategoriesConfig
from pipeline.step_4_classifier.drains import is_drain_item
from pipeline.step_4_classifier.prompts_discovery import (
    DiscoveredAttribute, DiscoveredFacet,
)
from pipeline.step_4_classifier.prompts_consolidation import (
    ConsolidatedAttribute, ConsolidatedFacet, ConsolidationResult,
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


def _structure(facets_per_domain):
    return {d: [f.model_dump() for f in facets]
            for d, facets in facets_per_domain.items()}


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
# CONSOLIDATIEGROEPEN
# =============================================================================

def test_groups_count_attributes_not_facets():
    """Thirty facets falls under the cap while five hundred attributes can hang
    beneath them, and that is where the judgement gives way."""
    clf = _clf(consolidation_max_items_per_call=10)
    kandidaten = [_facet(f"f{i}", *[f"a{j}" for j in range(6)]) for i in range(4)]
    groepen = clf._consolidation_groups(kandidaten)
    assert len(groepen) == 4
    assert all(len(g) == 1 for g in groepen)


def test_everything_in_one_group_when_it_fits():
    clf = _clf(consolidation_max_items_per_call=150)
    groepen = clf._consolidation_groups([_facet(f"f{i}", "a") for i in range(30)])
    assert len(groepen) == 1


def test_a_facet_never_travels_apart_from_its_attributes():
    clf = _clf(consolidation_max_items_per_call=2)
    groot = _facet("groot", *[f"a{j}" for j in range(9)])
    groepen = clf._consolidation_groups([groot])
    assert len(groepen) == 1
    assert len(groepen[0][0].attributes) == 9


def test_candidates_are_sorted_by_name_before_grouping():
    """Near-identical names end up next to each other, so they usually fall in
    the same group instead of missing each other for a whole round.

    Usually, not always: a greedy fill can still place the group boundary
    exactly between two neighbours. That is accepted — the next round puts the
    survivors together after all.
    """
    clf = _clf(consolidation_max_items_per_call=2)
    kandidaten = [_facet("Snelheid van afhandeling", "a"),
                  _facet("Bejegening", "b"), _facet("Snelheid", "c")]
    volgorde = [f.facet_name.lower()
                for g in clf._consolidation_groups(kandidaten) for f in g]
    assert volgorde == sorted(volgorde)


# =============================================================================
# CONSOLIDATIE-OVERLEVENDEN
# =============================================================================

def _survivor(name, *attrs, sources=(), attr_sources=None, question=""):
    """One returned facet, with the candidate ids it claims at both levels."""
    attr_sources = attr_sources or {}
    return ConsolidatedFacet(
        facet_name=name, facet_definition="d",
        facet_question=question or f"Wat zegt dit over {name}?",
        source_facet_ids=list(sources),
        attributes=[ConsolidatedAttribute(
            attribute_name=a, attribute_definition="d",
            example_observations=["e"],
            source_attribute_ids=list(attr_sources.get(a, []))) for a in attrs])


def _survivors(clf, candidates, facets):
    task = {"domain_label": "d", "candidates": candidates}
    return clf._consolidation_survivors(task, ConsolidationResult(
        decision_summary=[], facets=facets))


def _actions(clf, action):
    return [e for e in clf._action_log if e["action"] == action]


def test_an_unclaimed_facet_stays():
    """Merging and forgetting look identical in the output; without the net a
    facet the model never judged would vanish for good."""
    clf = _clf()
    kandidaten = [_facet("Snelheid", "wachttijd"), _facet("Bejegening", "toon")]
    overlevenden = _survivors(clf, kandidaten, [
        _survivor("Snelheid", "wachttijd", sources=["F1"],
                  attr_sources={"wachttijd": ["F1-A1"]})])
    assert {f.facet_name for f in overlevenden} == {"Snelheid", "Bejegening"}
    assert _actions(clf, "facet_kept_unclaimed")[0]["facet"] == "Bejegening"


def test_an_attribute_under_a_claimed_facet_stays():
    """The facet-level net reports full coverage here: the model absorbed the
    container and dropped its contents, and only an attribute-level check sees
    that."""
    clf = _clf()
    kandidaten = [_facet("Snelheid", "wachttijd", "doorlooptijd")]
    overlevenden = _survivors(clf, kandidaten, [
        _survivor("Snelheid", "wachttijd", sources=["F1"],
                  attr_sources={"wachttijd": ["F1-A1"]})])
    assert len(overlevenden) == 1
    assert [a.attribute_name for a in overlevenden[0].attributes] == [
        "wachttijd", "doorlooptijd"]
    gered = _actions(clf, "attribute_kept_unclaimed")[0]
    assert gered["attribute"] == "doorlooptijd"
    assert gered["landed_on"] == "Snelheid"


def test_a_rescued_attribute_lands_on_the_survivor_that_absorbed_its_facet():
    clf = _clf()
    kandidaten = [_facet("Snelheid", "wachttijd"), _facet("Tempo", "doorlooptijd")]
    overlevenden = _survivors(clf, kandidaten, [
        _survivor("Snelheid", "wachttijd", sources=["F1", "F2"],
                  attr_sources={"wachttijd": ["F1-A1"]})])
    assert len(overlevenden) == 1
    assert _actions(clf, "attribute_kept_unclaimed")[0]["landed_on"] == "Snelheid"


def test_a_wholly_kept_facet_brings_its_attributes_once():
    """Rescuing them again would list them twice."""
    clf = _clf()
    kandidaten = [_facet("Snelheid", "wachttijd"), _facet("Bejegening", "toon")]
    overlevenden = _survivors(clf, kandidaten, [
        _survivor("Snelheid", "wachttijd", sources=["F1"],
                  attr_sources={"wachttijd": ["F1-A1"]})])
    namen = [a.attribute_name for f in overlevenden for a in f.attributes]
    assert namen.count("toon") == 1
    assert not _actions(clf, "attribute_kept_unclaimed")


def test_a_relocated_attribute_counts_as_claimed():
    """Step 6 may move an attribute to the facet where it belongs; the claim is
    domain-wide, so that is not a loss."""
    clf = _clf()
    kandidaten = [_facet("Snelheid", "wachttijd"), _facet("Bejegening", "toon")]
    overlevenden = _survivors(clf, kandidaten, [
        _survivor("Snelheid", "wachttijd", sources=["F1"],
                  attr_sources={"wachttijd": ["F1-A1"]}),
        _survivor("Bejegening", "toon", sources=["F2"],
                  attr_sources={"toon": ["F2-A1"]})])
    assert not _actions(clf, "attribute_kept_unclaimed")
    assert sum(len(f.attributes) for f in overlevenden) == 2


def test_an_invented_source_id_is_reported():
    """An invented source claims nothing, which silently turns the candidate it
    was meant to cover into an unclaimed one. On ids the check is clean: the set
    was handed out in this very prompt."""
    clf = _clf()
    kandidaten = [_facet("Snelheid", "wachttijd")]
    _survivors(clf, kandidaten, [
        _survivor("Snelheid", "wachttijd", sources=["F1", "F9"],
                  attr_sources={"wachttijd": ["F1-A1"]})])
    assert _actions(clf, "unknown_source_id")[0]["facets"] == ["F9"]


def test_the_same_attribute_name_under_two_facets_is_two_claims():
    """Names are not unique. Claiming one used to count for both, so the other
    was silently treated as absorbed."""
    clf = _clf()
    kandidaten = [_facet("Ecologie", "verantwoordelijkheid"),
                  _facet("Samenleving", "verantwoordelijkheid")]
    _survivors(clf, kandidaten, [
        _survivor("Ecologie", "verantwoordelijkheid", sources=["F1", "F2"],
                  attr_sources={"verantwoordelijkheid": ["F1-A1"]})])
    gered = _actions(clf, "attribute_kept_unclaimed")
    assert [g["id"] for g in gered] == ["F2-A1"]


def test_provenance_pins_both_levels():
    """Survivors are rebuilt as plain structure, so `source_*` dies with the
    phase unless the action log keeps it."""
    clf = _clf()
    kandidaten = [_facet("Snelheid", "wachttijd"), _facet("Tempo", "doorlooptijd")]
    _survivors(clf, kandidaten, [
        _survivor("Snelheid", "wachttijd", sources=["F1", "F2"],
                  attr_sources={"wachttijd": ["F1-A1", "F2-A1"]})])
    entry = _actions(clf, "consolidation_provenance")[0]["facets"][0]
    assert entry["source_facet_ids"] == ["F1", "F2"]
    assert entry["facet_question"]
    assert entry["attributes"][0]["source_attribute_ids"] == ["F1-A1", "F2-A1"]


def test_the_structure_carries_no_source_fields():
    """The response model describes what an LLM proposed; the structure is a
    different thing and must not inherit its bookkeeping."""
    clf = _clf()
    overlevenden = _survivors(clf, [_facet("Snelheid", "wachttijd")], [
        _survivor("Snelheid", "wachttijd", sources=["F1"],
                  attr_sources={"wachttijd": ["F1-A1"]})])
    kaart = overlevenden[0].model_dump()
    assert "source_facet_ids" not in kaart
    assert "facet_question" not in kaart
    assert set(kaart["attributes"][0]) == {
        "attribute_name", "attribute_definition", "example_observations"}


def test_two_survivors_stating_one_question_are_reported():
    """Rule 1 forbids exactly this, and written-down questions are what make it
    visible. Logged, not repaired: merging here would override a judgement the
    model made with the candidates in view."""
    clf = _clf()
    kandidaten = [_facet("Snelheid", "wachttijd"), _facet("Tempo", "doorlooptijd")]
    _survivors(clf, kandidaten, [
        _survivor("Snelheid", "wachttijd", sources=["F1"],
                  attr_sources={"wachttijd": ["F1-A1"]},
                  question="Hoe snel gaat het?"),
        _survivor("Tempo", "doorlooptijd", sources=["F2"],
                  attr_sources={"doorlooptijd": ["F2-A1"]},
                  question="hoe snel gaat het? ")])
    melding = _actions(clf, "duplicate_facet_question")[0]
    assert melding["facets"] == ["Snelheid", "Tempo"]


def test_distinct_questions_are_not_reported():
    clf = _clf()
    kandidaten = [_facet("Snelheid", "wachttijd"), _facet("Bejegening", "toon")]
    _survivors(clf, kandidaten, [
        _survivor("Snelheid", "wachttijd", sources=["F1"],
                  attr_sources={"wachttijd": ["F1-A1"]},
                  question="Hoe snel gaat het?"),
        _survivor("Bejegening", "toon", sources=["F2"],
                  attr_sources={"toon": ["F2-A1"]},
                  question="Hoe wordt men bejegend?")])
    assert not _actions(clf, "duplicate_facet_question")


def test_a_candidate_split_across_survivors_is_reported():
    """Not a breach: step 6 may move an attribute, so a divided candidate is the
    honest record. It does leave its orphans an arbitrary landing spot."""
    clf = _clf()
    kandidaten = [_facet("Thema's", "ecologisch", "sociaal")]
    _survivors(clf, kandidaten, [
        _survivor("Ecologie", "ecologisch", sources=["F1"],
                  attr_sources={"ecologisch": ["F1-A1"]},
                  question="Waar gaat het ecologisch over?"),
        _survivor("Samenleving", "sociaal", sources=["F1"],
                  attr_sources={"sociaal": ["F1-A2"]},
                  question="Waar gaat het sociaal over?")])
    melding = _actions(clf, "divided_source_facet")[0]
    assert melding["source"] == "F1"
    assert melding["claimants"] == ["Ecologie", "Samenleving"]


# =============================================================================
# TOEWIJZING
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
# FACETTOEWIJZING IS AFGELEID
# =============================================================================

def test_facet_assignment_follows_where_the_attribute_lives():
    """One source. Two separately determined assignments could put an idea in
    facet F and in an attribute hanging under G."""
    structure = _structure({"d": [_facet("f1", "a1"), _facet("f2", "a2")]})
    assert derive_facet_assignments({"i1": "a2"}, structure) == {"d": {"i1": "f2"}}


def test_an_unknown_attribute_yields_no_facet():
    structure = _structure({"d": [_facet("f1", "a1")]})
    assert derive_facet_assignments({"i1": "verzonnen"}, structure) == {}


def test_facettoewijzing_matcht_hoofdletterongevoelig():
    structure = _structure({"d": [_facet("f1", "Wachttijd")]})
    assert derive_facet_assignments({"i1": "wachttijd"}, structure) == {"d": {"i1": "f1"}}


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
