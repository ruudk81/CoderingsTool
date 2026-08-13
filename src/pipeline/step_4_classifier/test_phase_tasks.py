"""Tests voor de taakvormen van de zes fasen (step 4).

De `_build_<fase>_tasks` zijn pure functies van hun argumenten, dus scope,
overslaan, chunking en tellingen zijn te controleren zonder een LLM-call.
"""
from pipeline.step_3_ideaExtractor.dimension_data import get_dimensions_in_decision_order
from pipeline.step_4_classifier.classifier import (
    PromptContext, TaxonomyClassifier, attribute_dicts, derive_facet_assignments,
    facet_dicts,
)
from pipeline.step_4_classifier.config_classifier import CategoriesConfig
from pipeline.step_4_classifier.drains import is_drain_item
from pipeline.step_4_classifier.prompts_discovery import (
    DiscoveredAttribute, DiscoveredFacet,
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

def test_discovery_slaat_de_vangnetdomeinen_over():
    """Step 3 definieert ze als bewust brede vangnetten; er structuur op leggen
    verzint onderscheid dat de antwoorden niet dragen."""
    ctx = _ctx({"inhoud": ["a"], "Overig": ["b"]}, drains=["overig"])
    tasks = _clf()._build_discovery_tasks(ctx)
    assert {t["domain_label"] for t in tasks} == {"inhoud"}


def test_vangnet_match_is_hoofdletterongevoelig():
    """Step 3 schrijft het label met hoofdletter, domeindiscovery maakt de
    partitienaam lowercase. Een exacte match vond geen van beide, stil."""
    ctx = _ctx({"Overig": ["b"]}, drains=["overig"])
    assert ctx.is_drain("Overig") is True
    assert _clf()._build_discovery_tasks(ctx) == []


def test_kleine_scope_krijgt_een_chunk():
    tasks = _clf()._build_discovery_tasks(_ctx({"d": ["a", "b", "c"]}))
    assert len(tasks) == 1
    assert tasks[0]["total_chunks"] == 1


def test_grote_scope_wordt_gechunkt_met_overlap():
    ctx = _ctx({"d": [f"obs{i}" for i in range(500)]})
    tasks = _clf()._build_discovery_tasks(ctx)
    assert len(tasks) > 1
    assert all(t["total_chunks"] == len(tasks) for t in tasks)
    gezien = {o for t in tasks for o in t["observations"]}
    assert gezien == set(ctx.domain("d")["observations"])


# =============================================================================
# CONSOLIDATIEGROEPEN
# =============================================================================

def test_groepen_tellen_attributen_niet_facetten():
    """Dertig facetten valt onder de cap terwijl er vijfhonderd attributen
    onder kunnen hangen, en daar bezwijkt het oordeel."""
    clf = _clf(consolidation_max_items_per_call=10)
    kandidaten = [_facet(f"f{i}", *[f"a{j}" for j in range(6)]) for i in range(4)]
    groepen = clf._consolidation_groups(kandidaten)
    assert len(groepen) == 4
    assert all(len(g) == 1 for g in groepen)


def test_alles_in_een_groep_wanneer_het_past():
    clf = _clf(consolidation_max_items_per_call=150)
    groepen = clf._consolidation_groups([_facet(f"f{i}", "a") for i in range(30)])
    assert len(groepen) == 1


def test_facet_reist_nooit_los_van_zijn_attributen():
    clf = _clf(consolidation_max_items_per_call=2)
    groot = _facet("groot", *[f"a{j}" for j in range(9)])
    groepen = clf._consolidation_groups([groot])
    assert len(groepen) == 1
    assert len(groepen[0][0].attributes) == 9


def test_kandidaten_worden_op_naam_gesorteerd_voor_het_groeperen():
    """Bijna-identieke namen komen naast elkaar te staan, zodat ze meestal in
    dezelfde groep vallen in plaats van elkaar een ronde lang mis te lopen.

    Meestal, niet altijd: een greedy vulling kan de groepsgrens nog steeds
    precies tussen twee buren leggen. Dat is aanvaard — de volgende ronde zet
    de overlevenden alsnog bij elkaar.
    """
    clf = _clf(consolidation_max_items_per_call=2)
    kandidaten = [_facet("Snelheid van afhandeling", "a"),
                  _facet("Bejegening", "b"), _facet("Snelheid", "c")]
    volgorde = [f.facet_name.lower()
                for g in clf._consolidation_groups(kandidaten) for f in g]
    assert volgorde == sorted(volgorde)


# =============================================================================
# TOEWIJZING
# =============================================================================

def test_een_taak_per_uniek_label():
    """Identieke tekst deelt een oordeel; dat is geen batch."""
    structure = _structure({"d": [_facet("f", "a1", "a2")]})
    tasks = _clf()._build_assignment_tasks(
        _ctx({"d": []}), structure,
        {"d": {"i1": "zelfde", "i2": "zelfde", "i3": "ander"}})
    assert len(tasks) == 2
    assert sorted(len(t["rep"].idea_ids) for t in tasks) == [1, 2]


def test_menu_van_een_krijgt_geen_taak():
    structure = _structure({"d": [_facet("f", "enige")]})
    tasks = _clf()._build_assignment_tasks(
        _ctx({"d": []}), structure, {"d": {"i1": "x"}})
    assert tasks == []


def test_menu_is_domeinbreed_en_per_facet_gegroepeerd():
    structure = _structure({"d": [_facet("f1", "a1"), _facet("f2", "a2")]})
    tasks = _clf()._build_assignment_tasks(
        _ctx({"d": []}), structure, {"d": {"i1": "x"}})
    id_map = tasks[0]["id_map"]
    assert {v["facet_name"] for v in id_map.values()} == {"f1", "f2"}


# =============================================================================
# VANGNETTEN AANHAKEN
# =============================================================================

def test_elk_facet_krijgt_een_other_en_elk_domein_een_other_facet():
    clf = _clf()
    structure = clf._add_drains(
        _ctx({"d": []}), _structure({"d": [_facet("f1", "a1"), _facet("f2", "a2")]}))
    drain_facets = [f for f in structure["d"] if is_drain_item(f)]
    assert len(drain_facets) == 1
    for facet in structure["d"]:
        assert any(is_drain_item(a) for a in facet["attributes"]), facet["facet_name"]


def test_vangnetten_komen_na_consolidatie_niet_ervoor():
    """Consolidatie beoordeelt wat de passes voorstelden; een bak die per
    constructie bestaat is geen voorstel."""
    clf = _clf()
    voor = _structure({"d": [_facet("f1", "a1")]})
    na = clf._add_drains(_ctx({"d": []}), voor)
    assert len(voor["d"][0]["attributes"]) == 1
    assert len(na["d"][0]["attributes"]) == 2


# =============================================================================
# FACETTOEWIJZING IS AFGELEID
# =============================================================================

def test_facettoewijzing_volgt_waar_het_attribuut_leeft():
    """Een bron. Twee los bepaalde toewijzingen konden een idee in facet F
    zetten en in een attribuut dat onder G hangt."""
    structure = _structure({"d": [_facet("f1", "a1"), _facet("f2", "a2")]})
    assert derive_facet_assignments({"i1": "a2"}, structure) == {"d": {"i1": "f2"}}


def test_onbekend_attribuut_levert_geen_facet_op():
    structure = _structure({"d": [_facet("f1", "a1")]})
    assert derive_facet_assignments({"i1": "verzonnen"}, structure) == {}


def test_facettoewijzing_matcht_hoofdletterongevoelig():
    structure = _structure({"d": [_facet("f1", "Wachttijd")]})
    assert derive_facet_assignments({"i1": "wachttijd"}, structure) == {"d": {"i1": "f1"}}


# =============================================================================
# UITPAKKEN NAAR DE TWEE CACHEREGISTERS
# =============================================================================

def test_facetkaarten_dragen_hun_attributen_niet_mee():
    """De cache houdt facetten en attributen in twee registers; de nesting die
    de structuur door de run draagt wordt aan het eind uitgepakt."""
    nested = _structure({"d": [_facet("f", "a1", "a2")]})["d"]
    assert "attributes" not in facet_dicts(nested)[0]
    assert [a["attribute_name"] for a in attribute_dicts(nested)["f"]] == ["a1", "a2"]


# =============================================================================
# STOPPUNTEN
# =============================================================================

def test_elke_fasenaam_is_een_geldig_stoppunt():
    for phase in TaxonomyClassifier.PHASES:
        _clf(stop_after_phase=phase)
