"""Taakopbouwtests voor de step-4-fasen — geen LLM-calls.

Wat er in bedrading fout gaat is de vorm van de taken: een domein dat wel of
niet meedoet, een chunk die niet gesplitst wordt, een batch die te groot is,
een niveau dat een taak krijgt terwijl er niets te kiezen valt. Dat is precies
wat hier getoetst wordt, en het is toetsbaar omdat elke `_build_<fase>_tasks`
een pure functie van zijn argumenten is.
"""
from pipeline.step_3_ideaExtractor.dimension_data import get_dimensions_in_decision_order
from pipeline.step_4_classifier.classifier import PromptContext, TaxonomyClassifier
from pipeline.step_4_classifier.config_classifier import CategoriesConfig
from pipeline.step_4_classifier.prompts_attribute import (
    ConsolidatedAttribute, DiscoveredAttribute,
)
from pipeline.step_4_classifier.prompts_facet import ConsolidatedFacet, DiscoveredFacet

DIM = get_dimensions_in_decision_order()[0]


def _facet(name):
    return DiscoveredFacet(
        facet_name=name, facet_definition="d", boundary_test="b?",
        exclusions=["x"], example_observations=["e"],
    )


def _consolidated(name):
    return ConsolidatedFacet(
        facet_name=name, facet_definition="d", boundary_test="b?",
        exclusions=["x"], example_observations=["e"], source_facets=[name],
    )


def _attr(name):
    return DiscoveredAttribute(
        attribute_name=name, attribute_definition="d", boundary_test="b?",
        exclusions=["x"], example_observations=["e"],
    )


def _consolidated_attr(name):
    return ConsolidatedAttribute(
        attribute_name=name, attribute_definition="d", boundary_test="b?",
        exclusions=["x"], example_observations=["e"], source_attributes=[name],
    )


def _fixture_context(domains, drain_labels=frozenset(), observations=None):
    """Minimale PromptContext: taal, vraag, de vijf specifiers, de dimensie,
    en per domein label/definitie/boundary_test/exclusions."""
    observations = observations or {}
    return PromptContext(
        language="Dutch", survey_question="Waar denkt u aan?",
        sector="finance", entity="e", topic="t",
        perspective="consumer", intent="associate",
        dimension=DIM,
        domains={d: {"label": d, "definition": "def", "boundary_test": "b?",
                     "exclusions": [], "observations": observations.get(d, [])}
                 for d in domains},
        drain_labels=set(drain_labels),
    )


# =============================================================================
# Facet discovery en consolidatie
# =============================================================================

def test_discovery_slaat_de_staande_domeinen_over():
    """De twee staande domeinen krijgen geen facetten: step 3 definieert ze als
    brede vangnetten, en daar hoor je geen structuur in aan te brengen."""
    clf = TaxonomyClassifier(CategoriesConfig())
    ctx = _fixture_context(["Dienstverlening", "Overig", "Niet bekend"],
                           drain_labels={"Overig", "Niet bekend"})
    tasks = clf._build_facet_discovery_tasks(ctx)
    assert {t["domain_label"] for t in tasks} == {"Dienstverlening"}


def test_discovery_chunkt_grote_domeinen():
    clf = TaxonomyClassifier(CategoriesConfig())
    ctx = _fixture_context(["A", "B"],
                           observations={"A": [f"a{i}" for i in range(600)],
                                         "B": ["b1", "b2"]})
    tasks = clf._build_facet_discovery_tasks(ctx)
    assert len([t for t in tasks if t["domain_label"] == "B"]) == 1
    assert len([t for t in tasks if t["domain_label"] == "A"]) > 1


def test_consolidatie_is_een_taak_per_domein():
    clf = TaxonomyClassifier(CategoriesConfig())
    ctx = _fixture_context(["A", "B"])
    raw = {"A": [_facet("a1"), _facet("a2")], "B": [_facet("b1")]}
    tasks = clf._build_facet_consolidation_tasks(ctx, raw)
    assert {t["domain_label"] for t in tasks} == {"A", "B"}


def test_consolidatie_krijgt_alle_kandidaten_van_zijn_domein():
    clf = TaxonomyClassifier(CategoriesConfig())
    ctx = _fixture_context(["A"])
    raw = {"A": [_facet("a1"), _facet("a2"), _facet("a3")]}
    tasks = clf._build_facet_consolidation_tasks(ctx, raw)
    assert len(tasks[0]["candidates"]) == 3


def test_domein_zonder_kandidaten_krijgt_geen_consolidatietaak():
    clf = TaxonomyClassifier(CategoriesConfig())
    ctx = _fixture_context(["A", "B"])
    tasks = clf._build_facet_consolidation_tasks(ctx, {"A": [_facet("a1")], "B": []})
    assert [t["domain_label"] for t in tasks] == ["A"]


def test_consolidatie_splitst_boven_de_grens_in_groepen():
    """Meer kandidaten dan in één call passen worden in rondes geconsolideerd;
    de eerste ronde is meerdere taken voor hetzelfde domein."""
    clf = TaxonomyClassifier(CategoriesConfig(consolidation_max_items_per_call=2))
    ctx = _fixture_context(["A"])
    raw = {"A": [_facet(f"a{i}") for i in range(5)]}
    tasks = clf._build_facet_consolidation_tasks(ctx, raw)
    assert len(tasks) == 3
    assert sum(len(t["candidates"]) for t in tasks) == 5


# =============================================================================
# Facet toewijzing
# =============================================================================

def test_toewijzing_batcht_unieke_labels():
    clf = TaxonomyClassifier(CategoriesConfig(assignment_batch_k=2))
    ctx = _fixture_context(["A"])
    facets = {"A": [_consolidated("f1"), _consolidated("f2")]}
    labels = {"A": {"i1": "groen", "i2": "groen", "i3": "duur", "i4": "snel"}}
    tasks = clf._build_facet_assignment_tasks(ctx, facets, labels)
    # drie unieke labels bij K=2 → twee batches
    assert len(tasks) == 2


def test_identiek_label_wordt_een_rep():
    clf = TaxonomyClassifier(CategoriesConfig(assignment_batch_k=5))
    ctx = _fixture_context(["A"])
    facets = {"A": [_consolidated("f1"), _consolidated("f2")]}
    labels = {"A": {"i1": "groen", "i2": "  GROEN "}}
    tasks = clf._build_facet_assignment_tasks(ctx, facets, labels)
    assert len(tasks) == 1
    assert len(tasks[0]["reps"]) == 1
    assert sorted(tasks[0]["reps"][0].idea_ids) == ["i1", "i2"]


def test_domein_met_een_facet_krijgt_geen_taak():
    """Auto-assign: bij één facet is er niets te kiezen."""
    clf = TaxonomyClassifier(CategoriesConfig())
    ctx = _fixture_context(["A"])
    tasks = clf._build_facet_assignment_tasks(
        ctx, {"A": [_consolidated("enig")]}, {"A": {"i1": "x"}})
    assert tasks == []


# =============================================================================
# Facet naslijpen
# =============================================================================

def test_naslijptaak_per_domein_met_minstens_twee_facetten():
    clf = TaxonomyClassifier(CategoriesConfig())
    ctx = _fixture_context(["A", "B"])
    facets = {"A": [_consolidated("f1"), _consolidated("f2")], "B": [_consolidated("g1")]}
    assignments = {"A": {"i1": "f1", "i2": "f2"}, "B": {"i3": "g1"}}
    tasks = clf._build_facet_refinement_tasks(ctx, facets, assignments, labels={})
    assert [t["domain_label"] for t in tasks] == ["A"]


def test_naslijptaak_draagt_aantallen_en_aandelen():
    clf = TaxonomyClassifier(CategoriesConfig())
    ctx = _fixture_context(["A"])
    facets = {"A": [_consolidated("f1"), _consolidated("f2")]}
    assignments = {"A": {"i1": "f1", "i2": "f1", "i3": "f2"}}
    labels = {"A": {"i1": "x", "i2": "y", "i3": "z"}}
    tasks = clf._build_facet_refinement_tasks(ctx, facets, assignments, labels)
    rows = {naam: (n, aandeel) for naam, n, aandeel, _ in tasks[0]["rows"]}
    assert rows["f1"][0] == 2
    assert abs(rows["f1"][1] - 2 / 3) < 0.01


# =============================================================================
# Attribuut discovery en consolidatie
# =============================================================================

def test_attribuutdiscovery_chunkt_grote_facetten():
    clf = TaxonomyClassifier(CategoriesConfig())
    ctx = _fixture_context(["A"])
    facets = {"A": [_consolidated("f1"), _consolidated("f2")]}
    ideas_per_facet = {("A", "f1"): ["x"] * 300, ("A", "f2"): ["y"] * 10}
    tasks = clf._build_attribute_discovery_tasks(ctx, facets, ideas_per_facet)
    assert len([t for t in tasks if t["facet_name"] == "f2"]) == 1
    assert len([t for t in tasks if t["facet_name"] == "f1"]) > 1


def test_attribuutconsolidatie_is_een_taak_per_facet():
    clf = TaxonomyClassifier(CategoriesConfig())
    ctx = _fixture_context(["A"])
    raw = {"A": {"f1": [_attr("a1"), _attr("a2")], "f2": [_attr("b1")]}}
    tasks = clf._build_attribute_consolidation_tasks(ctx, raw)
    assert {t["facet_name"] for t in tasks} == {"f1", "f2"}


# =============================================================================
# Attribuut toewijzing
# =============================================================================

def test_attribuuttoewijzing_batcht_binnen_het_facet():
    clf = TaxonomyClassifier(CategoriesConfig(assignment_batch_k=2))
    ctx = _fixture_context(["A"])
    attrs = {"A": {"f1": [_consolidated_attr("a1"), _consolidated_attr("a2")]}}
    ideas = {("A", "f1"): {"i1": "x", "i2": "y", "i3": "z"}}
    tasks = clf._build_attribute_assignment_tasks(ctx, attrs, ideas)
    assert len(tasks) == 2
    assert all(t["facet_name"] == "f1" for t in tasks)


def test_facet_met_een_attribuut_krijgt_geen_taak():
    clf = TaxonomyClassifier(CategoriesConfig())
    ctx = _fixture_context(["A"])
    tasks = clf._build_attribute_assignment_tasks(
        ctx, {"A": {"f1": [_consolidated_attr("enig")]}}, {("A", "f1"): {"i1": "x"}})
    assert tasks == []


# =============================================================================
# Attribuut naslijpen
# =============================================================================

def test_naslijptaak_per_facet_met_minstens_twee_attributen():
    clf = TaxonomyClassifier(CategoriesConfig())
    ctx = _fixture_context(["A"])
    attrs = {"A": {"f1": [_consolidated_attr("a1"), _consolidated_attr("a2")],
                   "f2": [_consolidated_attr("b1")]}}
    assignments = {"i1": "a1", "i2": "a2", "i3": "b1"}
    tasks = clf._build_attribute_refinement_tasks(ctx, attrs, assignments, labels={})
    assert [t["facet_name"] for t in tasks] == ["f1"]


def test_naslijptaak_krijgt_de_buurfacetten_mee():
    clf = TaxonomyClassifier(CategoriesConfig())
    ctx = _fixture_context(["A"])
    attrs = {"A": {"f1": [_consolidated_attr("a1"), _consolidated_attr("a2")],
                   "f2": [_consolidated_attr("b1")]}}
    tasks = clf._build_attribute_refinement_tasks(
        ctx, attrs, {"i1": "a1", "i2": "a2", "i3": "b1"}, labels={})
    assert "f2" in tasks[0]["neighbour_block"]
