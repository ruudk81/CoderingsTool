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
