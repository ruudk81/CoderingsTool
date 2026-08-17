"""Tests for relocating an attribute without merging it.

Every phase before cross-domain is scope-bound, and the two that could move an
attribute between facets both run before the attribute layer has its final
shape: `facet_settle` works on raw pools, and refinement lost its move exit on
2026-08-16. So an attribute that ends up under a facet whose question it does
not answer has nowhere left to go — unless it happens to be a duplicate of
something in another scope, because relocation was only reachable as a
side-effect of a merge.

Shown on 2026-08-16: `Betrouwbaarheid en veiligheid` (66 ideas) and
`Deskundigheid` sat under the facet `Schaal en ontwikkeling` and stayed there.

The dispatch is stubbed: what is under test is the rebuild, not the call.
"""
import asyncio
from types import SimpleNamespace

from pipeline.step_3_ideaExtractor.dimension_data import (
    get_dimensions_in_decision_order,
)
from pipeline.step_4_classifier.classifier import (
    Placement, PromptContext, TaxonomyClassifier,
)
from pipeline.step_4_classifier.config_classifier import CategoriesConfig

DIM = get_dimensions_in_decision_order()[0]


def _ctx(*domains):
    return PromptContext(
        language="Dutch", survey_question="?",
        dimension=DIM, dimension_name=DIM.key,
        dimension_description=DIM.dimension_description,
        domains={name: {"label": name, "definition": "d", "boundary_test": "",
                        "exclusions": [], "observations": []}
                 for name in domains},
        drain_labels=set(),
    )


def _card(facet_name, question, *attribute_names):
    return {"facet_name": facet_name, "facet_definition": "d",
            "facet_question": question,
            "attributes": [{"attribute_name": name, "attribute_definition": "d"}
                           for name in attribute_names]}


def _item(name, sources, home):
    return SimpleNamespace(name=name, definition="d",
                           source_ids=list(sources), home_id=home)


# Two domains, one attribute each. A1 is the misfit: it sits under a facet
# asking about size, and answers a question about trustworthiness.
STRUCTURE = {
    "organisatie": [_card("Schaal en ontwikkeling",
                          "Hoe groot en hoe ontwikkeld is de organisatie?",
                          "Betrouwbaarheid en veiligheid")],
    "vertrouwen": [_card("Betrouwbaarheid",
                         "Hoe betrouwbaar wordt de organisatie gevonden?",
                         "Deskundigheid")],
}
MISFIT = Placement("organisatie", "Schaal en ontwikkeling",
                   "Betrouwbaarheid en veiligheid")
NEIGHBOUR = Placement("vertrouwen", "Betrouwbaarheid", "Deskundigheid")
ASSIGNMENTS = {"i1": MISFIT, "i2": MISFIT, "i3": MISFIT, "i4": NEIGHBOUR}


def _run(items):
    """Run the phase against a canned answer; return structure, ideas, log."""
    clf = TaxonomyClassifier(CategoriesConfig())
    seen = {}

    async def dispatch(phase, tasks, prepare, parse, fallback, verbose):
        seen["inventory"] = tasks[0]["inventory_block"]
        return [SimpleNamespace(items=list(items))]

    clf._dispatch = dispatch
    structure, assignments = asyncio.run(clf._run_cross_domain(
        _ctx(*STRUCTURE), {d: list(f) for d, f in STRUCTURE.items()},
        dict(ASSIGNMENTS), verbose=False))
    return structure, assignments, clf._action_log, seen["inventory"]


def _names(structure, domain, facet):
    cards = {c["facet_name"]: c for c in structure.get(domain, [])}
    return [a["attribute_name"]
            for a in (cards.get(facet, {}).get("attributes") or [])]


# =============================================================================
# A group of one may name a different home
# =============================================================================

def test_a_lone_attribute_moves_to_the_facet_it_answers():
    """The whole point: no merge, no duplicate, just a better home."""
    structure, _, _, _ = _run([
        _item("Betrouwbaarheid en veiligheid", ["A1"], "A2"),
        _item("Deskundigheid", ["A2"], "A2")])

    assert sorted(_names(structure, "vertrouwen", "Betrouwbaarheid")) == [
        "Betrouwbaarheid en veiligheid", "Deskundigheid"]
    assert "organisatie" not in structure


def test_the_ideas_follow_the_attribute_that_moved():
    """A relocation that leaves the ideas behind would point them at a facet
    that no longer holds their attribute."""
    _, assignments, _, _ = _run([
        _item("Betrouwbaarheid en veiligheid", ["A1"], "A2"),
        _item("Deskundigheid", ["A2"], "A2")])

    moved = Placement("vertrouwen", "Betrouwbaarheid",
                      "Betrouwbaarheid en veiligheid")
    assert [assignments[i] for i in ("i1", "i2", "i3")] == [moved] * 3
    assert assignments["i4"] == NEIGHBOUR


def test_a_relocation_is_logged():
    """`cross_domain_merge` only fires on more than one source, so without its
    own line a relocation would happen invisibly — and churn is the risk this
    exit carries."""
    _, _, log, _ = _run([
        _item("Betrouwbaarheid en veiligheid", ["A1"], "A2"),
        _item("Deskundigheid", ["A2"], "A2")])

    row = next(e for e in log
               if e["action"] == "attribute_relocated_cross_domain")
    assert row["attribute"] == "Betrouwbaarheid en veiligheid"
    assert row["from"] == "organisatie › Schaal en ontwikkeling"
    assert row["to"] == "vertrouwen › Betrouwbaarheid"


def test_an_attribute_that_stays_put_is_not_logged_as_relocated():
    """A group of one naming itself is the normal case for most attributes."""
    structure, assignments, log, _ = _run([
        _item("Betrouwbaarheid en veiligheid", ["A1"], "A1"),
        _item("Deskundigheid", ["A2"], "A2")])

    assert _names(structure, "organisatie", "Schaal en ontwikkeling") == [
        "Betrouwbaarheid en veiligheid"]
    assert assignments == ASSIGNMENTS
    assert not [e for e in log
                if e["action"] == "attribute_relocated_cross_domain"]


# =============================================================================
# The judgement needs something to judge against
# =============================================================================

def test_the_inventory_shows_the_question_each_facet_answers():
    """Without it the model sees only facet NAMES, and 'does this attribute
    answer its facet's question' has nothing to test against."""
    _, _, _, inventory = _run([
        _item("Betrouwbaarheid en veiligheid", ["A1"], "A1"),
        _item("Deskundigheid", ["A2"], "A2")])

    assert "Hoe betrouwbaar wordt de organisatie gevonden?" in inventory
    assert "Hoe groot en hoe ontwikkeld is de organisatie?" in inventory
