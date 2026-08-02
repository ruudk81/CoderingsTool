"""
test_p9_position.py — P9 must keep the attribute layer inside the axis discipline.

The defect this pins down (measured 2026-08-01): P9 rebuilt the attribute layer
last and dropped every position tag on its merge/split/widen products — 18 of 144
survived — so the axis discipline that works at facet level died one level down.

Position is provenance, not a judgement: it is carried from the sources in code,
exactly like `parent_facet`, and never asked of the model. A merge whose sources
sit on two different positions of the refinement axis would launder an attribute
across that axis, so it is rejected before any construction (Guard 8).

Run:  cd src && python pipeline/step_4_classifier/test_p9_position.py
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(os.path.dirname(_HERE))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from pipeline.step_4_classifier.classifier import TaxonomyClassifier
from pipeline.step_4_classifier.config_classifier import CategoriesConfig
from pipeline.step_4_classifier.prompts_classifier import (
    DiscoveredAttribute, InFacetAttribute, InFacetConsolidatedResponse,
)

DOM, FAC = "samenstelling", "zoutgehalte"


def attr(name, position="", residual=False):
    return DiscoveredAttribute(
        attribute_name=name, attribute_description=f"desc {name}",
        parent_facet=FAC, example_observations=["obs"],
        position=position, is_residual_attr=residual,
    )


def out(name, sources, action="merge"):
    return InFacetAttribute(
        action=action, attribute_name=name, attribute_description=f"new {name}",
        example_observations=["obs"], source_attributes=sources,
    )


class Idea:
    """Minimal stand-in for an assigned idea (only these two fields are read)."""

    def __init__(self, idea_id, instance):
        self.idea_id = idea_id
        self.instance = instance


def run(attributes, outputs):
    """Drive _apply_p9_results on one facet; return (structure, log)."""
    clf = TaxonomyClassifier(CategoriesConfig())
    ideas = [Idea(f"i{n}", a.attribute_name) for n, a in enumerate(attributes)]
    tasks = [{"domain_name": DOM, "facet_name": FAC, "attributes": attributes,
              "facet_ideas": ideas}]
    results = [InFacetConsolidatedResponse(scratchpad="s", attributes=outputs, misfits=[])]
    dfa = {DOM: {FAC: list(attributes)}}
    pa = {DOM: {FAC: list(attributes)}}
    assignments = {f"i{n}": a.attribute_name for n, a in enumerate(attributes)}
    part_assignments = {DOM: dict(assignments)}
    clf._apply_p9_results(
        tasks=tasks, results=results, domain_facet_attributes=dfa,
        partition_attributes=pa, attribute_assignments=assignments,
        partition_assignments=part_assignments, verbose=False,
    )
    return dfa[DOM][FAC], []


def _names(structure):
    return sorted(a.attribute_name for a in structure)


def _pos(structure, name):
    return next(a.position for a in structure if a.attribute_name == name)


# =============================================================================

def test_keep_carries_its_position():
    """An untouched attribute keeps the position P6 gave it."""
    a = [attr("minder zout", position="reductie")]
    o = [out("minder zout", ["minder zout"], action="keep")]
    struct, _ = run(a, o)
    assert _pos(struct, "minder zout") == "reductie", _pos(struct, "minder zout")


def test_merge_within_one_position_carries_it():
    """Two sources on the same position merge and the position survives."""
    a = [attr("minder zout", position="reductie"),
         attr("zout beperken", position="reductie")]
    o = [out("zout verlagen", ["minder zout", "zout beperken"])]
    struct, _ = run(a, o)
    assert _names(struct) == ["zout verlagen"], _names(struct)
    assert _pos(struct, "zout verlagen") == "reductie"


def test_cross_position_merge_is_rejected():
    """Sources on two positions must not be merged — the facet stays as P8 left it."""
    a = [attr("minder zout", position="reductie"),
         attr("zout vervangen", position="substitutie")]
    o = [out("zoutaanpak", ["minder zout", "zout vervangen"])]
    struct, _ = run(a, o)
    assert _names(struct) == ["minder zout", "zout vervangen"], _names(struct)
    assert _pos(struct, "minder zout") == "reductie"
    assert _pos(struct, "zout vervangen") == "substitutie"


def test_split_children_inherit_the_parent_position():
    """A split stays inside its position; both children carry it."""
    a = [attr("zout", position="reductie")]
    o = [InFacetAttribute(action="split", attribute_name="minder zout",
                          attribute_description="d", example_observations=["o"],
                          source_attributes=["zout"], instance_texts=["minder zout"]),
         InFacetAttribute(action="split", attribute_name="geen zout",
                          attribute_description="d", example_observations=["o"],
                          source_attributes=["zout"], instance_texts=["geen zout"])]
    struct, _ = run(a, o)
    assert _names(struct) == ["geen zout", "minder zout"], _names(struct)
    assert all(a.position == "reductie" for a in struct), [a.position for a in struct]


def test_residual_status_follows_the_position():
    """is_residual_attr is re-derived from the sources, not remembered."""
    a = [attr("rest", position="unspecified", residual=True),
         attr("rest2", position="unspecified", residual=True)]
    o = [out("onbepaald", ["rest", "rest2"])]
    struct, _ = run(a, o)
    assert struct[0].is_residual_attr is True
    assert struct[0].position == "unspecified"


def test_untagged_path_is_unaffected():
    """Facets without positions merge exactly as before and never trip the guard."""
    a = [attr("a"), attr("b")]
    o = [out("c", ["a", "b"])]
    struct, _ = run(a, o)
    assert _names(struct) == ["c"], _names(struct)
    assert _pos(struct, "c") == ""
    assert struct[0].is_residual_attr is False


# =============================================================================

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {t.__name__}: {exc}")
        except Exception as exc:
            failed += 1
            print(f"ERROR {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
