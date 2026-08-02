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


def run(attributes, outputs, ideas=None):
    """Drive _apply_p9_results on one facet; return (structure, assignments).

    `ideas` is an optional list of (instance_text, attribute) pairs. The default
    puts one idea on each attribute, its text equal to the attribute name.
    """
    clf = TaxonomyClassifier(CategoriesConfig())
    if ideas is None:
        ideas = [(a.attribute_name, a.attribute_name) for a in attributes]
    objs = [Idea(f"i{n}", text) for n, (text, _) in enumerate(ideas)]
    assignments = {f"i{n}": att for n, (_, att) in enumerate(ideas)}
    tasks = [{"domain_name": DOM, "facet_name": FAC, "attributes": attributes,
              "facet_ideas": objs}]
    results = [InFacetConsolidatedResponse(scratchpad="s", attributes=outputs, misfits=[])]
    dfa = {DOM: {FAC: list(attributes)}}
    pa = {DOM: {FAC: list(attributes)}}
    # partition_assignments maps idea -> FACET (not attribute); _apply_p9_results
    # needs it to place an idea before it can remap or restore anything.
    part_assignments = {DOM: {o.idea_id: FAC for o in objs}}
    clf._apply_p9_results(
        tasks=tasks, results=results, domain_facet_attributes=dfa,
        partition_attributes=pa, attribute_assignments=assignments,
        partition_assignments=part_assignments, verbose=False,
    )
    return dfa[DOM][FAC], assignments


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
    """Sources on two positions must not be merged — both survive under their own name."""
    a = [attr("minder zout", position="reductie"),
         attr("zout vervangen", position="substitutie")]
    o = [out("zoutaanpak", ["minder zout", "zout vervangen"])]
    struct, _ = run(a, o)
    assert _names(struct) == ["minder zout", "zout vervangen"], _names(struct)
    assert _pos(struct, "minder zout") == "reductie"
    assert _pos(struct, "zout vervangen") == "substitutie"


def test_rejected_merge_leaves_no_orphaned_assignment():
    """The regression this pins: rejecting must never strand an idea on a name that
    is gone. A first attempt skipped the whole facet, which stopped the structure
    from being rebuilt — 354 of 774 assignments (46%) pointed at nothing."""
    a = [attr("minder zout", position="reductie"),
         attr("zout vervangen", position="substitutie")]
    o = [out("zoutaanpak", ["minder zout", "zout vervangen"])]
    struct, assignments = run(a, o)
    present = {x.attribute_name for x in struct}
    stranded = {n for n in assignments.values() if n and n not in present}
    assert not stranded, f"toewijzingen zonder attribuut: {stranded}"


def test_rejection_does_not_block_the_rest_of_the_facet():
    """A rejected merge is local: other consolidations in the same facet still apply."""
    a = [attr("minder zout", position="reductie"),
         attr("zout vervangen", position="substitutie"),
         attr("zout weglaten", position="reductie")]
    o = [out("zoutaanpak", ["minder zout", "zout vervangen"]),          # gekruist -> af
         out("zout eruit", ["zout weglaten"], action="keep")]           # blijft staan
    struct, assignments = run(a, o)
    assert "zout eruit" in _names(struct), _names(struct)
    assert "minder zout" in _names(struct) and "zout vervangen" in _names(struct)
    present = {x.attribute_name for x in struct}
    assert not {n for n in assignments.values() if n and n not in present}


def test_split_children_inherit_the_parent_position():
    """A split stays inside its position; both children carry it."""
    a = [attr("zout", position="reductie")]
    o = [InFacetAttribute(action="split", attribute_name="minder zout",
                          attribute_description="d", example_observations=["o"],
                          source_attributes=["zout"], instance_texts=["minder zout"]),
         InFacetAttribute(action="split", attribute_name="geen zout",
                          attribute_description="d", example_observations=["o"],
                          source_attributes=["zout"], instance_texts=["geen zout"])]
    # Both ideas must carry a text one of the children claims, or they stay on the
    # parent and the self-check rightly puts that node back.
    struct, _ = run(a, o, ideas=[("minder zout", "zout"), ("geen zout", "zout")])
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
