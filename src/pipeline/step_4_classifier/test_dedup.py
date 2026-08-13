"""Tests for exact-dedup of chunk discovery output (step 4)."""
from pipeline.step_4_classifier.dedup import dedup_exact_attributes, dedup_exact_facets
from pipeline.step_4_classifier.prompts_attribute import DiscoveredAttribute
from pipeline.step_4_classifier.prompts_facet import DiscoveredFacet


def make_facet(name, examples=None):
    return DiscoveredFacet(
        facet_name=name,
        facet_definition="d",
        boundary_test="b?",
        exclusions=["x"],
        example_observations=examples if examples is not None else ["e1"],
    )


def make_attribute(name, examples=None):
    return DiscoveredAttribute(
        attribute_name=name,
        attribute_definition="d",
        boundary_test="b?",
        exclusions=["x"],
        example_observations=examples if examples is not None else ["e1"],
    )


def test_merges_same_normalized_name():
    result = dedup_exact_facets([
        make_facet("Degelijk en betrouwbaar"),
        make_facet("  degelijk en betrouwbaar "),
    ])
    assert len(result) == 1


def test_does_not_merge_near_duplicates():
    result = dedup_exact_facets([
        make_facet("Warm en huiselijk"),
        make_facet("Warm en menselijk"),
    ])
    assert len(result) == 2


def test_unions_examples_preserving_order():
    result = dedup_exact_facets([
        make_facet("X", examples=["a", "b"]),
        make_facet("X", examples=["b", "c"]),
    ])
    assert result[0].example_observations == ["a", "b", "c"]


def test_first_seen_order_and_input_untouched():
    first = make_facet("B")
    inputs = [first, make_facet("A"), make_facet("B", examples=["extra"])]
    result = dedup_exact_facets(inputs)
    assert [f.facet_name for f in result] == ["B", "A"]
    assert first.example_observations == ["e1"]  # input niet gemuteerd


def test_attribute_dedup_merges_and_unions_examples():
    result = dedup_exact_attributes([
        make_attribute("Groen imago", examples=["a"]),
        make_attribute("groen imago", examples=["a", "b"]),
        make_attribute("Duurzaam beleid"),
    ])
    assert len(result) == 2
    assert result[0].example_observations == ["a", "b"]
