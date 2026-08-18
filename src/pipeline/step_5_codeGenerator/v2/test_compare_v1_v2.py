"""Tests voor het vergelijkingsoverzicht van beide codeboeken."""
from pipeline.step_5_codeGenerator.prompts_codeGenerator import ConsolidatedCode
from pipeline.step_5_codeGenerator.v2.compare_v1_v2 import format_comparison, summarise


def code(name, valence, sources):
    return ConsolidatedCode(code_name=name, definition="d", diagnostic_test="t",
                            valence=valence, typical_indicators=["i"],
                            source_attributes=list(sources))


def test_summary_counts_codes_by_valence():
    codes = [code("A", "positive", ["a1"]), code("B", "negative", ["a2"]),
             code("C", "neutral", ["a3"])]

    summary = summarise(codes)

    assert summary["n_codes"] == 3
    assert summary["n_positive"] == 1
    assert summary["n_negative"] == 1
    assert summary["n_neutral"] == 1


def test_summary_counts_codes_that_are_a_single_attribute():
    """Het symptoom waar v1 op faalde: 34 van de 42 codes waren één attribuut."""
    codes = [code("A", "positive", ["a1"]), code("B", "positive", ["a2", "a3"])]

    assert summarise(codes)["n_solo"] == 1


def test_summary_counts_distinct_attributes_covered():
    codes = [code("A", "positive", ["a1", "a2"]), code("B", "positive", ["a2"])]

    assert summarise(codes)["attributes_covered"] == 2


def test_comparison_shows_both_columns():
    v1 = summarise([code("A", "positive", ["a1"])])
    v2 = summarise([code("B", "neutral", ["a1", "a2"])])

    text = format_comparison(v1, v2)

    assert "v1" in text and "v2" in text


def test_where_each_attribute_landed_shows_both_sides():
    from pipeline.step_5_codeGenerator.v2.compare_v1_v2 import where_each_attribute_landed

    v1 = [code("Los", "positive", ["a1"]), code("Ook los", "positive", ["a2"])]
    v2 = [code("Samen", "positive", ["a1", "a2"])]

    landed = where_each_attribute_landed(v1, v2)

    assert landed == [("a1", "Los", "Samen"), ("a2", "Ook los", "Samen")]


def test_attribute_only_present_in_one_side_is_marked():
    from pipeline.step_5_codeGenerator.v2.compare_v1_v2 import where_each_attribute_landed

    landed = where_each_attribute_landed([code("A", "positive", ["a1"])], [])

    assert landed == [("a1", "A", "—")]
