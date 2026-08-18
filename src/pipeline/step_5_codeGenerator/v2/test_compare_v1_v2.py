"""Tests voor het vergelijkingsoverzicht van beide codeboeken."""
from pipeline.step_5_codeGenerator.prompts_codeGenerator import ConsolidatedCode
from pipeline.step_5_codeGenerator.v2.compare_v1_v2 import (
    code_sizes, find_overig_code_name, format_comparison, summarise,
)


def code(name, valence, sources):
    return ConsolidatedCode(code_name=name, definition="d", diagnostic_test="t",
                            valence=valence, typical_indicators=["i"],
                            source_attributes=list(sources))


def partition(attrs, assignments, valences):
    """Minimale `partition_results`-vorm die `codebook_verifier` verstaat:
    attrs = [(attribute_id, attribute_name), ...]
    assignments = {idea_id: attribute_name}
    valences = {idea_id: '+' | '-' | None}"""
    return {
        "domain": {
            "attributes": {"facet": [
                {"attribute_id": aid, "attribute_name": name} for aid, name in attrs
            ]},
            "attribute_assignments": assignments,
            "attribute_valence": valences,
        }
    }


def test_summary_counts_codes_by_valence():
    codes = [code("A", "positive", ["a1"]), code("B", "negative", ["a2"]),
             code("C", "neutral", ["a3"])]

    summary = summarise(codes, {})

    assert summary["n_codes"] == 3
    assert summary["n_positive"] == 1
    assert summary["n_negative"] == 1
    assert summary["n_neutral"] == 1


def test_summary_counts_codes_that_are_a_single_attribute():
    """Het symptoom waar v1 op faalde: 34 van de 42 codes waren één attribuut."""
    codes = [code("A", "positive", ["a1"]), code("B", "positive", ["a2", "a3"])]

    assert summarise(codes, {})["n_solo"] == 1


def test_comparison_shows_both_columns():
    v1 = summarise([code("A", "positive", ["a1"])], {})
    v2 = summarise([code("B", "neutral", ["a1", "a2"])], {})

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


def test_find_overig_code_name_is_language_agnostic():
    """Zelfde opzoeking als step 6 (`code_assignment.py:557`): de vergelijking
    kent de taal van de run niet, dus moet elke `MISCELLANEOUS_CODE_LABELS`-
    waarde herkennen, ongeacht hoofdlettergebruik."""
    assert find_overig_code_name([code("Other", "neutral", [])]) == "Other"
    assert find_overig_code_name([code("overig", "neutral", [])]) == "overig"
    assert find_overig_code_name([code("Hoofdcode", "positive", ["a1"])]) is None


def test_code_sizes_use_the_matching_valence_pole_of_the_source_attributes():
    """Dit is de grootteverdeling waar v1 op faalde: een staart van kleine
    codes naast een kop die bijna alle respondenten draagt."""
    attrs = [("A1", "a1"), ("A2", "a2")]
    assignments = {f"i{n}": "a1" for n in range(50)} | {f"j{n}": "a2" for n in range(12)}
    valences = {k: "+" for k in assignments}
    results = partition(attrs, assignments, valences)

    codes = [code("Groot", "positive", ["a1"]), code("Klein", "positive", ["a2"])]

    sizes = code_sizes(codes, results)

    assert sizes == {"Groot": 50, "Klein": 12}


def test_size_distribution_shows_min_median_max_across_codes():
    attrs = [("A1", "a1"), ("A2", "a2")]
    assignments = {f"i{n}": "a1" for n in range(50)} | {f"j{n}": "a2" for n in range(12)}
    valences = {k: "+" for k in assignments}
    results = partition(attrs, assignments, valences)

    codes = [code("Groot", "positive", ["a1"]), code("Klein", "positive", ["a2"])]
    summary = summarise(codes, results)

    assert summary["min_code_size"] == 12
    assert summary["max_code_size"] == 50


def test_overig_share_is_read_from_the_scorecard():
    """Overig-aandeel was niet af te lezen uit `raw_codes` alleen — het komt
    uit `build_scorecard`, dat op dezelfde cacherij se `partition_results`
    draait."""
    attrs = [("A1", "a1"), ("A2", "a2")]
    assignments = {f"i{n}": "a1" for n in range(8)} | {f"j{n}": "a2" for n in range(2)}
    valences = {k: "+" for k in assignments}
    results = partition(attrs, assignments, valences)

    codes = [code("Hoofdcode", "positive", ["a1"]), code("Overig", "neutral", ["a2"])]
    summary = summarise(codes, results)

    assert summary["overig_share_pct"] == 20.0
