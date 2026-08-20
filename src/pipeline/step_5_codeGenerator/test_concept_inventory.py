"""Tests for the concept inventory (step 1 of step 5)."""
from pipeline.step_5_codeGenerator.concept_inventory import build_inventory, t_keep
from pipeline.step_5_codeGenerator.taxonomy_input import AttributeRef, IdeaUnit


def unit(idea_id, respondent_id, attribute_id, valence):
    return IdeaUnit(idea_id=idea_id, respondent_id=respondent_id,
                    attribute_id=attribute_id, valence=valence,
                    instance="i", interpretation="t")


REFS = {
    "A1": AttributeRef("A1", "Prijs", "d", "Domein", "Facet"),
    "A2": AttributeRef("A2", "Service", "d", "Domein", "Facet"),
}


def test_one_respondent_with_three_ideas_counts_once():
    units = [unit(f"R1_{i}", "R1", "A1", "-") for i in range(3)]
    concept = build_inventory(units, REFS)[0]
    assert concept.n_iu == 3
    assert concept.n_resp == 1


def test_respondent_with_both_poles_counts_in_both():
    units = [unit("R1_1", "R1", "A1", "+"), unit("R1_2", "R1", "A1", "-")]
    concept = build_inventory(units, REFS)[0]
    assert concept.n_resp == 1
    assert concept.n_resp_pos == 1
    assert concept.n_resp_neg == 1


def test_empty_valence_counts_as_neutral():
    units = [unit("R1_1", "R1", "A1", ""), unit("R2_1", "R2", "A1", "0")]
    concept = build_inventory(units, REFS)[0]
    assert concept.n_resp_neu == 2


def test_attribute_without_ideas_is_absent():
    units = [unit("R1_1", "R1", "A1", "+")]
    ids = {c.attribute_id for c in build_inventory(units, REFS)}
    assert ids == {"A1"}


def test_unknown_attribute_id_is_skipped():
    units = [unit("R1_1", "R1", "A99", "+")]
    assert build_inventory(units, REFS) == []


def test_t_keep_uses_share_above_the_floor():
    class Cfg:
        t_keep_share = 0.01
        t_keep_min_respondents = 3
    assert t_keep(2000, Cfg) == 20


def test_t_keep_uses_the_floor_for_small_samples():
    class Cfg:
        t_keep_share = 0.01
        t_keep_min_respondents = 3
    assert t_keep(80, Cfg) == 3


def test_concept_erft_de_vangnetmarkering_van_zijn_ref():
    """Een vangnet houdt zijn Concept — de respondenten erop moeten in de
    boekhouding blijven — maar draagt de markering mee, zodat de kaartenfase
    hem kan overslaan."""
    from pipeline.step_5_codeGenerator.taxonomy_input import AttributeRef, IdeaUnit
    refs = {
        "A1": AttributeRef(attribute_id="A1", name="Prijs", definition="d",
                           domain="D", facet="F"),
        "A9": AttributeRef(attribute_id="A9", name="Overig — F", definition="d",
                           domain="D", facet="F", is_drain=True),
    }
    units = [IdeaUnit(idea_id="i1", respondent_id="r1", attribute_id="A1",
                      valence="+", instance="x", interpretation="y"),
             IdeaUnit(idea_id="i2", respondent_id="r2", attribute_id="A9",
                      valence="0", instance="x", interpretation="y")]

    by_id = {c.attribute_id: c for c in build_inventory(units, refs)}

    assert by_id["A9"].is_drain is True
    assert by_id["A1"].is_drain is False
    assert by_id["A9"].resp_ids == frozenset({"r2"})
