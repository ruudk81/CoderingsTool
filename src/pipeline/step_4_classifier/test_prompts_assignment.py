"""Tests voor de toewijzingsprompt (step 4)."""
from pipeline.step_4_classifier.drains import make_drain_attribute, make_drain_facet
from pipeline.step_4_classifier.prompts_assignment import (
    build_assignment_menu,
    build_assignment_model,
    build_assignment_prompt,
)
from pipeline.step_4_classifier.prompts_shared import INSTRUCTOR_HINT


def _attr(name, definition, example=None):
    return {"attribute_name": name, "attribute_definition": definition,
            "example_observations": [example] if example else []}


def _inventaris():
    """Twee inhoudelijke facetten met hun other, plus het domein-other."""
    return [
        {"facet_name": "Snelheid", "facet_definition": "Hoe snel er geleverd wordt.",
         "attributes": [_attr("Wachttijd", "De tijd tot antwoord.", "lange wachttijd"),
                        make_drain_attribute("Snelheid", "Dutch")]},
        {"facet_name": "Bejegening", "facet_definition": "Hoe men bejegend wordt.",
         "attributes": [_attr("Vriendelijkheid", "Of men vriendelijk is."),
                        make_drain_attribute("Bejegening", "Dutch")]},
        make_drain_facet("dienstverlening", "Dutch"),
    ]


def _kwargs(**overrides):
    menu_block, _ = build_assignment_menu(_inventaris())
    base = dict(
        language="Dutch",
        survey_question="Waar denkt u aan?",
        sector="finance", entity="asn_bank", topic="brand_association",
        perspective="consumer", intent="associate",
        domain_label="dienstverlening",
        domain_definition="Alles wat de organisatie aanbiedt en levert.",
        menu_block=menu_block,
        label="lange wachttijd bij de klantenservice",
    )
    base.update(overrides)
    return base


# =============================================================================
# HET MENU
# =============================================================================

def test_menu_groepeert_attributen_onder_hun_facet():
    block, _ = build_assignment_menu(_inventaris())
    assert block.index("Snelheid") < block.index("Wachttijd") < block.index("Bejegening")


def test_ids_lopen_door_over_facetten_heen():
    _, id_map = build_assignment_menu(_inventaris())
    assert list(id_map) == ["A1", "A2", "A3", "A4", "A5"]


def test_id_map_wijst_naar_facet_en_attribuut():
    _, id_map = build_assignment_menu(_inventaris())
    assert id_map["A1"]["facet_name"] == "Snelheid"
    assert id_map["A1"]["attribute_name"] == "Wachttijd"


def test_menu_bevat_per_facet_een_other_en_het_domein_other():
    _, id_map = build_assignment_menu(_inventaris())
    drains = [v for v in id_map.values() if v["is_drain"]]
    assert len(drains) == 3


def test_facet_zonder_attributen_valt_uit_het_menu():
    """Een facet zonder attributen is onbereikbaar — toewijzing kiest een
    attribuut, dus zo'n facet zou een dode regel in het menu zijn."""
    _, id_map = build_assignment_menu(
        [{"facet_name": "Leeg", "facet_definition": "…", "attributes": []}])
    assert id_map == {}


def test_menu_toont_definitie_en_voorbeeld():
    block, _ = build_assignment_menu(_inventaris())
    assert "De tijd tot antwoord." in block
    assert "lange wachttijd" in block


# =============================================================================
# HET RESPONSEMODEL
# =============================================================================

def test_model_dwingt_de_id_ruimte_af():
    model = build_assignment_model(["A1", "A2"])
    assert set(model.model_fields) == {
        "assigned_attribute_id", "confidence", "valence"}
    literal = str(model.model_fields["assigned_attribute_id"].annotation)
    assert "A1" in literal and "A2" in literal


def test_model_heeft_geen_none_uitgang():
    """other is een echte keuze in het menu, dus een aparte NONE-uitgang zou
    een tweede manier zijn om niets te kiezen — en die kwam als __UNASSIGNED__
    weer terug."""
    literal = str(build_assignment_model(
        ["A1", "A2"]).model_fields["assigned_attribute_id"].annotation)
    for verboden in ("NONE", "A_NONE", "UNASSIGNED"):
        assert verboden not in literal


def test_model_weigert_een_verzonnen_id():
    import pydantic
    model = build_assignment_model(["A1"])
    try:
        model(assigned_attribute_id="A9", confidence=1.0, valence="0")
    except pydantic.ValidationError:
        return
    raise AssertionError("een id buiten het menu had moeten falen")


# =============================================================================
# PROMPT ↔ MODEL SLUITEN AAN
# =============================================================================

def test_prompt_noemt_elk_veld_dat_het_model_kent():
    prompt = build_assignment_prompt(**_kwargs())
    for veld in ("assigned_attribute_id", "confidence", "valence"):
        assert veld in prompt, veld


def test_prompt_eindigt_op_de_instructor_zin():
    assert build_assignment_prompt(**_kwargs()).rstrip().endswith(INSTRUCTOR_HINT)


def test_prompt_vraagt_om_precies_een_attribuut():
    prompt = build_assignment_prompt(**_kwargs())
    assert "exactly one" in prompt.lower()


# =============================================================================
# DE OTHER-UITGANG
# =============================================================================

def test_prompt_legt_uit_wanneer_other_de_juiste_keuze_is():
    prompt = build_assignment_prompt(**_kwargs())
    assert "Overig" in prompt
    assert "last resort" in prompt.lower()


def test_prompt_zet_valence_neer_als_richting_niet_sentiment():
    prompt = build_assignment_prompt(**_kwargs())
    assert "not emotional sentiment" in prompt


def test_prompt_toont_het_label_en_de_domeingrens():
    prompt = build_assignment_prompt(**_kwargs())
    assert "lange wachttijd bij de klantenservice" in prompt
    assert "dienstverlening" in prompt


def test_prompt_bevat_geen_drempelgetallen():
    assert "%" not in build_assignment_prompt(**_kwargs())
