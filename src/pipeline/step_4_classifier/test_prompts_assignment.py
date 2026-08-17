"""Tests voor de toewijzingsprompt (step 4)."""
from pipeline.step_3_ideaExtractor.dimension_data import get_dimensions_in_decision_order
from pipeline.step_4_classifier.drains import make_drain_attribute, make_drain_facet
from pipeline.step_4_classifier.prompts_assignment import (
    build_assignment_menu,
    build_assignment_model,
    build_assignment_prompt,
    build_facet_assignment_model,
    build_facet_assignment_prompt,
    build_facet_menu,
)
from pipeline.step_4_classifier.prompts_shared import INSTRUCTOR_HINT
from pipeline.step_4_classifier.test_prompts_shared import (
    assert_every_field_is_described, assert_prompt_does_not_restate_the_schema,
)


DIM = get_dimensions_in_decision_order()[0]


def _attr(name, definition, example=None):
    return {"attribute_name": name, "attribute_definition": definition,
            "example_observations": [example] if example else []}


def _inventaris():
    """Two substantive facets with their other, plus the domain other."""
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
        dimension_name=DIM.key,
        dimension_description=DIM.dimension_description,
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

def test_menu_groups_attributes_under_their_facet():
    block, _ = build_assignment_menu(_inventaris())
    assert block.index("Snelheid") < block.index("Wachttijd") < block.index("Bejegening")


def test_ids_run_on_across_facets():
    _, id_map = build_assignment_menu(_inventaris())
    assert list(id_map) == ["A1", "A2", "A3", "A4", "A5"]


def test_the_id_map_points_at_facet_and_attribute():
    _, id_map = build_assignment_menu(_inventaris())
    assert id_map["A1"]["facet_name"] == "Snelheid"
    assert id_map["A1"]["attribute_name"] == "Wachttijd"


def test_menu_holds_an_other_per_facet_and_the_domain_other():
    _, id_map = build_assignment_menu(_inventaris())
    drains = [v for v in id_map.values() if v["is_drain"]]
    assert len(drains) == 3


def test_a_facet_without_attributes_drops_out_of_the_menu():
    """A facet without attributes is unreachable — assignment picks an
    attribute, so such a facet would be a dead line in the menu."""
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
    """other is a real choice in the menu, so a separate NONE exit would be a
    second way of choosing nothing — and that one kept coming back as
    __UNASSIGNED__."""
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

def test_the_model_describes_every_field_it_has():
    assert_every_field_is_described(build_assignment_model(["A1", "A2"]))


def test_the_prompt_does_not_restate_the_schema():
    assert_prompt_does_not_restate_the_schema(build_assignment_prompt(**_kwargs()))


def test_prompt_eindigt_op_de_instructor_zin():
    assert build_assignment_prompt(**_kwargs()).rstrip().endswith(INSTRUCTOR_HINT)


def test_prompt_asks_for_exactly_one_attribute():
    prompt = build_assignment_prompt(**_kwargs())
    assert "exactly one" in prompt.lower()


# =============================================================================
# DE OTHER-UITGANG
# =============================================================================

def test_prompt_explains_when_the_catch_all_is_the_right_choice():
    prompt = build_assignment_prompt(**_kwargs())
    assert "marked [CATCH-ALL]" in prompt
    assert "last resort" in prompt.lower()


def test_the_menu_marks_catch_alls_by_drain_key_not_by_name():
    """The name is in the survey language; the marker comes from drain_key."""
    block, _ = build_assignment_menu(_inventaris())
    gemarkeerd = [r for r in block.splitlines() if "[CATCH-ALL]" in r]
    # two facet others, the domain other facet and its attribute
    assert len(gemarkeerd) == 4
    assert not any("Wachttijd" in r or "Vriendelijkheid" in r for r in gemarkeerd)


def test_prompt_does_not_name_the_catch_all():
    """drains.py names the catch-all in the survey language — Overig, Other, or
    whatever the next language makes of it. A prompt pointing at "Overig" points,
    on an English dataset, at an option that is not in the menu."""
    prompt = build_assignment_prompt(**_kwargs(menu_block="<menu>"))
    for naam in ("Overig", "Other"):
        assert naam not in prompt, naam


def test_the_prompt_frames_valence_as_direction_not_sentiment():
    prompt = build_assignment_prompt(**_kwargs())
    assert "not emotional sentiment" in prompt


def test_prompt_shows_the_label_and_the_domain_boundary():
    prompt = build_assignment_prompt(**_kwargs())
    assert "lange wachttijd bij de klantenservice" in prompt
    assert "dienstverlening" in prompt


def test_prompt_bevat_geen_drempelgetallen():
    assert "%" not in build_assignment_prompt(**_kwargs())


# =============================================================================
# DE FACETTOEWIJZING
# =============================================================================

FACETS = [
    {"facet_name": "Snelheid", "facet_definition": "Hoe snel er geleverd wordt.",
     "facet_question": "Hoe snel wordt er geleverd?", "attributes": []},
    {"facet_name": "Bejegening", "facet_definition": "Hoe klanten worden aangesproken.",
     "facet_question": "", "attributes": []},
    make_drain_facet("dienstverlening", "Dutch"),
]


def _fkwargs(**overrides):
    menu_block, _ = build_facet_menu(FACETS)
    base = dict(
        language="Dutch",
        survey_question="Waar denkt u aan?",
        sector="finance", entity="asn_bank", topic="brand_association",
        perspective="consumer", intent="associate",
        dimension_name=DIM.key,
        dimension_description=DIM.dimension_description,
        domain_label="dienstverlening",
        domain_definition="Alles wat de organisatie aanbiedt en levert.",
        menu_block=menu_block,
        observation="lange wachttijd",
    )
    base.update(overrides)
    return base


def test_het_facetmenu_deelt_ids_uit():
    block, id_map = build_facet_menu(FACETS)
    assert list(id_map) == ["F1", "F2", "F3"]
    assert id_map["F1"]["facet_name"] == "Snelheid"
    assert "[F1]" in block


def test_het_facetmenu_toont_de_vraag_die_het_facet_beantwoordt():
    """De facetvraag is het sterkste signaal over de identiteit van een facet,
    en bestaat precies om die toetsbaar te maken."""
    block, _ = build_facet_menu(FACETS)
    assert "Hoe snel wordt er geleverd?" in block


def test_een_facet_zonder_vraag_rendert_geen_leeg_label():
    """Een domein met één kandidaat wordt zonder call vastgezet en heeft geen
    vraag; een label met niets erachter leest als een vraag die het facet niet
    wist te stellen."""
    block, _ = build_facet_menu(FACETS[1:2])
    assert "question it answers" not in block.lower()


def test_het_facetmenu_toont_de_attribuutnamen_niet():
    """De pool is op dit moment nog ongeconsolideerd — tot tientallen
    bijna-gelijke namen per facet — en zou het menu onleesbaar maken."""
    facets = [dict(FACETS[0], attributes=[{"attribute_name": "Wachttijd",
                                           "attribute_definition": "d"}])]
    block, _ = build_facet_menu(facets)
    assert "Wachttijd" not in block


def test_het_vangnetfacet_staat_gemarkeerd_in_het_menu():
    """Op de marker, nooit op de naam: die staat in de enquêtetaal.

    De marker staat op een eigen regel, dus de toets loopt over het blok van
    het facet en niet over de regel met zijn id.
    """
    block, id_map = build_facet_menu(FACETS)
    assert id_map["F3"]["is_drain"] is True
    gemarkeerd = [b for b in block.split("\n\n") if "[CATCH-ALL]" in b]
    assert len(gemarkeerd) == 1 and gemarkeerd[0].startswith("[F3]")


def test_het_responsemodel_laat_alleen_uitgedeelde_ids_toe():
    """Een verzonnen id is zo een schemafout die instructor overdoet, in plaats
    van een inhoudsfout die verderop opduikt."""
    model = build_facet_assignment_model(["F1", "F2"])
    assert str(model.model_fields["assigned_facet_id"].annotation) == (
        "typing.Literal['F1', 'F2']")


def test_het_responsemodel_draagt_geen_valence():
    """Valence wordt beoordeeld ten opzichte van het attribuut, niet het facet;
    hem hier vragen zou een oordeel verzinnen dat pas later te geven is."""
    model = build_facet_assignment_model(["F1"])
    assert set(model.model_fields) == {"assigned_facet_id", "confidence"}


def test_de_prompt_eindigt_op_de_instructor_zin():
    assert build_facet_assignment_prompt(**_fkwargs()).rstrip().endswith(INSTRUCTOR_HINT)


def test_de_prompt_draagt_geen_universele_regels():
    """Toewijzing kiest een id uit een menu en bedenkt geen naam — dezelfde
    uitzondering als de attribuuttoewijzing."""
    assert "<universal_rules>" not in build_facet_assignment_prompt(**_fkwargs())


def test_de_prompt_draagt_geen_taxonomieblok():
    """Dezelfde uitzondering, om dezelfde reden: er wordt geen naam bedacht,
    dus het taxonomieblok zou hier alleen ruis zijn. De facetvraag draagt al
    concreet, uit deze dataset zelf, wat het blok abstract zou zeggen."""
    assert "<taxonomy_structure>" not in build_facet_assignment_prompt(**_fkwargs())
