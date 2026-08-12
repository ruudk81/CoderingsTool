"""Tests voor de attribuutprompts (step 4)."""
import pytest
from pydantic import ValidationError
from pipeline.step_3_ideaExtractor.dimension_data import get_dimensions_in_decision_order
from pipeline.step_4_classifier.prompts_attribute import (
    AttributeConsolidationResult,
    AttributeDiscoveryResult,
    ConsolidatedAttribute,
    DiscoveredAttribute,
    build_attribute_assignment_model,
    build_attribute_assignment_prompt,
    build_attribute_consolidation_prompt,
    build_attribute_discovery_prompt,
    build_attribute_menu,
)

DIM = get_dimensions_in_decision_order()[0]

CTX = dict(
    language="Dutch", survey_question="Waar denkt u aan?",
    sector="finance", entity="asn_bank", topic="brand_association",
    perspective="consumer", intent="associate",
    dimension=DIM, dimension_name=DIM.key,
    dimension_description=DIM.dimension_description,
)

PARENT = dict(
    domain_label="Duurzaamheid",
    domain_definition="Antwoorden over het milieubeleid.",
    facet_name="Groene uitstraling",
    facet_definition="Hoe groen de entiteit oogt.",
    facet_boundary_test="Gaat dit over uitstraling?",
    facet_exclusions=["concreet beleid"],
)


def attribute(name="Windenergie", **kw):
    base = dict(
        attribute_name=name,
        attribute_definition="Verwijzingen naar windopwekking.",
        boundary_test="Noemt dit windopwekking?",
        exclusions=["zonne-energie"],
        example_observations=["windmolens"],
    )
    base.update(kw)
    return DiscoveredAttribute(**base)


def _prompt():
    return build_attribute_discovery_prompt(
        **CTX, **PARENT, observations=["windmolens", "groene site"]
    )


def test_prompt_bevat_de_attribuutdiagnostiek():
    assert DIM.prompt_rules.attribute_diagnostic in _prompt()


def test_prompt_bevat_domein_en_facet_als_ouders():
    prompt = _prompt()
    assert "Duurzaamheid" in prompt
    assert "Groene uitstraling" in prompt
    assert "Gaat dit over uitstraling?" in prompt


def test_prompt_bevat_de_volledige_taxonomie():
    for marker in ("L1", "L2", "L3", "L4"):
        assert marker in _prompt()


def test_prompt_verbiedt_splitsen_op_evaluatieve_richting():
    assert "evaluative direction" in _prompt().lower()


def test_prompt_bouwt_geen_assenapparaat():
    prompt = _prompt()
    for constructie in ("<axis", "Axis:", "axis_name", "axis system", "axis_system"):
        assert constructie not in prompt


def test_prompt_eindigt_op_de_instructor_hint():
    assert _prompt().rstrip().endswith(
        "provide your output as valid JSON following the response schema provided"
    )


def test_model_eist_boundary_test_en_exclusions():
    with pytest.raises(ValidationError):
        DiscoveredAttribute(
            attribute_name="X", attribute_definition="d", example_observations=["e"]
        )


def test_discovery_result_accepteert_lege_lijst():
    assert AttributeDiscoveryResult(scratchpad="s", attributes=[]).attributes == []


# =============================================================================
# Taak 7 — consolidatie over chunks
# =============================================================================

def _prompt_c():
    return build_attribute_consolidation_prompt(
        **CTX,
        domain_label=PARENT["domain_label"],
        domain_definition=PARENT["domain_definition"],
        facet_name=PARENT["facet_name"],
        facet_definition=PARENT["facet_definition"],
        candidates=[
            attribute("Windenergie", example_observations=["windmolens"]),
            attribute("Windmolens", example_observations=["molens in zee"]),
        ],
    )


def test_consolidatie_toont_kandidaten_met_voorbeelden():
    prompt = _prompt_c()
    assert "Windenergie" in prompt
    assert "molens in zee" in prompt


def test_consolidatie_bevat_de_attribuutdiagnostiek():
    assert DIM.prompt_rules.attribute_diagnostic in _prompt_c()


def test_consolidatie_bevat_de_mergetoets():
    prompt = _prompt_c().lower()
    assert "merge" in prompt
    assert "boundary" in prompt


def test_consolidatie_eindigt_op_de_instructor_hint():
    assert _prompt_c().rstrip().endswith(
        "provide your output as valid JSON following the response schema provided"
    )


def test_geconsolideerd_attribuut_noemt_zijn_bronnen():
    attr = ConsolidatedAttribute(
        attribute_name="Windenergie", attribute_definition="Windopwekking.",
        boundary_test="b?", exclusions=["zon"], example_observations=["windmolens"],
        source_attributes=["Windenergie", "Windmolens"],
    )
    assert len(attr.source_attributes) == 2


def test_consolidatieresultaat_accepteert_lege_lijst():
    assert AttributeConsolidationResult(scratchpad="s", attributes=[]).attributes == []


# =============================================================================
# Taak 8 — toewijzing
# =============================================================================

ASSIGN_CTX = dict(
    language="Dutch", survey_question="Waar denkt u aan?",
    sector="finance", entity="asn_bank", topic="brand_association",
    perspective="consumer", intent="associate",
    facet_name="Groene uitstraling",
    facet_definition="Hoe groen de entiteit oogt.",
)


def _attributen():
    return [ConsolidatedAttribute(
        attribute_name="Windenergie", attribute_definition="Windopwekking.",
        boundary_test="Noemt dit wind?", exclusions=["zonne-energie"],
        example_observations=["windmolens"], source_attributes=["Windenergie"],
    )]


def test_menu_nummert_en_toont_grens_en_uitsluiting():
    menu = build_attribute_menu(_attributen())
    assert "[A1]" in menu
    assert "Noemt dit wind?" in menu
    assert "zonne-energie" in menu


def test_toewijzingsprompt_bevat_menu_facet_en_ideeen():
    prompt = build_attribute_assignment_prompt(
        **ASSIGN_CTX, attributes=_attributen(), ideas=[("i1", "windmolens op zee")]
    )
    assert "[A1]" in prompt
    assert "Groene uitstraling" in prompt
    assert "[i1] windmolens op zee" in prompt
    assert "A_NONE" in prompt


def test_model_weigert_onbekend_attribuut_id():
    Model = build_attribute_assignment_model(["A1"], ["i1"])
    with pytest.raises(ValidationError):
        Model(assignments=[{"idea_id": "i1", "assigned_attribute_id": "A9",
                            "confidence": 0.9, "valence": "0"}])


def test_model_weigert_onbekend_idea_id():
    Model = build_attribute_assignment_model(["A1"], ["i1"])
    with pytest.raises(ValidationError):
        Model(assignments=[{"idea_id": "zz", "assigned_attribute_id": "A1",
                            "confidence": 0.9, "valence": "0"}])


def test_model_weigert_valence_buiten_de_drie_waarden():
    Model = build_attribute_assignment_model(["A1"], ["i1"])
    with pytest.raises(ValidationError):
        Model(assignments=[{"idea_id": "i1", "assigned_attribute_id": "A1",
                            "confidence": 0.9, "valence": "positief"}])


def test_model_accepteert_a_none():
    Model = build_attribute_assignment_model(["A1"], ["i1"])
    result = Model(assignments=[{"idea_id": "i1", "assigned_attribute_id": "A_NONE",
                                 "confidence": 0.1, "valence": "0"}])
    assert result.assignments[0].assigned_attribute_id == "A_NONE"
