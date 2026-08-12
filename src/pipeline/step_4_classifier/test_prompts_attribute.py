"""Tests voor de attribuutprompts (step 4)."""
import pytest
from pydantic import ValidationError
from pipeline.step_3_ideaExtractor.dimension_data import get_dimensions_in_decision_order
from pipeline.step_4_classifier.prompts_attribute import (
    AttributeDiscoveryResult,
    DiscoveredAttribute,
    build_attribute_discovery_prompt,
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
