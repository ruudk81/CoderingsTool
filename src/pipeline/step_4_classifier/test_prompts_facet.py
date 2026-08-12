"""Tests voor de facetprompts (step 4)."""
import pytest
from pydantic import ValidationError
from pipeline.step_3_ideaExtractor.dimension_data import get_dimensions_in_decision_order
from pipeline.step_4_classifier.prompts_facet import (
    DiscoveredFacet,
    FacetDiscoveryResult,
    build_facet_discovery_prompt,
)

DIM = get_dimensions_in_decision_order()[0]

CTX = dict(
    language="Dutch", survey_question="Waar denkt u aan?",
    sector="finance", entity="asn_bank", topic="brand_association",
    perspective="consumer", intent="associate",
    dimension=DIM, dimension_name=DIM.key,
    dimension_description=DIM.dimension_description,
)

DOMAIN = dict(
    domain_label="Duurzaamheid",
    domain_definition="Antwoorden over het milieubeleid van de entiteit.",
    domain_boundary_test="Gaat dit antwoord over milieubeleid?",
    domain_exclusions=["prijs en kosten"],
)


def facet(name="Groen imago", **kw):
    base = dict(
        facet_name=name,
        facet_definition="Antwoorden over de groene uitstraling.",
        boundary_test="Gaat dit over uitstraling en niet over beleid?",
        exclusions=["concreet milieubeleid"],
        example_observations=["groen"],
    )
    base.update(kw)
    return DiscoveredFacet(**base)


def _prompt():
    return build_facet_discovery_prompt(
        **CTX, **DOMAIN, observations=["groen", "duurzaam", "windmolens"]
    )


def test_prompt_bevat_de_facetdiagnostiek():
    assert DIM.prompt_rules.facet_diagnostic in _prompt()


def test_prompt_bevat_de_volledige_taxonomie():
    prompt = _prompt()
    for marker in ("L1", "L2", "L3", "L4"):
        assert marker in prompt


def test_prompt_bevat_domeincontext_en_grens():
    prompt = _prompt()
    assert "Duurzaamheid" in prompt
    assert "Gaat dit antwoord over milieubeleid?" in prompt
    assert "prijs en kosten" in prompt


def test_prompt_nummert_de_observaties():
    prompt = _prompt()
    assert "1. groen" in prompt
    assert "3. windmolens" in prompt


def test_prompt_eindigt_op_de_instructor_hint():
    assert _prompt().rstrip().endswith(
        "provide your output as valid JSON following the response schema provided"
    )


def test_prompt_bouwt_geen_assenapparaat():
    """Het woord 'axis' mag voorkomen — sommige dimensiedefinities in
    dimension_data.py gebruiken het ("one consistent target axis"). Wat weg moet
    is step 4's eigen assenconstructie: een assenblok, een as-kop boven het menu,
    of een as als veld. Die toetsen we hier."""
    prompt = _prompt()
    for constructie in ("<axis", "Axis:", "axis_name", "axis system", "axis_system"):
        assert constructie not in prompt


def test_model_eist_boundary_test_en_exclusions():
    with pytest.raises(ValidationError):
        DiscoveredFacet(
            facet_name="X", facet_definition="d", example_observations=["e"]
        )


def test_discovery_result_accepteert_lege_facetlijst():
    assert FacetDiscoveryResult(scratchpad="niets gevonden", facets=[]).facets == []
