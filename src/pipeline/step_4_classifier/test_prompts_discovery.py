"""Tests voor de gecombineerde discovery-prompt (step 4)."""
from pipeline.step_3_ideaExtractor.dimension_data import get_dimensions_in_decision_order
from pipeline.step_4_classifier.prompts_discovery import (
    DiscoveredAttribute,
    DiscoveredFacet,
    DiscoveryResult,
    build_discovery_prompt,
)
from pipeline.step_4_classifier.prompts_shared import INSTRUCTOR_HINT

DIM = get_dimensions_in_decision_order()[0]


def _kwargs(**overrides):
    base = dict(
        language="Dutch",
        survey_question="Waar denkt u aan?",
        sector="finance", entity="asn_bank", topic="brand_association",
        perspective="consumer", intent="associate",
        dimension=DIM,
        dimension_name=DIM.key,
        dimension_description=DIM.dimension_description,
        domain_label="dienstverlening",
        domain_definition="Alles wat de organisatie aanbiedt en levert.",
        domain_boundary_test="Gaat dit over wat er geleverd wordt?",
        domain_exclusions=["prijs en kosten", "communicatie"],
        observations=["snelle afhandeling", "lange wachttijd", "vriendelijk personeel"],
    )
    base.update(overrides)
    return base


# =============================================================================
# HET RESPONSEMODEL
# =============================================================================

def test_resultaat_is_scratchpad_plus_facetten():
    assert set(DiscoveryResult.model_fields) == {"scratchpad", "facets"}


def test_facet_draagt_zijn_attributen():
    assert set(DiscoveredFacet.model_fields) == {
        "facet_name", "facet_definition", "attributes"}


def test_attribuut_heet_definition_niet_description():
    """Step 5 leest `attribute_definition` uit de taxonomiecache; `*_description`
    was de oude naam en een hernoeming brak P8 stil."""
    assert set(DiscoveredAttribute.model_fields) == {
        "attribute_name", "attribute_definition", "example_observations"}


def test_geneste_structuur_valideert():
    result = DiscoveryResult(
        scratchpad="…",
        facets=[DiscoveredFacet(
            facet_name="Snelheid",
            facet_definition="Hoe snel er geleverd wordt.",
            attributes=[DiscoveredAttribute(
                attribute_name="Wachttijd",
                attribute_definition="De tijd tot antwoord.",
                example_observations=["lange wachttijd"],
            )],
        )],
    )
    assert result.facets[0].attributes[0].attribute_name == "Wachttijd"


# =============================================================================
# PROMPT ↔ MODEL SLUITEN AAN
# =============================================================================

def test_prompt_noemt_elk_veld_dat_het_model_kent():
    prompt = build_discovery_prompt(**_kwargs())
    for veld in ("scratchpad", "facets", "facet_name", "facet_definition",
                 "attributes", "attribute_name", "attribute_definition",
                 "example_observations"):
        assert veld in prompt, veld


def test_prompt_vraagt_geen_veld_dat_het_model_niet_heeft():
    prompt = build_discovery_prompt(**_kwargs())
    for verdwenen in ("facet_description", "attribute_description",
                      "parent_facet", "boundary_test", "exclusions\""):
        assert verdwenen not in prompt, verdwenen


def test_prompt_eindigt_op_de_instructor_zin():
    assert build_discovery_prompt(**_kwargs()).rstrip().endswith(INSTRUCTOR_HINT)


# =============================================================================
# WAT ER NIET IN MAG
# =============================================================================

def test_prompt_kent_geen_lens_en_geen_dimensie_opdracht():
    prompt = build_discovery_prompt(**_kwargs())
    assert "Lens" not in prompt
    assert "orthogonal" not in prompt.lower()
    assert "dimensions on which" not in prompt.lower()


def test_prompt_bevat_geen_drempelgetallen():
    """Een drempel als 'minstens 5% van zijn scope' is van een dataset
    afgelezen en valt onder hetzelfde verbod als een use-case-voorbeeld."""
    prompt = build_discovery_prompt(**_kwargs())
    assert "%" not in prompt


# =============================================================================
# WAT ER WEL IN MOET
# =============================================================================

def test_prompt_toont_de_observaties_genummerd():
    prompt = build_discovery_prompt(**_kwargs())
    assert "1. snelle afhandeling" in prompt
    assert "3. vriendelijk personeel" in prompt


def test_prompt_zet_de_domeingrens_neer():
    prompt = build_discovery_prompt(**_kwargs())
    assert "dienstverlening" in prompt
    assert "Gaat dit over wat er geleverd wordt?" in prompt
    assert "prijs en kosten" in prompt


def test_prompt_zonder_uitsluitingen_blijft_geldig():
    prompt = build_discovery_prompt(**_kwargs(
        domain_exclusions=[], domain_boundary_test=""))
    assert "dienstverlening" in prompt
    assert prompt.rstrip().endswith(INSTRUCTOR_HINT)


def test_prompt_vraagt_om_de_enquetetaal():
    assert build_discovery_prompt(**_kwargs()).count("Dutch") >= 2


def test_prompt_draagt_de_universele_regels():
    prompt = build_discovery_prompt(**_kwargs())
    assert "<universal_rules>" in prompt


def test_prompt_vraagt_beide_niveaus_in_een_beurt():
    prompt = build_discovery_prompt(**_kwargs())
    assert "fewest" in prompt.lower()
    assert prompt.index("facet") < prompt.rindex("attribute")
