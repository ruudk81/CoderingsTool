"""Tests voor de gecombineerde discovery-prompt (step 4)."""
from types import SimpleNamespace

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

def test_prompt_kent_geen_dimensie_opdracht():
    """De prompt vraagt naar facetten en attributen, nooit naar de dimensies of
    assen waarop antwoorden verschillen. Let op: 'orthogonaal' mag wél — dat gaat
    over de verhouding tussen facetten onderling, niet over het zoeken van assen."""
    prompt = build_discovery_prompt(**_kwargs())
    assert "Lens" not in prompt
    assert "dimensions on which" not in prompt.lower()
    assert "identify the dimensions" not in prompt.lower()


def test_prompt_bevat_geen_drempelgetallen():
    """Een drempel als 'minstens 5% van zijn scope' is van een dataset
    afgelezen en valt onder hetzelfde verbod als een use-case-voorbeeld."""
    prompt = build_discovery_prompt(**_kwargs())
    assert "%" not in prompt


def _skelet_prompt():
    """De prompt met élk dynamisch slot op een sentinel — ook de dimensie.

    Wat er dan nog aan woorden in staat is het statische instructieskelet: de
    tekst die voor iedere dataset én iedere dimensie ongewijzigd wordt verstuurd.
    De builder raakt van een dimensie alleen `prompt_rules` aan, dus een stub
    volstaat en is eerlijker dan een echte dimensie — die smokkelt haar eigen
    vocabulaire mee de meting in.
    """
    rules = SimpleNamespace(
        domain_instruction="Definition: DOMAINRULE\nKey idea: DOMAINIDEA",
        facet_instruction="Definition: FACETRULE\nKey idea: FACETIDEA",
        attribute_instruction="Definition: ATTRIBUTERULE\nKey idea: ATTRIBUTEIDEA",
    )
    return build_discovery_prompt(**_kwargs(
        language="LANGUAGE", survey_question="QUESTION",
        sector="SECTOR", entity="ENTITY", topic="TOPIC",
        perspective="PERSPECTIVE", intent="INTENT",
        dimension=SimpleNamespace(prompt_rules=rules),
        dimension_name="DIMENSION", dimension_description="DIMENSIONDESCRIPTION",
        domain_label="DOMAIN", domain_definition="DEFINITION",
        domain_boundary_test="", domain_exclusions=None,
        observations=["OBSERVATION"]))


def test_skelet_leent_geen_vocabulaire_van_dataset_of_dimensie():
    """Tripwire voor het lekpad uit CLAUDE.md: een diagnose op één dataset die
    als vuistregel in de prompt belandt en daar blijft staan.

    Vangt twee soorten lek in één meting. Onderwerpwoorden ('sustainability')
    horen bij één klant; dimensiewoorden ('association') horen bij één van de
    tien dimensies, terwijl deze prompt ze allemaal bedient.

    Geen bewijs van agnosticisme — een grendel op wat deze repo daadwerkelijk
    heeft gecodeerd. Komt er een nieuw voorbeeld in, breid de lijst dan uit in
    plaats van hem te laten verwateren.
    """
    ontleend = [
        # onderwerp van één dataset
        "sustainab", "green", "environmental", "social responsibility",
        "climate", "bank", "financial", "insurance", "mortgage",
        "festival", "supermarket",
        # vocabulaire van één dimensie
        "association", "motivation", "barrier", "prescriptive",
    ]
    prompt = _skelet_prompt().lower()
    gevonden = [w for w in ontleend if w in prompt]
    assert not gevonden, f"geleend vocabulaire in het statische skelet: {gevonden}"


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
