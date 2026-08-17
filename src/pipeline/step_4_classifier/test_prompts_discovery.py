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
from pipeline.step_4_classifier.test_prompts_shared import (
    assert_every_field_is_described, assert_prompt_does_not_restate_the_schema,
)

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

def test_the_result_is_facets_only():
    """`scratchpad` is parked, not deleted: discovery runs on `medium` effort, so
    the model reasons internally anyway, and `analytical_question` below is the
    lighter anchor being tried in its place. Restoring it means updating this."""
    assert set(DiscoveryResult.model_fields) == {"facets"}


def test_a_facet_carries_its_attributes_and_its_question():
    """The question is asked before the attributes are named — a field ordered
    after them would be a justification, not an anchor."""
    assert list(DiscoveredFacet.model_fields) == [
        "facet_name", "facet_definition", "analytical_question", "attributes"]


def test_the_attribute_field_is_definition_not_description():
    """Step 5 reads `attribute_definition` from the taxonomy cache;
    `*_description` was the old name and a rename broke step 5 silently."""
    assert set(DiscoveredAttribute.model_fields) == {
        "attribute_name", "attribute_definition", "example_observations"}


def test_the_nested_structure_validates():
    result = DiscoveryResult(
        facets=[DiscoveredFacet(
            facet_name="Snelheid",
            facet_definition="Hoe snel er geleverd wordt.",
            analytical_question="Hoe snel wordt er geleverd?",
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

def test_the_model_describes_every_field_it_has():
    assert_every_field_is_described(DiscoveryResult)


def test_the_prompt_does_not_restate_the_schema():
    assert_prompt_does_not_restate_the_schema(build_discovery_prompt(**_kwargs()))


def test_prompt_asks_for_no_field_the_model_lacks():
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
    """The prompt asks for facets and attributes, never for the dimensions or
    axes along which responses differ. Note: 'orthogonal' IS allowed — that is
    about the relation between facets, not about finding axes."""
    prompt = build_discovery_prompt(**_kwargs())
    assert "Lens" not in prompt
    assert "dimensions on which" not in prompt.lower()
    assert "identify the dimensions" not in prompt.lower()


def test_prompt_bevat_geen_drempelgetallen():
    """A threshold like 'at least 5% of its scope' is read off one dataset and
    falls under the same ban as a use-case example."""
    prompt = build_discovery_prompt(**_kwargs())
    assert "%" not in prompt


def _skelet_prompt():
    """The prompt with EVERY dynamic slot on a sentinel — the dimension too.

    What is then left in words is the static instruction skeleton: the text sent
    unchanged for every dataset and every dimension. The builder touches only
    `prompt_rules` of a dimension, so a stub suffices and is more honest than a
    real dimension — that one smuggles its own vocabulary into the measurement.
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


def test_skeleton_borrows_no_vocabulary_from_dataset_or_dimension():
    """Tripwire for the leak path from CLAUDE.md: a diagnosis on one dataset that
    ends up in the prompt as a rule of thumb and stays there.

    Catches two kinds of leak in one measurement. Subject words ('sustainability')
    belong to one client; dimension words ('association') belong to one of the ten
    dimensions, while this prompt serves all of them.

    Not proof of agnosticism — a latch on what this repo has actually coded. If a
    new example arrives, extend the list rather than letting it dilute.
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

def test_the_prompt_shows_the_observations_numbered():
    prompt = build_discovery_prompt(**_kwargs())
    assert "1. snelle afhandeling" in prompt
    assert "3. vriendelijk personeel" in prompt


def test_prompt_zet_de_domeingrens_neer():
    prompt = build_discovery_prompt(**_kwargs())
    assert "dienstverlening" in prompt
    assert "Gaat dit over wat er geleverd wordt?" in prompt
    assert "prijs en kosten" in prompt


def test_a_prompt_without_exclusions_stays_valid():
    prompt = build_discovery_prompt(**_kwargs(
        domain_exclusions=[], domain_boundary_test=""))
    assert "dienstverlening" in prompt
    assert prompt.rstrip().endswith(INSTRUCTOR_HINT)


def test_prompt_asks_for_the_survey_language():
    assert build_discovery_prompt(**_kwargs()).count("Dutch") >= 2


def test_the_prompt_carries_the_universal_rules():
    prompt = build_discovery_prompt(**_kwargs())
    assert "<universal_rules>" in prompt


def test_prompt_asks_for_both_levels_in_one_pass():
    prompt = build_discovery_prompt(**_kwargs())
    assert "fewest" in prompt.lower()
    assert prompt.index("facet") < prompt.rindex("attribute")
