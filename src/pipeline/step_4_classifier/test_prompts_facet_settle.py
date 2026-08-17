"""Tests voor het naslijpen van facetten (step 4)."""
import pydantic
import pytest

from pipeline.step_3_ideaExtractor.dimension_data import get_dimensions_in_decision_order
from pipeline.step_4_classifier.prompts_facet_settle import (
    SettledFacetCard,
    build_facet_settle_block,
    build_facet_settle_model,
    build_facet_settle_prompt,
)
from pipeline.step_4_classifier.prompts_shared import INSTRUCTOR_HINT
from pipeline.step_4_classifier.test_prompts_shared import (
    assert_every_field_is_described, assert_prompt_does_not_restate_the_schema,
)

DIM = get_dimensions_in_decision_order()[0]

FACETS = [
    {
        "facet_name": "Snelheid",
        "facet_definition": "Hoe snel er geleverd wordt.",
        "facet_question": "Hoe snel wordt er geleverd?",
        "attributes": [
            {"attribute_name": "Wachttijd", "attribute_id": "A1",
             "attribute_definition": "De tijd tot antwoord."},
            {"attribute_name": "Levertijd", "attribute_id": "A2",
             "attribute_definition": "De tijd tot levering."},
        ],
    },
]
# Keyed on the [F#] id this block assigns to `facets`, in that same order —
# never on the name: two facets of one domain may legally share one.
COUNTS = {"F1": 124}
SHARES = {"F1": 0.62}


def _blok(facets=FACETS, counts=COUNTS, shares=SHARES):
    return build_facet_settle_block(facets, counts, shares)


def _skwargs(**overrides):
    base = dict(
        language="Dutch", survey_question="Waar denkt u aan?",
        sector="finance", entity="asn_bank", topic="brand_association",
        perspective="consumer", intent="associate",
        dimension=DIM, dimension_name=DIM.key,
        dimension_description=DIM.dimension_description,
        domain_label="dienstverlening",
        domain_definition="Alles wat de organisatie aanbiedt en levert.",
        settle_block=_blok(),
    )
    base.update(overrides)
    return base


def _rules(prompt: str) -> str:
    """De eigen regels van deze prompt, zonder de universele eronder."""
    return prompt[prompt.index("\n# Decision rules\n"):
                  prompt.index("<universal_rules>")]


def _model():
    """Het model over dezelfde id-ruimte als `_blok()` uitdeelt."""
    return build_facet_settle_model(["F1"], ["A1", "A2"])


# =============================================================================
# HET BLOK
# =============================================================================

def test_het_blok_zet_de_facetvraag_naast_wat_het_facet_werkelijk_houdt():
    """Wat het facet claimt te beantwoorden, en daarna wat er feitelijk onder
    hangt — in die volgorde, want de claim is wat aan de inhoud getoetst wordt.
    De responsteksten zijn hier weg; de attributen zijn nu die inhoud."""
    blok = _blok()
    assert "Claims to answer" in blok
    assert "Holds these attributes" in blok
    assert blok.index("Claims to answer") < blok.index("Holds these attributes")


def test_het_blok_toont_aantal_en_aandeel_van_het_domein():
    """Het domein is de noemer die alle facetten ervan delen."""
    blok = _blok()
    assert "124 responses" in blok
    assert "62% of this domain" in blok


def test_het_blok_deelt_ids_uit_op_beide_niveaus():
    """De twee uitgangen wijzen naar allebei: een merge naar facet-ids, een
    verplaatsing naar een attribuut-id en een facet-id."""
    blok = _blok()
    assert "[F1]" in blok and "[A1]" in blok


def test_het_blok_toont_attribuutnamen_met_definities_en_zonder_voorbeelden():
    """Sinds de responsteksten uit dit blok zijn, zijn de attributen het enige
    kwalitatieve bewijs — en aan namen alleen is niet te zien of twee facetten
    dezelfde begrippen door elkaar gebruiken, wat de kernregel hier vraagt.

    Voorbeelden blijven eruit: dat is de laag waarmee de voorganger van
    facetconsolidatie de attributen óók ging vastzetten. Hier kan dat niet —
    `attribute_moves` verplaatst en verandert niets — maar het materiaal is
    hier evengoed niet nodig.
    """
    blok = _blok()
    assert "Wachttijd" in blok
    assert "De tijd tot antwoord." in blok
    assert "e.g." not in blok
    assert "example" not in blok.lower()


def test_twee_gelijknamige_facetten_delen_geen_telling():
    """`build_facet_menu` staat toe dat twee facetten van één domein dezelfde
    naam dragen. Op naam keyen zou de telling van de één naar de ander lekken
    — precies de invoer waarop deze fase moet oordelen."""
    facets = [
        {"facet_name": "Snelheid", "facet_definition": "d", "facet_question": "",
         "attributes": []},
        {"facet_name": "Snelheid", "facet_definition": "d", "facet_question": "",
         "attributes": []},
    ]
    blok = build_facet_settle_block(
        facets, {"F1": 10, "F2": 90}, {"F1": 0.1, "F2": 0.9})
    assert "10 responses" in blok and "90 responses" in blok
    assert "10% of this domain" in blok and "90% of this domain" in blok


# =============================================================================
# DE UITGANGEN
# =============================================================================

def test_de_uitgangen_lopen_op_ids():
    """Op 2026-08-16 noemde de misfit-uitgang van het naslijpen zijn
    bestemmingen op naam vóór de fase en zocht ze op ná de fase; 70% landde op
    een naam die de buurcall net had opgeslokt."""
    move = _model().model_fields["attribute_moves"].annotation.__args__[0]
    assert set(move.model_fields) == {"attribute_id", "to_facet_id"}


def test_een_verzonnen_bestemming_wordt_geweigerd():
    """Het hele punt van de fabriek: `to_facet_id` mag alleen een id zijn die
    dit blok heeft uitgedeeld, anders is een verzonnen bestemming niet te
    onderscheiden van een facet dat dezelfde call heeft weggevouwen."""
    model = _model()
    move = model.model_fields["attribute_moves"].annotation.__args__[0]
    with pytest.raises(pydantic.ValidationError):
        move(attribute_id="A1", to_facet_id="F99")
    with pytest.raises(pydantic.ValidationError):
        move(attribute_id="A99", to_facet_id="F1")
    move(attribute_id="A1", to_facet_id="F1")  # binnen de ruimte: geen fout


def test_een_overlevend_facet_noemt_zijn_bronnen_op_id():
    assert "source_facet_ids" in SettledFacetCard.model_fields


def test_een_overlevend_facet_schrijft_zijn_vraag_opnieuw_op():
    """Een samengevouwen facet dat de vraag van één van zijn bronnen overneemt,
    beschrijft de merge niet."""
    assert "facet_question" in SettledFacetCard.model_fields


def test_het_resultaat_draagt_facetten_en_verplaatsingen():
    """`decision_summary` hoort erbij sinds de prompt om een expliciete
    verantwoording vraagt — zonder veld kon die alleen in de scratchpad
    landen, en die is redeneerruimte en geen opleverpunt."""
    assert set(_model().model_fields) == {
        "scratchpad", "decision_summary", "facets", "attribute_moves"}


# =============================================================================
# DE PROMPT
# =============================================================================

def test_de_prompt_verbiedt_het_hernoemen_van_attributen():
    regels = _rules(build_facet_settle_prompt(**_skwargs()))
    assert "attribute" in regels.lower()
    assert "do not rename" in regels.lower()


def test_geen_drempelgetallen_in_de_regels():
    """Aandelen komen uit de data en mogen; een vast percentage is van één
    dataset afgelezen en mag niet."""
    regels = _rules(build_facet_settle_prompt(**_skwargs()))
    assert "%" not in regels


def test_de_prompt_beschrijft_zijn_eigen_schema_niet():
    assert_prompt_does_not_restate_the_schema(build_facet_settle_prompt(**_skwargs()))


def test_het_model_beschrijft_elk_veld_dat_het_heeft():
    assert_every_field_is_described(_model())


def test_de_prompt_eindigt_op_de_universele_regels_en_de_instructor_zin():
    prompt = build_facet_settle_prompt(**_skwargs())
    assert "<universal_rules>" in prompt
    assert prompt.rstrip().endswith(INSTRUCTOR_HINT)
