"""Tests voor de gedeelde promptbouwstenen (step 4)."""
import pytest
from pipeline.step_3_ideaExtractor.dimension_data import get_dimensions_in_decision_order
from pipeline.step_4_classifier.prompts_shared import (
    INSTRUCTOR_HINT,
    UNIVERSAL_RULES,
    build_context_block,
    build_taxonomy_block,
    level_diagnostic,
)

DIM = get_dimensions_in_decision_order()[0]


def test_context_block_bevat_alle_zeven_velden():
    block = build_context_block(
        language="Dutch", survey_question="Waar denkt u aan?",
        sector="finance", entity="asn_bank", topic="brand_association",
        perspective="consumer", intent="associate",
    )
    for value in ("Dutch", "Waar denkt u aan?", "finance",
                  "asn_bank", "brand_association", "consumer", "associate"):
        assert value in block


def test_taxonomy_block_bevat_alle_vier_niveaus():
    block = build_taxonomy_block(
        dimension=DIM, dimension_name=DIM.key,
        dimension_description=DIM.dimension_description,
    )
    for marker in ("L1", "L2", "L3", "L4"):
        assert marker in block


def test_level_diagnostic_kiest_de_juiste_vraag():
    assert level_diagnostic(DIM, "facet") == DIM.prompt_rules.facet_diagnostic
    assert level_diagnostic(DIM, "attribute") == DIM.prompt_rules.attribute_diagnostic


def test_level_diagnostic_weigert_onbekend_niveau():
    with pytest.raises(ValueError):
        level_diagnostic(DIM, "domain")


def test_universele_regels_dekken_de_drie_afspraken():
    tekst = UNIVERSAL_RULES.lower()
    assert "descriptive" in tekst
    assert "valence" in tekst
    assert "evaluative direction" in tekst


def test_instructor_hint_is_de_exacte_zin():
    assert INSTRUCTOR_HINT == (
        "provide your output as valid JSON following the response schema provided"
    )
