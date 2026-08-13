"""Tests voor de gedeelde promptbouwstenen (step 4)."""
from pipeline.step_3_ideaExtractor.dimension_data import get_dimensions_in_decision_order
from pipeline.step_4_classifier.prompts_shared import (
    INSTRUCTOR_HINT,
    UNIVERSAL_RULES,
    build_context_block,
    build_cross_scope_model,
    build_taxonomy_block,
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


def test_taxonomy_block_noemt_l1_de_dimensie_en_niet_de_lens():
    """De lens-benaming kwam uit de herbouw en gaat er weer uit: de prompts
    heten hier weer bij het niveau dat `dimension_data` zelf hanteert."""
    block = build_taxonomy_block(
        dimension=DIM, dimension_name=DIM.key,
        dimension_description=DIM.dimension_description,
    )
    assert "Lens" not in block
    assert "L1 — Dimension" in block


def test_level_diagnostic_bestaat_niet_meer():
    """De dimensie-opdracht is weg met de tweelaagse opzet: discovery vraagt
    naar facetten en attributen zelf, niet naar de as waarop ze verschillen."""
    import pipeline.step_4_classifier.prompts_shared as ps
    assert not hasattr(ps, "level_diagnostic")


def test_universele_regels_dekken_de_drie_afspraken():
    tekst = UNIVERSAL_RULES.lower()
    assert "descriptive" in tekst
    assert "valence" in tekst
    assert "evaluative direction" in tekst


def test_instructor_hint_is_de_exacte_zin():
    assert INSTRUCTOR_HINT == (
        "provide your output as valid JSON following the response schema provided"
    )


def test_cross_scope_model_dwingt_de_id_ruimte_af():
    model = build_cross_scope_model(["A1", "A2"], "attribute")
    fields = model.model_fields
    assert set(fields) == {"scratchpad", "items"}
    item = fields["items"].annotation.__args__[0]
    assert set(item.model_fields) == {"name", "definition", "source_ids", "home_id"}
