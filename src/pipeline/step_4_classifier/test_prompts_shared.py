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


def test_taxonomy_block_calls_l1_the_dimension_and_not_the_lens():
    """The lens naming came out of the rebuild and goes out again: the prompts
    call the level here by the name `dimension_data` itself uses."""
    block = build_taxonomy_block(
        dimension=DIM, dimension_name=DIM.key,
        dimension_description=DIM.dimension_description,
    )
    assert "Lens" not in block
    assert "L1 — Dimension" in block


def test_level_diagnostic_no_longer_exists():
    """The dimension instruction went with the two-layer design: discovery asks
    for facets and attributes themselves, not for the axis they differ along."""
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


def test_universele_regels_verbieden_een_zelfverzonnen_restcategorie():
    """The model made eight attributes literally named 'Overig', alongside the
    catch-alls the code already offers (measured 2026-08-13)."""
    tekst = UNIVERSAL_RULES
    assert "NEVER CREATE A LEFTOVER CATEGORY" in tekst
    assert '"Other"' in tekst


def test_the_ban_does_not_clash_with_the_residual_for_bare_judgments():
    """Rule 2 sends bare judgments to a single residual precisely — that one is
    defined by what they are, not by what they lack."""
    tekst = UNIVERSAL_RULES
    assert "residual overall-judgment item" in tekst
    assert "Overall judgment" in tekst
    assert "not a ban on abstraction" in tekst.lower()
