"""Tests voor naslijpen en cross-domein (step 4)."""
from pipeline.step_3_ideaExtractor.dimension_data import get_dimensions_in_decision_order
from pipeline.step_4_classifier.drains import make_drain_attribute
from pipeline.step_4_classifier.prompts_refinement import (
    RefinedAttribute,
    RefinementMisfitGroup,
    RefinementResult,
    build_contents_block,
    build_cross_domain_prompt,
    build_refinement_prompt,
)
from pipeline.step_4_classifier.prompts_shared import INSTRUCTOR_HINT

DIM = get_dimensions_in_decision_order()[0]

FACETS = [{
    "facet_name": "Snelheid",
    "facet_definition": "Hoe snel er geleverd wordt.",
    "attributes": [
        {"attribute_name": "Wachttijd", "attribute_definition": "De tijd tot antwoord."},
        make_drain_attribute("Snelheid", "Dutch"),
    ],
}]
CONTENTS = {"Wachttijd": ["lange wachttijd", "duurt lang", "snel geholpen"]}
SHARES = {"Wachttijd": 0.62}
COUNTS = {"Wachttijd": 124}


def _kwargs(**overrides):
    base = dict(
        language="Dutch", survey_question="Waar denkt u aan?",
        sector="finance", entity="asn_bank", topic="brand_association",
        perspective="consumer", intent="associate",
        dimension=DIM, dimension_name=DIM.key,
        dimension_description=DIM.dimension_description,
        domain_label="dienstverlening",
        domain_definition="Alles wat de organisatie aanbiedt en levert.",
        contents_block=build_contents_block(FACETS, CONTENTS, SHARES, COUNTS, 5),
    )
    base.update(overrides)
    return base


def _xkwargs(**overrides):
    base = dict(
        language="Dutch", survey_question="Waar denkt u aan?",
        sector="finance", entity="asn_bank", topic="brand_association",
        perspective="consumer", intent="associate",
        inventory_block="[A1] Wachttijd — 124 responses",
    )
    base.update(overrides)
    return base


# =============================================================================
# HET INHOUDSBLOK
# =============================================================================

def test_inhoudsblok_zet_claim_naast_inhoud():
    block = build_contents_block(FACETS, CONTENTS, SHARES, COUNTS, 5)
    assert "Claims to capture" in block
    assert "Actually holds" in block
    assert block.index("Claims to capture") < block.index("Actually holds")


def test_inhoudsblok_toont_aandeel_relatief():
    block = build_contents_block(FACETS, CONTENTS, SHARES, COUNTS, 5)
    assert "124 responses" in block
    assert "62% of this domain" in block


def test_inhoudsblok_kapt_op_top_n():
    block = build_contents_block(FACETS, CONTENTS, SHARES, COUNTS, 2)
    assert "snel geholpen" not in block


def test_leeg_attribuut_wordt_zo_benoemd():
    block = build_contents_block(FACETS, {}, {}, {}, 5)
    assert "nothing was assigned to it" in block


# =============================================================================
# DE HERSCHREVEN SPLITSCLAUSULE
# =============================================================================

def test_oude_splitsclausule_is_weg():
    """'a large share AND visibly diverse contents is too abstract: split it'
    vuurde voortdurend zodra discovery items bewust breed maakte."""
    prompt = build_refinement_prompt(**_kwargs())
    assert "too abstract" not in prompt


def test_nieuwe_splitsclausule_eist_twee_onderscheiden_antwoorden():
    prompt = build_refinement_prompt(**_kwargs())
    assert "two distinct ANSWERS to the same question" in prompt


def test_widen_is_de_standaard_bij_twijfel():
    prompt = build_refinement_prompt(**_kwargs())
    assert "Otherwise WIDEN" in prompt
    assert "prefer widen" in prompt


def test_breed_is_op_zichzelf_geen_probleem():
    prompt = build_refinement_prompt(**_kwargs())
    assert "not by itself a problem" in prompt


# =============================================================================
# DE VIJF UITGANGEN
# =============================================================================

def test_alle_vijf_uitgangen_staan_in_de_prompt():
    prompt = build_refinement_prompt(**_kwargs())
    for uitgang in ("merge", "widen", "split", "move", "out"):
        assert f'"{uitgang}"' in prompt, uitgang


def test_model_kent_de_vier_acties_en_de_twee_verdicts():
    assert str(RefinedAttribute.model_fields["action"].annotation) == (
        "typing.Literal['keep', 'merge', 'widen', 'split']")
    assert str(RefinementMisfitGroup.model_fields["verdict"].annotation) == (
        "typing.Literal['move', 'out']")


def test_resultaat_draagt_attributen_en_misfits():
    assert set(RefinementResult.model_fields) == {
        "scratchpad", "attributes", "misfits"}


# =============================================================================
# DOMEIN VAST, FACET NIET
# =============================================================================

def test_attribuut_mag_binnen_het_domein_van_facet_wisselen():
    prompt = build_refinement_prompt(**_kwargs())
    assert "THE DOMAIN IS FIXED, THE FACET IS NOT" in prompt
    assert "facet_name" in RefinedAttribute.model_fields


def test_prompt_noemt_elk_veld_dat_de_modellen_kennen():
    prompt = build_refinement_prompt(**_kwargs())
    for veld in ("scratchpad", "attributes", "misfits", "action", "facet_name",
                 "attribute_name", "attribute_definition", "example_observations",
                 "source_attributes", "instance_texts", "verdict",
                 "target_attribute"):
        assert veld in prompt, veld


# =============================================================================
# VANGNETTEN DOEN NIET MEE
# =============================================================================

def test_naslijpen_laat_de_vangnetten_met_rust():
    prompt = build_refinement_prompt(**_kwargs())
    assert "LEAVE THE CATCH-ALLS ALONE" in prompt


def test_cross_domein_laat_de_vangnetten_met_rust():
    prompt = build_cross_domain_prompt(**_xkwargs())
    assert "catch-all" in prompt


# =============================================================================
# CROSS-DOMEIN
# =============================================================================

def test_cross_domein_werkt_op_ids():
    prompt = build_cross_domain_prompt(**_xkwargs())
    assert "source_ids" in prompt and "home_id" in prompt


def test_cross_domein_zegt_dat_een_solist_een_groep_van_een_is():
    prompt = build_cross_domain_prompt(**_xkwargs())
    assert "group of one" in prompt


def test_cross_domein_eist_dat_elk_id_precies_een_keer_voorkomt():
    prompt = build_cross_domain_prompt(**_xkwargs())
    assert "exactly one" in prompt


# =============================================================================
# ALGEMEEN
# =============================================================================

def test_beide_prompts_eindigen_op_de_instructor_zin():
    assert build_refinement_prompt(**_kwargs()).rstrip().endswith(INSTRUCTOR_HINT)
    assert build_cross_domain_prompt(**_xkwargs()).rstrip().endswith(INSTRUCTOR_HINT)


def test_beide_prompts_dragen_de_universele_regels():
    assert "<universal_rules>" in build_refinement_prompt(**_kwargs())
    assert "<universal_rules>" in build_cross_domain_prompt(**_xkwargs())


def test_regels_gebruiken_niet_het_woord_dimension():
    prompt = build_refinement_prompt(**_kwargs())
    regels = prompt[prompt.index("# Rules"):prompt.index("# Required Process")]
    assert "dimension" not in regels.lower()


def test_geen_drempelgetallen_in_de_regels():
    """Aandelen komen uit de data en mogen; een vaste drempel is van een
    dataset afgelezen en mag niet."""
    prompt = build_refinement_prompt(**_kwargs())
    regels = prompt[prompt.index("# Rules"):prompt.index("# Required Process")]
    assert "%" not in regels
    assert "never against a fixed number" in " ".join(regels.split())


def test_vangnetten_worden_gemarkeerd_niet_op_naam_herkend():
    """De naam is vertaald en herschrijfbaar; de markering komt uit drain_key."""
    block = build_contents_block(FACETS, CONTENTS, SHARES, COUNTS, 5)
    gemarkeerd = [r for r in block.splitlines() if "[CATCH-ALL]" in r]
    assert len(gemarkeerd) == 1
    assert "Overig" in gemarkeerd[0]
    assert "Wachttijd" not in gemarkeerd[0]


def test_regel_negen_verwijst_naar_de_markering():
    prompt = build_refinement_prompt(**_kwargs())
    assert "marked [CATCH-ALL]" in prompt
    assert "Judge only what is not marked" in prompt
