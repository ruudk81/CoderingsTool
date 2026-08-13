"""Tests voor de facetprompts (step 4)."""
import pytest
from pydantic import ValidationError
from pipeline.step_3_ideaExtractor.dimension_data import get_dimensions_in_decision_order
from pipeline.step_4_classifier.prompts_facet import (
    ConsolidatedFacet,
    DiscoveredFacet,
    FacetConsolidationResult,
    FacetDiscoveryResult,
    FacetRefinementResult,
    RefinedFacet,
    build_facet_contents_block,
    build_facet_refinement_prompt,
    build_facet_assignment_model,
    build_facet_assignment_prompt,
    build_facet_consolidation_prompt,
    build_facet_discovery_prompt,
    build_facet_menu,
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


def test_discovery_stelt_de_diagnostische_vraag_niet():
    """De opdracht is dimensies vinden. De diagnostische vraag ernaast zetten gaf
    een tautologie: het model antwoordde in zes van de zeven domeinen dat de
    dimensie "type <domein>" was — de vraag zelf, teruggegeven als antwoord
    (gemeten ASN Qd1, 2026-08-12)."""
    assert DIM.prompt_rules.facet_diagnostic not in _prompt()


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


# =============================================================================
# Taak 3 — consolidatie over chunks
# =============================================================================

def _consolidatie_prompt():
    return build_facet_consolidation_prompt(
        **CTX,
        domain_label=DOMAIN["domain_label"],
        domain_definition=DOMAIN["domain_definition"],
        domain_boundary_test=DOMAIN["domain_boundary_test"],
        candidates=[
            facet("Groen imago", example_observations=["groen", "natuurlijk"]),
            facet("Groene uitstraling", example_observations=["oogt groen"]),
        ],
    )


def test_consolidatie_toont_elke_kandidaat_met_voorbeelden():
    prompt = _consolidatie_prompt()
    assert "Groen imago" in prompt
    assert "Groene uitstraling" in prompt
    assert "natuurlijk" in prompt
    assert "oogt groen" in prompt


def test_consolidatie_bevat_de_grenstoets_als_mergecriterium():
    prompt = _consolidatie_prompt().lower()
    assert "boundary" in prompt
    assert "merge" in prompt


def test_consolidatie_bevat_de_facetdiagnostiek():
    assert DIM.prompt_rules.facet_diagnostic in _consolidatie_prompt()


def test_consolidatie_eindigt_op_de_instructor_hint():
    assert _consolidatie_prompt().rstrip().endswith(
        "provide your output as valid JSON following the response schema provided"
    )


def test_geconsolideerd_facet_noemt_zijn_bronnen():
    f = ConsolidatedFacet(
        facet_name="Groene uitstraling", facet_definition="Hoe groen het oogt.",
        boundary_test="Gaat dit over hoe het oogt?", exclusions=["beleid"],
        example_observations=["groen"],
        source_facets=["Groen imago", "Groene uitstraling"],
    )
    assert len(f.source_facets) == 2


def test_consolidatieresultaat_accepteert_lege_lijst():
    assert FacetConsolidationResult(scratchpad="s", facets=[]).facets == []


# =============================================================================
# Taak 4 — toewijzing
# =============================================================================

ASSIGN_CTX = dict(
    language="Dutch", survey_question="Waar denkt u aan?",
    sector="finance", entity="asn_bank", topic="brand_association",
    perspective="consumer", intent="associate",
    domain_label="Duurzaamheid",
    domain_definition="Antwoorden over het milieubeleid van de entiteit.",
)


def _facetten():
    return [ConsolidatedFacet(
        facet_name="Groene uitstraling", facet_definition="Hoe groen het oogt.",
        boundary_test="Gaat dit over hoe het oogt?", exclusions=["concreet beleid"],
        example_observations=["groen"], source_facets=["Groen imago"],
    )]


def test_menu_nummert_en_toont_grens_en_uitsluiting():
    menu = build_facet_menu(_facetten())
    assert "[F1]" in menu
    assert "Groene uitstraling" in menu
    assert "Gaat dit over hoe het oogt?" in menu
    assert "concreet beleid" in menu


def test_toewijzingsprompt_bevat_menu_en_ideeen():
    prompt = build_facet_assignment_prompt(
        **ASSIGN_CTX, facets=_facetten(),
        ideas=[("i1", "heel groen"), ("i2", "dure bank")],
    )
    assert "[F1]" in prompt
    assert "[i1] heel groen" in prompt
    assert "[i2] dure bank" in prompt
    assert "F_NONE" in prompt


def test_model_weigert_onbekend_facet_id():
    Model = build_facet_assignment_model(["F1"], ["i1"])
    with pytest.raises(ValidationError):
        Model(assignments=[{"idea_id": "i1", "assigned_facet_id": "F9",
                            "confidence": 0.9, "valence": "0"}])


def test_model_weigert_onbekend_idea_id():
    Model = build_facet_assignment_model(["F1"], ["i1"])
    with pytest.raises(ValidationError):
        Model(assignments=[{"idea_id": "i7", "assigned_facet_id": "F1",
                            "confidence": 0.9, "valence": "0"}])


def test_model_weigert_valence_buiten_de_drie_waarden():
    Model = build_facet_assignment_model(["F1"], ["i1"])
    with pytest.raises(ValidationError):
        Model(assignments=[{"idea_id": "i1", "assigned_facet_id": "F1",
                            "confidence": 0.9, "valence": "positief"}])


def test_model_weigert_confidence_buiten_bereik():
    Model = build_facet_assignment_model(["F1"], ["i1"])
    with pytest.raises(ValidationError):
        Model(assignments=[{"idea_id": "i1", "assigned_facet_id": "F1",
                            "confidence": 1.5, "valence": "0"}])


def test_model_accepteert_f_none():
    Model = build_facet_assignment_model(["F1"], ["i1"])
    result = Model(assignments=[{"idea_id": "i1", "assigned_facet_id": "F_NONE",
                                 "confidence": 0.2, "valence": "0"}])
    assert result.assignments[0].assigned_facet_id == "F_NONE"


# =============================================================================
# Taak 5 — naslijpen na toewijzing
# =============================================================================

def _refined(name, action="keep", sources=None, texts=None):
    return RefinedFacet(
        action=action, facet_name=name, facet_definition="d",
        boundary_test="b?", exclusions=["x"], example_observations=["e"],
        source_facets=sources or [name], instance_texts=texts or [],
    )


def _naslijp_prompt():
    return build_facet_refinement_prompt(
        language="Dutch", survey_question="Waar denkt u aan?",
        sector="finance", entity="asn_bank", topic="brand_association",
        perspective="consumer", intent="associate", dimension=DIM,
        domain_label="Duurzaamheid", domain_definition="Milieubeleid.",
        facets_block=build_facet_contents_block([("A", 10, 0.5, ["t"])]),
    )


def test_contents_block_toont_aantal_aandeel_en_teksten():
    block = build_facet_contents_block([("Groene uitstraling", 120, 0.34, ["groen", "natuurlijk"])])
    assert "Groene uitstraling" in block
    assert "120" in block
    assert "34" in block
    assert "natuurlijk" in block


def test_naslijpprompt_bevat_de_vier_acties_en_twee_verdicts():
    prompt = _naslijp_prompt()
    for woord in ("keep", "merge", "widen", "split", "move", "out"):
        assert woord in prompt


def test_naslijpprompt_zegt_dat_het_domein_vaststaat():
    assert "Duurzaamheid" in _naslijp_prompt()


def test_naslijpprompt_gebruikt_contentless_test_voor_de_out_uitgang():
    """De 'out'-uitgang beschrijft inhoudsloosheid via de dimensie-eigen
    contentless_test, niet via standing_not_known.short — dat laatste zegt dat
    de respondent het onderwerp niet kent, en dat IS inhoud."""
    prompt = _naslijp_prompt()
    assert DIM.prompt_rules.contentless_test in prompt
    assert DIM.standing_not_known.short not in prompt


def test_split_zonder_instance_texts_wordt_geweigerd():
    with pytest.raises(ValidationError):
        FacetRefinementResult(scratchpad="s", facets=[_refined("A", action="split")], misfits=[])


def test_split_met_instance_texts_wordt_geaccepteerd():
    result = FacetRefinementResult(
        scratchpad="s", facets=[_refined("A", action="split", texts=["t1"])], misfits=[]
    )
    assert result.facets[0].instance_texts == ["t1"]


def test_bron_geclaimd_door_twee_facetten_zonder_teksten_wordt_geweigerd():
    with pytest.raises(ValidationError):
        FacetRefinementResult(
            scratchpad="s",
            facets=[_refined("A", sources=["X"]), _refined("B", sources=["X"])],
            misfits=[],
        )


def test_bron_geclaimd_door_twee_facetten_met_teksten_mag_wel():
    result = FacetRefinementResult(
        scratchpad="s",
        facets=[
            _refined("A", action="split", sources=["X"], texts=["t1"]),
            _refined("B", action="split", sources=["X"], texts=["t2"]),
        ],
        misfits=[],
    )
    assert len(result.facets) == 2
