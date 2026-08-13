"""Tests voor de chunk-consolidatieprompt (step 4)."""
from pipeline.step_3_ideaExtractor.dimension_data import get_dimensions_in_decision_order
from pipeline.step_4_classifier.prompts_discovery import (
    ConsolidatedAttribute,
    ConsolidatedFacet,
    ConsolidationResult,
    DiscoveredAttribute,
    DiscoveredFacet,
    build_candidate_block,
    build_chunk_consolidation_prompt,
)
from pipeline.step_4_classifier.prompts_shared import INSTRUCTOR_HINT

DIM = get_dimensions_in_decision_order()[0]


def _facet(name, *attr_names):
    return DiscoveredFacet(
        facet_name=name,
        facet_definition=f"Wat {name} vastlegt.",
        attributes=[DiscoveredAttribute(
            attribute_name=a,
            attribute_definition=f"De eigenschap {a}.",
            example_observations=[f"observatie over {a}"],
        ) for a in attr_names],
    )


CANDIDATES = [
    _facet("Snelheid", "Wachttijd", "Doorlooptijd"),
    _facet("Snelheid van afhandeling", "Wachttijd"),
    _facet("Bejegening", "Vriendelijkheid"),
]
RECURRENCE = {"Snelheid": 4, "Snelheid van afhandeling": 1, "Bejegening": 5}


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
        domain_exclusions=["prijs en kosten"],
        candidate_block=build_candidate_block(CANDIDATES, RECURRENCE, 6),
    )
    base.update(overrides)
    return base


# =============================================================================
# HET KANDIDATENBLOK
# =============================================================================

def test_kandidatenblok_toont_chunk_prevalentie_per_facet():
    block = build_candidate_block(CANDIDATES, RECURRENCE, 6)
    assert "Proposed in 4 of 6 independent passes" in block
    assert "Proposed in 1 of 6 independent passes" in block


def test_kandidatenblok_toont_de_attributen_genest():
    block = build_candidate_block(CANDIDATES, RECURRENCE, 6)
    assert "Wachttijd" in block
    assert "Doorlooptijd" in block
    assert block.index("Snelheid") < block.index("Wachttijd")


def test_kandidaat_zonder_telling_valt_terug_op_een_pas():
    block = build_candidate_block([_facet("Nieuw", "A")], {}, 6)
    assert "Proposed in 1 of 6 independent passes" in block


def test_kandidaat_zonder_attributen_breekt_het_blok_niet():
    block = build_candidate_block([_facet("Leeg")], {"Leeg": 2}, 3)
    assert "Leeg" in block


# =============================================================================
# PROMPT ↔ MODEL SLUITEN AAN
# =============================================================================

def test_resultaat_is_scratchpad_plus_facetten():
    assert set(ConsolidationResult.model_fields) == {"scratchpad", "facets"}


def test_geconsolideerd_facet_zegt_wat_erin_is_opgegaan():
    """Zonder dit veld ziet een samengevoegde kandidaat er identiek uit aan een
    vergeten kandidaat: allebei staan ze niet in het antwoord."""
    assert "source_facets" in ConsolidatedFacet.model_fields


def test_geconsolideerd_attribuut_zegt_hetzelfde_een_niveau_lager():
    assert "source_attributes" in ConsolidatedAttribute.model_fields


def test_geconsolideerd_facet_draagt_geconsolideerde_attributen():
    item = ConsolidatedFacet(
        facet_name="Snelheid", facet_definition="…", source_facets=["Snelheid"],
        attributes=[ConsolidatedAttribute(
            attribute_name="Wachttijd", attribute_definition="…",
            example_observations=["x"], source_attributes=["Wachttijd"])])
    assert item.attributes[0].source_attributes == ["Wachttijd"]


def test_prompt_legt_uit_dat_dekking_gecontroleerd_wordt():
    prompt = build_chunk_consolidation_prompt(**_kwargs())
    assert "source_facets" in prompt
    assert "source_attributes" in prompt
    assert "exactly one surviving" in prompt


def test_prompt_noemt_elk_veld_dat_het_model_kent():
    prompt = build_chunk_consolidation_prompt(**_kwargs())
    for veld in ("scratchpad", "facets", "facet_name", "facet_definition",
                 "attributes", "attribute_name", "attribute_definition",
                 "example_observations"):
        assert veld in prompt, veld


def test_prompt_eindigt_op_de_instructor_zin():
    assert build_chunk_consolidation_prompt(
        **_kwargs()).rstrip().endswith(INSTRUCTOR_HINT)


# =============================================================================
# DE VIER REGELS EN HUN PRECEDENTIE
# =============================================================================

def test_prompt_draagt_de_vier_groeperingsregels():
    prompt = build_chunk_consolidation_prompt(**_kwargs())
    for regel in ("UNDERLYING QUESTION FIRST", "PREVALENCE SETS GRANULARITY",
                  "LIFT, DON'T FLATTEN", "PLAIN, MEANINGFUL LABELS"):
        assert regel in prompt, regel


def test_prompt_noemt_de_precedentie_expliciet():
    """Zonder rangorde vechten de regels; met rangorde wint orthogonaliteit."""
    prompt = build_chunk_consolidation_prompt(**_kwargs())
    assert "Precedence when rules conflict" in prompt


def test_regel_een_gebruikt_niet_het_woord_dimension():
    """L1 heet Dimension in het taxonomieblok. Als regel 1 het woord ook voor
    'de as waarop een begrip varieert' gebruikt, betekent het twee dingen op
    een pagina — dat is precies wat de lens-benaming ooit moest oplossen."""
    prompt = build_chunk_consolidation_prompt(**_kwargs())
    regels = prompt[prompt.index("# Consolidation Rules"):
                    prompt.index("# Step-by-Step")]
    assert "dimension" not in regels.lower()
    assert "underlying question" in regels.lower()


# =============================================================================
# DE GENESTE STAP
# =============================================================================

def test_prompt_zegt_wat_er_met_attributen_gebeurt_bij_een_facetmerge():
    prompt = build_chunk_consolidation_prompt(**_kwargs())
    assert "pool" in prompt.lower()


def test_prompt_consolideert_ook_binnen_een_facet():
    prompt = build_chunk_consolidation_prompt(**_kwargs())
    stappen = prompt[prompt.index("# Step-by-Step"):]
    assert "attribute" in stappen.lower()


# =============================================================================
# WAT ER NIET IN MAG
# =============================================================================

def test_prompt_bevat_geen_drempelgetallen():
    prompt = build_chunk_consolidation_prompt(**_kwargs())
    assert "%" not in prompt


def test_prompt_kent_geen_lens():
    assert "Lens" not in build_chunk_consolidation_prompt(**_kwargs())


def test_prompt_draagt_de_universele_regels():
    assert "<universal_rules>" in build_chunk_consolidation_prompt(**_kwargs())


def test_prompt_zonder_uitsluitingen_blijft_geldig():
    prompt = build_chunk_consolidation_prompt(**_kwargs(domain_exclusions=[]))
    assert prompt.rstrip().endswith(INSTRUCTOR_HINT)
