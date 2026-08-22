"""Tests voor de runner. De LLM-fasen draaien hier niet; wat te toetsen valt is
de deterministische bedrading tussen consensus en de bestaande keten."""
import asyncio

from pipeline.step_5_codeGenerator.consensus.concept_inventory import Concept
from pipeline.step_5_codeGenerator.consensus import run_codebook as runner
from pipeline.step_5_codeGenerator.consensus.config_consensus import ConsensusConfig
from pipeline.step_5_codeGenerator.consensus.run_codebook import (
    GeneratedCodebook, groups_from_clusters, report_codebook_build,
)
from models import ConsolidatedCode


def concept(attribute_id, naam, n):
    respondenten = frozenset(f"r{attribute_id}{i}" for i in range(n))
    return Concept(attribute_id=attribute_id, name=naam, definition="d",
                   domain="D", facet="F", n_iu=n, resp_ids=respondenten,
                   resp_pos=respondenten, resp_neg=frozenset(),
                   resp_neu=frozenset(), is_drain=False)


def test_de_groepsnaam_is_het_zwaarste_lid():
    """Een consensusgroep is door geen enkele modelcall voorgesteld en heeft
    dus geen proposed_name. Die vult CodeShape.umbrella, en dat veld is de
    hernoemkandidaat zodra twee codes dezelfde naam krijgen — zonder
    vervanger staat daar een lege string op precies het moment dat er iets
    misgaat."""
    concepts = [concept("A1", "Klein", 2), concept("A2", "Groot", 40)]

    groepen = groups_from_clusters([("A1", "A2")], concepts)

    assert groepen[0].proposed_name == "Groot"


def test_een_onbekend_lid_laat_de_groep_niet_omvallen():
    """Een attribuut kan in de opgeslagen partities staan en geen Concept meer
    hebben — bijvoorbeeld na een herdraaide step 4."""
    concepts = [concept("A1", "Klein", 2)]

    groepen = groups_from_clusters([("A1", "A9")], concepts)

    assert groepen[0].member_ids == ("A1", "A9")
    assert groepen[0].proposed_name == "Klein"


def test_de_kandidaat_heeft_een_eigen_kosten_en_logidentiteit():
    """Deelt hij die met productie, dan lopen de twee routes in het
    kostenregister en in exports/verbose_logs door elkaar en is de vergelijking
    die deze keten moet mogelijk maken juist niet te maken."""
    assert runner.COST_STEP == "step_5_consensus"
    assert runner.CACHE_STEP == "mece_codes"


def test_de_verboselognaam_botst_niet_met_die_van_productie():
    from utils.saveVerbose import build_log_filename

    productie = build_log_filename("d.sav", "v", 100, 5)
    kandidaat = build_log_filename("d.sav", "v", 100, "5c")

    assert productie != kandidaat


def _leeg_codeboek(**overrides) -> GeneratedCodebook:
    velden = dict(
        shapes=[], overig_ids=[], codes=[], direction_loss=0, degeneration=None,
        partition_repairs=0, collisions=[], naming_mismatches=[],
        duplicate_definitions=[], vetoes=[], concept_by_id={}, runs_used=5,
        runs_failed=0, pool_log=[],
    )
    velden.update(overrides)
    return GeneratedCodebook(**velden)


def test_partition_repairs_is_een_geaggregeerde_telling_geen_lijst(capsys):
    """`generate_codebook` roept `repair_partition` N keer aan (één per run) —
    een reparatie per run is daar normaal, dus productie's regel-per-entry-lus
    zou bij N=30 het signaal verdrinken. Wat overblijft is de telling."""
    result = _leeg_codeboek(partition_repairs=7, runs_used=30)

    report_codebook_build(result, ConsensusConfig())

    output = capsys.readouterr().out
    assert "7 reparatie(s) over 30 runs" in output


def test_geen_partitieregel_zonder_reparaties(capsys):
    result = _leeg_codeboek(partition_repairs=0)

    report_codebook_build(result, ConsensusConfig())

    assert "PARTITIE" not in capsys.readouterr().out


def test_de_promptexportnaam_botst_niet_met_die_van_productie():
    """`save_prompts` opent zijn doel in 'w'-modus, zonder merge — delen de
    twee ketens één doctype, dan overschrijft wie als laatste draait stil het
    promptexport van de ander."""
    from utils.exportNaming import export_filename

    productie = export_filename("d.sav", "v", 100, "prompts_step5", "json")
    kandidaat = export_filename("d.sav", "v", 100, "prompts_step5c", "json")

    assert productie != kandidaat


def test_partities_meegeven_slaat_deel_1_over(monkeypatch):
    """De meetroute: je hebt de dertig runs al betaald en wilt er een codeboek
    uit, niet nog dertig calls. Zonder deze ingang moet de meetkant zijn eigen
    kopie van de keten onderhouden — en daar liepen er dertig regels uit
    elkaar."""
    async def nooit(*args, **kwargs):
        raise AssertionError("deel 1 had niet mogen draaien")

    concepts = [concept("A1", "Prijs", 40), concept("A2", "Service", 40)]
    concept_by_id = {c.attribute_id: c for c in concepts}

    async def stub_writer(shapes, *args, **kwargs):
        # `_match_shape` zoekt op (bronnaam-verzameling, valentie) — dus de
        # namen van ALLE leden van de vorm, niet alleen de eerste, en de naam
        # (niet het attribute_id) omdat `shape.members` id's zijn.
        return [ConsolidatedCode(
            code_name="X", definition="d", diagnostic_test="t",
            valence="non_negative", typical_indicators=[],
            source_attributes=[concept_by_id[m].name for m in shapes[0].members])]

    monkeypatch.setattr(runner, "resolve_consolidations", nooit)
    monkeypatch.setattr(runner, "write_codebook", stub_writer)
    partities = [[("A1", "A2")], [("A1", "A2")]]

    result = asyncio.run(runner.generate_codebook(
        concepts, {}, threshold=1, survey_question="V?", n_respondents=80,
        dimension_diagnostic="d", language="Dutch", config=ConsensusConfig(),
        verbose=False, prompt_printer=None, partitions=partities))

    assert result.runs_used == 2
    assert result.runs_failed == 0


def test_de_cache_zegt_waar_de_codes_vandaan_komen():
    """Zonder dit draait step 6 een week later op codes waarvan niet meer af te
    lezen is met welke tau of welke poolindeling ze gemaakt zijn — terwijl de
    cachesleutel gedeeld is met productie en dus ook door de andere keten
    beschreven kan zijn."""
    regel = runner.provenance(ConsensusConfig(tau=0.7, two_pole=True), runs_used=30)

    assert "consensus" in regel and "tau=0.7" in regel and "30 runs" in regel


def test_de_herkomst_landt_op_het_gecachte_codeboek(monkeypatch):
    """Niet genoeg dat `provenance()` een regel teruggeeft: die regel moet ook
    echt op het object staan dat `CacheManager` bewaart, anders bouwt de
    aanroep de herkomst en laat hem toch vallen."""
    from models import DomainSet
    from pipeline.step_5_codeGenerator.consensus import codebook_io

    captured = {}

    class FakeCacheManager:
        def save_metadata_to_cache(self, metadata, **kwargs):
            captured["metadata"] = metadata
            return True

    monkeypatch.setattr(codebook_io, "CacheManager", FakeCacheManager)

    regel = runner.provenance(ConsensusConfig(tau=0.7, two_pole=True), runs_used=30)
    runner.cache_mece_results(
        partition_set=DomainSet(partitions=[]), pydantic_results={}, codes=[],
        filename="f", variable="v", sample_size=10, variable_key="k",
        narrative=regel,
    )

    assert captured["metadata"].codebook_narrative == regel
