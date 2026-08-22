"""Tests voor de runner. De LLM-fasen draaien hier niet; wat te toetsen valt is
de deterministische bedrading tussen consensus en de bestaande keten."""
import asyncio

from pipeline.step_5_codeGenerator.consensus.code_shape import CodeShape
from pipeline.step_5_codeGenerator.consensus.codebook_io import apply_overig_sweep
from pipeline.step_5_codeGenerator.consensus.concept_inventory import Concept
from pipeline.step_5_codeGenerator.consensus.grouping import ShapingResult
from pipeline.step_5_codeGenerator.consensus import run_codebook as runner
from pipeline.step_5_codeGenerator.consensus.config_consensus import ConsensusConfig
from pipeline.step_5_codeGenerator.consensus.run_codebook import (
    GeneratedCodebook, groups_from_clusters, link_children_to_overig,
    report_codebook_build, report_true_overig,
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
        shapes=[], overig_ids=[], codes=[], coverage_recovered=0, degeneration=None,
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


# ---------------------------------------------------------------------------
# De tweede schrijfcall: kinderen onder Overig
# ---------------------------------------------------------------------------

def vorm(key, leden, valentie, herkomst, umbrella="F", resp=None):
    respondenten = frozenset(resp) if resp else frozenset({f"r{key}"})
    return CodeShape(key=key, members=tuple(leden), valence=valentie,
                     umbrella=umbrella, resp_ids=respondenten,
                     resp_pos=frozenset(), resp_neg=frozenset(),
                     resp_neu=frozenset(), origin=herkomst)


def code_voor(vorm_, concept_by_id, naam):
    """Wat een schrijfcall teruggeeft: de tekst, plús de twee dingen waarop
    `_match_shape` de code weer bij zijn vorm zoekt — de bronnamen en de
    valentie. Zou de stub die twee niet echoën, dan zou de test een match
    toetsen die de echte call nooit maakt."""
    return ConsolidatedCode(
        code_name=naam, definition=f"d {naam}", diagnostic_test="t",
        valence=vorm_.valence, typical_indicators=[],
        source_attributes=[concept_by_id[m].name for m in vorm_.members])


def bedraad(monkeypatch, vormen, concepts, hoofdnaam="Hoofd", kindnaam="Kind"):
    """Draait `generate_codebook` met beide schrijfcalls gestubd en geeft terug
    wat elke call te zien kreeg. `build_shapes` wordt vervangen omdat de vormen
    hier het onderwerp van de test zijn, niet hoe ze ontstaan."""
    concept_by_id = {c.attribute_id: c for c in concepts}
    gezien = {}

    async def nooit(*args, **kwargs):
        raise AssertionError("deel 1 had niet mogen draaien")

    def stub_shapes(*args, **kwargs):
        return ShapingResult(shapes=vormen, overig_ids=[], coverage_recovered=0)

    async def stub_hoofd(shapes, *args, **kwargs):
        gezien["hoofd"] = list(shapes)
        return [code_voor(v, concept_by_id, hoofdnaam) for v in shapes]

    async def stub_kind(shapes, *args, taken_names=None, **kwargs):
        gezien["kind"] = list(shapes)
        gezien["taken_names"] = taken_names
        return [code_voor(v, concept_by_id, kindnaam) for v in shapes]

    monkeypatch.setattr(runner, "resolve_consolidations", nooit)
    monkeypatch.setattr(runner, "build_shapes", stub_shapes)
    monkeypatch.setattr(runner, "write_codebook", stub_hoofd)
    monkeypatch.setattr(runner, "write_miscellaneous", stub_kind)

    result = asyncio.run(runner.generate_codebook(
        concepts, {}, threshold=1, survey_question="V?", n_respondents=80,
        dimension_diagnostic="d", language="Dutch", config=ConsensusConfig(),
        verbose=False, prompt_printer=None, partitions=[[("A1",)]]))
    return result, gezien


def test_een_kind_wordt_door_de_tweede_call_geschreven_en_niet_dubbel(monkeypatch):
    """Tot deze bedrading kreeg `write_codebook` ÁLLE vormen, kinderen
    inbegrepen — met de hoofdcodeprompt. Wie er een tweede call naast zet
    zonder te splitsen krijgt elk kind twee keer in het boek."""
    concepts = [concept("A1", "Prijs", 40), concept("A2", "Service", 40)]
    vormen = [vorm("V1", ["A1"], "non_negative", "solo"),
              vorm("V2", ["A2"], "negative", "child")]

    result, gezien = bedraad(monkeypatch, vormen, concepts)

    assert [v.key for v in gezien["hoofd"]] == ["V1"]
    assert [v.key for v in gezien["kind"]] == ["V2"]
    assert [c.code_name for c in result.codes] == ["Hoofd", "Kind"]


def test_de_kinderen_krijgen_de_hoofdnamen_als_verboden_namen(monkeypatch):
    """De tweede call ziet de hoofdvormen niet. Zonder `taken_names` kan hij
    een naam kiezen die een hoofdcode net heeft vastgelegd."""
    concepts = [concept("A1", "Prijs", 40), concept("A2", "Service", 40)]
    vormen = [vorm("V1", ["A1"], "non_negative", "solo"),
              vorm("V2", ["A2"], "negative", "child")]

    _result, gezien = bedraad(monkeypatch, vormen, concepts)

    assert gezien["taken_names"] == ["Hoofd"]


def test_elke_code_vindt_zijn_eigen_vorm_ook_als_twee_codes_gelijk_heten(monkeypatch):
    """De reden dat hier gematcht wordt en niet gezipt of op naam gesleuteld.

    Twee calls geven hun codes in hun eigen volgorde terug, en een dict op
    `code_name` klapt in elkaar zodra twee codes dezelfde naam dragen — dat is
    hier expres het geval. De koppeling loopt daarom over ÉÉN lookup op
    (bronnamen, valentie), over alle vormen samen.
    """
    concepts = [concept("A1", "Prijs", 40), concept("A2", "Service", 2)]
    vormen = [vorm("V1", ["A1"], "non_negative", "solo",
                   resp={"r1", "r2", "r3", "r4", "r5"}),
              vorm("V2", ["A2"], "negative", "child", umbrella="Bereikbaarheid",
                   resp={"r9"})]

    result, _gezien = bedraad(monkeypatch, vormen, concepts,
                              hoofdnaam="Zelfde", kindnaam="Zelfde")

    assert [v.key for v in result.shapes] == ["V1", "V2"]
    # De zwaarste houdt de naam; het kind wordt hernoemd — en niet naar het
    # kale facet, want dat claimt de kop die een restcategorie niet heeft.
    assert [c.code_name for c in result.codes] == ["Zelfde", "Overig — Bereikbaarheid"]


def test_de_bouw_meldt_hoofdcodes_kinderen_en_hun_respondenten(capsys):
    """Zonder deze regel is de meting handwerk: hoeveel codes een eigen kop
    dragen, hoeveel eronder hangen, en hoeveel respondenten daarin zitten."""
    kind_a = vorm("V2", ["A2"], "negative", "child", resp={"r1", "r2"})
    kind_b = vorm("V3", ["A3"], "negative", "child", resp={"r2", "r3"})
    result = _leeg_codeboek(
        shapes=[vorm("V1", ["A1"], "non_negative", "solo"), kind_a, kind_b],
        codes=[ConsolidatedCode(code_name=n, definition="d", diagnostic_test="t",
                                valence="neutral", typical_indicators=[])
               for n in ("a", "b", "c")])

    report_codebook_build(result, ConsensusConfig())

    # Drie respondenten, niet vier: r2 zit in beide kinderen en telt één keer.
    assert ("CODES: 1 hoofdcode(s) + 2 kind(eren) onder Overig, samen 3 "
            "respondent(en) in de kinderen") in capsys.readouterr().out


# ---------------------------------------------------------------------------
# De ouder-kindrelatie: een veld, nooit een naam
# ---------------------------------------------------------------------------

def _boek_met_een_kind():
    """Twee geschreven codes plus hun vormen: één hoofdcode, één kind."""
    codes = [
        ConsolidatedCode(code_name="Hoofd", definition="d", diagnostic_test="t",
                         valence="non_negative", typical_indicators=[],
                         source_attributes=["Prijs"]),
        ConsolidatedCode(code_name="Kind", definition="d", diagnostic_test="t",
                         valence="negative", typical_indicators=[],
                         source_attributes=["Service"]),
    ]
    shapes = [vorm("V1", ["A1"], "non_negative", "solo"),
              vorm("V2", ["A2"], "negative", "child")]
    return codes, shapes


def test_een_kind_wijst_met_een_veld_naar_de_overig_code():
    """Toetst de WAARDE, niet het bestaan van een kind.

    `models.py` negeert een onbekend init-argument stilzwijgend (pydantic's
    standaardgedrag, en `extra="forbid"` aanzetten zou elke instructor-call
    raken), dus een tikfout als `parent_code=` levert een OUDERLOZE code op
    zonder ergens te falen. Een assertie op "er is een kind" zou daar
    doorheen glippen; deze niet.
    """
    codes, shapes = _boek_met_een_kind()

    overig = apply_overig_sweep(codes, {}, "Dutch")
    kind_ids = link_children_to_overig(codes, shapes, overig)

    # De ids, niet een telling: de scorecard legt deze bedoeling naast het
    # `parent_code_id`-veld en heeft daar de `K#`'s zelf voor nodig.
    assert kind_ids == ["K2"]
    assert overig.code_id == "K3"
    assert codes[1].parent_code_id == "K3"
    assert codes[0].parent_code_id is None


def test_de_ouder_bestaat_pas_na_de_sweep():
    """De volgordebewaking: vóór `apply_overig_sweep` is er geen Overig-code en
    dus geen id om naar te wijzen. Deze test legt vast dat de sweep hem munt —
    zou dat pas bij de cache-write gebeuren, dan wees elk kind naar niets."""
    codes, _shapes = _boek_met_een_kind()

    assert [c.code_id for c in codes] == ["", ""]

    overig = apply_overig_sweep(codes, {}, "Dutch")

    assert overig.code_id
    assert [c.code_id for c in codes] == ["K1", "K2", "K3"]


def test_de_kinderen_krijgen_een_eigen_knummer_in_het_gecachte_codeboek():
    """`ensure_codebook_ids` mint K# op de dictvorm op schijf. Een kind is een
    code als elke andere en hoort dus een eigen nummer te krijgen — geen dat
    het met zijn ouder deelt — en de verwijzing naar die ouder moet de
    serialisatie overleven."""
    from models import CodingResultsCache, DomainSet
    from utils.identity import ensure_codebook_ids

    codes, shapes = _boek_met_een_kind()
    overig = apply_overig_sweep(codes, {}, "Dutch")
    link_children_to_overig(codes, shapes, overig)

    cache = CodingResultsCache(
        partition_set=DomainSet(partitions=[]), partition_results={},
        raw_codes=[c.model_dump() for c in codes])
    ensure_codebook_ids(cache)

    ids = [c["code_id"] for c in cache.raw_codes]
    assert ids == ["K1", "K2", "K3"]
    assert len(set(ids)) == 3
    assert cache.raw_codes[1]["parent_code_id"] == cache.raw_codes[2]["code_id"]


def test_de_koppeling_eist_dat_overig_de_enige_extra_code_is():
    """`codes` is op dit punt `shapes` plus precies de Overig-code. Klopt die
    lengte niet, dan is de positionele afspraak verschoven en zouden de
    verkeerde codes onder Overig komen te hangen — zonder ergens te falen."""
    import pytest

    codes, shapes = _boek_met_een_kind()

    with pytest.raises(ValueError):
        link_children_to_overig(codes, shapes, codes[0])


def test_een_attribuut_dat_alleen_een_kind_claimt_is_geen_wees():
    """De sweep leidt Overig af uit "wat geen enkele code noemt". Een kind IS
    een code, dus zijn attributen horen niet óók nog in Overig te belanden —
    geverifieerd in plaats van aangenomen, want dubbel geplaatst materiaal is
    precies wat de scorecard nooit meer terugvindt."""
    from models import DomainResultModel

    codes, shapes = _boek_met_een_kind()
    resultaten = {"D": DomainResultModel(
        partition_name="D", n_labels=2, n_batches=1,
        attributes={"F": [{"attribute_name": "Prijs", "attribute_id": "A1"},
                          {"attribute_name": "Service", "attribute_id": "A2"}]},
    )}

    overig = apply_overig_sweep(codes, resultaten, "Dutch")
    link_children_to_overig(codes, shapes, overig)

    assert overig.source_attributes == []


# ---------------------------------------------------------------------------
# Wat "echt-overig" vandaag écht betekent
# ---------------------------------------------------------------------------

def test_echt_overig_meldt_wat_niet_in_overig_belandde(capsys):
    """Het gat dat dit plan niet dicht: een attribuut wiens ene pool hoofdcode
    werd en wiens andere pool door de bodem zakte staat in `overig_ids`, maar
    de sweep ziet het niet als wees omdat de overlevende code het noemt. Die
    respondenten worden nergens geteld. Gemeld, niet stil."""
    concepts = [concept("A1", "Prijs", 40)]
    codes = [ConsolidatedCode(code_name="Hoofd", definition="d",
                              diagnostic_test="t", valence="non_negative",
                              typical_indicators=[], source_attributes=["Prijs"])]
    overig = apply_overig_sweep(codes, {}, "Dutch")
    result = _leeg_codeboek(overig_ids=["A1"],
                            concept_by_id={c.attribute_id: c for c in concepts})

    report_true_overig(result, overig)

    uit = capsys.readouterr().out
    assert "IN ATTRIBUTEN: 1 attribuut(en) bleven onder de bodem; 0 daarvan" in uit
    assert "LET OP: 1 niet" in uit and "Prijs" in uit


def test_geen_echt_overig_regel_zonder_gevallen_unies(capsys):
    result = _leeg_codeboek(overig_ids=[])
    overig = ConsolidatedCode(code_name="Overig", definition="d",
                              diagnostic_test="t", valence="neutral",
                              typical_indicators=[])

    report_true_overig(result, overig)

    assert "ECHT-OVERIG" not in capsys.readouterr().out


def test_echt_overig_meldt_het_gat_in_respondenten(capsys):
    """De kopregel, en de eenheid waarin het besluit valt. Een attribuuttelling
    zegt niets over omvang: op set 7 staan 9 attributen onder de bodem terwijl
    er 5 van 2317 respondenten werkelijk in geen enkele code voorkomen."""
    concepts = [concept("A1", "Prijs", 40)]
    concept_by_id = {c.attribute_id: c for c in concepts}
    codes = [ConsolidatedCode(code_name="Hoofd", definition="d",
                              diagnostic_test="t", valence="non_negative",
                              typical_indicators=[], source_attributes=["Prijs"])]
    overig = apply_overig_sweep(codes, {}, "Dutch")
    gedekt = vorm("v1", ["A1"], "non_negative", "solo",
                  resp=[f"rA1{i}" for i in range(38)])
    result = _leeg_codeboek(overig_ids=["A1"], shapes=[gedekt],
                            concept_by_id=concept_by_id)

    report_true_overig(result, overig)

    uit = capsys.readouterr().out
    assert "ECHT-OVERIG: 2 van 40 respondent(en) komen in geen enkele code voor" in uit
    assert "IN ATTRIBUTEN: 1 attribuut(en)" in uit


def test_het_respondentengat_wordt_ook_zonder_gevallen_unies_gemeld(capsys):
    """Zwijgen mag alleen als er niets te melden is. Een respondent die nergens
    staat is iets te melden, ook als geen enkel attribuut onder de bodem bleef —
    anders zou de zwijgregel het getal verbergen dat hij moest opleveren."""
    concepts = [concept("A1", "Prijs", 40)]
    codes = [ConsolidatedCode(code_name="Hoofd", definition="d",
                              diagnostic_test="t", valence="non_negative",
                              typical_indicators=[], source_attributes=["Prijs"])]
    overig = apply_overig_sweep(codes, {}, "Dutch")
    gedekt = vorm("v1", ["A1"], "non_negative", "solo",
                  resp=[f"rA1{i}" for i in range(38)])
    result = _leeg_codeboek(overig_ids=[], shapes=[gedekt],
                            concept_by_id={c.attribute_id: c for c in concepts})

    report_true_overig(result, overig)

    uit = capsys.readouterr().out
    assert "ECHT-OVERIG: 2 van 40 respondent(en)" in uit
    assert "IN ATTRIBUTEN" not in uit
