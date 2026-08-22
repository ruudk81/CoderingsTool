"""Tests voor de lezende acties (`analyse`, `vergelijk`) en de draaiende
acties (`verzamelen`, `codeboek`, `alles`). De lezende acties en de bedrading
van de draaiende acties draaien geen LLM-calls en zijn dus goedkoop om te
toetsen: het materiaal komt van schijf, niet van een call. `verzamelen` en het
LLM-gedeelte van `codeboek`/`alles` vragen een gevulde cache en draaien hier
niet."""
import builtins
import symtable
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline.step_5_codeGenerator.consensus import run_codebook as runner
from pipeline.step_5_codeGenerator.consensus.config_consensus import ConsensusConfig
from pipeline.step_5_codeGenerator.consensus.storage import RunSet, save_runset

RUNNER = Path(runner.__file__)


def _vrije_namen(table):
    """Elke naam die in dit bereik uit de module-globals of de builtins moet
    komen, plus dezelfde vraag voor alle geneste bereiken."""
    namen = {symbol.get_name() for symbol in table.get_symbols()
             if symbol.is_global() and not symbol.is_assigned()}
    for child in table.get_children():
        namen |= _vrije_namen(child)
    return namen


def test_elke_gebruikte_naam_in_de_runner_bestaat():
    """Een aanroep zonder import faalt pas op de regel zelf, en de dure paden
    van deze runner draaien niet in een test — dus wordt zo'n gat nooit
    gevangen. Dat is precies wat er gebeurde: `pool_thin_within_facet` kwam
    binnen met de facetpool, de import bleef achter, en het `codebook`-commando
    viel om na het laden van het materiaal maar vóór de schrijfcall."""
    beschikbaar = set(vars(runner)) | set(dir(builtins))

    gebruikt = _vrije_namen(symtable.symtable(
        RUNNER.read_text(encoding="utf-8"), str(RUNNER), "exec"))

    assert sorted(gebruikt - beschikbaar) == []


def test_setpad_draagt_configuratie_en_nummer():
    """Twee configuraties mogen nooit op hetzelfde bestand landen."""
    assert runner.runset_path("luna", 5).name == "consensus_luna_set5.json"
    assert runner.runset_path("gpt54", 1).name == "consensus_gpt54_set1.json"


def test_analyse_meldt_mislukte_runs(tmp_path, monkeypatch, capsys):
    """`n_failed` wordt bewaard maar tot deze fix nergens gelezen — een latere
    analyse zou dan `len(runs)` voor het gevraagde aantal aanzien en elke
    drempel stil laten verschuiven. Het hoort in de kop van `analyse` te staan,
    waar deze meting hem daadwerkelijk leest."""
    monkeypatch.setattr(runner, "OUT_DIR", tmp_path)
    runset = RunSet(
        model="gpt-5.6-luna", effort="medium",
        attribute_ids=["A1", "A2"], attribute_names={"A1": "x", "A2": "y"},
        n_respondents=10, runs=[[("A1", "A2")], [("A1",), ("A2",)]],
        salted=True, n_failed=3,
    )
    save_runset(runset, tmp_path / "consensus_luna_set9.json")

    runner.analyse(ConsensusConfig(config_name="luna"), 9)

    assert "3 mislukt" in capsys.readouterr().out


def test_verslagnaam_draagt_de_instellingen():
    """Twee varianten mogen nooit op hetzelfde bestand landen."""
    a = runner.report_path("luna", 3, "consensus", 0.6, "two")
    b = runner.report_path("luna", 3, "consensus", 0.7, "two")
    c = runner.report_path("luna", 3, "consensus", 0.6, "three")

    assert a.name == "codeboek_luna_set3_consensus_tau06_twopolen.txt"
    assert len({a.name, b.name, c.name}) == 3


def test_basislijn_krijgt_geen_tau_in_de_naam():
    """Zonder consensus doet tau niets, dus hij hoort niet in de naam."""
    assert runner.report_path("luna", 1, "baseline", 0.7, "three").name == \
        "codeboek_luna_set1_baseline_threepolen.txt"


def test_alles_weigert_bestaande_sets_te_overschrijven(tmp_path, monkeypatch):
    """`verzamelen` overschrijft zonder waarschuwing, en een set is 30 LLM-calls
    die je niet terugkrijgt. Een ronde die per ongeluk op set 3 landt wist het
    materiaal van gisteren."""
    monkeypatch.setattr(runner, "OUT_DIR", tmp_path)
    (tmp_path / "consensus_luna_set3.json").write_text("{}", encoding="utf-8")

    assert runner.bezette_sets("luna", 3, 6) == [3]
    assert runner.bezette_sets("luna", 5, 6) == []


def test_de_codeboekactie_draait_geen_deel_1(monkeypatch, tmp_path):
    """De partities zijn al betaald. Draaide deze actie deel 1 opnieuw, dan
    kost een codeboek uit bestaand materiaal dertig calls in plaats van één —
    en meet je bovendien een andere consensus dan je net analyseerde."""
    monkeypatch.setattr(runner, "OUT_DIR", tmp_path)
    runset = RunSet(
        model="gpt-5.6-luna", effort="medium",
        attribute_ids=["A1", "A2"], attribute_names={"A1": "x", "A2": "y"},
        n_respondents=10, runs=[[("A1", "A2")], [("A1",), ("A2",)]],
        salted=True, n_failed=0,
    )
    save_runset(runset, runner.runset_path("luna", 9))
    monkeypatch.setattr(runner, "load_material",
                        lambda config: {"cards": [SimpleNamespace(attribute_id="A1"),
                                                  SimpleNamespace(attribute_id="A2")]})

    aangeroepen = {}

    def spion(**kwargs):
        aangeroepen.update(kwargs)

    monkeypatch.setattr(runner, "run_codebook", spion)

    runner.codeboek(ConsensusConfig(config_name="luna"), 9, "consensus")

    assert aangeroepen["force_recalc"] is True
    assert aangeroepen["partitions"] == runset.runs


def test_codeboek_weigert_een_lege_partitieset(monkeypatch, tmp_path):
    """`partitions=[]` zou zonder deze wacht ongemerkt doorstromen (taak 2's
    review) tot diep in de consensusstap. Deze actie is de eerste echte
    aanroeper en hoort de weigering dus hier, met een duidelijke reden."""
    monkeypatch.setattr(runner, "OUT_DIR", tmp_path)
    runset = RunSet(
        model="gpt-5.6-luna", effort="medium",
        attribute_ids=["A1"], attribute_names={"A1": "x"},
        n_respondents=1, runs=[], salted=True, n_failed=1,
    )
    save_runset(runset, runner.runset_path("luna", 9))

    with pytest.raises(SystemExit):
        runner.codeboek(ConsensusConfig(config_name="luna"), 9, "consensus")


def test_codeboek_weigert_afwijkend_attribuutuniversum(monkeypatch, tmp_path):
    """`generate_codebook` telt paren alleen over de attributen van de HUIDIGE
    step-4-cache. Is de set tegen een andere boom verzameld, dan zouden
    verdwenen attributen stil uit de telling vallen en nieuwe automatisch solo
    worden — in het codeboek dat onder `mece_codes` belandt. `vergelijk`
    weigert dit al tussen twee sets; `codeboek` moet het weigeren tussen de
    set en de cache van nu, want dit is de actie die de gedeelde cache
    schrijft."""
    monkeypatch.setattr(runner, "OUT_DIR", tmp_path)
    runset = RunSet(
        model="gpt-5.6-luna", effort="medium",
        attribute_ids=["A1", "A2"], attribute_names={"A1": "x", "A2": "y"},
        n_respondents=10, runs=[[("A1", "A2")], [("A1",), ("A2",)]],
        salted=True, n_failed=0,
    )
    save_runset(runset, runner.runset_path("luna", 9))
    # De huidige step-4-cache heeft A2 niet meer en een nieuwe A3 wél.
    monkeypatch.setattr(runner, "load_material",
                        lambda config: {"cards": [SimpleNamespace(attribute_id="A1"),
                                                  SimpleNamespace(attribute_id="A3")]})

    with pytest.raises(SystemExit):
        runner.codeboek(ConsensusConfig(config_name="luna"), 9, "consensus")


def test_het_blok_levert_een_geldige_config():
    """Het blok is de enige plek waar knoppen staan; hij moet exact op
    ConsensusConfig passen, anders bestaat er alsnog een tweede tabel."""
    config = runner.config_uit_instellingen()

    assert config.config_name == runner.CONFIG
    assert config.runs == runner.RUNS
    assert config.tau == runner.TAU
    assert config.two_pole == (runner.POLES == "two")
    assert config.exclude_drains == (runner.DRAINS == "uit")
    assert config.salted == (runner.SALT == "aan")


def test_elke_actie_in_het_blok_heeft_een_afhandeling(monkeypatch):
    """Een tikfout in ACTIE moet een nette melding geven, geen stille no-op —
    en elke geldige naam moet daadwerkelijk zijn EIGEN functie bereiken. Een
    vergelijking van `ACTIES` tegen een letterlijke set zou nog steeds slagen
    als een tak uit `_draai_actie`'s if/elif-ketting zou verdwijnen; hier wordt
    daadwerkelijk dispatch getoetst."""
    assert set(runner.ACTIES) == {"alles", "verzamelen", "codeboek",
                                  "analyse", "vergelijk"}

    bereikt = {}
    for naam in runner.ACTIES:
        def spion(*args, _naam=naam, **kwargs):
            bereikt["actie"] = _naam
        monkeypatch.setattr(runner, naam, spion)

        monkeypatch.setattr(runner, "SET", 1)
        monkeypatch.setattr(runner, "SET_B", 2)
        monkeypatch.setattr(runner, "SOURCE", "consensus")

        bereikt.clear()
        runner._draai_actie(naam)
        assert bereikt.get("actie") == naam, \
            f"ACTIE {naam!r} bereikte niet zijn eigen functie"


def test_vrije_sets_telt_door_en_hergebruikt_geen_gaten(tmp_path, monkeypatch):
    """`alles` is de enige actie waarvan een knop bij ELKE run moest veranderen,
    met een weigering als faalmodus. Dat is geen instelling maar een teller die
    de gebruiker bijhield.

    Doortellen boven het hoogste nummer, NIET het laagste gat vullen. Een gat
    betekent meestal dat daar een set is weggegooid; hergebruik je dat nummer,
    dan verwijst "set 2" in je aantekeningen van vorige week naar ander
    materiaal dan "set 2" van vandaag. Nummers moeten één ding blijven
    aanwijzen."""
    monkeypatch.setattr(runner, "OUT_DIR", tmp_path)
    for n in (0, 1, 3):
        (tmp_path / f"consensus_luna_set{n}.json").write_text("{}", encoding="utf-8")

    assert runner.vrije_sets("luna", 2) == [4, 5]
    assert runner.vrije_sets("luna", 1) == [4]


def test_vrije_sets_kijkt_per_configuratie(tmp_path, monkeypatch):
    """luna en gpt54 hebben hun eigen nummerreeks; een bezette luna-set mag geen
    gpt54-nummer blokkeren."""
    monkeypatch.setattr(runner, "OUT_DIR", tmp_path)
    (tmp_path / "consensus_luna_set0.json").write_text("{}", encoding="utf-8")

    assert runner.vrije_sets("gpt54", 1) == [0]


def test_auto_kiest_vrije_nummers_in_plaats_van_te_weigeren(tmp_path, monkeypatch):
    """Met SET = "auto" hoort een tweede klik naast de vorige ronde te landen in
    plaats van op de klep te stuiten."""
    monkeypatch.setattr(runner, "OUT_DIR", tmp_path)
    (tmp_path / "consensus_luna_set0.json").write_text("{}", encoding="utf-8")
    gebruikt = []
    monkeypatch.setattr(runner, "verzamelen",
                        lambda c, n, **kw: gebruikt.append(n))
    monkeypatch.setattr(runner, "analyse", lambda c, n: None)
    monkeypatch.setattr(runner, "vergelijk", lambda c, a, b: None)
    monkeypatch.setattr(runner, "codeboek", lambda c, n, s, **kw: None)

    runner.alles(ConsensusConfig(), "auto", "auto")

    assert gebruikt == [1, 2]


def test_een_lezende_actie_weigert_auto():
    """"auto" betekent "kies vrije nummers om NAAR te schrijven". Een set die nog
    niet bestaat valt niet te analyseren, dus daar is het woord betekenisloos —
    en één woord met twee betekenissen is precies wat later bijt."""
    with pytest.raises(SystemExit, match="auto"):
        runner._eis_bestaand_setnummer("auto", "analyse")


class _NepPrinter:
    """Telt captures per soort, zoals PromptPrinter dat doet."""
    def __init__(self, **kwargs):
        self.prompts = []

    def capture_prompt(self, **kwargs):
        self.prompts.append(kwargs.get("prompt_type", "?"))


def test_een_ronde_deelt_een_printer_en_schrijft_hem_een_keer_weg(monkeypatch):
    """`save_prompts_to_json` opent in 'w' zonder merge, dus drie schrijvers in
    één ronde betekent dat alleen de laatste overleeft. Op de echte run van
    2026-08-22 bleven zo 0 van de 60 consolidatieprompts over."""
    bewaard = []
    monkeypatch.setattr(runner, "verzamelen",
                        lambda c, n, prompt_printer=None, **kw:
                            prompt_printer.capture_prompt(prompt_type="consolidation"))
    monkeypatch.setattr(runner, "analyse", lambda c, n: None)
    monkeypatch.setattr(runner, "vergelijk", lambda c, a, b: None)
    monkeypatch.setattr(runner, "codeboek",
                        lambda c, n, s, prompt_printer=None, **kw:
                            prompt_printer.capture_prompt(prompt_type="codebook_writer"))
    monkeypatch.setattr(runner, "PromptPrinter", _NepPrinter)
    monkeypatch.setattr(runner, "save_prompts_to_json",
                        lambda printer, doctype=None: bewaard.append(list(printer.prompts)))

    runner.alles(ConsensusConfig(), 90, 91)

    assert len(bewaard) == 1, "één ronde hoort één keer weg te schrijven"
    assert bewaard[0] == ["consolidation", "consolidation", "codebook_writer"]


def test_een_ronde_boekt_alle_consolidatiecalls_op_een_post(monkeypatch):
    """De kostensleutel is (stap, fase) en `record_phase` WIJST TOE. Twee
    `verzamelen`-aanroepen op dezelfde fasenaam betekent dus dat de eerste
    verdwijnt — gemeten op de echte run: 30 calls geboekt waar er 60 waren."""
    geboekt = []

    class _NepTracker:
        def __init__(self, **kwargs):
            pass

        def record_phase(self, stap, fase, voor, na, model=None):
            geboekt.append((stap, fase))

        def finalize_step(self, stap):
            pass

    monkeypatch.setattr(runner, "CostTracker", _NepTracker)
    monkeypatch.setattr(runner, "PromptPrinter", _NepPrinter)
    monkeypatch.setattr(runner, "save_prompts_to_json", lambda p, doctype=None: None)
    monkeypatch.setattr(runner, "verzamelen", lambda c, n, **kw: None)
    monkeypatch.setattr(runner, "analyse", lambda c, n: None)
    monkeypatch.setattr(runner, "vergelijk", lambda c, a, b: None)
    monkeypatch.setattr(runner, "codeboek", lambda c, n, s, **kw: None)

    runner.alles(ConsensusConfig(), 90, 91)

    fasen = [f for _, f in geboekt]
    assert fasen.count("consolidation") == 1, (
        "twee keer boeken op dezelfde fasenaam overschrijft de eerste")
