"""Tests voor de N-runs-dispatch.

De dispatch zelf (het netwerk) wordt hier niet gedraaid — wat te toetsen valt
is dat de N taken correct gebouwd worden en dat het faalcontract klopt.
"""
import asyncio
from pathlib import Path

import pytest

from pipeline.step_5_codeGenerator.consensus.attribute_cards import AttributeCard
from pipeline.step_5_codeGenerator.consensus import consolidation
from pipeline.step_5_codeGenerator.consensus.config_consensus import ConsensusConfig
from pipeline.step_5_codeGenerator.consensus.consolidation import build_tasks
from pipeline.step_5_codeGenerator.consensus.prompts_consolidation import build_consolidation_prompt


def kaart(attribute_id, naam):
    return AttributeCard(attribute_id=attribute_id, name=naam, definition="d",
                         domain="D", facet="F", n_resp=5, top_answers=())


CARDS = [kaart("A1", "Prijs"), kaart("A2", "Service"), kaart("A3", "Levertijd")]


def test_elke_run_krijgt_zijn_eigen_salt_in_de_taak():
    """De salt moet IN de taak zitten, niet in een closure. Zat hij in de
    closure, dan is één functie-aanroep gelijk aan één salt gelijk aan één
    taak — en dan kan er niets parallel."""
    cards = [kaart("A1", "Prijs"), kaart("A2", "Service")]

    taken = build_tasks(cards, ["run0", "run1", "run2"])

    assert [t["salt"] for t in taken] == ["run0", "run1", "run2"]
    assert all(t["cards"] == cards for t in taken)


def test_alle_taken_delen_dezelfde_kaarten():
    """Dertig trekkingen uit dezelfde urn. Verschilt de invoer per run, dan
    meet de co-associatiematrix twee dingen tegelijk."""
    cards = [kaart("A1", "Prijs")]

    taken = build_tasks(cards, ["a", "b"])

    assert taken[0]["cards"] is taken[1]["cards"]


class _FakeRequester:
    """Vervangt SmoothRequester: geeft `canned` terug zonder netwerk, vangt de
    constructor-kwargs op (`num_tasks`, `phase_key`) en roept `prepare_fn` op
    elke taak aan — precies zoals de echte requester dat vóór elke call doet —
    zodat `prepared` per taak het eigen prompt/model draagt in plaats van het
    laatst gebouwde."""
    captured = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _FakeRequester.captured = self

    async def process_all(self, tasks, prepare_fn, parse_fn, fallback_fn=None):
        self.prepared = [prepare_fn(task) for task in tasks]
        return self.canned


def test_een_enkele_geslaagde_run_is_een_harde_stop(monkeypatch):
    """Onder twee runs is er geen paar om te tellen. Stil doorgaan zou een
    'consensus' over één run opleveren — dat is geen consensus maar een run."""
    _FakeRequester.canned = ["ok", None, None]
    monkeypatch.setattr(consolidation, "SmoothRequester", _FakeRequester)

    with pytest.raises(RuntimeError, match="minstens twee"):
        asyncio.run(consolidation.resolve_consolidations(
            CARDS, "vraag", 100, "Dutch", ConsensusConfig(), ["a", "b", "c"]))


def test_mislukte_runs_worden_geteld_en_niet_verzwegen(monkeypatch):
    """Zonder telling leest een latere analyse len(runs) als het aantal
    gevraagde runs, en verschuift elke drempel stil."""
    _FakeRequester.canned = ["ok", "ok", None]
    monkeypatch.setattr(consolidation, "SmoothRequester", _FakeRequester)

    geslaagd, mislukt = asyncio.run(consolidation.resolve_consolidations(
        CARDS, "vraag", 100, "Dutch", ConsensusConfig(), ["a", "b", "c"]))

    assert (len(geslaagd), mislukt) == (2, 1)


def test_alle_runs_gaan_als_een_partij_naar_de_requester(monkeypatch):
    """Dit is de hele winst van deze taak: één requester met N taken, in plaats
    van N requesters met elk één taak."""
    _FakeRequester.canned = ["ok"] * 5
    monkeypatch.setattr(consolidation, "SmoothRequester", _FakeRequester)

    asyncio.run(consolidation.resolve_consolidations(
        CARDS, "vraag", 100, "Dutch", ConsensusConfig(), ["a", "b", "c", "d", "e"]))

    assert _FakeRequester.captured.kwargs["num_tasks"] == 5


def test_elke_taak_krijgt_zijn_eigen_prompt_op_de_eigen_phase_key(monkeypatch):
    """Zat de salt weer in een closure, dan zouden alle taken de laatst
    gebouwde prompt delen — hier wordt élk taak-prompt-paar apart nagerekend
    tegen de directe aanroep met diezelfde salt, dus een omwisseling of een
    stale closure-waarde faalt hier zichtbaar. `phase_key` is de andere helft
    van het contract: zonder eigen sleutel meet het ringbuffer de verkeerde
    call-vorm."""
    salts = ["run-a", "run-b", "run-c"]
    _FakeRequester.canned = ["ok"] * len(salts)
    monkeypatch.setattr(consolidation, "SmoothRequester", _FakeRequester)

    asyncio.run(consolidation.resolve_consolidations(
        CARDS, "Wat vond u van de service?", 100, "Dutch", ConsensusConfig(), salts))

    assert _FakeRequester.captured.kwargs["phase_key"] == "step5c_consolidation"
    for index, salt in enumerate(salts):
        verwacht_prompt = build_consolidation_prompt(
            CARDS, "Wat vond u van de service?", 100, "Dutch", salt)
        assert _FakeRequester.captured.prepared[index]["prompt"] == verwacht_prompt


# De docstring-alinea die de kopie zelf toevoegt aan die van productie — de
# ENIGE tekstuele afwijking naast de eigen relatieve import. Staat hier
# letterlijk, zodat een wijziging aan de alinea in het bronbestand deze test
# laat falen in plaats van hem stilzwijgend te laten slagen op een string die
# niet meer bestaat.
_TOEGEVOEGDE_ALINEA = (
    "\n\nDeze kopie is op 2026-08-21 byte-identiek begonnen aan die van de\n"
    "productieketen. Dat is met opzet en het is een MEETVOORWAARDE: wijkt hij af\n"
    "zonder dat dat een besluit was, dan schrijf je een winst toe aan consensus die\n"
    "van de prompt kwam. Elke afwijking hoort een expliciet besluit te zijn met een\n"
    "tekstuele vóór→na, zoals de promptregel van dit project voorschrijft."
)


def test_prompt_is_byte_identiek_aan_productie_op_import_en_docstring_na():
    """MEETVOORWAARDE, niet stijl: wijkt deze kopie af zonder dat dat een
    besluit was, dan schrijf je een winst toe aan consensus die eigenlijk van
    de prompt kwam. Deze test is het mechanische alternatief voor de `diff` uit
    het uitvoeringsplan, dat na afloop is verwijderd.

    Bij een BEWUSTE promptwijziging: werk deze test bij met een tekstuele
    vóór→na van de wijziging (zie de promptregel in het project-CLAUDE.md).
    Verwijder de test niet — dat is precies het stille-afwijking-risico dat
    hij moet vangen."""
    productie = (Path(__file__).parent.parent / "prompts_consolidation.py").read_text(
        encoding="utf-8")
    kandidaat = (Path(__file__).parent / "prompts_consolidation.py").read_text(
        encoding="utf-8")

    kandidaat = kandidaat.replace(
        "from ..prompts_common import INSTRUCTOR_HINT, _shuffled",
        "from .prompts_common import INSTRUCTOR_HINT, _shuffled",
    )
    assert _TOEGEVOEGDE_ALINEA in kandidaat, (
        "de docstring-alinea die de MEETVOORWAARDE vastlegt is weg of "
        "gewijzigd in consensus/prompts_consolidation.py — werk "
        "_TOEGEVOEGDE_ALINEA in deze test bij met de nieuwe tekst"
    )
    kandidaat = kandidaat.replace(_TOEGEVOEGDE_ALINEA, "")

    assert kandidaat == productie, (
        "consensus/prompts_consolidation.py is afgeweken van "
        "../prompts_consolidation.py buiten de eigen import en docstring-alinea "
        "om. Is dit een bewust besluit: werk deze test bij met een tekstuele "
        "vóór→na van de wijziging (de promptregel in CLAUDE.md verplicht dat) "
        "— verwijder de test niet, anders schrijft de volgende meting een "
        "winst toe aan consensus die van de prompt kwam."
    )
