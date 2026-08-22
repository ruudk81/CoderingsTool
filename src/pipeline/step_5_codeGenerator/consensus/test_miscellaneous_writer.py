"""Tests voor de tweede schrijfcall en voor het vetorecht dat erdoor verschuift.

Staat NIET in `test_codebook_writer.py`: dat bestand is een byte-identieke kopie
van productie's versie (bewaakt door `test_zelfstandigheid.py`), en deze twee
onderwerpen — `write_miscellaneous` en de herkomst `recovered` — bestaan alleen
in deze keten.
"""
import asyncio

from utils.smoothRequester import SmoothRequester

from pipeline.step_5_codeGenerator.consensus import codebook_writer
from pipeline.step_5_codeGenerator.consensus.code_shape import CodeShape
from pipeline.step_5_codeGenerator.consensus.concept_inventory import Concept
from pipeline.step_5_codeGenerator.consensus.config_codeGenerator import CodebookConfig
from pipeline.step_5_codeGenerator.consensus.prompts_miscellaneous import (
    MiscellaneousResult, MiscellaneousText,
)
from pipeline.step_5_codeGenerator.consensus.prompts_writer import CodeText, WriterResult


def concept_voor(attribute_id, naam, n_resp=6):
    resp = frozenset(f"R{i}" for i in range(n_resp))
    return Concept(attribute_id=attribute_id, name=naam, definition="def",
                   domain="Domein", facet="Facet", n_iu=n_resp,
                   resp_ids=resp, resp_pos=frozenset(), resp_neg=resp,
                   resp_neu=frozenset())


def vorm(key, members, origin, valence="negative", umbrella="Facet"):
    resp = frozenset(f"R{i}" for i in range(6))
    return CodeShape(key=key, members=tuple(members), valence=valence,
                     umbrella=umbrella, resp_ids=resp, resp_pos=frozenset(),
                     resp_neg=resp, resp_neu=frozenset(), origin=origin)


def kindtekst(key, naam="Restnaam"):
    return MiscellaneousText(key=key, code_name=naam, definition="d",
                             diagnostic_test="t", typical_indicators=["a"],
                             boundary_note="b")


class VangLog:
    def __init__(self):
        self.calls = []

    def add(self, **kwargs):
        self.calls.append(kwargs)


# ---------------------------------------------------------------------------
# write_miscellaneous
# ---------------------------------------------------------------------------

def test_elke_kindvorm_krijgt_een_code(monkeypatch):
    async def fake_process_all(self, tasks, prepare_fn, parse_fn, fallback_fn=None):
        call_params = prepare_fn(tasks[0])
        assert "prompt" in call_params
        assert "response_model" in call_params
        return [MiscellaneousResult(codes=[kindtekst("V1"), kindtekst("V2", "Tweede")])]

    monkeypatch.setattr(SmoothRequester, "process_all", fake_process_all)

    shapes = [vorm("V1", ["A1"], "child"), vorm("V2", ["A2"], "child")]
    concepts = [concept_voor("A1", "Wachttijd"), concept_voor("A2", "Openingstijden")]
    codes = asyncio.run(codebook_writer.write_miscellaneous(
        shapes, concepts, "stem", "nl-NL", CodebookConfig()))

    assert [c.code_name for c in codes] == ["Restnaam", "Tweede"]


def test_de_bronattributen_worden_door_de_code_gevuld_niet_door_het_model(monkeypatch):
    async def fake_process_all(self, tasks, prepare_fn, parse_fn, fallback_fn=None):
        return [MiscellaneousResult(codes=[kindtekst("V1")])]

    monkeypatch.setattr(SmoothRequester, "process_all", fake_process_all)

    shapes = [vorm("V1", ["A1", "A2"], "child")]
    concepts = [concept_voor("A1", "Wachttijd"), concept_voor("A2", "Openingstijden")]
    codes = asyncio.run(codebook_writer.write_miscellaneous(
        shapes, concepts, "stem", "nl-NL", CodebookConfig()))

    assert sorted(codes[0].source_attributes) == ["Openingstijden", "Wachttijd"]
    assert codes[0].valence == "negative"


def test_een_overgeslagen_vorm_valt_terug_op_zijn_onderwerp(monkeypatch):
    """De schrijver mag een sleutel overslaan; het kind verdwijnt dan niet.

    Anders dan bij `write_codebook` is de noodnaam het ONDERWERP en niet de
    naam van het eerste lid: een kind is per constructie een restcategorie van
    één facet, en de naam van één lid zou dat facet claimen.
    """
    async def fake_process_all(self, tasks, prepare_fn, parse_fn, fallback_fn=None):
        return [MiscellaneousResult(codes=[])]

    monkeypatch.setattr(SmoothRequester, "process_all", fake_process_all)

    shapes = [vorm("V1", ["A1", "A2"], "child", umbrella="Bereikbaarheid")]
    concepts = [concept_voor("A1", "Wachttijd"), concept_voor("A2", "Openingstijden")]
    codes = asyncio.run(codebook_writer.write_miscellaneous(
        shapes, concepts, "stem", "nl-NL", CodebookConfig()))

    assert len(codes) == 1
    assert codes[0].code_name == "Bereikbaarheid"


def test_een_lege_lijst_kost_geen_call(monkeypatch):
    def boem(*args, **kwargs):
        raise AssertionError("er is geen kind, dus er hoort geen call te zijn")

    monkeypatch.setattr(SmoothRequester, "process_all", boem)

    assert asyncio.run(codebook_writer.write_miscellaneous(
        [], [], "stem", "nl-NL", CodebookConfig())) == []


def test_de_fasesleutel_is_die_van_de_kandidaatketen(monkeypatch):
    """Een eigen fase, want de perf-ring is per (model, fase): de kinderprompt
    is korter dan de schrijverprompt en zou anders diens warmtestart vervuilen."""
    gezien = {}

    async def fake_process_all(self, tasks, prepare_fn, parse_fn, fallback_fn=None):
        gezien["phase_key"] = self.phase_key
        return [MiscellaneousResult(codes=[kindtekst("V1")])]

    monkeypatch.setattr(SmoothRequester, "process_all", fake_process_all)

    asyncio.run(codebook_writer.write_miscellaneous(
        [vorm("V1", ["A1"], "child")], [concept_voor("A1", "Wachttijd")],
        "stem", "nl-NL", CodebookConfig()))

    assert gezien["phase_key"] == "step5c_miscellaneous"


def test_een_mislukte_call_levert_alsnog_elk_kind_op(monkeypatch):
    async def fake_process_all(self, tasks, prepare_fn, parse_fn, fallback_fn=None):
        return [None]

    monkeypatch.setattr(SmoothRequester, "process_all", fake_process_all)

    codes = asyncio.run(codebook_writer.write_miscellaneous(
        [vorm("V1", ["A1"], "child", umbrella="Bereikbaarheid")],
        [concept_voor("A1", "Wachttijd")], "stem", "nl-NL", CodebookConfig()))

    assert [c.code_name for c in codes] == ["Bereikbaarheid"]


# ---------------------------------------------------------------------------
# Het veto en de herstelde hoofdcode
# ---------------------------------------------------------------------------

def test_een_herstelde_hoofdcode_is_niet_vetobaar(monkeypatch):
    """Bevinding uit taak 2's review: een unie uit `pool_minority_poles` die de
    drempel haalde kreeg `origin="pooled"` en was daarmee vetobaar. Bij veto
    stonden zijn respondenten wéér nergens — precies het defect dat dit plan
    opheft."""
    async def fake_process_all(self, tasks, prepare_fn, parse_fn, fallback_fn=None):
        return [WriterResult(codes=[CodeText(
            key="V1", code_name="Naam", definition="d", diagnostic_test="t",
            typical_indicators=["a"], boundary_note="b", nameable=False)])]

    monkeypatch.setattr(SmoothRequester, "process_all", fake_process_all)

    log = VangLog()
    codes = asyncio.run(codebook_writer.write_codebook(
        [vorm("V1", ["A1", "A2"], "recovered")],
        [concept_voor("A1", "Wachttijd"), concept_voor("A2", "Openingstijden")],
        "stem", "nl-NL", CodebookConfig(), log=log))

    assert len(codes) == 1
    assert log.calls == []


def test_een_gewone_gepoolde_vorm_blijft_wel_vetobaar(monkeypatch):
    """De tegenhanger van de test hierboven: het veto is niet uitgezet, alleen
    beperkt tot wat het beoordeelt — een door het model voorgestelde
    samenvoeging."""
    async def fake_process_all(self, tasks, prepare_fn, parse_fn, fallback_fn=None):
        return [WriterResult(codes=[CodeText(
            key="V1", code_name="Naam", definition="d", diagnostic_test="t",
            typical_indicators=["a"], boundary_note="b", nameable=False)])]

    monkeypatch.setattr(SmoothRequester, "process_all", fake_process_all)

    log = VangLog()
    codes = asyncio.run(codebook_writer.write_codebook(
        [vorm("V1", ["A1", "A2"], "pooled")],
        [concept_voor("A1", "Wachttijd"), concept_voor("A2", "Openingstijden")],
        "stem", "nl-NL", CodebookConfig(), log=log))

    assert codes == []
    assert [c["action"] for c in log.calls] == ["VETO"]
