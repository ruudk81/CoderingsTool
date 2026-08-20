"""Tests for the writing dispatch (codebook_writer.py) — step 4 of step 5."""
import asyncio

from utils.smoothRequester import SmoothRequester

from pipeline.step_5_codeGenerator import codebook_writer
from pipeline.step_5_codeGenerator.concept_inventory import Concept
from pipeline.step_5_codeGenerator.config_codeGenerator import CodebookConfig
from pipeline.step_5_codeGenerator.code_shape import CodeShape
from models import ConsolidatedCode
from pipeline.step_5_codeGenerator.prompts_writer import CodeText, WriterResult


def concept(attribute_id, name, n_resp=10):
    resp = frozenset(f"R{i}" for i in range(n_resp))
    return Concept(attribute_id=attribute_id, name=name, definition="def",
                   domain="Domein", facet="Facet", n_iu=n_resp,
                   resp_ids=resp, resp_pos=resp,
                   resp_neg=frozenset(), resp_neu=frozenset())


def shape(key, valence, umbrella, members, n_resp=40, origin="solo"):
    resp = frozenset(f"R{i}" for i in range(n_resp))
    return CodeShape(key=key, members=tuple(members), valence=valence,
                     umbrella=umbrella, resp_ids=resp, resp_pos=resp,
                     resp_neg=frozenset(), resp_neu=frozenset(), origin=origin)


def text(key, name="Naam", nameable=True):
    return CodeText(key=key, code_name=name, definition="d", diagnostic_test="t",
                    typical_indicators=["a"], boundary_note="b", nameable=nameable)


def code(name, valence="neutral"):
    return ConsolidatedCode(code_name=name, definition="d", diagnostic_test="t",
                            valence=valence, typical_indicators=["a"])


class FakeLog:
    def __init__(self):
        self.calls = []

    def add(self, **kwargs):
        self.calls.append(kwargs)


def test_sends_a_one_element_list_of_dicts(monkeypatch):
    captured = {}

    async def fake_process_all(self, tasks, prepare_fn, parse_fn, fallback_fn=None):
        captured["tasks"] = tasks
        call_params = prepare_fn(tasks[0])
        assert "prompt" in call_params
        assert "response_model" in call_params
        return [WriterResult(codes=[text("K1")])]

    monkeypatch.setattr(SmoothRequester, "process_all", fake_process_all)

    shapes = [shape("K1", "positive", "prijs", ["A1"])]
    concepts = [concept("A1", "Prijs")]
    codes = asyncio.run(
        codebook_writer.write_codebook(shapes, concepts, "stem", "nl-NL", CodebookConfig())
    )

    tasks = captured["tasks"]
    assert isinstance(tasks, list)
    assert len(tasks) == 1
    assert isinstance(tasks[0], dict)
    assert codes[0].code_name == "Naam"


def test_shape_count_and_valence_are_preserved_not_asked(monkeypatch):
    async def fake_process_all(self, tasks, prepare_fn, parse_fn, fallback_fn=None):
        return [WriterResult(codes=[text("K1"), text("K2")])]

    monkeypatch.setattr(SmoothRequester, "process_all", fake_process_all)

    shapes = [shape("K1", "positive", "prijs", ["A1"]),
              shape("K2", "negative", "prijs", ["A1"])]
    concepts = [concept("A1", "Prijs")]
    codes = asyncio.run(
        codebook_writer.write_codebook(shapes, concepts, "stem", "nl-NL", CodebookConfig())
    )
    assert len(codes) == 2
    assert sorted(c.valence for c in codes) == ["negative", "positive"]


def test_non_negative_valence_is_stored_as_neutral(monkeypatch):
    """`build_shapes(two_pole=True)` produces `non_negative`, a value
    `ConsolidatedCode.valence` does not accept — `_to_consolidated_code`
    translates it to `neutral`. The other three values must pass through
    unchanged."""
    async def fake_process_all(self, tasks, prepare_fn, parse_fn, fallback_fn=None):
        return [WriterResult(codes=[text("K1"), text("K2"), text("K3"), text("K4")])]

    monkeypatch.setattr(SmoothRequester, "process_all", fake_process_all)

    shapes = [shape("K1", "non_negative", "prijs", ["A1"]),
              shape("K2", "neutral", "prijs", ["A1"]),
              shape("K3", "positive", "prijs", ["A1"]),
              shape("K4", "negative", "prijs", ["A1"])]
    concepts = [concept("A1", "Prijs")]
    codes = asyncio.run(
        codebook_writer.write_codebook(shapes, concepts, "stem", "nl-NL", CodebookConfig())
    )
    assert codes[0].valence == "neutral"  # K1: non_negative -> neutral
    assert codes[1].valence == "neutral"  # K2: neutral -> neutral (unchanged)
    assert codes[2].valence == "positive"  # K3: unchanged
    assert codes[3].valence == "negative"  # K4: unchanged


def test_source_attributes_are_names_filled_in_code_not_by_the_model(monkeypatch):
    async def fake_process_all(self, tasks, prepare_fn, parse_fn, fallback_fn=None):
        return [WriterResult(codes=[text("K1")])]

    monkeypatch.setattr(SmoothRequester, "process_all", fake_process_all)

    shapes = [shape("K1", "positive", "prijs", ["A1", "A2"], origin="synonym")]
    concepts = [concept("A1", "Prijs"), concept("A2", "Kosten")]
    codes = asyncio.run(
        codebook_writer.write_codebook(shapes, concepts, "stem", "nl-NL", CodebookConfig())
    )
    assert sorted(codes[0].source_attributes) == ["Kosten", "Prijs"]


def test_a_veto_on_a_pooled_shape_drops_it_and_logs(monkeypatch):
    async def fake_process_all(self, tasks, prepare_fn, parse_fn, fallback_fn=None):
        return [WriterResult(codes=[text("K1", nameable=False)])]

    monkeypatch.setattr(SmoothRequester, "process_all", fake_process_all)

    shapes = [shape("K1", "neutral", "u", ["A1", "A2"], origin="pooled")]
    concepts = [concept("A1", "Prijs"), concept("A2", "Kosten")]
    log = FakeLog()
    codes = asyncio.run(
        codebook_writer.write_codebook(shapes, concepts, "stem", "nl-NL", CodebookConfig(), log=log)
    )
    assert codes == []
    assert len(log.calls) == 1
    assert log.calls[0]["action"] == "VETO"
    assert sorted(log.calls[0]["members"]) == ["A1", "A2"]


def test_a_veto_on_a_solo_shape_is_ignored(monkeypatch):
    async def fake_process_all(self, tasks, prepare_fn, parse_fn, fallback_fn=None):
        return [WriterResult(codes=[text("K1", nameable=False)])]

    monkeypatch.setattr(SmoothRequester, "process_all", fake_process_all)

    shapes = [shape("K1", "positive", "prijs", ["A1"], origin="solo")]
    concepts = [concept("A1", "Prijs")]
    codes = asyncio.run(
        codebook_writer.write_codebook(shapes, concepts, "stem", "nl-NL", CodebookConfig())
    )
    assert len(codes) == 1
    assert codes[0].code_name == "Naam"


def test_a_veto_on_a_synonym_shape_is_ignored(monkeypatch):
    async def fake_process_all(self, tasks, prepare_fn, parse_fn, fallback_fn=None):
        return [WriterResult(codes=[text("K1", nameable=False)])]

    monkeypatch.setattr(SmoothRequester, "process_all", fake_process_all)

    shapes = [shape("K1", "positive", "prijs", ["A1", "A2"], origin="synonym")]
    concepts = [concept("A1", "Prijs"), concept("A2", "Kosten")]
    codes = asyncio.run(
        codebook_writer.write_codebook(shapes, concepts, "stem", "nl-NL", CodebookConfig())
    )
    assert len(codes) == 1


def test_a_missing_shape_in_the_response_still_gets_a_code(monkeypatch):
    # The model only wrote text for K1; K2 must still surface as a code rather
    # than silently disappearing — this is the "shapes remain valid" contract.
    async def fake_process_all(self, tasks, prepare_fn, parse_fn, fallback_fn=None):
        return [WriterResult(codes=[text("K1")])]

    monkeypatch.setattr(SmoothRequester, "process_all", fake_process_all)

    shapes = [shape("K1", "positive", "prijs", ["A1"]),
              shape("K2", "negative", "service", ["A2"])]
    concepts = [concept("A1", "Prijs"), concept("A2", "Service")]
    codes = asyncio.run(
        codebook_writer.write_codebook(shapes, concepts, "stem", "nl-NL", CodebookConfig())
    )
    assert len(codes) == 2
    assert {c.valence for c in codes} == {"positive", "negative"}


def test_a_total_call_failure_still_returns_a_code_per_shape(monkeypatch):
    # "If the call fails entirely the shapes are still valid" — the codebook must
    # not fall just because the writer call failed outright.
    async def fake_process_all(self, tasks, prepare_fn, parse_fn, fallback_fn=None):
        return [fallback_fn(tasks[0], "boom")]

    monkeypatch.setattr(SmoothRequester, "process_all", fake_process_all)

    shapes = [shape("K1", "positive", "prijs", ["A1"])]
    concepts = [concept("A1", "Prijs")]
    codes = asyncio.run(
        codebook_writer.write_codebook(shapes, concepts, "stem", "nl-NL", CodebookConfig())
    )
    assert len(codes) == 1
    assert codes[0].valence == "positive"
    assert codes[0].source_attributes == ["Prijs"]


def test_no_shapes_returns_no_codes_without_a_call(monkeypatch):
    called = False

    async def fake_process_all(self, tasks, prepare_fn, parse_fn, fallback_fn=None):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(SmoothRequester, "process_all", fake_process_all)

    codes = asyncio.run(
        codebook_writer.write_codebook([], [], "stem", "nl-NL", CodebookConfig())
    )
    assert codes == []
    assert called is False


def test_taken_names_reach_the_rewrite_prompt(monkeypatch):
    captured = {}

    async def fake_process_all(self, tasks, prepare_fn, parse_fn, fallback_fn=None):
        call_params = prepare_fn(tasks[0])
        captured["prompt"] = call_params["prompt"]
        return [WriterResult(codes=[text("K1")])]

    monkeypatch.setattr(SmoothRequester, "process_all", fake_process_all)

    shapes = [shape("K1", "positive", "prijs", ["A1"])]
    concepts = [concept("A1", "Prijs")]
    asyncio.run(codebook_writer.write_codebook(
        shapes, concepts, "stem", "nl-NL", CodebookConfig(),
        taken_names=["Modern en toekomstgericht"],
    ))
    assert "Modern en toekomstgericht" in captured["prompt"]


# ---------------------------------------------------------------------------
# resolve_duplicate_names — deterministische achtervang op naamsbotsingen
# ---------------------------------------------------------------------------

def test_resolve_duplicate_names_is_a_noop_when_all_names_are_unique():
    codes = [code("Modern en toekomstgericht"), code("Dienstverlening en uitvoering")]
    shapes = [shape("K1", "neutral", "u1", ["A1"], n_resp=85),
              shape("K2", "neutral", "u2", ["A2"], n_resp=74)]
    log = FakeLog()

    resolved = codebook_writer.resolve_duplicate_names(codes, shapes, log=log)

    assert resolved == codes
    assert log.calls == []


def test_resolve_duplicate_names_repairs_a_collision():
    # Same name, different respondent counts and different constituent umbrellas —
    # the larger code keeps the name, the smaller one falls back to its umbrella.
    codes = [code("Modern en toekomstgericht"), code("Modern en toekomstgericht")]
    shapes = [shape("K1", "neutral", "innovatie", ["A1"], n_resp=85),
              shape("K2", "neutral", "vooruitstrevendheid", ["A2"], n_resp=32)]
    log = FakeLog()

    resolved = codebook_writer.resolve_duplicate_names(codes, shapes, log=log)

    names = [c.code_name for c in resolved]
    assert names.count("Modern en toekomstgericht") == 1
    assert names[0] == "Modern en toekomstgericht"
    assert names[1] == "vooruitstrevendheid"

    assert len(log.calls) == 1
    entry = log.calls[0]
    assert entry["action"] == "DUPLICATE_NAME_RESOLVED"
    assert entry["name"] == "Modern en toekomstgericht"
    assert entry["kept_n_resp"] == 85
    assert entry["renamed_to"] == "vooruitstrevendheid"
    assert entry["renamed_n_resp"] == 32


def test_resolve_duplicate_names_appends_a_number_when_the_umbrella_is_also_taken():
    codes = [code("Dienstverlening"), code("Dienstverlening"), code("service")]
    shapes = [shape("K1", "neutral", "u1", ["A1"], n_resp=74),
              shape("K2", "neutral", "service", ["A2"], n_resp=66),
              shape("K3", "neutral", "u3", ["A3"], n_resp=10)]
    log = FakeLog()

    resolved = codebook_writer.resolve_duplicate_names(codes, shapes, log=log)

    names = [c.code_name for c in resolved]
    assert names[0] == "Dienstverlening"
    assert names[1] == "service (2)"
    assert names[2] == "service"
    assert len(names) == len(set(names))


def test_resolve_duplicate_names_rejects_mismatched_list_lengths():
    codes = [code("Naam")]
    shapes = [shape("K1", "neutral", "u", ["A1"]), shape("K2", "neutral", "u", ["A2"])]
    try:
        codebook_writer.resolve_duplicate_names(codes, shapes)
    except ValueError:
        return
    raise AssertionError("een lengteverschil tussen codes en shapes had geweigerd moeten worden")


# ---------------------------------------------------------------------------
# find_naming_mismatches — deterministic backstop against a name that does not
# describe its own contents
# ---------------------------------------------------------------------------

def test_find_naming_mismatches_fires_on_the_real_example():
    # The live-run defect: a code named for communication/visibility whose
    # actual members are entirely a sustainability cluster — no word in the
    # name occurs in any member's name.
    sustainability_members = [
        "Algemene duurzame gerichtheid", "Toezicht op investeringen",
        "Commerciële gerichtheid", "Maatschappelijk-progressieve positionering",
        "Veranderingsgerichtheid", "Duurzaam imago", "Relatieve duurzaamheidspositie",
        "Geloofwaardigheid van duurzaamheid", "Ecologische focus",
        "Concrete ecologische inzet", "Transparantie en openheid",
    ]
    concept_by_id = {f"A{i}": concept(f"A{i}", name, n_resp=40)
                      for i, name in enumerate(sustainability_members)}
    codes = [code("Communicatie en zichtbaarheid", valence="positive")]
    shapes = [shape("K1", "positive", "communicatie", list(concept_by_id), n_resp=477)]

    mismatches = codebook_writer.find_naming_mismatches(codes, shapes, concept_by_id)

    assert len(mismatches) == 1
    assert mismatches[0]["code_name"] == "Communicatie en zichtbaarheid"
    assert mismatches[0]["n_resp"] == 477
    assert sorted(mismatches[0]["members"]) == sorted(sustainability_members)


def test_find_naming_mismatches_stays_silent_on_a_matching_code():
    concept_by_id = {
        "A1": concept("A1", "Heldere communicatie"),
        "A2": concept("A2", "Reclamekanaal"),
        "A3": concept("A3", "Reclame-uiting"),
    }
    codes = [code("Merkcommunicatie en reclame", valence="positive")]
    shapes = [shape("K1", "positive", "communicatie", ["A1", "A2", "A3"], n_resp=66)]

    mismatches = codebook_writer.find_naming_mismatches(codes, shapes, concept_by_id)

    assert mismatches == []


def test_find_naming_mismatches_skips_a_shape_with_no_resolvable_members():
    codes = [code("Naam")]
    shapes = [shape("K1", "neutral", "u", ["A1"])]

    mismatches = codebook_writer.find_naming_mismatches(codes, shapes, {})

    assert mismatches == []


def test_find_naming_mismatches_rejects_mismatched_list_lengths():
    codes = [code("Naam")]
    shapes = [shape("K1", "neutral", "u", ["A1"]), shape("K2", "neutral", "u", ["A2"])]
    try:
        codebook_writer.find_naming_mismatches(codes, shapes, {})
    except ValueError:
        return
    raise AssertionError("een lengteverschil tussen codes en shapes had geweigerd moeten worden")


# ---------------------------------------------------------------------------
# find_duplicate_definitions — deterministische achtervang tegen twee codes
# met dezelfde definitie
# ---------------------------------------------------------------------------

def _code_with_definition(name, definition, valence="neutral"):
    return ConsolidatedCode(code_name=name, definition=definition, diagnostic_test="t",
                            valence=valence, typical_indicators=["a"])


def test_find_duplicate_definitions_fires_on_a_duplicate():
    # The live-run defect: two codes over entirely different members, one
    # showing the other's definition verbatim.
    shared = ("ASN Bank krijgt een positieve waardering wanneer de uitstraling "
              "als natuurlijk, eigentijds, verzorgd, nuchter, alternatief of "
              "rustgevend wordt beleefd.")
    codes = [
        _code_with_definition("Stijl en merkbeleving", shared, valence="positive"),
        _code_with_definition("Merkuitstraling en stijl", shared, valence="positive"),
    ]
    shapes = [shape("K1", "positive", "u1", ["A1"], n_resp=94),
              shape("K2", "positive", "u2", ["A2"], n_resp=33)]

    duplicates = codebook_writer.find_duplicate_definitions(codes, shapes)

    assert len(duplicates) == 1
    names = {c["code_name"]: c["n_resp"] for c in duplicates[0]["codes"]}
    assert names == {"Stijl en merkbeleving": 94, "Merkuitstraling en stijl": 33}


def test_find_duplicate_definitions_catches_whitespace_and_case_only_differences():
    codes = [
        _code_with_definition("Code A", "ASN Bank is betrouwbaar."),
        _code_with_definition("Code B", "  asn bank  is   betrouwbaar.  "),
    ]
    shapes = [shape("K1", "neutral", "u1", ["A1"]), shape("K2", "neutral", "u2", ["A2"])]

    duplicates = codebook_writer.find_duplicate_definitions(codes, shapes)

    assert len(duplicates) == 1


def test_find_duplicate_definitions_stays_silent_when_all_definitions_differ():
    codes = [
        _code_with_definition("Code A", "ASN Bank is betrouwbaar."),
        _code_with_definition("Code B", "ASN Bank is duurzaam."),
        _code_with_definition("Code C", "ASN Bank is vriendelijk."),
    ]
    shapes = [shape("K1", "neutral", "u1", ["A1"]),
              shape("K2", "neutral", "u2", ["A2"]),
              shape("K3", "neutral", "u3", ["A3"])]

    assert codebook_writer.find_duplicate_definitions(codes, shapes) == []


def test_find_duplicate_definitions_rejects_mismatched_list_lengths():
    codes = [code("Naam")]
    shapes = [shape("K1", "neutral", "u", ["A1"]), shape("K2", "neutral", "u", ["A2"])]
    try:
        codebook_writer.find_duplicate_definitions(codes, shapes)
    except ValueError:
        return
    raise AssertionError("een lengteverschil tussen codes en shapes had geweigerd moeten worden")
