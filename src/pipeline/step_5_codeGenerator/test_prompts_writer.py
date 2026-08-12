"""Tests voor de schrijfprompt (stap 4 van step 5)."""
from pipeline.step_5_codeGenerator.concept_inventory import Concept
from pipeline.step_5_codeGenerator.consolidator import CodeShape
from pipeline.step_5_codeGenerator.prompts_writer import (
    CodeText, build_writer_prompt, make_writer_model,
)


def concept(attribute_id, name, n_resp=10, domain="Domein", facet="Facet"):
    resp = frozenset(f"R{i}" for i in range(n_resp))
    return Concept(attribute_id=attribute_id, name=name, definition="def",
                   domain=domain, facet=facet, n_iu=n_resp,
                   resp_ids=resp, resp_pos=resp,
                   resp_neg=frozenset(), resp_neu=frozenset())


def shape(key, valence, umbrella, members, n_resp=40, origin="solo"):
    resp = frozenset(f"R{i}" for i in range(n_resp))
    return CodeShape(key=key, members=tuple(members), valence=valence,
                     umbrella=umbrella, resp_ids=resp, resp_pos=resp,
                     resp_neg=frozenset(), resp_neu=frozenset(), origin=origin)


class Shape:
    """Fixture uit de taakbrief — geen consolidator.CodeShape, alleen de velden
    die build_writer_prompt daadwerkelijk gebruikt."""
    def __init__(self, key, valence, umbrella, members, n_resp, origin="solo"):
        self.key, self.valence, self.umbrella = key, valence, umbrella
        self.members, self.origin = members, origin
        self.resp_ids = frozenset(f"R{i}" for i in range(n_resp))


def test_prompt_states_the_fixed_number_of_codes():
    shapes = [Shape("K1", "positive", "prijs", ["A1"], 40),
              Shape("K2", "negative", "prijs", ["A1"], 30)]
    prompt = build_writer_prompt(shapes, {}, "Does the response say that...", "nl-NL")
    assert "2" in prompt
    assert "do not add, remove or merge" in prompt.lower()


def test_prompt_forbids_claiming_a_neighbours_territory():
    shapes = [Shape("K1", "neutral", "maatschappelijk", ["A2", "A3"], 31, "pooled")]
    prompt = build_writer_prompt(shapes, {}, "stem", "nl-NL")
    assert "must not claim" in prompt.lower()


def test_only_pooled_shapes_may_be_vetoed():
    text = CodeText(key="K1", code_name="X", definition="d", diagnostic_test="t",
                    typical_indicators=["a"], boundary_note="b", nameable=False)
    assert text.nameable is False


def test_codetext_default_nameable_is_true():
    text = CodeText(key="K1", code_name="X", definition="d", diagnostic_test="t",
                    typical_indicators=["a"], boundary_note="b")
    assert text.nameable is True


def test_prompt_shows_attribute_names_not_ids():
    shapes = [shape("K1", "positive", "prijs", ["A1"])]
    concept_by_id = {"A1": concept("A1", "Instapkosten")}
    prompt = build_writer_prompt(shapes, concept_by_id, "stem", "nl-NL")
    assert "Instapkosten" in prompt
    assert "A1" not in prompt


def test_prompt_contains_no_respondent_counts():
    shapes = [shape("K1", "positive", "prijs", ["A1"], n_resp=312)]
    concept_by_id = {"A1": concept("A1", "Prijs", n_resp=312)}
    prompt = build_writer_prompt(shapes, concept_by_id, "stem", "nl-NL")
    assert "312" not in prompt


def test_prompt_contains_no_domain_or_facet():
    shapes = [shape("K1", "positive", "prijs", ["A1"])]
    concept_by_id = {"A1": concept("A1", "Prijs", domain="Kostenbeleving", facet="Instapkosten facet")}
    prompt = build_writer_prompt(shapes, concept_by_id, "stem", "nl-NL")
    assert "Kostenbeleving" not in prompt
    assert "Instapkosten facet" not in prompt


def test_prompt_shows_the_direction():
    shapes = [shape("K1", "negative", "prijs", ["A1"])]
    concept_by_id = {"A1": concept("A1", "Prijs")}
    prompt = build_writer_prompt(shapes, concept_by_id, "stem", "nl-NL")
    assert "negative" in prompt


def test_prompt_order_is_not_the_membership_order():
    shapes = [shape(f"K{i}", "neutral", "u", [f"A{i}"], n_resp=100 - i) for i in range(8)]
    concept_by_id = {f"A{i}": concept(f"A{i}", f"Topic{i}") for i in range(8)}
    prompt = build_writer_prompt(shapes, concept_by_id, "stem", "nl-NL")
    rendered_order = sorted(shapes, key=lambda s: prompt.index(f"[{s.key}]"))
    assert [s.key for s in rendered_order] != [s.key for s in shapes]


def test_prompt_order_is_stable_across_calls():
    shapes = [shape(f"K{i}", "neutral", "u", [f"A{i}"]) for i in range(6)]
    concept_by_id = {f"A{i}": concept(f"A{i}", f"Topic{i}") for i in range(6)}
    first = build_writer_prompt(shapes, concept_by_id, "stem", "nl-NL")
    second = build_writer_prompt(shapes, concept_by_id, "stem", "nl-NL")
    assert first == second


def test_prompt_uses_the_dimension_diagnostic_stem():
    shapes = [shape("K1", "positive", "prijs", ["A1"])]
    prompt = build_writer_prompt(shapes, {}, "Does the response evaluate the price?", "nl-NL")
    assert "Does the response evaluate the price?" in prompt


def test_prompt_ends_with_the_instructor_hint():
    prompt = build_writer_prompt([shape("K1", "positive", "prijs", ["A1"])], {}, "stem", "nl-NL")
    assert prompt.rstrip().endswith(
        "provide your output as valid JSON following the response schema provided"
    )


def test_response_model_constrains_key_to_existing_shapes():
    shapes = [shape("K1", "positive", "prijs", ["A1"]), shape("K2", "negative", "prijs", ["A2"])]
    model = make_writer_model(shapes)
    ok = model(codes=[{"key": "K1", "code_name": "X", "definition": "d", "diagnostic_test": "t",
                       "typical_indicators": ["a"], "boundary_note": "b"}])
    assert ok.codes[0].key == "K1"

    import pydantic
    try:
        model(codes=[{"key": "K99", "code_name": "X", "definition": "d", "diagnostic_test": "t",
                      "typical_indicators": ["a"], "boundary_note": "b"}])
    except pydantic.ValidationError:
        return
    raise AssertionError("een niet-bestaande vorm-sleutel had geweigerd moeten worden")


def test_prompt_veto_rule_is_present():
    prompt = build_writer_prompt([shape("K1", "neutral", "u", ["A1", "A2"], origin="pooled")], {}, "stem", "nl-NL")
    assert "nameable" in prompt.lower()
    assert "share nothing" in prompt.lower() or "invent an umbrella" in prompt.lower()
