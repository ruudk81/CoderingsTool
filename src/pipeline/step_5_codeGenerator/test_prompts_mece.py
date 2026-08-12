"""Tests voor de MECE-prompts (stap 5 van step 5): Pass A (detectie) en
Pass B (adjudicatie)."""
from pipeline.step_5_codeGenerator.consolidator import CodeShape
from pipeline.step_5_codeGenerator.prompts_mece import (
    CandidatePair, CodeCandidate, build_overlap_prompt, build_pair_prompt,
    make_overlap_model, make_pair_model,
)


def shape(key, valence, members, n_resp=40, origin="solo"):
    resp = frozenset(f"R{i}" for i in range(n_resp))
    return CodeShape(key=key, members=tuple(members), valence=valence,
                     umbrella="u", resp_ids=resp, resp_pos=resp,
                     resp_neg=frozenset(), resp_neu=frozenset(), origin=origin)


def candidate(name, valence="positive", definition="def", indicators=("a", "b"),
             members=("A1",), n_resp=40):
    return CodeCandidate(name=name, definition=definition, indicators=tuple(indicators),
                         valence=valence, shape=shape(name, valence, members, n_resp))


# ---------------------------------------------------------------------------
# Pass A — detectie
# ---------------------------------------------------------------------------

def test_overlap_prompt_lists_every_code_with_definition_and_indicators():
    candidates = [candidate("Prijs", definition="Over prijs.", indicators=("duur", "goedkoop")),
                  candidate("Service", definition="Over service.", indicators=("vriendelijk",))]
    prompt = build_overlap_prompt(candidates)
    assert "Prijs" in prompt and "Over prijs." in prompt and "duur" in prompt
    assert "Service" in prompt and "Over service." in prompt and "vriendelijk" in prompt


def test_overlap_prompt_shows_valence():
    candidates = [candidate("Prijs", valence="negative")]
    prompt = build_overlap_prompt(candidates)
    assert "negative" in prompt


def test_overlap_prompt_contains_no_respondent_counts():
    candidates = [candidate("Prijs", n_resp=312)]
    prompt = build_overlap_prompt(candidates)
    assert "312" not in prompt


def test_overlap_prompt_contains_no_attribute_ids():
    candidates = [candidate("Prijs", members=("A17", "A42"))]
    prompt = build_overlap_prompt(candidates)
    assert "A17" not in prompt and "A42" not in prompt


def test_overlap_prompt_order_is_not_the_input_order():
    candidates = [candidate(f"Topic{i}") for i in range(8)]
    prompt = build_overlap_prompt(candidates)
    rendered_order = sorted(candidates, key=lambda c: prompt.index(c.name))
    assert [c.name for c in rendered_order] != [c.name for c in candidates]


def test_overlap_prompt_order_is_stable_across_calls():
    candidates = [candidate(f"Topic{i}") for i in range(6)]
    first = build_overlap_prompt(candidates)
    second = build_overlap_prompt(candidates)
    assert first == second


def test_overlap_prompt_ends_with_the_instructor_hint():
    prompt = build_overlap_prompt([candidate("Prijs")])
    assert prompt.rstrip().endswith(
        "provide your output as valid JSON following the response schema provided"
    )


def test_overlap_model_constrains_to_existing_code_names():
    model = make_overlap_model([candidate("Prijs"), candidate("Service")])
    ok = model(verdicts=[{"code": "Prijs", "hardest_to_separate_from": "Service"}])
    assert ok.verdicts[0].code == "Prijs"

    import pydantic
    try:
        model(verdicts=[{"code": "Verzonnen", "hardest_to_separate_from": None}])
    except pydantic.ValidationError:
        return
    raise AssertionError("een niet-bestaande codenaam had geweigerd moeten worden")


def test_overlap_model_allows_null_hardest_to_separate_from():
    model = make_overlap_model([candidate("Prijs")])
    ok = model(verdicts=[{"code": "Prijs", "hardest_to_separate_from": None}])
    assert ok.verdicts[0].hardest_to_separate_from is None


# ---------------------------------------------------------------------------
# Pass B — adjudicatie
# ---------------------------------------------------------------------------

def test_pair_prompt_shows_both_codes_of_every_pair():
    candidates = [candidate("Prijs", definition="Over prijs."),
                  candidate("Kosten", definition="Over kosten.")]
    candidate_by_name = {c.name: c for c in candidates}
    pairs = [CandidatePair(pair_id=1, code_a="Prijs", code_b="Kosten")]
    prompt = build_pair_prompt(pairs, candidate_by_name)
    assert "Prijs" in prompt and "Over prijs." in prompt
    assert "Kosten" in prompt and "Over kosten." in prompt
    assert "[1]" in prompt


def test_pair_prompt_forces_the_rule_before_the_verdict():
    candidate_by_name = {"Prijs": candidate("Prijs"), "Kosten": candidate("Kosten")}
    pairs = [CandidatePair(pair_id=1, code_a="Prijs", code_b="Kosten")]
    prompt = build_pair_prompt(pairs, candidate_by_name)
    assert "write" in prompt.lower() and "rule" in prompt.lower()


def test_pair_prompt_contains_no_respondent_counts():
    candidate_by_name = {"Prijs": candidate("Prijs", n_resp=312),
                         "Kosten": candidate("Kosten", n_resp=8)}
    pairs = [CandidatePair(pair_id=1, code_a="Prijs", code_b="Kosten")]
    prompt = build_pair_prompt(pairs, candidate_by_name)
    assert "312" not in prompt and "8" not in prompt


def test_pair_prompt_contains_no_attribute_ids():
    candidate_by_name = {"Prijs": candidate("Prijs", members=("A17",)),
                         "Kosten": candidate("Kosten", members=("A42",))}
    pairs = [CandidatePair(pair_id=1, code_a="Prijs", code_b="Kosten")]
    prompt = build_pair_prompt(pairs, candidate_by_name)
    assert "A17" not in prompt and "A42" not in prompt


def test_pair_prompt_order_is_not_the_input_order():
    candidate_by_name = {f"C{i}": candidate(f"C{i}") for i in range(10)}
    pairs = [CandidatePair(pair_id=i, code_a=f"C{i}", code_b=f"C{i+1}") for i in range(9)]
    prompt = build_pair_prompt(pairs, candidate_by_name)
    rendered_order = sorted(pairs, key=lambda p: prompt.index(f"[{p.pair_id}]"))
    assert [p.pair_id for p in rendered_order] != [p.pair_id for p in pairs]


def test_pair_prompt_order_is_stable_across_calls():
    candidate_by_name = {f"C{i}": candidate(f"C{i}") for i in range(6)}
    pairs = [CandidatePair(pair_id=i, code_a=f"C{i}", code_b=f"C{i+1}") for i in range(5)]
    first = build_pair_prompt(pairs, candidate_by_name)
    second = build_pair_prompt(pairs, candidate_by_name)
    assert first == second


def test_pair_prompt_ends_with_the_instructor_hint():
    candidate_by_name = {"Prijs": candidate("Prijs"), "Kosten": candidate("Kosten")}
    pairs = [CandidatePair(pair_id=1, code_a="Prijs", code_b="Kosten")]
    prompt = build_pair_prompt(pairs, candidate_by_name)
    assert prompt.rstrip().endswith(
        "provide your output as valid JSON following the response schema provided"
    )


def test_pair_model_constrains_pair_id_to_existing_pairs():
    pairs = [CandidatePair(pair_id=1, code_a="Prijs", code_b="Kosten"),
             CandidatePair(pair_id=2, code_a="Service", code_b="Klantcontact")]
    model = make_pair_model(pairs)
    ok = model(verdicts=[{"pair_id": 1, "separation_rule": "r", "one_dimension": False}])
    assert ok.verdicts[0].pair_id == 1

    import pydantic
    try:
        model(verdicts=[{"pair_id": 99, "separation_rule": "r", "one_dimension": False}])
    except pydantic.ValidationError:
        return
    raise AssertionError("een niet-bestaand pair_id had geweigerd moeten worden")
