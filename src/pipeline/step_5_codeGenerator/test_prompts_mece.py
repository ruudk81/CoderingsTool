"""Tests voor de MECE-prompts (stap 5 van step 5): Pass A (detectie) en
Pass B (blinde toewijzingsproef)."""
from pipeline.step_5_codeGenerator.consolidator import CodeShape
from pipeline.step_5_codeGenerator.prompts_mece import (
    CandidatePair, CodeCandidate, ProbeIdea, build_overlap_prompt, build_probe_prompt,
    make_overlap_model, make_probe_model,
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
# Pass B — blinde toewijzingsproef
# ---------------------------------------------------------------------------

def test_probe_idea_carries_only_a_ref_and_text():
    import dataclasses
    names = {f.name for f in dataclasses.fields(ProbeIdea)}
    assert names == {"idea_ref", "text"}


def test_probe_prompt_shows_both_code_names_and_definitions():
    candidate_by_name = {"Prijs": candidate("Prijs", definition="Over prijs."),
                         "Kosten": candidate("Kosten", definition="Over kosten.")}
    pair = CandidatePair(pair_id=1, code_a="Prijs", code_b="Kosten")
    prompt = build_probe_prompt(pair, candidate_by_name, [ProbeIdea(idea_ref=1, text="tekst")])
    assert "Prijs" in prompt and "Over prijs." in prompt
    assert "Kosten" in prompt and "Over kosten." in prompt


def test_probe_prompt_shows_every_idea_with_its_ref():
    candidate_by_name = {"Prijs": candidate("Prijs"), "Kosten": candidate("Kosten")}
    pair = CandidatePair(pair_id=1, code_a="Prijs", code_b="Kosten")
    ideas = [ProbeIdea(idea_ref=1, text="tekst een"), ProbeIdea(idea_ref=2, text="tekst twee")]
    prompt = build_probe_prompt(pair, candidate_by_name, ideas)
    assert "[1]" in prompt and "tekst een" in prompt
    assert "[2]" in prompt and "tekst twee" in prompt


def test_probe_prompt_does_not_say_which_code_an_idea_came_from():
    candidate_by_name = {"Prijs": candidate("Prijs"), "Kosten": candidate("Kosten")}
    pair = CandidatePair(pair_id=1, code_a="Prijs", code_b="Kosten")
    ideas = [ProbeIdea(idea_ref=1, text="tekst een"), ProbeIdea(idea_ref=2, text="tekst twee")]
    prompt = build_probe_prompt(pair, candidate_by_name, ideas)
    line1 = next(l for l in prompt.splitlines() if "tekst een" in l)
    line2 = next(l for l in prompt.splitlines() if "tekst twee" in l)
    assert "Prijs" not in line1 and "Kosten" not in line1
    assert "Prijs" not in line2 and "Kosten" not in line2


def test_probe_prompt_contains_no_respondent_counts():
    candidate_by_name = {"Prijs": candidate("Prijs", n_resp=312),
                         "Kosten": candidate("Kosten", n_resp=8)}
    pair = CandidatePair(pair_id=1, code_a="Prijs", code_b="Kosten")
    prompt = build_probe_prompt(pair, candidate_by_name, [ProbeIdea(idea_ref=1, text="tekst")])
    assert "312" not in prompt and "8" not in prompt


def test_probe_prompt_contains_no_attribute_ids():
    candidate_by_name = {"Prijs": candidate("Prijs", members=("A17",)),
                         "Kosten": candidate("Kosten", members=("A42",))}
    pair = CandidatePair(pair_id=1, code_a="Prijs", code_b="Kosten")
    prompt = build_probe_prompt(pair, candidate_by_name, [ProbeIdea(idea_ref=1, text="tekst")])
    assert "A17" not in prompt and "A42" not in prompt


def test_probe_prompt_ends_with_the_instructor_hint():
    candidate_by_name = {"Prijs": candidate("Prijs"), "Kosten": candidate("Kosten")}
    pair = CandidatePair(pair_id=1, code_a="Prijs", code_b="Kosten")
    prompt = build_probe_prompt(pair, candidate_by_name, [ProbeIdea(idea_ref=1, text="tekst")])
    assert prompt.rstrip().endswith(
        "provide your output as valid JSON following the response schema provided"
    )


def test_probe_model_constrains_idea_ref_to_shown_ideas():
    pair = CandidatePair(pair_id=1, code_a="Prijs", code_b="Kosten")
    ideas = [ProbeIdea(idea_ref=1, text="a"), ProbeIdea(idea_ref=2, text="b")]
    model = make_probe_model(pair, ideas)
    ok = model(assignments=[{"idea_ref": 1, "assigned_to": "Prijs"},
                             {"idea_ref": 2, "assigned_to": "Kosten"}])
    assert ok.assignments[0].idea_ref == 1

    import pydantic
    try:
        model(assignments=[{"idea_ref": 99, "assigned_to": "Prijs"}])
    except pydantic.ValidationError:
        return
    raise AssertionError("een niet-getoond idea_ref had geweigerd moeten worden")


def test_probe_model_constrains_assigned_to_to_the_pairs_two_codes():
    pair = CandidatePair(pair_id=1, code_a="Prijs", code_b="Kosten")
    ideas = [ProbeIdea(idea_ref=1, text="a")]
    model = make_probe_model(pair, ideas)
    ok = model(assignments=[{"idea_ref": 1, "assigned_to": "Kosten"}])
    assert ok.assignments[0].assigned_to == "Kosten"

    import pydantic
    try:
        model(assignments=[{"idea_ref": 1, "assigned_to": "Service"}])
    except pydantic.ValidationError:
        return
    raise AssertionError("een codenaam buiten dit paar had geweigerd moeten worden")
