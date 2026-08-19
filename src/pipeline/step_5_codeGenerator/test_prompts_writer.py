"""Tests voor de writerprompt: richting moet in naam en definitie landen."""
from pipeline.step_5_codeGenerator.concept_inventory import Concept
from pipeline.step_5_codeGenerator.code_shape import CodeShape
from pipeline.step_5_codeGenerator.prompts_writer import build_writer_prompt


def shape(key, valence, members=("A1",)):
    resp = frozenset({"r1", "r2"})
    return CodeShape(key=key, members=members, valence=valence, umbrella="U",
                     resp_ids=resp, resp_pos=resp, resp_neg=frozenset(),
                     resp_neu=frozenset(), origin="solo")


CONCEPTS = {"A1": Concept(attribute_id="A1", name="Kosten", definition="over prijs",
                          domain="D", facet="F", n_iu=2, resp_ids=frozenset({"r1"}),
                          resp_pos=frozenset({"r1"}), resp_neg=frozenset(),
                          resp_neu=frozenset())}


def test_prompt_requires_direction_in_name_and_definition():
    prompt = build_writer_prompt([shape("V1", "negative")], CONCEPTS,
                                    "vooral in ...", "Dutch")

    # Rule 1 specifically requires direction in BOTH name and definition
    assert "must be readable in" in prompt
    assert "BOTH its name and its definition" in prompt
    # the production prompt has three rules (Rule 1 about direction), v1 has two
    assert "Three rules:" in prompt


def test_neutral_codes_are_explicitly_exempt_from_carrying_direction():
    """Richting hoort in de naam 'mits relevant' — een beschrijvende code
    verzint er geen."""
    prompt = build_writer_prompt([shape("V1", "neutral")], CONCEPTS,
                                    "vooral in ...", "Dutch")

    # Rule 1 explicitly exempts neutral codes from invented evaluation
    assert "do not invent an evaluation it does not carry" in prompt


def test_prompt_still_shows_members_and_ends_with_the_hint():
    prompt = build_writer_prompt([shape("V1", "positive")], CONCEPTS,
                                    "vooral in ...", "Dutch")

    assert "Kosten" in prompt
    assert "over prijs" in prompt
    assert prompt.rstrip().endswith(
        "provide your output as valid JSON following the response schema provided")


def test_taken_names_are_passed_through():
    prompt = build_writer_prompt([shape("V1", "positive")], CONCEPTS,
                                    "vooral in ...", "Dutch",
                                    taken_names=["Al vergeven"])

    assert "Al vergeven" in prompt
