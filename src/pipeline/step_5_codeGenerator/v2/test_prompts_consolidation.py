"""Tests voor de consolidatieprompt en zijn responsemodel."""
import pytest
from pydantic import ValidationError

from pipeline.step_5_codeGenerator.v2.attribute_cards import AttributeCard
from pipeline.step_5_codeGenerator.v2.prompts_consolidation import (
    build_consolidation_prompt, make_consolidation_model,
)


def card(attribute_id, name, n_resp=10, answers=(("iets", 3),)):
    return AttributeCard(attribute_id=attribute_id, name=name,
                         definition=f"def van {name}", domain="D", facet="F",
                         n_resp=n_resp, top_answers=answers)


CARDS = [card("A1", "Duurzaamheid", 400, (("groen", 135), ("duurzaam", 107))),
         card("A2", "Sparen", 15),
         card("A3", "Spaarproducten", 15)]


def test_prompt_shows_question_count_answers_and_address():
    prompt = build_consolidation_prompt(CARDS, "Wat vind je van X?", 1092, "Dutch")

    assert "Wat vind je van X?" in prompt
    assert "1092" in prompt
    assert "[A1] Duurzaamheid" in prompt
    assert "400 respondents" in prompt
    assert "groen (135)" in prompt
    assert "D > F" in prompt


def test_prompt_ends_with_instructor_hint():
    prompt = build_consolidation_prompt(CARDS, "V?", 100, "Dutch")

    assert prompt.rstrip().endswith(
        "provide your output as valid JSON following the response schema provided")


def test_prompt_names_no_target_number_of_codes():
    """Een aantal is een symptoom, nooit een doelvariabele — noem je er een,
    dan krijg je dat aantal en weet je niets."""
    prompt = build_consolidation_prompt(CARDS, "V?", 100, "Dutch").lower()

    for forbidden in ("about 20", "roughly", "approximately", "aim for", "target"):
        assert forbidden not in prompt


def test_prompt_does_not_mention_valence():
    """Richting wordt deterministisch afgeleid; twee onbekenden tegelijk maken
    een slecht resultaat onleesbaar."""
    prompt = build_consolidation_prompt(CARDS, "V?", 100, "Dutch").lower()

    for forbidden in ("valence", "positive", "negative", "direction"):
        assert forbidden not in prompt


def test_prompt_order_is_independent_of_input_order():
    """Volgorde mag geen signaal over prevalentie of domein dragen."""
    forward = build_consolidation_prompt(CARDS, "V?", 100, "Dutch")
    backward = build_consolidation_prompt(list(reversed(CARDS)), "V?", 100, "Dutch")

    assert forward == backward


def test_model_accepts_only_offered_tags():
    model = make_consolidation_model(CARDS)

    ok = model(codes=[{"code_name": "Sparen", "explanation": "e",
                       "topics": ["[A2] Sparen", "[A3] Spaarproducten"]}])
    assert len(ok.codes) == 1

    with pytest.raises(ValidationError):
        model(codes=[{"code_name": "Verzonnen", "explanation": "e",
                      "topics": ["[A9] Bestaat niet"]}])
