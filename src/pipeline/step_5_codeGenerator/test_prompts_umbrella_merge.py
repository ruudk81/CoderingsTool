"""Tests for cleaning up umbrella names (step 2b of step 5).

A per-item question ("is there another name that means the same?"), not a
grouping question — see the module docstring of prompts_umbrella_merge.py for
why. The canonical name is not chosen by the model but derived deterministically
in code (relations.py); that is tested here via apply_umbrella_merge.
"""
import pydantic
import pytest

from pipeline.step_5_codeGenerator.prompts_umbrella_merge import (
    Umbrella, build_umbrella_merge_prompt, make_umbrella_merge_model,
)
from pipeline.step_5_codeGenerator.relations import apply_umbrella_merge


def umbrella(name, members, definition="def"):
    return Umbrella(name=name, definition=definition, member_names=tuple(members))


UMBRELLAS = [
    umbrella("Bankdiensten", ["Betalen", "Sparen"]),
    umbrella("Bankdiensten en aanbod", ["Hypotheek"]),
    umbrella("Visuele merkidentiteit", ["Logo", "Kleurgebruik", "Beeldmerk"]),
]


def test_prompt_contains_no_member_counts():
    # Form-independent: the fixture's names and member names contain no digit,
    # so ANY digit in the rendered prompt would have to be a count — "2", "(2)",
    # "2 topics", all equally caught. A substring check for " 2 " specifically
    # would miss "(2)".
    prompt = build_umbrella_merge_prompt(UMBRELLAS)
    assert not any(char.isdigit() for char in prompt)
    assert "aantal" not in prompt.lower() and "count" not in prompt.lower()


def test_prompt_lists_every_umbrella_with_its_members():
    prompt = build_umbrella_merge_prompt(UMBRELLAS)
    for u in UMBRELLAS:
        assert u.name in prompt
        for member in u.member_names:
            assert member in prompt


def test_prompt_order_is_not_the_input_order():
    many = [umbrella(f"Naam {i}", [f"Attr {i}"]) for i in range(12)]
    prompt = build_umbrella_merge_prompt(many)
    positions = [prompt.index(u.name) for u in many]
    assert positions != sorted(positions)


def test_prompt_order_is_stable_across_calls():
    a = build_umbrella_merge_prompt(UMBRELLAS)
    b = build_umbrella_merge_prompt(UMBRELLAS)
    assert a == b


def test_prompt_ends_with_the_instructor_hint():
    prompt = build_umbrella_merge_prompt(UMBRELLAS)
    assert prompt.rstrip().endswith(
        "provide your output as valid JSON following the response schema provided"
    )


def test_model_accepts_a_same_as_verdict_between_existing_names():
    model = make_umbrella_merge_model(UMBRELLAS)
    ok = model(scratchpad="", verdicts=[
        {"umbrella": "Bankdiensten", "same_as": "Bankdiensten en aanbod"},
        {"umbrella": "Bankdiensten en aanbod", "same_as": None},
        {"umbrella": "Visuele merkidentiteit", "same_as": None},
    ])
    assert ok.verdicts[0].same_as == "Bankdiensten en aanbod"


def test_model_rejects_a_same_as_target_outside_the_list():
    model = make_umbrella_merge_model(UMBRELLAS)
    with pytest.raises(pydantic.ValidationError):
        model(scratchpad="", verdicts=[
            {"umbrella": "Bankdiensten", "same_as": "Bestaat niet"},
        ])


def test_model_rejects_an_umbrella_outside_the_list():
    model = make_umbrella_merge_model(UMBRELLAS)
    with pytest.raises(pydantic.ValidationError):
        model(scratchpad="", verdicts=[
            {"umbrella": "Bestaat niet", "same_as": None},
        ])


class FakeRelation:
    def __init__(self, attribute, umbrella_name, umbrella_definition="d", synonym_of=None):
        self.attribute = attribute
        self.umbrella_name = umbrella_name
        self.umbrella_definition = umbrella_definition
        self.synonym_of = synonym_of


class FakeRelations:
    def __init__(self, relations):
        self.scratchpad = ""
        self.relations = relations


class FakeVerdict:
    def __init__(self, umbrella, same_as=None):
        self.umbrella = umbrella
        self.same_as = same_as


class FakeMerge:
    def __init__(self, verdicts):
        self.scratchpad = ""
        self.verdicts = verdicts


def test_apply_rewrites_member_umbrellas_to_the_canonical_name():
    relations = FakeRelations([
        FakeRelation("[A1] Betalen", "Bankdiensten"),
        FakeRelation("[A2] Hypotheek", "Bankdiensten en aanbod"),
        FakeRelation("[A3] Logo", "Visuele merkidentiteit"),
    ])
    merged = apply_umbrella_merge(relations, FakeMerge([
        FakeVerdict("Bankdiensten", same_as=None),
        FakeVerdict("Bankdiensten en aanbod", same_as="Bankdiensten"),
        FakeVerdict("Visuele merkidentiteit", same_as=None),
    ]))
    names = [r.umbrella_name for r in merged.relations]
    # Both umbrellas carry one attribute each — tied — so the shorter name wins.
    assert names == ["Bankdiensten", "Bankdiensten", "Visuele merkidentiteit"]


def test_apply_leaves_umbrellas_that_are_in_no_group_untouched():
    relations = FakeRelations([FakeRelation("[A1] Logo", "Visuele merkidentiteit")])
    merged = apply_umbrella_merge(relations, FakeMerge([
        FakeVerdict("Visuele merkidentiteit", same_as=None),
    ]))
    assert merged.relations[0].umbrella_name == "Visuele merkidentiteit"


def test_apply_collapses_a_same_as_chain_into_one_group():
    relations = FakeRelations([
        FakeRelation("[A1] Een", "A"),
        FakeRelation("[A2] Twee", "B"),
        FakeRelation("[A3] Drie", "C"),
    ])
    merged = apply_umbrella_merge(relations, FakeMerge([
        FakeVerdict("A", same_as="B"),
        FakeVerdict("B", same_as="C"),
        FakeVerdict("C", same_as=None),
    ]))
    names = {r.umbrella_name for r in merged.relations}
    assert len(names) == 1


def test_canonical_name_is_the_member_with_the_most_attributes_even_if_longer():
    relations = FakeRelations([
        FakeRelation("[A1] Een", "Langenaam"),
        FakeRelation("[A2] Twee", "Langenaam"),
        FakeRelation("[A3] Drie", "Langenaam"),
        FakeRelation("[A4] Vier", "Kort"),
    ])
    merged = apply_umbrella_merge(relations, FakeMerge([
        FakeVerdict("Langenaam", same_as="Kort"),
        FakeVerdict("Kort", same_as=None),
    ]))
    names = {r.umbrella_name for r in merged.relations}
    assert names == {"Langenaam"}


def test_canonical_name_tie_break_prefers_the_shortest_name():
    relations = FakeRelations([
        FakeRelation("[A1] Een", "Bbb"),
        FakeRelation("[A2] Twee", "Aaaa"),
    ])
    merged = apply_umbrella_merge(relations, FakeMerge([
        FakeVerdict("Bbb", same_as="Aaaa"),
        FakeVerdict("Aaaa", same_as=None),
    ]))
    names = {r.umbrella_name for r in merged.relations}
    assert names == {"Bbb"}


def test_canonical_name_tie_break_falls_back_to_alphabetical():
    relations = FakeRelations([
        FakeRelation("[A1] Een", "Zorg"),
        FakeRelation("[A2] Twee", "Aard"),
    ])
    merged = apply_umbrella_merge(relations, FakeMerge([
        FakeVerdict("Zorg", same_as="Aard"),
        FakeVerdict("Aard", same_as=None),
    ]))
    names = {r.umbrella_name for r in merged.relations}
    assert names == {"Aard"}


def test_apply_does_not_mutate_the_input():
    relations = FakeRelations([
        FakeRelation("[A1] Betalen", "Bankdiensten"),
        FakeRelation("[A2] Sparen", "Diensten"),
        FakeRelation("[A3] Hypotheek", "Diensten"),
    ])
    apply_umbrella_merge(relations, FakeMerge([
        FakeVerdict("Bankdiensten", same_as="Diensten"),
        FakeVerdict("Diensten", same_as=None),
    ]))
    assert relations.relations[0].umbrella_name == "Bankdiensten"
