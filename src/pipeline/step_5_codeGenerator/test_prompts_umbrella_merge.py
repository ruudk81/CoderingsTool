"""Tests voor het opschonen van verzamelnamen (stap 2b van step 5)."""
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
    prompt = build_umbrella_merge_prompt(UMBRELLAS, "nl-NL")
    assert " 2 " not in prompt and " 3 " not in prompt
    assert "aantal" not in prompt.lower() and "count" not in prompt.lower()


def test_prompt_lists_every_umbrella_with_its_members():
    prompt = build_umbrella_merge_prompt(UMBRELLAS, "nl-NL")
    for u in UMBRELLAS:
        assert u.name in prompt
        for member in u.member_names:
            assert member in prompt


def test_prompt_order_is_not_the_input_order():
    many = [umbrella(f"Naam {i}", [f"Attr {i}"]) for i in range(12)]
    prompt = build_umbrella_merge_prompt(many, "nl-NL")
    positions = [prompt.index(u.name) for u in many]
    assert positions != sorted(positions)


def test_prompt_order_is_stable_across_calls():
    a = build_umbrella_merge_prompt(UMBRELLAS, "nl-NL")
    b = build_umbrella_merge_prompt(UMBRELLAS, "nl-NL")
    assert a == b


def test_prompt_ends_with_the_instructor_hint():
    prompt = build_umbrella_merge_prompt(UMBRELLAS, "nl-NL")
    assert prompt.rstrip().endswith(
        "provide your output as valid JSON following the response schema provided"
    )


def test_model_constrains_members_to_existing_umbrella_names():
    model = make_umbrella_merge_model(UMBRELLAS)
    ok = model(scratchpad="", groups=[{
        "canonical_name": "Bankdiensten",
        "canonical_definition": "d",
        "members": ["Bankdiensten", "Bankdiensten en aanbod"],
    }])
    assert ok.groups[0].members == ["Bankdiensten", "Bankdiensten en aanbod"]

    with pytest.raises(pydantic.ValidationError):
        model(scratchpad="", groups=[{
            "canonical_name": "X", "canonical_definition": "d",
            "members": ["Bestaat niet"],
        }])


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


class FakeGroup:
    def __init__(self, canonical_name, members, canonical_definition="d"):
        self.canonical_name = canonical_name
        self.canonical_definition = canonical_definition
        self.members = members


class FakeMerge:
    def __init__(self, groups):
        self.scratchpad = ""
        self.groups = groups


def test_apply_rewrites_member_umbrellas_to_the_canonical_name():
    relations = FakeRelations([
        FakeRelation("[A1] Betalen", "Bankdiensten"),
        FakeRelation("[A2] Hypotheek", "Bankdiensten en aanbod"),
        FakeRelation("[A3] Logo", "Visuele merkidentiteit"),
    ])
    merged = apply_umbrella_merge(relations, FakeMerge([
        FakeGroup("Bankdiensten", ["Bankdiensten", "Bankdiensten en aanbod"]),
    ]))
    names = [r.umbrella_name for r in merged.relations]
    assert names == ["Bankdiensten", "Bankdiensten", "Visuele merkidentiteit"]


def test_apply_leaves_umbrellas_that_are_in_no_group_untouched():
    relations = FakeRelations([FakeRelation("[A1] Logo", "Visuele merkidentiteit")])
    merged = apply_umbrella_merge(relations, FakeMerge([]))
    assert merged.relations[0].umbrella_name == "Visuele merkidentiteit"


def test_apply_does_not_mutate_the_input():
    relations = FakeRelations([FakeRelation("[A1] Betalen", "Bankdiensten")])
    apply_umbrella_merge(relations, FakeMerge([
        FakeGroup("Diensten", ["Bankdiensten"]),
    ]))
    assert relations.relations[0].umbrella_name == "Bankdiensten"
