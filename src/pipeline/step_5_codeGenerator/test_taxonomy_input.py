"""Tests for step 5's step-4 entry point."""
import pytest

from pipeline.step_5_codeGenerator.taxonomy_input import (
    IdeaUnit, build_idea_units, build_attribute_refs,
)


class FakeIdea:
    def __init__(self, idea_id, attribute, attribute_id, valence,
                 instance="i", interpretation="t"):
        self.idea_id = idea_id
        self.attribute = attribute
        self.attribute_id = attribute_id
        self.valence = valence
        self.instance = instance
        self.interpretation = interpretation


class FakeResponse:
    def __init__(self, respondent_id, ideas):
        self.respondent_id = respondent_id
        self.response_ideas = ideas


def test_idea_units_carry_respondent_id():
    classified = [FakeResponse("R1", [FakeIdea("R1_1", "Prijs", "A1", "-")])]
    units = build_idea_units(classified)
    assert units == [IdeaUnit(idea_id="R1_1", respondent_id="R1", attribute_id="A1",
                              valence="-", instance="i", interpretation="t")]


def test_ideas_without_attribute_id_are_skipped():
    classified = [FakeResponse("R1", [FakeIdea("R1_1", "", "", "+")])]
    assert build_idea_units(classified) == []


def test_responses_without_ideas_are_skipped():
    classified = [FakeResponse("R1", None)]
    assert build_idea_units(classified) == []


def test_a_renamed_idea_field_raises_instead_of_going_empty():
    """A field step 4 renames must break loudly. Empty is a valid value, absent is
    a breach of contract — without that distinction a renamed `valence` silently
    became a neutral idea."""
    class IdeaWithoutValence:
        idea_id = "R1_1"
        attribute_id = "A1"
        instance = "i"
        interpretation = "t"

    classified = [FakeResponse("R1", [IdeaWithoutValence()])]
    with pytest.raises(AttributeError):
        build_idea_units(classified)


def test_a_renamed_attribute_name_key_raises():
    taxonomy = {
        "Domein A": {
            "attributes": {
                "Facet X": [{"attribute_id": "A1", "attribute_definition": "d"}]
            }
        }
    }
    with pytest.raises(KeyError):
        build_attribute_refs(taxonomy)


def test_attribute_refs_read_new_and_old_definition_field():
    taxonomy = {
        "Domein A": {
            "attributes": {
                "Facet X": [
                    {"attribute_id": "A1", "attribute_name": "Prijs",
                     "attribute_definition": "nieuw veld"},
                    {"attribute_id": "A2", "attribute_name": "Service",
                     "attribute_description": "oud veld"},
                ]
            }
        }
    }
    refs = build_attribute_refs(taxonomy)
    assert refs["A1"].definition == "nieuw veld"
    assert refs["A2"].definition == "oud veld"
    assert refs["A1"].domain == "Domein A" and refs["A1"].facet == "Facet X"
