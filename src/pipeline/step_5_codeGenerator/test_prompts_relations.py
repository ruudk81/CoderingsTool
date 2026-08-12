"""Tests voor de relatieprompt (stap 2 van step 5)."""
import re

from pipeline.step_5_codeGenerator.concept_inventory import Concept
from pipeline.step_5_codeGenerator.prompts_relations import (
    build_relations_prompt, make_relations_model,
)


def concept(attribute_id, name, n_resp=10):
    resp = frozenset(f"R{i}" for i in range(n_resp))
    return Concept(attribute_id=attribute_id, name=name, definition="def",
                   domain="Domein", facet="Facet", n_iu=n_resp,
                   resp_ids=resp, resp_pos=resp,
                   resp_neg=frozenset(), resp_neu=frozenset())


def test_prompt_contains_no_counts():
    concepts = [concept("A1", "Prijs", 240), concept("A2", "Service", 9)]
    prompt = build_relations_prompt(concepts, "nl-NL")
    assert "240" not in prompt
    assert "9" not in re.sub(r"\{\{.*?\}\}", "", prompt)


def test_prompt_lists_every_attribute():
    concepts = [concept("A1", "Prijs"), concept("A2", "Service")]
    prompt = build_relations_prompt(concepts, "nl-NL")
    assert "Prijs (Domein)" in prompt
    assert "Service (Domein)" in prompt


def test_prompt_ends_with_the_instructor_hint():
    prompt = build_relations_prompt([concept("A1", "Prijs")], "nl-NL")
    assert prompt.rstrip().endswith(
        "provide your output as valid JSON following the response schema provided"
    )


def test_response_model_constrains_to_existing_attributes():
    model = make_relations_model([concept("A1", "Prijs"), concept("A2", "Service")])
    ok = model(relations=[{"attribute": "Prijs (Domein)", "synonym_of": None,
                           "umbrella_name": "Kosten", "umbrella_definition": "d"}])
    assert ok.relations[0].attribute == "Prijs (Domein)"

    import pydantic
    try:
        model(relations=[{"attribute": "Verzonnen (Domein)", "synonym_of": None,
                          "umbrella_name": "Kosten", "umbrella_definition": "d"}])
    except pydantic.ValidationError:
        return
    raise AssertionError("een niet-bestaand attribuut had geweigerd moeten worden")
