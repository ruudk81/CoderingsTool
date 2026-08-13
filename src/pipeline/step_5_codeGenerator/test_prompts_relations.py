"""Tests voor de relatieprompt (stap 2 van step 5)."""
import re

from pipeline.step_5_codeGenerator.concept_inventory import Concept
from pipeline.step_5_codeGenerator.prompts_relations import (
    build_relations_prompt, make_relations_model, tagged,
)


def concept(attribute_id, name, n_resp=10, domain="Domein"):
    resp = frozenset(f"R{i}" for i in range(n_resp))
    return Concept(attribute_id=attribute_id, name=name, definition="def",
                   domain=domain, facet="Facet", n_iu=n_resp,
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
    assert "[A1] Prijs" in prompt
    assert "[A2] Service" in prompt


def test_prompt_order_is_not_the_prevalence_order():
    # Input arrives prevalence-sorted, as build_inventory produces it: highest
    # n_resp first. The rendered order must not match that.
    concepts = [concept(f"A{i}", f"Topic{i}", n_resp=100 - i) for i in range(8)]
    prompt = build_relations_prompt(concepts, "nl-NL")
    rendered_order = sorted(concepts, key=lambda c: prompt.index(tagged(c)))
    assert [c.attribute_id for c in rendered_order] != [c.attribute_id for c in concepts]


def test_prompt_order_is_stable_across_calls():
    concepts = [concept(f"A{i}", f"Topic{i}", n_resp=100 - i) for i in range(8)]
    first = build_relations_prompt(concepts, "nl-NL")
    second = build_relations_prompt(concepts, "nl-NL")
    assert first == second


def test_prompt_breaks_up_domain_contiguity():
    # attribute_id is minted sequentially PER DOMAIN (identity.py), so ordering
    # by id alone would place every domain's attributes in one unbroken block —
    # reproducing the domain grouping in list position instead of in the label.
    # This is the property the fix exists for, not an implementation detail.
    concepts = (
        [concept(f"A{i}", f"Een{i}", domain="Domein1") for i in range(1, 4)]
        + [concept(f"A{i}", f"Twee{i}", domain="Domein2") for i in range(4, 7)]
    )
    prompt = build_relations_prompt(concepts, "nl-NL")
    rendered_order = sorted(concepts, key=lambda c: prompt.index(tagged(c)))
    domain_sequence = [c.domain for c in rendered_order]

    runs = 1 + sum(a != b for a, b in zip(domain_sequence, domain_sequence[1:]))
    assert runs > len(set(domain_sequence)), (
        "id order would give exactly one run per domain — the fix must break "
        "at least one domain's attributes into more than one run"
    )


def test_prompt_ends_with_the_instructor_hint():
    prompt = build_relations_prompt([concept("A1", "Prijs")], "nl-NL")
    assert prompt.rstrip().endswith(
        "provide your output as valid JSON following the response schema provided"
    )


def test_response_model_constrains_to_existing_attributes():
    model = make_relations_model([concept("A1", "Prijs"), concept("A2", "Service")])
    ok = model(relations=[{"attribute": "[A1] Prijs", "synonym_of": None,
                           "umbrella_name": "Kosten", "umbrella_definition": "d"}])
    assert ok.relations[0].attribute == "[A1] Prijs"

    import pydantic
    try:
        model(relations=[{"attribute": "[A99] Verzonnen", "synonym_of": None,
                          "umbrella_name": "Kosten", "umbrella_definition": "d"}])
    except pydantic.ValidationError:
        return
    raise AssertionError("een niet-bestaand attribuut had geweigerd moeten worden")
