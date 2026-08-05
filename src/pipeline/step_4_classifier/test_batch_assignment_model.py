"""Tests for the runtime-built batch facet-assignment response model."""
import pytest
from pydantic import ValidationError

from pipeline.step_4_classifier.prompts_classifier import (
    build_batch_facet_assignment_model,
    build_facet_assignment_prompt_batch,
)

FACET_IDS = ["F1", "F2", "F3"]
IDEA_IDS = ["id_a", "id_b"]


def make_item(idea_id="id_a", facet_id="F1"):
    return {"idea_id": idea_id, "assigned_facet_id": facet_id,
            "confidence": 0.9, "valence": "+"}


def test_accepts_valid_assignments():
    model = build_batch_facet_assignment_model(FACET_IDS, IDEA_IDS)
    parsed = model(assignments=[make_item("id_a", "F2"), make_item("id_b", "F_NONE")])
    assert parsed.assignments[0].assigned_facet_id == "F2"
    assert parsed.assignments[1].assigned_facet_id == "F_NONE"


def test_rejects_hallucinated_facet_id():
    model = build_batch_facet_assignment_model(FACET_IDS, IDEA_IDS)
    with pytest.raises(ValidationError):
        model(assignments=[make_item("id_a", "F99")])


def test_rejects_hallucinated_idea_id():
    model = build_batch_facet_assignment_model(FACET_IDS, IDEA_IDS)
    with pytest.raises(ValidationError):
        model(assignments=[make_item("id_onbekend", "F1")])


def test_rejects_out_of_range_confidence():
    model = build_batch_facet_assignment_model(FACET_IDS, IDEA_IDS)
    bad = make_item()
    bad["confidence"] = 1.4
    with pytest.raises(ValidationError):
        model(assignments=[bad])


def test_batch_prompt_contains_menu_ideas_escape_and_schema_hint():
    from pipeline.step_4_classifier.prompts_classifier import DiscoveredFacet
    facets = [
        DiscoveredFacet(facet_name=f"Facet {i}", facet_description="d",
                        example_observations=["e"]) for i in (1, 2)
    ]
    prompt = build_facet_assignment_prompt_batch(
        survey_question="Q?", language="nl-NL", dataset_context_section="",
        domain_name="dom", domain_definition="def", facets=facets,
        ideas=[("id_a", "label a"), ("id_b", "label b")],
    )
    assert "[F1] Facet 1" in prompt and "[F2] Facet 2" in prompt
    assert "[id_a] label a" in prompt and "[id_b] label b" in prompt
    assert "F_NONE" in prompt
    assert prompt.rstrip().endswith(
        "Provide your output as valid JSON following the response schema provided.")
