"""Tests for the pure batch-assignment helpers (step 4, P4 batch mode)."""
import numpy as np

from pipeline.step_4_classifier.assignment_batching import (
    facet_card_text,
    group_label_reps,
    make_batches,
    shortlist_indices,
    validate_batch_response,
)


class FakeIdea:
    def __init__(self, idea_id, idea):
        self.idea_id = idea_id
        self.idea = idea
        self.instance = ""
        self.interpretation = ""
        self.abstraction = ""
        self.facet = ""
        self.domain = ""


class FakeItem:
    def __init__(self, idea_id, facet_id):
        self.idea_id = idea_id
        self.assigned_facet_id = facet_id


class FakeResponse:
    def __init__(self, items):
        self.assignments = items


def test_group_label_reps_dedups_normalized_labels():
    ideas = [FakeIdea("a", "Warm gevoel"), FakeIdea("b", " warm gevoel "),
             FakeIdea("c", "ouderwets")]
    reps = group_label_reps(ideas, "idea", "", dedup=True)
    assert [r.idea_ids for r in reps] == [["a", "b"], ["c"]]
    assert reps[0].label == "Warm gevoel"  # eerste-gezien label, niet de genormaliseerde


def test_group_label_reps_without_dedup_is_one_rep_per_idea():
    ideas = [FakeIdea("a", "x"), FakeIdea("b", "x")]
    reps = group_label_reps(ideas, "idea", "", dedup=False)
    assert [r.idea_ids for r in reps] == [["a"], ["b"]]


def test_group_label_reps_never_merges_empty_labels():
    ideas = [FakeIdea("a", ""), FakeIdea("b", "")]
    reps = group_label_reps(ideas, "idea", "", dedup=True)
    assert [r.idea_ids for r in reps] == [["a"], ["b"]]


def test_make_batches_splits_and_covers_all_indices():
    assert make_batches(7, 3) == [[0, 1, 2], [3, 4, 5], [6]]
    assert make_batches(0, 3) == []


def test_shortlist_indices_unions_per_row_topk_sorted():
    cards = np.eye(4, dtype=np.float32)
    labels = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
    assert shortlist_indices(labels, cards, 1) == [0, 2]
    assert shortlist_indices(labels, cards, 4) == [0, 1, 2, 3]


def test_validate_batch_response_routes_ok_missing_duplicate_fnone():
    response = FakeResponse([
        FakeItem("a", "F1"),
        FakeItem("b", "F_NONE"),
        FakeItem("c", "F2"), FakeItem("c", "F3"),
    ])
    ok, escalate = validate_batch_response(["a", "b", "c", "d"], response)
    assert list(ok) == ["a"] and ok["a"].assigned_facet_id == "F1"
    assert escalate == {"b": "f_none", "c": "duplicate", "d": "missing"}


def test_facet_card_text_joins_menu_fields():
    text = facet_card_text({
        "facet_name": "N", "facet_description": "D",
        "inclusion_rule": "I", "example_observations": ["e1", "e2"],
    })
    assert "N" in text and "D" in text and "I" in text and "e1" in text
