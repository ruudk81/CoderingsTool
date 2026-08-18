"""Tests voor fase 2 en 3: partitiereparatie, valentiesplitsing, degeneratie."""
from pipeline.step_5_codeGenerator.v2.attribute_cards import AttributeCard
from pipeline.step_5_codeGenerator.v2.grouping import Group, repair_partition
from pipeline.step_5_codeGenerator.v2.prompts_consolidation import (
    ConsolidationResult, ProposedCode,
)


def card(attribute_id, name, n_resp=10):
    return AttributeCard(attribute_id=attribute_id, name=name, definition="d",
                         domain="D", facet="F", n_resp=n_resp, top_answers=())


class _Log:
    def __init__(self):
        self.entries = []

    def add(self, **kwargs):
        self.entries.append(kwargs)


def result(*groups):
    """groups: reeks (naam, [tags])."""
    return ConsolidationResult(codes=[
        ProposedCode(code_name=name, explanation="e", topics=list(tags))
        for name, tags in groups
    ])


def test_clean_proposal_passes_through_unchanged():
    cards = [card("A1", "Een"), card("A2", "Twee")]
    groups = repair_partition(result(("G", ["[A1] Een", "[A2] Twee"])), cards)

    assert groups == [Group(member_ids=("A1", "A2"), proposed_name="G", explanation="e")]


def test_forgotten_attribute_becomes_its_own_group():
    cards = [card("A1", "Een"), card("A2", "Twee")]
    log = _Log()

    groups = repair_partition(result(("G", ["[A1] Een"])), cards, log=log)

    assert ("A2",) in [g.member_ids for g in groups]
    assert log.entries[0]["action"] == "PARTITION_MISSING"
    assert log.entries[0]["attribute_id"] == "A2"


def test_forgotten_attribute_keeps_its_own_name_as_proposal():
    cards = [card("A1", "Een"), card("A2", "Twee")]
    groups = repair_partition(result(("G", ["[A1] Een"])), cards)

    orphan = next(g for g in groups if g.member_ids == ("A2",))
    assert orphan.proposed_name == "Twee"


def test_double_placed_attribute_goes_to_the_group_with_most_respondents():
    cards = [card("A1", "Groot", 100), card("A2", "Klein", 5), card("A3", "Deler", 10)]
    log = _Log()

    groups = repair_partition(result(
        ("Grote groep", ["[A1] Groot", "[A3] Deler"]),
        ("Kleine groep", ["[A2] Klein", "[A3] Deler"]),
    ), cards, log=log)

    by_name = {g.proposed_name: g.member_ids for g in groups}
    assert by_name["Grote groep"] == ("A1", "A3")
    assert by_name["Kleine groep"] == ("A2",)
    assert log.entries[0]["action"] == "PARTITION_DOUBLE"


def test_double_placement_tie_is_broken_reproducibly():
    """Gelijke respondentaantallen: meeste leden wint, dan alfabetisch op naam."""
    cards = [card("A1", "Een", 10), card("A2", "Twee", 10), card("A3", "Deler", 10)]

    groups = repair_partition(result(
        ("Zebra", ["[A1] Een", "[A3] Deler"]),
        ("Alfa", ["[A2] Twee", "[A3] Deler"]),
    ), cards)

    by_name = {g.proposed_name: g.member_ids for g in groups}
    assert by_name["Alfa"] == ("A2", "A3")
    assert by_name["Zebra"] == ("A1",)


def test_group_emptied_by_repair_is_dropped():
    cards = [card("A1", "Groot", 100), card("A2", "Deler", 10)]

    groups = repair_partition(result(
        ("Houdt hem", ["[A1] Groot", "[A2] Deler"]),
        ("Raakt leeg", ["[A2] Deler"]),
    ), cards)

    assert [g.proposed_name for g in groups] == ["Houdt hem"]
