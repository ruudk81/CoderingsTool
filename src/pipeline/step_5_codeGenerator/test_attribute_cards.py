"""Tests voor de attribuutkaarten die fase 1 te zien krijgt."""
from pipeline.step_5_codeGenerator.concept_inventory import Concept
from pipeline.step_5_codeGenerator.taxonomy_input import IdeaUnit
from pipeline.step_5_codeGenerator.attribute_cards import build_cards


def concept(attribute_id, name, pos=0, neg=0, neu=0):
    def resp(prefix, n):
        return frozenset(f"{attribute_id}{prefix}{i}" for i in range(n))
    p, g, u = resp("P", pos), resp("G", neg), resp("U", neu)
    return Concept(attribute_id=attribute_id, name=name, definition=f"def van {name}",
                   domain="D", facet="F", n_iu=pos + neg + neu,
                   resp_ids=p | g | u, resp_pos=p, resp_neg=g, resp_neu=u)


def unit(attribute_id, respondent_id, instance):
    return IdeaUnit(idea_id=f"{attribute_id}-{respondent_id}", respondent_id=respondent_id,
                    attribute_id=attribute_id, valence="+", instance=instance,
                    interpretation="i")


def test_card_carries_name_definition_address_and_respondent_count():
    concepts = [concept("A1", "Duurzaamheid", pos=3)]
    units = {"A1": [unit("A1", "r1", "groen"), unit("A1", "r2", "groen"),
                    unit("A1", "r3", "milieu")]}

    cards = build_cards(concepts, units)

    assert len(cards) == 1
    card = cards[0]
    assert card.attribute_id == "A1"
    assert card.name == "Duurzaamheid"
    assert card.definition == "def van Duurzaamheid"
    assert card.domain == "D"
    assert card.facet == "F"
    assert card.n_resp == 3
    assert card.tag == "[A1] Duurzaamheid"


def test_top_answers_are_the_most_frequent_instances_with_their_counts():
    concepts = [concept("A1", "Duurzaamheid", pos=4)]
    units = {"A1": [unit("A1", "r1", "groen"), unit("A1", "r2", "groen"),
                    unit("A1", "r3", "groen"), unit("A1", "r4", "milieu")]}

    card = build_cards(concepts, units)[0]

    assert card.top_answers == (("groen", 3), ("milieu", 1))


def test_top_answers_is_capped_at_top_n():
    concepts = [concept("A1", "Iets", pos=6)]
    units = {"A1": [unit("A1", f"r{i}", f"antwoord{i}") for i in range(6)]}

    card = build_cards(concepts, units, top_n=2)[0]

    assert len(card.top_answers) == 2


def test_empty_instances_are_not_offered_as_answers():
    concepts = [concept("A1", "Iets", pos=2)]
    units = {"A1": [unit("A1", "r1", ""), unit("A1", "r2", "wel iets")]}

    card = build_cards(concepts, units)[0]

    assert card.top_answers == (("wel iets", 1),)


def test_attribute_without_ideas_still_gets_a_card_without_answers():
    """Een Concept bestaat alleen als er ideeën zijn, maar de idee-index kan
    achterlopen op een gefilterde run. De kaart mag daar niet op crashen."""
    cards = build_cards([concept("A1", "Iets", pos=2)], {})

    assert cards[0].top_answers == ()


def test_vangnetten_komen_niet_op_een_kaart():
    """Een vangnet is per constructie restant, geen onderwerp. Zijn definitie
    luidt letterlijk 'responsen die nergens pasten', dus het model vragen om
    hem thematisch te groeperen is een onbeantwoordbare vraag — en hij kreeg
    daar gemeten 28-van-30-zekerheid op, over een bakje met een respondent.
    De ideeen erop vallen in de Overig-sweep, die 100% dekking garandeert."""
    gewoon = concept("A1", "Prijs", pos=3)
    vangnet = Concept(attribute_id="A9", name="Overig — F", definition="rest",
                      domain="D", facet="F", n_iu=1,
                      resp_ids=frozenset({"r9"}), resp_pos=frozenset(),
                      resp_neg=frozenset(), resp_neu=frozenset({"r9"}),
                      is_drain=True)

    cards = build_cards([gewoon, vangnet], {}, exclude_drains=True)

    assert [c.attribute_id for c in cards] == ["A1"]


def test_zonder_de_vlag_staan_vangnetten_gewoon_op_een_kaart():
    """De standaard is het gedrag van voor 2026-08-20, tot het promotiebesluit
    over het consensus-experiment valt."""
    gewoon = concept("A1", "Prijs", pos=3)
    vangnet = Concept(attribute_id="A9", name="Overig — F", definition="rest",
                      domain="D", facet="F", n_iu=1,
                      resp_ids=frozenset({"r9"}), resp_pos=frozenset(),
                      resp_neg=frozenset(), resp_neu=frozenset({"r9"}),
                      is_drain=True)

    cards = build_cards([gewoon, vangnet], {})

    assert [c.attribute_id for c in cards] == ["A1", "A9"]
