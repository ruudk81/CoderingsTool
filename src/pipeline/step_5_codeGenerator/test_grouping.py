"""Tests voor fase 2 en 3: partitiereparatie, valentiesplitsing, degeneratie."""
from pipeline.step_5_codeGenerator.concept_inventory import Concept
from pipeline.step_5_codeGenerator.attribute_cards import AttributeCard
from pipeline.step_5_codeGenerator.grouping import Group, build_shapes, check_degeneration, repair_partition
from pipeline.step_5_codeGenerator.prompts_consolidation import (
    ConsolidationResult, ProposedCode,
)


def card(attribute_id, name, n_resp=10):
    return AttributeCard(attribute_id=attribute_id, name=name, definition="d",
                         domain="D", facet="F", n_resp=n_resp, top_answers=())


def concept(attribute_id, resp_ids):
    """Concept met expliciete respondent-ids, voor volledige controle over overlap."""
    ids = frozenset(resp_ids)
    return Concept(attribute_id=attribute_id, name=attribute_id, definition="d",
                   domain="D", facet="F", n_iu=len(ids), resp_ids=ids,
                   resp_pos=frozenset(), resp_neg=frozenset(), resp_neu=frozenset())


def concepts_for(cards):
    """Concepten met onderling disjuncte respondentsets, ter grootte van elke
    kaart z'n n_resp — genoeg voor tests die de union-vs-som-fix niet raken."""
    return [concept(c.attribute_id, [f"{c.attribute_id}R{i}" for i in range(c.n_resp)])
            for c in cards]


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
    groups = repair_partition(result(("G", ["[A1] Een", "[A2] Twee"])), cards,
                              concepts_for(cards))

    assert groups == [Group(member_ids=("A1", "A2"), proposed_name="G", explanation="e")]


def test_forgotten_attribute_becomes_its_own_group():
    cards = [card("A1", "Een"), card("A2", "Twee")]
    log = _Log()

    groups = repair_partition(result(("G", ["[A1] Een"])), cards, concepts_for(cards),
                              log=log)

    assert ("A2",) in [g.member_ids for g in groups]
    assert log.entries[0]["action"] == "PARTITION_MISSING"
    assert log.entries[0]["attribute_id"] == "A2"


def test_forgotten_attribute_keeps_its_own_name_as_proposal():
    cards = [card("A1", "Een"), card("A2", "Twee")]
    groups = repair_partition(result(("G", ["[A1] Een"])), cards, concepts_for(cards))

    orphan = next(g for g in groups if g.member_ids == ("A2",))
    assert orphan.proposed_name == "Twee"


def test_double_placed_attribute_goes_to_the_group_with_most_respondents():
    cards = [card("A1", "Groot", 100), card("A2", "Klein", 5), card("A3", "Deler", 10)]
    log = _Log()

    groups = repair_partition(result(
        ("Grote groep", ["[A1] Groot", "[A3] Deler"]),
        ("Kleine groep", ["[A2] Klein", "[A3] Deler"]),
    ), cards, concepts_for(cards), log=log)

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
    ), cards, concepts_for(cards))

    by_name = {g.proposed_name: g.member_ids for g in groups}
    assert by_name["Alfa"] == ("A2", "A3")
    assert by_name["Zebra"] == ("A1",)


def test_group_emptied_by_repair_is_dropped():
    cards = [card("A1", "Groot", 100), card("A2", "Deler", 10)]

    groups = repair_partition(result(
        ("Houdt hem", ["[A1] Groot", "[A2] Deler"]),
        ("Raakt leeg", ["[A2] Deler"]),
    ), cards, concepts_for(cards))

    assert [g.proposed_name for g in groups] == ["Houdt hem"]


def test_double_placement_uses_respondent_union_not_naive_sum():
    """Dubbeltelling: A3's respondenten zitten al in A1. Op papier (naieve som,
    50+30=80) lijkt "Lijkt Groter" de grootste groep, maar in de unie is dat
    maar 50 — kleiner dan de 70 van "Is Groter" (40+30, disjunct). De unie
    moet winnen, dus "Is Groter" houdt A3, niet "Lijkt Groter"."""
    cards = [card("A1", "Een"), card("A2", "Twee"), card("A3", "Deler")]
    a1_ids = [f"R{i}" for i in range(50)]
    a3_ids = a1_ids[:30]                    # volledig overlappend met A1
    a2_ids = [f"S{i}" for i in range(40)]   # disjunct van A1 en A3
    concepts = [concept("A1", a1_ids), concept("A2", a2_ids), concept("A3", a3_ids)]

    groups = repair_partition(result(
        ("Lijkt Groter", ["[A1] Een", "[A3] Deler"]),
        ("Is Groter", ["[A2] Twee", "[A3] Deler"]),
    ), cards, concepts)

    by_name = {g.proposed_name: g.member_ids for g in groups}
    assert by_name["Is Groter"] == ("A2", "A3")
    assert by_name["Lijkt Groter"] == ("A1",)


def test_duplicate_tag_within_group_is_collapsed_and_logged():
    cards = [card("A1", "Een"), card("A2", "Twee")]
    log = _Log()

    groups = repair_partition(result(
        ("G", ["[A1] Een", "[A1] Een", "[A2] Twee"]),
    ), cards, concepts_for(cards), log=log)

    g = next(g for g in groups if g.proposed_name == "G")
    assert g.member_ids == ("A1", "A2")
    assert log.entries[0]["action"] == "PARTITION_DUPLICATE_IN_GROUP"
    assert log.entries[0]["attribute_id"] == "A1"


def valence_concept(attribute_id, name, pos=0, neg=0, neu=0):
    """Concept met poolgrootte per valentie, voor de valentiesplitsing-tests.
    Anders dan `concept()` hierboven (expliciete resp_ids, geen valentie) —
    hier is de poolgrootte per valentie precies wat wordt getest, dus een
    tweede helper in plaats van hergebruik met een andere signatuur."""
    def resp(prefix, n):
        return frozenset(f"{attribute_id}{prefix}{i}" for i in range(n))
    p, g, u = resp("P", pos), resp("G", neg), resp("U", neu)
    return Concept(attribute_id=attribute_id, name=name, definition="d",
                   domain="D", facet="F", n_iu=pos + neg + neu,
                   resp_ids=p | g | u, resp_pos=p, resp_neg=g, resp_neu=u)


def group(*ids, name="G"):
    return Group(member_ids=tuple(ids), proposed_name=name, explanation="e")


def test_both_poles_above_threshold_become_two_pure_codes():
    concepts = [valence_concept("A1", "Iets", pos=30, neg=20)]

    out = build_shapes([group("A1")], concepts, threshold=12)

    by_valence = {s.valence: s for s in out.shapes}
    assert set(by_valence) == {"positive", "negative"}
    assert len(by_valence["positive"].resp_ids) == 30
    assert by_valence["positive"].resp_neg == frozenset()
    assert len(by_valence["negative"].resp_ids) == 20
    assert by_valence["negative"].resp_pos == frozenset()


def test_minority_pole_below_threshold_goes_to_overig_not_into_the_code():
    """Dit is de eis die v1 stilzwijgend overtrad: een code die 'positive' heet
    en de negatieve respondenten meedraagt."""
    concepts = [valence_concept("A1", "Iets", pos=30, neg=8)]

    out = build_shapes([group("A1")], concepts, threshold=12)

    assert [s.valence for s in out.shapes] == ["positive"]
    assert out.shapes[0].resp_neg == frozenset()
    assert len(out.shapes[0].resp_ids) == 30
    assert out.direction_loss == 8


def test_group_where_no_pole_clears_the_threshold_lands_entirely_in_overig():
    concepts = [valence_concept("A1", "Iets", pos=5, neg=4, neu=3)]

    out = build_shapes([group("A1")], concepts, threshold=12)

    assert out.shapes == []
    assert out.overig_ids == ["A1"]
    assert out.direction_loss == 12


def test_members_of_a_group_are_unioned_by_respondent_not_summed():
    """Een respondent die in twee samengevoegde attributen zit telt één keer."""
    shared = frozenset({"r1", "r2"})
    concepts = [
        Concept(attribute_id="A1", name="Een", definition="d", domain="D", facet="F",
                n_iu=2, resp_ids=shared, resp_pos=shared, resp_neg=frozenset(),
                resp_neu=frozenset()),
        Concept(attribute_id="A2", name="Twee", definition="d", domain="D", facet="F",
                n_iu=2, resp_ids=shared, resp_pos=shared, resp_neg=frozenset(),
                resp_neu=frozenset()),
    ]

    out = build_shapes([group("A1", "A2")], concepts, threshold=2)

    assert len(out.shapes) == 1
    assert len(out.shapes[0].resp_ids) == 2


def test_direction_loss_unions_dropped_poles_instead_of_summing_them():
    """De unie-niet-som-fix in `direction_loss` (regel 154) had geen test die
    een terugval op `sum()` zou opmerken — elke bestaande fixture gebruikt
    onderling disjuncte respondenten per pool. Hier deelt dezelfde respondent
    twee gedropte polen van dezelfde groep (negatief via A1, neutraal via A2):
    naief opgeteld (3 + 3) zou 6 zijn, de unie is 5."""
    shared = "shared1"
    concept_a1 = Concept(
        attribute_id="A1", name="Een", definition="d", domain="D", facet="F",
        n_iu=33,
        resp_ids=frozenset(f"P{i}" for i in range(30)) | frozenset({shared, "negA", "negB"}),
        resp_pos=frozenset(f"P{i}" for i in range(30)),
        resp_neg=frozenset({shared, "negA", "negB"}),
        resp_neu=frozenset(),
    )
    concept_a2 = Concept(
        attribute_id="A2", name="Twee", definition="d", domain="D", facet="F",
        n_iu=3, resp_ids=frozenset({shared, "neuA", "neuB"}),
        resp_pos=frozenset(), resp_neg=frozenset(),
        resp_neu=frozenset({shared, "neuA", "neuB"}),
    )

    out = build_shapes([group("A1", "A2")], [concept_a1, concept_a2], threshold=12)

    assert len(out.shapes) == 1
    assert out.shapes[0].valence == "positive"
    assert out.direction_loss == 5


def test_shape_keys_are_unique_across_all_groups_and_poles():
    concepts = [valence_concept("A1", "Een", pos=30, neg=20), valence_concept("A2", "Twee", pos=30)]

    out = build_shapes([group("A1", name="G1"), group("A2", name="G2")],
                       concepts, threshold=12)

    keys = [s.key for s in out.shapes]
    assert len(keys) == len(set(keys))


def test_group_with_no_matching_concept_routes_its_ids_to_overig_instead_of_vanishing():
    """Onbereikbaar via de normale route (elk lid van een `Group` komt uit
    `repair_partition`, dat zelf uit `cards`/`concepts` put), maar als die
    aanname ooit breekt moet de boekhouding heel blijven: een groep zonder een
    enkele bijpassende Concept moet naar Overig, niet stilzwijgend verdwijnen."""
    concepts = [valence_concept("A1", "Iets", pos=30)]

    out = build_shapes([group("A1"), group("Onbekend")], concepts, threshold=12)

    assert out.overig_ids == ["Onbekend"]
    assert len(out.shapes) == 1


def test_shape_carries_the_proposed_name_as_umbrella():
    """`umbrella` is waar resolve_duplicate_names op terugvalt bij een
    naamsbotsing, dus het moet de voorgestelde naam dragen."""
    concepts = [valence_concept("A1", "Iets", pos=30)]

    out = build_shapes([group("A1", name="Voorstel")], concepts, threshold=12)

    assert out.shapes[0].umbrella == "Voorstel"


def test_healthy_consolidation_is_not_flagged():
    assert check_degeneration(n_groups=26, n_attributes=66) is None


def test_no_consolidation_at_all_is_flagged():
    assert "geen consolidatie" in check_degeneration(n_groups=64, n_attributes=66)


def test_everything_on_one_heap_is_flagged():
    assert "één hoop" in check_degeneration(n_groups=2, n_attributes=66)


def test_exact_ceiling_boundary_counts_as_healthy():
    """De vergelijking is strikt (`>`): exact 90% van de attributen is nog
    gezond, 90% + 1 groep niet."""
    assert check_degeneration(n_groups=90, n_attributes=100) is None
    assert check_degeneration(n_groups=91, n_attributes=100) is not None


def test_exact_floor_boundary_counts_as_healthy():
    """De vergelijking is strikt (`<`): exact 5% van de attributen is nog
    gezond, 5% - 1 groep niet."""
    assert check_degeneration(n_groups=5, n_attributes=100) is None
    assert check_degeneration(n_groups=4, n_attributes=100) is not None


def test_bounds_are_relative_so_a_small_tree_is_judged_on_its_own_scale():
    """Een absolute ondergrens ('minder dan 3 codes is fout') zou op een kleine
    dataset een correcte uitkomst afkeuren. Dat is precies de use-case-
    afhankelijkheid die dit ontwerp moet vermijden."""
    assert check_degeneration(n_groups=3, n_attributes=20) is None
    assert check_degeneration(n_groups=3, n_attributes=200) is not None


def test_no_attributes_is_not_a_degeneration_verdict():
    assert check_degeneration(n_groups=0, n_attributes=0) is None


def pooled_concept(attribute_id, pos=(), neg=(), neu=()):
    """Concept met expliciet gevulde valentiepolen."""
    pos, neg, neu = frozenset(pos), frozenset(neg), frozenset(neu)
    return Concept(attribute_id=attribute_id, name=attribute_id, definition="d",
                   domain="D", facet="F", n_iu=len(pos | neg | neu),
                   resp_ids=pos | neg | neu,
                   resp_pos=pos, resp_neg=neg, resp_neu=neu)


def test_driedeling_laat_een_te_kleine_pool_vallen():
    """Vandaag: pos=2 en neu=1 halen de drempel van 3 niet en gaan verloren."""
    c = pooled_concept("A1", pos={"r1", "r2"}, neu={"r3"}, neg={"r4", "r5", "r6"})
    groups = [Group(member_ids=("A1",), proposed_name="G", explanation="e")]

    result = build_shapes(groups, [c], threshold=3)

    assert [s.valence for s in result.shapes] == ["negative"]
    assert result.direction_loss == 3


def test_tweedeling_redt_dezelfde_pool():
    """pos ∪ neu = 3 haalt de drempel wel, dus er gaat niets verloren."""
    c = pooled_concept("A1", pos={"r1", "r2"}, neu={"r3"}, neg={"r4", "r5", "r6"})
    groups = [Group(member_ids=("A1",), proposed_name="G", explanation="e")]

    result = build_shapes(groups, [c], threshold=3, two_pole=True)

    assert sorted(s.valence for s in result.shapes) == ["negative", "non_negative"]
    assert result.direction_loss == 0


def test_tweedeling_telt_een_respondent_met_twee_ideeen_een_keer():
    """r1 heeft zowel een positief als een neutraal idee: unie, geen som."""
    c = pooled_concept("A1", pos={"r1", "r2"}, neu={"r1", "r3"})
    groups = [Group(member_ids=("A1",), proposed_name="G", explanation="e")]

    result = build_shapes(groups, [c], threshold=3, two_pole=True)

    shape = next(s for s in result.shapes if s.valence == "non_negative")
    assert len(shape.resp_ids) == 3


def test_tweedeling_bewaart_de_onderverdeling_op_de_shape():
    """De samengevoegde pool moet nog steeds laten zien wat + en 0 was."""
    c = pooled_concept("A1", pos={"r1", "r2", "r3"}, neu={"r4"})
    groups = [Group(member_ids=("A1",), proposed_name="G", explanation="e")]

    shape = build_shapes(groups, [c], threshold=3, two_pole=True).shapes[0]

    assert shape.resp_pos == frozenset({"r1", "r2", "r3"})
    assert shape.resp_neu == frozenset({"r4"})
    assert shape.resp_neg == frozenset()
