"""Tests voor fase 2 en 3: partitiereparatie, valentiesplitsing, degeneratie."""
from dataclasses import replace

from pipeline.step_5_codeGenerator.consensus.concept_inventory import Concept
from pipeline.step_5_codeGenerator.consensus.attribute_cards import AttributeCard
from pipeline.step_5_codeGenerator.consensus.grouping import (
    Group, build_shapes, check_degeneration, pool_thin_within_facet, repair_partition,
)
from pipeline.step_5_codeGenerator.consensus.prompts_consolidation import (
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


def test_minority_pole_below_threshold_stays_out_of_the_majority_code():
    """Dit is de eis die v1 stilzwijgend overtrad: een code die 'positive' heet
    en de negatieve respondenten meedraagt.

    De afgevallen pool verdwijnt sinds 2026-08-22 niet meer: hij is de enige
    van zijn facet en valentie, haalt de drempel van 12 niet maar wel de bodem
    van 3, en wordt daarmee een kind. De eis hierboven staat los daarvan — de
    positieve code draagt die acht respondenten nog steeds niet."""
    concepts = [valence_concept("A1", "Iets", pos=30, neg=8)]

    out = build_shapes([group("A1")], concepts, threshold=12)

    positief = out.shapes[0]
    assert positief.valence == "positive"
    assert positief.resp_neg == frozenset()
    assert len(positief.resp_ids) == 30

    kind = out.shapes[1]
    assert (kind.valence, kind.origin) == ("negative", "child")
    assert len(kind.resp_ids) == 8
    assert out.coverage_recovered == 8


def test_group_where_no_pole_clears_the_threshold_still_feeds_the_facet_pool():
    """Deze test heette tot 2026-08-22 `..._lands_entirely_in_overig` en eiste
    het omgekeerde: geen enkele vorm, alle attributen naar Overig.

    Dat was de smalle regel van taak 2 — afgevallen polen alleen oppakken waar
    een zusterpool overleefde, want juist daar telt materiaal mee onder een code
    die het tegenovergestelde beweert. De regel is verbreed omdat een groep die
    in zijn geheel in Overig verdween zijn minderheidsmateriaal ononderscheiden
    achterliet: het doel is dat zulk materiaal een eigen naam krijgt, desnoods
    als kind onder Overig. Dit is dus geen aangepaste verwachting maar een
    gewijzigd besluit.

    Alle drie de polen (5, 4, 3) halen de drempel van 12 niet en blijven boven
    de bodem van 3, dus ze worden alle drie een kind — en Overig blijft leeg.
    """
    concepts = [valence_concept("A1", "Iets", pos=5, neg=4, neu=3)]

    out = build_shapes([group("A1")], concepts, threshold=12)

    assert sorted((s.valence, s.origin) for s in out.shapes) == [
        ("negative", "child"), ("neutral", "child"), ("positive", "child")]
    assert out.overig_ids == []
    assert out.coverage_recovered == 12


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


def test_coverage_recovered_unions_recovered_poles_instead_of_summing_them():
    """De unie-niet-som-eis had geen test die een terugval op `sum()` zou
    opmerken — elke bestaande fixture gebruikt onderling disjuncte respondenten
    per pool. Hier deelt dezelfde respondent twee AFGEVALLEN polen van dezelfde
    groep (negatief via A1, neutraal via A2). Beide worden een kind, en de
    respondent die in allebei zit telt in de dekkingsmaat één keer: naief
    opgeteld (3 + 3) zou 6 zijn, de unie is 5."""
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

    assert out.shapes[0].valence == "positive"
    assert sorted((s.valence, s.origin) for s in out.shapes[1:]) == [
        ("negative", "child"), ("neutral", "child")]
    assert out.coverage_recovered == 5


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
    # pos=2 en neu=1 halen ook samengenomen de bodem van 3 niet: echt-overig.
    assert result.coverage_recovered == 0


def test_tweedeling_redt_dezelfde_pool():
    """pos ∪ neu = 3 haalt de drempel wel, dus er gaat niets verloren."""
    c = pooled_concept("A1", pos={"r1", "r2"}, neu={"r3"}, neg={"r4", "r5", "r6"})
    groups = [Group(member_ids=("A1",), proposed_name="G", explanation="e")]

    result = build_shapes(groups, [c], threshold=3, two_pole=True)

    assert sorted(s.valence for s in result.shapes) == ["negative", "non_negative"]
    assert result.coverage_recovered == 0


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


def facet_concept(attribute_id, facet, resp, valence="neu"):
    """Concept met een expliciet facet en al zijn respondenten op een pool."""
    ids = frozenset(resp)
    return Concept(attribute_id=attribute_id, name=attribute_id, definition="d",
                   domain="D", facet=facet, n_iu=len(ids), resp_ids=ids,
                   resp_pos=ids if valence == "pos" else frozenset(),
                   resp_neg=ids if valence == "neg" else frozenset(),
                   resp_neu=ids if valence == "neu" else frozenset())


def solo(attribute_id):
    return Group(member_ids=(attribute_id,), proposed_name=attribute_id, explanation="e")


def test_dunne_facetgenoten_worden_een_groep():
    """Twee attributen die elk te dun zijn maar hetzelfde facet delen, halen
    de drempel samen wel. Vandaag verdwijnen ze allebei in Overig."""
    concepts = [facet_concept("A1", "Bancair", {"r1", "r2"}),
                facet_concept("A2", "Bancair", {"r3", "r4"})]

    groups, log = pool_thin_within_facet([solo("A1"), solo("A2")], concepts, threshold=4)

    assert [g.member_ids for g in groups] == [("A1", "A2")]
    assert log[0]["facet"] == "Bancair"


def test_een_groep_boven_de_drempel_blijft_ongemoeid():
    concepts = [facet_concept("A1", "Bancair", {"r1", "r2", "r3", "r4"}),
                facet_concept("A2", "Bancair", {"r5"})]

    groups, _ = pool_thin_within_facet([solo("A1"), solo("A2")], concepts, threshold=4)

    assert ("A1",) in [g.member_ids for g in groups]
    assert ("A2",) in [g.member_ids for g in groups]


def test_dunne_attributen_uit_verschillende_facetten_blijven_apart():
    """De pool mag nooit een facetgrens oversteken — hij erft step 4's
    structuur en verzint er geen nieuwe."""
    concepts = [facet_concept("A1", "Bancair", {"r1", "r2"}),
                facet_concept("A2", "Merkbeeld", {"r3", "r4"})]

    groups, log = pool_thin_within_facet([solo("A1"), solo("A2")], concepts, threshold=4)

    assert sorted(g.member_ids for g in groups) == [("A1",), ("A2",)]
    assert log == []


def test_een_pool_die_ook_samen_te_dun_blijft_wordt_niet_gevormd():
    concepts = [facet_concept("A1", "Bancair", {"r1"}),
                facet_concept("A2", "Bancair", {"r2"})]

    groups, log = pool_thin_within_facet([solo("A1"), solo("A2")], concepts, threshold=9)

    assert sorted(g.member_ids for g in groups) == [("A1",), ("A2",)]
    assert log == []


def test_respondenten_worden_verenigd_niet_opgeteld():
    """r1 antwoordde op beide attributen: samen zijn het er drie, niet vier."""
    concepts = [facet_concept("A1", "Bancair", {"r1", "r2"}),
                facet_concept("A2", "Bancair", {"r1", "r3"})]

    groups, _ = pool_thin_within_facet([solo("A1"), solo("A2")], concepts, threshold=4)

    assert [g.member_ids for g in groups] == [("A1",), ("A2",)]


def test_een_groep_over_twee_facetten_wordt_niet_gepoold():
    """Een dunne groep die zelf al twee facetten omvat heeft geen eenduidig
    facet om bij te horen; die blijft zoals hij is."""
    concepts = [facet_concept("A1", "Bancair", {"r1"}),
                facet_concept("A2", "Merkbeeld", {"r2"}),
                facet_concept("A3", "Bancair", {"r3", "r4"})]
    gemengd = Group(member_ids=("A1", "A2"), proposed_name="g", explanation="e")

    groups, _ = pool_thin_within_facet([gemengd, solo("A3")], concepts, threshold=4)

    assert ("A1", "A2") in [g.member_ids for g in groups]


def gemengd_concept(attribute_id, facet, pos, neg):
    p, n = frozenset(pos), frozenset(neg)
    return Concept(attribute_id=attribute_id, name=attribute_id, definition="d",
                   domain="D", facet=facet, n_iu=len(p | n), resp_ids=p | n,
                   resp_pos=p, resp_neg=n, resp_neu=frozenset())


def test_een_groep_die_alleen_op_het_TOTAAL_de_drempel_haalt_telt_als_dun():
    """De regressietest op een gemeten ontwerpfout: deze fase oordeelde eerst
    op het groepstotaal, terwijl `build_shapes` een POOL eist. Een groep met
    vier respondenten verdeeld over twee polen van twee haalt de drempel van
    vier dus niet, en moet gepoold worden in plaats van blijven staan."""
    concepts = [gemengd_concept("A1", "F", {"r1", "r2"}, {"r3", "r4"}),
                gemengd_concept("A2", "F", {"r5", "r6"}, {"r7", "r8"})]

    groups, log = pool_thin_within_facet([solo("A1"), solo("A2")], concepts,
                                         threshold=4, two_pole=True)

    assert [g.member_ids for g in groups] == [("A1", "A2")]
    assert log[0]["facet"] == "F"


# ---------------------------------------------------------------------------
# De afgevallen polen: per facet samengenomen in plaats van weggevallen.
# ---------------------------------------------------------------------------

def minderheids_concept(attribute_id, naam, facet, pos=0, neg=0):
    """Concept met een expliciet facet en een pos/neg-verdeling.

    Heet niet `concept`: die naam is in deze module al bezet (regel 19) en een
    tweede definitie zou de eerste stil overschrijven — inclusief
    `concepts_for`, dat er tien tests eerder op leunt.
    """
    p = frozenset(f"p{attribute_id}{i}" for i in range(pos))
    n = frozenset(f"n{attribute_id}{i}" for i in range(neg))
    return Concept(attribute_id=attribute_id, name=naam, definition="d",
                   domain="D", facet=facet, n_iu=pos + neg,
                   resp_ids=p | n, resp_pos=p, resp_neg=n,
                   resp_neu=frozenset(), is_drain=False)


def losse_groep(attribute_id):
    return Group(member_ids=(attribute_id,), proposed_name="", explanation="")


def test_afgevallen_polen_van_hetzelfde_facet_worden_samengenomen():
    """Het materiaal ligt over groepen, niet in één groep. Op de ASN-set had
    één facet drie groepen met 3, 9 en 15 negatieve respondenten — elk te dun,
    samen 27 en dus ruim boven de drempel van 23."""
    cs = [minderheids_concept("A1", "Een", "F", pos=30, neg=5),
          minderheids_concept("A2", "Twee", "F", pos=30, neg=5)]
    groepen = [losse_groep("A1"), losse_groep("A2")]

    res = build_shapes(groepen, cs, threshold=8, two_pole=True)

    # beide niet-negatieve polen halen 30 en worden hoofdcode; de twee
    # negatieve polen van 5 halen los niets, samen 10 en dus wel
    hoofd = [s for s in res.shapes if s.origin != "child"]
    assert len(hoofd) == 3
    assert sum(1 for s in hoofd if s.valence == "negative") == 1


def test_een_unie_die_de_drempel_haalt_wordt_een_HOOFDcode():
    """Eén drempel, één regel. `pool_thin_within_facet` levert vandaag al
    hoofdcodes op uit een facetpool; twee verschillende regels voor dezelfde
    constructie zou inconsistent zijn."""
    cs = [minderheids_concept("A1", "Een", "F", pos=30, neg=5),
          minderheids_concept("A2", "Twee", "F", pos=30, neg=5)]
    groepen = [losse_groep("A1"), losse_groep("A2")]

    res = build_shapes(groepen, cs, threshold=8, two_pole=True)

    unie = next(s for s in res.shapes if s.valence == "negative")
    assert unie.origin != "child"
    assert unie.members == ("A1", "A2")
    assert len(unie.resp_ids) == 10
    assert res.coverage_recovered == 10


def test_first_time_covered_is_a_set_difference_not_the_bucket_size():
    """`coverage_recovered` is de omvang van de unie; `first_time_covered` is
    ervan AFGETROKKEN wie al ergens stond. Vijf van de tien respondenten in de
    negatieve unie zijn dezelfde mensen als vijf die de niet-negatieve pool van
    hetzelfde attribuut al `solo` maakte — diezelfde persoon prees en
    bekritiseerde hetzelfde attribuut. Werd `first_time_covered` per ongeluk
    gelijk aan `coverage_recovered` (de emmer in plaats van het verschil), dan
    zou dit 10 zijn in plaats van 5."""
    a1 = Concept(
        attribute_id="A1", name="Een", definition="d", domain="D", facet="F",
        n_iu=35,
        resp_pos=frozenset(f"p1_{i}" for i in range(30)),
        resp_neg=frozenset(["p1_0", "p1_1", "p1_2", "n1_0", "n1_1"]),
        resp_neu=frozenset(),
        resp_ids=frozenset(f"p1_{i}" for i in range(30)) | {"n1_0", "n1_1"},
        is_drain=False,
    )
    a2 = Concept(
        attribute_id="A2", name="Twee", definition="d", domain="D", facet="F",
        n_iu=35,
        resp_pos=frozenset(f"p2_{i}" for i in range(30)),
        resp_neg=frozenset(["p2_0", "p2_1", "n2_0", "n2_1", "n2_2"]),
        resp_neu=frozenset(),
        resp_ids=frozenset(f"p2_{i}" for i in range(30)) | {"n2_0", "n2_1", "n2_2"},
        is_drain=False,
    )
    groepen = [losse_groep("A1"), losse_groep("A2")]

    res = build_shapes(groepen, [a1, a2], threshold=8, two_pole=True)

    unie = next(s for s in res.shapes if s.valence == "negative")
    assert unie.origin != "child"
    assert len(unie.resp_ids) == 10          # p1_0/1/2, p2_0/1, n1_0/1, n2_0/1/2
    assert res.coverage_recovered == 10
    assert res.first_time_covered == 5       # alleen de n1_*/n2_* zijn nieuw


def test_een_unie_eronder_wordt_een_kind():
    """origin == "child", en de vorm draagt de gepoolde respondenten."""
    cs = [minderheids_concept("A1", "Een", "F", pos=30, neg=2),
          minderheids_concept("A2", "Twee", "F", pos=30, neg=2)]
    groepen = [losse_groep("A1"), losse_groep("A2")]

    # unie = 4: boven de bodem (3), onder de drempel (8)
    res = build_shapes(groepen, cs, threshold=8, two_pole=True)

    kind = next(s for s in res.shapes if s.valence == "negative")
    assert kind.origin == "child"
    assert len(kind.resp_ids) == 4
    assert kind.resp_neg == kind.resp_ids
    assert res.coverage_recovered == 4


def test_een_unie_onder_de_bodem_wordt_echt_overig():
    """Bodem is t_keep_min_respondents — bestaande constante, geen nieuwe knop.

    Dekt sinds 2026-08-22 BEIDE routes naar echt-overig, want de verbreding gaf
    er een tweede bij. A1/A2 hebben een overlevende zusterpool; A3 heeft er geen
    en kwam onder de smalle regel als hele groep in Overig terecht. Nu levert
    ook A3 zijn pool aan de facetpool, en dan is Overig geen vanzelfsprekendheid
    meer maar het gevolg van de bodem — precies het gat waar een attribuut stil
    zou kunnen verdwijnen. Eigen facet, anders zou A3's ene respondent de unie
    van F op de bodem tillen en er een kind van maken.
    """
    cs = [minderheids_concept("A1", "Een", "F", pos=30, neg=1),
          minderheids_concept("A2", "Twee", "F", pos=30, neg=1),
          minderheids_concept("A3", "Drie", "G", neg=1)]
    groepen = [losse_groep("A1"), losse_groep("A2"), losse_groep("A3")]

    # unie F = 2 en unie G = 1, allebei onder de bodem van 3
    res = build_shapes(groepen, cs, threshold=8, two_pole=True)

    assert [s.valence for s in res.shapes] == ["non_negative", "non_negative"]
    assert res.overig_ids == ["A1", "A2", "A3"]
    assert res.coverage_recovered == 0


def test_de_unie_telt_een_gedeelde_respondent_een_keer():
    """Wie in twee groepen van hetzelfde facet een negatief idee had telt één
    keer. Op de ASN-set was de som 29 en de unie 27.

    Heet niet `test_respondenten_worden_verenigd_niet_opgeteld`, zoals de
    taakopdracht schetste: die naam draagt in deze module al de gelijknamige
    eis voor `pool_thin_within_facet`, en een tweede definitie zou de eerste
    stil vervangen.
    """
    gedeeld = frozenset({"r1", "r2", "r3"})
    a = minderheids_concept("A1", "Een", "F", pos=30)
    b = minderheids_concept("A2", "Twee", "F", pos=30)
    a = replace(a, resp_neg=gedeeld, resp_ids=a.resp_ids | gedeeld)
    b = replace(b, resp_neg=gedeeld, resp_ids=b.resp_ids | gedeeld)
    groepen = [losse_groep("A1"), losse_groep("A2")]

    res = build_shapes(groepen, [a, b], threshold=8, two_pole=True)

    kind = next(s for s in res.shapes if s.valence == "negative")
    assert len(kind.resp_ids) == 3      # niet 6
    assert kind.origin == "child"


def test_een_groep_zonder_eenduidig_facet_gaat_rechtstreeks_naar_overig():
    """`pool_thin_within_facet` laat zo'n groep met rust omdat er geen facet is
    om op te groeperen; dezelfde regel geldt hier."""
    # neg=3 per attribuut: samen 6, dus de negatieve pool van de groep valt af
    # (drempel 8). Zou hij het halen, dan was er niets om te poolen.
    cs = [minderheids_concept("A1", "Een", "F", pos=30, neg=3),
          minderheids_concept("A2", "Twee", "G", pos=30, neg=3)]
    groepen = [Group(member_ids=("A1", "A2"), proposed_name="", explanation="")]

    res = build_shapes(groepen, cs, threshold=8, two_pole=True)

    assert [s.valence for s in res.shapes] == ["non_negative"]
    assert res.overig_ids == ["A1", "A2"]
    assert res.coverage_recovered == 0


def test_de_bestaande_groepen_blijven_ongemoeid():
    """Deze operatie voegt vormen toe en verandert de indeling NIET. De groepen
    waarvan we de minderheidspool oppakken zijn op hun andere kant een
    volwaardige hoofdcode."""
    cs = [minderheids_concept("A1", "Een", "F", pos=30, neg=5),
          minderheids_concept("A2", "Twee", "F", pos=30, neg=5)]
    groepen = [losse_groep("A1"), losse_groep("A2")]

    res = build_shapes(groepen, cs, threshold=8, two_pole=True)

    eigen = [s for s in res.shapes if s.valence == "non_negative"]
    assert [s.members for s in eigen] == [("A1",), ("A2",)]
    assert [len(s.resp_ids) for s in eigen] == [30, 30]
    assert all(s.resp_neg == frozenset() for s in eigen)


def test_een_herstelde_hoofdcode_draagt_een_eigen_herkomst():
    """`origin == "recovered"`, en dus niet `"pooled"`.

    Taak 2 gaf een unie die de drempel haalde `"pooled"` — dezelfde herkomst
    als een door het model voorgestelde samenvoeging, en daarmee vetobaar in
    `codebook_writer`. Bij veto stonden de respondenten wéér nergens: in de
    regel blijft het attribuut bron van zijn overlevende zusterpool, en anders
    — sinds de verbreding van 2026-08-22 levert ook een groep zonder overlevende
    pool aan — ononderscheiden in Overig. Een facetunie is bovendien geen
    modelvoorstel maar step 4's eigen structuur; het veto beoordeelt daarmee
    iets wat het niet beoordeelt.
    """
    cs = [minderheids_concept("A1", "Een", "F", pos=30, neg=5),
          minderheids_concept("A2", "Twee", "F", pos=30, neg=5)]
    groepen = [losse_groep("A1"), losse_groep("A2")]

    res = build_shapes(groepen, cs, threshold=8, two_pole=True)

    unie = next(s for s in res.shapes if s.valence == "negative")
    assert unie.origin == "recovered"


def test_geen_enkele_vorm_uit_de_facetpool_is_vetobaar():
    """De unie wordt hoofdcode óf kind; geen van beide mag `"pooled"` heten,
    want dat is de enige herkomst die `codebook_writer` mag weigeren.

    Toetst de eigenschap en niet één tak ervan: een vorm uit deze pool bestaat
    omdat zijn respondenten anders onder een code met de tegengestelde richting
    geteld worden, en dat geldt aan beide kanten van de drempel.
    """
    cs = [minderheids_concept("A1", "Een", "F", pos=30, neg=5),
          minderheids_concept("A2", "Twee", "F", pos=30, neg=5),
          minderheids_concept("A3", "Drie", "G", pos=30, neg=2),
          minderheids_concept("A4", "Vier", "G", pos=30, neg=2)]
    groepen = [losse_groep(i) for i in ("A1", "A2", "A3", "A4")]

    # facet F: unie 10, boven de drempel → hoofdcode
    # facet G: unie 4, boven de bodem (3) en onder de drempel → kind
    res = build_shapes(groepen, cs, threshold=8, two_pole=True)

    uit_de_pool = [s for s in res.shapes if s.valence == "negative"]
    assert sorted(s.origin for s in uit_de_pool) == ["child", "recovered"]


def test_een_groep_zonder_overlevende_pool_levert_zijn_polen_ook_aan_de_pool():
    """Ook een groep waar GEEN pool de drempel haalt draagt bij aan de facetpool.

    Taak 2 verzamelde afgevallen polen alleen uit groepen waar een zusterpool
    overleefde: daar telt kritiek mee onder een code die het tegenovergestelde
    beweert, en dat is het scherpste defect. Op 2026-08-22 is dat besluit
    verbreed. Een groep die in zijn geheel in Overig verdween liet zijn
    minderheidsmateriaal ononderscheiden achter, terwijl het doel is dat zulk
    materiaal een eigen naam krijgt — desnoods als kind onder Overig. De grens
    blijft het facet; alleen de herkomst van de afgevallen pool telt niet meer
    mee.

    Hier haalt geen van beide groepen iets: elk attribuut heeft alleen een
    negatieve pool van 5, onder de drempel van 8. Samen halen ze 10 en worden
    ze één herstelde hoofdcode in plaats van twee naamloze Overig-attributen.
    """
    cs = [minderheids_concept("A1", "Een", "F", neg=5),
          minderheids_concept("A2", "Twee", "F", neg=5)]
    groepen = [losse_groep("A1"), losse_groep("A2")]

    res = build_shapes(groepen, cs, threshold=8, two_pole=True)

    unie = next(s for s in res.shapes if s.valence == "negative")
    assert unie.origin == "recovered"
    assert unie.members == ("A1", "A2")
    assert len(unie.resp_ids) == 10
    assert res.overig_ids == []
    assert res.coverage_recovered == 10


def test_zonder_overlevende_pool_blijft_de_facetgrens_toch_de_grens():
    """De verbreding van 2026-08-22 raakt WELKE polen worden verzameld, niet
    waar ze samen mogen komen. Het facet blijft de enige groeperingsgrens.

    Deze groep omvat twee facetten en heeft geen enkele overlevende pool — het
    geval dat onder de smalle regel niet bestond, want zo'n groep ging in zijn
    geheel naar Overig zonder ooit langs `pool_minority_poles` te komen. Ook nu
    gaat hij daarheen, maar via de andere route: geen eenduidig facet, dus
    rechtstreeks naar overig in plaats van gepoold. De boekhouding blijft heel.

    neg=3 per attribuut: de negatieve pool van de GROEP is 6 en haalt de drempel
    van 8 niet. Haalde hij hem wel, dan was het een gewone `pooled` hoofdcode en
    viel er niets te poolen.
    """
    cs = [minderheids_concept("A1", "Een", "F", neg=3),
          minderheids_concept("A2", "Twee", "G", neg=3)]
    groepen = [Group(member_ids=("A1", "A2"), proposed_name="", explanation="")]

    res = build_shapes(groepen, cs, threshold=8, two_pole=True)

    assert res.shapes == []
    assert res.overig_ids == ["A1", "A2"]
    assert res.coverage_recovered == 0
