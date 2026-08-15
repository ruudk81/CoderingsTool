"""Tests for merging and direction (step 3 of step 5)."""
from pipeline.step_5_codeGenerator.concept_inventory import Concept
from pipeline.step_5_codeGenerator.consolidator import consolidate


def concept(attribute_id, name, pos=0, neg=0, neu=0):
    def resp(prefix, n):
        return frozenset(f"{attribute_id}{prefix}{i}" for i in range(n))
    p, g, u = resp("P", pos), resp("G", neg), resp("U", neu)
    return Concept(attribute_id=attribute_id, name=name, definition="d",
                   domain="D", facet="F", n_iu=pos + neg + neu,
                   resp_ids=p | g | u, resp_pos=p, resp_neg=g, resp_neu=u)


def relations(mapping, synonyms=None):
    """mapping: attribute_id -> umbrella; synonyms: attribute_id -> attribute_id."""
    from pipeline.step_5_codeGenerator.consolidator import RelationMap
    return RelationMap(umbrella=mapping, synonym_of=synonyms or {})


def test_large_concept_gets_its_own_code_and_does_not_climb():
    concepts = [concept("A1", "Duurzaamheid", pos=240),
                concept("A2", "Goede doelen", pos=14),
                concept("A3", "Sponsoring", pos=9),
                concept("A4", "Eerlijke handel", pos=8)]
    rel = relations({"A1": "maatschappelijk", "A2": "maatschappelijk",
                     "A3": "maatschappelijk", "A4": "maatschappelijk"})
    shapes, overig = consolidate(concepts, rel, threshold=20)

    solo = [s for s in shapes if s.origin == "solo"]
    pooled = [s for s in shapes if s.origin == "pooled"]
    assert [s.members for s in solo] == [("A1",)]
    assert set(pooled[0].members) == {"A2", "A3", "A4"}
    assert overig == []


def test_two_large_concepts_never_merge():
    concepts = [concept("A1", "Prijs", pos=200), concept("A2", "Service", pos=180)]
    rel = relations({"A1": "beleving", "A2": "beleving"})
    shapes, _ = consolidate(concepts, rel, threshold=20)
    assert sorted(s.members for s in shapes) == [("A1",), ("A2",)]


def test_synonyms_merge_regardless_of_size():
    concepts = [concept("A1", "Warm en huiselijk", pos=312),
                concept("A2", "Warm en menselijk", pos=287)]
    rel = relations({"A1": "sfeer", "A2": "sfeer"}, synonyms={"A2": "A1"})
    shapes, _ = consolidate(concepts, rel, threshold=20)
    assert len(shapes) == 1
    assert set(shapes[0].members) == {"A1", "A2"}
    assert shapes[0].origin == "synonym"


def test_merged_respondents_are_a_union_not_a_sum():
    shared = concept("A1", "Prijs", pos=10)
    same = Concept(attribute_id="A2", name="Kosten", definition="d", domain="D",
                   facet="F", n_iu=10, resp_ids=shared.resp_ids,
                   resp_pos=shared.resp_pos, resp_neg=frozenset(),
                   resp_neu=frozenset())
    rel = relations({"A1": "geld", "A2": "geld"}, synonyms={"A2": "A1"})
    shapes, _ = consolidate([shared, same], rel, threshold=5)
    assert len(shapes[0].resp_ids) == 10  # niet 20


def test_pool_below_threshold_goes_to_overig():
    concepts = [concept("A1", "Website", pos=4), concept("A2", "Reclame", pos=3)]
    rel = relations({"A1": "overig contact", "A2": "overig contact"})
    shapes, overig = consolidate(concepts, rel, threshold=20)
    assert shapes == []
    assert sorted(overig) == ["A1", "A2"]


def test_both_poles_above_threshold_gives_two_codes():
    concepts = [concept("A1", "Prijsniveau", pos=60, neg=80)]
    rel = relations({"A1": "prijs"})
    shapes, _ = consolidate(concepts, rel, threshold=20)
    assert sorted(s.valence for s in shapes) == ["negative", "positive"]


def test_one_pole_above_threshold_gives_one_directed_code():
    concepts = [concept("A1", "Prijsniveau", pos=5, neg=80)]
    rel = relations({"A1": "prijs"})
    shapes, _ = consolidate(concepts, rel, threshold=20)
    assert [s.valence for s in shapes] == ["negative"]


def test_no_pole_above_threshold_gives_one_neutral_code():
    concepts = [concept("A1", "Prijsniveau", pos=15, neg=15, neu=5)]
    rel = relations({"A1": "prijs"})
    shapes, _ = consolidate(concepts, rel, threshold=20)
    assert [s.valence for s in shapes] == ["neutral"]


def test_neutral_pool_above_threshold_gets_its_own_code():
    concepts = [concept("A1", "Prijsniveau", pos=60, neg=80, neu=40)]
    rel = relations({"A1": "prijs"})
    shapes, _ = consolidate(concepts, rel, threshold=20)
    assert sorted(s.valence for s in shapes) == ["negative", "neutral", "positive"]


def test_normalize_maps_qualified_names_back_to_ids():
    from pipeline.step_5_codeGenerator.consolidator import normalize_relations
    concepts = [concept("A1", "Prijs"), concept("A2", "Kosten")]

    class Result:
        relations = [
            type("R", (), {"attribute": "[A1] Prijs", "synonym_of": None,
                           "umbrella_name": "geld", "umbrella_definition": "d"})(),
            type("R", (), {"attribute": "[A2] Kosten", "synonym_of": "[A1] Prijs",
                           "umbrella_name": "geld", "umbrella_definition": "d"})(),
        ]

    rel = normalize_relations(Result(), concepts)
    assert rel.umbrella == {"A1": "geld", "A2": "geld"}
    assert rel.synonym_of == {"A2": "A1"}


def test_normalize_ignores_a_synonym_pointing_outside_the_list():
    from pipeline.step_5_codeGenerator.consolidator import normalize_relations
    concepts = [concept("A1", "Prijs")]

    class Result:
        relations = [
            type("R", (), {"attribute": "[A1] Prijs", "synonym_of": "[A99] Bestaat niet",
                           "umbrella_name": "geld", "umbrella_definition": "d"})(),
        ]

    assert normalize_relations(Result(), concepts).synonym_of == {}
