"""Tests voor de codeboek-reassemblage in run_codeGenerator.py — het
samenvoegen van geschreven tekst terug op de vorm waar hij bij hoort, na de
MECE-ronde.

De echte fout (live ASN-run): twee compleet verschillende vormen kregen
dezelfde geschreven codenaam ("Stijl en merkbeleving" / "Merkuitstraling en
stijl" was zo'n paar). Een woordenboek gekeyd op die naam
(`{code.code_name: code for code in codes}`) versmelt de twee entries tot één
object zodra dat gebeurt — beide vormen erven dan de definitie van welke van
de twee toevallig als laatste in de iteratie stond, inclusief de vorm wiens
eigen leden die tekst niet beschrijven. `_index_codes_by_shape_key` keyt in
plaats daarvan op `shape.key` (uniek per run, nooit hergebruikt), dus een
naamcollision kan de mapping niet meer laten instorten.
"""
from pipeline.step_5_codeGenerator import run_codeGenerator as rcg
from pipeline.step_5_codeGenerator.concept_inventory import Concept
from pipeline.step_5_codeGenerator.consolidator import CodeShape
from pipeline.step_5_codeGenerator.prompts_codeGenerator import ConsolidatedCode


def concept(attribute_id, name, n_resp=10):
    resp = frozenset(f"R{i}" for i in range(n_resp))
    return Concept(attribute_id=attribute_id, name=name, definition="def",
                   domain="Domein", facet="Facet", n_iu=n_resp,
                   resp_ids=resp, resp_pos=resp,
                   resp_neg=frozenset(), resp_neu=frozenset())


def shape(key, valence, umbrella, members, n_resp=40, origin="solo"):
    resp = frozenset(f"R{i}" for i in range(n_resp))
    return CodeShape(key=key, members=tuple(members), valence=valence,
                     umbrella=umbrella, resp_ids=resp, resp_pos=resp,
                     resp_neg=frozenset(), resp_neu=frozenset(), origin=origin)


def test_index_codes_by_shape_key_keeps_two_same_named_codes_distinct():
    # Two shapes with entirely different members — the writer coincidentally
    # gave them the identical code_name in one batch call.
    shape_a = shape("K1", "positive", "u1", ["A1"], n_resp=94)  # "Stijl en merkbeleving"
    shape_b = shape("K2", "positive", "u2", ["A2"], n_resp=33)  # "Merkuitstraling en stijl"
    concept_by_id = {"A1": concept("A1", "Commerciële gerichtheid"),
                      "A2": concept("A2", "Natuurlijke uitstraling")}
    code_a = ConsolidatedCode(code_name="Stijl en merkbeleving", definition="def over A1",
                              diagnostic_test="t", valence="positive",
                              typical_indicators=["x"], source_attributes=["Commerciële gerichtheid"])
    code_b = ConsolidatedCode(code_name="Stijl en merkbeleving", definition="def over A2",
                              diagnostic_test="t", valence="positive",
                              typical_indicators=["y"], source_attributes=["Natuurlijke uitstraling"])
    lookup = rcg._shape_lookup([shape_a, shape_b], concept_by_id)

    indexed = rcg._index_codes_by_shape_key([code_a, code_b], lookup)

    assert indexed[shape_a.key].definition == "def over A1"
    assert indexed[shape_b.key].definition == "def over A2"

    # The mechanism this replaces: a dict keyed on the written name instead
    # of the shape collapses the two entries into one — both shapes would
    # then resolve to whichever code happened to be written last.
    code_by_name = {c.code_name: c for c in [code_a, code_b]}
    assert code_by_name[code_a.code_name] is code_by_name[code_b.code_name]
    assert len(code_by_name) == 1


def test_index_codes_by_shape_key_is_unaffected_by_a_name_collision_among_others():
    # A third, uniquely-named shape is unaffected by a collision elsewhere.
    shape_a = shape("K1", "positive", "u1", ["A1"], n_resp=94)
    shape_b = shape("K2", "positive", "u2", ["A2"], n_resp=33)
    shape_c = shape("K3", "neutral", "u3", ["A3"], n_resp=10)
    concept_by_id = {"A1": concept("A1", "N1"), "A2": concept("A2", "N2"), "A3": concept("A3", "N3")}
    code_a = ConsolidatedCode(code_name="Dup", definition="def A", diagnostic_test="t",
                              valence="positive", typical_indicators=["x"], source_attributes=["N1"])
    code_b = ConsolidatedCode(code_name="Dup", definition="def B", diagnostic_test="t",
                              valence="positive", typical_indicators=["y"], source_attributes=["N2"])
    code_c = ConsolidatedCode(code_name="Uniek", definition="def C", diagnostic_test="t",
                              valence="neutral", typical_indicators=["z"], source_attributes=["N3"])
    lookup = rcg._shape_lookup([shape_a, shape_b, shape_c], concept_by_id)

    indexed = rcg._index_codes_by_shape_key([code_a, code_b, code_c], lookup)

    assert indexed[shape_c.key].definition == "def C"
    assert indexed[shape_a.key].definition == "def A"
    assert indexed[shape_b.key].definition == "def B"
