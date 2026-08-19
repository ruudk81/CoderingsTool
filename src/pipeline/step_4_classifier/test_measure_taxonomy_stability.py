"""Tests voor het stabiliteitsinstrument van step 4.

Een meetinstrument dat stil verkeerd meet is erger dan geen instrument: de
uitkomst gaat hier een ontwerpbeslissing dragen (snijden door de boom, ja of
nee), en een ARI van 0,4 die eigenlijk een leesfout is, sluit dat ontwerp af op
grond van niets. Vandaar dat hier de afleiding wordt vastgelegd, niet de
getallen — die komen uit de runs.
"""
from types import SimpleNamespace

from pipeline.step_4_classifier.measure_taxonomy_stability import (
    _placement_by_attribute, block_move_violations, build_snapshot,
    inherited_spread, thin_facets,
)


def idea(idea_id, attribute_id, instance="", interpretation=""):
    return SimpleNamespace(idea_id=idea_id, attribute_id=attribute_id, idea=instance,
                           instance=instance, interpretation=interpretation)


def response(respondent_id, *ideas):
    return SimpleNamespace(respondent_id=respondent_id, response_ideas=list(ideas))


def taxonomy(structure):
    """structure: {domein: {facet: [attribuut-id, ...]}}"""
    return SimpleNamespace(partition_results={
        domain: {"attributes": {
            facet: [{"attribute_id": a, "attribute_name": a} for a in attrs]
            for facet, attrs in facets.items()}}
        for domain, facets in structure.items()})


BOOM = taxonomy({"Duurzaamheid": {"Ecologie": ["A1", "A2"], "Ethiek": ["A3"]},
                 "Bank": {"Producten": ["A4"]}})


def test_placement_komt_uit_de_structuur_niet_uit_het_idee():
    """De velden op het idee kunnen achterlopen op de boom; de structuur is de
    afgesproken enige bron. Een idee dat zijn oude facetnaam meedraagt mag de
    meting niet met een fantoomfacet vervuilen."""
    placement = _placement_by_attribute(BOOM)

    assert placement["A1"] == ("Duurzaamheid", "Ecologie")
    assert placement["A3"] == ("Duurzaamheid", "Ethiek")
    assert placement["A4"] == ("Bank", "Producten")


def test_idee_zonder_bekend_attribuut_telt_als_ongeplaatst():
    """Niet stil weglaten: een idee dat nergens landt is informatie over de run,
    en het verschil tussen 'ongeplaatst' en 'bestaat niet' bepaalt of de ARI
    over dezelfde eenheden gaat."""
    snapshot = build_snapshot(
        [response("r1", idea("r1_1", "A1", "groen"), idea("r1_2", "", "leeg"),
                  idea("r1_3", "A99", "verdwenen"))], BOOM)

    assert snapshot["ideas_total"] == 3
    assert snapshot["ideas_placed"] == 1
    assert snapshot["ideas_unplaced"] == 2


def test_facet_label_bevat_het_domein():
    """Facetnamen zijn alleen binnen hun domein uniek. Zonder domeinprefix zou
    een gelijknamig facet in twee domeinen als één cluster meetellen en de ARI
    optisch stabieler maken dan de boom is."""
    boom = taxonomy({"D1": {"Zelfde": ["A1"]}, "D2": {"Zelfde": ["A2"]}})
    snapshot = build_snapshot(
        [response("r1", idea("r1_1", "A1", "x")), response("r2", idea("r2_1", "A2", "y"))],
        boom)

    assert snapshot["n_facets"] == 2
    assert set(snapshot["labels"]["facet"].values()) == {"D1 > Zelfde", "D2 > Zelfde"}


def test_respondent_telt_eenmaal_per_facet():
    """Twee ideeën van dezelfde respondent in één facet is één respondent. Het
    bereik is een unie, nooit een som — dezelfde regel als in step 5."""
    snapshot = build_snapshot(
        [response("r1", idea("r1_1", "A1", "groen"), idea("r1_2", "A2", "natuur"))],
        BOOM)

    assert snapshot["facet_reach"]["Duurzaamheid > Ecologie"] == 1
    assert snapshot["facet_attributes"]["Duurzaamheid > Ecologie"] == 2


def test_blokinvariant_meldt_niets_bij_identiek_label_en_attribuut():
    """(domein, label) -> één attribuut is hoe step 4 toewijst. Dit is de
    gezonde toestand."""
    snapshot = build_snapshot(
        [response("r1", idea("r1_1", "A1", "groen", "milieuvriendelijk")),
         response("r2", idea("r2_1", "A1", "groen", "milieuvriendelijk"))],
        BOOM)

    assert block_move_violations(snapshot)["violations"] == 0


def test_blokinvariant_meldt_identiek_label_op_twee_attributen():
    """Kan vandaag niet gebeuren — en juist daarom moet het gemeld worden als het
    tóch gebeurt: dan is de blokverplaatsing stuk en meet de ARI een bug in
    plaats van instabiliteit."""
    snapshot = build_snapshot(
        [response("r1", idea("r1_1", "A1", "groen", "milieuvriendelijk")),
         response("r2", idea("r2_1", "A2", "groen", "milieuvriendelijk"))],
        BOOM)

    assert block_move_violations(snapshot)["violations"] == 1


def test_gelijke_span_met_andere_interpretatie_is_geen_invariantbreuk():
    """Dezelfde span, andere interpretatie: dat zijn twee verschillende labels,
    dus twee reps, dus twee toewijzingen. Geen bug — geërfde spreiding."""
    snapshot = build_snapshot(
        [response("r1", idea("r1_1", "A1", "groen", "milieuvriendelijk")),
         response("r2", idea("r2_1", "A2", "groen", "kleur van het logo"))],
        BOOM)

    assert block_move_violations(snapshot)["violations"] == 0
    spread = inherited_spread(snapshot)
    assert spread["split_spans"] == 1
    assert spread["within_domain"] == 1
    assert spread["across_domains"] == 0


def test_geerfde_spreiding_scheidt_domeinoorzaak_van_bewoordingsoorzaak():
    """De twee oorzaken vragen om een andere ingreep, dus ze worden apart
    geteld: uiteenvallen over domeinen is step 3's toewijzing, uiteenvallen
    binnen één domein is step 3's bewoording."""
    snapshot = build_snapshot(
        [response("r1", idea("r1_1", "A1", "groen", "een")),
         response("r2", idea("r2_1", "A4", "groen", "twee")),
         response("r3", idea("r3_1", "A1", "eerlijk", "drie")),
         response("r4", idea("r4_1", "A3", "eerlijk", "vier"))],
        BOOM)
    spread = inherited_spread(snapshot)

    assert spread["split_spans"] == 2
    assert spread["across_domains"] == 1   # "groen": Duurzaamheid vs Bank
    assert spread["within_domain"] == 1    # "eerlijk": twee facetten, één domein


def test_thin_facets_gebruikt_de_drempel_van_step_5():
    """De dunne-facetten-teller is de tweede falsificatiegrens van het
    snede-ontwerp, dus hij moet op step 5's codedrempel staan en niet op een
    eigen getal."""
    responses = [response(f"r{i}", idea(f"r{i}_1", "A1", "x")) for i in range(200)]
    responses.append(response("r200", idea("r200_1", "A3", "y")))  # facet met 1
    snapshot = build_snapshot(responses, BOOM)

    thin, total = thin_facets(snapshot)
    assert total == 2                # alleen facetten met ideeën tellen mee
    assert thin == 1                 # 1 respondent haalt de bodem van 3 niet
