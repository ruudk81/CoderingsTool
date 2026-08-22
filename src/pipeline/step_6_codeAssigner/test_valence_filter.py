"""Tests for the opposing-pole filter (step 6, after assignment).

The filter moves an idea to Overig when it sits under a code carrying the
OPPOSITE direction. Everything else stays put — the point is to remove active
errors ("slecht bereikbaar" counted under a code labelled positive), not to
empty the codebook.
"""

import pytest

from models import CodeAssignedModel, CodeAssignedSubmodel, ConsolidatedCode
from pipeline.step_6_codeAssigner.valence_filter import (
    carries,
    find_overig_code,
    opposes,
    route_opposing_poles,
)


# =============================================================================
# FIXTURES
# =============================================================================

def code(name: str, valence: str, code_id: str) -> ConsolidatedCode:
    return ConsolidatedCode(
        code_name=name, definition="d", diagnostic_test="t",
        valence=valence, typical_indicators=[], code_id=code_id,
    )


def idea(idea_id: str, valence: str, code_name: str, code_id: str) -> CodeAssignedSubmodel:
    return CodeAssignedSubmodel(
        idea_id=idea_id, idea="i", valence=valence,
        assigned_code=code_name, assigned_code_id=code_id,
    )


def respondent(*ideas) -> CodeAssignedModel:
    return CodeAssignedModel(respondent_id=1, response="r", response_ideas=list(ideas))


CODES = [
    code("Sterke klantrelatie", "positive", "K1"),
    code("Ongunstige voorwaarden", "negative", "K2"),
    code("Sparen en rente", "neutral", "K3"),
    code("Overig", "neutral", "K9"),
]


# =============================================================================
# THE DIRECTION RULE
# =============================================================================

@pytest.mark.parametrize("code_valence,idea_valence,verwacht", [
    ("positive", "-", True),      # negatief idee onder positief label — de fout
    ("negative", "+", True),      # en andersom
    ("positive", "+", False),
    ("negative", "-", False),
    ("positive", "0", False),     # beschrijvend, geen oordeel
    ("negative", "0", False),
    ("positive", "", False),      # geen valentie gemeten — nooit verhuizen
    ("negative", "", False),
    ("neutral", "+", False),      # neutrale code doet geen richtingsuitspraak
    ("neutral", "-", False),
])
def test_opposes(code_valence, idea_valence, verwacht):
    assert opposes(code_valence, idea_valence) is verwacht


def test_opposes_negeert_onbekende_valentie():
    """Een onverwachte waarde is geen tegenpool — geen else-tak die alles vangt."""
    assert opposes("positive", "negatief") is False
    assert opposes("gemengd", "-") is False


# =============================================================================
# OVERIG OPZOEKEN
# =============================================================================

def test_find_overig_code():
    assert find_overig_code(CODES).code_id == "K9"


def test_find_overig_code_taalvarianten():
    """De catch-all heet niet in elke taal 'Overig' — zelfde regel als de assigner."""
    engels = [code("Sterke klantrelatie", "positive", "K1"),
              code("Other", "neutral", "K9")]
    assert find_overig_code(engels).code_id == "K9"
    duits = [code("Sonstiges", "neutral", "K9")]
    assert find_overig_code(duits).code_id == "K9"


def test_find_overig_code_afwezig():
    assert find_overig_code([code("Sterke klantrelatie", "positive", "K1")]) is None


# =============================================================================
# DE PAS
# =============================================================================

def test_verhuist_negatief_idee_onder_positieve_code():
    resp = [respondent(idea("i1", "-", "Sterke klantrelatie", "K1"))]
    rapport = route_opposing_poles(resp, CODES)

    verhuisd = resp[0].response_ideas[0]
    assert verhuisd.assigned_code == "Overig"
    assert verhuisd.assigned_code_id == "K9"
    assert rapport.moved == 1


def test_verhuist_positief_idee_onder_negatieve_code():
    resp = [respondent(idea("i1", "+", "Ongunstige voorwaarden", "K2"))]
    route_opposing_poles(resp, CODES)
    assert resp[0].response_ideas[0].assigned_code_id == "K9"


def test_werkt_naam_en_id_allebei_bij():
    """view_codebook joint op assigned_code_id maar valt terug op assigned_code:
    laat je er één achter, dan telt hetzelfde idee twee kanten op."""
    resp = [respondent(idea("i1", "-", "Sterke klantrelatie", "K1"))]
    route_opposing_poles(resp, CODES)
    i = resp[0].response_ideas[0]
    assert (i.assigned_code, i.assigned_code_id) == ("Overig", "K9")


@pytest.mark.parametrize("valence,code_name,code_id", [
    ("+", "Sterke klantrelatie", "K1"),      # richting klopt
    ("0", "Sterke klantrelatie", "K1"),      # beschrijvend
    ("", "Sterke klantrelatie", "K1"),       # niet gemeten
    ("-", "Sparen en rente", "K3"),          # neutrale code
    ("+", "Sparen en rente", "K3"),
])
def test_laat_de_rest_staan(valence, code_name, code_id):
    resp = [respondent(idea("i1", valence, code_name, code_id))]
    rapport = route_opposing_poles(resp, CODES)
    assert resp[0].response_ideas[0].assigned_code_id == code_id
    assert rapport.moved == 0


def test_idee_al_bij_overig_blijft_en_telt_niet_mee():
    resp = [respondent(idea("i1", "-", "Overig", "K9"))]
    rapport = route_opposing_poles(resp, CODES)
    assert rapport.moved == 0
    assert rapport.overig_after == rapport.overig_before == 1


def test_ongecodeerd_idee_blijft():
    resp = [respondent(idea("i1", "-", "__UNASSIGNED__", "__UNASSIGNED__"))]
    rapport = route_opposing_poles(resp, CODES)
    assert rapport.moved == 0
    assert resp[0].response_ideas[0].assigned_code_id == "__UNASSIGNED__"


def test_zonder_overig_gebeurt_er_niets():
    """Geen catch-all om naartoe te routeren: de pas doet niets en zegt dat."""
    zonder = [code("Sterke klantrelatie", "positive", "K1")]
    resp = [respondent(idea("i1", "-", "Sterke klantrelatie", "K1"))]
    rapport = route_opposing_poles(resp, zonder)
    assert rapport.skipped_no_overig is True
    assert rapport.moved == 0
    assert resp[0].response_ideas[0].assigned_code_id == "K1"


# =============================================================================
# RAPPORTAGE
# =============================================================================

def test_telt_per_code_van_herkomst():
    resp = [respondent(
        idea("i1", "-", "Sterke klantrelatie", "K1"),
        idea("i2", "-", "Sterke klantrelatie", "K1"),
        idea("i3", "+", "Ongunstige voorwaarden", "K2"),
        idea("i4", "+", "Sterke klantrelatie", "K1"),   # blijft
    )]
    rapport = route_opposing_poles(resp, CODES)
    assert rapport.moved == 3
    assert rapport.per_code == {"Sterke klantrelatie": 2, "Ongunstige voorwaarden": 1}


def test_overig_aandeel_voor_en_na():
    resp = [respondent(
        idea("i1", "-", "Sterke klantrelatie", "K1"),
        idea("i2", "+", "Sterke klantrelatie", "K1"),
        idea("i3", "0", "Overig", "K9"),
    )]
    rapport = route_opposing_poles(resp, CODES)
    assert (rapport.overig_before, rapport.overig_after) == (1, 2)
    assert rapport.coded_ideas == 3
    assert rapport.overig_share_after == pytest.approx(2 / 3)


def test_meten_muteert_niet_maar_telt_hetzelfde():
    """apply=False is dezelfde code, alleen zonder de schrijfactie — anders
    kan de meting gaan afwijken van wat de filter werkelijk doet."""
    def verse_responses():
        return [respondent(
            idea("i1", "-", "Sterke klantrelatie", "K1"),
            idea("i2", "+", "Ongunstige voorwaarden", "K2"),
        )]

    droog_resp = verse_responses()
    droog = route_opposing_poles(droog_resp, CODES, apply=False)
    nat = route_opposing_poles(verse_responses(), CODES, apply=True)

    assert droog == nat
    assert [i.assigned_code_id for i in droog_resp[0].response_ideas] == ["K1", "K2"]


def test_lege_invoer():
    rapport = route_opposing_poles([], CODES)
    assert rapport.moved == 0
    assert rapport.overig_share_after == 0.0


def test_een_negatief_idee_botst_met_een_niet_negatieve_code():
    """De tweedeling maakt `non_negative` (positief ∪ neutraal). Zo'n code zegt
    letterlijk dat een klacht er niet in hoort, dus een negatief idee is daar
    wél een conflict — anders dan bij een echt neutrale code.

    Tot 2026-08-22 werd `non_negative` als `neutral` opgeslagen omdat het
    contract maar drie waarden kende, en dan vuurde deze bewaking nooit. Dat is
    het pad dat in de praktijk gedraaid werd."""
    assert opposes("non_negative", "-") is True


def test_een_niet_negatieve_code_botst_niet_met_positief_of_beschrijvend():
    assert opposes("non_negative", "+") is False
    assert opposes("non_negative", "0") is False


def test_een_echt_neutrale_code_botst_met_niets():
    """Beschrijvend materiaal heeft geen tegenpool. Dat `neutral` buiten de
    tabel valt is dus correct en moet zo blijven — het onderscheid met
    `non_negative` is precies waarom er een vierde waarde nodig was."""
    assert opposes("neutral", "-") is False
    assert opposes("neutral", "+") is False


def test_een_negatieve_code_botst_ook_zonder_positieve_code_in_het_boek():
    """Er wordt vergeleken met de POOL van het idee, niet met de valentie van
    een andere code. In een tweedelingscodeboek bestaat geen positieve code, en
    toch hoort een positief idee niet onder een negatieve."""
    assert opposes("negative", "+") is True


# =============================================================================
# KINDEREN ONDER OVERIG
# =============================================================================

def kind(name: str, valence: str, code_id: str, attr_ids,
         parent: str = "K9") -> ConsolidatedCode:
    """Een restcategorie onder Overig: volwaardige code, maar met een ouder.
    Zijn `source_attribute_ids` zijn de attributen van het facet waarvan deze
    pool de drempel niet haalde — dat is de aanhechting die de router gebruikt."""
    return ConsolidatedCode(
        code_name=name, definition="d", diagnostic_test="t",
        valence=valence, typical_indicators=[], code_id=code_id,
        parent_code_id=parent, source_attribute_ids=list(attr_ids),
    )


def idee_met_attribuut(idea_id: str, valence: str, code_name: str,
                       code_id: str, attribute_id) -> CodeAssignedSubmodel:
    return CodeAssignedSubmodel(
        idea_id=idea_id, idea="i", valence=valence,
        assigned_code=code_name, assigned_code_id=code_id,
        attribute_id=attribute_id,
    )


# Een codeboek met kinderen: K1 draait op attribuut A1, en de negatieve pool van
# datzelfde attribuut kreeg geen eigen kop maar een kind (K31).
CODES_MET_KIND = [
    code("Sterke klantrelatie", "positive", "K1"),
    code("Ongunstige voorwaarden", "negative", "K2"),
    kind("Negatieve overige klantrelatie", "negative", "K31", ["A1"]),
    kind("Overige kostenwaardering", "non_negative", "K32", ["A2"]),
    kind("Overige beschrijving", "neutral", "K33", ["A3"]),
    code("Overig", "neutral", "K9"),
]
CODES_MET_KIND[0].source_attribute_ids = ["A1"]
CODES_MET_KIND[1].source_attribute_ids = ["A2"]


def test_tegenpool_gaat_naar_het_kind_met_de_juiste_richting():
    """Het kind bestaat precies voor dit materiaal: dezelfde attributen, de
    andere pool. Het naar de ouder sturen zou het opnieuw ononderscheiden
    maken."""
    resp = [respondent(idee_met_attribuut("i1", "-", "Sterke klantrelatie", "K1", "A1"))]
    rapport = route_opposing_poles(resp, CODES_MET_KIND)

    verhuisd = resp[0].response_ideas[0]
    assert (verhuisd.assigned_code, verhuisd.assigned_code_id) == (
        "Negatieve overige klantrelatie", "K31")
    assert rapport.moved == 1
    assert rapport.per_child["Negatieve overige klantrelatie"] == 1


def test_een_niet_negatief_kind_vangt_een_positief_idee():
    """`non_negative` is de positieve helft van een tweedelingscodeboek: dáár
    bestaat geen code met valentie `positive`. Zou de bestemmingstoets gewoon
    `not opposes()` zijn, dan zou ook een neutraal kind meetellen en zou in een
    tweedeling nooit een positieve bestemming gevonden worden."""
    resp = [respondent(idee_met_attribuut("i1", "+", "Ongunstige voorwaarden", "K2", "A2"))]
    route_opposing_poles(resp, CODES_MET_KIND)
    assert resp[0].response_ideas[0].assigned_code_id == "K32"


def test_neutraal_kind_is_nooit_een_bestemming():
    """Beschrijvend materiaal draagt geen richting, dus een neutraal kind kan de
    richting van een idee niet dragen. Het idee valt terug op de ouder."""
    codes = [code("Beschouwing positief", "positive", "K5"), CODES_MET_KIND[4],
             code("Overig", "neutral", "K9")]
    codes[0].source_attribute_ids = ["A3"]
    resp = [respondent(idee_met_attribuut("i1", "-", "Beschouwing positief", "K5", "A3"))]
    route_opposing_poles(resp, codes)
    assert resp[0].response_ideas[0].assigned_code_id == "K9"


def test_kind_met_de_verkeerde_richting_is_geen_bestemming():
    """Een negatief idee hoort niet in een `non_negative` kind — dat is exact de
    fout die deze pas opruimt, één laag lager."""
    resp = [respondent(idee_met_attribuut("i1", "-", "Sterke klantrelatie", "K1", "A2"))]
    route_opposing_poles(resp, CODES_MET_KIND)
    assert resp[0].response_ideas[0].assigned_code_id == "K9"


def test_zonder_passend_kind_blijft_de_ouder_de_bestemming():
    resp = [respondent(idee_met_attribuut("i1", "-", "Sterke klantrelatie", "K1", "A77"))]
    rapport = route_opposing_poles(resp, CODES_MET_KIND)
    assert resp[0].response_ideas[0].assigned_code_id == "K9"
    assert rapport.per_child["Negatieve overige klantrelatie"] == 0


def test_idee_zonder_attribuut_id_gaat_naar_de_ouder():
    """De aanhechting is een A#, nooit een naam: een naam staat in de
    enquêtetaal en kan herschreven worden. Ontbreekt de id (voor-id-artefact),
    dan is de uitkomst het oude gedrag — de ouder — en dus nooit fout."""
    resp = [respondent(idee_met_attribuut("i1", "-", "Sterke klantrelatie", "K1", None))]
    route_opposing_poles(resp, CODES_MET_KIND)
    assert resp[0].response_ideas[0].assigned_code_id == "K9"


def test_een_idee_onder_een_kind_met_de_verkeerde_richting_verhuist_ook():
    """Een kind is een volwaardige code en dus even goed een plek waar de
    verkeerde richting kan landen."""
    resp = [respondent(idee_met_attribuut(
        "i1", "+", "Negatieve overige klantrelatie", "K31", "A1"))]
    rapport = route_opposing_poles(resp, CODES_MET_KIND)
    assert resp[0].response_ideas[0].assigned_code_id == "K9"
    assert rapport.moved == 1


def test_overig_aandeel_telt_de_kinderen_mee():
    """Step 5's Overig-plafond telt de ouder én zijn kinderen; deze meting moet
    hetzelfde begrip gebruiken, anders meten twee stappen twee dingen."""
    resp = [respondent(
        idee_met_attribuut("i1", "-", "Negatieve overige klantrelatie", "K31", "A1"),
        idee_met_attribuut("i2", "0", "Overig", "K9", None),
        idee_met_attribuut("i3", "+", "Sterke klantrelatie", "K1", "A1"),
    )]
    rapport = route_opposing_poles(resp, CODES_MET_KIND)
    assert rapport.moved == 0
    assert (rapport.overig_before, rapport.overig_after) == (2, 2)
    assert rapport.coded_ideas == 3


def test_find_overig_code_kiest_de_ouder_niet_het_kind():
    """Een kind draagt een door een LLM geschreven naam. Treft die toevallig een
    catch-all-woord, dan mag hij de ouder niet verdringen — de hiërarchie zit in
    het veld."""
    codes = [kind("Other", "negative", "K31", ["A1"]),
             code("Overig", "neutral", "K9")]
    assert find_overig_code(codes).code_id == "K9"


def test_kinderloos_codeboek_routeert_exact_als_voorheen():
    """DE PRODUCTIEVOORWAARDE. Productie-step-5 levert nog codeboeken zónder
    kinderen — `parent_code_id` is daar op élke code None — en de gebruiker
    draait die keten. Op zo'n codeboek moet deze pas exact hetzelfde doen als
    vóór 2026-08-22: iedere tegenpool naar de ouder, punt.

    Faalt deze test, dan is kindbewustzijn naar het kinderloze geval gelekt.
    Vereenvoudig de tak dus niet weg.
    """
    assert all(c.parent_code_id is None for c in CODES)

    resp = [respondent(
        idea("i1", "-", "Sterke klantrelatie", "K1"),
        idea("i2", "+", "Ongunstige voorwaarden", "K2"),
        idea("i3", "-", "Sparen en rente", "K3"),
        idea("i4", "0", "Overig", "K9"),
    )]
    rapport = route_opposing_poles(resp, CODES)

    assert [i.assigned_code_id for i in resp[0].response_ideas] == [
        "K9", "K9", "K3", "K9"]
    assert (rapport.moved, rapport.coded_ideas) == (2, 4)
    assert (rapport.overig_before, rapport.overig_after) == (1, 3)
    assert rapport.per_code == {"Sterke klantrelatie": 1, "Ongunstige voorwaarden": 1}
    assert rapport.per_child == {}


@pytest.mark.parametrize("code_valence,idea_valence,verwacht", [
    ("positive", "+", True),
    ("non_negative", "+", True),      # de positieve helft van een tweedeling
    ("negative", "-", True),
    ("neutral", "+", False),          # zegt niets over richting, draagt dus niets
    ("neutral", "-", False),
    ("non_negative", "-", False),
    ("positive", "-", False),
    ("positive", "0", False),         # het idee heeft zelf geen richting
    ("positive", "", False),
])
def test_carries(code_valence, idea_valence, verwacht):
    """`carries` is de bestemmingstoets en NIET de ontkenning van `opposes`:
    `neutral` botst nergens mee én draagt niets, en `non_negative` draagt wel
    een positief idee maar botst met een negatief."""
    assert carries(code_valence, idea_valence) is verwacht
