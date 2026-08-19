"""Tests for the opposing-pole filter (step 6, after assignment).

The filter moves an idea to Overig when it sits under a code carrying the
OPPOSITE direction. Everything else stays put — the point is to remove active
errors ("slecht bereikbaar" counted under a code labelled positive), not to
empty the codebook.
"""

import pytest

from models import CodeAssignedModel, CodeAssignedSubmodel, ConsolidatedCode
from pipeline.step_6_codeAssigner.valence_filter import (
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
