"""Tests voor de opmaak van verbose-rapportblokken.

Deze tests bestaan doordat de renderer puur is. Aan een string die de aanroeper
zelf in elkaar zette valt niets te toetsen; aan `render_block()` gegeven
dezelfde records valt alles te toetsen.
"""
import pytest

from utils.reportBlocks import (
    MARK_DRAIN,
    MARK_DROPPED,
    Group,
    Metric,
    _width,
    measure,
    render_block,
    render_flow,
)


def _blok():
    return [
        Group("weggegooid", [Metric("zonder inhoud", 7, of=2210)],
              marker=MARK_DROPPED, total=Metric("weggegooid", 7, of=2210)),
        Group("vangnetten", [
            Metric("zonder onderwerp", 138, of=2203),
            Metric("onbekend met het onderwerp", 80, of=2203),
            Metric("ander onderwerp", 14, of=2203),
        ], marker=MARK_DRAIN, total=Metric("vangnetten", 232, of=2203)),
    ]


def _zichtbaar(lijn):
    """Kolompositie zoals je hem ziet, niet zoals Python hem indexeert.

    Een emoji is één teken voor `len()` en ~twee cellen op het scherm; toetsen
    op een string-index zou dus juist de regels met een marker fout beoordelen.
    """
    return _width(lijn)


# =============================================================================
# UITLIJNING — de reden dat dit een eigen module is
# =============================================================================

def test_het_hele_blok_deelt_een_kolom():
    """Per groep uitlijnen lijnt elke groep tegen zichzelf uit en tegen niets
    anders; dan kan het oog geen kolom volgen, en dat was het hele doel."""
    assert render_block(_blok()) == [
        "    \U0001f9f9  weggegooid                    7   0,3%",
        "        zonder inhoud                 7   0,3%",
        "    \U0001f573\ufe0f  vangnetten                  232  10,5%",
        "        zonder onderwerp            138   6,3%",
        "        onbekend met het onderwerp   80   3,6%",
        "        ander onderwerp              14   0,6%",
    ]


def test_elke_regel_zet_zijn_percentage_op_dezelfde_cel():
    """De toets die telt: zichtbare breedte, niet string-index."""
    einden = {_zichtbaar(l[:l.index("%") + 1]) for l in render_block(_blok())}
    assert len(einden) == 1, f"percentages eindigen op {einden}"


def test_de_groepskop_telt_mee_voor_de_kolombreedte():
    """Regressie: measure() keek alleen naar rijen. Een kop van 10,5% tegen
    rijen van 6,3% duwde de kop uit de kolom die de rijen hadden gekozen."""
    w = measure([Group("g", [Metric("r", 1, of=1000)],
                       total=Metric("g", 500, of=1000))])
    assert w.share == len("50,0%")
    assert w.value == len("500")


# =============================================================================
# MARKERS EN BREEDTE
# =============================================================================

def test_emoji_telt_als_twee_cellen():
    """`len()` is hier fout: 🕳️ is twee codepoints en beslaat ~twee cellen, dus
    padding uit len() zou juist die regel scheeftrekken."""
    assert _width(MARK_DRAIN) == 2
    assert _width(MARK_DROPPED) == 2
    assert _width("abc") == 3


def test_marker_verschuift_de_getalkolom_niet():
    """De kop draagt een marker en staat een niveau hoger dan zijn rijen; de
    getallen moeten toch op dezelfde cel eindigen. Zichtbare breedte, want een
    string-index telt de emoji als één teken en het scherm als twee."""
    for lijnen in (
        render_block([Group("t", [Metric("r", 1, of=10)],
                            marker=MARK_DROPPED, total=Metric("t", 1, of=10))]),
        render_block([Group("t", [Metric("r", 1, of=10)],
                            total=Metric("t", 1, of=10))]),
    ):
        einden = {_zichtbaar(l[:l.index("%") + 1]) for l in lijnen}
        assert len(einden) == 1, f"{einden} in {lijnen}"


# =============================================================================
# GETALLEN EN VOORBEELDEN
# =============================================================================

def test_percentage_gebruikt_een_decimale_komma():
    lijnen = render_block([Group("g", [Metric("r", 1, of=3)])])
    assert "33,3%" in lijnen[1]
    assert "33.3" not in lijnen[1]


def test_zonder_noemer_geen_percentage():
    lijnen = render_block([Group("g", [Metric("r", 7)])])
    assert "%" not in lijnen[1]


def test_voorbeelden_worden_afgekapt_met_een_telling():
    lijnen = render_block([Group("g", [
        Metric("r", 9, of=9, examples=[f"v{i}" for i in range(9)])])])
    assert "(+3)" in lijnen[1]
    assert "v6" not in lijnen[1]


def test_voorbeelden_gescheiden_door_een_punt():
    lijnen = render_block([Group("g", [Metric("r", 2, of=2, examples=["a", "b"])])])
    assert "a · b" in lijnen[1]


def test_geen_regel_eindigt_op_spaties():
    for lijn in render_block(_blok()):
        assert lijn == lijn.rstrip()


# =============================================================================
# FLOW
# =============================================================================

def test_flow_toont_de_tussenstap():
    """Splitsen en filteren zijn één beweging; alleen begin en eind tonen
    verbergt dat er iets gebeurde."""
    assert render_flow([1236, "responsen", 2210, "fragmenten", 2203, "ideeën"]).strip() \
        == "1236 responsen  →  2210 fragmenten  →  2203 ideeën"
