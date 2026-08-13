"""Tests voor de twee deterministische other-vangnetten (step 4)."""
from pipeline.step_4_classifier.drains import (
    DRAIN_ATTRIBUTE_KEY,
    DRAIN_FACET_KEY,
    is_drain_item,
    make_drain_attribute,
    make_drain_facet,
    strip_empty_drains,
)


# =============================================================================
# CONSTRUCTIE
# =============================================================================

def test_drain_attribuut_draagt_sleutel_vlag_naam_en_definitie():
    item = make_drain_attribute("Bereikbaarheid", "Dutch")
    assert item["drain_key"] == DRAIN_ATTRIBUTE_KEY
    assert item["is_drain"] is True
    assert item["attribute_name"].strip()
    assert item["attribute_definition"].strip()


def test_drain_facet_draagt_sleutel_vlag_naam_en_definitie():
    item = make_drain_facet("Dienstverlening", "Dutch")
    assert item["drain_key"] == DRAIN_FACET_KEY
    assert item["is_drain"] is True
    assert item["facet_name"].strip()
    assert item["facet_definition"].strip()


def test_drain_facet_brengt_zijn_eigen_attribuut_mee():
    """Een idee dat in geen enkel facet past moet toch een attribuut krijgen —
    anders is het domein-vangnet zelf een gat."""
    item = make_drain_facet("Dienstverlening", "Dutch")
    assert len(item["attributes"]) == 1
    assert item["attributes"][0]["drain_key"] == DRAIN_ATTRIBUTE_KEY


def test_naam_volgt_de_enquetetaal_met_engelse_terugval():
    nl = make_drain_attribute("F", "Dutch")["attribute_name"]
    en = make_drain_attribute("F", "English")["attribute_name"]
    onbekend = make_drain_attribute("F", "Klingon")["attribute_name"]
    assert nl != en
    assert onbekend == en


def test_de_twee_sleutels_verschillen():
    assert DRAIN_ATTRIBUTE_KEY != DRAIN_FACET_KEY


# =============================================================================
# HERKENNING
# =============================================================================

def test_gewoon_item_is_geen_drain():
    assert is_drain_item({"attribute_name": "Wachttijd"}) is False


def test_drain_wordt_herkend_op_sleutel_niet_op_naam():
    """De naam is vertaald en door naslijpen te herschrijven; de sleutel niet."""
    item = make_drain_attribute("F", "Dutch")
    item["attribute_name"] = "Iets heel anders"
    assert is_drain_item(item) is True


# =============================================================================
# OPRUIMEN
# =============================================================================

def _facets_met_drain():
    facets = {"D": [
        {"facet_name": "F", "is_drain": False},
        make_drain_facet("D", "Dutch"),
    ]}
    attributes = {"D": {
        "F": [
            {"attribute_name": "A", "is_drain": False},
            make_drain_attribute("F", "Dutch"),
        ],
        facets["D"][1]["facet_name"]: list(facets["D"][1]["attributes"]),
    }}
    return facets, attributes


def test_lege_drains_vallen_weg():
    facets, attributes = _facets_met_drain()
    facets, attributes, assigns = strip_empty_drains(
        facets, attributes, {"i1": "A"})
    assert [f["facet_name"] for f in facets["D"]] == ["F"]
    assert [a["attribute_name"] for a in attributes["D"]["F"]] == ["A"]


def test_gevulde_drain_blijft_staan():
    facets, attributes = _facets_met_drain()
    drain_name = attributes["D"]["F"][1]["attribute_name"]
    facets, attributes, assigns = strip_empty_drains(
        facets, attributes, {"i1": "A", "i2": drain_name})
    assert drain_name in [a["attribute_name"] for a in attributes["D"]["F"]]


def test_gevuld_drainfacet_blijft_met_zijn_attribuut_staan():
    facets, attributes = _facets_met_drain()
    drain_facet = facets["D"][1]["facet_name"]
    drain_attr = attributes["D"][drain_facet][0]["attribute_name"]
    facets, attributes, assigns = strip_empty_drains(
        facets, attributes, {"i1": drain_attr})
    assert drain_facet in [f["facet_name"] for f in facets["D"]]
    assert attributes["D"][drain_facet]


def test_opruimen_raakt_de_toewijzingen_niet():
    facets, attributes = _facets_met_drain()
    assigns_in = {"i1": "A"}
    _, _, assigns_out = strip_empty_drains(facets, attributes, assigns_in)
    assert assigns_out == assigns_in


def test_opruimen_muteert_de_invoer_niet():
    facets, attributes = _facets_met_drain()
    before = len(facets["D"])
    strip_empty_drains(facets, attributes, {"i1": "A"})
    assert len(facets["D"]) == before


def test_drainfacet_en_zijn_attribuut_delen_naam_en_definitie():
    """Het is een bak, geen twee. Een verzonnen onderscheid ("Overig — Overig
    — <domein>") beloofde een verfijning die er niet is."""
    facet = make_drain_facet("dienstverlening", "Dutch")
    attribuut = facet["attributes"][0]
    assert attribuut["attribute_name"] == facet["facet_name"]
    assert attribuut["attribute_definition"] == facet["facet_definition"]
    assert facet["facet_name"].count("Overig") == 1
