"""Tests voor de canonieke exportnaam.

De fixtures zijn synthetisch (M000000-reeks, zie saveVerbose.py en concatOpenEnds.py).
Wat hier getest wordt is het formaat, niet een dataset: de parser kent de vorm
`M<nummer> <omschrijving>`, niet welk onderzoek erachter zit. Gebruik daarom nooit
een echte bestandsnaam als fixture.
"""

import pytest

from utils.exportNaming import (
    DOCTYPES,
    ExportName,
    export_filename,
    parse_export_filename,
)

STEMS = [
    "M000001 Merkonderzoek Voorbeeldklant 2024 databestand.sav",
    "M000002 Associatiemonitor Merk X tabellenbestand.sav",
    "M000002 Associatiemonitor Merk X tabellenbestand vergelijkend met Qd1.sav",
    "M000003_Tabellenbestand_Casus.sav",
    "M000004 Associatiemonitor Merk X net databestand.sav",
]


def test_shape():
    assert export_filename(
        "M000001 Merkonderzoek Voorbeeldklant 2024 databestand.sav",
        "Q20", 1000, "codering", "xlsx",
    ) == "M000001_Merkonderzoek_Voorbeeldklant_2024_databestand_Q20_1000_codering.xlsx"


def test_no_timestamp_so_a_rerun_overwrites():
    a = export_filename("d.sav", "Q1", 100, "log_step4", "txt")
    b = export_filename("d.sav", "Q1", 100, "log_step4", "txt")
    assert a == b


def test_sample_size_forms():
    assert export_filename("d.sav", "Q1", None, "kosten", "json") == "d_Q1_full_kosten.json"
    assert export_filename("d.sav", "Q1", "full", "kosten", "json") == "d_Q1_full_kosten.json"
    assert export_filename("d.sav", "Q1", 0, "kosten", "json") == "d_Q1_0_kosten.json"


def test_datasets_sharing_a_long_prefix_stay_distinct():
    """Zonder afkapping kunnen deze twee elkaars bestand niet overschrijven."""
    kort = export_filename(
        "M000002 Associatiemonitor Merk X tabellenbestand.sav",
        "Qd1", 500, "codering", "xlsx")
    lang = export_filename(
        "M000002 Associatiemonitor Merk X tabellenbestand vergelijkend met Qd1.sav",
        "Qd1", 500, "codering", "xlsx")
    assert kort != lang


def test_unknown_doctype_is_rejected():
    with pytest.raises(ValueError, match="unknown doctype"):
        export_filename("d.sav", "Q1", 100, "verzonnen", "txt")


@pytest.mark.parametrize("doctype", sorted(DOCTYPES))
def test_roundtrip_for_every_doctype(doctype):
    naam = export_filename(
        "M000003_Tabellenbestand_Casus.sav", "xQ1_Open_tevr", 2500, doctype, "json")
    uit = parse_export_filename(naam, STEMS)
    assert uit == ExportName(
        dataset="M000003_Tabellenbestand_Casus",
        var_name="xQ1_Open_tevr",
        sample="2500",
        doctype=doctype,
        ext="json",
    )


def test_longest_doctype_wins():
    """'taxonomie_fijn' mag niet als 'taxonomie' gelezen worden."""
    naam = export_filename("d.sav", "Q1", 100, "taxonomie_fijn", "sav")
    uit = parse_export_filename(naam, ["d.sav"])
    assert uit.doctype == "taxonomie_fijn"
    assert uit.var_name == "Q1"


def test_longest_dataset_wins():
    """De langere datasetnaam begint met de kortere; die mag niet winnen."""
    naam = export_filename(
        "M000002 Associatiemonitor Merk X tabellenbestand vergelijkend met Qd1.sav",
        "Qd1", 4586, "codeboek", "sav")
    uit = parse_export_filename(naam, STEMS)
    assert uit.dataset == (
        "M000002_Associatiemonitor_Merk_X_tabellenbestand_vergelijkend_met_Qd1")
    assert uit.var_name == "Qd1"
    assert uit.sample == "4586"


def test_underscores_in_variable_name_survive():
    naam = export_filename(
        "M000004 Associatiemonitor Merk X net databestand.sav",
        "Qd1_combined", 2000, "prompts_step3", "json")
    uit = parse_export_filename(naam, STEMS)
    assert uit.var_name == "Qd1_combined"
    assert uit.sample == "2000"


def test_legacy_names_do_not_parse():
    """Oude namen mogen niet per ongeluk als canoniek gelezen worden."""
    for oud in (
        "step3_Q20_Q20_1000.json",
        "M000001_Merkonderzoek_Voorbeeldklant_Q20_1000_step7_20260801_064213.txt",
        "codebook_M000001_Merkonderzoek_Voorbeeldklant_Q20_1000.xlsx",
        "zonder_extensie",
    ):
        assert parse_export_filename(oud, STEMS) is None


def test_sample_must_be_a_number_or_full():
    """Anders leest een restant als een analyse met een verzonnen steekproef."""
    assert parse_export_filename("d_Q1_2500 v1_codeboek.xlsx", ["d.sav"]) is None
    assert parse_export_filename("d_Q1_kladversie_codeboek.xlsx", ["d.sav"]) is None
    assert parse_export_filename("d_Q1_2500_codeboek.xlsx", ["d.sav"]).sample == "2500"
    assert parse_export_filename("d_Q1_full_codeboek.xlsx", ["d.sav"]).sample == "full"


def test_unknown_dataset_does_not_parse():
    naam = export_filename("Onbekend bestand.sav", "Q1", 100, "kosten", "json")
    assert parse_export_filename(naam, STEMS) is None
