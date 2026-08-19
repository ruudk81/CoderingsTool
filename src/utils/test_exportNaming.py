"""Tests for the canonical export name.

The fixtures are synthetic (M000000 series, see saveVerbose.py and
concatOpenEnds.py). What is tested here is the format, not a dataset: the parser
knows the shape `M<number> <description>`, not which study sits behind it. So
never use a real file name as a fixture.
"""

import pytest

from utils import exportNaming
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
    """Without truncation these two cannot overwrite each other's file."""
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


def test_longest_doctype_wins(monkeypatch):
    """A doctype ending in another doctype must not be read as the shorter one.

    No pair in the live vocabulary overlaps that way today, so the guard is tested
    against a synthetic pair — it is what keeps adding one from silently breaking
    every name already on disk."""
    monkeypatch.setattr(exportNaming, "DOCTYPES", frozenset({"taxonomie", "ruwe_taxonomie"}))
    uit = parse_export_filename("d_Q1_100_ruwe_taxonomie.sav", ["d.sav"])
    assert uit.doctype == "ruwe_taxonomie"
    assert uit.var_name == "Q1"


def test_longest_dataset_wins():
    """The longer dataset name starts with the shorter one; that must not win."""
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
    """Old names must not accidentally read as canonical."""
    for oud in (
        "step3_Q20_Q20_1000.json",
        "M000001_Merkonderzoek_Voorbeeldklant_Q20_1000_step7_20260801_064213.txt",
        "codebook_M000001_Merkonderzoek_Voorbeeldklant_Q20_1000.xlsx",
        "zonder_extensie",
    ):
        assert parse_export_filename(oud, STEMS) is None


def test_sample_must_be_a_number_or_full():
    """Otherwise a leftover reads as an analysis with an invented sample."""
    assert parse_export_filename("d_Q1_2500 v1_codeboek.xlsx", ["d.sav"]) is None
    assert parse_export_filename("d_Q1_kladversie_codeboek.xlsx", ["d.sav"]) is None
    assert parse_export_filename("d_Q1_2500_codeboek.xlsx", ["d.sav"]).sample == "2500"
    assert parse_export_filename("d_Q1_full_codeboek.xlsx", ["d.sav"]).sample == "full"


def test_unknown_dataset_does_not_parse():
    naam = export_filename("Onbekend bestand.sav", "Q1", 100, "kosten", "json")
    assert parse_export_filename(naam, STEMS) is None
