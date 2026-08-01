"""Tests voor de canonieke verbose-lognaam."""

from utils.saveVerbose import build_log_filename


def test_name_has_no_timestamp():
    """Twee aanroepen op verschillende momenten geven dezelfde naam —
    dat is wat overschrijven mogelijk maakt."""
    a = build_log_filename("M241030 Vezet 2024 databestand.sav", "Q20", 1000, 7)
    b = build_log_filename("M241030 Vezet 2024 databestand.sav", "Q20", 1000, 7)
    assert a == b
    assert a == "M241030_Vezet_2024_databestand_Q20_1000_step7.txt"


def test_sample_size_appears_once():
    """De samplesize staat precies één keer in de naam."""
    name = build_log_filename("dataset.sav", "Qd1", 4586, 5)
    assert name == "dataset_Qd1_4586_step5.txt"
    assert name.count("4586") == 1


def test_no_sample_size_becomes_full():
    name = build_log_filename("dataset.sav", "Qd1", None, 3)
    assert name == "dataset_Qd1_full_step3.txt"


def test_long_names_stay_distinct():
    """Zonder afkapping blijven twee datasets met een gelijk begin
    onderscheiden. Mét afkapping op 50 tekens zouden ze elkaar overschrijven."""
    lang = build_log_filename(
        "M260502 Associatiemonitor ASN Bank tabellenbestand vergelijkend met Qd1.sav",
        "Qd1", 4586, 4)
    kort = build_log_filename(
        "M260502 Associatiemonitor ASN Bank tabellenbestand.sav",
        "Qd1", 4586, 4)
    assert lang != kort


def test_spaces_become_underscores():
    name = build_log_filename("met spaties erin.sav", "Q1", 100, 0)
    assert " " not in name
    assert name == "met_spaties_erin_Q1_100_step0.txt"


def test_sample_size_as_string_full():
    """test_data.py gebruikt letterlijk de string "full" als samplesize."""
    assert build_log_filename("dataset.sav", "Qd1", "full", 3) == "dataset_Qd1_full_step3.txt"


def test_sample_size_zero_is_not_full():
    """0 is een getal, geen ontbrekende waarde."""
    assert build_log_filename("dataset.sav", "Qd1", 0, 3) == "dataset_Qd1_0_step3.txt"
