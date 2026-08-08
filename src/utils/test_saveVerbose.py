"""Tests voor de canonieke verbose-lognaam."""

import pytest

from utils.saveVerbose import VerboseCapture, build_log_filename


def test_name_has_no_timestamp():
    """Twee aanroepen op verschillende momenten geven dezelfde naam —
    dat is wat overschrijven mogelijk maakt."""
    a = build_log_filename("M000001 Merkonderzoek 2024 databestand.sav", "Q20", 1000, 7)
    b = build_log_filename("M000001 Merkonderzoek 2024 databestand.sav", "Q20", 1000, 7)
    assert a == b
    assert a == "M000001_Merkonderzoek_2024_databestand_Q20_1000_log_step7.txt"


def test_sample_size_appears_once():
    """De samplesize staat precies één keer in de naam."""
    name = build_log_filename("dataset.sav", "Qd1", 4586, 5)
    assert name == "dataset_Qd1_4586_log_step5.txt"
    assert name.count("4586") == 1


def test_no_sample_size_becomes_full():
    name = build_log_filename("dataset.sav", "Qd1", None, 3)
    assert name == "dataset_Qd1_full_log_step3.txt"


def test_long_names_stay_distinct():
    """Zonder afkapping blijven twee datasets met een gelijk begin
    onderscheiden. Mét afkapping op 50 tekens zouden ze elkaar overschrijven."""
    lang = build_log_filename(
        "M000002 Associatiemonitor Merk X tabellenbestand vergelijkend met Qd1.sav",
        "Qd1", 4586, 4)
    kort = build_log_filename(
        "M000002 Associatiemonitor Merk X tabellenbestand.sav",
        "Qd1", 4586, 4)
    assert lang != kort


def test_spaces_become_underscores():
    name = build_log_filename("met spaties erin.sav", "Q1", 100, 0)
    assert " " not in name
    assert name == "met_spaties_erin_Q1_100_log_step0.txt"


def test_sample_size_as_string_full():
    """test_data.py gebruikt letterlijk de string "full" als samplesize."""
    assert build_log_filename("dataset.sav", "Qd1", "full", 3) == "dataset_Qd1_full_log_step3.txt"


def test_sample_size_zero_is_not_full():
    """0 is een getal, geen ontbrekende waarde."""
    assert build_log_filename("dataset.sav", "Qd1", 0, 3) == "dataset_Qd1_0_log_step3.txt"


def test_find_latest_log_is_exact(tmp_path):
    """Regressie: de oude glob `{base}_{varkey}_*step{N}_*.txt` slokte het
    samplesize-segment op, waardoor een log van sample 500 werd gevonden
    terwijl 4586 was gevraagd."""
    (tmp_path / "dataset_Qd1_500_log_step7.txt").write_text("verkeerde sample")
    doel = tmp_path / "dataset_Qd1_4586_log_step7.txt"
    doel.write_text("juiste sample")

    gevonden = VerboseCapture.find_latest_log(
        "dataset.sav", "Qd1", 4586, 7, output_dir=tmp_path)

    assert gevonden == doel
    assert gevonden.read_text() == "juiste sample"


def test_find_latest_log_returns_none_when_absent(tmp_path):
    assert VerboseCapture.find_latest_log(
        "dataset.sav", "Qd1", 4586, 7, output_dir=tmp_path) is None


def test_capture_overwrites_on_rerun(tmp_path):
    """Een tweede run van dezelfde stap vervangt het log, plakt er niet achter."""
    for tekst in ("eerste run", "tweede run"):
        with VerboseCapture("dataset.sav", "Q1", 100, 2, output_dir=tmp_path):
            print(tekst)

    logs = list(tmp_path.glob("*.txt"))
    assert len(logs) == 1
    inhoud = logs[0].read_text()
    assert "tweede run" in inhoud
    assert "eerste run" not in inhoud


def test_verbose_capture_rejects_removed_parameters():
    """append_mode en session_id bestaan niet meer."""
    with pytest.raises(TypeError):
        VerboseCapture("d.sav", "Q1", 100, 2, append_mode=True)
