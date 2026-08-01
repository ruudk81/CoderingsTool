"""Tests voor het opruimen per analyse."""

import os
import sqlite3
import time

import pytest

from utils import retention
from utils.exportNaming import export_filename


# =============================================================================
# Opzet: een tmp-repo met een minimale cache-database
# =============================================================================

def _repo(tmp_path, datasets=("data.sav",)):
    """Een projectmap met data/, data/cache/cache.db en exports/."""
    (tmp_path / "data").mkdir()
    for d in datasets:
        (tmp_path / "data" / d).write_bytes(b"")
    (tmp_path / "data" / "cache").mkdir()
    con = sqlite3.connect(tmp_path / "data" / "cache" / "cache.db")
    con.execute("CREATE TABLE cache_metadata (filename TEXT, variable_key TEXT, status TEXT)")
    for d in datasets:
        con.execute("INSERT INTO cache_metadata VALUES (?, ?, 'valid')", (d, "Q1_100"))
    con.commit()
    con.close()
    return tmp_path


def _bestand(root, mapnaam, dataset, var, sample, doctype, ext="txt", mtime=None, grootte=10):
    d = root / "exports" / mapnaam
    d.mkdir(parents=True, exist_ok=True)
    p = d / export_filename(dataset, var, sample, doctype, ext)
    p.write_bytes(b"x" * grootte)
    if mtime is not None:
        os.utime(p, (mtime, mtime))
    return p


_MAPPEN = {"log_step4": "verbose_logs", "kosten": "costs", "codering": "coderingen"}


def _analyse(root, dataset, var, sample, mtime):
    """Eén analyse met bestanden in drie verschillende mappen."""
    return [_bestand(root, _MAPPEN[dt], dataset, var, sample, dt, mtime=mtime)
            for dt in _MAPPEN]


@pytest.fixture(autouse=True)
def _standaard_instellingen(monkeypatch):
    """Elke test begint met de opgeleverde stand: alle plafonds uit."""
    monkeypatch.setattr(retention, "RETENTION_ENABLED", True)
    monkeypatch.setattr(retention, "MAX_ANALYSES", None)
    monkeypatch.setattr(retention, "TRASH_MAX_MB", None)
    monkeypatch.setattr(retention, "PROTECT_DAYS", 7)


# =============================================================================
# Groeperen
# =============================================================================

def test_bestanden_groeperen_per_analyse(tmp_path):
    root = _repo(tmp_path)
    nu = time.time()
    _analyse(root, "data.sav", "Q1", 100, nu - 30 * 86400)
    _analyse(root, "data.sav", "Q1", 500, nu - 60 * 86400)

    analyses, restanten = retention.collect(root)

    assert len(analyses) == 2
    assert restanten == []
    assert {a.key.sample for a in analyses} == {"100", "500"}
    assert all(len(a.entries) == 3 for a in analyses)


def test_analyse_is_zo_oud_als_haar_nieuwste_bestand(tmp_path):
    root = _repo(tmp_path)
    nu = time.time()
    _bestand(root, "verbose_logs", "data.sav", "Q1", 100, "log_step4", mtime=nu - 60 * 86400)
    _bestand(root, "costs", "data.sav", "Q1", 100, "kosten", ext="json", mtime=nu - 86400)

    analyses, _ = retention.collect(root)
    assert (nu - analyses[0].mtime) / 86400 == pytest.approx(1, abs=0.1)


def test_analyses_staan_nieuwste_eerst(tmp_path):
    root = _repo(tmp_path)
    nu = time.time()
    _analyse(root, "data.sav", "Q1", 100, nu - 90 * 86400)
    _analyse(root, "data.sav", "Q1", 500, nu - 30 * 86400)

    analyses, _ = retention.collect(root)
    assert [a.key.sample for a in analyses] == ["500", "100"]


def test_onbeheerde_mappen_blijven_buiten_beeld(tmp_path):
    root = _repo(tmp_path)
    for m in ("adhoc", "diagnostics", "experiment_logs"):
        d = root / "exports" / m
        d.mkdir(parents=True)
        (d / "van_alles.txt").write_text("blijf")

    analyses, restanten = retention.collect(root)
    assert analyses == [] and restanten == []


# =============================================================================
# Plafond
# =============================================================================

def test_zonder_plafond_verhuist_er_niets(tmp_path):
    """De opgeleverde stand: het gereedschap rapporteert maar kan niets."""
    root = _repo(tmp_path)
    nu = time.time()
    for i in range(5):
        _analyse(root, "data.sav", "Q1", 100 + i, nu - (30 + i) * 86400)

    analyses, _ = retention.collect(root)
    assert retention.select_analyses_for_removal(analyses, now=nu) == []


def test_plafond_houdt_de_n_nieuwste(tmp_path, monkeypatch):
    monkeypatch.setattr(retention, "MAX_ANALYSES", 2)
    root = _repo(tmp_path)
    nu = time.time()
    for i in range(4):
        _analyse(root, "data.sav", "Q1", 100 + i, nu - (30 + 10 * i) * 86400)

    analyses, _ = retention.collect(root)
    weg = retention.select_analyses_for_removal(analyses, now=nu)
    assert [a.key.sample for a in weg] == ["102", "103"]


def test_verse_analyse_blijft_ook_voorbij_het_plafond(tmp_path, monkeypatch):
    """Het venster wint van het plafond, ook als dat het plafond overschrijdt."""
    monkeypatch.setattr(retention, "MAX_ANALYSES", 1)
    root = _repo(tmp_path)
    nu = time.time()
    _analyse(root, "data.sav", "Q1", 100, nu - 3600)
    _analyse(root, "data.sav", "Q1", 200, nu - 2 * 86400)
    _analyse(root, "data.sav", "Q1", 300, nu - 90 * 86400)

    analyses, _ = retention.collect(root)
    weg = retention.select_analyses_for_removal(analyses, now=nu)
    assert [a.key.sample for a in weg] == ["300"]


def test_nieuwste_analyse_overleeft_een_plafond_van_nul(tmp_path, monkeypatch):
    monkeypatch.setattr(retention, "MAX_ANALYSES", 0)
    root = _repo(tmp_path)
    nu = time.time()
    _analyse(root, "data.sav", "Q1", 100, nu - 90 * 86400)
    _analyse(root, "data.sav", "Q1", 200, nu - 91 * 86400)

    analyses, _ = retention.collect(root)
    weg = retention.select_analyses_for_removal(analyses, now=nu)
    assert [a.key.sample for a in weg] == ["200"]


def test_een_analyse_verhuist_in_zijn_geheel(tmp_path, monkeypatch):
    """Geen halve analyse: de kosten mogen niet blijven als het logboek weggaat."""
    monkeypatch.setattr(retention, "MAX_ANALYSES", 1)
    root = _repo(tmp_path)
    nu = time.time()
    _analyse(root, "data.sav", "Q1", 100, nu - 30 * 86400)
    _analyse(root, "data.sav", "Q1", 200, nu - 90 * 86400)

    retention.run(root, apply=True, now=nu)

    bak = root / "exports" / "_prullenbak"
    verhuisd = sorted(p.name for p in bak.rglob("*") if p.is_file())
    assert len(verhuisd) == 3
    assert all("_200_" in n for n in verhuisd)
    blijft = [p.name for p in (root / "exports").rglob("*")
              if p.is_file() and "_prullenbak" not in str(p)]
    assert blijft and all("_100_" in n for n in blijft)


# =============================================================================
# Restanten
# =============================================================================

def test_oud_restant_verhuist_vers_restant_blijft(tmp_path, monkeypatch):
    monkeypatch.setattr(retention, "MAX_ANALYSES", 10)
    root = _repo(tmp_path)
    nu = time.time()
    _analyse(root, "data.sav", "Q1", 100, nu - 86400)      # anders te veel restanten
    d = root / "exports" / "verbose_logs"
    oud = d / "oude_naamgeving_20260301_120000.txt"
    oud.write_text("oud"); os.utime(oud, (nu - 90 * 86400,) * 2)
    vers = d / "ook_oude_naamgeving_20260801_120000.txt"
    vers.write_text("vers"); os.utime(vers, (nu - 3600,) * 2)

    _, restanten = retention.collect(root)
    assert len(restanten) == 2
    weg = retention.select_orphans_for_removal(restanten, now=nu)
    assert [e.path.name for e in weg] == [oud.name]


def test_lege_datasetlijst_stopt_de_run(tmp_path):
    """Zonder datasets lijkt elk bestand een restant — dan liever stoppen."""
    root = _repo(tmp_path)
    (root / "data" / "data.sav").unlink()
    con = sqlite3.connect(root / "data" / "cache" / "cache.db")
    con.execute("DELETE FROM cache_metadata")
    con.commit()
    con.close()

    with pytest.raises(retention.RetentionError, match="geen enkele dataset"):
        retention.collect(root)


def test_ontbrekende_cache_stopt_de_run(tmp_path):
    root = _repo(tmp_path)
    (root / "data" / "cache" / "cache.db").unlink()
    with pytest.raises(retention.RetentionError, match="cache-database"):
        retention.collect(root)


# =============================================================================
# Prullenbak
# =============================================================================

def test_bak_krijgt_respijt_binnen_dezelfde_run(tmp_path, monkeypatch):
    """Wat in deze run naar de bak gaat, mag er niet meteen uit gewist worden."""
    monkeypatch.setattr(retention, "MAX_ANALYSES", 1)
    monkeypatch.setattr(retention, "TRASH_MAX_MB", 0)
    root = _repo(tmp_path)
    nu = time.time()
    _analyse(root, "data.sav", "Q1", 100, nu - 30 * 86400)
    _analyse(root, "data.sav", "Q1", 200, nu - 90 * 86400)

    retention.run(root, apply=True, now=nu)

    bak = root / "exports" / "_prullenbak"
    assert sum(1 for p in bak.rglob("*") if p.is_file()) == 3


def test_bak_boven_zijn_plafond_gaat_definitief_weg(tmp_path, monkeypatch):
    monkeypatch.setattr(retention, "TRASH_MAX_MB", 1)
    root = _repo(tmp_path)
    nu = time.time()
    bak = root / "exports" / "_prullenbak" / "verbose_logs"
    bak.mkdir(parents=True)
    for i in range(2):
        p = bak / f"oud_{i}.txt"
        p.write_bytes(b"x" * 1024 * 1024)
        os.utime(p, (nu - (10 + i) * 86400,) * 2)

    entries = retention.trash_entries(root)
    weg = retention.select_trash_for_removal(entries, now=nu)
    assert [e.path.name for e in weg] == ["oud_1.txt"]


def test_schakelaar_uit_doet_niets(tmp_path, monkeypatch):
    monkeypatch.setattr(retention, "RETENTION_ENABLED", False)
    monkeypatch.setattr(retention, "MAX_ANALYSES", 0)
    root = _repo(tmp_path)
    nu = time.time()
    paden = _analyse(root, "data.sav", "Q1", 100, nu - 90 * 86400)

    verslag = retention.run(root, apply=True, now=nu)

    assert verslag["uit"] is True
    assert all(p.exists() for p in paden)


# =============================================================================
# Invariant
# =============================================================================

def test_pad_buiten_exports_stopt_de_run(tmp_path):
    root = _repo(tmp_path)
    buiten = root / "data" / "geheim.txt"
    buiten.write_text("niet aankomen")
    d = root / "exports" / "verbose_logs"
    d.mkdir(parents=True)
    (d / export_filename("data.sav", "Q1", 100, "log_step4", "txt")).symlink_to(buiten)

    with pytest.raises(retention.RetentionError, match="buiten exports"):
        retention.collect(root)
