"""Tests voor het opruimen per analyse."""

import os
import sqlite3
import time

import pytest

from utils import retention
from utils.exportNaming import export_filename


# =============================================================================
# Setup: a tmp repo with a minimal cache database
# =============================================================================

def _repo(tmp_path, datasets=("data.sav",)):
    """Een projectmap met data/, data/cache/cache.db en exports/."""
    (tmp_path / "data").mkdir()
    for d in datasets:
        (tmp_path / "data" / d).write_bytes(b"")
    cache = tmp_path / "data" / "cache"
    cache.mkdir()
    con = sqlite3.connect(cache / "cache.db")
    con.execute("CREATE TABLE cache_metadata "
                "(filename TEXT, variable_key TEXT, status TEXT, cache_path TEXT)")
    for d in datasets:
        # cache_path deliberately points at a file that does not exist: this row
        # is only here to make the dataset known, not to claim a pickle.
        con.execute("INSERT INTO cache_metadata VALUES (?, ?, 'valid', ?)",
                    (d, "Q1_100", str(cache / f"000_seed_{d}.pkl")))
    con.commit()
    con.close()
    return tmp_path


def _cache_bestand(root, dataset, variable_key, naam, mtime=None,
                   status="valid", in_db=True, grootte=10):
    """Eén pickle in data/cache/, met of zonder geldige db-rij."""
    cache = root / "data" / "cache"
    p = cache / f"{naam}.pkl"
    p.write_bytes(b"x" * grootte)
    if mtime is not None:
        os.utime(p, (mtime, mtime))
    if in_db:
        con = sqlite3.connect(cache / "cache.db")
        con.execute("INSERT INTO cache_metadata VALUES (?, ?, ?, ?)",
                    (dataset, variable_key, status, str(p)))
        con.commit()
        con.close()
    return p


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
    monkeypatch.setattr(retention, "CACHE_MAX_ANALYSES", None)
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


def test_an_analysis_is_as_old_as_its_newest_file(tmp_path):
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

def test_without_a_cap_nothing_moves(tmp_path):
    """The delivered state: the tool reports but can do nothing."""
    root = _repo(tmp_path)
    nu = time.time()
    for i in range(5):
        _analyse(root, "data.sav", "Q1", 100 + i, nu - (30 + i) * 86400)

    analyses, _ = retention.collect(root)
    assert retention.select_analyses_for_removal(analyses, now=nu) == []


def test_the_cap_keeps_the_n_newest(tmp_path, monkeypatch):
    monkeypatch.setattr(retention, "MAX_ANALYSES", 2)
    root = _repo(tmp_path)
    nu = time.time()
    for i in range(4):
        _analyse(root, "data.sav", "Q1", 100 + i, nu - (30 + 10 * i) * 86400)

    analyses, _ = retention.collect(root)
    weg = retention.select_analyses_for_removal(analyses, now=nu)
    assert [a.key.sample for a in weg] == ["102", "103"]


def test_a_fresh_analysis_stays_even_beyond_the_cap(tmp_path, monkeypatch):
    """The window beats the cap, even when that exceeds the cap."""
    monkeypatch.setattr(retention, "MAX_ANALYSES", 1)
    root = _repo(tmp_path)
    nu = time.time()
    _analyse(root, "data.sav", "Q1", 100, nu - 3600)
    _analyse(root, "data.sav", "Q1", 200, nu - 2 * 86400)
    _analyse(root, "data.sav", "Q1", 300, nu - 90 * 86400)

    analyses, _ = retention.collect(root)
    weg = retention.select_analyses_for_removal(analyses, now=nu)
    assert [a.key.sample for a in weg] == ["300"]


def test_the_newest_analysis_survives_a_cap_of_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(retention, "MAX_ANALYSES", 0)
    root = _repo(tmp_path)
    nu = time.time()
    _analyse(root, "data.sav", "Q1", 100, nu - 90 * 86400)
    _analyse(root, "data.sav", "Q1", 200, nu - 91 * 86400)

    analyses, _ = retention.collect(root)
    weg = retention.select_analyses_for_removal(analyses, now=nu)
    assert [a.key.sample for a in weg] == ["200"]


def test_an_analysis_moves_as_a_whole(tmp_path, monkeypatch):
    """No half analysis: the costs must not stay when the log goes."""
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

def test_an_old_leftover_moves_a_fresh_one_stays(tmp_path, monkeypatch):
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


def test_an_old_name_attaches_to_its_analysis(tmp_path, monkeypatch):
    """A log under the old name belongs to the analysis, not to the leftovers.

    Without this rule such a file is protected only by its own age and disappears
    within a week, while its analysis stays.
    """
    monkeypatch.setattr(retention, "MAX_ANALYSES", 10)
    root = _repo(tmp_path)
    nu = time.time()
    _analyse(root, "data.sav", "Q1", 100, nu - 86400)
    oud = root / "exports" / "verbose_logs" / "data_Q1_100_100_step3_20260301_120000.txt"
    oud.write_text("oud logboek"); os.utime(oud, (nu - 90 * 86400,) * 2)

    analyses, restanten = retention.collect(root)

    assert restanten == []
    assert oud in [e.path for e in analyses[0].entries]
    assert retention.select_orphans_for_removal(restanten, now=nu) == []


def test_an_ambiguous_old_name_stays_a_leftover(tmp_path, monkeypatch):
    """Two samples on the same question: the name does not say which. Do not guess."""
    monkeypatch.setattr(retention, "MAX_ANALYSES", 10)
    root = _repo(tmp_path)
    nu = time.time()
    _analyse(root, "data.sav", "Q1", 100, nu - 86400)
    _analyse(root, "data.sav", "Q1", 500, nu - 86400)
    # oude coderingen-naam: dataset + variabele, geen steekproef
    oud = root / "exports" / "coderingen" / "data_Q1_codeboek.sav"
    oud.write_text("welke steekproef?"); os.utime(oud, (nu - 90 * 86400,) * 2)

    _, restanten = retention.collect(root)

    assert [e.path for e in restanten] == [oud]


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
    """What goes to the bin in this run must not be erased from it immediately."""
    monkeypatch.setattr(retention, "MAX_ANALYSES", 1)
    monkeypatch.setattr(retention, "TRASH_MAX_MB", 0)
    root = _repo(tmp_path)
    nu = time.time()
    _analyse(root, "data.sav", "Q1", 100, nu - 30 * 86400)
    _analyse(root, "data.sav", "Q1", 200, nu - 90 * 86400)

    retention.run(root, apply=True, now=nu)

    bak = root / "exports" / "_prullenbak"
    assert sum(1 for p in bak.rglob("*") if p.is_file()) == 3


def test_a_bin_over_its_cap_goes_for_good(tmp_path, monkeypatch):
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


# =============================================================================
# data/cache — hetzelfde analyse-begrip, een eigen plafond
# =============================================================================

def test_cache_and_exports_together_form_one_analysis(tmp_path):
    """A cache file belongs to the same analysis as its exports."""
    root = _repo(tmp_path)
    nu = time.time()
    _analyse(root, "data.sav", "Q1", 100, nu - 30 * 86400)
    _cache_bestand(root, "data.sav", "Q1_100", "004_extracted_ideas_data_Q1_100",
                   mtime=nu - 30 * 86400)

    analyses, restanten = retention.collect(root)

    assert len(analyses) == 1
    assert len(analyses[0].entries) == 3
    assert len(analyses[0].cache_entries) == 1
    assert restanten == []


def test_a_cache_without_exports_is_an_analysis_of_its_own(tmp_path):
    """If the exports are already cleaned up, the cache stays an analysis on its own."""
    root = _repo(tmp_path)
    nu = time.time()
    _cache_bestand(root, "data.sav", "Q1_500", "004_extracted_ideas_data_Q1_500",
                   mtime=nu - 30 * 86400)

    analyses, restanten = retention.collect(root)

    assert [str(a.key) for a in analyses] == ["data · Q1 · 500"]
    assert analyses[0].entries == []
    assert restanten == []


def test_an_unreachable_pickle_is_a_leftover(tmp_path):
    """An 'invalid' row means no code path can read the file any more."""
    root = _repo(tmp_path)
    nu = time.time()
    dood = _cache_bestand(root, "data.sav", "Q1_100", "006_mece_codes_metadata_oud",
                          mtime=nu - 30 * 86400, status="invalid")

    analyses, restanten = retention.collect(root)

    assert [e.path for e in restanten] == [dood]
    assert analyses == []


def test_a_pickle_without_a_db_row_is_a_leftover(tmp_path):
    root = _repo(tmp_path)
    nu = time.time()
    vreemd = _cache_bestand(root, "data.sav", "Q1_100", "007_taxonomy_codes_onbekend",
                            mtime=nu - 30 * 86400, in_db=False)

    _, restanten = retention.collect(root)

    assert [e.path for e in restanten] == [vreemd]


def test_the_cache_has_its_own_wider_cap(tmp_path, monkeypatch):
    """Exports beyond their cap go; the cache of the same analysis stays."""
    monkeypatch.setattr(retention, "MAX_ANALYSES", 1)
    monkeypatch.setattr(retention, "CACHE_MAX_ANALYSES", 2)
    root = _repo(tmp_path)
    nu = time.time()
    for sample, dagen in ((100, 30), (500, 60)):
        _analyse(root, "data.sav", "Q1", sample, nu - dagen * 86400)
        _cache_bestand(root, "data.sav", f"Q1_{sample}", f"004_ideas_{sample}",
                       mtime=nu - dagen * 86400)

    retention.run(root, apply=True, now=nu)

    oud = root / "data" / "cache" / "004_ideas_500.pkl"
    assert oud.exists(), "cache van de oudste analyse moet binnen haar eigen plafond blijven"
    assert not list((root / "exports" / "coderingen").glob("*_500_*")), \
        "exports van de oudste analyse moesten wél weg"


def test_cache_beyond_its_cap_moves_and_is_invalidated(tmp_path, monkeypatch):
    """The file goes to data_cache/ in the bin, the db row goes to invalid."""
    monkeypatch.setattr(retention, "CACHE_MAX_ANALYSES", 1)
    root = _repo(tmp_path)
    nu = time.time()
    _cache_bestand(root, "data.sav", "Q1_100", "004_ideas_100", mtime=nu - 30 * 86400)
    oud = _cache_bestand(root, "data.sav", "Q1_500", "004_ideas_500", mtime=nu - 60 * 86400)

    retention.run(root, apply=True, now=nu)

    assert not oud.exists()
    assert (root / "exports" / "_prullenbak" / "data_cache" / "004_ideas_500.pkl").exists()

    con = sqlite3.connect(root / "data" / "cache" / "cache.db")
    status = con.execute("SELECT status FROM cache_metadata WHERE cache_path = ?",
                         (str(oud),)).fetchone()[0]
    con.close()
    assert status == "invalid"


def test_a_fresh_cache_stays_even_beyond_the_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(retention, "CACHE_MAX_ANALYSES", 1)
    root = _repo(tmp_path)
    nu = time.time()
    _cache_bestand(root, "data.sav", "Q1_100", "004_ideas_100", mtime=nu - 1 * 86400)
    vers = _cache_bestand(root, "data.sav", "Q1_500", "004_ideas_500", mtime=nu - 2 * 86400)

    retention.run(root, apply=True, now=nu)

    assert vers.exists()


def test_an_analysis_without_exports_costs_no_export_slot(tmp_path, monkeypatch):
    """A cache-only analysis must not eat up a MAX_ANALYSES slot."""
    monkeypatch.setattr(retention, "MAX_ANALYSES", 2)
    root = _repo(tmp_path)
    nu = time.time()
    # nieuwste: alleen cache. Daarna twee analyses mét exports.
    _cache_bestand(root, "data.sav", "Q1_900", "004_ideas_900", mtime=nu - 10 * 86400)
    for sample, dagen in ((100, 30), (500, 60)):
        _analyse(root, "data.sav", "Q1", sample, nu - dagen * 86400)

    analyses, _ = retention.collect(root)
    weg = retention.select_analyses_for_removal(analyses, now=nu)

    assert weg == [], "beide export-analyses passen binnen het plafond van 2"
