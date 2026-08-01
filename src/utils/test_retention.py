"""Tests voor het retentiegereedschap."""

import pytest

from utils.retention import (
    RetentionError,
    RetentionRule,
    resolve_entries,
    select_for_removal,
    run,
)


def _maak(root, relpad, grootte=10, mtime=None):
    p = root / relpad
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x" * grootte)
    if mtime is not None:
        import os
        os.utime(p, (mtime, mtime))
    return p


def test_rule_outside_exports_raises(tmp_path):
    """De invariant: alles onder exports/. Een regel die daarbuiten wijst
    stopt het script, in plaats van stilzwijgend overgeslagen te worden."""
    _maak(tmp_path, "data/cache/iets.pkl")
    regel = RetentionRule("data/cache/*", max_entries=1)
    with pytest.raises(RetentionError, match="buiten exports"):
        resolve_entries(regel, tmp_path)


def test_entries_sorted_newest_first(tmp_path):
    _maak(tmp_path, "exports/logs/oud.txt", mtime=1000)
    _maak(tmp_path, "exports/logs/nieuw.txt", mtime=2000)
    namen = [e.path.name for e in resolve_entries(RetentionRule("exports/logs/*"), tmp_path)]
    assert namen == ["nieuw.txt", "oud.txt"]


def test_none_ceilings_remove_nothing(tmp_path):
    """Zo wordt het gereedschap opgeleverd: geen plafond, geen macht."""
    for i in range(5):
        _maak(tmp_path, f"exports/logs/l{i}.txt", mtime=1000 + i)
    entries = resolve_entries(RetentionRule("exports/logs/*"), tmp_path)
    assert select_for_removal(entries, RetentionRule("exports/logs/*")) == []


def test_max_entries_keeps_newest(tmp_path):
    for i in range(5):
        _maak(tmp_path, f"exports/logs/l{i}.txt", mtime=1000 + i)
    regel = RetentionRule("exports/logs/*", max_entries=2)
    entries = resolve_entries(regel, tmp_path)
    weg = [e.path.name for e in select_for_removal(entries, regel)]
    assert weg == ["l2.txt", "l1.txt", "l0.txt"]


def test_max_mb_counts_cumulative(tmp_path):
    mb = 1024 * 1024
    for i in range(4):
        _maak(tmp_path, f"exports/big/b{i}.bin", grootte=mb, mtime=1000 + i)
    regel = RetentionRule("exports/big/*", max_mb=2)
    entries = resolve_entries(regel, tmp_path)
    weg = [e.path.name for e in select_for_removal(entries, regel)]
    assert weg == ["b1.bin", "b0.bin"]


def test_apply_moves_to_trash_preserving_path(tmp_path):
    for i in range(3):
        _maak(tmp_path, f"exports/logs/l{i}.txt", mtime=1000 + i)
    regels = [RetentionRule("exports/logs/*", max_entries=1)]

    run(tmp_path, regels, apply=True)

    assert (tmp_path / "exports/logs/l2.txt").exists()
    assert (tmp_path / "exports/_prullenbak/logs/l0.txt").exists()
    assert (tmp_path / "exports/_prullenbak/logs/l1.txt").exists()
    assert not (tmp_path / "exports/logs/l0.txt").exists()


def test_dry_run_moves_nothing(tmp_path):
    for i in range(3):
        _maak(tmp_path, f"exports/logs/l{i}.txt", mtime=1000 + i)
    regels = [RetentionRule("exports/logs/*", max_entries=1)]

    verslag = run(tmp_path, regels, apply=False)

    assert (tmp_path / "exports/logs/l0.txt").exists()
    assert not (tmp_path / "exports/_prullenbak").exists()
    assert verslag[0]["te_verwijderen"] == 2


def test_recent_entries_are_never_removed(tmp_path):
    """Het beschermingsvenster: wat deze week is gewijzigd blijft, ook als het
    plafond daardoor wordt overschreden."""
    import time
    nu = time.time()
    for i in range(5):
        _maak(tmp_path, f"exports/logs/vers{i}.txt", mtime=nu - i * 3600)

    regel = RetentionRule("exports/logs/*", max_entries=1)
    entries = resolve_entries(regel, tmp_path)

    assert select_for_removal(entries, regel, now=nu) == []


def test_ceiling_may_be_exceeded_by_protected_entries(tmp_path):
    """Twee verse bestanden bij een plafond van 1: het plafond accepteert dat."""
    import time
    nu = time.time()
    _maak(tmp_path, "exports/logs/vers_a.txt", mtime=nu - 3600)
    _maak(tmp_path, "exports/logs/vers_b.txt", mtime=nu - 7200)
    _maak(tmp_path, "exports/logs/oud.txt", mtime=nu - 30 * 86400)

    regel = RetentionRule("exports/logs/*", max_entries=1)
    entries = resolve_entries(regel, tmp_path)
    weg = [e.path.name for e in select_for_removal(entries, regel, now=nu)]

    assert weg == ["oud.txt"]


def test_protected_entries_count_toward_the_ceiling(tmp_path):
    """Eén vers bestand vult het plafond van 2 half; van de oude blijft er
    dus nog één over, de rest gaat weg."""
    import time
    nu = time.time()
    _maak(tmp_path, "exports/logs/vers.txt", mtime=nu - 3600)
    _maak(tmp_path, "exports/logs/oud1.txt", mtime=nu - 30 * 86400)
    _maak(tmp_path, "exports/logs/oud2.txt", mtime=nu - 31 * 86400)
    _maak(tmp_path, "exports/logs/oud3.txt", mtime=nu - 32 * 86400)

    regel = RetentionRule("exports/logs/*", max_entries=2)
    entries = resolve_entries(regel, tmp_path)
    weg = [e.path.name for e in select_for_removal(entries, regel, now=nu)]

    assert weg == ["oud2.txt", "oud3.txt"]


def test_trash_deletes_instead_of_recursing(tmp_path):
    """De prullenbak is de laatste schakel: wat daar buiten het plafond valt
    gaat echt weg, en belandt niet in een prullenbak binnen de prullenbak.
    Gebruikt de mappenstructuur die move_to_trash zelf aanmaakt."""
    import time
    oud = time.time() - 60 * 86400
    for i in range(3):
        _maak(tmp_path, f"exports/_prullenbak/logs/oud{i}.txt", mtime=oud + i)
    regels = [RetentionRule("exports/_prullenbak/**/*", max_entries=1)]

    run(tmp_path, regels, apply=True)

    assert (tmp_path / "exports/_prullenbak/logs/oud2.txt").exists()
    assert not (tmp_path / "exports/_prullenbak/logs/oud0.txt").exists()
    assert not (tmp_path / "exports/_prullenbak/_prullenbak").exists()


def test_newest_entry_survives_a_tiny_max_mb(tmp_path):
    """Een plafond kleiner dan het nieuwste bestand mag niet de hele map wissen."""
    import time
    mb, oud = 1024 * 1024, time.time() - 60 * 86400
    _maak(tmp_path, "exports/big/nieuw.bin", grootte=5 * mb, mtime=oud + 10)
    _maak(tmp_path, "exports/big/oud.bin", grootte=1 * mb, mtime=oud)
    regel = RetentionRule("exports/big/*", max_mb=2)
    weg = [e.path.name for e in select_for_removal(resolve_entries(regel, tmp_path), regel)]
    assert weg == ["oud.bin"]


def test_rule_validated_even_when_glob_matches_nothing(tmp_path):
    """Een foute regel moet stoppen vóórdat er iets verplaatst wordt, ook als
    de glob toevallig niets matcht (en de per-entry-controle dus niet afgaat)."""
    regel = RetentionRule("data/cache/*", max_entries=1)
    with pytest.raises(RetentionError, match="buiten exports"):
        run(tmp_path, [regel], apply=False)
