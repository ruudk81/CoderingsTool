"""
retention.py - Clean up exports/ and data/cache/ per analysis, with a trash folder.

The unit is the **analysis** — dataset + question + sample — not the directory.
One analysis scatters its artefacts over five export directories (the log, the
prompts, the codebook, the coding, the costs) plus nine cache files; a per-
directory cap can prune prompts/ while leaving codebook/ in place, and then
yields half an analysis. So an analysis is kept or moved as a whole.

Two domains, one notion of an analysis, two caps:

  MAX_ANALYSES = 10        exports/     — freely restorable (step 7, seconds)
  CACHE_MAX_ANALYSES = 25  data/cache/  — restorable only by rerunning the whole
                                          chain 0-6

The difference is deliberate, not sloppiness. Throwing away an export costs you
thirty seconds of compute; throwing away a cached analysis costs a full chain —
minutes to hours, and real money — to reclaim ~5 MB. The wider cache cap also
guarantees "cache superset of exports": every export kept can always still be
regenerated.

Analyses are sorted by their newest file; per domain the N newest that have files
in *that* domain are kept, the rest go to exports/_prullenbak/.

Two floors underneath, in both domains:
  - Anything modified within PROTECT_DAYS stays regardless, even when that
    exceeds the cap. Protected analyses do count towards N.
  - The newest analysis always stays.

Files that belong to no known analysis (leftovers from an older naming scheme or
from a deleted dataset) also move to the trash, but only once they are older than
PROTECT_DAYS. In the cache one extra kind counts as a leftover: a .pkl whose db
row reads 'invalid'. CacheManager does not delete the file on invalidation, so
such files linger while no code path can read them any more. They follow their own
age rather than their analysis — unlike an export under an old name, this is no
longer an artefact of the analysis but a corpse.

The trash has a cap of its own, in MB. There — and only there — something goes
irreversibly. Note: **restoring from the trash is not symmetric for the cache.**
The file comes back, the db row stays invalid, so the cache stays invalid. If you
really want such an analysis back, rerun it.

Safety invariant: every path this script touches lies under exports/ (and there
only inside MANAGED_DIRS) or is a .pkl directly under data/cache/. The two .db
files are never touched, only updated.

Invocation from src/:
    python -m utils.retention              # dry-run
    python -m utils.retention --apply
    python -m utils.retention --status
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from utils.exportNaming import parse_export_filename

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPORTS_DIRNAME = "exports"
TRASH_DIRNAME = "_prullenbak"

# The cache lives outside exports/, so it gets its own branch in the same trash
# folder — one bin, one cap, one place to look. Cache names are flat (no
# subdirectories), so the branch is flat too.
CACHE_DIRNAME = Path("data") / "cache"
CACHE_TRASH_SUBDIR = "data_cache"
CACHE_DB_NAME = "cache.db"

# Deliberately directly under exports/, not in the trash: otherwise it falls
# under the trash rule and the script could erase its own audit trail.
LOG_PATH = Path(EXPORTS_DIRNAME) / "retention.log"

# =============================================================================
# SETTINGS — adjust these right before a run
# =============================================================================

RETENTION_ENABLED = True

# How many analyses do we keep? None = off; then nothing moves.
# Protected analyses (see PROTECT_DAYS) count towards this number: with six
# protected and a cap of 10, four more are kept on top.
MAX_ANALYSES: Optional[int] = 10

# The same, but for data/cache/. Deliberately wider than MAX_ANALYSES: an export
# is free to restore (step 7), a cached analysis only by rerunning the whole
# chain 0-6. As long as this number exceeds MAX_ANALYSES, "cache superset of
# exports" holds and every export kept can still be reproduced.
CACHE_MAX_ANALYSES: Optional[int] = 25

# Cap on the trash in MB. None = off. This is the only place where something
# disappears irreversibly — as long as this is None everything merely moves and
# you can put it all back.
TRASH_MAX_MB: Optional[int] = None

# Floor: anything modified within this window never goes, even when that exceeds
# the cap. This is the guarantee against a badly set cap: however wrong the
# number, this week's work survives it.
# 0 = off.
PROTECT_DAYS = 7

# The managed directories. Deliberately NOT managed: exports/adhoc/,
# exports/diagnostics/, exports/experiment_logs/, and everything outside exports/.
MANAGED_DIRS = ("verbose_logs", "prompts", "codebook", "coderingen", "costs")



class RetentionError(Exception):
    """Unsafe state: the script would rather stop than move anything."""


@dataclass(frozen=True)
class Entry:
    """One measurement of one file: path, mtime and size in bytes.

    Everything is measured once. Were the mtime read from disk again later, a
    file touched in the meantime — the pipeline writing while this runs — could
    slip past the protection assumption.
    """
    path: Path
    mtime: float
    size: int


@dataclass(frozen=True)
class AnalysisKey:
    dataset: str
    var_name: str
    sample: str

    def __str__(self) -> str:
        return f"{self.dataset} · {self.var_name} · {self.sample}"


@dataclass
class Analysis:
    """One analysis, with its files in both domains.

    `entries` are the exports, `cache_entries` the pickles under data/cache/. An
    analysis need not exist in both domains: a fresh run has no exports yet, and
    an analysis whose exports were already cleaned up keeps its cache.
    """
    key: AnalysisKey
    entries: list[Entry] = field(default_factory=list)
    cache_entries: list[Entry] = field(default_factory=list)

    @property
    def alle(self) -> list[Entry]:
        return self.entries + self.cache_entries

    @property
    def mtime(self) -> float:
        """An analysis is as old as its newest file, in whichever domain.

        Deliberately across both domains: an analysis where only step 7 was just
        rerun is recent as a whole, and its cache should not suddenly count as
        old.
        """
        return max(e.mtime for e in self.alle)

    @property
    def size(self) -> int:
        return sum(e.size for e in self.alle)


# =============================================================================
# INVENTARISATIE
# =============================================================================

def known_stems(root: Path) -> set[str]:
    """Known datasets: what the cache knows plus what sits in data/.

    Both sources are needed. The cache knows analyses whose .sav file has been
    moved; data/ holds files nothing has been run on yet. If the cache is
    unreadable the script stops: with an incomplete list, suddenly everything
    looks like a leftover.
    """
    stems = {p.name for p in (root / "data").glob("*.sav")}
    db = root / "data" / "cache" / "cache.db"
    if not db.exists():
        raise RetentionError(f"cache-database niet gevonden: {db}")
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            stems |= {r[0] for r in con.execute(
                "SELECT DISTINCT filename FROM cache_metadata WHERE status='valid'")}
        finally:
            con.close()
    except sqlite3.Error as e:
        raise RetentionError(f"cache-database onleesbaar ({db}): {e}") from e
    if not stems:
        raise RetentionError(
            "geen enkele dataset gevonden in data/ of in de cache — zonder die "
            "lijst is elk bestand een restant, en dat is bijna nooit waar")
    return stems


def collect(root: Path) -> tuple[list[Analysis], list[Entry]]:
    """All managed files from both domains, per analysis, plus the leftovers.

    An export belongs to an analysis when its name reads according to the
    canonical convention (utils.exportNaming). If it does not, it is a leftover:
    an older naming scheme, or a dataset that no longer exists.

    A cache file is not read from its name but from cache.db, where filename and
    variable_key sit as separate columns. More reliable than the export side:
    there is nothing to split wrongly.
    """
    exports = (root / EXPORTS_DIRNAME).resolve()
    stems = known_stems(root)

    per_key: dict[AnalysisKey, Analysis] = {}
    restanten: list[Entry] = []

    for mapnaam in MANAGED_DIRS:
        d = root / EXPORTS_DIRNAME / mapnaam
        if not d.is_dir():
            continue
        for pad in sorted(d.glob("*")):
            if not pad.is_file():
                continue
            if not pad.resolve().is_relative_to(exports):
                raise RetentionError(f"pad wijst buiten exports/: {pad}")
            stat = pad.stat()
            entry = Entry(path=pad, mtime=stat.st_mtime, size=stat.st_size)
            geparsed = parse_export_filename(pad.name, stems)
            if geparsed is None:
                restanten.append(entry)
                continue
            key = AnalysisKey(geparsed.dataset, geparsed.var_name, geparsed.sample)
            per_key.setdefault(key, Analysis(key)).entries.append(entry)

    cache_restanten = _verzamel_cache(root, per_key)
    analyses = sorted(per_key.values(), key=lambda a: a.mtime, reverse=True)
    # Only the export leftovers go through the attach step: it matches on old
    # file names, and a cache file has none by definition.
    restanten = _hecht_restanten_aan_analyses(restanten, analyses, stems)
    return analyses, restanten + cache_restanten


def _cache_analyse_key(filename: str, variable_key: str) -> Optional[AnalysisKey]:
    """The analysis a cache entry belongs to.

    The cache does not have the parsing problem exports have: filename and
    variable_key sit as separate columns in cache.db, so only the sample still
    has to be separated from the variable name. Same requirement as in
    exportNaming: a number or "full", otherwise it is not a readable key.

    Edge case: for merged variables, variable_key also carries the merge
    configuration (Q1+Q2_concat_semicolon_skip_500), while the export uses the
    bare variable name. Such an analysis then does not match its exports and
    becomes an analysis of its own. That is the safe side: matching only decides
    the grouping, never whether something may go.
    """
    var_name, _, sample = variable_key.rpartition("_")
    if not var_name or (sample != "full" and not sample.isdigit()):
        return None
    return AnalysisKey(Path(filename).stem.replace(" ", "_"), var_name, sample)


def _verzamel_cache(root: Path, per_key: dict[AnalysisKey, Analysis]) -> list[Entry]:
    """Attach the cache files to their analysis. Returns the cache leftovers.

    A leftover here is anything no live code path can still reach: a .pkl without
    a db row, and a .pkl whose row reads 'invalid'. The second happens more often
    than you would think — CacheManager invalidates an entry but does not delete
    the file, so discarded caches linger.
    """
    cache = (root / CACHE_DIRNAME).resolve()
    if not cache.is_dir():
        return []
    pickles = sorted(cache.glob("*.pkl"))
    if not pickles:
        return []

    db = cache / CACHE_DB_NAME
    if not db.exists():
        raise RetentionError(f"cache-database niet gevonden: {db}")
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            rijen = {
                str(Path(pad).resolve()): (bestand, vk, status)
                for pad, bestand, vk, status in con.execute(
                    "SELECT cache_path, filename, variable_key, status FROM cache_metadata")
            }
        finally:
            con.close()
    except sqlite3.Error as e:
        raise RetentionError(f"cache-database onleesbaar ({db}): {e}") from e

    restanten: list[Entry] = []
    for pad in pickles:
        if not pad.resolve().is_relative_to(cache):
            raise RetentionError(f"pad wijst buiten data/cache/: {pad}")
        stat = pad.stat()
        entry = Entry(path=pad, mtime=stat.st_mtime, size=stat.st_size)
        rij = rijen.get(str(pad.resolve()))
        key = _cache_analyse_key(rij[0], rij[1]) if rij and rij[2] == "valid" else None
        if key is None:
            restanten.append(entry)
            continue
        per_key.setdefault(key, Analysis(key)).cache_entries.append(entry)
    return restanten


def _hecht_restanten_aan_analyses(
    restanten: list[Entry],
    analyses: list[Analysis],
    stems: set[str],
) -> list[Entry]:
    """Attach every unambiguously assignable leftover to its analysis.

    A file under an old name often does belong to an analysis you are keeping —
    the log of a step that has not been rerun since the rename. Without this step
    such a file is protected only by its own age and disappears within a week,
    while the analysis stays. That conflicts with the principle that the analysis
    is the unit.

    Assignment happens only on certainty: the name must start with dataset +
    variable of exactly one analysis, and that analysis's sample must appear as a
    separate name part. That leaves the codings whose sample was never written
    down untouched — several analyses are possible there, and guessing is how you
    lose a file under the wrong label.
    """
    origineel = {Path(s).stem.replace(" ", "_"): Path(s).stem for s in stems}
    overig: list[Entry] = []

    for entry in restanten:
        naam = entry.path.name
        delen = set(naam.replace(".", "_").split("_"))
        treffers = [
            a for a in analyses
            if a.key.sample in delen
            and any(naam.startswith(f"{p}_{a.key.var_name}_")
                    for p in _prefixen(a.key.dataset, origineel))
        ]
        if len(treffers) == 1:
            treffers[0].entries.append(entry)
        else:
            overig.append(entry)

    return overig


def _prefixen(dataset_slug: str, origineel: dict[str, str]) -> set[str]:
    """The forms a dataset name can take at the front of an old file name.

    Old names sometimes used spaces, sometimes underscores, and the verbose logs
    truncated the name at 50 characters.
    """
    vormen = {dataset_slug, origineel.get(dataset_slug, dataset_slug)}
    return vormen | {v[:50] for v in vormen}


def trash_entries(root: Path) -> list[Entry]:
    """Alles wat in de prullenbak ligt, nieuwste eerst."""
    bak = root / EXPORTS_DIRNAME / TRASH_DIRNAME
    entries: list[Entry] = []
    if bak.is_dir():
        for pad in bak.rglob("*"):
            if pad.is_file():
                stat = pad.stat()
                entries.append(Entry(path=pad, mtime=stat.st_mtime, size=stat.st_size))
    entries.sort(key=lambda e: e.mtime, reverse=True)
    return entries


# =============================================================================
# SELECTIE
# =============================================================================

def _grens(now: Optional[float]) -> float:
    return (now if now is not None else time.time()) - PROTECT_DAYS * 86400


def _selecteer(
    analyses: list[Analysis],
    plafond: Optional[int],
    entries_van,
    now: Optional[float],
) -> list[Analysis]:
    """The analyses beyond the cap in one domain, excluding the protected ones.

    The list is sorted on mtime, so the protected analyses and the newest one
    together form the head of the queue. A protected analysis counts towards the
    cap but is never selected itself — the cap would rather accept an overrun.

    Analyses without files in *this* domain are skipped: an analysis whose
    exports were already cleaned up must not keep occupying an export slot, or
    the effective cap shrinks with every cleanup.
    """
    if plafond is None:
        return []

    grens = _grens(now)
    behouden = 0
    weg: list[Analysis] = []
    for analyse in analyses:
        if not entries_van(analyse):
            continue
        beschermd = analyse.mtime >= grens
        if beschermd or behouden == 0 or behouden < plafond:
            behouden += 1
            continue
        weg.append(analyse)
    return weg


def select_analyses_for_removal(
    analyses: list[Analysis],
    now: Optional[float] = None,
) -> list[Analysis]:
    """The exports beyond MAX_ANALYSES, excluding the protected ones."""
    return _selecteer(analyses, MAX_ANALYSES, lambda a: a.entries, now)


def select_cache_for_removal(
    analyses: list[Analysis],
    now: Optional[float] = None,
) -> list[Analysis]:
    """The same for data/cache/, against the wider CACHE_MAX_ANALYSES.

    As long as CACHE_MAX_ANALYSES exceeds MAX_ANALYSES, an analysis's cache never
    disappears before its exports, and every export kept stays reproducible.
    """
    return _selecteer(analyses, CACHE_MAX_ANALYSES, lambda a: a.cache_entries, now)


def select_orphans_for_removal(
    restanten: list[Entry],
    now: Optional[float] = None,
) -> list[Entry]:
    """Restanten ouder dan PROTECT_DAYS. Verse restanten blijven staan."""
    if MAX_ANALYSES is None and CACHE_MAX_ANALYSES is None:
        return []
    grens = _grens(now)
    return [e for e in restanten if e.mtime < grens]


def select_trash_for_removal(
    entries: list[Entry],
    now: Optional[float] = None,
) -> list[Entry]:
    """What may go from the trash for good, beyond TRASH_MAX_MB.

    Here too the newest always stays, so that a cap set too tight does not empty
    the bin in one go.
    """
    if TRASH_MAX_MB is None:
        return []
    limiet = TRASH_MAX_MB * 1024 * 1024
    behouden: list[Entry] = []
    cumulatief = 0
    for entry in entries:
        if behouden and cumulatief + entry.size > limiet:
            break
        behouden.append(entry)
        cumulatief += entry.size
    return entries[len(behouden):]


# =============================================================================
# UITVOEREN
# =============================================================================

def _invalideer_cache_rij(root: Path, pad: Path) -> None:
    """Set the db row of a moved cache file to 'invalid'.

    Without this a 'valid' row remains, pointing at a file that is gone.
    CacheManager catches that on the next use, but by then it is a malfunction
    rather than a decision. Comparison happens on the resolved path, because the
    db stores the path as it was assembled at write time — not necessarily in the
    same form.

    Note the asymmetry: restoring the file from the trash does not make this row
    valid again. See the module docstring.
    """
    db = root / CACHE_DIRNAME / CACHE_DB_NAME
    if not db.exists():
        return
    doel = str(pad.resolve())
    try:
        with sqlite3.connect(db, timeout=30) as con:
            ids = [rid for rid, p in con.execute("SELECT rowid, cache_path FROM cache_metadata")
                   if str(Path(p).resolve()) == doel]
            con.executemany(
                "UPDATE cache_metadata SET status = 'invalid' WHERE rowid = ?",
                [(i,) for i in ids])
    except sqlite3.Error as e:
        raise RetentionError(f"cache-database niet bij te werken ({db}): {e}") from e


def move_to_trash(pad: Path, root: Path) -> None:
    """Move to exports/_prullenbak/, preserving where it came from.

    Exports keep their path under exports/; cache files go to the flat branch
    data_cache/ (cache names have no subdirectories) and their db row goes along
    as 'invalid'.

    If the path is already in the bin it is deleted — the bin is the last link.
    The membership test uses the unresolved path, so that a symlink inside the
    bin does not take the move branch anyway via .resolve().
    """
    bak = root / EXPORTS_DIRNAME / TRASH_DIRNAME

    if pad.is_relative_to(bak):
        pad.unlink()
        return

    if pad.is_relative_to(root / CACHE_DIRNAME):
        _invalideer_cache_rij(root, pad)
        relatief = Path(CACHE_TRASH_SUBDIR) / pad.name
    else:
        relatief = pad.relative_to(root / EXPORTS_DIRNAME)

    doel = bak / relatief
    doel.parent.mkdir(parents=True, exist_ok=True)
    if doel.exists():
        _log_regel(root, f"overschreven in prullenbak: {relatief}")
        doel.unlink()
    shutil.move(str(pad), str(doel))


def run(root: Path, apply: bool, now: Optional[float] = None) -> dict:
    """Inventory, select and (on apply) move.

    The order is deliberate: everything is inventoried and selected first, and
    only then moved. Were the trash inspected *after* the move, it would see the
    files that arrived in *this* run — and shutil.move preserves the mtime, so
    PROTECT_DAYS does not protect them there. A single --apply could then move a
    file and immediately delete it for good, so the bin never granted any
    reprieve.
    """
    if not RETENTION_ENABLED:
        return {"uit": True, "analyses": [], "restanten": 0, "prullenbak": {}}

    analyses, restanten = collect(root)
    bak_voor = trash_entries(root)

    weg_analyses = select_analyses_for_removal(analyses, now)
    weg_cache = select_cache_for_removal(analyses, now)
    weg_restanten = select_orphans_for_removal(restanten, now)
    weg_bak = select_trash_for_removal(bak_voor, now)

    if apply:
        for analyse in weg_analyses:
            for entry in analyse.entries:
                move_to_trash(entry.path, root)
        for analyse in weg_cache:
            for entry in analyse.cache_entries:
                move_to_trash(entry.path, root)
        for entry in weg_restanten:
            move_to_trash(entry.path, root)
        for entry in weg_bak:
            move_to_trash(entry.path, root)

    grens = _grens(now)
    return {
        "uit": False,
        "analyses": [
            {
                "key": str(a.key),
                "bestanden": len(a.entries),
                "mb": round(sum(e.size for e in a.entries) / 1024 / 1024, 1),
                "cache_bestanden": len(a.cache_entries),
                "cache_mb": round(sum(e.size for e in a.cache_entries) / 1024 / 1024, 1),
                "laatst": datetime.fromtimestamp(a.mtime).strftime("%Y-%m-%d %H:%M"),
                "beschermd": a.mtime >= grens,
                "weg": a in weg_analyses,
                "cache_weg": a in weg_cache,
            }
            for a in analyses
        ],
        "restanten": len(restanten),
        "restanten_weg": len(weg_restanten),
        "prullenbak": {
            "bestanden": len(bak_voor),
            "mb": round(sum(e.size for e in bak_voor) / 1024 / 1024, 1),
            "weg": len(weg_bak),
        },
    }


# =============================================================================
# LOG + UITVOER
# =============================================================================

def _log_regel(root: Path, tekst: str) -> None:
    log = root / LOG_PATH
    log.parent.mkdir(parents=True, exist_ok=True)
    stempel = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"{stempel}  {tekst}\n")


STEMPEL_LEN = len("2026-08-01 00:00:00")


def _laatste_run(root: Path) -> str:
    """Timestamp of the last actual execution.

    The log also holds lines from move_to_trash and from a failed run; only the
    line with the "RUN  " prefix counts here.
    """
    log = root / LOG_PATH
    if not log.exists():
        return "nooit"
    runs = [r for r in log.read_text(encoding="utf-8").splitlines()
            if r[STEMPEL_LEN + 2:].startswith("RUN  ")]
    return runs[-1][:STEMPEL_LEN] if runs else "nooit"


def _kort(key: str, breedte: int = 62) -> str:
    """Shorten the dataset name, never the variable and the sample.

    Those two are exactly what tells two analyses of the same dataset apart; a
    blind truncation on width cuts them off and makes the lines unreadably
    identical.
    """
    dataset, _, staart = key.partition(" · ")
    ruimte = breedte - len(staart) - 3
    if len(dataset) > ruimte:
        dataset = dataset[: max(ruimte - 1, 1)] + "…"
    return f"{dataset} · {staart}"


def _print_verslag(verslag: dict, apply: bool) -> None:
    if verslag.get("uit"):
        print("RETENTION_ENABLED staat op False — niets gedaan.")
        return

    kop = "VERWIJDERD" if apply else "ZOU VERWIJDEREN (dry-run — gebruik --apply)"
    plafond = f"{MAX_ANALYSES} analyses" if MAX_ANALYSES is not None else "uit (None)"
    cache_plafond = (f"{CACHE_MAX_ANALYSES} analyses"
                     if CACHE_MAX_ANALYSES is not None else "uit (None)")
    venster = (f"beschermd: alles van de afgelopen {PROTECT_DAYS} dagen"
               if PROTECT_DAYS else "beschermingsvenster staat uit")
    print(f"\n{kop}")
    print(f"plafond exports: {plafond}   plafond cache: {cache_plafond}   ({venster})\n")

    print(f"{'#':>3s}  {'analyse':52s} {'exports':>9s} {'MB':>7s} "
          f"{'cache':>7s} {'MB':>7s} {'laatst':>16s}  ")
    print("-" * 118)
    for i, a in enumerate(verslag["analyses"], 1):
        if a["weg"] and a["cache_weg"]:
            vlag = "WEG"
        elif a["weg"]:
            vlag = "exports WEG"
        elif a["cache_weg"]:
            vlag = "cache WEG"
        else:
            vlag = "beschermd" if a["beschermd"] else ""
        print(f"{i:3d}  {_kort(a['key'], 52):52s} {a['bestanden']:9d} {a['mb']:7.1f} "
              f"{a['cache_bestanden']:7d} {a['cache_mb']:7.1f} {a['laatst']:>16s}  {vlag}")

    print(f"\nrestanten (bij geen analyse): {verslag['restanten']}"
          f"   waarvan weg: {verslag['restanten_weg']}")
    bak = verslag["prullenbak"]
    bak_plafond = f"{TRASH_MAX_MB} MB" if TRASH_MAX_MB is not None else "uit (None)"
    print(f"prullenbak: {bak['bestanden']} bestanden, {bak['mb']} MB"
          f"   plafond: {bak_plafond}   definitief weg: {bak['weg']}")


def opruimen(root: Path = PROJECT_ROOT, aanleiding: str = "") -> str:
    """Run the cleanup, log the result, and return a single line.

    This is the hook for the app. An error is not swallowed: it goes to
    exports/retention.log AND comes back as text, so the caller can show it. The
    previous cleaner swallowed its own ImportError and was therefore broken for
    months without anyone noticing.
    """
    try:
        verslag = run(root, apply=True)
    except RetentionError as e:
        _log_regel(root, f"FOUT ({aanleiding}): {e}")
        return f"opruimen mislukt: {e}"

    if verslag.get("uit"):
        return "opruimen staat uit"

    verplaatst = sum(a["bestanden"] for a in verslag["analyses"] if a["weg"])
    verplaatst += sum(a["cache_bestanden"] for a in verslag["analyses"] if a["cache_weg"])
    verplaatst += verslag["restanten_weg"]
    definitief = verslag["prullenbak"]["weg"]
    _log_regel(root, f"RUN  ({aanleiding}) {verplaatst} bestanden verplaatst, "
                     f"{definitief} definitief verwijderd")
    if not verplaatst and not definitief:
        return "opruimen: niets te doen"
    return (f"opruimen: {verplaatst} naar de prullenbak, "
            f"{definitief} definitief verwijderd")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="verplaats daadwerkelijk (standaard is dry-run)")
    parser.add_argument("--status", action="store_true",
                        help="toon de analyses en wanneer er laatst gedraaid is")
    args = parser.parse_args(argv)

    doe_het = args.apply and not args.status
    try:
        verslag = run(PROJECT_ROOT, apply=doe_het)
    except RetentionError as e:
        print(f"FOUT: {e}", file=sys.stderr)
        _log_regel(PROJECT_ROOT, f"FOUT: {e}")
        return 1

    if args.status:
        print(f"\nLaatste uitvoering: {_laatste_run(PROJECT_ROOT)}")
    _print_verslag(verslag, apply=doe_het)

    if doe_het and not verslag.get("uit"):
        n = sum(a["bestanden"] for a in verslag["analyses"] if a["weg"])
        n += sum(a["cache_bestanden"] for a in verslag["analyses"] if a["cache_weg"])
        n += verslag["restanten_weg"]
        _log_regel(PROJECT_ROOT,
                   f"RUN  {n} bestanden verplaatst, "
                   f"{verslag['prullenbak']['weg']} definitief verwijderd")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
