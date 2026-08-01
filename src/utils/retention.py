"""
retention.py - Opruimen van exports/ per analyse, met een prullenbak.

De eenheid is de **analyse** — dataset + vraag + steekproef — niet de map. De
artefacten van één analyse liggen verspreid over vijf mappen (het logboek, de
prompts, het codeboek, de codering, de kosten); een plafond per map kan
prompts/ snoeien en codebook/ laten staan en levert dan een halve analyse op.
Daarom bewaren of verhuizen we een analyse in zijn geheel.

Eén knop: MAX_ANALYSES. Analyses worden gesorteerd op hun nieuwste bestand; de
N nieuwste blijven, de rest gaat naar exports/_prullenbak/.

Twee bodems eronder:
  - Alles binnen PROTECT_DAYS blijft hoe dan ook staan, ook als het plafond
    daardoor wordt overschreden. Beschermde analyses tellen wél mee voor N.
  - De nieuwste analyse blijft altijd staan.

Bestanden die niet bij een bekende analyse horen (restanten van een oudere
naamgeving of van een verwijderde dataset) verhuizen ook naar de prullenbak,
maar pas als ze ouder zijn dan PROTECT_DAYS.

De prullenbak heeft zijn eigen plafond in MB. Dáár — en alleen daar — gaat iets
onherroepelijk weg.

Veiligheidsinvariant: elk pad dat dit script aanraakt ligt onder exports/, en
alleen in de mappen in MANAGED_DIRS. Er wordt nergens iets geïnventariseerd
buiten die lijst.

Aanroep vanuit src/:
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

# Ligt bewust direct onder exports/, niet in de prullenbak: anders valt het
# onder de bak-regel en kan het script zijn eigen audit-spoor wissen.
LOG_PATH = Path(EXPORTS_DIRNAME) / "retention.log"

# =============================================================================
# INSTELLINGEN — pas deze aan vlak vóór een run
# =============================================================================

RETENTION_ENABLED = True

# Hoeveel analyses bewaren we? None = uit; dan verhuist er niets.
# Opgeleverd op None: het gereedschap rapporteert wel (--status) maar kan niets
# verwijderen. Het getal wordt vastgesteld in een apart beslismoment, met de
# --status-uitvoer erbij.
MAX_ANALYSES: Optional[int] = None

# Plafond op de prullenbak in MB. None = uit. Dit is de enige plek waar iets
# onherroepelijk verdwijnt.
TRASH_MAX_MB: Optional[int] = None

# Bodem: wat hierbinnen is gewijzigd gaat nooit weg, ook niet als het plafond
# daardoor wordt overschreden. Dit is de zekerheid tegen een verkeerd ingesteld
# plafond: hoe fout het getal ook is, het werk van deze week overleeft het.
# 0 = uit.
PROTECT_DAYS = 7

# De beheerde mappen. Bewust NIET beheerd: exports/adhoc/,
# exports/diagnostics/, exports/experiment_logs/, en alles buiten exports/.
MANAGED_DIRS = ("verbose_logs", "prompts", "codebook", "coderingen", "costs")



class RetentionError(Exception):
    """Onveilige toestand: het script stopt liever dan dat het iets verplaatst."""


@dataclass(frozen=True)
class Entry:
    """Eén meting van één bestand: pad, mtime en omvang in bytes.

    Alles wordt één keer gemeten. Zou de mtime later opnieuw van disk worden
    gelezen, dan kan een bestand dat tussentijds wordt aangeraakt — de pipeline
    die schrijft terwijl dit draait — de beschermingsaanname omzeilen.
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
    key: AnalysisKey
    entries: list[Entry] = field(default_factory=list)

    @property
    def mtime(self) -> float:
        """De analyse is zo oud als haar nieuwste bestand."""
        return max(e.mtime for e in self.entries)

    @property
    def size(self) -> int:
        return sum(e.size for e in self.entries)


# =============================================================================
# INVENTARISATIE
# =============================================================================

def known_stems(root: Path) -> set[str]:
    """Bekende datasets: wat de cache kent én wat er in data/ ligt.

    Beide bronnen zijn nodig. De cache kent analyses waarvan het .sav-bestand
    is verplaatst; data/ bevat bestanden waarvoor nog niets gedraaid is. Is de
    cache onleesbaar, dan stopt het script: met een onvolledige lijst lijkt
    ineens alles een restant.
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
    """Alle beheerde bestanden, gegroepeerd per analyse, plus de restanten.

    Een bestand hoort bij een analyse als zijn naam volgens de canonieke
    conventie te lezen is (utils.exportNaming). Lukt dat niet, dan is het een
    restant: een oudere naamgeving, of een dataset die niet meer bestaat.
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

    analyses = sorted(per_key.values(), key=lambda a: a.mtime, reverse=True)
    return analyses, restanten


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


def select_analyses_for_removal(
    analyses: list[Analysis],
    now: Optional[float] = None,
) -> list[Analysis]:
    """De analyses voorbij MAX_ANALYSES, exclusief de beschermde.

    De lijst staat op mtime gesorteerd, dus de beschermde analyses en de
    nieuwste vormen samen het begin van de rij. Een beschermde analyse telt
    mee voor het plafond maar wordt zelf nooit geselecteerd — het plafond
    accepteert dan liever een overschrijding.
    """
    if MAX_ANALYSES is None:
        return []

    grens = _grens(now)
    behouden = 0
    weg: list[Analysis] = []
    for analyse in analyses:
        beschermd = analyse.mtime >= grens
        if beschermd or behouden == 0 or behouden < MAX_ANALYSES:
            behouden += 1
            continue
        weg.append(analyse)
    return weg


def select_orphans_for_removal(
    restanten: list[Entry],
    now: Optional[float] = None,
) -> list[Entry]:
    """Restanten ouder dan PROTECT_DAYS. Verse restanten blijven staan."""
    if MAX_ANALYSES is None:
        return []
    grens = _grens(now)
    return [e for e in restanten if e.mtime < grens]


def select_trash_for_removal(
    entries: list[Entry],
    now: Optional[float] = None,
) -> list[Entry]:
    """Wat er uit de prullenbak definitief weg mag, boven TRASH_MAX_MB.

    Ook hier blijft de nieuwste altijd staan, zodat een te krap plafond de bak
    niet in één keer leegt.
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

def move_to_trash(pad: Path, root: Path) -> None:
    """Verplaats naar exports/_prullenbak/ met behoud van het pad onder exports/.

    Ligt het pad al in de bak, dan wordt het verwijderd — de bak is de laatste
    schakel. De membership-test gebruikt het ongeresolvede pad, zodat een
    symlink in de bak niet via .resolve() alsnog de verplaats-tak neemt.
    """
    bak = root / EXPORTS_DIRNAME / TRASH_DIRNAME

    if pad.is_relative_to(bak):
        pad.unlink()
        return

    relatief = pad.relative_to(root / EXPORTS_DIRNAME)
    doel = bak / relatief
    doel.parent.mkdir(parents=True, exist_ok=True)
    if doel.exists():
        _log_regel(root, f"overschreven in prullenbak: {relatief}")
        doel.unlink()
    shutil.move(str(pad), str(doel))


def run(root: Path, apply: bool, now: Optional[float] = None) -> dict:
    """Inventariseer, selecteer en (bij apply) verplaats.

    De volgorde is bewust: eerst wordt álles geïnventariseerd en geselecteerd,
    daarna pas verplaatst. Zou de prullenbak ná de verhuizing worden bekeken,
    dan zag hij de bestanden die in déze run net binnenkwamen — en shutil.move
    behoudt de mtime, dus PROTECT_DAYS beschermt ze daar niet. Eén --apply zou
    dan een bestand kunnen verplaatsen én meteen definitief wissen, waarmee de
    bak nooit respijt bood.
    """
    if not RETENTION_ENABLED:
        return {"uit": True, "analyses": [], "restanten": 0, "prullenbak": {}}

    analyses, restanten = collect(root)
    bak_voor = trash_entries(root)

    weg_analyses = select_analyses_for_removal(analyses, now)
    weg_restanten = select_orphans_for_removal(restanten, now)
    weg_bak = select_trash_for_removal(bak_voor, now)

    if apply:
        for analyse in weg_analyses:
            for entry in analyse.entries:
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
                "mb": round(a.size / 1024 / 1024, 1),
                "laatst": datetime.fromtimestamp(a.mtime).strftime("%Y-%m-%d %H:%M"),
                "beschermd": a.mtime >= grens,
                "weg": a in weg_analyses,
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
    """Tijdstip van de laatste dáádwerkelijke uitvoering.

    Het log bevat ook regels van move_to_trash en van een mislukte run; alleen
    de regel met het "RUN  "-voorvoegsel telt hier.
    """
    log = root / LOG_PATH
    if not log.exists():
        return "nooit"
    runs = [r for r in log.read_text(encoding="utf-8").splitlines()
            if r[STEMPEL_LEN + 2:].startswith("RUN  ")]
    return runs[-1][:STEMPEL_LEN] if runs else "nooit"


def _kort(key: str, breedte: int = 62) -> str:
    """Kort de datasetnaam in, nooit de variabele en de steekproef.

    Die twee zijn juist wat twee analyses van dezelfde dataset onderscheidt;
    een blinde afkapping op breedte snijdt ze eraf en maakt de regels
    onleesbaar gelijk.
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
    venster = (f"beschermd: alles van de afgelopen {PROTECT_DAYS} dagen"
               if PROTECT_DAYS else "beschermingsvenster staat uit")
    print(f"\n{kop}   plafond: {plafond}   ({venster})\n")

    print(f"{'#':>3s}  {'analyse':62s} {'bestanden':>9s} {'MB':>7s} {'laatst':>16s}  ")
    print("-" * 106)
    for i, a in enumerate(verslag["analyses"], 1):
        vlag = "WEG" if a["weg"] else ("beschermd" if a["beschermd"] else "")
        print(f"{i:3d}  {_kort(a['key']):62s} {a['bestanden']:9d} {a['mb']:7.1f} "
              f"{a['laatst']:>16s}  {vlag}")

    print(f"\nrestanten (bij geen analyse): {verslag['restanten']}"
          f"   waarvan weg: {verslag['restanten_weg']}")
    bak = verslag["prullenbak"]
    bak_plafond = f"{TRASH_MAX_MB} MB" if TRASH_MAX_MB is not None else "uit (None)"
    print(f"prullenbak: {bak['bestanden']} bestanden, {bak['mb']} MB"
          f"   plafond: {bak_plafond}   definitief weg: {bak['weg']}")


def opruimen(root: Path = PROJECT_ROOT, aanleiding: str = "") -> str:
    """Voer de opruiming uit, log het resultaat, en geef één regel terug.

    Dit is het haakje voor de app. Een fout wordt niet verzwegen: hij gaat naar
    exports/retention.log én komt terug als tekst, zodat de aanroeper hem kan
    tonen. De vorige opruimer slikte zijn eigen ImportError en was daardoor
    maandenlang stuk zonder dat iemand het merkte.
    """
    try:
        verslag = run(root, apply=True)
    except RetentionError as e:
        _log_regel(root, f"FOUT ({aanleiding}): {e}")
        return f"opruimen mislukt: {e}"

    if verslag.get("uit"):
        return "opruimen staat uit"

    verplaatst = sum(a["bestanden"] for a in verslag["analyses"] if a["weg"])
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
        n += verslag["restanten_weg"]
        _log_regel(PROJECT_ROOT,
                   f"RUN  {n} bestanden verplaatst, "
                   f"{verslag['prullenbak']['weg']} definitief verwijderd")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
