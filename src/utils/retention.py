"""
retention.py - Plafonds op exports/, met een prullenbak.

Eén primitief: een regel is een glob plus een plafond. De entries worden
gesorteerd op mtime (nieuwste eerst) en van boven naar beneden opgeteld; zodra
een plafond wordt overschreden gaat alles vanaf die entry naar
exports/_prullenbak/. De nieuwste blijven dus altijd staan.

Boven op het plafond ligt één bodem: alles binnen PROTECT_DAYS blijft hoe dan
ook staan, ook als het plafond daardoor wordt overschreden.

Veiligheidsinvariant: elk pad dat dit script aanraakt ligt onder exports/. Er
wordt nergens een map geïnventariseerd die niet in RULES staat.

Aanroep vanuit src/:
    python -m utils.retention              # dry-run
    python -m utils.retention --apply
    python -m utils.retention --status
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPORTS_DIRNAME = "exports"
TRASH_DIRNAME = "_prullenbak"
LOG_NAME = "retention.log"

# =============================================================================
# REGELS — pas deze aan vlak vóór een run
# =============================================================================
#
# max_entries : hard plafond op het aantal entries
# max_mb      : hard plafond op de cumulatieve omvang in MB
# beide None  : die regel staat uit
#
# Opgeleverd met alles op None: het gereedschap rapporteert wel (--status)
# maar verwijdert niets. De getallen worden vastgesteld in een apart
# beslismoment, met de --status-uitvoer erbij.

RETENTION_ENABLED = True

# Bodem onder alle regels: wat hierbinnen is gewijzigd gaat nooit weg, ook niet
# als het plafond daardoor wordt overschreden. Beschermde entries tellen wél mee
# voor het plafond. Dit is de zekerheid tegen een verkeerd ingesteld plafond:
# hoe fout het getal ook is, het werk van deze week overleeft het. 0 = uit.
PROTECT_DAYS = 7


@dataclass(frozen=True)
class RetentionRule:
    glob: str
    max_entries: Optional[int] = None
    max_mb: Optional[int] = None


RULES: list[RetentionRule] = [
    RetentionRule("exports/verbose_logs/*", max_entries=None, max_mb=None),
    RetentionRule("exports/prompts/*",      max_entries=None, max_mb=None),
    RetentionRule("exports/codebook/*",     max_entries=None, max_mb=None),
    RetentionRule("exports/coderingen/*",   max_entries=None, max_mb=None),
    RetentionRule("exports/costs/*",        max_entries=None, max_mb=None),
    RetentionRule("exports/_prullenbak/*",  max_entries=None, max_mb=None),
]

# Bewust niet beheerd: exports/adhoc/, exports/diagnostics/,
# exports/experiment_logs/ en alles onder data/.


class RetentionError(Exception):
    """Een regel wijst buiten exports/, of een verplaatsing mislukte."""


def entry_size(path: Path) -> int:
    """Omvang in bytes; voor een map de som van alle bestanden erin."""
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return path.stat().st_size


def resolve_entries(rule: RetentionRule, root: Path) -> list[Path]:
    """De entries van een regel, nieuwste eerst.

    Toetst elke entry aan de invariant. Ligt er één buiten exports/, dan stopt
    het script — een genegeerde regel is een regel die je niet opmerkt.
    """
    exports = (root / EXPORTS_DIRNAME).resolve()
    entries = sorted(root.glob(rule.glob), key=lambda p: p.stat().st_mtime,
                     reverse=True)
    for entry in entries:
        if not entry.resolve().is_relative_to(exports):
            raise RetentionError(
                f"regel {rule.glob!r} wijst buiten exports/: {entry}")
    return entries


def select_for_removal(
    entries: list[Path],
    rule: RetentionRule,
    now: Optional[float] = None,
) -> list[Path]:
    """De entries die weg mogen: alles voorbij het strengste plafond.

    Entries binnen PROTECT_DAYS worden nooit geselecteerd, ook niet als het
    plafond daardoor wordt overschreden. Ze tellen wel mee voor dat plafond.
    Omdat de lijst op mtime is gesorteerd vormen ze altijd het begin van de rij,
    dus een `break` na de eerste onbeschermde entry slaat er nooit één over.

    `now` bestaat alleen zodat tests een tijdstip kunnen vastzetten.
    """
    if rule.max_entries is None and rule.max_mb is None:
        return []

    grens = (now if now is not None else time.time()) - PROTECT_DAYS * 86400
    limiet_bytes = rule.max_mb * 1024 * 1024 if rule.max_mb is not None else None
    behouden: list[Path] = []
    cumulatief = 0

    for entry in entries:
        beschermd = entry.stat().st_mtime >= grens
        grootte = entry_size(entry) if limiet_bytes is not None else 0

        if not beschermd:
            if rule.max_entries is not None and len(behouden) >= rule.max_entries:
                break
            if limiet_bytes is not None and cumulatief + grootte > limiet_bytes:
                break

        behouden.append(entry)
        cumulatief += grootte

    return entries[len(behouden):]


def move_to_trash(entry: Path, root: Path) -> None:
    """Verplaats naar exports/_prullenbak/ met behoud van het pad.

    Ligt de entry al in de prullenbak, dan wordt hij verwijderd. Dat is de
    enige plek in dit script waar iets onherroepelijk weggaat — de bak is de
    laatste schakel, dus daar moet de keten eindigen.
    """
    trash = (root / EXPORTS_DIRNAME / TRASH_DIRNAME).resolve()

    if entry.resolve().is_relative_to(trash):
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()
        return

    relatief = entry.relative_to(root / EXPORTS_DIRNAME)
    doel = root / EXPORTS_DIRNAME / TRASH_DIRNAME / relatief
    doel.parent.mkdir(parents=True, exist_ok=True)
    if doel.exists():
        shutil.rmtree(doel) if doel.is_dir() else doel.unlink()
    shutil.move(str(entry), str(doel))


def run(root: Path, rules: list[RetentionRule], apply: bool) -> list[dict]:
    """Voer de regels uit (of toon alleen wat er zou gebeuren)."""
    verslag: list[dict] = []

    for rule in rules:
        entries = resolve_entries(rule, root)
        weg = select_for_removal(entries, rule)

        # Omvang meten vóór het verplaatsen: daarna bestaan de paden niet meer.
        totaal_bytes = sum(entry_size(p) for p in entries)
        bytes_weg = sum(entry_size(p) for p in weg)

        if apply:
            for entry in weg:
                move_to_trash(entry, root)

        verslag.append({
            "glob": rule.glob,
            "aanwezig": len(entries),
            "omvang_mb": round(totaal_bytes / 1024 / 1024, 1),
            "te_verwijderen": len(weg),
            "vrijgemaakt_mb": round(bytes_weg / 1024 / 1024, 1),
            "plafond": _plafond_tekst(rule),
        })

    return verslag


def _plafond_tekst(rule: RetentionRule) -> str:
    delen = []
    if rule.max_entries is not None:
        delen.append(f"{rule.max_entries} entries")
    if rule.max_mb is not None:
        delen.append(f"{rule.max_mb} MB")
    return " + ".join(delen) if delen else "uit (None)"


def _log_regel(root: Path, tekst: str) -> None:
    log = root / EXPORTS_DIRNAME / TRASH_DIRNAME / LOG_NAME
    log.parent.mkdir(parents=True, exist_ok=True)
    stempel = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"{stempel}  {tekst}\n")


def _laatste_run(root: Path) -> str:
    log = root / EXPORTS_DIRNAME / TRASH_DIRNAME / LOG_NAME
    if not log.exists():
        return "nooit"
    regels = [r for r in log.read_text(encoding="utf-8").splitlines() if r.strip()]
    return regels[-1][:19] if regels else "nooit"


def _print_verslag(verslag: list[dict], apply: bool) -> None:
    kop = "VERWIJDERD" if apply else "ZOU VERWIJDEREN (dry-run — gebruik --apply)"
    venster = (f"beschermd: alles van de afgelopen {PROTECT_DAYS} dagen"
               if PROTECT_DAYS else "beschermingsvenster staat uit")
    print(f"\n{kop}  ({venster})\n")
    print(f"{'regel':32s} {'aanwezig':>9s} {'MB':>7s} {'plafond':>18s} {'weg':>5s}")
    print("-" * 76)
    for r in verslag:
        print(f"{r['glob']:32s} {r['aanwezig']:9d} {r['omvang_mb']:7.1f} "
              f"{r['plafond']:>18s} {r['te_verwijderen']:5d}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="verplaats daadwerkelijk (standaard is dry-run)")
    parser.add_argument("--status", action="store_true",
                        help="toon vulling per regel en wanneer er laatst gedraaid is")
    args = parser.parse_args(argv)

    if not RETENTION_ENABLED:
        print("RETENTION_ENABLED staat op False — niets gedaan.")
        return 0

    try:
        verslag = run(PROJECT_ROOT, RULES, apply=args.apply and not args.status)
    except RetentionError as e:
        print(f"FOUT: {e}", file=sys.stderr)
        _log_regel(PROJECT_ROOT, f"FOUT: {e}")
        return 1

    if args.status:
        print(f"\nLaatste uitvoering: {_laatste_run(PROJECT_ROOT)}")
        _print_verslag(verslag, apply=False)
        return 0

    _print_verslag(verslag, apply=args.apply)

    if args.apply:
        totaal = sum(r["te_verwijderen"] for r in verslag)
        mb = sum(r["vrijgemaakt_mb"] for r in verslag)
        _log_regel(PROJECT_ROOT, f"{totaal} entries verplaatst ({mb:.1f} MB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
