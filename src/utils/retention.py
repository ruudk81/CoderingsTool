"""
retention.py - Plafonds op exports/, met een prullenbak.

Eén primitief: een regel is een glob plus een plafond. De entries (bestanden;
mappen worden overgeslagen) worden één keer opgemeten — pad, mtime, omvang —
en vervolgens gesorteerd op mtime (nieuwste eerst) en van boven naar beneden
opgeteld. Zodra een plafond wordt overschreden gaat alles vanaf die entry naar
exports/_prullenbak/. De nieuwste blijft altijd staan, ook als hij zelf al het
plafond overschrijdt: het plafond accepteert liever een overschrijding dan dat
het de hele map leegveegt.

Boven op het plafond ligt één bodem: alles binnen PROTECT_DAYS blijft hoe dan
ook staan, ook als het plafond daardoor wordt overschreden.

Veiligheidsinvariant: elk pad dat dit script aanraakt ligt onder exports/. Een
regel wordt getoetst vóórdat er iets verplaatst wordt (tekstueel, op de glob
zelf) én per gevonden entry (op het geresolvede pad, dus ook tegen symlinks en
'..'). Er wordt nergens een map geïnventariseerd die niet in RULES staat.

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

# Ligt bewust direct onder exports/, niet onder exports/_prullenbak/: dat valt
# onder de bak-regel, en het script mag zijn eigen audit-spoor niet kunnen
# wissen door zijn eigen retentie op zichzelf toe te passen.
LOG_PATH = Path(EXPORTS_DIRNAME) / "retention.log"

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


@dataclass(frozen=True)
class Entry:
    """Eén meting van één bestand: pad, mtime en omvang in bytes.

    resolve_entries meet dit één keer per bestand. select_for_removal leest
    daarna nergens opnieuw van disk — zou dat wel gebeuren, dan zou een
    bestand dat tussen de twee stappen in wordt aangeraakt (bijvoorbeeld de
    pipeline die naar verbose_logs/ schrijft terwijl dit gereedschap draait)
    de beschermingsaanname kunnen omzeilen.
    """
    path: Path
    mtime: float
    size: int


RULES: list[RetentionRule] = [
    RetentionRule("exports/verbose_logs/*",    max_entries=None, max_mb=None),
    RetentionRule("exports/prompts/*",         max_entries=None, max_mb=None),
    RetentionRule("exports/codebook/*",        max_entries=None, max_mb=None),
    RetentionRule("exports/coderingen/*",      max_entries=None, max_mb=None),
    RetentionRule("exports/costs/*",           max_entries=None, max_mb=None),
    RetentionRule("exports/_prullenbak/**/*",  max_entries=None, max_mb=None),
]

# Bewust niet beheerd: exports/adhoc/, exports/diagnostics/,
# exports/experiment_logs/ en alles onder data/.


class RetentionError(Exception):
    """Een regel wijst buiten exports/, of een verplaatsing mislukte."""


def _valideer_regel(rule: RetentionRule) -> None:
    """Tekstuele voorcontrole, vóór er iets verplaatst wordt.

    Een regel die niets matcht (een lege map, een tikfout) glipt anders door
    de per-entry-controle in resolve_entries heen — die gaat pas af op een
    gevonden entry. Deze controle raakt elke regel aan vóórdat regel 1 al iets
    verplaatst heeft; de per-entry-controle blijft daarnaast bestaan, want die
    vangt wat een tekstuele check niet ziet: symlinks en '..'.
    """
    if not rule.glob.startswith(f"{EXPORTS_DIRNAME}/"):
        raise RetentionError(
            f"regel {rule.glob!r} wijst buiten exports/ "
            f"(moet beginnen met '{EXPORTS_DIRNAME}/')")


def resolve_entries(rule: RetentionRule, root: Path) -> list[Entry]:
    """De entries van een regel, nieuwste eerst.

    Mappen worden overgeslagen: alleen bestanden worden een Entry. Dat
    voorkomt dat de gespiegelde submappen die move_to_trash zelf binnen
    _prullenbak/ aanmaakt (waarvan de mtime bij elke verplaatsing ververst)
    als permanent-beschermde entries meetellen. Dit filter staat bewust vóór
    de invariant-toets: een kapotte symlink is geen bestand (is_file() is
    False voor een symlink zonder geldig doel) en wordt zo overgeslagen in
    plaats van dat .resolve() erop stukloopt en de hele run met een
    RetentionError afbreekt.

    Wat wél een bestand is, wordt getoetst aan de invariant (elk pad ligt
    onder exports/) — ook als het een symlink is. Ligt er één buiten, dan
    stopt het script — een genegeerde regel is een regel die je niet opmerkt.
    """
    exports = (root / EXPORTS_DIRNAME).resolve()
    entries: list[Entry] = []
    for pad in root.glob(rule.glob):
        if not pad.is_file():
            continue
        if not pad.resolve().is_relative_to(exports):
            raise RetentionError(
                f"regel {rule.glob!r} wijst buiten exports/: {pad}")
        stat = pad.stat()
        entries.append(Entry(path=pad, mtime=stat.st_mtime, size=stat.st_size))
    entries.sort(key=lambda e: e.mtime, reverse=True)
    return entries


def select_for_removal(
    entries: list[Entry],
    rule: RetentionRule,
    now: Optional[float] = None,
) -> list[Entry]:
    """De entries die weg mogen: alles voorbij het strengste plafond.

    Entries binnen PROTECT_DAYS worden nooit geselecteerd, ook niet als het
    plafond daardoor wordt overschreden. Ze tellen wel mee voor dat plafond.
    Dezelfde bodem geldt voor de allereerste (nieuwste) entry: die blijft
    altijd staan, ook als hij zelf al het plafond overschrijdt — anders zou
    een max_mb kleiner dan het nieuwste bestand de hele map leegvegen in
    plaats van te snoeien.

    Omdat de lijst op mtime is gesorteerd vormen beschermde entries en de
    eerste entry altijd het begin van de rij, dus een `break` bij de eerste
    onbeschermde, niet-eerste entry die het plafond zou overschrijden slaat er
    nooit één over.

    `now` bestaat alleen zodat tests een tijdstip kunnen vastzetten.
    """
    if rule.max_entries is None and rule.max_mb is None:
        return []

    grens = (now if now is not None else time.time()) - PROTECT_DAYS * 86400
    limiet_bytes = rule.max_mb * 1024 * 1024 if rule.max_mb is not None else None
    behouden: list[Entry] = []
    cumulatief = 0

    for entry in entries:
        beschermd = entry.mtime >= grens

        if not beschermd and behouden:
            if rule.max_entries is not None and len(behouden) >= rule.max_entries:
                break
            if limiet_bytes is not None and cumulatief + entry.size > limiet_bytes:
                break

        behouden.append(entry)
        cumulatief += entry.size

    return entries[len(behouden):]


def move_to_trash(entry: Path, root: Path) -> None:
    """Verplaats naar exports/_prullenbak/ met behoud van het pad.

    Ligt de entry al in de prullenbak, dan wordt hij verwijderd — dat is de
    enige plek in dit script waar iets onherroepelijk weggaat, want de bak is
    de laatste schakel. Een entry is hier altijd een bestand (resolve_entries
    slaat mappen over), dus unlink() volstaat.

    De trash-membership-test gebruikt het ongeresolvede pad: een symlink in
    _prullenbak/ die elders binnen exports/ heen wijst mag niet via .resolve()
    buiten de bak belanden en zo alsnog de verplaats-tak nemen (en zo
    _prullenbak/_prullenbak/ opleveren).
    """
    trash = root / EXPORTS_DIRNAME / TRASH_DIRNAME

    if entry.is_relative_to(trash):
        entry.unlink()
        return

    relatief = entry.relative_to(root / EXPORTS_DIRNAME)
    doel = root / EXPORTS_DIRNAME / TRASH_DIRNAME / relatief
    doel.parent.mkdir(parents=True, exist_ok=True)
    if doel.exists():
        _log_regel(root, f"overschreven in prullenbak: {relatief}")
        doel.unlink()
    shutil.move(str(entry), str(doel))


def run(root: Path, rules: list[RetentionRule], apply: bool) -> list[dict]:
    """Voer de regels uit (of toon alleen wat er zou gebeuren).

    De schakelaar geldt hier, niet alleen in main(): wie deze functie
    rechtstreeks importeert mag RETENTION_ENABLED niet kunnen omzeilen.

    Twee lussen, bewust gescheiden: eerst wordt van élke regel geïnventariseerd
    wat er zou weggaan, en pas dáárna wordt er verplaatst. Zou dat in één lus
    gebeuren, dan ziet de bak-regel (exports/_prullenbak/**/*, laatste in
    RULES) entries die in déze run net uit een andere map de bak in zijn
    verplaatst — en shutil.move behoudt de mtime, dus PROTECT_DAYS beschermt
    ze daar niet. Eén --apply zou zo een bestand kunnen verplaatsen én
    definitief wissen, zonder dat de bak ooit respijt bood. Met de
    inventarisatie vooraf ziet de bak-regel alleen wat er vóór deze run al in
    de bak lag.
    """
    if not RETENTION_ENABLED:
        return []

    for rule in rules:
        _valideer_regel(rule)

    inventaris: list[tuple[RetentionRule, list[Entry], list[Entry]]] = []
    for rule in rules:
        entries = resolve_entries(rule, root)
        inventaris.append((rule, entries, select_for_removal(entries, rule)))

    verslag: list[dict] = []

    for rule, entries, weg in inventaris:
        totaal_bytes = sum(e.size for e in entries)
        bytes_weg = sum(e.size for e in weg)

        if apply:
            for entry in weg:
                move_to_trash(entry.path, root)

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
    log = root / LOG_PATH
    log.parent.mkdir(parents=True, exist_ok=True)
    stempel = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"{stempel}  {tekst}\n")


STEMPEL_LEN = len("2026-08-01 00:00:00")


def _laatste_run(root: Path) -> str:
    """Tijdstip van de laatste dáádwerkelijke uitvoering.

    retention.log bevat ook regels van move_to_trash ("overschreven in
    prullenbak: ...") en van een mislukte run ("FOUT: ..."). Alleen de
    uitvoeringsregel die main() na afloop van --apply schrijft draagt het
    "RUN  "-voorvoegsel; dat is de enige regel die hier telt.
    """
    log = root / LOG_PATH
    if not log.exists():
        return "nooit"
    runs = [
        r for r in log.read_text(encoding="utf-8").splitlines()
        if r[STEMPEL_LEN + 2:].startswith("RUN  ")
    ]
    return runs[-1][:STEMPEL_LEN] if runs else "nooit"


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
        _log_regel(PROJECT_ROOT, f"RUN  {totaal} entries verplaatst ({mb:.1f} MB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
