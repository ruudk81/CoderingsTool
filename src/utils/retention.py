"""
retention.py - Opruimen van exports/ en data/cache/ per analyse, met een prullenbak.

De eenheid is de **analyse** — dataset + vraag + steekproef — niet de map. De
artefacten van één analyse liggen verspreid over vijf exportmappen (het logboek,
de prompts, het codeboek, de codering, de kosten) plus negen cachebestanden; een
plafond per map kan prompts/ snoeien en codebook/ laten staan en levert dan een
halve analyse op. Daarom bewaren of verhuizen we een analyse in zijn geheel.

Twee domeinen, één analyse-begrip, twee plafonds:

  MAX_ANALYSES = 10        exports/     — vrij te herstellen (stap 7, seconden)
  CACHE_MAX_ANALYSES = 25  data/cache/  — alleen te herstellen door de hele
                                          keten 0-6 opnieuw te draaien

Het verschil is opzet, geen slordigheid. Een export weggooien kost je dertig
seconden rekenwerk; een cache-analyse weggooien kost een volledige keten —
minuten tot uren en echt geld — om ~5 MB terug te winnen. Het ruimere
cacheplafond garandeert bovendien "cache ⊇ exports": elke bewaarde export is
altijd nog opnieuw te genereren.

Analyses worden gesorteerd op hun nieuwste bestand; per domein blijven de N
nieuwste die in dát domein bestanden hebben, de rest gaat naar
exports/_prullenbak/.

Twee bodems eronder, in beide domeinen:
  - Alles binnen PROTECT_DAYS blijft hoe dan ook staan, ook als het plafond
    daardoor wordt overschreden. Beschermde analyses tellen wél mee voor N.
  - De nieuwste analyse blijft altijd staan.

Bestanden die niet bij een bekende analyse horen (restanten van een oudere
naamgeving of van een verwijderde dataset) verhuizen ook naar de prullenbak,
maar pas als ze ouder zijn dan PROTECT_DAYS. In de cache telt daar één soort
extra mee: een .pkl waarvan de db-rij op 'invalid' staat. CacheManager
verwijdert het bestand niet bij invalidatie, dus zulke bestanden blijven liggen
terwijl geen enkel codepad ze nog kan lezen. Ze volgen niet hun analyse maar hun
eigen leeftijd — anders dan een export met een oude naam is dit geen artefact
van de analyse meer, maar een lijk.

De prullenbak heeft zijn eigen plafond in MB. Dáár — en alleen daar — gaat iets
onherroepelijk weg. Let op: **terugzetten uit de prullenbak is voor de cache
niet symmetrisch.** Het bestand komt terug, de db-rij blijft ongeldig, dus de
cache blijft ongeldig. Wil je zo'n analyse echt terug, draai hem opnieuw.

Veiligheidsinvariant: elk pad dat dit script aanraakt ligt onder exports/ (en
daar alleen in MANAGED_DIRS) óf is een .pkl direct onder data/cache/. De twee
.db-bestanden worden nooit aangeraakt, alleen bijgewerkt.

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

# De cache ligt buiten exports/, dus krijgt hij een eigen tak in dezelfde
# prullenbak — één bak, één plafond, één plek om te zoeken. Cachenamen zijn plat
# (geen submappen), dus de tak is dat ook.
CACHE_DIRNAME = Path("data") / "cache"
CACHE_TRASH_SUBDIR = "data_cache"
CACHE_DB_NAME = "cache.db"

# Ligt bewust direct onder exports/, niet in de prullenbak: anders valt het
# onder de bak-regel en kan het script zijn eigen audit-spoor wissen.
LOG_PATH = Path(EXPORTS_DIRNAME) / "retention.log"

# =============================================================================
# INSTELLINGEN — pas deze aan vlak vóór een run
# =============================================================================

RETENTION_ENABLED = True

# Hoeveel analyses bewaren we? None = uit; dan verhuist er niets.
# Beschermde analyses (zie PROTECT_DAYS) tellen mee voor dit getal: staan er
# zes beschermd en is het plafond 10, dan komen er nog vier bij.
MAX_ANALYSES: Optional[int] = 10

# Hetzelfde, maar voor data/cache/. Bewust ruimer dan MAX_ANALYSES: een export
# is gratis te herstellen (stap 7), een cache-analyse alleen door de hele keten
# 0-6 opnieuw te draaien. Zolang dit getal groter is dan MAX_ANALYSES geldt
# "cache ⊇ exports" en is elke bewaarde export nog te reproduceren.
CACHE_MAX_ANALYSES: Optional[int] = 25

# Plafond op de prullenbak in MB. None = uit. Dit is de enige plek waar iets
# onherroepelijk verdwijnt — zolang dit None is verhuist alles alleen maar en
# kun je alles terugzetten.
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
    """Eén analyse, met haar bestanden in beide domeinen.

    `entries` zijn de exports, `cache_entries` de pickles onder data/cache/. Een
    analyse hoeft niet in beide domeinen te bestaan: een verse run heeft nog geen
    exports, en een analyse waarvan de exports al zijn opgeruimd houdt haar cache.
    """
    key: AnalysisKey
    entries: list[Entry] = field(default_factory=list)
    cache_entries: list[Entry] = field(default_factory=list)

    @property
    def alle(self) -> list[Entry]:
        return self.entries + self.cache_entries

    @property
    def mtime(self) -> float:
        """De analyse is zo oud als haar nieuwste bestand, in welk domein ook.

        Bewust over beide domeinen heen: een analyse waarvan alleen stap 7 net
        opnieuw is gedraaid is als geheel recent, en haar cache hoort dan niet
        opeens als oud te gelden.
        """
        return max(e.mtime for e in self.alle)

    @property
    def size(self) -> int:
        return sum(e.size for e in self.alle)


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
    """Alle beheerde bestanden uit beide domeinen, per analyse, plus de restanten.

    Een export hoort bij een analyse als zijn naam volgens de canonieke
    conventie te lezen is (utils.exportNaming). Lukt dat niet, dan is het een
    restant: een oudere naamgeving, of een dataset die niet meer bestaat.

    Een cachebestand wordt niet uit zijn naam gelezen maar uit cache.db, waar
    filename en variable_key als losse kolommen staan. Betrouwbaarder dan de
    exportkant: er valt niets verkeerd te splitsen.
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
    # Alleen de export-restanten gaan langs de hecht-stap: die zoekt op oude
    # bestandsnamen, en een cachebestand heeft er per definitie geen.
    restanten = _hecht_restanten_aan_analyses(restanten, analyses, stems)
    return analyses, restanten + cache_restanten


def _cache_analyse_key(filename: str, variable_key: str) -> Optional[AnalysisKey]:
    """De analyse waar een cache-ingang bij hoort.

    De cache heeft de parseerproblematiek van exports niet: filename en
    variable_key staan als losse kolommen in cache.db, dus alleen de steekproef
    hoeft nog van de variabelenaam te worden gescheiden. Dezelfde eis als in
    exportNaming: een getal of "full", anders is het geen leesbare sleutel.

    Randgeval: bij samengevoegde variabelen bevat variable_key ook de
    merge-configuratie (Q1+Q2_concat_semicolon_skip_500), terwijl de export de
    kále variabelenaam gebruikt. Zo'n analyse matcht dan niet met haar exports
    en wordt een eigen analyse. Dat is de veilige kant: matchen bepaalt alleen
    de groepering, nooit of iets weg mag.
    """
    var_name, _, sample = variable_key.rpartition("_")
    if not var_name or (sample != "full" and not sample.isdigit()):
        return None
    return AnalysisKey(Path(filename).stem.replace(" ", "_"), var_name, sample)


def _verzamel_cache(root: Path, per_key: dict[AnalysisKey, Analysis]) -> list[Entry]:
    """Voeg de cachebestanden bij hun analyse. Geeft de cache-restanten terug.

    Restant is hier alles wat geen levend codepad meer kan bereiken: een .pkl
    zonder db-rij, en een .pkl waarvan de rij op 'invalid' staat. Dat tweede
    komt vaker voor dan je zou denken — CacheManager invalideert een ingang maar
    verwijdert het bestand niet, dus afgedankte caches blijven liggen.
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
    """Voeg elk eenduidig toewijsbaar restant bij zijn analyse.

    Een bestand met een oude naam hoort vaak wel degelijk bij een analyse die
    je bewaart — het logboek van een stap die sinds de hernoeming niet opnieuw
    is gedraaid. Zonder deze stap wordt zo'n bestand alleen door zijn eigen
    leeftijd beschermd en verdwijnt het binnen een week, terwijl de analyse
    blijft. Dat botst met het uitgangspunt dat de analyse de eenheid is.

    Toewijzen gebeurt alleen bij zekerheid: de naam moet beginnen met dataset +
    variabele van precies één analyse, én de steekproef van die analyse moet
    als los naamdeel voorkomen. Zo blijven de coderingen waarvan de steekproef
    nooit is opgeschreven ongemoeid — daar zijn meerdere analyses mogelijk, en
    gokken is hoe je een bestand onder een verkeerd etiket kwijtraakt.
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
    """De vormen waarin een datasetnaam vooraan een oude bestandsnaam kan staan.

    Oude namen gebruikten soms spaties, soms underscores, en de verbose logs
    kapten de naam af op 50 tekens.
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
    """De analyses voorbij het plafond in één domein, exclusief de beschermde.

    De lijst staat op mtime gesorteerd, dus de beschermde analyses en de
    nieuwste vormen samen het begin van de rij. Een beschermde analyse telt
    mee voor het plafond maar wordt zelf nooit geselecteerd — het plafond
    accepteert dan liever een overschrijding.

    Analyses zonder bestanden in dít domein slaan we over: een analyse waarvan
    de exports al zijn opgeruimd mag geen exportplek meer bezet houden, anders
    krimpt het effectieve plafond met elke opruiming.
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
    """De exports voorbij MAX_ANALYSES, exclusief de beschermde."""
    return _selecteer(analyses, MAX_ANALYSES, lambda a: a.entries, now)


def select_cache_for_removal(
    analyses: list[Analysis],
    now: Optional[float] = None,
) -> list[Analysis]:
    """Hetzelfde voor data/cache/, tegen het ruimere CACHE_MAX_ANALYSES.

    Zolang CACHE_MAX_ANALYSES groter is dan MAX_ANALYSES verdwijnt de cache van
    een analyse nooit vóór haar exports, en blijft elke bewaarde export
    reproduceerbaar.
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

def _invalideer_cache_rij(root: Path, pad: Path) -> None:
    """Zet de db-rij van een verhuisd cachebestand op 'invalid'.

    Zonder dit blijft er een 'valid'-rij staan die naar een verdwenen bestand
    wijst. CacheManager vangt dat op bij het eerstvolgende gebruik, maar dan is
    het een storing in plaats van een besluit. Vergelijken gebeurt op het
    geresolvede pad, want de db bewaart het pad zoals het bij het schrijven
    werd samengesteld — niet noodzakelijk in dezelfde vorm.

    Let op de asymmetrie: het bestand terugzetten uit de prullenbak maakt deze
    rij niet weer geldig. Zie de moduledocstring.
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
    """Verplaats naar exports/_prullenbak/, met behoud van waar het vandaan kwam.

    Exports houden hun pad onder exports/; cachebestanden gaan naar de platte
    tak data_cache/ (cachenamen kennen geen submappen) en hun db-rij gaat mee op
    'invalid'.

    Ligt het pad al in de bak, dan wordt het verwijderd — de bak is de laatste
    schakel. De membership-test gebruikt het ongeresolvede pad, zodat een
    symlink in de bak niet via .resolve() alsnog de verplaats-tak neemt.
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
