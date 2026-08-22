"""
Tegenpool-filter — step 6, na de toewijzing.

Een idee dat onder een code met de OMGEKEERDE richting terechtkomt, is een
actieve fout in de oplevering: de respondent die schreef dat de bank slecht
bereikbaar is, krijgt een `1` op een kolom met het label "Toegankelijk en
bereikbaar", en de tabel zegt iets dat niet waar is. Deze pas haalt zulke
ideeën weg bij hun code en routeert ze naar Overig.

Waarom hier en niet in step 5: het codeboek is attribuut-korrelig
(`source_attributes` is een lijst attributen), terwijl dit onderscheid
idee-korrelig is — de positieve ideeën van een attribuut horen bij hun code te
blijven en alleen de negatieve moeten weg. Alleen step 6 kent de toewijzing per
idee, en alleen daar dragen beide kanten hun richting: het idee via `valence`,
de code via `valence` uit step 5.

STRENG, met opzet. Alleen een tegengesteld GERICHT label telt. Een gericht idee
onder een NEUTRALE code is onvolledig maar niet onwaar — het label doet geen
uitspraak over richting, dus er valt niets om te keren. Die ruimere variant zou
op de gemeten dataset ook het Overig-plafond van 10% overschrijden.

Sinds 2026-08-22 kan het codeboek KINDEREN dragen: volwaardige codes die via
`parent_code_id` onder Overig hangen en de afgevallen pool van een facet
vertegenwoordigen. Voor deze pas verandert dat de BESTEMMING, niet de regel: een
tegenpool-idee gaat naar het kind dat zijn richting wél draagt als dat bestaat,
en anders naar de ouder zoals altijd. Zie `route_opposing_poles`.

De pas verandert geen enkele definitie, alleen lidmaatschap. Overig's
`source_attributes` in step 5's cache blijft dus ongemoeid — `view_codebook`
bouwt zijn attribuutregels uit de toewijzingen zelf en vult die alleen áán met
die bronlijst, dus verhuisde ideeën verschijnen vanzelf onder hun eigen
attribuutnaam. Step 6 hoeft (en mag) niet in step 5's cache schrijven.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config import MISCELLANEOUS_CODE_LABELS
from utils.verboseReporter import VerboseReporter

UNASSIGNED_SENTINEL = "__UNASSIGNED__"

# Twee vocabulaires: ideeën spreken '+' / '-' / '0' / '', codes spreken
# 'positive' / 'negative' / 'neutral'. Alles wat hier niet in staat heeft geen
# pool en kan dus nooit een tegenpool zijn — geen else-tak die de rest opveegt.
_IDEA_POLE = {"+": "positive", "-": "negative"}
# `neutral` staat hier bewust NIET in: beschrijvend materiaal heeft geen
# tegenpool. `non_negative` (positief u neutraal, uit de tweedeling van
# `build_shapes`) wel — die code zegt letterlijk dat een klacht er niet in
# hoort, dus een negatief idee is daar een conflict.
_OPPOSITE = {
    "positive": "negative",
    "negative": "positive",
    "non_negative": "negative",
}

# De spiegelvraag van `_OPPOSITE`, en met opzet GEEN afgeleide ervan: welke
# codevalenties DRAGEN de richting van een idee? Dat is niet "alles wat niet
# tegengesteld is" — een `neutral` code botst nergens mee maar draagt ook niets,
# en zou zo als bestemming binnensluipen. En `non_negative` moet er aan de
# positieve kant wél in: in een tweedelingscodeboek (positief u neutraal tegen
# negatief) bestaat geen enkele code met valentie `positive`, dus zonder deze
# regel zou een positief idee daar nooit een bestemming vinden.
_CARRIES = {
    "positive": {"positive", "non_negative"},
    "negative": {"negative"},
}

_OVERIG_NAMES = {v.strip().lower() for v in MISCELLANEOUS_CODE_LABELS.values()} | {"overig"}


@dataclass
class FilterReport:
    """Wat de pas deed (of zou doen, bij apply=False)."""
    moved: int = 0
    per_code: Dict[str, int] = field(default_factory=dict)   # code van herkomst → n
    # Elk kind van Overig staat hier, ook op nul — een kind dat niets opvangt is
    # een bevinding en moet leesbaar zijn zonder de codes ernaast te leggen.
    per_child: Dict[str, int] = field(default_factory=dict)  # kind → n opgevangen
    coded_ideas: int = 0                                     # ideeën met een echte code
    overig_before: int = 0
    overig_after: int = 0
    overig_code_name: str = ""
    skipped_no_overig: bool = False

    @property
    def overig_share_after(self) -> float:
        return (self.overig_after / self.coded_ideas) if self.coded_ideas else 0.0

    @property
    def overig_share_before(self) -> float:
        return (self.overig_before / self.coded_ideas) if self.coded_ideas else 0.0


def pole_of(idea_valence: Any) -> Optional[str]:
    """De pool van een idee, of None als het er geen heeft. Eén plek, want
    `opposes`, `carries` en de bestemmingskeuze moeten het over dezelfde ideeën
    eens zijn."""
    return _IDEA_POLE.get(str(idea_valence).strip() if idea_valence else "")


def opposes(code_valence: Any, idea_valence: Any) -> bool:
    """Draagt dit idee de tegengestelde richting van zijn code?

    Alleen waar als BEIDE kanten een pool hebben en die polen elkaars
    tegengestelde zijn. Een neutrale code, een beschrijvend idee ('0') en een
    niet-gemeten valentie ('') vallen er allemaal buiten.
    """
    pool = pole_of(idea_valence)
    if pool is None:
        return False
    return _OPPOSITE.get(str(code_valence).strip()) == pool


def carries(code_valence: Any, idea_valence: Any) -> bool:
    """Draagt deze code de richting van dit idee?

    Gebruikt door de bestemmingskeuze, NIET door de foutdetectie. `opposes` en
    `carries` zijn geen elkaars ontkenning: tussen "botst" en "draagt" ligt
    "zegt er niets over" (een neutrale code), en dat is precies de ruimte waar
    een bestemming niet mag landen.
    """
    pool = pole_of(idea_valence)
    if pool is None:
        return False
    return str(code_valence).strip() in _CARRIES.get(pool, frozenset())


def find_overig_code(codes: List[Any]) -> Optional[Any]:
    """De catch-all uit het codeboek, of None als die er niet is.

    Zelfde herkenningsregel als `CodeAssigner._build_provenance_maps` — de
    no-fit-optie en deze pas moeten op dezelfde code uitkomen, anders ontstaan
    er twee verschillende restbakken.

    Een code met een `parent_code_id` valt af: kindnamen worden door een LLM
    geschreven en kunnen een catch-all-woord treffen, en de ouder draagt per
    definitie geen ouder. De hiërarchie zit in het veld, nooit in de naam.
    """
    for code in codes:
        if getattr(code, "parent_code_id", None):
            continue
        if (getattr(code, "code_name", "") or "").strip().lower() in _OVERIG_NAMES:
            return code
    return None


def route_opposing_poles(
    responses: List[Any],
    codes: List[Any],
    *,
    apply: bool = True,
) -> FilterReport:
    """Verhuis elk tegenpool-idee naar de Overig-familie. Muteert `responses`
    in place.

    De BESTEMMING is de ouder — behalve wanneer er een kind onder die ouder
    bestaat dat de richting van het idee wél draagt én dat over hetzelfde
    attribuut gaat. Dan is dát de plaats: zo'n kind is per constructie de
    afgevallen pool van hetzelfde facet, dus letterlijk gemaakt voor dit
    materiaal. Naar de ouder sturen zou het opnieuw ononderscheiden maken,
    terwijl het kind bestaat om er een naam aan te geven.

    De aanhechting is het attribuut-id (A#) van het idee tegen de
    `source_attribute_ids` van het kind — nooit een naam, want een naam staat in
    de enquêtetaal en kan herschreven worden. Levert de aanhechting niets op
    (geen id, of geen kind dat dit attribuut dekt), dan is de bestemming de
    ouder: de uitkomst is dan exact het gedrag van vóór de kinderen.

    `apply=False` doorloopt exact dezelfde beslissingen maar schrijft niets — de
    meting kan daardoor niet gaan afwijken van wat de filter werkelijk doet.
    """
    overig = find_overig_code(codes)
    report = FilterReport(
        overig_code_name=(getattr(overig, "code_name", "") or "") if overig else "",
        skipped_no_overig=overig is None,
    )

    code_valence = {}
    for code in codes:
        naam = (getattr(code, "code_name", "") or "").strip()
        if naam:
            code_valence[naam] = getattr(code, "valence", "") or ""

    overig_name = report.overig_code_name
    overig_id = getattr(overig, "code_id", "") if overig else ""

    # De kinderen van DEZE ouder. Op een productiecodeboek is deze lijst leeg en
    # blijft er van de hele kindtak niets over dan een lege dict-lookup.
    children = [c for c in codes
                if overig_id and getattr(c, "parent_code_id", None) == overig_id]
    report.per_child = {(getattr(c, "code_name", "") or "").strip(): 0 for c in children}
    # (attribuut-id, pool van het idee) → kind. Eerste kind wint; per facet en
    # valentie bestaat er maar één, dus die volgorde is geen echte keuze.
    child_by_attr: Dict[tuple, Any] = {}
    for c in children:
        for idea_valence, pool in _IDEA_POLE.items():
            if not carries(getattr(c, "valence", ""), idea_valence):
                continue
            for attr_id in (getattr(c, "source_attribute_ids", None) or []):
                child_by_attr.setdefault((attr_id, pool), c)

    family = {overig_name} | set(report.per_child)

    for resp in responses:
        for idea in (getattr(resp, "response_ideas", None) or []):
            naam = (getattr(idea, "assigned_code", "") or "").strip()
            if not naam or naam == UNASSIGNED_SENTINEL:
                continue
            report.coded_ideas += 1
            if naam in family:
                report.overig_before += 1
            # De ouder is de bestemming en kan dus niet zelf een fout dragen; een
            # kind wel — dat is een volwaardige code met een eigen richting.
            botst = (naam != overig_name and not report.skipped_no_overig
                     and opposes(code_valence.get(naam), idea.valence))

            eind_naam = naam
            if botst:
                doel = child_by_attr.get(
                    (getattr(idea, "attribute_id", None), pole_of(idea.valence)))
                doel_id = getattr(doel, "code_id", "") if doel is not None else overig_id
                eind_naam = ((getattr(doel, "code_name", "") or "").strip()
                             if doel is not None else overig_name)
                report.moved += 1
                report.per_code[naam] = report.per_code.get(naam, 0) + 1
                if doel is not None:
                    report.per_child[eind_naam] += 1
                if apply:
                    idea.assigned_code = eind_naam
                    idea.assigned_code_id = doel_id

            if eind_naam in family:
                report.overig_after += 1

    return report


def report_filter(report: FilterReport, reporter: Optional[VerboseReporter] = None) -> None:
    """Print wat de pas deed. Dit is de enige plek waar het Overig-aandeel ná
    toewijzing gemeten wordt: de scorecard van step 5 draait ervóór en blijft
    het aandeel van vóór de verhuizing melden."""
    reporter = reporter or VerboseReporter(True)
    reporter.section_header("TEGENPOOL-FILTER")

    if report.skipped_no_overig:
        reporter.stat_line("Geen Overig-code in het codeboek — pas overgeslagen.")
        return

    reporter.stat_line(f"Verhuisd naar {report.overig_code_name}: {report.moved} ideeën "
                       f"van {report.coded_ideas} gecodeerd "
                       f"({report.moved / report.coded_ideas * 100:.1f}%)"
                       if report.coded_ideas else "Geen gecodeerde ideeën.")
    reporter.stat_line(f"Overig-aandeel: {report.overig_share_before * 100:.1f}% "
                       f"→ {report.overig_share_after * 100:.1f}%")
    for naam, n in sorted(report.per_code.items(), key=lambda kv: -kv[1]):
        reporter.stat_line(f"  {naam}: {n}")

    if report.per_child:
        opgevangen = sum(report.per_child.values())
        reporter.stat_line(f"Waarvan opgevangen door een kind van "
                           f"{report.overig_code_name}: {opgevangen}")
        for naam, n in sorted(report.per_child.items(), key=lambda kv: (-kv[1], kv[0])):
            reporter.stat_line(f"  {naam}: {n}")


def measure_from_cache(filename=None, var_name=None, sample_size=None) -> FilterReport:
    """Meet wat de filter zou doen op de bestaande step-6-cache, zonder iets te
    schrijven en zonder één API-call.

    Bedoeld om het effect te zien vóór je step 6 met `force_recalc=True`
    opnieuw draait. Draait dezelfde `route_opposing_poles` met `apply=False`,
    dus de meting kan niet uiteenlopen met de pas zelf.
    """
    from utils.cacheManager import CacheManager, generate_enhanced_variable_key
    from models import CodeAssignedModel, CodingResultsCache, ConsolidatedCode
    from test_data import TEST_DATA

    filename = filename or TEST_DATA.filename
    var_name = var_name or TEST_DATA.var_name
    sample_size = sample_size if sample_size is not None else TEST_DATA.sample_size

    variable_key = generate_enhanced_variable_key(
        selected_variables=[var_name], is_merged=False, sample_size=sample_size)
    cm = CacheManager()
    responses = cm.load_from_cache(filename, "taxonomy_codes", variable_key, CodeAssignedModel)
    if not responses:
        raise FileNotFoundError("Geen taxonomy_codes-cache — draai eerst step 6.")
    mece = cm.load_metadata_from_cache(filename, "mece_codes", variable_key, CodingResultsCache)
    if not mece:
        raise FileNotFoundError("Geen mece_codes-cache — draai eerst step 5.")
    codes = [ConsolidatedCode(**c) if isinstance(c, dict) else c for c in (mece.raw_codes or [])]

    return route_opposing_poles(responses, codes, apply=False)


if __name__ == "__main__":
    # Droogloop tegen de huidige cache: cd src && python -m pipeline.step_6_codeAssigner.valence_filter
    report_filter(measure_from_cache())
