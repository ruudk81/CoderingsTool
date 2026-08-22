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

_OVERIG_NAMES = {v.strip().lower() for v in MISCELLANEOUS_CODE_LABELS.values()} | {"overig"}


@dataclass
class FilterReport:
    """Wat de pas deed (of zou doen, bij apply=False)."""
    moved: int = 0
    per_code: Dict[str, int] = field(default_factory=dict)   # code van herkomst → n
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


def opposes(code_valence: Any, idea_valence: Any) -> bool:
    """Draagt dit idee de tegengestelde richting van zijn code?

    Alleen waar als BEIDE kanten een pool hebben en die polen elkaars
    tegengestelde zijn. Een neutrale code, een beschrijvend idee ('0') en een
    niet-gemeten valentie ('') vallen er allemaal buiten.
    """
    pool = _IDEA_POLE.get(str(idea_valence).strip() if idea_valence else "")
    if pool is None:
        return False
    return _OPPOSITE.get(str(code_valence).strip()) == pool


def find_overig_code(codes: List[Any]) -> Optional[Any]:
    """De catch-all uit het codeboek, of None als die er niet is.

    Zelfde herkenningsregel als `CodeAssigner._build_provenance_maps` — de
    no-fit-optie en deze pas moeten op dezelfde code uitkomen, anders ontstaan
    er twee verschillende restbakken.
    """
    for code in codes:
        if (getattr(code, "code_name", "") or "").strip().lower() in _OVERIG_NAMES:
            return code
    return None


def route_opposing_poles(
    responses: List[Any],
    codes: List[Any],
    *,
    apply: bool = True,
) -> FilterReport:
    """Verhuis elk tegenpool-idee naar Overig. Muteert `responses` in place.

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

    for resp in responses:
        for idea in (getattr(resp, "response_ideas", None) or []):
            naam = (getattr(idea, "assigned_code", "") or "").strip()
            if not naam or naam == UNASSIGNED_SENTINEL:
                continue
            report.coded_ideas += 1
            if naam == overig_name:
                report.overig_before += 1
                continue
            if report.skipped_no_overig or not opposes(code_valence.get(naam), idea.valence):
                continue
            report.moved += 1
            report.per_code[naam] = report.per_code.get(naam, 0) + 1
            if apply:
                idea.assigned_code = overig_name
                idea.assigned_code_id = overig_id

    report.overig_after = report.overig_before + report.moved
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
