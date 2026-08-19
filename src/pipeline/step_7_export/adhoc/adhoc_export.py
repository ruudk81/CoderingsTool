#%%

"""
Adhoc SPSS-levering — EXPERIMENT (zie ADHOC.md).

Neemt de output van step 7 en maakt er een SPSS-klare levering van in
exports/adhoc/:

  1. verzamelen  — de bestanden van één run kopiëren (coderingen/ + codebook/)
  2. verrijken   — koppen en achtergrondvariabelen uit het bronbestand erbij
  3. hercoderen  — codeboek en taxonomie naar de m-variabeleconventie
  4. splitsen    — per meting, plus het ongesplitste bestand

Raakt de productiecode niet aan: leest step 7's output, schrijft ernaast.

Draaien:
    cd src && python -m pipeline.step_7_export.adhoc.adhoc_export
"""

import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

src_dir = Path(__file__).resolve().parents[3]
project_root = src_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pandas as pd
import pyreadstat

from test_data import TEST_DATA
from utils.exportNaming import export_filename
from utils.verboseReporter import VerboseReporter
from pipeline.step_7_export.resultsExporter import (
    FILTER_CODE_LABELS, Catalog, build_catalog, _clean_var_lab)
from pipeline.step_7_export.run_export import StepConfig, get_var_lab, load_step6_cache


# =============================================================================
# PROJECT_SPECIFIEK — alles wat bij promotie naar productie moet wijken.
# Dit blok is het enige dat deze dataset kent; de rest van het bestand is
# mechaniek en verhuist ongewijzigd. Zie ADHOC.md, sectie "Naar productie".
# =============================================================================
BRON_ID = "xDLNMID"                 # respondent-id in het bronbestand
MEETVARIABELE = "nMeting"           # variabele waarop gesplitst wordt
METINGEN = {1.0: "2025", 2.0: "2026"}

# De banner: (bronvariabele, kopnaam), in de volgorde waarin ze in het
# tabellenboek moeten staan. Ze worden hernoemd naar nKOP1, nKOP2, … en houden
# hun eigen waardelabels — die beginnen bij 1, zoals elke categorische
# variabele in de bron. Het bronbestand had alleen nKOP1 (A. METING); die wordt
# hier opnieuw opgebouwd uit nMeting, zodat er één regel geldt voor alle koppen.
KOPPEN = [
    ("nMeting",       "A. METING"),
    ("nGeslacht",     "B. GESLACHT"),
    ("nLftcat",       "C. LEEFTIJD"),
    ("nOplVoltCat",   "D. OPLEIDING"),
    ("nNielsen5",     "E. REGIO"),
    ("nMentality",    "F. MENTALITY"),
    ("nDoelgroepREC", "G. DOELGROEP"),
    ("nQa1_4",        "H. KENT ASN BANK"),
]

# Variabelen die ongewijzigd meegaan — naam, label en waarden blijven zoals ze
# in de bron staan. De weegfactor is geen kop, maar zonder hem kun je in SPSS
# geen gewogen tabellen draaien.
EXTRA_BRONVARIABELEN = ["weegvar"]

# Basistekst in het variabelelabel: de conditie waaronder de vraag gesteld is.
# Leeg laten als er geen basisrestrictie is — dan blijft het bracket-slot leeg,
# precies zoals in het bronbestand ("[]Qa1_4 Kan je aangeven ...").
BASIS = "Basis - Kent ASN Bank"


# =============================================================================
# CONVENTIE — mechaniek, dataset-onafhankelijk
# =============================================================================
# Waardelabels van een m-variabele, letterlijk zoals in het bronbestand.
M_WAARDELABELS = {0.0: "Niet Genoemd", 1.0: "Wel Genoemd", 99999999.0: "Missing"}
M_FORMAT = "F8.2"
M_MEASURE = "nominal"

# SPSS kapt een variabelelabel af op 256 tekens; liever zelf kappen en tellen.
MAX_LABEL = 256

# De vier lagen: (laagcode in de variabelenaam, kwalificatie in het label).
# De laag hoort VAST aan de vraag-id, zonder underscore: de tabelleringssoftware
# groepeert op de stam vóór het laatste _<nummer>, dus "mQd1_dom_1" en
# "mQd1_att_1" belanden in één tabel en "mQd1DOM_1" / "mQd1ATT_1" niet.
LAAG_CODE = "COD"
LAAG_DOMEIN = "DOM"
LAAG_FACET = "FAC"
LAAG_ATTRIBUUT = "ATT"

# Het id draagt in de bron een x; step 7 laat die vallen.
ID_EXPORT = "DLNMID"
ID_DOEL = "xDLNMID"

# Prefix per meetniveau, afgelezen aan het bronbestand:
#   m = dichotome set   n = enkelvoudig categorisch   x = tekst/ruw
# Het lange bestand (gecombineerd) heeft geen dichotome sets, dus n en x.
GECOMBINEERD_LAGEN = [
    ("code",           "n", LAAG_CODE,        "Code"),
    ("domain",         "n", LAAG_DOMEIN,      "Domein"),
    ("facet",          "n", LAAG_FACET,       "Facet"),
    ("attribute",      "n", LAAG_ATTRIBUUT,   "Attribuut"),
    ("valence",        "n", "VAL",            "Valentie"),
    ("instance",       "x", "INSTANCE",       "Instance"),
    ("interpretation", "x", "INTERPRETATIE",  "Interpretatie"),
    ("abstraction",    "x", "ABSTRACTIE",     "Abstractie"),
]


def m_naam(var_name: str, laag: str, nummer: int) -> str:
    """m<VraagID><LAAG>_<nummer> — het volgnummer is altijd het laatste token,
    en de laag zit vast aan de vraag-id zodat elke laag zijn eigen tabel wordt."""
    return f"m{var_name}{laag}_{nummer}"


def m_label(var_name: str, laag: str, nummer, vraag: str, antwoord: str,
            kwalificatie: str = "") -> str:
    """[]<basis>[][]<VraagID><LAAG>_<n> <kwalificatie><vraagtekst> <antwoord>.

    Zonder basis blijft er één leeg bracket-paar over, zoals in het bronbestand.
    """
    ident = f"{var_name}{laag}_{nummer}"
    kop = f"[]{BASIS}[][]" if BASIS else "[]"
    return f"{kop}{ident} {kwalificatie}{vraag} {antwoord}".strip()[:MAX_LABEL]


# =============================================================================
# 1. VERZAMELEN
# =============================================================================
def adhoc_dir() -> Path:
    return project_root / "exports" / "adhoc"


def bron_bestanden(config: StepConfig) -> Dict[str, Path]:
    """De bestanden van déze run — bepaald door TEST_DATA, niet door mtime."""
    exports = project_root / "exports"
    naam = lambda doctype, ext: export_filename(
        config.filename, config.var_name, config.sample_size, doctype, ext)
    return {
        "codeboek_sav": exports / "coderingen" / naam("codeboek", "sav"),
        "taxonomie_sav": exports / "coderingen" / naam("taxonomie", "sav"),
        "gecombineerd_sav": exports / "coderingen" / naam("gecombineerd", "sav"),
        "codering_xlsx": exports / "coderingen" / naam("codering", "xlsx"),
        "codeboek_xlsx": exports / "codebook" / naam("codeboek", "xlsx"),
        "codeboek_csv": exports / "codebook" / naam("codeboek", "csv"),
        "taxonomie_csv": exports / "codebook" / naam("taxonomie", "csv"),
    }


def kopieer_documentatie(bronnen: Dict[str, Path], doel: Path, rep: VerboseReporter):
    """Excel + CSV gaan ongewijzigd mee: codeboek en legenda zijn gedeeld over
    beide metingen, dus die worden niet gesplitst en niet hercodeerd."""
    for sleutel in ("codering_xlsx", "codeboek_xlsx", "codeboek_csv", "taxonomie_csv"):
        bron = bronnen[sleutel]
        shutil.copy2(bron, doel / bron.name)
        rep.stat_line(f"gekopieerd: {bron.name}")


# =============================================================================
# 2. VERRIJKEN
# =============================================================================
class Bron:
    """Het importbestand: de banner, de weegfactor, en de meting per respondent.

    De koppen worden hier gemaakt: elke bronvariabele uit KOPPEN wordt nKOP<n>
    met de opgegeven kopnaam als label. Waarden en waardelabels gaan ongewijzigd
    mee — categorisch en 1-based, zoals in de bron.
    """

    def __init__(self, config: StepConfig):
        pad = project_root / "data" / config.filename
        bronnamen = [b for b, _ in KOPPEN]
        kolommen = list(dict.fromkeys(
            [BRON_ID, MEETVARIABELE] + bronnamen + EXTRA_BRONVARIABELEN))
        df, self.meta = pyreadstat.read_sav(str(pad), usecols=kolommen)
        df = df.set_index(BRON_ID)
        self.meting = df[MEETVARIABELE]

        self.kolommen: List[str] = []
        self.labels: Dict[str, str] = {}
        self.waardelabels: Dict[str, Dict] = {}
        self.measures: Dict[str, str] = {}
        self.formats: Dict[str, str] = {}
        self.df = pd.DataFrame(index=df.index)

        for nummer, (bronnaam, kopnaam) in enumerate(KOPPEN, 1):
            kop = f"nKOP{nummer}"
            self._neem_over(df, bronnaam, kop, label=kopnaam)
        for naam in EXTRA_BRONVARIABELEN:
            self._neem_over(df, naam, naam)

    def _neem_over(self, df, bronnaam: str, doelnaam: str, label: Optional[str] = None):
        if bronnaam not in df.columns:
            raise KeyError(f"bronvariabele ontbreekt in het importbestand: {bronnaam}")
        self.df[doelnaam] = df[bronnaam]
        self.kolommen.append(doelnaam)
        self.labels[doelnaam] = label or self.meta.column_names_to_labels.get(bronnaam, bronnaam)
        if bronnaam in self.meta.variable_value_labels:
            self.waardelabels[doelnaam] = self.meta.variable_value_labels[bronnaam]
        self.measures[doelnaam] = self.meta.variable_measure.get(bronnaam, "nominal")
        self.formats[doelnaam] = self.meta.original_variable_types.get(bronnaam, M_FORMAT)


def verrijk(df: pd.DataFrame, bron: Bron) -> pd.DataFrame:
    """Koppen direct achter het id, de rest van de kolommen ongewijzigd erna."""
    verrijkt = df.copy()
    for kolom in bron.kolommen:
        verrijkt[kolom] = verrijkt[ID_DOEL].map(bron.df[kolom])
    rest = [c for c in df.columns if c != ID_DOEL]
    return verrijkt[[ID_DOEL] + bron.kolommen + rest]


# =============================================================================
# 3. HERCODEREN — codeboek en taxonomie naar de m-conventie
# =============================================================================
def hernoem_codeboek(cat: Catalog, var_name: str, vraag: str) -> Dict[str, Tuple[str, str]]:
    """oude kolomnaam -> (m-naam, m-label) voor het codeboek."""
    mapping = {}
    for e in sorted(cat.codes.values(), key=lambda x: x.number):
        mapping[f"{var_name}code_{e.number}"] = (
            m_naam(var_name, LAAG_CODE, e.number),
            m_label(var_name, LAAG_CODE, e.number, vraag, e.name))
    for filtercode, label in FILTER_CODE_LABELS.items():
        mapping[f"{var_name}code_{filtercode}"] = (
            m_naam(var_name, LAAG_CODE, filtercode),
            m_label(var_name, LAAG_CODE, filtercode, vraag, label))
    return mapping


def hernoem_taxonomie(cat: Catalog, var_name: str, vraag: str) -> Dict[str, Tuple[str, str]]:
    """oude kolomnaam -> (m-naam, m-label) voor de taxonomie.

    Het domeinnummer verdwijnt uit de variabelenaam — het volgnummer moet het
    laatste token zijn — en komt als domeinNAAM terug in het label. Dat kan
    zonder verlies: facet- en attribuutnummers zijn doorgenummerd over alle
    domeinen heen en dus al uniek.
    """
    mapping = {}
    for e in sorted(cat.domains.values(), key=lambda x: x.number):
        mapping[f"{var_name}domain_{e.number}"] = (
            m_naam(var_name, LAAG_DOMEIN, e.number),
            m_label(var_name, LAAG_DOMEIN, e.number, vraag, e.name, "Domein — "))
    for laag, entries, woord in (
        (LAAG_FACET, cat.facets, "Facet"),
        (LAAG_ATTRIBUUT, cat.attributes, "Attribuut"),
    ):
        oud = "facet" if laag == LAAG_FACET else "attr"
        for e in sorted(entries.values(), key=lambda x: x.number):
            mapping[f"{var_name}{oud}_{e.number}_{e.domain_number}"] = (
                m_naam(var_name, laag, e.number),
                m_label(var_name, laag, e.number, vraag, e.name,
                        f"{woord} (domein '{e.domain_name}') — "))
    return mapping


def hernoem_basis(var_name: str, vraag: str) -> Dict[str, Tuple[str, str]]:
    """Id en responstekst — beide dragen in de bron een x-prefix."""
    kop = f"[]{BASIS}[][]" if BASIS else "[]"
    return {
        ID_EXPORT: (ID_DOEL, "[]Deelname ID"),
        var_name: (f"x{var_name}", f"{kop}{var_name} {vraag}"[:MAX_LABEL]),
    }


def hernoem_gecombineerd(var_name: str, vraag: str) -> Dict[str, Tuple[str, str]]:
    """Het lange bestand: enkelvoudig categorisch wordt n, tekst wordt x.

    Geen m: er is hier geen dichotome set, elke rij draagt één idee.
    De waardelabels die step 7 al zette (codenamen, domeinnamen, valentie)
    blijven ongewijzigd staan.
    """
    kop = f"[]{BASIS}[][]" if BASIS else "[]"
    mapping = {}
    for kolom, prefix, achtervoegsel, woord in GECOMBINEERD_LAGEN:
        ident = f"{var_name}{achtervoegsel}"
        mapping[kolom] = (f"{prefix}{ident}",
                          f"{kop}{ident} {woord} — {vraag}"[:MAX_LABEL])
    return mapping


def hercodeer(df: pd.DataFrame, meta,
              m_mapping: Dict[str, Tuple[str, str]],
              hernoem_mapping: Dict[str, Tuple[str, str]]
              ) -> Tuple[pd.DataFrame, Dict, Dict, Dict, Dict]:
    """Hernoem kolommen en bouw de SPSS-metadata opnieuw op.

    `m_mapping` krijgt de dichotome conventie opgelegd (0/1/99999999, F8.2,
    nominaal); `hernoem_mapping` wordt alleen hernoemd en gelabeld en houdt zijn
    eigen waardelabels. Kolommen in geen van beide blijven ongemoeid.
    """
    alle = {**m_mapping, **hernoem_mapping}
    ontbreekt = [k for k in alle if k not in df.columns]
    if ontbreekt:
        raise KeyError(f"verwachte kolommen ontbreken in de .sav: {ontbreekt[:5]}"
                       f" ({len(ontbreekt)} totaal)")

    labels, waardelabels, measures, formats = {}, {}, {}, {}
    for kolom in df.columns:
        if kolom in m_mapping:
            nieuw, label = m_mapping[kolom]
            labels[nieuw] = label
            waardelabels[nieuw] = M_WAARDELABELS
            measures[nieuw] = M_MEASURE
            formats[nieuw] = M_FORMAT
        else:
            nieuw, label = hernoem_mapping.get(
                kolom, (kolom, meta.column_names_to_labels.get(kolom, kolom)))
            labels[nieuw] = label
            if kolom in meta.variable_value_labels:
                waardelabels[nieuw] = meta.variable_value_labels[kolom]
            measures[nieuw] = meta.variable_measure.get(kolom, "nominal")
            formats[nieuw] = meta.original_variable_types.get(kolom, M_FORMAT)

    hernoemd = df.rename(columns={oud: nieuw for oud, (nieuw, _) in alle.items()})
    return hernoemd, labels, waardelabels, measures, formats


# =============================================================================
# 4. SPLITSEN + SCHRIJVEN
# =============================================================================
def schrijf_sav(df: pd.DataFrame, pad: Path, labels, waardelabels, measures, formats):
    pyreadstat.write_sav(
        df, str(pad),
        column_labels=[labels.get(c, c) for c in df.columns],
        variable_value_labels={k: v for k, v in waardelabels.items() if k in df.columns},
        variable_measure={k: v for k, v in measures.items() if k in df.columns},
        variable_format={k: v for k, v in formats.items() if k in df.columns},
    )


def schrijf_alle_metingen(df: pd.DataFrame, basisnaam: str, doel: Path, bron: Bron,
                          labels, waardelabels, measures, formats,
                          rep: VerboseReporter) -> List[Path]:
    """Het ongesplitste bestand plus één bestand per meting."""
    geschreven = []
    pad = doel / basisnaam
    schrijf_sav(df, pad, labels, waardelabels, measures, formats)
    rep.stat_line(f"{pad.name}: {len(df)} rijen, {len(df.columns)} variabelen")
    geschreven.append(pad)

    meting_per_rij = df[ID_DOEL].map(bron.meting)
    for waarde, naam in METINGEN.items():
        deel = df[meting_per_rij == waarde]
        pad = doel / f"{Path(basisnaam).stem}_{naam}{Path(basisnaam).suffix}"
        schrijf_sav(deel, pad, labels, waardelabels, measures, formats)
        rep.stat_line(f"{pad.name}: {len(deel)} rijen")
        geschreven.append(pad)
    return geschreven


# =============================================================================
# HOOFDROUTINE
# =============================================================================
def run(config: Optional[StepConfig] = None) -> Dict[str, List[Path]]:
    config = config or StepConfig()
    rep = VerboseReporter(True)
    rep.section_header("ADHOC EXPORT — SPSS-levering")

    doel = adhoc_dir()
    doel.mkdir(parents=True, exist_ok=True)
    bronnen = bron_bestanden(config)
    ontbreekt = [p.name for p in bronnen.values() if not p.exists()]
    if ontbreekt:
        raise FileNotFoundError(
            "Deze run is niet volledig geëxporteerd; draai step 7 opnieuw.\n"
            + "\n".join(f"  ontbreekt: {n}" for n in ontbreekt))

    # catalogus uit dezelfde cache als de export — gelijke nummering gegarandeerd
    responses, codes, partition_set, partition_results, metadata, _, _ = load_step6_cache(config)
    cat = build_catalog(codes, partition_set, partition_results, metadata, responses)
    vraag = _clean_var_lab(get_var_lab(config))
    rep.stat_line(f"Catalogus: {len(cat.codes)} codes, {len(cat.domains)} domeinen, "
                  f"{len(cat.facets)} facetten, {len(cat.attributes)} attributen")

    bron = Bron(config)
    rep.stat_line(f"Bron: {len(bron.df)} respondenten | koppen: "
                  + ", ".join(f"nKOP{i} {naam}" for i, (_, naam) in enumerate(KOPPEN, 1)))

    rep.stat_line("")
    kopieer_documentatie(bronnen, doel, rep)

    basis = hernoem_basis(config.var_name, vraag)
    geschreven = {}
    for sleutel, m_mapping in (
        ("codeboek", hernoem_codeboek(cat, config.var_name, vraag)),
        ("taxonomie", hernoem_taxonomie(cat, config.var_name, vraag)),
        ("gecombineerd", {}),
    ):
        rep.stat_line("")
        hernoem = dict(basis)
        if not m_mapping:
            hernoem.update(hernoem_gecombineerd(config.var_name, vraag))

        df, meta = pyreadstat.read_sav(str(bronnen[f"{sleutel}_sav"]))
        df, labels, waardelabels, measures, formats = hercodeer(df, meta, m_mapping, hernoem)
        rep.stat_line(f"{sleutel}: {len(m_mapping)} m-variabelen, "
                      f"{len(hernoem)} overige hernoemd")

        df = verrijk(df, bron)
        labels.update(bron.labels)
        measures.update(bron.measures)
        formats.update(bron.formats)
        waardelabels.update(bron.waardelabels)

        geschreven[sleutel] = schrijf_alle_metingen(
            df, bronnen[f"{sleutel}_sav"].name, doel, bron,
            labels, waardelabels, measures, formats, rep)

    rep.stat_line("")
    rep.stat_line(f"Map: {doel}")
    return geschreven


if __name__ == "__main__":
    print("=" * 70)
    print("Adhoc export — SPSS-levering (experiment)")
    print("=" * 70)
    run()
