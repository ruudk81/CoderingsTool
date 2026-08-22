"""Bewaakt de belofte achter consensus/: dat de map zichzelf bedruipt.

De afspraak (zie dev/WORK.md en de SDD-taken 1-3 van 2026-08-22): verwijder
step 5's ketenmodules, verhuis consensus/ naar hun plek, en run_pipeline
draait door zonder wijziging. Dat houdt alleen stand als drie dingen tegelijk
waar blijven, en niemand controleert dat met het blote oog:

1. Niets in consensus/ importeert iets van buiten consensus/ binnen step 5
   (test_consensus_leent_niets_meer_uit_step_5).
2. Elke kopie is en blijft inhoudelijk gelijk aan haar origineel in step 5,
   op de imports na — anders staat er straks een meetuitkomst op naam van de
   verkeerde keten (test_de_kopie_is_gelijk_aan_het_origineel).
3. `codebook_io.py` rekent zijn `project_root` naar de repo-root uit. Bewaker 2
   ziet dat niet: dat bestand is sinds 2026-08-22 een eigen versie en wordt dus
   niet meer met het origineel vergeleken. Bewaker 3 toetst daarom niet HOEVEEL
   `.parent`-stappen er staan, maar WAAR ze uitkomen — en dat klopt zowel hier
   als na de verhuizing (test_project_root_wijst_naar_de_repo_root).

Zonder deze drie tests is de afspraak een voornemen; met deze drie is hij een
test die faalt zodra iemand een import, een kopie, of het anker van de
promptexport laat afwijken.
"""
import ast
import pathlib

import pytest


# ---------------------------------------------------------------------------
# Guard 1: consensus leent niets uit step 5.
# ---------------------------------------------------------------------------

STAP5 = "pipeline.step_5_codeGenerator"
EIGEN = f"{STAP5}.consensus"


def _leent_buiten_consensus(doel: str) -> bool:
    """Wijst deze modulenaam naar step 5 maar buiten consensus/?

    Vergelijkt op PAKKETGRENS, niet op tekenprefix. `doel.startswith(EIGEN)`
    alleen zou een toekomstige zustermap `consensus_experiment_2` als "eigen"
    aanzien en er stil langs laten; dezelfde fout aan de buitenkant zou
    `step_5_codeGeneratorX` binnenhalen. Een naam telt pas als binnen een
    pakket wanneer hij dat pakket IS of erop volgt met een punt.
    """
    def binnen(pakket: str) -> bool:
        return doel == pakket or doel.startswith(pakket + ".")

    return binnen(STAP5) and not binnen(EIGEN)


def test_consensus_leent_niets_meer_uit_step_5():
    """De afspraak: verwijder step 5's ketenmodules, verplaats consensus/
    omhoog, en run_pipeline draait door. Dat kan alleen als consensus/ niets
    buiten zichzelf gebruikt binnen step 5.

    Drie vormen breken bij zo'n verhuizing, elk op hun eigen manier:
    - `from ..x import y` gaat na de verhuizing naar `pipeline.x` wijzen
      (een `ast.ImportFrom` met `level >= 2`)
    - `from pipeline.step_5_codeGenerator.x import y` blijft wijzen naar een
      bestand dat er dan niet meer is (een `ast.ImportFrom`)
    - `import pipeline.step_5_codeGenerator.x` heeft hetzelfde probleem maar
      als kale `ast.Import` — geen `from`, dus een aparte AST-knoop. Task 2's
      review wees erop dat de oorspronkelijke schets alleen `ImportFrom`
      doorliep; er staat vandaag geen `import`-statement van die vorm in de
      map, dus deze tak sluit een gat vóórdat het ooit gebruikt wordt.

    Loopt met `ast`, niet met een regex op de brontekst: Task 1 werd betrapt
    op een `^from`-patroon dat een import miste die binnen een functie op een
    niet-nul kolom stond — hij slaagde toch, door toevallige duck typing, en
    zou pas bij de verhuizing zelf zijn geknapt. Een AST-wandeling ziet elke
    import ongeacht insprong of nesting.
    """
    pkg = pathlib.Path(__file__).parent
    fouten = []
    for f in sorted(pkg.glob("*.py")):
        boom = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        for node in ast.walk(boom):
            if isinstance(node, ast.ImportFrom):
                if node.level >= 2:
                    fouten.append(
                        f"{f.name}: relatieve import buiten het pakket "
                        f"({'.' * node.level}{node.module or ''})"
                    )
                if _leent_buiten_consensus(node.module or ""):
                    fouten.append(f"{f.name}: leent {node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if _leent_buiten_consensus(alias.name):
                        fouten.append(f"{f.name}: leent {alias.name}")

    assert fouten == [], (
        "consensus/ leent iets van buiten zichzelf binnen step 5 — dat breekt "
        "zodra step_5_codeGenerator's ketenmodules verdwijnen. Maak van de "
        "import een absolute `pipeline.step_5_codeGenerator.consensus.<module>` "
        "of een lokale `.<module>`, nooit een pad dat buiten consensus/ uitkomt.\n"
        + "\n".join(fouten)
    )


# ---------------------------------------------------------------------------
# Guard 2: elke kopie blijft gelijk aan haar origineel.
# ---------------------------------------------------------------------------

# Bestandsnamen die consensus/ met step 5 DEELT, maar die welbewust EIGEN zijn
# — consensus' eigen versie van een productiebestand, geen kopie ervan. Elk
# van deze dertien hoort dus NIET in GEKOPIEERD, en het waarom staat per bestand
# hieronder; dat maakt de uitsluiting een geschreven claim in plaats van een
# stilzwijgende omissie.
EIGEN_VERSIE = {
    # Draagt de VORM van deze opzet (N runs, eigen dispatch/cache/log-identiteit)
    # — dat is precies wat consensus/ toevoegt aan productie, geen kopie ervan.
    "run_codebook", "consolidation", "prompts_consolidation",
    # Lezen het codeboek resp. de prompts terug; die van consensus/ moeten de
    # eigen cache- en promptexport-namen kennen (`step5c_consolidation`,
    # `prompts_step5c`) die productie's versie niet kent.
    "view_codebook", "view_prompts",
    # Pakket-docstring: beschrijft de submap zelf, niet een ketenstap — heeft
    # dus geen zinnig productie-origineel om tegen te vergelijken.
    "__init__",
    # Testen van de drie bovenstaande eigen modules, dus zelf ook eigen.
    "test_consolidation", "test_run_codebook",
    # Vangt sinds 2026-08-22 de afgevallen valentiepolen op: `pool_minority_poles`
    # neemt ze per (facet, valentie) samen, `build_shapes` kreeg er een `floor`
    # bij, en `direction_loss` is vervangen door `coverage_recovered`. Productie
    # laat een afgevallen pool nog steeds vallen — dat is het verschil dat deze
    # keten moet meten, dus geen kopie meer.
    "grouping", "test_grouping",
    # `CodeShape.origin` kent hier twee extra waarden, "recovered" en "child",
    # die productie niet kent; het vetorecht van `codebook_writer` hangt eraan.
    "code_shape",
    # Draagt sinds 2026-08-22 een TWEEDE schrijfcall, `write_miscellaneous`, die
    # de kinderen onder Overig schrijft met een eigen prompt, een eigen fase
    # (`step5c_miscellaneous`) en zonder veto. Bovendien is de veto-toets zelf
    # verscherpt: alleen `pooled` is nog vetobaar, want een `recovered` facetunie
    # weigeren zet zijn respondenten weer onder de zusterpool die het
    # tegenovergestelde beweert. Productie kent geen kinderen en geen facetunie.
    "codebook_writer",
    # `apply_overig_sweep` geeft sinds 2026-08-22 de Overig-CODE terug in plaats
    # van zijn naam, en munt daar de K#'s. Nodig omdat een kind onder Overig
    # hangt via `parent_code_id` en dus de id van zijn ouder moet kennen vóór de
    # cache-write — de hiërarchie leeft in een veld, nooit in een naam. Productie
    # kent geen kinderen en heeft aan de naam genoeg.
    "codebook_io",
}


def _module_stammen(map_: pathlib.Path) -> set[str]:
    return {p.stem for p in map_.glob("*.py")}


# Afgeleid, niet met de hand bijgehouden: alles wat consensus/ met step 5 DEELT
# (zelfde bestandsnaam) en niet welbewust eigen is, is een kopie en hoort dus
# bewaakt te worden. Een module die ooit wordt overgekopieerd komt hier vanzelf
# bij; een module die verdwijnt valt er vanzelf uit — geen van beide vereist dat
# iemand deze lijst met de hand bijwerkt.
GEKOPIEERD = sorted(
    (_module_stammen(pathlib.Path(__file__).parent)
     & _module_stammen(pathlib.Path(__file__).parent.parent))
    - EIGEN_VERSIE
)

# Modules waarvan een afwijking een vastgelegd besluit is, in plaats van
# ruis: naam -> reden (commit). Leeg bij aanvang — dit is geen skip-lijst
# voor codebook_io, want die afwijking wordt hieronder genormaliseerd in
# plaats van overgeslagen (zie _normaliseer_project_root). Een module komt
# hier pas bij als de vergelijking zelf niet kan worden gered door
# normalisatie, bijvoorbeeld wanneer `grouping.py` op een dag inhoudelijk
# uiteenloopt van het origineel.
AFGEWEKEN = {}


def _zonder_imports(tekst: str) -> str:
    """Alles behalve de importregels. De kopieën verschillen per definitie in
    hun imports — daar gaat deze test niet over."""
    return "\n".join(r for r in tekst.splitlines()
                     if not r.lstrip().startswith(("import ", "from ")))


@pytest.mark.parametrize("naam", GEKOPIEERD)
def test_de_kopie_is_gelijk_aan_het_origineel(naam):
    """Zolang beide ketens naast elkaar draaien staat deze code dubbel. Een
    ongemerkt verschil is de manier waarop een meetuitkomst aan het ontwerp
    wordt toegeschreven terwijl hij van de kopie kwam."""
    hier = pathlib.Path(__file__).parent

    if naam in AFGEWEKEN:
        pytest.skip(f"bewuste afwijking: {AFGEWEKEN[naam]}")

    kopie = _zonder_imports((hier / f"{naam}.py").read_text(encoding="utf-8"))
    origineel = _zonder_imports(
        (hier.parent / f"{naam}.py").read_text(encoding="utf-8"))

    verschil = [] if kopie == origineel else [naam]

    assert verschil == [], (
        f"{naam} wijkt af van het origineel in step 5.\n"
        "Is dat een BEWUSTE afwijking? Werk dan deze test bij met een "
        "vastgelegde vóór/ná — verwijder hem niet. Zolang beide ketens naast "
        "elkaar draaien is een ongemerkt verschil de manier waarop een "
        "meetuitkomst aan het ontwerp wordt toegeschreven terwijl hij van de "
        "kopie kwam."
    )


def test_afgeleide_lijst_is_compleet():
    """Bewaakt de afleiding zelf, niet alleen wat ze afleidt.

    `GEKOPIEERD` wordt berekend uit de bestanden die op schijf staan; wie een
    kopie verwijdert (of `EIGEN_VERSIE` te ruim maakt) krijgt daardoor een
    KORTERE lijst, geen falende test — `test_de_kopie_is_gelijk_aan_het_origineel`
    parametriseert immers over wat er ook is, en een module die er niet meer in
    zit wordt simpelweg niet meer getoetst. Deze test legt vast wat er hoort te
    staan, zodat een stillere lijst zichtbaar wordt als FOUT in plaats van als
    kortere testrun.
    """
    verwachte_kopieen = {
        "attribute_cards", "codebook_verifier",
        "concept_inventory", "config_codeGenerator",
        "prompts_common", "prompts_writer", "taxonomy_input",
    }
    verwachte_testkopieen = {
        "test_attribute_cards", "test_codebook_writer", "test_concept_inventory",
        "test_prompts_writer", "test_taxonomy_input",
    }
    verwacht = verwachte_kopieen | verwachte_testkopieen

    assert set(GEKOPIEERD) == verwacht, (
        f"GEKOPIEERD is {sorted(set(GEKOPIEERD) - verwacht)} te veel en "
        f"{sorted(verwacht - set(GEKOPIEERD))} te weinig t.o.v. de verwachte "
        "zeven ketenmodules plus vijf testkopieën. Ontbreekt er iets: is een "
        "kopie verwijderd, of hoort de nieuwe naam in EIGEN_VERSIE? Staat er "
        "iets te veel in: is er een nieuwe kopie bijgekomen die deze lijst "
        "(en de verwachting hier) terecht moet zien groeien."
    )
    assert len(verwachte_kopieen) == 7
    assert len(verwachte_testkopieen) == 5


def test_pakketgrens_wordt_op_de_punt_getrokken():
    """De grens tussen 'eigen' en 'geleend' loopt op een punt, niet op tekens.

    Zonder deze toets zou een tekenprefix volstaan, en dan glipt een
    toekomstige zustermap (`consensus_experiment_2`) langs bewaker 1 omdat
    zijn naam toevallig met `consensus` begint. Datzelfde geldt aan de
    buitenkant voor een map die met `step_5_codeGenerator` begint maar het
    niet is.
    """
    assert _leent_buiten_consensus(f"{STAP5}.grouping")
    assert _leent_buiten_consensus(STAP5)
    assert _leent_buiten_consensus(f"{STAP5}.consensus_experiment_2.grouping")

    assert not _leent_buiten_consensus(f"{EIGEN}.grouping")
    assert not _leent_buiten_consensus(EIGEN)
    assert not _leent_buiten_consensus("pipeline.step_4_classifier.classifier")
    assert not _leent_buiten_consensus("pipeline.step_5_codeGeneratorX.iets")


def test_project_root_wijst_naar_de_repo_root():
    """De ene regel die bewaker 2 niet dekt.

    `codebook_io.project_root` telt `.parent`-stappen vanaf `__file__` — een
    positieafhankelijke constante, die bij de promotie met de hand terug moet
    naar vier omdat het bestand dan een map hoger ligt. Bewaker 2 komt er niet
    aan toe: `codebook_io` is sinds 2026-08-22 een eigen versie en wordt niet
    meer regel voor regel met het origineel vergeleken.

    De repetitie van 2026-08-22 vond dat als losse ingreep in het verhuisrecept.
    Een recept is een voornemen; deze toets is een test. Hij vraagt niet hoeveel
    stappen er staan maar waar ze uitkomen, en klopt daarom op beide dieptes —
    hier én na de verhuizing.
    """
    from pipeline.step_5_codeGenerator.consensus import codebook_io

    wortel = codebook_io.project_root
    assert (wortel / "src" / "pipeline").is_dir(), (
        f"project_root komt uit op {wortel}, en daar staat geen src/pipeline. "
        "Tel de `.parent`-stappen in codebook_io.py na: ze horen op de repo-root "
        "uit te komen, en dat aantal hangt af van hoe diep het bestand ligt."
    )
    assert (wortel / ".git").exists(), (
        f"project_root komt uit op {wortel}, en dat is niet de repo-root. "
        "Promptexport en logs belanden dan naast de repo in plaats van erin."
    )
