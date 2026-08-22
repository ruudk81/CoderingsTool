"""Bewaakt de belofte achter consensus/: dat de map zichzelf bedruipt.

De afspraak (zie dev/WORK.md en de SDD-taken 1-3 van 2026-08-22): verwijder
step 5's ketenmodules, verhuis consensus/ naar hun plek, en run_pipeline
draait door zonder wijziging. Dat houdt alleen stand als twee dingen tegelijk
waar blijven, en niemand controleert dat met het blote oog:

1. Niets in consensus/ importeert iets van buiten consensus/ binnen step 5
   (test_consensus_leent_niets_meer_uit_step_5).
2. Elke kopie is en blijft inhoudelijk gelijk aan haar origineel in step 5,
   op de imports na — anders staat er straks een meetuitkomst op naam van de
   verkeerde keten (test_de_kopie_is_gelijk_aan_het_origineel).

Zonder deze twee tests is de afspraak een voornemen; met deze twee is hij een
test die faalt zodra iemand een import of een kopie laat afwijken.
"""
import ast
import pathlib
import re

import pytest


# ---------------------------------------------------------------------------
# Guard 1: consensus leent niets uit step 5.
# ---------------------------------------------------------------------------

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
                doel = node.module or ""
                if (doel.startswith("pipeline.step_5_codeGenerator")
                        and not doel.startswith("pipeline.step_5_codeGenerator.consensus")):
                    fouten.append(f"{f.name}: leent {doel}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    doel = alias.name
                    if (doel.startswith("pipeline.step_5_codeGenerator")
                            and not doel.startswith("pipeline.step_5_codeGenerator.consensus")):
                        fouten.append(f"{f.name}: leent {doel}")

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

GEKOPIEERD = [
    "prompts_common", "code_shape", "taxonomy_input", "attribute_cards",
    "concept_inventory", "prompts_writer", "grouping", "codebook_io",
    "codebook_writer", "codebook_verifier", "config_codeGenerator",
]

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


_PROJECT_ROOT_BLOK = re.compile(
    r"(?:^#.*\n)*^project_root = Path\(__file__\)((?:\.parent)+)\s*$",
    re.MULTILINE,
)


def _normaliseer_project_root(tekst: str, *, is_kopie: bool) -> str:
    """`codebook_io.py` berekent `project_root` als `Path(__file__)` plus een
    vaste keten `.parent`-stappen. Dat is een positieafhankelijke constante:
    de kopie in `consensus/` ligt één map dieper dan het origineel in
    `step_5_codeGenerator/`, dus heeft daar terecht één `.parent` méér nodig
    om dezelfde map (de repo-root) te bereiken. Vastgelegd in commit
    f43c3969.

    De byte-identiteitsregel gaat over logica, niet over positie — dus
    normaliseren we hier alleen die ene regel (en de toelichting die erboven
    staat) naar een kanonieke vorm, in plaats van het hele bestand van de
    vergelijking uit te sluiten. Alle 344 andere regels blijven gewoon
    vergeleken: een échte inhoudelijke afwijking in codebook_io.py valt hier
    nog steeds doorheen."""
    def vervang(match: re.Match) -> str:
        aantal = match.group(1).count(".parent")
        if is_kopie:
            aantal -= 1  # de kopie ligt één map dieper dan het origineel
        return f"project_root = Path(__file__){'.parent' * aantal}"

    return _PROJECT_ROOT_BLOK.sub(vervang, tekst)


def _genormaliseerd(naam: str, tekst: str, *, is_kopie: bool) -> str:
    tekst = _zonder_imports(tekst)
    if naam == "codebook_io":
        tekst = _normaliseer_project_root(tekst, is_kopie=is_kopie)
    return tekst


@pytest.mark.parametrize("naam", GEKOPIEERD)
def test_de_kopie_is_gelijk_aan_het_origineel(naam):
    """Zolang beide ketens naast elkaar draaien staat deze code dubbel. Een
    ongemerkt verschil is de manier waarop een meetuitkomst aan het ontwerp
    wordt toegeschreven terwijl hij van de kopie kwam."""
    hier = pathlib.Path(__file__).parent

    if naam in AFGEWEKEN:
        pytest.skip(f"bewuste afwijking: {AFGEWEKEN[naam]}")

    kopie = _genormaliseerd(
        naam, (hier / f"{naam}.py").read_text(encoding="utf-8"), is_kopie=True
    )
    origineel = _genormaliseerd(
        naam, (hier.parent / f"{naam}.py").read_text(encoding="utf-8"), is_kopie=False
    )

    verschil = [] if kopie == origineel else [naam]

    assert verschil == [], (
        f"{naam} wijkt af van het origineel in step 5.\n"
        "Is dat een BEWUSTE afwijking? Werk dan deze test bij met een "
        "vastgelegde vóór/ná — verwijder hem niet. Zolang beide ketens naast "
        "elkaar draaien is een ongemerkt verschil de manier waarop een "
        "meetuitkomst aan het ontwerp wordt toegeschreven terwijl hij van de "
        "kopie kwam."
    )
