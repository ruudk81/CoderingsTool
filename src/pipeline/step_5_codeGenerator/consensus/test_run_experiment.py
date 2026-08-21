"""Tests voor de runner zelf — de laag tussen de opgeslagen partities en de
step-5-keten.

De zwaarste paden hier (`collect`, `codebook`) vragen een gevulde cache en
LLM-calls en zijn dus niet in een test te draaien. Wat wél te toetsen is: dat
elke naam die de runner gebruikt bestaat, en dat de CLI de sets kan aanwijzen
die op schijf staan.
"""
import builtins
import symtable
from pathlib import Path

from pipeline.step_5_codeGenerator.consensus import run_experiment

RUNNER = Path(run_experiment.__file__)


def _vrije_namen(table):
    """Elke naam die in dit bereik uit de module-globals of de builtins moet
    komen, plus dezelfde vraag voor alle geneste bereiken."""
    namen = {symbol.get_name() for symbol in table.get_symbols()
             if symbol.is_global() and not symbol.is_assigned()}
    for child in table.get_children():
        namen |= _vrije_namen(child)
    return namen


def test_elke_gebruikte_naam_in_de_runner_bestaat():
    """Een aanroep zonder import faalt pas op de regel zelf, en de dure paden
    van deze runner draaien niet in een test — dus wordt zo'n gat nooit
    gevangen. Dat is precies wat er gebeurde: `pool_thin_within_facet` kwam
    binnen met de facetpool, de import bleef achter, en het `codebook`-commando
    viel om na het laden van het materiaal maar vóór de schrijfcall."""
    beschikbaar = set(vars(run_experiment)) | set(dir(builtins))

    gebruikt = _vrije_namen(symtable.symtable(
        RUNNER.read_text(encoding="utf-8"), str(RUNNER), "exec"))

    assert sorted(gebruikt - beschikbaar) == []


def test_compare_kan_twee_willekeurige_sets_aanwijzen():
    """De twee 30-runssets staan als set 3 en set 4 op schijf. Met set 1 en 2
    vastgeschroefd is de meting die daarop is gedaan niet over te doen — en
    voor luna bestaat set 2 niet eens, dus het commando viel om op een
    ontbrekend bestand."""
    args = run_experiment.build_parser().parse_args(
        ["compare", "--config", "luna", "--tau", "0.7", "--set-a", "3",
         "--set-b", "4"])

    assert (args.set_a, args.set_b) == (3, 4)


def test_compare_vergelijkt_zonder_vlaggen_nog_steeds_set_1_en_2():
    """De reproduceerstappen in het verslag noemen geen setnummers."""
    args = run_experiment.build_parser().parse_args(["compare", "--config", "gpt54"])

    assert (args.set_a, args.set_b) == (1, 2)


def test_een_klik_op_run_draait_de_instellingen_bovenaan_het_bestand():
    """Code Runner geeft geen argumenten mee. Zonder terugval faalt het bestand
    dan op een verplicht subcommando — en dit project draait zijn runners door
    erop te klikken."""
    argv = run_experiment.settings_argv()

    args = run_experiment.build_parser().parse_args(argv)

    assert args.command == run_experiment.ACTIE
    assert args.config == run_experiment.CONFIG


def test_argumenten_van_de_commandoregel_winnen_van_de_instellingen():
    """Anders is het verslag niet meer te reproduceren met de commando's die
    erin staan."""
    assert run_experiment.gekozen_argv(["compare", "--config", "gpt54"]) == \
        ["compare", "--config", "gpt54"]
    assert run_experiment.gekozen_argv([]) == run_experiment.settings_argv()


def test_alles_draait_de_hele_ronde_met_een_actie():
    """De losse acties zijn er om een ronde te kunnen onderbreken; wie hem
    gewoon wil uitvoeren hoort niet zes keer een bestand te hoeven bewerken."""
    args = run_experiment.build_parser().parse_args(
        ["alles", "--config", "luna", "--runs", "30", "--set-a", "5",
         "--set-b", "6", "--tau", "0.7"])

    assert args.command == "alles"
    assert (args.runs, args.set_a, args.set_b, args.tau) == (30, 5, 6, 0.7)
