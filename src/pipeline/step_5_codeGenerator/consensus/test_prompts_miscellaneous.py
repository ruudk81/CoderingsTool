"""Tests voor de kinderprompt: een restcategorie moet zijn onderwerp dragen.

Naar het model van `test_prompts_writer.py` — elke vorm komt in de prompt, de
instructor-hint sluit hem af, en het instructieraamwerk bevat geen woord uit de
dataset. Dat laatste is hier expliciet getoetst en niet bij de schrijverprompt:
deze prompt is nieuw, en het lekpad (diagnose → vuistregel → prompt) begint bij
een nieuwe prompt.
"""
from pipeline.step_5_codeGenerator.consensus.concept_inventory import Concept
from pipeline.step_5_codeGenerator.consensus.code_shape import CodeShape
from pipeline.step_5_codeGenerator.consensus.prompts_miscellaneous import (
    MiscellaneousText, build_miscellaneous_prompt, make_miscellaneous_model,
)


def kindvorm(key, valence, members=("A1",), umbrella="Bereikbaarheid"):
    resp = frozenset({"r1", "r2", "r3"})
    return CodeShape(key=key, members=members, valence=valence, umbrella=umbrella,
                     resp_ids=resp, resp_pos=frozenset(), resp_neg=resp,
                     resp_neu=frozenset(), origin="child")


def concept_voor(attribute_id, naam, definitie):
    return Concept(attribute_id=attribute_id, name=naam, definition=definitie,
                   domain="D", facet="F", n_iu=2, resp_ids=frozenset({"r1"}),
                   resp_pos=frozenset(), resp_neg=frozenset({"r1"}),
                   resp_neu=frozenset())


CONCEPTS = {
    "A1": concept_voor("A1", "Wachttijd", "hoe lang je moet wachten"),
    "A2": concept_voor("A2", "Openingstijden", "wanneer je terecht kunt"),
}


def test_elke_vorm_komt_met_naam_en_definitie_in_de_prompt():
    """Een kind kan meerdere attributen dragen; een kale namenlijst geeft het
    model te weinig om een eerlijke restnaam op te baseren."""
    prompt = build_miscellaneous_prompt(
        [kindvorm("V1", "negative", ("A1", "A2"))], CONCEPTS, "vooral in ...", "Dutch")

    assert "Wachttijd" in prompt
    assert "hoe lang je moet wachten" in prompt
    assert "Openingstijden" in prompt
    assert "wanneer je terecht kunt" in prompt


def test_twee_vormen_komen_allebei_in_de_prompt():
    prompt = build_miscellaneous_prompt(
        [kindvorm("V1", "negative", ("A1",)), kindvorm("V2", "non_negative", ("A2",))],
        CONCEPTS, "vooral in ...", "Dutch")

    assert "[V1]" in prompt
    assert "[V2]" in prompt


def test_de_prompt_eindigt_op_de_instructorhint():
    prompt = build_miscellaneous_prompt(
        [kindvorm("V1", "negative")], CONCEPTS, "vooral in ...", "Dutch")

    assert prompt.rstrip().endswith(
        "provide your output as valid JSON following the response schema provided")


def test_de_naam_moet_een_restcategorie_binnen_een_onderwerp_dragen():
    """De ene regel die van de schrijverprompt moet verschillen: deze codes
    zijn restcategorieën binnen een onderwerp, en de naam mag geen hoofdthema
    suggereren."""
    prompt = " ".join(build_miscellaneous_prompt(
        [kindvorm("V1", "negative")], CONCEPTS, "vooral in ...", "Dutch").split())

    assert "rest category within its subject" in prompt
    assert "not as a theme of its own" in prompt


def test_het_gedeelde_onderwerp_staat_bij_de_vorm():
    """Zonder het onderwerp kan het model geen restnaam op dat niveau schrijven
    — het zou terugvallen op het eerste lid en daarmee een hoofdthema claimen."""
    prompt = build_miscellaneous_prompt(
        [kindvorm("V1", "negative", ("A1", "A2"), umbrella="Bereikbaarheid")],
        CONCEPTS, "vooral in ...", "Dutch")

    assert "Bereikbaarheid" in prompt


def test_de_richting_van_de_vorm_staat_in_de_prompt():
    """Een kind draagt vaak juist de kritiek die anders onder een positieve
    code viel; die richting is een feit uit de data, geen suggestie."""
    prompt = build_miscellaneous_prompt(
        [kindvorm("V1", "negative")], CONCEPTS, "vooral in ...", "Dutch")

    assert "direction: negative" in prompt


def test_al_vergeven_namen_gaan_mee():
    prompt = build_miscellaneous_prompt(
        [kindvorm("V1", "negative")], CONCEPTS, "vooral in ...", "Dutch",
        taken_names=["Al vergeven"])

    assert "Al vergeven" in prompt


def test_het_instructieraamwerk_draagt_geen_dataset_vocabulaire():
    """Twee totaal verschillende datasets moeten hetzelfde raamwerk opleveren.

    Dit is de mechanische versie van de huisregel: prompts blijven
    use-case-agnostisch. Zodra iemand een diagnose op één dataset als
    vuistregel in de instructie schrijft, gaan de twee raamwerken uit elkaar
    lopen óf duikt het woord op in het raamwerk van de ander.
    """
    een = build_miscellaneous_prompt(
        [kindvorm("V1", "negative", ("A1",), umbrella="Bereikbaarheid")],
        CONCEPTS, "vooral in ...", "Dutch")
    ander_concepts = {
        "B1": concept_voor("B1", "Rentestand", "wat je krijgt of betaalt"),
    }
    ander = build_miscellaneous_prompt(
        [kindvorm("V1", "negative", ("B1",), umbrella="Voorwaarden")],
        ander_concepts, "vooral in ...", "Dutch")

    raamwerk_een = een.split("Codes:")[0]
    raamwerk_ander = ander.split("Codes:")[0]
    assert raamwerk_een == raamwerk_ander

    for woord in ("Wachttijd", "Openingstijden", "Bereikbaarheid",
                  "Rentestand", "Voorwaarden"):
        assert woord not in raamwerk_een


def test_het_responsemodel_kent_geen_veto():
    """Een kind is niet vetobaar — het bestaat omdat deze respondenten anders
    nergens staan. Een `nameable`-veld in het schema zou het model uitnodigen
    een oordeel te vellen dat de keten toch negeert."""
    assert "nameable" not in MiscellaneousText.model_fields


def test_het_responsemodel_beperkt_de_sleutels_tot_de_aangeboden_vormen():
    """Zo kan het model geen sleutel verzinnen en er geen overslaan."""
    model = make_miscellaneous_model(
        [kindvorm("V1", "negative"), kindvorm("V2", "non_negative")])
    entry = model.model_fields["codes"].annotation.__args__[0]

    assert set(entry.model_fields["key"].annotation.__args__) == {"V1", "V2"}
