"""Tests voor de zeven die bepalen wát er als spelfout wordt aangeboden.

Het zijn pure functies over losse woorden, dus ze zijn te testen zonder
Hunspell, spaCy of een LLM. Dat is precies de bedoeling: dit is de laag waar
elke datacorruptie in deze stap vandaan kwam.
"""

import pytest

from pipeline.step_1_preProcessor.spellChecker import SpellChecker


@pytest.mark.parametrize("token, verwacht", [
    # Letters met een cijfer ertussen: de respondent typte een letter verkeerd.
    ("2eet", True),
    ("N8ks", True),
    ("Go4ed", True),
    # Gewone woorden blijven vanzelfsprekend door.
    ("eekhoorn", True),
    ("café", True),
    # Alleen cijfers: geen woord, niets te corrigeren.
    ("60", False),
    ("110", False),
    # Leestekens erin: samenstellingen en afkortingen die correct zijn zoals ze
    # staan. Die willen we hier juist niet binnenhalen.
    ("zzp-ers", False),
    ("i.o.", False),
    ("n.v.t.", False),
])
def test_is_checkable(token, verwacht):
    assert SpellChecker.is_checkable(token) is verwacht


@pytest.mark.parametrize("word, verwacht", [
    # Een cijfer staat voor een letter die we niet kunnen zien. De klinker- en
    # medeklinkertoets kunnen daarover niets concluderen, dus zwijgen ze.
    ("N8ks", False),
    ("Go4ed", False),
    ("2eet", False),
    # Zonder cijfer verandert er niets aan het oordeel.
    ("Xxx", True),
    ("Jsisjdkdjd", True),
    ("Fvvvb", True),
    ("BLG", True),          # afkorting zonder klinker, blijft beschermd
    ("eekhorn", False),     # gewone typefout, moet gecorrigeerd worden
    ("maatschappelihjk", False),
])
def test_is_unrepairable(word, verwacht):
    assert SpellChecker.is_unrepairable(word) is verwacht
