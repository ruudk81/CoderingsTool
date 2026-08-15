"""Tests for the sieves that decide WHAT is offered as a spelling error.

They are pure functions over single words, so they can be tested without
Hunspell, spaCy or an LLM. That is exactly the point: this is the layer every
data corruption in this step came from.
"""

import pytest

from pipeline.step_1_preProcessor.spellChecker import SpellChecker


@pytest.mark.parametrize("token, verwacht", [
    # Letters with a digit among them: the respondent mistyped a letter.
    ("2eet", True),
    ("N8ks", True),
    ("Go4ed", True),
    # Gewone woorden blijven vanzelfsprekend door.
    ("eekhoorn", True),
    ("café", True),
    # Digits only: not a word, nothing to correct.
    ("60", False),
    ("110", False),
    # Punctuation inside: compounds and abbreviations that are correct as they
    # stand. Those are exactly what we do not want to pull in here.
    ("zzp-ers", False),
    ("i.o.", False),
    ("n.v.t.", False),
])
def test_is_checkable(token, verwacht):
    assert SpellChecker.is_checkable(token) is verwacht


@pytest.mark.parametrize("word, verwacht", [
    # A digit stands for a letter we cannot see. The vowel and
    # medeklinkertoets kunnen daarover niets concluderen, dus zwijgen ze.
    ("N8ks", False),
    ("Go4ed", False),
    ("2eet", False),
    # Without a digit the verdict does not change.
    ("Xxx", True),
    ("Jsisjdkdjd", True),
    ("Fvvvb", True),
    ("BLG", True),          # afkorting zonder klinker, blijft beschermd
    ("eekhorn", False),     # gewone typefout, moet gecorrigeerd worden
    ("maatschappelihjk", False),
])
def test_is_unrepairable(word, verwacht):
    assert SpellChecker.is_unrepairable(word) is verwacht


@pytest.mark.parametrize("tekst, verwacht", [
    ("oké", "oke"),
    ("ideëel", "ideeel"),
    ("georiënteerd", "georienteerd"),
    ("één", "een"),
    ("Café", "cafe"),
    # Without an accent only the capitalisation changes.
    ("Eekhoorn", "eekhoorn"),
    ("asn", "asn"),
])
def test_deaccent(tekst, verwacht):
    assert SpellChecker.deaccent(tekst) == verwacht


@pytest.mark.parametrize("woord, suggesties, verwacht", [
    # An accent difference only: this is a typo, not a name.
    ("oke", ["oké", "koe", "ode"], True),
    ("ideeel", ["ideëel"], True),
    # Hunspell sometimes offers only a different capitalisation. That is not an
    # accent error, and a brand name must not lose its protection over it.
    ("sns", ["SNS", "sos"], False),
    ("asn", ["ASN", "aan"], False),
    # Een echt ander woord laat de bescherming staan.
    ("nvt", ["nv", "n.v.t."], False),
])
def test_accentfout_herkennen(woord, suggesties, verwacht):
    assert any(SpellChecker._is_diacritic_variant(woord, s) for s in suggesties) is verwacht
