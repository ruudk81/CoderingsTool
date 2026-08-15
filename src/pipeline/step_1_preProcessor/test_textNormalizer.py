"""Tests for TextNormalizer.

Fixtures are standalone text fragments, not responses from a client file.
"""

import pytest

from pipeline.step_1_preProcessor.textNormalizer import TextNormalizer


@pytest.fixture
def normalizer():
    return TextNormalizer()


def test_benoemde_entiteit_wordt_ontsleuteld(normalizer):
    assert normalizer.normalize_response("&quot;groen&quot;, ooit anders") == '"groen", ooit anders'


def test_a_numeric_entity_becomes_a_character(normalizer):
    # &#304; is the Turkish dotted I. Unescaping puts the error within reach of
    # the speller; it does not yet make the text correct. That is the goal here.
    assert normalizer.normalize_response("&#304;ets meer menselijkheid") == "İets meer menselijkheid"


def test_losse_ampersand_blijft_staan(normalizer):
    # Geen geldige entiteit, dus niets te ontsleutelen.
    assert normalizer.normalize_response("A&O, prima winkel") == "A&O, prima winkel"


def test_an_ampersand_entity_becomes_an_ampersand(normalizer):
    assert normalizer.normalize_response("kop &amp; schotel") == "kop & schotel"


def test_text_without_an_entity_does_not_change(normalizer):
    assert normalizer.normalize_response("gewoon een bank") == "gewoon een bank"
