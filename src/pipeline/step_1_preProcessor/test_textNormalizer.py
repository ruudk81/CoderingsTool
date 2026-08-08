"""Tests voor TextNormalizer.

Fixtures zijn losse tekstfragmenten, geen responses uit een klantbestand.
"""

import pytest

from pipeline.step_1_preProcessor.textNormalizer import TextNormalizer


@pytest.fixture
def normalizer():
    return TextNormalizer()


def test_benoemde_entiteit_wordt_ontsleuteld(normalizer):
    assert normalizer.normalize_response("&quot;groen&quot;, ooit anders") == '"groen", ooit anders'


def test_numerieke_entiteit_wordt_een_teken(normalizer):
    # &#304; is de Turkse punt-I. Ontsleutelen maakt de fout bereikbaar voor de
    # speller; het maakt de tekst nog niet correct. Dat is het doel hier.
    assert normalizer.normalize_response("&#304;ets meer menselijkheid") == "İets meer menselijkheid"


def test_losse_ampersand_blijft_staan(normalizer):
    # Geen geldige entiteit, dus niets te ontsleutelen.
    assert normalizer.normalize_response("A&O, prima winkel") == "A&O, prima winkel"


def test_ampersand_entiteit_wordt_een_ampersand(normalizer):
    assert normalizer.normalize_response("kop &amp; schotel") == "kop & schotel"


def test_tekst_zonder_entiteit_verandert_niet(normalizer):
    assert normalizer.normalize_response("gewoon een bank") == "gewoon een bank"
