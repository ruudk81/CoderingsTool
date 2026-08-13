"""Tests voor de configuratie van step 4."""
import pytest

from config import STEP_MODEL
from pipeline.step_4_classifier.classifier import TaxonomyClassifier
from pipeline.step_4_classifier.config_classifier import CategoriesConfig


def test_elke_fase_heeft_een_modelsleutel():
    """PHASES en de config-attributen zijn een tabel, geen twee registers die
    uit elkaar kunnen lopen."""
    config = CategoriesConfig()
    for phase in TaxonomyClassifier.PHASES:
        assert hasattr(config, f"model_{phase}"), phase


def test_geen_modelsleutel_zonder_fase():
    config = CategoriesConfig()
    model_attrs = {a[len("model_"):] for a in vars(CategoriesConfig)
                   if a.startswith("model_")}
    assert model_attrs == set(TaxonomyClassifier.PHASES)


def test_elke_modelsleutel_bestaat_in_config_step_model():
    """Een rung die niet bestaat is een RuntimeError bij import, nooit een
    stille omleiding."""
    for phase in TaxonomyClassifier.PHASES:
        assert f"classifier_{phase}" in STEP_MODEL, phase


def test_consolidatiegrenzen_zijn_aanwezig():
    config = CategoriesConfig()
    assert config.consolidation_max_items_per_call > 0
    assert config.consolidation_max_rounds > 0


def test_chunking_heeft_een_register_niet_twee():
    """Facetten en attributen komen uit dezelfde call, dus er is een
    chunkbron."""
    attrs = set(vars(CategoriesConfig))
    assert "batch_size_min" in attrs
    assert not any(a.startswith("attribute_chunk") for a in attrs)


def test_batch_en_shortlistknoppen_zijn_weg():
    """Er is geen batch meer om te valideren, en een getrimd menu zou de
    catch-all de makkelijke uitweg maken."""
    attrs = set(vars(CategoriesConfig))
    for verdwenen in ("assignment_batch_k", "assignment_shortlist_enabled",
                      "assignment_shortlist_k"):
        assert verdwenen not in attrs, verdwenen


def test_stopfase_neemt_een_naam():
    config = CategoriesConfig()
    assert config.stop_after_phase is None
    TaxonomyClassifier(CategoriesConfig(stop_after_phase="discovery"))


def test_onbekende_stopfase_is_een_valueerror():
    """De numerieke voorganger draaide de volledige pijplijn bij elke waarde
    die geen stoppunt was, en dat kostte een volle run om te ontdekken."""
    with pytest.raises(ValueError):
        TaxonomyClassifier(CategoriesConfig(stop_after_phase="facet_discovery"))
