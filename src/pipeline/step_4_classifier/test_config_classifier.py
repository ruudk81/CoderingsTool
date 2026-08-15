"""Tests voor de configuratie van step 4."""
import pytest

from config import STEP_MODEL
from pipeline.step_4_classifier.classifier import TaxonomyClassifier
from pipeline.step_4_classifier.config_classifier import CategoriesConfig


def test_elke_fase_heeft_een_modelsleutel():
    """PHASES and the config attributes are one table, not two registers that
    can drift apart."""
    config = CategoriesConfig()
    for phase in TaxonomyClassifier.PHASES:
        assert hasattr(config, f"model_{phase}"), phase


def test_geen_modelsleutel_zonder_fase():
    config = CategoriesConfig()
    model_attrs = {a[len("model_"):] for a in vars(CategoriesConfig)
                   if a.startswith("model_")}
    assert model_attrs == set(TaxonomyClassifier.PHASES)


def test_elke_modelsleutel_bestaat_in_config_step_model():
    """A rung that does not exist is a RuntimeError at import, never a silent
    reroute."""
    for phase in TaxonomyClassifier.PHASES:
        assert f"classifier_{phase}" in STEP_MODEL, phase


def test_consolidatiegrenzen_zijn_aanwezig():
    config = CategoriesConfig()
    assert config.consolidation_max_items_per_call > 0
    assert config.consolidation_max_rounds > 0


def test_chunking_has_one_register_not_two():
    """Facetten en attributen komen uit dezelfde call, dus er is een
    chunkbron."""
    attrs = set(vars(CategoriesConfig))
    assert "batch_size_min" in attrs
    assert not any(a.startswith("attribute_chunk") for a in attrs)


def test_batch_en_shortlistknoppen_zijn_weg():
    """There is no batch left to validate, and a trimmed menu would make the
    catch-all the easy way out."""
    attrs = set(vars(CategoriesConfig))
    for verdwenen in ("assignment_batch_k", "assignment_shortlist_enabled",
                      "assignment_shortlist_k"):
        assert verdwenen not in attrs, verdwenen


def test_stopfase_neemt_een_naam():
    config = CategoriesConfig()
    assert config.stop_after_phase is None
    TaxonomyClassifier(CategoriesConfig(stop_after_phase="discovery"))


def test_onbekende_stopfase_is_een_valueerror():
    """The numeric predecessor ran the full pipeline for every value that was
    not a stop point, and that cost a full run to discover."""
    with pytest.raises(ValueError):
        TaxonomyClassifier(CategoriesConfig(stop_after_phase="facet_discovery"))
