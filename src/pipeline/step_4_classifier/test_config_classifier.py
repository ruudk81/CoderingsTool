"""Tests for the configuration of step 4."""
import pytest

from config import STEP_MODEL
from pipeline.step_4_classifier.classifier import TaxonomyClassifier
from pipeline.step_4_classifier.config_classifier import CategoriesConfig


def test_the_phases_name_both_consolidation_calls():
    """One name for one job: facets and attributes are settled by two calls."""
    assert TaxonomyClassifier.PHASES == (
        "discovery", "facet_consolidation", "facet_assignment", "facet_settle",
        "attribute_consolidation", "assignment", "refinement", "cross_domain",
        "valence_merge")


def test_every_phase_has_a_model_key():
    """PHASES and the config attributes are one table, not two registers that
    can drift apart."""
    config = CategoriesConfig()
    for phase in TaxonomyClassifier.PHASES:
        assert hasattr(config, f"model_{phase}"), phase


def test_no_model_key_without_a_phase():
    model_attrs = {a[len("model_"):] for a in vars(CategoriesConfig)
                   if a.startswith("model_")}
    assert model_attrs == set(TaxonomyClassifier.PHASES)


def test_every_model_key_exists_in_config_step_model():
    """A rung that does not exist is a RuntimeError at import, never a silent
    reroute."""
    for phase in TaxonomyClassifier.PHASES:
        assert f"classifier_{phase}" in STEP_MODEL, phase


def test_the_consolidation_caps_are_present():
    """Two phases, two scopes, two caps — and one round budget over both."""
    config = CategoriesConfig()
    assert config.consolidation_max_rounds > 0
    assert config.facet_consolidation_max_facets_per_call > 0
    assert config.attribute_consolidation_max_attributes_per_call > 0


def test_chunking_has_one_register_not_two():
    """Facets and attributes come from the same discovery call, so there is one
    chunking source."""
    attrs = set(vars(CategoriesConfig))
    assert "batch_size_min" in attrs
    assert not any(a.startswith("attribute_chunk") for a in attrs)


def test_the_batch_and_shortlist_knobs_are_gone():
    """There is no batch left to validate, and a trimmed menu would make the
    catch-all the easy way out."""
    attrs = set(vars(CategoriesConfig))
    for verdwenen in ("assignment_batch_k", "assignment_shortlist_enabled",
                      "assignment_shortlist_k"):
        assert verdwenen not in attrs, verdwenen


def test_the_stop_phase_takes_a_name():
    config = CategoriesConfig()
    assert config.stop_after_phase is None
    TaxonomyClassifier(CategoriesConfig(stop_after_phase="discovery"))


def test_both_consolidation_phases_are_valid_stop_points():
    """A partial result is a real result, so each of the two must be nameable —
    stopping after the facets is how the pooled-but-unsettled state is read."""
    for phase in ("facet_consolidation", "attribute_consolidation"):
        clf = TaxonomyClassifier(CategoriesConfig(stop_after_phase=phase))
        assert clf._stop_after_phase == phase


def test_an_unknown_stop_phase_is_a_valueerror():
    """The numeric predecessor ran the full pipeline for every value that was
    not a stop point, and that cost a full run to discover."""
    with pytest.raises(ValueError):
        TaxonomyClassifier(CategoriesConfig(stop_after_phase="facet_discovery"))


def test_chunk_consolidation_is_no_longer_a_phase():
    """One name for one job: the old name did two, and a run that still asked
    for it would silently have run everything."""
    with pytest.raises(ValueError, match="not a phase"):
        TaxonomyClassifier(
            CategoriesConfig(stop_after_phase="chunk_consolidation"))
