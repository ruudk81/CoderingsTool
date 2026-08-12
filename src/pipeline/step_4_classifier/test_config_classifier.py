"""Tests voor de step-4-configuratie."""
import dataclasses
from pipeline.step_4_classifier.config_classifier import CategoriesConfig, DEFAULT_CONFIG

PHASES = (
    "facet_discovery", "facet_consolidation", "facet_assignment", "facet_refinement",
    "attribute_discovery", "attribute_consolidation", "attribute_assignment",
    "attribute_refinement", "valence_merge",
)


def test_elke_fase_heeft_een_modelsleutel():
    for phase in PHASES:
        assert hasattr(DEFAULT_CONFIG, f"model_{phase}")


def test_geen_veld_verwijst_nog_naar_een_fasenummer():
    for f in dataclasses.fields(CategoriesConfig):
        assert not f.name.startswith("qr_model_p")
        assert not f.name.startswith("p4_")
        assert not f.name.startswith("p9_")


def test_assenvlag_bestaat_niet_meer():
    assert not hasattr(DEFAULT_CONFIG, "axis_first_enabled")


def test_toewijzingsvlaggen_gelden_voor_beide_niveaus():
    """Ze sturen zowel facet- als attribuuttoewijzing aan, dus geen facet_-prefix."""
    for f in ("assignment_batch_k", "assignment_shortlist_enabled",
              "assignment_shortlist_k"):
        assert hasattr(DEFAULT_CONFIG, f)
    for f in ("facet_assignment_batch_enabled", "facet_assignment_label_dedup"):
        assert not hasattr(DEFAULT_CONFIG, f)


def test_consolidatiegrenzen_zijn_aanwezig():
    for f in ("consolidation_max_chunks_per_call",
              "consolidation_max_items_per_call",
              "consolidation_max_rounds"):
        assert hasattr(DEFAULT_CONFIG, f)


def test_elke_modelsleutel_bestaat_in_config_step_model():
    """Een sport die niet in STEP_MODEL staat gooit bij import — die fout moet
    hier opduiken en niet halverwege een betaalde run."""
    from config import get_step_model
    for phase in PHASES:
        assert get_step_model(f"classifier_{phase}")
