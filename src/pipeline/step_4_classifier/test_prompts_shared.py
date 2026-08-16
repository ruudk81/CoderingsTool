"""Tests voor de gedeelde promptbouwstenen (step 4)."""
from pipeline.step_3_ideaExtractor.dimension_data import get_dimensions_in_decision_order
from pipeline.step_4_classifier.prompts_shared import (
    INSTRUCTOR_HINT,
    UNIVERSAL_RULES,
    build_context_block,
    build_cross_scope_model,
    build_taxonomy_block,
)

DIM = get_dimensions_in_decision_order()[0]


def test_the_context_block_holds_all_seven_fields():
    block = build_context_block(
        language="Dutch", survey_question="Waar denkt u aan?",
        sector="finance", entity="asn_bank", topic="brand_association",
        perspective="consumer", intent="associate",
    )
    for value in ("Dutch", "Waar denkt u aan?", "finance",
                  "asn_bank", "brand_association", "consumer", "associate"):
        assert value in block


def test_the_taxonomy_block_holds_all_four_levels():
    block = build_taxonomy_block(
        dimension=DIM, dimension_name=DIM.key,
        dimension_description=DIM.dimension_description,
    )
    for marker in ("L1", "L2", "L3", "L4"):
        assert marker in block


def test_taxonomy_block_calls_l1_the_dimension_and_not_the_lens():
    """The lens naming came out of the rebuild and goes out again: the prompts
    call the level here by the name `dimension_data` itself uses."""
    block = build_taxonomy_block(
        dimension=DIM, dimension_name=DIM.key,
        dimension_description=DIM.dimension_description,
    )
    assert "Lens" not in block
    assert "L1 — Dimension" in block


def test_level_diagnostic_no_longer_exists():
    """The dimension instruction went with the two-layer design: discovery asks
    for facets and attributes themselves, not for the axis they differ along."""
    import pipeline.step_4_classifier.prompts_shared as ps
    assert not hasattr(ps, "level_diagnostic")


def test_the_universal_rules_cover_the_agreements():
    tekst = UNIVERSAL_RULES.lower()
    assert "descriptive" in tekst
    assert "valence" in tekst
    assert "evaluative direction" in tekst


def test_the_instructor_hint_is_the_exact_sentence():
    assert INSTRUCTOR_HINT == (
        "provide your output as valid JSON following the response schema provided"
    )


def test_the_cross_scope_model_enforces_the_id_space():
    model = build_cross_scope_model(["A1", "A2"], "attribute")
    fields = model.model_fields
    assert set(fields) == {"scratchpad", "items"}
    item = fields["items"].annotation.__args__[0]
    assert set(item.model_fields) == {"name", "definition", "source_ids", "home_id"}


def test_the_universal_rules_forbid_a_self_invented_residual():
    """The model made eight attributes literally named 'Overig', alongside the
    catch-alls the code already offers (measured 2026-08-13)."""
    tekst = UNIVERSAL_RULES
    assert "NEVER CREATE A LEFTOVER CATEGORY" in tekst
    assert '"Other"' in tekst


def test_the_ban_does_not_clash_with_the_residual_for_bare_judgments():
    """Rule 2 sends bare judgments to a single residual precisely — that one is
    defined by what they are, not by what they lack."""
    tekst = UNIVERSAL_RULES
    assert "residual overall-judgment item" in tekst
    assert "Overall judgment" in tekst
    assert "not a ban on abstraction" in tekst.lower()


# =============================================================================
# HET UITVOERCONTRACT — één plek, en dat is het responsemodel
# =============================================================================
#
# Elke prompt beschreef zijn JSON-vorm ook zelf, in een `# Output`-blok. Dat is
# dezelfde instructie op twee plekken: instructor rendert het schema mét zijn
# descriptions al in de call. Twee plekken lopen uit elkaar — en dat gebeurde:
# het kandidatenblok toonde één voorbeeld terwijl de outputspec er 2-3 eiste.
#
# De blokken zijn weg. Wat een veld betekent staat in zijn `Field(description=)`
# en nergens anders. Regels die naar een veld verwijzen blijven wel in de prompt
# staan — "elk id moet in `source_facet_ids` van minstens één survivor staan" is
# een regel, geen vormvoorschrift.

def describable_fields(model, _seen=None):
    """Elk veld van een responsemodel, ook die van geneste modellen."""
    _seen = set() if _seen is None else _seen
    if model in _seen:
        return
    _seen.add(model)
    for name, field in model.model_fields.items():
        yield f"{model.__name__}.{name}", field
        for nested in _nested_models(field.annotation):
            yield from describable_fields(nested, _seen)


def _nested_models(annotation):
    import typing
    import pydantic
    if isinstance(annotation, type) and issubclass(annotation, pydantic.BaseModel):
        yield annotation
        return
    for arg in typing.get_args(annotation):
        yield from _nested_models(arg)


def assert_every_field_is_described(model):
    for where, field in describable_fields(model):
        assert field.description, where


def assert_prompt_does_not_restate_the_schema(prompt):
    assert "# Output" not in prompt
    assert "Return a JSON object" not in prompt
