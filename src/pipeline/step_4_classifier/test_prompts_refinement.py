"""Tests voor naslijpen en cross-domein (step 4)."""
from pipeline.step_3_ideaExtractor.dimension_data import get_dimensions_in_decision_order
from pipeline.step_4_classifier.drains import make_drain_attribute
from pipeline.step_4_classifier.prompts_refinement import (
    RefinedAttribute,
    RefinementMisfitGroup,
    RefinementResult,
    build_contents_block,
    build_cross_domain_prompt,
    build_refinement_prompt,
)
from pipeline.step_4_classifier.prompts_shared import (
    INSTRUCTOR_HINT, build_cross_scope_model,
)
from pipeline.step_4_classifier.test_prompts_shared import (
    assert_every_field_is_described, assert_prompt_does_not_restate_the_schema,
)

DIM = get_dimensions_in_decision_order()[0]

FACETS = [{
    "facet_name": "Snelheid",
    "facet_definition": "Hoe snel er geleverd wordt.",
    "attributes": [
        {"attribute_name": "Wachttijd", "attribute_definition": "De tijd tot antwoord."},
        make_drain_attribute("Snelheid", "Dutch"),
    ],
}]
CONTENTS = {"Wachttijd": ["lange wachttijd", "duurt lang", "snel geholpen"]}
SHARES = {"Wachttijd": 0.62}
COUNTS = {"Wachttijd": 124}


def _kwargs(**overrides):
    base = dict(
        language="Dutch", survey_question="Waar denkt u aan?",
        sector="finance", entity="asn_bank", topic="brand_association",
        perspective="consumer", intent="associate",
        dimension=DIM, dimension_name=DIM.key,
        dimension_description=DIM.dimension_description,
        domain_label="dienstverlening",
        domain_definition="Alles wat de organisatie aanbiedt en levert.",
        contents_block=build_contents_block(FACETS, CONTENTS, SHARES, COUNTS, 5),
    )
    base.update(overrides)
    return base


def _xkwargs(**overrides):
    base = dict(
        language="Dutch", survey_question="Waar denkt u aan?",
        sector="finance", entity="asn_bank", topic="brand_association",
        perspective="consumer", intent="associate",
        inventory_block="[A1] Wachttijd — 124 responses",
    )
    base.update(overrides)
    return base


# =============================================================================
# HET INHOUDSBLOK
# =============================================================================

def test_inhoudsblok_zet_claim_naast_inhoud():
    block = build_contents_block(FACETS, CONTENTS, SHARES, COUNTS, 5)
    assert "Claims to capture" in block
    assert "Actually holds" in block
    assert block.index("Claims to capture") < block.index("Actually holds")


def test_inhoudsblok_toont_aandeel_relatief():
    block = build_contents_block(FACETS, CONTENTS, SHARES, COUNTS, 5)
    assert "124 responses" in block
    assert "62% of this domain" in block


def test_inhoudsblok_kapt_op_top_n():
    block = build_contents_block(FACETS, CONTENTS, SHARES, COUNTS, 2)
    assert "snel geholpen" not in block


def test_an_empty_attribute_is_named_as_such():
    block = build_contents_block(FACETS, {}, {}, {}, 5)
    assert "nothing was assigned to it" in block


# =============================================================================
# DE HERSCHREVEN SPLITSCLAUSULE
# =============================================================================

def test_the_old_split_clause_is_gone():
    """'a large share AND visibly diverse contents is too abstract: split it'
    vuurde voortdurend zodra discovery items bewust breed maakte."""
    prompt = build_refinement_prompt(**_kwargs())
    assert "too abstract" not in prompt


def test_the_new_split_clause_requires_two_distinct_answers():
    prompt = build_refinement_prompt(**_kwargs())
    assert "two distinct ANSWERS to the same question" in prompt


def test_widen_is_the_default_on_doubt():
    prompt = build_refinement_prompt(**_kwargs())
    assert "Otherwise WIDEN" in prompt
    assert "prefer widen" in prompt


def test_broad_is_not_a_problem_in_itself():
    prompt = build_refinement_prompt(**_kwargs())
    assert "not by itself a problem" in prompt


# =============================================================================
# DE VIJF UITGANGEN
# =============================================================================

def test_all_five_exits_are_in_the_prompt():
    prompt = build_refinement_prompt(**_kwargs())
    for uitgang in ("merge", "widen", "split", "move", "out"):
        assert f'"{uitgang}"' in prompt, uitgang


def test_model_knows_the_four_actions_and_the_two_verdicts():
    assert str(RefinedAttribute.model_fields["action"].annotation) == (
        "typing.Literal['keep', 'merge', 'widen', 'split']")
    assert str(RefinementMisfitGroup.model_fields["verdict"].annotation) == (
        "typing.Literal['move', 'out']")


def test_resultaat_draagt_attributen_en_misfits():
    assert set(RefinementResult.model_fields) == {
        "scratchpad", "attributes", "misfits"}


# =============================================================================
# DOMEIN VAST, FACET NIET
# =============================================================================

def test_an_attribute_may_change_facet_within_the_domain():
    prompt = build_refinement_prompt(**_kwargs())
    assert "THE DOMAIN IS FIXED, THE FACET IS NOT" in prompt
    assert "facet_name" in RefinedAttribute.model_fields


def test_the_refinement_prompt_asks_whether_each_attribute_sits_right():
    """Consolidation used to check this while it held every facet of the
    domain. It now settles one facet per call, so the check moved to the phase
    that still sees them all."""
    prompt = build_refinement_prompt(**_kwargs())
    assert "sits under the facet it belongs to" in prompt
    assert "saw one facet at a time" in prompt


def test_the_placement_check_lives_in_the_facet_rule():
    """A placement check parked outside the rule that grants the facet move is
    a sentence without a mechanism, so pin it to rule 5."""
    prompt = build_refinement_prompt(**_kwargs())
    rule_five = prompt[
        prompt.index("**5. THE DOMAIN IS FIXED"):prompt.index("**6. FIVE EXITS")]
    assert "sits under the facet it belongs to" in rule_five
    assert "saw one facet at a time" in rule_five


def test_the_model_describes_every_field_it_has():
    assert_every_field_is_described(RefinementResult)


def test_the_prompt_does_not_restate_the_schema():
    assert_prompt_does_not_restate_the_schema(build_refinement_prompt(**_kwargs()))


# =============================================================================
# VANGNETTEN DOEN NIET MEE
# =============================================================================

def test_refinement_leaves_the_catch_alls_alone():
    prompt = build_refinement_prompt(**_kwargs())
    assert "LEAVE THE CATCH-ALLS ALONE" in prompt


def test_cross_domain_leaves_the_catch_alls_alone():
    prompt = build_cross_domain_prompt(**_xkwargs())
    assert "catch-all" in prompt


# =============================================================================
# CROSS-DOMEIN
# =============================================================================

def test_cross_domein_werkt_op_ids():
    """Groups come back as ids plus a home, never as free text that has to be
    matched back. That is a property of the model, and since the schema is the
    only place the output shape is stated, this is where it is checked."""
    item = build_cross_scope_model(["A1", "A2"], "attribute").model_fields[
        "items"].annotation.__args__[0]
    assert {"source_ids", "home_id"} <= set(item.model_fields)


def test_cross_domain_says_a_soloist_is_a_group_of_one():
    prompt = build_cross_domain_prompt(**_xkwargs())
    assert "group of one" in prompt


def test_cross_domain_requires_every_id_exactly_once():
    """A forgotten id is what makes this round lose an attribute silently, so
    the demand is stated twice in the schema: once as the reasoning step that
    checks it, once on the field that has to satisfy it."""
    model = build_cross_scope_model(["A1", "A2"], "attribute")
    assert "every id appears exactly once" in model.model_fields[
        "scratchpad"].description
    assert "Every input id appears exactly once" in model.model_fields[
        "items"].description


# =============================================================================
# ALGEMEEN
# =============================================================================

def test_both_prompts_end_on_the_instructor_sentence():
    assert build_refinement_prompt(**_kwargs()).rstrip().endswith(INSTRUCTOR_HINT)
    assert build_cross_domain_prompt(**_xkwargs()).rstrip().endswith(INSTRUCTOR_HINT)


def test_both_prompts_carry_the_universal_rules():
    assert "<universal_rules>" in build_refinement_prompt(**_kwargs())
    assert "<universal_rules>" in build_cross_domain_prompt(**_xkwargs())


def test_rules_do_not_use_the_word_dimension():
    prompt = build_refinement_prompt(**_kwargs())
    regels = prompt[prompt.index("# Rules"):prompt.index("# Required Process")]
    assert "dimension" not in regels.lower()


def test_no_threshold_numbers_in_the_rules():
    """Shares come from the data and are allowed; a fixed threshold is read off
    one dataset and is not."""
    prompt = build_refinement_prompt(**_kwargs())
    regels = prompt[prompt.index("# Rules"):prompt.index("# Required Process")]
    assert "%" not in regels
    assert "never against a fixed number" in " ".join(regels.split())


def test_catch_alls_are_marked_not_recognised_by_name():
    """The name is translated and rewritable; the marker comes from drain_key."""
    block = build_contents_block(FACETS, CONTENTS, SHARES, COUNTS, 5)
    gemarkeerd = [r for r in block.splitlines() if "[CATCH-ALL]" in r]
    assert len(gemarkeerd) == 1
    assert "Overig" in gemarkeerd[0]
    assert "Wachttijd" not in gemarkeerd[0]


def test_rule_nine_points_at_the_marker():
    prompt = build_refinement_prompt(**_kwargs())
    assert "marked [CATCH-ALL]" in prompt
    assert "Judge only what is not marked" in prompt
