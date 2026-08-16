"""Tests for the consolidation prompts of step 4.

Covers both prompts in the module: the facet call, which settles which facets a
domain has, and the attribute call, which settles the pool inside one of them.
Two sections guard the wording each half of the split ruled on. Their negative
assertions used to be paired against the combined prompt that both replaced;
that prompt is gone, so each pair is now anchored on the replacement wording in
the same prompt — a bare `not in` passes vacuously the moment the literal is
mistyped.
"""
from pipeline.step_3_ideaExtractor.dimension_data import get_dimensions_in_decision_order
from pipeline.step_4_classifier.prompts_discovery import DiscoveredAttribute
from pipeline.step_4_classifier.prompts_consolidation import (
    AttributeConsolidationResult,
    FacetConsolidationResult,
    FacetPool,
    SettledAttribute,
    SettledFacet,
    build_attribute_candidate_block,
    build_attribute_candidate_index,
    build_attribute_consolidation_prompt,
    build_facet_candidate_block,
    build_facet_candidate_index,
    build_facet_consolidation_prompt,
)
from pipeline.step_4_classifier.test_prompts_shared import (
    assert_every_field_is_described, assert_prompt_does_not_restate_the_schema,
)
from pipeline.step_4_classifier.prompts_shared import INSTRUCTOR_HINT

DIM = get_dimensions_in_decision_order()[0]

RECURRENCE = {"Snelheid": 4, "Snelheid van afhandeling": 1, "Bejegening": 5}


def _pool(name, *attr_names, question=""):
    return FacetPool(
        facet_name=name,
        facet_definition=f"Wat {name} vastlegt.",
        facet_question=question,
        attributes=[DiscoveredAttribute(
            attribute_name=a,
            attribute_definition=f"De eigenschap {a}.",
            example_observations=[f"observatie over {a}"],
        ) for a in attr_names],
    )


POOLS = [
    _pool("Snelheid", "Wachttijd", "Doorlooptijd"),
    _pool("Snelheid van afhandeling", "Wachttijd"),
    _pool("Bejegening", "Vriendelijkheid"),
]


def _facet_kwargs(**overrides):
    base = dict(
        language="Dutch",
        survey_question="Waar denkt u aan?",
        sector="finance", entity="asn_bank", topic="brand_association",
        perspective="consumer", intent="associate",
        dimension=DIM,
        dimension_name=DIM.key,
        dimension_description=DIM.dimension_description,
        domain_label="dienstverlening",
        domain_definition="Alles wat de organisatie aanbiedt en levert.",
        domain_exclusions=["prijs en kosten"],
        candidate_block=build_facet_candidate_block(POOLS, RECURRENCE, 6),
    )
    base.update(overrides)
    return base


# =============================================================================
# THE FACET PHASE
# =============================================================================

def test_facet_ids_are_positional_and_stable():
    index = build_facet_candidate_index(POOLS)
    assert list(index) == ["F1", "F2", "F3"]
    assert index["F2"].facet_name == "Snelheid van afhandeling"


def test_the_facet_block_shows_prevalence_per_facet():
    block = build_facet_candidate_block(POOLS, RECURRENCE, 6)
    assert "proposed under this exact name in 4 of 6 independent passes" in block
    assert "proposed under this exact name in 1 of 6 independent passes" in block


def test_the_facet_block_names_the_attributes_as_evidence():
    """A facet is only a good facet if what sits under it is one kind of thing.
    The names are shown for that judgement — they are not what this call
    returns."""
    block = build_facet_candidate_block(POOLS, RECURRENCE, 6)
    assert "Wachttijd" in block
    assert "Doorlooptijd" in block


def test_the_facet_block_leaves_out_definitions_and_examples():
    """Evidence, not material to consolidate: rendering each attribute in full
    is what made this call carry two jobs at once."""
    block = build_facet_candidate_block(POOLS, RECURRENCE, 6)
    assert "De eigenschap Wachttijd" not in block
    assert "observatie over Wachttijd" not in block


def test_a_pool_carries_its_question_once_a_round_has_set_one():
    block = build_facet_candidate_block(
        [_pool("Snelheid", "Wachttijd", question="Hoe snel ging het?")],
        RECURRENCE, 6)
    assert "Hoe snel ging het?" in block


def test_the_facet_result_is_a_decision_summary_plus_facets():
    assert set(FacetConsolidationResult.model_fields) == {
        "decision_summary", "facets"}


def test_a_consolidated_facet_carries_no_attributes():
    """The whole point of the split: this call decides the inventory, not what
    hangs under it."""
    assert "attributes" not in SettledFacet.model_fields


def test_a_consolidated_facet_states_its_question_and_its_sources():
    assert "facet_question" in SettledFacet.model_fields
    assert "source_facet_ids" in SettledFacet.model_fields


def test_the_facet_model_describes_every_field_it_has():
    assert_every_field_is_described(FacetConsolidationResult)


def test_the_facet_prompt_does_not_restate_the_schema():
    assert_prompt_does_not_restate_the_schema(
        build_facet_consolidation_prompt(**_facet_kwargs()))


def test_the_facet_model_carries_no_attribute_field():
    """This call settles the inventory and hands the pools on untouched. An
    attribute field would let it settle both levels again."""
    assert "source_attribute_ids" not in SettledFacet.model_fields


def test_the_facet_prompt_carries_the_merge_test():
    """Facet merging had no test at all while the merge test sat under the
    attribute step. Asserted on the facet-level phrasing: the bare heading would
    also pass on a merge test that still spoke about attributes."""
    prompt = build_facet_consolidation_prompt(**_facet_kwargs())
    assert "run it on any two facets" in prompt


def test_the_facet_prompt_does_not_ask_for_a_placement_check():
    """Placement moved to refinement, which is domain-scoped and can move an
    attribute between facets."""
    prompt = build_facet_consolidation_prompt(**_facet_kwargs())
    assert "must sit inside the facet" not in prompt


def test_the_facet_prompt_ends_on_the_universal_rules_and_the_hint():
    prompt = build_facet_consolidation_prompt(**_facet_kwargs())
    assert prompt.rstrip().endswith(INSTRUCTOR_HINT)
    assert "DESCRIPTIVE, NEVER EVALUATIVE" in prompt


def test_the_facet_prompt_asks_coverage_on_ids():
    """`source_facet_ids` is asserted by the field test above; what is specific
    here is that coverage is checked on the ids and not on the names."""
    prompt = build_facet_consolidation_prompt(**_facet_kwargs())
    assert "never on names" in prompt
    assert "at least one surviving" in prompt


def test_a_facet_without_a_count_falls_back_to_one_pass():
    """A survivor of round one carries no recurrence of its own, and a facet
    rendered without a count would read as one the passes never proposed."""
    block = build_facet_candidate_block([_pool("Nieuw", "A")], {}, 6)
    assert "proposed under this exact name in 1 of 6 independent passes" in block


def test_a_facet_without_attributes_does_not_break_the_block():
    """Not hypothetical: a survivor claimed by a second survivor hands its pool
    to the first and re-enters round two holding nothing."""
    block = build_facet_candidate_block([_pool("Leeg")], {"Leeg": 2}, 3)
    assert "Leeg" in block


def test_the_facet_prompt_carries_its_four_grouping_rules():
    prompt = build_facet_consolidation_prompt(**_facet_kwargs())
    for rule in ("UNDERLYING QUESTION FIRST", "PREVALENCE SETS GRANULARITY",
                 "LIFT, DON'T FLATTEN", "PLAIN, MEANINGFUL LABELS"):
        assert rule in prompt, rule


def test_the_facet_precedence_covers_every_rule_and_ends_on_the_count():
    """Without a precedence the rules fight; with one, orthogonality wins. It
    once read `1 > 2 > 4` while four numbered rules stood above it, so LIFT,
    DON'T FLATTEN had no place in the ordering at all.

    Minimisation comes last because this is the phase licensed to merge: a
    smaller inventory that has lost a distinction is not a better one.
    """
    prompt = build_facet_consolidation_prompt(**_facet_kwargs())
    order = prompt[prompt.index("When these conflict"):
                   prompt.index("# Step-by-Step")]
    for rule in ("Orthogonality", "Prevalence", "Lifting", "Label clarity"):
        assert rule in order, rule
    assert order.index("Orthogonality") < order.index("Fewest facets")
    assert "Never merge distinct concepts" in order


def test_the_facet_question_must_be_written_down():
    """Rule 1 asks whether two candidates answer the same question. Unless the
    question is stated, that test is a matter of feel and not checkable — and
    two survivors stating the same one is the visible violation the classifier
    logs as `duplicate_facet_question`. The rule travelled inside the removed
    `# Output` block; it lives on the field it constrains."""
    assert ("No two surviving facets may state the same one"
            in SettledFacet.model_fields["facet_question"].description)
    prompt = build_facet_consolidation_prompt(**_facet_kwargs())
    assert "`facet_question`" in prompt


def test_the_facet_question_is_tested_against_being_a_subject():
    """Sorting by what the material is ABOUT belongs one level up, and produces
    facets that overlap wherever a response touches two topics."""
    prompt = build_facet_consolidation_prompt(**_facet_kwargs())
    assert "answered by naming a subject" in prompt


def test_facet_labels_are_not_asked_to_avoid_nominalisations():
    """Nearly every usable Dutch or German taxonomy label is a nominalisation,
    so the rule could not be satisfied in the languages this runs on."""
    prompt = build_facet_consolidation_prompt(**_facet_kwargs())
    assert "nominalisation" not in prompt.lower()
    assert "ordinary noun phrase" in prompt


def test_rule_one_does_not_call_opposite_answers_a_reason_to_split():
    """`Mutually exclusive` and `opposite` are not the same thing, and the
    universal rules forbid splitting one concept by evaluative direction — in
    this same prompt."""
    prompt = build_facet_consolidation_prompt(**_facet_kwargs())
    assert "opposite answers" not in prompt
    assert "erase what" in prompt


def test_rule_one_does_not_use_the_word_dimension():
    """L1 is called Dimension in the taxonomy block. If rule 1 also uses the
    word for 'the axis along which a concept varies', it means two things on one
    page — which is exactly what the lens naming was once meant to solve."""
    prompt = build_facet_consolidation_prompt(**_facet_kwargs())
    rules = prompt[prompt.index("# Consolidation Rules"):
                   prompt.index("# Step-by-Step")]
    assert "dimension" not in rules.lower()
    assert "underlying question" in rules.lower()


def test_prevalence_is_not_an_absolute_veto():
    """Two concepts can both be well supported and still be one concept said
    twice; `never dissolve` made support outrank deduplication."""
    prompt = build_facet_consolidation_prompt(**_facet_kwargs())
    assert "unless it demonstrably" in prompt
    assert "never a reason to keep a duplicate" in prompt


def test_the_facet_prompt_separates_disposition_action_and_outcome():
    """A broad facet can otherwise treat an outcome as if it were an act. The
    discovery prompt draws this line; the phase that merges must hold it."""
    prompt = build_facet_consolidation_prompt(**_facet_kwargs())
    assert "different KINDS of statement" in prompt
    assert "Do not infer one from another either" in prompt


def test_the_facet_definitions_are_not_rendered_twice():
    """They already stand as L3 and L4 in the taxonomy block; the same sentence
    twice within a few hundred words is not an anchor."""
    prompt = build_facet_consolidation_prompt(**_facet_kwargs())
    assert "# What a facet is" not in prompt
    assert prompt.count("L3 — Facet:") == 1


def test_the_facet_prompt_carries_no_threshold_numbers_and_no_lens():
    """A percentage invites arithmetic where the phase needs judgement, and
    `lens` is a name the taxonomy no longer uses."""
    prompt = build_facet_consolidation_prompt(**_facet_kwargs())
    assert "%" not in prompt
    assert "Lens" not in prompt


def test_a_facet_prompt_without_exclusions_stays_valid():
    prompt = build_facet_consolidation_prompt(**_facet_kwargs(domain_exclusions=[]))
    assert prompt.rstrip().endswith(INSTRUCTOR_HINT)


# =============================================================================
# WHAT THE SPLIT RULED
#
# These guard decisions, not implementation details. Each ruling names the
# wording that was removed AND the wording that replaced it, in one prompt. The
# positive half is what keeps the negative half honest: on its own, `not in`
# starts passing the moment the literal is mistyped. The combined prompt these
# were once paired against no longer exists, so where a ruling was a pure
# removal the anchor is the surviving sentence it was removed from.
# =============================================================================

def test_ruling_the_facet_rules_do_not_point_at_an_attribute_step():
    """Rule 2 ended on `The same reasoning governs the attributes in step 6`.
    This prompt has no attribute step, and its step 6 is the coverage step — so
    the pointer resolved to something real and wrong."""
    facet = build_facet_consolidation_prompt(**_facet_kwargs())
    assert "The same reasoning governs the attributes in step 6" not in facet
    assert "PREVALENCE SETS GRANULARITY" in facet


def test_ruling_the_merge_test_asks_about_attributes_not_examples():
    """The facet candidate block renders attribute names and no examples, so
    item 4 asked about material the call cannot see."""
    facet = build_facet_consolidation_prompt(**_facet_kwargs())
    assert "does every attribute named under them still have" in facet
    assert "does every example still have" not in facet


def test_ruling_the_merge_test_closes_on_the_two_levels_it_has():
    """`not of items, not of examples` was vague where the prompt has two
    concrete levels, and named material it does not render."""
    facet = build_facet_consolidation_prompt(**_facet_kwargs())
    assert "not of facets, not of attributes." in facet
    assert "not of items, not of examples." not in facet


def test_ruling_rule_four_does_not_ask_for_attribute_names():
    """This call returns no attributes: `SettledFacet` has no such field. Asking
    for attribute names invites output the model cannot deliver."""
    facet = build_facet_consolidation_prompt(**_facet_kwargs())
    assert "Name every surviving facet in everyday language" in facet
    assert "Name every surviving facet and attribute" not in facet


def test_ruling_the_precedence_minimises_facets_not_items():
    """Same vagueness as the merge-test closing line, one screen above it."""
    facet = build_facet_consolidation_prompt(**_facet_kwargs())
    assert "6. Fewest facets —" in facet
    assert "6. Fewest items —" not in facet


def test_ruling_the_facet_prompt_never_mentions_examples_in_its_own_body():
    """The two surviving hits are rule 3 of UNIVERSAL_RULES, `NO BORROWED
    EXAMPLES` — a prohibition on importing outside material, which is the
    opposite failure mode and is shared by every step-4 prompt."""
    prompt = build_facet_consolidation_prompt(**_facet_kwargs())
    body = prompt[:prompt.index("<universal_rules>")]
    assert "example" not in body.lower()


POOL_ATTRIBUTES = [
    DiscoveredAttribute(
        attribute_name="Wachttijd",
        attribute_definition="Hoe lang iemand moest wachten.",
        example_observations=["lang wachten", "snel geholpen", "eindeloos"]),
    DiscoveredAttribute(
        attribute_name="Doorlooptijd",
        attribute_definition="Hoe lang de afhandeling duurde.",
        example_observations=["duurde weken"]),
]
POOL_RECURRENCE = {"Wachttijd": 5, "Doorlooptijd": 1}


def _attribute_kwargs(**overrides):
    base = dict(
        language="Dutch",
        survey_question="Waar denkt u aan?",
        sector="finance", entity="asn_bank", topic="brand_association",
        perspective="consumer", intent="associate",
        dimension=DIM,
        facet_name="Snelheid",
        facet_definition="Wat Snelheid vastlegt.",
        facet_question="Hoe snel ging het?",
        candidate_block=build_attribute_candidate_block(
            POOL_ATTRIBUTES, POOL_RECURRENCE, 6),
    )
    base.update(overrides)
    return base


# =============================================================================
# THE ATTRIBUTE PHASE
# =============================================================================

def test_attribute_ids_are_flat_within_one_facet():
    """One call is one facet, so an id needs no facet part."""
    index = build_attribute_candidate_index(POOL_ATTRIBUTES)
    assert list(index) == ["A1", "A2"]


def test_the_attribute_block_shows_prevalence():
    block = build_attribute_candidate_block(POOL_ATTRIBUTES, POOL_RECURRENCE, 6)
    assert "Wachttijd [5/6 passes]" in block
    assert "Doorlooptijd [1/6 passes]" in block


def test_the_attribute_block_shows_definition_and_examples():
    """This call consolidates the attributes, so it needs them in full — the
    facet call did not."""
    block = build_attribute_candidate_block(POOL_ATTRIBUTES, POOL_RECURRENCE, 6)
    assert "Hoe lang iemand moest wachten." in block
    assert "lang wachten" in block


def test_the_attribute_block_shows_at_most_three_examples():
    """Consolidation can only pass on what it is shown: showing one while the
    spec asks for up to three left the model inventing examples."""
    many = [DiscoveredAttribute(
        attribute_name="Wachttijd",
        attribute_definition="…",
        example_observations=[f"voorbeeld {i}" for i in range(9)])]
    block = build_attribute_candidate_block(many, POOL_RECURRENCE, 6)
    assert "voorbeeld 2" in block
    assert "voorbeeld 3" not in block


def test_an_attribute_without_a_count_falls_back_to_one_pass():
    block = build_attribute_candidate_block(POOL_ATTRIBUTES, {}, 6)
    assert "Wachttijd [1/6 passes]" in block


def test_the_attribute_result_is_a_decision_summary_plus_attributes():
    assert set(AttributeConsolidationResult.model_fields) == {
        "decision_summary", "attributes"}


def test_a_consolidated_attribute_states_what_folded_into_it():
    assert "source_attribute_ids" in SettledAttribute.model_fields


def test_a_settled_attribute_keeps_the_discovered_attribute_fields():
    """It is what the taxonomy carries onward, so it must still be a complete
    attribute and not just a provenance record."""
    item = SettledAttribute(
        attribute_name="Wachttijd", attribute_definition="…",
        example_observations=["lang wachten"], source_attribute_ids=["A1"])
    assert item.attribute_name == "Wachttijd"


def test_the_attribute_prompt_states_the_facet_it_works_inside():
    prompt = build_attribute_consolidation_prompt(**_attribute_kwargs())
    assert "Snelheid" in prompt
    assert "Hoe snel ging het?" in prompt


def test_the_attribute_prompt_omits_the_question_when_the_facet_has_none():
    """A facet settled without a call keeps the raw candidate, which carries no
    question. Rendering the label anyway left a bare `The question this facet
    answers:` above a rule that leans on it."""
    prompt = build_attribute_consolidation_prompt(
        **_attribute_kwargs(facet_question=""))
    assert "The question this facet answers" not in prompt
    assert "Facet: Snelheid — Wat Snelheid vastlegt.\n</taxonomy_facet>" in prompt


def test_the_attribute_model_describes_every_field_it_has():
    assert_every_field_is_described(AttributeConsolidationResult)


def test_the_attribute_prompt_does_not_restate_the_schema():
    assert_prompt_does_not_restate_the_schema(
        build_attribute_consolidation_prompt(**_attribute_kwargs()))


def test_the_attribute_prompt_forbids_dropping():
    """One facet in view cannot judge where something else belongs. Merging is
    allowed here; losing is not."""
    prompt = build_attribute_consolidation_prompt(**_attribute_kwargs())
    assert "never drop" in prompt.lower()


def test_the_attribute_prompt_carries_the_merge_test():
    prompt = build_attribute_consolidation_prompt(**_attribute_kwargs())
    assert "MERGE TEST" in prompt


def test_the_attribute_prompt_forbids_a_hierarchy_under_one_facet():
    """Step 5 carries what used to be rule 3: a general item and a specific one
    inside it are one level too many, and let the same response be coded twice."""
    prompt = build_attribute_consolidation_prompt(**_attribute_kwargs())
    assert "No survivor may sit inside another" in prompt


def test_the_attribute_prompt_carries_its_six_steps():
    """The five numbered rules became a six-step process on 2026-08-16. Three of
    them survive as steps 3, 4 and 5; the ranking, `lift don't flatten` and the
    plain-label rule were dropped deliberately."""
    prompt = build_attribute_consolidation_prompt(**_attribute_kwargs())
    for step in ("Step 1 — Scan the pool",
                 "Step 2 — Group what means the same",
                 "Step 3 — Apply the same-question test",
                 "Step 4 — Let prevalence set the granularity",
                 "Step 5 — Check for hierarchy",
                 "Step 6 — Account for every candidate"):
        assert step in prompt, step


def test_the_attribute_prompt_carries_no_threshold_numbers():
    prompt = build_attribute_consolidation_prompt(**_attribute_kwargs())
    assert "%" not in prompt


def test_the_attribute_prompt_asks_coverage_on_ids():
    prompt = build_attribute_consolidation_prompt(**_attribute_kwargs())
    assert "never on names" in prompt
    assert "at least one surviving" in prompt


def test_examples_are_one_to_three_and_never_a_reason_to_merge():
    """Every candidate attribute usually carries a single example. Demanding
    two or three left merging semantically distinct attributes as the cheapest
    way to comply. The field is overridden here rather than inherited from
    discovery: this is the phase that may merge at all."""
    described = SettledAttribute.model_fields["example_observations"].description
    assert "1-3 observations carried over" in described
    assert "NEVER merge attributes that mean" in described
    assert "2-3 observations" not in described


def test_the_attribute_prompt_ends_on_the_universal_rules_and_the_hint():
    prompt = build_attribute_consolidation_prompt(**_attribute_kwargs())
    assert prompt.rstrip().endswith(INSTRUCTOR_HINT)
    assert "DESCRIPTIVE, NEVER EVALUATIVE." in prompt


# =============================================================================
# WHAT THE ATTRIBUTE SPLIT RULED
#
# Same technique as the facet section above: every ruling names the wording that
# went and the wording that replaced it, so a mistyped literal cannot make the
# negative half pass on its own.
#
# Note the polarity is the reverse of the facet section on one point: this call
# DOES render definitions and examples per candidate, so wording about examples
# belongs here and was only wrong one level up. That pair still has its twin —
# the facet prompt is the one that must not mention them.
# =============================================================================

def test_ruling_the_attribute_prompt_runs_no_placement_check():
    """The old step 6 could move an attribute to another facet or drop it. One
    facet in view can do neither, so the whole clause goes — and what replaced
    it is the flat prohibition on losing anything."""
    attribute = build_attribute_consolidation_prompt(**_attribute_kwargs())
    assert "or drop it if no facet fits" not in attribute
    assert "never drop" in attribute.lower()


def test_ruling_the_attribute_prompt_checks_no_domain_boundary():
    """A facet-scoped call has no neighbouring domains in view: it is not given
    the exclusions, and the facet's domain was settled two phases ago. What it
    is given instead is the one facet it works inside."""
    attribute = build_attribute_consolidation_prompt(**_attribute_kwargs())
    assert "and not to a neighbouring domain" not in attribute
    assert "prijs en kosten" not in attribute
    assert "Hoe snel ging het?" in attribute


def test_ruling_the_merge_test_runs_on_attributes_not_on_items():
    """`Any two items` was right where the prompt merged at two levels. Here
    there is one level, and naming it is what makes the test applicable."""
    attribute = build_attribute_consolidation_prompt(**_attribute_kwargs())
    assert "run it on any two attributes" in attribute
    assert "run it on any two items" not in attribute


def test_ruling_the_merge_test_closes_on_the_levels_this_call_has():
    """`Not of items` is vague for the same reason. The counts this call can be
    tempted to chase are attributes and the examples carried under them."""
    attribute = build_attribute_consolidation_prompt(**_attribute_kwargs())
    assert "not of attributes, not of examples." in attribute
    assert "not of items, not of examples." not in attribute


def test_ruling_this_call_counts_attributes_and_never_items():
    """`Items` was the combined predecessor's word for two levels at once. This
    call holds one, so anything it counts is an attribute. The explicit ranking
    that used to carry the ruling is gone; the merge test still carries it."""
    attribute = build_attribute_consolidation_prompt(**_attribute_kwargs())
    assert "not of attributes, not of examples" in attribute
    assert "Fewest items" not in attribute


def test_ruling_the_prompt_never_asks_for_a_facet_name():
    """This call returns no facets, so naming one is output it cannot deliver —
    the mirror image of the ruling on the facet prompt."""
    attribute = build_attribute_consolidation_prompt(**_attribute_kwargs())
    assert "Name every surviving facet" not in attribute


def test_ruling_the_call_may_not_touch_its_facet():
    """The old wording let the call collapse a facet into its single attribute.
    The facet is settled before this call and is not in its output model. The
    lone-attribute note was dropped on 2026-08-16; the ruling it enforced is a
    negative and stands without it."""
    attribute = build_attribute_consolidation_prompt(**_attribute_kwargs())
    assert "Collapse it only when you can say plainly" not in attribute
    assert "facet_name" not in AttributeConsolidationResult.model_fields
    assert "facet_name" not in SettledAttribute.model_fields


def test_ruling_the_attribute_prompt_may_speak_of_examples():
    """The reverse of the facet ruling, and the reason these sections cannot
    share assertions: the attribute candidate block renders up to three examples
    per candidate, so item 4 of the merge test asks about material in view."""
    clause = "does every example still have"
    attribute = build_attribute_consolidation_prompt(**_attribute_kwargs())
    assert clause in attribute
    assert clause not in build_facet_consolidation_prompt(**_facet_kwargs())
    assert "e.g. \"lang wachten\"" in _attribute_kwargs()["candidate_block"]


def test_ruling_the_attribute_prompt_points_at_no_step_of_another_prompt():
    """Rule 2 of the combined prompt ended on a pointer into its own step 6.
    Copied here it would name a step this prompt does not have."""
    attribute = build_attribute_consolidation_prompt(**_attribute_kwargs())
    assert "The same reasoning governs the attributes in step 6" not in attribute
    assert "Step 4 — Let prevalence set the granularity" in attribute
