"""Tests voor de chunk-consolidatieprompt (step 4)."""
from pipeline.step_3_ideaExtractor.dimension_data import get_dimensions_in_decision_order
from pipeline.step_4_classifier.prompts_discovery import (
    ConsolidatedAttribute,
    ConsolidatedFacet,
    ConsolidationResult,
    DiscoveredAttribute,
    DiscoveredFacet,
    build_candidate_block,
    build_candidate_index,
    build_chunk_consolidation_prompt,
)
from pipeline.step_4_classifier.prompts_shared import INSTRUCTOR_HINT

DIM = get_dimensions_in_decision_order()[0]


def _facet(name, *attr_names):
    return DiscoveredFacet(
        facet_name=name,
        facet_definition=f"Wat {name} vastlegt.",
        attributes=[DiscoveredAttribute(
            attribute_name=a,
            attribute_definition=f"De eigenschap {a}.",
            example_observations=[f"observatie over {a}"],
        ) for a in attr_names],
    )


CANDIDATES = [
    _facet("Snelheid", "Wachttijd", "Doorlooptijd"),
    _facet("Snelheid van afhandeling", "Wachttijd"),
    _facet("Bejegening", "Vriendelijkheid"),
]
RECURRENCE = {"Snelheid": 4, "Snelheid van afhandeling": 1, "Bejegening": 5}
ATTR_RECURRENCE = {"Wachttijd": 5, "Doorlooptijd": 1, "Vriendelijkheid": 3}


def _kwargs(**overrides):
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
        candidate_block=build_candidate_block(
            CANDIDATES, RECURRENCE, ATTR_RECURRENCE, 6),
    )
    base.update(overrides)
    return base


# =============================================================================
# HET KANDIDATENBLOK
# =============================================================================

def test_the_candidate_block_shows_chunk_prevalence_per_facet():
    block = build_candidate_block(CANDIDATES, RECURRENCE, ATTR_RECURRENCE, 6)
    assert "proposed under this exact name in 4 of 6 independent passes" in block
    assert "proposed under this exact name in 1 of 6 independent passes" in block


def test_the_count_says_it_is_per_exact_name():
    """A concept five passes proposed under five wordings arrives as five
    candidates of one pass each — exactly the case this phase resolves. A label
    promising support for the concept would mislead on the wrong candidates."""
    block = build_candidate_block(CANDIDATES, RECURRENCE, ATTR_RECURRENCE, 6)
    assert "under this exact name" in block


def test_the_candidate_block_shows_prevalence_per_attribute_too():
    """Step 6 asks which attributes are well supported; until this count existed
    that judgement had no data behind it."""
    block = build_candidate_block(CANDIDATES, RECURRENCE, ATTR_RECURRENCE, 6)
    assert "Wachttijd [5/6 passes]" in block
    assert "Doorlooptijd [1/6 passes]" in block


def test_the_candidate_block_shows_the_attributes_nested():
    block = build_candidate_block(CANDIDATES, RECURRENCE, ATTR_RECURRENCE, 6)
    assert "Wachttijd" in block
    assert "Doorlooptijd" in block
    assert block.index("Snelheid") < block.index("Wachttijd")


def test_a_candidate_without_a_count_falls_back_to_one_pass():
    block = build_candidate_block([_facet("Nieuw", "A")], {}, {}, 6)
    assert "proposed under this exact name in 1 of 6 independent passes" in block
    assert "A [1/6 passes]" in block


def test_a_candidate_without_attributes_does_not_break_the_block():
    block = build_candidate_block([_facet("Leeg")], {"Leeg": 2}, {}, 3)
    assert "Leeg" in block


# =============================================================================
# PROMPT ↔ MODEL SLUITEN AAN
# =============================================================================

def test_the_result_is_a_decision_summary_plus_facets():
    """A free-form scratchpad on a phase that already reasons internally gives
    long, uneven output in which the result is the smaller part."""
    assert set(ConsolidationResult.model_fields) == {
        "decision_summary", "facets"}


def test_a_consolidated_facet_states_what_folded_into_it():
    """Without this field a merged candidate looks identical to a forgotten one:
    neither appears in the answer."""
    assert "source_facet_ids" in ConsolidatedFacet.model_fields


def test_a_consolidated_attribute_says_the_same_one_level_down():
    assert "source_attribute_ids" in ConsolidatedAttribute.model_fields


def test_a_consolidated_facet_carries_consolidated_attributes():
    item = ConsolidatedFacet(
        facet_name="Snelheid", facet_definition="…", source_facet_ids=["F1"],
        facet_question="Hoe snel gaat het?",
        attributes=[ConsolidatedAttribute(
            attribute_name="Wachttijd", attribute_definition="…",
            example_observations=["x"], source_attribute_ids=["F1-A1"])])
    assert item.attributes[0].source_attribute_ids == ["F1-A1"]


def test_prompt_explains_that_coverage_is_checked():
    prompt = build_chunk_consolidation_prompt(**_kwargs())
    assert "source_facet_ids" in prompt
    assert "source_attribute_ids" in prompt
    assert "at least one surviving" in prompt


def test_a_divided_candidate_may_be_claimed_by_several_survivors():
    """Step 6 lets an attribute move to the facet where it belongs, so a
    candidate facet whose contents divide cannot honestly be pinned to one
    survivor. `exactly one` made the model claim an absorption that never
    happened."""
    prompt = build_chunk_consolidation_prompt(**_kwargs())
    assert "exactly one surviving" not in prompt
    assert "listed by every survivor that took part" in prompt


def test_examples_are_one_to_three_and_never_a_reason_to_merge():
    """Every candidate attribute usually carries a single example. Demanding
    two or three left merging semantically distinct attributes as the cheapest
    way to comply."""
    prompt = build_chunk_consolidation_prompt(**_kwargs())
    assert "1-3 observations carried over" in prompt
    assert "NEVER merge attributes that mean" in prompt
    assert "2-3 observations" not in prompt


def test_the_candidate_block_shows_more_than_one_example():
    """Consolidation can only carry over what it is shown."""
    rich = DiscoveredFacet(
        facet_name="Snelheid", facet_definition="…",
        attributes=[DiscoveredAttribute(
            attribute_name="Wachttijd", attribute_definition="…",
            example_observations=["lang wachten", "traag", "duurt eeuwen"])])
    block = build_candidate_block([rich], {}, {}, 4)
    for text in ("lang wachten", "traag", "duurt eeuwen"):
        assert f'e.g. "{text}"' in block


def test_prompt_names_every_field_the_model_knows():
    prompt = build_chunk_consolidation_prompt(**_kwargs())
    for veld in ("decision_summary", "facets", "facet_name", "facet_definition",
                 "facet_question", "source_facet_ids", "attributes",
                 "attribute_name", "attribute_definition",
                 "example_observations", "source_attribute_ids"):
        assert veld in prompt, veld


def test_prompt_eindigt_op_de_instructor_zin():
    assert build_chunk_consolidation_prompt(
        **_kwargs()).rstrip().endswith(INSTRUCTOR_HINT)


# =============================================================================
# DE VIER REGELS EN HUN PRECEDENTIE
# =============================================================================

def test_the_prompt_carries_the_four_grouping_rules():
    prompt = build_chunk_consolidation_prompt(**_kwargs())
    for regel in ("UNDERLYING QUESTION FIRST", "PREVALENCE SETS GRANULARITY",
                  "LIFT, DON'T FLATTEN", "PLAIN, MEANINGFUL LABELS"):
        assert regel in prompt, regel


def test_the_prompt_states_the_precedence_explicitly():
    """Without a precedence the rules fight; with one, orthogonality wins."""
    prompt = build_chunk_consolidation_prompt(**_kwargs())
    assert "When these conflict, decide in this order" in prompt


def test_the_precedence_covers_every_rule():
    """It read `1 > 2 > 4` while four numbered rules stood above it, so LIFT,
    DON'T FLATTEN had no place in the ordering at all."""
    prompt = build_chunk_consolidation_prompt(**_kwargs())
    order = prompt[prompt.index("When these conflict"):
                   prompt.index("# Step-by-Step")]
    for rule in ("Orthogonality", "Prevalence", "Lifting", "Label clarity"):
        assert rule in order, rule


def test_minimisation_comes_last_in_the_ordering():
    """This is the phase licensed to merge, so the brake belongs here: a smaller
    inventory that has lost a distinction is not a better one."""
    prompt = build_chunk_consolidation_prompt(**_kwargs())
    order = prompt[prompt.index("When these conflict"):
                   prompt.index("# Step-by-Step")]
    assert order.index("Orthogonality") < order.index("Fewest items")
    assert "Never merge distinct concepts" in order


def test_the_facet_question_must_be_written_down():
    """Rule 1 asks whether two candidates answer the same question. Unless the
    question is stated, that test is a matter of feel and not checkable."""
    prompt = build_chunk_consolidation_prompt(**_kwargs())
    assert "facet_question" in prompt
    assert "No two surviving facets may state the same one." in prompt


def test_labels_are_not_asked_to_avoid_nominalisations():
    """Nearly every usable Dutch or German taxonomy label is a nominalisation,
    so the rule could not be satisfied in the languages this runs on."""
    prompt = build_chunk_consolidation_prompt(**_kwargs())
    assert "nominalisation" not in prompt.lower()
    assert "ordinary noun phrase" in prompt


def test_rule_one_does_not_call_opposite_answers_a_reason_to_split():
    """`Mutually exclusive` and `opposite` are not the same thing, and the
    universal rules forbid splitting one concept by evaluative direction — in
    this same prompt."""
    prompt = build_chunk_consolidation_prompt(**_kwargs())
    assert "opposite answers" not in prompt
    assert "erase what" in prompt


def test_rule_one_does_not_use_the_word_dimension():
    """L1 is called Dimension in the taxonomy block. If rule 1 also uses the word
    for 'the axis along which a concept varies', it means two things on one page
    — which is exactly what the lens naming was once meant to solve."""
    prompt = build_chunk_consolidation_prompt(**_kwargs())
    regels = prompt[prompt.index("# Consolidation Rules"):
                    prompt.index("# Step-by-Step")]
    assert "dimension" not in regels.lower()
    assert "underlying question" in regels.lower()


# =============================================================================
# DE GENESTE STAP
# =============================================================================

def test_prompt_says_what_happens_to_attributes_on_a_facet_merge():
    prompt = build_chunk_consolidation_prompt(**_kwargs())
    assert "pool" in prompt.lower()


def test_prompt_also_consolidates_within_a_facet():
    prompt = build_chunk_consolidation_prompt(**_kwargs())
    stappen = prompt[prompt.index("# Step-by-Step"):]
    assert "attribute" in stappen.lower()


# =============================================================================
# WAT ER NIET IN MAG
# =============================================================================

def test_prompt_bevat_geen_drempelgetallen():
    prompt = build_chunk_consolidation_prompt(**_kwargs())
    assert "%" not in prompt


def test_prompt_kent_geen_lens():
    assert "Lens" not in build_chunk_consolidation_prompt(**_kwargs())


def test_the_prompt_carries_the_universal_rules():
    assert "<universal_rules>" in build_chunk_consolidation_prompt(**_kwargs())


def test_a_prompt_without_exclusions_stays_valid():
    prompt = build_chunk_consolidation_prompt(**_kwargs(domain_exclusions=[]))
    assert prompt.rstrip().endswith(INSTRUCTOR_HINT)


# =============================================================================
# WAT DE REVIEW VAN 2026-08-15 OPLOSTE
# =============================================================================

def test_the_candidate_block_hands_out_ids():
    """Names are not unique: the same attribute name can sit under two candidate
    facets, and a list of two identical strings is something a JSON layer may
    quietly collapse to one."""
    block = build_candidate_block(CANDIDATES, RECURRENCE, ATTR_RECURRENCE, 6)
    assert "[F1] Snelheid" in block
    assert "[F1-A1] Wachttijd" in block
    assert "[F3-A1] Vriendelijkheid" in block


def test_the_index_is_stable_for_a_given_task():
    facets, attributes = build_candidate_index(CANDIDATES)
    assert list(facets) == ["F1", "F2", "F3"]
    assert attributes["F1-A2"][0] == "F1"
    assert attributes["F1-A2"][1].attribute_name == "Doorlooptijd"


def test_coverage_is_checked_on_ids_not_names():
    prompt = build_chunk_consolidation_prompt(**_kwargs())
    assert "never on names" in prompt


def test_prevalence_is_not_an_absolute_veto():
    """Two concepts can both be well supported and still be one concept said
    twice; `never dissolve` made support outrank deduplication."""
    prompt = build_chunk_consolidation_prompt(**_kwargs())
    assert "unless it demonstrably" in prompt
    assert "never a reason to keep a duplicate" in prompt


def test_the_facet_question_is_tested_against_being_a_subject():
    """Sorting by what the material is ABOUT belongs one level up, and produces
    facets that overlap wherever a response touches two topics."""
    prompt = build_chunk_consolidation_prompt(**_kwargs())
    assert "answered by naming a subject" in prompt


def test_a_merge_test_is_given():
    """`Minimal` on its own is not measurable, and the phase licensed to merge
    needs a test it can apply."""
    prompt = build_chunk_consolidation_prompt(**_kwargs())
    assert "MERGE TEST" in prompt
    assert "Would the same observation be coded under both?" in prompt


def test_nesting_between_sibling_attributes_is_forbidden():
    """A general item and a specific one inside it let the same response be
    coded twice, honestly."""
    prompt = build_chunk_consolidation_prompt(**_kwargs())
    assert "subtype, a component or a concrete instance" in prompt


def test_a_lone_attribute_is_a_warning_not_a_verdict():
    """A real lens can hold one attribute in this material and several in the
    next batch."""
    prompt = build_chunk_consolidation_prompt(**_kwargs())
    assert "is a WARNING, not a verdict" in prompt


def test_the_definitions_are_not_rendered_twice():
    """They already stand as L3 and L4 in the taxonomy block; the same sentence
    twice within a few hundred words is not an anchor."""
    prompt = build_chunk_consolidation_prompt(**_kwargs())
    assert "# What a facet is" not in prompt
    assert prompt.count("L3 — Facet:") == 1
