"""Tests for the consolidation prompts of step 4.

Covers both prompts in the module: the facet call, which settles which facets a
domain has, and the attribute call, which settles the pool inside one of them.

WHAT IS TESTED HERE, AND WHAT IS NOT
------------------------------------
Only coherence between two artefacts. Does the block render what the prompt
tells the model to use? Does the prompt state the rule that governs a field the
schema demands? Is a builder still callable with the arguments it is given?

That is the one failure class this step keeps producing: a prompt that points at
material which is not in front of the model. It has now happened four times —
`build_context_block` losing seven callers, the facet-consolidation prompt asking
for attributes while returning facets, facet_settle clustering response texts that
had been switched off, and both candidate blocks dropping the prevalence its own
rules lean on. None of it raises; the run finishes and the answer is worse.

Assertions on the wording of a prompt are deliberately NOT here. They fail when
someone rewrites a sentence, which is a normal and wanted event, and the red they
leave behind is what made a real break invisible: 33 tests had been failing for
long enough that "the suite is red" stopped carrying information. What a prompt
says belongs in git history and in the step's dev docs, not in a string compare.

The same reasoning retired the two `WHAT THE SPLIT RULED` sections. They guarded
the 2026-08-15 split against leftovers of the prompt it replaced. That migration
is finished.
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
# THE FACET PHASE — the block
# =============================================================================

def test_facet_ids_are_positional_and_stable():
    """Provenance runs on these ids, and names are not unique within a domain."""
    index = build_facet_candidate_index(POOLS)
    assert list(index) == ["F1", "F2", "F3"]
    assert index["F2"].facet_name == "Snelheid van afhandeling"


def test_the_facet_block_shows_prevalence_per_facet():
    """The prompt has a rule that uses prevalence to set granularity. If the
    block stops rendering it, that rule asks about material the model cannot
    see — and nothing raises.

    The line used to read `proposed under this exact name in …`. The block no
    longer renders facet names — only the definition, so that the judgement runs
    on the concept — and a sentence pointing at a name the model cannot see is
    the very incoherence this test exists to catch.
    """
    block = build_facet_candidate_block(POOLS, RECURRENCE, 6)
    assert "Proposed in 4 of 6 independent passes" in block
    assert "Proposed in 1 of 6 independent passes" in block


def test_a_facet_without_a_count_falls_back_to_one_pass():
    """A survivor of round one carries no recurrence of its own, and a facet
    rendered without a count would read as one the passes never proposed."""
    block = build_facet_candidate_block([_pool("Nieuw", "A")], {}, 6)
    assert "Proposed in 1 of 6 independent passes" in block


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
    """Round one comes from discovery and has no question; round two candidates
    do, and the block must show it or rule 1 has nothing to test."""
    block = build_facet_candidate_block(
        [_pool("Snelheid", "Wachttijd", question="Hoe snel ging het?")],
        RECURRENCE, 6)
    assert "Hoe snel ging het?" in block


def test_a_facet_without_attributes_does_not_break_the_block():
    """Not hypothetical: a survivor claimed by a second survivor hands its pool
    to the first and re-enters round two holding nothing."""
    block = build_facet_candidate_block([_pool("Leeg")], {"Leeg": 2}, 3)
    assert "Leeg" in block


# =============================================================================
# THE FACET PHASE — the schema, and what the prompt owes it
# =============================================================================

def test_the_facet_result_is_a_decision_summary_plus_facets():
    assert set(FacetConsolidationResult.model_fields) == {
        "decision_summary", "facets"}


def test_a_consolidated_facet_carries_no_attributes():
    """The whole point of the split: this call decides the inventory, not what
    hangs under it."""
    assert "attributes" not in SettledFacet.model_fields


def test_the_facet_model_carries_no_attribute_field():
    """This call settles the inventory and hands the pools on untouched. An
    attribute field would let it settle both levels again."""
    assert "source_attribute_ids" not in SettledFacet.model_fields


def test_a_consolidated_facet_states_its_question_and_its_sources():
    assert "facet_question" in SettledFacet.model_fields
    assert "source_facet_ids" in SettledFacet.model_fields


def test_the_facet_question_may_not_be_stated_twice():
    """The classifier logs `duplicate_facet_question` when two survivors state
    the same one, so the rule that forbids it has to be somewhere the model
    reads. Without a stated question, rule 1 — do these two answer the same
    question — is a matter of feel and not checkable."""
    assert ("No two surviving facets may state the same one"
            in SettledFacet.model_fields["facet_question"].description)


def test_the_facet_prompt_asks_coverage_on_ids():
    """`source_facet_ids` is the safety net that tells a merged candidate from a
    forgotten one. The rule that every id must be accounted for governs a field,
    so it belongs in the prompt — and coverage is checked on the ids, never on
    the names, which are not unique."""
    prompt = build_facet_consolidation_prompt(**_facet_kwargs())
    assert "never on names" in prompt
    assert "at least one surviving" in prompt


def test_the_facet_model_describes_every_field_it_has():
    assert_every_field_is_described(FacetConsolidationResult)


def test_the_facet_prompt_does_not_restate_the_schema():
    """One place for the output contract, and that is the response model:
    instructor renders the schema with its descriptions into the call already.
    Two copies drifted apart once — the candidate block showed one example
    while the output spec demanded 2-3."""
    assert_prompt_does_not_restate_the_schema(
        build_facet_consolidation_prompt(**_facet_kwargs()))


def test_the_facet_prompt_ends_on_the_universal_rules_and_the_hint():
    """Without the instructor sentence, 23 of 56 tasks fail outright."""
    prompt = build_facet_consolidation_prompt(**_facet_kwargs())
    assert prompt.rstrip().endswith(INSTRUCTOR_HINT)
    assert "DESCRIPTIVE, NEVER EVALUATIVE" in prompt


def test_a_facet_prompt_without_exclusions_stays_valid():
    """Step 3 does not always name neighbouring domains."""
    prompt = build_facet_consolidation_prompt(**_facet_kwargs(domain_exclusions=[]))
    assert prompt.rstrip().endswith(INSTRUCTOR_HINT)


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
        dimension_name=DIM.key,
        dimension_description=DIM.dimension_description,
        facet_name="Snelheid",
        facet_definition="Wat Snelheid vastlegt.",
        facet_question="Hoe snel ging het?",
        candidate_block=build_attribute_candidate_block(
            POOL_ATTRIBUTES, POOL_RECURRENCE, 6),
    )
    base.update(overrides)
    return base


# =============================================================================
# THE ATTRIBUTE PHASE — the block
# =============================================================================

def test_attribute_ids_are_flat_within_one_facet():
    """One call is one facet, so an id needs no facet part."""
    index = build_attribute_candidate_index(POOL_ATTRIBUTES)
    assert list(index) == ["A1", "A2"]


def test_the_attribute_block_shows_prevalence():
    """Same coherence as one level up: the prompt lets prevalence set the
    granularity, so the block has to carry it."""
    block = build_attribute_candidate_block(POOL_ATTRIBUTES, POOL_RECURRENCE, 6)
    assert "Wachttijd [5/6 passes]" in block
    assert "Doorlooptijd [1/6 passes]" in block


def test_an_attribute_without_a_count_falls_back_to_one_pass():
    block = build_attribute_candidate_block(POOL_ATTRIBUTES, {}, 6)
    assert "Wachttijd [1/6 passes]" in block


def test_the_attribute_block_shows_definition_and_examples():
    """This call is the one that decides whether two attributes mean the same
    thing, so it needs them in full — which is exactly why the facet block one
    level up may show names only."""
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


# =============================================================================
# THE ATTRIBUTE PHASE — the schema, and what the prompt owes it
# =============================================================================

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
        example_observations=["lang wachten"], source_attribute_ids=["A1"],
        boundary_rules=[])
    assert item.attribute_name == "Wachttijd"


def test_examples_are_one_to_three_and_never_a_reason_to_merge():
    """Every candidate attribute usually carries a single example. Demanding
    two or three left merging semantically distinct attributes as the cheapest
    way to comply. The field is overridden here rather than inherited from
    discovery: this is the phase that may merge at all."""
    described = SettledAttribute.model_fields["example_observations"].description
    assert "1-3 observations carried over" in described
    assert "NEVER merge attributes that mean" in described
    assert "2-3 observations" not in described


def test_the_attribute_prompt_states_the_facet_it_works_inside():
    """A facet-scoped call has no domain and no neighbours in view; the one
    facet it works inside is the whole of its scope."""
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


def test_the_attribute_prompt_forbids_dropping():
    """One facet in view cannot judge where something else belongs. Merging is
    allowed here; losing is not — and losing is invisible in the answer, which
    is why the prohibition has to be stated."""
    prompt = build_attribute_consolidation_prompt(**_attribute_kwargs())
    assert "never drop" in prompt.lower()


def test_the_attribute_prompt_asks_coverage_on_ids():
    """The mirror of the facet coverage rule, on `source_attribute_ids`."""
    prompt = build_attribute_consolidation_prompt(**_attribute_kwargs())
    assert "never on names" in prompt
    assert "at least one surviving" in prompt


def test_the_attribute_model_describes_every_field_it_has():
    assert_every_field_is_described(AttributeConsolidationResult)


def test_the_attribute_prompt_does_not_restate_the_schema():
    assert_prompt_does_not_restate_the_schema(
        build_attribute_consolidation_prompt(**_attribute_kwargs()))


def test_the_attribute_prompt_ends_on_the_universal_rules_and_the_hint():
    prompt = build_attribute_consolidation_prompt(**_attribute_kwargs())
    assert prompt.rstrip().endswith(INSTRUCTOR_HINT)
    assert "DESCRIPTIVE, NEVER EVALUATIVE." in prompt
