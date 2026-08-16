"""Tests voor naslijpen en cross-domein (step 4)."""
from pipeline.step_3_ideaExtractor.dimension_data import get_dimensions_in_decision_order
from pipeline.step_4_classifier.drains import (
    make_drain_attribute, make_drain_facet,
)
from pipeline.step_4_classifier.prompts_refinement import (
    RefinedAttribute,
    RefinementMisfitGroup,
    RefinementResult,
    build_contents_block,
    build_cross_domain_prompt,
    build_move_targets_block,
    build_refinement_prompt,
)
from pipeline.step_4_classifier.prompts_shared import (
    INSTRUCTOR_HINT, build_cross_scope_model,
)
from pipeline.step_4_classifier.test_prompts_shared import (
    assert_every_field_is_described, assert_prompt_does_not_restate_the_schema,
)

DIM = get_dimensions_in_decision_order()[0]

ATTRIBUTES = [
    {"attribute_name": "Wachttijd", "attribute_definition": "De tijd tot antwoord."},
    make_drain_attribute("Snelheid", "Dutch"),
]
# One domain, three facets: the one under judgement, a neighbour that may serve
# as a move target, and a catch-all that may not.
FACETS = [
    {"facet_name": "Snelheid",
     "facet_definition": "Hoe snel er geleverd wordt.",
     "facet_question": "Hoe snel wordt er geleverd?",
     "attributes": ATTRIBUTES},
    {"facet_name": "Bejegening",
     "facet_definition": "Hoe klanten worden aangesproken.",
     "attributes": [{"attribute_name": "Vriendelijkheid",
                     "attribute_definition": "De toon in het contact."}]},
    make_drain_facet("dienstverlening", "Dutch"),
]
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
        facet_name="Snelheid",
        facet_definition="Hoe snel er geleverd wordt.",
        facet_question="Hoe snel wordt er geleverd?",
        contents_block=build_contents_block(ATTRIBUTES, CONTENTS, SHARES, COUNTS, 5),
        move_targets_block=build_move_targets_block(FACETS, 0),
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
    block = build_contents_block(ATTRIBUTES, CONTENTS, SHARES, COUNTS, 5)
    assert "Claims to capture" in block
    assert "Actually holds" in block
    assert block.index("Claims to capture") < block.index("Actually holds")


def test_inhoudsblok_toont_aandeel_relatief():
    """Het aandeel is het aandeel in het domein, breder dan de call zelf: binnen
    het facet gemeten zou elk facet even zwaar lijken, en zet prevalentie de
    granulariteit op een schaal die per call verschuift."""
    block = build_contents_block(ATTRIBUTES, CONTENTS, SHARES, COUNTS, 5)
    assert "124 responses" in block
    assert "62% of this domain" in block


def test_inhoudsblok_kapt_op_top_n():
    block = build_contents_block(ATTRIBUTES, CONTENTS, SHARES, COUNTS, 2)
    assert "snel geholpen" not in block


def test_an_empty_attribute_is_named_as_such():
    block = build_contents_block(ATTRIBUTES, {}, {}, {}, 5)
    assert "nothing was assigned to it" in block


def test_inhoudsblok_draagt_geen_facetkop_meer():
    """Het facet is de scope van de hele call en staat één keer in de context;
    een kop erboven zou hem een tweede keer neerzetten."""
    block = build_contents_block(ATTRIBUTES, CONTENTS, SHARES, COUNTS, 5)
    assert "Facet:" not in block


# =============================================================================
# HET VERHUISROOSTER
# =============================================================================

def test_verhuisrooster_toont_de_buren():
    block = build_move_targets_block(FACETS, 0)
    assert "Bejegening" in block
    assert "Vriendelijkheid" in block


def test_verhuisrooster_laat_het_eigen_facet_weg():
    """Op positie, niet op naam: een domein mag twee facetten met dezelfde naam
    dragen, en dan zou uitsluiten op naam ook de buurman verbergen."""
    block = build_move_targets_block(FACETS, 0)
    assert "Wachttijd" not in block


def test_verhuisrooster_laat_de_vangnetten_weg():
    block = build_move_targets_block(FACETS, 0)
    assert "Overig" not in block


def test_verhuisrooster_draagt_geen_aantallen():
    """Een bestemmingslijst, geen materiaal om te beoordelen: met aantallen
    erbij gaat het model deze attributen mee wegen, en dat is de call van een
    ander facet."""
    block = build_move_targets_block(FACETS, 0)
    assert "responses" not in block
    assert "%" not in block


def test_verhuisrooster_zegt_het_als_er_geen_buren_zijn():
    block = build_move_targets_block(FACETS[:1], 0)
    assert "no other facet" in block


# =============================================================================
# DE SPLITSCLAUSULE
# =============================================================================

def test_the_old_split_clause_is_gone():
    """'a large share AND visibly diverse contents is too abstract: split it'
    vuurde voortdurend zodra discovery items bewust breed maakte. De regel is
    nooit teruggekomen en mag dat ook niet."""
    prompt = build_refinement_prompt(**_kwargs())
    assert "too abstract" not in prompt


# =============================================================================
# DE UITGANGEN
# =============================================================================

def test_a_misfit_group_moves_or_goes_out():
    assert str(RefinementMisfitGroup.model_fields["verdict"].annotation) == (
        "typing.Literal['move', 'out']")


def test_a_refined_attribute_states_no_action():
    """Wat er gebeurde is af te lezen uit `source_attributes`: meer dan één bron
    is een merge, één bron onder een andere naam een hernoeming. Het model laten
    classificeren wat de code al ziet, leverde labels op die hun regels
    overleefden."""
    assert "action" not in RefinedAttribute.model_fields


def test_a_refined_attribute_routes_no_texts():
    """`instance_texts` bestond alleen voor `split`, en die uitgang is weg. Op
    de misfit blijft het veld staan: daar dráágt het de verhuizing."""
    assert "instance_texts" not in RefinedAttribute.model_fields
    assert "instance_texts" in RefinementMisfitGroup.model_fields


def test_source_attributes_states_what_leaving_one_out_costs():
    """De dekkingseis stond in regel 7 van het oude regelblok. Nu de regels weg
    zijn is het veld de enige plek waar hij nog kan staan — en de juiste, want
    het is het uitvoercontract."""
    omschrijving = " ".join(
        RefinedAttribute.model_fields["source_attributes"].description.split())
    assert "exactly once" in omschrijving
    assert "it stays where it was" in omschrijving


def test_resultaat_draagt_attributen_en_misfits():
    assert set(RefinementResult.model_fields) == {
        "scratchpad", "attributes", "misfits"}


def test_the_scratchpad_walks_the_rules_in_order():
    """De scratchpad is er voor regeldiscipline, dus hij loopt de regels van de
    prompt af in hun eigen volgorde: samenvoegen, MECE, prevalentie, misfits."""
    stappen = RefinementResult.model_fields["scratchpad"].description
    volgorde = ["Minimize the number of containers", "must be MECE",
                "prevalence", "<move_targets>"]
    posities = [stappen.index(s) for s in volgorde]
    assert posities == sorted(posities), volgorde


def test_the_scratchpad_routes_misfits_after_merging():
    """Andersom breekt stil: `_apply_refinement` zoekt een misfitdoel op onder de
    attributen die nog bestaan, en een doel dat het model zelf heeft
    weggemerged valt terug op waar de responses al stonden — zonder logregel."""
    stappen = RefinementResult.model_fields["scratchpad"].description
    assert stappen.index("Merge candidate attributes") < stappen.index(
        "<move_targets>")


def test_the_scratchpad_ends_on_the_coverage_check():
    """Een vergeten input is het enige waarmee deze fase stil een attribuut
    verliest, dus de controle staat er dubbel: hier en op het veld zelf."""
    stappen = RefinementResult.model_fields["scratchpad"].description
    assert "exactly one source_attributes list" in stappen


# =============================================================================
# HET FACET IS DE SCOPE, EN LIGT DAARMEE VAST
# =============================================================================

def test_a_refined_attribute_names_no_facet():
    """Eén call is één facet, dus er is niets om naartoe te verplaatsen. Het
    veld weglaten is de enige manier om dat waar te maken: stond het er nog,
    dan zou het model een facet kunnen noemen dat deze call niet bezit."""
    assert "facet_name" not in RefinedAttribute.model_fields


def test_the_prompt_renders_the_facet_it_judges():
    prompt = build_refinement_prompt(**_kwargs())
    assert "<facet_contents>" in prompt
    assert "Snelheid — Hoe snel er geleverd wordt." in prompt
    assert "Hoe snel wordt er geleverd?" in prompt


def test_a_facet_without_a_question_renders_no_label():
    """Een facet dat zonder call gezet werd draagt geen vraag; een label met
    niets erachter leest als een vraag die het facet niet wist te stellen."""
    prompt = build_refinement_prompt(**_kwargs(facet_question=""))
    assert "The question it answers:" not in prompt


def test_the_prompt_carries_the_domain_as_parent_context():
    """Het facet deelt een domein op, dus de facetvraag betekent alleen iets
    binnen dat domein."""
    prompt = build_refinement_prompt(**_kwargs())
    assert "<taxonomy_domain>" in prompt
    assert "dienstverlening" in prompt


def test_the_prompt_offers_the_neighbours_as_move_targets():
    prompt = build_refinement_prompt(**_kwargs())
    assert "<move_targets>" in prompt
    assert "Vriendelijkheid" in prompt


def test_the_move_targets_are_offered_as_a_destination_not_as_material():
    """Zonder die zin leest het rooster als een tweede stapel om te beoordelen,
    en dat is de call van een ander facet."""
    prompt = build_refinement_prompt(**_kwargs())
    assert "destination list" in prompt
    assert "not material to judge" in prompt


def test_the_rules_make_the_misfit_exit_reachable():
    """Het schema biedt misfits en het rooster staat er; zonder een regel die
    zegt dat ze bestaan, is de uitgang onbereikbaar."""
    regels = _rules(build_refinement_prompt(**_kwargs()))
    assert "misfit" in regels
    assert "<move_targets>" in regels


def test_the_model_describes_every_field_it_has():
    assert_every_field_is_described(RefinementResult)


def test_the_prompt_does_not_restate_the_schema():
    assert_prompt_does_not_restate_the_schema(build_refinement_prompt(**_kwargs()))


# =============================================================================
# VANGNETTEN DOEN NIET MEE
# =============================================================================

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


def _rules(prompt: str) -> str:
    """De eigen regels van deze prompt, zonder de universele eronder."""
    return prompt[prompt.index("\nRules\n"):prompt.index("<universal_rules>")]


def test_rules_do_not_use_the_word_dimension():
    assert "dimension" not in _rules(build_refinement_prompt(**_kwargs())).lower()


def test_no_threshold_numbers_in_the_rules():
    """Aandelen komen uit de data en mogen; een vast percentage is van één
    dataset afgelezen en mag niet."""
    assert "%" not in _rules(build_refinement_prompt(**_kwargs()))


def test_catch_alls_are_marked_not_recognised_by_name():
    """The name is translated and rewritable; the marker comes from drain_key."""
    block = build_contents_block(ATTRIBUTES, CONTENTS, SHARES, COUNTS, 5)
    gemarkeerd = [r for r in block.splitlines() if "[CATCH-ALL]" in r]
    assert len(gemarkeerd) == 1
    assert "Overig" in gemarkeerd[0]
    assert "Wachttijd" not in gemarkeerd[0]
