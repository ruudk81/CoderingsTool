"""Tests voor naslijpen en cross-domein (step 4)."""
import re

from pipeline.step_3_ideaExtractor.dimension_data import get_dimensions_in_decision_order
from pipeline.step_4_classifier.drains import make_drain_attribute
from pipeline.step_4_classifier.prompts_refinement import (
    RefinedAttribute,
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

ATTRIBUTES = [
    {"attribute_name": "Wachttijd", "attribute_definition": "De tijd tot antwoord."},
    make_drain_attribute("Snelheid", "Dutch"),
]
CONTENTS = {"Wachttijd": ["lange wachttijd", "duurt lang", "snel geholpen"]}
SHARES = {"Wachttijd": 0.62}
COUNTS = {"Wachttijd": 124}
FACET_TOTAL = 124
DOMAIN_TOTAL = 200


def _contents(attributes=ATTRIBUTES, contents=CONTENTS, shares=SHARES,
              counts=COUNTS, top_n=5, facet_total=FACET_TOTAL,
              domain_total=DOMAIN_TOTAL):
    return build_contents_block(attributes, contents, shares, counts, top_n,
                                facet_total, domain_total)


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
        contents_block=_contents(),
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
    block = _contents()
    assert "Claims to capture" in block
    assert "Actually holds" in block
    assert block.index("Claims to capture") < block.index("Actually holds")


def test_inhoudsblok_toont_aandeel_relatief():
    """Het aandeel is het aandeel in het domein, breder dan de call zelf: binnen
    het facet gemeten zou elk facet even zwaar lijken, en zet prevalentie de
    granulariteit op een schaal die per call verschuift."""
    block = _contents()
    assert "124 responses" in block
    assert "62% of this domain" in block


def test_de_kop_zet_het_facet_naast_zijn_domein():
    """Zonder dat tweede getal heeft "klein" geen ijkpunt binnen de call: een
    facet met een twintigste van zijn domein toont àl zijn attributen als lage
    percentages, en een prevalentieregel leest dan als 'voeg alles samen'."""
    block = _contents(facet_total=124, domain_total=200)
    kop = block.splitlines()[0]
    assert "124 of the 200 responses" in kop
    assert "62%" in kop


def test_de_kop_zegt_welke_noemer_de_percentages_gebruiken():
    assert "share of the DOMAIN, not of this facet" in _contents()


def test_een_leeg_domein_maakt_de_kop_niet_stuk():
    assert "0 of the 0 responses" in _contents(facet_total=0, domain_total=0)


def test_inhoudsblok_kapt_op_top_n():
    assert "snel geholpen" not in _contents(top_n=2)


def test_an_empty_attribute_is_named_as_such():
    block = _contents(contents={}, shares={}, counts={})
    assert "nothing was assigned to it" in block


def test_inhoudsblok_draagt_geen_facetkop_meer():
    """Het facet is de scope van de hele call en staat één keer in de context;
    een kop erboven zou hem een tweede keer neerzetten."""
    assert "Facet:" not in _contents()


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

def test_a_refined_attribute_states_no_action():
    """Wat er gebeurde is af te lezen uit `source_attributes`: meer dan één bron
    is een merge, één bron onder een andere naam een hernoeming. Het model laten
    classificeren wat de code al ziet, leverde labels op die hun regels
    overleefden."""
    assert "action" not in RefinedAttribute.model_fields


def test_this_phase_routes_no_texts_at_all():
    """`instance_texts` droeg eerst de split, daarna de verhuizing. Beide
    uitgangen zijn weg, dus het veld heeft nergens meer een drager."""
    assert "instance_texts" not in RefinedAttribute.model_fields
    assert "instance_texts" not in str(RefinementResult.model_fields)


def test_source_attributes_states_what_leaving_one_out_costs():
    """De dekkingseis stond in regel 7 van het oude regelblok. Nu de regels weg
    zijn is het veld de enige plek waar hij nog kan staan — en de juiste, want
    het is het uitvoercontract."""
    omschrijving = " ".join(
        RefinedAttribute.model_fields["source_attributes"].description.split())
    assert "exactly once" in omschrijving
    assert "it stays where it was" in omschrijving


def test_het_resultaat_draagt_alleen_nog_attributen():
    assert set(RefinementResult.model_fields) == {"scratchpad", "attributes"}


def test_the_scratchpad_points_at_the_rules_instead_of_repeating_them():
    """De regels stonden hier én in de promptbody. De twee kopieën liepen uit
    elkaar — de ene droeg een drempel die de andere niet kende — en het model
    kreeg ze allebei. Eén plek, en dat is de prompt."""
    stappen = RefinementResult.model_fields["scratchpad"].description
    assert "numbered rules of the prompt" in stappen
    assert "Minimize the number of containers" not in stappen


def test_the_rules_are_stated_exactly_once():
    prompt = build_refinement_prompt(**_kwargs())
    assert prompt.count("Minimize the number of containers") == 1


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


def test_the_prompt_shows_no_neighbouring_facet():
    """Het verhuisrooster is weg met de uitgang die het bediende. Bleef het
    staan, dan zou het model attributen zien die het niet kan noemen."""
    prompt = build_refinement_prompt(**_kwargs())
    assert "move_targets" not in prompt
    assert "destination list" not in prompt


def test_no_rule_offers_an_exit_out_of_the_facet():
    """Zonder rooster is een regel die verhuizen noemt een uitgang naar niets."""
    regels = _rules(build_refinement_prompt(**_kwargs()))
    assert "misfit" not in regels.lower()
    assert "moved to" not in regels


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
    return prompt[prompt.index("\n# Rules\n"):prompt.index("<universal_rules>")]


def test_rules_do_not_use_the_word_dimension():
    assert "dimension" not in _rules(build_refinement_prompt(**_kwargs())).lower()


def test_no_threshold_numbers_in_the_rules():
    """Aandelen komen uit de data en mogen; een vast percentage is van één
    dataset afgelezen en mag niet. Er heeft er één gestaan — '<10% in the
    domain' — en die maakte in elk facet dat een klein deel van zijn domein
    houdt élk attribuut samenvoegbaar."""
    regels = _rules(build_refinement_prompt(**_kwargs()))
    assert "%" not in regels
    # Alleen de nummering van de regels zelf mag een cijfer zijn.
    assert not re.search(r"(?<!^)(?<!\n)\d", regels, re.MULTILINE)


def test_prevalence_is_judged_against_the_siblings():
    regels = _rules(build_refinement_prompt(**_kwargs()))
    assert "far less prevalent than the others" in regels


def test_a_rule_tells_the_model_what_the_catch_all_marker_means():
    """De marker werd een tijd gerenderd zonder dat één regel hem noemde. Het
    model noemde een vangnet toen als bron, en zijn ideeën liepen leeg in een
    inhoudelijk attribuut."""
    regels = _rules(build_refinement_prompt(**_kwargs()))
    assert "[CATCH-ALL]" in regels
    assert "never name one as a source" in regels


def test_catch_alls_are_marked_not_recognised_by_name():
    """The name is translated and rewritable; the marker comes from drain_key."""
    gemarkeerd = [r for r in _contents().splitlines() if "[CATCH-ALL]" in r]
    assert len(gemarkeerd) == 1
    assert "Overig" in gemarkeerd[0]
    assert "Wachttijd" not in gemarkeerd[0]
