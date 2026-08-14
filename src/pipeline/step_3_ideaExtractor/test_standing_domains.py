"""Tests voor de dimensie-specifieke standing domains.

Structureel, geen LLM. Wat hier bewezen wordt is de CONSTRUCTIE: elke dimensie
levert er drie, ze bereiken de prompt, en de keys overleven. Of de teksten goed
GEFORMULEERD zijn is niet mechanisch te toetsen — dat blijkt pas op data die een
andere dimensie kiest.
"""

import pytest

from pipeline.step_3_ideaExtractor.dimension_data import (
    DIMENSIONS,
    StandingDomain,
    get_dimension,
)
from pipeline.step_3_ideaExtractor.ideaExtractor import IdeaExtractor
from pipeline.step_3_ideaExtractor.prompts_ideaExtractor import (
    NON_ANSWER_DOMAIN,
    STANDING_NOT_KNOWN_KEY,
    STANDING_NO_SUBJECT_KEY,
    STANDING_OTHER_KEY,
    DiscoveredDomainItem,
    DomainChunkResponse,
    DomainConsolidatedResponse,
    DomainItem,
    MenuEntryRenderResponse,
    ReformulatedDomains,
    build_domain_consolidation_prompt,
    build_domain_discovery_prompt,
    build_orthogonalize_domains_prompt,
    build_standing_labels_prompt,
)

ALL_KEYS = sorted(DIMENSIONS)


# ── 1. Volledigheid ────────────────────────────────────────────────────────

@pytest.mark.parametrize("dimension_key", ALL_KEYS)
def test_every_dimension_carries_all_standing_domains(dimension_key):
    """Een dimensie zonder standing domains laat step 3 zonder afvoerdomein draaien."""
    d = get_dimension(dimension_key)
    for spec in (d.standing_not_known, d.standing_other, d.standing_no_subject):
        assert isinstance(spec, StandingDomain)
        for field in ("fallback_label", "definition", "short"):
            value = getattr(spec, field)
            assert value and value.strip(), f"{dimension_key}.{field} is leeg"


@pytest.mark.parametrize("dimension_key", ALL_KEYS)
def test_the_standing_domains_are_pairwise_distinct(dimension_key):
    """Samengevallen definities maken de afvoeren ononderscheidbaar.

    De drie vangen drie verschillende faalvormen van de domeinas: het onderwerp
    niet kennen, een onderwerp noemen dat geen domein dekt, en geen onderwerp
    noemen. Lopen er twee in elkaar, dan verliest een van de drie zijn eigen
    categorie.
    """
    d = get_dimension(dimension_key)
    specs = (d.standing_not_known, d.standing_other, d.standing_no_subject)
    assert len({s.definition for s in specs}) == 3
    assert len({s.short for s in specs}) == 3
    assert len({s.fallback_label for s in specs}) == 3


@pytest.mark.parametrize("dimension_key", ALL_KEYS)
def test_contentless_test_is_present_and_distinct_from_not_known(dimension_key):
    """`contentless_test` (P8) en `standing_not_known.short` (step 3) zijn andere concepten.

    De eerste is "noemt niets op de as", de tweede "kent het onderwerp niet". Ze
    mogen niet stilzwijgend samenvallen — dat zou P8 de verkeerde toets geven.
    """
    d = get_dimension(dimension_key)
    contentless_test = d.prompt_rules.contentless_test
    assert contentless_test and contentless_test.strip()
    assert contentless_test != d.standing_not_known.short


def test_standing_domains_are_required_fields():
    """Zonder default kan een nieuwe dimensie ze niet vergeten: TypeError bij import."""
    fields = DIMENSIONS[ALL_KEYS[0]].__dataclass_fields__
    import dataclasses
    for name in ("standing_not_known", "standing_other", "standing_no_subject"):
        assert fields[name].default is dataclasses.MISSING
        assert fields[name].default_factory is dataclasses.MISSING


# ── 2. Resolutie ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("dimension_key", ALL_KEYS)
def test_resolve_falls_back_when_there_is_no_translation(dimension_key):
    """Geen vertaling (call gefaald of overgeslagen): het Engelse fallback-label."""
    d = get_dimension(dimension_key)
    out = IdeaExtractor._resolve_standing_domains(None, d)

    assert [c.key for c in out] == [
        STANDING_NOT_KNOWN_KEY, STANDING_OTHER_KEY, STANDING_NO_SUBJECT_KEY]
    assert out[0].label == d.standing_not_known.fallback_label
    assert out[1].label == d.standing_other.fallback_label
    assert out[2].label == d.standing_no_subject.fallback_label
    assert all(c.boundary_test.strip() for c in out)


@pytest.mark.parametrize("dimension_key", ALL_KEYS)
def test_resolve_prefers_the_rendered_text_but_keeps_the_english_as_source(dimension_key):
    """Label, definitie én lidmaatschapstoets komen uit de weergave als die er is."""
    d = get_dimension(dimension_key)
    rendered = MenuEntryRenderResponse(
        not_known_label="Kent het merk niet", not_known_definition="NL definitie een.",
        not_known_boundary_test="Meldt de respondent het merk niet te kennen?",
        other_label="Overig onderwerp", other_definition="NL definitie twee.",
        other_boundary_test="Noemt het antwoord een onderwerp dat geen domein dekt?",
        no_subject_label="Zonder onderwerp", no_subject_definition="NL definitie drie.",
        no_subject_boundary_test="Noemt het antwoord geen onderwerp?",
        non_answer_label="Geen inhoud", non_answer_definition="NL definitie vier.",
        non_answer_boundary_test="Zegt het fragment alleen dat er geen antwoord is?")
    out = IdeaExtractor._resolve_standing_domains(rendered, d)
    assert [c.label for c in out] == [
        "Kent het merk niet", "Overig onderwerp", "Zonder onderwerp"]
    assert out[0].definition == "NL definitie een."
    assert out[0].boundary_test == "Meldt de respondent het merk niet te kennen?"
    assert [c.key for c in out] == [
        STANDING_NOT_KNOWN_KEY, STANDING_OTHER_KEY, STANDING_NO_SUBJECT_KEY]


@pytest.mark.parametrize("dimension_key", ALL_KEYS)
def test_resolve_falls_back_to_the_dimension_on_a_blank_rendering(dimension_key):
    """Geen weergave (call gefaald of overgeslagen): de Engelse dimensietekst."""
    d = get_dimension(dimension_key)
    out = IdeaExtractor._resolve_standing_domains(None, d)
    assert out[0].definition == d.standing_not_known.definition
    assert out[0].label == d.standing_not_known.fallback_label
    assert out[0].boundary_test.strip()


def test_resolve_standing_domains_have_no_exclusions():
    """Pin voor bevinding C: de menu-regel ✗ mag alleen verschijnen als er iets in staat.

    `_resolve_standing_domains` levert altijd `exclusions=[]` — niet per dimensie
    verschillend, dus één dimensie volstaat om de constructie vast te leggen.
    """
    d = get_dimension(ALL_KEYS[0])
    out = IdeaExtractor._resolve_standing_domains(None, d)
    assert all(c.exclusions == [] for c in out)


@pytest.mark.parametrize("dimension_key", ALL_KEYS)
def test_resolve_ignores_an_empty_translated_label(dimension_key):
    """Een leeg of blank label mag geen naamloos domein op het menu zetten — en
    dat mag onafhankelijk zijn van of de definitie/boundary_test wél gerenderd is."""
    d = get_dimension(dimension_key)
    out = IdeaExtractor._resolve_standing_domains(
        MenuEntryRenderResponse(
            not_known_label="   ", not_known_definition="d1", not_known_boundary_test="t1",
            other_label="", other_definition="d2", other_boundary_test="t2",
            no_subject_label=" ", no_subject_definition="d3", no_subject_boundary_test="t3",
            non_answer_label="", non_answer_definition="d4", non_answer_boundary_test="t4"), d)

    assert out[0].label == d.standing_not_known.fallback_label
    assert out[1].label == d.standing_other.fallback_label
    assert out[2].label == d.standing_no_subject.fallback_label
    assert out[0].definition == "d1"
    assert out[1].definition == "d2"
    assert out[2].definition == "d3"


# ── 2b. De normalisatie ná consolidatie ────────────────────────────────────

def test_set_domain_keys_derives_from_label_but_spares_the_standing_two():
    """Ontdekte domeinen krijgen hun key uit het label, de staande twee houden de hunne.

    Regressie (2026-08-09): deze normalisatie liep onvoorwaardelijk over álle
    domeinen en wiste de staande keys vóórdat _orthogonalize_domains ze kon
    beschermen. De guard daar bewaakte dus iets dat al vernietigd was, en step 4's
    DRAIN_KEYS vond niets meer — zonder foutmelding.
    """
    domains = [
        DomainItem(key="", label="Duurzaamheid",
                   definition="d", boundary_test="t", exclusions=[]),
        DomainItem(key=STANDING_NOT_KNOWN_KEY, label="Algemene beoordeling",
                   definition="d", boundary_test="t", exclusions=[]),
        DomainItem(key=STANDING_OTHER_KEY, label="Niet-geclassificeerd onderwerp",
                   definition="d", boundary_test="t", exclusions=[]),
    ]
    IdeaExtractor._set_domain_keys(domains)

    assert [c.key for c in domains] == [
        "Duurzaamheid", STANDING_NOT_KNOWN_KEY, STANDING_OTHER_KEY]


def test_set_domain_keys_accepts_no_domains():
    """Fase 3 kan overgeslagen zijn; dan is er niets te normaliseren."""
    IdeaExtractor._set_domain_keys(None)
    IdeaExtractor._set_domain_keys([])


# ── 3. De teksten bereiken de prompt ───────────────────────────────────────

@pytest.mark.parametrize("dimension_key", ALL_KEYS)
def test_consolidation_prompt_carries_this_dimensions_wording(dimension_key):
    """De builder leest de vangnetten uit de meegegeven dimensie en zet ze als grond."""
    d = get_dimension(dimension_key)
    prompt = build_domain_consolidation_prompt(
        language="nl-NL", survey_question="Vraag?", sector="s", entity="e",
        topic="t", perspective="p", intent="i", primary_dimension=dimension_key,
        chunk_results="chunk", dimension=d,
    )
    assert d.standing_not_known.definition in prompt
    assert d.standing_other.definition in prompt
    assert d.standing_not_known.short in prompt
    assert d.prompt_rules.domain_diagnostic in prompt
    # De eenrichtingsregel: de verplichting ligt bij de ontdekte domeinen.
    assert "must not reach into" in prompt
    # Het model levert ze niet meer op — de prompt zegt dat met zoveel woorden.
    assert "you do NOT return them" in prompt


def test_consolidation_response_has_no_slot_for_the_standing_domains():
    """Constructie, geen instructie: er is geen veld om ze in te herschrijven."""
    assert set(DomainConsolidatedResponse.model_fields) == {"domains"}


def test_orthogonalize_response_has_no_slot_for_the_standing_domains():
    """Zelfde constructie-toets, voor het herformuleringsmodel."""
    assert set(ReformulatedDomains.model_fields) == {"domains"}


# ── 4. Elke dimensie stelt de structurele toets, geen inhoudelijke ─────────
#
# Verving de no-op-pin op ATTRIBUTES_ASSOCIATIONS (2026-08-09). Die legde de tekst
# byte voor byte vast om te bewijzen dat de per-dimensie-refactor daar niets
# veranderde. Dat bewijs is geleverd — en de pin bevroor intussen een fout: de
# refactor verbreedde het concept naar "noemt niets op de as" en schreef tien nieuwe
# definities, terwijl de elfde op de oude, smallere tekst bleef staan. Een snapshot
# bewaakt de letter; deze tests bewaken de vorm.

@pytest.mark.parametrize("dimension_key", ALL_KEYS)
def test_standing_other_sends_axis_failures_back_to_the_bare_domain(dimension_key):
    """`other` is voor een genoemd onderwerp dat geen domein dekt — niet voor leegte.

    Zonder die afbakening lopen de twee vol elkaar in: alles wat nergens past wordt
    `other`, en de kale antwoorden verliezen hun eigen categorie.
    """
    d = get_dimension(dimension_key)
    assert "not for" in d.standing_other.definition


CORE_GUARD = ("A response that gives no answer at all, or that states there is nothing "
              "to report, belongs to the quality filter and never reaches this domain.")


@pytest.mark.parametrize("dimension_key", ALL_KEYS)
def test_not_known_carries_the_filter_boundary_verbatim(dimension_key):
    """Deze zin is de grens met filtercode 97 en 98. Zonder hem loopt het vangnet vol."""
    assert CORE_GUARD in get_dimension(dimension_key).standing_not_known.definition


@pytest.mark.parametrize("dimension_key", ALL_KEYS)
def test_not_known_avoids_the_category_one_magnets(dimension_key):
    """'I don't know' is letterlijk categorie 1 van het filter — dat woord trekt het aan."""
    t = get_dimension(dimension_key).standing_not_known.definition.lower()
    for magnet in ("i don't know", "i do not know", "not sure", "no opinion"):
        assert magnet not in t, f"{dimension_key}: bevat '{magnet}'"


@pytest.mark.parametrize("dimension_key", ALL_KEYS)
def test_not_known_describes_what_the_respondent_reports(dimension_key):
    """Een uitspraak over het onderwerp, niet een constatering over het antwoord.

    'contains no content' zou samenvallen met filtercode 98; 'reports not knowing'
    niet.
    """
    d = get_dimension(dimension_key).standing_not_known
    assert "reports" in d.definition or "reports" in d.short
    assert "contains no content" not in d.definition


@pytest.mark.parametrize("dimension_key", ALL_KEYS)
def test_standing_other_points_at_the_not_known_domain(dimension_key):
    """De oude staart verwees naar het vangnet dat niet meer bestaat."""
    t = get_dimension(dimension_key).standing_other.definition
    assert "not-known domain" in t
    assert "unplaced" not in t.lower()


# ── 5. De vertaalcall ─────────────────────────────────────────────────────

@pytest.mark.parametrize("dimension_key", ALL_KEYS)
def test_standing_labels_prompt_carries_both_fixed_definitions(dimension_key):
    """De vertaalcall krijgt de canonieke tekst mee — anders vertaalt hij een gok."""
    d = get_dimension(dimension_key)
    prompt = build_standing_labels_prompt(language="nl-NL", entity="e", dimension=d)

    assert d.standing_not_known.definition in prompt
    assert d.standing_other.definition in prompt
    assert d.standing_no_subject.definition in prompt
    assert d.standing_not_known.short in prompt
    assert d.standing_other.short in prompt
    assert d.standing_no_subject.short in prompt
    assert "nl-NL" in prompt
    assert prompt.rstrip().endswith(
        "provide your output as valid JSON following the response schema provided.")


@pytest.mark.parametrize("dimension_key", ALL_KEYS)
def test_standing_labels_prompt_also_carries_the_non_answer_bucket(dimension_key):
    """Bevinding C: de tijdelijke bak krijgt dezelfde vertaalbehandeling."""
    d = get_dimension(dimension_key)
    prompt = build_standing_labels_prompt(language="nl-NL", entity="e", dimension=d)

    assert NON_ANSWER_DOMAIN.definition in prompt
    assert NON_ANSWER_DOMAIN.short in prompt


def test_menu_entry_render_response_carries_three_fields_per_entry():
    """Label, definitie én boundary_test — voor alle drie de vangnetten én de
    non-answer-bak."""
    assert set(MenuEntryRenderResponse.model_fields) == {
        "not_known_label", "not_known_definition", "not_known_boundary_test",
        "other_label", "other_definition", "other_boundary_test",
        "no_subject_label", "no_subject_definition", "no_subject_boundary_test",
        "non_answer_label", "non_answer_definition", "non_answer_boundary_test"}


# ── 6. Orthogonalisatie raakt de vangnetten niet ──────────────────────────

def _mk(key, label):
    return DomainItem(key=key, label=label, definition=f"def {label}",
                      boundary_test="t?", exclusions=[])


def test_partition_standing_splits_and_keeps_order():
    domains = [_mk("Duurzaamheid", "Duurzaamheid"),
               _mk(STANDING_NOT_KNOWN_KEY, "Kale associatie"),
               _mk("Aanbod", "Aanbod"),
               _mk(STANDING_OTHER_KEY, "Overig")]
    discovered, standing = IdeaExtractor._partition_standing(domains)

    assert [d.label for d in discovered] == ["Duurzaamheid", "Aanbod"]
    assert [d.label for d in standing] == ["Kale associatie", "Overig"]


def test_merge_orthogonalized_leaves_the_standing_two_untouched():
    """Het vangnet mag niet herschreven terugkomen — dat is hoe de definitie versmalde."""
    discovered = [_mk("Duurzaamheid", "Duurzaamheid")]
    standing = [_mk(STANDING_NOT_KNOWN_KEY, "Kale associatie"),
                _mk(STANDING_OTHER_KEY, "Overig")]
    new_discovered = [_mk("", "Ecologische koers")]

    merged, rename = IdeaExtractor._merge_orthogonalized(
        new_discovered, discovered, standing)

    assert [d.label for d in merged] == ["Ecologische koers", "Kale associatie", "Overig"]
    assert [d.key for d in merged] == [
        "Ecologische koers", STANDING_NOT_KNOWN_KEY, STANDING_OTHER_KEY]
    assert merged[1].definition == "def Kale associatie"
    assert rename == {"Duurzaamheid": "Ecologische koers"}


def test_merge_orthogonalized_refuses_a_count_mismatch():
    """De telcontrole vergelijkt tegen de ONTDEKTE domeinen, niet tegen het totaal.

    Regressie: de guard telde tegen de volledige lijst inclusief de twee
    vangnetten. Zodra het responsemodel er twee minder teruggeeft, slaat hij aan
    en draait orthogonalisatie helemaal niet meer — zonder foutmelding.
    """
    discovered = [_mk("A", "A"), _mk("B", "B")]
    standing = [_mk(STANDING_NOT_KNOWN_KEY, "Kale associatie"),
                _mk(STANDING_OTHER_KEY, "Overig")]

    assert IdeaExtractor._merge_orthogonalized([_mk("", "A2")], discovered, standing) \
        == (None, None)
    merged, _ = IdeaExtractor._merge_orthogonalized(
        [_mk("", "A2"), _mk("", "B2")], discovered, standing)
    assert merged is not None


def test_orthogonalize_prompt_shows_the_standing_two_as_fixed():
    """Zichtbaar zodat de andere zich ervan wegformuleren, met de eenrichtingsregel.

    De builder neemt geen dimensie — parametriseren over alle elf gaf elf keer
    dezelfde run.
    """
    prompt = build_orthogonalize_domains_prompt(
        language="nl-NL", survey_question="Vraag?", sector="s", entity="e",
        topic="t", perspective="p", intent="i", primary_dimension="ATTRIBUTES_ASSOCIATIONS",
        domain_diagnostic="Welk onderwerpsgebied?",
        domains_block="  Duurzaamheid: def",
        standing_block="  Kale associatie: vangnet-definitie",
    )
    assert "Kale associatie" in prompt
    assert "vangnet-definitie" in prompt
    assert "do not return them" in prompt
    assert "must not reach into" in prompt


# ── 7. `key` is niet meer door het model te schrijven (bevinding A) ────────

def test_response_schemas_expose_no_key_property_to_the_model():
    """Dit is het echte contract: de JSON schema die instructor naar het model stuurt.

    Een veld dat wél op `DomainItem` bestaat maar hier ontbreekt kan nooit
    modeloutput worden, ongeacht wat de prompttekst zegt.
    """
    for cls in (DomainChunkResponse, DomainConsolidatedResponse, ReformulatedDomains):
        schema = cls.model_json_schema()
        item_schema = schema["$defs"]["DiscoveredDomainItem"]
        assert "key" not in item_schema["properties"]


def test_model_supplied_key_cannot_reach_self_domains():
    """Zelfs een kwaadwillig/verward 'key': 'other' in de ruwe modeloutput haalt
    `self.domains` niet: het schema heeft er geen plek voor, dus Pydantic laat het
    veld vallen bij het parsen, en de enige plek waar `key` daarna gezet wordt is
    `_set_domain_keys`, vanuit het label.
    """
    raw = DiscoveredDomainItem.model_validate({
        "key": STANDING_OTHER_KEY,
        "label": "Duurzaamheid",
        "definition": "d", "boundary_test": "t", "exclusions": [],
    })
    assert not hasattr(raw, "key")

    domain = DomainItem(**raw.model_dump())
    IdeaExtractor._set_domain_keys([domain])

    assert domain.key == "Duurzaamheid"
    assert domain.key != STANDING_OTHER_KEY


# ── 8. Labelbotsing met een staand domein (bevinding B) ────────────────────

def test_disambiguate_against_standing_renames_the_discovered_one():
    """Eerste botsingsrichting: bij het samenstellen van de lijst kiest een ontdekt
    domein toevallig hetzelfde label als een staand domein. De staande twee komen
    uit een aparte, parallelle vertaalcall — niets garandeert dat de labels
    verschillen."""
    standing = [_mk(STANDING_NOT_KNOWN_KEY, "Overig"), _mk(STANDING_OTHER_KEY, "Kale associatie")]
    discovered = [_mk("Overig", "Overig")]

    renames = IdeaExtractor._disambiguate_against_standing(discovered, standing)

    assert renames == [("Overig", "Overig (2)")]
    assert discovered[0].label == "Overig (2)"
    # De staande twee blijven onaangeroerd.
    assert [d.label for d in standing] == ["Overig", "Kale associatie"]


def test_disambiguate_against_standing_leaves_non_colliding_labels_alone():
    standing = [_mk(STANDING_NOT_KNOWN_KEY, "Kale associatie"), _mk(STANDING_OTHER_KEY, "Overig")]
    discovered = [_mk("Duurzaamheid", "Duurzaamheid"), _mk("Aanbod", "Aanbod")]

    renames = IdeaExtractor._disambiguate_against_standing(discovered, standing)

    assert renames == []
    assert [d.label for d in discovered] == ["Duurzaamheid", "Aanbod"]


def test_disambiguate_and_remap_fixes_a_relabel_that_collides_after_orthogonalize():
    """Tweede botsingsrichting: de herformulering beschrijft een ontdekt domein
    opnieuw en komt toevallig op een staand label uit. Dat mag geen dubbel label op
    het toewijzingsmenu zetten, en een idee dat tegelijk naar dat label wordt
    hernoemd moet op het uiteindelijke (ontdubbelde) label uitkomen."""
    standing = [_mk(STANDING_NOT_KNOWN_KEY, "Kale associatie"), _mk(STANDING_OTHER_KEY, "Overig")]
    new_discovered = [_mk("Overig", "Overig")]  # _merge_orthogonalized zette de key al
    rename = {"Duurzaamheid": "Overig"}

    new_domains, rename2, collisions = IdeaExtractor._disambiguate_and_remap(
        list(new_discovered) + list(standing), rename)

    assert collisions == [("Overig", "Overig (2)")]
    assert rename2 is rename
    assert rename2 == {"Duurzaamheid": "Overig (2)"}
    assert [d.label for d in new_domains] == ["Overig (2)", "Kale associatie", "Overig"]
    # De key volgt het uiteindelijke label, niet het botsende tussenlabel.
    assert new_domains[0].key == "Overig (2)"


def test_disambiguate_and_remap_is_a_noop_without_a_collision():
    standing = [_mk(STANDING_NOT_KNOWN_KEY, "Kale associatie"), _mk(STANDING_OTHER_KEY, "Overig")]
    new_discovered = [_mk("Duurzaamheid", "Duurzaamheid")]
    rename = {"Duurzaam": "Duurzaamheid"}

    new_domains, rename2, collisions = IdeaExtractor._disambiguate_and_remap(
        list(new_discovered) + list(standing), rename)

    assert collisions == []
    assert rename2 == {"Duurzaam": "Duurzaamheid"}
    assert [d.label for d in new_domains] == ["Duurzaamheid", "Kale associatie", "Overig"]


# ── 9. Het toewijzingsmenu (bevinding C) ────────────────────────────────────

def test_domain_table_omits_the_dangling_exclusion_marker():
    """Een staand domein heeft `exclusions=[]` — de ✗-regel moet dan wegvallen in
    plaats van kaal '✗ ' te tonen voor precies de twee gevoeligste domeinen."""
    d = get_dimension(ALL_KEYS[0])
    standing = IdeaExtractor._resolve_standing_domains(None, d)
    domain_with_exclusions = _mk("Duurzaamheid", "Duurzaamheid")
    domain_with_exclusions.exclusions = ["Aanbod"]

    table = IdeaExtractor.build_domain_table([domain_with_exclusions] + standing)

    lines = table.splitlines()
    assert any(line.strip() == "✗ Aanbod" for line in lines)
    assert not any(line.strip() == "✗" for line in lines)
    assert "✗ \n" not in table
    assert not table.rstrip().endswith("✗")


# ── 10. Domeinoverzicht na orthogonalisatie ─────────────────────────────────

def test_format_domain_overview_shows_arrow_for_a_renamed_discovered_domain():
    """Een hernoemd ontdekt domein toont `oud label → nieuw label`."""
    d = _mk("Duurzaamheid", "Ecologische koers")
    rename = {"Duurzaamheid": "Ecologische koers"}

    lines = IdeaExtractor._format_domain_overview_lines(
        [d], rename, (STANDING_NOT_KNOWN_KEY, STANDING_OTHER_KEY))

    assert lines[0] == "    • Duurzaamheid → Ecologische koers"
    assert lines[1] == "        def: def Ecologische koers"
    assert lines[2] == "        ✓ t?"


def test_format_domain_overview_shows_the_label_alone_when_unchanged():
    """Geen botsing van bewoording nodig: een ongewijzigd label staat er kaal."""
    d = _mk("Aanbod", "Aanbod")
    rename = {"Aanbod": "Aanbod"}

    lines = IdeaExtractor._format_domain_overview_lines(
        [d], rename, (STANDING_NOT_KNOWN_KEY, STANDING_OTHER_KEY))

    assert lines[0] == "    • Aanbod"


def test_format_domain_overview_marks_a_standing_domain_and_omits_its_exclusion_line():
    """Een staand domein krijgt de markering en nooit een ✗-regel (exclusions=[])."""
    standing = _mk(STANDING_NOT_KNOWN_KEY, "Kale associatie")

    lines = IdeaExtractor._format_domain_overview_lines(
        [standing], {}, (STANDING_NOT_KNOWN_KEY, STANDING_OTHER_KEY))

    assert lines[0] == "    • Kale associatie (standing)"
    assert not any(line.strip().startswith("✗") for line in lines)


def test_format_domain_overview_shows_the_exclusion_line_when_present():
    """Een ontdekt domein mét exclusions krijgt wél de ✗-regel."""
    d = _mk("Duurzaamheid", "Duurzaamheid")
    d.exclusions = ["Aanbod", "Prijs"]

    lines = IdeaExtractor._format_domain_overview_lines(
        [d], {"Duurzaamheid": "Duurzaamheid"}, (STANDING_NOT_KNOWN_KEY, STANDING_OTHER_KEY))

    assert lines[-1] == "        ✗ Aanbod, Prijs"


# ── 11. De oude sleutel mag nergens achterblijven ───────────────────────────

def test_no_file_in_src_still_mentions_the_old_key():
    """Een achtergebleven oude sleutel matcht nergens meer en faalt stil.

    Bouwt de oude sleutel uit delen op: anders bevat deze testfile zelf het
    verboden woord (in deze docstring, in de vergelijking hieronder) en zou
    de test zichzelf altijd als overtreder aanwijzen.

    Matcht op woordgrens, niet als kale substring: `measure_stability.py` leest
    historische snapshotregels (`data/step3_stability.jsonl`, geschreven vóór
    deze hernoeming) tolerant terug via het oude veld `bare_evaluation_pct` —
    dat bestand blijft ongewijzigd en mag dus niet herschreven worden. Die
    veldnaam is geen sleutel-gebruik en moet dit niet laten falen.
    """
    import re
    from pathlib import Path
    old_key = "bare_" + "evaluation"
    pattern = re.compile(r"\b" + old_key + r"\b")
    this_file = Path(__file__).resolve()
    src = this_file.parents[2]
    offenders = [
        str(p.relative_to(src))
        for p in src.rglob("*.py")
        if p != this_file and "__pycache__" not in str(p) and pattern.search(p.read_text(encoding="utf-8"))
    ]
    assert offenders == [], f"nog aanwezig in: {offenders}"


# ── 12. Vergaarbak-verbod toetst de grensvorm, niet het onderwerp ──────────

def test_both_prompts_ban_a_residual_boundary_not_a_subject_type():
    """De grensvorm is de toets, niet het onderwerp — anders blokkeert de regel
    het eigenschappen-domein dat we juist willen."""
    d = get_dimension("ATTRIBUTES_ASSOCIATIONS")
    disc = build_domain_discovery_prompt(
        language="nl-NL", survey_question="Vraag?", chunk_responses="x", chunk_size=10,
        perspective="p", intent="i", sector="s", entity="e", topic="t",
        primary_dimension="ATTRIBUTES_ASSOCIATIONS",
        primary_dimension_description="beschrijving", dimension=d)
    cons = build_domain_consolidation_prompt(
        language="nl-NL", survey_question="Vraag?", sector="s", entity="e", topic="t",
        perspective="p", intent="i", primary_dimension="ATTRIBUTES_ASSOCIATIONS",
        chunk_results="chunk", dimension=d)
    for prompt in (disc, cons):
        assert "defined by what it contains, never by what the other domains do not" in prompt
        assert '"character"' not in prompt


# ── 13. De tijdelijke non-answer-bak bij de toewijzing ─────────────────────

def test_domain_table_offers_the_non_answer_bucket():
    """Zonder zichtbare bak kan het model zo'n fragment nergens kwijt."""
    doms = [DomainItem(key="Duurzaamheid", label="Duurzaamheid", definition="d",
                       boundary_test="t?", exclusions=["x"])]
    non_answer = DomainItem(key="non_answer", label="Geen inhoud", definition="d?",
                            boundary_test="t2?", exclusions=[])
    table = IdeaExtractor.build_domain_table(doms, non_answer)
    assert non_answer.label in table
    assert non_answer.boundary_test in table


def test_domain_table_falls_back_to_canonical_english_without_a_rendering():
    """Geen `non_answer` meegegeven: het Engelse fallback-label, niet stil niets."""
    doms = [DomainItem(key="Duurzaamheid", label="Duurzaamheid", definition="d",
                       boundary_test="t?", exclusions=[])]
    table = IdeaExtractor.build_domain_table(doms)
    assert NON_ANSWER_DOMAIN.fallback_label in table


def test_drop_non_answer_ideas_removes_them_and_reports_what_went():
    """Verwijderen mag, ongemerkt verwijderen niet."""
    import models

    def _idea(idea_id, instance, domain):
        return models.IdeasExtractedSubmodel(
            idea_id=str(idea_id), idea=instance, instance=instance,
            interpretation=instance, abstraction=instance, domain=domain)

    label = "Geen inhoud"
    rows = [models.IdeasExtractedModel(
        respondent_id=1, response="Eekhoorn, Niks.", response_type="text",
        quality_filter=False, response_ideas=[
            _idea(1, "Eekhoorn", "Merkuitingen"),
            _idea(2, "Niks", label),
        ], idea_count=2)]

    dropped, texts = IdeaExtractor._drop_non_answer_ideas(rows, label)

    assert dropped == 1
    assert texts == ["Niks"]
    assert [i.instance for i in rows[0].response_ideas] == ["Eekhoorn"]
    assert rows[0].idea_count == 1


def test_drop_non_answer_ideas_is_a_noop_without_them():
    import models
    rows = [models.IdeasExtractedModel(
        respondent_id=1, response="Eekhoorn.", response_type="text",
        quality_filter=False, response_ideas=[models.IdeasExtractedSubmodel(
            idea_id="1", idea="Eekhoorn", instance="Eekhoorn",
            interpretation="Eekhoorn", abstraction="Eekhoorn", domain="Merkuitingen")],
        idea_count=1)]
    assert IdeaExtractor._drop_non_answer_ideas(rows, "Geen inhoud") == (0, [])
    assert rows[0].idea_count == 1


def test_drop_non_answer_ideas_handles_a_response_with_no_ideas():
    """Bevinding H: `response_ideas=None` mag niet crashen — de `or []`-guard."""
    import models
    rows = [models.IdeasExtractedModel(
        respondent_id=1, response="", response_type="text",
        quality_filter=True, response_ideas=None, idea_count=0)]

    assert IdeaExtractor._drop_non_answer_ideas(rows, "Geen inhoud") == (0, [])
    assert rows[0].response_ideas is None
    assert rows[0].idea_count == 0


def test_drop_non_answer_ideas_can_drop_every_idea_in_a_response():
    """Bevinding H: alle ideeën van één respons zitten in de non-answer-bak."""
    import models
    label = "Geen inhoud"

    def _idea(idea_id, instance):
        return models.IdeasExtractedSubmodel(
            idea_id=str(idea_id), idea=instance, instance=instance,
            interpretation=instance, abstraction=instance, domain=label)

    rows = [models.IdeasExtractedModel(
        respondent_id=1, response="Niks, niks.", response_type="text",
        quality_filter=False, response_ideas=[_idea(1, "Niks"), _idea(2, "niks")],
        idea_count=2)]

    dropped, texts = IdeaExtractor._drop_non_answer_ideas(rows, label)

    assert dropped == 2
    assert texts == ["Niks", "niks"]
    assert rows[0].response_ideas == []
    assert rows[0].idea_count == 0


# ── 14. De drie sleutel-definities blijven synchroon (bevinding F) ─────────

def test_drain_key_literals_agree_across_the_three_definitions():
    """`prompts_ideaExtractor`, `measure_stability` en `taxonomy_health` houden
    elk hun eigen kopie van dezelfde drie sleutels — niets anders bewaakte dat
    een hernoeming ze alle drie raakt."""
    from pipeline.step_3_ideaExtractor.measure_stability import DRAIN_KEYS as stability_keys
    from pipeline.step_4_classifier.taxonomy_health import DRAIN_KEYS as health_keys

    canonical = {STANDING_NOT_KNOWN_KEY, STANDING_OTHER_KEY, STANDING_NO_SUBJECT_KEY}
    assert set(stability_keys) == canonical
    assert set(health_keys) == canonical


@pytest.mark.parametrize("dimension_key", ALL_KEYS)
def test_consolidation_prompt_grounds_all_three_standing_domains(dimension_key):
    """Consolidatie moet alle drie zien, anders maakt hij er alsnog een duplicaat van.

    Regressie-vorm: tot 2026-08-13 zag hij er twee, en de zin erbij zei dat
    "answers reporting no knowledge" al gedekt waren. Dat las als dekking voor
    alles wat leeg leek, waardoor het ontdekte geen-inhoud-domein sneuvelde en
    de onderwerploze antwoorden over de inhoudelijke domeinen werden uitgesmeerd.
    """
    d = get_dimension(dimension_key)
    prompt = build_domain_consolidation_prompt(
        language="nl-NL", survey_question="Vraag?", sector="s", entity="e",
        topic="t", perspective="p", intent="i", primary_dimension=dimension_key,
        chunk_results="chunk", dimension=d,
    )
    for spec in (d.standing_not_known, d.standing_other, d.standing_no_subject):
        assert spec.definition in prompt, f"{dimension_key}: {spec.fallback_label}"
        assert spec.short in prompt


# ── 15. Eén bron voor "is dit een vangnet?" ────────────────────────────────

def test_standing_keys_bevat_alle_drie():
    from pipeline.step_3_ideaExtractor.prompts_ideaExtractor import STANDING_KEYS
    assert set(STANDING_KEYS) == {
        STANDING_NOT_KNOWN_KEY, STANDING_OTHER_KEY, STANDING_NO_SUBJECT_KEY}


def test_set_domain_keys_spaart_alle_drie_de_staande_domeinen():
    """Regressie 2026-08-13: `no_subject` kwam erbij maar `_set_domain_keys`
    kende er twee, dus het derde vangnet kreeg zijn label als key. Step 4's
    drain_domains() zag het daardoor niet en gaf het een volledige facetlaag —
    precies waar een vangnet van gevrijwaard hoort te zijn."""
    domains = [
        DomainItem(key="", label="Duurzaamheid", definition="d",
                   boundary_test="t", exclusions=[]),
        DomainItem(key=STANDING_NOT_KNOWN_KEY, label="Onbekend",
                   definition="d", boundary_test="t", exclusions=[]),
        DomainItem(key=STANDING_OTHER_KEY, label="Ander onderwerp",
                   definition="d", boundary_test="t", exclusions=[]),
        DomainItem(key=STANDING_NO_SUBJECT_KEY, label="Geen genoemd onderwerp",
                   definition="d", boundary_test="t", exclusions=[]),
    ]
    IdeaExtractor._set_domain_keys(domains)

    assert [d.key for d in domains] == [
        "Duurzaamheid", STANDING_NOT_KNOWN_KEY, STANDING_OTHER_KEY,
        STANDING_NO_SUBJECT_KEY]


def test_partition_standing_herkent_alle_drie():
    domains = [_mk("Duurzaamheid", "Duurzaamheid"),
               _mk(STANDING_NOT_KNOWN_KEY, "Onbekend"),
               _mk(STANDING_NO_SUBJECT_KEY, "Geen genoemd onderwerp"),
               _mk(STANDING_OTHER_KEY, "Ander onderwerp")]
    discovered, standing = IdeaExtractor._partition_standing(domains)

    assert [d.label for d in discovered] == ["Duurzaamheid"]
    assert len(standing) == 3


def test_niemand_hardcodeert_nog_een_deelverzameling():
    """De vorige twee bugs ontstonden allebei doordat een tweede plek zijn eigen
    lijstje bijhield. Deze test faalt zodra er weer een los paar verschijnt."""
    from pathlib import Path
    here = Path(__file__).parent
    verboden = "STANDING_NOT_KNOWN_KEY, STANDING_OTHER_KEY)"
    overtreders = [
        p.name for p in here.glob("*.py")
        if p.name != Path(__file__).name and verboden in p.read_text(encoding="utf-8")
    ]
    assert overtreders == [], f"los paar in: {overtreders}"
