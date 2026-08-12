"""Tests voor de dimensie-specifieke standing domains.

Structureel, geen LLM. Wat hier bewezen wordt is de CONSTRUCTIE: elke dimensie
levert er twee, ze bereiken de prompt, en de keys overleven. Of de teksten goed
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
    STANDING_BARE_KEY,
    STANDING_OTHER_KEY,
    DiscoveredDomainItem,
    DomainChunkResponse,
    DomainConsolidatedResponse,
    DomainItem,
    ReformulatedDomains,
    StandingLabelsResponse,
    build_domain_consolidation_prompt,
    build_orthogonalize_domains_prompt,
    build_standing_labels_prompt,
)

ALL_KEYS = sorted(DIMENSIONS)


# ── 1. Volledigheid ────────────────────────────────────────────────────────

@pytest.mark.parametrize("dimension_key", ALL_KEYS)
def test_every_dimension_carries_both_standing_domains(dimension_key):
    """Een dimensie zonder standing domains laat step 3 zonder afvoerdomein draaien."""
    d = get_dimension(dimension_key)
    for spec in (d.standing_bare, d.standing_other):
        assert isinstance(spec, StandingDomain)
        for field in ("fallback_label", "definition", "short"):
            value = getattr(spec, field)
            assert value and value.strip(), f"{dimension_key}.{field} is leeg"


@pytest.mark.parametrize("dimension_key", ALL_KEYS)
def test_the_two_standing_domains_are_distinct(dimension_key):
    """Samengevallen definities maken de twee afvoeren ononderscheidbaar."""
    d = get_dimension(dimension_key)
    assert d.standing_bare.definition != d.standing_other.definition
    assert d.standing_bare.short != d.standing_other.short


def test_standing_domains_are_required_fields():
    """Zonder default kan een nieuwe dimensie ze niet vergeten: TypeError bij import."""
    fields = DIMENSIONS[ALL_KEYS[0]].__dataclass_fields__
    import dataclasses
    for name in ("standing_bare", "standing_other"):
        assert fields[name].default is dataclasses.MISSING
        assert fields[name].default_factory is dataclasses.MISSING


# ── 2. Resolutie ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("dimension_key", ALL_KEYS)
def test_resolve_falls_back_when_there_is_no_translation(dimension_key):
    """Geen vertaling (call gefaald of overgeslagen): het Engelse fallback-label."""
    d = get_dimension(dimension_key)
    out = IdeaExtractor._resolve_standing_domains(None, d)

    assert [c.key for c in out] == [STANDING_BARE_KEY, STANDING_OTHER_KEY]
    assert out[0].label == d.standing_bare.fallback_label
    assert out[1].label == d.standing_other.fallback_label
    assert all(c.boundary_test.strip() for c in out)


@pytest.mark.parametrize("dimension_key", ALL_KEYS)
def test_resolve_takes_the_label_and_nothing_else(dimension_key):
    """Het label komt van de vertaling, de betekenis uit dimension_data."""
    d = get_dimension(dimension_key)
    out = IdeaExtractor._resolve_standing_domains(
        StandingLabelsResponse(bare_label="Kale associatie", other_label="Overig"), d)

    assert [c.label for c in out] == ["Kale associatie", "Overig"]
    assert out[0].definition == d.standing_bare.definition
    assert out[1].definition == d.standing_other.definition
    assert [c.key for c in out] == [STANDING_BARE_KEY, STANDING_OTHER_KEY]


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
    """Een leeg of blank label mag geen naamloos domein op het menu zetten."""
    d = get_dimension(dimension_key)
    out = IdeaExtractor._resolve_standing_domains(
        StandingLabelsResponse(bare_label="   ", other_label=""), d)

    assert out[0].label == d.standing_bare.fallback_label
    assert out[1].label == d.standing_other.fallback_label


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
        DomainItem(key=STANDING_BARE_KEY, label="Algemene beoordeling",
                   definition="d", boundary_test="t", exclusions=[]),
        DomainItem(key=STANDING_OTHER_KEY, label="Niet-geclassificeerd onderwerp",
                   definition="d", boundary_test="t", exclusions=[]),
    ]
    IdeaExtractor._set_domain_keys(domains)

    assert [c.key for c in domains] == [
        "Duurzaamheid", STANDING_BARE_KEY, STANDING_OTHER_KEY]


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
    assert d.standing_bare.definition in prompt
    assert d.standing_other.definition in prompt
    assert d.standing_bare.short in prompt
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
def test_standing_bare_states_an_axis_failure_not_a_content_type(dimension_key):
    """De as-faalmodus, niet een opsomming van inhoudsvormen.

    De vorm is: [de handeling van deze dimensie] + [maar geen eenheid op de as].
    Wie inhoudsvormen opsomt sluit stilzwijgend uit wat er niet in staat, en die
    ideeën worden dan in een inhoudelijk domein geperst.
    """
    d = get_dimension(dimension_key)
    t = d.standing_bare.definition

    assert t.rstrip().endswith("it simply names nothing the other domains could cover.")
    assert "names no" in t or "no sphere" in t or "with no" in t


@pytest.mark.parametrize("dimension_key", ALL_KEYS)
def test_standing_other_sends_axis_failures_back_to_the_bare_domain(dimension_key):
    """`other` is voor een genoemd onderwerp dat geen domein dekt — niet voor leegte.

    Zonder die afbakening lopen de twee vol elkaar in: alles wat nergens past wordt
    `other`, en de kale antwoorden verliezen hun eigen categorie.
    """
    d = get_dimension(dimension_key)
    assert "not for" in d.standing_other.definition


def test_attributes_associations_no_longer_enumerates_affective_forms():
    """Regressie: de dimensie van kwaliteiten en beelden noemde er drie op.

    'evaluation, a feeling or a general impression' liet elke associatie die geen
    van drieën is buiten de definitie vallen — de kale categorie-associatie het
    duidelijkst.
    """
    d = get_dimension("ATTRIBUTES_ASSOCIATIONS")
    assert "a feeling or a general impression" not in d.standing_bare.definition
    assert "general-impression domain" not in d.standing_other.definition


# ── 5. De vertaalcall ─────────────────────────────────────────────────────

@pytest.mark.parametrize("dimension_key", ALL_KEYS)
def test_standing_labels_prompt_carries_both_fixed_definitions(dimension_key):
    """De vertaalcall krijgt de canonieke tekst mee — anders vertaalt hij een gok."""
    d = get_dimension(dimension_key)
    prompt = build_standing_labels_prompt(language="nl-NL", entity="e", dimension=d)

    assert d.standing_bare.definition in prompt
    assert d.standing_other.definition in prompt
    assert d.standing_bare.short in prompt
    assert d.standing_other.short in prompt
    assert "nl-NL" in prompt
    assert prompt.rstrip().endswith(
        "provide your output as valid JSON following the response schema provided.")


def test_standing_labels_response_carries_labels_only():
    """Alleen labels. Een definitie-veld hier zou het vangnet weer laten schuiven."""
    assert set(StandingLabelsResponse.model_fields) == {"bare_label", "other_label"}


# ── 6. Orthogonalisatie raakt de vangnetten niet ──────────────────────────

def _mk(key, label):
    return DomainItem(key=key, label=label, definition=f"def {label}",
                      boundary_test="t?", exclusions=[])


def test_partition_standing_splits_and_keeps_order():
    domains = [_mk("Duurzaamheid", "Duurzaamheid"),
               _mk(STANDING_BARE_KEY, "Kale associatie"),
               _mk("Aanbod", "Aanbod"),
               _mk(STANDING_OTHER_KEY, "Overig")]
    discovered, standing = IdeaExtractor._partition_standing(domains)

    assert [d.label for d in discovered] == ["Duurzaamheid", "Aanbod"]
    assert [d.label for d in standing] == ["Kale associatie", "Overig"]


def test_merge_orthogonalized_leaves_the_standing_two_untouched():
    """Het vangnet mag niet herschreven terugkomen — dat is hoe de definitie versmalde."""
    discovered = [_mk("Duurzaamheid", "Duurzaamheid")]
    standing = [_mk(STANDING_BARE_KEY, "Kale associatie"),
                _mk(STANDING_OTHER_KEY, "Overig")]
    new_discovered = [_mk("", "Ecologische koers")]

    merged, rename = IdeaExtractor._merge_orthogonalized(
        new_discovered, discovered, standing)

    assert [d.label for d in merged] == ["Ecologische koers", "Kale associatie", "Overig"]
    assert [d.key for d in merged] == [
        "Ecologische koers", STANDING_BARE_KEY, STANDING_OTHER_KEY]
    assert merged[1].definition == "def Kale associatie"
    assert rename == {"Duurzaamheid": "Ecologische koers"}


def test_merge_orthogonalized_refuses_a_count_mismatch():
    """De telcontrole vergelijkt tegen de ONTDEKTE domeinen, niet tegen het totaal.

    Regressie: de guard telde tegen de volledige lijst inclusief de twee
    vangnetten. Zodra het responsemodel er twee minder teruggeeft, slaat hij aan
    en draait orthogonalisatie helemaal niet meer — zonder foutmelding.
    """
    discovered = [_mk("A", "A"), _mk("B", "B")]
    standing = [_mk(STANDING_BARE_KEY, "Kale associatie"),
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
    standing = [_mk(STANDING_BARE_KEY, "Overig"), _mk(STANDING_OTHER_KEY, "Kale associatie")]
    discovered = [_mk("Overig", "Overig")]

    renames = IdeaExtractor._disambiguate_against_standing(discovered, standing)

    assert renames == [("Overig", "Overig (2)")]
    assert discovered[0].label == "Overig (2)"
    # De staande twee blijven onaangeroerd.
    assert [d.label for d in standing] == ["Overig", "Kale associatie"]


def test_disambiguate_against_standing_leaves_non_colliding_labels_alone():
    standing = [_mk(STANDING_BARE_KEY, "Kale associatie"), _mk(STANDING_OTHER_KEY, "Overig")]
    discovered = [_mk("Duurzaamheid", "Duurzaamheid"), _mk("Aanbod", "Aanbod")]

    renames = IdeaExtractor._disambiguate_against_standing(discovered, standing)

    assert renames == []
    assert [d.label for d in discovered] == ["Duurzaamheid", "Aanbod"]


def test_disambiguate_and_remap_fixes_a_relabel_that_collides_after_orthogonalize():
    """Tweede botsingsrichting: de herformulering beschrijft een ontdekt domein
    opnieuw en komt toevallig op een staand label uit. Dat mag geen dubbel label op
    het toewijzingsmenu zetten, en een idee dat tegelijk naar dat label wordt
    hernoemd moet op het uiteindelijke (ontdubbelde) label uitkomen."""
    standing = [_mk(STANDING_BARE_KEY, "Kale associatie"), _mk(STANDING_OTHER_KEY, "Overig")]
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
    standing = [_mk(STANDING_BARE_KEY, "Kale associatie"), _mk(STANDING_OTHER_KEY, "Overig")]
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
