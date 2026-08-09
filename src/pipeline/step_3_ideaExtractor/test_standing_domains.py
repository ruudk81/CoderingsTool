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
    DomainConsolidatedResponse,
    DomainItem,
    build_domain_consolidation_prompt,
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
def test_resolve_falls_back_when_the_llm_returns_nothing(dimension_key):
    """Het fallback-pad: de consolidatie leverde geen standing_domains."""
    d = get_dimension(dimension_key)
    out = IdeaExtractor._resolve_standing_domains(
        DomainConsolidatedResponse(domains=[]), d)

    assert [c.key for c in out] == [STANDING_BARE_KEY, STANDING_OTHER_KEY]
    assert out[0].label == d.standing_bare.fallback_label
    assert out[0].definition == d.standing_bare.definition
    assert out[1].definition == d.standing_other.definition
    assert all(c.boundary_test.strip() for c in out)


@pytest.mark.parametrize("dimension_key", ALL_KEYS)
def test_resolve_keeps_llm_labels_but_forces_the_keys(dimension_key):
    """Het normale pad: labels in enquêtetaal, keys onaantastbaar.

    De key is het downstream-contract (step 4 DRAIN_KEYS). Wat het model ook
    terugstuurt, de key wordt gezet.
    """
    d = get_dimension(dimension_key)
    consolidated = DomainConsolidatedResponse(
        domains=[],
        standing_domains=[
            DomainItem(key=STANDING_BARE_KEY, label="Algemene indruk",
                       definition="Vertaalde definitie.", boundary_test="Test?",
                       exclusions=[]),
            DomainItem(key=STANDING_OTHER_KEY, label="Overig",
                       definition="Vertaalde definitie twee.", boundary_test="Test?",
                       exclusions=[]),
        ],
    )
    out = IdeaExtractor._resolve_standing_domains(consolidated, d)

    assert [c.key for c in out] == [STANDING_BARE_KEY, STANDING_OTHER_KEY]
    assert [c.label for c in out] == ["Algemene indruk", "Overig"]


@pytest.mark.parametrize("dimension_key", ALL_KEYS)
def test_resolve_repairs_a_mangled_key(dimension_key):
    """Zet het model de key op het label, dan wordt hij teruggezet, niet overgenomen."""
    d = get_dimension(dimension_key)
    consolidated = DomainConsolidatedResponse(
        domains=[],
        standing_domains=[
            DomainItem(key="Algemene indruk zonder onderwerp", label="Algemene indruk",
                       definition="Vertaald.", boundary_test="Test?", exclusions=[]),
            DomainItem(key=STANDING_OTHER_KEY, label="Overig",
                       definition="Vertaald.", boundary_test="Test?", exclusions=[]),
        ],
    )
    out = IdeaExtractor._resolve_standing_domains(consolidated, d)
    assert [c.key for c in out] == [STANDING_BARE_KEY, STANDING_OTHER_KEY]


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
    """De builder leest de standing domains uit de meegegeven dimensie."""
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
    assert f'key "{STANDING_BARE_KEY}"' in prompt
    assert f'key "{STANDING_OTHER_KEY}"' in prompt


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
