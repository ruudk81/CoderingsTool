"""Tests for the dimension-specific standing domains.

Structural, no LLM. What is proven here is the CONSTRUCTION: every dimension
supplies three, they reach the prompt, and the keys survive. Whether the texts
are well PHRASED cannot be tested mechanically — that only shows on data that
picks a different dimension.
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
    STANDING_KEYS,
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


# ── 1. Completeness ────────────────────────────────────────────────────────

@pytest.mark.parametrize("dimension_key", ALL_KEYS)
def test_every_dimension_carries_all_standing_domains(dimension_key):
    """A dimension without standing domains leaves step 3 running with no drain."""
    d = get_dimension(dimension_key)
    for spec in (d.standing_not_known, d.standing_other, d.standing_no_subject):
        assert isinstance(spec, StandingDomain)
        for field in ("fallback_label", "definition", "short"):
            value = getattr(spec, field)
            assert value and value.strip(), f"{dimension_key}.{field} is empty"


@pytest.mark.parametrize("dimension_key", ALL_KEYS)
def test_the_standing_domains_are_pairwise_distinct(dimension_key):
    """Collapsed definitions make the drains indistinguishable.

    The three catch three different failure modes of the domain axis: not
    knowing the subject, naming a subject no domain covers, and naming no
    subject at all. If two run into each other, one of the three loses its own
    category.
    """
    d = get_dimension(dimension_key)
    specs = (d.standing_not_known, d.standing_other, d.standing_no_subject)
    assert len({s.definition for s in specs}) == 3
    assert len({s.short for s in specs}) == 3
    assert len({s.fallback_label for s in specs}) == 3


@pytest.mark.parametrize("dimension_key", ALL_KEYS)
def test_contentless_test_is_present_and_distinct_from_not_known(dimension_key):
    """`contentless_test` and `standing_not_known.short` are different concepts.

    The first is "names nothing on the axis", the second "does not know the
    subject". They must not silently coincide — that would give the contentless
    check the wrong test.
    """
    d = get_dimension(dimension_key)
    contentless_test = d.prompt_rules.contentless_test
    assert contentless_test and contentless_test.strip()
    assert contentless_test != d.standing_not_known.short


def test_standing_domains_are_required_fields():
    """Without a default a new dimension cannot forget them: TypeError at import."""
    fields = DIMENSIONS[ALL_KEYS[0]].__dataclass_fields__
    import dataclasses
    for name in ("standing_not_known", "standing_other", "standing_no_subject"):
        assert fields[name].default is dataclasses.MISSING
        assert fields[name].default_factory is dataclasses.MISSING


# ── 2. Resolution ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("dimension_key", ALL_KEYS)
def test_resolve_falls_back_when_there_is_no_translation(dimension_key):
    """No translation (call failed or skipped): the English fallback label."""
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
    """Label, definition and membership test all come from the rendering, if any."""
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
    """No rendering (call failed or skipped): the English dimension text."""
    d = get_dimension(dimension_key)
    out = IdeaExtractor._resolve_standing_domains(None, d)
    assert out[0].definition == d.standing_not_known.definition
    assert out[0].label == d.standing_not_known.fallback_label
    assert out[0].boundary_test.strip()


def test_resolve_standing_domains_have_no_exclusions():
    """Pin for finding C: the ✗ menu line may only appear when it holds something.

    `_resolve_standing_domains` always yields `exclusions=[]` — not different per
    dimension, so one dimension suffices to pin the construction.
    """
    d = get_dimension(ALL_KEYS[0])
    out = IdeaExtractor._resolve_standing_domains(None, d)
    assert all(c.exclusions == [] for c in out)


@pytest.mark.parametrize("dimension_key", ALL_KEYS)
def test_resolve_ignores_an_empty_translated_label(dimension_key):
    """An empty or blank label must not put a nameless domain on the menu — and
    that holds independently of whether definition/boundary_test did render."""
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


# ── 2b. The normalisation after consolidation ──────────────────────────────

def test_set_domain_keys_derives_from_label_but_spares_the_standing_two():
    """Discovered domains get their key from the label, the standing ones keep theirs.

    Regression (2026-08-09): this normalisation ran unconditionally over ALL
    domains and wiped the standing keys before _orthogonalize_domains could
    protect them. The guard there was therefore protecting something already
    destroyed, and step 4's DRAIN_KEYS found nothing — without an error.
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
    """Phase 3 may have been skipped; then there is nothing to normalise."""
    IdeaExtractor._set_domain_keys(None)
    IdeaExtractor._set_domain_keys([])


# ── 3. The texts reach the prompt ──────────────────────────────────────────

@pytest.mark.parametrize("dimension_key", ALL_KEYS)
def test_consolidation_prompt_carries_this_dimensions_wording(dimension_key):
    """The builder reads the drains from the given dimension and lays them down."""
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
    # The one-way rule: the obligation sits with the discovered domains.
    assert "must not reach into" in prompt
    # The model no longer returns them — the prompt says so in as many words.
    assert "you do NOT return them" in prompt


def test_consolidation_response_has_no_slot_for_the_standing_domains():
    """Construction, not instruction: there is no field to rewrite them into."""
    assert set(DomainConsolidatedResponse.model_fields) == {"domains"}


def test_orthogonalize_response_has_no_slot_for_the_standing_domains():
    """Same construction test, for the reformulation model."""
    assert set(ReformulatedDomains.model_fields) == {"domains"}


# ── 4. Every dimension makes the structural test, not a substantive one ────
#
# Replaced the no-op pin on ATTRIBUTES_ASSOCIATIONS (2026-08-09). That one fixed
# the text byte for byte to prove the per-dimension refactor changed nothing
# there. That proof has been delivered — and meanwhile the pin froze a bug: the
# refactor widened the concept to "names nothing on the axis" and wrote ten new
# definitions, while the eleventh stayed on the old, narrower text. A snapshot
# guards the letter; these tests guard the shape.

@pytest.mark.parametrize("dimension_key", ALL_KEYS)
def test_standing_other_sends_axis_failures_back_to_the_bare_domain(dimension_key):
    """`other` is for a named subject no domain covers — not for emptiness.

    Without that boundary the two run into each other: everything that fits
    nowhere becomes `other`, and the bare answers lose their own category.
    """
    d = get_dimension(dimension_key)
    assert "not for" in d.standing_other.definition


CORE_GUARD = ("A response that gives no answer at all, or that states there is nothing "
              "to report, belongs to the quality filter and never reaches this domain.")


@pytest.mark.parametrize("dimension_key", ALL_KEYS)
def test_not_known_carries_the_filter_boundary_verbatim(dimension_key):
    """This sentence is the boundary with filter codes 97 and 98. Without it the
    drain fills up."""
    assert CORE_GUARD in get_dimension(dimension_key).standing_not_known.definition


@pytest.mark.parametrize("dimension_key", ALL_KEYS)
def test_not_known_avoids_the_category_one_magnets(dimension_key):
    """'I don't know' is literally category 1 of the filter — that phrase attracts it."""
    t = get_dimension(dimension_key).standing_not_known.definition.lower()
    for magnet in ("i don't know", "i do not know", "not sure", "no opinion"):
        assert magnet not in t, f"{dimension_key}: contains '{magnet}'"


@pytest.mark.parametrize("dimension_key", ALL_KEYS)
def test_not_known_describes_what_the_respondent_reports(dimension_key):
    """A statement about the subject, not an observation about the answer.

    'contains no content' would coincide with filter code 98; 'reports not
    knowing' does not.
    """
    d = get_dimension(dimension_key).standing_not_known
    assert "reports" in d.definition or "reports" in d.short
    assert "contains no content" not in d.definition


@pytest.mark.parametrize("dimension_key", ALL_KEYS)
def test_standing_other_points_at_the_not_known_domain(dimension_key):
    """The old tail pointed at the drain that no longer exists."""
    t = get_dimension(dimension_key).standing_other.definition
    assert "not-known domain" in t
    assert "unplaced" not in t.lower()


# ── 5. The translation call ───────────────────────────────────────────────

@pytest.mark.parametrize("dimension_key", ALL_KEYS)
def test_standing_labels_prompt_carries_both_fixed_definitions(dimension_key):
    """The translation call is given the canonical text — otherwise it translates
    a guess."""
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
    """Finding C: the temporary bucket gets the same translation treatment."""
    d = get_dimension(dimension_key)
    prompt = build_standing_labels_prompt(language="nl-NL", entity="e", dimension=d)

    assert NON_ANSWER_DOMAIN.definition in prompt
    assert NON_ANSWER_DOMAIN.short in prompt


def test_menu_entry_render_response_carries_three_fields_per_entry():
    """Label, definition and boundary_test — for all three drains and the
    non-answer bucket."""
    assert set(MenuEntryRenderResponse.model_fields) == {
        "not_known_label", "not_known_definition", "not_known_boundary_test",
        "other_label", "other_definition", "other_boundary_test",
        "no_subject_label", "no_subject_definition", "no_subject_boundary_test",
        "non_answer_label", "non_answer_definition", "non_answer_boundary_test"}


# ── 6. Orthogonalisation does not touch the drains ────────────────────────

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
    """The drain must not come back rewritten — that is how the definition narrowed."""
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
    """The count check compares against the DISCOVERED domains, not the total.

    Regression: the guard counted against the full list including the two
    drains. As soon as the response model returns two fewer, it trips and
    orthogonalisation stops running altogether — without an error.
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
    """Visible so the others phrase themselves away from it, with the one-way rule.

    The builder takes no dimension — parametrising over all eleven gave the same
    run eleven times.
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


# ── 7. `key` can no longer be written by the model (finding A) ────────────

def test_response_schemas_expose_no_key_property_to_the_model():
    """This is the real contract: the JSON schema instructor sends to the model.

    A field that does exist on `DomainItem` but is missing here can never become
    model output, whatever the prompt text says.
    """
    for cls in (DomainChunkResponse, DomainConsolidatedResponse, ReformulatedDomains):
        schema = cls.model_json_schema()
        item_schema = schema["$defs"]["DiscoveredDomainItem"]
        assert "key" not in item_schema["properties"]


def test_model_supplied_key_cannot_reach_self_domains():
    """Even a malicious/confused 'key': 'other' in the raw model output does not
    reach `self.domains`: the schema has no place for it, so Pydantic drops the
    field while parsing, and the only place `key` is set afterwards is
    `_set_domain_keys`, from the label.
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


# ── 8. Label collision with a standing domain (finding B) ──────────────────

def test_disambiguate_against_standing_renames_the_discovered_one():
    """First collision direction: while assembling the list a discovered domain
    happens to pick the same label as a standing domain. The standing two come
    from a separate, parallel translation call — nothing guarantees the labels
    differ."""
    standing = [_mk(STANDING_NOT_KNOWN_KEY, "Overig"), _mk(STANDING_OTHER_KEY, "Kale associatie")]
    discovered = [_mk("Overig", "Overig")]

    renames = IdeaExtractor._disambiguate_against_standing(discovered, standing)

    assert renames == [("Overig", "Overig (2)")]
    assert discovered[0].label == "Overig (2)"
    # The standing two remain untouched.
    assert [d.label for d in standing] == ["Overig", "Kale associatie"]


def test_disambiguate_against_standing_leaves_non_colliding_labels_alone():
    standing = [_mk(STANDING_NOT_KNOWN_KEY, "Kale associatie"), _mk(STANDING_OTHER_KEY, "Overig")]
    discovered = [_mk("Duurzaamheid", "Duurzaamheid"), _mk("Aanbod", "Aanbod")]

    renames = IdeaExtractor._disambiguate_against_standing(discovered, standing)

    assert renames == []
    assert [d.label for d in discovered] == ["Duurzaamheid", "Aanbod"]


def test_disambiguate_and_remap_fixes_a_relabel_that_collides_after_orthogonalize():
    """Second collision direction: the reformulation redescribes a discovered
    domain and happens to land on a standing label. That must not put a duplicate
    label on the assignment menu, and an idea being renamed to that label at the
    same time must end up on the final (deduplicated) label."""
    standing = [_mk(STANDING_NOT_KNOWN_KEY, "Kale associatie"), _mk(STANDING_OTHER_KEY, "Overig")]
    new_discovered = [_mk("Overig", "Overig")]  # _merge_orthogonalized already set the key
    rename = {"Duurzaamheid": "Overig"}

    new_domains, rename2, collisions = IdeaExtractor._disambiguate_and_remap(
        list(new_discovered) + list(standing), rename)

    assert collisions == [("Overig", "Overig (2)")]
    assert rename2 is rename
    assert rename2 == {"Duurzaamheid": "Overig (2)"}
    assert [d.label for d in new_domains] == ["Overig (2)", "Kale associatie", "Overig"]
    # The key follows the final label, not the colliding intermediate one.
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


# ── 9. The assignment menu (finding C) ──────────────────────────────────────

def test_domain_table_omits_the_dangling_exclusion_marker():
    """A standing domain has `exclusions=[]` — the ✗ line must then drop out
    rather than show a bare '✗ ' for exactly the two most sensitive domains."""
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


# ── 10. Domain overview after orthogonalisation ─────────────────────────────

def test_format_domain_overview_shows_arrow_for_a_renamed_discovered_domain():
    """A renamed discovered domain shows `old label → new label`."""
    d = _mk("Duurzaamheid", "Ecologische koers")
    rename = {"Duurzaamheid": "Ecologische koers"}

    lines = IdeaExtractor._format_domain_overview_lines(
        [d], rename, (STANDING_NOT_KNOWN_KEY, STANDING_OTHER_KEY))

    assert lines == ["    • Duurzaamheid → Ecologische koers"]


def test_format_domain_overview_shows_the_label_alone_when_unchanged():
    """No wording clash needed: an unchanged label stands there bare."""
    d = _mk("Aanbod", "Aanbod")
    rename = {"Aanbod": "Aanbod"}

    lines = IdeaExtractor._format_domain_overview_lines(
        [d], rename, (STANDING_NOT_KNOWN_KEY, STANDING_OTHER_KEY))

    assert lines[0] == "    • Aanbod"


def test_format_domain_overview_marks_a_standing_domain_and_omits_its_exclusion_line():
    """A standing domain gets the marker and never a ✗ line (exclusions=[])."""
    standing = _mk(STANDING_NOT_KNOWN_KEY, "Kale associatie")

    lines = IdeaExtractor._format_domain_overview_lines(
        [standing], {}, (STANDING_NOT_KNOWN_KEY, STANDING_OTHER_KEY))

    assert lines[0] == "    • Kale associatie (standing)"
    assert not any(line.strip().startswith("✗") for line in lines)


def test_format_domain_overview_shows_no_prompt_texts():
    """Definition, boundary_test and exclusions are already in the prompt export.
    Four lines per domain — one of them five lines long since 2026-08-14 — buried
    the only thing this overview is for: seeing what got renamed."""
    d = _mk("Duurzaamheid", "Duurzaamheid")
    d.exclusions = ["Aanbod", "Prijs"]

    lines = IdeaExtractor._format_domain_overview_lines(
        [d], {"Duurzaamheid": "Duurzaamheid"}, STANDING_KEYS)

    assert lines == ["    • Duurzaamheid"]


# ── 11. The old key must be left nowhere ────────────────────────────────────

def test_no_file_in_src_still_mentions_the_old_key():
    """A leftover old key matches nothing any more and fails silently.

    Assembles the old key from parts: otherwise this test file itself contains
    the forbidden word (in this docstring, in the comparison below) and the test
    would always point at itself as an offender.

    Matches on a word boundary, not as a bare substring: `measure_stability.py`
    reads historical snapshot lines (`data/step3_stability.jsonl`, written before
    this rename) back tolerantly via the old field `bare_evaluation_pct` — that
    file stays unchanged and must therefore not be rewritten. That field name is
    not a use of the key and must not make this fail.
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
    assert offenders == [], f"still present in: {offenders}"


# ── 12. The catch-all ban tests the boundary form, not the subject ─────────

def test_both_prompts_ban_a_residual_boundary_not_a_subject_type():
    """The boundary form is the test, not the subject — otherwise the rule blocks
    the very attributes domain we want."""
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


# ── 13. The temporary non-answer bucket at assignment ──────────────────────

def test_domain_table_offers_the_non_answer_bucket():
    """Without a visible bucket the model has nowhere to put such a fragment."""
    doms = [DomainItem(key="Duurzaamheid", label="Duurzaamheid", definition="d",
                       boundary_test="t?", exclusions=["x"])]
    non_answer = DomainItem(key="non_answer", label="Geen inhoud", definition="d?",
                            boundary_test="t2?", exclusions=[])
    table = IdeaExtractor.build_domain_table(doms, non_answer)
    assert non_answer.label in table
    assert non_answer.boundary_test in table


def test_domain_table_falls_back_to_canonical_english_without_a_rendering():
    """No `non_answer` supplied: the English fallback label, not silently nothing."""
    doms = [DomainItem(key="Duurzaamheid", label="Duurzaamheid", definition="d",
                       boundary_test="t?", exclusions=[])]
    table = IdeaExtractor.build_domain_table(doms)
    assert NON_ANSWER_DOMAIN.fallback_label in table


def test_drop_non_answer_ideas_removes_them_and_reports_what_went():
    """Removing is allowed, removing unnoticed is not."""
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
    """Finding H: `response_ideas=None` must not crash — the `or []` guard."""
    import models
    rows = [models.IdeasExtractedModel(
        respondent_id=1, response="", response_type="text",
        quality_filter=True, response_ideas=None, idea_count=0)]

    assert IdeaExtractor._drop_non_answer_ideas(rows, "Geen inhoud") == (0, [])
    assert rows[0].response_ideas is None
    assert rows[0].idea_count == 0


def test_drop_non_answer_ideas_can_drop_every_idea_in_a_response():
    """Finding H: every idea of one response sits in the non-answer bucket."""
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


# ── 14. The three key definitions stay in sync (finding F) ────────────────

def test_drain_key_literals_agree_across_the_three_definitions():
    """`prompts_ideaExtractor`, `measure_stability` and `taxonomy_health` each
    keep their own copy of the same three keys — nothing else guarded that a
    rename reaches all three."""
    from pipeline.step_3_ideaExtractor.measure_stability import DRAIN_KEYS as stability_keys
    from pipeline.step_4_classifier.taxonomy_health import DRAIN_KEYS as health_keys

    canonical = {STANDING_NOT_KNOWN_KEY, STANDING_OTHER_KEY, STANDING_NO_SUBJECT_KEY}
    assert set(stability_keys) == canonical
    assert set(health_keys) == canonical


@pytest.mark.parametrize("dimension_key", ALL_KEYS)
def test_consolidation_prompt_grounds_all_three_standing_domains(dimension_key):
    """Consolidation must see all three, or it makes a duplicate of one anyway.

    Regression shape: until 2026-08-13 it saw two, and the sentence alongside said
    "answers reporting no knowledge" were already covered. That read as coverage
    for everything that looked empty, so the discovered no-content domain died and
    the subject-less answers were smeared over the substantive domains.
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


# ── 15. One source for "is this a drain?" ──────────────────────────────────

def test_standing_keys_holds_all_three():
    from pipeline.step_3_ideaExtractor.prompts_ideaExtractor import STANDING_KEYS
    assert set(STANDING_KEYS) == {
        STANDING_NOT_KNOWN_KEY, STANDING_OTHER_KEY, STANDING_NO_SUBJECT_KEY}


def test_set_domain_keys_spares_all_three_standing_domains():
    """Regression 2026-08-13: `no_subject` was added but `_set_domain_keys` knew
    of two, so the third drain got its label as key. Step 4's drain_domains()
    therefore did not see it and gave it a full facet layer — exactly what a drain
    should be spared."""
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


def test_partition_standing_recognises_all_three():
    domains = [_mk("Duurzaamheid", "Duurzaamheid"),
               _mk(STANDING_NOT_KNOWN_KEY, "Onbekend"),
               _mk(STANDING_NO_SUBJECT_KEY, "Geen genoemd onderwerp"),
               _mk(STANDING_OTHER_KEY, "Ander onderwerp")]
    discovered, standing = IdeaExtractor._partition_standing(domains)

    assert [d.label for d in discovered] == ["Duurzaamheid"]
    assert len(standing) == 3


def test_nobody_hardcodes_a_subset_any_more():
    """The previous two bugs both arose from a second place keeping its own little
    list. This test fails as soon as a loose pair appears again."""
    from pathlib import Path
    here = Path(__file__).parent
    forbidden = "STANDING_NOT_KNOWN_KEY, STANDING_OTHER_KEY)"
    offenders = [
        p.name for p in here.glob("*.py")
        if p.name != Path(__file__).name and forbidden in p.read_text(encoding="utf-8")
    ]
    assert offenders == [], f"loose pair in: {offenders}"


# ── 16. Content versus placement: the four fixed entries ───────────────────

def test_non_answer_is_about_emptiness_not_about_subjectlessness():
    """The four entries answer two different questions. non_answer: was anything
    said at all? The three standing domains: where does what was said belong?

    Until 2026-08-14 non_answer was phrased in placement terms ("without naming
    the subject") and moreover mentioned 'does not know' — which made it clash
    with both no_subject and not_known, and bare numbers ended up in the
    no-subject domain instead of being discarded.
    """
    t = NON_ANSWER_DOMAIN.definition
    assert "carries no statement at all" in t
    assert "does not know the subject is also a real answer" in t
    assert "no-subject domain instead" in t


def test_non_answer_is_not_per_dimension():
    """A remnant of the splitting is an artefact of the splitting, not of the
    dimension. What does differ per dimension sits in standing_no_subject."""
    from pipeline.step_3_ideaExtractor import prompts_ideaExtractor as mod
    assert isinstance(mod.NON_ANSWER_DOMAIN, StandingDomain)
    assert not hasattr(DIMENSIONS[ALL_KEYS[0]], "non_answer")


@pytest.mark.parametrize("dimension_key", ALL_KEYS)
def test_no_subject_points_emptiness_back_to_the_non_answer_entry(dimension_key):
    """The other side of the same boundary, in every dimension."""
    t = get_dimension(dimension_key).standing_no_subject.definition
    assert "communicates nothing at all" in t
    assert "no-content entry" in t


@pytest.mark.parametrize("dimension_key", ALL_KEYS)
def test_the_four_fixed_entries_reference_each_other_without_overlap(dimension_key):
    """Each of the four must say what it is NOT, or they run into each other."""
    d = get_dimension(dimension_key)
    assert "not for" in d.standing_other.definition
    assert "not-known domain" in d.standing_other.definition
    assert "no-content entry" in d.standing_no_subject.definition
    assert "quality filter" in d.standing_not_known.definition
