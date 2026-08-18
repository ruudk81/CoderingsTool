"""Tests voor de v2-keten: volgorde van de fasen en de cachecontracten."""
from pipeline.step_5_codeGenerator.concept_inventory import Concept
from pipeline.step_5_codeGenerator.config_codeGenerator import CodebookConfig
from pipeline.step_5_codeGenerator.code_shape import CodeShape
from pipeline.step_5_codeGenerator.prompts_codeGenerator import ConsolidatedCode
from pipeline.step_5_codeGenerator.v2 import run_codebook_v2 as runner


class _NullCostTracker:
    """De productie-ingang meet kosten; in een test hoeft dat niet naar schijf."""

    def record_phase(self, *args, **kwargs):
        pass

    def finalize_step(self, *args, **kwargs):
        pass
from pipeline.step_5_codeGenerator.v2.grouping import ShapingResult


def test_writer_prompt_builder_defaults_to_v1_behaviour():
    """De enige aanpassing in v1-code is een optionele parameter met de
    bestaande builder als default."""
    import inspect
    from pipeline.step_5_codeGenerator import codebook_writer
    from pipeline.step_5_codeGenerator.prompts_writer import build_writer_prompt

    signature = inspect.signature(codebook_writer.write_codebook)
    assert signature.parameters["prompt_builder"].default is build_writer_prompt


def test_degeneration_is_reported_not_repaired(capsys):
    """Een stille terugval zou precies verbergen wat je moet weten."""
    result = runner.GeneratedCodebookV2(
        shapes=[], overig_ids=[], codes=[], direction_loss=0,
        degeneration="geen consolidatie: 64 groepen op 66 attributen (grens 90%)",
        partition_repairs=[], collisions=[], naming_mismatches=[],
        duplicate_definitions=[], vetoes=[], concept_by_id={},
    )

    runner.report_codebook_build_v2(result)

    assert "DEGENERATIE" in capsys.readouterr().out


def test_direction_loss_is_reported_when_nonzero(capsys):
    result = runner.GeneratedCodebookV2(
        shapes=[], overig_ids=[], codes=[], direction_loss=42, degeneration=None,
        partition_repairs=[], collisions=[], naming_mismatches=[],
        duplicate_definitions=[], vetoes=[], concept_by_id={},
    )

    runner.report_codebook_build_v2(result)

    assert "42" in capsys.readouterr().out


def test_direction_loss_wording_does_not_claim_everything_goes_to_overig(capsys):
    """I3: `grouping.py` only routes a group's members into `overig_ids` when
    NO pole clears the threshold. When one pole clears, the dropped pole is
    counted into `direction_loss` but the attribute stays a source of the
    surviving code — those respondents reach the surviving, oppositely-signed
    code in step 6, not Overig. The message must not claim otherwise."""
    result = runner.GeneratedCodebookV2(
        shapes=[], overig_ids=[], codes=[], direction_loss=8, degeneration=None,
        partition_repairs=[], collisions=[], naming_mismatches=[],
        duplicate_definitions=[], vetoes=[], concept_by_id={},
    )

    runner.report_codebook_build_v2(result)

    out = capsys.readouterr().out
    assert "geen eigen code" in out
    assert "overblijvende" in out


def test_partition_duplicate_in_group_is_reported_not_raised(capsys):
    """C1: `repair_partition` emits three log actions, but the report used to
    branch on only two (`PARTITION_MISSING` / else), so a
    `PARTITION_DUPLICATE_IN_GROUP` entry — which carries `attribute_id` and
    `group`, not `kept_in`/`removed_from` — fell into the PARTITION_DOUBLE
    branch and raised `KeyError: 'kept_in'`. Reachable model output: nothing
    stops the same tag appearing twice in one code's `topics`."""
    result = runner.GeneratedCodebookV2(
        shapes=[], overig_ids=[], codes=[], direction_loss=0, degeneration=None,
        partition_repairs=[{"action": "PARTITION_DUPLICATE_IN_GROUP",
                            "attribute_id": "A1", "group": "G"}],
        collisions=[], naming_mismatches=[], duplicate_definitions=[], vetoes=[],
        concept_by_id={},
    )

    runner.report_codebook_build_v2(result)

    out = capsys.readouterr().out
    assert "A1" in out and "G" in out


def test_vetoes_are_reported_when_present(capsys):
    """F2: elke samengevoegde groep in v2 is `pooled` — een veto is de normale
    route, niet een randgeval — en zonder deze melding verdwijnt een
    afgekeurde code stil in de Overig-sweep, ononderscheidbaar van een
    attribuut dat de consolidatie zelf al niet had samengevoegd."""
    result = runner.GeneratedCodebookV2(
        shapes=[], overig_ids=[], codes=[], direction_loss=0, degeneration=None,
        partition_repairs=[], collisions=[], naming_mismatches=[],
        duplicate_definitions=[],
        vetoes=[{"action": "VETO", "members": ["A1", "A2"], "umbrella": "u",
                "reason": "grouped topics share nothing that can be named honestly"}],
        concept_by_id={},
    )

    runner.report_codebook_build_v2(result)

    out = capsys.readouterr().out
    assert "1 pooled code" in out
    assert "A1" in out and "A2" in out


def test_veto_survivor_is_paired_with_its_own_shape(monkeypatch):
    """F4: `write_codebook` kan een `pooled` shape veto'en en 'm weglaten uit
    wat het teruggeeft. De plandraft gaf `shaped.shapes` (twee shapes)
    integraal door aan `resolve_duplicate_names` terwijl `codes` na een veto
    er nog maar één had — en zou hier met een `ValueError` zijn gecrasht
    ("codes and shapes must be positional pairs of equal length"). Deze test
    toetst dat de overgebleven code aan de JUISTE van de twee shapes wordt
    gekoppeld, niet alleen dat er geen crash is."""
    concept_a = Concept(attribute_id="A1", name="Kleur", definition="d",
                        domain="D", facet="F", n_iu=1, resp_ids=frozenset(),
                        resp_pos=frozenset(), resp_neg=frozenset(), resp_neu=frozenset())
    concept_b = Concept(attribute_id="A2", name="Vorm", definition="d",
                        domain="D", facet="F", n_iu=1, resp_ids=frozenset(),
                        resp_pos=frozenset(), resp_neg=frozenset(), resp_neu=frozenset())
    shape_a = CodeShape(key="V1", members=("A1",), valence="positive", umbrella="u",
                        resp_ids=frozenset(), resp_pos=frozenset(), resp_neg=frozenset(),
                        resp_neu=frozenset(), origin="pooled")
    shape_b = CodeShape(key="V2", members=("A2",), valence="positive", umbrella="u",
                        resp_ids=frozenset(), resp_pos=frozenset(), resp_neg=frozenset(),
                        resp_neu=frozenset(), origin="pooled")
    shaping_result = ShapingResult(shapes=[shape_a, shape_b], overig_ids=[], direction_loss=0)

    # write_codebook vetoes shape_a's code — only shape_b's comes back.
    code_for_b = ConsolidatedCode(
        code_name="Vorm-code", definition="def", diagnostic_test="t",
        valence="positive", typical_indicators=["Vorm"], source_attributes=["Vorm"],
    )

    async def fake_resolve_consolidation(*args, **kwargs):
        return object()

    async def fake_write_codebook(*args, **kwargs):
        return [code_for_b]

    monkeypatch.setattr(runner, "resolve_consolidation", fake_resolve_consolidation)
    monkeypatch.setattr(runner, "repair_partition", lambda *a, **k: [])
    monkeypatch.setattr(runner, "build_shapes", lambda *a, **k: shaping_result)
    monkeypatch.setattr(runner, "write_codebook", fake_write_codebook)

    result = runner.generate_codebook_v2(
        [concept_a, concept_b], {}, threshold=1, survey_question="q", n_respondents=2,
        dimension_diagnostic="d", language="Dutch", config=CodebookConfig(), verbose=False,
    )

    assert len(result.shapes) == 1
    assert result.shapes[0].key == "V2"
    assert result.codes[0].source_attributes == ["Vorm"]


def test_run_codebook_v2_pins_step_on_cache_call(monkeypatch):
    """Step 6 en step 7 lezen `mece_codes`. Dit toetst de daadwerkelijke
    `cache_mece_results`-aanroep op die letterlijke sleutel, niet op de
    `CACHE_STEP`-constante: een assert tegen de constante zou meebewegen met
    elke wijziging en dus niets bewaken."""
    class FakeMetadata:
        lang = "Dutch"
        var_lab = "Wat vindt u van dit merk?"
        primary_dimension = ""

    class FakeTaxonomy:
        partition_set = object()
        partition_results = {}

    class FakeCacheManager:
        def is_metadata_cache_valid(self, *args, **kwargs):
            return False

    empty_result = runner.GeneratedCodebookV2(
        shapes=[], overig_ids=[], codes=[], direction_loss=0, degeneration=None,
        partition_repairs=[], collisions=[], naming_mismatches=[],
        duplicate_definitions=[], vetoes=[], concept_by_id={},
    )

    captured = {}

    monkeypatch.setattr(runner, "CacheManager", FakeCacheManager)
    monkeypatch.setattr(runner, "save_prompts_to_json", lambda printer: None)
    monkeypatch.setattr(runner, "CostTracker", lambda **kwargs: _NullCostTracker())
    monkeypatch.setattr(runner, "generate_codebook_v2", lambda *a, **k: empty_result)
    monkeypatch.setattr(runner, "load_extraction_metadata", lambda *a, **k: FakeMetadata())
    monkeypatch.setattr(runner, "load_classified_ideas", lambda *a, **k: [])
    monkeypatch.setattr(runner, "load_taxonomy_cache", lambda *a, **k: FakeTaxonomy())
    monkeypatch.setattr(runner, "apply_overig_sweep", lambda codes, results, language: "Overig")
    monkeypatch.setattr(runner, "print_codebook_results", lambda codes: None)
    monkeypatch.setattr(runner, "run_scorecard", lambda *a, **k: None)
    monkeypatch.setattr(runner, "cache_mece_results", lambda *a, **k: captured.update(k))

    runner.run_codebook_v2(filename="f", var_name="v", sample_size=10, force_recalc=True)

    assert captured.get("step") == "mece_codes"


def test_degenerate_proposal_is_not_cached(monkeypatch, capsys):
    """I2: degeneratie is een harde FAIL — een ontaard voorstel mag niet onder
    CACHE_STEP landen waar step 6 het stilzwijgend zou inlezen. Reporting
    (codebook + scorecard) blijft draaien; alleen de cache-write wordt
    overgeslagen, en dat moet met zoveel woorden gemeld worden."""
    class FakeMetadata:
        lang = "Dutch"
        var_lab = "Wat vindt u van dit merk?"
        primary_dimension = ""

    class FakeTaxonomy:
        partition_set = object()
        partition_results = {}

    class FakeCacheManager:
        def is_metadata_cache_valid(self, *args, **kwargs):
            return False

    degenerate_result = runner.GeneratedCodebookV2(
        shapes=[], overig_ids=[], codes=[], direction_loss=0,
        degeneration="geen consolidatie: 64 groepen op 66 attributen (grens 90%)",
        partition_repairs=[], collisions=[], naming_mismatches=[],
        duplicate_definitions=[], vetoes=[], concept_by_id={},
    )

    cache_calls = []

    monkeypatch.setattr(runner, "CacheManager", FakeCacheManager)
    monkeypatch.setattr(runner, "save_prompts_to_json", lambda printer: None)
    monkeypatch.setattr(runner, "CostTracker", lambda **kwargs: _NullCostTracker())
    monkeypatch.setattr(runner, "generate_codebook_v2", lambda *a, **k: degenerate_result)
    monkeypatch.setattr(runner, "load_extraction_metadata", lambda *a, **k: FakeMetadata())
    monkeypatch.setattr(runner, "load_classified_ideas", lambda *a, **k: [])
    monkeypatch.setattr(runner, "load_taxonomy_cache", lambda *a, **k: FakeTaxonomy())
    monkeypatch.setattr(runner, "apply_overig_sweep", lambda codes, results, language: "Overig")
    monkeypatch.setattr(runner, "print_codebook_results", lambda codes: None)
    monkeypatch.setattr(runner, "run_scorecard", lambda *a, **k: None)
    monkeypatch.setattr(runner, "cache_mece_results", lambda *a, **k: cache_calls.append(k))

    runner.run_codebook_v2(filename="f", var_name="v", sample_size=10, force_recalc=True)

    assert cache_calls == []
    out = capsys.readouterr().out
    assert "NIET gecached" in out
    assert "degeneratie" in out


def test_richtingsverlies_is_paired_with_the_scorecards_under_split_count(monkeypatch, capsys):
    """I3: `under_split_codes` is the number that measures RICHTINGSVERLIES's
    effect on this run — print it alongside, using the scorecard `run_scorecard`
    (v1) already builds, without touching v1 itself."""
    class FakeMetadata:
        lang = "Dutch"
        var_lab = "Wat vindt u van dit merk?"
        primary_dimension = ""

    class FakeTaxonomy:
        partition_set = object()
        partition_results = {}

    class FakeCacheManager:
        def is_metadata_cache_valid(self, *args, **kwargs):
            return False

    class FakeScorecard:
        under_split_codes = [object(), object(), object()]

    result_with_loss = runner.GeneratedCodebookV2(
        shapes=[], overig_ids=[], codes=[], direction_loss=8, degeneration=None,
        partition_repairs=[], collisions=[], naming_mismatches=[],
        duplicate_definitions=[], vetoes=[], concept_by_id={},
    )

    monkeypatch.setattr(runner, "CacheManager", FakeCacheManager)
    monkeypatch.setattr(runner, "save_prompts_to_json", lambda printer: None)
    monkeypatch.setattr(runner, "CostTracker", lambda **kwargs: _NullCostTracker())
    monkeypatch.setattr(runner, "generate_codebook_v2", lambda *a, **k: result_with_loss)
    monkeypatch.setattr(runner, "load_extraction_metadata", lambda *a, **k: FakeMetadata())
    monkeypatch.setattr(runner, "load_classified_ideas", lambda *a, **k: [])
    monkeypatch.setattr(runner, "load_taxonomy_cache", lambda *a, **k: FakeTaxonomy())
    monkeypatch.setattr(runner, "apply_overig_sweep", lambda codes, results, language: "Overig")
    monkeypatch.setattr(runner, "print_codebook_results", lambda codes: None)
    monkeypatch.setattr(runner, "run_scorecard", lambda *a, **k: FakeScorecard())
    monkeypatch.setattr(runner, "cache_mece_results", lambda *a, **k: None)

    runner.run_codebook_v2(filename="f", var_name="v", sample_size=10, force_recalc=True)

    out = capsys.readouterr().out
    assert "3 under-split code(s)" in out


def test_no_unconditional_success_claim_when_cache_save_fails(monkeypatch, capsys):
    """I5: `cache_mece_results` prints its own honest ERROR line and returns
    None on a failed save; `run_codebook_v2` used to print a "v2-codeboek
    gecached ..." success line right after, unconditionally. The announcement
    must not claim success when the save failed."""
    class FakeMetadata:
        lang = "Dutch"
        var_lab = "Wat vindt u van dit merk?"
        primary_dimension = ""

    class FakeTaxonomy:
        partition_set = object()
        partition_results = {}

    class FakeCacheManager:
        def is_metadata_cache_valid(self, *args, **kwargs):
            return False

    empty_result = runner.GeneratedCodebookV2(
        shapes=[], overig_ids=[], codes=[], direction_loss=0, degeneration=None,
        partition_repairs=[], collisions=[], naming_mismatches=[],
        duplicate_definitions=[], vetoes=[], concept_by_id={},
    )

    def fake_failed_cache(*args, **kwargs):
        print("ERROR: codebook NOT cached (0 codes) — downstream steps will regenerate.")

    monkeypatch.setattr(runner, "CacheManager", FakeCacheManager)
    monkeypatch.setattr(runner, "save_prompts_to_json", lambda printer: None)
    monkeypatch.setattr(runner, "CostTracker", lambda **kwargs: _NullCostTracker())
    monkeypatch.setattr(runner, "generate_codebook_v2", lambda *a, **k: empty_result)
    monkeypatch.setattr(runner, "load_extraction_metadata", lambda *a, **k: FakeMetadata())
    monkeypatch.setattr(runner, "load_classified_ideas", lambda *a, **k: [])
    monkeypatch.setattr(runner, "load_taxonomy_cache", lambda *a, **k: FakeTaxonomy())
    monkeypatch.setattr(runner, "apply_overig_sweep", lambda codes, results, language: "Overig")
    monkeypatch.setattr(runner, "print_codebook_results", lambda codes: None)
    monkeypatch.setattr(runner, "run_scorecard", lambda *a, **k: None)
    monkeypatch.setattr(runner, "cache_mece_results", fake_failed_cache)

    runner.run_codebook_v2(filename="f", var_name="v", sample_size=10, force_recalc=True)

    out = capsys.readouterr().out
    assert "NOT cached" in out
    assert "v2-codeboek gecached" not in out
