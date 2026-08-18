"""Tests voor de v2-keten: volgorde van de fasen en de cachecontracten."""
from pipeline.step_5_codeGenerator.concept_inventory import Concept
from pipeline.step_5_codeGenerator.config_codeGenerator import CodebookConfig
from pipeline.step_5_codeGenerator.consolidator import CodeShape
from pipeline.step_5_codeGenerator.prompts_codeGenerator import ConsolidatedCode
from pipeline.step_5_codeGenerator.v2 import run_codebook_v2 as runner
from pipeline.step_5_codeGenerator.v2.grouping import ShapingResult


def test_cache_step_is_separate_from_v1():
    """De v1-cache mag nooit overschreven worden — beide codeboeken moeten op
    dezelfde taxonomie naast elkaar te leggen zijn."""
    assert runner.CACHE_STEP == "mece_codes_v2"
    assert runner.CACHE_STEP != "mece_codes"


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
    """M6: de hardste eis van deze taak is dat v2 de v1-cache nooit
    overschrijft. Dit toetst de daadwerkelijke `cache_mece_results`-aanroep
    binnen `run_codebook_v2`, niet alleen de `CACHE_STEP`-constante — een
    wijziging die `step=CACHE_STEP` uit die aanroep haalt zou de constante
    ongemoeid laten en `test_cache_step_is_separate_from_v1` laten slagen."""
    from pipeline.step_5_codeGenerator import run_codeGenerator as v1

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
    monkeypatch.setattr(runner, "generate_codebook_v2", lambda *a, **k: empty_result)
    monkeypatch.setattr(v1, "load_extraction_metadata", lambda *a, **k: FakeMetadata())
    monkeypatch.setattr(v1, "load_classified_ideas", lambda *a, **k: [])
    monkeypatch.setattr(v1, "load_taxonomy_cache", lambda *a, **k: FakeTaxonomy())
    monkeypatch.setattr(v1, "apply_overig_sweep", lambda codes, results, language: "Overig")
    monkeypatch.setattr(v1, "print_codebook_results", lambda codes: None)
    monkeypatch.setattr(v1, "run_scorecard", lambda *a, **k: None)
    monkeypatch.setattr(v1, "cache_mece_results", lambda *a, **k: captured.update(k))

    runner.run_codebook_v2(filename="f", var_name="v", sample_size=10, force_recalc=True)

    assert captured.get("step") == runner.CACHE_STEP
