"""Tests voor de v2-keten: volgorde van de fasen en de cachecontracten."""
from pipeline.step_5_codeGenerator.concept_inventory import Concept
from pipeline.step_5_codeGenerator.v2 import run_codebook_v2 as runner


def concept(attribute_id, name, pos=0, neg=0):
    def resp(prefix, n):
        return frozenset(f"{attribute_id}{prefix}{i}" for i in range(n))
    p, g = resp("P", pos), resp("G", neg)
    return Concept(attribute_id=attribute_id, name=name, definition="d",
                   domain="D", facet="F", n_iu=pos + neg,
                   resp_ids=p | g, resp_pos=p, resp_neg=g, resp_neu=frozenset())


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
        duplicate_definitions=[], concept_by_id={},
    )

    runner.report_codebook_build_v2(result)

    assert "DEGENERATIE" in capsys.readouterr().out


def test_direction_loss_is_reported_when_nonzero(capsys):
    result = runner.GeneratedCodebookV2(
        shapes=[], overig_ids=[], codes=[], direction_loss=42, degeneration=None,
        partition_repairs=[], collisions=[], naming_mismatches=[],
        duplicate_definitions=[], concept_by_id={},
    )

    runner.report_codebook_build_v2(result)

    assert "42" in capsys.readouterr().out
