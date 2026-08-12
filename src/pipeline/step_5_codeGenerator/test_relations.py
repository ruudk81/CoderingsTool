"""Tests voor de relatiedispatch (relations.py) — het contract met SmoothRequester.

process_all verwacht List[Dict]: _execute_task/_worker doen task.get(...) op elk
element. resolve_relations gaf ooit [None] door — dat crasht op None.get(...)
voordat er ook maar één LLM-call vertrekt. Deze test pint het echte contract door
process_all te stubben en het doorgegeven `tasks`-argument te inspecteren.
"""
import asyncio

from utils.smoothRequester import SmoothRequester

from pipeline.step_5_codeGenerator import relations
from pipeline.step_5_codeGenerator.concept_inventory import Concept
from pipeline.step_5_codeGenerator.config_codeGenerator import CodebookConfig
from pipeline.step_5_codeGenerator.prompts_relations import RelationsResult
from pipeline.step_5_codeGenerator.prompts_umbrella_merge import Umbrella


def concept(attribute_id, name):
    resp = frozenset({"R1"})
    return Concept(attribute_id=attribute_id, name=name, definition="def",
                   domain="Domein", facet="Facet", n_iu=1,
                   resp_ids=resp, resp_pos=resp,
                   resp_neg=frozenset(), resp_neu=frozenset())


def test_resolve_relations_sends_a_one_element_list_of_dicts(monkeypatch):
    captured = {}

    async def fake_process_all(self, tasks, prepare_fn, parse_fn, fallback_fn=None):
        captured["tasks"] = tasks
        # Exercise prepare_fn exactly as SmoothRequester's worker would, so a
        # task shape prepare_fn can't consume also fails this test.
        call_params = prepare_fn(tasks[0])
        assert "prompt" in call_params
        assert "response_model" in call_params
        return [RelationsResult(relations=[])]

    monkeypatch.setattr(SmoothRequester, "process_all", fake_process_all)

    concepts = [concept("A1", "Prijs")]
    result = asyncio.run(
        relations.resolve_relations(concepts, CodebookConfig(), "nl-NL")
    )

    tasks = captured["tasks"]
    assert isinstance(tasks, list)
    assert len(tasks) == 1
    assert isinstance(tasks[0], dict)
    assert result.relations == []


def test_resolve_relations_raises_when_the_call_fails(monkeypatch):
    async def fake_process_all(self, tasks, prepare_fn, parse_fn, fallback_fn=None):
        return [fallback_fn(tasks[0], "boom")]

    monkeypatch.setattr(SmoothRequester, "process_all", fake_process_all)

    concepts = [concept("A1", "Prijs")]
    try:
        asyncio.run(relations.resolve_relations(concepts, CodebookConfig(), "nl-NL"))
    except RuntimeError:
        return
    raise AssertionError("een mislukte call had een RuntimeError moeten geven")


def test_resolve_umbrella_merge_returns_none_when_the_call_fails_instead_of_raising(monkeypatch):
    # The one contract this dispatch function is meant to break with
    # resolve_relations: a failed call must NOT hard-stop the pipeline. A missed
    # cleanup gives a finer-grained codebook, not a broken one. This pins that a
    # refactor that copies resolve_relations's `raise RuntimeError(...)` in here
    # would break this test.
    async def fake_process_all(self, tasks, prepare_fn, parse_fn, fallback_fn=None):
        return [fallback_fn(tasks[0], "boom")]

    monkeypatch.setattr(SmoothRequester, "process_all", fake_process_all)

    umbrellas = [Umbrella(name="Bankdiensten", definition="def", member_names=("Betalen",))]
    result = asyncio.run(
        relations.resolve_umbrella_merge(umbrellas, CodebookConfig(), "nl-NL")
    )
    assert result is None
