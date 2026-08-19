"""Tests voor de dispatch van de consolidatiecall."""
import asyncio
import pytest

from pipeline.step_5_codeGenerator.config_codeGenerator import CodebookConfig
from pipeline.step_5_codeGenerator.attribute_cards import AttributeCard
from pipeline.step_5_codeGenerator import consolidation


def card(attribute_id, name):
    return AttributeCard(attribute_id=attribute_id, name=name, definition="d",
                         domain="D", facet="F", n_resp=10, top_answers=(("x", 1),))


CARDS = [card("A1", "Een"), card("A2", "Twee")]


class _FakeRequester:
    """Vervangt SmoothRequester: legt de prepare_fn-uitkomst vast en geeft
    `canned` terug, zodat de dispatch getest wordt zonder LLM."""
    captured = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _FakeRequester.captured = self

    async def process_all(self, tasks, prepare_fn, parse_fn, fallback_fn):
        self.prepared = prepare_fn(tasks[0])
        return [self.canned]


def test_prepared_call_carries_prompt_model_and_reasoning_params(monkeypatch):
    _FakeRequester.canned = object()
    monkeypatch.setattr(consolidation, "SmoothRequester", _FakeRequester)

    asyncio.run(consolidation.resolve_consolidation(
        CARDS, "V?", 100, "Dutch", CodebookConfig()))

    prepared = _FakeRequester.captured.prepared
    assert "V?" in prepared["prompt"]
    assert prepared["max_retries"] == 2
    assert "extra_kwargs" in prepared
    assert _FakeRequester.captured.kwargs["phase_key"] == "step5_consolidation"


def test_failed_call_is_a_hard_stop(monkeypatch):
    """Zonder groepering is er geen codeboek — geen fallback, net als
    resolve_relations in v1."""
    _FakeRequester.canned = None
    monkeypatch.setattr(consolidation, "SmoothRequester", _FakeRequester)

    with pytest.raises(RuntimeError, match="consolidatie"):
        asyncio.run(consolidation.resolve_consolidation(
            CARDS, "V?", 100, "Dutch", CodebookConfig()))
