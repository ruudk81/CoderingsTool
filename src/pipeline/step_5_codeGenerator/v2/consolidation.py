"""Fase 1 — dispatch van de consolidatiecall.

Faalcontract: hard. Zonder groepering is er geen codeboek, precies zoals
`resolve_relations` in v1 — dit is de enige fase in v2 zonder fallback.
"""
from __future__ import annotations

from typing import List

from config import get_reasoning_params
from utils.smoothRequester import SmoothRequester

from ..config_codeGenerator import CodebookConfig
from .attribute_cards import AttributeCard
from .prompts_consolidation import (
    ConsolidationResult, build_consolidation_prompt, make_consolidation_model,
)

PHASE = "step5_v2_consolidation"


async def resolve_consolidation(
    cards: List[AttributeCard],
    survey_question: str,
    n_respondents: int,
    language: str,
    config: CodebookConfig,
    verbose: bool = False,
    prompt_printer=None,
) -> ConsolidationResult:
    """Eén call over de hele attribuutinventaris."""

    def prepare_fn(task):
        prompt = build_consolidation_prompt(
            task["cards"], survey_question, n_respondents, language)
        if prompt_printer is not None:
            prompt_printer.capture_prompt(
                step_name="code_generator_v2",
                utility_name="resolve_consolidation",
                prompt_content=prompt,
                prompt_type="consolidation",
                metadata={
                    "model": config.model_relations,
                    "n_cards": len(task["cards"]),
                    "card_ids": [c.attribute_id for c in task["cards"]],
                    "language": language,
                },
            )
        return {
            "prompt": prompt,
            "response_model": make_consolidation_model(task["cards"]),
            "temperature": config.temperature_relations,
            "max_tokens": config.max_tokens_relations,
            "max_retries": 2,
            "extra_kwargs": get_reasoning_params(
                config.model_relations, phase="codegen_relations"),
        }

    requester = SmoothRequester(
        model=config.model_relations, phase_key=PHASE, num_tasks=1,
        verbose=verbose, quiet=True,
    )
    results = await requester.process_all(
        [{"cards": cards}], prepare_fn,
        lambda _task, response: response, lambda _task, _reason: None,
    )
    if not results or results[0] is None:
        raise RuntimeError(
            "Step 5 v2 fase 1 (consolidatie) is mislukt. Zonder groepering is er "
            "geen codeboek — dit is een harde stop, geen fallback."
        )
    return results[0]
