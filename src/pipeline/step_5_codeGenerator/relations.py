"""Stap 2 — dispatch van de relatiecall."""
from __future__ import annotations

from typing import List, Optional

from config import get_reasoning_params
from utils.llm import RateLimits
from utils.smoothRequester import SmoothRequester

from .concept_inventory import Concept
from .config_codeGenerator import CodebookConfig
from .prompts_relations import RelationsResult, build_relations_prompt, make_relations_model

PHASE = "step5_relations"


async def resolve_relations(
    concepts: List[Concept],
    config: CodebookConfig,
    language: str,
    known_limits: Optional[RateLimits] = None,
    has_server_headers: Optional[bool] = None,
    verbose: bool = False,
) -> RelationsResult:
    """One call across the whole concept inventory. If it fails there is no
    grouping, so step 5 stops here — no fallback."""
    model = make_relations_model(concepts)
    prompt = build_relations_prompt(concepts, language)

    def prepare_fn(_task):
        return {
            "prompt": prompt,
            "response_model": model,
            "temperature": config.temperature_relations,
            "max_tokens": config.max_tokens_relations,
            "max_retries": 2,
            "extra_kwargs": get_reasoning_params(config.model_relations, phase="codegen_relations"),
        }

    def parse_fn(_task, response):
        return response

    def fallback_fn(_task, _reason):
        return None

    requester = SmoothRequester(
        model=config.model_relations,
        phase_key=PHASE,
        num_tasks=1,
        verbose=verbose,
        known_limits=known_limits,
        has_server_headers=has_server_headers,
        quiet=True,
    )
    results = await requester.process_all([None], prepare_fn, parse_fn, fallback_fn)
    if not results or results[0] is None:
        raise RuntimeError(
            "Step 5 stap 2 (relaties) is mislukt. Zonder groepering is er geen "
            "codeboek — dit is een harde stop, geen fallback."
        )
    return results[0]
