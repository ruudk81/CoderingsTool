"""Stap 2 — dispatch van de relatiecall. Stap 2b — verzamelnamen consolideren."""
from __future__ import annotations

import copy
from typing import List, Optional

from config import get_reasoning_params
from utils.llm import RateLimits
from utils.smoothRequester import SmoothRequester

from .concept_inventory import Concept
from .config_codeGenerator import CodebookConfig
from .prompts_relations import RelationsResult, build_relations_prompt, make_relations_model
from .prompts_umbrella_merge import (
    Umbrella, UmbrellaMergeResult, build_umbrella_merge_prompt, make_umbrella_merge_model,
)

PHASE = "step5_relations"
UMBRELLA_MERGE_PHASE = "step5_umbrella_merge"


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

    def prepare_fn(task):
        return {
            "prompt": build_relations_prompt(task["concepts"], task["language"]),
            "response_model": make_relations_model(task["concepts"]),
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
    tasks = [{"concepts": concepts, "language": language}]
    results = await requester.process_all(tasks, prepare_fn, parse_fn, fallback_fn)
    if not results or results[0] is None:
        raise RuntimeError(
            "Step 5 stap 2 (relaties) is mislukt. Zonder groepering is er geen "
            "codeboek — dit is een harde stop, geen fallback."
        )
    return results[0]


async def resolve_umbrella_merge(
    umbrellas: List[Umbrella],
    config: CodebookConfig,
    language: str,
    known_limits: Optional[RateLimits] = None,
    has_server_headers: Optional[bool] = None,
    verbose: bool = False,
) -> Optional[UmbrellaMergeResult]:
    """One call that consolidates step 2's umbrella names before pooling. A
    failed call means a finer-grained codebook (unconsolidated names), not a
    broken one — this returns None instead of raising, unlike resolve_relations."""

    def prepare_fn(task):
        return {
            "prompt": build_umbrella_merge_prompt(task["umbrellas"], task["language"]),
            "response_model": make_umbrella_merge_model(task["umbrellas"]),
            "temperature": config.temperature_umbrella_merge,
            "max_tokens": config.max_tokens_umbrella_merge,
            "max_retries": 2,
            "extra_kwargs": get_reasoning_params(
                config.model_umbrella_merge, phase="codegen_umbrella_merge"
            ),
        }

    def parse_fn(_task, response):
        return response

    def fallback_fn(_task, _reason):
        return None

    requester = SmoothRequester(
        model=config.model_umbrella_merge,
        phase_key=UMBRELLA_MERGE_PHASE,
        num_tasks=1,
        verbose=verbose,
        known_limits=known_limits,
        has_server_headers=has_server_headers,
        quiet=True,
    )
    tasks = [{"umbrellas": umbrellas, "language": language}]
    results = await requester.process_all(tasks, prepare_fn, parse_fn, fallback_fn)
    if not results or results[0] is None:
        return None
    return results[0]


def apply_umbrella_merge(relations_result, merge_result) -> RelationsResult:
    """Deterministic rewrite of `relations_result` with each merge group's
    members replaced by its canonical name/definition. Umbrella names that
    belong to no group are left untouched. Does not mutate its input."""
    canonical = {
        member: (group.canonical_name, group.canonical_definition)
        for group in merge_result.groups
        for member in group.members
    }

    def rewrite(relation):
        if relation.umbrella_name not in canonical:
            return relation
        name, definition = canonical[relation.umbrella_name]
        rewritten = copy.copy(relation)
        rewritten.umbrella_name = name
        rewritten.umbrella_definition = definition
        return rewritten

    merged = copy.copy(relations_result)
    merged.relations = [rewrite(relation) for relation in relations_result.relations]
    return merged
