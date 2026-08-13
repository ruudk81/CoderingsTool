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
    prompt_printer=None,
) -> RelationsResult:
    """One call across the whole concept inventory. If it fails there is no
    grouping, so step 5 stops here — no fallback."""

    def prepare_fn(task):
        prompt = build_relations_prompt(task["concepts"], task["language"])
        if prompt_printer is not None:
            prompt_printer.capture_prompt(
                step_name="code_generator",
                utility_name="resolve_relations",
                prompt_content=prompt,
                prompt_type="relations",
                metadata={
                    "model": config.model_relations,
                    "temperature": config.temperature_relations,
                    "max_tokens": config.max_tokens_relations,
                    "language": task["language"],
                    "n_concepts": len(task["concepts"]),
                    "concept_ids": [c.attribute_id for c in task["concepts"]],
                    "concept_names": [c.name for c in task["concepts"]],
                },
            )
        return {
            "prompt": prompt,
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
    known_limits: Optional[RateLimits] = None,
    has_server_headers: Optional[bool] = None,
    verbose: bool = False,
    prompt_printer=None,
) -> Optional[UmbrellaMergeResult]:
    """One call that asks, for every umbrella name, whether another name in the
    list means the same thing — a per-item lookup, not a partitioning question
    (see prompts_umbrella_merge.py for why). A failed call means a finer-grained
    codebook (unconsolidated names), not a broken one — this returns None
    instead of raising, unlike resolve_relations."""

    def prepare_fn(task):
        prompt = build_umbrella_merge_prompt(task["umbrellas"])
        if prompt_printer is not None:
            prompt_printer.capture_prompt(
                step_name="code_generator",
                utility_name="resolve_umbrella_merge",
                prompt_content=prompt,
                prompt_type="umbrella_merge",
                metadata={
                    "model": config.model_umbrella_merge,
                    "temperature": config.temperature_umbrella_merge,
                    "max_tokens": config.max_tokens_umbrella_merge,
                    "n_umbrellas": len(task["umbrellas"]),
                    "umbrella_names": [u.name for u in task["umbrellas"]],
                },
            )
        return {
            "prompt": prompt,
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
    tasks = [{"umbrellas": umbrellas}]
    results = await requester.process_all(tasks, prepare_fn, parse_fn, fallback_fn)
    if not results or results[0] is None:
        return None
    return results[0]


def _umbrella_groups(verdicts) -> List[set]:
    """Connected components over `same_as` verdicts (union-find): a chain
    A same_as B, B same_as C collapses into one group regardless of verdict
    order. Umbrella names with no `same_as` link form a group of size one."""
    parent: dict = {}

    def find(name):
        parent.setdefault(name, name)
        root = name
        while parent[root] != root:
            root = parent[root]
        while parent[name] != root:
            parent[name], name = root, parent[name]
        return root

    def union(a, b):
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_a] = root_b

    for verdict in verdicts:
        find(verdict.umbrella)
        if verdict.same_as is not None:
            union(verdict.umbrella, verdict.same_as)

    groups: dict = {}
    for name in parent:
        groups.setdefault(find(name), set()).add(name)
    return list(groups.values())


def _canonical_name(members: set, attribute_counts: dict) -> str:
    """Most attributes wins; ties broken by shortest name, then alphabetically.
    Counting is fine here — this runs in code, not in a prompt."""
    return min(members, key=lambda name: (-attribute_counts.get(name, 0), len(name), name))


def apply_umbrella_merge(relations_result, merge_result) -> RelationsResult:
    """Deterministic rewrite of `relations_result`: umbrella names the verdicts
    tie together (directly, or through a same_as chain) are rewritten to one
    canonical name per group — picked in code from the group's own existing
    names and definitions, never authored by the model. Umbrella names in no
    group of more than one are left untouched. Does not mutate its input."""
    attribute_counts: dict = {}
    definitions: dict = {}
    for relation in relations_result.relations:
        attribute_counts[relation.umbrella_name] = attribute_counts.get(relation.umbrella_name, 0) + 1
        definitions.setdefault(relation.umbrella_name, relation.umbrella_definition)

    canonical_for = {}
    for members in _umbrella_groups(merge_result.verdicts):
        if len(members) <= 1:
            continue
        canonical = _canonical_name(members, attribute_counts)
        for member in members:
            canonical_for[member] = canonical

    def rewrite(relation):
        canonical = canonical_for.get(relation.umbrella_name)
        if canonical is None:
            return relation
        rewritten = copy.copy(relation)
        rewritten.umbrella_name = canonical
        rewritten.umbrella_definition = definitions[canonical]
        return rewritten

    merged = copy.copy(relations_result)
    merged.relations = [rewrite(relation) for relation in relations_result.relations]
    return merged
