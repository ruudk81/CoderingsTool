"""Narrow LLM judgment layer for the step-5 experiment (phases 2-4: membership,
noise, naming), plus a generic 3-vote helper.

`vote()` is the only orchestration logic here: it fires `n_votes` independent
LLM calls (one per `vote_idx`, each with its own deterministically-shuffled
evidence order for diversity), tolerates per-vote failures, and reduces the
survivors to a majority via an injectable key function. It takes `llm_call`
as a plain async callable so tests can inject fakes — no real LLM call, no
API key, ever touches the test suite.

`make_llm_call()` is the production `llm_call` factory: it follows the
standard create_client + llm_create_async + get_reasoning_params pattern of
utils/llm.py, as used by utils/smoothRequester.py (and directly by scripts
such as step_4's view_consolidation_ab.py), with its own bounded retry per
vote.
"""
from __future__ import annotations

import asyncio
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Type

from pydantic import BaseModel, Field

SCHEMA_HINT = "Provide your output as valid JSON following the response schema provided."


# =============================================================================
# Response models
# =============================================================================
class MembershipVote(BaseModel):
    choice: Literal["A", "B"] = Field(description="Which group the attribute belongs to.")
    reason: str = Field(description="Short justification for the choice.")


class NoiseVote(BaseModel):
    genuine_opposition: bool = Field(
        description="True if the positive/negative statements express genuine "
                    "opposition on the same dimension; False if the split is noise.")
    reason: str = Field(description="Short justification for the judgment.")


class CodeNaming(BaseModel):
    code_name: str = Field(description="Short, plain-language name for the code.")
    definition: str = Field(description="One-sentence definition of what the code covers.")
    diagnostic_test: str = Field(description="A test a coder can apply to decide membership.")
    typical_indicators: List[str] = Field(description="Typical indicator phrases for this code.")


# =============================================================================
# Vote outcome + generic 3-vote helper
# =============================================================================
@dataclass
class VoteOutcome:
    votes: List[Any]
    majority: Optional[Any]
    unanimous: bool
    failed: int


async def vote(
    build_prompt: Callable[[int], str],
    response_model: Type[BaseModel],
    llm_call: Callable[[str, Type[BaseModel]], Any],
    majority_key: Callable[[Any], Any],
    n_votes: int = 3,
) -> VoteOutcome:
    """Fire `n_votes` independent votes concurrently and reduce to a majority.

    Args:
        build_prompt: builds the prompt for a given vote_idx (0..n_votes-1);
            responsible for its own evidence-order shuffle per vote_idx.
        response_model: Pydantic model each vote must resolve to.
        llm_call: async (prompt, response_model) -> instance. Injectable —
            tests pass fakes; production passes `make_llm_call(...)`.
        majority_key: maps a successful vote's response to the (hashable)
            value the majority is computed over (e.g. `lambda v: v.choice`
            for `MembershipVote`, `lambda v: v.genuine_opposition` for
            `NoiseVote`). Required — the response models this module defines
            are Pydantic models, which are unhashable, so there is no safe
            identity-function default: computing `Counter` over raw instances
            would raise `TypeError` before a single vote could be tallied.
        n_votes: number of independent votes to fire (default 3).

    Returns:
        VoteOutcome with the successful vote instances, the majority value
        (None if every vote failed), whether it was unanimous, and the count
        of failed votes.
    """
    async def _one(vote_idx: int):
        prompt = build_prompt(vote_idx)
        return await llm_call(prompt, response_model)

    raw = await asyncio.gather(
        *(_one(i) for i in range(n_votes)), return_exceptions=True)

    votes: List[Any] = [r for r in raw if not isinstance(r, Exception)]
    failed = len(raw) - len(votes)

    if not votes:
        return VoteOutcome(votes=[], majority=None, unanimous=False, failed=failed)

    counts = Counter(majority_key(v) for v in votes)
    majority_value, majority_count = counts.most_common(1)[0]
    unanimous = majority_count == len(votes)
    return VoteOutcome(votes=votes, majority=majority_value, unanimous=unanimous, failed=failed)


# =============================================================================
# Evidence-order shuffle (stem-diversiteit)
# =============================================================================
def _shuffled(items: List[Any], vote_idx: int) -> List[Any]:
    """Deterministic per-vote shuffle of an evidence list — same items, an
    order that depends only on vote_idx so repeated builds are reproducible."""
    copy = list(items)
    random.Random(vote_idx).shuffle(copy)
    return copy


# =============================================================================
# Prompt builders
# =============================================================================
def membership_prompt(
    attr: str,
    definition: str,
    samples: List[str],
    group_a: str,
    group_b: str,
    language: str,
    vote_idx: int,
) -> str:
    """Prompt: does `attr` belong to Group A or Group B, given evidence samples."""
    samples_block = "\n".join(f"- {s}" for s in _shuffled(samples, vote_idx))
    return (
        "You are classifying which of two phenomenon groups an attribute belongs to.\n\n"
        f"Attribute: {attr}\n"
        f"Definition: {definition}\n\n"
        f"Group A: {group_a}\n"
        f"Group B: {group_b}\n\n"
        "Evidence (verbatim respondent statements assigned to this attribute):\n"
        f"{samples_block}\n\n"
        "Decide whether this attribute belongs to Group A or Group B based on the "
        "evidence above. Choose the group whose meaning best matches what the "
        "evidence expresses.\n\n"
        f"All output MUST be in {language}.\n\n"
        f"{SCHEMA_HINT}"
    )


def noise_prompt(
    phenomenon_desc: str,
    pos_texts: List[str],
    neg_texts: List[str],
    language: str,
    vote_idx: int,
) -> str:
    """Prompt: is the positive/negative split genuine opposition, or noise."""
    pos_block = "\n".join(f"- {t}" for t in _shuffled(pos_texts, vote_idx))
    neg_block = "\n".join(f"- {t}" for t in _shuffled(neg_texts, vote_idx))
    return (
        "You are checking whether a phenomenon shows genuine opposing positions, "
        "or whether the apparent split is noise.\n\n"
        f"Phenomenon: {phenomenon_desc}\n\n"
        f"Positive-valence statements:\n{pos_block}\n\n"
        f"Negative-valence statements:\n{neg_block}\n\n"
        "Decide: do the positive and negative statements express genuinely "
        "opposing views on the same underlying dimension (true opposition), or "
        "is the apparent split noise (e.g. miscoded outliers, unrelated remarks)?\n\n"
        f"All output MUST be in {language}.\n\n"
        f"{SCHEMA_HINT}"
    )


def naming_prompt(
    members: List[str],
    samples_per_pole: Dict[str, List[str]],
    valence: str,
    language: str,
    survey_question: str,
    avoid_names: List[str],
    vote_idx: int = 0,
) -> str:
    """Prompt: name a code covering `members`, with sample evidence per member."""
    # Only the top-level evidence order (which member's block comes first)
    # is shuffled per vote_idx; a member's own sample list stays in its given
    # order so each line's content — not just its position — stays stable.
    ordered_members = _shuffled(members, vote_idx)
    lines = []
    for m in ordered_members:
        sample_block = "; ".join(f'"{s}"' for s in samples_per_pole.get(m, []))
        lines.append(f"- {m}: {sample_block}")
    members_block = "\n".join(lines)
    avoid_block = ", ".join(avoid_names) if avoid_names else "(none)"
    return (
        "You are naming a code for a codebook derived from open-ended survey "
        "responses.\n\n"
        f"Survey question: {survey_question}\n"
        f"Valence: {valence}\n\n"
        "The code groups these attributes with representative respondent "
        f"statements:\n{members_block}\n\n"
        "Avoid these existing code names (must not duplicate or closely resemble "
        f"them): {avoid_block}\n\n"
        "Provide a short, plain-language code name, a one-sentence definition, "
        "a diagnostic test a coder can apply to decide membership, and a list "
        "of typical indicator phrases.\n\n"
        f"All output MUST be in {language}.\n\n"
        f"{SCHEMA_HINT}"
    )


# =============================================================================
# Production llm_call factory (wraps utils/llm.py)
# =============================================================================
def make_llm_call(
    model_key: str,
    phase: str,
    retries: int = 2,
    temperature: float = 0.0,
    max_tokens: int = 2000,
) -> Callable[[str, Type[BaseModel]], Any]:
    """Build the production `llm_call(prompt, response_model)` for `vote()`.

    Follows the standard client-creation / llm_create_async / get_reasoning_params
    pattern of utils/llm.py (the same one utils/smoothRequester.py wraps for
    its task dispatch, and that step_4_classifier/view_consolidation_ab.py
    calls directly). `model_key` resolves the model via `get_step_model()`
    (e.g. "code_assignment" for votes,
    "codegen_p8" for naming); `phase` is forwarded to `get_reasoning_params()`
    for per-step verbosity. A vote that keeps failing after `retries`
    attempts re-raises its last exception — `vote()` counts it as failed.
    """
    from config import get_reasoning_params, get_step_model
    from utils.llm import create_client, llm_create_async

    model = get_step_model(model_key)
    client = create_client(model, async_mode=True)

    async def _call(prompt: str, response_model: Type[BaseModel]) -> Any:
        last_exc: Optional[Exception] = None
        for _ in range(retries):
            try:
                return await llm_create_async(
                    client=client,
                    model=model,
                    prompt=prompt,
                    response_model=response_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **get_reasoning_params(model, phase=phase),
                )
            except Exception as exc:  # noqa: BLE001 - retried and re-raised below
                last_exc = exc
        if last_exc is None:
            raise RuntimeError(f"make_llm_call: no attempts made (retries={retries})")
        raise last_exc

    return _call
