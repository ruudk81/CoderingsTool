"""Group ideas whose instance text means the same thing, so per-idea phases run once.

WHY
    Step 3 costs per RESPONSE; steps 4 and 6 cost per IDEA. On a short-answer survey
    most ideas repeat: ASN Qd1 has 4833 ideas over 1658 distinct instance texts. P3,
    P6 and code assignment then pay 4833 times to answer the same question.

    Grouping is safe exactly where the consuming phase is blind: two ideas whose
    instance text carries the same meaning produce the same input, so deciding them
    separately is noise, not signal — that is what scattered one word over five
    domains before.

HOW
    The survey question is prepended to every instance before embedding. Without it
    the ranking breaks: measured on ASN, `vriendelijk`/`klantvriendelijk` (0.79, do
    NOT belong together) outranked `milieu`/`voor het milieu` (0.55, do). With it the
    order is correct.

    Ideas link to a group REPRESENTATIVE, not to each other. Single-link chaining put
    `duurzaam`, `groen`, `natuur`, `bank` and `goed` in one group of 2780 at 0.97.

THRESHOLD
    0.99, measured on ASN. The workable window is narrow — the lowest true pair sat
    at 0.9882 and the highest false pair at 0.9857 — so treat this as calibrated, not
    derived, and re-check it on a dataset unlike the ones it was set on. Too high
    fails safe: fewer groups, no wrong merges.

SCOPE
    Instance text only. Deduplicating on the abstraction rung collapses paraphrases in
    long answers, but on short answers it merges away real distinctions (it put
    "claimt een groene bank te zijn" with "duurzaam"). Out of scope here.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from utils.llm import create_embedding_client

EMBEDDING_MODEL = "text-embedding-3-large"
SIMILARITY_THRESHOLD = 0.99
EMBED_BATCH = 256


@dataclass
class DedupResult:
    """Which idea speaks for which, and what that saves."""

    # idea_id -> idea_id of the representative that will actually be processed
    representative_of: Dict[str, str] = field(default_factory=dict)
    # representative idea_id -> every idea_id it speaks for (including itself)
    members_of: Dict[str, List[str]] = field(default_factory=dict)
    # representative idea_id -> the distinct instance texts merged into it
    texts_of: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def n_ideas(self) -> int:
        return len(self.representative_of)

    @property
    def n_groups(self) -> int:
        return len(self.members_of)

    @property
    def saved_fraction(self) -> float:
        return 1.0 - (self.n_groups / self.n_ideas) if self.n_ideas else 0.0

    def summary(self) -> str:
        return (f"{self.n_ideas} ideas -> {self.n_groups} groups "
                f"({self.saved_fraction:.0%} fewer per-idea calls)")


def _norm(text: Optional[str]) -> str:
    return (text or "").strip()


async def _embed(texts: Sequence[str]) -> np.ndarray:
    client = create_embedding_client(async_mode=True)
    try:
        out: List[List[float]] = []
        for i in range(0, len(texts), EMBED_BATCH):
            resp = await client.embeddings.create(
                model=EMBEDDING_MODEL, input=list(texts[i:i + EMBED_BATCH])
            )
            out.extend(d.embedding for d in resp.data)
        return np.asarray(out, dtype=np.float32)
    finally:
        # close inside the loop that opened it; asyncio.run() tears the loop down
        # first otherwise, and httpx raises "Event loop is closed" into the log
        close = getattr(client, "close", None)
        if close is not None:
            maybe = close()
            if asyncio.iscoroutine(maybe):
                await maybe


def _group_texts(
    matrix: np.ndarray,
    threshold: float,
) -> List[int]:
    """Assign each text to a group, linking to a representative.

    The densest text becomes a representative and absorbs everything within
    `threshold` of IT. A member never absorbs further members, which is what stops
    A~B~C from putting A and C together when A and C are not alike.
    """
    n = matrix.shape[0]
    assigned = [-1] * n
    n_groups = 0
    for idx in np.argsort(-matrix.max(axis=1)):
        if assigned[idx] != -1:
            continue
        group = n_groups
        n_groups += 1
        assigned[idx] = group
        for other in np.where(matrix[idx] >= threshold)[0]:
            if assigned[other] == -1:
                assigned[other] = group
    return assigned


async def dedup_ideas_async(
    ideas: Iterable,
    survey_question: str = "",
    threshold: float = SIMILARITY_THRESHOLD,
) -> DedupResult:
    """Group ideas by instance meaning. Ideas without instance text stand alone."""
    ideas = list(ideas)
    result = DedupResult()

    by_text: Dict[str, List] = defaultdict(list)
    for idea in ideas:
        text = _norm(getattr(idea, "instance", ""))
        if text:
            by_text[text].append(idea)
        else:
            # nothing to compare on — never merged, never silently dropped
            iid = idea.idea_id
            result.representative_of[iid] = iid
            result.members_of[iid] = [iid]
            result.texts_of[iid] = []

    texts = sorted(by_text)
    if not texts:
        return result

    payload = [f"{survey_question} {t}".strip() if survey_question else t for t in texts]
    vectors = await _embed(payload)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    sim = vectors @ vectors.T
    np.fill_diagonal(sim, -1.0)

    assignment = _group_texts(sim, threshold)

    texts_by_group: Dict[int, List[str]] = defaultdict(list)
    for text, group in zip(texts, assignment):
        texts_by_group[group].append(text)

    for group, group_texts in texts_by_group.items():
        # the most frequent text speaks for the group — the representative idea is a
        # real idea, so downstream phases see genuine data, not a synthetic centroid
        group_texts.sort(key=lambda t: (-len(by_text[t]), t))
        members = [idea for t in group_texts for idea in by_text[t]]
        rep = by_text[group_texts[0]][0].idea_id
        result.members_of[rep] = [i.idea_id for i in members]
        result.texts_of[rep] = group_texts
        for idea in members:
            result.representative_of[idea.idea_id] = rep

    return result


def dedup_ideas(ideas: Iterable, survey_question: str = "",
                threshold: float = SIMILARITY_THRESHOLD) -> DedupResult:
    """Synchronous wrapper around dedup_ideas_async."""
    return asyncio.run(dedup_ideas_async(ideas, survey_question, threshold))


def apply_to_assignments(
    assignments: Dict[str, str],
    dedup: DedupResult,
) -> Dict[str, str]:
    """Spread a representative's assignment to every idea it speaks for.

    `assignments` maps idea_id -> whatever was decided (facet, attribute, code).
    Only representatives need to appear; members inherit. An idea whose
    representative was never decided is left out rather than guessed at.
    """
    out: Dict[str, str] = {}
    for rep, members in dedup.members_of.items():
        value = assignments.get(rep)
        if value is None:
            continue
        for iid in members:
            out[iid] = value
    # anything decided outside the dedup map (shouldn't happen) is preserved
    for iid, value in assignments.items():
        out.setdefault(iid, value)
    return out
