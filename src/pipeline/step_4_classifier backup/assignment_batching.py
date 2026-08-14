"""Pure helpers for P4 batch facet assignment (no LLM, no I/O).

Batch mode assigns unique (domain, label) representatives in groups of K
against a shortlist menu; every helper here is deterministic and unit-tested.
The LLM plumbing lives in classifier.py; the response schema in
prompts_classifier.py (build_batch_facet_assignment_model).
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from pipeline.step_4_classifier.partition_labels import format_label


@dataclass
class LabelRep:
    """One unique normalized label with every idea instance that carries it."""
    label: str
    idea_ids: List[str]


def group_label_reps(ideas, label_source: str, label_prefix: str,
                     dedup: bool) -> List[LabelRep]:
    """Group ideas into one rep per normalized label (first-seen order).

    The rep keeps the first-seen original label text. Identical labels share
    one call and thus one facet/valence/confidence — consistent with the
    pipeline's block-move semantics for identical normalized text. Empty
    labels never merge (nothing to judge them equal on). dedup=False yields
    one rep per idea (byte-identical behavior lever).
    """
    reps: List[LabelRep] = []
    by_key: Dict[str, LabelRep] = {}
    for idea in ideas:
        label = format_label(idea, label_source, label_prefix)
        key = label.strip().lower()
        if dedup and key and key in by_key:
            by_key[key].idea_ids.append(idea.idea_id)
            continue
        rep = LabelRep(label=label, idea_ids=[idea.idea_id])
        reps.append(rep)
        if key:
            by_key[key] = rep
    return reps


def make_batches(count: int, k: int) -> List[List[int]]:
    """Split range(count) into consecutive index groups of at most k."""
    return [list(range(start, min(start + k, count)))
            for start in range(0, count, k)]


def shortlist_indices(label_vectors: np.ndarray, card_vectors: np.ndarray,
                      k: int) -> List[int]:
    """Union of each label's top-k most similar cards, as sorted indices.

    Cosine via normalized dot product; vectors need not be pre-normalized.
    """
    cards = card_vectors / np.linalg.norm(card_vectors, axis=1, keepdims=True)
    labels = label_vectors / np.linalg.norm(label_vectors, axis=1, keepdims=True)
    similarities = labels @ cards.T
    keep = set()
    for row in similarities:
        keep.update(np.argsort(-row)[:k].tolist())
    return sorted(keep)


def validate_batch_response(batch_ids: List[str],
                            response) -> Tuple[Dict[str, object], Dict[str, str]]:
    """Split a batch response into accepted items and escalations-with-reason.

    Accepted: the id appears exactly once with a real facet id. Escalation
    reasons: "missing" (id absent), "duplicate" (id appears 2+; all its items
    rejected), "f_none" (model says no facet fits).
    """
    seen: Dict[str, list] = {}
    for item in response.assignments:
        seen.setdefault(item.idea_id, []).append(item)
    ok: Dict[str, object] = {}
    escalate: Dict[str, str] = {}
    for batch_id in batch_ids:
        items = seen.get(batch_id, [])
        if not items:
            escalate[batch_id] = "missing"
        elif len(items) > 1:
            escalate[batch_id] = "duplicate"
        elif items[0].assigned_facet_id == "F_NONE":
            escalate[batch_id] = "f_none"
        else:
            ok[batch_id] = items[0]
    return ok, escalate


def facet_card_text(facet: dict) -> str:
    """The same content P4's menu renders per facet, joined for embedding."""
    parts = [facet.get("facet_name", ""), facet.get("facet_description", "")]
    if facet.get("inclusion_rule"):
        parts.append(facet["inclusion_rule"])
    examples = facet.get("example_observations") or []
    if examples:
        parts.append("; ".join(examples[:3]))
    return ". ".join(p for p in parts if p)
