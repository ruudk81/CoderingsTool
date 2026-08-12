"""Pure helpers for batched assignment (no LLM, no I/O).

Used by both assignment phases: ideas are grouped into one representative per
unique normalized label, reps are batched in groups of K, and the menu can be
shortlisted by embedding similarity. Every helper here is deterministic and
unit-tested. The LLM plumbing lives in classifier.py; the response schemas in
prompts_facet.py and prompts_attribute.py.
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


def group_label_reps(ideas, label_source: str, label_prefix: str) -> List[LabelRep]:
    """Group ideas into one rep per normalized label (first-seen order).

    The rep keeps the first-seen original label text. Identical labels share
    one call and thus one facet/valence/confidence — consistent with the
    pipeline's block-move semantics for identical normalized text. Empty labels
    never merge: there is nothing to judge them equal on.
    """
    reps: List[LabelRep] = []
    by_key: Dict[str, LabelRep] = {}
    for idea in ideas:
        label = format_label(idea, label_source, label_prefix)
        key = label.strip().lower()
        if key and key in by_key:
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


def validate_batch_response(
    batch_ids: List[str],
    response,
    *,
    id_field: str = "assigned_facet_id",
    none_id: str = "F_NONE",
) -> Tuple[Dict[str, object], Dict[str, str]]:
    """Split a batch response into accepted items and escalations-with-reason.

    Accepted: the id appears exactly once with a real menu id. Escalation
    reasons: "missing" (id absent), "duplicate" (id appears 2+; all its items
    rejected), "none" (model says nothing on the menu fits).

    `id_field` and `none_id` differ per level — the facet level answers with
    `assigned_facet_id` / "F_NONE", the attribute level with
    `assigned_attribute_id` / "A_NONE". Parameters rather than a second copy of
    the function: the routing logic is identical and should stay that way.
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
        elif getattr(items[0], id_field) == none_id:
            escalate[batch_id] = "none"
        else:
            ok[batch_id] = items[0]
    return ok, escalate


def _card_text(item: dict, name_key: str, definition_key: str) -> str:
    """Join the four boundary fields plus examples, for embedding.

    Mirrors what the menu renders. The shortlist compares labels against these
    cards, so a card that leaves out what the menu shows would rank on
    different evidence than the model decides on.
    """
    parts = [item.get(name_key, ""), item.get(definition_key, "")]
    if item.get("boundary_test"):
        parts.append(item["boundary_test"])
    exclusions = item.get("exclusions") or []
    if exclusions:
        parts.append("; ".join(exclusions))
    examples = item.get("example_observations") or []
    if examples:
        parts.append("; ".join(examples[:3]))
    return ". ".join(p for p in parts if p)


def facet_card_text(facet: dict) -> str:
    """The content the facet menu renders per facet, joined for embedding."""
    return _card_text(facet, "facet_name", "facet_definition")


def attribute_card_text(attribute: dict) -> str:
    """The content the attribute menu renders per attribute, joined for embedding."""
    return _card_text(attribute, "attribute_name", "attribute_definition")
