"""Pure helper for assignment input (no LLM, no I/O).

Ideas are grouped into one representative per unique normalized label, so
identical text is judged once. Deterministic and unit-tested; the LLM plumbing
lives in classifier.py and the response schema in prompts_assignment.py.

This module used to also batch reps in groups of K, validate a batch response,
and shortlist the menu by embedding similarity. All three are gone with the
single-answer assignment: there is no batch to validate, and a shortlisted menu
would make the catch-all the easy way out for an attribute that was merely
trimmed away.
"""
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass
class LabelRep:
    """One unique normalized label with every idea instance that carries it."""
    label: str
    idea_ids: List[str]


def group_label_reps(items: Iterable[Tuple[str, str]]) -> List[LabelRep]:
    """Group (idea_id, label) pairs into one rep per normalized label.

    First-seen order, and the rep keeps the first-seen original label text.
    Identical labels share one call and thus one facet/valence/confidence —
    consistent with the pipeline's block-move semantics for identical normalized
    text. Empty labels never merge: there is nothing to judge them equal on.

    Takes rendered labels rather than idea objects so the caller that builds
    assignment tasks can be a pure function of plain data, and so both
    assignment levels feed it the same way.
    """
    reps: List[LabelRep] = []
    by_key: Dict[str, LabelRep] = {}
    for idea_id, label in items:
        label = label or ""
        key = label.strip().lower()
        if key and key in by_key:
            by_key[key].idea_ids.append(idea_id)
            continue
        rep = LabelRep(label=label, idea_ids=[idea_id])
        reps.append(rep)
        if key:
            by_key[key] = rep
    return reps
