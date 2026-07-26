"""
Deterministic direction-assignment logic for codebook generation (phase 3).

Pure module: no I/O, no external imports except standard library.
Implements the decision tree for directional vs. neutral coding.
"""

import math
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class DirectionDecision:
    """Outcome of the deterministic direction assignment."""

    attributes: List[str]
    pos: int
    neu: int
    neg: int
    total: int
    floor: int
    outcome: str  # 'dimensional', 'too_small', or 'needs_noise_check'
    neutral_third: bool
    dominant_pole: str  # 'positive' or 'negative'


def resolve_direction(
    attrs: List[str],
    attr_valence: Dict[str, Dict[str, int]],
    total_assigned: int,
) -> DirectionDecision:
    """
    Resolve the direction assignment for a set of attributes.

    Args:
        attrs: List of attribute names.
        attr_valence: Dict mapping each attribute name to {"positive": int, "neutral": int, "negative": int}.
        total_assigned: The total number of assigned ideas (population).

    Returns:
        DirectionDecision with outcome, floor, neutral_third, and dominant_pole.
    """
    # Aggregate valence across all attributes
    pos = sum(attr_valence.get(a, {}).get("positive", 0) for a in attrs)
    neu = sum(attr_valence.get(a, {}).get("neutral", 0) for a in attrs)
    neg = sum(attr_valence.get(a, {}).get("negative", 0) for a in attrs)
    total = pos + neu + neg

    # Compute sample floor: max(2, int(log(total)))
    floor = max(2, int(math.log(total))) if total > 0 else 2

    # Compute population floor: max(2, int(log(total_assigned)))
    pop_floor = max(2, int(math.log(total_assigned))) if total_assigned > 0 else 2

    # Determine outcome
    both_poles_above_floor = pos >= floor and neg >= floor

    if both_poles_above_floor:
        # Both poles clear the floor
        if min(pos, neg) < pop_floor:
            outcome = "too_small"
        else:
            outcome = "needs_noise_check"
    else:
        # Not both poles clear the floor
        outcome = "dimensional"

    # Determine neutral_third: only when outcome='needs_noise_check' and neu/total >= 0.30
    neutral_third = False
    if outcome == "needs_noise_check" and total > 0:
        if neu / total >= 0.30:
            neutral_third = True

    # Determine dominant_pole: 'positive' if pos >= neg, else 'negative'
    dominant_pole = "positive" if pos >= neg else "negative"

    return DirectionDecision(
        attributes=attrs,
        pos=pos,
        neu=neu,
        neg=neg,
        total=total,
        floor=floor,
        outcome=outcome,
        neutral_third=neutral_third,
        dominant_pole=dominant_pole,
    )


def codes_for(decision: DirectionDecision, split: bool) -> List[dict]:
    """
    Generate code distribution plan based on direction decision.

    Args:
        decision: The DirectionDecision from resolve_direction.
        split: If False, all ideas coded as neutral. If True, distribute by polarity.

    Returns:
        List of dicts {"valence": str, "expected": int}, ordered as:
        - split=False: [{"valence": "neutral", "expected": total}]
        - split=True, no neutral_third: [positive, negative] (always this order);
          dominant pool absorbs the neutrals (expected tuning only)
        - split=True, with neutral_third: [positive, neutral, negative] (always this order)
    """
    if not split:
        # All as neutral
        return [{"valence": "neutral", "expected": decision.total}]

    if decision.neutral_third:
        # Three buckets: positive, neutral, negative (in this order)
        return [
            {"valence": "positive", "expected": decision.pos},
            {"valence": "neutral", "expected": decision.neu},
            {"valence": "negative", "expected": decision.neg},
        ]

    # split=True, no neutral_third: dominant pole gets the neutrals
    # Order is always [positive, negative] regardless of dominance
    if decision.dominant_pole == "positive":
        return [
            {"valence": "positive", "expected": decision.pos + decision.neu},
            {"valence": "negative", "expected": decision.neg},
        ]
    else:  # dominant_pole == "negative"
        return [
            {"valence": "positive", "expected": decision.pos},
            {"valence": "negative", "expected": decision.neg + decision.neu},
        ]
