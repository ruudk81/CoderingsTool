"""
Test suite for direction_rules.py (TDD approach).

Cases a-f from task-3-brief.md, plus edge case: total=0.
"""

import math

import pytest
from pipeline.step_5_codeGenerator_experiment.direction_rules import (
    resolve_direction,
    codes_for,
)


class TestResolveDirection:
    """Test the deterministic direction-assignment logic."""

    def test_case_a_needs_noise_check_no_neutral_third(self):
        """Case (a): +88/○6/−17, n=111 → needs_noise_check, no neutral_third."""
        attrs = ["a"]
        attr_valence = {"a": {"positive": 88, "neutral": 6, "negative": 17}}
        total_assigned = 111

        decision = resolve_direction(attrs, attr_valence, total_assigned)

        assert decision.attributes == ["a"]
        assert decision.pos == 88
        assert decision.neu == 6
        assert decision.neg == 17
        assert decision.total == 111
        assert decision.outcome == "needs_noise_check"
        assert decision.neutral_third is False
        assert decision.dominant_pole == "positive"
        # floor = max(2, int(log(111))) = max(2, 4) = 4
        assert decision.floor == 4

    def test_case_b_dimensional(self):
        """Case (b): +67/○0/−2 → dimensional."""
        attrs = ["b"]
        attr_valence = {"b": {"positive": 67, "neutral": 0, "negative": 2}}
        total_assigned = 69

        decision = resolve_direction(attrs, attr_valence, total_assigned)

        assert decision.pos == 67
        assert decision.neu == 0
        assert decision.neg == 2
        assert decision.total == 69
        assert decision.outcome == "dimensional"
        assert decision.dominant_pole == "positive"

    def test_case_c_too_small_minority_below_pop_floor(self):
        """Case (c): +2/−4, total_assigned=4833 → too_small (minority < pop_floor=8)."""
        attrs = ["c"]
        attr_valence = {"c": {"positive": 2, "neutral": 0, "negative": 4}}
        total_assigned = 4833

        decision = resolve_direction(attrs, attr_valence, total_assigned)

        assert decision.pos == 2
        assert decision.neu == 0
        assert decision.neg == 4
        assert decision.total == 6
        assert decision.outcome == "too_small"
        # pop_floor = max(2, int(log(4833))) = max(2, 8) = 8
        # min(pos, neg) = min(2, 4) = 2 < 8 ✓
        assert decision.floor == 2  # floor for total=6: max(2, int(log(6))) = max(2, 1) = 2
        assert decision.dominant_pole == "negative"

    def test_case_d_needs_noise_check_with_neutral_third(self):
        """Case (d): +100/○180/−60 → needs_noise_check and neutral_third=True (180/340≥0.30)."""
        attrs = ["d"]
        attr_valence = {"d": {"positive": 100, "neutral": 180, "negative": 60}}
        total_assigned = 340

        decision = resolve_direction(attrs, attr_valence, total_assigned)

        assert decision.pos == 100
        assert decision.neu == 180
        assert decision.neg == 60
        assert decision.total == 340
        assert decision.outcome == "needs_noise_check"
        assert decision.neutral_third is True
        # 180 / 340 ≈ 0.529 >= 0.30 ✓
        assert decision.dominant_pole == "positive"
        # floor = max(2, int(log(340))) = max(2, 5) = 5
        assert decision.floor == 5

    def test_case_e_floor_formula(self):
        """Case (e): verify floor = max(2, int(log(n))) via resolve_direction."""
        # Test spot cases: floor = max(2, int(log(n)))
        test_cases = [
            (7, 2),      # log(7)≈1.94, int=1, max(2, 1)=2
            (111, 4),    # log(111)≈4.71, int=4, max(2, 4)=4
            (4833, 8),   # log(4833)≈8.48, int=8, max(2, 8)=8
            (340, 5),    # log(340)≈5.82, int=5, max(2, 5)=5
        ]

        for total, expected_floor in test_cases:
            attrs = ["test"]
            attr_valence = {"test": {"positive": total // 2, "neutral": 0, "negative": total // 2}}
            decision = resolve_direction(attrs, attr_valence, total)
            assert decision.floor == expected_floor, f"n={total}: expected floor={expected_floor}, got {decision.floor}"

    def test_case_f_codes_for_split_scenarios(self):
        """Case (f): codes_for distributes expected correctly in all three scenarios."""
        # Scenario 1: split=False → all neutral
        attrs = ["x"]
        attr_valence = {"x": {"positive": 50, "neutral": 30, "negative": 20}}
        decision_a = resolve_direction(attrs, attr_valence, 100)

        codes_no_split = codes_for(decision_a, split=False)
        assert len(codes_no_split) == 1
        assert codes_no_split[0] == {"valence": "neutral", "expected": 100}

        # Scenario 2: split=True, no neutral_third → dominant gets neutrals
        attrs = ["y"]
        attr_valence = {"y": {"positive": 80, "neutral": 10, "negative": 10}}
        decision_b = resolve_direction(attrs, attr_valence, 100)
        # neutral_third check: outcome='needs_noise_check', total>0, neu/total=10/100=0.10 < 0.30 → False

        codes_split_no_third = codes_for(decision_b, split=True)
        assert decision_b.neutral_third is False
        # dominant_pole='positive', so positive gets 80+10=90, negative gets 10
        assert len(codes_split_no_third) == 2
        # Order: positive first, then negative
        assert codes_split_no_third[0] == {"valence": "positive", "expected": 90}
        assert codes_split_no_third[1] == {"valence": "negative", "expected": 10}

        # Scenario 3: split=True, with neutral_third → three buckets
        attrs = ["z"]
        attr_valence = {"z": {"positive": 100, "neutral": 180, "negative": 60}}
        decision_c = resolve_direction(attrs, attr_valence, 340)
        # neutral_third check: outcome='needs_noise_check', total>0, neu/total=180/340≈0.529 >= 0.30 → True

        codes_split_with_third = codes_for(decision_c, split=True)
        assert len(codes_split_with_third) == 3
        assert codes_split_with_third[0] == {"valence": "positive", "expected": 100}
        assert codes_split_with_third[1] == {"valence": "neutral", "expected": 180}
        assert codes_split_with_third[2] == {"valence": "negative", "expected": 60}

    def test_case_f_codes_for_dominant_negative(self):
        """Case (f) variant: codes_for with dominant_pole='negative'.

        Order is ALWAYS [positive, negative], but dominant negative absorbs neutrals.
        """
        attrs = ["n"]
        # Use values where neg > pos and neu/total < 0.30 (e.g., neu=8/100=0.08)
        attr_valence = {"n": {"positive": 20, "neutral": 8, "negative": 50}}
        decision = resolve_direction(attrs, attr_valence, 78)
        # dominant_pole should be 'negative' (50 > 20)

        codes_split = codes_for(decision, split=True)
        # Check that split=True without neutral_third preserves [positive, negative] order
        assert decision.neutral_third is False
        assert len(codes_split) == 2
        # Order: positive first, negative gets neutrals
        assert codes_split[0] == {"valence": "positive", "expected": 20}
        assert codes_split[1] == {"valence": "negative", "expected": 50 + 8}

    def test_edge_case_total_zero(self):
        """Edge case: total=0 → no crash, outcome='dimensional', expected=0."""
        attrs = ["empty"]
        attr_valence = {"empty": {"positive": 0, "neutral": 0, "negative": 0}}
        total_assigned = 0

        decision = resolve_direction(attrs, attr_valence, total_assigned)

        assert decision.attributes == ["empty"]
        assert decision.pos == 0
        assert decision.neu == 0
        assert decision.neg == 0
        assert decision.total == 0
        assert decision.outcome == "dimensional"
        assert decision.floor == 2
        # codes_for should also work
        codes = codes_for(decision, split=False)
        assert codes[0]["expected"] == 0

    def test_multiple_attributes_aggregation(self):
        """Test aggregation across multiple attributes."""
        attrs = ["attr1", "attr2", "attr3"]
        attr_valence = {
            "attr1": {"positive": 30, "neutral": 5, "negative": 10},
            "attr2": {"positive": 40, "neutral": 0, "negative": 5},
            "attr3": {"positive": 18, "neutral": 2, "negative": 5},
        }
        total_assigned = 115

        decision = resolve_direction(attrs, attr_valence, total_assigned)

        assert decision.attributes == attrs
        assert decision.pos == 30 + 40 + 18  # 88
        assert decision.neu == 5 + 0 + 2  # 7
        assert decision.neg == 10 + 5 + 5  # 20
        assert decision.total == 115


class TestCodesFor:
    """Test the codes_for function in isolation."""

    def test_codes_for_split_false_returns_neutral_only(self):
        """split=False always returns [{'valence':'neutral', 'expected':total}]."""
        from dataclasses import dataclass

        @dataclass
        class FakeDecision:
            total: int
            neutral_third: bool
            dominant_pole: str

        decision = FakeDecision(total=99, neutral_third=False, dominant_pole="positive")
        codes = codes_for(decision, split=False)

        assert codes == [{"valence": "neutral", "expected": 99}]

    def test_codes_for_with_equal_poles_positive_wins_dominant(self):
        """When pos==neg, positive is dominant (pos >= neg)."""
        from dataclasses import dataclass

        @dataclass
        class FakeDecision:
            pos: int
            neu: int
            neg: int
            neutral_third: bool
            dominant_pole: str

        decision = FakeDecision(
            pos=50, neu=20, neg=50, neutral_third=False, dominant_pole="positive"
        )
        codes = codes_for(decision, split=True)

        # dominant_pole='positive' (50 >= 50)
        # split=True, no neutral_third → positive gets 50+20=70, negative gets 50
        assert len(codes) == 2
        assert codes[0] == {"valence": "positive", "expected": 70}
        assert codes[1] == {"valence": "negative", "expected": 50}
