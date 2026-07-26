"""Tests for judgments.py's vote() helper and prompt builders.

All tests use fake, injected `llm_call` callables — no real LLM calls, no
import of anything that requires an API key at import time.
"""
import pytest

from pipeline.step_5_codeGenerator_experiment.judgments import (
    CodeNaming,
    MembershipVote,
    NoiseVote,
    SCHEMA_HINT,
    membership_prompt,
    naming_prompt,
    noise_prompt,
    vote,
)


def _idx_from_prompt(prompt: str) -> int:
    return int(prompt.rsplit(" ", 1)[-1])


# =============================================================================
# (a) 3x identical -> unanimous
# =============================================================================
@pytest.mark.asyncio
async def test_three_identical_votes_are_unanimous():
    async def llm_call(prompt, response_model):
        return MembershipVote(choice="A", reason="matches group A")

    outcome = await vote(
        build_prompt=lambda i: f"membership prompt {i}",
        response_model=MembershipVote,
        llm_call=llm_call,
        majority_key=lambda v: v.choice,
    )
    assert outcome.unanimous is True
    assert outcome.majority == "A"
    assert outcome.failed == 0
    assert len(outcome.votes) == 3


# =============================================================================
# (b) 2-1 split -> correct majority, unanimous=False
# =============================================================================
@pytest.mark.asyncio
async def test_two_one_split_gives_majority_not_unanimous():
    choices = {0: "A", 1: "A", 2: "B"}

    async def llm_call(prompt, response_model):
        return MembershipVote(choice=choices[_idx_from_prompt(prompt)], reason="r")

    outcome = await vote(
        build_prompt=lambda i: f"membership prompt {i}",
        response_model=MembershipVote,
        llm_call=llm_call,
        majority_key=lambda v: v.choice,
    )
    assert outcome.majority == "A"
    assert outcome.unanimous is False
    assert outcome.failed == 0
    assert len(outcome.votes) == 3


# =============================================================================
# (c) 1 exception -> failed==1, majority computed over the remaining 2
# =============================================================================
@pytest.mark.asyncio
async def test_one_failure_counts_as_failed_majority_over_survivors():
    async def llm_call(prompt, response_model):
        idx = _idx_from_prompt(prompt)
        if idx == 0:
            raise RuntimeError("boom")
        return NoiseVote(genuine_opposition=True, reason="r")

    outcome = await vote(
        build_prompt=lambda i: f"noise prompt {i}",
        response_model=NoiseVote,
        llm_call=llm_call,
        majority_key=lambda v: v.genuine_opposition,
    )
    assert outcome.failed == 1
    assert len(outcome.votes) == 2
    assert outcome.majority is True
    assert outcome.unanimous is True  # both survivors agree


# =============================================================================
# (d) 3 exceptions -> majority is None
# =============================================================================
@pytest.mark.asyncio
async def test_all_failures_give_majority_none():
    async def llm_call(prompt, response_model):
        raise RuntimeError("boom")

    outcome = await vote(
        build_prompt=lambda i: f"noise prompt {i}",
        response_model=NoiseVote,
        llm_call=llm_call,
        majority_key=lambda v: v.genuine_opposition,
    )
    assert outcome.failed == 3
    assert outcome.votes == []
    assert outcome.majority is None
    assert outcome.unanimous is False


# =============================================================================
# majority_key is required (no unhashable-identity default) and works for
# each of the three response models with their obvious key.
# =============================================================================
@pytest.mark.asyncio
async def test_majority_key_required_for_membership_vote():
    async def llm_call(prompt, response_model):
        return MembershipVote(choice="A", reason="r")

    outcome = await vote(
        build_prompt=lambda i: f"prompt {i}",
        response_model=MembershipVote,
        llm_call=llm_call,
        majority_key=lambda v: v.choice,
    )
    assert outcome.majority == "A"
    assert outcome.unanimous is True


@pytest.mark.asyncio
async def test_majority_key_required_for_noise_vote():
    async def llm_call(prompt, response_model):
        return NoiseVote(genuine_opposition=False, reason="r")

    outcome = await vote(
        build_prompt=lambda i: f"prompt {i}",
        response_model=NoiseVote,
        llm_call=llm_call,
        majority_key=lambda v: v.genuine_opposition,
    )
    assert outcome.majority is False
    assert outcome.unanimous is True


@pytest.mark.asyncio
async def test_majority_key_required_for_code_naming():
    async def llm_call(prompt, response_model):
        return CodeNaming(
            code_name="friendly service", definition="def", diagnostic_test="test",
            typical_indicators=["ind1"],
        )

    outcome = await vote(
        build_prompt=lambda i: f"prompt {i}",
        response_model=CodeNaming,
        llm_call=llm_call,
        majority_key=lambda v: v.code_name,
    )
    assert outcome.majority == "friendly service"
    assert outcome.unanimous is True


def test_vote_requires_majority_key_keyword():
    """majority_key has no default — omitting it is a TypeError, not a
    silent unhashable-Pydantic-model crash inside Counter."""
    import inspect
    sig = inspect.signature(vote)
    assert sig.parameters["majority_key"].default is inspect.Parameter.empty


# =============================================================================
# (e) prompts for vote_idx 0/1/2 differ (evidence shuffle) but carry the same lines
# =============================================================================
def test_membership_prompt_shuffles_evidence_but_keeps_same_lines():
    samples = ["s1 says one thing", "s2 says another", "s3 says a third",
               "s4 says a fourth", "s5 says a fifth"]
    prompts = [
        membership_prompt(
            attr="attr_x", definition="def of attr_x", samples=samples,
            group_a="Group A description", group_b="Group B description",
            language="English", vote_idx=i,
        )
        for i in range(3)
    ]
    assert len(set(prompts)) == 3, "each vote_idx should shuffle to a distinct prompt"
    reference = sorted(prompts[0].splitlines())
    for p in prompts[1:]:
        assert sorted(p.splitlines()) == reference


def test_noise_prompt_shuffles_evidence_but_keeps_same_lines():
    pos_texts = ["p1", "p2", "p3", "p4", "p5"]
    neg_texts = ["n1", "n2", "n3", "n4", "n5"]
    prompts = [
        noise_prompt(
            phenomenon_desc="phenomenon X", pos_texts=pos_texts, neg_texts=neg_texts,
            language="English", vote_idx=i,
        )
        for i in range(3)
    ]
    assert len(set(prompts)) == 3
    reference = sorted(prompts[0].splitlines())
    for p in prompts[1:]:
        assert sorted(p.splitlines()) == reference


def test_naming_prompt_shuffles_evidence_but_keeps_same_lines():
    members = ["attr_a", "attr_b", "attr_c", "attr_d", "attr_e"]
    samples_per_pole = {m: [f"{m} sample 1", f"{m} sample 2"] for m in members}
    prompts = [
        naming_prompt(
            members=members, samples_per_pole=samples_per_pole, valence="positive",
            language="English", survey_question="What do you think?",
            avoid_names=["existing code"], vote_idx=i,
        )
        for i in range(3)
    ]
    assert len(set(prompts)) == 3
    reference = sorted(prompts[0].splitlines())
    for p in prompts[1:]:
        assert sorted(p.splitlines()) == reference


# =============================================================================
# (f) every prompt ends on the literal schema-hint sentence
# =============================================================================
def test_all_prompt_builders_end_on_schema_hint():
    assert SCHEMA_HINT == "Provide your output as valid JSON following the response schema provided."

    m = membership_prompt(
        attr="attr_x", definition="def", samples=["s1", "s2"],
        group_a="A desc", group_b="B desc", language="English", vote_idx=0,
    )
    n = noise_prompt(
        phenomenon_desc="phenomenon", pos_texts=["p1"], neg_texts=["n1"],
        language="English", vote_idx=0,
    )
    c = naming_prompt(
        members=["attr_a"], samples_per_pole={"attr_a": ["s1"]}, valence="neutral",
        language="English", survey_question="Q?", avoid_names=[], vote_idx=0,
    )
    for prompt in (m, n, c):
        assert prompt.endswith(SCHEMA_HINT)
