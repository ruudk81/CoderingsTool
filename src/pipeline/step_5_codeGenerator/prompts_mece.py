"""Step 5 — MECE enforcement across the set of codes: prompts + response models.

Pass A (`OverlapVerdict`) asks each code for its hardest neighbour: a forced
lookup, not a partition — never a grouping question. A grouping question ("which
codes overlap?") has been tried before in this codebase, for umbrella names (the
first form of `prompts_umbrella_merge.py`), and yielded nothing on a real run:
45 names in, 45 groups out, because "belongs with nothing" is as valid an answer
as a real group. A per-item question with a forced lookup does work — `synonym_of`
in `prompts_relations.py` found three genuine pairs in that same run.

Pass B was once a written separation rule plus a boolean (`one_dimension`). That
could not work: a model asked to write a rule always writes one, and then
concludes — because it just wrote the rule itself — that the codes are separable.
On a live run Pass A produced 31 pairs and Pass B zero merges: the "rules" were
the two definitions restated, not a rule. Generating a justification is not a test
as long as the model decides whether it passes.

Pass B then became a blind binary assignment probe: real idea texts from both
codes pooled, shuffled, presented without provenance, and scored against the known
provenance. That measured the wrong thing too: on a live run 31 pairs averaged 91%
accuracy — and the codebook still kept four codes on one theme, four on another and
four on a third. Separability is not orthogonality: the probe shows ideas from each
code's OWN attributes, and those are lexically distinguishable even when both codes
cover the same dimension. A model sorting on wording scores high without the codes
pulling a real dimension apart.

Pass B is now a blind assignment probe with a THIRD choice (source specification
§4.8): besides code A and code B the model may answer "BOTH" — the idea fits both
equally well. Two deterministic signals (Python, never a claim by the model itself;
see `score_probe` in `mece.py`): `accuracy` (among the ideas that did fall on one
side, the share that landed on the correct side — catches "cannot be told apart")
and `both_rate` (the share of ALL probed ideas that got BOTH — catches "can be told
apart, but genuinely belongs to both"). Merge when accuracy is at/below its
threshold OR both_rate at/above its own — see `mece.py` (`score_probe`,
`is_one_dimension`). The null option is to merge, not to keep apart (source
specification §2.5, compression preference): a pair must prove its separate right
to exist, not the other way round.

Leak discipline: no respondent counts, idea counts, domain, facet or attribute ids.
Pass A shows a code as name + definition + indicators. Direction (valence) IS shown
in Pass A: it is already decided and determines which pairs are comparable at all
(two codes with opposite direction are distinguished by their direction alone, and
are never a merge candidate). Pass B shows a code as name + definition, and shows
the shuffled idea texts without any indication of which side they came from — that
stays exclusively in Python (`PairProbe.truth` in `mece.py`)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, create_model

from .consolidator import CodeShape
from .prompts_relations import INSTRUCTOR_HINT, _shuffled


@dataclass(frozen=True)
class CodeCandidate:
    """One code as the MECE step sees it. `shape` is never shown to the model —
    it is used only to merge members and respondent sets deterministically after
    a merge verdict (see `mece.py`)."""
    name: str
    definition: str
    indicators: Tuple[str, ...]
    valence: str
    shape: CodeShape

    @property
    def attribute_id(self) -> str:
        """Lets the shared `_shuffled` (which sorts on `.attribute_id`) order
        codes as well, without a second hashing implementation."""
        return self.name


@dataclass(frozen=True)
class CandidatePair:
    """One candidate pair for Pass B. `pair_id` is a run-local key following the
    `_shuffled` order (see `mece.build_candidate_pairs`), so that prompt order and
    the `Literal` enum in the response model always stay in step."""
    pair_id: int
    code_a: str
    code_b: str

    @property
    def attribute_id(self) -> str:
        return f"{self.code_a}||{self.code_b}"


# ---------------------------------------------------------------------------
# Pass A — detectie: per code, de moeilijkste buur
# ---------------------------------------------------------------------------

class OverlapVerdict(BaseModel):
    code: str = Field(..., description="The code this verdict is about")
    hardest_to_separate_from: Optional[str] = Field(
        None,
        description=(
            "The other code in the list a coder would most easily confuse this "
            "one with — the one where you would hesitate longest deciding "
            "between the two. Null only if there is genuinely none."
        ),
    )


class OverlapDetectionResult(BaseModel):
    scratchpad: str = Field(default="", description="Brief reasoning before the verdicts")
    verdicts: List[OverlapVerdict] = Field(
        ..., description="Exactly one entry per code in the list"
    )


def make_overlap_model(candidates: List[CodeCandidate]) -> type:
    """OverlapDetectionResult met `code` en `hardest_to_separate_from` beperkt
    tot bestaande codenamen."""
    names: Tuple[str, ...] = tuple(c.name for c in _shuffled(candidates))
    constrained_verdict = create_model(
        "ConstrainedOverlapVerdict",
        __base__=OverlapVerdict,
        code=(Literal[names], Field(..., description=(
            OverlapVerdict.model_fields["code"].description))),
        hardest_to_separate_from=(Optional[Literal[names]], Field(None, description=(
            OverlapVerdict.model_fields["hardest_to_separate_from"].description))),
    )
    return create_model(
        "ConstrainedOverlapDetectionResult",
        __base__=OverlapDetectionResult,
        verdicts=(List[constrained_verdict], Field(..., description=(
            OverlapDetectionResult.model_fields["verdicts"].description))),
    )


def _code_block(candidate: CodeCandidate) -> str:
    indicators = ", ".join(candidate.indicators) or "—"
    return (f'- "{candidate.name}" ({candidate.valence}): {candidate.definition}\n'
            f"  Indicators: {indicators}")


def build_overlap_prompt(candidates: List[CodeCandidate]) -> str:
    inventory = "\n".join(_code_block(c) for c in _shuffled(candidates))

    return f"""You are reviewing a set of qualitative codes from a coding scheme. Codes in
this scheme must be mutually exclusive: a coder faced with one idea should be able to
place it under exactly one code.

For EVERY code below, name the other code in the list that a coder would most easily
confuse it with — the one where you would hesitate longest deciding between the two.
If there is genuinely no code you would ever confuse this one with, say null.

Rules:
- Each code's evaluative direction is shown in parentheses. Only codes that share the
  same direction can ever be confused with each other — direction alone already tells
  two codes apart.
- Judge on the definition and indicators, not on the name alone: two codes can sound
  alike while covering different things, or sound different while covering the same
  thing.
- Every code gets exactly one entry.

Codes:
{inventory}

{INSTRUCTOR_HINT}"""


# ---------------------------------------------------------------------------
# Pass B — blinde toewijzingsproef: per kandidaat-paar, echte ideeën toewijzen
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProbeIdea:
    """One pooled idea as the model gets to see it: a sequence number and the
    text, nothing more. Which code this idea actually comes from appears nowhere
    on this object — that lives exclusively in `mece.PairProbe.truth`, in Python,
    never in the prompt or the response model."""
    idea_ref: int
    text: str


class IdeaAssignment(BaseModel):
    idea_ref: int = Field(..., description="Which idea, by the number shown in the list, this is about")
    assigned_to: str = Field(
        ...,
        description=(
            "Which of the two codes this idea belongs to, or \"BOTH\" if it genuinely "
            "fits either equally well."
        ),
    )


class ProbeResult(BaseModel):
    scratchpad: str = Field(default="", description="Brief reasoning before the assignments")
    assignments: List[IdeaAssignment] = Field(
        ..., description="Exactly one entry per idea in the list"
    )


def make_probe_model(pair: CandidatePair, ideas: List[ProbeIdea]) -> type:
    """ProbeResult with `idea_ref` restricted to the ideas shown and
    `assigned_to` restricted to this pair's two code names plus "BOTH" — the
    third, equally valid choice (see the module docstring)."""
    idea_refs: Tuple[int, ...] = tuple(idea.idea_ref for idea in ideas)
    choices: Tuple[str, str, str] = (pair.code_a, pair.code_b, "BOTH")
    constrained_assignment = create_model(
        "ConstrainedIdeaAssignment",
        __base__=IdeaAssignment,
        idea_ref=(Literal[idea_refs], Field(..., description=(
            IdeaAssignment.model_fields["idea_ref"].description))),
        assigned_to=(Literal[choices], Field(..., description=(
            IdeaAssignment.model_fields["assigned_to"].description))),
    )
    return create_model(
        "ConstrainedProbeResult",
        __base__=ProbeResult,
        assignments=(List[constrained_assignment], Field(..., description=(
            ProbeResult.model_fields["assignments"].description))),
    )


def build_probe_prompt(
    pair: CandidatePair, candidate_by_name: Dict[str, CodeCandidate], ideas: List[ProbeIdea],
) -> str:
    a, b = candidate_by_name[pair.code_a], candidate_by_name[pair.code_b]
    idea_lines = "\n".join(f'[{idea.idea_ref}] "{idea.text}"' for idea in ideas)

    return f"""Below are two codes from a coding scheme, and a pooled, shuffled list of real
survey responses that were coded under one or the other. You are not told which response
came from which code.

Codes:
- "{a.name}": {a.definition}
- "{b.name}": {b.definition}

For EVERY response below, decide which of the two codes above it actually belongs to.
If it genuinely fits BOTH codes equally well, answer "BOTH" — that is the correct
answer whenever it applies, not a fallback for when you are unsure. A response that is
merely hard to place still belongs on one side, so give it that side instead.

Responses:
{idea_lines}

{INSTRUCTOR_HINT}"""
