"""Stap 5 — MECE-afdwinging over de verzameling codes: prompts + responsemodellen.

Twee vragen, allebei per item — nooit een groepeervraag. Een groepeervraag
("welke codes overlappen?") is in deze codebase eerder geprobeerd voor
verzamelnamen (`prompts_umbrella_merge.py`'s eerste vorm) en leverde op een
echte run niets op: 45 namen in, 45 groepen uit, want "hoort bij niets" is een
even geldig antwoord als een echte groep. Een per-item vraag met een gedwongen
opzoeking werkt wél — `synonym_of` in `prompts_relations.py` vond in dezelfde
run drie echte paren.

Pass A (`OverlapVerdict`) vraagt per code naar de moeilijkste buur: een
gedwongen opzoeking, geen partitionering. Pass B (`PairVerdict`) dwingt het
model de scheidingsregel EERST op te schrijven, vóór het oordeelt — dat
schrijven is de forcing function; een los boolean zonder geschreven regel zou
dezelfde vrije-nulantwoord-fout herhalen.

Lekdiscipline: geen respondenttellingen, ideetellingen, domein, facet of
attribuut-ids — een code wordt getoond als naam + definitie + indicatoren.
Richting (valence) wordt WEL getoond: die is al besloten en bepaalt welke
paren überhaupt vergelijkbaar zijn (twee codes met tegengestelde richting zijn
door hun richting alleen al onderscheiden, nooit een samenvoegkandidaat)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, create_model

from .consolidator import CodeShape
from .prompts_relations import INSTRUCTOR_HINT, _shuffled


@dataclass(frozen=True)
class CodeCandidate:
    """Eén code zoals de MECE-stap 'm ziet. `shape` is nooit aan het model
    getoond — alleen gebruikt om na een merge-oordeel deterministisch de
    leden en respondentverzamelingen samen te voegen (zie `mece.py`)."""
    name: str
    definition: str
    indicators: Tuple[str, ...]
    valence: str
    shape: CodeShape

    @property
    def attribute_id(self) -> str:
        """Laat de gedeelde `_shuffled` (op `.attribute_id` gesorteerd) ook
        codes ordenen, zonder een tweede hashing-implementatie."""
        return self.name


@dataclass(frozen=True)
class CandidatePair:
    """Eén kandidaat-paar voor Pass B. `pair_id` is een run-lokale sleutel die
    de `_shuffled`-volgorde volgt (zie `mece.build_candidate_pairs`), zodat
    promptvolgorde en de `Literal`-enum in het responsemodel altijd gelijk
    lopen."""
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
# Pass B — adjudicatie: per kandidaat-paar, de scheidingsregel
# ---------------------------------------------------------------------------

class PairVerdict(BaseModel):
    pair_id: int = Field(..., description="Which candidate pair this verdict is about")
    separation_rule: str = Field(
        ..., description=(
            "The rule a coder could apply to assign any given idea to exactly "
            "one of the two codes, never both. Write this BEFORE deciding "
            "one_dimension — the decision follows from whether this rule is "
            "real, not the other way around."
        )
    )
    one_dimension: bool = Field(
        ..., description=(
            "True only if no real separation rule can be written — every rule "
            "you can state would leave real ideas belonging under both, or "
            "under neither more than the other. In that case the two codes "
            "are one dimension and should merge. False if the rule you wrote "
            "actually separates them."
        )
    )


class PairAdjudicationResult(BaseModel):
    scratchpad: str = Field(default="", description="Brief reasoning before the verdicts")
    verdicts: List[PairVerdict] = Field(
        ..., description="Exactly one entry per candidate pair in the list"
    )


def make_pair_model(pairs: List[CandidatePair]) -> type:
    """PairAdjudicationResult met `pair_id` beperkt tot bestaande kandidaat-
    paren, zodat het model er geen kan verzinnen of overslaan."""
    ids: Tuple[int, ...] = tuple(p.pair_id for p in _shuffled(pairs))
    constrained_verdict = create_model(
        "ConstrainedPairVerdict",
        __base__=PairVerdict,
        pair_id=(Literal[ids], Field(..., description=(
            PairVerdict.model_fields["pair_id"].description))),
    )
    return create_model(
        "ConstrainedPairAdjudicationResult",
        __base__=PairAdjudicationResult,
        verdicts=(List[constrained_verdict], Field(..., description=(
            PairAdjudicationResult.model_fields["verdicts"].description))),
    )


def _pair_block(pair: CandidatePair, candidate_by_name: Dict[str, CodeCandidate]) -> str:
    def block(c: CodeCandidate) -> str:
        indicators = ", ".join(c.indicators) or "—"
        return f'  "{c.name}": {c.definition}\n    Indicators: {indicators}'

    a, b = candidate_by_name[pair.code_a], candidate_by_name[pair.code_b]
    return f"[{pair.pair_id}] (both {a.valence})\n{block(a)}\n{block(b)}"


def build_pair_prompt(pairs: List[CandidatePair], candidate_by_name: Dict[str, CodeCandidate]) -> str:
    inventory = "\n\n".join(_pair_block(p, candidate_by_name) for p in _shuffled(pairs))

    return f"""Below are pairs of codes from the same coding scheme, flagged as easy to
confuse with each other. For EVERY pair, first WRITE the rule a coder could apply to
assign any given idea to exactly one of the two codes — never both. Only after writing
that rule, decide:

- If you could state a real rule, `one_dimension` is false — the codes stay separate.
- If no such rule exists — every rule you try leaves real ideas belonging under both,
  or under neither more than the other — `one_dimension` is true, and the two codes
  should merge into one.

A rule that only rephrases one code's name, without pointing to something that
distinguishes actual ideas, is not a real rule.

Pairs:
{inventory}

{INSTRUCTOR_HINT}"""
