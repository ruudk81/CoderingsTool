"""Stap 5 — MECE-afdwinging over de verzameling codes: prompts + responsemodellen.

Pass A (`OverlapVerdict`) vraagt per code naar de moeilijkste buur: een
gedwongen opzoeking, geen partitionering — nooit een groepeervraag. Een
groepeervraag ("welke codes overlappen?") is in deze codebase eerder
geprobeerd voor verzamelnamen (`prompts_umbrella_merge.py`'s eerste vorm) en
leverde op een echte run niets op: 45 namen in, 45 groepen uit, want "hoort
bij niets" is een even geldig antwoord als een echte groep. Een per-item vraag
met een gedwongen opzoeking werkt wél — `synonym_of` in `prompts_relations.py`
vond in dezelfde run drie echte paren.

Pass B was ooit een geschreven scheidingsregel + een boolean (`one_dimension`).
Dat kon niet werken: een model dat gevraagd wordt een regel te schrijven,
schrijft er altijd één, en concludeert daarna — omdat het de regel zelf net
heeft geschreven — dat de codes scheidbaar zijn. Op een live run kwamen er 31
paren uit Pass A en nul samenvoegingen uit Pass B: de "regels" waren de twee
definities herhaald, geen regel. Het genereren van een verantwoording is geen
toets zolang de model zelf bepaalt of hij slaagt.

Pass B is nu een blinde toewijzingsproef (bronspecificatie §4.8): echte
ideeteksten van beide codes worden gepoold, geschud, en zonder herkomst aan
het model voorgelegd; het model wijst elk idee toe aan code A of code B. De
score tegen de bekende herkomst (Python, nooit het model zelf) is de toets —
zie `mece.py` (`score_assignments`, `is_one_dimension`) voor de scoring en de
drempel.

Lekdiscipline: geen respondenttellingen, ideetellingen, domein, facet of
attribuut-ids. Pass A toont een code als naam + definitie + indicatoren.
Richting (valence) wordt WEL getoond in Pass A: die is al besloten en bepaalt
welke paren überhaupt vergelijkbaar zijn (twee codes met tegengestelde
richting zijn door hun richting alleen al onderscheiden, nooit een
samenvoegkandidaat). Pass B toont een code als naam + definitie, en toont de
geschudde ideeteksten zonder enige aanduiding van welke kant ze vandaan
komen — dat blijft uitsluitend in Python (`PairProbe.truth` in `mece.py`)."""
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
# Pass B — blinde toewijzingsproef: per kandidaat-paar, echte ideeën toewijzen
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProbeIdea:
    """Eén gepoold ideetje zoals het model het te zien krijgt: alleen een
    volgnummer en de tekst. Welke code dit idee werkelijk levert, staat
    nergens op dit object — dat leeft uitsluitend in `mece.PairProbe.truth`,
    in Python, nooit in de prompt of het responsemodel."""
    idea_ref: int
    text: str


class IdeaAssignment(BaseModel):
    idea_ref: int = Field(..., description="Which idea, by the number shown in the list, this is about")
    assigned_to: str = Field(..., description="Which of the two codes this idea actually belongs to")


class ProbeResult(BaseModel):
    scratchpad: str = Field(default="", description="Brief reasoning before the assignments")
    assignments: List[IdeaAssignment] = Field(
        ..., description="Exactly one entry per idea in the list"
    )


def make_probe_model(pair: CandidatePair, ideas: List[ProbeIdea]) -> type:
    """ProbeResult met `idea_ref` beperkt tot de getoonde ideeën en
    `assigned_to` beperkt tot precies de twee codenamen van dit paar."""
    idea_refs: Tuple[int, ...] = tuple(idea.idea_ref for idea in ideas)
    code_names: Tuple[str, str] = (pair.code_a, pair.code_b)
    constrained_assignment = create_model(
        "ConstrainedIdeaAssignment",
        __base__=IdeaAssignment,
        idea_ref=(Literal[idea_refs], Field(..., description=(
            IdeaAssignment.model_fields["idea_ref"].description))),
        assigned_to=(Literal[code_names], Field(..., description=(
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

Responses:
{idea_lines}

{INSTRUCTOR_HINT}"""
