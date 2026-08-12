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

Pass B werd daarna een blinde binaire toewijzingsproef: echte ideeteksten van
beide codes gepoold, geschud, zonder herkomst voorgelegd, en gescoord tegen de
bekende herkomst. Ook dát mat het verkeerde: op een live run haalde het 31
paren gemiddeld 91% accuracy — en het codeboek hield alsnog vier codes over
duurzaamheid, vier over persoonlijk contact en vier over visuele identiteit
over. Scheidbaarheid is geen orthogonaliteit: de proef toont ideeën uit elke
code's EIGEN attributen, en die zijn lexicaal te onderscheiden zelfs wanneer
beide codes dezelfde dimensie dekken. Een model dat op bewoording sorteert,
scoort hoog zonder dat de codes een echte dimensie uiteen leggen.

Pass B is nu een blinde toewijzingsproef met een DERDE keuze (bronspecificatie
§4.8): naast code A en code B mag het model "BOTH" antwoorden — het idee past
bij beide even goed. Twee deterministische signalen (Python, nooit een claim
van het model zelf; zie `mece.py`'s `score_probe`): `accuracy` (onder de
ideeën die wél op één kant vielen, het aandeel dat op de juiste kant kwam —
vangt "kan niet uit elkaar gehouden worden") en `both_rate` (het aandeel van
ALLE bevraagde ideeën dat BOTH kreeg — vangt "kan wel uit elkaar gehouden
worden, maar hoort allebei écht bij beide", precies het duurzaamheidsgeval).
Samenvoegen bij accuracy op/onder zijn drempel ÓF both_rate op/boven de zijne
— zie `mece.py` (`score_probe`, `is_one_dimension`). De nuloptie is
samenvoegen, niet apart houden (bronspecificatie §2.5, compressievoorkeur):
een paar moet zijn aparte bestaansrecht bewijzen, niet andersom.

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
    """ProbeResult met `idea_ref` beperkt tot de getoonde ideeën en
    `assigned_to` beperkt tot de twee codenamen van dit paar plus "BOTH" —
    de derde, gelijkwaardige keuze (zie moduledocstring)."""
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
