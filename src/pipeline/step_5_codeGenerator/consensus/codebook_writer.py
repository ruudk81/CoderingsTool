"""Step 4 — dispatch of the writing call.

The shape (how many codes, which members, which direction) is already fixed; this
module only fills in the texts and applies the one permitted veto: `nameable:
false` on a pooled shape drops that shape. If the call fails, or a shape is
missing from the answer, that shape gets a deterministic filling rather than the
codebook losing a code — the shapes stay valid, only the text gets less rich.

Leak discipline: this module never passes a respondent count, domain, facet or
attribute id to the LLM — see `prompts_writer.py` for the prompt construction
itself.

A rewrite that sees only PART of the codebook (e.g. only the MECE-merged codes)
can land on a name a non-rewritten code already carries: `write_codebook`'s
`taken_names` passes those other names along so the LLM avoids them, and
`resolve_duplicate_names` is the deterministic backstop the caller runs over the
COMPLETE, reunited codebook after rewriting — a prompt rule is never trusted here
as the only guarantee.

Sinds 2026-08-22 staan hier TWEE schrijfcalls naast elkaar. `write_codebook`
schrijft de hoofdcodes; `write_miscellaneous` schrijft de kinderen — de
facetunies van afgevallen valentiepolen die onder Overig komen te hangen. Twee
calls en niet één, omdat de instructie verschilt: een kind is een restcategorie
binnen een onderwerp en mag geen eigen kop claimen. De aanroeper splitst de
vormen op `origin` en geeft elke helft aan zijn eigen call; geen van beide
functies filtert zelf, want een call die stil vormen laat vallen is precies de
faalvorm die deze module elders met een fallbacktekst voorkomt.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Union

from config import get_reasoning_params
from utils.llm import RateLimits
from utils.smoothRequester import SmoothRequester

from .codebook_io import rest_label
from .concept_inventory import Concept
from .config_codeGenerator import CodebookConfig
from .code_shape import CodeShape, stored_valence
from models import ConsolidatedCode
from .prompts_miscellaneous import (
    MiscellaneousText, build_miscellaneous_prompt, make_miscellaneous_model,
)
from .prompts_writer import CodeText, build_writer_prompt, make_writer_model

PHASE = "step5_writer"
# Een eigen fase, en niet `PHASE`: `perfModel` houdt zijn ringbuffer per
# (model, fase) bij, en de kinderprompt is korter en levert kortere teksten op
# dan de schrijverprompt. Op één sleutel zouden ze elkaars warmtestart
# vervuilen. De `c` volgt `consolidation.PHASE` (`step5c_consolidation`): in
# deze kandidaatketen draagt elke eigen fase die letter, zodat een meting nooit
# op naam van de productieketen komt te staan.
MISCELLANEOUS_PHASE = "step5c_miscellaneous"


def _topic_names(shape: CodeShape, concept_by_id: Dict[str, Concept]) -> List[str]:
    return [concept_by_id[member_id].name
            for member_id in shape.members if member_id in concept_by_id]


def _fallback_text(shape: CodeShape, concept_by_id: Dict[str, Concept],
                   dimension_diagnostic: str) -> CodeText:
    """Deterministic stand-in for a shape the model never wrote — either the
    whole call failed, or it simply omitted this key. Always nameable: without
    a model judgment there is no basis for a veto, and dropping a shape here
    would silently shrink the codebook."""
    names = _topic_names(shape, concept_by_id)
    code_name = names[0] if names else shape.umbrella
    return CodeText(
        key=shape.key,
        code_name=code_name,
        definition=f"Responses about {', '.join(names) if names else code_name}.",
        diagnostic_test=dimension_diagnostic,
        typical_indicators=names or [code_name],
        boundary_note="",
        nameable=True,
    )


def _miscellaneous_name(umbrella: str, language: str) -> str:
    """De noodnaam van een kind: de restaanduiding ÉN zijn onderwerp.

    Niet het kale onderwerp. `shape.umbrella` is het facet, en dat is precies
    wat regel 1 van de kinderprompt verbiedt — een kale onderwerpsnaam claimt
    de kop van dat hele facet, terwijl een kind juist draagt wat daar buiten de
    hoofdcodes viel. Niet de kale restaanduiding evenmin: dan is niet te zien
    bij welk onderwerp hij hoort, en er zijn er meerdere.
    """
    return f"{rest_label(language)} — {umbrella}"


def _miscellaneous_fallback_text(shape: CodeShape, concept_by_id: Dict[str, Concept],
                                 dimension_diagnostic: str,
                                 language: str) -> MiscellaneousText:
    """Deterministische noodtekst voor een kind dat het model oversloeg.

    De noodnaam is de restaanduiding plus het ONDERWERP (`shape.umbrella`, door
    `build_shapes` gevuld met het facet), waar `_fallback_text` de naam van het
    eerste lid neemt. Dat verschil is de reden dat deze functie bestaat: een
    kind is per constructie de restcategorie van één facet, en de naam van één
    lid zou de kop van dat hele facet claimen — precies wat de kinderprompt
    verbiedt. Alleen als het onderwerp ontbreekt (per constructie onbereikbaar)
    valt hij terug op het eerste lid, want een naamloze code is erger dan een te
    specifieke.
    """
    names = _topic_names(shape, concept_by_id)
    code_name = (_miscellaneous_name(shape.umbrella, language) if shape.umbrella
                 else (names[0] if names else ""))
    return MiscellaneousText(
        key=shape.key,
        code_name=code_name,
        definition=f"Responses about {', '.join(names) if names else code_name}.",
        diagnostic_test=dimension_diagnostic,
        typical_indicators=names or [code_name],
        boundary_note="",
    )


def _to_consolidated_code(text: Union[CodeText, MiscellaneousText], shape: CodeShape,
                          concept_by_id: Dict[str, Concept]) -> ConsolidatedCode:
    """Gedeeld door beide schrijfcalls: de velden die hier worden uitgelezen
    dragen de twee teksten allebei. `nameable` staat er niet bij — dat veld is
    een oordeel over de vorm, geen tekst van de code, en het wordt afgehandeld
    waar het veto valt.

    `parent_code_id` blijft leeg, ook voor een kind: de ouder is de
    Overig-code, en die bestaat pas nadat `apply_overig_sweep` hem heeft
    gemaakt. De aanroeper hangt de kinderen daarna aan hun ouder.
    """
    return ConsolidatedCode(
        code_name=text.code_name,
        definition=text.definition,
        diagnostic_test=text.diagnostic_test,
        valence=stored_valence(shape.valence),
        typical_indicators=text.typical_indicators,
        source_attributes=_topic_names(shape, concept_by_id),
    )


def _record_veto(log, shape: CodeShape, concept_by_id: Dict[str, Concept]) -> None:
    if log is None:
        return
    log.add(
        action="VETO",
        members=list(shape.members),
        umbrella=shape.umbrella,
        reason="grouped topics share nothing that can be named honestly",
    )


def _writing_requester(config: CodebookConfig, phase: str, verbose: bool,
                       known_limits: Optional[RateLimits],
                       has_server_headers: Optional[bool]) -> SmoothRequester:
    """De opstelling die beide schrijfcalls delen: één taak, hetzelfde model,
    stil. Alleen de fase verschilt. Staat apart zodat de twee calls niet elk
    hun eigen kopie van deze zeven regels dragen — dan zou een aanpassing aan
    de ene stil langs de andere gaan."""
    return SmoothRequester(
        model=config.model_writer,
        phase_key=phase,
        num_tasks=1,
        verbose=verbose,
        known_limits=known_limits,
        has_server_headers=has_server_headers,
        quiet=True,
    )


async def write_codebook(
    shapes: List[CodeShape],
    concepts: List[Concept],
    dimension_diagnostic: str,
    language: str,
    config: CodebookConfig,
    log=None,
    known_limits: Optional[RateLimits] = None,
    has_server_headers: Optional[bool] = None,
    verbose: bool = False,
    taken_names: Optional[List[str]] = None,
    prompt_printer=None,
    prompt_builder=build_writer_prompt,
) -> List[ConsolidatedCode]:
    """One call across all fixed code shapes. A `nameable: false` verdict on a
    `pooled` shape drops it (recorded in `log` as a VETO); the same verdict on
    a `solo` or `synonym` shape is ignored — those are single attributes and
    are by definition nameable. `recovered` en `child` zijn evenmin vetobaar;
    de reden staat bij de veto-toets zelf.

    `taken_names` is for a re-write that only sees a SUBSET of the book (e.g.
    the MECE-merged codes) — the names already committed for the codes NOT in
    `shapes` this call, so the model doesn't land on one of them. This is a
    prompt-level ask, not a guarantee; see `resolve_duplicate_names` for the
    deterministic backstop the caller must still run over the full, reassembled
    codebook.

    `prompt_builder` defaults to the production writing prompt; the retired v1
    chain passes its own `build_writer_prompt_v1` (same five positional params)
    to reuse this whole dispatch without duplicating it."""
    if not shapes:
        return []

    concept_by_id = {concept.attribute_id: concept for concept in concepts}

    def prepare_fn(task):
        prompt = prompt_builder(
            task["shapes"], task["concept_by_id"],
            task["dimension_diagnostic"], task["language"], task["taken_names"],
        )
        if prompt_printer is not None:
            prompt_printer.capture_prompt(
                step_name="code_generator",
                utility_name="write_codebook",
                prompt_content=prompt,
                prompt_type="codebook_writer",
                metadata={
                    "model": config.model_writer,
                    "temperature": config.temperature_writer,
                    "max_tokens": config.max_tokens_writer,
                    "language": task["language"],
                    "n_shapes": len(task["shapes"]),
                    "shape_keys": [shape.key for shape in task["shapes"]],
                },
            )
        return {
            "prompt": prompt,
            "response_model": make_writer_model(task["shapes"]),
            "temperature": config.temperature_writer,
            "max_tokens": config.max_tokens_writer,
            "max_retries": 2,
            "extra_kwargs": get_reasoning_params(config.model_writer, phase="codegen_writer"),
        }

    def parse_fn(_task, response):
        return response

    def fallback_fn(_task, _reason):
        return None

    requester = _writing_requester(config, PHASE, verbose, known_limits,
                                   has_server_headers)
    tasks = [{
        "shapes": shapes, "concept_by_id": concept_by_id,
        "dimension_diagnostic": dimension_diagnostic, "language": language,
        "taken_names": taken_names,
    }]
    results = await requester.process_all(tasks, prepare_fn, parse_fn, fallback_fn)
    result = results[0] if results else None
    text_by_key = {text.key: text for text in result.codes} if result is not None else {}

    codes: List[ConsolidatedCode] = []
    for shape in shapes:
        text = text_by_key.get(shape.key) or _fallback_text(
            shape, concept_by_id, dimension_diagnostic
        )
        # Alleen `pooled` — en dat is sinds 2026-08-22 een scherpere grens dan
        # "meer dan één attribuut". Het veto beoordeelt of een door het MODEL
        # voorgestelde samenvoeging een code was; valt hij weg, dan blijven de
        # attributen los over en veegt `apply_overig_sweep` ze naar Overig.
        #
        # Voor een facetunie uit `pool_minority_poles` (`recovered`) klopt geen
        # van beide. Hij is geen modelvoorstel maar step 4's eigen structuur,
        # en zijn attributen zijn ook bron van de OVERLEVENDE zusterpool — die
        # vaak het tegenovergestelde beweert. De sweep ziet ze daarom niet als
        # wees, dus een veto zet deze respondenten wéér onder een code die iets
        # anders zegt: precies het defect dat de facetunie opheft. Een `child`
        # valt om dezelfde reden buiten het veto (zie `code_shape.CodeShape`).
        #
        # Daar hangt ook een telling aan: `ShapingResult.coverage_recovered`
        # wordt in `build_shapes` berekend, vóór deze call. Dat getal is alleen
        # waar zolang geen enkele vorm uit `pool_minority_poles` hier nog kan
        # sneuvelen. Wie het veto ooit tot `recovered` uitbreidt, moet die
        # telling ná de schrijfcall opnieuw bepalen — anders claimt hij dekking
        # die er niet is.
        if not text.nameable and shape.origin == "pooled":
            _record_veto(log, shape, concept_by_id)
            continue
        codes.append(_to_consolidated_code(text, shape, concept_by_id))
    return codes


async def write_miscellaneous(
    shapes: List[CodeShape],
    concepts: List[Concept],
    dimension_diagnostic: str,
    language: str,
    config: CodebookConfig,
    known_limits: Optional[RateLimits] = None,
    has_server_headers: Optional[bool] = None,
    verbose: bool = False,
    taken_names: Optional[List[str]] = None,
    prompt_printer=None,
) -> List[ConsolidatedCode]:
    """Eén call over de kindvormen: de restcategorieën die onder Overig hangen.

    Dezelfde vorm als `write_codebook` — één call over alle vormen tegelijk,
    tekst teruggevonden op `shape.key`, een deterministische noodtekst voor een
    vorm die het model oversloeg — met drie bewuste verschillen:

    - **Een andere prompt.** Een kind is een restcategorie binnen een
      onderwerp; de naam mag geen hoofdthema suggereren. Zie
      `prompts_miscellaneous.py`.
    - **Geen veto, en dus geen `log`.** Een kind bestaat omdat zijn
      respondenten anders nergens staan; het mogen weigeren zou ze daar weer
      neerzetten. Het responsemodel kent daarom geen `nameable`-veld.
    - **Een eigen noodnaam.** Het onderwerp, niet het eerste lid — zie
      `_miscellaneous_fallback_text`.

    De aanroeper bepaalt WELKE vormen hier binnenkomen (`origin == "child"`);
    deze functie filtert niet. Hij zet ook `parent_code_id`: de ouder is de
    Overig-code en die bestaat op dit moment nog niet. Om de teruggegeven codes
    aan hun vorm te koppelen gebruikt de aanroeper `_shape_lookup`/
    `_match_shape` uit `code_shape.py`, net als na `write_codebook` — nooit
    zippen, want de volgorde is geen identiteit.
    """
    if not shapes:
        return []

    concept_by_id = {concept.attribute_id: concept for concept in concepts}

    def prepare_fn(task):
        prompt = build_miscellaneous_prompt(
            task["shapes"], task["concept_by_id"],
            task["dimension_diagnostic"], task["language"], task["taken_names"],
        )
        if prompt_printer is not None:
            prompt_printer.capture_prompt(
                step_name="code_generator",
                utility_name="write_miscellaneous",
                prompt_content=prompt,
                prompt_type="miscellaneous_writer",
                metadata={
                    "model": config.model_writer,
                    "temperature": config.temperature_writer,
                    "max_tokens": config.max_tokens_writer,
                    "language": task["language"],
                    "n_shapes": len(task["shapes"]),
                    "shape_keys": [shape.key for shape in task["shapes"]],
                },
            )
        return {
            "prompt": prompt,
            "response_model": make_miscellaneous_model(task["shapes"]),
            "temperature": config.temperature_writer,
            "max_tokens": config.max_tokens_writer,
            "max_retries": 2,
            # Dezelfde fase-instelling als de schrijver: het is hetzelfde
            # model dat hetzelfde soort werk doet. Een eigen `STEP_EFFORT`-
            # sleutel zou een knop zijn die op één dataset is afgesteld.
            "extra_kwargs": get_reasoning_params(config.model_writer, phase="codegen_writer"),
        }

    def parse_fn(_task, response):
        return response

    def fallback_fn(_task, _reason):
        return None

    requester = _writing_requester(config, MISCELLANEOUS_PHASE, verbose,
                                   known_limits, has_server_headers)
    tasks = [{
        "shapes": shapes, "concept_by_id": concept_by_id,
        "dimension_diagnostic": dimension_diagnostic, "language": language,
        "taken_names": taken_names,
    }]
    results = await requester.process_all(tasks, prepare_fn, parse_fn, fallback_fn)
    result = results[0] if results else None
    text_by_key = {text.key: text for text in result.codes} if result is not None else {}

    return [
        _to_consolidated_code(
            text_by_key.get(shape.key)
            or _miscellaneous_fallback_text(shape, concept_by_id,
                                            dimension_diagnostic, language),
            shape, concept_by_id,
        )
        for shape in shapes
    ]


def resolve_duplicate_names(
    codes: List[ConsolidatedCode], shapes: List[CodeShape], log=None,
    language: str = "",
) -> List[ConsolidatedCode]:
    """Deterministic backstop for `taken_names`: the prompt asks the model not
    to reuse a name, but nothing here depends on it having obeyed. `codes[i]`
    must be the text written for `shapes[i]` — the caller's positional
    pairing (e.g. the untouched codes followed by a re-write's output, in the
    same order the shapes were passed to `write_codebook`), not re-derived
    here.

    Within each group of codes sharing a name, the code with the most
    respondents keeps it (ties broken by `shape.key` for a reproducible
    result); every other code in the group is renamed to its own shape's
    umbrella term — the name consolidation proposed for that group
    (`grouping.py`; the quarantined v1 chain fills the same field with the
    group name its consolidator climbed to) — with a number appended only if
    even that is already taken. Every rename is reported via `log.add(...)`
    (duck-typed, like `write_codebook`'s own `log`), so a resolved collision is
    always visible, never a silent rename. A codebook with no duplicate names
    is returned unchanged.

    Voor een KIND is de kale umbrella geen geldige naam — dat is het facet, en
    regel 1 van de kinderprompt verbiedt juist dat een restcategorie de kop van
    haar onderwerp claimt. Een verliezend kind valt daarom terug op dezelfde
    naamvorm als zijn noodtekst (`_miscellaneous_name`), en daarvoor is
    `language` nodig. Blijft die leeg, dan zijn er per constructie ook geen
    kinderen: alleen de aanroeper die de tweede schrijfcall bedraadt levert ze,
    en die geeft de taal mee."""
    if len(codes) != len(shapes):
        raise ValueError("codes and shapes must be positional pairs of equal length")

    groups: Dict[str, List[int]] = defaultdict(list)
    for i, code in enumerate(codes):
        groups[code.code_name].append(i)
    duplicate_names = sorted(name for name, indices in groups.items() if len(indices) > 1)
    if not duplicate_names:
        return codes

    resolved = list(codes)
    taken = {code.code_name for code in codes}
    for name in duplicate_names:
        winner_idx, *loser_indices = sorted(
            groups[name], key=lambda i: (-len(shapes[i].resp_ids), shapes[i].key)
        )
        for loser_idx in loser_indices:
            shape = shapes[loser_idx]
            basis = (_miscellaneous_name(shape.umbrella, language)
                     if shape.origin == "child" and shape.umbrella else shape.umbrella)
            candidate = basis
            suffix = 2
            while candidate in taken:
                candidate = f"{basis} ({suffix})"
                suffix += 1
            taken.add(candidate)
            resolved[loser_idx] = resolved[loser_idx].model_copy(update={"code_name": candidate})
            if log is not None:
                log.add(
                    action="DUPLICATE_NAME_RESOLVED",
                    name=name,
                    kept_n_resp=len(shapes[winner_idx].resp_ids),
                    renamed_to=candidate,
                    renamed_n_resp=len(shape.resp_ids),
                )
    return resolved


# ---------------------------------------------------------------------------
# find_naming_mismatches — deterministic backstop against a name that no longer
# eigen inhoud niet beschrijft
# ---------------------------------------------------------------------------

_STOPWORDS = {
    # Dutch
    "de", "het", "een", "en", "van", "in", "op", "voor", "met", "aan", "bij",
    "over", "tot", "als", "naar", "door", "om", "uit", "is", "zijn", "haar",
    "hun", "wordt", "worden", "niet", "geen", "die", "dat", "dit", "deze",
    "ook", "meer", "wel", "nog",
    # English
    "the", "a", "an", "and", "of", "in", "on", "for", "with", "to", "by",
    "at", "is", "are", "as", "or", "this", "that", "these", "those", "not",
}


def _meaningful_words(text: str) -> Set[str]:
    """Lowercased word tokens, minus stopwords and words too short to carry
    meaning (2 letters or fewer) — generic linguistic filtering, not use-case
    vocabulary."""
    return {word for word in re.findall(r"[^\W\d_]+", text.lower())
            if len(word) > 2 and word not in _STOPWORDS}


def find_naming_mismatches(
    codes: List[ConsolidatedCode], shapes: List[CodeShape], concept_by_id: Dict[str, Concept],
) -> List[dict]:
    """Deterministic check: does a written code name share a meaningful word
    with the name of at least one of its own member attributes? A codebook
    entry whose name has nothing lexically in common with what it actually
    contains is the signature of the writer having named the wrong material —
    a stale label, or content meant for a different code in the same batch.
    Prompt rules have failed silently before in this codebase, so this runs
    regardless of what the model claimed about itself.

    Not an auto-correction: a false positive here (two honestly-worded terms
    for the same thing that happen not to share a stem) costs one printed
    line; a false negative would be a wrong code shipped silently, which is
    what this exists to catch. `codes[i]` must be the text written for
    `shapes[i]` — the same positional contract as `resolve_duplicate_names`.
    A shape with no resolvable member names is skipped, not flagged: there is
    nothing to compare the name against."""
    if len(codes) != len(shapes):
        raise ValueError("codes and shapes must be positional pairs of equal length")

    mismatches: List[dict] = []
    for code, shape in zip(codes, shapes):
        code_words = _meaningful_words(code.code_name)
        member_names = [concept_by_id[member_id].name
                        for member_id in shape.members if member_id in concept_by_id]
        if not code_words or not member_names:
            continue
        if any(code_words & _meaningful_words(name) for name in member_names):
            continue
        mismatches.append({
            "code_name": code.code_name,
            "n_resp": len(shape.resp_ids),
            "members": member_names,
        })
    return mismatches


# ---------------------------------------------------------------------------
# find_duplicate_definitions — deterministische achtervang tegen twee codes
# met dezelfde definitie
# ---------------------------------------------------------------------------

def _normalized_definition(definition: str) -> str:
    """Lowercased, whitespace-collapsed comparison key — catches a definition
    that is identical apart from capitalization or incidental whitespace,
    without attempting full fuzzy matching."""
    return re.sub(r"\s+", " ", definition.strip().lower())


def find_duplicate_definitions(
    codes: List[ConsolidatedCode], shapes: List[CodeShape],
) -> List[dict]:
    """Deterministic check: do two different codes in the assembled codebook
    carry the same definition? Two codes cannot both be coded against with
    the same definition — a coder has no way to choose between them, so the
    codebook is unusable at those entries. This is not hypothetical: on a
    real run, a code's definition was a byte-for-byte copy of a different
    code's text, describing that other code's members and not its own.
    Whether the writer itself produced the duplicate text or an assembly step
    attached the wrong shape's text, nothing downstream noticed — this check
    exists to always notice. `codes[i]` must be the text for `shapes[i]`, the
    same positional contract as `resolve_duplicate_names` and
    `find_naming_mismatches`."""
    if len(codes) != len(shapes):
        raise ValueError("codes and shapes must be positional pairs of equal length")

    groups: Dict[str, List[int]] = defaultdict(list)
    for i, code in enumerate(codes):
        groups[_normalized_definition(code.definition)].append(i)

    duplicates: List[dict] = []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        duplicates.append({
            "definition": codes[indices[0]].definition,
            "codes": [
                {"code_name": codes[i].code_name, "n_resp": len(shapes[i].resp_ids)}
                for i in indices
            ],
        })
    return duplicates
