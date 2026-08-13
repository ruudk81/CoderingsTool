"""Stap 4 — dispatch van de schrijfcall.

De vorm (hoeveel codes, welke leden, welke richting) is al vast; deze module
vult alleen de teksten in en past het ene toegestane veto toe: `nameable: false`
op een gepoolde vorm laat die vorm vervallen. Faalt de call, of ontbreekt een
vorm in het antwoord, dan krijgt die vorm een deterministische invulling in
plaats van dat het codeboek een code verliest — de vormen blijven geldig, alleen
de tekst wordt minder rijk.

Lekdiscipline: deze module geeft nooit een respondenttelling, domein, facet of
attribuut-id aan de LLM door — zie `prompts_writer.py` voor de promptbouw zelf.

Een herschrijving die maar een DEEL van het codeboek ziet (bijv. alleen de
MECE-samengevoegde codes) kan op een naam landen die een niet-herschreven code
al draagt: `write_codebook`'s `taken_names` geeft die andere namen door zodat
de LLM ze mijdt, en `resolve_duplicate_names` is de deterministische achtervang
die de aanroeper ná het herschrijven over het VOLLEDIGE, herenigde codeboek
draait — een prompt-regel wordt hier nooit vertrouwd als enige garantie.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional, Set

from config import get_reasoning_params
from utils.llm import RateLimits
from utils.smoothRequester import SmoothRequester

from .concept_inventory import Concept
from .config_codeGenerator import CodebookConfig
from .consolidator import CodeShape
from .prompts_codeGenerator import ConsolidatedCode
from .prompts_writer import CodeText, build_writer_prompt, make_writer_model

PHASE = "step5_writer"


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


def _to_consolidated_code(text: CodeText, shape: CodeShape,
                          concept_by_id: Dict[str, Concept]) -> ConsolidatedCode:
    return ConsolidatedCode(
        code_name=text.code_name,
        definition=text.definition,
        diagnostic_test=text.diagnostic_test,
        valence=shape.valence,
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
) -> List[ConsolidatedCode]:
    """One call across all fixed code shapes. A `nameable: false` verdict on a
    `pooled` shape drops it (recorded in `log` as a VETO); the same verdict on
    a `solo` or `synonym` shape is ignored — those are single attributes and
    are by definition nameable.

    `taken_names` is for a re-write that only sees a SUBSET of the book (e.g.
    the MECE-merged codes) — the names already committed for the codes NOT in
    `shapes` this call, so the model doesn't land on one of them. This is a
    prompt-level ask, not a guarantee; see `resolve_duplicate_names` for the
    deterministic backstop the caller must still run over the full, reassembled
    codebook."""
    if not shapes:
        return []

    concept_by_id = {concept.attribute_id: concept for concept in concepts}

    def prepare_fn(task):
        prompt = build_writer_prompt(
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

    requester = SmoothRequester(
        model=config.model_writer,
        phase_key=PHASE,
        num_tasks=1,
        verbose=verbose,
        known_limits=known_limits,
        has_server_headers=has_server_headers,
        quiet=True,
    )
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
        if not text.nameable and shape.origin == "pooled":
            _record_veto(log, shape, concept_by_id)
            continue
        codes.append(_to_consolidated_code(text, shape, concept_by_id))
    return codes


def resolve_duplicate_names(
    codes: List[ConsolidatedCode], shapes: List[CodeShape], log=None,
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
    umbrella term — the constituent group name it climbed from in
    `consolidator.py` — with a number appended only if even that is already
    taken. Every rename is reported via `log.add(...)` (duck-typed, like
    `write_codebook`'s own `log`), so a resolved collision is always visible,
    never a silent rename. A codebook with no duplicate names is returned
    unchanged."""
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
            candidate = shape.umbrella
            suffix = 2
            while candidate in taken:
                candidate = f"{shape.umbrella} ({suffix})"
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
# find_naming_mismatches — deterministische achtervang tegen een naam die zijn
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
