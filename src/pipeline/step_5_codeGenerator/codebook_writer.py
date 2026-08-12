"""Stap 4 — dispatch van de schrijfcall.

De vorm (hoeveel codes, welke leden, welke richting) is al vast; deze module
vult alleen de teksten in en past het ene toegestane veto toe: `nameable: false`
op een gepoolde vorm laat die vorm vervallen. Faalt de call, of ontbreekt een
vorm in het antwoord, dan krijgt die vorm een deterministische invulling in
plaats van dat het codeboek een code verliest — de vormen blijven geldig, alleen
de tekst wordt minder rijk.

Lekdiscipline: deze module geeft nooit een respondenttelling, domein, facet of
attribuut-id aan de LLM door — zie `prompts_writer.py` voor de promptbouw zelf.
"""
from __future__ import annotations

from typing import Dict, List, Optional

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
) -> List[ConsolidatedCode]:
    """One call across all fixed code shapes. A `nameable: false` verdict on a
    `pooled` shape drops it (recorded in `log` as a VETO); the same verdict on
    a `solo` or `synonym` shape is ignored — those are single attributes and
    are by definition nameable."""
    if not shapes:
        return []

    concept_by_id = {concept.attribute_id: concept for concept in concepts}

    def prepare_fn(task):
        return {
            "prompt": build_writer_prompt(
                task["shapes"], task["concept_by_id"],
                task["dimension_diagnostic"], task["language"],
            ),
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
