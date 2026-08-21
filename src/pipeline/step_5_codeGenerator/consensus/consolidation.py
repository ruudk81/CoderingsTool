"""Fase 1 van de consensuskandidaat — N runs, één requester.

Productie's `consolidation.py` bouwt een `SmoothRequester(num_tasks=1)` met een
takenlijst van één, en de salt zit in de closure van `prepare_fn`. Eén aanroep
is daar dus per constructie één taak. Deze variant zet de salt IN de taak, en
dan is N parallel wat de component altijd al deed — `process_all` leidt zijn
concurrency af uit `len(tasks)`.

Waarom dat meer is dan snelheid: dertig losse aanroepen bouwen dertig losse
requesters die elk één taak zien. De adaptieve doorvoerregeling waar die
component voor bestaat heeft dan niets te regelen.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from config import get_reasoning_params
from utils.smoothRequester import SmoothRequester

from ..attribute_cards import AttributeCard
from .config_consensus import ConsensusConfig
from .prompts_consolidation import (
    ConsolidationResult, build_consolidation_prompt, make_consolidation_model,
)

PHASE = "step5c_consolidation"


def build_tasks(cards: List[AttributeCard], salts: List[str]) -> List[Dict]:
    """Eén taak per run. Alle taken delen dezelfde kaarten — dertig trekkingen
    uit dezelfde urn — en verschillen alleen in de volgorde waarin het
    materiaal wordt aangeboden."""
    return [{"cards": cards, "salt": salt} for salt in salts]


async def resolve_consolidations(
    cards: List[AttributeCard],
    survey_question: str,
    n_respondents: int,
    language: str,
    config: ConsensusConfig,
    salts: List[str],
    verbose: bool = False,
    prompt_printer=None,
) -> Tuple[List[ConsolidationResult], int]:
    """N calls over dezelfde inventaris, parallel. Geeft de geslaagde
    resultaten terug plus hoeveel er mislukten.

    Faalcontract, en het wijkt bewust af van productie. Daar is één mislukte
    call het einde van het codeboek en dus een harde `RuntimeError`. Hier is
    één mislukte run van dertig geen ramp — de uitkomst is een gemiddelde. Maar
    onder de twee runs valt er niets te middelen: er is dan geen paar om te
    tellen en geen matrix om te vullen. Dus: hard stoppen onder twee, en
    daarboven doorgaan mét een telling die naar de `RunSet` gaat, zodat elke
    latere drempel over het juiste aantal rekent.
    """
    def prepare_fn(task):
        prompt = build_consolidation_prompt(
            task["cards"], survey_question, n_respondents, language, task["salt"])
        if prompt_printer is not None:
            prompt_printer.capture_prompt(
                step_name="code_generator_consensus",
                utility_name="resolve_consolidations",
                prompt_content=prompt,
                prompt_type="consolidation",
                metadata={
                    "model": config.model_relations,
                    "n_cards": len(task["cards"]),
                    "card_ids": [c.attribute_id for c in task["cards"]],
                    "card_names": [c.name for c in task["cards"]],
                    "language": language,
                    "salt": task["salt"],
                },
            )
        return {
            "prompt": prompt,
            "response_model": make_consolidation_model(task["cards"], task["salt"]),
            "temperature": config.temperature_relations,
            "max_tokens": config.max_tokens_relations,
            "max_retries": 2,
            "extra_kwargs": get_reasoning_params(
                config.model_relations, phase="codegen_relations"),
        }

    tasks = build_tasks(cards, salts)
    requester = SmoothRequester(
        model=config.model_relations, phase_key=PHASE, num_tasks=len(tasks),
        verbose=verbose, quiet=True,
    )
    results = await requester.process_all(
        tasks, prepare_fn,
        lambda _task, response: response, lambda _task, _reason: None,
    )
    geslaagd = [r for r in results if r is not None]
    mislukt = len(tasks) - len(geslaagd)
    if len(geslaagd) < 2:
        raise RuntimeError(
            f"Consensus heeft minstens twee geslaagde consolidatieruns nodig; "
            f"{len(geslaagd)} van {len(tasks)} kwamen terug. Zonder twee runs "
            f"is er geen paar om te tellen — dit is een harde stop, geen "
            f"fallback."
        )
    return geslaagd, mislukt
