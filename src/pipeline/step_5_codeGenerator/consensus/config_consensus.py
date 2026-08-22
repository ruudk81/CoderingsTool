"""De enige knoppentabel die deze keten heeft en productie niet."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import config as project_config

from .config_codeGenerator import CodebookConfig

# De twee beproefde configuraties: welk model draait, en met welke reasoning
# effort. Dit was tot 2026-08-21 een tweede defaultlijst in de losse
# argparse-runner die dit pakket toen nog had, los van deze dataclass —
# dezelfde keten gaf dan een ander antwoord (tau 0,7 tegen 0,8, twee polen
# tegen drie) afhankelijk van welke deur je binnenkwam. Nu is
# `ConsensusConfig` de enige plek.
CONFIGS = {
    "luna":  {"model": project_config.MODELS[("5.6", 3)].name, "effort": "medium"},
    "gpt54": {"model": project_config.MODELS[("5.4", 5)].name, "effort": "high"},
}


@dataclass
class ConsensusConfig(CodebookConfig):
    """`CodebookConfig` plus wat consensus nodig heeft.

    `tau` staat hier en niet in `config.py`, en dat is een besluit met een
    reden: hij ziet er neutraal uit en is dat niet. De eerdere keuze voor 0,5
    werd mede beargumenteerd met "dan blijft `Kosten en prijsstelling` staan"
    — afstellen op de inhoud van één dataset, in een vorm die de regel over
    use-case-agnostische prompts niet vangt. Zolang er op een tweede dataset
    niets gemeten is, blijft hij een expliciete invoer.

    `runs=30` omdat de overeenstemming tussen samenvoegingen doorstijgt tot
    dertig: 42-63% bij N=10, 53-65% bij N=15, 73% bij N=30.
    """
    runs: int = 30
    tau: float = 0.7
    two_pole: bool = True
    exclude_drains: bool = True

    # Welke van de twee beproefde configuraties draait. Bepaalt zowel het model
    # als de reasoning effort; `__post_init__` zet `model_relations` erop.
    # Gemeten: gpt-5.4 bij `high` is NIET stabieler dan luna bij `medium`
    # (mediane ARI 0,422 tegen 0,440 over 45 vergelijkingen) bij 12,5x de
    # tokenprijs — vandaar luna als standaard.
    config_name: str = "luna"

    # De enige overgebleven variatiebron tussen twee runs op identieke invoer.
    # Temperatuur bereikt de API niet (redeneermodel), en zonder salt sorteert
    # `_shuffled` op `md5(attribute_id)` — dus elke run krijgt exact dezelfde
    # volgorde en meet je alleen servergrilligheid. Met salt verschillen de runs
    # om een reden die we snappen, en is de spreiding meteen een meting: verandert
    # de indeling sterk met de volgorde, dan was ze deels een volgorde-artefact.
    salted: bool = True

    def __post_init__(self):
        if self.config_name not in CONFIGS:
            raise ValueError(
                f"onbekende config {self.config_name!r} — kies uit "
                f"{', '.join(sorted(CONFIGS))}")
        self.model_relations = CONFIGS[self.config_name]["model"]

    @property
    def effort(self) -> str:
        return CONFIGS[self.config_name]["effort"]


@contextmanager
def effort_van(config: ConsensusConfig):
    """Zet tijdelijk de reasoning effort van de consolidatiefase.

    `STEP_EFFORT` heeft de FASE als sleutel, niet het model, dus deze waarde
    geldt globaal zolang de call loopt. Vandaar het herstel in `finally`: zonder
    dat lekt de effort van een experimentconfiguratie de rest van het proces in.
    """
    origineel = project_config.STEP_EFFORT.get("codegen_relations")
    project_config.STEP_EFFORT["codegen_relations"] = config.effort
    try:
        yield
    finally:
        project_config.STEP_EFFORT["codegen_relations"] = origineel
