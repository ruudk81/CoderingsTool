"""De vier knoppen die deze keten heeft en productie niet."""
from __future__ import annotations

from dataclasses import dataclass

from ..config_codeGenerator import CodebookConfig


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
