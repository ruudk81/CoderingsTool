"""Van N partities naar één indeling — deterministisch, geen LLM.

`together_from_runs` bouwt de co-associatiematrix (per attribuutpaar, in
hoeveel runs de twee samen zaten); `consensus_partition` snijdt haar. Beide
horen in deze module, niet slechts de tweede: de paartelling IS de claim van
de kandidaat, dus lenen uit `../stability.py` zou de kern van deze keten laten
steunen op de module die een consensusindeling juist afwijst.

`stability.py` wijst een consensusindeling af omdat transitieve sluiting A-B-C
aan elkaar plakt zodra A-B en B-C vastliggen, ook wanneer A-C juist vast op
apart staat. Dat bezwaar treft ENKELVOUDIGE koppeling. Hier staat volledige
koppeling: een groep ontstaat alleen als élk paar erin de drempel haalt, dus
A-C op apart sluit {A,B,C} uit. De ketenvorming kan niet ontstaan in plaats van
achteraf gerepareerd te worden — dezelfde soort garantie als de
valentiesplitsing in `build_shapes`.
"""
from __future__ import annotations

import math
from collections import Counter
from itertools import combinations
from typing import Dict, FrozenSet, List, Sequence, Tuple


def together_from_runs(
    runs: Sequence[Sequence[Sequence[str]]],
    attribute_ids: Sequence[str],
) -> Dict[FrozenSet[str], int]:
    """Co-associatiematrix uit opgeslagen partities: per attribuutpaar, in
    hoeveel runs de twee samen in één cluster zaten.

    Dit IS stage 2 van de kandidaat — de paartelling is precies de claim die
    `consensus_partition` hierboven snijdt — en hoort daarom hier, niet
    geleend uit `../stability.py`. Die module telt hetzelfde soort paar, maar
    wijst een consensusindeling juist AF (zie haar eigen docstring); haar
    rekenkern gebruiken zou deze indeling laten steunen op de module die
    beargumenteert waarom zo'n indeling niet volstaat, en zou bij promotie
    meteen een probleem geven — `stability.py` gaat dan met de rest van de
    huidige keten met pensioen naar `_quarantine_v1/`.

    `attribute_ids` bepaalt welke paren geteld worden, zodat een attribuut dat
    in één run ontbreekt als "niet samen" meetelt in plaats van uit de meting
    te verdwijnen — dezelfde zorgvuldigheid als `stability.measure_stability`.
    """
    counts: Counter = Counter()
    for clusters in runs:
        counts.update(
            frozenset(pair)
            for cluster in clusters
            for pair in combinations(sorted(cluster), 2)
        )
    return {
        frozenset(pair): counts.get(frozenset(pair), 0)
        for pair in combinations(sorted(set(attribute_ids)), 2)
    }


def min_together(tau: float, runs: int) -> int:
    """Hoe vaak een paar minstens samen moet zijn geweest om te mogen koppelen.

    Naar boven afgerond met een marge, want `0.7 * 10` is in drijvende komma
    7.000000000000001: zonder die marge zou een paar dat precies 7 van de 10
    keer samen zat de drempel missen.
    """
    return math.ceil(tau * runs - 1e-9)


def consensus_partition(
    together: Dict[FrozenSet[str], int],
    attribute_ids: Sequence[str],
    runs: int,
    tau: float,
) -> List[Tuple[str, ...]]:
    """Clusters waarin ELK paar minstens `tau * runs` keer samen zat.

    Agglomeratief: begin met losse attributen, voeg steeds de twee clusters
    samen met de hoogste MINIMALE paartelling, tot die onder de drempel zakt.
    Een attribuut dat met niemand de drempel haalt blijft alleen — dat is het
    eerlijke antwoord voor een twijfelgeval, in plaats van het stilletjes naar
    de sterkste buurman te schuiven.

    Deterministisch, want dat is wat hier gemeten wordt: gelijkspel wordt
    gebroken op (hoogste minimum, grootste gecombineerde omvang, lexicografisch
    kleinste lid), en de uitvoer is gesorteerd. Dezelfde matrix geeft dezelfde
    partitie, ongeacht de volgorde van `attribute_ids`.
    """
    threshold = min_together(tau, runs)
    clusters: List[Tuple[str, ...]] = [(a,) for a in sorted(set(attribute_ids))]

    def link(left: Tuple[str, ...], right: Tuple[str, ...]) -> int:
        return min(together.get(frozenset((a, b)), 0) for a in left for b in right)

    while True:
        best: Tuple[int, int] | None = None
        best_key = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                score = link(clusters[i], clusters[j])
                if score < threshold:
                    continue
                key = (-score, -(len(clusters[i]) + len(clusters[j])),
                       clusters[i][0], clusters[j][0])
                if best_key is None or key < best_key:
                    best, best_key = (i, j), key
        if best is None:
            return sorted(clusters)
        i, j = best
        merged = tuple(sorted(clusters[i] + clusters[j]))
        clusters = [c for k, c in enumerate(clusters) if k not in (i, j)]
        clusters.append(merged)
        clusters.sort()


def labels_from_clusters(clusters: Sequence[Sequence[str]]) -> Dict[str, str]:
    """Attribuut -> groepslabel, voor ARI. De labelnamen zijn betekenisloos;
    ARI vergelijkt alleen wie bij wie zit."""
    return {member: f"g{index}"
            for index, cluster in enumerate(clusters)
            for member in cluster}


def dominant_member(cluster: Sequence[str], weight_by_id: Dict[str, int]) -> str:
    """Het zwaarste lid van een cluster, als vervanger voor de naam die deel 1
    niet meer levert.

    Een consensusgroep is niet door één modelcall voorgesteld en heeft dus geen
    `proposed_name`. Die vult in productie `CodeShape.umbrella`, en dat veld
    wordt gebruikt als hernoemkandidaat zodra twee codes dezelfde naam krijgen.
    Zonder vervanger staat daar een lege string op precies het moment dat er
    iets misgaat. Gelijkspel gaat naar het lexicografisch kleinste id, zodat de
    keuze reproduceert.
    """
    return min(cluster, key=lambda member: (-weight_by_id.get(member, 0), member))
