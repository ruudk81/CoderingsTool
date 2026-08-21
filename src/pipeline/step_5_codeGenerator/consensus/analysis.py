"""Wat er uit de runs af te lezen valt — pure functies, geen LLM, geen IO.

Vier metingen, elk met een eigen doel:
- `pairwise_ari`: hoe onstabiel is deel 1 op deze configuratie? Tien runs geven
  45 vergelijkingen, dus een verdeling in plaats van één punt.
- `histogram`: heeft de matrix de goede vorm? Bijna-binair betekent dat
  consensus niets toevoegt; overwegend middenin betekent dat er niets te
  koppelen valt. Consensus werkt alleen bij een duidelijke kern met een dunne
  onzekere schil.
- `tau_sweep`: welke drempel levert een bruikbare indeling?
- `consensus_ari`: de hoofdmaat — hoe goed komt de consensusindeling van de ene
  set overeen met die van de andere?
- `merge_recurrence`: dezelfde vergelijking op de eenheid die het codeboek
  bepaalt — welke SAMENVOEGINGEN komen terug. ARI en ruwe paarovereenstemming
  tellen ook de honderden beslissingen over attributen die toch alleen blijven,
  en op dit materiaal domineren die het getal.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, FrozenSet, List, Sequence, Set, Tuple

from pipeline.step_3_ideaExtractor.measure_stability import adjusted_rand_index

from ..grouping import Group
from ..stability import measure_stability
from .consensus import consensus_partition, labels_from_clusters


def pairwise_ari(runs: Sequence[Sequence[Sequence[str]]]) -> List[float]:
    """ARI tussen elk paar runs. Bij tien runs zijn dat 45 waarden.

    Dat is bewust geen gemiddelde: de spreiding is de boodschap. Één
    vergelijking tussen twee runs is een punt, en een punt zegt niets over hoe
    breed de wolk is.
    """
    labellings = [labels_from_clusters(clusters) for clusters in runs]
    return [adjusted_rand_index(a, b) for a, b in combinations(labellings, 2)]


def histogram(together: Dict[FrozenSet[str], int], runs: int) -> List[int]:
    """Hoeveel paren zaten 0, 1, ... runs samen. Index = aantal runs samen."""
    counts = [0] * (runs + 1)
    for n in together.values():
        counts[n] += 1
    return counts


def consensus_ari(a: Sequence[Sequence[str]], b: Sequence[Sequence[str]]) -> float:
    """ARI tussen twee consensusindelingen — het getal waar `compare` om draait.

    Let op: een indeling van louter solo's scoort hier 1.0, niet NaN. Dat is
    geen bug van `adjusted_rand_index` — bij louter solo's vallen het maximum
    en de kansverwachting samen, dus is 1.0 wiskundig de uitkomst. Het is wél
    een vals perfecte score, en precies waarom `compare` de degeneratieverdict
    naast de ARI moet drukken in plaats van het getal op zichzelf te tonen.
    """
    return adjusted_rand_index(labels_from_clusters(a), labels_from_clusters(b))


def _merges(clusters: Sequence[Sequence[str]]) -> Set[FrozenSet[str]]:
    """De groepen die daadwerkelijk iets samenvoegen. Een solo is er geen."""
    return {frozenset(cluster) for cluster in clusters if len(cluster) >= 2}


def _together_pairs(clusters: Sequence[Sequence[str]]) -> Set[FrozenSet[str]]:
    """Elk paar dat in deze indeling in dezelfde groep zit."""
    return {frozenset(pair)
            for cluster in clusters
            for pair in combinations(sorted(cluster), 2)}


def merge_recurrence(a: Sequence[Sequence[str]],
                     b: Sequence[Sequence[str]]) -> dict:
    """Welke samenvoegingen komen in beide indelingen terug.

    Dit is de vraag waar het codeboek van afhangt, en het is niet de vraag die
    ARI beantwoordt. ARI weegt élke paarbeslissing even zwaar, dus op een dunne
    indeling — 49 attributen, 26 groepen, 13 solo's — gaat het getal vooral
    over attributen die toch alleen blijven. Dezelfde verwarring zat in de
    ruwe paarovereenstemming van 89-90% in `WORK.md`: bijna volledig te danken
    aan paren die in béíde runs apart zaten.

    Twee getallen, beide met alleen samengevoegd materiaal in de noemer:

    - `identical`: hoeveel samenvoegingen letterlijk in allebei staan. Streng —
      {A,B,C} tegenover {A,B} telt niet mee, want dat is een andere code.
    - `pair_agreement`: van alle paren die minstens één van de twee samenvoegde,
      welk deel voegden ze allebei samen. Dit vangt de gedeeltelijke overlap die
      `identical` per definitie mist.

    `pair_agreement` is `None` wanneer geen van beide indelingen ook maar iets
    samenvoegt. Dat is geen 1.0: twee verzamelingen solo's zijn het nergens
    oneens omdat er niets te beslissen viel — dezelfde vals perfecte score die
    `consensus_ari` daar wel afdrukt, en die daar alleen te verdragen is omdat
    `compare` de degeneratieverdict ernaast zet.
    """
    pairs_a, pairs_b = _together_pairs(a), _together_pairs(b)
    beide = pairs_a & pairs_b
    minstens_een = pairs_a | pairs_b
    return {
        "merges_a": len(_merges(a)),
        "merges_b": len(_merges(b)),
        "identical": len(_merges(a) & _merges(b)),
        "pair_agreement": len(beide) / len(minstens_een) if minstens_een else None,
    }


def tau_sweep(
    together: Dict[FrozenSet[str], int],
    attribute_ids: Sequence[str],
    runs: int,
    taus: Sequence[float],
) -> List[dict]:
    """Vorm van de indeling per drempel: hoeveel groepen, hoe groot de grootste,
    hoeveel attributen blijven alleen."""
    rows = []
    for tau in taus:
        clusters = consensus_partition(together, attribute_ids, runs, tau)
        rows.append({
            "tau": tau,
            "n_groups": len(clusters),
            "largest": max((len(c) for c in clusters), default=0),
            "n_solo": sum(1 for c in clusters if len(c) == 1),
        })
    return rows


def together_from_runs(
    runs: Sequence[Sequence[Tuple[str, ...]]],
    attribute_ids: Sequence[str],
) -> Dict[FrozenSet[str], int]:
    """Co-associatiematrix uit opgeslagen partities.

    `measure_stability` telt per attribuutpaar in hoeveel runs de twee samen
    zaten — precies deze matrix — inclusief de zorgvuldigheid dat een attribuut
    dat in één run ontbreekt als "niet samen" meetelt in plaats van uit de
    meting te verdwijnen. Het verwacht `Group`-objecten en op schijf staan
    tuples; die vertaling is alles wat hier gebeurt.
    """
    as_groups = [
        [Group(member_ids=tuple(cluster), proposed_name="", explanation="")
         for cluster in run]
        for run in runs
    ]
    return measure_stability(as_groups, list(attribute_ids)).together
