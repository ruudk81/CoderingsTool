"""Tests voor de analysefuncties van het consensus-experiment."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from analysis import consensus_ari, histogram, pairwise_ari, tau_sweep


def matrix(pairs):
    return {frozenset(pair): n for pair, n in pairs.items()}


def test_pairwise_ari_geeft_een_waarde_per_runpaar():
    """Drie runs -> drie vergelijkingen; tien runs zouden er 45 geven."""
    runs = [[("A", "B"), ("C",)],
            [("A", "B"), ("C",)],
            [("A",), ("B", "C")]]

    waarden = pairwise_ari(runs)

    assert len(waarden) == 3
    assert waarden[0] == 1.0  # run 1 tegen run 2: identiek


def test_histogram_telt_paren_per_aantal_runs_samen():
    together = matrix({("A", "B"): 10, ("A", "C"): 0, ("B", "C"): 5})

    telling = histogram(together, runs=10)

    assert len(telling) == 11  # 0 t/m 10
    assert telling[0] == 1
    assert telling[5] == 1
    assert telling[10] == 1


def test_tau_sweep_geeft_per_drempel_de_vorm_van_de_indeling():
    together = matrix({("A", "B"): 10, ("A", "C"): 5, ("B", "C"): 5})

    rijen = tau_sweep(together, ["A", "B", "C"], runs=10, taus=[1.0, 0.5])

    streng = next(r for r in rijen if r["tau"] == 1.0)
    los = next(r for r in rijen if r["tau"] == 0.5)

    assert streng["n_groups"] == 2      # {A,B} en {C}
    assert streng["n_solo"] == 1
    assert streng["largest"] == 2
    assert los["n_groups"] == 1         # alles haalt 5/10


def test_consensus_ari_is_1_voor_twee_identieke_indelingen():
    a = [("A", "B"), ("C",)]
    b = [("A", "B"), ("C",)]

    assert consensus_ari(a, b) == 1.0


def test_consensus_ari_is_ook_1_voor_twee_indelingen_van_louter_solos():
    """Louter solo's is geen NaN maar een vals perfecte score: maximum en
    kansverwachting vallen samen (zie `adjusted_rand_index`), dus ARI is 1.0.
    Dat is exact de reden dat `compare` de degeneratieverdict naast de ARI
    moet drukken — het getal alleen kan een gedegenereerde indeling niet
    verraden."""
    a = [("A",), ("B",), ("C",)]
    b = [("A",), ("B",), ("C",)]

    assert consensus_ari(a, b) == 1.0
