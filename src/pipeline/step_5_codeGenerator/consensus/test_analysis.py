"""Tests voor de analysefuncties van het consensus-experiment."""
from pipeline.step_5_codeGenerator.consensus.analysis import (
    consensus_ari, histogram, merge_recurrence, pairwise_ari, tau_sweep,
    together_from_runs)


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


def test_merge_recurrentie_telt_alleen_samenvoegingen():
    """Een solo is geen samenvoeging. Telt hij mee, dan meet je vooral hoeveel
    attributen alleen bleven — en dat is precies het bezwaar tegen ARI op dit
    materiaal."""
    a = [("A", "B"), ("C", "D"), ("E",)]
    b = [("A", "B"), ("C",), ("D",), ("E",)]

    uitkomst = merge_recurrence(a, b)

    assert uitkomst["merges_a"] == 2      # {A,B} en {C,D}
    assert uitkomst["merges_b"] == 1      # alleen {A,B}
    assert uitkomst["identical"] == 1


def test_een_deels_overlappende_groep_telt_niet_als_identiek():
    """Identiek is identiek. {A,B,C} tegenover {A,B} is een andere code, geen
    reproductie ervan — de paarovereenstemming vangt de gedeeltelijke
    overlap."""
    a = [("A", "B", "C")]
    b = [("A", "B"), ("C",)]

    uitkomst = merge_recurrence(a, b)

    assert uitkomst["identical"] == 0
    assert uitkomst["pair_agreement"] == 1 / 3   # A-B van de drie paren


def test_paarovereenstemming_negeert_wat_in_beide_apart_zit():
    """De maat die in WORK.md misleidde telde apart-apart mee en kwam daardoor
    op 89-90% terwijl de indelingen nauwelijks op elkaar leken. Hier staat
    alleen wat minstens één van de twee samenvoegde in de noemer."""
    a = [("A", "B"), ("C",), ("D",), ("E",), ("F",)]
    b = [("A", "B"), ("C",), ("D",), ("E",), ("F",)]

    uitkomst = merge_recurrence(a, b)

    # Er zijn 15 paren; 14 zitten in beide apart en tellen niet mee.
    assert uitkomst["pair_agreement"] == 1.0


def test_zonder_enige_samenvoeging_is_paarovereenstemming_ongedefinieerd():
    """Twee indelingen van louter solo's zijn het overal 'eens', maar er valt
    niets overeen te stemmen. 1.0 zou hier dezelfde vals perfecte score zijn
    die `consensus_ari` bij louter solo's geeft — daarom None, zodat de
    aanroeper het moet benoemen in plaats van het te kunnen aflezen."""
    a = [("A",), ("B",)]
    b = [("A",), ("B",)]

    assert merge_recurrence(a, b)["pair_agreement"] is None


def test_brug_telt_paren_over_runs():
    runs = [[("A1", "A2"), ("A3",)],
            [("A1", "A2"), ("A3",)],
            [("A1",), ("A2", "A3")]]

    together = together_from_runs(runs, ["A1", "A2", "A3"])

    assert together[frozenset({"A1", "A2"})] == 2
    assert together[frozenset({"A2", "A3"})] == 1
    assert together[frozenset({"A1", "A3"})] == 0
