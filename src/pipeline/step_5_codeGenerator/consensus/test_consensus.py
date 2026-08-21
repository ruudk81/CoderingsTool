"""Tests voor de consensusindeling: volledige koppeling op de paartelling."""
from pipeline.step_5_codeGenerator.consensus.consensus import (
    consensus_partition, dominant_member, labels_from_clusters, min_together,
    together_from_runs)


def matrix(pairs):
    """pairs: {(a, b): aantal runs samen}."""
    return {frozenset(pair): n for pair, n in pairs.items()}


def test_drempel_rondt_naar_boven_af_zonder_drijvende_kommafout():
    """0.7 * 10 is in float 7.000000000000001; een paar dat precies 7 keer
    samen zat moet de drempel wél halen."""
    assert min_together(0.7, 10) == 7
    assert min_together(1.0, 10) == 10
    assert min_together(0.85, 10) == 9


def test_een_paar_dat_altijd_samen_zat_wordt_een_groep():
    together = matrix({("A", "B"): 10})

    assert consensus_partition(together, ["A", "B"], runs=10, tau=1.0) == [("A", "B")]


def test_ketenvorming_kan_niet_ontstaan():
    """A-B en B-C liggen vast, A-C ligt vast op APART. Enkelvoudige koppeling
    zou hier {A,B,C} maken — een groep die geen enkele run voorstelde.

    A-B en B-C zijn hier gelijk sterk (10/10); de tie-break (kleinste
    lexicografische eerste lid) legt de uitkomst vast op ("A","B") + ("C",),
    dus de hele partitie is asserteerbaar in plaats van "één van beide"."""
    together = matrix({("A", "B"): 10, ("B", "C"): 10, ("A", "C"): 0})

    clusters = consensus_partition(together, ["A", "B", "C"], runs=10, tau=0.8)

    assert clusters == [("A", "B"), ("C",)]


def test_een_attribuut_dat_nergens_de_drempel_haalt_blijft_alleen():
    """5 van de 10 bij A, 4 bij B: het twijfelgeval wordt een eigen code."""
    together = matrix({("A", "B"): 10, ("A", "X"): 5, ("B", "X"): 4})

    clusters = consensus_partition(together, ["A", "B", "X"], runs=10, tau=0.8)

    assert clusters == [("A", "B"), ("X",)]


def test_alles_samen_geeft_een_groep():
    together = matrix({("A", "B"): 10, ("B", "C"): 10, ("A", "C"): 10})

    assert consensus_partition(together, ["A", "B", "C"], runs=10, tau=1.0) == [("A", "B", "C")]


def test_uitkomst_is_onafhankelijk_van_de_invoervolgorde():
    together = matrix({("A", "B"): 9, ("B", "C"): 9, ("A", "C"): 8,
                       ("A", "D"): 1, ("B", "D"): 0, ("C", "D"): 2})

    een = consensus_partition(together, ["A", "B", "C", "D"], runs=10, tau=0.8)
    twee = consensus_partition(together, ["D", "C", "B", "A"], runs=10, tau=0.8)

    assert een == twee


def test_elk_attribuut_komt_precies_een_keer_voor():
    """De uitkomst is een partitie — dat is de MECE-garantie."""
    together = matrix({("A", "B"): 10, ("B", "C"): 3, ("A", "C"): 2})
    ids = ["A", "B", "C", "D"]

    clusters = consensus_partition(together, ids, runs=10, tau=0.8)

    geplaatst = [a for cluster in clusters for a in cluster]
    assert sorted(geplaatst) == sorted(ids)
    assert len(geplaatst) == len(set(geplaatst))


def test_labels_zijn_bruikbaar_voor_ari():
    labels = labels_from_clusters([("A", "B"), ("C",)])

    assert labels["A"] == labels["B"]
    assert labels["C"] != labels["A"]


def test_dominant_lid_is_het_zwaarste_attribuut():
    """Een consensusgroep heeft geen voorgestelde naam; `CodeShape.umbrella`
    heeft er wel een nodig voor de hernoemroute bij dubbele namen."""
    assert dominant_member(("A", "B", "C"), {"A": 10, "B": 400, "C": 25}) == "B"


def test_dominant_lid_breekt_gelijkspel_deterministisch():
    assert dominant_member(("B", "A"), {"A": 10, "B": 10}) == "A"


def test_paartelling_telt_over_alle_runs_en_negeert_ontbrekende_leden():
    """Een attribuut dat in één run ontbreekt telt daar als 'niet samen' mee
    in plaats van uit de meting te verdwijnen — dezelfde zorgvuldigheid als
    `stability.measure_stability`, waarvan deze functie niet meer leent."""
    runs = [[("A1", "A2"), ("A3",)],
            [("A1", "A2"), ("A3",)],
            [("A1",), ("A2", "A3")]]

    together = together_from_runs(runs, ["A1", "A2", "A3"])

    assert together[frozenset({"A1", "A2"})] == 2
    assert together[frozenset({"A2", "A3"})] == 1
    assert together[frozenset({"A1", "A3"})] == 0
