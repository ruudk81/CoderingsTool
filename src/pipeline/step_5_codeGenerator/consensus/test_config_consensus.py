"""De kandidaat heeft vier knoppen die productie niet heeft."""
import pytest

from pipeline.step_5_codeGenerator.consensus.config_consensus import ConsensusConfig


def test_consensus_config_erft_de_productievelden():
    """Zonder erfenis lopen de twee ketens uit elkaar op drempel, model en
    max_tokens — en dan vergelijk je twee dingen die niet vergelijkbaar zijn."""
    config = ConsensusConfig()

    assert hasattr(config, "model_relations")
    assert hasattr(config, "model_writer")
    assert hasattr(config, "max_tokens_relations")


def test_consensus_config_draagt_de_vier_eigen_knoppen():
    """tau blijft een expliciete invoer en gaat NIET naar config.py — zie
    dev/WORK.md: de keuze voor een waarde is op één dataset beredeneerd."""
    config = ConsensusConfig(runs=10, tau=0.5, two_pole=False, exclude_drains=False)

    assert (config.runs, config.tau) == (10, 0.5)
    assert (config.two_pole, config.exclude_drains) == (False, False)


def test_standaard_is_dertig_runs_en_tau_zeven():
    """De schaalcurve stijgt door tot N=30 (42-63% bij N=10, 73% bij N=30),
    dus tien runs meet het middel op zijn zwakst."""
    config = ConsensusConfig()

    assert config.runs == 30
    assert config.tau == 0.7


def test_salted_hoort_bij_de_config_en_niet_bij_de_aanroep():
    """De volgordesalt is een knop van de consolidatie, geen argument dat elke
    aanroeper opnieuw moet doorgeven. Zonder salt meet je kale servervariatie;
    dat is een geldige meting, maar het moet een keuze in het blok zijn."""
    assert ConsensusConfig().salted is True
    assert ConsensusConfig(salted=False).salted is False


def test_alle_vijf_de_knoppen_staan_op_een_plek():
    """De enige knoppentabel. Stond er een tweede defaultlijst naast — zoals de
    argparse-vlaggen deden — dan geeft dezelfde keten een ander antwoord
    afhankelijk van welke deur je binnenkomt (tau 0,7 tegen 0,8, twee polen
    tegen drie)."""
    config = ConsensusConfig()

    assert (config.runs, config.tau) == (30, 0.7)
    assert (config.two_pole, config.exclude_drains, config.salted) == (True, True, True)


def test_onbekende_config_naam_faalt_hard():
    """Een stille verkeerde route (bv. een tikfout) mag geen default worden."""
    with pytest.raises(ValueError):
        ConsensusConfig(config_name="nope")
