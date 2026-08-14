"""Tests voor de dispatch-spreiding van SmoothRequester.

De spreiding bestaat om één ding te voorkomen: dat de eerste golf zware
requests als één muur bij de server aankomt. Twee dingen gingen daarin mis en
worden hier vastgelegd — hij stond aan op een geleende schatting, en hij stopte
nooit.
"""
import pytest

from utils.llm import RateLimits
from utils.perfModel import Prediction
from utils.smoothRequester import (
    DISPATCH_DELAY_P50_THRESHOLD,
    DISPATCH_DELAY_SPREAD_FACTOR,
    SmoothRequester,
)


def _requester(monkeypatch, *, p50, origin, num_tasks=1451):
    """Een requester met een voorgeschreven warm start, zonder netwerk."""
    pred = Prediction()
    pred.p50_latency_s = p50
    pred.avg_tokens = 1500
    pred.expected_input_tokens = 1200
    pred.expected_output_tokens = 300
    pred.origins = {"avg_tokens": origin, "p50_latency_s": "curve"}

    import utils.smoothRequester as mod
    monkeypatch.setattr(mod.perf_model, "predict", lambda model, phase: pred)

    return SmoothRequester(
        model="gpt-5.6-luna", phase_key="step4_assignment", num_tasks=num_tasks,
        verbose=False, known_limits=RateLimits(105_000, 1_000),
        has_server_headers=True, show_setup=False, quiet=True,
    )


# =============================================================================
# FIX 2 — niet spreiden op een geleende schatting
# =============================================================================

def test_geen_spreiding_wanneer_de_schatting_uit_de_pool_komt():
    """Zonder eigen historie is de tokenschatting een gemiddelde over fasen die
    1,5k en 5,5k tokens doen. Een lichte fase erft dan de voorzichtigheid van
    een zware — dat kostte op 2026-08-13 zes minuten op 8% tokenbudget."""
    def check(monkeypatch):
        r = _requester(monkeypatch, p50=7.85, origin="pool")
        assert r._dispatch_delay == 0.0
    with pytest.MonkeyPatch.context() as mp:
        check(mp)


def test_wel_spreiding_wanneer_de_fase_zichzelf_zwaar_heeft_getoond():
    with pytest.MonkeyPatch.context() as mp:
        r = _requester(mp, p50=17.0, origin="phase")
        verwacht = (17.0 - DISPATCH_DELAY_P50_THRESHOLD) / DISPATCH_DELAY_SPREAD_FACTOR
        assert r._dispatch_delay == pytest.approx(verwacht)


def test_lichte_fase_met_eigen_historie_spreidt_niet():
    """Onder de drempel is er niets te ontzien."""
    with pytest.MonkeyPatch.context() as mp:
        r = _requester(mp, p50=2.8, origin="phase")
        assert r._dispatch_delay == 0.0


def test_zonder_p50_geen_spreiding():
    """Een koude fase gokt niet dat ze zwaar is."""
    with pytest.MonkeyPatch.context() as mp:
        r = _requester(mp, p50=None, origin="phase")
        assert r._dispatch_delay == 0.0


# =============================================================================
# FIX 1 — spreiden stopt zodra de pijplijn vol is
# =============================================================================

def test_spreiding_geldt_tijdens_het_vullen():
    with pytest.MonkeyPatch.context() as mp:
        r = _requester(mp, p50=17.0, origin="phase")
        r.optimal_concurrency = 200
        r._dispatch_start = 1000.0
        assert r._stagger_target(1) == pytest.approx(1000.0 + r._dispatch_delay)
        assert r._stagger_target(199) == pytest.approx(1000.0 + 199 * r._dispatch_delay)


def test_spreiding_stopt_zodra_de_pijplijn_vol_is():
    """De teller liep door over de hele fase en werd daarmee een permanent
    doorvoerplafond van 1/delay, dat niets van RPM of TPM weet."""
    with pytest.MonkeyPatch.context() as mp:
        r = _requester(mp, p50=17.0, origin="phase")
        r.optimal_concurrency = 200
        r._dispatch_start = 1000.0
        assert r._stagger_target(200) is None
        assert r._stagger_target(1450) is None


def test_eerste_dispatch_wacht_nooit():
    with pytest.MonkeyPatch.context() as mp:
        r = _requester(mp, p50=17.0, origin="phase")
        r.optimal_concurrency = 200
        r._dispatch_start = 1000.0
        assert r._stagger_target(0) is None


def test_zonder_vertraging_wacht_niemand():
    with pytest.MonkeyPatch.context() as mp:
        r = _requester(mp, p50=2.8, origin="phase")
        r.optimal_concurrency = 200
        r._dispatch_start = 1000.0
        assert r._stagger_target(5) is None


def test_concurrency_nul_blokkeert_niet():
    """optimal_concurrency staat op 0 tot _probe_and_setup draait; een
    deling-door-nul of een eeuwige wachtrij is daar geen acceptabele uitkomst."""
    with pytest.MonkeyPatch.context() as mp:
        r = _requester(mp, p50=17.0, origin="phase")
        r.optimal_concurrency = 0
        r._dispatch_start = 1000.0
        assert r._stagger_target(1) is None


def test_groeiende_concurrency_spreidt_de_nieuwe_slots():
    """Workers komen erbij als de regelaar de concurrency verhoogt; die nieuwe
    golf is een echte golf en mag gespreid worden."""
    with pytest.MonkeyPatch.context() as mp:
        r = _requester(mp, p50=17.0, origin="phase")
        r._dispatch_start = 1000.0
        r.optimal_concurrency = 200
        assert r._stagger_target(250) is None
        r.optimal_concurrency = 330
        assert r._stagger_target(250) is not None


# =============================================================================
# Een onbekende limiet klemt de gemeten capaciteit niet af
# =============================================================================

def _setup(monkeypatch, *, probe_limits, num_tasks=1236, knee=267):
    """Een requester die _probe_and_setup heeft doorlopen, zonder netwerk."""
    pred = Prediction()
    pred.p50_latency_s = 2.4
    pred.avg_tokens = 1500
    pred.expected_input_tokens, pred.expected_output_tokens = 1200, 300
    pred.concurrency = knee
    pred.origins = {"avg_tokens": "phase", "p50_latency_s": "curve"}

    import utils.smoothRequester as mod
    monkeypatch.setattr(mod.perf_model, "predict", lambda model, phase: pred)

    async def fake_fetch(model):
        return probe_limits, False

    monkeypatch.setattr(mod, "llm_fetch_rate_limits", fake_fetch)
    r = mod.SmoothRequester(
        model="gpt-5.6-luna", phase_key="step3_idea_extraction",
        num_tasks=num_tasks, verbose=False, show_setup=False, quiet=True)
    import asyncio
    asyncio.run(r._probe_and_setup(num_tasks))
    return r


def test_echte_limieten_klemmen_wel():
    """Met een limiet van de API zelf is min(rate, server, tasks) juist goed."""
    with pytest.MonkeyPatch.context() as mp:
        r = _setup(mp, probe_limits=RateLimits(7_000_000, 7_000))
        assert r._limits_are_known is True
        assert r.optimal_concurrency == min(
            r._rate_limit_concurrency, r._server_concurrency, 1236)


def test_onbekende_limiet_klemt_de_gemeten_knie_niet_af():
    """Een geraden limiet is geen bewijs. Gemeten 2026-08-14: de quota-headers
    vielen weg, FALLBACK_RPM=100 gaf rate_concurrency=3 tegen een gemeten
    server_concurrency van 267, en een fase van 41 seconden duurde veertien
    minuten."""
    with pytest.MonkeyPatch.context() as mp:
        r = _setup(mp, probe_limits=RateLimits(0, 0), knee=267)
        assert r._limits_are_known is False
        assert r._rate_limit_concurrency < 10       # de geraden limiet is krap
        assert r.optimal_concurrency == 267         # en telt niet mee


def test_onbekende_limiet_blijft_begrensd_door_het_aantal_taken():
    with pytest.MonkeyPatch.context() as mp:
        r = _setup(mp, probe_limits=RateLimits(0, 0), knee=267, num_tasks=12)
        assert r.optimal_concurrency == 12


def test_zonder_gemeten_knie_valt_het_terug_op_de_koude_bovengrens():
    """Geen limieten én geen historie: dan is de koude bovengrens het enige
    getal dat er is — nog steeds beter dan een verzonnen RPM."""
    import utils.smoothRequester as mod
    with pytest.MonkeyPatch.context() as mp:
        r = _setup(mp, probe_limits=RateLimits(0, 0), knee=None)
        assert r.optimal_concurrency == min(mod.COLD_START_CAP, 1236)


def test_de_tokenbucket_gebruikt_de_terugval_wel():
    """Iets moet de bucket voeden; alleen de concurrency-conclusie vervalt."""
    with pytest.MonkeyPatch.context() as mp:
        from config import FALLBACK_TPM
        r = _setup(mp, probe_limits=RateLimits(0, 0))
        assert r.rate_limits.tokens_per_minute == FALLBACK_TPM
