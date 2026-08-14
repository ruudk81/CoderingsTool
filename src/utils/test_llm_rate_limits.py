"""Tests voor het uitlezen van het quotum (geen netwerk).

Azure serveert de Responses API op `/openai/v1/`, en die route geeft geen
`x-ratelimit-*` terug — Microsoft houdt dat bij als known issue. De klassieke
deployment-route rapporteert ze nog wel, voor hetzelfde quotum. Wat hier
vastligt is dat de tweede poging alleen vuurt als de eerste niets oplevert.
"""
import asyncio

import pytest

import utils.llm as llm
from utils.llm import RateLimits, extract_rate_limits_from_response


class _Resp:
    def __init__(self, headers):
        self.headers = headers


V1_MET_QUOTUM = {"x-ratelimit-limit-tokens": "7000000",
                 "x-ratelimit-limit-requests": "7000",
                 "openai-processing-ms": "120"}
V1_ZONDER_QUOTUM = {"apim-request-id": "x", "x-ms-region": "Sweden Central"}


def _patch(monkeypatch, v1_headers, classic_limits, calls):
    """Vervangt de v1-call en de klassieke probe; telt wie er aangeroepen wordt."""
    class _Raw:
        async def create(self, **kw):
            calls.append("v1")
            return _Resp(v1_headers)

    class _Responses:
        with_raw_response = _Raw()

    class _Client:
        responses = _Responses()

    monkeypatch.setattr(llm, "API_PROVIDER", "azure")
    monkeypatch.setattr(llm, "get_azure_route",
                        lambda m: ("https://x.example.com/", "k", "dep"))
    monkeypatch.setattr(llm, "AsyncOpenAI", lambda **kw: _Client())

    async def fake_classic(endpoint, api_key, deployment):
        calls.append("classic")
        return classic_limits

    monkeypatch.setattr(llm, "_probe_azure_classic", fake_classic)


# =============================================================================
# HET UITLEZEN ZELF
# =============================================================================

def test_ontbrekende_headers_geven_nul():
    limits = extract_rate_limits_from_response(_Resp(V1_ZONDER_QUOTUM))
    assert limits.tokens_per_minute == 0
    assert limits.requests_per_minute == 0


def test_aanwezige_headers_worden_gelezen():
    limits = extract_rate_limits_from_response(_Resp(V1_MET_QUOTUM))
    assert limits.tokens_per_minute == 7_000_000
    assert limits.requests_per_minute == 7_000


# =============================================================================
# DE TWEEDE POGING
# =============================================================================

def test_v1_levert_quotum_dan_geen_tweede_poging(monkeypatch):
    """Repareert Azure de headers op v1, dan stopt de tweede probe vanzelf."""
    calls = []
    _patch(monkeypatch, V1_MET_QUOTUM, RateLimits(1, 1), calls)
    limits, has_headers = asyncio.run(llm.fetch_rate_limits("gpt-5.6-luna"))
    assert calls == ["v1"]
    assert limits.tokens_per_minute == 7_000_000
    assert has_headers is True


def test_v1_zonder_quotum_valt_terug_op_de_klassieke_route(monkeypatch):
    calls = []
    _patch(monkeypatch, V1_ZONDER_QUOTUM, RateLimits(7_000_000, 7_000), calls)
    limits, has_headers = asyncio.run(llm.fetch_rate_limits("gpt-5.6-luna"))
    assert calls == ["v1", "classic"]
    assert limits.tokens_per_minute == 7_000_000
    assert limits.requests_per_minute == 7_000


def test_has_server_headers_blijft_van_de_echte_route(monkeypatch):
    """De klassieke route zegt niets over per-request timing op v1. Zou die
    vlag meeliften, dan koos de requester de header-aware controller voor een
    route die geen headers stuurt."""
    calls = []
    _patch(monkeypatch, V1_ZONDER_QUOTUM, RateLimits(7_000_000, 7_000), calls)
    _, has_headers = asyncio.run(llm.fetch_rate_limits("gpt-5.6-luna"))
    assert has_headers is False


def test_beide_routes_leeg_geeft_nul_terug(monkeypatch):
    """Geen verzonnen getal hier — de aanroeper beslist wat er zonder quotum
    gebeurt, en die kiest de gemeten capaciteit."""
    calls = []
    _patch(monkeypatch, V1_ZONDER_QUOTUM, RateLimits(0, 0), calls)
    limits, _ = asyncio.run(llm.fetch_rate_limits("gpt-5.6-luna"))
    assert calls == ["v1", "classic"]
    assert limits.tokens_per_minute == 0


def test_half_quotum_telt_als_geen_quotum(monkeypatch):
    """Alleen tokens en geen requests is niet bruikbaar: de requester deelt
    door allebei."""
    calls = []
    _patch(monkeypatch, {"x-ratelimit-limit-tokens": "7000000"},
           RateLimits(7_000_000, 7_000), calls)
    limits, _ = asyncio.run(llm.fetch_rate_limits("gpt-5.6-luna"))
    assert calls == ["v1", "classic"]
    assert limits.requests_per_minute == 7_000


# =============================================================================
# DE KLASSIEKE PROBE FAALT NOOIT LUID
# =============================================================================

def test_klassieke_probe_slikt_een_fout_en_geeft_nul(monkeypatch):
    """Die route verdwijnt ooit. Als dat gebeurt mag de run niet omvallen op
    het ophalen van een getal dat we ook kunnen missen."""
    class _Boom:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **kw):
            raise RuntimeError("route weg")

    monkeypatch.setattr(llm.httpx, "AsyncClient", lambda **kw: _Boom())
    limits = asyncio.run(llm._probe_azure_classic("https://x/", "k", "dep"))
    assert limits.tokens_per_minute == 0
    assert limits.requests_per_minute == 0


def test_klassieke_probe_pint_zijn_api_version():
    """Deze route wordt alleen gebruikt om headers te lezen, nooit om werk te
    doen — hij mag niet meebewegen met wat de Responses API doet."""
    assert llm.AZURE_CLASSIC_PROBE_API_VERSION == "2024-10-21"
