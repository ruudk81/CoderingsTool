"""Tests for reading the quota (no network).

Azure serves the Responses API on `/openai/v1/`, and that route returns no
`x-ratelimit-*` — Microsoft tracks this as a known issue. The classic deployment
route still reports them, for the same quota. What is pinned here is that the
second attempt only fires when the first yields nothing.
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
    """Replaces the v1 call and the classic probe; counts who gets called."""
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

def test_v1_yields_quota_then_no_second_attempt(monkeypatch):
    """Repareert Azure de headers op v1, dan stopt de tweede probe vanzelf."""
    calls = []
    _patch(monkeypatch, V1_MET_QUOTUM, RateLimits(1, 1), calls)
    limits, has_headers = asyncio.run(llm.fetch_rate_limits("gpt-5.6-luna"))
    assert calls == ["v1"]
    assert limits.tokens_per_minute == 7_000_000
    assert has_headers is True


def test_v1_without_quota_falls_back_to_the_classic_route(monkeypatch):
    calls = []
    _patch(monkeypatch, V1_ZONDER_QUOTUM, RateLimits(7_000_000, 7_000), calls)
    limits, has_headers = asyncio.run(llm.fetch_rate_limits("gpt-5.6-luna"))
    assert calls == ["v1", "classic"]
    assert limits.tokens_per_minute == 7_000_000
    assert limits.requests_per_minute == 7_000


def test_has_server_headers_stays_from_the_real_route(monkeypatch):
    """The classic route says nothing about per-request timing on v1. If that
    flag hitched a ride, the requester would pick the header-aware controller for
    a route that sends no headers."""
    calls = []
    _patch(monkeypatch, V1_ZONDER_QUOTUM, RateLimits(7_000_000, 7_000), calls)
    _, has_headers = asyncio.run(llm.fetch_rate_limits("gpt-5.6-luna"))
    assert has_headers is False


def test_beide_routes_leeg_geeft_nul_terug(monkeypatch):
    """No invented number here — the caller decides what happens without a quota,
    and it picks the measured capacity."""
    calls = []
    _patch(monkeypatch, V1_ZONDER_QUOTUM, RateLimits(0, 0), calls)
    limits, _ = asyncio.run(llm.fetch_rate_limits("gpt-5.6-luna"))
    assert calls == ["v1", "classic"]
    assert limits.tokens_per_minute == 0


def test_a_half_quota_counts_as_no_quota(monkeypatch):
    """Tokens only and no requests is unusable: the requester divides by both."""
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
    """That route will disappear one day. When it does, the run must not fall
    over fetching a number we can also do without."""
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
    """This route is used only to read headers, never to do work — it must not
    move along with whatever the Responses API does."""
    assert llm.AZURE_CLASSIC_PROBE_API_VERSION == "2024-10-21"
