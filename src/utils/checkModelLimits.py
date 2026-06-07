"""
Model limits checker — verify config.py against the live OpenAI API.

Audits the three kinds of "limits" stored in config.py, each against its only
authoritative source:

  1. Model existence  -> GET /v1/models (free, read-only). Confirms every tier
     of the current MODEL_FAMILY still exists on this account.
  2. Rate limits      -> response headers (x-ratelimit-limit-*). OpenAI has no
     dedicated endpoint, so we make one tiny probe call per model and read the
     headers, then compare against FALLBACK_RPM/TPM.
  3. Pricing / context -> NOT exposed by any API. We print the configured values
     next to the official pricing URL for a manual eyeball check.

Usage (from src/):
    python utils/checkModelLimits.py

Only meaningful for API_PROVIDER == "openai"; Azure routes through deployment
names with separate, deployment-scoped limits.
"""

import os
import sys

# Ensure src/ is on sys.path when run directly (python utils/checkModelLimits.py)
_src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from openai import OpenAI

import config
from config import (
    API_PROVIDER,
    MODEL_FAMILY,
    get_model,
    get_reasoning_params,
    OPENAI_MODEL_LIMITS,
    MODEL_PRICING,
    FALLBACK_RPM,
    FALLBACK_TPM,
)

PRICING_URL = "https://openai.com/api/pricing/"

OK = "✅"     # ✅
WARN = "⚠️"  # ⚠️
MISS = "❌"   # ❌


def _tier_models() -> list[tuple[str, str]]:
    """Resolve (tier, model) for the current MODEL_FAMILY, deduped by model.

    FAMILY_TIER_OVERRIDES can collapse tiers (e.g. gpt-4.1 nano->mini), so two
    tiers may resolve to the same model; we keep the first tier that maps to it.
    """
    seen = set()
    pairs = []
    for tier in ("default", "mini", "nano"):
        model = get_model(tier)
        if model not in seen:
            seen.add(model)
            pairs.append((tier, model))
    return pairs


def _fmt(n) -> str:
    """Human-readable thousands grouping, or '?' when missing."""
    return f"{int(n):,}" if n is not None else "?"


def fetch_rate_limits(client: OpenAI, model: str) -> tuple[str | None, str | None]:
    """Probe one model with a minimal call and read its rate-limit headers.

    Mirrors how the pipeline calls the model (same reasoning params), so a model
    that works in the pipeline works here. Returns (rpm, tpm) as raw strings, or
    (None, None) on error.
    """
    try:
        raw = client.responses.with_raw_response.create(
            model=model,
            input="ping",
            max_output_tokens=16,
            **get_reasoning_params(model),
        )
        h = raw.headers
        return h.get("x-ratelimit-limit-requests"), h.get("x-ratelimit-limit-tokens")
    except Exception as e:
        return None, f"ERROR: {str(e)[:80]}"


def main() -> int:
    """Returns an exit code: 0 = all tiers exist, 1 = a configured model is
    missing from the live API (config stale), 2 = could not verify (no key)."""
    print("=" * 72)
    print("MODEL LIMITS CHECK")
    print("=" * 72)
    print(f"Provider:     {API_PROVIDER}")
    print(f"Model family: {MODEL_FAMILY}")
    print("=" * 72)

    if API_PROVIDER != "openai":
        print(
            f"\n{WARN} API_PROVIDER is '{API_PROVIDER}'. Live checks target OpenAI only; "
            "Azure uses deployment-scoped limits. Showing config values without live probe.\n"
        )
        _print_pricing_section()
        return 0

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(f"\n{MISS} OPENAI_API_KEY not found in environment / .env.")
        return 2

    client = OpenAI(api_key=api_key)

    # --- 1. Existence via /v1/models (one call) ---
    available = {m.id for m in client.models.list()}

    # --- 2. Live rate limits via response headers (one probe per model) ---
    print(f"\nFallback (used only when headers are unavailable): "
          f"RPM={_fmt(FALLBACK_RPM)}  TPM={_fmt(FALLBACK_TPM)}\n")
    print(f"{'tier':9s} {'model':16s} {'exists':7s} {'live RPM':>12s} {'live TPM':>16s}")
    print("-" * 72)
    missing = []
    for tier, model in _tier_models():
        if model not in available:
            missing.append(model)
        exists = OK if model in available else MISS
        rpm, tpm = (None, None)
        if model in available:
            rpm, tpm = fetch_rate_limits(client, model)
        if tpm and str(tpm).startswith("ERROR"):
            print(f"{tier:9s} {model:16s} {exists:7s}  {tpm}")
        else:
            print(f"{tier:9s} {model:16s} {exists:7s} {_fmt(rpm):>12s} {_fmt(tpm):>16s}")

    # --- 3. Pricing & context window: manual reference ---
    _print_pricing_section()

    # --- Verdict (drives exit code) ---
    if missing:
        print(f"{MISS} Stale config: not on live API -> {', '.join(missing)}\n")
        return 1
    print(f"{OK} All {MODEL_FAMILY} tiers exist on the live API.\n")
    return 0


def _print_pricing_section() -> None:
    print(f"\n{'-' * 72}")
    print(f"Pricing & context window (config values — verify against {PRICING_URL})")
    print("-" * 72)
    print(f"{'model':16s} {'$in/Mtok':>9s} {'$out/Mtok':>10s} {'context':>12s} {'max_out':>10s}")
    print("-" * 72)
    for _, model in _tier_models():
        price = MODEL_PRICING.get(model, {})
        limits = OPENAI_MODEL_LIMITS.get(model, {})
        print(
            f"{model:16s} "
            f"{price.get('input', '?')!s:>9s} "
            f"{price.get('output', '?')!s:>10s} "
            f"{_fmt(limits.get('context_window')):>12s} "
            f"{_fmt(limits.get('max_output')):>10s}"
        )
    print()


if __name__ == "__main__":
    sys.exit(main())
