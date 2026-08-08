"""
Model limits checker — verify config.py against the live OpenAI API.

Audits the three kinds of "limits" stored in config.py, each against its only
authoritative source:

  1. Model existence  -> GET /v1/models (free, read-only). Confirms every model
     the pipeline is configured to call still exists on this account.
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

from config import (
    ACTIVE_GENERATIONS,
    API_PROVIDER,
    STEP_MODEL,
    get_reasoning_params,
    get_step_model,
    OPENAI_MODEL_LIMITS,
    MODEL_PRICING,
    FALLBACK_RPM,
    FALLBACK_TPM,
)

PRICING_URL = "https://openai.com/api/pricing/"

OK = "✅"     # ✅
WARN = "⚠️"  # ⚠️
MISS = "❌"   # ❌


def _configured_models() -> list[tuple[str, str]]:
    """Resolve (rung, model) for every model the phases actually use, deduped.

    Checking the whole MODELS table would flag rungs that are merely available;
    what matters is whether the models this pipeline is configured to call really
    exist. Several phases share a rung, hence the dedup.
    """
    seen = set()
    pairs = []
    for phase, (generation, level) in STEP_MODEL.items():
        model = get_step_model(phase)
        if model not in seen:
            seen.add(model)
            pairs.append((f"{generation}/{level}", model))
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
    """Returns an exit code: 0 = all configured models exist, 1 = one is missing
    from the live API (config stale), 2 = could not verify (no key)."""
    print("=" * 72)
    print("MODEL LIMITS CHECK")
    print("=" * 72)
    print(f"Provider:     {API_PROVIDER}")
    print(f"Generations:  {ACTIVE_GENERATIONS}")
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
    print(f"{'rung':9s} {'model':16s} {'exists':7s} {'live RPM':>12s} {'live TPM':>16s}")
    print("-" * 72)
    missing = []
    for rung, model in _configured_models():
        if model not in available:
            missing.append(model)
        exists = OK if model in available else MISS
        rpm, tpm = (None, None)
        if model in available:
            rpm, tpm = fetch_rate_limits(client, model)
        if tpm and str(tpm).startswith("ERROR"):
            print(f"{rung:9s} {model:16s} {exists:7s}  {tpm}")
        else:
            print(f"{rung:9s} {model:16s} {exists:7s} {_fmt(rpm):>12s} {_fmt(tpm):>16s}")

    # --- 3. Pricing & context window: manual reference ---
    _print_pricing_section()

    # --- Verdict (drives exit code) ---
    if missing:
        print(f"{MISS} Stale config: not on live API -> {', '.join(missing)}\n")
        return 1
    print(f"{OK} All configured models ({ACTIVE_GENERATIONS}) exist on the live API.\n")
    return 0


def _print_pricing_section() -> None:
    print(f"\n{'-' * 72}")
    print(f"Pricing & context window (config values — verify against {PRICING_URL})")
    print("-" * 72)
    print(f"{'model':16s} {'$in/Mtok':>9s} {'$out/Mtok':>10s} {'context':>12s} {'max_out':>10s}")
    print("-" * 72)
    for _, model in _configured_models():
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
