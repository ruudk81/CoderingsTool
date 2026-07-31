"""
Centralized LLM client that abstracts provider differences.

Features:
- OpenAI: Uses responses.create() with input= parameter
- Azure: Uses chat.completions.create() with messages= parameter
- Automatic retries (3x) via instructor
- Token counting and cost tracking

Usage:
    from utils.llm import create_client, llm_create_async, token_tracker

    client = create_client(model="gpt-4.1-mini", async_mode=True)
    response = await llm_create_async(
        client=client,
        model="gpt-4.1-mini",
        prompt="Extract the name",
        response_model=MyPydanticModel,
        temperature=0.0
    )

    # At end of pipeline:
    print(token_tracker.get_summary())
"""

import uuid
import time

import httpx
import instructor
from openai import OpenAI, AsyncOpenAI
from typing import Any, Optional, Type
from pydantic import BaseModel
from dataclasses import dataclass, field
from threading import Lock
from collections import OrderedDict

from config import (
    API_PROVIDER,
    OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDING,
    get_model_for_api,
    ModelConfig,
    MODEL_PRICING,
    DEFAULT_PRICING,
)

# Model types for determining if temperature is supported
_MODEL_TYPES = ModelConfig.MODEL_TYPES


def _is_reasoning_model(model: str) -> bool:
    """Check if model is a reasoning model that doesn't support temperature."""
    return _MODEL_TYPES.get(model, "chat") == "reasoning"


# =============================================================================
# Debug Logging for LLM Requests
# =============================================================================
DEBUG_LLM_REQUESTS = False  # Set to True to print full LLM requests
DEBUG_LLM_FIRST_ONLY = True  # Only print the first matching request
DEBUG_LLM_FILTER_MODEL = None # Only print requests with specific pydantic / esponse_model name (None = all)
_debug_request_count = 0
_debug_filtered_count = 0  # Count of matching requests printed


def _debug_print_request(params: dict, provider: str, model: str):
    """Print the API request parameters for debugging.

    NOTE: This shows what WE send to instructor. Instructor then:
    1. Takes the response_model
    2. Converts it to a tool/function definition
    3. Adds 'tools' and 'tool_choice' to the actual API call
    4. Removes 'response_model' from the call

    The [RESPONSE_MODEL SCHEMA] section shows the Pydantic JSON schema,
    which is close to (but not identical to) the tool schema instructor generates.
    """
    global _debug_request_count, _debug_filtered_count

    if not DEBUG_LLM_REQUESTS:
        return

    _debug_request_count += 1

    # Filter by response_model name if specified
    if DEBUG_LLM_FILTER_MODEL:
        rm = params.get("response_model")
        if rm is None:
            return
        rm_name = getattr(rm, "__name__", str(rm))
        if DEBUG_LLM_FILTER_MODEL not in rm_name:
            return

    # Check if we should only print the first matching request
    _debug_filtered_count += 1
    if DEBUG_LLM_FIRST_ONLY and _debug_filtered_count > 1:
        return

    import json

    print("\n" + "=" * 80)
    print(f"[DEBUG] LLM REQUEST #{_debug_request_count} - Provider: {provider}, Model: {model}")
    print("=" * 80)

    # Print messages/input
    if "messages" in params:
        print("\n[MESSAGES]")
        for i, msg in enumerate(params["messages"]):
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            print(f"  Message {i} ({role}):")
            if len(content) > 2000:
                # Show first 1000 chars and last 500 chars to see both start and end
                print(f"    {content[:1000]}")
                print(f"    [...middle truncated...]")
                print(f"    {content[-500:]}")
                print(f"    [total {len(content)} chars]")
            else:
                print(f"    {content}")
    elif "input" in params:
        print("\n[INPUT]")
        content = params["input"]
        if len(content) > 1000:
            print(f"  {content[:1000]}...")
            print(f"  [...truncated, total {len(content)} chars]")
        else:
            print(f"  {content}")

    # Print tools (THE KEY PART - what instructor adds)
    if "tools" in params:
        print("\n[TOOLS] (added by instructor)")
        try:
            print(json.dumps(params["tools"], indent=2, default=str))
        except Exception as e:
            print(f"  Could not serialize tools: {e}")
            print(f"  Raw: {params['tools']}")

    # Print tool_choice
    if "tool_choice" in params:
        print("\n[TOOL_CHOICE]")
        try:
            print(json.dumps(params["tool_choice"], indent=2, default=str))
        except Exception:
            print(f"  {params['tool_choice']}")

    # Print response_model schema if present
    if "response_model" in params:
        print("\n[RESPONSE_MODEL SCHEMA]")
        rm = params["response_model"]
        if hasattr(rm, "model_json_schema"):
            try:
                schema = rm.model_json_schema()
                print(json.dumps(schema, indent=2))
            except Exception as e:
                print(f"  Could not get schema: {e}")
        else:
            print(f"  {rm}")

    # Print other params
    other_keys = ["model", "max_tokens", "max_output_tokens", "temperature", "reasoning", "text"]
    other_params = {k: v for k, v in params.items() if k in other_keys}
    if other_params:
        print("\n[OTHER PARAMS]")
        print(json.dumps(other_params, indent=2, default=str))

    print("\n" + "=" * 80 + "\n")


# =============================================================================
# Minimal Response Model for Probe Calls
# =============================================================================
class ProbeResponse(BaseModel):
    """Minimal response model for bootstrap/probe calls that measure latency and tokens."""
    content: str


# =============================================================================
# Header Capture Transport — transparent middleware for API response headers
# =============================================================================

class HeaderCaptureTransport(httpx.AsyncBaseTransport):
    """Wraps an httpx async transport to capture OpenAI response headers.

    Stores entries in a bounded dict keyed by client_request_id (from the
    outgoing X-Client-Request-Id header). Provides O(1) lookup for correlating
    headers with specific API calls under high concurrency.

    Used by the header-aware concurrency controller to read:
    - openai-processing-ms: server-side processing time
    - x-ratelimit-remaining-requests/tokens: live budget counters
    - x-ratelimit-reset-requests/tokens: window refill timing
    """

    def __init__(self, wrapped: httpx.AsyncBaseTransport, maxlen: int = 500):
        self._wrapped = wrapped
        self._maxlen = maxlen
        self._store: OrderedDict = OrderedDict()
        # Convenience: last response's processing_ms (racy under concurrency, use for trends only)
        self.last_processing_ms: float = 0.0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        client_id = request.headers.get('x-client-request-id', '')
        response = await self._wrapped.handle_async_request(request)

        entry = {
            'client_request_id': client_id,
            'server_request_id': response.headers.get('x-request-id', ''),
            'processing_ms': float(response.headers.get('openai-processing-ms', 0)),
            'remaining_requests': int(response.headers.get('x-ratelimit-remaining-requests', 0)),
            'remaining_tokens': int(response.headers.get('x-ratelimit-remaining-tokens', 0)),
            'reset_requests': response.headers.get('x-ratelimit-reset-requests', ''),
            'reset_tokens': response.headers.get('x-ratelimit-reset-tokens', ''),
            'limit_requests': int(response.headers.get('x-ratelimit-limit-requests', 0)),
            'limit_tokens': int(response.headers.get('x-ratelimit-limit-tokens', 0)),
            'timestamp': time.monotonic(),
        }
        self.last_processing_ms = entry['processing_ms']

        if client_id:
            self._store[client_id] = entry
            # Evict oldest if over capacity
            while len(self._store) > self._maxlen:
                self._store.popitem(last=False)

        return response

    def get(self, client_request_id: str) -> Optional[dict]:
        """Look up captured headers by client request ID. O(1)."""
        return self._store.get(client_request_id)


# =============================================================================
# Dynamic Rate Limits - Fetched from API Response Headers
# =============================================================================
@dataclass
class RateLimits:
    """Rate limits fetched from API response headers.

    Works for both OpenAI and Azure OpenAI - both providers use the same
    x-ratelimit-limit-tokens and x-ratelimit-limit-requests headers.
    """
    tokens_per_minute: int
    requests_per_minute: int
    tokens_per_day: int = 0  # Optional, not always available


def extract_rate_limits_from_response(response) -> RateLimits:
    """Extract TPM/RPM limits from API response headers.

    Works for both OpenAI and Azure OpenAI providers.

    Args:
        response: Raw API response object with headers attribute

    Returns:
        RateLimits with tokens_per_minute and requests_per_minute.
        Returns zeros if headers not present.
    """
    # Get headers from response - handle different response types
    headers = {}
    if hasattr(response, 'headers'):
        headers = response.headers
    elif hasattr(response, '_headers'):
        headers = response._headers

    # Extract rate limits from standard headers
    tpm = int(headers.get('x-ratelimit-limit-tokens', 0))
    rpm = int(headers.get('x-ratelimit-limit-requests', 0))

    return RateLimits(
        tokens_per_minute=tpm,
        requests_per_minute=rpm,
        tokens_per_day=tpm * 60 * 24 if tpm > 0 else 0  # Estimate daily from per-minute
    )


async def fetch_rate_limits(model: str) -> tuple:
    """Probe API for rate limits and header availability.

    Makes a minimal API call ("Hi") and extracts rate limits from
    response headers. Also checks for openai-processing-ms header.

    Args:
        model: Model name (OpenAI) or deployment name (Azure)

    Returns:
        (RateLimits, has_server_headers: bool)
    """
    if API_PROVIDER == "azure":
        # Quota is per deployment, so probe the deployment this model resolves to
        deployment = get_model_for_api(model)
        client = AsyncOpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            base_url=f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/deployments/{deployment}/",
            default_query={"api-version": "2024-10-21"},
        )
        response = await client.chat.completions.with_raw_response.create(
            model=deployment,
            messages=[{"role": "user", "content": "Hi"}],
            max_completion_tokens=5,
        )
    else:
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        response = await client.responses.with_raw_response.create(
            model=model,
            input="Hi",
        )

    rate_limits = extract_rate_limits_from_response(response)
    has_headers = 'openai-processing-ms' in response.headers
    return rate_limits, has_headers


@dataclass
class TokenTracker:
    """Thread-safe token and cost tracking across all LLM calls."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    call_count: int = 0
    costs_by_model: dict = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def record(self, model: str, input_tokens: int, output_tokens: int):
        """Record tokens and calculate cost for a single call."""
        pricing = MODEL_PRICING.get(model, DEFAULT_PRICING)

        # Cost = tokens / 1M * price_per_1M
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        call_cost = input_cost + output_cost

        with self._lock:
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_cost_usd += call_cost
            self.call_count += 1

            if model not in self.costs_by_model:
                self.costs_by_model[model] = {"calls": 0, "cost": 0.0, "tokens": 0}
            self.costs_by_model[model]["calls"] += 1
            self.costs_by_model[model]["cost"] += call_cost
            self.costs_by_model[model]["tokens"] += input_tokens + output_tokens

    def get_summary(self) -> str:
        """Get formatted summary of token usage and costs, broken down by model."""
        with self._lock:
            by_model = sorted(self.costs_by_model.items(), key=lambda kv: -kv[1]["cost"])
            input_tokens = self.total_input_tokens
            output_tokens = self.total_output_tokens
            total_cost = self.total_cost_usd
            call_count = self.call_count

        lines = [
            "=" * 50,
            "LLM USAGE SUMMARY",
            "=" * 50,
            f"Provider: {'Azure OpenAI' if API_PROVIDER == 'azure' else 'OpenAI'}",
        ]

        # Name the models this run actually called. On Azure quota and billing
        # attach to the deployment, which does not match the model name.
        for model, stats in by_model:
            label = model
            if API_PROVIDER == "azure":
                label = f"{model} (deployment: {get_model_for_api(model)})"
            lines.append(
                f"  {label}: {stats['calls']} calls, "
                f"{stats['tokens']:,} tokens, ${stats['cost']:.4f}"
            )

        lines += [
            f"Total API calls: {call_count}",
            f"Total tokens: {input_tokens + output_tokens:,}",
            f"  - Input: {input_tokens:,}",
            f"  - Output: {output_tokens:,}",
            f"Total cost: ${total_cost:.4f}",
            "=" * 50,
        ]
        return "\n".join(lines)

    def snapshot(self) -> dict:
        """Return current cumulative state as a plain dict (thread-safe copy)."""
        with self._lock:
            return {
                "input_tokens": self.total_input_tokens,
                "output_tokens": self.total_output_tokens,
                "cost_usd": self.total_cost_usd,
                "calls": self.call_count,
            }

    def reset(self):
        """Reset all counters."""
        with self._lock:
            self.total_input_tokens = 0
            self.total_output_tokens = 0
            self.total_cost_usd = 0.0
            self.call_count = 0
            self.costs_by_model = {}


# Global token tracker instance
token_tracker = TokenTracker()


# =============================================================================
# Client Creation (with instructor + retries)
# =============================================================================
DEFAULT_MAX_RETRIES = 3


def create_client(
    model: str,
    async_mode: bool = True,
    max_retries: int = DEFAULT_MAX_RETRIES,
    azure_deployment: Optional[str] = None,
    capture_headers: bool = False,
) -> Any:
    """
    Create an instructor-wrapped client for the configured provider.

    Args:
        model: Model name (used for OpenAI, ignored for Azure which uses deployment)
        async_mode: Whether to create async client
        max_retries: Number of retries for failed requests (default: 3)
        azure_deployment: Optional Azure deployment name override (e.g., for codeGenerator)
        capture_headers: If True, inject HeaderCaptureTransport to capture response headers.
            The transport is accessible on the returned client as `_header_transport`.
            Default False — no overhead for steps that don't need headers.

    Returns:
        Instructor-wrapped client with automatic retries
    """
    transport = None

    if API_PROVIDER == "azure":
        # Azure: use TOOLS mode with chat.completions.create
        # (West Europe doesn't support Responses API yet)
        deployment = azure_deployment or get_model_for_api(model)
        azure_base_url = f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/deployments/{deployment}/"

        client_kwargs = {
            'api_key': AZURE_OPENAI_API_KEY,
            'base_url': azure_base_url,
            'default_query': {"api-version": "2024-10-21"},
            'max_retries': max_retries,
        }

        if capture_headers and async_mode:
            base_http = httpx.AsyncClient()
            transport = HeaderCaptureTransport(base_http._transport)
            client_kwargs['http_client'] = httpx.AsyncClient(transport=transport)

        if async_mode:
            base_client = AsyncOpenAI(**client_kwargs)
        else:
            base_client = OpenAI(**client_kwargs)

        wrapped = instructor.from_openai(base_client, mode=instructor.Mode.TOOLS)
    else:
        # OpenAI: use RESPONSES_TOOLS mode with responses.create
        if capture_headers and async_mode:
            base_http = httpx.AsyncClient()
            transport = HeaderCaptureTransport(base_http._transport)
            http_client = httpx.AsyncClient(transport=transport)
            base_client = AsyncOpenAI(api_key=OPENAI_API_KEY, http_client=http_client)
            wrapped = instructor.from_openai(base_client, mode=instructor.Mode.RESPONSES_TOOLS)
        else:
            wrapped = instructor.from_provider(
                f"openai/{model}",
                mode=instructor.Mode.RESPONSES_TOOLS,
                async_client=async_mode,
                api_key=OPENAI_API_KEY
            )

    # Attach transport reference for callers that need header data
    if transport:
        wrapped._header_transport = transport

    return wrapped


# =============================================================================
# LLM Request Functions (with automatic token tracking)
# =============================================================================

def _extract_and_track_usage(completion: Any, model: str):
    """Extract usage from completion object and record in token_tracker.

    Args:
        completion: Raw completion object from create_with_completion() or direct API response
        model: Model name for pricing lookup
    """
    # Get usage directly from completion object
    usage = getattr(completion, "usage", None)

    if usage:
        if API_PROVIDER == "azure":
            # Chat completions API uses prompt_tokens/completion_tokens
            input_tokens = getattr(usage, "prompt_tokens", 0)
            output_tokens = getattr(usage, "completion_tokens", 0)
        else:
            # Responses API uses input_tokens/output_tokens
            input_tokens = getattr(usage, "input_tokens", 0)
            output_tokens = getattr(usage, "output_tokens", 0)

        token_tracker.record(model, input_tokens, output_tokens)


async def llm_create_async(
    client: Any,
    model: str,
    prompt: str,
    response_model: Optional[Type[BaseModel]] = None,
    temperature: float = 0.0,
    max_tokens: int = 4000,
    track_usage: bool = True,
    **kwargs
) -> Any:
    """
    Make an async LLM request with provider-appropriate parameters.

    Args:
        client: Instructor-wrapped client from create_client()
        model: Model name (flexible - passed from config.py or app.py)
        prompt: The prompt text
        response_model: Optional Pydantic model for structured output
        temperature: Temperature setting
        max_tokens: Maximum output tokens
        track_usage: Whether to record tokens/cost (default: True)
        **kwargs: Additional parameters passed to the API

    Returns:
        Response (Pydantic model if response_model provided, else raw response)
    """
    completion = None  # Will hold raw completion for token tracking

    # Generate client request ID for header correlation (used by HeaderCaptureTransport)
    client_request_id = str(uuid.uuid4())
    kwargs.setdefault('extra_headers', {})['X-Client-Request-Id'] = client_request_id

    # Check if response_model is a List type (instructor's create_with_completion has a bug with List types)
    is_list_response = (response_model is not None and
                        hasattr(response_model, '__origin__') and
                        response_model.__origin__ is list)

    # Reasoning models (gpt-5-mini, gpt-5, etc.) don't support temperature parameter
    is_reasoning = _is_reasoning_model(model)

    if API_PROVIDER == "azure":
        # Azure: chat.completions.create with messages
        params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            **kwargs
        }
        # Reasoning models reject max_tokens and temperature
        if is_reasoning:
            params["max_completion_tokens"] = max_tokens
        else:
            params["max_tokens"] = max_tokens
            params["temperature"] = temperature
        if response_model:
            params["response_model"] = response_model

        # DEBUG: Print full request before API call
        _debug_print_request(params, API_PROVIDER, model)

        if response_model:
            if is_list_response:
                # For List types, use regular create() as create_with_completion has a bug
                response = await client.chat.completions.create(**params)
                # Try to get _raw_response if instructor attached it
                completion = getattr(response, "_raw_response", None)
            else:
                # Use create_with_completion to get raw response with usage data
                response, completion = await client.chat.completions.create_with_completion(**params)
        else:
            response = await client.chat.completions.create(**params)
            completion = response  # Raw response when no response_model
    else:
        # OpenAI: responses.create with input
        params = {
            "model": model,
            "input": prompt,
            "max_output_tokens": max_tokens,
            **kwargs
        }
        # Only add temperature for non-reasoning models
        if not is_reasoning:
            params["temperature"] = temperature
        if response_model:
            params["response_model"] = response_model

        # DEBUG: Print full request before API call
        _debug_print_request(params, API_PROVIDER, model)

        if response_model:
            if is_list_response:
                # For List types, use regular create() as create_with_completion has a bug
                response = await client.responses.create(**params)
                completion = getattr(response, "_raw_response", None)
            else:
                # Use create_with_completion to get raw response with usage data
                response, completion = await client.responses.create_with_completion(**params)
        else:
            response = await client.responses.create(**params)
            completion = response  # Raw response when no response_model

    if track_usage and completion:
        _extract_and_track_usage(completion, model)

    # Attach _raw_response and _client_request_id to response
    # _raw_response: backwards compatibility for token reconciliation
    # _client_request_id: correlation key for HeaderCaptureTransport lookup
    if completion and response is not completion:
        try:
            if isinstance(response, list):
                class ResponseList(list):
                    pass
                wrapped = ResponseList(response)
                wrapped._raw_response = completion
                wrapped._client_request_id = client_request_id
                return wrapped
            else:
                response._raw_response = completion
                response._client_request_id = client_request_id
        except (AttributeError, TypeError):
            pass  # Some responses don't allow attribute setting
    else:
        try:
            response._client_request_id = client_request_id
        except (AttributeError, TypeError):
            pass

    return response


# =============================================================================
# Embedding Client (separate from instructor-wrapped clients)
# =============================================================================

def create_embedding_client(async_mode: bool = True) -> Any:
    """
    Create a raw OpenAI client for embeddings (not instructor-wrapped).

    Args:
        async_mode: Whether to create async client

    Returns:
        OpenAI or AsyncOpenAI client
    """
    if API_PROVIDER == "azure":
        azure_base_url = f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDING}/"
        if async_mode:
            return AsyncOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                base_url=azure_base_url,
                default_query={"api-version": "2024-10-21"}
            )
        return OpenAI(
            api_key=AZURE_OPENAI_API_KEY,
            base_url=azure_base_url,
            default_query={"api-version": "2024-10-21"}
        )
    else:
        if async_mode:
            return AsyncOpenAI(api_key=OPENAI_API_KEY)
        return OpenAI(api_key=OPENAI_API_KEY)
