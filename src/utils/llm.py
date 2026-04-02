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

import instructor
from openai import OpenAI, AsyncOpenAI
from typing import Any, Optional, Type
from pydantic import BaseModel
from dataclasses import dataclass, field
from threading import Lock

from config import (
    API_PROVIDER,
    OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT_NAME,
    AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDING,
    AZURE_OPENAI_DEPLOYMENT_NAME_CODEDESIGNER,
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
    other_keys = ["model", "max_tokens", "max_output_tokens", "temperature"]
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
        """Get formatted summary of token usage and costs."""
        # Get provider info for display
        provider_label = API_PROVIDER.upper()
        if API_PROVIDER == "azure":
            deployment = AZURE_OPENAI_DEPLOYMENT_NAME_CODEDESIGNER or AZURE_OPENAI_DEPLOYMENT_NAME
            provider_label = f"Azure OpenAI (deployment: {deployment})"
        else:
            provider_label = "OpenAI"

        lines = [
            "=" * 50,
            "LLM USAGE SUMMARY",
            "=" * 50,
            f"Provider: {provider_label}",
            f"Total API calls: {self.call_count}",
            f"Total tokens: {self.total_input_tokens + self.total_output_tokens:,}",
            f"  - Input: {self.total_input_tokens:,}",
            f"  - Output: {self.total_output_tokens:,}",
            f"Total cost: ${self.total_cost_usd:.4f}",
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
    azure_deployment: Optional[str] = None
) -> Any:
    """
    Create an instructor-wrapped client for the configured provider.

    Args:
        model: Model name (used for OpenAI, ignored for Azure which uses deployment)
        async_mode: Whether to create async client
        max_retries: Number of retries for failed requests (default: 3)
        azure_deployment: Optional Azure deployment name override (e.g., for codeGenerator)

    Returns:
        Instructor-wrapped client with automatic retries
    """
    if API_PROVIDER == "azure":
        # Azure: use TOOLS mode with chat.completions.create
        # (West Europe doesn't support Responses API yet)
        deployment = azure_deployment or AZURE_OPENAI_DEPLOYMENT_NAME
        azure_base_url = f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/deployments/{deployment}/"

        if async_mode:
            base_client = AsyncOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                base_url=azure_base_url,
                default_query={"api-version": "2024-10-21"},
                max_retries=max_retries
            )
        else:
            base_client = OpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                base_url=azure_base_url,
                default_query={"api-version": "2024-10-21"},
                max_retries=max_retries
            )
        return instructor.from_openai(base_client, mode=instructor.Mode.TOOLS)
    else:
        # OpenAI: use RESPONSES_TOOLS mode with responses.create
        return instructor.from_provider(
            f"openai/{model}",
            mode=instructor.Mode.RESPONSES_TOOLS,
            async_client=async_mode,
            api_key=OPENAI_API_KEY
        )


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
            "max_tokens": max_tokens,
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

    # Attach _raw_response to response for backwards compatibility with code that accesses usage directly
    # This allows utilities like qualityFilter to still do token reconciliation
    if completion and response is not completion:
        try:
            # For list responses, we can't set attributes directly, so wrap in a list subclass
            if isinstance(response, list):
                class ResponseList(list):
                    pass
                wrapped = ResponseList(response)
                wrapped._raw_response = completion
                return wrapped
            else:
                response._raw_response = completion
        except (AttributeError, TypeError):
            pass  # Some responses don't allow attribute setting

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
