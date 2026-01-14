"""
Centralized LLM client that abstracts provider differences.

Features:
- OpenAI: Uses responses.create() with input= parameter
- Azure: Uses chat.completions.create() with messages= parameter
- Automatic retries (3x) via instructor
- Token counting and cost tracking
- Model limits (static for OpenAI, dynamic from ARM for Azure)

Usage:
    from utils.llm import create_client, llm_create_async, token_tracker, get_model_limits

    # Get model limits for batching/chunking strategies
    limits = get_model_limits("gpt-4.1-mini")
    # Returns: {"context_window": 1_000_000, "max_output": 32_000}

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
)


# =============================================================================
# Model Limits - Provider-Specific Approach
#
# OpenAI: Static limits from config.py (no API available to fetch dynamically)
# Azure: Can fetch dynamically from Azure Resource Manager (ARM) API
# =============================================================================

# Try to import model limits from config, fall back to defaults
try:
    from config import OPENAI_MODEL_LIMITS
except ImportError:
    OPENAI_MODEL_LIMITS = None

try:
    from config import AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP
except ImportError:
    AZURE_SUBSCRIPTION_ID = None
    AZURE_RESOURCE_GROUP = None

# Fallback defaults if not configured in config.py
DEFAULT_OPENAI_LIMITS = {
    # GPT-4.1 family - 1M context window
    "gpt-4.1": {"context_window": 1_000_000, "max_output": 32_000},
    "gpt-4.1-mini": {"context_window": 1_000_000, "max_output": 32_000},
    "gpt-4.1-nano": {"context_window": 1_000_000, "max_output": 32_000},
    # GPT-5 family - 400K total (272K input + 128K output)
    "gpt-5": {"context_window": 272_000, "max_output": 128_000},
    "gpt-5.1": {"context_window": 272_000, "max_output": 128_000},
    "gpt-5.2": {"context_window": 272_000, "max_output": 128_000},
    "gpt-5-mini": {"context_window": 272_000, "max_output": 128_000},
    "gpt-5-nano": {"context_window": 128_000, "max_output": 32_000},
    "gpt-5-chat-latest": {"context_window": 272_000, "max_output": 128_000},
    # GPT-4o family (legacy)
    "gpt-4o": {"context_window": 128_000, "max_output": 16_000},
    "gpt-4o-mini": {"context_window": 128_000, "max_output": 16_000},
    # Embeddings
    "text-embedding-3-large": {"context_window": 8_191, "max_output": 0},
    "text-embedding-3-small": {"context_window": 8_191, "max_output": 0},
}

DEFAULT_LIMITS = {"context_window": 128_000, "max_output": 16_000}

# Cache for Azure limits (fetched once per session)
_azure_limits_cache: Optional[dict] = None


def _fetch_azure_deployment_limits() -> dict:
    """
    Fetch deployment limits from Azure Resource Manager API.

    Uses ARM endpoint to get deployment info including rate limits.
    Requires azure-mgmt-cognitiveservices package and proper Azure credentials.

    Returns:
        dict mapping deployment_name -> {"context_window": N, "max_output": N, "tpm": N, "rpm": N}
    """
    global _azure_limits_cache
    if _azure_limits_cache is not None:
        return _azure_limits_cache

    # Check if ARM access is configured
    if not AZURE_SUBSCRIPTION_ID or not AZURE_RESOURCE_GROUP:
        _azure_limits_cache = {}
        return {}

    try:
        from azure.identity import DefaultAzureCredential
        from azure.mgmt.cognitiveservices import CognitiveServicesManagementClient

        credential = DefaultAzureCredential()
        client = CognitiveServicesManagementClient(credential, AZURE_SUBSCRIPTION_ID)

        # Extract account name from endpoint
        account_name = AZURE_OPENAI_ENDPOINT.split("//")[1].split(".")[0]

        deployments = client.deployments.list(AZURE_RESOURCE_GROUP, account_name)

        limits = {}
        for deployment in deployments:
            name = deployment.name
            # Extract rate limits from deployment properties
            rate_limits = getattr(deployment.properties, 'rate_limits', [])
            tpm = rpm = 0
            for limit in rate_limits:
                if hasattr(limit, 'key'):
                    if limit.key == 'token':
                        tpm = getattr(limit, 'count', 0)
                    elif limit.key == 'request':
                        rpm = getattr(limit, 'count', 0)

            # Model-specific context windows
            model_name = getattr(deployment.properties.model, 'name', '')
            model_limits_lookup = OPENAI_MODEL_LIMITS or DEFAULT_OPENAI_LIMITS
            model_limits = model_limits_lookup.get(model_name, DEFAULT_LIMITS)

            limits[name] = {
                "context_window": model_limits["context_window"],
                "max_output": model_limits["max_output"],
                "tpm": tpm,  # Tokens per minute quota
                "rpm": rpm,  # Requests per minute quota
            }

        _azure_limits_cache = limits
        return limits

    except ImportError:
        print("Warning: azure-mgmt-cognitiveservices not installed. Using static limits.")
        _azure_limits_cache = {}
        return {}
    except Exception as e:
        # Fall back to static limits if ARM access fails
        print(f"Warning: Could not fetch Azure limits from ARM: {e}")
        _azure_limits_cache = {}
        return {}


def get_model_limits(model: str) -> dict:
    """
    Get context window and max output tokens for a model.

    - OpenAI: Returns static limits from config.py (adjustable per subscription)
    - Azure: Fetches from ARM API if configured, caches for session

    Args:
        model: Model name (e.g., "gpt-4.1-mini")

    Returns:
        dict with 'context_window', 'max_output' keys (Azure also has 'tpm', 'rpm')
    """
    limits_lookup = OPENAI_MODEL_LIMITS or DEFAULT_OPENAI_LIMITS

    if API_PROVIDER == "azure":
        azure_limits = _fetch_azure_deployment_limits()
        if AZURE_OPENAI_DEPLOYMENT_NAME in azure_limits:
            return azure_limits[AZURE_OPENAI_DEPLOYMENT_NAME]
        # Fall back to static limits for the model
        return limits_lookup.get(model, DEFAULT_LIMITS)
    else:
        # OpenAI: static limits
        return limits_lookup.get(model, DEFAULT_LIMITS)


# =============================================================================
# Token Pricing (per 1M tokens) - Updated Jan 2026
# Source: https://www.finout.io/blog/openai-pricing-in-2026
# =============================================================================
MODEL_PRICING = {
    # GPT-4.1 family
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    # GPT-5 family
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-5.1": {"input": 1.25, "output": 10.00},
    "gpt-5.2": {"input": 1.25, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    "gpt-5-chat-latest": {"input": 1.25, "output": 10.00},
    # GPT-4o family (legacy)
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    # Embeddings
    "text-embedding-3-large": {"input": 0.13, "output": 0.0},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
}

# Default for unknown models
DEFAULT_PRICING = {"input": 1.00, "output": 4.00}


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
        lines = [
            "=" * 50,
            "LLM USAGE SUMMARY",
            "=" * 50,
            f"Total API calls: {self.call_count}",
            f"Total tokens: {self.total_input_tokens + self.total_output_tokens:,}",
            f"  - Input: {self.total_input_tokens:,}",
            f"  - Output: {self.total_output_tokens:,}",
            f"Total cost: ${self.total_cost_usd:.4f}",
            "",
            "By model:",
        ]
        for model, data in sorted(self.costs_by_model.items()):
            lines.append(f"  {model}: {data['calls']} calls, {data['tokens']:,} tokens, ${data['cost']:.4f}")
        lines.append("=" * 50)
        return "\n".join(lines)

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


def create_client(model: str, async_mode: bool = True, max_retries: int = DEFAULT_MAX_RETRIES) -> Any:
    """
    Create an instructor-wrapped client for the configured provider.

    Args:
        model: Model name (used for OpenAI, ignored for Azure which uses deployment)
        async_mode: Whether to create async client
        max_retries: Number of retries for failed requests (default: 3)

    Returns:
        Instructor-wrapped client with automatic retries
    """
    if API_PROVIDER == "azure":
        # Azure: use TOOLS mode with chat.completions.create
        # (West Europe doesn't support Responses API yet)
        azure_base_url = f"{AZURE_OPENAI_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT_NAME}/"

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

def _extract_and_track_usage(response: Any, model: str):
    """Extract usage from response and record in token_tracker."""
    # Try to get usage from _raw_response first (instructor pattern)
    usage = None
    raw_response = getattr(response, "_raw_response", None)
    if raw_response:
        usage = getattr(raw_response, "usage", None)

    # Fall back to direct usage attribute
    if not usage:
        usage = getattr(response, "usage", None)

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
    if API_PROVIDER == "azure":
        # Azure: chat.completions.create with messages
        params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        if response_model:
            params["response_model"] = response_model
        response = await client.chat.completions.create(**params)
    else:
        # OpenAI: responses.create with input
        params = {
            "model": model,
            "input": prompt,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            **kwargs
        }
        if response_model:
            params["response_model"] = response_model
        response = await client.responses.create(**params)

    if track_usage:
        _extract_and_track_usage(response, model)

    return response


def llm_create_sync(
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
    Synchronous version of llm_create_async.

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
    if API_PROVIDER == "azure":
        # Azure: chat.completions.create with messages
        params = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        if response_model:
            params["response_model"] = response_model
        response = client.chat.completions.create(**params)
    else:
        # OpenAI: responses.create with input
        params = {
            "model": model,
            "input": prompt,
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            **kwargs
        }
        if response_model:
            params["response_model"] = response_model
        response = client.responses.create(**params)

    if track_usage:
        _extract_and_track_usage(response, model)

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
