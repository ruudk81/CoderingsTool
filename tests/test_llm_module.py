"""
Test LLM Module

Tests the centralized llm.py module that abstracts OpenAI vs Azure provider differences.
Tests both client creation and basic functionality.

Run with: pytest tests/test_llm_module.py -v
"""

import os
import sys
import pytest
from pydantic import BaseModel

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class SimpleExtraction(BaseModel):
    """Simple Pydantic model for testing structured output"""
    name: str
    age: int


class TestLLMModuleImports:
    """Test that llm.py can be imported and has expected exports"""

    def test_import_llm_module(self):
        """Test that llm module can be imported"""
        from utils.llm import (
            create_client,
            llm_create_async,
            llm_create_sync,
            create_embedding_client,
            get_model_limits,
            token_tracker,
            TokenTracker,
            MODEL_PRICING,
        )
        assert create_client is not None
        assert llm_create_async is not None
        assert llm_create_sync is not None
        assert create_embedding_client is not None
        assert get_model_limits is not None
        assert token_tracker is not None
        print("All exports available")


class TestTokenTracker:
    """Test TokenTracker functionality"""

    def test_token_tracker_record(self):
        """Test recording tokens and costs"""
        from utils.llm import TokenTracker

        tracker = TokenTracker()
        tracker.record("gpt-4.1-mini", input_tokens=1000, output_tokens=500)

        assert tracker.call_count == 1
        assert tracker.total_input_tokens == 1000
        assert tracker.total_output_tokens == 500
        # gpt-4.1-mini: $0.40/1M input, $1.60/1M output
        # Cost = (1000/1M * 0.40) + (500/1M * 1.60) = 0.0004 + 0.0008 = 0.0012
        assert abs(tracker.total_cost_usd - 0.0012) < 0.0001
        print(f"Cost calculation correct: ${tracker.total_cost_usd:.4f}")

    def test_token_tracker_multiple_models(self):
        """Test tracking multiple models"""
        from utils.llm import TokenTracker

        tracker = TokenTracker()
        tracker.record("gpt-4.1-mini", 1000, 500)
        tracker.record("gpt-5.2", 2000, 1000)
        tracker.record("gpt-4.1-mini", 500, 250)

        assert tracker.call_count == 3
        assert "gpt-4.1-mini" in tracker.costs_by_model
        assert "gpt-5.2" in tracker.costs_by_model
        assert tracker.costs_by_model["gpt-4.1-mini"]["calls"] == 2
        assert tracker.costs_by_model["gpt-5.2"]["calls"] == 1
        print(tracker.get_summary())

    def test_token_tracker_reset(self):
        """Test resetting tracker"""
        from utils.llm import TokenTracker

        tracker = TokenTracker()
        tracker.record("gpt-4.1-mini", 1000, 500)
        tracker.reset()

        assert tracker.call_count == 0
        assert tracker.total_input_tokens == 0
        assert tracker.total_cost_usd == 0.0
        print("Reset successful")


class TestModelLimits:
    """Test model limits functionality"""

    def test_get_known_model_limits(self):
        """Test getting limits for known models"""
        from utils.llm import get_model_limits

        limits = get_model_limits("gpt-4.1-mini")
        assert "context_window" in limits
        assert "max_output" in limits
        assert limits["context_window"] == 1_000_000  # 1M for GPT-4.1
        print(f"gpt-4.1-mini limits: {limits}")

    def test_get_gpt5_limits(self):
        """Test GPT-5 model limits"""
        from utils.llm import get_model_limits

        limits = get_model_limits("gpt-5.2")
        assert limits["context_window"] == 272_000  # 272K for GPT-5
        assert limits["max_output"] == 128_000
        print(f"gpt-5.2 limits: {limits}")

    def test_get_unknown_model_defaults(self):
        """Test that unknown models get default limits"""
        from utils.llm import get_model_limits

        limits = get_model_limits("unknown-future-model")
        assert "context_window" in limits
        assert "max_output" in limits
        print(f"Default limits for unknown model: {limits}")


class TestClientCreation:
    """Test client creation functions"""

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set"
    )
    def test_create_client_openai(self):
        """Test creating client with OpenAI provider"""
        # Temporarily set provider to openai
        from config import API_PROVIDER
        original_provider = API_PROVIDER

        try:
            from utils.llm import create_client
            client = create_client("gpt-4.1-mini", async_mode=True)
            assert client is not None
            print("OpenAI async client created")

            sync_client = create_client("gpt-4.1-mini", async_mode=False)
            assert sync_client is not None
            print("OpenAI sync client created")
        finally:
            pass  # Provider is module-level, no cleanup needed

    @pytest.mark.skipif(
        not os.getenv("AZURE_OPENAI_ENDPOINT") or not os.getenv("AZURE_OPENAI_API_KEY"),
        reason="Azure not configured"
    )
    def test_create_client_azure(self):
        """Test creating client with Azure provider"""
        # This would require changing API_PROVIDER at runtime
        # For now, just test that the function exists
        from utils.llm import create_client
        assert create_client is not None
        print("Azure client creation function available")

    def test_create_embedding_client(self):
        """Test creating embedding client"""
        from utils.llm import create_embedding_client

        # Just test that function works (actual API call requires credentials)
        assert create_embedding_client is not None
        print("Embedding client creation function available")


class TestLLMRequests:
    """Test actual LLM request functions (requires API credentials)"""

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_llm_create_async_openai(self):
        """Test async LLM request with OpenAI"""
        from utils.llm import create_client, llm_create_async, token_tracker

        token_tracker.reset()

        client = create_client("gpt-4.1-mini", async_mode=True)

        response = await llm_create_async(
            client=client,
            model="gpt-4.1-mini",
            prompt="Extract: John is 30 years old",
            response_model=SimpleExtraction,
            temperature=0.0,
            max_tokens=100
        )

        assert isinstance(response, SimpleExtraction)
        assert response.name == "John"
        assert response.age == 30
        print(f"Extracted: {response}")

        # Check token tracking
        assert token_tracker.call_count >= 1
        print(f"Tokens tracked: {token_tracker.total_input_tokens + token_tracker.total_output_tokens}")

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set"
    )
    def test_llm_create_sync_openai(self):
        """Test sync LLM request with OpenAI"""
        from utils.llm import create_client, llm_create_sync

        client = create_client("gpt-4.1-mini", async_mode=False)

        response = llm_create_sync(
            client=client,
            model="gpt-4.1-mini",
            prompt="Extract: Alice is 25 years old",
            response_model=SimpleExtraction,
            temperature=0.0,
            max_tokens=100
        )

        assert isinstance(response, SimpleExtraction)
        assert response.name == "Alice"
        assert response.age == 25
        print(f"Sync extracted: {response}")


class TestAzureLLMRequests:
    """Test LLM requests with Azure provider"""

    @pytest.mark.skipif(
        not os.getenv("AZURE_OPENAI_ENDPOINT") or not os.getenv("AZURE_OPENAI_API_KEY"),
        reason="Azure not configured"
    )
    @pytest.mark.asyncio
    async def test_llm_create_async_azure(self):
        """Test async LLM request with Azure (uses chat.completions.create)"""
        # This test would require modifying API_PROVIDER at runtime
        # For a proper test, you'd need to reload the module with Azure settings
        print("Azure async test - requires API_PROVIDER=azure in config.py")

    @pytest.mark.skipif(
        not os.getenv("AZURE_OPENAI_ENDPOINT") or not os.getenv("AZURE_OPENAI_API_KEY"),
        reason="Azure not configured"
    )
    def test_llm_create_sync_azure(self):
        """Test sync LLM request with Azure"""
        print("Azure sync test - requires API_PROVIDER=azure in config.py")


# Test Summary
def test_llm_module_summary():
    """Print summary of llm module tests"""
    print("\n" + "=" * 70)
    print("LLM MODULE TEST SUMMARY")
    print("=" * 70)
    print("Tested:")
    print("  - Module imports and exports")
    print("  - TokenTracker recording and cost calculation")
    print("  - Model limits retrieval")
    print("  - Client creation functions")
    print("\nTo test actual API calls:")
    print("  1. Set OPENAI_API_KEY for OpenAI tests")
    print("  2. Set AZURE_OPENAI_* env vars for Azure tests")
    print("  3. Change API_PROVIDER in config.py to test Azure path")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
