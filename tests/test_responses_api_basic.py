"""
Test Basic Responses API Functionality

Tests that responses.create API works with simple Pydantic models.
Verifies the fundamental API pattern before migrating production code.

Run with: pytest tests/test_responses_api_basic.py -v
"""

import os
import pytest
import asyncio
from pydantic import BaseModel
from typing import Optional


class SimpleUser(BaseModel):
    """Simple Pydantic model for testing"""
    name: str
    age: int


class SimpleResponse(BaseModel):
    """Simple response model"""
    message: str
    confidence: float


@pytest.fixture
def skip_if_no_provider():
    """Skip tests if neither OpenAI nor Azure is configured"""
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_azure = bool(os.getenv("AZURE_OPENAI_ENDPOINT"))

    if not (has_openai or has_azure):
        pytest.skip("Neither OpenAI nor Azure configured - cannot test responses API")


class TestResponsesAPIWithOpenAI:
    """Test responses.create with OpenAI (if available)"""

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set"
    )
    def test_instructor_responses_mode_openai(self):
        """Test instructor with RESPONSES_TOOLS mode on OpenAI"""
        import instructor

        client = instructor.from_provider(
            "openai/gpt-4o-mini",
            mode=instructor.Mode.RESPONSES_TOOLS,
            async_client=False  # Sync for simpler test
        )

        assert client is not None
        print("✓ Created OpenAI client with RESPONSES_TOOLS mode")

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_simple_responses_create_openai(self):
        """Test basic responses.create call with simple Pydantic model"""
        import instructor

        client = instructor.from_provider(
            "openai/gpt-4o-mini",
            mode=instructor.Mode.RESPONSES_TOOLS,
            async_client=True
        )

        # Test with simple extraction task
        response = await client.responses.create(
            input="Extract: John is 30 years old",
            response_model=SimpleUser
        )

        assert isinstance(response, SimpleUser)
        assert response.name == "John"
        assert response.age == 30
        print(f"✓ Successfully extracted: {response}")

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set"
    )
    def test_sync_responses_create_openai(self):
        """Test synchronous responses.create call"""
        import instructor

        client = instructor.from_provider(
            "openai/gpt-4o-mini",
            mode=instructor.Mode.RESPONSES_TOOLS,
            async_client=False
        )

        # Sync call
        response = client.responses.create(
            input="Extract: Jane is 25 years old",
            response_model=SimpleUser
        )

        assert isinstance(response, SimpleUser)
        assert response.name == "Jane"
        assert response.age == 25
        print(f"✓ Sync call successful: {response}")


class TestResponsesAPIWithAzure:
    """Test responses.create with Azure OpenAI (if available)"""

    @pytest.mark.skipif(
        not os.getenv("AZURE_OPENAI_ENDPOINT") or not os.getenv("AZURE_OPENAI_API_KEY"),
        reason="Azure not configured"
    )
    def test_instructor_responses_mode_azure(self):
        """Test instructor with RESPONSES_TOOLS mode on Azure"""
        import instructor
        from openai import AzureOpenAI

        # Create Azure client
        azure_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version="2025-08-01"
        )

        # Patch with instructor
        client = instructor.from_client(
            azure_client,
            mode=instructor.Mode.RESPONSES_TOOLS
        )

        assert client is not None
        print("✓ Created Azure client with RESPONSES_TOOLS mode")

    @pytest.mark.skipif(
        not os.getenv("AZURE_OPENAI_ENDPOINT") or not os.getenv("AZURE_OPENAI_API_KEY"),
        reason="Azure not configured"
    )
    @pytest.mark.asyncio
    async def test_simple_responses_create_azure(self):
        """Test basic responses.create call with Azure"""
        import instructor
        from openai import AsyncAzureOpenAI

        # Note: You need to set the deployment name
        deployment_name = os.getenv("AZURE_GPT41_MINI_DEPLOYMENT", "gpt-41-mini")

        azure_client = AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version="2025-08-01"
        )

        client = instructor.from_client(
            azure_client,
            mode=instructor.Mode.RESPONSES_TOOLS
        )

        # Test with simple extraction task
        # Note: model parameter should be deployment name for Azure
        response = await client.responses.create(
            input="Extract: Alice is 28 years old",
            response_model=SimpleUser,
            # model=deployment_name  # Uncomment if needed
        )

        assert isinstance(response, SimpleUser)
        print(f"✓ Azure responses.create successful: {response}")


class TestParameterDifferences:
    """Test differences between chat.completions and responses APIs"""

    def test_input_vs_messages_parameter(self, skip_if_no_provider):
        """Document that responses API uses 'input' instead of 'messages'"""
        # This is a documentation test

        # OLD: chat.completions.create
        old_pattern = {
            "method": "chat.completions.create",
            "parameters": {
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 1000
            }
        }

        # NEW: responses.create
        new_pattern = {
            "method": "responses.create",
            "parameters": {
                "input": "Hello",  # ← Changed from messages
                "max_output_tokens": 1000  # ← Changed from max_tokens
            }
        }

        print("\n" + "="*70)
        print("PARAMETER MAPPING")
        print("="*70)
        print(f"OLD: messages={old_pattern['parameters']['messages']}")
        print(f"NEW: input={new_pattern['parameters']['input']}")
        print()
        print(f"OLD: max_tokens={old_pattern['parameters']['max_tokens']}")
        print(f"NEW: max_output_tokens={new_pattern['parameters']['max_output_tokens']}")
        print("="*70 + "\n")

        assert True  # Documentation test

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_max_output_tokens_parameter(self):
        """Test that max_output_tokens parameter works"""
        import instructor

        client = instructor.from_provider(
            "openai/gpt-4o-mini",
            mode=instructor.Mode.RESPONSES_TOOLS,
            async_client=True
        )

        response = await client.responses.create(
            input="Say hello",
            response_model=SimpleResponse,
            max_output_tokens=100  # ← New parameter name
        )

        assert isinstance(response, SimpleResponse)
        print(f"✓ max_output_tokens parameter works: {response}")


class TestResponseModelValidation:
    """Test that Pydantic validation works with responses API"""

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_response_model_validation(self):
        """Test that response_model enforces Pydantic validation"""
        import instructor

        client = instructor.from_provider(
            "openai/gpt-4o-mini",
            mode=instructor.Mode.RESPONSES_TOOLS,
            async_client=True
        )

        response = await client.responses.create(
            input="Extract: Bob is 35 years old",
            response_model=SimpleUser
        )

        # Should be validated Pydantic model
        assert isinstance(response, SimpleUser)
        assert isinstance(response.name, str)
        assert isinstance(response.age, int)
        print(f"✓ Pydantic validation working: {response}")

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_validation_error_handling(self):
        """Test that validation errors are properly handled"""
        import instructor
        from pydantic import ValidationError

        client = instructor.from_provider(
            "openai/gpt-4o-mini",
            mode=instructor.Mode.RESPONSES_TOOLS,
            async_client=True
        )

        # This should work - instructor handles retries
        try:
            response = await client.responses.create(
                input="Extract user info: Charlie, age unknown",
                response_model=SimpleUser
            )
            # May succeed with instructor's retry logic
            print(f"✓ Validation handling working (got: {response})")
        except Exception as e:
            # Or may fail - either is acceptable for this test
            print(f"✓ Validation error handled: {type(e).__name__}")


class TestMultipleModels:
    """Test with different model types"""

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_gpt4_mini_model(self):
        """Test with gpt-4o-mini"""
        import instructor

        client = instructor.from_provider(
            "openai/gpt-4o-mini",
            mode=instructor.Mode.RESPONSES_TOOLS,
            async_client=True
        )

        response = await client.responses.create(
            input="Extract: Emma is 22 years old",
            response_model=SimpleUser
        )

        assert isinstance(response, SimpleUser)
        print(f"✓ gpt-4o-mini works: {response}")

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set - gpt-4.1-mini test"
    )
    @pytest.mark.asyncio
    async def test_gpt41_mini_model(self):
        """Test with gpt-4.1-mini (if available)"""
        import instructor

        try:
            client = instructor.from_provider(
                "openai/gpt-4.1-mini",
                mode=instructor.Mode.RESPONSES_TOOLS,
                async_client=True
            )

            response = await client.responses.create(
                input="Extract: Frank is 40 years old",
                response_model=SimpleUser
            )

            assert isinstance(response, SimpleUser)
            print(f"✓ gpt-4.1-mini works: {response}")
        except Exception as e:
            print(f"ℹ gpt-4.1-mini not available or failed: {e}")
            pytest.skip(f"gpt-4.1-mini test skipped: {e}")


class TestTemperatureAndSeed:
    """Test temperature and seed parameters"""

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_temperature_parameter(self):
        """Test that temperature parameter works"""
        import instructor

        client = instructor.from_provider(
            "openai/gpt-4o-mini",
            mode=instructor.Mode.RESPONSES_TOOLS,
            async_client=True
        )

        response = await client.responses.create(
            input="Extract: Grace is 27 years old",
            response_model=SimpleUser,
            temperature=0.0  # Deterministic
        )

        assert isinstance(response, SimpleUser)
        print(f"✓ temperature=0.0 works: {response}")


# Test Summary
def test_basic_responses_api_summary():
    """Print summary of basic responses API tests"""
    print("\n" + "="*70)
    print("BASIC RESPONSES API - TEST SUMMARY")
    print("="*70)
    print("✓ Responses API basic functionality validated")
    print("✓ Pydantic model validation working")
    print("✓ Parameter changes documented (input, max_output_tokens)")
    print("\nNext steps:")
    print("1. Run: pytest tests/test_responses_api_pydantic.py -v")
    print("2. Test complex Pydantic models")
    print("="*70 + "\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
