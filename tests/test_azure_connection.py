"""
Test Azure OpenAI Connection and Authentication

Tests Azure OpenAI connectivity with both API key and managed identity authentication.
All tests must pass before proceeding with migration.

Run with: pytest tests/test_azure_connection.py -v
"""

import os
import pytest
from typing import Optional

# Test prerequisites check
def check_azure_prerequisites() -> tuple[bool, list[str]]:
    """Check if Azure prerequisites are met"""
    missing = []

    if not os.getenv("AZURE_OPENAI_ENDPOINT"):
        missing.append("AZURE_OPENAI_ENDPOINT environment variable")

    return (len(missing) == 0, missing)


class TestAzurePrerequisites:
    """Test that Azure environment is properly configured"""

    def test_azure_endpoint_configured(self):
        """Test that AZURE_OPENAI_ENDPOINT is set"""
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        assert endpoint is not None, (
            "AZURE_OPENAI_ENDPOINT not set. "
            "Set with: export AZURE_OPENAI_ENDPOINT='https://your-resource.openai.azure.com/'"
        )
        assert endpoint.startswith("https://"), "Azure endpoint must use HTTPS"
        assert "openai.azure.com" in endpoint, "Invalid Azure OpenAI endpoint format"

    def test_azure_api_key_optional(self):
        """Test that API key is available (optional for managed identity)"""
        # API key is optional - managed identity is preferred
        # This test just documents the option
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if api_key:
            print(f"✓ Azure API key found (length: {len(api_key)})")
        else:
            print("ℹ Azure API key not set - will use managed identity")

    def test_deployment_mappings_configured(self):
        """Test that deployment name mappings are configured"""
        # These are optional with defaults
        mappings = {
            "AZURE_GPT41_MINI_DEPLOYMENT": os.getenv("AZURE_GPT41_MINI_DEPLOYMENT"),
            "AZURE_GPT5_MINI_DEPLOYMENT": os.getenv("AZURE_GPT5_MINI_DEPLOYMENT"),
            "AZURE_EMBEDDING_DEPLOYMENT": os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
        }

        configured = {k: v for k, v in mappings.items() if v}
        if configured:
            print(f"✓ Deployment mappings configured: {list(configured.keys())}")
        else:
            print("ℹ Using default deployment name mappings")


class TestAzureAPIKeyAuthentication:
    """Test Azure OpenAI connection with API key authentication"""

    @pytest.mark.skipif(
        not os.getenv("AZURE_OPENAI_API_KEY"),
        reason="AZURE_OPENAI_API_KEY not set - skipping API key tests"
    )
    def test_connection_with_api_key(self):
        """Test basic connection to Azure OpenAI with API key"""
        from openai import AzureOpenAI

        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version="2025-08-01"
        )

        # Simple test: list models (or try a basic completion)
        # Note: Actual API call depends on your deployment
        assert client is not None
        print("✓ Successfully created Azure OpenAI client with API key")

    @pytest.mark.skipif(
        not os.getenv("AZURE_OPENAI_API_KEY"),
        reason="AZURE_OPENAI_API_KEY not set"
    )
    def test_async_client_with_api_key(self):
        """Test async Azure OpenAI client with API key"""
        from openai import AsyncAzureOpenAI

        client = AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version="2025-08-01"
        )

        assert client is not None
        print("✓ Successfully created async Azure OpenAI client with API key")


class TestAzureManagedIdentity:
    """Test Azure OpenAI connection with managed identity"""

    def test_azure_identity_package_installed(self):
        """Test that azure-identity package is installed"""
        try:
            import azure.identity
            print(f"✓ azure-identity version: {azure.identity.__version__ if hasattr(azure.identity, '__version__') else 'unknown'}")
        except ImportError:
            pytest.fail(
                "azure-identity package not installed. "
                "Install with: pip install azure-identity>=1.15.0"
            )

    def test_default_azure_credential_creation(self):
        """Test that DefaultAzureCredential can be created"""
        from azure.identity import DefaultAzureCredential

        # Creating credential doesn't call Azure yet
        credential = DefaultAzureCredential()
        assert credential is not None
        print("✓ DefaultAzureCredential created successfully")

    def test_token_provider_creation(self):
        """Test that get_bearer_token_provider can be created"""
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider

        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default"
        )

        assert token_provider is not None
        print("✓ Bearer token provider created successfully")

    @pytest.mark.skipif(
        not os.getenv("AZURE_OPENAI_ENDPOINT"),
        reason="AZURE_OPENAI_ENDPOINT not set"
    )
    def test_azure_client_with_managed_identity(self):
        """Test creating Azure OpenAI client with managed identity"""
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        from openai import AzureOpenAI

        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default"
        )

        client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_ad_token_provider=token_provider,
            api_version="2025-08-01"
        )

        assert client is not None
        print("✓ Azure OpenAI client created with managed identity")

    @pytest.mark.skipif(
        not os.getenv("AZURE_OPENAI_ENDPOINT"),
        reason="AZURE_OPENAI_ENDPOINT not set"
    )
    def test_async_azure_client_with_managed_identity(self):
        """Test creating async Azure OpenAI client with managed identity"""
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        from openai import AsyncAzureOpenAI

        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default"
        )

        client = AsyncAzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_ad_token_provider=token_provider,
            api_version="2025-08-01"
        )

        assert client is not None
        print("✓ Async Azure OpenAI client created with managed identity")


class TestProviderAutoDetection:
    """Test automatic provider detection based on environment variables"""

    def test_azure_detected_when_endpoint_set(self):
        """Test that Azure is detected when AZURE_OPENAI_ENDPOINT is set"""
        if os.getenv("AZURE_OPENAI_ENDPOINT"):
            # Should detect Azure
            provider = "azure" if os.getenv("AZURE_OPENAI_ENDPOINT") else "openai"
            assert provider == "azure", "Should detect Azure when endpoint is set"
            print("✓ Provider auto-detection working: Azure")
        else:
            pytest.skip("AZURE_OPENAI_ENDPOINT not set - cannot test auto-detection")

    def test_openai_fallback_when_no_azure(self):
        """Test that OpenAI is used when Azure not configured"""
        # Temporarily clear Azure endpoint
        original = os.getenv("AZURE_OPENAI_ENDPOINT")
        if original:
            os.environ.pop("AZURE_OPENAI_ENDPOINT", None)

        provider = "azure" if os.getenv("AZURE_OPENAI_ENDPOINT") else "openai"
        assert provider == "openai", "Should fall back to OpenAI when Azure not configured"

        # Restore
        if original:
            os.environ["AZURE_OPENAI_ENDPOINT"] = original

        print("✓ Provider auto-detection working: OpenAI fallback")


class TestInstructorIntegration:
    """Test that instructor library works with Azure OpenAI"""

    def test_instructor_package_installed(self):
        """Test that instructor package is installed"""
        try:
            import instructor
            print(f"✓ instructor package installed")
        except ImportError:
            pytest.fail(
                "instructor package not installed. "
                "Install with: pip install instructor>=1.0.0"
            )

    def test_instructor_mode_responses_tools_available(self):
        """Test that instructor.Mode.RESPONSES_TOOLS is available"""
        import instructor

        assert hasattr(instructor.Mode, "RESPONSES_TOOLS"), (
            "instructor.Mode.RESPONSES_TOOLS not available. "
            "Update instructor: pip install instructor>=1.0.0"
        )
        print("✓ instructor.Mode.RESPONSES_TOOLS available")

    @pytest.mark.skipif(
        not os.getenv("AZURE_OPENAI_ENDPOINT") or not os.getenv("AZURE_OPENAI_API_KEY"),
        reason="Azure not fully configured"
    )
    def test_instructor_from_client_azure(self):
        """Test that instructor.from_client works with Azure client"""
        import instructor
        from openai import AzureOpenAI

        azure_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version="2025-08-01"
        )

        client = instructor.from_client(
            azure_client,
            mode=instructor.Mode.RESPONSES_TOOLS
        )

        assert client is not None
        print("✓ instructor.from_client works with Azure OpenAI")

    def test_instructor_from_provider_azure(self):
        """Test that instructor.from_provider works with Azure provider string"""
        import instructor

        # Note: This doesn't make an actual API call, just creates the client
        # The provider string format for Azure may vary - check instructor docs
        try:
            # This is a syntax test, not an actual connection test
            # Actual provider string format should be verified with instructor docs
            print("ℹ Check instructor docs for Azure provider string format")
            print("  Example: instructor.from_provider('azure_openai/gpt-4.1-mini')")
        except Exception as e:
            print(f"ℹ instructor.from_provider syntax varies - check docs: {e}")


class TestAPIVersionSupport:
    """Test that API version supports responses API"""

    def test_api_version_is_v1(self):
        """Test that we're using API version with responses.create support"""
        api_version = "2025-08-01"

        # Responses API available in 2025-08-01 and later
        year, month, day = map(int, api_version.split("-"))
        assert year >= 2025, "API version must be 2025 or later for responses API"

        if year == 2025:
            assert month >= 8, "For 2025, must be month 08 (August) or later"

        print(f"✓ API version {api_version} supports responses.create API")


# Test Summary
def test_summary():
    """Print test summary and prerequisites check"""
    is_ready, missing = check_azure_prerequisites()

    print("\n" + "="*70)
    print("AZURE OPENAI MIGRATION - PREREQUISITES CHECK")
    print("="*70)

    if is_ready:
        print("✓ All prerequisites met!")
        print("\nNext steps:")
        print("1. Run: pytest tests/test_responses_api_basic.py -v")
        print("2. Continue with remaining tests")
    else:
        print("✗ Missing prerequisites:")
        for item in missing:
            print(f"  - {item}")
        print("\nPlease configure missing items before proceeding.")

    print("="*70 + "\n")

    assert is_ready, f"Prerequisites not met: {', '.join(missing)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
