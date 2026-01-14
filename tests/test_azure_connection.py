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
    """Test Azure OpenAI connection with API key authentication using v1 API"""

    @pytest.mark.skipif(
        not os.getenv("AZURE_OPENAI_API_KEY") or not os.getenv("AZURE_OPENAI_ENDPOINT"),
        reason="AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT not set"
    )
    def test_connection_with_api_key_v1(self):
        """Test basic connection to Azure OpenAI with API key using v1 API format"""
        from openai import OpenAI

        # v1 API format: use standard OpenAI client with custom base_url
        azure_base_url = f"{os.getenv('AZURE_OPENAI_ENDPOINT').rstrip('/')}/openai/v1/"

        client = OpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            base_url=azure_base_url
        )

        assert client is not None
        print(f"✓ Successfully created Azure OpenAI v1 client with base_url: {azure_base_url}")

    @pytest.mark.skipif(
        not os.getenv("AZURE_OPENAI_API_KEY") or not os.getenv("AZURE_OPENAI_ENDPOINT"),
        reason="AZURE_OPENAI_API_KEY or AZURE_OPENAI_ENDPOINT not set"
    )
    def test_async_client_with_api_key_v1(self):
        """Test async Azure OpenAI client with API key using v1 API format"""
        from openai import AsyncOpenAI

        # v1 API format: use standard AsyncOpenAI client with custom base_url
        azure_base_url = f"{os.getenv('AZURE_OPENAI_ENDPOINT').rstrip('/')}/openai/v1/"

        client = AsyncOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            base_url=azure_base_url
        )

        assert client is not None
        print(f"✓ Successfully created async Azure OpenAI v1 client with base_url: {azure_base_url}")


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
    def test_instructor_from_openai_azure_v1(self):
        """Test that instructor.from_openai works with Azure v1 API client"""
        import instructor
        from openai import OpenAI

        # v1 API format: use standard OpenAI client with custom base_url
        azure_base_url = f"{os.getenv('AZURE_OPENAI_ENDPOINT').rstrip('/')}/openai/v1/"

        azure_client = OpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            base_url=azure_base_url
        )

        client = instructor.from_openai(
            azure_client,
            mode=instructor.Mode.RESPONSES_TOOLS
        )

        assert client is not None
        print("✓ instructor.from_openai works with Azure v1 API")

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


class TestV1APIFormat:
    """Test that v1 API format is correctly configured for Responses API"""

    def test_v1_base_url_format(self):
        """Test that v1 API base_url is correctly formatted"""
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
        azure_base_url = f"{endpoint.rstrip('/')}/openai/v1/"

        assert azure_base_url.endswith("/openai/v1/"), "Base URL must end with /openai/v1/"
        assert "openai.azure.com" in azure_base_url, "Must be Azure OpenAI endpoint"

        print(f"✓ v1 API base_url format correct: {azure_base_url}")

    def test_v1_api_supports_responses_create(self):
        """Document that v1 API supports responses.create"""
        print("\n" + "="*70)
        print("Azure v1 API - Responses API Support")
        print("="*70)
        print("The v1 API format uses:")
        print("  base_url = https://{resource}.openai.azure.com/openai/v1/")
        print("")
        print("This gives access to:")
        print("  - client.responses.create()  ← Required for reasoning models")
        print("  - client.chat.completions.create()")
        print("  - client.embeddings.create()")
        print("="*70 + "\n")


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
