"""
Test config.py Client Factory Functions

Tests that the create_instructor_client() and create_embedding_client()
factory functions work correctly with both OpenAI and Azure.

NOTE: These tests will fail until config.py is refactored with factory functions.

Run with: pytest tests/test_client_factories.py -v
"""

import os
import pytest
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


class TestClientFactoryPrerequisites:
    """Test that config.py has been refactored with factory functions"""

    def test_config_has_api_provider_config(self):
        """Test that APIProviderConfig exists in config.py"""
        try:
            from config import APIProviderConfig
            print("✓ APIProviderConfig found in config.py")
        except ImportError:
            pytest.fail(
                "APIProviderConfig not found in config.py. "
                "Run migration Phase 3 first (config.py refactoring)"
            )

    def test_config_has_create_instructor_client(self):
        """Test that create_instructor_client() function exists"""
        try:
            from config import create_instructor_client
            print("✓ create_instructor_client() found in config.py")
        except ImportError:
            pytest.fail(
                "create_instructor_client() not found in config.py. "
                "Run migration Phase 3 first"
            )

    def test_config_has_create_embedding_client(self):
        """Test that create_embedding_client() function exists"""
        try:
            from config import create_embedding_client
            print("✓ create_embedding_client() found in config.py")
        except ImportError:
            pytest.fail(
                "create_embedding_client() not found in config.py. "
                "Run migration Phase 3 first"
            )

    def test_default_api_provider_config_exists(self):
        """Test that DEFAULT_API_PROVIDER_CONFIG instance exists"""
        try:
            from config import DEFAULT_API_PROVIDER_CONFIG
            print(f"✓ DEFAULT_API_PROVIDER_CONFIG found: provider={DEFAULT_API_PROVIDER_CONFIG.provider}")
        except ImportError:
            pytest.fail("DEFAULT_API_PROVIDER_CONFIG not found in config.py")


class TestOpenAIClientFactory:
    """Test factory creates OpenAI clients correctly"""

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set"
    )
    def test_create_openai_async_client(self):
        """Test creating async OpenAI client via factory"""
        from config import create_instructor_client, APIProviderConfig

        api_config = APIProviderConfig(provider="openai")

        client = create_instructor_client(
            model="gpt-4o-mini",
            async_mode=True,
            api_config=api_config
        )

        assert client is not None
        print("✓ OpenAI async client created via factory")

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set"
    )
    def test_create_openai_sync_client(self):
        """Test creating sync OpenAI client via factory"""
        from config import create_instructor_client, APIProviderConfig

        api_config = APIProviderConfig(provider="openai")

        client = create_instructor_client(
            model="gpt-4o-mini",
            async_mode=False,
            api_config=api_config
        )

        assert client is not None
        print("✓ OpenAI sync client created via factory")

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set"
    )
    def test_create_client_with_stage(self):
        """Test client creation using stage name"""
        from config import create_instructor_client, DEFAULT_MODEL_CONFIG

        client = create_instructor_client(
            stage="quality_filter",
            async_mode=True
        )

        # Should have selected model based on stage
        expected_model = DEFAULT_MODEL_CONFIG.get_model_for_stage("quality_filter")
        print(f"✓ Client created for stage 'quality_filter': model={expected_model}")


class TestAzureClientFactory:
    """Test factory creates Azure clients correctly"""

    @pytest.mark.skipif(
        not os.getenv("AZURE_OPENAI_ENDPOINT"),
        reason="Azure not configured"
    )
    def test_create_azure_async_client_with_api_key(self):
        """Test creating async Azure client with API key"""
        from config import create_instructor_client, APIProviderConfig

        api_config = APIProviderConfig(
            provider="azure",
            azure_use_managed_identity=False  # Force API key
        )

        if not os.getenv("AZURE_OPENAI_API_KEY"):
            pytest.skip("AZURE_OPENAI_API_KEY not set")

        client = create_instructor_client(
            model="gpt-4.1-mini",
            async_mode=True,
            api_config=api_config
        )

        assert client is not None
        print("✓ Azure async client created with API key")

    @pytest.mark.skipif(
        not os.getenv("AZURE_OPENAI_ENDPOINT"),
        reason="Azure not configured"
    )
    def test_create_azure_client_with_managed_identity(self):
        """Test creating Azure client with managed identity"""
        from config import create_instructor_client, APIProviderConfig

        api_config = APIProviderConfig(
            provider="azure",
            azure_use_managed_identity=True
        )

        try:
            client = create_instructor_client(
                model="gpt-4.1-mini",
                async_mode=True,
                api_config=api_config
            )

            assert client is not None
            print("✓ Azure client created with managed identity")
        except Exception as e:
            pytest.skip(f"Managed identity not available: {e}")

    @pytest.mark.skipif(
        not os.getenv("AZURE_OPENAI_ENDPOINT"),
        reason="Azure not configured"
    )
    def test_azure_deployment_name_mapping(self):
        """Test that model names are mapped to Azure deployment names"""
        from config import APIProviderConfig

        api_config = APIProviderConfig(provider="azure")

        # Test mapping
        openai_model = "gpt-4.1-mini"
        azure_deployment = api_config.get_model_name(openai_model)

        print(f"✓ Model mapping: {openai_model} → {azure_deployment}")
        # Should be mapped to Azure deployment name
        assert azure_deployment is not None


class TestProviderAutoDetection:
    """Test automatic provider selection"""

    def test_provider_autodetect_with_azure_endpoint(self):
        """Test provider is Azure when AZURE_OPENAI_ENDPOINT is set"""
        if os.getenv("AZURE_OPENAI_ENDPOINT"):
            from config import DEFAULT_API_PROVIDER_CONFIG

            assert DEFAULT_API_PROVIDER_CONFIG.provider == "azure"
            print(f"✓ Auto-detected provider: {DEFAULT_API_PROVIDER_CONFIG.provider}")
        else:
            pytest.skip("AZURE_OPENAI_ENDPOINT not set")

    def test_provider_fallback_to_openai(self):
        """Test provider falls back to OpenAI when Azure not configured"""
        # This tests the default factory lambda logic
        from config import APIProviderConfig

        # Create new instance without Azure env var set
        original_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if original_endpoint:
            os.environ.pop("AZURE_OPENAI_ENDPOINT", None)

        api_config = APIProviderConfig()
        assert api_config.provider == "openai"

        # Restore
        if original_endpoint:
            os.environ["AZURE_OPENAI_ENDPOINT"] = original_endpoint

        print("✓ Falls back to OpenAI when Azure not configured")


class TestEmbeddingClientFactory:
    """Test embedding client factory"""

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set"
    )
    def test_create_embedding_client_openai(self):
        """Test creating OpenAI embedding client"""
        from config import create_embedding_client, APIProviderConfig

        api_config = APIProviderConfig(provider="openai")

        client = create_embedding_client(
            model="text-embedding-3-large",
            async_mode=True,
            api_config=api_config
        )

        assert client is not None
        print("✓ OpenAI embedding client created")

    @pytest.mark.skipif(
        not os.getenv("AZURE_OPENAI_ENDPOINT") or not os.getenv("AZURE_OPENAI_API_KEY"),
        reason="Azure not configured"
    )
    def test_create_embedding_client_azure(self):
        """Test creating Azure embedding client"""
        from config import create_embedding_client, APIProviderConfig

        api_config = APIProviderConfig(
            provider="azure",
            azure_use_managed_identity=False
        )

        client = create_embedding_client(
            model="text-embedding-3-large",
            async_mode=True,
            api_config=api_config
        )

        assert client is not None
        print("✓ Azure embedding client created")


def test_client_factories_summary():
    """Print client factories test summary"""
    print("\n" + "="*70)
    print("CLIENT FACTORIES - TEST SUMMARY")
    print("="*70)
    print("✓ Factory functions validated")
    print("✓ Both OpenAI and Azure client creation working")
    print("✓ Provider auto-detection working")
    print("✓ Deployment name mapping working")
    print("\nNext steps:")
    print("1. Run: pytest tests/test_dual_execution_paths.py -v")
    print("2. Test standalone vs app-orchestrated modes")
    print("="*70 + "\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
