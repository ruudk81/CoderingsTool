# Azure OpenAI + Responses API Migration Tests

This directory contains comprehensive test suites to validate the migration from OpenAI `chat.completions` API to Azure OpenAI `responses.create` API.

## Prerequisites

Before running these tests:

1. **Install required packages:**
```bash
pip install azure-identity>=1.15.0
pip install instructor>=1.0.0
pip install openai>=1.50.0
pip install pytest pytest-asyncio
```

2. **Configure environment variables:**

For OpenAI testing:
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

For Azure testing:
```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-azure-api-key"  # Optional for managed identity

# Deployment name mappings
export AZURE_GPT41_MINI_DEPLOYMENT="your-gpt41-mini-deployment"
export AZURE_GPT5_MINI_DEPLOYMENT="your-gpt5-mini-deployment"
export AZURE_EMBEDDING_DEPLOYMENT="your-embedding-deployment"
```

## Test Files Overview

### 1. test_azure_connection.py
**Purpose:** Validate Azure OpenAI connectivity and authentication

**Tests:**
- Azure endpoint configuration
- API key authentication
- Managed identity authentication
- Provider auto-detection
- instructor library integration

**Run:**
```bash
pytest tests/test_azure_connection.py -v
```

### 2. test_responses_api_basic.py
**Purpose:** Test basic responses.create functionality

**Tests:**
- Simple Pydantic model extraction
- Parameter differences (input vs messages)
- Sync and async clients
- Both OpenAI and Azure providers

**Run:**
```bash
pytest tests/test_responses_api_basic.py -v
```

### 3. test_responses_api_pydantic.py
**Purpose:** Test complex Pydantic structures (production critical)

**Tests:**
- `List[Model]` response patterns
- Nested models
- Field validation and constraints
- Production patterns (qualityFilter, ideaExtractor)
- Edge cases (empty lists, large lists)

**Run:**
```bash
pytest tests/test_responses_api_pydantic.py -v
```

### 4. test_gpt5_reasoning.py
**Purpose:** Test GPT-5 reasoning model parameters

**Tests:**
- `reasoning_effort` parameter (minimal, medium, high)
- `text_verbosity` parameter (low, medium, high)
- Integration with ModelConfig

**Run:**
```bash
pytest tests/test_gpt5_reasoning.py -v
```

**Note:** Some tests may be skipped if GPT-5 models not yet available.

### 5. test_client_factories.py
**Purpose:** Test config.py client factory functions

**Tests:**
- `create_instructor_client()` for both OpenAI and Azure
- `create_embedding_client()` for embeddings
- Provider auto-detection
- Deployment name mapping
- Stage-based model selection

**Run:**
```bash
pytest tests/test_client_factories.py -v
```

**Note:** These tests require config.py to be refactored first (Phase 3).

### 6. test_dual_execution_paths.py
**Purpose:** Test standalone vs app-orchestrated execution

**Tests:**
- Standalone pipeline (direct config.py)
- App-orchestrated (via cached_resources.py)
- User override handling
- Cache behavior

**Run:**
```bash
pytest tests/test_dual_execution_paths.py -v
```

### 7. test_migration_compatibility.py
**Purpose:** Test actual production utility patterns

**Tests:**
- qualityFilter.py API pattern
- ideaExtractor.py API pattern
- codeAssigner.py API pattern
- Token counting compatibility
- Rate limiting compatibility
- Probe calls for bootstrap

**Run:**
```bash
pytest tests/test_migration_compatibility.py -v
```

## Running All Tests

### Run all tests in order:
```bash
pytest tests/test_azure_*.py -v -s
```

### Run specific test categories:
```bash
# Connection and authentication
pytest tests/test_azure_connection.py -v

# API functionality
pytest tests/test_responses_api_*.py -v

# Migration readiness
pytest tests/test_migration_compatibility.py -v
```

### Run with detailed output:
```bash
pytest tests/ -v -s --tb=short
```

## Success Criteria

**ALL tests must PASS before proceeding with migration!**

If any tests fail:
1. ❌ **DO NOT** proceed with migration
2. 🔍 Debug the failing test
3. 📝 Update documentation with findings
4. 🔄 Re-run all tests

## Test Execution Order

For best results, run tests in this order:

1. **test_azure_connection.py** - Verify Azure is configured
2. **test_responses_api_basic.py** - Verify basic API works
3. **test_responses_api_pydantic.py** - Verify complex models work
4. **test_gpt5_reasoning.py** - Verify GPT-5 parameters (optional)
5. **test_client_factories.py** - After config.py refactoring
6. **test_dual_execution_paths.py** - After cached_resources update
7. **test_migration_compatibility.py** - Final validation

## Skipped Tests

Some tests may be skipped due to missing configuration:

- **Azure tests** - Skip if `AZURE_OPENAI_ENDPOINT` not set
- **OpenAI tests** - Skip if `OPENAI_API_KEY` not set
- **GPT-5 tests** - Skip if models not yet available
- **Factory tests** - Skip if config.py not refactored

Skipped tests are **not failures** - they simply indicate missing prerequisites.

## Debugging Failed Tests

### Azure connection failures:
```bash
# Check endpoint format
echo $AZURE_OPENAI_ENDPOINT  # Should be https://...openai.azure.com/

# Test managed identity
az account show  # Should show your Azure account

# Check RBAC role
az role assignment list --assignee <your-identity-id>
```

### API call failures:
- Check API version is 2025-08-01 or later
- Verify deployment names match your Azure resource
- Check model availability in your region

### Import errors:
```bash
# Reinstall packages
pip install --upgrade azure-identity instructor openai pytest pytest-asyncio
```

## After All Tests Pass

Once all tests pass:

1. ✅ Read `docs/azure_responses_migration_guide.md`
2. ✅ Begin Phase 3: config.py refactoring
3. ✅ Activate specialized migration agent
4. ✅ Proceed with module-by-module migration

## Questions?

Refer to:
- **Migration Guide:** `docs/azure_responses_migration_guide.md`
- **Test Output:** Run with `-v -s` for detailed output
- **Agent Instructions:** See specialized migration agent documentation
