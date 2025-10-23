# Azure OpenAI + Responses API Migration Guide

**Version:** 1.0
**Date:** 2025-01-19
**Status:** Preparation Phase (No Migration Yet)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Deep Dive](#2-architecture-deep-dive)
3. [Prerequisites & Dependencies](#3-prerequisites--dependencies)
4. [Azure Setup Guide](#4-azure-setup-guide)
5. [API Migration Reference](#5-api-migration-reference)
6. [config.py Refactoring Plan](#6-configpy-refactoring-plan)
7. [cached_resources.py Updates](#7-cached_resourcespy-updates)
8. [Module-by-Module Migration Guide](#8-module-by-module-migration-guide)
9. [Testing Strategy](#9-testing-strategy)
10. [Rollback Plan](#10-rollback-plan)
11. [Specialized Agent Instructions](#11-specialized-agent-instructions)

---

## 1. Executive Summary

### Why Migrate?

**Three primary goals:**

1. **Azure Managed Identity Support**
   - Eliminate API key management (more secure)
   - Use Azure Active Directory authentication
   - Seamless integration when deployed on Azure infrastructure

2. **GPT-5 Reasoning Model Support**
   - GPT-5 models require the Responses API for full reasoning capabilities
   - Access to chain-of-thought reasoning (server-side managed)
   - Better performance with reasoning carryover

3. **Unified API Experience**
   - Single API pattern for both chat models (GPT-4) and reasoning models (GPT-5)
   - Server-managed conversation state
   - Future-proof architecture

### What Changes?

**API Level:**
- `chat.completions.create()` → `responses.create()`
- `messages` parameter → `input` parameter
- `max_tokens` → `max_output_tokens`
- Response structure differences

**Infrastructure Level:**
- OpenAI direct connection → Azure OpenAI endpoint
- API key authentication → Managed identity (DefaultAzureCredential)
- Model names → Azure deployment names

**Code Level:**
- Centralized client factories in `config.py`
- Updated `cached_resources.py` to use factories
- 7 utility files need migration

### When to Migrate?

**Critical Prerequisites:**
1. ✅ App is fully working and tested
2. ✅ App deployed to Azure infrastructure
3. ✅ Azure OpenAI resource provisioned
4. ✅ Managed identity configured with correct RBAC roles
5. ✅ All test scripts pass (`tests/test_azure_*.py`)

**DO NOT** migrate until all prerequisites are met!

---

## 2. Architecture Deep Dive

### Current Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CoderingsTool                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Execution Path 1: Standalone Pipeline                      │
│  ┌────────────┐                                             │
│  │ pipeline.py│ ──────> config.py ──────> OpenAI API        │
│  └────────────┘         (direct)           (API key)        │
│                                                              │
│  Execution Path 2: App-Orchestrated (Streamlit)             │
│  ┌──────────┐         ┌──────────────────┐                 │
│  │  app.py  │ ─────>  │cached_resources.py│ ──> config.py  │
│  │ (UI picks│         │  (adds caching)   │     (fallback) │
│  │  models) │         └──────────────────┘                 │
│  └──────────┘                  │                            │
│                                 v                            │
│                            OpenAI API                        │
│                            (API key)                         │
└─────────────────────────────────────────────────────────────┘

config.py = Source of Truth
- Model selection (ModelConfig)
- Rate limits
- API keys
- All step-specific configs

cached_resources.py = Runtime Performance Layer
- Session-wide client caching
- User UI override handling
- Wraps config.py with @conditional_cache_resource
```

### Target Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CoderingsTool                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Execution Path 1: Standalone Pipeline                      │
│  ┌────────────┐                                             │
│  │ pipeline.py│ ──> config.py ──> Azure OpenAI              │
│  └────────────┘      (factories)   (managed identity)       │
│                                                              │
│  Execution Path 2: App-Orchestrated (Streamlit)             │
│  ┌──────────┐       ┌──────────────────┐                   │
│  │  app.py  │ ───>  │cached_resources.py│ ──> config.py    │
│  │ (UI picks│       │  (wraps factories)│     (factories)   │
│  │  models) │       └──────────────────┘                   │
│  └──────────┘                 │                             │
│                                v                             │
│                         Azure OpenAI                         │
│                      (managed identity)                      │
└─────────────────────────────────────────────────────────────┘

config.py Additions:
+ APIProviderConfig (Azure + OpenAI settings)
+ create_instructor_client() factory
+ create_embedding_client() factory
+ Auto-detection of provider via env vars

cached_resources.py Updates:
+ Calls config.py factories instead of direct instantiation
+ Applies user overrides to factory parameters
+ Maintains caching behavior
```

### Key Architectural Principles

1. **config.py = Source of Truth**
   - All configuration lives here
   - Client factories defined here
   - No hardcoded API keys or endpoints

2. **cached_resources.py = Performance Layer**
   - Wraps config factories with caching
   - Handles user UI selections
   - Does NOT contain business logic

3. **Dual Execution Path Support**
   - Both standalone and app modes must work
   - Test both paths for every change
   - Config overrides flow correctly

4. **Provider Agnostic**
   - Auto-detect based on environment variables
   - Fallback to OpenAI if Azure not configured
   - Explicit override option available

---

## 3. Prerequisites & Dependencies

### Python Packages

```bash
# Azure authentication
pip install azure-identity>=1.15.0

# Latest instructor with RESPONSES_TOOLS mode
pip install instructor>=1.0.0

# Latest OpenAI SDK with responses.create support
pip install openai>=1.50.0
```

### Azure Resources Required

1. **Azure OpenAI Resource**
   - Provisioned in supported region (e.g., East US2, Sweden Central)
   - API version 2025-08-01 or later (for responses API)

2. **Model Deployments**
   - Deploy required models (gpt-4.1-mini, gpt-5-mini, etc.)
   - Note deployment names (may differ from OpenAI model names)

3. **Managed Identity Setup**
   - System-assigned or user-assigned managed identity
   - RBAC role: "Cognitive Services OpenAI User"
   - Scoped to Azure OpenAI resource

4. **Environment Variables**
   - `AZURE_OPENAI_ENDPOINT` - Your resource endpoint
   - Optional: `AZURE_OPENAI_API_KEY` - For testing before managed identity
   - Model deployment mappings (see Section 4)

---

## 4. Azure Setup Guide

### 4.1 Environment Variables

**Required:**
```bash
# Azure endpoint URL
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
```

**Optional (for testing with API key before managed identity):**
```bash
export AZURE_OPENAI_API_KEY="your-api-key-here"
```

**Model Deployment Mappings:**
```bash
# Map OpenAI model names to Azure deployment names
export AZURE_GPT41_MINI_DEPLOYMENT="my-gpt41-mini-deployment"
export AZURE_GPT5_MINI_DEPLOYMENT="my-gpt5-mini-deployment"
export AZURE_EMBEDDING_DEPLOYMENT="my-embedding-deployment"
```

### 4.2 Managed Identity Setup

#### Step 1: Assign RBAC Role

```bash
# Get your resource ID
RESOURCE_ID=$(az cognitiveservices account show \
  --name your-openai-resource \
  --resource-group your-rg \
  --query id -o tsv)

# Get your managed identity principal ID
PRINCIPAL_ID=$(az identity show \
  --name your-identity \
  --resource-group your-rg \
  --query principalId -o tsv)

# Assign role
az role assignment create \
  --role "Cognitive Services OpenAI User" \
  --assignee $PRINCIPAL_ID \
  --scope $RESOURCE_ID
```

#### Step 2: Test Managed Identity

```python
from azure.identity import DefaultAzureCredential

# This will try multiple credential sources in order:
# 1. Environment variables
# 2. Managed identity
# 3. Azure CLI
# 4. Interactive browser

credential = DefaultAzureCredential()
token = credential.get_token("https://cognitiveservices.azure.com/.default")
print(f"Successfully obtained token: {token.token[:20]}...")
```

### 4.3 Code Example: Azure OpenAI with Managed Identity

```python
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI
import instructor
import os

# Create token provider
token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default"
)

# Create Azure OpenAI client
azure_client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_ad_token_provider=token_provider,  # No API key needed!
    api_version="2025-08-01"  # v1 API for responses.create
)

# Patch with instructor for structured outputs
client = instructor.from_client(
    azure_client,
    mode=instructor.Mode.RESPONSES_TOOLS
)

# Now client.responses.create() works with managed identity!
```

### 4.4 Deployment Name Mapping

**Problem:** Azure uses deployment names, not model names

```python
# OpenAI direct
model = "gpt-4.1-mini"  # ✅ Works

# Azure OpenAI
model = "gpt-4.1-mini"  # ❌ Fails - no deployment with this name
model = "my-gpt41-mini-deployment"  # ✅ Works - actual deployment name
```

**Solution:** Environment variable mapping

```python
# In config.py
azure_deployments = {
    "gpt-4.1-mini": os.getenv("AZURE_GPT41_MINI_DEPLOYMENT", "gpt-41-mini"),
    "gpt-5-mini": os.getenv("AZURE_GPT5_MINI_DEPLOYMENT", "gpt-5-mini"),
    "text-embedding-3-large": os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
}

# Usage
def get_azure_model_name(openai_model: str) -> str:
    return azure_deployments.get(openai_model, openai_model)
```

---

## 5. API Migration Reference

### 5.1 Core API Change: chat.completions → responses

#### Current Pattern (chat.completions.create)

```python
import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel
from typing import List

# Client setup
client = instructor.patch(AsyncOpenAI(api_key=OPENAI_API_KEY))

# Pydantic model
class QualityFilteredModel(BaseModel):
    respondent_id: str
    is_high_quality: bool
    confidence: float

# API call
response = await client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a quality filter."},
        {"role": "user", "content": prompt}
    ],
    response_model=List[QualityFilteredModel],
    temperature=0.0,
    max_tokens=4000,
    seed=42
)

# Access usage
usage = response._raw_response.usage
print(f"Tokens: {usage.total_tokens}")
```

#### Target Pattern (responses.create)

```python
import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel
from typing import List

# Client setup
client = instructor.from_provider(
    "openai/gpt-4o-mini",
    mode=instructor.Mode.RESPONSES_TOOLS,
    async_client=True
)

# Pydantic model (unchanged)
class QualityFilteredModel(BaseModel):
    respondent_id: str
    is_high_quality: bool
    confidence: float

# API call
response = await client.responses.create(
    input=prompt,  # ← Changed from messages
    response_model=List[QualityFilteredModel],
    temperature=0.0,
    max_output_tokens=4000,  # ← Changed from max_tokens
    # seed parameter handled differently in responses API
)

# Access usage (structure differs)
# Response API returns usage in different format
# Check actual response structure
```

### 5.2 Parameter Mapping

| chat.completions.create | responses.create | Notes |
|------------------------|------------------|-------|
| `messages=[...]` | `input="..."` | String instead of array |
| `max_tokens=4000` | `max_output_tokens=4000` | Renamed parameter |
| `seed=42` | Handled differently | Check latest API docs |
| `response._raw_response.usage` | Response format differs | Verify structure |

### 5.3 Instructor Mode Differences

#### Old: instructor.patch()

```python
from openai import AsyncOpenAI
import instructor

client = instructor.patch(AsyncOpenAI(api_key=API_KEY))
# Uses chat.completions.create internally
```

#### New: instructor.from_provider() with RESPONSES_TOOLS

```python
import instructor

# Method 1: from_provider (easiest)
client = instructor.from_provider(
    "openai/gpt-4.1-mini",
    mode=instructor.Mode.RESPONSES_TOOLS,
    async_client=True
)

# Method 2: from_client (more control)
from openai import AzureOpenAI

azure_client = AzureOpenAI(...)
client = instructor.from_client(
    azure_client,
    mode=instructor.Mode.RESPONSES_TOOLS
)
```

### 5.4 GPT-5 Reasoning Models

**Special parameters for reasoning models:**

```python
# GPT-5 models support reasoning parameters
response = await client.responses.create(
    input=prompt,
    response_model=MyModel,
    # Reasoning parameters (GPT-5 only)
    reasoning={"effort": "medium"},  # minimal | medium | high
    text={"verbosity": "medium"}     # low | medium | high
)
```

**Integration with ModelConfig:**

```python
# In config.py ModelConfig
def get_reasoning_effort_for_stage(self, stage: str) -> str:
    """Get GPT-5 reasoning effort for specific stage"""
    stage_efforts = {
        'theme_extraction': self.theme_extraction_reasoning_effort,
        'candidate_selection': self.candidate_selection_reasoning_effort,
        ...
    }
    return stage_efforts.get(stage, self.gpt5_reasoning_effort)
```

### 5.5 Error Handling Differences

```python
# chat.completions errors
from openai import RateLimitError, APIConnectionError, APITimeoutError

# Same errors apply to responses API
# But server-side state may affect retry behavior
```

---

## 6. config.py Refactoring Plan

### 6.1 Add APIProviderConfig (After line 117)

```python
@dataclass
class APIProviderConfig:
    """
    Azure + OpenAI provider configuration.

    Auto-detects provider based on environment variables:
    - If AZURE_OPENAI_ENDPOINT is set → use Azure
    - Otherwise → use OpenAI
    """

    # Provider selection
    provider: str = field(default_factory=lambda:
        "azure" if os.getenv("AZURE_OPENAI_ENDPOINT") else "openai"
    )

    # =========================================================================
    # OPENAI CONFIGURATION
    # =========================================================================
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))

    # =========================================================================
    # AZURE CONFIGURATION
    # =========================================================================

    # Azure endpoint (e.g., https://your-resource.openai.azure.com/)
    azure_endpoint: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_ENDPOINT"))

    # Azure API key (optional - for testing before managed identity)
    azure_api_key: str = field(default_factory=lambda: os.getenv("AZURE_OPENAI_API_KEY"))

    # Use managed identity for authentication (preferred for production)
    azure_use_managed_identity: bool = True

    # API version for responses.create support
    azure_api_version: str = "2025-08-01"

    # Model deployment name mappings (Azure deployment names ≠ OpenAI model names)
    azure_deployments: Dict[str, str] = field(default_factory=lambda: {
        # Chat models
        "gpt-4.1-mini": os.getenv("AZURE_GPT41_MINI_DEPLOYMENT", "gpt-41-mini"),
        "gpt-4.1": os.getenv("AZURE_GPT41_DEPLOYMENT", "gpt-41"),
        "gpt-4o-mini": os.getenv("AZURE_GPT4O_MINI_DEPLOYMENT", "gpt-4o-mini"),

        # Reasoning models
        "gpt-5": os.getenv("AZURE_GPT5_DEPLOYMENT", "gpt-5"),
        "gpt-5-mini": os.getenv("AZURE_GPT5_MINI_DEPLOYMENT", "gpt-5-mini"),
        "gpt-5-nano": os.getenv("AZURE_GPT5_NANO_DEPLOYMENT", "gpt-5-nano"),

        # Embedding models
        "text-embedding-3-large": os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"),
        "text-embedding-3-small": os.getenv("AZURE_EMBEDDING_SMALL_DEPLOYMENT", "text-embedding-3-small"),
    })

    def get_model_name(self, openai_model: str) -> str:
        """
        Get the appropriate model/deployment name based on provider.

        For OpenAI: returns model name as-is
        For Azure: returns mapped deployment name
        """
        if self.provider == "azure":
            return self.azure_deployments.get(openai_model, openai_model)
        return openai_model
```

### 6.2 Add Client Factory Functions (Before DEFAULT_MODEL_CONFIG)

```python
def create_instructor_client(
    model: str,
    stage: str = None,
    provider: str = None,
    async_mode: bool = True,
    mode: instructor.Mode = instructor.Mode.RESPONSES_TOOLS,
    api_config: APIProviderConfig = None,
    model_config: ModelConfig = None
) -> Any:
    """
    Create instructor-patched client for structured outputs with responses.create API.

    Supports both OpenAI and Azure OpenAI with automatic provider detection.

    Args:
        model: OpenAI model name (e.g., "gpt-4.1-mini")
        stage: Pipeline stage name for model selection
        provider: Override auto-detection ("openai" or "azure")
        async_mode: Return async client if True, sync if False
        mode: Instructor mode (RESPONSES_TOOLS recommended)
        api_config: API provider configuration (uses default if None)
        model_config: Model configuration (uses default if None)

    Returns:
        Instructor-patched client ready for responses.create() calls

    Example:
        # Auto-detect provider
        client = create_instructor_client(
            model="gpt-4.1-mini",
            stage="quality_filter",
            async_mode=True
        )

        # Explicit provider
        client = create_instructor_client(
            model="gpt-5-mini",
            provider="azure",
            async_mode=True
        )
    """
    import instructor
    from openai import OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI

    # Use defaults if not provided
    api_config = api_config or APIProviderConfig()
    model_config = model_config or ModelConfig()

    # Override provider if specified
    if provider:
        api_config.provider = provider

    # Get model name for stage if stage provided
    if stage:
        model = model_config.get_model_for_stage(stage)

    # Get appropriate model/deployment name
    model_name = api_config.get_model_name(model)

    # Create base client based on provider
    if api_config.provider == "azure":
        # Azure OpenAI client

        # Prepare authentication
        if api_config.azure_use_managed_identity and not api_config.azure_api_key:
            # Use managed identity (preferred)
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider

            token_provider = get_bearer_token_provider(
                DefaultAzureCredential(),
                "https://cognitiveservices.azure.com/.default"
            )

            if async_mode:
                base_client = AsyncAzureOpenAI(
                    azure_endpoint=api_config.azure_endpoint,
                    azure_ad_token_provider=token_provider,
                    api_version=api_config.azure_api_version
                )
            else:
                base_client = AzureOpenAI(
                    azure_endpoint=api_config.azure_endpoint,
                    azure_ad_token_provider=token_provider,
                    api_version=api_config.azure_api_version
                )
        else:
            # Use API key (for testing)
            if async_mode:
                base_client = AsyncAzureOpenAI(
                    api_key=api_config.azure_api_key,
                    azure_endpoint=api_config.azure_endpoint,
                    api_version=api_config.azure_api_version
                )
            else:
                base_client = AzureOpenAI(
                    api_key=api_config.azure_api_key,
                    azure_endpoint=api_config.azure_endpoint,
                    api_version=api_config.azure_api_version
                )
    else:
        # OpenAI client
        if async_mode:
            base_client = AsyncOpenAI(api_key=api_config.openai_api_key)
        else:
            base_client = OpenAI(api_key=api_config.openai_api_key)

    # Patch with instructor for structured outputs
    client = instructor.from_client(base_client, mode=mode)

    return client


def create_embedding_client(
    model: str = None,
    provider: str = None,
    async_mode: bool = True,
    api_config: APIProviderConfig = None,
    model_config: ModelConfig = None
) -> Any:
    """
    Create client for embeddings API.

    Note: Embeddings use embeddings.create(), NOT responses.create()

    Args:
        model: Embedding model name
        provider: Override auto-detection ("openai" or "azure")
        async_mode: Return async client if True, sync if False
        api_config: API provider configuration (uses default if None)
        model_config: Model configuration (uses default if None)

    Returns:
        OpenAI/AzureOpenAI client for embeddings
    """
    from openai import OpenAI, AsyncOpenAI, AzureOpenAI, AsyncAzureOpenAI

    # Use defaults if not provided
    api_config = api_config or APIProviderConfig()
    model_config = model_config or ModelConfig()

    # Override provider if specified
    if provider:
        api_config.provider = provider

    # Get embedding model
    if not model:
        model = model_config.embedding_model

    # Get appropriate model/deployment name
    model_name = api_config.get_model_name(model)

    # Create client based on provider
    # (Same logic as create_instructor_client but NO instructor patching)
    if api_config.provider == "azure":
        if api_config.azure_use_managed_identity and not api_config.azure_api_key:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider

            token_provider = get_bearer_token_provider(
                DefaultAzureCredential(),
                "https://cognitiveservices.azure.com/.default"
            )

            if async_mode:
                return AsyncAzureOpenAI(
                    azure_endpoint=api_config.azure_endpoint,
                    azure_ad_token_provider=token_provider,
                    api_version=api_config.azure_api_version
                )
            else:
                return AzureOpenAI(
                    azure_endpoint=api_config.azure_endpoint,
                    azure_ad_token_provider=token_provider,
                    api_version=api_config.azure_api_version
                )
        else:
            if async_mode:
                return AsyncAzureOpenAI(
                    api_key=api_config.azure_api_key,
                    azure_endpoint=api_config.azure_endpoint,
                    api_version=api_config.azure_api_version
                )
            else:
                return AzureOpenAI(
                    api_key=api_config.azure_api_key,
                    azure_endpoint=api_config.azure_endpoint,
                    api_version=api_config.azure_api_version
                )
    else:
        if async_mode:
            return AsyncOpenAI(api_key=api_config.openai_api_key)
        else:
            return OpenAI(api_key=api_config.openai_api_key)
```

### 6.3 Add Default Instance (After line 777)

```python
# API Provider configuration
DEFAULT_API_PROVIDER_CONFIG = APIProviderConfig()
```

### 6.4 Summary of config.py Changes

**Lines to add:**
- ~Line 118: `APIProviderConfig` dataclass (~80 lines)
- Before `DEFAULT_MODEL_CONFIG`: Client factory functions (~150 lines)
- After line 777: `DEFAULT_API_PROVIDER_CONFIG` (1 line)

**Total additions:** ~230 lines

**No deletions** - all existing code remains

---

## 7. cached_resources.py Updates

### 7.1 Current Implementation

```python
# cached_resources.py (lines 10-15)
@conditional_cache_resource
def get_openai_client(api_key: str = None):
    """Create cached OpenAI instructor client for session-wide reuse"""
    api_key = api_key or OPENAI_API_KEY
    with conditional_spinner("Initializing OpenAI client..."):
        return instructor.patch(AsyncOpenAI(api_key=api_key))
```

### 7.2 Target Implementation

```python
# cached_resources.py (updated)
@conditional_cache_resource
def get_openai_client(
    api_key: str = None,
    model: str = None,
    stage: str = None,
    provider: str = None,
    async_mode: bool = True
):
    """
    Create cached instructor client for session-wide reuse.

    Now calls config.py client factory instead of direct instantiation.
    Supports both OpenAI and Azure OpenAI with automatic provider detection.

    Args:
        api_key: Optional API key override (for UI selection)
        model: Optional model override (for UI selection)
        stage: Pipeline stage name for model selection
        provider: Optional provider override ("openai" or "azure")
        async_mode: Return async client if True

    Returns:
        Instructor-patched client ready for responses.create()
    """
    from config import (
        create_instructor_client,
        DEFAULT_API_PROVIDER_CONFIG,
        DEFAULT_MODEL_CONFIG
    )

    # Create API config with overrides
    api_config = DEFAULT_API_PROVIDER_CONFIG

    # Apply user overrides from UI
    if provider:
        api_config.provider = provider
    if api_key:
        if api_config.provider == "azure":
            api_config.azure_api_key = api_key
            api_config.azure_use_managed_identity = False  # Using key, not managed identity
        else:
            api_config.openai_api_key = api_key

    # Display appropriate spinner
    provider_name = api_config.provider.upper()
    with conditional_spinner(f"Initializing {provider_name} client..."):
        return create_instructor_client(
            model=model or DEFAULT_MODEL_CONFIG.get_model_for_stage(stage) if stage else "gpt-4.1-mini",
            stage=stage,
            async_mode=async_mode,
            api_config=api_config,
            mode=instructor.Mode.RESPONSES_TOOLS
        )
```

### 7.3 Add get_embedding_client (if needed)

```python
@conditional_cache_resource
def get_embedding_client(
    provider: str = None,
    model: str = None,
    async_mode: bool = True
):
    """
    Create cached embedding client for session-wide reuse.

    Args:
        provider: Optional provider override ("openai" or "azure")
        model: Optional embedding model override
        async_mode: Return async client if True

    Returns:
        OpenAI/AzureOpenAI client for embeddings
    """
    from config import (
        create_embedding_client,
        DEFAULT_API_PROVIDER_CONFIG,
        DEFAULT_MODEL_CONFIG
    )

    api_config = DEFAULT_API_PROVIDER_CONFIG
    if provider:
        api_config.provider = provider

    provider_name = api_config.provider.upper()
    with conditional_spinner(f"Initializing {provider_name} embedding client..."):
        return create_embedding_client(
            model=model or DEFAULT_MODEL_CONFIG.embedding_model,
            async_mode=async_mode,
            api_config=api_config
        )
```

### 7.4 Update Cache Keys

**Important:** Cache keys should include provider + API version to avoid conflicts

```python
# Add provider info to cache key generation
# This ensures OpenAI and Azure clients are cached separately
```

---

## 8. Module-by-Module Migration Guide

### 8.1 Migration Order

1. ✅ **qualityFilter.py** - Simplest async pattern, good test case
2. ✅ **ideaExtractor.py** - Similar async pattern
3. ✅ **codeAssigner.py** - Similar async pattern
4. ✅ **spellChecker.py** - More complex async with token counting
5. ✅ **codeGenerator.py** - **Special case: uses sync client**
6. ✅ **embedder.py** - **Special case: uses embeddings API (may not need changes)**
7. ✅ **speculativeStarterCodes.py** - Small utility

### 8.2 qualityFilter.py Migration

**File:** `src/utils/qualityFilter.py`

#### Current Code (Lines 22-27)

```python
# === CONFIG ========================================================================================================
from config import OPENAI_API_KEY, DEFAULT_LANGUAGE, ModelConfig, QualityFilterConfig, DEFAULT_QUALITY_FILTER_CONFIG, get_openai_rate_limits
from prompts import GRADER_INSTRUCTIONS

# === UTILS ========================================================================================================
from .verboseReporter import VerboseReporter, ProcessingStats
from .cached_resources import get_openai_client, get_tiktoken_encoding
```

#### Target Code

```python
# === CONFIG ========================================================================================================
from config import (
    DEFAULT_LANGUAGE,
    ModelConfig,
    QualityFilterConfig,
    DEFAULT_QUALITY_FILTER_CONFIG,
    get_openai_rate_limits,
    create_instructor_client,  # ← NEW
    DEFAULT_API_PROVIDER_CONFIG  # ← NEW
)
from prompts import GRADER_INSTRUCTIONS

# === UTILS ========================================================================================================
from .verboseReporter import VerboseReporter, ProcessingStats
from .cached_resources import get_tiktoken_encoding
# Note: get_openai_client can still be used, it now wraps config factory
```

#### Current Code (Lines 165-175, client initialization)

```python
class QualityFilter:
    def __init__(
        self,
        var_lab: str,
        verbose: bool = True,
        client: Any = None,
        config: QualityFilterConfig = None,
        model_config: ModelConfig = None
    ):
        self.config = config or DEFAULT_QUALITY_FILTER_CONFIG
        self.model_config = model_config or ModelConfig()
        self.model = self.model_config.get_model_for_stage('quality_filter')
        self.client = client or get_openai_client(api_key=OPENAI_API_KEY)
```

#### Target Code

```python
class QualityFilter:
    def __init__(
        self,
        var_lab: str,
        verbose: bool = True,
        client: Any = None,
        config: QualityFilterConfig = None,
        model_config: ModelConfig = None,
        api_config: APIProviderConfig = None  # ← NEW parameter
    ):
        self.config = config or DEFAULT_QUALITY_FILTER_CONFIG
        self.model_config = model_config or ModelConfig()
        self.api_config = api_config or DEFAULT_API_PROVIDER_CONFIG  # ← NEW
        self.model = self.model_config.get_model_for_stage('quality_filter')

        # Use provided client or create new one via factory
        self.client = client or create_instructor_client(
            stage='quality_filter',
            async_mode=True,
            api_config=self.api_config,
            model_config=self.model_config
        )
```

#### Current Code (Lines 405-414, API call)

```python
response = await asyncio.wait_for(
    self.client.chat.completions.create(
        model=self.model,
        response_model=List[models.QualityFilteredModel],
        messages=[{"role": "user", "content": prompt}],
        temperature=self.config.temperature,
        max_tokens=self.config.max_tokens,
        seed=self.model_config.seed
    ),
    timeout=timeout
)
```

#### Target Code

```python
response = await asyncio.wait_for(
    self.client.responses.create(
        input=prompt,  # ← Changed from messages
        response_model=List[models.QualityFilteredModel],
        temperature=self.config.temperature,
        max_output_tokens=self.config.max_tokens,  # ← Changed parameter name
        # Note: seed handling may differ in responses API
    ),
    timeout=timeout
)
```

#### Current Code (Lines 482-490, probe call for token estimation)

```python
# For probes: avoid response_model so we can read .usage
resp = await self.client.chat.completions.create(
    model=self.model,
    messages=[{"role": "user", "content": prompt}],
    temperature=self.config.temperature,
    seed=self.model_config.seed
)

u = getattr(resp, "usage", None)
return {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens}
```

#### Target Code

```python
# For probes: avoid response_model so we can read .usage
resp = await self.client.responses.create(
    input=prompt,  # ← Changed from messages
    temperature=self.config.temperature,
)

# Note: Response API usage format may differ - verify structure
u = getattr(resp, "usage", None)
return {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens}
```

#### Testing Checklist for qualityFilter.py

- [ ] Standalone pipeline mode works
- [ ] App-orchestrated mode works
- [ ] User model selection in UI is respected
- [ ] Token counting still accurate
- [ ] Rate limiting still works
- [ ] Cache behavior unchanged
- [ ] Bootstrap probes succeed
- [ ] Pydantic validation errors handled correctly

---

### 8.3 ideaExtractor.py Migration

**File:** `src/utils/ideaExtractor.py`

**Pattern:** Very similar to qualityFilter.py

#### Key Changes

1. **Import changes** (add create_instructor_client, DEFAULT_API_PROVIDER_CONFIG)
2. **Client initialization** (use factory instead of get_openai_client)
3. **Subject extraction call** (line 305): `chat.completions.create` → `responses.create`
4. **Idea extraction call** (line 556): `chat.completions.create` → `responses.create`
5. **Probe calls** (line 468): Update for responses API
6. **Parameter changes**: `messages` → `input`, `max_tokens` → `max_output_tokens`

#### Testing Checklist

- [ ] Subject extraction works correctly
- [ ] Idea segmentation works correctly
- [ ] Multi-idea responses handled properly
- [ ] Token estimation accurate
- [ ] Both execution paths work

---

### 8.4 codeAssigner.py Migration

**File:** `src/utils/codeAssigner.py`

**Pattern:** Same as qualityFilter.py and ideaExtractor.py

#### Key Changes

1. Import updates
2. Client initialization with factory
3. Line 608: `chat.completions.create` → `responses.create`
4. Line 536 (probe): Update for responses API
5. Parameter changes

#### Testing Checklist

- [ ] Code assignment works correctly
- [ ] Top-k similar codes retrieved properly
- [ ] Confidence scores calculated correctly
- [ ] Token counting accurate

---

### 8.5 spellChecker.py Migration

**File:** `src/utils/spellChecker.py`

**Pattern:** Similar to others, but more complex token counting logic

#### Key Changes

1. Import updates
2. Client initialization (line ~115)
3. Line 1023 (probe): Update for responses API
4. Line 1089 (correction call): `chat.completions.create` → `responses.create`
5. **Important:** Token counting logic extensively used - verify all calculations

#### Special Considerations

- Token bucket logic must be updated if usage format differs
- Adaptive timeout calculations based on tokens
- Pre-validation logic may be affected

#### Testing Checklist

- [ ] OOV detection works
- [ ] Suggestion generation works
- [ ] LLM correction calls work
- [ ] Token estimation highly accurate
- [ ] Rate limiting smooth
- [ ] No performance regression

---

### 8.6 codeGenerator.py Migration

**File:** `src/utils/codeGenerator.py`

**⚠️ SPECIAL CASE:** Uses **sync client**, not async

#### Current Code (Line 36)

```python
client = OpenAI()
```

#### Target Code

```python
from config import create_instructor_client, DEFAULT_API_PROVIDER_CONFIG

client = create_instructor_client(
    model="gpt-4.1-mini",  # Or appropriate model
    async_mode=False,  # ← SYNC CLIENT
    api_config=DEFAULT_API_PROVIDER_CONFIG
)
```

#### Key Changes

1. Line 36: Global client initialization
2. Line 864, 885: Additional client instances
3. Line 1311: Embedding client (separate - uses embeddings API)
4. Multiple completion calls throughout need `chat.completions` → `responses` migration

#### Special Considerations

- Sync vs async carefully maintained
- SharedCodebook class may have client handling
- Embedding calls stay on embeddings.create (not responses)

#### Testing Checklist

- [ ] All 4 LLM prompts work (theme extraction, decision, generation, validation)
- [ ] Sync client behavior maintained
- [ ] Embedding generation separate and working
- [ ] Codebook versioning works
- [ ] No blocking or performance issues

---

### 8.7 embedder.py Migration

**File:** `src/utils/embedder.py`

**⚠️ SPECIAL CASE:** Uses **embeddings API**, not chat/responses API

#### Current Code (Lines 54-55)

```python
if self.provider == "openai":
    self.client = client or AsyncOpenAI(api_key=OPENAI_API_KEY)
```

#### Target Code

```python
if self.provider == "openai":
    from config import create_embedding_client, DEFAULT_API_PROVIDER_CONFIG
    self.client = client or create_embedding_client(
        model=self.embedding_model,
        async_mode=True,
        api_config=DEFAULT_API_PROVIDER_CONFIG
    )
```

#### Key Considerations

- **Embeddings use `client.embeddings.create()`, NOT `responses.create()`**
- May need minimal changes (just client initialization)
- Gemini provider also supported - test both

#### Testing Checklist

- [ ] OpenAI embeddings work
- [ ] Gemini embeddings work (if used)
- [ ] Question-aware embeddings work
- [ ] Batch processing works
- [ ] Both providers can be used

---

### 8.8 speculativeStarterCodes.py Migration

**File:** `src/utils/speculativeStarterCodes.py`

**Pattern:** Similar to qualityFilter.py

#### Current Code (Line 42)

```python
self.client = instructor.patch(AsyncOpenAI(api_key=OPENAI_API_KEY))
```

#### Target Code

```python
from config import create_instructor_client, DEFAULT_API_PROVIDER_CONFIG

self.client = create_instructor_client(
    stage='speculative_codes',
    async_mode=True,
    api_config=DEFAULT_API_PROVIDER_CONFIG
)
```

#### Testing Checklist

- [ ] Speculative code generation works
- [ ] Structured output validation works

---

### 8.9 codebookRefinement.py Migration

**File:** `src/utils/codebookRefinement.py`

**Pattern:** Uses both async and sync clients

#### Current Code (Lines 42, 150)

```python
# Line 42 (async)
self.client = AsyncOpenAI(api_key=self.api_key)

# Line 150 (sync)
client = OpenAI(api_key=self.api_key)
```

#### Target Code

```python
from config import create_instructor_client, DEFAULT_API_PROVIDER_CONFIG

# Line 42 (async)
self.client = create_instructor_client(
    stage='codebook_refinement',
    async_mode=True,
    api_config=api_config
)

# Line 150 (sync)
client = create_instructor_client(
    stage='codebook_refinement',
    async_mode=False,
    api_config=api_config
)
```

#### Testing Checklist

- [ ] Async refinement works
- [ ] Sync refinement works
- [ ] Both modes produce same results

---

## 9. Testing Strategy

### 9.1 Test Hierarchy

```
Level 1: Unit Tests (test_azure_*.py)
├── Connection tests
├── API compatibility tests
└── Client factory tests

Level 2: Integration Tests
├── Standalone pipeline tests
└── App-orchestrated tests

Level 3: End-to-End Tests
├── Full pipeline with Azure
└── Cache behavior validation
```

### 9.2 Required Test Files

See `tests/` directory for:

1. `test_azure_connection.py` - Azure connectivity
2. `test_responses_api_basic.py` - Basic responses API
3. `test_responses_api_pydantic.py` - Complex Pydantic models
4. `test_gpt5_reasoning.py` - GPT-5 specific features
5. `test_client_factories.py` - config.py factories
6. `test_dual_execution_paths.py` - Standalone vs app modes
7. `test_migration_compatibility.py` - Actual utility patterns

### 9.3 Test Execution Order

```bash
# Step 1: Verify Azure connection
pytest tests/test_azure_connection.py -v

# Step 2: Verify responses API basics
pytest tests/test_responses_api_basic.py -v

# Step 3: Verify Pydantic compatibility
pytest tests/test_responses_api_pydantic.py -v

# Step 4: Verify GPT-5 reasoning
pytest tests/test_gpt5_reasoning.py -v

# Step 5: Verify client factories
pytest tests/test_client_factories.py -v

# Step 6: Verify both execution paths
pytest tests/test_dual_execution_paths.py -v

# Step 7: Verify migration patterns
pytest tests/test_migration_compatibility.py -v

# All tests must pass before migration!
pytest tests/test_azure_*.py -v
```

### 9.4 Success Criteria

**All tests MUST pass before proceeding with migration:**

- ✅ Azure managed identity authentication works
- ✅ Responses API creates structured outputs correctly
- ✅ Complex Pydantic models validated properly
- ✅ GPT-5 reasoning parameters accepted
- ✅ Client factories create correct clients
- ✅ Both execution paths work identically
- ✅ Token counting accurate
- ✅ Rate limiting functional

**If ANY test fails:**
1. Do NOT proceed with migration
2. Debug the failing test
3. Update documentation with findings
4. Re-run all tests

---

## 10. Rollback Plan

### 10.1 Environment Variable Fallback

**Keep OpenAI as fallback option:**

```bash
# Force use of OpenAI instead of Azure
unset AZURE_OPENAI_ENDPOINT

# Or explicit override
export FORCE_PROVIDER="openai"
```

### 10.2 Dual-Mode Operation

**During transition, support both APIs:**

```python
# In config.py
USE_RESPONSES_API = os.getenv("USE_RESPONSES_API", "true").lower() == "true"

if USE_RESPONSES_API:
    # New: responses.create
    response = await client.responses.create(input=prompt, ...)
else:
    # Old: chat.completions.create
    response = await client.chat.completions.create(messages=[...], ...)
```

### 10.3 Git Workflow

**Before starting migration:**

```bash
# Create feature branch
git checkout -b feature/azure-responses-migration

# Commit after each module migration
git add src/utils/qualityFilter.py
git commit -m "refactor: migrate qualityFilter to responses API"

# If rollback needed
git revert HEAD
# Or full branch abandon
git checkout main
git branch -D feature/azure-responses-migration
```

### 10.4 Backup Strategy

**Before any changes:**

```bash
# Backup entire utils directory
cp -r src/utils src/utils_backup_$(date +%Y%m%d)

# Or use git tags
git tag pre-migration-backup
```

---

## 11. Specialized Agent Instructions

### 11.1 Agent Activation Conditions

**The specialized migration agent should ONLY activate when:**

1. ✅ User explicitly requests migration help
2. ✅ All prerequisite tests have passed
3. ✅ Azure environment is configured
4. ✅ This documentation has been read

**Agent should REFUSE to help if:**

- ❌ Any test is failing
- ❌ Azure not configured
- ❌ User hasn't confirmed readiness

### 11.2 Agent Capabilities

**The agent can:**

1. **Read and reference this documentation**
   - Quote relevant sections
   - Provide code examples
   - Explain architecture decisions

2. **Identify files needing migration**
   - Search for `chat.completions.create` patterns
   - Find client instantiation code
   - Locate parameter usage

3. **Suggest refactorings**
   - Show current vs target code
   - Explain changes needed
   - Provide migration steps

4. **Validate changes**
   - Check against documentation
   - Ensure both execution paths considered
   - Verify test coverage

5. **Track progress**
   - Mark completed modules
   - Suggest next module to migrate
   - Identify blockers

**The agent should NOT:**

- Make changes without showing current vs target code
- Skip testing steps
- Proceed if tests failing
- Ignore dual execution path requirement

### 11.3 Agent Workflow

```
User Request
    ↓
Check Prerequisites (tests passed?)
    ↓
Read Migration Guide
    ↓
Identify Module to Migrate
    ↓
Show Current Code vs Target Code
    ↓
Get User Approval
    ↓
Make Changes
    ↓
Suggest Testing Steps
    ↓
Move to Next Module
```

### 11.4 Agent Prompt Template

```markdown
You are a specialized Azure OpenAI migration agent. Your purpose is to help migrate
CoderingsTool from OpenAI chat.completions API to Azure OpenAI responses API.

## Prerequisites Check

Before helping, verify:
1. All tests in tests/test_azure_*.py have passed
2. AZURE_OPENAI_ENDPOINT environment variable is set
3. User has confirmed they are ready to migrate

If any prerequisite fails, explain what's needed and refuse to proceed.

## Migration Approach

1. Reference docs/azure_responses_migration_guide.md for all decisions
2. Follow the module migration order specified in Section 8
3. For each module:
   - Show current code
   - Show target code
   - Explain changes
   - Get user approval
   - Make changes
   - Suggest testing steps

## Safety Rules

- Never proceed if tests are failing
- Always show code diffs before making changes
- Test both standalone and app-orchestrated modes
- Keep backups via git commits
- Provide rollback instructions if issues occur

## Communication Style

- Be explicit about what you're changing and why
- Reference section numbers from the migration guide
- Show code examples from the guide
- Ask for confirmation before major changes
```

---

## Appendix A: Quick Reference

### Environment Variables Needed

```bash
# Azure configuration
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-key"  # Optional, for testing

# Deployment mappings
export AZURE_GPT41_MINI_DEPLOYMENT="my-gpt41-mini"
export AZURE_GPT5_MINI_DEPLOYMENT="my-gpt5-mini"
export AZURE_EMBEDDING_DEPLOYMENT="my-embedding"
```

### Key Code Patterns

```python
# Create client
from config import create_instructor_client, DEFAULT_API_PROVIDER_CONFIG

client = create_instructor_client(
    stage='quality_filter',
    async_mode=True
)

# Make API call
response = await client.responses.create(
    input=prompt,
    response_model=MyModel,
    max_output_tokens=4000
)
```

### Testing Commands

```bash
# Run all tests
pytest tests/test_azure_*.py -v

# Run specific test
pytest tests/test_azure_connection.py::test_azure_managed_identity -v
```

---

## Appendix B: Troubleshooting

### Issue: "No module named 'azure.identity'"

```bash
pip install azure-identity>=1.15.0
```

### Issue: "DefaultAzureCredential failed to retrieve a token"

1. Check managed identity is assigned to resource
2. Verify RBAC role "Cognitive Services OpenAI User" is assigned
3. Try Azure CLI login: `az login`

### Issue: "Deployment not found"

Check deployment name mapping in environment variables:
```bash
export AZURE_GPT41_MINI_DEPLOYMENT="actual-deployment-name"
```

### Issue: Tests failing with "responses API not available"

Verify API version is 2025-08-01 or later:
```python
api_version="2025-08-01"
```

---

## Document Version History

- **v1.0** (2025-01-19): Initial comprehensive migration guide

---

**END OF MIGRATION GUIDE**
