# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CoderingsTool is an AI-powered qualitative data analysis tool that automates the coding of open-ended survey responses. It uses LLMs (GPT-4.1/GPT-5) to process text responses through a multi-stage pipeline: preprocessing, quality filtering, idea extraction, embedding generation, clustering, codebook generation, refinement, and code assignment.

The tool supports both OpenAI and Azure OpenAI APIs and provides a bilingual (Dutch/English) Streamlit web interface alongside standalone pipeline execution.

## Environment Setup

This project uses a per-project virtual environment managed with `uv`:

```bash
# Setup and activate environment (run from project root)
chmod +x setup.sh
source ./setup.sh
```

The setup script creates/updates `.venv`, installs dependencies from [requirements.txt](requirements.txt), and activates the environment.

## Running the Application

### Streamlit Web Interface

```bash
# Activate environment first
source ./setup.sh

# Run the Streamlit app
streamlit run src/app.py
```

The Streamlit app ([src/app.py](src/app.py)) provides an interactive UI with step-by-step navigation through the pipeline.

### Standalone Pipeline Execution

Edit the configuration variables at the top of [src/pipeline.py](src/pipeline.py):

```python
filename = "your_data_file.sav"
id_column = "respondentid"
var_name = "Q1"
sample_size = 50  # or None for full dataset
RUN_UNTIL_STEP = 8  # Stop at specific step (0-9)
FORCE_RECALCULATE_ALL = False
```

Then run:

```bash
# From src directory
cd src
python pipeline.py
```

## Architecture

### Core Pipeline Structure

The pipeline is organized into 10 sequential steps in [src/pipeline.py](src/pipeline.py):

- **Step 0** (`step_0_load_data`): Load data from SPSS files (.sav)
- **Step 1** (`step_1_preprocess`): Spell checking using Hunspell + LLM correction
- **Step 2** (`step_2_quality_filter`): Filter out gibberish, empty, and "don't know" responses
- **Step 3** (`step_3_extract_ideas`): Extract individual ideas/concepts from responses
- **Step 4** (`step_4_generate_embeddings`): Generate text embeddings using OpenAI models
- **Step 5** (`step_5_cluster`): Cluster ideas using UMAP + HDBSCAN
- **Step 6** (`step_6_generate_codebook`): Generate initial codebook from clusters
- **Step 7** (`step_7_refine_codebook`): Refine codebook with LLM reasoning
- **Step 8** (`step_8_assign_codes`): Assign codes to all ideas
- **Step 9** (`step_9_export_results`): Export results to Excel

Each step takes the output from the previous step, applies transformations, and returns results as Pydantic models defined in [src/models.py](src/models.py).

### Data Flow & Pydantic Models

Data flows through progressively enriched Pydantic models ([src/models.py](src/models.py)):

```
ResponseModel → PreprocessedModel → QualityFilteredModel → IdeasExtractedModel
→ EmbeddingsModel → ClusterModel → CodeAssignedModel
```

Each model extends the previous, adding new fields as data progresses through the pipeline.

### LLM Integration

All LLM calls go through the centralized [src/utils/llm.py](src/utils/llm.py) module:

- **Dual Provider Support**: Switches between OpenAI and Azure OpenAI via `API_PROVIDER` setting in [src/config.py](src/config.py)
- **OpenAI**: Uses Responses API (`responses.create()`) with `input=` parameter
- **Azure**: Uses Chat Completions API (`chat.completions.create()`) with `messages=` parameter
- **Structured Outputs**: All calls use `instructor` library with Pydantic response models
- **Token Tracking**: Global `token_tracker` monitors usage and costs across all LLM calls

Main functions:
- `create_client(model, async_mode)` - Create instructor-wrapped client
- `llm_create_async()` - Async LLM call with structured output
- `get_model_limits(model)` - Get context window/max output for a model

### Configuration System

[src/config.py](src/config.py) centralizes all configuration via dataclasses:

- **API Configuration**: `API_PROVIDER`, OpenAI/Azure credentials, deployment names
- **Model Configuration**: `ModelConfig` - model selection, rate limits, model types (chat vs reasoning)
- **Processing Configuration**: Step-specific configs (SpellCheckConfig, QualityFilterConfig, EmbeddingConfig, HDBSCANConfig, etc.)
- **Hunspell Configuration**: Cross-platform paths for spell checking dictionaries
- **Language Configuration**: Multilingual labels (Dutch, English, German, French, Spanish)

The config is loaded from environment variables via `.env` file at project root.

### Caching System

[src/utils/cacheManager.py](src/utils/cacheManager.py) implements SQLite-backed caching:

- **Cache Keys**: Generated from `(filename, step_name, variable_key)` tuple
- **Variable Keys**: Handle single variables (`Q18`) and merged variables (`Q18+Q19+Q20_concat_semicolon_skip_250`)
- **Binary Storage**: Pickled Pydantic models stored in SQLite BLOBs
- **Invalidation**: Cascade invalidation - updating a step invalidates all downstream steps
- **Concurrency**: Thread-safe with context managers and connection pooling

Key functions:
- `save_to_cache()` - Save step results
- `load_from_cache()` - Load cached results
- `invalidate_cache()` - Force recalculation
- `generate_enhanced_variable_key()` - Generate cache keys with merge config

### Utility Modules

Key utilities in [src/utils/](src/utils/):

- **[dataLoader.py](src/utils/dataLoader.py)**: SPSS file loading, variable merging, sampling
- **[spellChecker.py](src/utils/spellChecker.py)**: Hunspell + LLM spell correction with parallel processing
- **[qualityFilter.py](src/utils/qualityFilter.py)**: Filter low-quality responses (gibberish, empty, "don't know")
- **[ideaExtractor.py](src/utils/ideaExtractor.py)**: Extract individual ideas from responses with LLM
- **[embedder.py](src/utils/embedder.py)**: Generate embeddings via OpenAI/Azure embedding models
- **[clusterer.py](src/utils/clusterer.py)**: UMAP + HDBSCAN clustering with parallel UMAP processing
- **[codeGenerator.py](src/utils/codeGenerator.py)**: Multi-phase codebook generation with batching strategies
- **[codebookRefinement.py](src/utils/codebookRefinement.py)**: LLM-based codebook refinement with reasoning models
- **[codeAssigner.py](src/utils/codeAssigner.py)**: Assign codes to ideas with parallel processing
- **[resultsExporter.py](src/utils/resultsExporter.py)**: Export to Excel with formatted sheets

### Streamlit App Architecture

[src/app.py](src/app.py) implements a step-based navigation system:

- **Session State Management**: Stores configuration, results, and progress in `st.session_state`
- **Step Pages**: Each pipeline step has a dedicated UI page function (`render_step_0_page()`, etc.)
- **Configuration Management**: `DatasetConfig` dataclass manages dataset configuration with validation
- **Bilingual UI**: All text defined in [src/ui_text.py](src/ui_text.py) with Dutch/English support
- **Cache Recovery**: Automatic cache corruption detection and recovery
- **Progress Tracking**: Visual progress indicators with completed steps, current step, and max step reached

### Prompts System

[src/prompts.py](src/prompts.py) contains all LLM prompt templates:

- **Template Variables**: Prompts use `{variable}` placeholders for runtime substitution
- **Multi-step Prompts**: Some utilities (e.g., codeGenerator) use multi-phase prompt chains
- **Language Support**: Prompts include language-specific instructions
- **Structured Output**: All prompts designed for Pydantic model outputs via instructor

## API Provider Configuration

The tool supports dual providers via the `API_PROVIDER` setting in [src/config.py](src/config.py):

```python
API_PROVIDER = "azure"  # or "openai"
```

### OpenAI Configuration

Set in `.env`:
```
OPENAI_API_KEY=your_key_here
```

### Azure OpenAI Configuration

Set in `.env`:
```
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your_key_here
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4.1
AZURE_OPENAI_DEPLOYMENT_NAME_EMBEDDING=text-embedding-3-large
AZURE_OPENAI_DEPLOYMENT_NAME_CODEDESIGNER=gpt-4.1

# Optional: For dynamic limit fetching via ARM API
AZURE_SUBSCRIPTION_ID=your_subscription_id
AZURE_RESOURCE_GROUP=your_resource_group
```

## Model Selection

Models are configured in [src/config.py](src/config.py) `ModelConfig`:

- **Reasoning Models** (e.g., gpt-5, o1): No temperature parameter, used for complex reasoning tasks
- **Chat Models** (e.g., gpt-4.1, gpt-5-chat): Support temperature, used for most tasks
- **Model Type Detection**: `ModelConfig.MODEL_TYPES` dict maps model names to types
- **Rate Limits**: `OPENAI_RATE_LIMITS` defines tokens/min, requests/min, tokens/day per model

When modifying LLM calls, check if the model is a reasoning model using `_is_reasoning_model()` in [src/utils/llm.py](src/utils/llm.py).

## Data Directory Structure

```
data/               # SPSS files (.sav) - gitignored
exports/            # Excel exports - gitignored
hunspell/          # Spell check dictionaries (nl_NL, en_GB)
  dict/
    nl_NL/
    en_GB/
  hunspell.exe     # Windows only
src/
  utils/           # Pipeline utilities
  backup/          # Version history
  app.py           # Streamlit web UI
  pipeline.py      # Standalone pipeline
  config.py        # Configuration
  models.py        # Pydantic models
  prompts.py       # LLM prompts
  ui_text.py       # Bilingual UI text
```

## Important Development Notes

### When Modifying Utility Files

1. **Check Provider Compatibility**: Ensure changes work with both OpenAI and Azure providers
2. **Use Centralized LLM Module**: All LLM calls MUST go through [src/utils/llm.py](src/utils/llm.py)
3. **Update Cache Keys**: If changing input parameters, update cache key generation to avoid stale caches
4. **Maintain Pydantic Models**: Keep model chain intact (each model extends previous)
5. **Test Both Modes**: Test changes in both Streamlit app and standalone pipeline

### When Adding New Pipeline Steps

1. Add step function to [src/pipeline.py](src/pipeline.py) following `step_N_name()` pattern
2. Update `STEP_NAMES` dict in both [src/pipeline.py](src/pipeline.py) and [src/ui_text.py](src/ui_text.py)
3. Add corresponding Pydantic model to [src/models.py](src/models.py)
4. Create step page function in [src/app.py](src/app.py): `render_step_N_page()`
5. Add prompts to [src/prompts.py](src/prompts.py) if using LLM

### When Modifying Configuration

1. Update dataclass in [src/config.py](src/config.py)
2. Update session state initialization in [src/app.py](src/app.py) if needed
3. Update any UI elements that reference the config
4. Consider cache invalidation if config changes affect results

### Cross-Platform Compatibility

The codebase supports macOS, Linux, and Windows:

- **Hunspell**: Cross-platform paths in [src/config.py](src/config.py) (`_get_hunspell_paths()`)
- **CPU Detection**: Uses `os.cpu_count()` and `multiprocessing.cpu_count()` (both cross-platform)
- **Path Handling**: Uses `pathlib.Path` for cross-platform path operations

## Common Gotchas

1. **Cache Invalidation**: When changing step logic, you may need to manually delete cache or set `FORCE_RECALCULATE_ALL = True`
2. **Session State vs Config**: Streamlit app uses `st.session_state` for runtime state; [src/config.py](src/config.py) for persistent settings
3. **Async vs Sync**: Most utilities use async LLM calls; ensure proper `await` and event loop handling
4. **Model Limits**: Check model context windows before batching; use `get_model_limits()` from [src/utils/llm.py](src/utils/llm.py)
5. **Provider Differences**: OpenAI uses Responses API; Azure uses Chat Completions API - differences abstracted in [src/utils/llm.py](src/utils/llm.py)
6. **Widget Key Conflicts**: Streamlit app uses `_config` suffix for storage keys to avoid widget conflicts (see `DatasetConfig` in [src/app.py](src/app.py))

## Testing

There is no automated test suite. Test manually:

1. **Standalone Pipeline**: Run [src/pipeline.py](src/pipeline.py) with a small sample size
2. **Streamlit App**: Run `streamlit run src/app.py` and walk through steps
3. **Provider Switching**: Test with both `API_PROVIDER = "openai"` and `API_PROVIDER = "azure"`
4. **Cache Behavior**: Test with fresh cache and with existing cache

## Key Dependencies

- **streamlit**: Web UI framework
- **pandas**: Data manipulation
- **pyreadstat**: SPSS file reading
- **openai**: OpenAI API client (used for both OpenAI and Azure)
- **instructor**: Structured LLM outputs with Pydantic
- **scikit-learn**: Machine learning utilities
- **hdbscan**: Clustering algorithm
- **umap-learn**: Dimensionality reduction
- **tiktoken**: Token counting
- **spacy**: Advanced NLP (optional, with language models)
- **azure-identity, azure-mgmt-cognitiveservices**: Azure ARM API access (optional)

See [requirements.txt](requirements.txt) for full dependency list with version constraints.
