#%%
#
"""
Debug script for Code Generator prompts (P8-P9): Full LLM Request Inspector
Shows exactly what the LLM receives: prompt text + instructor-generated Pydantic schemas.

Usage:
    cd src && python -m steps.step_5_codeGenerator.debug_codebook_prompts
"""

import sys
import json
from pathlib import Path
from typing import Optional, Tuple, Type
from instructor.function_calls import openai_schema

src_dir = Path(__file__).parent.parent.parent
project_root = src_dir.parent
sys.path.insert(0, str(src_dir))

from utils.cacheManager import generate_enhanced_variable_key

# Import centralized test data config
from test_data import TEST_DATA

# Import response models (codebook only)
from pipeline.step_5_codeGenerator.prompts_codeGenerator import (
    CodeGenerationFromAttributesResult,
    CodebookConsolidationResult,
)


# Configuration (from centralized test_data.py)
FILENAME = TEST_DATA.filename
VAR_NAME = TEST_DATA.var_name
SAMPLE_SIZE = TEST_DATA.sample_size


# =============================================================================
# Prompt Type → Response Model Mapping
# =============================================================================

STATIC_PROMPT_MODELS = {
    "code_generation_from_attributes": CodeGenerationFromAttributesResult,
    "codebook_consolidation": CodebookConsolidationResult,
}


def resolve_response_model(prompt_entry: dict) -> Tuple[Optional[Type], bool, str]:
    """Resolve the Pydantic response model for a prompt entry."""
    prompt_type = prompt_entry.get("prompt_type", "")

    if prompt_type in STATIC_PROMPT_MODELS:
        model = STATIC_PROMPT_MODELS[prompt_type]
        return (model, False, f"Static model: {model.__name__}")

    return (None, False, f"Unknown prompt type: {prompt_type}")


# =============================================================================
# File Loading
# =============================================================================

def get_prompts_file() -> Path:
    """Get the codebook prompts JSON file path."""
    variable_key = generate_enhanced_variable_key([VAR_NAME], False, SAMPLE_SIZE)
    prompts_dir = project_root / "exports" / "prompts"
    return prompts_dir / f"step5_codeGenerator_{variable_key}_codebook.json"


def load_prompts(filepath: Path) -> Optional[dict]:
    """Load prompts from JSON file."""
    if not filepath.exists():
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


# =============================================================================
# Display
# =============================================================================

def print_full_prompt(prompt_entry: dict, index: int, total: int) -> None:
    """Print a single prompt entry with prompt text AND response model schema."""
    prompt_type = prompt_entry.get("prompt_type", "unknown")

    print(f"\n{'='*100}")
    print(f"PROMPT {index}/{total}: {prompt_type}")
    print(f"{'='*100}")

    metadata = prompt_entry.get("metadata", {})
    if "model" in metadata:
        print(f"Model:          {metadata['model']}")
    if "temperature" in metadata:
        print(f"Temperature:    {metadata['temperature']}")
    if "max_tokens" in metadata:
        print(f"Max tokens:     {metadata['max_tokens']}")
    if "language" in metadata:
        print(f"Language:       {metadata['language']}")
    if "n_domains" in metadata:
        print(f"N domains:      {metadata['n_domains']}")
    if "n_total_attributes" in metadata:
        print(f"N total attrs:  {metadata['n_total_attributes']}")
    if "n_raw_codes" in metadata:
        print(f"N raw codes:    {metadata['n_raw_codes']}")
    if "dimension_name" in metadata:
        print(f"Dimension:      {metadata['dimension_name']}")

    # Prompt content
    content = prompt_entry.get("prompt_content", "(no content)")
    print(f"\n[Prompt Content]")
    print("-" * 100)
    print(content)
    print("-" * 100)

    # Response model schema
    model, is_list, note = resolve_response_model(prompt_entry)

    print(f"\n[Response Model]")
    print(f"  {note}")

    schema_str = ""
    if model is not None:
        try:
            schema = openai_schema(model).openai_schema
            schema_str = json.dumps(schema, indent=2)
            print(f"\n[OpenAI Tool Definition (exact schema injected by instructor)]")
            print("-" * 100)
            print(schema_str)
            print("-" * 100)
        except Exception as e:
            print(f"  ERROR generating schema: {e}")
    else:
        print("  (no model available)")

    # Stats
    print(f"\n[Stats]")
    print(f"  Prompt: {len(content):,} chars (~{len(content) // 4:,} tokens)")
    if schema_str:
        print(f"  Schema: {len(schema_str):,} chars (~{len(schema_str) // 4:,} tokens)")


# =============================================================================
# Main
# =============================================================================

def main():
    prompts_file = get_prompts_file()

    print("=" * 100)
    print("DEBUG: Code Generator Prompt Inspector (P8-P9)")
    print("Shows prompt text + Pydantic response model schemas (as seen by instructor)")
    print("=" * 100)
    print(f"Variable:     {VAR_NAME}")
    print(f"Sample size:  {SAMPLE_SIZE}")
    print(f"Prompts file: {prompts_file.name}")
    print("=" * 100)

    data = load_prompts(prompts_file)
    if data is None:
        print(f"\nNo codebook prompts file found at: {prompts_file}")
        print("\nTo generate prompts, run:")
        print("  cd src && python -m steps.step_5_codeGenerator.run_codeGenerator")
        return

    print(f"\nSession ID:    {data.get('session_id', 'unknown')}")
    print(f"Capture time:  {data.get('capture_time', 'unknown')}")
    print(f"Total prompts: {data.get('total_prompts', 0)}")

    prompts = data.get("prompts", [])
    if not prompts:
        print("\nNo prompts found in file.")
        return

    print(f"\n{'#'*100}")
    print(f"# CODEBOOK PROMPTS ({len(prompts)} total)")
    print(f"{'#'*100}")

    for i, entry in enumerate(prompts, 1):
        print_full_prompt(entry, i, len(prompts))


if __name__ == "__main__":
    main()

# %%
