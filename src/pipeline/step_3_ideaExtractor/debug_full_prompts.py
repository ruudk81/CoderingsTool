#%%
#
"""
Debug script for Step 3: Full LLM Request Inspector
Shows exactly what the LLM receives: prompt text + instructor-generated Pydantic schemas.

For each captured prompt, displays:
  1. The filled prompt text (what goes into the 'input' / 'messages' parameter)
  2. The Pydantic response model's JSON schema (what instructor injects as tool definition)

Usage:
    cd src && python -m steps.step_3_ideaExtractor.debug_full_prompts
"""

import sys
import json
from pathlib import Path
from typing import Optional, Tuple, Type

src_dir = Path(__file__).parent.parent.parent
project_root = src_dir.parent
sys.path.insert(0, str(src_dir))

from utils.cacheManager import generate_enhanced_variable_key

# Import centralized test data config
try:
    from pipeline.test_data import TEST_DATA
except ImportError:
    exp_root = Path(__file__).parent.parent
    if str(exp_root) not in sys.path:
        sys.path.insert(0, str(exp_root))
    from test_data import TEST_DATA

try:
    from pipeline.step_3_ideaExtractor.prompts_ideaExtractor import (
        GenericSpecifierGroup1Response,
        GenericSpecifierGroup2Response,
        PrimaryDimensionChunkResponse,
        PrimaryDimensionConsolidatedResponse,
        DomainChunkResponse,
        DomainConsolidatedResponse,
        create_extraction_model,
    )
    from pipeline.step_3_ideaExtractor.dimension_data import get_dimension
except ImportError:
    from pipeline.step_3_ideaExtractor.prompts_ideaExtractor import (
        GenericSpecifierGroup1Response,
        GenericSpecifierGroup2Response,
        PrimaryDimensionChunkResponse,
        PrimaryDimensionConsolidatedResponse,
        DomainChunkResponse,
        DomainConsolidatedResponse,
        create_extraction_model,
    )
    from pipeline.step_3_ideaExtractor.dimension_data import get_dimension


# Configuration (from centralized test_data.py)
FILENAME = TEST_DATA.filename
VAR_NAME = TEST_DATA.var_name
SAMPLE_SIZE = TEST_DATA.sample_size


# =============================================================================
# Prompt Type → Response Model Mapping
# =============================================================================

STATIC_PROMPT_MODELS = {
    "context_specifier_group1": GenericSpecifierGroup1Response,
    "context_specifier_group2": GenericSpecifierGroup2Response,
    "consolidate_specifiers_group1": GenericSpecifierGroup1Response,
    "consolidate_specifiers_group2": GenericSpecifierGroup2Response,
    "dimension_chunk_decision_tree": PrimaryDimensionChunkResponse,
    "dimension_consolidation": PrimaryDimensionConsolidatedResponse,
    "domain_chunk": DomainChunkResponse,
    "domain_consolidation": DomainConsolidatedResponse,
}


def resolve_response_model(prompt_entry: dict) -> Tuple[Optional[Type], bool, str]:
    """Resolve the Pydantic response model for a prompt entry.

    Returns:
        (model_class, is_list, description_note)
        - model_class: The Pydantic model class
        - is_list: Whether the actual API call uses List[model_class]
        - description_note: Human-readable note about the model
    """
    prompt_type = prompt_entry.get("prompt_type", "")
    metadata = prompt_entry.get("metadata", {})

    # Static models
    if prompt_type in STATIC_PROMPT_MODELS:
        model = STATIC_PROMPT_MODELS[prompt_type]
        return (model, False, f"Static model: {model.__name__}")

    # Dynamic: idea_extraction
    if prompt_type in ("idea_extraction", "idea_extraction_v3"):
        dimension_key = metadata.get("primary_dimension", "ATTRIBUTES_ASSOCIATIONS")
        dimension = get_dimension(dimension_key)
        template_prefix = metadata.get("template_prefix", "")
        model = create_extraction_model(dimension=dimension, template_prefix=template_prefix)
        return (model, True, f"Dynamic model: List[DimensionTaxonomy_{dimension_key}] (dimension={dimension_key})")

    return (None, False, f"Unknown prompt type: {prompt_type}")


# =============================================================================
# File Loading (same pattern as debug_prompts.py)
# =============================================================================

def get_prompts_file() -> Path:
    """Get the prompts file path for current config."""
    variable_key = generate_enhanced_variable_key([VAR_NAME], False, SAMPLE_SIZE)
    prompts_dir = project_root / "exports" / "prompts"
    return prompts_dir / f"step3_{VAR_NAME}_{variable_key}.json"


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

    # Key metadata inline
    metadata = prompt_entry.get("metadata", {})
    print(f"Step:      {prompt_entry.get('step_name', 'unknown')}")
    if "model" in metadata:
        print(f"Model:     {metadata['model']}")
    if "language" in metadata:
        print(f"Language:  {metadata['language']}")
    if "primary_dimension" in metadata:
        print(f"Dimension: {metadata['primary_dimension']}")

    # Other metadata
    other_meta = {k: v for k, v in metadata.items()
                  if k not in ("model", "language", "primary_dimension")}
    if other_meta:
        print(f"\n[Other Metadata]")
        for key, value in other_meta.items():
            print(f"  {key}: {value}")

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
            schema = model.model_json_schema()
            schema_str = json.dumps(schema, indent=2)
            if is_list:
                print(f"  NOTE: Instructor receives List[{model.__name__}] - wraps this schema in an array.")
            print(f"\n[JSON Schema (what instructor injects as tool definition)]")
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
    print("DEBUG: Step 3 - Full LLM Request Inspector")
    print("Shows prompt text + Pydantic response model schemas (as seen by instructor)")
    print("=" * 100)
    print(f"Variable:     {VAR_NAME}")
    print(f"Sample size:  {SAMPLE_SIZE}")
    print(f"Prompts file: {prompts_file}")
    print("=" * 100)

    if not prompts_file.exists():
        print(f"\nNo prompts file found at: {prompts_file}")
        print("\nTo generate prompts, run:")
        print("  cd src && python -m steps.step_3_ideaExtractor.run_ideaExtractor")
        print("\nMake sure PRINT_PROMPTS = True in run file")
        return

    data = load_prompts(prompts_file)
    if data is None:
        print("Failed to load prompts file.")
        return

    # Summary
    print(f"\nSession ID:   {data.get('session_id', 'unknown')}")
    print(f"Capture time: {data.get('capture_time', 'unknown')}")
    print(f"Total prompts: {data.get('total_prompts', 0)}")

    summary = data.get("summary", {})
    if summary:
        print("\n[Prompts by Step]")
        for step, count in summary.get("by_step", {}).items():
            print(f"  {step}: {count}")
        print("\n[Prompts by Type]")
        for ptype, count in summary.get("by_utility", {}).items():
            print(f"  {ptype}: {count}")

    # Print each prompt with full schema
    prompts = data.get("prompts", [])
    if prompts:
        print(f"\n{'#'*100}")
        print(f"# FULL LLM REQUEST DETAILS ({len(prompts)} prompts)")
        print(f"{'#'*100}")

        for i, prompt_entry in enumerate(prompts, 1):
            print_full_prompt(prompt_entry, i, len(prompts))
    else:
        print("\nNo prompts found in file.")


if __name__ == "__main__":
    main()
