#%%
#
"""
Debug script for Taxonomy Classifier prompts (P1-P6 + P5b): Full LLM Request Inspector
Shows exactly what the LLM receives: prompt text + instructor-generated Pydantic schemas.

Usage:
    cd src && python -m pipeline.step_4_classifier.debug_taxonomy_prompts
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

# Import response models
from pipeline.step_4_classifier.prompts_classifier import (
    FacetDiscoveryResult,
    FacetConsolidatedResponse,
    FacetAssignmentResult,
    AttributeAssignmentResult,
    AttributeDiscoveryResult,
    AttributeChunkConsolidatedResponse,
    InFacetConsolidatedResponse,
)


# Configuration (from centralized test_data.py)
FILENAME = TEST_DATA.filename
VAR_NAME = TEST_DATA.var_name
SAMPLE_SIZE = TEST_DATA.sample_size


# =============================================================================
# Prompt Type → Response Model Mapping
# =============================================================================

STATIC_PROMPT_MODELS = {
    "facet_discovery": FacetDiscoveryResult,
    "facet_consolidation": FacetConsolidatedResponse,
    "facet_assignment": FacetAssignmentResult,
    "attribute_assignment": AttributeAssignmentResult,
    "attribute_discovery": AttributeDiscoveryResult,
    "attribute_chunk_consolidation": AttributeChunkConsolidatedResponse,
    "in_facet_consolidation": InFacetConsolidatedResponse,
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

def get_prompts_files() -> list[Path]:
    """Get taxonomy prompts file path(s)."""
    variable_key = generate_enhanced_variable_key([VAR_NAME], False, SAMPLE_SIZE)
    prompts_dir = project_root / "exports" / "prompts"
    base = f"step4_classifier_{variable_key}"

    files = []
    f = prompts_dir / f"{base}_taxonomy.json"
    if f.exists():
        files.append(f)
    return files


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
    print(f"Step:           {prompt_entry.get('step_name', 'unknown')}")
    if "model" in metadata:
        print(f"Model:          {metadata['model']}")
    if "temperature" in metadata:
        print(f"Temperature:    {metadata['temperature']}")
    if "max_tokens" in metadata:
        print(f"Max tokens:     {metadata['max_tokens']}")
    if "language" in metadata:
        print(f"Language:       {metadata['language']}")
    if "partition_name" in metadata:
        print(f"Partition:      {metadata['partition_name']}")
    if "dimension_name" in metadata:
        print(f"Dimension:      {metadata['dimension_name']}")
    if "batch_number" in metadata:
        print(f"Batch:          {metadata['batch_number']} / {metadata.get('total_batches', '?')}")
    if "n_batches" in metadata:
        print(f"N batches:      {metadata['n_batches']}")
    if "n_facets" in metadata:
        print(f"N facets:       {metadata['n_facets']}")
    if "facet_name" in metadata:
        print(f"Facet:          {metadata['facet_name']}")
    if "n_observations" in metadata:
        print(f"N observations: {metadata['n_observations']}")
    if "n_domains" in metadata:
        print(f"N domains:      {metadata['n_domains']}")
    if "n_total_attributes" in metadata:
        print(f"N total attrs:  {metadata['n_total_attributes']}")
    if "n_categories" in metadata:
        print(f"N categories:   {metadata['n_categories']}")
    if "n_labels" in metadata:
        print(f"N labels:       {metadata['n_labels']}")

    # Other metadata
    other_meta = {k: v for k, v in metadata.items()
                  if k not in ("model", "temperature", "max_tokens",
                               "language", "partition_name",
                               "dimension_name",
                               "batch_number", "total_batches",
                               "n_batches", "n_clusters", "n_codes",
                               "n_domains", "n_total_codes",
                               "n_categories", "n_labels",
                               "survey_question")}
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
            schema = openai_schema(model).openai_schema
            schema_str = json.dumps(schema, indent=2)
            if is_list:
                print(f"  NOTE: Instructor receives List[{model.__name__}] - wraps this schema in an array.")
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
    prompts_files = get_prompts_files()

    print("=" * 100)
    print("DEBUG: Taxonomy Classifier Prompt Inspector (P1-P6 + P5b)")
    print("Shows prompt text + Pydantic response model schemas (as seen by instructor)")
    print("=" * 100)
    print(f"Variable:     {VAR_NAME}")
    print(f"Sample size:  {SAMPLE_SIZE}")
    print(f"Prompts files: {[f.name for f in prompts_files] if prompts_files else '(none found)'}")
    print("=" * 100)

    if not prompts_files:
        print("\nNo prompts files found.")
        print("\nTo generate prompts, run:")
        print("  cd src && python -m pipeline.step_4_classifier.run_classifier")
        return

    # Merge prompts from all files
    all_prompts = []
    for pf in prompts_files:
        data = load_prompts(pf)
        if data is None:
            print(f"Failed to load: {pf.name}")
            continue
        print(f"\n[{pf.name}]")
        print(f"  Session ID:    {data.get('session_id', 'unknown')}")
        print(f"  Capture time:  {data.get('capture_time', 'unknown')}")
        print(f"  Total prompts: {data.get('total_prompts', 0)}")
        summary = data.get("summary", {})
        if summary:
            for ptype, count in summary.get("by_utility", {}).items():
                print(f"    {ptype}: {count}")
        all_prompts.extend(data.get("prompts", []))

    prompts = all_prompts
    if not prompts:
        print("\nNo prompts found in files.")
        return

    # Group by prompt_type, keeping first instance of each
    by_type = {}
    for entry in prompts:
        ptype = entry.get("prompt_type", "unknown")
        if ptype not in by_type:
            by_type[ptype] = entry

    print(f"\n{'#'*100}")
    print(f"# PROMPT DISPLAY ({len(prompts)} total prompts, "
          f"{len(by_type)} unique types)")
    print(f"{'#'*100}")
    for ptype in by_type:
        count = sum(1 for p in prompts if p.get("prompt_type") == ptype)
        print(f"  {ptype}: {count} instance(s)")

    for i, (ptype, entry) in enumerate(by_type.items(), 1):
        print(f"\n{'#'*100}")
        print(f"# {ptype} (1 of "
              f"{sum(1 for p in prompts if p.get('prompt_type') == ptype)} instances)")
        print(f"{'#'*100}")
        print_full_prompt(entry, i, len(by_type))


if __name__ == "__main__":
    main()
