#%%
#
"""
Debug script for Single-Idea Dual Assignment prompts.

Loads captured assignment prompts from the JSON file saved during
the assignment run, and displays them with response model schema.

Usage:
    cd src && python -m development.step_4_classNcoder.debug_assignment_prompt
"""

import sys
import json
from pathlib import Path
from typing import Optional

src_dir = Path(__file__).parent.parent.parent
project_root = src_dir.parent
sys.path.insert(0, str(src_dir))

from utils.cacheManager import generate_enhanced_variable_key

# Import centralized test data config
try:
    from development.test_data import TEST_DATA
except ImportError:
    exp_root = Path(__file__).parent.parent
    if str(exp_root) not in sys.path:
        sys.path.insert(0, str(exp_root))
    from test_data import TEST_DATA

from development.step_4_classNcoder.prompts_exp import CodeAttributeAssignment

# Configuration
FILENAME = TEST_DATA.filename
VAR_NAME = TEST_DATA.var_name
SAMPLE_SIZE = TEST_DATA.sample_size

MAX_PROMPTS = 5  # How many prompts to display (None = all)


# =============================================================================
# FILE LOADING
# =============================================================================

def get_assignment_prompts_file() -> Path:
    """Get the captured assignment prompts JSON file path."""
    variable_key = generate_enhanced_variable_key([VAR_NAME], False, SAMPLE_SIZE)
    prompts_dir = project_root / "exports" / "prompts"
    return prompts_dir / f"step4_classNcoder_{variable_key}_assignment.json"


def load_prompts(filepath: Path) -> Optional[dict]:
    """Load prompts from JSON file."""
    if not filepath.exists():
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


# =============================================================================
# DISPLAY
# =============================================================================

def print_prompt(prompt_entry: dict, index: int, total: int):
    """Display one captured assignment prompt."""
    prompt_type = prompt_entry.get("prompt_type", "unknown")
    metadata = prompt_entry.get("metadata", {})

    print(f"\n{'='*100}")
    print(f"ASSIGNMENT PROMPT {index}/{total}: {prompt_type}")
    print(f"{'='*100}")

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
    if "n_codes" in metadata:
        print(f"N codes:        {metadata['n_codes']}")

    # Full prompt
    print(f"\n[Full Prompt]")
    print("-" * 100)
    print(prompt_entry.get("prompt_content", "(no content)"))
    print("-" * 100)

    # Stats
    content = prompt_entry.get("prompt_content", "")
    print(f"\n[Stats]")
    print(f"  Prompt: {len(content):,} chars (~{len(content) // 4:,} tokens)")


def print_response_schema():
    """Display the Pydantic response model schema."""
    print(f"\n{'='*100}")
    print(f"RESPONSE MODEL: CodeAttributeAssignment")
    print(f"{'='*100}")
    schema = CodeAttributeAssignment.model_json_schema()
    schema_str = json.dumps(schema, indent=2)
    print(schema_str)
    print(f"\n  Schema: {len(schema_str):,} chars (~{len(schema_str) // 4:,} tokens)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 100)
    print("DEBUG: Assignment Prompt Inspector (from captured prompts)")
    print("Shows the actual prompts sent to the LLM during assignment")
    print("=" * 100)
    print(f"Variable:     {VAR_NAME}")
    print(f"Sample size:  {SAMPLE_SIZE}")
    print(f"Max prompts:  {MAX_PROMPTS or 'all'}")
    print("=" * 100)

    prompts_file = get_assignment_prompts_file()
    print(f"\nPrompts file: {prompts_file.name}")

    data = load_prompts(prompts_file)
    if data is None:
        print(f"\nERROR: No captured assignment prompts found at:")
        print(f"  {prompts_file}")
        print("\nRun assignment first:")
        print("  cd src && python -m development.step_4_classNcoder.run_experiment")
        print("  (with RUN_MODE = 'assignment' or 'all')")
        return

    print(f"Session ID:    {data.get('session_id', 'unknown')}")
    print(f"Capture time:  {data.get('capture_time', 'unknown')}")
    print(f"Total prompts: {data.get('total_prompts', 0)}")

    prompts = data.get("prompts", [])
    if not prompts:
        print("\nNo prompts found in file.")
        return

    # Limit display count
    show_prompts = prompts[:MAX_PROMPTS] if MAX_PROMPTS else prompts
    total = len(prompts)

    print(f"\nShowing {len(show_prompts)} of {total} captured assignment prompts")

    for i, entry in enumerate(show_prompts, 1):
        print_prompt(entry, i, len(show_prompts))

    # Response model schema (once)
    print_response_schema()

    print(f"\n{'='*100}")
    print("Done.")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()

# %%
