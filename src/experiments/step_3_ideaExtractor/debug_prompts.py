#%%
#
"""
Debug script for Step 3: Idea Extractor Prompts
Loads and displays prompts saved from a previous run_experiment execution.

Usage:
    cd src && python -m experiments.step_3_ideaExtractor.debug_prompts
"""

import sys
import json
from pathlib import Path
src_dir = Path(__file__).parent.parent.parent
project_root = src_dir.parent
sys.path.insert(0, str(src_dir))

from utils.cacheManager import generate_enhanced_variable_key

# Import centralized test data config
try:
    from experiments.test_data import TEST_DATA
except ImportError:
    exp_root = Path(__file__).parent.parent
    if str(exp_root) not in sys.path:
        sys.path.insert(0, str(exp_root))
    from test_data import TEST_DATA

# Configuration (from centralized test_data.py)
FILENAME = TEST_DATA.filename
VAR_NAME = TEST_DATA.var_name
SAMPLE_SIZE = TEST_DATA.sample_size


def get_prompts_file() -> Path:
    """Get the prompts file path for current config."""
    variable_key = generate_enhanced_variable_key([VAR_NAME], False, SAMPLE_SIZE)
    prompts_dir = project_root / "exports" / "prompts"
    prompts_file = prompts_dir / f"step3_{VAR_NAME}_{variable_key}.json"
    return prompts_file


def load_prompts(filepath: Path) -> dict:
    """Load prompts from JSON file."""
    if not filepath.exists():
        return None
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def print_prompt(prompt_entry: dict, index: int, total: int) -> None:
    """Print a single prompt entry with formatting."""
    print(f"\n{'='*80}")
    print(f"PROMPT {index}/{total}: {prompt_entry.get('prompt_type', 'unknown')}")
    print(f"{'='*80}")
    print(f"Step:      {prompt_entry.get('step_name', 'unknown')}")
    print(f"Utility:   {prompt_entry.get('utility_name', 'unknown')}")
    print(f"Timestamp: {prompt_entry.get('timestamp', 'unknown')}")

    # Print metadata if available
    metadata = prompt_entry.get('metadata', {})
    if metadata:
        print(f"\n[Metadata]")
        for key, value in metadata.items():
            print(f"  {key}: {value}")

    # Print prompt content
    print(f"\n[Prompt Content]")
    print("-" * 80)
    print(prompt_entry.get('prompt_content', '(no content)'))
    print("-" * 80)


def main():
    prompts_file = get_prompts_file()

    print("=" * 70)
    print("DEBUG: Step 3 - Idea Extractor Prompts")
    print("=" * 70)
    print(f"Variable: {VAR_NAME}")
    print(f"Sample size: {SAMPLE_SIZE}")
    print(f"Prompts file: {prompts_file}")
    print("=" * 70)

    if not prompts_file.exists():
        print(f"\nNo prompts file found at: {prompts_file}")
        print("\nTo generate prompts, run:")
        print("  cd src && python -m experiments.step_3_ideaExtractor.run_experiment")
        print("\nMake sure PRINT_PROMPTS = True in run_experiment.py")
        return

    data = load_prompts(prompts_file)

    if data is None:
        print("Failed to load prompts file.")
        return

    # Print summary
    print(f"\nSession ID: {data.get('session_id', 'unknown')}")
    print(f"Capture time: {data.get('capture_time', 'unknown')}")
    print(f"Total prompts: {data.get('total_prompts', 0)}")

    summary = data.get('summary', {})
    if summary:
        print("\n[Prompts by Step]")
        for step, count in summary.get('by_step', {}).items():
            print(f"  {step}: {count}")

        print("\n[Prompts by Type]")
        for ptype, count in summary.get('by_utility', {}).items():
            print(f"  {ptype}: {count}")

    # Print each prompt
    prompts = data.get('prompts', [])
    if prompts:
        print(f"\n{'#'*80}")
        print(f"# CAPTURED PROMPTS ({len(prompts)} total)")
        print(f"{'#'*80}")

        for i, prompt_entry in enumerate(prompts, 1):
            print_prompt(prompt_entry, i, len(prompts))
    else:
        print("\nNo prompts found in file.")


if __name__ == "__main__":
    main()
