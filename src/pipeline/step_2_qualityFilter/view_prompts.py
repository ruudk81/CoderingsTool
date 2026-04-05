#%%

"""
Debug script for Quality Filter prompts: Full LLM Request Inspector
Shows exactly what the LLM receives: prompt text + instructor-generated Pydantic schema.

Usage:
    cd src && python -m steps.step_2_qualityFilter.debug_quality_prompts
"""

import sys
import json
from pathlib import Path
from typing import Optional, Type
from instructor.function_calls import openai_schema

src_dir = Path(__file__).parent.parent.parent
project_root = src_dir.parent
sys.path.insert(0, str(src_dir))

from utils.cacheManager import generate_enhanced_variable_key

from test_data import TEST_DATA

from pipeline.step_2_qualityFilter.prompts_qualityFilter import (
    GRADER_INSTRUCTIONS, QualityFilterLLMResponseExp,
)
from config import DEFAULT_LANGUAGE

# Configuration
FILENAME = TEST_DATA.filename
VAR_NAME = TEST_DATA.var_name
SAMPLE_SIZE = TEST_DATA.sample_size


# =============================================================================
# Build a sample prompt (what the LLM actually sees)
# =============================================================================

def build_sample_prompt(var_lab: str, response_text: str = "De sfeer was geweldig en de muziek was top.") -> str:
    """Build a sample prompt as the LLM would receive it."""
    return GRADER_INSTRUCTIONS.format(
        language=DEFAULT_LANGUAGE,
        var_lab=var_lab,
        response_text=response_text,
    )


# =============================================================================
# Display
# =============================================================================

def main():
    from utils import dataLoader
    data_dir = project_root / "data"

    print("=" * 100)
    print("DEBUG: Quality Filter Prompt Inspector")
    print("Shows prompt text + Pydantic response model schema (as seen by instructor)")
    print("=" * 100)
    print(f"Variable:     {VAR_NAME}")
    print(f"Sample size:  {SAMPLE_SIZE}")
    print("=" * 100)

    # Get variable label
    loader = dataLoader.DataLoader(data_dir=str(data_dir), verbose=False)
    var_lab = loader.get_varlab(filename=FILENAME, var_name=VAR_NAME)
    print(f"Question:     {var_lab}")

    # --- Sample prompts ---
    sample_responses = [
        ("Meaningful", "De sfeer was geweldig en de muziek was top."),
        ("Don't know", "Weet ik niet"),
        ("Gibberish", "asdfkj jjjjj"),
        ("Off-topic", "Ik werk bij de bank."),
        ("Empty", ""),
        ("Minimal", "Niets"),
    ]

    for label, response_text in sample_responses:
        prompt = build_sample_prompt(var_lab, response_text)

        print(f"\n{'='*100}")
        print(f"SAMPLE: {label}")
        print(f"Response: \"{response_text}\"")
        print(f"{'='*100}")

        print(f"\n[Prompt Content] ({len(prompt):,} chars, ~{len(prompt)//4:,} tokens)")
        print("-" * 100)
        print(prompt)
        print("-" * 100)

    # --- Response model schema (same for all calls) ---
    print(f"\n{'='*100}")
    print("RESPONSE MODEL (Pydantic schema injected by instructor)")
    print(f"{'='*100}")

    try:
        schema = openai_schema(QualityFilterLLMResponseExp).openai_schema
        schema_str = json.dumps(schema, indent=2)
        print(f"\n[OpenAI Tool Definition]")
        print("-" * 100)
        print(schema_str)
        print("-" * 100)
        print(f"\nSchema: {len(schema_str):,} chars (~{len(schema_str)//4:,} tokens)")
    except Exception as e:
        print(f"ERROR generating schema: {e}")

    # --- Also show saved prompts if available ---
    variable_key = generate_enhanced_variable_key([VAR_NAME], False, SAMPLE_SIZE)
    prompts_dir = project_root / "exports" / "prompts"
    prompts_file = prompts_dir / f"step2_{VAR_NAME}_{variable_key}.json"

    if prompts_file.exists():
        print(f"\n{'='*100}")
        print(f"SAVED PROMPTS: {prompts_file.name}")
        print(f"{'='*100}")
        with open(prompts_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"  Session ID:    {data.get('session_id', 'unknown')}")
        print(f"  Capture time:  {data.get('capture_time', 'unknown')}")
        print(f"  Total prompts: {data.get('total_prompts', 0)}")
    else:
        print(f"\nNo saved prompts found at: {prompts_file}")
        print("Run step 2 with prompt_printer_enabled=True to generate.")


if __name__ == "__main__":
    main()
