"""
Debug script for Step 7: Codebook Refinement - Prompt Display
Displays captured prompts from the refinement step.

Note: This requires running the refinement with prompt_printer_enabled=True
and accessing the prompt_printer object. For cached results, prompts may
not be available.

Usage:
    cd src && python -m development.step_7_codebookRefinement.debug_prompts

For full prompt inspection, run run_experiment.py with:
    prompt_printer_enabled=True
"""

import sys
from pathlib import Path
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

print("=" * 70)
print("STEP 7 PROMPT DEBUGGING")
print("=" * 70)
print()
print("To inspect Step 7 prompts, run the experiment with prompt capture:")
print()
print("  1. Edit run_experiment.py and set:")
print("     prompt_printer_enabled=True")
print()
print("  2. Run the experiment:")
print("     cd src && python -m development.step_7_codebookRefinement.run_experiment")
print()
print("  3. The prompts will be printed during execution.")
print()
print("Alternatively, modify run_experiment.py to store prompts to a file.")
print("=" * 70)

# If you have a stored prompts file, you can load and display it here:
# Example:
# import json
# with open("step7_prompts.json") as f:
#     prompts = json.load(f)
# for i, prompt in enumerate(prompts, 1):
#     print(f"PROMPT {i}: {prompt.get('step_name')}")
#     print(prompt['prompt_content'])
#     print("-" * 70)
