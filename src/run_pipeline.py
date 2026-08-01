"""
Pipeline Runner — Runs all steps sequentially using test_data configuration.

Each step handles its own caching. Set force_recalc=True in step configs
to force reprocessing, or False to use cached results.

Usage:
    cd src && python pipeline.py
"""

import os, sys

# Ensure src/ is on sys.path
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import time
import warnings
import nest_asyncio
nest_asyncio.apply()

warnings.filterwarnings("ignore", message="To exit: use 'exit', 'quit', or Ctrl-D.")

from test_data import TEST_DATA
from utils.saveVerbose import VerboseCapture
from utils.llm import token_tracker

# =============================================================================
# PIPELINE CONFIGURATION
# =============================================================================

RUN_UNTIL_STEP = 4           
FORCE_RECALCULATE_ALL = False
VERBOSE = True

# =============================================================================
# STEP NAMES (for display)
# =============================================================================

STEP_NAMES = {
    0: "Data Loading",
    1: "Preprocessing",
    2: "Quality Filter",
    3: "Idea Extraction",
    4: "Taxonomy Classification",
    5: "Codebook Generation",
    6: "Code Assignment",
    7: "Export",
}

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(
    run_until_step: int = RUN_UNTIL_STEP,
    force_recalc: bool = FORCE_RECALCULATE_ALL,
    verbose: bool = VERBOSE,
):
    """Run the full pipeline from step 0 through run_until_step.

    Cache semantics:
      - Steps 0..run_until_step-1 load from cache when valid.
      - Step run_until_step (the target) always recomputes.
      - force_recalc=True (FORCE_RECALCULATE_ALL) makes every step recompute.
    """

    pipeline_start = time.time()

    def step_force(step_i: int) -> bool:
        return force_recalc or (step_i == run_until_step)

    print("=" * 70)
    print("PIPELINE RUNNER")
    print("=" * 70)
    print(f"Dataset:      {TEST_DATA.filename}")
    print(f"Variable:     {TEST_DATA.var_name}")
    print(f"Sample size:  {TEST_DATA.sample_size}")
    print(f"Run until:    Step {run_until_step} ({STEP_NAMES.get(run_until_step, '?')})")
    print(f"Force recalc: {force_recalc} (target step {run_until_step} always recomputes)")
    print("=" * 70)

    # --- Step 0: Load Data ---
    if run_until_step >= 0:
        print(f"\n{'='*70}\nStep 0 — {STEP_NAMES[0]}\n{'='*70}")
        from pipeline.step_0_dataLoader.run_dataLoader import run_step as run_step_0, StepConfig as Step0Config
        config_0 = Step0Config(force_recalc=step_force(0))
        raw_data = run_step_0(config_0)
        print(f"  → {len(raw_data)} responses loaded")

    # --- Step 1: Preprocess ---
    if run_until_step >= 1:
        print(f"\n{'='*70}\nStep 1 — {STEP_NAMES[1]}\n{'='*70}")
        from pipeline.step_1_preProcessor.run_preProcessor import run_step as run_step_1, StepConfig as Step1Config
        config_1 = Step1Config(force_recalc=step_force(1))
        preprocessed = run_step_1(config_1)
        print(f"  → {len(preprocessed)} responses preprocessed")

    # --- Step 2: Quality Filter ---
    if run_until_step >= 2:
        print(f"\n{'='*70}\nStep 2 — {STEP_NAMES[2]}\n{'='*70}")
        from pipeline.step_2_qualityFilter.run_qualityFilter import run_step as run_step_2, StepConfig as Step2Config
        config_2 = Step2Config(force_recalc=step_force(2))
        filtered = run_step_2(config_2)
        print(f"  → {len(filtered)} responses after filtering")

    # --- Step 3: Idea Extraction ---
    if run_until_step >= 3:
        print(f"\n{'='*70}\nStep 3 — {STEP_NAMES[3]}\n{'='*70}")
        from pipeline.step_3_ideaExtractor.run_ideaExtractor import run_step as run_step_3, StepConfig as Step3Config
        config_3 = Step3Config(force_recalc=step_force(3))
        result_3 = run_step_3(config_3)
        # run_step returns (ideas_models, extractor, prompt_printer)
        if isinstance(result_3, tuple):
            ideas = result_3[0]
        else:
            ideas = result_3
        total_ideas = sum(len(r.response_ideas) for r in ideas if hasattr(r, 'response_ideas'))
        print(f"  → {total_ideas} ideas extracted from {len(ideas)} responses")

    # --- Step 4: Taxonomy Classification ---
    if run_until_step >= 4:
        print(f"\n{'='*70}\nStep 4 — {STEP_NAMES[4]}\n{'='*70}")
        from pipeline.step_4_classifier.run_classifier import run_taxonomy
        run_taxonomy(force_recalc=step_force(4))

    # --- Step 5: Codebook Generation ---
    if run_until_step >= 5:
        print(f"\n{'='*70}\nStep 5 — {STEP_NAMES[5]}\n{'='*70}")
        from pipeline.step_5_codeGenerator.run_codeGenerator import run_codebook
        run_codebook(force_recalc=step_force(5))

    # --- Step 6: Code Assignment ---
    if run_until_step >= 6:
        print(f"\n{'='*70}\nStep 6 — {STEP_NAMES[6]}\n{'='*70}")
        from pipeline.step_6_codeAssigner.run_codeAssigner import run_assignment
        run_assignment(force_recalc=step_force(6))

    # --- Step 7: Export ---
    if run_until_step >= 7:
        print(f"\n{'='*70}\nStep 7 — {STEP_NAMES[7]}\n{'='*70}")
        from pipeline.step_7_export.run_export import run_step as run_step_7, StepConfig as Step7Config
        config_7 = Step7Config(force_recalc=step_force(7))
        run_step_7(config_7)
        # Also write the standalone codebook/taxonomy readout to exports/codebook/
        # (same call the Streamlit app makes at the Export step — reused as-is).
        from pipeline.step_6_codeAssigner.view_codebook import export_codebook
        export_codebook(
            filename=TEST_DATA.filename,
            var_name=TEST_DATA.var_name,
            sample_size=TEST_DATA.sample_size,
        )

    # --- Summary ---
    elapsed = time.time() - pipeline_start
    print(f"\n{'='*70}")
    print(f"PIPELINE COMPLETE — {elapsed:.1f}s total")
    print(f"Steps completed: 0–{run_until_step}")
    print(token_tracker.get_summary())
    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    verbose_capture = VerboseCapture(
        filename=TEST_DATA.filename,
        var_name=TEST_DATA.var_name,
        sample_size=TEST_DATA.sample_size,
        step=RUN_UNTIL_STEP,
    )

    with verbose_capture:
        run_pipeline()
