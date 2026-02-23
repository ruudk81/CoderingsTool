#%%  
"""
Debug script for Step 8: Code Assigner
Loads code assignments from cache and displays all ideas with full abstraction
ladder, assigned codes, and confidence scores.

Usage:
    cd src && python -m experiments.step_8_codeAssigner.debug_samples

Grouping options (set GROUP_BY below):
    "partition"  - group by concept_type partition
    "confidence" - group by confidence band (high/medium/low/unknown)
    "none"       - flat list, sorted by confidence descending
"""

import sys
from pathlib import Path
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

from collections import defaultdict
from experiments import models_exp as models
from config import CacheConfig
from utils.cacheManager import CacheManager, generate_enhanced_variable_key

# Import centralized test data config
try:
    from experiments.test_data import TEST_DATA
except ImportError:
    exp_root = Path(__file__).parent.parent
    if str(exp_root) not in sys.path:
        sys.path.insert(0, str(exp_root))
    from test_data import TEST_DATA

# =============================================================================
# CONFIGURATION
# =============================================================================
FILENAME = TEST_DATA.filename
VAR_NAME = TEST_DATA.var_name
SAMPLE_SIZE = TEST_DATA.sample_size

GROUP_BY = "partition"       # "partition" | "confidence" | "none"
FILTER_CODE = None           # e.g., "Overig" to show only that code
FILTER_PARTITION = None      # e.g., "sfeer en sociale beleving"
SHOW_RATIONALE = True        # show assignment rationale
MAX_IDEAS = None             # limit total ideas shown (None = all)


def format_idea(idea, response: str) -> str:
    """Format a single idea with full ladder and assignment info."""
    codes = idea.assigned_codes or ["(none)"]
    themes = idea.assigned_themes or []
    confidence = getattr(idea, 'assignment_confidence', None)
    rationale = getattr(idea, 'assignment_rationale', None)

    conf_str = f"{confidence:.2f}" if confidence is not None else "N/A"

    # Confidence indicator
    if confidence is not None:
        if confidence >= 0.9:
            indicator = "A"
        elif confidence >= 0.7:
            indicator = "B"
        elif confidence >= 0.5:
            indicator = "C"
        else:
            indicator = "D"
    else:
        indicator = "?"

    lines = []
    lines.append(f"  [{indicator}] conf={conf_str}  {idea.idea_id}")

    # Abstraction ladder (what codes are assigned TO)
    lines.append(f"      LADDER:")
    lines.append(f"        instance:          {idea.instance or '(empty)'}")
    lines.append(f"        concept:           {idea.concept or '(empty)'}")
    lines.append(f"        concept_type:      {idea.concept_type or '(empty)'}")
    ct_def = getattr(idea, 'concept_type_definition', None)
    if ct_def:
        lines.append(f"        concept_type_def:  {ct_def}")

    # Assignment result
    lines.append(f"      ASSIGNED:")
    lines.append(f"        code:              {', '.join(codes)}")
    if themes:
        lines.append(f"        theme:             {', '.join(themes)}")

    if SHOW_RATIONALE and rationale:
        rat_display = rationale[:150] + "..." if len(rationale) > 150 else rationale
        lines.append(f"      RATIONALE: {rat_display}")

    lines.append(f"      response: {response[:80]}{'...' if len(response) > 80 else ''}")

    return "\n".join(lines)


def main():
    variable_key = generate_enhanced_variable_key([VAR_NAME], False, SAMPLE_SIZE)
    cache_manager = CacheManager(CacheConfig())

    # Load code assigned results
    code_assigned_results = cache_manager.load_from_cache(
        FILENAME, "code_assignment_direct", variable_key, models.CodeAssignedModel
    )

    if not code_assigned_results:
        print("No code assignment results found in cache.")
        print(f"  filename: {FILENAME}")
        print(f"  variable_key: {variable_key}")
        print("Run step 8 first: cd src && python -m experiments.step_8_codeAssigner.run_experiment")
        return

    print(f"Loaded {len(code_assigned_results)} responses from cache")

    # Collect all ideas
    all_ideas = []
    for result in code_assigned_results:
        if result.response_ideas:
            for idea in result.response_ideas:
                all_ideas.append((idea, result.respondent_id, result.response or ""))

    print(f"Total ideas: {len(all_ideas)}")

    # Apply filters
    if FILTER_CODE:
        all_ideas = [(i, r, resp) for i, r, resp in all_ideas
                     if i.assigned_codes and FILTER_CODE in i.assigned_codes]
        print(f"Filtered to code '{FILTER_CODE}': {len(all_ideas)} ideas")

    if FILTER_PARTITION:
        all_ideas = [(i, r, resp) for i, r, resp in all_ideas
                     if (i.concept_type or "").lower() == FILTER_PARTITION.lower()]
        print(f"Filtered to partition '{FILTER_PARTITION}': {len(all_ideas)} ideas")

    if MAX_IDEAS and len(all_ideas) > MAX_IDEAS:
        all_ideas = all_ideas[:MAX_IDEAS]
        print(f"Limited to {MAX_IDEAS} ideas")

    if not all_ideas:
        print("No ideas found matching criteria")
        return

    # Summary stats
    confidences = [getattr(i, 'assignment_confidence', None) or 0.0 for i, _, _ in all_ideas]
    assigned = sum(1 for i, _, _ in all_ideas
                   if i.assigned_codes and i.assigned_codes[0] not in ("Overig", "Other", "Sonstiges"))
    unknown = len(all_ideas) - assigned

    print(f"\n{'=' * 80}")
    print(f"CODE ASSIGNMENT DEBUG — {len(all_ideas)} ideas")
    print(f"{'=' * 80}")
    print(f"Assigned: {assigned} ({assigned/len(all_ideas)*100:.1f}%)  |  "
          f"Unknown: {unknown} ({unknown/len(all_ideas)*100:.1f}%)  |  "
          f"Avg confidence: {sum(confidences)/len(confidences):.2f}")
    print(f"{'=' * 80}")

    # Group and display
    if GROUP_BY == "partition":
        _print_by_partition(all_ideas)
    elif GROUP_BY == "confidence":
        _print_by_confidence(all_ideas)
    else:
        _print_flat(all_ideas)


def _print_by_partition(all_ideas):
    """Group ideas by concept_type partition."""
    partitions = defaultdict(list)
    for idea, resp_id, response in all_ideas:
        partition = idea.concept_type or "(no partition)"
        partitions[partition].append((idea, resp_id, response))

    for partition in sorted(partitions.keys()):
        items = partitions[partition]
        confs = [getattr(i, 'assignment_confidence', None) or 0.0 for i, _, _ in items]
        assigned = sum(1 for i, _, _ in items
                       if i.assigned_codes and i.assigned_codes[0] not in ("Overig", "Other", "Sonstiges"))
        avg_conf = sum(confs) / len(confs) if confs else 0

        print(f"\n{'_' * 80}")
        print(f"PARTITION: {partition}")
        print(f"  {len(items)} ideas | {assigned} assigned | "
              f"{len(items) - assigned} unknown | avg conf: {avg_conf:.2f}")
        print(f"{'_' * 80}")

        # Sort by confidence descending within partition
        items.sort(key=lambda x: getattr(x[0], 'assignment_confidence', None) or 0.0, reverse=True)

        for idea, resp_id, response in items:
            print(format_idea(idea, response))
            print()


def _print_by_confidence(all_ideas):
    """Group ideas by confidence band."""
    bands = {
        "A: Explicit (0.90-1.00)": [],
        "B: Paraphrase (0.70-0.89)": [],
        "C: Weak/Implied (0.50-0.69)": [],
        "D: No Fit / Unknown (0.00-0.49)": [],
    }

    for idea, resp_id, response in all_ideas:
        conf = getattr(idea, 'assignment_confidence', None) or 0.0
        if conf >= 0.90:
            bands["A: Explicit (0.90-1.00)"].append((idea, resp_id, response))
        elif conf >= 0.70:
            bands["B: Paraphrase (0.70-0.89)"].append((idea, resp_id, response))
        elif conf >= 0.50:
            bands["C: Weak/Implied (0.50-0.69)"].append((idea, resp_id, response))
        else:
            bands["D: No Fit / Unknown (0.00-0.49)"].append((idea, resp_id, response))

    for band_name, items in bands.items():
        if not items:
            continue

        print(f"\n{'_' * 80}")
        print(f"CONFIDENCE BAND: {band_name} — {len(items)} ideas")
        print(f"{'_' * 80}")

        items.sort(key=lambda x: getattr(x[0], 'assignment_confidence', None) or 0.0, reverse=True)

        for idea, resp_id, response in items:
            print(format_idea(idea, response))
            print()


def _print_flat(all_ideas):
    """Flat list sorted by confidence descending."""
    all_ideas.sort(key=lambda x: getattr(x[0], 'assignment_confidence', None) or 0.0, reverse=True)

    for idea, resp_id, response in all_ideas:
        print(format_idea(idea, resp_id, response))
        print()


if __name__ == "__main__":
    main()
