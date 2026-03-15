#%% 
"""
Summary helper for Step 8: Code Assigner

Loads code assignment results and the refined codebook from cache, then prints
a frequency summary organized by theme, showing both idea-level and
respondent-level counts (deduplicated per respondent).

Usage (standalone):
    cd src && python -m development.step_8_codeAssigner.summary

Usage (from run_experiment.py):
    from .summary import print_code_summary
    print_code_summary(config)
"""

import sys
from pathlib import Path
from collections import defaultdict

src_dir = Path(__file__).parent.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from development import models_exp as models
from config import CacheConfig
from utils.cacheManager import CacheManager, generate_enhanced_variable_key

# Import centralized test data config
try:
    from development.test_data import TEST_DATA
except ImportError:
    exp_root = Path(__file__).parent.parent
    if str(exp_root) not in sys.path:
        sys.path.insert(0, str(exp_root))
    from test_data import TEST_DATA

# Unknown / catch-all labels used by the code assigner
_UNKNOWN_LABELS = {"Overig", "Other", "Sonstiges", "Unassigned"}


def print_code_summary(config) -> None:
    """Print a code-frequency summary organized by the refined codebook's theme hierarchy.

    Deduplication rule: duplicate codes within a single respondent are collapsed
    first.  A respondent with 3 ideas all coded "Speed" produces ONE count for
    "Speed".  Both columns are derived from the deduplicated set:

      - Count:  number of (respondent, code) pairs after dedup
      - % ideas:        count / total deduplicated code-assignments (sums to 100%)
      - % respondents:  count / total respondents (can sum to >100%)

    Args:
        config: ExperimentConfig (or any object with filename, var_name, sample_size).
    """
    variable_key = generate_enhanced_variable_key(
        selected_variables=[config.var_name],
        is_merged=False,
        sample_size=config.sample_size,
    )
    cache_manager = CacheManager(CacheConfig())

    # ------------------------------------------------------------------
    # Load step 7 codebook (for theme grouping structure)
    # ------------------------------------------------------------------
    codebook_list = cache_manager.load_from_cache(
        config.filename, "codebook_refinement_enriched", variable_key,
        models.ThemeEnrichedCodebookModelExp,
    )
    if not codebook_list:
        print("[summary] No codebook found in cache — skipping summary.")
        return
    codebook = codebook_list[0]

    # ------------------------------------------------------------------
    # Load step 8 assignment results
    # ------------------------------------------------------------------
    results = cache_manager.load_from_cache(
        config.filename, "code_assignment_direct", variable_key,
        models.CodeAssignedModel,
    )
    if not results:
        print("[summary] No code assignment results found in cache — skipping summary.")
        return

    # ------------------------------------------------------------------
    # Build theme → [codes] ordering from the codebook
    # ------------------------------------------------------------------
    theme_to_codes: dict[str, list[str]] = defaultdict(list)
    code_set: set[str] = set()
    for entry in codebook.codes:
        theme = entry.theme or "(no theme)"
        if entry.code and entry.code not in code_set:
            theme_to_codes[theme].append(entry.code)
            code_set.add(entry.code)

    # ------------------------------------------------------------------
    # Count frequencies (deduplicated per respondent first)
    # ------------------------------------------------------------------
    # Step 1: collect unique codes per respondent
    respondent_codes: dict[str, set[str]] = defaultdict(set)  # respondent_id → {codes}

    for result in results:
        resp_id = str(result.respondent_id)
        respondent_codes[resp_id]  # ensure key exists even if no ideas
        if not result.response_ideas:
            continue
        for idea in result.response_ideas:
            code = (idea.assigned_codes[0]
                    if idea.assigned_codes else "Unassigned")
            respondent_codes[resp_id].add(code)

    # Step 2: derive counts from the deduplicated sets
    code_counts: dict[str, int] = defaultdict(int)
    for codes in respondent_codes.values():
        for code in codes:
            code_counts[code] += 1

    total_respondents = len(respondent_codes)
    total_deduped = sum(code_counts.values())  # total (respondent, code) pairs

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("CODE ASSIGNMENT SUMMARY  (deduplicated per respondent)")
    print(f"Total code-assignments: {total_deduped}  |  Total respondents: {total_respondents}")
    print("=" * 78)

    # Determine column widths
    max_code_len = max(
        (len(c) for c in code_set),
        default=20,
    )
    max_code_len = max(max_code_len, 20)  # minimum width

    # Print per theme
    for theme, codes in theme_to_codes.items():
        # Theme header with totals
        theme_count = sum(code_counts.get(c, 0) for c in codes)
        theme_resps = _unique_respondents_for_codes(codes, respondent_codes)
        theme_count_pct = _pct(theme_count, total_deduped)
        theme_resp_pct = _pct(theme_resps, total_respondents)

        print()
        print(f"  THEME: {theme}  "
              f"[n={theme_count} ({theme_count_pct})  |  "
              f"{theme_resps} resp ({theme_resp_pct})]")

        for i, code in enumerate(codes):
            is_last = (i == len(codes) - 1)
            prefix = "└─" if is_last else "├─"
            cc = code_counts.get(code, 0)
            cp = _pct(cc, total_deduped)
            rp = _pct(cc, total_respondents)
            print(f"  {prefix} {code:<{max_code_len}}  "
                  f"n={cc:>3} ({cp:>5})   "
                  f"{cc:>4} resp ({rp:>5})")

    # Unknown / unassigned bucket
    unknown_codes = [c for c in code_counts if c in _UNKNOWN_LABELS or c not in code_set]
    if unknown_codes:
        unk_count = sum(code_counts[c] for c in unknown_codes)
        unk_resps = _unique_respondents_for_codes(unknown_codes, respondent_codes)
        unk_cp = _pct(unk_count, total_deduped)
        unk_rp = _pct(unk_resps, total_respondents)

        print()
        print(f"  {'─' * 74}")
        label = " / ".join(sorted(unknown_codes))
        print(f"  {label:<{max_code_len + 4}}"
              f"n={unk_count:>3} ({unk_cp:>5})   "
              f"{unk_resps:>4} resp ({unk_rp:>5})")

    print()
    print("=" * 78)
    print(f"  n / % codes: deduplicated count & % of all assignments ({total_deduped})")
    print(f"  resp:        % of respondents ({total_respondents}) — may sum to >100%")
    print("=" * 78)
    print()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _pct(count: int, total: int) -> str:
    """Format a percentage string like '14.3%'."""
    if total == 0:
        return " 0.0%"
    return f"{count / total * 100:>4.1f}%"


def _unique_respondents_for_codes(
    codes: list[str],
    respondent_codes: dict[str, set[str]],
) -> int:
    """Count how many respondents have at least one of the given codes."""
    return sum(
        1 for resp_codes in respondent_codes.values()
        if resp_codes & set(codes)
    )


# ------------------------------------------------------------------
# Standalone entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    from dataclasses import dataclass
    from typing import Optional

    @dataclass
    class _StandaloneConfig:
        filename: str = TEST_DATA.filename
        var_name: str = TEST_DATA.var_name
        sample_size: Optional[int] = TEST_DATA.sample_size

    print_code_summary(_StandaloneConfig())
