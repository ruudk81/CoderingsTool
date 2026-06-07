#%%

"""
View code divergence: find words (instances) that received DIFFERENT codes
across their occurrences — the "same word, different code" inconsistency.

Read-only, deterministic, no LLM. Groups ideas by their verbatim instance and
flags groups whose occurrences were assigned more than one distinct code. This
sizes the multi-coding problem and pinpoints which words are ambiguous.

LEVEL switches the source:
  - "code":      step 6 output (taxonomy_codes) — the real target (run step 6 first)
  - "attribute": step 4 output (taxonomy_classified) — proxy, available before step 6

Usage:
    cd src && python -m pipeline.step_6_codeAssigner.view_code_divergence
"""

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, List, Optional

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from pipeline.step_6_codeAssigner.models_codeAssigner import CodeAssignedModel
from pipeline.step_4_classifier.models_classifier import TaxonomyClassifiedModel

from test_data import TEST_DATA


# =============================================================================
# CONFIGURATION
# =============================================================================

LEVEL = "code"            # "code" (taxonomy_codes) or "attribute" (taxonomy_classified, proxy)
GROUPING = "exact"        # "exact" or "loose" (crude suffix-strip for plurals/diminutives)
MIN_OCCURRENCES = 2       # a word must occur at least this often to be considered
EXCLUDE_OTHER = True      # ignore "Other"/sentinel codes when judging divergence
TOP_N = 20                # how many divergent words to list

FILENAME = TEST_DATA.filename
VARIABLE = TEST_DATA.var_name
SAMPLE_SIZE = TEST_DATA.sample_size

# Code/attribute values that are not real assignments
_SENTINELS = {"__unassigned__", "(no attribute)", "no code assigned", ""}


# =============================================================================
# NORMALIZATION
# =============================================================================

def _norm_exact(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _norm_loose(s: str) -> str:
    s = _norm_exact(s)
    if " " not in s:  # only stem single words, not phrases
        s = re.sub(r"(tje|pje|je|en|s)$", "", s)
    return s


def _normalizer() -> Callable[[str], str]:
    return _norm_loose if GROUPING == "loose" else _norm_exact


# =============================================================================
# DATA LOADING
# =============================================================================

def load_ideas():
    """Load ideas + a value accessor for the configured LEVEL.

    Returns (ideas, value_of) where value_of(idea) -> the assigned code or
    attribute string.
    """
    variable_key = generate_enhanced_variable_key(
        selected_variables=[VARIABLE], is_merged=False, sample_size=SAMPLE_SIZE,
    )
    cm = CacheManager()

    if LEVEL == "code":
        data = cm.load_from_cache(FILENAME, "taxonomy_codes", variable_key, CodeAssignedModel)
        if not data:
            raise FileNotFoundError(
                "No taxonomy_codes cache — run step 6 first (this dataset has no "
                "code assignments yet). Set LEVEL='attribute' for the step-4 proxy."
            )
        value_of = lambda idea: idea.assigned_code
    else:
        data = cm.load_from_cache(FILENAME, "taxonomy_classified", variable_key, TaxonomyClassifiedModel)
        if not data:
            raise FileNotFoundError("No taxonomy_classified cache — run step 4 first.")
        value_of = lambda idea: idea.attribute

    ideas = [i for r in data for i in (r.response_ideas or [])]
    return ideas, value_of


# =============================================================================
# ANALYSIS
# =============================================================================

def _is_real(value: str) -> bool:
    v = (value or "").strip().lower()
    if v in _SENTINELS:
        return False
    if EXCLUDE_OTHER and v == "other":
        return False
    return True


def _trunc(s: str, n: int = 34) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def analyze(ideas: List, value_of: Callable) -> None:
    norm = _normalizer()

    groups: dict = defaultdict(list)  # normalized instance -> [code/attr, ...]
    n_with_value = 0
    for idea in ideas:
        value = (value_of(idea) or "").strip()
        if not _is_real(value):
            continue
        n_with_value += 1
        key = norm(idea.instance)
        if key:
            groups[key].append(value)

    multi = {k: v for k, v in groups.items() if len(v) >= MIN_OCCURRENCES}
    divergent = {k: v for k, v in multi.items() if len(set(v)) > 1}
    affected = sum(len(v) for v in divergent.values())

    print(f"\n{'=' * 78}")
    print(f"CODE DIVERGENCE — level={LEVEL}, grouping={GROUPING}")
    print(f"{FILENAME}  |  {VARIABLE}  |  n={SAMPLE_SIZE}")
    print(f"{'=' * 78}")
    print(f"ideas: {len(ideas)}  (with a real {LEVEL}: {n_with_value})")
    print(f"words occurring >={MIN_OCCURRENCES}x : {len(multi)}")
    print(f"  consistent (1 {LEVEL})     : {len(multi) - len(divergent)}")
    print(f"  divergent (>=2 {LEVEL}s)   : {len(divergent)}")
    pct = 100 * affected / max(1, n_with_value)
    print(f"ideas hit by divergence    : {affected}  ({pct:.1f}% of coded)")

    if not divergent:
        print("\nNo divergent words. Every repeated word got a single consistent code.")
        return

    print(f"\nTop {TOP_N} divergent words  (fan-out: 'dual'=2 codes, 'wide'=>2):")
    ranked = sorted(divergent.items(), key=lambda kv: -len(kv[1]))
    for key, values in ranked[:TOP_N]:
        counts = Counter(values)
        fanout = "dual" if len(counts) == 2 else "wide"
        split = " / ".join(f"{n} {_trunc(code)}" for code, n in counts.most_common())
        print(f"  [{fanout}] \"{key}\" ({len(values)}x, {len(counts)} codes): {split}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    ideas, value_of = load_ideas()
    analyze(ideas, value_of)

# %%
