#%%

"""
View valence-split attributes: find attribute PAIRS within a facet that differ
only in evaluative direction — a valence artifact baked into the taxonomy
(e.g. "Algemene positieve waardering" vs "Algemene niet-positieve waardering").

A pair is flagged when BOTH hold (see valence_consolidator.detect_valence_splits):
  - label similarity (token-set Jaccard or char ratio) >= LABEL_SIM_THRESHOLD, AND
  - the two attributes skew to OPPOSITE valence (one mostly "+", the other mostly not-"+").

Both signals are needed: label-similarity alone over-flags descriptive near-
duplicates (Sparen/Betalen); valence-skew alone over-flags homogeneous-but-solo
attributes (Concrete natuurbeelden, which has no opposite sibling).

Read-only, deterministic, no LLM. Detection only — it does NOT merge. The
deterministic merge (with LLM renaming) lives in valence_consolidator.py and
runs in the pipeline after P9. See dev/DESIGN_VALENCE_NEUTRALITY.md.

Usage:
    cd src && python -m pipeline.step_4_classifier.view_valence_split
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from models import TaxonomyClassifiedModel
from pipeline.step_4_classifier.valence_consolidator import detect_valence_splits

from test_data import TEST_DATA


# =============================================================================
# CONFIGURATION
# =============================================================================

LABEL_SIM_THRESHOLD = 0.6   # min label similarity (max of token-set Jaccard / char ratio)
MIN_SKEW = 0.7              # an attribute "skews +" if >= this fraction of ideas are "+"
MIN_COUNT = 5              # ignore attributes with fewer ideas
AUTO_SAFE_SIM = 0.8        # label sim above which a flagged pair is auto-merge-safe

FILENAME = TEST_DATA.filename
VARIABLE = TEST_DATA.var_name
SAMPLE_SIZE = TEST_DATA.sample_size


def _valence_str(counter, total: int) -> str:
    return f"+{counter.get('+', 0)} -{counter.get('-', 0)} 0{counter.get('0', 0)} (n={total})"


def main():
    variable_key = generate_enhanced_variable_key(
        selected_variables=[VARIABLE], is_merged=False, sample_size=SAMPLE_SIZE,
    )
    cm = CacheManager()
    data = cm.load_from_cache(FILENAME, "taxonomy_classified", variable_key, TaxonomyClassifiedModel)
    if not data:
        raise FileNotFoundError("No taxonomy_classified cache — run step 4 first.")

    pairs = detect_valence_splits(
        data, LABEL_SIM_THRESHOLD, MIN_SKEW, MIN_COUNT, AUTO_SAFE_SIM,
    )

    print(f"\n{'=' * 78}")
    print(f"VALENCE-SPLIT ATTRIBUTES  (label_sim>={LABEL_SIM_THRESHOLD}, skew>={MIN_SKEW})")
    print(f"{FILENAME}  |  {VARIABLE}  |  n={SAMPLE_SIZE}")
    print(f"{'=' * 78}")

    for p in sorted(pairs, key=lambda x: -x.sim):
        # mergeable by the pipeline = auto-safe AND a clean single-token diff
        if p.auto_safe and p.fallback_name:
            tag = "AUTO-MERGE-SAFE"
        elif p.auto_safe:
            tag = "auto-safe (no single-token name -> review)"
        else:
            tag = "review"
        print(f"\n[{tag}]  domain: {p.domain}  |  facet: {p.facet}  |  label_sim={p.sim:.2f}")
        print(f"   - \"{p.name_a}\"   {_valence_str(p.val_a, p.total_a)}")
        print(f"       e.g. {', '.join(repr(x) for x in p.samples_a)}")
        print(f"   - \"{p.name_b}\"   {_valence_str(p.val_b, p.total_b)}")
        print(f"       e.g. {', '.join(repr(x) for x in p.samples_b)}")

    print(f"\n{'-' * 78}")
    print(f"{len(pairs)} valence-split candidate pair(s) flagged.")
    if not pairs:
        print("No attribute pairs differ only in evaluative direction.")


if __name__ == "__main__":
    main()

# %%
