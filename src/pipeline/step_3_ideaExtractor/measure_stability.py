#%%

"""Measure step 3's run-to-run stability. Read-only, no LLM calls.

Run this after every step 3 run. It records a snapshot and prints the comparison
against every earlier snapshot of the same dataset+variable, because a run
overwrites the cache and the verbose log of the previous one — without a snapshot
taken at the time, the previous run is simply gone.

Three questions, in order of weight:

  1. Is the domain partition stable across runs?
     Compared with the Adjusted Rand Index over respondents, which needs no label
     matching: it asks whether respondents grouped together in run A are still
     grouped together in run B. Labels change between runs, groupings are what
     downstream steps actually build on. 1.0 = identical partition, 0.0 = no better
     than chance.

  2. What is the noise floor on the domain layer?
     Free from repeated answer texts within one run: identical text, same run, no
     context that could justify a different domain. Reported as the share of
     repeated ideas sitting on the minority side.

  3. Where does the mass sit, and are the drains behaving?
     A large `other` means the discovered menu is missing something. A large
     `bare_evaluation` is only healthy if those answers genuinely name no subject —
     it is the counter-metric for any change that widens the drains.

Only respondents with exactly one idea are used for (1) and (2): then the domain is
a function of the whole response text, with nothing else to explain a difference.
"""

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, List

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from utils.cacheManager import CacheManager, generate_enhanced_variable_key
import models

from test_data import TEST_DATA


# =============================================================================
# CONFIGURATION
# =============================================================================

FILENAME = TEST_DATA.filename
VARIABLE = TEST_DATA.var_name
SAMPLE_SIZE = TEST_DATA.sample_size

SNAPSHOT_FILE = project_root / "data" / "step3_stability.jsonl"

DRAIN_KEYS = ("bare_evaluation", "other")


# =============================================================================
# MEASUREMENT
# =============================================================================

def build_snapshot(rows, meta) -> Dict:
    """One run, reduced to the numbers worth comparing against another run."""
    ideas = [i for r in rows for i in (r.response_ideas or [])]
    key_by_label = {d["label"]: d.get("key", "") for d in (meta.domains or [])}

    counts = Counter(i.domain for i in ideas)
    total = sum(counts.values()) or 1

    def drain_share(key: str) -> float:
        return 100 * sum(n for l, n in counts.items()
                         if key_by_label.get(l) == key) / total

    # Respondents whose whole response produced exactly one idea: the domain is then
    # a function of the text alone, comparable within and across runs.
    single = {r.respondent_id: r.response_ideas[0].domain
              for r in rows if r.response_ideas and len(r.response_ideas) == 1}
    text_of = {r.respondent_id: (r.response or "").strip().lower()
               for r in rows if r.respondent_id in single}

    return {
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "filename": FILENAME,
        "variable": VARIABLE,
        "sample_size": SAMPLE_SIZE,
        "respondents": len(rows),
        "ideas": len(ideas),
        "processing_errors": sum(1 for i in ideas
                                 if (i.idea or "").startswith("PROCESSING_ERROR")),
        "respondents_without_ideas": sum(1 for r in rows if not (r.response_ideas or [])),
        "domains": [{"key": d.get("key", ""), "label": d.get("label", "")}
                    for d in (meta.domains or [])],
        "substantive_domains": sum(1 for d in (meta.domains or [])
                                   if d.get("key") not in DRAIN_KEYS),
        "shares": {l: round(100 * n / total, 1) for l, n in counts.most_common()},
        "bare_evaluation_pct": round(drain_share("bare_evaluation"), 1),
        "other_pct": round(drain_share("other"), 1),
        "assignments": single,      # respondent_id -> domain label
        "texts": text_of,           # respondent_id -> response text
    }


def noise_floor(snapshot: Dict) -> Dict:
    """Share of repeated-answer ideas sitting on the minority side.

    Identical text, one run, different domain: there is no context left that could
    justify the difference, so no judgement call is needed to read it as an error.
    """
    by_text = defaultdict(list)
    for rid, domain in snapshot["assignments"].items():
        by_text[snapshot["texts"][rid]].append(domain)

    repeated = {t: v for t, v in by_text.items() if len(v) > 1}
    inconsistent = {t: Counter(v) for t, v in repeated.items() if len(set(v)) > 1}

    n_repeated = sum(len(v) for v in repeated.values())
    minority = sum(sum(c.values()) - max(c.values()) for c in inconsistent.values())

    return {
        "repeated_ideas": n_repeated,
        "repeated_texts": len(repeated),
        "inconsistent_texts": len(inconsistent),
        "minority": minority,
        "pct": round(100 * minority / n_repeated, 1) if n_repeated else 0.0,
        "detail": sorted(((t, dict(c)) for t, c in inconsistent.items()),
                         key=lambda kv: -sum(kv[1].values())),
    }


def adjusted_rand_index(a: Dict[str, str], b: Dict[str, str]) -> float:
    """ARI between two labellings of the same units. Label names are irrelevant.

    Counts pairs of respondents that both runs put together, or both apart, and
    corrects for the agreement expected by chance. Written out here rather than
    pulled from a library: it is fifteen lines, and the correction term is the
    whole point — a raw agreement rate looks high whenever one domain dominates.
    """
    shared = sorted(set(a) & set(b))
    if len(shared) < 2:
        return float("nan")

    table = Counter((a[u], b[u]) for u in shared)
    rows = Counter(a[u] for u in shared)
    cols = Counter(b[u] for u in shared)

    def c2(n: int) -> int:
        return n * (n - 1) // 2

    index = sum(c2(n) for n in table.values())
    exp_rows = sum(c2(n) for n in rows.values())
    exp_cols = sum(c2(n) for n in cols.values())
    total = c2(len(shared))

    expected = exp_rows * exp_cols / total
    maximum = (exp_rows + exp_cols) / 2
    if maximum == expected:
        return 1.0
    return (index - expected) / (maximum - expected)


# =============================================================================
# REPORTING
# =============================================================================

def print_run(snapshot: Dict, nf: Dict) -> None:
    print(f"\n{'=' * 72}\nTHIS RUN  ({snapshot['recorded_at']})\n{'=' * 72}")
    print(f"respondents {snapshot['respondents']} | ideas {snapshot['ideas']} | "
          f"domains {len(snapshot['domains'])} "
          f"({snapshot['substantive_domains']} substantive + 2 standing)")
    print(f"PROCESSING_ERROR {snapshot['processing_errors']} | "
          f"respondents without ideas {snapshot['respondents_without_ideas']}")

    print("\nmass per domain")
    for label, pct in snapshot["shares"].items():
        key = next((d["key"] for d in snapshot["domains"] if d["label"] == label), "??")
        mark = "  <-- standing" if key in DRAIN_KEYS else ""
        print(f"  {pct:>5.1f}%  {label}{mark}")

    print(f"\nbare_evaluation {snapshot['bare_evaluation_pct']}%   "
          f"other {snapshot['other_pct']}%")
    print(f"noise floor     {nf['pct']}%  "
          f"({nf['minority']} of {nf['repeated_ideas']} repeated ideas on the "
          f"minority side, over {nf['inconsistent_texts']} inconsistent texts)")

    if nf["detail"]:
        print("\n  the inconsistent texts, by frequency")
        for text, counts in nf["detail"][:10]:
            print(f"    {sum(counts.values()):>3}x  {text[:40]:<40} {counts}")


def print_comparison(history: List[Dict]) -> None:
    print(f"\n{'=' * 72}\nACROSS {len(history)} RUNS\n{'=' * 72}")

    header = f"{'run':<22}{'subst':>7}{'ideas':>8}{'bare%':>8}{'other%':>8}{'noise%':>8}{'errors':>8}"
    print(header)
    for snap in history:
        nf = noise_floor(snap)
        print(f"{snap['recorded_at']:<22}{snap['substantive_domains']:>7}"
              f"{snap['ideas']:>8}{snap['bare_evaluation_pct']:>8}"
              f"{snap['other_pct']:>8}{nf['pct']:>8}{snap['processing_errors']:>8}")

    print("\npartition stability between consecutive runs (Adjusted Rand Index)")
    print("  1.00 = same grouping of respondents; 0.00 = chance")
    for earlier, later in zip(history, history[1:]):
        ari = adjusted_rand_index(earlier["assignments"], later["assignments"])
        shared = len(set(earlier["assignments"]) & set(later["assignments"]))
        print(f"  {earlier['recorded_at']} -> {later['recorded_at']}: "
              f"ARI {ari:.3f}  (over {shared} respondents)")

    print("\ndomain labels per run")
    for snap in history:
        subst = [d["label"] for d in snap["domains"] if d["key"] not in DRAIN_KEYS]
        print(f"  {snap['recorded_at']}: {subst}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    variable_key = generate_enhanced_variable_key(
        selected_variables=[VARIABLE], is_merged=False, sample_size=SAMPLE_SIZE)
    cache_manager = CacheManager()

    rows = cache_manager.load_from_cache(
        FILENAME, "extracted_ideas", variable_key, models.IdeasExtractedModel)
    meta = cache_manager.load_metadata_from_cache(
        FILENAME, "extracted_ideas", variable_key, models.ExtractionMetadata)

    if not rows or not meta:
        print("No step 3 cache for this dataset+variable. Run step 3 first.")
        return

    snapshot = build_snapshot(rows, meta)
    print_run(snapshot, noise_floor(snapshot))

    # Append first, read back after: the snapshot of this run must survive the next
    # one, which overwrites the cache it was computed from.
    SNAPSHOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with SNAPSHOT_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(snapshot, ensure_ascii=False) + "\n")

    history = [
        s for s in (json.loads(line) for line in
                    SNAPSHOT_FILE.read_text(encoding="utf-8").splitlines() if line.strip())
        if s.get("filename") == FILENAME and s.get("variable") == VARIABLE
    ]
    if len(history) > 1:
        print_comparison(history)
    else:
        print(f"\nFirst snapshot recorded in {SNAPSHOT_FILE.name}. "
              f"Run step 3 again and re-run this to get the comparison.")


if __name__ == "__main__":
    main()
