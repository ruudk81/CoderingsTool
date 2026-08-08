"""Score step 1's output against the hand-labelled answer key.

Read-only. Run from src/:
    python -m pipeline.step_1_preProcessor.score_against_key

Why this exists: two runs of identical code differ on 16% of the responses the LLM
touches, so a single before/after comparison cannot tell an improvement from a dice
roll. The key fixes what the right answer is, independently of any run, and every
change is scored the same way afterwards.

Three outcomes per response:
    correct       output matches the key
    uncorrected   output equals the input while the key asked for a change
    wrong         output differs from both

Words and pairs labelled "?" in the key are excluded rather than guessed at.
"""
import json
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from test_data import TEST_DATA                                     # noqa: E402
from config import CacheConfig                                      # noqa: E402
from utils.cacheManager import CacheManager, generate_enhanced_variable_key  # noqa: E402
import models                                                       # noqa: E402

KEY_DIR = Path(__file__).parent / "dev"


def normalize(text: str) -> str:
    """TextNormalizer, minus the parts that only affect presentation."""
    text = re.sub(r'\s*/\s*|/', ' , ', text)
    text = " ".join(text.split())
    return re.sub(r"\s+([,.;!?])", r"\1", text).strip()


def comparable(text: str) -> str:
    """Strip what step 1 is allowed to decide freely: outer casing and end punctuation."""
    text = re.sub(r"\s*([,.;:!?])\s*", r"\1 ", str(text))     # spacing around punctuation is free
    text = " ".join(text.split()).strip().rstrip(".").strip()
    return text.lower()


def expected_text(raw: str, key: dict) -> tuple[str, bool]:
    """Apply the key to one raw response. Returns (expected, is_scorable)."""
    text = normalize(raw)
    scorable = True

    for word, target in key["words"].items():
        if not re.search(rf'\b{re.escape(word)}\b', text):
            continue
        if target == "?":
            scorable = False
        elif target is not None:
            text = re.sub(rf'\b{re.escape(word)}\b', target, text)

    for pair, join in key["pairs"].items():
        first, second = pair.split(" ", 1)
        pattern = rf'\b{re.escape(first)}\s+{re.escape(second)}\b'
        if join and re.search(pattern, text, re.I):
            text = re.sub(pattern, first + second, text, flags=re.I)

    return text, scorable


def main():
    config = TEST_DATA
    key_files = sorted(KEY_DIR.glob("answer_key_*.json"))
    if not key_files:
        raise SystemExit(f"No answer key in {KEY_DIR}")
    key = json.loads(key_files[0].read_text(encoding="utf-8"))
    print(f"Answer key: {key_files[0].name}")
    print(f"  {len(key['words'])} words, {len(key['pairs'])} pairs, "
          f"{sum(1 for v in key['words'].values() if v == '?')} undecided\n")

    variable_key = generate_enhanced_variable_key(
        selected_variables=[config.var_name], is_merged=False, sample_size=config.sample_size)
    cache = CacheManager(CacheConfig())
    raw = cache.load_from_cache(config.filename, "data", variable_key, models.ResponseModel)
    out = cache.load_from_cache(config.filename, "preprocessed", variable_key, models.PreprocessedModel)
    if not raw or not out:
        raise SystemExit("Step 0 or step 1 cache missing — run step 1 first.")

    raw_by_id = {r.respondent_id: r.response for r in raw}
    verdicts, wrong, uncorrected = Counter(), [], []

    for item in out:
        if item.quality_filter:
            continue
        source, actual = raw_by_id.get(item.respondent_id), item.response
        if not isinstance(source, str) or not source.strip() or not isinstance(actual, str):
            continue

        want, scorable = expected_text(source, key)
        if not scorable:
            verdicts["excluded"] += 1
            continue
        if comparable(want) == comparable(normalize(source)):
            continue                                   # key asks for nothing here

        if comparable(actual) == comparable(want):
            verdicts["correct"] += 1
        elif comparable(actual) == comparable(normalize(source)):
            verdicts["uncorrected"] += 1
            uncorrected.append((source, want))
        else:
            verdicts["wrong"] += 1
            wrong.append((source, want, actual))

    scored = verdicts["correct"] + verdicts["uncorrected"] + verdicts["wrong"]
    print(f"{'':14s} {'n':>5}   {'aandeel':>8}")
    for name in ("correct", "uncorrected", "wrong"):
        print(f"  {name:12s} {verdicts[name]:>5}   {verdicts[name] / max(scored, 1) * 100:>7.1f}%")
    print(f"  {'excluded':12s} {verdicts['excluded']:>5}   (labelled '?')")
    print(f"\n  scored: {scored}")

    if wrong:
        print(f"\nWRONG ({len(wrong)}):")
        for source, want, actual in wrong:
            print(f"  in   : {source[:70]}")
            print(f"  want : {want[:70]}")
            print(f"  got  : {actual[:70]}\n")
    if uncorrected:
        print(f"UNCORRECTED ({len(uncorrected)}):")
        for source, want in uncorrected:
            print(f"  {source[:48]:50s} -> want: {want[:48]}")


if __name__ == "__main__":
    main()
