#%%

"""Where does the run-to-run variance in the domain set come from?

Step 3 produces a different L2 partition on every run (ARI ~0.72 at best, see
dev/WORK.md). Two stages can be responsible and they need different fixes:

  discovery      each chunk sees a different slice of the data, so each proposes a
                 different set. Eleven chunks, eleven plausible answers.
  consolidation  one call merges those proposals into the final set.

This isolates them. Discovery runs once and its proposals are frozen to disk; then
only consolidation is repeated N times on that identical input. Everything else is
already deterministic — the grounding sample is drawn from a seeded RNG, and the
chunk boundaries are a fixed slicing of the response order — so whatever varies here
is consolidation itself.

  spread across the N runs  ->  consolidation is the source, and it is the cheap one
                                to fix (one prompt, one call)
  N near-identical results  ->  consolidation is stable and the spread arrives at its
                                door. A sharper consolidation prompt cannot help;
                                what is needed is consensus across whole runs.

Reads context and dimension from the existing step 3 metadata cache, so it costs one
discovery pass (~11 calls) the first time and N consolidation calls per invocation.
Both are cheap: the whole context+discovery phase of a production run is about $0.02.

Usage, from src/:
    python -m pipeline.step_3_ideaExtractor.exp_consolidation_variance
    python -m pipeline.step_3_ideaExtractor.exp_consolidation_variance --refresh
"""

import asyncio
import json
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Dict, List

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from config import get_step_model, get_reasoning_params
from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from utils.llm import create_client, llm_create_async
import models

from test_data import TEST_DATA

from .dimension_data import get_dimension
from .ideaExtractor import IdeaExtractor, GENERIC_SPECIFIER_CHUNK_SIZE, SAMPLING_SEED
from .prompts_ideaExtractor import (
    build_domain_discovery_prompt,
    build_domain_consolidation_prompt,
    DomainChunkResponse,
    DomainConsolidatedResponse,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

N_CONSOLIDATIONS = 10
MAX_CONCURRENT = 5

FROZEN_FILE = project_root / "data" / "step3_frozen_chunk_domains.json"

MODEL = get_step_model("idea_extraction_taxonomy")

# Words that say nothing about which theme a label names. Only used for the crude
# overlap number at the end, never to decide anything.
STOPWORDS = {"en", "van", "de", "het", "een", "in", "op", "of", "als", "voor", "bij"}


# =============================================================================
# INPUT
# =============================================================================

def load_inputs():
    """Responses from step 2, context and dimension from step 3's metadata."""
    variable_key = generate_enhanced_variable_key(
        selected_variables=[TEST_DATA.var_name], is_merged=False,
        sample_size=TEST_DATA.sample_size)
    cm = CacheManager()

    filtered = cm.load_from_cache(TEST_DATA.filename, "quality_filter", variable_key,
                                  models.QualityFilteredModel)
    meta = cm.load_metadata_from_cache(TEST_DATA.filename, "extracted_ideas", variable_key,
                                       models.ExtractionMetadata)
    if not filtered or not meta:
        raise SystemExit("Need a step 2 cache and a step 3 metadata cache. Run both first.")

    responses = [r for r in filtered if not r.quality_filter]
    return responses, meta


def context_of(meta) -> Dict[str, str]:
    """The six specifiers, under the key names the prompt builders expect."""
    return {
        "domain": meta.sector, "entity": meta.entity, "topic": meta.topic,
        "perspective": meta.perspective, "intent": meta.intent,
    }


# =============================================================================
# STAGE 1 — DISCOVERY, RUN ONCE AND FROZEN
# =============================================================================

async def discover(responses, meta) -> List[Dict]:
    """One discovery call per chunk. Chunking is production's own function."""
    chunks = IdeaExtractor.build_domain_chunks(responses)
    dimension = get_dimension(meta.primary_dimension)
    ctx = context_of(meta)
    client = create_client(MODEL)
    gate = asyncio.Semaphore(MAX_CONCURRENT)

    async def one(chunk):
        prompt = build_domain_discovery_prompt(
            language=meta.lang, survey_question=meta.var_lab,
            chunk_responses="\n".join(f"- {r.response}" for r in chunk),
            chunk_size=len(chunk), perspective=ctx["perspective"], intent=ctx["intent"],
            sector=ctx["domain"], entity=ctx["entity"], topic=ctx["topic"],
            primary_dimension=meta.primary_dimension,
            primary_dimension_description=meta.primary_dimension_description,
            dimension=dimension)
        async with gate:
            res = await llm_create_async(
                client=client, model=MODEL, response_model=DomainChunkResponse,
                prompt=prompt, temperature=0.0,
                **get_reasoning_params(MODEL, phase="idea_extraction_taxonomy"))
        return [{"label": d.label, "definition": d.definition} for d in res.domains]

    print(f"Discovery over {len(chunks)} chunks "
          f"({sum(len(c) for c in chunks)} response slots, {len(responses)} responses)...")
    return await asyncio.gather(*(one(c) for c in chunks))


def frozen_proposals(responses, meta, refresh: bool) -> List[List[Dict]]:
    if FROZEN_FILE.exists() and not refresh:
        data = json.loads(FROZEN_FILE.read_text(encoding="utf-8"))
        print(f"Reusing frozen proposals from {FROZEN_FILE.name} "
              f"({len(data['chunks'])} chunks). --refresh to redo discovery.")
        return data["chunks"]

    chunks = asyncio.run(discover(responses, meta))
    FROZEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    FROZEN_FILE.write_text(json.dumps(
        {"filename": TEST_DATA.filename, "variable": TEST_DATA.var_name,
         "dimension": meta.primary_dimension, "chunks": chunks},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Frozen to {FROZEN_FILE.name}")
    return chunks


# =============================================================================
# STAGE 2 — CONSOLIDATION, REPEATED ON THAT SAME INPUT
# =============================================================================

async def consolidate_n_times(chunk_domains: List[List[Dict]], responses, meta,
                              n: int) -> List[DomainConsolidatedResponse]:
    """N consolidations of one identical prompt. Only the model varies.

    Returns the full responses, not just labels: `exp_menu_wording_variance.py`
    needs the definitions, boundary tests and exclusions, because those are what the
    extraction prompt actually shows the model.
    """
    dimension = get_dimension(meta.primary_dimension)
    ctx = context_of(meta)

    formatted = "\n\n".join(
        "Chunk {}:\n  Domains:\n{}".format(
            idx + 1,
            "\n".join(f'    - "{d["label"]}" — {d["definition"]}' for d in chunk))
        for idx, chunk in enumerate(chunk_domains))

    # Same seeded draw production makes, so the grounding sample is identical too.
    import random
    rng = random.Random(SAMPLING_SEED)
    grounding = rng.sample(responses, min(GENERIC_SPECIFIER_CHUNK_SIZE, len(responses)))

    prompt = build_domain_consolidation_prompt(
        language=meta.lang, survey_question=meta.var_lab, sector=ctx["domain"],
        entity=ctx["entity"], topic=ctx["topic"], perspective=ctx["perspective"],
        intent=ctx["intent"], primary_dimension=meta.primary_dimension,
        chunk_results=formatted, dimension=dimension,
        chunk_responses="\n".join(f"- {r.response}" for r in grounding))

    client = create_client(MODEL)
    gate = asyncio.Semaphore(MAX_CONCURRENT)

    async def one():
        async with gate:
            res = await llm_create_async(
                client=client, model=MODEL, response_model=DomainConsolidatedResponse,
                prompt=prompt, temperature=0.0,
                **get_reasoning_params(MODEL, phase="idea_extraction_taxonomy"))
        return res

    print(f"Consolidating {n}x on that identical input ({len(prompt)} chars)...")
    return await asyncio.gather(*(one() for _ in range(n)))


# =============================================================================
# REPORTING
# =============================================================================

def content_words(label: str) -> set:
    return {w for w in re.findall(r"[a-zà-ÿ]+", label.lower()) if w not in STOPWORDS}


def crude_overlap(a: List[str], b: List[str]) -> float:
    """Jaccard over the content words of two label sets.

    Deliberately crude and lexical: it is a reading aid for the printed sets below,
    not a criterion. Two runs that name the same theme differently ("Bankdiensten" /
    "Financiële activiteiten") score low here while being the same carving — which is
    exactly why the sets are printed in full and why the real stability number is the
    ARI in `measure_stability.py`, measured over respondents rather than labels.
    """
    wa = set().union(*(content_words(l) for l in a)) if a else set()
    wb = set().union(*(content_words(l) for l in b)) if b else set()
    return len(wa & wb) / len(wa | wb) if (wa or wb) else 1.0


def report(chunk_domains: List[List[Dict]], consolidations) -> None:
    print(f"\n{'=' * 72}\nDISCOVERY (frozen, one pass)\n{'=' * 72}")
    counts = [len(c) for c in chunk_domains]
    print(f"{len(chunk_domains)} chunks proposed {counts} domains "
          f"(min {min(counts)}, max {max(counts)})")
    for idx, chunk in enumerate(chunk_domains):
        print(f"  chunk {idx + 1:>2}: {[d['label'] for d in chunk]}")

    results = [[d.label for d in c.domains] for c in consolidations]
    print(f"\n{'=' * 72}\nCONSOLIDATION ({len(results)}x on that identical input)\n{'=' * 72}")
    sizes = Counter(len(r) for r in results)
    print(f"domain count per run: {dict(sorted(sizes.items()))}")
    for i, labels in enumerate(results, 1):
        print(f"  run {i:>2} ({len(labels)}): {labels}")

    pairs = [crude_overlap(a, b) for a, b in combinations(results, 2)]
    if pairs:
        print(f"\ncrude lexical overlap between runs: "
              f"min {min(pairs):.2f}  median {sorted(pairs)[len(pairs) // 2]:.2f}  "
              f"max {max(pairs):.2f}")
        print("  (a reading aid only — same carving under different names scores low)")

    print("\nRead it as: one repeated answer means consolidation is stable and the "
          "spread arrives from discovery.\nA different answer each time means "
          "consolidation is itself a source, and the cheaper one to fix.")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    refresh = "--refresh" in sys.argv
    responses, meta = load_inputs()
    print(f"dimension {meta.primary_dimension} | {len(responses)} meaningful responses")

    chunk_domains = frozen_proposals(responses, meta, refresh)
    results = asyncio.run(consolidate_n_times(chunk_domains, responses, meta,
                                              N_CONSOLIDATIONS))
    report(chunk_domains, results)


if __name__ == "__main__":
    main()
