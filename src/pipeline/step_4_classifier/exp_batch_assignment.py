#%%
"""Batch-assignment experiment: does K=5 batching match per-idea quality?

Compares batch facet assignment (one menu, five ideas, Literal-schema
response) against the cached per-idea baseline on a seed-42 sample of 300
ideas. Disagreements go to a blind A/B LLM judge. Read-only w.r.t. the
cache; results land in exports/diagnostics/. Cost: < $0.50 per arm.

Gate (printed at the end):
  - <= 2% of sampled ideas end __UNASSIGNED__ after the escalation ladder
  - of the judged disagreements, <= 1/3 prefer the baseline assignment

Run from src/:
  python -m pipeline.step_4_classifier.exp_batch_assignment            # arm 1
  python -m pipeline.step_4_classifier.exp_batch_assignment --shortlist  # + arm 2
"""
import asyncio
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "src"))

import models
from config import get_reasoning_params
from test_data import TEST_DATA
from utils.cacheManager import CacheManager, generate_enhanced_variable_key
from utils.embedder import SharedEmbedder
from utils.llm import create_client, llm_create_async
from pipeline.step_4_classifier.config_classifier import CategoriesConfig
from pipeline.step_4_classifier.partition_labels import format_label
from pipeline.step_4_classifier.prompts_classifier import (
    DiscoveredFacet,
    FacetAssignmentResult,
    build_batch_facet_assignment_model,
    build_facet_assignment_prompt_batch,
    build_facet_assignment_prompt_single,
)
from pipeline.step_4_classifier.view_shortlist_recall import facet_card_text

SAMPLE_SIZE = 300
BATCH_K = 5
SHORTLIST_K = 10
SEED = 42
UNASSIGNED = "__UNASSIGNED__"


class JudgeVerdict(BaseModel):
    """Blind A/B verdict on which facet fits the idea better."""
    verdict: Literal["A", "B", "equal"] = Field(
        ..., description="A if facet A fits the idea better, B if facet B does, "
                         "equal if both fit about equally well")


def build_judge_prompt(idea_label: str, domain_name: str, domain_definition: str,
                       card_a: str, card_b: str, language: str) -> str:
    return f"""You are a qualitative coding auditor. Two candidate facets are proposed for one survey response idea. Judge which facet fits the idea better, based only on the idea text and the facet cards.

<domain_context>
Domain: {domain_name} -- {domain_definition}
Language: {language}
</domain_context>

<idea>
{idea_label}
</idea>

<facet_A>
{card_a}
</facet_A>

<facet_B>
{card_b}
</facet_B>

Answer "A" or "B" for the better-fitting facet, or "equal" when both fit about equally well.

Provide your output as valid JSON following the response schema provided."""


def facets_as_models(raw_facets: List) -> List[DiscoveredFacet]:
    out = []
    for f in raw_facets:
        d = f if isinstance(f, dict) else f.model_dump()
        out.append(DiscoveredFacet(
            facet_name=d.get("facet_name", ""),
            facet_description=d.get("facet_description", ""),
            inclusion_rule=d.get("inclusion_rule", "") or "",
            exclusion_rule=d.get("exclusion_rule", "") or "",
            example_observations=d.get("example_observations") or [""],
            axis=d.get("axis", "") or "",
        ))
    return out


async def assign_single(client, model, config, prompt_kwargs) -> Optional[FacetAssignmentResult]:
    prompt = build_facet_assignment_prompt_single(**prompt_kwargs)
    try:
        return await llm_create_async(
            client, model, prompt,
            response_model=FacetAssignmentResult,
            temperature=config.qr_temperature,
            max_tokens=config.qr_max_tokens_facet_assignment,
            **get_reasoning_params(model, phase="classifier_p4"),
        )
    except Exception:
        return None


async def main() -> None:
    shortlist_enabled = "--shortlist" in sys.argv

    filename = TEST_DATA.filename
    variable_key = generate_enhanced_variable_key(
        selected_variables=[TEST_DATA.var_name], is_merged=False,
        sample_size=TEST_DATA.sample_size,
    )
    cache_manager = CacheManager()
    taxonomy = cache_manager.load_metadata_from_cache(
        filename=filename, step="taxonomy", variable_key=variable_key,
        model_cls=models.TaxonomyResultsCache,
    )
    ideas_models = cache_manager.load_from_cache(
        filename, "extracted_ideas", variable_key, models.IdeasExtractedModel,
    )
    if taxonomy is None or not ideas_models:
        raise SystemExit(
            "taxonomy of extracted_ideas cache ontbreekt — draai eerst steps 3+4")

    idea_by_id = {
        idea.idea_id: idea
        for m in ideas_models for idea in (m.response_ideas or [])
    }
    meta = cache_manager.load_metadata_from_cache(
        filename=filename, step="extracted_ideas", variable_key=variable_key,
        model_cls=models.ExtractionMetadata,
    )
    survey_question = (meta.var_lab if meta and meta.var_lab else "") or ""
    language = (meta.lang if meta and meta.lang else "") or "nl-NL"

    domains: Dict[str, dict] = {}
    pool: List[Tuple[str, str]] = []
    for domain_name in sorted(taxonomy.partition_results):
        result = taxonomy.partition_results[domain_name]
        facets = facets_as_models(result.facets)
        if len(facets) < 2:
            continue
        description = ""
        for part in taxonomy.partition_set.partitions:
            if part.partition_name == domain_name:
                description = part.inclusion_definition
                break
        valid = {
            idea_id: facet_name
            for idea_id, facet_name in result.facet_assignments.items()
            if idea_id in idea_by_id
            and facet_name in {f.facet_name for f in facets}
        }
        if not valid:
            continue
        domains[domain_name] = {
            "facets": facets, "definition": description, "baseline": valid,
        }
        pool.extend((domain_name, idea_id) for idea_id in valid)

    rng = random.Random(SEED)
    sample = rng.sample(pool, min(SAMPLE_SIZE, len(pool)))
    by_domain: Dict[str, List[str]] = defaultdict(list)
    for domain_name, idea_id in sample:
        by_domain[domain_name].append(idea_id)

    config = CategoriesConfig()
    assign_model = config.qr_model_p4
    judge_model = config.qr_model_p5
    client = create_client(assign_model, async_mode=True)
    judge_client = create_client(judge_model, async_mode=True)
    semaphore = asyncio.Semaphore(8)

    shortlist_cards: Dict[str, np.ndarray] = {}
    idea_embeddings: Dict[str, np.ndarray] = {}
    if shortlist_enabled:
        embedder = SharedEmbedder()
        for domain_name, idea_ids in by_domain.items():
            info = domains[domain_name]
            cards = await embedder.embed_texts(
                [facet_card_text(f.model_dump()) for f in info["facets"]])
            labels = [format_label(idea_by_id[i], "ladder", "") for i in idea_ids]
            ideas_emb = await embedder.embed_texts(labels)
            cards_n = cards / np.linalg.norm(cards, axis=1, keepdims=True)
            ideas_n = ideas_emb / np.linalg.norm(ideas_emb, axis=1, keepdims=True)
            shortlist_cards[domain_name] = cards_n
            for row, idea_id in enumerate(idea_ids):
                idea_embeddings[idea_id] = ideas_n[row]

    async def run_batch(domain_name: str, idea_ids: List[str]) -> Dict[str, dict]:
        info = domains[domain_name]
        facets = info["facets"]
        if shortlist_enabled:
            keep = set()
            cards_n = shortlist_cards[domain_name]
            for idea_id in idea_ids:
                sims = idea_embeddings[idea_id] @ cards_n.T
                keep.update(np.argsort(-sims)[:SHORTLIST_K].tolist())
            facets = [facets[i] for i in sorted(keep)]
        facet_ids = [f"F{i}" for i in range(1, len(facets) + 1)]
        id_to_name = dict(zip(facet_ids, (f.facet_name for f in facets)))
        labels = [(i, format_label(idea_by_id[i], "ladder", "")) for i in idea_ids]

        prompt = build_facet_assignment_prompt_batch(
            survey_question=survey_question, language=language,
            dataset_context_section="", domain_name=domain_name,
            domain_definition=info["definition"], facets=facets, ideas=labels,
        )
        response_model = build_batch_facet_assignment_model(facet_ids, idea_ids)

        results: Dict[str, dict] = {}
        need_single: List[str] = []
        async with semaphore:
            try:
                response = await llm_create_async(
                    client, assign_model, prompt,
                    response_model=response_model,
                    temperature=config.qr_temperature,
                    max_tokens=config.qr_max_tokens_facet_assignment * 2,
                    **get_reasoning_params(assign_model, phase="classifier_p4"),
                )
            except Exception:
                response = None
        if response is None:
            need_single = list(idea_ids)
        else:
            seen: Dict[str, list] = defaultdict(list)
            for item in response.assignments:
                seen[item.idea_id].append(item)
            for idea_id in idea_ids:
                items = seen.get(idea_id, [])
                if len(items) == 1 and items[0].assigned_facet_id != "F_NONE":
                    item = items[0]
                    results[idea_id] = {
                        "facet": id_to_name[item.assigned_facet_id],
                        "valence": item.valence, "confidence": item.confidence,
                        "route": "batch",
                    }
                else:
                    need_single.append(idea_id)

        for idea_id in need_single:
            full = domains[domain_name]["facets"]
            single_ids = {f"F{i}": f.facet_name for i, f in enumerate(full, 1)}
            async with semaphore:
                single = await assign_single(client, assign_model, config, dict(
                    survey_question=survey_question, language=language,
                    dataset_context_section="", domain_name=domain_name,
                    domain_definition=domains[domain_name]["definition"],
                    facets=full,
                    idea_label=format_label(idea_by_id[idea_id], "ladder", ""),
                ))
            name = single_ids.get(single.assigned_facet_id) if single else None
            results[idea_id] = {
                "facet": name or UNASSIGNED,
                "valence": single.valence if single else "0",
                "confidence": single.confidence if single else 0.0,
                "route": "escalated_single" if name else "unassigned",
            }
        return results

    batch_jobs = []
    for domain_name, idea_ids in by_domain.items():
        for start in range(0, len(idea_ids), BATCH_K):
            batch_jobs.append(run_batch(domain_name, idea_ids[start:start + BATCH_K]))
    batch_results: Dict[str, dict] = {}
    for partial in await asyncio.gather(*batch_jobs):
        batch_results.update(partial)

    async def judge(domain_name: str, idea_id: str, batch_facet: str,
                    baseline_facet: str) -> str:
        info = domains[domain_name]
        cards = {f.facet_name: facet_card_text(f.model_dump()) for f in info["facets"]}
        order = random.Random(idea_id)
        pair = [("batch", batch_facet), ("baseline", baseline_facet)]
        order.shuffle(pair)
        (side_a, facet_a), (side_b, facet_b) = pair
        prompt = build_judge_prompt(
            idea_label=format_label(idea_by_id[idea_id], "ladder", ""),
            domain_name=domain_name, domain_definition=info["definition"],
            card_a=cards.get(facet_a, facet_a), card_b=cards.get(facet_b, facet_b),
            language=language,
        )
        async with semaphore:
            try:
                verdict = await llm_create_async(
                    judge_client, judge_model, prompt,
                    response_model=JudgeVerdict, temperature=0.0, max_tokens=1000,
                    **get_reasoning_params(judge_model, phase="classifier_p5"),
                )
            except Exception:
                return "judge_failed"
        if verdict.verdict == "equal":
            return "equal"
        return side_a if verdict.verdict == "A" else side_b

    disagreements = [
        (domain_name, idea_id, batch_results[idea_id]["facet"],
         domains[domain_name]["baseline"][idea_id])
        for domain_name, idea_id in sample
        if batch_results[idea_id]["facet"] not in (
            UNASSIGNED, domains[domain_name]["baseline"][idea_id])
    ]
    verdicts = await asyncio.gather(*[
        judge(d, i, bf, blf) for d, i, bf, blf in disagreements])

    rows = []
    verdict_by_id = {i: v for (_, i, _, _), v in zip(disagreements, verdicts)}
    for domain_name, idea_id in sample:
        entry = batch_results[idea_id]
        rows.append({
            "domain": domain_name, "idea_id": idea_id,
            "label": format_label(idea_by_id[idea_id], "ladder", ""),
            "baseline_facet": domains[domain_name]["baseline"][idea_id],
            "batch_facet": entry["facet"], "route": entry["route"],
            "valence": entry["valence"], "confidence": entry["confidence"],
            "judge": verdict_by_id.get(idea_id, ""),
        })

    n = len(sample)
    n_unassigned = sum(1 for r in rows if r["batch_facet"] == UNASSIGNED)
    n_agree = sum(1 for r in rows if r["batch_facet"] == r["baseline_facet"])
    counts = defaultdict(int)
    for v in verdicts:
        counts[v] += 1
    n_judged = sum(v for k, v in counts.items() if k != "judge_failed")

    out_dir = project_root / "exports" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    arm = "shortlist" if shortlist_enabled else "full-menu"
    out_path = out_dir / f"2026-08-05-batch-assignment-experiment-{arm}.json"
    out_path.write_text(json.dumps({
        "arm": arm, "sample_size": n, "batch_k": BATCH_K, "seed": SEED,
        "model": assign_model, "judge_model": judge_model,
        "agreement": n_agree / n if n else 0.0,
        "unassigned": n_unassigned,
        "judge_counts": dict(counts),
        "rows": rows,
    }, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== batch-assignment experiment ({arm}) ===")
    print(f"sample: {n} ideeen | agreement met baseline: {n_agree / n * 100:.1f}%")
    print(f"routes: batch={sum(1 for r in rows if r['route'] == 'batch')}, "
          f"escalated={sum(1 for r in rows if r['route'] == 'escalated_single')}, "
          f"unassigned={n_unassigned}")
    print(f"disagreements: {len(disagreements)} | judge: {dict(counts)}")
    gate_a = n_unassigned / n <= 0.02 if n else False
    gate_b = (counts["baseline"] / n_judged <= 1 / 3) if n_judged else True
    print(f"GATE a (<=2% unassigned): {'PASS' if gate_a else 'FAIL'}")
    print(f"GATE b (judge kiest <=1/3 baseline): {'PASS' if gate_b else 'FAIL'}")
    print(f"resultaat: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
