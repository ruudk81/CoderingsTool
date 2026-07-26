"""Task 6 — orchestrator for the step-5 experiment.

Wires phases 1-5 (phenomenon_clusterer, direction_rules, judgments, assembler)
into a single run: load inputs -> attribute centroids -> phenomenon discovery
-> membership votes on ambiguous attributes -> per-cluster direction
resolution -> noise votes on clusters that need one -> code plans -> naming
(single call per code, with a case-insensitive collision re-ask) -> assemble
-> save -> scorecard.

`run_experiment(filename, var_name, sample_size, llm_call=None, embedder=None)`
is the production entry point (loads real caches via `data_io.load_inputs`
and the taxonomy cache's `partition_set`). `run_from_inputs(...)` is the
injectable orchestration core it delegates to — tests call it directly with
synthetic `ExperimentInputs`/`partition_set`, bypassing the cache entirely.

`llm_call=None` builds the production callables via `judgments.make_llm_call`:
model_key "code_assignment" for the membership/noise votes, "codegen_p8" for
naming. A single injected `llm_call` (as fake-LLM tests do) serves all three
call sites — `vote()`'s callers and the direct naming call all share the same
`(prompt, response_model)` signature, so one fake that switches on
`response_model` is enough.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from config import MISCELLANEOUS_CODE_LABELS
from models import CodingResultsCache, DomainSet, TaxonomyResultsCache
from identity import ensure_codebook_ids
from utils.cacheManager import CacheManager
from utils.embedder import SharedEmbedder
from pipeline.step_5_codeGenerator.config_codeGenerator import CodebookConfig
from pipeline.step_5_codeGenerator.codebook_verifier import collect_taxonomy_attributes

from pipeline.step_5_codeGenerator_experiment.data_io import ExperimentInputs, load_inputs
from pipeline.step_5_codeGenerator_experiment.phenomenon_clusterer import (
    ClusterResult, attribute_centroids, discover_phenomena, missing_attributes,
)
from pipeline.step_5_codeGenerator_experiment.direction_rules import codes_for, resolve_direction
from pipeline.step_5_codeGenerator_experiment.judgments import (
    CodeNaming, MembershipVote, NoiseVote,
    make_llm_call, membership_prompt, naming_prompt, noise_prompt, vote,
)
from pipeline.step_5_codeGenerator_experiment.assembler import (
    Decision, assemble_codebook, run_scorecard_on, save_experiment,
)

_BASELINE_CONFIG = CodebookConfig()
MAX_SAMPLES = 8


# =============================================================================
# Input helpers — evidence lookup for prompt builders
# =============================================================================
def _load_partition_set(filename: str, variable_key: str) -> DomainSet:
    """Load `partition_set` from the taxonomy cache — the one field
    `data_io.load_inputs` deliberately does not carry (it only reads
    `partition_results`). Same load route: same `CacheManager`, same
    variable_key (reused from `ExperimentInputs.variable_key` rather than
    recomputed)."""
    cm = CacheManager()
    tax = cm.load_metadata_from_cache(filename, "taxonomy", variable_key, TaxonomyResultsCache)
    if tax is None:
        raise RuntimeError("no taxonomy cache — run step 4 first")
    return tax.partition_set


def _attribute_descriptions(partition_results: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for r in partition_results.values():
        for attrs in (getattr(r, "attributes", None) or {}).values():
            for a in attrs:
                name = a.get("attribute_name")
                if name and name not in out:
                    out[name] = a.get("attribute_description", "") or ""
    return out


def _idea_valence_map(partition_results: Dict[str, Any]) -> Dict[str, str]:
    """idea_id -> "+"/"-" (or absent for neutral), across all domains."""
    out: Dict[str, str] = {}
    for r in partition_results.values():
        out.update(getattr(r, "attribute_valence", None) or {})
    return out


def _ideas_for(idea_assignments: Dict[str, str], attrs) -> List[str]:
    attr_set = {attrs} if isinstance(attrs, str) else set(attrs)
    return sorted(i for i, a in idea_assignments.items() if a in attr_set)


def _samples_for_attr(inputs: ExperimentInputs, attr: str) -> List[str]:
    ids = _ideas_for(inputs.idea_assignments, attr)
    return [inputs.idea_texts[i] for i in ids if i in inputs.idea_texts][:MAX_SAMPLES]


def _pole_texts(
    inputs: ExperimentInputs, valence_map: Dict[str, str], members: List[str],
) -> Tuple[List[str], List[str]]:
    """Positive/negative statement texts for a cluster's members (noise check)."""
    pos, neg = [], []
    for i in _ideas_for(inputs.idea_assignments, members):
        text = inputs.idea_texts.get(i)
        if text is None:
            continue
        v = valence_map.get(i)
        if v == "+":
            pos.append(text)
        elif v == "-":
            neg.append(text)
    return pos[:MAX_SAMPLES], neg[:MAX_SAMPLES]


def _samples_per_pole_for_naming(
    inputs: ExperimentInputs, valence_map: Dict[str, str], members: List[str], valence: str,
    neutral_third: bool = False,
) -> Dict[str, List[str]]:
    """Per-member sample texts restricted to this code's own valence bucket.

    "neutral" has two distinct meanings depending on `neutral_third`:
    - False (no split, or a two-way split with no middle bucket): this
      "neutral" code IS the whole dimensional phenomenon — it gets every
      idea as naming evidence, regardless of valence.
    - True (a three-way split's middle bucket): this "neutral" code covers
      only the ideas WITHOUT +/- valence — the +/- ideas are naming
      evidence for the positive/negative codes, not this one.
    """
    out: Dict[str, List[str]] = {}
    for m in members:
        texts = []
        for i in _ideas_for(inputs.idea_assignments, m):
            text = inputs.idea_texts.get(i)
            if text is None:
                continue
            if valence == "neutral":
                if neutral_third and valence_map.get(i) in ("+", "-"):
                    continue
                texts.append(text)
            else:
                v = valence_map.get(i)
                bucket = "positive" if v == "+" else "negative" if v == "-" else None
                if bucket == valence:
                    texts.append(text)
        out[m] = texts[:MAX_SAMPLES]
    return out


async def _fallback_embeddings(inputs: ExperimentInputs, embedder) -> Dict[str, List[float]]:
    """Compute idea embeddings when nothing was reused from a baseline cache.

    Reuses the baseline's embedding model (`CodebookConfig.embedding_model`).
    `data_io.load_inputs` has already reduced each idea to a single text
    string (instance-or-idea) — there is no upstream code_source formatting
    left to replay, so that text is embedded as-is."""
    if not inputs.idea_texts:
        return {}
    ids = sorted(inputs.idea_texts)
    texts = [inputs.idea_texts[i] for i in ids]
    emb = embedder or SharedEmbedder(model=_BASELINE_CONFIG.embedding_model)
    vectors = await emb.embed_texts(texts)
    return {idea_id: vectors[idx].tolist() for idx, idea_id in enumerate(ids)}


# =============================================================================
# Orchestration core — injectable, no cache access
# =============================================================================
async def run_from_inputs(
    inputs: ExperimentInputs,
    partition_set: DomainSet,
    filename: str,
    llm_call: Optional[Callable[[str, type], Any]] = None,
    embedder=None,
    project_root: Optional[Path] = None,
) -> CodingResultsCache:
    """Run phases 1-5 against already-loaded inputs. No `CacheManager`/disk
    access of its own (`save_experiment` does the one real write, patchable
    via `project_root` + monkeypatching `assembler.CacheManager`)."""
    decisions: List[Decision] = []

    if not inputs.idea_embeddings:
        inputs.idea_embeddings = await _fallback_embeddings(inputs, embedder)

    membership_llm = llm_call or make_llm_call("code_assignment", phase="membership")
    noise_llm = llm_call or make_llm_call("code_assignment", phase="noise")
    naming_llm = llm_call or make_llm_call("codegen_p8", phase="naming")

    # --- Phase 1: centroids + phenomenon discovery ---
    centroids = attribute_centroids(inputs.idea_embeddings, inputs.idea_assignments)
    missing = missing_attributes(inputs.idea_assignments, centroids)
    for attr in missing:
        decisions.append(Decision(
            phase="clustering", subject=attr, outcome="routed_to_overig",
            evidence={"reason": "no embeddings"},
        ))

    # Taxonomy attributes with ZERO idea assignments never appear in
    # `idea_assignments.values()` at all, so `missing_attributes()` (which
    # only looks at assigned-but-uncentered attributes) can't see them
    # either — they would otherwise never be placed anywhere (not clustered,
    # not routed), showing up as an orphan attribute in the scorecard. Route
    # them to Overig the same way, with their own reason.
    taxonomy_attrs = collect_taxonomy_attributes(inputs.partition_results)
    unassigned = sorted(set(taxonomy_attrs) - set(centroids) - set(missing))
    for attr in unassigned:
        decisions.append(Decision(
            phase="clustering", subject=attr, outcome="routed_to_overig",
            evidence={"reason": "no assigned ideas"},
        ))
    overig_routed = missing + unassigned

    cluster_result: ClusterResult = discover_phenomena(centroids)
    attr_descriptions = _attribute_descriptions(inputs.partition_results)

    # --- Phase 2: membership votes for ambiguous attributes ---
    for attr in list(cluster_result.ambiguous):
        own = cluster_result.labels[attr]
        neighbor = cluster_result.neighbor.get(attr)
        if neighbor is None or neighbor == own:
            continue
        group_a = [m for m in cluster_result.clusters.get(own, []) if m != attr]
        group_b = cluster_result.clusters.get(neighbor, [])
        samples = _samples_for_attr(inputs, attr)
        definition = attr_descriptions.get(attr, "")

        outcome = await vote(
            build_prompt=lambda i, a=attr, d=definition, s=samples, ga=group_a, gb=group_b: membership_prompt(
                attr=a, definition=d, samples=s,
                group_a=", ".join(ga) or "(none)", group_b=", ".join(gb) or "(none)",
                language=inputs.language, vote_idx=i,
            ),
            response_model=MembershipVote,
            llm_call=membership_llm,
            majority_key=lambda v: v.choice,
        )
        moved = False
        if outcome.majority == "B":
            cluster_result.clusters[own] = [m for m in cluster_result.clusters[own] if m != attr]
            cluster_result.clusters.setdefault(neighbor, []).append(attr)
            cluster_result.labels[attr] = neighbor
            moved = True
        decisions.append(Decision(
            phase="membership", subject=attr,
            outcome="moved_to_neighbor" if moved else "kept",
            evidence={"own_cluster": own, "neighbor_cluster": neighbor, "failed": outcome.failed},
            votes={"choices": [v.choice for v in outcome.votes]},
            is_borderline=moved or not outcome.unanimous,
        ))

    # --- Phase 3+4: direction resolution + noise check per cluster ---
    valence_map = _idea_valence_map(inputs.partition_results)
    total_assigned = len(inputs.idea_assignments)
    code_plans: Dict[int, List[dict]] = {}
    naming_targets: List[Tuple[int, str]] = []
    neutral_third_by_label: Dict[int, bool] = {}

    for label in sorted(cluster_result.clusters):
        members = cluster_result.clusters[label]
        if not members:
            continue
        decision = resolve_direction(members, inputs.attr_valence, total_assigned)
        decisions.append(Decision(
            phase="direction", subject=f"cluster {label}", outcome=decision.outcome,
            evidence={"pos": decision.pos, "neu": decision.neu, "neg": decision.neg,
                      "floor": decision.floor, "members": members},
        ))

        split = False
        if decision.outcome == "needs_noise_check":
            pos_texts, neg_texts = _pole_texts(inputs, valence_map, members)
            noise_outcome = await vote(
                build_prompt=lambda i, pt=pos_texts, nt=neg_texts, m=members: noise_prompt(
                    phenomenon_desc=", ".join(m), pos_texts=pt, neg_texts=nt,
                    language=inputs.language, vote_idx=i,
                ),
                response_model=NoiseVote,
                llm_call=noise_llm,
                majority_key=lambda v: v.genuine_opposition,
            )
            split = bool(noise_outcome.majority)
            decisions.append(Decision(
                phase="noise", subject=f"cluster {label}",
                outcome="split" if split else "dimensional",
                evidence={"failed": noise_outcome.failed},
                votes={"genuine": [v.genuine_opposition for v in noise_outcome.votes]},
                is_borderline=noise_outcome.majority is None or not noise_outcome.unanimous,
            ))

        plan = codes_for(decision, split)
        code_plans[label] = plan
        neutral_third_by_label[label] = split and decision.neutral_third
        for entry in plan:
            naming_targets.append((label, entry["valence"]))

    # --- Phase 5a: naming (one call per code, collision re-ask) ---
    # Seed with the reserved catch-all label (same source assembler.py reads
    # for the Overig code) so an LLM-named code can never silently collide
    # with it — the verifier exempts Overig from its own checks, so a
    # collision here would otherwise go undetected.
    overig_label = MISCELLANEOUS_CODE_LABELS.get(inputs.language, "Overig")
    namings: Dict[Tuple[int, str], CodeNaming] = {}
    used_names_lower: set = {overig_label.strip().lower()}

    for label, valence in naming_targets:
        members = cluster_result.clusters[label]
        samples_per_pole = _samples_per_pole_for_naming(
            inputs, valence_map, members, valence,
            neutral_third=neutral_third_by_label.get(label, False),
        )
        avoid_names = sorted({n.code_name for n in namings.values()} | {overig_label})

        prompt = naming_prompt(
            members=members, samples_per_pole=samples_per_pole, valence=valence,
            language=inputs.language, survey_question=inputs.survey_question,
            avoid_names=avoid_names, vote_idx=0,
        )
        naming = await naming_llm(prompt, CodeNaming)

        if naming.code_name.strip().lower() in used_names_lower:
            retry_prompt = naming_prompt(
                members=members, samples_per_pole=samples_per_pole, valence=valence,
                language=inputs.language, survey_question=inputs.survey_question,
                avoid_names=avoid_names + [naming.code_name], vote_idx=1,
            )
            naming = await naming_llm(retry_prompt, CodeNaming)
            decisions.append(Decision(
                phase="naming", subject=f"cluster {label} {valence}",
                outcome="renamed_after_collision", evidence={"final_name": naming.code_name},
            ))

        namings[(label, valence)] = naming
        used_names_lower.add(naming.code_name.strip().lower())
        decisions.append(Decision(
            phase="naming", subject=naming.code_name, outcome="named",
            evidence={"cluster": label, "valence": valence},
        ))

    # --- Phase 5b: assemble, route missing-embedding attributes to Overig ---
    cache = assemble_codebook(
        inputs=inputs, cluster_result=cluster_result, code_plans=code_plans,
        namings=namings, decisions=decisions, partition_set=partition_set,
    )
    if overig_routed and cache.raw_codes:
        overig = cache.raw_codes[-1]
        existing = set(overig.get("source_attributes") or [])
        added = [a for a in overig_routed if a not in existing]
        if added:
            overig["source_attributes"] = list(overig.get("source_attributes") or []) + added
            overig["source_attribute_ids"] = []  # force ensure_codebook_ids to re-resolve
            ensure_codebook_ids(cache)

    save_experiment(cache, filename, inputs.variable_key, decisions, project_root=project_root)
    run_scorecard_on(cache, inputs.partition_results)
    return cache


# =============================================================================
# Production entry point
# =============================================================================
async def run_experiment(
    filename: str,
    var_name: str,
    sample_size,
    llm_call: Optional[Callable[[str, type], Any]] = None,
    embedder=None,
) -> CodingResultsCache:
    """Load real inputs (`data_io.load_inputs` + the taxonomy cache's
    `partition_set`) and run the full experiment. `llm_call=None` uses the
    production judgments (`make_llm_call`); `embedder=None` uses
    `SharedEmbedder` for any embedding fallback."""
    inputs = load_inputs(filename, var_name, sample_size)
    partition_set = _load_partition_set(filename, inputs.variable_key)
    return await run_from_inputs(inputs, partition_set, filename, llm_call=llm_call, embedder=embedder)


# =============================================================================
# CLI — mirrors run_codeGenerator.py's __main__ (TEST_DATA + token_tracker)
# =============================================================================
if __name__ == "__main__":
    from test_data import TEST_DATA
    from utils.llm import token_tracker

    token_tracker.reset()
    cache = asyncio.run(run_experiment(TEST_DATA.filename, TEST_DATA.var_name, TEST_DATA.sample_size))

    if token_tracker.call_count > 0:
        print(token_tracker.get_summary())
