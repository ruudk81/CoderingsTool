#%%
# Cell 0: Imports + configuration
"""
MECE Topic Extractor — Fine-grained topics grouped into themes.

Pipeline:
  Cell 0: Config
  Cell 1: Load data + generate clusters (quiet)
  Cell 2: (A) Extract fine-grained MECE topics per cluster
  Cell 3: (B) Group topics into thematic clusters via embeddings

Usage:
    Run cells sequentially in VS Code interactive mode.
"""

import io
import sys
import asyncio
import time
import pickle
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

import numpy as np
import nest_asyncio
from aiolimiter import AsyncLimiter

nest_asyncio.apply()

# Path setup
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "src" / "experiments"))

from experiments import models_exp as models
from utils.cacheManager import generate_enhanced_variable_key, CacheManager
from utils.llm import create_client, llm_create_async, create_embedding_client
from config import get_embedding_model_for_api

from experiments.step_5_clusterer_v2.clusterer_exp import Clusterer
from experiments.step_5_clusterer_v2.config_clusterer_exp import ClustererConfig
from experiments.step_5_clusterer_v2.prompts_exp import (
    CLUSTER_MECE_TOPIC_EXTRACTION_PROMPT,
    ClusterMECETopics,
)

# Dataset (from test_data.py)
try:
    from experiments.test_data import TEST_DATA
except ImportError:
    exp_root = Path(__file__).parent.parent
    if str(exp_root) not in sys.path:
        sys.path.insert(0, str(exp_root))
    from test_data import TEST_DATA

FILENAME = TEST_DATA.filename
VARIABLE = TEST_DATA.var_name
SAMPLE_SIZE = TEST_DATA.sample_size

# --- Config ---
TOTAL_SAMPLE_BUDGET = 30
LLM_MODEL = "gpt-4.1"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 4000
CONCURRENCY = 5
RPM_LIMIT = 30

PROBABILITY_BANDS = {
    "core":       (0.7, 1.01),
    "moderate":   (0.4, 0.7),
    "peripheral": (0.0, 0.4),
}
BAND_LABELS = {
    "core":       "core members (high confidence)",
    "moderate":   "boundary members (moderate confidence)",
    "peripheral": "peripheral members (low confidence)",
}


class TeeOutput:
    """Capture stdout while also printing to console."""

    def __init__(self, original_stdout):
        self.original_stdout = original_stdout
        self.buffer = io.StringIO()

    def write(self, text):
        self.original_stdout.write(text)
        self.buffer.write(text)

    def flush(self):
        self.original_stdout.flush()

    def get_output(self) -> str:
        return self.buffer.getvalue()


def save_results_to_file(output: str, base_name: str) -> Path:
    """Save all console output to a text file."""
    output_dir = project_root / "exports" / "mece_topic_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_str = str(SAMPLE_SIZE) if SAMPLE_SIZE else "full"
    date_str = datetime.now().strftime("%Y%m%d")
    output_filename = f"mece_topics_{base_name}_{VARIABLE}_{sample_str}_{date_str}.txt"
    output_path = output_dir / output_filename

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output)
    return output_path


#%% Cell 1: Load data + generate clusters (quiet)

# Activate TeeOutput only when running as script (not interactive cells)
_tee = None
if __name__ == "__main__":
    _tee = TeeOutput(sys.stdout)
    sys.stdout = _tee


def load_step4_embeddings():
    """Load Step 4 embeddings from cache."""
    variable_key = generate_enhanced_variable_key(
        selected_variables=[VARIABLE], is_merged=False, sample_size=SAMPLE_SIZE
    )
    cache_dir = project_root / "data" / "cache"
    base_name = Path(FILENAME).stem
    cache_path = cache_dir / f"005_embeddings_{base_name}_{variable_key}.pkl"

    if not cache_path.exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")

    with open(cache_path, "rb") as f:
        serializable_data = pickle.load(f)

    return [models.EmbeddingsModel.model_validate(item) for item in serializable_data]


def load_extraction_metadata():
    """Load ExtractionMetadata from cache."""
    variable_key = generate_enhanced_variable_key(
        selected_variables=[VARIABLE], is_merged=False, sample_size=SAMPLE_SIZE
    )
    cm = CacheManager()
    return cm.load_metadata_from_cache(
        filename=FILENAME, step="extracted_ideas",
        variable_key=variable_key, model_cls=models.ExtractionMetadata
    )


print("Loading data...")
embeddings_models = load_step4_embeddings()
extraction_metadata = load_extraction_metadata()
print(f"  {len(embeddings_models)} respondents loaded")

print("Generating clusters...")
config = ClustererConfig(
    generate_ctfidf=True,       # keywords used as LLM context in step A
    generate_llm_labels=False,  # we do our own topic extraction
    generate_mece_topics=False,
    verbose=False,              # suppress optimization logs
)
clusterer = Clusterer(embeddings_models, config=config, extraction_metadata=extraction_metadata)
clusterer.run()

unique_labels = sorted(set(clusterer._labels))
cluster_ids = [c for c in unique_labels if c >= 0]
n_noise = int(np.sum(clusterer._labels == -1))
n_total = len(clusterer._labels)
print(f"  {len(cluster_ids)} clusters, {n_noise} noise ({n_noise/n_total:.0%}), {n_total} ideas total")

# Shared cache variables (used by Cells 2 and 3)
variable_key = generate_enhanced_variable_key(
    selected_variables=[VARIABLE], is_merged=False, sample_size=SAMPLE_SIZE
)
var_lab = getattr(extraction_metadata, "var_lab", None)
cache_manager = CacheManager()
cache_dir = project_root / "data" / "cache"
base_name = Path(FILENAME).stem

# Show detailed cluster information
clusterer.print_all_clusters(n_samples=10)


#%% Cell 2: (A) Extract fine-grained MECE topics per cluster

@dataclass
class SampledIdea:
    idea_text: str
    node: str
    semantic_category: str
    root: str
    probability: float
    band: str


def stratified_sample_cluster(
    clusterer: Clusterer, cluster_id: int, budget: int = TOTAL_SAMPLE_BUDGET,
) -> List[SampledIdea]:
    """Sample ideas from a cluster stratified across probability bands."""
    cluster_mask = clusterer._labels == cluster_id
    cluster_indices = np.where(cluster_mask)[0]
    if len(cluster_indices) == 0:
        return []

    has_probs = (
        clusterer._hdbscan_model is not None
        and hasattr(clusterer._hdbscan_model, "probabilities_")
    )

    all_ideas: Dict[str, List[SampledIdea]] = {"core": [], "moderate": [], "peripheral": []}
    seen_texts = set()

    for global_idx in cluster_indices:
        prob = float(clusterer._hdbscan_model.probabilities_[global_idx]) if has_probs else 1.0
        resp_idx, idea_idx = clusterer._idea_indices[global_idx]
        idea_obj = clusterer._input_list[resp_idx].response_ideas[idea_idx]
        idea_text = clusterer._idea_texts[global_idx]

        if idea_text in seen_texts:
            continue
        seen_texts.add(idea_text)

        band = "core"
        if has_probs:
            for band_name, (low, high) in PROBABILITY_BANDS.items():
                if low <= prob < high:
                    band = band_name
                    break

        all_ideas[band].append(SampledIdea(
            idea_text=idea_text,
            node=getattr(idea_obj, "node", "") or "",
            semantic_category=getattr(idea_obj, "semantic_category", "") or "",
            root=getattr(idea_obj, "root", "") or "",
            probability=prob, band=band,
        ))

    for band in all_ideas:
        all_ideas[band].sort(key=lambda x: -x.probability)

    non_empty = {k: v for k, v in all_ideas.items() if v}
    if not non_empty:
        return []

    n_bands = len(non_empty)
    per_band = budget // n_bands
    remainder = budget % n_bands

    band_budgets = {}
    rem_idx = 0
    for band_name in ["core", "moderate", "peripheral"]:
        if band_name in non_empty:
            band_budgets[band_name] = per_band + (1 if rem_idx < remainder else 0)
            rem_idx += 1

    result = []
    for band_name in ["core", "moderate", "peripheral"]:
        if band_name not in non_empty:
            continue
        ideas = non_empty[band_name]
        result.extend(ideas[:min(band_budgets[band_name], len(ideas))])
    return result


def format_stratified_sample(samples: List[SampledIdea]) -> str:
    sections = []
    for band_name in ["core", "moderate", "peripheral"]:
        band_items = [s for s in samples if s.band == band_name]
        if not band_items:
            continue
        lines = []
        for i, item in enumerate(band_items, 1):
            parts = [p for p in [item.semantic_category, item.node] if p]
            ontology_str = f"  [{' > '.join(parts)}]" if parts else ""
            lines.append(f"{i}. {item.idea_text}{ontology_str}")
        sections.append(f"{BAND_LABELS[band_name]}:\n" + "\n".join(lines))
    return "\n\n".join(sections)


def build_prompt_for_cluster(
    clusterer: Clusterer, cluster_id: int,
    extraction_metadata: models.ExtractionMetadata,
    keywords: Optional[Dict[int, List[Tuple[str, float]]]] = None,
) -> Tuple[List[SampledIdea], str]:
    samples = stratified_sample_cluster(clusterer, cluster_id)
    if not samples:
        return samples, ""

    stratified_text = format_stratified_sample(samples)
    total_size = int((clusterer._labels == cluster_id).sum())

    keywords_section = ""
    if keywords and cluster_id in keywords:
        kw_formatted = ", ".join(kw for kw, _ in keywords[cluster_id][:10])
        keywords_section = f"\n<statistical_keywords>\nThese terms statistically differentiate this cluster from others:\n{kw_formatted}\n</statistical_keywords>\n"

    dataset_context_section = ""
    if extraction_metadata:
        parts = []
        for field in ["domain", "entity", "topic", "perspective", "intent"]:
            val = getattr(extraction_metadata, field, "")
            if val:
                parts.append(f"{field.capitalize()}: {val}")
        if parts:
            dataset_context_section = "\n" + "\n".join(parts)

    taxonomy_context = ""
    tax_axis = getattr(extraction_metadata, "taxonomy_axis", None)
    tax_desc = getattr(extraction_metadata, "taxonomy_axis_description", None)
    if tax_axis:
        taxonomy_context = f"\n<taxonomy_context>\nPrimary coding dimension: {tax_axis}\nDefinition: {tax_desc or 'Not specified'}\nTopics MUST describe content within this dimension ONLY.\n</taxonomy_context>\n"

    prompt = CLUSTER_MECE_TOPIC_EXTRACTION_PROMPT.format(
        survey_question=getattr(extraction_metadata, "var_lab", "") or "",
        language="Dutch",
        dataset_context_section=dataset_context_section,
        taxonomy_context=taxonomy_context,
        cluster_id=cluster_id,
        total_cluster_size=total_size,
        sample_size=len(samples),
        keywords_section=keywords_section,
        stratified_sample_text=stratified_text,
    )
    return samples, prompt


async def extract_all_clusters_parallel(
    clusterer: Clusterer, cluster_ids: List[int],
    extraction_metadata: models.ExtractionMetadata,
    keywords: Optional[Dict[int, List[Tuple[str, float]]]] = None,
) -> Dict[int, ClusterMECETopics]:
    prompts: Dict[int, str] = {}
    for cid in cluster_ids:
        _, prompt = build_prompt_for_cluster(clusterer, cid, extraction_metadata, keywords)
        prompts[cid] = prompt

    client = create_client(model=LLM_MODEL, async_mode=True)
    semaphore = asyncio.Semaphore(CONCURRENCY)
    rate_limiter = AsyncLimiter(RPM_LIMIT, time_period=60)
    results: Dict[int, ClusterMECETopics] = {}

    async def process_one(cluster_id: int):
        prompt = prompts[cluster_id]
        if not prompt:
            results[cluster_id] = ClusterMECETopics(semantic_theme=f"Empty cluster {cluster_id}", topics=[])
            return
        async with semaphore:
            async with rate_limiter:
                try:
                    result = await llm_create_async(
                        client=client, model=LLM_MODEL, prompt=prompt,
                        response_model=ClusterMECETopics,
                        temperature=LLM_TEMPERATURE, max_tokens=LLM_MAX_TOKENS,
                    )
                    results[cluster_id] = result
                    n_ideas = int((clusterer._labels == cluster_id).sum())
                    print(f"  cluster {cluster_id} ({n_ideas} ideas) → {len(result.topics)} topics")
                except Exception as e:
                    print(f"  cluster {cluster_id} → FAILED: {type(e).__name__}: {e}")

    await asyncio.gather(*(process_one(cid) for cid in cluster_ids))
    return results


# --- Run extraction ---
all_keywords = clusterer.get_all_cluster_keywords()
keywords_for_llm = (all_keywords.get("mmr") or all_keywords.get("ctfidf")) if all_keywords else None

print(f"\n{'='*80}")
print(f"(A) EXTRACTING FINE-GRAINED TOPICS")
print(f"{'='*80}")
print(f"Extracting from {len(cluster_ids)} clusters ({LLM_MODEL}, {TOTAL_SAMPLE_BUDGET} samples/cluster)...")

start_time = time.time()
cluster_mece_results = asyncio.run(
    extract_all_clusters_parallel(clusterer, cluster_ids, extraction_metadata, keywords_for_llm)
)
elapsed = time.time() - start_time

# Collect all topics
all_topics = []
for cluster_id in sorted(cluster_mece_results.keys()):
    result = cluster_mece_results[cluster_id]
    for topic in result.topics:
        all_topics.append({
            "label": topic.topic_label,
            "definition": topic.inclusion_definition,
            "source_cluster_id": cluster_id,
        })

print(f"\nDone in {elapsed:.1f}s — {len(all_topics)} topics extracted from {len(cluster_mece_results)} clusters")

# --- Structured Phase A reporting ---
print(f"\n{'='*80}")
print(f"PHASE A RESULTS: {len(all_topics)} topics from {len(cluster_mece_results)} clusters")
print(f"{'='*80}")

for cluster_id in sorted(cluster_mece_results.keys()):
    result = cluster_mece_results[cluster_id]
    n_ideas = int((clusterer._labels == cluster_id).sum())

    print(f"\n{'─'*80}")
    print(f"CLUSTER {cluster_id} (n={n_ideas}) | Theme: {result.semantic_theme}")
    print(f"{'─'*80}")

    for j, topic in enumerate(result.topics, 1):
        print(f"\n  [{j}] {topic.topic_label}")
        print(f"      Definition: {topic.inclusion_definition}")
        if topic.key_expressions:
            expr_str = "; ".join(topic.key_expressions[:5])
            print(f"      Key expressions: {expr_str}")

print(f"\n{'='*80}")
print(f"PHASE A SUMMARY: {len(cluster_mece_results)} clusters -> {len(all_topics)} topics ({elapsed:.1f}s)")
print(f"{'='*80}")

# --- Cache Phase A results ---
phase_a_path = cache_dir / f"mece_phase_a_{base_name}_{variable_key}.pkl"
phase_a_serializable = {
    cid: result.model_dump() for cid, result in cluster_mece_results.items()
}
with open(phase_a_path, "wb") as f:
    pickle.dump(phase_a_serializable, f)
print(f"\nCACHED: Phase A results ({len(cluster_mece_results)} clusters) -> '{phase_a_path.name}'")


#%% Cell 3: (B) Consolidate topics via LLM map-reduce

from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

from experiments.step_5_clusterer_v2.prompts_exp import (
    MAP_CONSOLIDATION_PROMPT,
    REDUCE_CONSOLIDATION_PROMPT,
    MapBatchResult,
    ReduceResult,
    ConsolidatedTopic,
)


async def embed_topic_texts(texts: List[str]) -> np.ndarray:
    client = create_embedding_client(async_mode=True)
    model = get_embedding_model_for_api()
    response = await client.embeddings.create(input=texts, model=model)
    return np.array([item.embedding for item in response.data], dtype=np.float32)


def create_similarity_batches(
    topic_embeddings: np.ndarray,
    n_topics: int,
    max_batch_size: int = 20,
    min_similarity: float = 0.65,
) -> List[List[int]]:
    """Group topics into similarity-based batches (similar topics together)."""
    sim_matrix = sklearn_cosine_similarity(topic_embeddings)

    # Count neighbors per topic
    neighbor_counts = [
        int(np.sum(sim_matrix[i, :] >= min_similarity)) - 1
        for i in range(n_topics)
    ]
    sorted_indices = sorted(range(n_topics), key=lambda i: -neighbor_counts[i])

    batches = []
    assigned = set()

    for seed in sorted_indices:
        if seed in assigned:
            continue

        batch = [seed]
        assigned.add(seed)

        # Add similar topics to this batch
        # Sort candidates by similarity to seed (most similar first)
        candidates = [
            (i, sim_matrix[seed, i])
            for i in range(n_topics)
            if i not in assigned
        ]
        candidates.sort(key=lambda x: -x[1])

        for cand_idx, cand_sim in candidates:
            if cand_idx in assigned or len(batch) >= max_batch_size:
                break
            # Check similarity to ANY batch member
            if any(sim_matrix[cand_idx, m] >= min_similarity for m in batch):
                batch.append(cand_idx)
                assigned.add(cand_idx)

        batches.append(batch)

    # Merge tiny batches (<3 topics) into the nearest batch
    final_batches = []
    tiny = []
    for batch in batches:
        if len(batch) < 3:
            tiny.extend(batch)
        else:
            final_batches.append(batch)

    if tiny and final_batches:
        # Assign each tiny topic to the most similar batch
        for idx in tiny:
            best_batch = 0
            best_sim = -1
            for bi, batch in enumerate(final_batches):
                avg_sim = float(np.mean([sim_matrix[idx, m] for m in batch]))
                if avg_sim > best_sim:
                    best_sim = avg_sim
                    best_batch = bi
            final_batches[best_batch].append(idx)
    elif tiny:
        final_batches.append(tiny)

    return final_batches


def format_topics_for_prompt(indices: List[int], all_topics: List[Dict]) -> str:
    lines = []
    for i, idx in enumerate(indices, 1):
        t = all_topics[idx]
        lines.append(
            f"{i}. [Cluster {t['source_cluster_id']}] \"{t['label']}\"\n"
            f"   Definition: {t['definition']}"
        )
    return "\n\n".join(lines)


def format_map_outputs_for_reduce(map_results: List[MapBatchResult]) -> str:
    lines = []
    idx = 1
    for batch_result in map_results:
        for topic in batch_result.consolidated_topics:
            merged_str = ", ".join(f'"{l}"' for l in topic.merged_from)
            cluster_str = ", ".join(str(c) for c in sorted(topic.source_cluster_ids))
            lines.append(
                f"{idx}. \"{topic.topic_label}\"\n"
                f"   Definition: {topic.inclusion_definition}\n"
                f"   Merged from: [{merged_str}]\n"
                f"   Source clusters: [{cluster_str}]"
            )
            idx += 1
    return "\n\n".join(lines)


def build_context_sections(extraction_metadata):
    dataset_context = ""
    if extraction_metadata:
        parts = []
        for field in ["domain", "entity", "topic", "perspective", "intent"]:
            val = getattr(extraction_metadata, field, "")
            if val:
                parts.append(f"{field.capitalize()}: {val}")
        if parts:
            dataset_context = "\n" + "\n".join(parts)

    taxonomy_context = ""
    tax_axis = getattr(extraction_metadata, "taxonomy_axis", None)
    tax_desc = getattr(extraction_metadata, "taxonomy_axis_description", None)
    if tax_axis:
        taxonomy_context = (
            f"\n<taxonomy_context>\nPrimary coding dimension: {tax_axis}\n"
            f"Definition: {tax_desc or 'Not specified'}\n"
            f"Topics MUST describe content within this dimension ONLY.\n</taxonomy_context>\n"
        )

    survey_question = getattr(extraction_metadata, "var_lab", "") or ""
    return survey_question, dataset_context, taxonomy_context


async def run_map_phase(
    batches: List[List[int]], all_topics: List[Dict],
    extraction_metadata,
) -> List[MapBatchResult]:
    survey_q, dataset_ctx, taxonomy_ctx = build_context_sections(extraction_metadata)
    client = create_client(model=LLM_MODEL, async_mode=True)
    semaphore = asyncio.Semaphore(CONCURRENCY)
    rate_limiter = AsyncLimiter(RPM_LIMIT, time_period=60)
    results: List[Optional[MapBatchResult]] = [None] * len(batches)

    async def process_batch(batch_idx: int, batch_indices: List[int]):
        topics_list = format_topics_for_prompt(batch_indices, all_topics)
        prompt = MAP_CONSOLIDATION_PROMPT.format(
            survey_question=survey_q, language="Dutch",
            dataset_context_section=dataset_ctx, taxonomy_context=taxonomy_ctx,
            n_input_topics=len(batch_indices), topics_list=topics_list,
        )
        async with semaphore:
            async with rate_limiter:
                try:
                    result = await llm_create_async(
                        client=client, model=LLM_MODEL, prompt=prompt,
                        response_model=MapBatchResult,
                        temperature=LLM_TEMPERATURE, max_tokens=LLM_MAX_TOKENS,
                    )
                    results[batch_idx] = result
                    print(f"  batch {batch_idx+1}: {len(batch_indices)} → {len(result.consolidated_topics)} topics")
                except Exception as e:
                    print(f"  batch {batch_idx+1}: FAILED ({type(e).__name__}: {e})")
                    # Fallback: no merging
                    results[batch_idx] = MapBatchResult(consolidated_topics=[
                        ConsolidatedTopic(
                            topic_label=all_topics[i]["label"],
                            merged_from=[all_topics[i]["label"]],
                            inclusion_definition=all_topics[i]["definition"],
                            merge_rationale="standalone (error fallback)",
                            source_cluster_ids=[all_topics[i]["source_cluster_id"]],
                        ) for i in batch_indices
                    ])

    await asyncio.gather(*(process_batch(i, b) for i, b in enumerate(batches)))
    return [r for r in results if r is not None]


async def run_reduce_phase(
    map_results: List[MapBatchResult], extraction_metadata,
) -> ReduceResult:
    survey_q, dataset_ctx, taxonomy_ctx = build_context_sections(extraction_metadata)
    map_outputs_list = format_map_outputs_for_reduce(map_results)
    n_map_outputs = sum(len(r.consolidated_topics) for r in map_results)

    prompt = REDUCE_CONSOLIDATION_PROMPT.format(
        survey_question=survey_q, language="Dutch",
        dataset_context_section=dataset_ctx, taxonomy_context=taxonomy_ctx,
        n_map_outputs=n_map_outputs, map_outputs_list=map_outputs_list,
    )
    client = create_client(model=LLM_MODEL, async_mode=True)
    try:
        result = await llm_create_async(
            client=client, model=LLM_MODEL, prompt=prompt,
            response_model=ReduceResult,
            temperature=LLM_TEMPERATURE, max_tokens=8000,
        )
        print(f"  reduce: {n_map_outputs} → {len(result.consolidated_topics)} final topics")
        return result
    except Exception as e:
        print(f"  reduce FAILED ({type(e).__name__}: {e})")
        all_consolidated = []
        for r in map_results:
            all_consolidated.extend(r.consolidated_topics)
        return ReduceResult(consolidated_topics=all_consolidated)


# --- Main execution ---

print(f"\n{'='*80}")
print(f"(B) CONSOLIDATING TOPICS VIA LLM MAP-REDUCE")
print(f"{'='*80}")

n_topics = len(all_topics)

# 1. Embed for similarity-based batching
print(f"\nEmbedding {n_topics} topics...")
embed_texts_list = [f"{t['label']}: {t['definition']}" for t in all_topics]
topic_embeddings = asyncio.run(embed_topic_texts(embed_texts_list))

# 2. Create similarity-based batches
batches = create_similarity_batches(topic_embeddings, n_topics)
print(f"Created {len(batches)} batches: {[len(b) for b in batches]}")

# 3. Map phase (parallel)
print(f"\nMap phase ({len(batches)} batches)...")
map_start = time.time()
map_results = asyncio.run(run_map_phase(batches, all_topics, extraction_metadata))
map_elapsed = time.time() - map_start
n_map_outputs = sum(len(r.consolidated_topics) for r in map_results)
print(f"Map done in {map_elapsed:.1f}s: {n_topics} → {n_map_outputs} topics")

print(f"\n{'─'*80}")
print(f"MAP PHASE OUTPUT: {n_map_outputs} consolidated topics")
print(f"{'─'*80}")
map_idx = 1
for batch_result in map_results:
    for topic in batch_result.consolidated_topics:
        merged_str = f"({len(topic.merged_from)} merged)" if len(topic.merged_from) > 1 else "(standalone)"
        print(f"  {map_idx:3}. {topic.topic_label}  {merged_str}")
        map_idx += 1

# 4. Reduce phase
print(f"\nReduce phase...")
reduce_start = time.time()
final_result = asyncio.run(run_reduce_phase(map_results, extraction_metadata))
reduce_elapsed = time.time() - reduce_start

# 5. Results
final_topics = final_result.consolidated_topics

print(f"\n{'='*80}")
print(f"FINAL CONSOLIDATED TOPICS ({len(final_topics)} topics)")
print(f"{'='*80}")

for i, topic in enumerate(final_topics, 1):
    n_merged = len(topic.merged_from)
    clusters = sorted(set(topic.source_cluster_ids))
    cluster_str = ", ".join(str(c) for c in clusters)

    print(f"\n{'─'*80}")
    print(f"TOPIC {i}: {topic.topic_label}")
    print(f"{'─'*80}")
    print(f"  Source clusters ({len(clusters)}): [{cluster_str}]")
    print(f"  Inclusion: {topic.inclusion_definition}")
    print(f"  Rationale: {topic.merge_rationale}")
    if n_merged > 1:
        print(f"  Merged from ({n_merged}):")
        for label in topic.merged_from:
            print(f"    - {label}")

# 6. Validation + orphan reassignment
all_original = set(t["label"] for t in all_topics)
all_merged = set()
for topic in final_topics:
    all_merged.update(topic.merged_from)

missing = all_original - all_merged
extra = all_merged - all_original

if missing:
    print(f"\nReassigning {len(missing)} orphan topics to nearest theme...")
    # Build embedding lookup from step B's topic_embeddings
    label_to_embedding = {}
    for i, t in enumerate(all_topics):
        label_to_embedding[t["label"]] = topic_embeddings[i]

    # For each final theme, compute centroid from its merged_from embeddings
    theme_centroids = []
    for topic in final_topics:
        member_embeddings = [
            label_to_embedding[lbl] for lbl in topic.merged_from
            if lbl in label_to_embedding
        ]
        if member_embeddings:
            theme_centroids.append(np.mean(member_embeddings, axis=0))
        else:
            theme_centroids.append(np.zeros_like(topic_embeddings[0]))
    theme_centroids = np.array(theme_centroids)

    # Assign each orphan to nearest theme
    for orphan_label in sorted(missing):
        if orphan_label not in label_to_embedding:
            continue
        orphan_emb = label_to_embedding[orphan_label].reshape(1, -1)
        sims = sklearn_cosine_similarity(orphan_emb, theme_centroids)[0]
        best_idx = int(np.argmax(sims))
        final_topics[best_idx].merged_from.append(orphan_label)
        # Also add source cluster
        for t in all_topics:
            if t["label"] == orphan_label:
                if t["source_cluster_id"] not in final_topics[best_idx].source_cluster_ids:
                    final_topics[best_idx].source_cluster_ids.append(t["source_cluster_id"])
                break
        print(f"  '{orphan_label}' → '{final_topics[best_idx].topic_label}' (sim={sims[best_idx]:.3f})")

    # Revalidate
    all_merged = set()
    for topic in final_topics:
        all_merged.update(topic.merged_from)
    missing = all_original - all_merged

if extra:
    print(f"WARNING: {len(extra)} unknown topics in merged_from: {extra}")
if not missing and not extra:
    print(f"All {len(all_original)} original topics accounted for.")

# --- Pipeline summary ---
print(f"\n{'='*80}")
print(f"PIPELINE SUMMARY")
print(f"{'='*80}")
print(f"  {n_total} ideas -> {len(cluster_ids)} clusters -> {len(all_topics)} topics -> {n_map_outputs} (map) -> {len(final_topics)} (reduce)")
print(f"  Noise: {n_noise} ideas ({n_noise/n_total:.1%})")
total_time = elapsed + map_elapsed + reduce_elapsed
print(f"  Time: {total_time:.1f}s total (A: {elapsed:.1f}s, B-map: {map_elapsed:.1f}s, B-reduce: {reduce_elapsed:.1f}s)")

# List ideas per final topic (like object_discovery lists nodes per MECE object)
print(f"\nIDEAS PER CONSOLIDATED TOPIC:")
for i, topic in enumerate(final_topics, 1):
    source_ideas = []
    for cid in topic.source_cluster_ids:
        mask = clusterer._labels == cid
        for idx in np.where(mask)[0]:
            source_ideas.append(clusterer._idea_texts[idx])

    seen = set()
    unique_ideas = []
    for idea in source_ideas:
        if idea not in seen:
            seen.add(idea)
            unique_ideas.append(idea)

    print(f"\n  [{i}] {topic.topic_label} ({len(unique_ideas)} ideas)")
    for idea in unique_ideas[:10]:
        truncated = idea[:80] + "..." if len(idea) > 80 else idea
        print(f"      - {truncated}")
    if len(unique_ideas) > 10:
        print(f"      ... and {len(unique_ideas) - 10} more unique ideas")

# --- Cache Phase B results ---
# Final result (single Pydantic model)
cache_manager.save_metadata_to_cache(
    metadata=final_result,
    filename=FILENAME,
    step="mece_consolidated_topics",
    variable_key=variable_key,
    processing_time=map_elapsed + reduce_elapsed,
    var_lab=var_lab,
)
print(f"\nCACHED: Phase B final result ({len(final_topics)} topics) -> 'mece_consolidated_topics'")

# Topic embeddings (numpy array + aligned labels)
embeddings_path = cache_dir / f"mece_topic_embeddings_{base_name}_{variable_key}.pkl"
with open(embeddings_path, "wb") as f:
    pickle.dump({
        "embeddings": topic_embeddings,
        "topic_labels": [t["label"] for t in all_topics],
        "source_cluster_ids": [t["source_cluster_id"] for t in all_topics],
    }, f)
print(f"CACHED: Topic embeddings ({topic_embeddings.shape}) -> '{embeddings_path.name}'")

# All topics intermediate list
all_topics_path = cache_dir / f"mece_all_topics_{base_name}_{variable_key}.pkl"
with open(all_topics_path, "wb") as f:
    pickle.dump(all_topics, f)
print(f"CACHED: All topics list ({len(all_topics)} topics) -> '{all_topics_path.name}'")


#%% Main entry point — save captured output to file

if __name__ == "__main__" and _tee is not None:
    sys.stdout = _tee.original_stdout
    output_path = save_results_to_file(_tee.get_output(), base_name)
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_path}")
