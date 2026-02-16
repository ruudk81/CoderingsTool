#%%

"""
Step 5: Clusterer V4 Experiment Runner — Object-Aware Map-Reduce MECE

Three-stage pipeline:
  Stage 1: Object Discovery
    - Extract unique categories from Step 4 embeddings
    - Cluster categories using V4 Clusterer (phases 1-5)
    - Generate per-cluster themes → consolidate into MECE objects

  Stage 2: Map Objects to Ideas
    - Map each MECE object back to all idea instances
    - Chain: MECE object → source_cluster_ids → categories → ideas

  Stage 3: Object-Aware Map-Reduce MECE
    - For each MECE object, run MAP/REDUCE/MECE on its ideas
    - All prompts include object context (label, inclusion/exclusion, peer objects)
    - Result: MECE topics per MECE object

This experiment runner uses LOCAL COPIES that can be modified
without affecting the production pipeline.

Dataset configuration is centralized in experiments/test_data.py.

Usage:
    cd src && python -m experiments.step_5_clusterer_v4.run_experiment
"""

import sys
import io
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime
import pickle

import numpy as np

# Add parent paths for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "src" / "experiments"))

from experiments import models_exp as models
from utils.cacheManager import CacheManager, generate_enhanced_variable_key

# Import V4 components
try:
    from .config_clusterer_exp import ClustererConfig
    from .object_discovery import ObjectDiscoverer
    from .object_mapper import ObjectMapper
    from .map_reduce_mece import ObjectAwareMapReduceMECE, ObjectMECEResult
    from .prompts_exp import MECEObjectSet
    from .category_discovery import CategoryBasedDiscoverer
except ImportError:
    from experiments.step_5_clusterer_v4.config_clusterer_exp import ClustererConfig
    from experiments.step_5_clusterer_v4.object_discovery import ObjectDiscoverer
    from experiments.step_5_clusterer_v4.object_mapper import ObjectMapper
    from experiments.step_5_clusterer_v4.map_reduce_mece import ObjectAwareMapReduceMECE, ObjectMECEResult
    from experiments.step_5_clusterer_v4.prompts_exp import MECEObjectSet
    from experiments.step_5_clusterer_v4.category_discovery import CategoryBasedDiscoverer

# =============================================================================
# DATASET CONFIGURATION (centralized in experiments/test_data.py)
# =============================================================================
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


# =============================================================================
# CONFIGURATION
# =============================================================================
# All defaults defined in config_clusterer_exp.py.
# Override individual params here only for one-off experiments.
CONFIG = ClustererConfig(
    # Discovery mode: "clustering" (default) or "semantic_category" (skip Stages 1+2)
    object_discovery_mode="semantic_category",
    # Stage 1 (only used when mode="clustering"): cluster on category-level embeddings
    object_discovery_level="category",
    # Stage 3: use idea.node text for map-reduce
    mapreduce_text_source="node",
)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_step4_embeddings(
    filename: str = FILENAME,
    variable: str = VARIABLE,
    sample_size: Optional[int] = SAMPLE_SIZE,
    variable_key: Optional[str] = None,
) -> List[models.EmbeddingsModel]:
    """Load Step 4 embeddings from cache."""
    if variable_key is None:
        variable_key = generate_enhanced_variable_key(
            selected_variables=[variable],
            is_merged=False,
            sample_size=sample_size
        )

    cache_dir = project_root / "data" / "cache"
    base_name = Path(filename).stem
    cache_filename = f"005_embeddings_{base_name}_{variable_key}.pkl"
    cache_path = cache_dir / cache_filename

    print(f"Loading embeddings from: {cache_path}")

    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cache file not found: {cache_path}\n"
            f"Run pipeline step 4 first to generate embeddings."
        )

    with open(cache_path, 'rb') as f:
        serializable_data = pickle.load(f)

    embeddings_models = [models.EmbeddingsModel.model_validate(item) for item in serializable_data]

    # Log cached format
    embedding_format = "idea"
    if embeddings_models and hasattr(embeddings_models[0], 'embedding_text_format'):
        embedding_format = embeddings_models[0].embedding_text_format or "idea"
    print(f"Cached embedding format: {embedding_format}")

    return embeddings_models


def load_extraction_metadata(
    filename: str = FILENAME,
    variable: str = VARIABLE,
    sample_size: Optional[int] = SAMPLE_SIZE,
    variable_key: Optional[str] = None
) -> Optional[models.ExtractionMetadata]:
    """Load ExtractionMetadata from cache (if available)."""
    if variable_key is None:
        variable_key = generate_enhanced_variable_key(
            selected_variables=[variable],
            is_merged=False,
            sample_size=sample_size
        )

    cache_manager = CacheManager()
    metadata = cache_manager.load_metadata_from_cache(
        filename=filename,
        step="extracted_ideas",
        variable_key=variable_key,
        model_cls=models.ExtractionMetadata
    )

    if metadata:
        print(f"Loaded ExtractionMetadata: taxonomy_axis={metadata.taxonomy_axis}")
        if metadata.var_lab:
            print(f"  Survey question (var_lab): {metadata.var_lab}")
    else:
        print("ExtractionMetadata not found in cache (optional)")

    return metadata


# =============================================================================
# RESULTS PRINTING
# =============================================================================

def print_results(
    mece_objects: MECEObjectSet,
    object_mappings: Dict,
    results: Dict[str, ObjectMECEResult],
):
    """Print the complete V4 results: MECE objects → MECE topics per object."""
    total_topics = sum(len(r.topics) for r in results.values())

    print(f"\n{'='*80}")
    print(f"PER-OBJECT MECE TOPICS ({len(mece_objects.topics)} objects, {total_topics} total topics)")
    print(f"{'='*80}")

    for i, obj in enumerate(mece_objects.topics, 1):
        label = obj.topic_label
        mapping = object_mappings.get(label)
        result = results.get(label)

        print(f"\n{'─'*80}")
        # Object header with pipeline metadata
        n_ideas = result.n_ideas if result else (mapping.idea_count if mapping else 0)
        if result:
            reduce_str = "reduce skipped" if result.reduce_skipped else "reduce applied"
            print(f"OBJECT {i}: {label} (n={n_ideas}, {result.n_batches} batch(es), {reduce_str})")
        else:
            print(f"OBJECT {i}: {label} (n={n_ideas})")
        print(f"{'─'*80}")

        print(f"  Inclusion: {obj.inclusion_definition}")
        print(f"  Boundary test: {obj.boundary_test}")
        signals_str = ", ".join(obj.diagnostic_signals[:5]) if obj.diagnostic_signals else "(none)"
        print(f"  Diagnostic signals: {signals_str}")
        print(f"  Source clusters: {obj.source_cluster_ids}")
        print(f"  Merge rationale: {obj.merge_rationale}")

        if mapping:
            print(f"  Categories: {len(mapping.category_names)} → {mapping.idea_count} ideas")
            if mapping.category_names:
                cat_preview = mapping.category_names[:8]
                cat_str = ", ".join(cat_preview)
                if len(mapping.category_names) > 8:
                    cat_str += f", ... (+{len(mapping.category_names) - 8} more)"
                print(f"    {cat_str}")

        if result:
            print(f"\n  MECE Topics ({len(result.topics)}):")
            for j, topic in enumerate(result.topics, 1):
                print(f"\n    [{j}] {topic.topic_label}")
                print(f"        Inclusion: {topic.inclusion_definition}")
                print(f"        Boundary test: {topic.boundary_test}")
                signals = ", ".join(topic.diagnostic_signals[:5]) if topic.diagnostic_signals else "(none)"
                print(f"        Diagnostic signals: {signals}")
                if topic.tiebreaker_rules:
                    print(f"        Tiebreaker rules:")
                    for rule in topic.tiebreaker_rules:
                        print(f"          - {rule}")
                if topic.key_expressions:
                    print(f"        Expressions:")
                    for expr in topic.key_expressions[:3]:
                        truncated = expr[:80] + "..." if len(expr) > 80 else expr
                        print(f"          - {truncated}")

            # Print MECE verifications
            if hasattr(result, 'mece_verifications') and result.mece_verifications:
                print(f"\n  MECE Verifications ({len(result.mece_verifications)}):")
                for v in result.mece_verifications:
                    print(f"    [{v.topic_a}] vs [{v.topic_b}]")
                    truncated_ex = v.ambiguous_example[:80] + "..." if len(v.ambiguous_example) > 80 else v.ambiguous_example
                    print(f"      Example: \"{truncated_ex}\"")
                    print(f"      → Assigned to: {v.assigned_to}")
                    print(f"      → Reasoning: {v.reasoning}")
        else:
            print(f"\n  (no MECE topics — processing may have failed)")

    # Grand summary
    total_ideas = sum(m.idea_count for m in object_mappings.values())
    print(f"\n{'='*80}")
    print(f"GRAND SUMMARY")
    print(f"{'='*80}")
    print(f"  MECE Objects: {len(mece_objects.topics)}")
    print(f"  Total Ideas:  {total_ideas}")
    print(f"  Total Topics: {total_topics}")
    if mece_objects.topics:
        print(f"  Average:      {total_topics / len(mece_objects.topics):.1f} topics per object")
    print(f"{'='*80}\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the V4 Object-Aware Map-Reduce MECE pipeline."""
    print("=" * 70)
    print("Clustering Pipeline V4 (Object-Aware Map-Reduce MECE)")
    print("=" * 70)
    print(f"\nDataset: {FILENAME}")
    print(f"Variable: {VARIABLE}")
    print(f"Sample size: {SAMPLE_SIZE}")
    print(f"Discovery mode: {CONFIG.object_discovery_mode}")
    if CONFIG.object_discovery_mode == "clustering":
        print(f"Object discovery level: {CONFIG.object_discovery_level}")
    print(f"Map-reduce text source: {CONFIG.mapreduce_text_source}")
    print(f"Batch size: {CONFIG.mapreduce_batch_size}")
    print()

    # Load data
    embeddings_models = load_step4_embeddings()
    extraction_metadata = load_extraction_metadata()

    # =========================================================================
    # Stages 1+2: Object Discovery + Mapping
    # =========================================================================
    # Grouping instructions: only used in semantic_category mode
    grouping_instructions = None

    if CONFIG.object_discovery_mode == "semantic_category":
        # Category-based: partition by semantic_category (no clustering)
        discoverer = CategoryBasedDiscoverer(CONFIG, extraction_metadata)
        mece_objects, object_mappings = discoverer.discover(embeddings_models)
        taxonomy_axis_info = discoverer.get_taxonomy_axis_info()
        grouping_instructions = discoverer.get_grouping_instructions()

    else:
        # Default: clustering-based object discovery (Stages 1+2)
        discoverer = ObjectDiscoverer(CONFIG, extraction_metadata)
        mece_objects, cat_clusterer, cat_names, cat_labels = discoverer.discover(embeddings_models)

        # Stage 1 Summary: Algorithm recommendation + metrics
        rec = cat_clusterer.get_algorithm_recommendation()
        if rec:
            print(f"\nAlgorithm Recommendation:")
            print(f"  Recommended: {rec.recommended_algorithm} ({rec.confidence} confidence)")
            print(f"  DVC: {rec.dvc_value:.3f} → {rec.dvc_recommendation}")
            print(f"  Knee: y_diff={rec.y_difference:.2f}, sharp={rec.has_sharp_knee}")
            if rec.is_forced:
                print(f"  FORCED: Algorithm selection was forced by hard DVC rule")
            print(f"  Reasoning: {rec.reasoning}")

        metrics = cat_clusterer.get_metrics()
        if metrics:
            print(f"\nClustering Metrics:")
            print(f"  Clusters: {metrics.n_clusters}")
            print(f"  Noise: {metrics.noise_count} ({metrics.noise_rate:.1%})")
            print(f"  Coherence: {metrics.mean_coherence:.3f} ({metrics.coherence_breakdown})")
            if metrics.dbcv is not None:
                print(f"  DBCV: {metrics.dbcv:.3f}")
            if metrics.silhouette is not None and not np.isnan(metrics.silhouette):
                print(f"  Silhouette: {metrics.silhouette:.3f}")
            if metrics.mean_persistence is not None:
                print(f"  Persistence: mean={metrics.mean_persistence:.3f}, weighted={metrics.weighted_persistence:.3f}")
            if metrics.mean_probability is not None:
                print(f"  Probability: mean={metrics.mean_probability:.3f}, low_ratio={metrics.low_prob_ratio:.1%}")
            if metrics.mean_outlier_score is not None:
                print(f"  Outliers: mean_score={metrics.mean_outlier_score:.3f}, high_ratio={metrics.high_outlier_ratio:.1%}")
            print(f"  Cluster sizes: min={metrics.min_cluster_size}, median={metrics.median_cluster_size}, max={metrics.max_cluster_size}")

        # Stage 2: Map Objects to Ideas
        mapper = ObjectMapper(mece_objects, cat_names, cat_labels, embeddings_models)
        object_mappings = mapper.map_objects_to_ideas(text_source=CONFIG.mapreduce_text_source)
        taxonomy_axis_info = discoverer.get_taxonomy_axis_info()

    # =========================================================================
    # Stage 3: Object-Aware Map-Reduce MECE
    # =========================================================================
    # Build context from extraction metadata
    survey_question = ""
    language = "Dutch"
    dataset_context = None
    taxonomy_axis = None
    taxonomy_description = None
    taxonomy_actionable_type = None

    if extraction_metadata:
        meta = extraction_metadata
        survey_question = getattr(meta, 'var_lab', '') or ''
        language = getattr(meta, 'lang', 'Dutch') or 'Dutch'
        dataset_context = {}
        for field in ('domain', 'entity', 'topic', 'perspective', 'intent'):
            val = getattr(meta, field, None)
            if val:
                dataset_context[field] = val
        taxonomy_axis = getattr(meta, 'taxonomy_axis', None)
        taxonomy_description = getattr(meta, 'taxonomy_axis_description', None)
        taxonomy_actionable_type = getattr(meta, 'taxonomy_actionable_type', None)

    processor = ObjectAwareMapReduceMECE(CONFIG)
    results = processor.process_all_objects(
        object_mappings=object_mappings,
        mece_objects=mece_objects,
        survey_question=survey_question,
        language=language,
        dataset_context=dataset_context,
        taxonomy_axis=taxonomy_axis,
        taxonomy_description=taxonomy_description,
        taxonomy_axis_info=taxonomy_axis_info,
        taxonomy_actionable_type=taxonomy_actionable_type,
        grouping_instructions=grouping_instructions,
        verbose=CONFIG.verbose,
    )

    # =========================================================================
    # Print results
    # =========================================================================
    print_results(mece_objects, object_mappings, results)

    return mece_objects, object_mappings, results, embeddings_models


# =============================================================================
# OUTPUT CAPTURE
# =============================================================================

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


def save_results_to_file(output: str, filename: str, variable: str, sample_size: Optional[int]) -> Path:
    """Save results to a text file."""
    output_dir = project_root / "exports" / "cluster_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(filename).stem
    sample_str = str(sample_size) if sample_size else "full"
    date_str = datetime.now().strftime("%Y%m%d")

    output_filename = f"cluster_results_v4_{base_name}_{variable}_{sample_str}_{date_str}.txt"
    output_path = output_dir / output_filename

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output)

    return output_path


if __name__ == "__main__":
    # Capture all output while also printing to console
    tee = TeeOutput(sys.stdout)
    sys.stdout = tee

    try:
        mece_objects, object_mappings, results, embeddings_models = main()
    finally:
        sys.stdout = tee.original_stdout

    # Save results to file
    output_path = save_results_to_file(
        output=tee.get_output(),
        filename=FILENAME,
        variable=VARIABLE,
        sample_size=SAMPLE_SIZE
    )
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_path}")

# %%
