#%%

"""
Step 5: Category Discovery V5 — Partition-Aware Map-Reduce MECE

Always partitions by semantic_category (6 fixed groups), then discovers
coding categories within each partition via MAP/REDUCE/MECE.

Two processing modes:
  Mode A ("direct"):    MAP/REDUCE/MECE on labels directly
  Mode B ("clustered"): Pre-cluster labels via UMAP+HDBSCAN,
                        then MAP/REDUCE/MECE with cluster hints

Dataset configuration is centralized in experiments/test_data.py.

Usage:
    cd src && python -m experiments.step_5_categories_v5.run_experiment
"""

import sys
import io
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime
import pickle

# Add parent paths for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "src" / "experiments"))

from experiments import models_exp as models
from utils.cacheManager import CacheManager, generate_enhanced_variable_key

# Import V5 components
try:
    from .config_categories_exp import CategoriesConfig
    from .partition_discoverer import PartitionDiscoverer, PartitionLabelMapping
    from .map_reduce_mece import MapReduceMECE, PartitionMECEResult
    from .prompts_exp import PartitionSet
except ImportError:
    from experiments.step_5_categories_v5.config_categories_exp import CategoriesConfig
    from experiments.step_5_categories_v5.partition_discoverer import PartitionDiscoverer, PartitionLabelMapping
    from experiments.step_5_categories_v5.map_reduce_mece import MapReduceMECE, PartitionMECEResult
    from experiments.step_5_categories_v5.prompts_exp import PartitionSet

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
# All defaults defined in config_categories_exp.py.
# Override individual params here only for one-off experiments.
CONFIG = CategoriesConfig(
    processing_mode="direct",        # "direct" (Mode A) or "clustered" (Mode B)
    label_source="category_label",   # "category_label", "node", "idea", "instance"
    label_prefix="",                 # "" or "{root}: " for dynamic prefix
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

    embeddings_models = [
        models.EmbeddingsModel.model_validate(item)
        for item in serializable_data
    ]

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
    variable_key: Optional[str] = None,
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
    partition_set: PartitionSet,
    label_mappings: Dict[str, PartitionLabelMapping],
    results: Dict[str, PartitionMECEResult],
):
    """Print the complete V5 results: partitions → MECE categories per partition."""
    total_categories = sum(len(r.categories) for r in results.values())

    print(f"\n{'='*80}")
    print(f"PER-PARTITION MECE CATEGORIES "
          f"({len(partition_set.partitions)} partitions, "
          f"{total_categories} total categories)")
    print(f"{'='*80}")

    for i, part in enumerate(partition_set.partitions, 1):
        name = part.partition_name
        mapping = label_mappings.get(name)
        result = results.get(name)

        print(f"\n{'─'*80}")
        n_labels = result.n_labels if result else (mapping.label_count if mapping else 0)
        if result:
            reduce_str = ("reduce skipped"
                          if result.reduce_skipped
                          else "reduce applied")
            print(f"PARTITION {i}: {name} "
                  f"(n={n_labels}, {result.n_batches} batch(es), "
                  f"{reduce_str})")
        else:
            print(f"PARTITION {i}: {name} (n={n_labels})")
        print(f"{'─'*80}")

        print(f"  Inclusion: {part.inclusion_definition}")
        print(f"  Boundary test: {part.boundary_test}")
        signals_str = (", ".join(part.diagnostic_signals[:5])
                       if part.diagnostic_signals else "(none)")
        print(f"  Diagnostic signals: {signals_str}")

        if mapping:
            print(f"  Labels: {mapping.label_count} unique")
            if mapping.labels:
                label_preview = mapping.labels[:8]
                label_str = ", ".join(label_preview)
                if len(mapping.labels) > 8:
                    label_str += f", ... (+{len(mapping.labels) - 8} more)"
                print(f"    {label_str}")

        if result:
            print(f"\n  MECE Categories ({len(result.categories)}):")
            for j, cat in enumerate(result.categories, 1):
                print(f"\n    [{j}] {cat.category_label}")
                print(f"        Inclusion: {cat.inclusion_definition}")
                print(f"        Boundary test: {cat.boundary_test}")
                signals = (", ".join(cat.diagnostic_signals[:5])
                           if cat.diagnostic_signals else "(none)")
                print(f"        Diagnostic signals: {signals}")
                if cat.tiebreaker_rules:
                    print(f"        Tiebreaker rules:")
                    for rule in cat.tiebreaker_rules:
                        print(f"          - {rule}")
                if cat.key_expressions:
                    print(f"        Key expressions:")
                    for expr in cat.key_expressions[:3]:
                        truncated = (expr[:80] + "..."
                                     if len(expr) > 80 else expr)
                        print(f"          - {truncated}")

            # Print MECE verifications
            if (hasattr(result, 'mece_verifications')
                    and result.mece_verifications):
                print(f"\n  MECE Verifications "
                      f"({len(result.mece_verifications)}):")
                for v in result.mece_verifications:
                    print(f"    [{v.category_a}] vs [{v.category_b}]")
                    truncated_ex = (v.ambiguous_example[:80] + "..."
                                    if len(v.ambiguous_example) > 80
                                    else v.ambiguous_example)
                    print(f"      Example: \"{truncated_ex}\"")
                    print(f"      → Assigned to: {v.assigned_to}")
                    print(f"      → Reasoning: {v.reasoning}")
        else:
            print(f"\n  (no MECE categories — processing may have failed)")

    # Grand summary
    total_labels = sum(m.label_count for m in label_mappings.values())
    print(f"\n{'='*80}")
    print(f"GRAND SUMMARY")
    print(f"{'='*80}")
    print(f"  Partitions:       {len(partition_set.partitions)}")
    print(f"  Total Labels:     {total_labels}")
    print(f"  Total Categories: {total_categories}")
    if partition_set.partitions:
        print(f"  Average:          "
              f"{total_categories / len(partition_set.partitions):.1f} "
              f"categories per partition")
    print(f"{'='*80}\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the V5 Category Discovery pipeline."""
    print("=" * 70)
    print("Category Discovery V5 (Partition-Aware Map-Reduce MECE)")
    print("=" * 70)
    print(f"\nDataset: {FILENAME}")
    print(f"Variable: {VARIABLE}")
    print(f"Sample size: {SAMPLE_SIZE}")
    print(f"Processing mode: {CONFIG.processing_mode}")
    print(f"Label source: {CONFIG.label_source}")
    if CONFIG.label_prefix:
        print(f"Label prefix: {CONFIG.label_prefix!r}")
    print(f"Batch size: {CONFIG.mapreduce_batch_size}")
    print()

    # Load data
    embeddings_models = load_step4_embeddings()
    extraction_metadata = load_extraction_metadata()

    # =========================================================================
    # Stage 1: Partition Discovery
    # =========================================================================
    discoverer = PartitionDiscoverer(CONFIG, extraction_metadata)
    partition_set, label_mappings, precluster_results = discoverer.discover(
        embeddings_models
    )
    taxonomy_axis_info = discoverer.get_taxonomy_axis_info()
    grouping_instructions = discoverer.get_grouping_instructions()

    # =========================================================================
    # Stage 2: Map-Reduce MECE (per partition)
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
        for f in ('domain', 'entity', 'topic', 'perspective', 'intent'):
            val = getattr(meta, f, None)
            if val:
                dataset_context[f] = val
        taxonomy_axis = getattr(meta, 'taxonomy_axis', None)
        taxonomy_description = getattr(meta, 'taxonomy_axis_description', None)
        taxonomy_actionable_type = getattr(meta, 'taxonomy_actionable_type', None)

    processor = MapReduceMECE(CONFIG)
    results = processor.process_all_partitions(
        label_mappings=label_mappings,
        partition_set=partition_set,
        survey_question=survey_question,
        language=language,
        dataset_context=dataset_context,
        taxonomy_axis=taxonomy_axis,
        taxonomy_description=taxonomy_description,
        taxonomy_axis_info=taxonomy_axis_info,
        taxonomy_actionable_type=taxonomy_actionable_type,
        grouping_instructions=grouping_instructions,
        precluster_results=precluster_results,
        verbose=CONFIG.verbose,
    )

    # =========================================================================
    # Print results
    # =========================================================================
    print_results(partition_set, label_mappings, results)

    return partition_set, label_mappings, results, embeddings_models


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


def save_results_to_file(
    output: str,
    filename: str,
    variable: str,
    sample_size: Optional[int],
) -> Path:
    """Save results to a text file."""
    output_dir = project_root / "exports" / "cluster_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(filename).stem
    sample_str = str(sample_size) if sample_size else "full"
    date_str = datetime.now().strftime("%Y%m%d")

    output_filename = (
        f"category_results_v5_{base_name}_{variable}"
        f"_{sample_str}_{date_str}.txt"
    )
    output_path = output_dir / output_filename

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(output)

    return output_path


if __name__ == "__main__":
    # Capture all output while also printing to console
    tee = TeeOutput(sys.stdout)
    sys.stdout = tee

    try:
        partition_set, label_mappings, results, embeddings_models = main()
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
