#%%
# Cell 0: Imports + configuration
"""
MECE Topic Discovery from Nodes

Clusters unique ontology nodes to discover MECE topics using the
Clusterer pipeline (HDBSCAN → Phase B theme labeling → Phase C MECE consolidation).

TEXT_FORMAT controls what text is embedded and clustered.
If the format matches a pre-computed embedding field, it is reused;
otherwise embeddings are generated on the fly.

Usage:
    Run cells sequentially in VS Code interactive mode.
    Or: cd src && python -m experiments.step_5_clusterer_v2.topic_discovery_nodes
"""

import sys
from pathlib import Path

import nest_asyncio
nest_asyncio.apply()

# Path setup
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "src" / "experiments"))

from experiments.step_5_clusterer_v2.clusterer_exp import Clusterer
from experiments.step_5_clusterer_v2.config_clusterer_exp import ClustererConfig
from experiments.step_5_clusterer_v2.clusterer_helpers_exp import ThemeGenerator
from experiments.step_5_clusterer_v2.prompts_exp import (
    CLUSTER_THEME_PROMPT, ClusterThemeDescription,
    MECE_CONSOLIDATION_PROMPT, MECETopicSet,
)
from experiments.step_5_clusterer_v2.discovery_helpers import (
    resolve_embedding_strategy, extract_unique_items,
    wrap_as_embeddings_models, load_step4_embeddings,
    load_extraction_metadata, get_template_prefix,
    print_discovery_results,
)

# Dataset
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

# ── Configurable text format ──
# Examples: "{node}", "{template_prefix}{node}", "{root} -> {category}", "{category}"
TEXT_FORMAT = "{node}"


#%% Cell 1: Load data

embeddings_models = load_step4_embeddings(FILENAME, VARIABLE, SAMPLE_SIZE, project_root)
extraction_metadata = load_extraction_metadata(FILENAME, VARIABLE, SAMPLE_SIZE)
template_prefix = get_template_prefix(embeddings_models, extraction_metadata)


#%% Cell 2: Extract unique items + build synthetic models

_, embedding_source = resolve_embedding_strategy(TEXT_FORMAT)
item_names, item_embeddings, item_metadata = extract_unique_items(
    embeddings_models, TEXT_FORMAT, template_prefix
)
synthetic_models = wrap_as_embeddings_models(item_names, item_embeddings, embedding_source)
print(f"Created {len(synthetic_models)} synthetic EmbeddingsModel records")


#%% Cell 3: Run Clusterer with TOPIC prompts

config = ClustererConfig(
    embedding_source=embedding_source,
    generate_ctfidf=True,
    generate_llm_labels=True,
    generate_mece_topics=True,
)

theme_generator = ThemeGenerator(
    config,
    prompt_template=CLUSTER_THEME_PROMPT,
    response_model=ClusterThemeDescription,
)

clusterer = Clusterer(
    synthetic_models,
    config=config,
    extraction_metadata=extraction_metadata,
    theme_generator=theme_generator,
    mece_prompt_template=MECE_CONSOLIDATION_PROMPT,
    mece_response_model=MECETopicSet,
)

clusterer.run()


#%% Cell 4: Results

print_discovery_results(clusterer, item_names, item_metadata, "TOPIC")


#%% Main entry point

if __name__ == "__main__":
    pass
