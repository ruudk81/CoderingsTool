#%%
"""
PRACTICAL EXAMPLE: How to run representation model comparison

This shows the SIMPLEST way to compare all 5 keyword extraction models
using data you already have from cluster_analysis.py
"""

# ============================================================================
# METHOD 1: Using cluster_analysis.py output (RECOMMENDED)
# ============================================================================

from cluster_analysis import run_experiment, ExperimentConfig
from representation_comparison import compare_all_models

# Step 1: Configure and run cluster_analysis
# TODO: Update these values to match YOUR data!
config = ExperimentConfig(
    filename="M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav",  # ← CHANGE THIS to your SPSS file
    var_name="Q20",             # ← CHANGE THIS to your question variable
    sample_size=50,             # ← CHANGE THIS to match your Step 5 sample_size (or None)
    keyword_method="ctfidf",    # Use c-TF-IDF
    description_model="gpt-5.2-chat",
    n_sample_clusters=None,     # Display all
    verbose=True
)

print("Running cluster_analysis.py...")
experiment_data = run_experiment(config)

# Step 2: Run comparison on the same data
print("\nRunning representation_comparison.py...")
comparison = compare_all_models(
    cluster_results=experiment_data["cluster_results"],  # Pass data directly!
    n_sample_clusters=10,  # Compare on 10 clusters
    export_excel=True
)

print("\n✅ Done! Check: exports/representation_comparison.xlsx")

#%%

# ============================================================================
# METHOD 2: Using cached data (if you already ran Step 5)
# ============================================================================

from representation_comparison import compare_all_models, ComparisonConfig

# Configure comparison to load from cache
# TODO: Update these values to match YOUR data!
config = ComparisonConfig(
    filename="YOUR_FILE.sav",  # ← CHANGE THIS
    var_name="Q20",             # ← CHANGE THIS
    sample_size=50,             # ← MUST match your Step 5 sample_size!
    n_sample_clusters=10,
    export_excel=True,
    verbose=True
)

# Run comparison (loads from cache)
print("Loading from cache and running comparison...")
comparison = compare_all_models(config=config)

print("\n✅ Done! Check: exports/representation_comparison.xlsx")

#%%

# ============================================================================
# What you get:
# ============================================================================

# comparison dict contains:
# - comparison['results']: Keywords from each model
# - comparison['metrics']: Coverage, diversity, etc.
# - comparison['clusters']: Cluster data

# Excel file has:
# - Overview sheet: All models side-by-side
# - Standard_TF-IDF sheet: Detailed Standard TF-IDF keywords
# - c-TF-IDF sheet: Detailed c-TF-IDF keywords
# - c-TF-IDF_MMR sheet: Detailed MMR keywords
# - c-TF-IDF_KeyBERT sheet: Detailed KeyBERT keywords
# - c-TF-IDF_LLM sheet: Detailed LLM-enhanced keywords

#%%
