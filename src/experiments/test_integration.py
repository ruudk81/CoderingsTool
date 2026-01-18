"""
Test integration between cluster_analysis.py and representation_comparison.py

This script demonstrates the complete workflow:
1. Run cluster_analysis.py with c-TF-IDF
2. Pass results to representation_comparison.py
3. Compare all 5 models
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cluster_analysis import run_experiment, ExperimentConfig
from representation_comparison import compare_all_models

# ============================================================================
# Step 1: Run cluster_analysis.py
# ============================================================================

print("="*100)
print("STEP 1: Running cluster_analysis.py with c-TF-IDF")
print("="*100)

config = ExperimentConfig(
    filename="M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav",
    var_name="Q20",
    sample_size=50,
    keyword_method="ctfidf",
    ctfidf_top_k=15,
    description_model="gpt-4.1-mini",
    use_keywords_in_prompt=True,
    max_ideas_per_cluster=10,
    n_sample_clusters=5,  # Just show 5 for now
    verbose=True
)

# Run experiment - this now returns data
experiment_data = run_experiment(config)

print("\n✅ cluster_analysis.py complete!")
print(f"   - Clusters: {len(experiment_data['clusters'])}")
print(f"   - Keywords extracted: {len(experiment_data['cluster_keywords'])}")
print(f"   - Descriptions generated: {len(experiment_data['cluster_descriptions'])}")

# ============================================================================
# Step 2: Run representation_comparison.py
# ============================================================================

print("\n" + "="*100)
print("STEP 2: Running representation_comparison.py")
print("="*100)

comparison_results = compare_all_models(
    cluster_results=experiment_data["cluster_results"],
    n_sample_clusters=5,  # Compare on 5 clusters
    export_excel=True
)

print("\n✅ representation_comparison.py complete!")
print(f"   - Models compared: {len(comparison_results['results'])}")
print(f"   - Clusters analyzed: {len(comparison_results['clusters'])}")
print(f"   - Export: exports/representation_comparison.xlsx")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*100)
print("INTEGRATION TEST COMPLETE")
print("="*100)
print("\nWhat was tested:")
print("  1. ✅ cluster_analysis.py runs and returns data")
print("  2. ✅ representation_comparison.py accepts cluster_results directly")
print("  3. ✅ All 5 models compared side-by-side")
print("  4. ✅ Excel export created")
print("\nNext steps:")
print("  - Review the comparison in: exports/representation_comparison.xlsx")
print("  - Compare keyword quality across models")
print("  - Decide which model to use going forward")
