# test_labeller_determinism.py
"""
Standalone script to test labeller determinism
Run from project root: python test_labeller_determinism.py
"""

import sys
sys.path.append('src')

import json
import hashlib
from collections import defaultdict
from pathlib import Path

# Import your modules
from cache_manager import CacheManager
from config import CacheConfig, LabellerConfig
from utils.labeller import Labeller
from utils.data_io import DataLoader
import models

# Test configuration
FILENAME = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
VAR_NAME = "Q20"
NUM_RUNS = 3


def main():
    """Main test function"""
    print("🔄 Loading data...")
    
    # Initialize cache manager
    cache_config = CacheConfig()
    cache_manager = CacheManager(cache_config)
    
    # Load variable label
    data_loader = DataLoader()
    var_lab = data_loader.get_varlab(filename=FILENAME, var_name=VAR_NAME)
    print(f"Variable label: {var_lab}")
    
    # Load cluster results from cache
    cluster_results = cache_manager.load_from_cache(FILENAME, "clusters", models.ClusterModel)
    
    if not cluster_results:
        print("❌ No cluster data found in cache. Please run pipeline steps 1-5 first.")
        return False
    
    print(f"✅ Loaded {len(cluster_results)} cluster models")
    
    # Run the determinism test
    success = run_determinism_test(cluster_results, var_lab, cache_manager, NUM_RUNS)
    
    if success:
        print("\n🎉 Your labeller is DETERMINISTIC!")
        print("The same input produces the same output every time.")
    else:
        print("\n⚠️ Your labeller is NOT deterministic!")
        print("Results vary between runs - this needs investigation.")
    
    return success


def run_determinism_test(cluster_results, var_lab, cache_manager, num_runs=3):
    """Run the labeller multiple times and compare results"""
    print(f"\n🧪 STARTING DETERMINISM TEST")
    print(f"Will run labeller {num_runs} times and compare results\n")
    
    results = []
    hierarchies = []
    
    for run in range(num_runs):
        print(f"\n{'='*60}")
        print(f"RUN {run + 1}/{num_runs}")
        print('='*60)
        
        # Clear intermediate cache to force fresh calculation
        intermediate_dir = cache_manager.config.cache_dir / "intermediate"
        if intermediate_dir.exists():
            for file in intermediate_dir.glob(f"*hierarchy*.pkl"):
                file.unlink()
                print(f"Cleared cache: {file.name}")
        
        # Create fresh labeller config
        labeller_config = LabellerConfig(
            temperature=0.0,
            seed=42,
            batch_size=8,
            max_retries=3,
            use_sequential_processing=True,
            validation_threshold=0.95
        )
        
        # Create new labeller instance (no cache for testing)
        label_generator = Labeller(
            config=labeller_config,
            cache_manager=None,  # Disable cache for testing
            filename=FILENAME
        )
        
        try:
            # Run labelling
            labeled_results = label_generator.run_pipeline(cluster_results, var_lab)
            
            # Extract hierarchy structure
            hierarchy = extract_hierarchy_structure(labeled_results)
            
            results.append(labeled_results)
            hierarchies.append(hierarchy)
            
            # Print summary for this run
            print(f"\nRun {run + 1} Summary:")
            print(f"- Themes: {len(hierarchy['themes'])}")
            print(f"- Total clusters assigned: {hierarchy['total_clusters']}")
            print(f"- Direct assignment themes: {len(hierarchy['direct_themes'])}")
            
        except Exception as e:
            print(f"❌ Error in run {run + 1}: {str(e)}")
            return False
    
    # Compare results
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    # 1. Check if theme structures are identical
    identical_themes = all(h['themes'] == hierarchies[0]['themes'] for h in hierarchies[1:])
    
    # 2. Check cluster assignments
    identical_assignments = compare_cluster_assignments(results)
    
    # 3. Generate hashes for detailed comparison
    hashes = [generate_hierarchy_hash(h) for h in hierarchies]
    identical_hashes = len(set(hashes)) == 1
    
    # Print results
    if identical_themes and identical_assignments and identical_hashes:
        print("✅ SUCCESS: All runs produced IDENTICAL results!")
        print(f"   Consistent hierarchy hash: {hashes[0][:12]}")
    else:
        print("❌ FAILURE: Runs produced DIFFERENT results!")
        
        if not identical_themes:
            print("\n⚠️ Theme structures differ:")
            for i, h in enumerate(hierarchies):
                print(f"   Run {i+1}: {sorted(h['themes'].keys())}")
        
        if not identical_assignments:
            print("\n⚠️ Cluster assignments differ")
        
        if not identical_hashes:
            print("\n⚠️ Different hierarchy hashes:")
            for i, hash_val in enumerate(hashes):
                print(f"   Run {i+1}: {hash_val[:12]}")
    
    # Show detailed differences if any
    if not identical_hashes:
        print("\n📊 DETAILED DIFFERENCES:")
        show_differences(hierarchies)
    
    return identical_themes and identical_assignments and identical_hashes


def extract_hierarchy_structure(labeled_results):
    """Extract the hierarchy structure from labeled results"""
    structure = {
        'themes': {},
        'topics': {},
        'cluster_assignments': {},
        'total_clusters': 0,
        'direct_themes': []
    }
    
    for result in labeled_results:
        if result.response_segment:
            for segment in result.response_segment:
                if segment.Theme and segment.micro_cluster:
                    theme_id = list(segment.Theme.keys())[0]
                    theme_text = list(segment.Theme.values())[0]
                    cluster_id = list(segment.micro_cluster.keys())[0]
                    
                    structure['themes'][theme_id] = theme_text
                    
                    if segment.Topic:
                        topic_id = list(segment.Topic.keys())[0]
                        topic_text = list(segment.Topic.values())[0]
                        
                        # Check if this is a direct assignment
                        if theme_text == topic_text:
                            if theme_id not in structure['direct_themes']:
                                structure['direct_themes'].append(theme_id)
                        
                        if topic_id not in structure['topics']:
                            structure['topics'][topic_id] = {
                                'text': topic_text,
                                'theme_id': theme_id,
                                'clusters': []
                            }
                        
                        if cluster_id not in structure['topics'][topic_id]['clusters']:
                            structure['topics'][topic_id]['clusters'].append(cluster_id)
                    
                    structure['cluster_assignments'][cluster_id] = {
                        'theme_id': theme_id,
                        'topic_id': list(segment.Topic.keys())[0] if segment.Topic else None
                    }
    
    structure['total_clusters'] = len(structure['cluster_assignments'])
    return structure


def generate_hierarchy_hash(hierarchy):
    """Generate a hash of the hierarchy for comparison"""
    # Create a deterministic string representation
    data = {
        'themes': sorted([(k, v) for k, v in hierarchy['themes'].items()]),
        'topics': sorted([(k, v['text'], sorted(v['clusters'])) 
                         for k, v in hierarchy['topics'].items()]),
        'assignments': sorted([(k, v) for k, v in hierarchy['cluster_assignments'].items()])
    }
    
    json_str = json.dumps(data, sort_keys=True)
    return hashlib.md5(json_str.encode()).hexdigest()


def compare_cluster_assignments(results_list):
    """Compare cluster assignments across runs"""
    all_assignments = []
    
    for results in results_list:
        assignments = {}
        for result in results:
            if result.response_segment:
                for segment in result.response_segment:
                    if segment.micro_cluster and segment.Theme:
                        cluster_id = list(segment.micro_cluster.keys())[0]
                        theme_id = list(segment.Theme.keys())[0]
                        topic_id = list(segment.Topic.keys())[0] if segment.Topic else None
                        assignments[cluster_id] = (theme_id, topic_id)
        all_assignments.append(assignments)
    
    # Check if all are identical
    if not all_assignments:
        return False
    
    first = all_assignments[0]
    return all(a == first for a in all_assignments[1:])


def show_differences(hierarchies):
    """Show specific differences between hierarchies"""
    # Compare themes
    all_theme_ids = set()
    for h in hierarchies:
        all_theme_ids.update(h['themes'].keys())
    
    print("\nTheme Differences:")
    for theme_id in sorted(all_theme_ids):
        theme_texts = []
        for i, h in enumerate(hierarchies):
            if theme_id in h['themes']:
                theme_texts.append(h['themes'][theme_id])
            else:
                theme_texts.append("[MISSING]")
        
        if len(set(theme_texts)) > 1:
            print(f"\nTheme {theme_id}:")
            for i, text in enumerate(theme_texts):
                print(f"  Run {i+1}: {text}")
    
    # Compare cluster assignments
    all_clusters = set()
    for h in hierarchies:
        all_clusters.update(h['cluster_assignments'].keys())
    
    different_clusters = []
    for cluster_id in sorted(all_clusters):
        assignments = []
        for h in hierarchies:
            if cluster_id in h['cluster_assignments']:
                assignments.append(h['cluster_assignments'][cluster_id])
            else:
                assignments.append(None)
        
        if len(set(str(a) for a in assignments)) > 1:
            different_clusters.append((cluster_id, assignments))
    
    if different_clusters:
        print(f"\nCluster Assignment Differences ({len(different_clusters)} clusters differ):")
        # Show first 5 differences
        for cluster_id, assignments in different_clusters[:5]:
            print(f"\nCluster {cluster_id}:")
            for i, assignment in enumerate(assignments):
                if assignment:
                    theme = assignment['theme_id']
                    topic = assignment['topic_id'] if assignment['topic_id'] else "direct"
                    print(f"  Run {i+1}: Theme {theme}, Topic {topic}")
                else:
                    print(f"  Run {i+1}: [NOT ASSIGNED]")
        
        if len(different_clusters) > 5:
            print(f"\n... and {len(different_clusters) - 5} more clusters with differences")


if __name__ == "__main__":
    # Run the test
    success = main()