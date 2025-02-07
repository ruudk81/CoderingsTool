"""Test the new labeller with cached cluster data"""
import asyncio
import sys
from pathlib import Path

# Add project paths
sys.path.append(str(Path(__file__).parents[2]))  # Add src directory

import models
from cache_manager import CacheManager
from cache_config import CacheConfig
import data_io
from modules.utils.labeller import Labeller

async def main():
    # Initialize cache manager
    cache_config = CacheConfig()
    cache_manager = CacheManager(cache_config)
    
    filename = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
    var_name = "Q20"
    
    # Load clusters from cache
    cluster_results = cache_manager.load_from_cache(filename, "clusters", models.ClusterModel)
    
    if cluster_results:
        print(f"Loaded {len(cluster_results)} cluster results from cache")
        
        # Count total clusters
        unique_clusters = set()
        for result in cluster_results:
            for segment in result.response_segment:
                if segment.mirco_cluster:
                    cluster_id = list(segment.mirco_cluster.keys())[0]
                    unique_clusters.add(cluster_id)
        
        print(f"Found {len(unique_clusters)} unique clusters")
        
        # Get variable label
        data_loader = data_io.DataLoader()
        var_lab = data_loader.get_varlab(filename=filename, var_name=var_name)
        print(f"Variable label: {var_lab}")
        
        # Initialize labeller
        print("\n=== Running new hierarchical labelling pipeline ===")
        labeller = Labeller()
        
        # Run the pipeline (limit to first 10 results for testing)
        test_results = cluster_results[:10]
        label_results = labeller.create_hierarchical_labels(test_results, var_lab)
        
        # Save to cache
        cache_manager.save_to_cache(label_results, filename, 'labels_test')
        print(f"\nSaved {len(label_results)} label results to cache")
        
        # Print summary
        print("\n=== Label Summary ===")
        print(f"Total results: {len(label_results)}")
        
        # Show first result
        if label_results:
            first_result = label_results[0]
            print(f"\nFirst result - Respondent {first_result.respondent_id}:")
            print(f"Summary: {first_result.summary[:200]}...")
            
            if first_result.response_segment:
                first_segment = first_result.response_segment[0]
                print(f"\nFirst segment:")
                print(f"Theme: {first_segment.Theme}")
                print(f"Topic: {first_segment.Topic}")
                print(f"Code: {first_segment.Code}")
        
    else:
        print("No cached cluster data found.")
        print("Please run the clustering pipeline first:")
        print("  python pipeline.py --force-step clusters")

if __name__ == "__main__":
    asyncio.run(main())