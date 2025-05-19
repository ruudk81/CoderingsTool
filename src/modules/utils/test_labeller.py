"""Test the new labeller with cached cluster data"""
import asyncio
import sys
from pathlib import Path
from collections import defaultdict, Counter

# Add project paths
sys.path.append(str(Path(__file__).parents[2]))  # Add src directory

import models
from cache_manager import CacheManager
from cache_config import CacheConfig
import data_io
from modules.utils.labeller import Labeller


if __name__ == "__main__":
    """Test the labeller with cached cluster data"""
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
        print("\n=== Running hierarchical labelling pipeline ===")
        labeller = Labeller()
        
        # Run the pipeline
        label_results = labeller.create_hierarchical_labels(cluster_results, var_lab)
        
        # Save to cache
        cache_key = 'labels_all'
        cache_manager.save_to_cache(label_results, filename, cache_key)
        print(f"\nSaved {len(label_results)} label results to cache with key '{cache_key}'")
        
        # Unpack and analyze the results
        print("\n=== UNPACKING HIERARCHICAL LABELS ===")
        
        # Data structures to collect hierarchy
        themes = {}
        theme_summaries = {}
        theme_topics = defaultdict(lambda: {})
        topic_codes = defaultdict(lambda: defaultdict(list))
        
        # Process each result to extract hierarchy
        for result in label_results:
            # Extract theme summary if available
            if result.summary:
                # Store all summaries we find (they might be theme-specific)
                for segment in result.response_segment:
                    if segment.Theme:
                        theme_id = list(segment.Theme.keys())[0]
                        if theme_id not in theme_summaries:
                            theme_summaries[theme_id] = result.summary
            
            # Extract hierarchical structure
            for segment in result.response_segment:
                if segment.Theme:
                    theme_id, theme_label = list(segment.Theme.items())[0]
                    themes[theme_id] = theme_label
                    
                    if segment.Topic:
                        topic_id, topic_label = list(segment.Topic.items())[0]
                        theme_topics[theme_id][topic_id] = topic_label
                        
                        if segment.Code:
                            code_id, code_label = list(segment.Code.items())[0]
                            topic_codes[theme_id][topic_id].append((code_id, code_label))
        
        # Display the hierarchical structure
        print(f"\nFound {len(themes)} themes:")
        
        for theme_id in sorted(themes.keys()):
            print(f"\n{'='*60}")
            print(f"THEME {theme_id}: {themes[theme_id]}")
            print(f"{'='*60}")
            
            # Theme summary
            if theme_id in theme_summaries:
                print(f"\nSummary:")
                print(f"{theme_summaries[theme_id][:500]}...")
            
            # Topics in this theme
            topics_in_theme = theme_topics[theme_id]
            print(f"\nTopics ({len(topics_in_theme)}):")
            
            for topic_id in sorted(topics_in_theme.keys()):
                topic_label = topics_in_theme[topic_id]
                print(f"\n  TOPIC {topic_id}: {topic_label}")
                
                codes_in_topic = topic_codes[theme_id][topic_id]
                code_counts = Counter(codes_in_topic)
                sorted_code_counts = sorted(code_counts.items())
                
                print(f"  Unique Codes ({len(sorted_code_counts)}):")
                for (code_id, code_label), count in sorted_code_counts:
                    print(f"    CODE {code_id}: {code_label} (#{count})")
        
        # Summary statistics
        print("\n=== SUMMARY STATISTICS ===")
        print(f"Total respondents processed: {len(label_results)}")
        print(f"Total segments processed: {sum(len(r.response_segment) for r in label_results)}")
        print(f"Total themes: {len(themes)}")
        total_topics = sum(len(topics) for topics in theme_topics.values())
        print(f"Total topics: {total_topics}")
        total_codes = len(set([code for theme in topic_codes.values() for topic in theme.values() for code in topic]))
        print(f"Total codes: {total_codes}")
        print(f"Original clusters: {len(unique_clusters)}")
        
    else:
        print("No cached cluster data found.")
        print("Please run the clustering pipeline first:")
        print("  python clusterer.py")