"""Test the new labeller with cached cluster data"""
import asyncio
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project paths
sys.path.append(str(Path(__file__).parents[2]))  # Add src directory

import models
from cache_manager import CacheManager
from cache_config import CacheConfig
import data_io
from modules.utils.labeller import Labeller

async def main(limit=None, save_overview=False):
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
        
        # Run the pipeline
        if limit:
            print(f"\nLimiting to first {limit} results for testing...")
            test_results = cluster_results[:limit]
        else:
            print(f"\nProcessing all {len(cluster_results)} results...")
            test_results = cluster_results
            
        label_results = labeller.create_hierarchical_labels(test_results, var_lab)
        
        # Save to cache
        cache_manager.save_to_cache(label_results, filename, 'labels_test')
        print(f"\nSaved {len(label_results)} label results to cache")
        
        # Print summary
        print("\n=== Label Summary ===")
        print(f"Total results: {len(label_results)}")
        
        # Analyze hierarchical structure
        themes = {}
        topics = {}
        codes = {}
        theme_summaries = {}
        
        # Collect all labels
        for result in label_results:
            # Extract theme summaries
            if result.summary and result.summary.strip():
                theme_summaries[result.respondent_id] = result.summary
            
            for segment in result.response_segment:
                if segment.Theme:
                    for theme_id, theme_label in segment.Theme.items():
                        themes[theme_id] = theme_label
                
                if segment.Topic:
                    for topic_id, topic_label in segment.Topic.items():
                        topics[topic_id] = {
                            'label': topic_label,
                            'theme_id': list(segment.Theme.keys())[0] if segment.Theme else None
                        }
                
                if segment.Code:
                    for code_id, code_label in segment.Code.items():
                        codes[code_id] = {
                            'label': code_label,
                            'topic_id': list(segment.Topic.keys())[0] if segment.Topic else None,
                            'theme_id': list(segment.Theme.keys())[0] if segment.Theme else None
                        }
        
        # Display hierarchical overview
        print("\n=== HIERARCHICAL LABELLING OVERVIEW ===")
        print(f"\nTotal Themes: {len(themes)}")
        print(f"Total Topics: {len(topics)}")
        print(f"Total Codes: {len(codes)}")
        
        # Display by theme
        for theme_id in sorted(themes.keys()):
            theme_label = themes[theme_id]
            print(f"\n{'='*60}")
            print(f"THEME {theme_id}: {theme_label}")
            print(f"{'='*60}")
            
            # Find theme summary
            theme_summary = None
            for resp_id, summary in theme_summaries.items():
                if summary:  # Just use the first available summary for this theme
                    theme_summary = summary
                    break
            
            if theme_summary:
                print(f"\nTheme Summary:")
                print(f"{theme_summary[:500]}...")
                print()
            
            # Display topics under this theme
            theme_topics = {tid: tdata for tid, tdata in topics.items() 
                          if tdata['theme_id'] == theme_id}
            
            for topic_id in sorted(theme_topics.keys()):
                topic_data = theme_topics[topic_id]
                print(f"\n  TOPIC {topic_id}: {topic_data['label']}")
                
                # Display codes under this topic
                topic_codes = {cid: cdata for cid, cdata in codes.items() 
                             if cdata['topic_id'] == topic_id}
                
                for code_id in sorted(topic_codes.keys()):
                    code_data = topic_codes[code_id]
                    print(f"    CODE {code_id}: {code_data['label']}")
            
            print()
        
        # Show cluster mapping
        print("\n=== CLUSTER TO LABEL MAPPING ===")
        cluster_mapping = {}
        
        for result in label_results:
            for segment in result.response_segment:
                if segment.mirco_cluster:
                    cluster_id = list(segment.mirco_cluster.keys())[0]
                    theme_label = list(segment.Theme.values())[0] if segment.Theme else "Unknown"
                    topic_label = list(segment.Topic.values())[0] if segment.Topic else "Unknown"
                    code_label = list(segment.Code.values())[0] if segment.Code else "Unknown"
                    
                    if cluster_id not in cluster_mapping:
                        cluster_mapping[cluster_id] = {
                            'theme': theme_label,
                            'topic': topic_label,
                            'code': code_label,
                            'count': 0
                        }
                    cluster_mapping[cluster_id]['count'] += 1
        
        # Display cluster mappings
        print(f"\nTotal Original Clusters Mapped: {len(cluster_mapping)}")
        print("\nOriginal Clusters → Labels (showing all):")
        for i, (cluster_id, mapping) in enumerate(sorted(cluster_mapping.items())):
            print(f"Cluster {cluster_id} ({mapping['count']} segments):")
            print(f"  → Theme: {mapping['theme']}")
            print(f"  → Topic: {mapping['topic']}")
            print(f"  → Code: {mapping['code']}")
            if i < len(cluster_mapping) - 1:  # Don't print separator after last item
                print()
        
        # Summary statistics
        print("\n=== SUMMARY STATISTICS ===")
        print(f"Total respondents processed: {len(label_results)}")
        print(f"Total segments processed: {sum(len(r.response_segment) for r in label_results)}")
        print(f"Total themes created: {len(themes)}")
        print(f"Total topics created: {len(topics)}")
        print(f"Total codes created: {len(codes)}")
        print(f"Original clusters: {len(unique_clusters)}")
        print(f"Merged clusters: {len(cluster_mapping)}")
        print(f"Merge ratio: {1 - (len(cluster_mapping) / len(unique_clusters)):.2%}")
        
        # Save overview to file if requested
        if save_overview:
            overview_file = Path(__file__).parent / "labeller_overview.txt"
            with open(overview_file, 'w', encoding='utf-8') as f:
                f.write(f"=== HIERARCHICAL LABELLING OVERVIEW ===\n")
                f.write(f"Question: {var_lab}\n")
                f.write(f"Processing date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write(f"Total Themes: {len(themes)}\n")
                f.write(f"Total Topics: {len(topics)}\n")
                f.write(f"Total Codes: {len(codes)}\n\n")
                
                for theme_id in sorted(themes.keys()):
                    theme_label = themes[theme_id]
                    f.write(f"\n{'='*60}\n")
                    f.write(f"THEME {theme_id}: {theme_label}\n")
                    f.write(f"{'='*60}\n")
                    
                    # Find theme summary
                    theme_summary = None
                    for resp_id, summary in theme_summaries.items():
                        if summary:
                            theme_summary = summary
                            break
                    
                    if theme_summary:
                        f.write(f"\nTheme Summary:\n{theme_summary}\n\n")
                    
                    # Topics and codes
                    theme_topics = {tid: tdata for tid, tdata in topics.items() 
                                  if tdata['theme_id'] == theme_id}
                    
                    for topic_id in sorted(theme_topics.keys()):
                        topic_data = theme_topics[topic_id]
                        f.write(f"\n  TOPIC {topic_id}: {topic_data['label']}\n")
                        
                        topic_codes = {cid: cdata for cid, cdata in codes.items() 
                                     if cdata['topic_id'] == topic_id}
                        
                        for code_id in sorted(topic_codes.keys()):
                            code_data = topic_codes[code_id]
                            f.write(f"    CODE {code_id}: {code_data['label']}\n")
                
                f.write(f"\n\n=== CLUSTER MAPPING ===\n")
                for cluster_id, mapping in sorted(cluster_mapping.items()):
                    f.write(f"\nCluster {cluster_id} ({mapping['count']} segments):\n")
                    f.write(f"  → Theme: {mapping['theme']}\n")
                    f.write(f"  → Topic: {mapping['topic']}\n")
                    f.write(f"  → Code: {mapping['code']}\n")
                
                f.write(f"\n\n=== SUMMARY STATISTICS ===\n")
                f.write(f"Total respondents processed: {len(label_results)}\n")
                f.write(f"Total segments processed: {sum(len(r.response_segment) for r in label_results)}\n")
                f.write(f"Original clusters: {len(unique_clusters)}\n")
                f.write(f"Merged clusters: {len(cluster_mapping)}\n")
                f.write(f"Merge ratio: {1 - (len(cluster_mapping) / len(unique_clusters)):.2%}\n")
                
            print(f"\nOverview saved to: {overview_file}")
            
            # Also save as Excel-friendly CSV
            import csv
            csv_file = Path(__file__).parent / "labeller_hierarchy.csv"
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Original_Cluster_ID', 'Theme_ID', 'Theme_Label', 
                               'Topic_ID', 'Topic_Label', 'Code_ID', 'Code_Label', 
                               'Segment_Count'])
                
                for cluster_id, mapping in sorted(cluster_mapping.items()):
                    theme_id = next((tid for tid, tlabel in themes.items() 
                                   if tlabel == mapping['theme']), '')
                    topic_id = next((tid for tid, tdata in topics.items() 
                                   if tdata['label'] == mapping['topic']), '')
                    code_id = next((cid for cid, cdata in codes.items() 
                                  if cdata['label'] == mapping['code']), '')
                    
                    writer.writerow([cluster_id, theme_id, mapping['theme'],
                                   topic_id, mapping['topic'], code_id, mapping['code'],
                                   mapping['count']])
            
            print(f"CSV hierarchy saved to: {csv_file}")
        
    else:
        print("No cached cluster data found.")
        print("Please run the clustering pipeline first:")
        print("  python pipeline.py --force-step clusters")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the labeller with cached cluster data')
    parser.add_argument('--limit', type=int, default=None, 
                        help='Limit number of results to process (default: all)')
    parser.add_argument('--all', action='store_true', 
                        help='Process all results (default if no limit specified)')
    parser.add_argument('--save', action='store_true',
                        help='Save overview to file (labeller_overview.txt and labeller_hierarchy.csv)')
    
    args = parser.parse_args()
    
    # Run with limit or all
    asyncio.run(main(limit=args.limit, save_overview=args.save))