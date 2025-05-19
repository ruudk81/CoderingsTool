import asyncio
from typing import List, Dict, Set, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import AsyncOpenAI
import logging

try:
    # When running as a script
    from labeller import (
        LabellerConfig, ClusterData, InitialLabel, MergeMapping
    )
except ImportError:
    # When imported as a module
    from .labeller import (
        LabellerConfig, ClusterData, InitialLabel, MergeMapping
    )

logger = logging.getLogger(__name__)


class Phase2Merger:
    """Phase 2: Merge clusters that are not meaningfully differentiated"""
    
    def __init__(self, config: LabellerConfig):
        self.config = config
        self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        self.openai_client = AsyncOpenAI(api_key=config.api_key)
    
    async def merge_similar_clusters(self,
                                   cluster_data: Dict[int, ClusterData],
                                   initial_labels: Dict[int, InitialLabel],
                                   var_lab: str) -> MergeMapping:
        """Main method to identify and merge similar clusters"""
        logger.info("Phase 2: Analyzing cluster similarity and merging based on labels...")
        
        # Get label embeddings instead of using centroids
        label_embeddings, cluster_ids = await self._get_label_embeddings(initial_labels)
        
        # Calculate similarities based on label embeddings
        similarities = cosine_similarity(label_embeddings)
        
        # Show similarity distribution analysis for label embeddings
        self._show_label_similarity_distribution(similarities, cluster_ids)
        
        # Auto-merge clusters based on label similarity
        auto_merge_groups = self._auto_merge_by_label_similarity(similarities, cluster_ids)
        logger.info(f"Auto-merged {len(auto_merge_groups)} groups based on label similarity")
        
        # Create final merge mapping
        merge_mapping = self._create_merge_mapping(auto_merge_groups, initial_labels)
        
        logger.info(f"Final merge: {len(cluster_data)} → {len(merge_mapping.merge_groups)} clusters")
        
        return merge_mapping
    
    
    async def _get_label_embeddings(self, initial_labels: Dict[int, InitialLabel]) -> Tuple[np.ndarray, List[int]]:
        """Get embeddings for labels enriched with keywords"""
        
        cluster_ids = list(initial_labels.keys())
        
        # Create enriched text: label + keywords
        enriched_texts = []
        for cid in cluster_ids:
            label = initial_labels[cid]
            # Combine label with top 5 keywords
            text = f"{label.label}. Keywords: {', '.join(label.keywords[:5])}"
            enriched_texts.append(text)
        
        logger.info(f"Getting embeddings for {len(enriched_texts)} enriched labels...")
        
        # Get embeddings in batches
        embeddings = []
        batch_size = 100  # OpenAI can handle up to 2048
        
        for i in range(0, len(enriched_texts), batch_size):
            batch = enriched_texts[i:i + batch_size]
            response = await self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=batch
            )
            embeddings.extend([e.embedding for e in response.data])
        
        logger.info(f"Successfully obtained {len(embeddings)} label embeddings")
        
        return np.array(embeddings), cluster_ids
    
    def _show_label_similarity_distribution(self, similarities: np.ndarray, cluster_ids: List[int]):
        """Show distribution analysis of label similarities"""
        if len(cluster_ids) <= 1:
            return
        
        # Extract upper triangle (excluding diagonal)
        all_similarities = []
        for i in range(len(cluster_ids)):
            for j in range(i+1, len(cluster_ids)):
                all_similarities.append(similarities[i, j])
        
        all_similarities = np.array(all_similarities)
        
        # Calculate statistics
        stats = {
            'min': np.min(all_similarities),
            'max': np.max(all_similarities),
            'mean': np.mean(all_similarities),
            'median': np.median(all_similarities),
            'std': np.std(all_similarities),
            'p90': np.percentile(all_similarities, 90),
            'p95': np.percentile(all_similarities, 95),
            'p99': np.percentile(all_similarities, 99)
        }
        
        # Create histogram bins: 0.05 steps up to 0.9, then 0.01 steps from 0.9 to 1.0
        bins_coarse = np.arange(0, 0.9, 0.05)
        bins_fine = np.arange(0.9, 1.01, 0.01)
        bins = np.concatenate([bins_coarse, bins_fine])
        hist, bin_edges = np.histogram(all_similarities, bins=bins)
        
        logger.info("\n=== Label Cosine Similarity Distribution Analysis ===")
        logger.info(f"Total pairs analyzed: {len(all_similarities)}")
        logger.info("\nStatistics:")
        logger.info(f"  Min:    {stats['min']:.4f}")
        logger.info(f"  Max:    {stats['max']:.4f}")
        logger.info(f"  Mean:   {stats['mean']:.4f}")
        logger.info(f"  Median: {stats['median']:.4f}")
        logger.info(f"  Std:    {stats['std']:.4f}")
        logger.info(f"  90th percentile: {stats['p90']:.4f}")
        logger.info(f"  95th percentile: {stats['p95']:.4f}")
        logger.info(f"  99th percentile: {stats['p99']:.4f}")
        
        logger.info("\nDistribution:")
        max_count = max(hist)
        for i, count in enumerate(hist):
            start = bin_edges[i]
            end = bin_edges[i+1]
            bar_length = int(40 * count / max_count) if max_count > 0 else 0
            bar = '█' * bar_length
            logger.info(f"  {start:.3f}-{end:.3f}: {bar} ({count} pairs)")
        
        # Count pairs above certain thresholds
        thresholds = [0.9, 0.95, 0.98, 0.99]
        logger.info("\nPairs above thresholds:")
        for threshold in thresholds:
            count = np.sum(all_similarities > threshold)
            percentage = 100 * count / len(all_similarities)
            logger.info(f"  > {threshold}: {count} pairs ({percentage:.1f}%)")
        
        logger.info("=" * 50)
    
    
    def _auto_merge_by_label_similarity(self, similarities: np.ndarray, cluster_ids: List[int]) -> List[List[int]]:
        """Auto-merge clusters based on label similarity"""
        if len(cluster_ids) <= 1:
            return [[cid] for cid in cluster_ids]
        
        # Log high similarity pairs
        high_similarity_pairs = []
        for i in range(len(cluster_ids)):
            for j in range(i+1, len(cluster_ids)):
                sim_score = similarities[i, j]
                if sim_score > 0.9:  # Log pairs above 0.9 similarity
                    high_similarity_pairs.append((cluster_ids[i], cluster_ids[j], sim_score))
        
        # Sort by similarity score
        high_similarity_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Log the highest similarity pairs
        logger.info("High label similarity pairs (>0.9):")
        for i, (cid1, cid2, score) in enumerate(high_similarity_pairs[:10]):
            logger.info(f"  Clusters {cid1} & {cid2}: {score:.4f}")
        if len(high_similarity_pairs) > 10:
            logger.info(f"  ... and {len(high_similarity_pairs) - 10} more pairs")
        
        # Find groups to merge
        merged = set()
        merge_groups = []
        merge_decisions = []  # Track merge decisions for logging
        
        for i, cid1 in enumerate(cluster_ids):
            if cid1 in merged:
                continue
            
            group = [cid1]
            merged.add(cid1)
            
            for j, cid2 in enumerate(cluster_ids[i+1:], i+1):
                if cid2 in merged:
                    continue
                
                sim_score = similarities[i, j]
                if sim_score > self.config.similarity_threshold:
                    group.append(cid2)
                    merged.add(cid2)
                    merge_decisions.append((cid1, cid2, sim_score))
            
            merge_groups.append(group)
        
        # Log merge decisions
        logger.info(f"\nAuto-merge decisions based on labels (threshold={self.config.similarity_threshold}):")
        for cid1, cid2, score in merge_decisions[:10]:
            logger.info(f"  Merging {cid1} & {cid2} (label similarity: {score:.4f})")
        if len(merge_decisions) > 10:
            logger.info(f"  ... and {len(merge_decisions) - 10} more merges")
        
        # Add any remaining clusters as singleton groups
        for cid in cluster_ids:
            if cid not in merged:
                merge_groups.append([cid])
        
        return merge_groups
    
    def _create_merge_mapping(self,
                            merge_groups: List[List[int]],
                            initial_labels: Dict[int, InitialLabel]) -> MergeMapping:
        """Create final merge mapping"""
        cluster_to_merged = {}
        merge_reasons = {}
        
        for i, group in enumerate(merge_groups):
            # All clusters in group map to the same merged ID
            for cid in group:
                cluster_to_merged[cid] = i
            
            # Create merge reason
            if len(group) > 1:
                labels = [initial_labels[cid].label for cid in group[:3]]
                merge_reasons[i] = f"Merged similar clusters: {', '.join(labels)}"
            else:
                merge_reasons[i] = f"No merge needed: {initial_labels[group[0]].label}"
        
        return MergeMapping(
            merge_groups=merge_groups,
            cluster_to_merged=cluster_to_merged,
            merge_reasons=merge_reasons
        )


if __name__ == "__main__":
    """Test Phase 2 with actual cached data from Phase 1"""
    import sys
    from pathlib import Path
    import json
    import pickle
    
    # Add project paths
    sys.path.append(str(Path(__file__).parents[2]))  # Add src directory
    
    from config import OPENAI_API_KEY, DEFAULT_MODEL
    from cache_manager import CacheManager
    from cache_config import CacheConfig
    import data_io
    import models
    
    # Initialize cache manager
    cache_config = CacheConfig()
    cache_manager = CacheManager(cache_config)
    
    # File and variable info
    filename = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
    var_name = "Q20"
    
    # Load cluster results from cache
    cluster_results = cache_manager.load_from_cache(filename, "clusters", models.ClusterModel)
    
    # Load phase 1 labels from cache
    phase1_labels = cache_manager.load_intermediate_data(filename, "phase1_labels", dict)
    
    if cluster_results and phase1_labels:
        print(f"Loaded {len(cluster_results)} cluster results from cache")
        print(f"Loaded {len(phase1_labels)} phase 1 labels from cache")
        print(f"Type of phase1_labels: {type(phase1_labels)}")
        
        # Print first item to debug
        first_key = list(phase1_labels.keys())[0]
        print(f"First key: {first_key}, type: {type(first_key)}")
        print(f"First value type: {type(phase1_labels[first_key])}")
        if hasattr(phase1_labels[first_key], '__dict__'):
            print(f"First value attributes: {phase1_labels[first_key].__dict__}")
        
        try:
            # Extract cluster data from the results
            from labeller import Labeller
            temp_labeller = Labeller()
            cluster_data = temp_labeller.extract_cluster_data(cluster_results)
            print(f"Extracted data for {len(cluster_data)} clusters")
        except Exception as e:
            print(f"Error extracting cluster data: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        try:
            # Convert labels to InitialLabel objects if needed
            initial_labels = {}
            for cluster_id, label_data in phase1_labels.items():
                try:
                    # Check if it's already an InitialLabel instance
                    if hasattr(label_data, 'label') and hasattr(label_data, 'keywords') and hasattr(label_data, 'confidence'):
                        # Already an InitialLabel object
                        initial_labels[cluster_id] = label_data
                    elif isinstance(label_data, dict):
                        # Convert from dict
                        initial_labels[cluster_id] = InitialLabel(**label_data)
                    else:
                        print(f"Unexpected type for label_data: {type(label_data)}")
                        # Try to use it directly
                        initial_labels[cluster_id] = label_data
                except Exception as e:
                    print(f"Error processing cluster {cluster_id}: {e}")
                    print(f"Type of label_data: {type(label_data)}")
                    print(f"Content: {label_data}")
                    raise
        except Exception as e:
            print(f"Error converting labels: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Get variable label
        data_loader = data_io.DataLoader()
        var_lab = data_loader.get_varlab(filename=filename, var_name=var_name)
        print(f"Variable label: {var_lab}")
        
        # Initialize configuration
        config = LabellerConfig(
            api_key=OPENAI_API_KEY,
            model=DEFAULT_MODEL,
            batch_size=10,
            similarity_threshold=0.98,  # For auto-merge - raised to be less aggressive
            merge_score_threshold=0.7   # For LLM merge (not used anymore)
        )
    
        # Initialize phase 2 merger
        phase2 = Phase2Merger(config)
        
        async def run_test():
            """Run the test"""
            print("=== Testing Phase 2: Cluster Merging with Real Data ===")
            print(f"Variable label: {var_lab}")
            print(f"Number of clusters: {len(cluster_data)}")
            
            try:
                # Set up logging for the test
                import logging
                logging.basicConfig(level=logging.INFO, format='%(message)s')
                
                print("\n=== Starting detailed merge analysis ===\n")
                
                # Now we'll use label embeddings for merge analysis
                print("=== LABEL-BASED MERGE ANALYSIS ===")
                
                # Get label embeddings
                label_embeddings, cluster_ids = await phase2._get_label_embeddings(initial_labels)
                
                # Calculate pairwise similarities of labels
                similarities = cosine_similarity(label_embeddings)
                
                # Capture all similarity scores
                label_merge_scores = []
                for i in range(len(cluster_ids)):
                    for j in range(i+1, len(cluster_ids)):
                        sim_score = similarities[i, j]
                        label_merge_scores.append({
                            'cluster1': cluster_ids[i],
                            'cluster2': cluster_ids[j],
                            'score': sim_score,
                            'merged': sim_score > config.similarity_threshold
                        })
                
                # Sort by score descending
                label_merge_scores.sort(key=lambda x: x['score'], reverse=True)
                
                print(f"\nTop 20 Label similarity scores (threshold={config.similarity_threshold}):")
                for i, score_info in enumerate(label_merge_scores[:20]):
                    status = "MERGED" if score_info['merged'] else "not merged"
                    cid1 = score_info['cluster1']
                    cid2 = score_info['cluster2']
                    label1_obj = initial_labels[cid1]
                    label2_obj = initial_labels[cid2]
                    print(f"  {i+1}. Clusters {cid1} & {cid2}: {score_info['score']:.4f} ({status})")
                    print(f"     Label 1: {label1_obj.label} | Keywords: {', '.join(label1_obj.keywords[:5])}")
                    print(f"     Label 2: {label2_obj.label} | Keywords: {', '.join(label2_obj.keywords[:5])}")
                
                # Create histogram of label merge scores
                label_scores_only = [s['score'] for s in label_merge_scores]
                
                # Create bins: 0.05 steps up to 0.9, then 0.01 steps from 0.9 to 1.0
                bins_coarse = np.arange(0, 0.9, 0.05)
                bins_fine = np.arange(0.9, 1.01, 0.01)
                bins = np.concatenate([bins_coarse, bins_fine])
                hist, bin_edges = np.histogram(label_scores_only, bins=bins)
                
                print("\nLabel similarity score distribution:")
                max_count = max(hist)
                for i, count in enumerate(hist):
                    start = bin_edges[i]
                    end = bin_edges[i+1]
                    bar_length = int(40 * count / max_count) if max_count > 0 else 0
                    bar = '█' * bar_length
                    print(f"  {start:.3f}-{end:.3f}: {bar} ({count} pairs)")
                
                # Now run merge process (label similarity)
                print("\n=== MERGE PROCESS (Label Similarity) ===")
                
                # Run merge process
                merge_mapping = await phase2.merge_similar_clusters(
                    cluster_data, initial_labels, var_lab
                )
                
                print(f"\nThreshold Analysis:")
                print(f"Auto-merge threshold: {config.similarity_threshold}")
                
                label_merged_count = sum(1 for s in label_merge_scores if s['merged'])
                print(f"Label-based merge merged: {label_merged_count} pairs")
                
                # Display results
                print("\n=== Merge Results ===")
                print(f"Number of merge groups: {len(merge_mapping.merge_groups)}")
                
                # Show some examples of merged clusters
                print(f"\nSample cluster mappings (first 10):")
                for i, (cid, merged_id) in enumerate(merge_mapping.cluster_to_merged.items()):
                    if i >= 10:
                        break
                    print(f"  Cluster {cid} → Merged cluster {merged_id}")
                
                # Calculate statistics
                original_count = len(cluster_data)
                merged_count = len(merge_mapping.merge_groups)
                reduction = (1 - merged_count/original_count) * 100
                
                print(f"\nStatistics:")
                print(f"  Original clusters: {original_count}")
                print(f"  Merged clusters: {merged_count}")
                print(f"  Reduction: {reduction:.1f}%")
                
                # Provide threshold recommendations
                print("\n=== THRESHOLD RECOMMENDATIONS ===")
                
                # Analyze label-merge scores to suggest threshold
                label_scores_sorted = sorted([s['score'] for s in label_merge_scores], reverse=True)
                percentiles = np.percentile(label_scores_sorted, [99, 98, 97, 96, 95])
                
                print("\nLabel similarity score percentiles:")
                for i, (p, val) in enumerate(zip([99, 98, 97, 96, 95], percentiles)):
                    print(f"  {p}th percentile: {val:.4f}")
                
                print("\nSuggested adjustments to reduce merging:")
                print(f"  Current label-merge threshold: {config.similarity_threshold}")
                print(f"  Consider raising to: {percentiles[1]:.4f} (98th percentile) or {percentiles[0]:.4f} (99th percentile)")
                
                # Save to cache for phase 3
                cache_key = 'phase2_merge_mapping'
                cache_data = {
                    'merge_mapping': merge_mapping,
                    'cluster_data': cluster_data,
                    'initial_labels': initial_labels
                }
                cache_manager.cache_intermediate_data(cache_data, filename, cache_key)
                print(f"\nSaved results to cache with key '{cache_key}'")
                
                # Save to JSON for inspection
                output_data = {
                    "merge_groups": merge_mapping.merge_groups,
                    "cluster_mapping": merge_mapping.cluster_to_merged,
                    "merge_reasons": merge_mapping.merge_reasons,
                    "statistics": {
                        "original": original_count,
                        "merged": merged_count,
                        "reduction_percent": reduction
                    }
                }
                
                output_file = Path("phase2_test_results.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                
                print(f"Results saved to: {output_file}")
                
            except Exception as e:
                print(f"Error during testing: {e}")
                import traceback
                traceback.print_exc()
        
        # Run the test
        asyncio.run(run_test())
    else:
        print("Missing required cached data.")
        print("Please ensure you have run:")
        print("  1. python clusterer.py")
        print("  2. python phase1_labeller.py")