import asyncio
from typing import List, Dict, Set, Tuple, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import AsyncOpenAI
import logging
from tqdm.asyncio import tqdm
import networkx as nx

try:
    # When running as a script
    from labeller import (
        LabellerConfig, ClusterData, InitialLabel, MergeMapping,
        BatchSimilarityResponse, SimilarityScore
    )
except ImportError:
    # When imported as a module
    from .labeller import (
        LabellerConfig, ClusterData, InitialLabel, MergeMapping,
        BatchSimilarityResponse, SimilarityScore
    )
from prompts import SIMILARITY_SCORING_PROMPT

logger = logging.getLogger(__name__)


class Phase2Merger:
    """Phase 2: Merge clusters that are not meaningfully differentiated from the perspective of the research question"""
    
    def __init__(self, config: LabellerConfig, client=None):
        self.config = config
        self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        self.client = client or AsyncOpenAI(api_key=config.api_key)
    
    async def merge_similar_clusters(self,
                                   cluster_data: Dict[int, ClusterData],
                                   initial_labels: Dict[int, InitialLabel],
                                   var_lab: str) -> MergeMapping:
        """Main method to identify and merge similar clusters"""
        logger.info(f"Phase 2: Analyzing cluster similarity for research question: '{var_lab}'")
        
        # Step 1: First identify very similar clusters by embedding cosine similarity (fast pre-filter)
        auto_merge_groups = await self._identify_highly_similar_clusters(cluster_data)
        logger.info(f"Auto-merged {len(auto_merge_groups)} groups based on embedding similarity > {self.config.similarity_threshold}")
        
        # Step 2: Create initial cluster ID groups from auto-merge results
        cluster_groups = self._initialize_cluster_groups(auto_merge_groups, list(cluster_data.keys()))
        logger.info(f"Starting with {len(cluster_groups)} cluster groups after auto-merging")
        
        # Step 3: Use LLM to analyze semantic similarity between remaining groups
        merge_graph = await self._analyze_group_similarity(cluster_groups, cluster_data, initial_labels, var_lab)
        
        # Step 4: Use graph-based connected components to find merge groups
        final_merge_groups = self._find_connected_components(merge_graph, cluster_groups)
        logger.info(f"Final merge: {len(cluster_data)} clusters → {len(final_merge_groups)} groups")
        
        # Step 5: Create final merge mapping
        merge_mapping = self._create_merge_mapping(final_merge_groups, initial_labels)
        
        return merge_mapping
    
    async def _identify_highly_similar_clusters(self, cluster_data: Dict[int, ClusterData]) -> List[List[int]]:
        """Identify clusters with very high embedding similarity for auto-merging"""
        logger.info("Pre-filtering clusters based on embedding similarity...")
        
        # Get all cluster IDs
        cluster_ids = list(cluster_data.keys())
        if len(cluster_ids) <= 1:
            return [[cid] for cid in cluster_ids]
        
        # Calculate pairwise similarities between cluster centroids
        centroids = np.array([cluster_data[cid].centroid for cid in cluster_ids])
        similarities = cosine_similarity(centroids)
        
        # Find clusters to merge based on high similarity
        similarity_threshold = self.config.similarity_threshold
        merge_pairs = []
        
        for i in range(len(cluster_ids)):
            for j in range(i+1, len(cluster_ids)):
                sim_score = similarities[i, j]
                if sim_score > similarity_threshold:
                    merge_pairs.append((cluster_ids[i], cluster_ids[j]))
                    logger.info(f"Auto-merge candidates: Clusters {cluster_ids[i]} & {cluster_ids[j]} (similarity: {sim_score:.4f})")
        
        # Use graph-based connected components to find groups
        if not merge_pairs:
            return []  # No highly similar clusters found
            
        G = nx.Graph()
        G.add_nodes_from(cluster_ids)
        G.add_edges_from(merge_pairs)
        
        # Get connected components as merge groups
        merge_groups = [list(comp) for comp in nx.connected_components(G) if len(comp) > 1]
        
        return merge_groups
    
    def _initialize_cluster_groups(self, auto_merge_groups: List[List[int]], all_cluster_ids: List[int]) -> List[List[int]]:
        """Initialize cluster groups based on auto-merge results"""
        # Create a set of all clusters that are already in merge groups
        merged_clusters = set()
        for group in auto_merge_groups:
            merged_clusters.update(group)
        
        # Add singleton groups for clusters not in any merge group
        result_groups = list(auto_merge_groups)  # Make a copy
        for cid in all_cluster_ids:
            if cid not in merged_clusters:
                result_groups.append([cid])
        
        return result_groups
    
    async def _analyze_group_similarity(self,
                                      cluster_groups: List[List[int]],
                                      cluster_data: Dict[int, ClusterData],
                                      initial_labels: Dict[int, InitialLabel],
                                      var_lab: str) -> nx.Graph:
        """Analyze semantic similarity between cluster groups using LLM"""
        logger.info(f"Analyzing semantic similarity between {len(cluster_groups)} cluster groups...")
        
        # Create similarity graph
        G = nx.Graph()
        
        # Add all clusters as nodes
        for group in cluster_groups:
            for cid in group:
                G.add_node(cid)
        
        # Add edges within auto-merged groups (they're already determined to be similar)
        for group in cluster_groups:
            if len(group) > 1:
                for i in range(len(group)):
                    for j in range(i+1, len(group)):
                        G.add_edge(group[i], group[j], 
                                  weight=1.0,
                                  reason="Auto-merged based on embedding similarity")
        
        # Create representative for each group (use the first cluster in each group)
        group_representatives = [group[0] for group in cluster_groups]
        
        # Only compare representatives if more than one group exists
        if len(group_representatives) > 1:
            # Create pairs for comparison (only comparing between groups)
            comparison_pairs = []
            for i in range(len(group_representatives)):
                for j in range(i+1, len(group_representatives)):
                    # Get original groups these representatives belong to
                    group1 = cluster_groups[i]
                    group2 = cluster_groups[j]
                    
                    # Add pair for comparison
                    comparison_pairs.append((group_representatives[i], group_representatives[j]))
            
            # Check similarity between representative pairs
            G = await self._analyze_pair_similarity(
                comparison_pairs, cluster_data, initial_labels, var_lab, G)
        
        return G
    
    async def _analyze_pair_similarity(self,
                                     pairs: List[Tuple[int, int]],
                                     cluster_data: Dict[int, ClusterData],
                                     initial_labels: Dict[int, InitialLabel],
                                     var_lab: str,
                                     graph: nx.Graph) -> nx.Graph:
        """Analyze semantic similarity between cluster pairs using LLM"""
        if not pairs:
            return graph
            
        logger.info(f"Analyzing {len(pairs)} cluster pairs for semantic similarity...")
        
        # Create batches of pairs
        batches = [pairs[i:i + self.config.batch_size] for i in range(0, len(pairs), self.config.batch_size)]
        
        # Process batches with progress bar
        with tqdm(total=len(pairs), desc="Analyzing cluster pairs") as pbar:
            for batch in batches:
                # Create prompt for batch analysis
                prompt = self._create_similarity_prompt(batch, cluster_data, initial_labels, var_lab)
                
                # Get similarity scores
                try:
                    async with self.semaphore:
                        scores = await self._get_similarity_scores(prompt)
                        
                        # Update graph with similarity scores
                        for score in scores.scores:
                            # Only add edges for clusters that should be merged
                            if score.merge_suggested:
                                graph.add_edge(
                                    score.cluster_id_1,
                                    score.cluster_id_2,
                                    weight=score.score,
                                    reason=score.reason
                                )
                                logger.info(f"Merge suggested: Clusters {score.cluster_id_1} & {score.cluster_id_2} "
                                           f"(score: {score.score:.2f}, reason: {score.reason})")
                            else:
                                logger.debug(f"No merge: Clusters {score.cluster_id_1} & {score.cluster_id_2} "
                                           f"(score: {score.score:.2f})")
                        
                        # Update progress
                        pbar.update(len(batch))
                except Exception as e:
                    logger.error(f"Error analyzing similarity batch: {e}")
                    # Continue with next batch
        
        return graph
    
    def _create_similarity_prompt(self,
                                pairs: List[Tuple[int, int]],
                                cluster_data: Dict[int, ClusterData],
                                initial_labels: Dict[int, InitialLabel],
                                var_lab: str) -> str:
        """Create prompt for similarity scoring between cluster pairs"""
        prompt = SIMILARITY_SCORING_PROMPT.replace("{var_lab}", var_lab)
        prompt = prompt.replace("{language}", self.config.language)
        
        # Create cluster pair descriptions
        pair_descriptions = []
        
        for cluster_id_1, cluster_id_2 in pairs:
            # Get data for first cluster
            cluster1 = cluster_data[cluster_id_1]
            label1 = initial_labels[cluster_id_1].label
            
            # Get representative items for first cluster (codes and descriptions)
            codes1 = cluster1.descriptive_codes[:5]  # Use top 5 codes
            descriptions1 = cluster1.code_descriptions[:5]  # Use top 5 descriptions
            
            # Get data for second cluster
            cluster2 = cluster_data[cluster_id_2]
            label2 = initial_labels[cluster_id_2].label
            
            # Get representative items for second cluster (codes and descriptions)
            codes2 = cluster2.descriptive_codes[:5]  # Use top 5 codes
            descriptions2 = cluster2.code_descriptions[:5]  # Use top 5 descriptions
            
            # Create pair description
            description = f"\n\nCOMPARISON: Cluster {cluster_id_1} vs Cluster {cluster_id_2}\n"
            description += f"\nCluster {cluster_id_1}:"
            description += f"\n- Label: {label1}"
            description += f"\n- Representative codes:"
            for i, code in enumerate(codes1):
                description += f"\n  {i+1}. {code}"
            description += f"\n- Code descriptions:"
            for i, desc in enumerate(descriptions1):
                description += f"\n  {i+1}. {desc}"
            
            description += f"\n\nCluster {cluster_id_2}:"
            description += f"\n- Label: {label2}"
            description += f"\n- Representative codes:"
            for i, code in enumerate(codes2):
                description += f"\n  {i+1}. {code}"
            description += f"\n- Code descriptions:"
            for i, desc in enumerate(descriptions2):
                description += f"\n  {i+1}. {desc}"
            
            pair_descriptions.append(description)
        
        prompt = prompt.replace("{cluster_pairs}", "\n".join(pair_descriptions))
        
        return prompt
    
    async def _get_similarity_scores(self, prompt: str) -> BatchSimilarityResponse:
        """Get similarity scores from LLM"""
        import instructor
        
        # Use the client from initialization if it's already an instructor client, otherwise wrap it
        client = self.client
        if not hasattr(self.client, 'chat') or not hasattr(self.client.chat, 'completions') or not hasattr(self.client.chat.completions, 'create'):
            client = instructor.from_openai(self.client)
        
        messages = [
            {"role": "system", "content": "You are an expert in thematic analysis and survey response clustering."},
            {"role": "user", "content": prompt}
        ]
        
        # With retry logic
        for attempt in range(self.config.max_retries):
            try:
                response = await client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    response_model=BatchSimilarityResponse,
                    temperature=0.3,
                    max_tokens=4000
                )
                return response
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    logger.error(f"Failed to get similarity scores after {self.config.max_retries} attempts: {e}")
                    raise e
    
    def _find_connected_components(self, graph: nx.Graph, cluster_groups: List[List[int]]) -> List[List[int]]:
        """Find connected components in similarity graph"""
        # Get connected components
        connected_components = list(nx.connected_components(graph))
        
        # Convert components to lists
        component_lists = [list(comp) for comp in connected_components]
        
        return component_lists
    
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
            
            # Create merge reason based on whether it's a merge or not
            if len(group) > 1:
                # Get labels of clusters in this group (limit to first 3 for readability)
                labels = [initial_labels[cid].label for cid in group[:3]]
                if len(group) > 3:
                    labels.append(f"and {len(group) - 3} more")
                
                merge_reasons[i] = f"Merged similar clusters: {', '.join(labels)}"
            else:
                # Single cluster, no merging
                merge_reasons[i] = f"No similar clusters found: {initial_labels[group[0]].label}"
        
        return MergeMapping(
            merge_groups=merge_groups,
            cluster_to_merged=cluster_to_merged,
            merge_reasons=merge_reasons
        )


if __name__ == "__main__":
    """Test Phase 2 with actual cached cluster data"""
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
    import instructor
    
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
        
        # Extract cluster data from the results
        from labeller import Labeller
        temp_labeller = Labeller()
        cluster_data = temp_labeller.extract_cluster_data(cluster_results)
        print(f"Extracted data for {len(cluster_data)} clusters")
        
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
        
        # Get variable label
        data_loader = data_io.DataLoader()
        var_lab = data_loader.get_varlab(filename=filename, var_name=var_name)
        print(f"Variable label: {var_lab}")
        
        # Initialize configuration
        config = LabellerConfig(
            api_key=OPENAI_API_KEY,
            model=DEFAULT_MODEL,
            batch_size=5,  # Process a smaller batch size for merge analysis
            similarity_threshold=0.95,  # For auto-merge by embedding similarity
            merge_score_threshold=0.7   # For LLM merge threshold
        )
        
        # Initialize phase 2 merger
        client = instructor.from_openai(AsyncOpenAI(api_key=config.api_key))
        phase2 = Phase2Merger(config, client)
        
        async def run_test():
            """Run the test"""
            print("=== Testing Phase 2: LLM-Based Cluster Merging with Real Data ===")
            print(f"Variable label: {var_lab}")
            print(f"Number of clusters: {len(cluster_data)}")
            
            # Set up logging for test
            import logging
            logging.basicConfig(level=logging.INFO, format='%(message)s')
            
            try:
                # Run merge process
                merge_mapping = await phase2.merge_similar_clusters(
                    cluster_data, initial_labels, var_lab
                )
                
                # Display results
                print("\n=== Merge Results ===")
                print(f"Original clusters: {len(cluster_data)}")
                print(f"Merged clusters: {len(merge_mapping.merge_groups)}")
                
                # Calculate reduction
                reduction = (1 - len(merge_mapping.merge_groups)/len(cluster_data)) * 100
                print(f"Reduction: {reduction:.1f}%")
                
                # Show some examples of merged clusters
                print(f"\nSample merge groups (first 10):")
                for i, group in enumerate(merge_mapping.merge_groups):
                    if i >= 10:
                        break
                    
                    if len(group) > 1:
                        group_labels = [initial_labels[cid].label for cid in group]
                        print(f"\nGroup {i} ({len(group)} clusters):")
                        for j, (cid, label) in enumerate(zip(group, group_labels)):
                            print(f"  {j+1}. Cluster {cid}: {label}")
                    else:
                        print(f"\nGroup {i} (1 cluster):")
                        print(f"  Cluster {group[0]}: {initial_labels[group[0]].label}")
                
                # Count distribution of group sizes
                group_sizes = [len(group) for group in merge_mapping.merge_groups]
                size_counts = {}
                for size in group_sizes:
                    size_counts[size] = size_counts.get(size, 0) + 1
                
                print("\nGroup size distribution:")
                for size, count in sorted(size_counts.items()):
                    print(f"  Size {size}: {count} groups")
                
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
                    "merge_groups": [[int(cid) for cid in group] for group in merge_mapping.merge_groups],
                    "cluster_mapping": {str(k): v for k, v in merge_mapping.cluster_to_merged.items()},
                    "merge_reasons": {str(k): v for k, v in merge_mapping.merge_reasons.items()},
                    "statistics": {
                        "original": len(cluster_data),
                        "merged": len(merge_mapping.merge_groups),
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