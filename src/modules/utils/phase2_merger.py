import asyncio
from typing import List, Dict, Set, Tuple, Any, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import AsyncOpenAI
import logging
from tqdm.asyncio import tqdm
import networkx as nx
import sys
import os
from pathlib import Path

# Disable HTTP request logging from OpenAI client
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Import nest_asyncio to allow nested event loops (for Spyder/IPython)
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass  # Not required outside of Jupyter/IPython environments

# Add the src directory to the path for imports
try:
    # When running as a script
    src_dir = str(Path(__file__).resolve().parents[2])
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
except (NameError, AttributeError):
    # When running in interactive environment like Spyder
    # where __file__ might not be defined
    current_dir = os.getcwd()
    if current_dir.endswith('utils'):
        src_dir = str(Path(current_dir).parents[1])
    else:
        src_dir = os.path.abspath('src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    print(f"Added {src_dir} to path for imports")

from pydantic import BaseModel, Field, ConfigDict

try:
    # Try direct import (from utils folder)
    from labeller import (
        LabellerConfig, ClusterData, InitialLabel, MergeMapping
    )
    from prompts import SIMILARITY_SCORING_PROMPT
    
    # Define new response models for binary merging decision
    class MergeDecision(BaseModel):
        """Decision about whether to merge two clusters"""
        cluster_id_1: int
        cluster_id_2: int
        should_merge: bool
        reason: str
        
        model_config = ConfigDict(from_attributes=True)

    class BatchMergeDecisionResponse(BaseModel):
        """Batch response for merge decisions"""
        decisions: list[MergeDecision]
        
        model_config = ConfigDict(from_attributes=True)
except ImportError:
    try:
        # Try relative import (as module)
        from .labeller import (
            LabellerConfig, ClusterData, InitialLabel, MergeMapping
        )
        from ..prompts import SIMILARITY_SCORING_PROMPT
        
        # Define new response models for binary merging decision
        class MergeDecision(BaseModel):
            """Decision about whether to merge two clusters"""
            cluster_id_1: int
            cluster_id_2: int
            should_merge: bool
            reason: str
            
            model_config = ConfigDict(from_attributes=True)

        class BatchMergeDecisionResponse(BaseModel):
            """Batch response for merge decisions"""
            decisions: list[MergeDecision]
            
            model_config = ConfigDict(from_attributes=True)
    except ImportError:
        # Try absolute import (from any directory)
        from src.modules.utils.labeller import (
            LabellerConfig, ClusterData, InitialLabel, MergeMapping
        )
        from src.prompts import SIMILARITY_SCORING_PROMPT
        
        # Define new response models for binary merging decision
        class MergeDecision(BaseModel):
            """Decision about whether to merge two clusters"""
            cluster_id_1: int
            cluster_id_2: int
            should_merge: bool
            reason: str
            
            model_config = ConfigDict(from_attributes=True)

        class BatchMergeDecisionResponse(BaseModel):
            """Batch response for merge decisions"""
            decisions: list[MergeDecision]
            
            model_config = ConfigDict(from_attributes=True)

logger = logging.getLogger(__name__)


class Phase2Merger:
    """Phase 2: Merge clusters that are not meaningfully differentiated from the perspective of the research question"""
    
    def __init__(self, config: LabellerConfig, client=None):
        self.config = config
        self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        self.client = client or AsyncOpenAI(api_key=config.api_key)
        self.max_merge_group_size = 5      # Maximum size for any merge group
        self.min_similarity_threshold = 0.8  # Initial threshold for candidate filtering
        self.learned_threshold = 0.8       # Dynamic threshold that will be updated based on LLM judgments
    
    async def merge_similar_clusters(self,
                                   cluster_data: Dict[int, ClusterData],
                                   initial_labels: Dict[int, InitialLabel],
                                   var_lab: str) -> MergeMapping:
        """Main method to identify and merge similar clusters with dynamic similarity threshold learning"""
        logger.info(f"Phase 2: Analyzing cluster similarity for research question: '{var_lab}'")
        
        # Calculate all pairwise similarities between clusters
        similarities_dict = await self._calculate_all_similarities(cluster_data)
        logger.info(f"Calculated similarities between all {len(cluster_data)} clusters")
        logger.info(f"Starting with similarity threshold of {self.learned_threshold}")
        
        # Merge clusters using similarity-guided LLM decisions
        merge_groups = await self._similarity_guided_merging(
            list(cluster_data.keys()), 
            similarities_dict, 
            cluster_data, 
            initial_labels, 
            var_lab
        )
        
        logger.info(f"Final merge: {len(cluster_data)} clusters → {len(merge_groups)} groups")
        logger.info(f"Final learned similarity threshold: {self.learned_threshold}")
        
        # Create final merge mapping
        merge_mapping = self._create_merge_mapping(merge_groups, initial_labels)
        
        return merge_mapping
    
    async def _calculate_all_similarities(self, cluster_data: Dict[int, ClusterData]) -> Dict[Tuple[int, int], float]:
        """Calculate all pairwise similarities between cluster centroids and store in a dictionary"""
        # Get all cluster IDs
        cluster_ids = list(cluster_data.keys())
        if len(cluster_ids) <= 1:
            return {}
        
        # Calculate pairwise similarities between cluster centroids
        centroids = np.array([cluster_data[cid].centroid for cid in cluster_ids])
        similarities = cosine_similarity(centroids)
        
        # Create dictionary of similarities
        similarity_dict = {}
        
        # Print similarity distribution statistics
        all_sim_values = []
        for i in range(len(cluster_ids)):
            for j in range(i+1, len(cluster_ids)):
                sim_score = similarities[i, j]
                all_sim_values.append(sim_score)
                # Store in dictionary with tuple of IDs as key
                similarity_dict[(cluster_ids[i], cluster_ids[j])] = sim_score
        
        # Log similarity distribution
        if all_sim_values:
            all_sim_values = np.array(all_sim_values)
            logger.info(f"Similarity statistics:")
            logger.info(f"  Min: {np.min(all_sim_values):.4f}")
            logger.info(f"  Max: {np.max(all_sim_values):.4f}")
            logger.info(f"  Mean: {np.mean(all_sim_values):.4f}")
            logger.info(f"  Median: {np.median(all_sim_values):.4f}")
            logger.info(f"  Percentiles:")
            for p in [50, 75, 90, 95, 99]:
                logger.info(f"    {p}%: {np.percentile(all_sim_values, p):.4f}")
        
        return similarity_dict
    
    def _get_similarity(self, cluster1: int, cluster2: int, similarities: Dict[Tuple[int, int], float]) -> float:
        """Get similarity between two clusters from the similarities dictionary"""
        # Check if pair exists in dictionary (in either order)
        if (cluster1, cluster2) in similarities:
            return similarities[(cluster1, cluster2)]
        elif (cluster2, cluster1) in similarities:
            return similarities[(cluster2, cluster1)]
        else:
            # Should not happen, but return 0 as fallback
            logger.error(f"No similarity found for clusters {cluster1} and {cluster2}")
            return 0.0
    
    async def _similarity_guided_merging(self,
                                     cluster_ids: List[int],
                                     similarities: Dict[Tuple[int, int], float],
                                     cluster_data: Dict[int, ClusterData],
                                     initial_labels: Dict[int, InitialLabel],
                                     var_lab: str) -> List[List[int]]:
        """Merge clusters using similarity-guided LLM decisions with dynamic threshold learning"""
        logger.info(f"Starting similarity-guided merging for {len(cluster_ids)} clusters...")
        
        # Set of clusters we've already processed
        processed_clusters = set()
        
        # Final merge groups - we'll build this up as we go
        final_groups = []
        
        # Process each cluster sequentially
        for current_cluster in cluster_ids:
            # Skip if already processed
            if current_cluster in processed_clusters:
                continue
            
            logger.info(f"Processing cluster {current_cluster}...")
            
            # Create a new group with just this cluster
            new_group = [current_cluster]
            processed_clusters.add(current_cluster)
            
            # Find candidate clusters for comparison
            candidates = []
            for compare_cluster in cluster_ids:
                # Skip if already processed or same as current cluster
                if compare_cluster in processed_clusters:
                    continue
                
                # Get similarity
                sim = self._get_similarity(current_cluster, compare_cluster, similarities)
                
                # Only consider candidates above learned threshold
                if sim >= self.learned_threshold:
                    candidates.append((compare_cluster, sim))
            
            # Sort candidates by similarity (highest first)
            candidates.sort(key=lambda x: x[1], reverse=True)
            logger.info(f"Found {len(candidates)} candidates with similarity >= {self.learned_threshold:.4f}")
            
            # Process candidates in order of similarity
            for compare_cluster, sim in candidates:
                # Skip if would exceed max size
                if len(new_group) >= self.max_merge_group_size:
                    logger.info(f"Group would exceed max size, stopping comparisons for cluster {current_cluster}")
                    break
                
                # Skip if already processed (could have been added in a previous iteration)
                if compare_cluster in processed_clusters:
                    continue
                
                # Check if these should be merged
                logger.info(f"Checking clusters {current_cluster} and {compare_cluster} (similarity: {sim:.4f})...")
                should_merge = await self._check_should_merge(
                    current_cluster, compare_cluster, cluster_data, initial_labels, var_lab)
                
                if should_merge:
                    # Add to the current group and mark as processed
                    logger.info(f"Merging cluster {compare_cluster} into group with {current_cluster}")
                    new_group.append(compare_cluster)
                    processed_clusters.add(compare_cluster)
                else:
                    # Update learned threshold if this pair was more similar than our current threshold
                    if sim > self.learned_threshold:
                        logger.info(f"Updating learned threshold from {self.learned_threshold:.4f} to {sim:.4f}")
                        self.learned_threshold = sim
            
            # Add the new group to final groups
            final_groups.append(new_group)
            logger.info(f"Created merge group with {len(new_group)} clusters")
        
        return final_groups
    
# Removed: _hierarchical_merging function is no longer needed - replaced by _similarity_guided_merging
    
    async def _check_should_merge(self, 
                               cluster_id_1: int, 
                               cluster_id_2: int, 
                               cluster_data: Dict[int, ClusterData],
                               initial_labels: Dict[int, InitialLabel],
                               var_lab: str) -> bool:
        """Check if two clusters should be merged based on LLM decision"""
        # Create the prompt
        prompt = self._create_merge_decision_prompt(
            [(cluster_id_1, cluster_id_2)], cluster_data, initial_labels, var_lab)
        
        # Get decision from LLM
        async with self.semaphore:
            try:
                decisions = await self._get_merge_decisions(prompt)
                
                # Should only have one decision
                if decisions and decisions.decisions and len(decisions.decisions) > 0:
                    decision = decisions.decisions[0]
                    logger.info(f"Merge decision for clusters {cluster_id_1} & {cluster_id_2}: "
                              f"{decision.should_merge} (Reason: {decision.reason})")
                    return decision.should_merge
                
                # Default to not merging if no clear decision
                return False
            except Exception as e:
                logger.error(f"Error getting merge decision: {e}")
                # Be conservative in case of errors
                return False
    
    def _create_merge_decision_prompt(self,
                                 pairs: List[Tuple[int, int]],
                                 cluster_data: Dict[int, ClusterData],
                                 initial_labels: Dict[int, InitialLabel],
                                 var_lab: str) -> str:
        """Create prompt for binary merge decisions"""
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
    
    async def _get_merge_decisions(self, prompt: str) -> BatchMergeDecisionResponse:
        """Get binary merge decisions from LLM"""
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
                    response_model=BatchMergeDecisionResponse,
                    temperature=0.3,
                    max_tokens=4000
                )
                return response
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    logger.error(f"Failed to get merge decisions after {self.config.max_retries} attempts: {e}")
                    raise e
    
    # Removed _find_connected_components method as it's no longer needed
    
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
    import json
    import pickle
    import instructor
    
    # Try different import patterns for maximum compatibility
    try:
        # Direct imports (from src folder)
        from config import OPENAI_API_KEY, DEFAULT_MODEL
        from cache_manager import CacheManager
        from cache_config import CacheConfig
        from modules.utils import data_io
        import models
    except ImportError:
        try:
            # Absolute imports (with src prefix)
            from src.config import OPENAI_API_KEY, DEFAULT_MODEL
            from src.cache_manager import CacheManager
            from src.cache_config import CacheConfig
            from src.modules.utils import data_io
            import src.models as models
        except ImportError:
            # In Spyder or interactive environments
            print("Using sys.path manipulation for imports")
            import sys
            import os
            from pathlib import Path
            current_dir = os.getcwd()
            if current_dir.endswith('utils'):
                # If in utils folder, go up to src
                src_dir = str(Path(current_dir).parents[1])
            else:
                # If elsewhere, try to find src
                src_dir = os.path.abspath('src')
            if src_dir not in sys.path:
                sys.path.insert(0, src_dir)
            
            # Try imports again
            from config import OPENAI_API_KEY, DEFAULT_MODEL
            from cache_manager import CacheManager
            from cache_config import CacheConfig
            from modules.utils import data_io
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
        
        # Extract cluster data from the results
        try:
            from labeller import Labeller
        except ImportError:
            from src.modules.utils.labeller import Labeller
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
            similarity_threshold=0.98  # For auto-merge by embedding similarity (will be overridden in init)
        )
        
        # Initialize phase 2 merger with quiet HTTP logging
        openai_client = AsyncOpenAI(api_key=config.api_key)
        client = instructor.from_openai(openai_client)
        
        # Ensure logging is disabled for HTTP requests
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        
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
                # Create distribution histogram of similarity scores function
                async def analyze_similarity_distribution(cluster_data):
                    """Analyze the distribution of cluster similarities"""
                    similarities_dict = await phase2._calculate_all_similarities(cluster_data)
                    
                    # Extract all similarity values
                    all_similarities = list(similarities_dict.values())
                    
                    if not all_similarities:
                        print("No similarities to analyze")
                        return
                    
                    # Create histogram bins
                    bins = np.arange(0, 1.01, 0.05)
                    hist, bin_edges = np.histogram(all_similarities, bins=bins)
                    
                    # Print histogram
                    print("\nSimilarity Distribution Histogram:")
                    max_count = max(hist)
                    for i, count in enumerate(hist):
                        start = bin_edges[i]
                        end = bin_edges[i+1]
                        bar_length = int(50 * count / max_count) if max_count > 0 else 0
                        bar = '█' * bar_length
                        print(f"  {start:.2f}-{end:.2f}: {bar} ({count} pairs)")
                    
                    # Print statistics
                    print(f"\nTotal pairs: {len(all_similarities)}")
                    print(f"Min: {min(all_similarities):.4f}")
                    print(f"Max: {max(all_similarities):.4f}")
                    print(f"Mean: {np.mean(all_similarities):.4f}")
                    print(f"Median: {np.median(all_similarities):.4f}")
                    print(f"Std dev: {np.std(all_similarities):.4f}")
                    
                    # Print percentiles
                    print("\nPercentiles:")
                    for p in [25, 50, 75, 90, 95, 98, 99]:
                        print(f"  {p}%: {np.percentile(all_similarities, p):.4f}")
                    
                    return similarities_dict
                
                # Analyze similarity distribution
                print("\n=== Analyzing similarity distribution ===")
                await analyze_similarity_distribution(cluster_data)
                
                # Run merge process with similarity-guided approach
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
        
        # Handle asyncio in Spyder/IPython which already has a running event loop
        try:
            # Get the current event loop
            loop = asyncio.get_event_loop()
            
            # If running in IPython/Spyder, use nest_asyncio to avoid event loop issues
            try:
                import nest_asyncio
                nest_asyncio.apply()
                print("Applied nest_asyncio patch for running in Spyder/IPython")
            except ImportError:
                print("Consider installing nest_asyncio for better compatibility in Spyder/IPython")
            
            # Run the test
            if loop.is_running():
                # Already in an event loop (e.g., IPython/Spyder)
                print("Running in existing event loop")
                future = asyncio.ensure_future(run_test())
                # For IPython, we need to return the result
                # This is equivalent to 'await future' in async code
                import IPython
                IPython.display.display(IPython.display.HTML("Running async test... check output below"))
                loop.run_until_complete(future)
            else:
                # Normal case outside of Jupyter/IPython
                asyncio.run(run_test())
        except RuntimeError as e:
            # Last resort for running in Spyder/IPython
            if "asyncio.run() cannot be called from a running event loop" in str(e):
                print("Working around asyncio.run() in running event loop...")
                import nest_asyncio
                nest_asyncio.apply()
                asyncio.run(run_test())
            else:
                raise
    else:
        print("Missing required cached data.")
        print("Please ensure you have run:")
        print("  1. python clusterer.py")
        print("  2. python phase1_labeller.py")