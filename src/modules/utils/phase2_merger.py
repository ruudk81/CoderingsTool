import asyncio
from typing import List, Dict, Set, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.asyncio import tqdm
import logging

try:
    # When running as a script
    from labeller import (
        LabellerConfig, ClusterData, InitialLabel, MergeMapping,
        SimilarityScore, BatchSimilarityResponse
    )
except ImportError:
    # When imported as a module
    from .labeller import (
        LabellerConfig, ClusterData, InitialLabel, MergeMapping,
        SimilarityScore, BatchSimilarityResponse
    )
from prompts import SIMILARITY_SCORING_PROMPT

logger = logging.getLogger(__name__)


class Phase2Merger:
    """Phase 2: Merge clusters that are not meaningfully differentiated"""
    
    def __init__(self, config: LabellerConfig, client):
        self.config = config
        self.client = client
        self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)
    
    async def merge_similar_clusters(self,
                                   cluster_data: Dict[int, ClusterData],
                                   initial_labels: Dict[int, InitialLabel],
                                   var_lab: str) -> MergeMapping:
        """Main method to identify and merge similar clusters"""
        logger.info("Phase 2: Analyzing cluster similarity and merging...")
        
        # First, show similarity distribution analysis
        self._show_similarity_distribution(cluster_data)
        
        # Step 1: Auto-merge highly similar clusters based on embedding similarity
        auto_merge_groups = self._auto_merge_by_similarity(cluster_data)
        logger.info(f"Auto-merged {len(auto_merge_groups)} groups based on embedding similarity")
        
        # Step 2: Get remaining clusters that need LLM analysis
        already_merged = set()
        for group in auto_merge_groups:
            already_merged.update(group)
        
        remaining_clusters = [
            cid for cid in cluster_data.keys()
            if cid not in already_merged
        ]
        
        # Add singleton auto-merge groups to remaining
        for group in auto_merge_groups:
            if len(group) == 1:
                remaining_clusters.extend(group)
                already_merged.difference_update(group)
        
        logger.info(f"{len(remaining_clusters)} clusters need LLM similarity analysis")
        
        # Step 3: Use LLM to score remaining clusters
        llm_merge_groups = []
        if remaining_clusters:
            similarity_scores = await self._llm_score_clusters(
                remaining_clusters, cluster_data, initial_labels, var_lab
            )
            llm_merge_groups = self._create_merge_groups_from_scores(
                similarity_scores, self.config.merge_score_threshold
            )
        
        # Step 4: Combine auto-merge and LLM merge groups
        all_merge_groups = self._combine_merge_groups(auto_merge_groups, llm_merge_groups)
        
        # Step 5: Create final merge mapping
        merge_mapping = self._create_merge_mapping(all_merge_groups, initial_labels)
        
        logger.info(f"Final merge: {len(cluster_data)} → {len(merge_mapping.merge_groups)} clusters")
        
        return merge_mapping
    
    def _show_similarity_distribution(self, cluster_data: Dict[int, ClusterData]):
        """Show distribution analysis of pairwise similarities"""
        cluster_ids = list(cluster_data.keys())
        if len(cluster_ids) <= 1:
            return
        
        # Create centroid matrix
        centroids = np.array([cluster_data[cid].centroid for cid in cluster_ids])
        
        # Calculate all pairwise similarities
        similarities = cosine_similarity(centroids)
        
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
        
        # Create histogram bins
        bins = np.arange(0, 1.05, 0.05)
        hist, bin_edges = np.histogram(all_similarities, bins=bins)
        
        logger.info("\n=== Cosine Similarity Distribution Analysis ===")
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
            logger.info(f"  {start:.2f}-{end:.2f}: {bar} ({count} pairs)")
        
        # Count pairs above certain thresholds
        thresholds = [0.9, 0.95, 0.98, 0.99]
        logger.info("\nPairs above thresholds:")
        for threshold in thresholds:
            count = np.sum(all_similarities > threshold)
            percentage = 100 * count / len(all_similarities)
            logger.info(f"  > {threshold}: {count} pairs ({percentage:.1f}%)")
        
        logger.info("=" * 50)
    
    def _auto_merge_by_similarity(self, cluster_data: Dict[int, ClusterData]) -> List[List[int]]:
        """Auto-merge clusters with very high embedding similarity"""
        cluster_ids = list(cluster_data.keys())
        if len(cluster_ids) <= 1:
            return [[cid] for cid in cluster_ids]
        
        # Create centroid matrix
        centroids = np.array([cluster_data[cid].centroid for cid in cluster_ids])
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(centroids)
        
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
        logger.info("High cosine similarity pairs (>0.9):")
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
        logger.info(f"\nAuto-merge decisions (threshold={self.config.similarity_threshold}):")
        for cid1, cid2, score in merge_decisions[:10]:
            logger.info(f"  Merging {cid1} & {cid2} (similarity: {score:.4f})")
        if len(merge_decisions) > 10:
            logger.info(f"  ... and {len(merge_decisions) - 10} more merges")
        
        # Add any remaining clusters as singleton groups
        for cid in cluster_ids:
            if cid not in merged:
                merge_groups.append([cid])
        
        return merge_groups
    
    async def _llm_score_clusters(self,
                                cluster_ids: List[int],
                                cluster_data: Dict[int, ClusterData],
                                initial_labels: Dict[int, InitialLabel],
                                var_lab: str) -> List[SimilarityScore]:
        """Use LLM to score cluster similarities"""
        # Create batches for scoring
        batches = self._create_scoring_batches(cluster_ids, self.config.batch_size)
        
        # Process batches concurrently
        tasks = []
        for batch_ids in batches:
            task = self._score_batch(batch_ids, cluster_data, initial_labels, var_lab)
            tasks.append(task)
        
        # Execute all tasks with progress bar
        all_scores = []
        with tqdm(total=len(batches), desc="Scoring cluster similarities") as pbar:
            for coro in asyncio.as_completed(tasks):
                batch_scores = await coro
                all_scores.extend(batch_scores)
                pbar.update(1)
        
        return all_scores
    
    async def _score_batch(self,
                         batch_ids: List[int],
                         cluster_data: Dict[int, ClusterData],
                         initial_labels: Dict[int, InitialLabel],
                         var_lab: str) -> List[SimilarityScore]:
        """Score similarities for a batch of clusters"""
        async with self.semaphore:
            try:
                # Create all pairs within the batch
                pairs = []
                for i, cid1 in enumerate(batch_ids):
                    for cid2 in batch_ids[i+1:]:
                        pairs.append((cid1, cid2))
                
                if not pairs:
                    return []
                
                # Create prompt
                prompt = self._create_scoring_prompt(pairs, cluster_data, initial_labels, var_lab)
                
                # Get scores from LLM
                response = await self._get_llm_response(prompt)
                
                # Validate and return scores
                valid_scores = []
                for score in response.scores:
                    # Ensure the cluster IDs match our pairs
                    if (score.cluster_id_1, score.cluster_id_2) in pairs or \
                       (score.cluster_id_2, score.cluster_id_1) in pairs:
                        valid_scores.append(score)
                        # Log LLM scores
                        if score.score > 0.5:  # Log scores above 0.5
                            logger.info(f"LLM similarity: Clusters {score.cluster_id_1} & {score.cluster_id_2}: {score.score:.3f} - {score.reason[:50]}...")
                
                return valid_scores
                
            except Exception as e:
                logger.error(f"Error scoring batch: {e}")
                return []
    
    def _create_scoring_prompt(self,
                             pairs: List[Tuple[int, int]],
                             cluster_data: Dict[int, ClusterData],
                             initial_labels: Dict[int, InitialLabel],
                             var_lab: str) -> str:
        """Create prompt for similarity scoring"""
        # Start with the template
        prompt = SIMILARITY_SCORING_PROMPT.replace("{var_lab}", var_lab)
        prompt = prompt.replace("{language}", self.config.language)
        
        # Add cluster pairs information
        pairs_info = []
        for cid1, cid2 in pairs:
            cluster1 = cluster_data[cid1]
            cluster2 = cluster_data[cid2]
            label1 = initial_labels[cid1]
            label2 = initial_labels[cid2]
            
            # Get representative items
            reps1 = self._get_representative_items(cluster1, n=3)
            reps2 = self._get_representative_items(cluster2, n=3)
            
            pair_text = f"\nCompare Cluster {cid1} vs Cluster {cid2}:"
            
            # Cluster 1 info
            pair_text += f"\n\nCluster {cid1}:"
            pair_text += f"\n- Label: {label1.label}"
            pair_text += f"\n- Keywords: {', '.join(label1.keywords[:5])}"
            pair_text += f"\n- Size: {cluster1.size} items"
            pair_text += "\n- Representative codes:"
            for i, rep in enumerate(reps1, 1):
                pair_text += f"\n  {i}. {rep['code']}: {rep['description']}"
            
            # Cluster 2 info
            pair_text += f"\n\nCluster {cid2}:"
            pair_text += f"\n- Label: {label2.label}"
            pair_text += f"\n- Keywords: {', '.join(label2.keywords[:5])}"
            pair_text += f"\n- Size: {cluster2.size} items"
            pair_text += "\n- Representative codes:"
            for i, rep in enumerate(reps2, 1):
                pair_text += f"\n  {i}. {rep['code']}: {rep['description']}"
            
            pairs_info.append(pair_text)
        
        prompt = prompt.replace("{cluster_pairs}", "\n".join(pairs_info))
        
        return prompt
    
    def _get_representative_items(self, cluster: ClusterData, n: int = 3) -> List[Dict[str, str]]:
        """Get most representative items (same as Phase1)"""
        similarities = cosine_similarity([cluster.centroid], cluster.embeddings)[0]
        top_indices = np.argsort(similarities)[-n:][::-1]
        
        representatives = []
        for idx in top_indices:
            if idx < len(cluster.descriptive_codes):
                representatives.append({
                    'code': cluster.descriptive_codes[idx],
                    'description': cluster.code_descriptions[idx]
                })
        
        return representatives
    
    async def _get_llm_response(self, prompt: str) -> BatchSimilarityResponse:
        """Get response from LLM with retry logic"""
        messages = [
            {"role": "system", "content": "You are an expert in semantic analysis and clustering."},
            {"role": "user", "content": prompt}
        ]
        
        for attempt in range(self.config.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    response_model=BatchSimilarityResponse,
                    temperature=0.1,  # Lower temperature for consistency
                    max_tokens=4000
                )
                return response
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    raise e
    
    def _create_merge_groups_from_scores(self,
                                       scores: List[SimilarityScore],
                                       threshold: float) -> List[List[int]]:
        """Create merge groups from similarity scores"""
        # First, show LLM score distribution
        if scores:
            all_llm_scores = [score.score for score in scores]
            logger.info(f"\nLLM Similarity Score Summary (threshold={threshold}):")
            logger.info(f"  Total scored pairs: {len(all_llm_scores)}")
            logger.info(f"  Min score: {min(all_llm_scores):.3f}")
            logger.info(f"  Max score: {max(all_llm_scores):.3f}")
            logger.info(f"  Mean score: {np.mean(all_llm_scores):.3f}")
            
            # Count scores above threshold
            above_threshold = sum(1 for s in all_llm_scores if s >= threshold)
            logger.info(f"  Scores >= {threshold}: {above_threshold} ({100*above_threshold/len(all_llm_scores):.1f}%)")
        
        # Build adjacency list
        adjacency = {}
        merge_candidates = []
        for score in scores:
            if score.score >= threshold:
                cid1, cid2 = score.cluster_id_1, score.cluster_id_2
                merge_candidates.append((cid1, cid2, score.score))
                
                if cid1 not in adjacency:
                    adjacency[cid1] = set()
                if cid2 not in adjacency:
                    adjacency[cid2] = set()
                
                adjacency[cid1].add(cid2)
                adjacency[cid2].add(cid1)
        
        # Log LLM merge decisions
        if merge_candidates:
            logger.info(f"\nLLM merge decisions (showing first 10):")
            for cid1, cid2, score in merge_candidates[:10]:
                logger.info(f"  Merging {cid1} & {cid2} (LLM score: {score:.3f})")
            if len(merge_candidates) > 10:
                logger.info(f"  ... and {len(merge_candidates) - 10} more merges")
        
        # Find connected components
        visited = set()
        merge_groups = []
        
        for cid in adjacency:
            if cid not in visited:
                # DFS to find all connected clusters
                group = []
                stack = [cid]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        group.append(current)
                        stack.extend(adjacency[current] - visited)
                
                merge_groups.append(sorted(group))
        
        return merge_groups
    
    def _combine_merge_groups(self,
                            auto_groups: List[List[int]],
                            llm_groups: List[List[int]]) -> List[List[int]]:
        """Combine auto-merge and LLM merge groups"""
        # Create mapping of cluster to group
        cluster_to_group = {}
        group_id = 0
        
        # Add auto-merge groups
        for group in auto_groups:
            if len(group) > 1:  # Only multi-cluster groups
                for cid in group:
                    cluster_to_group[cid] = group_id
                group_id += 1
        
        # Add LLM merge groups
        for group in llm_groups:
            # Check if any cluster is already in a group
            existing_groups = set()
            for cid in group:
                if cid in cluster_to_group:
                    existing_groups.add(cluster_to_group[cid])
            
            if existing_groups:
                # Merge with existing group(s)
                main_group = min(existing_groups)
                for cid in group:
                    cluster_to_group[cid] = main_group
                # Update other groups to point to main group
                for gid in existing_groups:
                    if gid != main_group:
                        for cid, group_id in cluster_to_group.items():
                            if group_id == gid:
                                cluster_to_group[cid] = main_group
            else:
                # Create new group
                for cid in group:
                    cluster_to_group[cid] = group_id
                group_id += 1
        
        # Convert back to list of groups
        groups_dict = {}
        for cid, gid in cluster_to_group.items():
            if gid not in groups_dict:
                groups_dict[gid] = []
            groups_dict[gid].append(cid)
        
        # Add singleton clusters
        all_clusters = set()
        for group in auto_groups:
            all_clusters.update(group)
        for group in llm_groups:
            all_clusters.update(group)
        
        final_groups = list(groups_dict.values())
        
        # Add any missing clusters as singletons
        grouped_clusters = set()
        for group in final_groups:
            grouped_clusters.update(group)
        
        for cid in all_clusters - grouped_clusters:
            final_groups.append([cid])
        
        return [sorted(group) for group in final_groups]
    
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
    
    def _create_scoring_batches(self, items: List[int], batch_size: int) -> List[List[int]]:
        """Create batches that respect pair generation within batches"""
        # Smaller batches since we need to compare all pairs within batch
        # For n items, we get n(n-1)/2 pairs
        # So for batch_size=10, we get 45 pairs
        actual_batch_size = min(batch_size, 10)  # Limit to avoid too many pairs
        return [items[i:i + actual_batch_size] for i in range(0, len(items), actual_batch_size)]


if __name__ == "__main__":
    """Test Phase 2 with actual cached data from Phase 1"""
    import sys
    from pathlib import Path
    import json
    import pickle
    
    # Add project paths
    sys.path.append(str(Path(__file__).parents[2]))  # Add src directory
    
    from openai import AsyncOpenAI
    import instructor
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
            merge_score_threshold=0.7   # For LLM merge
        )
    
        # Initialize phase 2 merger
        client = instructor.from_openai(AsyncOpenAI(api_key=config.api_key))
        phase2 = Phase2Merger(config, client)
        
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
                
                # First run auto-merge analysis separately to capture scores
                print("=== AUTO-MERGE ANALYSIS (Cosine Similarity) ===")
                
                # Create centroid matrix
                cluster_ids = list(cluster_data.keys())
                centroids = np.array([cluster_data[cid].centroid for cid in cluster_ids])
                
                # Calculate pairwise similarities
                similarities = cosine_similarity(centroids)
                
                # Capture all similarity scores
                auto_merge_scores = []
                for i in range(len(cluster_ids)):
                    for j in range(i+1, len(cluster_ids)):
                        sim_score = similarities[i, j]
                        auto_merge_scores.append({
                            'cluster1': cluster_ids[i],
                            'cluster2': cluster_ids[j],
                            'score': sim_score,
                            'merged': sim_score > config.similarity_threshold
                        })
                
                # Sort by score descending
                auto_merge_scores.sort(key=lambda x: x['score'], reverse=True)
                
                print(f"\nTop 20 Auto-merge similarity scores (threshold={config.similarity_threshold}):")
                for i, score_info in enumerate(auto_merge_scores[:20]):
                    status = "MERGED" if score_info['merged'] else "not merged"
                    print(f"  {i+1}. Clusters {score_info['cluster1']} & {score_info['cluster2']}: {score_info['score']:.4f} ({status})")
                
                # Create histogram of auto-merge scores
                auto_scores_only = [s['score'] for s in auto_merge_scores]
                bins = np.arange(0, 1.05, 0.05)
                hist, bin_edges = np.histogram(auto_scores_only, bins=bins)
                
                print("\nAuto-merge score distribution:")
                max_count = max(hist)
                for i, count in enumerate(hist):
                    start = bin_edges[i]
                    end = bin_edges[i+1]
                    bar_length = int(40 * count / max_count) if max_count > 0 else 0
                    bar = '█' * bar_length
                    print(f"  {start:.2f}-{end:.2f}: {bar} ({count} pairs)")
                
                # Now run full merge process to get LLM scores
                print("\n=== FULL MERGE PROCESS (Including LLM Analysis) ===")
                
                # Temporarily store LLM scores for analysis
                llm_scores = []
                
                # Monkey-patch the phase2 instance to capture LLM scores
                original_score_batch = phase2._score_similarity_batch
                
                async def capture_scores(*args, **kwargs):
                    result = await original_score_batch(*args, **kwargs)
                    # Capture the scores for our analysis
                    if hasattr(result, 'scores'):
                        for score in result.scores:
                            llm_scores.append({
                                'cluster1': score.cluster_id_1,
                                'cluster2': score.cluster_id_2,
                                'score': score.score,
                                'merged': score.score >= config.merge_score_threshold,
                                'reason': score.reason
                            })
                    return result
                
                phase2._score_similarity_batch = capture_scores
                
                # Run full merge process
                merge_mapping = await phase2.merge_similar_clusters(
                    cluster_data, initial_labels, var_lab
                )
                
                # Restore original method
                phase2._score_similarity_batch = original_score_batch
                
                # Analyze LLM scores
                if llm_scores:
                    llm_scores.sort(key=lambda x: x['score'], reverse=True)
                    
                    print(f"\nTop 20 LLM similarity scores (threshold={config.merge_score_threshold}):")
                    for i, score_info in enumerate(llm_scores[:20]):
                        status = "MERGED" if score_info['merged'] else "not merged"
                        reason = score_info['reason'][:50] + "..." if len(score_info['reason']) > 50 else score_info['reason']
                        print(f"  {i+1}. Clusters {score_info['cluster1']} & {score_info['cluster2']}: {score_info['score']:.3f} ({status}) - {reason}")
                    
                    # Create histogram of LLM scores
                    llm_scores_only = [s['score'] for s in llm_scores]
                    bins = np.arange(0, 1.05, 0.05)
                    hist, bin_edges = np.histogram(llm_scores_only, bins=bins)
                    
                    print("\nLLM score distribution:")
                    max_count = max(hist)
                    for i, count in enumerate(hist):
                        start = bin_edges[i]
                        end = bin_edges[i+1]
                        bar_length = int(40 * count / max_count) if max_count > 0 else 0
                        bar = '█' * bar_length
                        print(f"  {start:.2f}-{end:.2f}: {bar} ({count} pairs)")
                    
                    # Compare thresholds
                    print(f"\nThreshold Analysis:")
                    print(f"Auto-merge threshold: {config.similarity_threshold}")
                    print(f"LLM merge threshold: {config.merge_score_threshold}")
                    
                    auto_merged_count = sum(1 for s in auto_merge_scores if s['merged'])
                    llm_merged_count = sum(1 for s in llm_scores if s['merged'])
                    
                    print(f"Auto-merge would merge: {auto_merged_count} pairs")
                    print(f"LLM merge would merge: {llm_merged_count} pairs")
                
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
                
                # Analyze auto-merge scores to suggest threshold
                auto_scores_sorted = sorted([s['score'] for s in auto_merge_scores], reverse=True)
                percentiles = np.percentile(auto_scores_sorted, [99, 98, 97, 96, 95])
                
                print("\nAuto-merge score percentiles:")
                for i, (p, val) in enumerate(zip([99, 98, 97, 96, 95], percentiles)):
                    print(f"  {p}th percentile: {val:.4f}")
                
                # Analyze LLM scores to suggest threshold
                if llm_scores:
                    llm_scores_sorted = sorted([s['score'] for s in llm_scores], reverse=True)
                    llm_percentiles = np.percentile(llm_scores_sorted, [90, 85, 80, 75, 70])
                    
                    print("\nLLM score percentiles:")
                    for i, (p, val) in enumerate(zip([90, 85, 80, 75, 70], llm_percentiles)):
                        print(f"  {p}th percentile: {val:.3f}")
                
                print("\nSuggested adjustments to reduce merging:")
                print(f"  Current auto-merge threshold: {config.similarity_threshold}")
                print(f"  Consider raising to: {percentiles[1]:.4f} (98th percentile) or {percentiles[0]:.4f} (99th percentile)")
                print(f"  Current LLM threshold: {config.merge_score_threshold}")
                if llm_scores:
                    print(f"  Consider raising to: {llm_percentiles[1]:.3f} (85th percentile) or {llm_percentiles[0]:.3f} (90th percentile)")
                
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