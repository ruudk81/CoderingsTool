import asyncio
from typing import List, Dict, Set, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.asyncio import tqdm
import logging

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
    
    def _auto_merge_by_similarity(self, cluster_data: Dict[int, ClusterData]) -> List[List[int]]:
        """Auto-merge clusters with very high embedding similarity"""
        cluster_ids = list(cluster_data.keys())
        if len(cluster_ids) <= 1:
            return [[cid] for cid in cluster_ids]
        
        # Create centroid matrix
        centroids = np.array([cluster_data[cid].centroid for cid in cluster_ids])
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(centroids)
        
        # Find groups to merge
        merged = set()
        merge_groups = []
        
        for i, cid1 in enumerate(cluster_ids):
            if cid1 in merged:
                continue
            
            group = [cid1]
            merged.add(cid1)
            
            for j, cid2 in enumerate(cluster_ids[i+1:], i+1):
                if cid2 in merged:
                    continue
                
                if similarities[i, j] > self.config.similarity_threshold:
                    group.append(cid2)
                    merged.add(cid2)
            
            merge_groups.append(group)
        
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
        # Build adjacency list
        adjacency = {}
        for score in scores:
            if score.score >= threshold:
                cid1, cid2 = score.cluster_id_1, score.cluster_id_2
                
                if cid1 not in adjacency:
                    adjacency[cid1] = set()
                if cid2 not in adjacency:
                    adjacency[cid2] = set()
                
                adjacency[cid1].add(cid2)
                adjacency[cid2].add(cid1)
        
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