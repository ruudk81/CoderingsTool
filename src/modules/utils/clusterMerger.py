import asyncio
from typing import List, Dict, Set, Tuple, Any, Optional
from collections import defaultdict
import numpy as np
import numpy.typing as npt
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI, AsyncOpenAI
import instructor
import logging
from tqdm.asyncio import tqdm
import models
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

# Internal data model for merging
class MergeResultMapper(BaseModel):
    # Original cluster data
    respondent_id: Any
    segment_id: str
    descriptive_code: str
    code_description: str
    code_embedding: npt.NDArray[np.float32]
    description_embedding: npt.NDArray[np.float32]
    
    # Cluster data
    original_cluster_id: Optional[int] = None
    merged_cluster_id: Optional[int] = None
    
    # Config
    model_config = ConfigDict(arbitrary_types_allowed=True)  # for arrays with embeddings

# Import prompts with fallbacks
try:
    from prompts import SIMILARITY_SCORING_PROMPT
except ImportError:
    try:
        from ..prompts import SIMILARITY_SCORING_PROMPT
    except ImportError:
        from src.prompts import SIMILARITY_SCORING_PROMPT

# Configuration for the merger
from config import OPENAI_API_KEY, DEFAULT_MODEL, DEFAULT_LANGUAGE

# Define configuration and data models directly to avoid circular imports
class MergerConfig(BaseModel):
    """Configuration for the ClusterMerger"""
    model: str = DEFAULT_MODEL
    api_key: str = OPENAI_API_KEY
    max_concurrent_requests: int = 10
    batch_size: int = 20
    similarity_threshold: float = 0.95  # Auto-merge threshold
    merge_score_threshold: float = 0.7  # LLM merge threshold
    max_retries: int = 3
    retry_delay: int = 2
    language: str = DEFAULT_LANGUAGE

# Define models for cluster data
class ClusterData(BaseModel):
    """Internal representation of cluster with extracted data"""
    cluster_id: int
    descriptive_codes: List[str]
    code_descriptions: List[str]
    embeddings: np.ndarray
    centroid: np.ndarray
    size: int
    
    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)

class InitialLabel(BaseModel):
    """Initial label for a cluster"""
    cluster_id: int
    label: str
    keywords: List[str]
    confidence: float = Field(ge=0.0, le=1.0)
    
    model_config = ConfigDict(from_attributes=True)

class MergeMapping(BaseModel):
    """Mapping of cluster merges"""
    merge_groups: List[List[int]]  # Groups of cluster IDs to merge
    cluster_to_merged: Dict[int, int]  # Original cluster ID → Merged cluster ID
    merge_reasons: Dict[int, str]  # Merged cluster ID → Reason
    
    model_config = ConfigDict(from_attributes=True)

# Define response models for binary merging decision
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


class ClusterMerger:
    """Merges clusters that are not meaningfully differentiated from the perspective of the research question"""
    
    def __init__(self, input_list=None, var_lab=None, config=None, client=None):
        self.config = config or MergerConfig()
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        self.client = client
        self.max_merge_group_size = 5      # Maximum size for any merge group
        self.similarity_threshold = 0.95   # Fixed threshold for high similarity pairs
        
        # Validate input_list is a list of ClusterModel objects if provided
        if input_list is not None and not all(isinstance(item, models.ClusterModel) for item in input_list):
            raise ValueError("input_list must be a list of ClusterModel objects")
            
        # Store input data if provided
        self.input_clusters = input_list
        self.var_lab = var_lab
        self.cluster_data = None
        self.initial_labels = None
    
    async def merge_similar_clusters(self,
                                   cluster_data: Dict[int, ClusterData],
                                   initial_labels: Dict[int, InitialLabel],
                                   var_lab: str) -> MergeMapping:
        """Main method to identify and merge similar clusters using sorted similarity pairs"""
        logger.info(f"Analyzing cluster similarity for research question: '{var_lab}'")
        
        # Calculate all pairwise similarities between clusters
        similarities_dict = await self._calculate_all_similarities(cluster_data)
        logger.info(f"Calculated similarities between all {len(cluster_data)} clusters")
        
        # Sort all pairs by similarity (highest first) and filter by threshold
        sorted_pairs = self._get_sorted_similarity_pairs(similarities_dict)
        high_similarity_pairs = [(c1, c2, sim) for c1, c2, sim in sorted_pairs if sim >= self.similarity_threshold]
        logger.info(f"Found {len(high_similarity_pairs)} pairs with similarity ≥ {self.similarity_threshold}")
        
        # Merge clusters by processing sorted pairs
        merge_groups = await self._sorted_pairs_merging(
            high_similarity_pairs,
            cluster_data, 
            initial_labels, 
            var_lab
        )
        
        logger.info(f"Final merge: {len(cluster_data)} clusters → {len(merge_groups)} groups")
        
        # Create final merge mapping
        merge_mapping = self._create_merge_mapping(merge_groups, initial_labels)
        
        return merge_mapping
        
    def _get_sorted_similarity_pairs(self, similarities: Dict[Tuple[int, int], float]) -> List[Tuple[int, int, float]]:
        """Get all similarity pairs sorted by similarity (highest first)"""
        # Convert dictionary to list of (cluster1, cluster2, similarity) tuples
        pairs = [(c1, c2, sim) for (c1, c2), sim in similarities.items()]
        
        # Sort by similarity (highest first)
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        return pairs
    
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
    
    async def _sorted_pairs_merging(self,
                                sorted_pairs: List[Tuple[int, int, float]],
                                cluster_data: Dict[int, ClusterData],
                                initial_labels: Dict[int, InitialLabel],
                                var_lab: str) -> List[List[int]]:
        """Merge clusters by processing similarity pairs in sorted order with batched LLM requests
        
        This improved implementation:
        1. Batches LLM requests for efficiency (fewer API calls)
        2. Prioritizes the most promising pairs based on embedding similarity
        3. Updates merge groups dynamically as decisions come in
        4. Provides detailed progress logging
        """
        logger.info(f"Starting optimized pair-based merging with {len(sorted_pairs)} high-similarity pairs...")
        
        # Initialize cluster groups (initially each cluster is in its own group)
        cluster_to_group = {}  # Maps cluster ID to its group index
        merge_groups = []      # List of merge groups (each group is a list of cluster IDs)
        
        # Initialize with singleton groups
        all_clusters = set()
        for c1, c2, _ in sorted_pairs:
            all_clusters.add(c1)
            all_clusters.add(c2)
        
        # Add all clusters (including those not in any pair)
        all_clusters.update(cluster_data.keys())
        logger.info(f"Processing a total of {len(all_clusters)} clusters")
        
        # Start with individual groups
        for i, cid in enumerate(all_clusters):
            merge_groups.append([cid])
            cluster_to_group[cid] = i
        
        # Create batches of pairs to evaluate
        batch_size = min(10, self.config.batch_size)  # Limit batch size for prompt size
        
        # Filter pairs that would exceed max group size
        valid_pairs = []
        for c1, c2, similarity in sorted_pairs:
            g1 = cluster_to_group.get(c1)
            g2 = cluster_to_group.get(c2)
            
            # Skip if both clusters are in the same group
            if g1 == g2:
                continue
            
            # Skip if merging would exceed max group size
            if len(merge_groups[g1]) + len(merge_groups[g2]) > self.max_merge_group_size:
                logger.debug(f"Skipping evaluation: merging clusters {c1} & {c2} would exceed max size")
                continue
            
            valid_pairs.append((c1, c2, similarity))
        
        logger.info(f"Found {len(valid_pairs)} valid pairs to evaluate (after filtering for size constraints)")
        
        # Process pairs in batches
        llm_calls = 0
        total_batches = (len(valid_pairs) + batch_size - 1) // batch_size
        merge_decisions = {}  # Cache decisions to avoid redundant API calls
        
        # Process in batches with logging
        for batch_idx in range(0, len(valid_pairs), batch_size):
            batch_end = min(batch_idx + batch_size, len(valid_pairs))
            current_batch = valid_pairs[batch_idx:batch_end]
            
            logger.info(f"Processing batch {batch_idx // batch_size + 1}/{total_batches} "
                      f"with {len(current_batch)} pairs...")
            
            # Extract pairs for the batch
            batch_pairs = [(c1, c2) for c1, c2, _ in current_batch]
            
            # Create batch prompt
            prompt = self._create_merge_decision_prompt(batch_pairs, cluster_data, initial_labels, var_lab)
            
            # Get decisions from LLM
            try:
                batch_decisions = await self._get_merge_decisions(prompt)
                llm_calls += 1
                
                # Create a mapping from pair to decision
                if batch_decisions and batch_decisions.decisions:
                    for decision in batch_decisions.decisions:
                        pair_key = (decision.cluster_id_1, decision.cluster_id_2)
                        merge_decisions[pair_key] = decision.should_merge
                        
                        # Log the decision
                        status = "MERGED" if decision.should_merge else "KEPT SEPARATE"
                        reason = decision.reason[:50] + "..." if len(decision.reason) > 50 else decision.reason
                        logger.info(f"Decision for clusters {decision.cluster_id_1} & {decision.cluster_id_2}: "
                                  f"{status} - {reason}")
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                # Continue with next batch
                continue
            
            # Apply merge decisions for this batch
            for c1, c2, similarity in current_batch:
                # Get current groups (they might have changed from previous merges)
                g1 = cluster_to_group.get(c1)
                g2 = cluster_to_group.get(c2)
                
                # Skip if already in same group or would exceed size
                if g1 == g2:
                    continue
                
                if len(merge_groups[g1]) + len(merge_groups[g2]) > self.max_merge_group_size:
                    continue
                
                # Get decision (from either order of pairs)
                should_merge = merge_decisions.get((c1, c2)) or merge_decisions.get((c2, c1), False)
                
                if should_merge:
                    # Merge the two groups
                    logger.info(f"Applying merge for clusters {c1} & {c2} (similarity: {similarity:.4f})")
                    
                    # Always keep the smaller group index and merge the larger one into it
                    if g1 > g2:
                        g1, g2 = g2, g1  # Swap to ensure g1 < g2
                    
                    # Merge group g2 into g1
                    merge_groups[g1].extend(merge_groups[g2])
                    
                    # Update group assignments for merged clusters
                    for cid in merge_groups[g2]:
                        cluster_to_group[cid] = g1
                    
                    # Mark group g2 as empty
                    merge_groups[g2] = []
        
        # Filter out empty groups
        final_groups = [group for group in merge_groups if group]
        
        # Calculate statistics
        total_merged = sum(1 for g in final_groups if len(g) > 1)
        merge_reduction = (len(all_clusters) - len(final_groups)) / len(all_clusters) * 100 if all_clusters else 0
        
        logger.info(f"Merging completed with {llm_calls} LLM batch calls")
        logger.info(f"Created {len(final_groups)} merge groups ({total_merged} merged groups) from {len(all_clusters)} clusters")
        logger.info(f"Reduction: {merge_reduction:.1f}%")
        
        return final_groups
    
    def _create_merge_decision_prompt(self,
                                 pairs: List[Tuple[int, int]],
                                 cluster_data: Dict[int, ClusterData],
                                 initial_labels: Dict[int, InitialLabel],
                                 var_lab: str) -> str:
        """Create prompt for binary merge decisions with support for multiple pairs
        
        This updated version:
        1. Can handle batches of pairs in a single prompt for efficiency
        2. Selects the 5 most representative codes based on cosine similarity to centroid
        3. Focuses explicitly on the research question context
        4. Presents clear decision criteria based on meaningful differentiation
        """
        prompt = SIMILARITY_SCORING_PROMPT.replace("{var_lab}", var_lab)
        prompt = prompt.replace("{language}", self.config.language)
        
        # Create cluster pair descriptions
        pair_descriptions = []
        
        for idx, (cluster_id_1, cluster_id_2) in enumerate(pairs):
            # Get data for first cluster
            cluster1 = cluster_data[cluster_id_1]
            label1 = initial_labels[cluster_id_1].label
            
            # Get the 5 most representative items based on cosine similarity to centroid
            representative_items1 = self._get_representative_items(cluster1, n=5)
            
            # Get data for second cluster
            cluster2 = cluster_data[cluster_id_2]
            label2 = initial_labels[cluster_id_2].label
            
            # Get the 5 most representative items based on cosine similarity to centroid
            representative_items2 = self._get_representative_items(cluster2, n=5)
            
            # Create pair description with a number for batches
            description = f"\n\nPAIR {idx+1}: Cluster {cluster_id_1} vs Cluster {cluster_id_2}\n"
            
            # First cluster
            description += f"\nCluster {cluster_id_1}:"
            description += f"\n- Label: {label1}"
            description += f"\n- Most representative responses (by similarity to cluster centroid):"
            
            for i, item in enumerate(representative_items1):
                description += f"\n  {i+1}. {item['code']}: {item['description']}"
            
            # Second cluster
            description += f"\n\nCluster {cluster_id_2}:"
            description += f"\n- Label: {label2}"
            description += f"\n- Most representative responses (by similarity to cluster centroid):"
            
            for i, item in enumerate(representative_items2):
                description += f"\n  {i+1}. {item['code']}: {item['description']}"
            
            # Add explicit question about this pair
            description += f"\n\nQuestion for this pair: Do these clusters represent meaningfully different responses to the research question \"{var_lab}\", or are they essentially saying the same thing?"
            
            pair_descriptions.append(description)
        
        # Join all pair descriptions
        prompt = prompt.replace("{cluster_pairs}", "\n".join(pair_descriptions))
        
        # Add reminder about format for batch responses
        if len(pairs) > 1:
            prompt += "\n\nRemember: For each pair, include both cluster_id_1 and cluster_id_2 in your response " \
                      "along with your should_merge decision and reason. Ensure cluster IDs match exactly."
        
        return prompt
    
    def _get_representative_items(self, cluster: ClusterData, n: int = 5) -> List[Dict[str, str]]:
        """Get most representative items from a cluster based on cosine similarity to centroid
        
        Args:
            cluster: The cluster data containing embeddings, codes, and descriptions
            n: Number of representative items to return (default: 5)
            
        Returns:
            List of dictionaries with 'code' and 'description' keys for the most representative items
        """
        # Calculate similarity between each item's embedding and the cluster centroid
        similarities = cosine_similarity([cluster.centroid], cluster.embeddings)[0]
        
        # Get indices of the top n most similar items
        top_indices = np.argsort(similarities)[-n:][::-1]
        
        # Create list of representative items
        representatives = []
        for idx in top_indices:
            if idx < len(cluster.descriptive_codes) and idx < len(cluster.code_descriptions):
                representatives.append({
                    'code': cluster.descriptive_codes[idx],
                    'description': cluster.code_descriptions[idx],
                    'similarity': similarities[idx]  # Include similarity score for reference
                })
        
        return representatives
    
    async def _get_merge_decisions(self, prompt: str) -> BatchMergeDecisionResponse:
        """Get binary merge decisions from LLM"""
        # Create an instructor-patched client
        client = instructor.from_openai(
            AsyncOpenAI(api_key=self.config.api_key)
        )
        
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


    def merge_clusters(self, input_list=None, var_lab=None):
        """Main entry point for merging clusters
        
        Args:
            input_list: List of ClusterModel objects with initial clustering
            var_lab: Variable label/research question
            
        Returns:
            Tuple containing:
            - List of ClusterModel objects with merged clusters
            - MergeMapping object with merge information (for caching)
        """
        # Update input data if provided
        if input_list is not None:
            # Validate input_list is a list of ClusterModel objects
            if not all(isinstance(item, models.ClusterModel) for item in input_list):
                raise ValueError("input_list must be a list of ClusterModel objects")
            self.input_clusters = input_list
            
        if var_lab is not None:
            self.var_lab = var_lab
            
        # Validate input
        if not self.input_clusters or not self.var_lab:
            raise ValueError("Input clusters and var_lab must be provided")
        
        # Run the actual merging process
        merged_clusters = asyncio.run(self._merge_clusters_async())
        
        # Verify output is a list of ClusterModel objects
        if not all(isinstance(item, models.ClusterModel) for item in merged_clusters):
            raise ValueError("Output must be a list of ClusterModel objects")
            
        # Return both the merged clusters and the merge mapping for caching
        return merged_clusters, self.merge_mapping
        
    async def _merge_clusters_async(self):
        """Async implementation of cluster merging"""
        # Extract cluster data
        self.cluster_data = self._extract_cluster_data(self.input_clusters)
        logger.info(f"Extracted data for {len(self.cluster_data)} clusters")
        
        # Generate simple labels based on embeddings
        self.initial_labels = await self._generate_simple_labels()
        logger.info(f"Generated simple labels for {len(self.initial_labels)} clusters")
        
        # Merge similar clusters
        merge_mapping = await self.merge_similar_clusters(
            self.cluster_data, self.initial_labels, self.var_lab
        )
        
        # Convert to ClusterModel using to_cluster_model
        logger.info("Converting merged clusters to ClusterModel format...")
        merged_clusters = self.to_cluster_model(merge_mapping)
        
        # Calculate and log statistics
        total_initial = len(self.cluster_data)
        total_final = len(set(merge_mapping.cluster_to_merged.values()))
        merged_groups = [g for g in merge_mapping.merge_groups if len(g) > 1]
        
        logger.info(f"Cluster merging statistics:")
        logger.info(f"Initial clusters: {total_initial}")
        logger.info(f"Merge groups: {len(merged_groups)}")
        logger.info(f"Final clusters after merging: {total_final}")
        logger.info(f"Reduction: {(1 - total_final/total_initial) * 100:.1f}%")
        
        # Store merge mapping for reference
        self.merge_mapping = merge_mapping
        
        return merged_clusters
    
    def _extract_cluster_data(self, cluster_results):
        """Extract cluster data from cluster results
        Similar to Labeller.extract_cluster_data but without dependencies
        """
        cluster_data = defaultdict(lambda: {
            'descriptive_codes': [],
            'code_descriptions': [],
            'embeddings': []
        })
        
        # Collect data by cluster ID
        for result in cluster_results:
            for segment in result.response_segment:
                if segment.mirco_cluster is not None:
                    cluster_id = list(segment.mirco_cluster.keys())[0]
                    cluster_data[cluster_id]['descriptive_codes'].append(segment.descriptive_code)
                    cluster_data[cluster_id]['code_descriptions'].append(segment.code_description)
                    # Use description embeddings by default
                    cluster_data[cluster_id]['embeddings'].append(segment.description_embedding)
        
        # Convert to ClusterData objects
        clusters = {}
        for cluster_id, data in cluster_data.items():
            embeddings_array = np.array(data['embeddings'])
            centroid = np.mean(embeddings_array, axis=0)
            
            clusters[cluster_id] = ClusterData(
                cluster_id=cluster_id,
                descriptive_codes=data['descriptive_codes'],
                code_descriptions=data['code_descriptions'],
                embeddings=embeddings_array,
                centroid=centroid,
                size=len(data['descriptive_codes'])
            )
        
        return clusters
    
    async def _generate_simple_labels(self):
        """Generate simple labels for each cluster based on most frequent codes"""
        labels = {}
        
        for cluster_id, cluster in self.cluster_data.items():
            # Count codes
            code_counts = defaultdict(int)
            for code in cluster.descriptive_codes:
                code_counts[code] += 1
            
            # Get the most common code
            if code_counts:
                most_common_code, _ = max(code_counts.items(), key=lambda x: x[1])
                label = most_common_code[:50]  # Limit length
            else:
                label = f"Cluster {cluster_id}"
            
            # Create a simple initial label
            labels[cluster_id] = InitialLabel(
                cluster_id=cluster_id,
                label=label,
                keywords=[k for k, v in sorted(code_counts.items(), key=lambda x: x[1], reverse=True)[:5]],
                confidence=0.8  # Default confidence
            )
        
        return labels
    
    def _convert_to_mappers(self, merge_mapping):
        """Convert input cluster data to internal MergeResultMapper objects"""
        output_list = []
        
        # Process each cluster result
        for result in self.input_clusters:
            for segment in result.response_segment:
                if segment.mirco_cluster is not None:
                    old_cluster_id = list(segment.mirco_cluster.keys())[0]
                    
                    # Get the merged cluster ID
                    merged_cluster_id = None
                    if old_cluster_id in merge_mapping.cluster_to_merged:
                        merged_cluster_id = merge_mapping.cluster_to_merged[old_cluster_id]
                    
                    # Create mapper object
                    mapper = MergeResultMapper(
                        respondent_id=result.respondent_id,
                        segment_id=segment.segment_id,
                        descriptive_code=segment.descriptive_code,
                        code_description=segment.code_description,
                        code_embedding=segment.code_embedding,
                        description_embedding=segment.description_embedding,
                        original_cluster_id=old_cluster_id,
                        merged_cluster_id=merged_cluster_id
                    )
                    
                    output_list.append(mapper)
        
        return output_list
    
    def to_cluster_model(self, merge_mapping) -> List[models.ClusterModel]:
        """Convert internal data to ClusterModel format
        
        Similar to ClusterGenerator.to_cluster_model, but handles merged clusters
        
        Args:
            merge_mapping: MergeMapping object with merge groups and mapping information
            
        Returns:
            List of ClusterModel objects with updated cluster assignments
        """
        # Convert input data to internal format
        output_list = self._convert_to_mappers(merge_mapping)
        
        if not output_list:
            raise ValueError("Output list is empty. Nothing to convert.")
        
        # Group by respondent ID
        items_by_respondent = defaultdict(list)
        for item in output_list:
            items_by_respondent[item.respondent_id].append(item)
        
        # Create mapping of respondent_id to original response data
        response_mapping = {}
        segment_mapping = {}
        for original_item in self.input_clusters:
            response_mapping[original_item.respondent_id] = original_item.response
            if original_item.response_segment:
                for segment in original_item.response_segment:
                    segment_key = (original_item.respondent_id, segment.segment_id)
                    segment_mapping[segment_key] = segment.segment_response
        
        # Create ClusterModel instances
        result_models = []
        
        for respondent_id, items in items_by_respondent.items():
            # Get the original response from mapping
            response = response_mapping.get(respondent_id, "")
            
            # Create submodels for each segment
            submodels = []
            for item in items:
                # Get original segment response
                segment_key = (respondent_id, item.segment_id)
                segment_response = segment_mapping.get(segment_key, "")
                
                # Use merged cluster ID for micro_cluster
                micro_cluster = None
                if item.merged_cluster_id is not None:
                    # Store the cluster ID with an empty string (will be populated with labels later)
                    micro_cluster = {item.merged_cluster_id: ""}
                
                # Create submodel
                submodel = models.ClusterSubmodel(
                    segment_id=item.segment_id,
                    segment_response=segment_response,
                    descriptive_code=item.descriptive_code,
                    code_description=item.code_description,
                    code_embedding=item.code_embedding,
                    description_embedding=item.description_embedding,
                    meta_cluster=None,  # No meta clusters
                    meso_cluster=None,  # No meso clusters
                    mirco_cluster=micro_cluster  # Use merged cluster ID
                )
                
                submodels.append(submodel)
            
            # Create ClusterModel
            model = models.ClusterModel(
                respondent_id=respondent_id,
                response=response,
                response_segment=submodels
            )
            
            result_models.append(model)
            
        logger.info(f"Converted results to {len(result_models)} ClusterModel objects")
        return result_models


if __name__ == "__main__":
    """Test with actual cached cluster data"""
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
            batch_size=5  # Process a smaller batch size for merge analysis
        )
        
        # Initialize cluster merger with quiet HTTP logging
        openai_client = AsyncOpenAI(api_key=config.api_key)
        client = instructor.from_openai(openai_client)
        
        # Ensure logging is disabled for HTTP requests
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        
        cluster_merger = ClusterMerger(config, client)
        
        async def run_test():
            """Run the test"""
            print("=== Testing Cluster Merging with Real Data ===")
            print(f"Variable label: {var_lab}")
            print(f"Number of clusters: {len(cluster_data)}")
            
            # Set up logging for test
            import logging
            logging.basicConfig(level=logging.INFO, format='%(message)s')
            
            try:
                # Create distribution histogram of similarity scores function
                async def analyze_similarity_distribution(cluster_data):
                    """Analyze the distribution of cluster similarities"""
                    similarities_dict = await cluster_merger._calculate_all_similarities(cluster_data)
                    
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
                merge_mapping = await cluster_merger.merge_similar_clusters(
                    cluster_data, initial_labels, var_lab
                )
                
                # Display results
                print("\n=== Merge Results ===")
                print(f"Original clusters: {len(cluster_data)}")
                print(f"Merged clusters: {len(merge_mapping.merge_groups)}")
                
                # Calculate reduction
                reduction = (1 - len(merge_mapping.merge_groups)/len(cluster_data)) * 100
                print(f"Reduction: {reduction:.1f}%")
                
                # Show all multi-cluster groups first
                multi_cluster_groups = [group for group in merge_mapping.merge_groups if len(group) > 1]
                if multi_cluster_groups:
                    print(f"\nMerged groups ({len(multi_cluster_groups)}):")
                    for i, group in enumerate(multi_cluster_groups):
                        group_labels = [initial_labels[cid].label for cid in group]
                        print(f"\nMerged Group {i+1} ({len(group)} clusters):")
                        for j, (cid, label) in enumerate(zip(group, group_labels)):
                            print(f"  {j+1}. Cluster {cid}: {label}")
                else:
                    print("\nNo clusters were merged.")
                
                # Then show sample of singleton groups
                singleton_groups = [group for group in merge_mapping.merge_groups if len(group) == 1]
                if singleton_groups:
                    sample_size = min(5, len(singleton_groups))
                    print(f"\nSample of singleton groups ({len(singleton_groups)} total):")
                    for i, group in enumerate(singleton_groups[:sample_size]):
                        print(f"  Group {i+1}: Cluster {group[0]} - {initial_labels[group[0]].label}")
                
                # Count distribution of group sizes
                group_sizes = [len(group) for group in merge_mapping.merge_groups]
                size_counts = {}
                for size in group_sizes:
                    size_counts[size] = size_counts.get(size, 0) + 1
                
                print("\nGroup size distribution:")
                for size, count in sorted(size_counts.items()):
                    print(f"  Size {size}: {count} groups")
                
                # Save to cache for phase 3
                cache_key = 'cluster_merge_mapping'
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
                
                output_file = Path("cluster_merge_results.json")
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