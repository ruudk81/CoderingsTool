import asyncio
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict
import numpy as np
import numpy.typing as npt
from sklearn.metrics.pairwise import cosine_similarity
from openai import AsyncOpenAI #Openai
import instructor
import models

from pydantic import BaseModel, ConfigDict #Field
from prompts import MERGE_PROMPT
from config import OPENAI_API_KEY, DEFAULT_MODEL, DEFAULT_LANGUAGE

# models for structured data
class MergerConfig(BaseModel):
    """Configuration for the ClusterMerger"""
    model: str = DEFAULT_MODEL
    api_key: str = OPENAI_API_KEY
    max_concurrent_requests: int = 10
    batch_size: int = 20
    similarity_threshold: float = 0.95   
    max_retries: int = 3
    retry_delay: int = 2
    language: str = DEFAULT_LANGUAGE
    verbose: bool = True  
    
class ClusterData(BaseModel):
    """Internal representation of cluster with extracted data"""
    cluster_id: int
    descriptive_codes: List[str]
    code_descriptions: List[str]
    cluster_embeddings: np.ndarray
    centroid: np.ndarray
    size: int
    
    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)

class ResultMapper(BaseModel):
    """Internal representation of cluster with extracted data"""
    respondent_id: Any
    segment_id: str
    descriptive_code: str
    code_description: str
    code_embedding: npt.NDArray[np.float32]
    description_embedding: npt.NDArray[np.float32]
    original_cluster_id: Optional[int] = None
    merged_cluster_id: Optional[int] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)  # for arrays with embeddings

class MergeMap(BaseModel):
    """Mapping of cluster merges"""
    merge_groups: List[List[int]]  # Groups of cluster IDs to merge
    cluster_to_merged: Dict[int, int]  # Original cluster ID → Merged cluster ID
    merge_reasons: Dict[int, str]  # Merged cluster ID → Reason
    model_config = ConfigDict(from_attributes=True)
  
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

class ClusterMerger:
    """Merges clusters that are not meaningfully differentiated from the perspective of the research question"""
    
    def __init__(
        self, 
        input_list: List[models.ClusterModel] = None, 
        var_lab: str = None, 
        config: MergerConfig = None, 
        client = None, 
        verbose: bool = True):
        
        # Initialize all attributes first to ensure they exist
        self.config = config or MergerConfig()
        self.verbose = verbose
        self.var_lab = var_lab
        self.client = client
        self.similarity_threshold = 0.95
        self.cluster_data: Dict[int, ClusterData] = {}
        self.input_clusters: List[models.ClusterModel] = []
        self.merge_mapping: Optional[MergeMap] = None
        
        # Initialize async semaphore
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        # Process input data if provided
        if input_list:
            if not isinstance(input_list, list):
                raise ValueError("input_list must be a list")
            if not all(isinstance(item, models.ClusterModel) for item in input_list):
                raise ValueError("All items in input_list must be ClusterModel objects")
            
            self.input_clusters = input_list  # Store the input clusters
            self.populate_from_input_list(input_list)
            
            if self.verbose:
                print(f"Initialized ClusterMerger with {len(input_list)} clusters")
   
    def populate_from_input_list(self, input_list: List[models.ClusterModel]) -> None:
        if self.verbose:
            print("Populating output list from input models...")
  
        input_data = defaultdict(lambda: {
            'descriptive_codes': [],
            'code_descriptions': [],
            'cluster_embeddings': [] })
     
        for response_item in input_list:
            for segment in response_item.response_segment:
                if segment.micro_cluster is not None:
                    cluster_id = list(segment.micro_cluster.keys())[0]
                    input_data[cluster_id]['descriptive_codes'].append(segment.descriptive_code)
                    input_data[cluster_id]['code_descriptions'].append(segment.code_description)
                    input_data[cluster_id]['cluster_embeddings'].append(segment.description_embedding)
        
        cluster_data = {}
        for cluster_id, data in input_data.items():
            embeddings_array = np.array(data['cluster_embeddings'])
            centroid = np.mean(embeddings_array, axis=0)
            
            cluster_data[cluster_id] = ClusterData(
                cluster_id=cluster_id,
                descriptive_codes=data['descriptive_codes'],
                code_descriptions=data['code_descriptions'],
                cluster_embeddings=embeddings_array,
                centroid=centroid,
                size=len(data['descriptive_codes']))
            
        self.cluster_data = cluster_data
  
    async def merge_similar_clusters(self,
                                   cluster_data: Dict[int, models.ClusterModel],
                                   var_lab: str) -> MergeMap:
        """Main method to identify and merge similar clusters using sorted similarity pairs"""
        if self.verbose:
            print(f"Analyzing cluster similarity for research question: '{var_lab}'")
        
        # Calculate all pairwise similarities between clusters
        similarities_dict = await self._calculate_all_similarities(cluster_data)
        if self.verbose:
            print(f"Calculated similarities between all {len(cluster_data)} clusters")
        
        # Sort all pairs by similarity (highest first) and filter by threshold
        sorted_pairs = self._get_sorted_similarity_pairs(similarities_dict)
        high_similarity_pairs = [(c1, c2, sim) for c1, c2, sim in sorted_pairs if sim >= self.similarity_threshold]
        if self.verbose:
            print(f"Found {len(high_similarity_pairs)} pairs with similarity ≥ {self.similarity_threshold}")
        
        # Merge clusters by processing sorted pairs
        merge_groups = await self._sorted_pairs_merging(
            high_similarity_pairs,
            cluster_data, 
            var_lab)
        
        if self.verbose:
            print(f"Final merge: {len(cluster_data)} clusters → {len(merge_groups)} groups")
        
        # Create final merge mapping
        merge_mapping = self._create_merge_mapping(merge_groups)
        
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
        if self.verbose:
            if all_sim_values:
                all_sim_values = np.array(all_sim_values)
                print("Similarity statistics:")
                print(f"  Min: {np.min(all_sim_values):.4f}")
                print(f"  Max: {np.max(all_sim_values):.4f}")
                print(f"  Mean: {np.mean(all_sim_values):.4f}")
                print(f"  Median: {np.median(all_sim_values):.4f}")
                print("  Percentiles:")
                for p in [50, 75, 90, 95, 99]:
                    print(f"    {p}%: {np.percentile(all_sim_values, p):.4f}")
            
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
            print(f"No similarity found for clusters {cluster1} and {cluster2}")
            return 0.0
    
    async def _sorted_pairs_merging(self,
                                sorted_pairs: List[Tuple[int, int, float]],
                                cluster_data: Dict[int, ClusterData],
                                var_lab: str) -> List[List[int]]:
        """Merge clusters by processing similarity pairs in sorted order with batched LLM requests
        
        This improved implementation:
        1. Batches LLM requests for efficiency (fewer API calls)
        2. Prioritizes the most promising pairs based on embedding similarity
        3. Updates merge groups dynamically as decisions come in
        4. Provides detailed progress logging
        """
        if self.verbose:
            print(f"Starting optimized pair-based merging with {len(sorted_pairs)} high-similarity pairs...")
        
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
        if self.verbose: 
            print(f"Processing a total of {len(all_clusters)} clusters")
        
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
            
            # # Skip if merging would exceed max group size
            # if len(merge_groups[g1]) + len(merge_groups[g2]) > 5:
            #     if self.verbose:
            #         print(f"Skipping evaluation: merging clusters {c1} & {c2} would exceed max size")
            #     continue
            
            valid_pairs.append((c1, c2, similarity))
        
        if self.verbose:
            print(f"Found {len(valid_pairs)} valid pairs to evaluate (after filtering for size constraints)")
        
        # Process pairs in batches
        llm_calls = 0
        total_batches = (len(valid_pairs) + batch_size - 1) // batch_size
        merge_decisions = {}  # Cache decisions to avoid redundant API calls
        
        # Process in batches with logging
        for batch_idx in range(0, len(valid_pairs), batch_size):
            batch_end = min(batch_idx + batch_size, len(valid_pairs))
            current_batch = valid_pairs[batch_idx:batch_end]
            
            if self.verbose:
                print(f"Processing batch {batch_idx // batch_size + 1}/{total_batches} "
                      f"with {len(current_batch)} pairs...")
            
            # Extract pairs for the batch
            batch_pairs = [(c1, c2) for c1, c2, _ in current_batch]
            
            # Create batch prompt
            prompt = self._create_merge_decision_prompt(batch_pairs, cluster_data, var_lab)
            
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
                        if self.verbose:
                            print(f"Decision for clusters {decision.cluster_id_1} & {decision.cluster_id_2}: "
                                  f"{status} - {reason}")
            except Exception as e:
                if self.verbose:
                    print(f"Error processing batch: {e}")
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
   
                # Get decision (from either order of pairs)
                should_merge = merge_decisions.get((c1, c2)) or merge_decisions.get((c2, c1), False)
                
                if should_merge:
                    # Merge the two groups
                    if self.verbose:
                        print(f"Applying merge for clusters {c1} & {c2} (similarity: {similarity:.4f})")
                    
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
        
        if self.verbose:
            print(f"Merging completed with {llm_calls} LLM batch calls")
            print(f"Created {len(final_groups)} merge groups ({total_merged} merged groups) from {len(all_clusters)} clusters")
            print(f"Reduction: {merge_reduction:.1f}%")
            
        return final_groups
    
    def _create_merge_decision_prompt(self,
                                 pairs: List[Tuple[int, int]],
                                 cluster_data: Dict[int, ClusterData],
                                 var_lab: str) -> str:
        """Create prompt for binary merge decisions with support for multiple pairs
        
        This updated version:
        1. Can handle batches of pairs in a single prompt for efficiency
        2. Selects the 5 most representative codes based on cosine similarity to centroid
        3. Focuses explicitly on the research question context
        4. Presents clear decision criteria based on meaningful differentiation
        """
        prompt = MERGE_PROMPT.replace("{var_lab}", var_lab)
        prompt = prompt.replace("{language}", self.config.language)
        
        # Create cluster pair descriptions
        pair_descriptions = []
        
        for idx, (cluster_id_1, cluster_id_2) in enumerate(pairs):
            # Get data for first cluster
            cluster1 = cluster_data[cluster_id_1]
            representative_items1 = self._get_representative_items(cluster1, n=5)
            
            # Get data for second cluster
            cluster2 = cluster_data[cluster_id_2]
            representative_items2 = self._get_representative_items(cluster2, n=5)
            
            # Create pair description with a number for batches
            description = f"\n\nPAIR {idx+1}: Cluster {cluster_id_1} vs Cluster {cluster_id_2}\n"
            
            # First cluster
            description += f"\nCluster {cluster_id_1}:"
            description += "\n- Most representative responses (by similarity to cluster centroid):"
            
            for i, item in enumerate(representative_items1):
                description += f"\n  {i+1}. {item['code']}: {item['description']}"
            
            # Second cluster
            description += f"\n\nCluster {cluster_id_2}:"
            description += "\n- Most representative responses (by similarity to cluster centroid):"
            
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
        similarities = cosine_similarity([cluster.centroid], cluster.cluster_embeddings)[0]
        
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
                    if self.verbose:
                        print(f"Failed to get merge decisions after {self.config.max_retries} attempts: {e}")
                    raise e
    
    def _create_merge_mapping(self,
                           merge_groups: List[List[int]]) -> MergeMap:
        """Create final merge mapping"""
        cluster_to_merged = {}
        merge_reasons = {}
        
        for i, group in enumerate(merge_groups):
            # All clusters in group map to the same merged ID
            for cid in group:
                cluster_to_merged[cid] = i
        
        return MergeMap(
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
        import time
        start_time = time.time()
        
        if self.verbose:
            print(f"Starting cluster merging process with {len(input_list) if input_list else 0} clusters")
        
        # Update input data if provided
        if input_list is not None:
            # Validate input_list is a list of ClusterModel objects
            if not all(isinstance(item, models.ClusterModel) for item in input_list):
                raise ValueError("input_list must be a list of ClusterModel objects")
            self.input_clusters = input_list
            if self.verbose:
                print(f"Received {len(input_list)} ClusterModel objects as input")
            
        if var_lab is not None:
            self.var_lab = var_lab
            if self.verbose:
                print(f"Using research question: '{var_lab}'")
            
        # Validate input
        if not self.input_clusters or not self.var_lab:
            raise ValueError("Input clusters and var_lab must be provided")
        
        # Run the actual merging process
        if self.verbose:
            print("Beginning cluster merge analysis...")
        merged_clusters = asyncio.run(self._merge_clusters_async())
        
        # Verify output is a list of ClusterModel objects
        if not all(isinstance(item, models.ClusterModel) for item in merged_clusters):
            raise ValueError("Output must be a list of ClusterModel objects")
        
        # Calculate time taken
        end_time = time.time()
        elapsed_time = end_time - start_time
        if self.verbose:
            print(f"Cluster merging completed in {elapsed_time:.2f} seconds")
            print(f"Produced {len(merged_clusters)} ClusterModel objects with merged clusters")
            
        # Return both the merged clusters and the merge mapping for caching
        return merged_clusters, self.merge_mapping
        
    async def _merge_clusters_async(self):
        """Async implementation of cluster merging"""
        # Extract cluster data
     
        if self.verbose:
            print(f"Extracted data for {len(self.cluster_data)} clusters")
        
 
        # Merge similar clusters
        merge_mapping = await self.merge_similar_clusters(self.cluster_data, self.var_lab)
        
        # Convert to ClusterModel using to_cluster_model
        if self.verbose:
            print("Converting merged clusters to ClusterModel format...")
        merged_clusters = self.to_cluster_model(merge_mapping)
        
        # Calculate and log statistics
        total_initial = len(self.cluster_data)
        total_final = len(set(merge_mapping.cluster_to_merged.values()))
        merged_groups = [g for g in merge_mapping.merge_groups if len(g) > 1]
        
        if self.verbose:
            print("Cluster merging statistics:")
            print(f"Initial clusters: {total_initial}")
            print(f"Merge groups: {len(merged_groups)}")
            print(f"Final clusters after merging: {total_final}")
            print(f"Reduction: {(1 - total_final/total_initial) * 100:.1f}%")
                	
        # Store merge mapping for reference
        self.merge_mapping = merge_mapping
        
        return merged_clusters

    
    def _convert_to_mappers(self, merge_mapping):
        """Convert input cluster data to internal MergeResultMapper objects"""
        output_list = []
        
        # Process each cluster result
        for result in self.input_clusters:
            for segment in result.response_segment:
                if segment.micro_cluster is not None:
                    old_cluster_id = list(segment.micro_cluster.keys())[0]
                    
                    # Get the merged cluster ID
                    merged_cluster_id = None
                    if old_cluster_id in merge_mapping.cluster_to_merged:
                        merged_cluster_id = merge_mapping.cluster_to_merged[old_cluster_id]
                    
                    # Create mapper object
                    mapper = ResultMapper(
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
                    macro_cluster=None,  # No meso clusters
                    micro_cluster=micro_cluster  # Use merged cluster ID
                )
                
                submodels.append(submodel)
            
            # Create ClusterModel
            model = models.ClusterModel(
                respondent_id=respondent_id,
                response=response,
                response_segment=submodels
            )
            
            result_models.append(model)
            
        if self.verbose:
            print(f"Converted results to {len(result_models)} ClusterModel objects")
        return result_models
   