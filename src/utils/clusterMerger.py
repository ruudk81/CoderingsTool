import asyncio
import functools
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict
import numpy as np
import numpy.typing as npt
from sklearn.metrics.pairwise import cosine_similarity
from openai import AsyncOpenAI
import instructor
import models

from pydantic import BaseModel, ConfigDict
from prompts import MERGE_PROMPT
from config import OPENAI_API_KEY, DEFAULT_MODEL, DEFAULT_LANGUAGE

class MergerConfig(BaseModel):
    """Configuration for the ClusterMerger"""
    model: str = DEFAULT_MODEL
    api_key: str = OPENAI_API_KEY
    max_concurrent_requests: int = 5 #10
    batch_size: int = 5  # Reduced for better prompt management
    similarity_threshold: float = 0.95   
    max_retries: int = 3
    retry_delay: int = 2
    language: str = DEFAULT_LANGUAGE
    verbose: bool = True
    temperature: float = 0.3
    max_tokens: int = 4000

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
    model_config = ConfigDict(arbitrary_types_allowed=True)

class MergeMap(BaseModel):
    """Mapping of cluster merges"""
    merge_groups: List[List[int]]
    cluster_to_merged: Dict[int, int]
    merge_reasons: Dict[int, str]
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
    decisions: List[MergeDecision]
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
        
        self.config = config or MergerConfig()
        self.verbose = verbose
        self.var_lab = var_lab
        self.client = client
        self.similarity_threshold = self.config.similarity_threshold
        self.cluster_data: Dict[int, ClusterData] = {}
        self.input_clusters: List[models.ClusterModel] = []
        self.merge_mapping: Optional[MergeMap] = None
        
        # Initialize async semaphore for rate limiting
        self.semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        # Initialize instructor-patched client
        self.instructor_client = None
        
        if input_list:
            if not isinstance(input_list, list):
                raise ValueError("input_list must be a list")
            if not all(isinstance(item, models.ClusterModel) for item in input_list):
                raise ValueError("All items in input_list must be ClusterModel objects")
            
            self.input_clusters = input_list
            self.populate_from_input_list(input_list)
            
            if self.verbose:
                print(f"Initialized ClusterMerger with {len(input_list)} clusters")
   
    def populate_from_input_list(self, input_list: List[models.ClusterModel]) -> None:
        """Extract cluster data from input ClusterModel objects"""
        if self.verbose:
            print("Populating cluster data from input models...")
  
        input_data = defaultdict(lambda: {
            'descriptive_codes': [],
            'code_descriptions': [],
            'cluster_embeddings': []
        })
     
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
                size=len(data['descriptive_codes'])
            )
            
        self.cluster_data = cluster_data
  
    async def merge_similar_clusters(self, cluster_data: Dict[int, ClusterData], var_lab: str) -> MergeMap:
        """Main method to identify and merge similar clusters using concurrent processing"""
        if self.verbose:
            print(f"Analyzing cluster similarity for research question: '{var_lab}'")
        
        # Calculate all pairwise similarities between clusters
        similarities_dict = await self._calculate_all_similarities(cluster_data)
        if self.verbose:
            print(f"Calculated similarities between all {len(cluster_data)} clusters")
        
        # Sort all pairs by similarity and filter by threshold
        sorted_pairs = self._get_sorted_similarity_pairs(similarities_dict)
        high_similarity_pairs = [(c1, c2, sim) for c1, c2, sim in sorted_pairs 
                               if sim >= self.similarity_threshold]
        
        if self.verbose:
            print(f"Found {len(high_similarity_pairs)} pairs with similarity ≥ {self.similarity_threshold}")
        
        # Process all merge decisions concurrently
        merge_groups = await self._concurrent_pairs_merging(
            high_similarity_pairs, cluster_data, var_lab)
        
        if self.verbose:
            print(f"Final merge: {len(cluster_data)} clusters → {len(merge_groups)} groups")
        
        # Create final merge mapping
        merge_mapping = self._create_merge_mapping(merge_groups)
        return merge_mapping
        
    def _get_sorted_similarity_pairs(self, similarities: Dict[Tuple[int, int], float]) -> List[Tuple[int, int, float]]:
        """Get all similarity pairs sorted by similarity (highest first)"""
        pairs = [(c1, c2, sim) for (c1, c2), sim in similarities.items()]
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs
    
    async def _calculate_all_similarities(self, cluster_data: Dict[int, ClusterData]) -> Dict[Tuple[int, int], float]:
        """Calculate all pairwise similarities between cluster centroids"""
        cluster_ids = list(cluster_data.keys())
        if len(cluster_ids) <= 1:
            return {}
        
        # Calculate pairwise similarities between cluster centroids
        centroids = np.array([cluster_data[cid].centroid for cid in cluster_ids])
        similarities = cosine_similarity(centroids)
        
        # Create dictionary of similarities
        similarity_dict = {}
        all_sim_values = []
        
        for i in range(len(cluster_ids)):
            for j in range(i+1, len(cluster_ids)):
                sim_score = similarities[i, j]
                all_sim_values.append(sim_score)
                similarity_dict[(cluster_ids[i], cluster_ids[j])] = sim_score
        
        # Log similarity distribution
        if self.verbose and all_sim_values:
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
    
    async def _concurrent_pairs_merging(self,
                                      sorted_pairs: List[Tuple[int, int, float]],
                                      cluster_data: Dict[int, ClusterData],
                                      var_lab: str) -> List[List[int]]:
        """
        Concurrent version of pairs merging using the Grader's pattern.
        
        This implementation:
        1. Creates batches of pair evaluation tasks
        2. Runs all batches concurrently for maximum speed
        3. Handles failures gracefully without stopping the process
        4. Applies merge decisions after all evaluations complete
        """
        if self.verbose:
            print(f"Starting concurrent pair-based merging with {len(sorted_pairs)} high-similarity pairs...")
        
        # Initialize cluster groups (each cluster starts in its own group)
        all_clusters = set()
        for c1, c2, _ in sorted_pairs:
            all_clusters.add(c1)
            all_clusters.add(c2)
        all_clusters.update(cluster_data.keys())
        
        if self.verbose:
            print(f"Processing a total of {len(all_clusters)} clusters")
        
        # Filter valid pairs (not already in same group, size constraints, etc.)
        valid_pairs = self._filter_valid_pairs(sorted_pairs)
        
        if not valid_pairs:
            if self.verbose:
                print("No valid pairs to evaluate after filtering")
            return [[cid] for cid in all_clusters]
        
        if self.verbose:
            print(f"Found {len(valid_pairs)} valid pairs to evaluate")
        
        # Get all merge decisions concurrently
        merge_decisions = await self._process_all_merge_batches(valid_pairs, cluster_data, var_lab)
        
        # Apply merge decisions to create final groups
        final_groups = self._apply_merge_decisions(all_clusters, valid_pairs, merge_decisions)
        
        # Calculate and log statistics
        total_merged = sum(1 for g in final_groups if len(g) > 1)
        merge_reduction = (len(all_clusters) - len(final_groups)) / len(all_clusters) * 100 if all_clusters else 0
        
        if self.verbose:
            print(f"Created {len(final_groups)} merge groups ({total_merged} merged groups) from {len(all_clusters)} clusters")
            print(f"Reduction: {merge_reduction:.1f}%")
            
        return final_groups
    
    def _filter_valid_pairs(self, sorted_pairs: List[Tuple[int, int, float]]) -> List[Tuple[int, int, float]]:
        """Filter pairs for validity (can be extended with more criteria)"""
        # For now, just return all pairs - can add size constraints or other filters here
        return sorted_pairs
    
    async def _process_all_merge_batches(self,
                                       valid_pairs: List[Tuple[int, int, float]],
                                       cluster_data: Dict[int, ClusterData],
                                       var_lab: str) -> Dict[Tuple[int, int], bool]:
        """
        Process all merge decision batches concurrently (following Grader pattern).
        
        Returns:
            Dictionary mapping (cluster1, cluster2) tuples to merge decisions
        """
        # Create batches
        batch_size = self.config.batch_size
        batches = [valid_pairs[i:i + batch_size] 
                  for i in range(0, len(valid_pairs), batch_size)]
        
        if self.verbose:
            print(f"Processing {len(batches)} batches concurrently with batch size {batch_size}")
        
        # Create concurrent tasks for all batches
        tasks = [self._process_merge_batch(batch, cluster_data, var_lab) 
                for batch in batches]
        
        # Run all batches concurrently with exception handling (like Grader)
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results and handle failures
        all_decisions = {}
        failed_batches = 0
        
        for i, batch_result in enumerate(batch_results):
            if isinstance(batch_result, Exception):
                if self.verbose:
                    print(f"Batch {i+1} processing failed: {str(batch_result)}")
                failed_batches += 1
                continue
            
            # Merge the decisions from this batch
            all_decisions.update(batch_result)
        
        if self.verbose:
            if failed_batches > 0:
                print(f"{failed_batches} out of {len(batches)} batches failed completely")
            print(f"Successfully collected {len(all_decisions)} merge decisions")
        
        return all_decisions
    
    async def _process_merge_batch(self,
                                 batch_pairs: List[Tuple[int, int, float]],
                                 cluster_data: Dict[int, ClusterData],
                                 var_lab: str) -> Dict[Tuple[int, int], bool]:
        """
        Process a single batch of merge decisions with retry logic (like Grader).
        
        Returns:
            Dictionary mapping (cluster1, cluster2) tuples to merge decisions for this batch
        """
        # Extract just the pair tuples for prompt creation
        pairs_for_prompt = [(c1, c2) for c1, c2, _ in batch_pairs]
        prompt = self._create_merge_decision_prompt(pairs_for_prompt, cluster_data, var_lab)
        
        # Retry logic similar to Grader._call_openai_api
        tries = 0
        max_tries = self.config.max_retries
        
        while tries < max_tries:
            tries += 1
            try:
                # Use semaphore for rate limiting
                async with self.semaphore:
                    batch_decisions = await self._call_merge_api(prompt)
                
                # Convert to dictionary format
                decisions_dict = {}
                if batch_decisions and batch_decisions.decisions:
                    for decision in batch_decisions.decisions:
                        pair_key = (decision.cluster_id_1, decision.cluster_id_2)
                        decisions_dict[pair_key] = decision.should_merge
                        
                        # Log individual decisions
                        if self.verbose:
                            status = "MERGE" if decision.should_merge else "SEPARATE"
                            reason = decision.reason #[:50] + "..." if len(decision.reason) > 50 else decision.reason
                            print(f"  {decision.cluster_id_1} & {decision.cluster_id_2}: {status} - {reason}")
                
                return decisions_dict
                
            except Exception as e:
                if self.verbose:
                    print(f"Batch API call failed on attempt {tries}/{max_tries}: {str(e)}")
                
                if tries >= max_tries:
                    raise
                
                # Exponential backoff (like Grader)
                await asyncio.sleep(2 ** tries)
                continue
    
    async def _call_merge_api(self, prompt: str) -> BatchMergeDecisionResponse:
        """Make API call for merge decisions with instructor"""
        # Initialize instructor client if needed
        if self.instructor_client is None:
            self.instructor_client = instructor.from_openai(
                AsyncOpenAI(api_key=self.config.api_key)
            )
        
        messages = [
            {"role": "system", "content": "You are an expert in thematic analysis and survey response clustering."},
            {"role": "user", "content": prompt}
        ]
        
        # Use run_in_executor pattern like Grader for consistency
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            functools.partial(
                self._sync_api_call,
                messages
            )
        )
        return response
    
    def _sync_api_call(self, messages: List[Dict]) -> BatchMergeDecisionResponse:
        """Synchronous API call wrapper"""
        import asyncio
        # Create new event loop for sync context
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(
                self.instructor_client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    response_model=BatchMergeDecisionResponse,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    seed=42
                )
            )
        finally:
            loop.close()
    
    def _apply_merge_decisions(self,
                             all_clusters: set,
                             valid_pairs: List[Tuple[int, int, float]],
                             merge_decisions: Dict[Tuple[int, int], bool]) -> List[List[int]]:
        """Apply merge decisions to create final cluster groups"""
        # Initialize with individual groups
        cluster_to_group = {}
        merge_groups = []
        
        for i, cid in enumerate(all_clusters):
            merge_groups.append([cid])
            cluster_to_group[cid] = i
        
        # Apply merge decisions
        merges_applied = 0
        for c1, c2, similarity in valid_pairs:
            # Get decision (check both orientations)
            should_merge = merge_decisions.get((c1, c2)) or merge_decisions.get((c2, c1), False)
            
            if should_merge:
                g1 = cluster_to_group.get(c1)
                g2 = cluster_to_group.get(c2)
                
                # Skip if already in same group
                if g1 == g2:
                    continue
                
                # Merge the groups (always merge larger index into smaller)
                if g1 > g2:
                    g1, g2 = g2, g1
                
                # Merge group g2 into g1
                merge_groups[g1].extend(merge_groups[g2])
                
                # Update group assignments
                for cid in merge_groups[g2]:
                    cluster_to_group[cid] = g1
                
                # Mark group g2 as empty
                merge_groups[g2] = []
                merges_applied += 1
                
                if self.verbose:
                    print(f"Applied merge: clusters {c1} & {c2} (similarity: {similarity:.4f})")
        
        # Filter out empty groups
        final_groups = [group for group in merge_groups if group]
        
        if self.verbose:
            print(f"Applied {merges_applied} merges successfully")
        
        return final_groups
    
    def _create_merge_decision_prompt(self,
                                    pairs: List[Tuple[int, int]],
                                    cluster_data: Dict[int, ClusterData],
                                    var_lab: str) -> str:
        """Create prompt for binary merge decisions with support for multiple pairs"""
        prompt = MERGE_PROMPT.replace("{var_lab}", var_lab)
        prompt = prompt.replace("{language}", self.config.language)
        
        # Create cluster pair descriptions
        pair_descriptions = []
        
        for idx, (cluster_id_1, cluster_id_2) in enumerate(pairs):
            cluster1 = cluster_data[cluster_id_1]
            representative_items1 = self._get_representative_items(cluster1, n=5)
            
            cluster2 = cluster_data[cluster_id_2]
            representative_items2 = self._get_representative_items(cluster2, n=5)
            
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
            
            description += f"\n\nQuestion for this pair: Do these clusters represent meaningfully different responses to the research question \"{var_lab}\", or are they essentially saying the same thing?"
            
            pair_descriptions.append(description)
        
        prompt = prompt.replace("{cluster_pairs}", "\n".join(pair_descriptions))
        
        if len(pairs) > 1:
            prompt += "\n\nRemember: For each pair, include both cluster_id_1 and cluster_id_2 in your response " \
                      "along with your should_merge decision and reason. Ensure cluster IDs match exactly."
        
        return prompt
    
    def _get_representative_items(self, cluster: ClusterData, n: int = 5) -> List[Dict[str, str]]:
        """Get most representative items from a cluster based on cosine similarity to centroid"""
        similarities = cosine_similarity([cluster.centroid], cluster.cluster_embeddings)[0]
        top_indices = np.argsort(similarities)[-n:][::-1]
            
        representatives = []
        for idx in top_indices:
            if idx < len(cluster.descriptive_codes) and idx < len(cluster.code_descriptions):
                representatives.append({
                    'code': cluster.descriptive_codes[idx],
                    'description': cluster.code_descriptions[idx],
                    'similarity': similarities[idx]
                })
        
        return representatives
    
    def _create_merge_mapping(self, merge_groups: List[List[int]]) -> MergeMap:
        """Create final merge mapping"""
        cluster_to_merged = {}
        merge_reasons = {}
        
        for i, group in enumerate(merge_groups):
            for cid in group:
                cluster_to_merged[cid] = i
        
        return MergeMap(
            merge_groups=merge_groups,
            cluster_to_merged=cluster_to_merged,
            merge_reasons=merge_reasons
        )

    def merge_clusters(self, input_list=None, var_lab=None):
        """Main entry point for merging clusters"""
        import time
        start_time = time.time()
        
        if self.verbose:
            print(f"Starting cluster merging process with {len(input_list) if input_list else 0} clusters")
        
        # Update input data if provided
        if input_list is not None:
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
        
        # Run the merging process
        if self.verbose:
            print("Beginning concurrent cluster merge analysis...")
        merged_clusters = asyncio.run(self._merge_clusters_async())
        
        # Verify output
        if not all(isinstance(item, models.ClusterModel) for item in merged_clusters):
            raise ValueError("Output must be a list of ClusterModel objects")
        
        # Calculate time taken
        end_time = time.time()
        elapsed_time = end_time - start_time
        if self.verbose:
            print(f"Cluster merging completed in {elapsed_time:.2f} seconds")
            print(f"Produced {len(merged_clusters)} ClusterModel objects with merged clusters")
            
        return merged_clusters, self.merge_mapping
        
    async def _merge_clusters_async(self):
        """Async implementation of cluster merging"""
        if self.verbose:
            print(f"Extracted data for {len(self.cluster_data)} clusters")
        
        # Merge similar clusters
        merge_mapping = await self.merge_similar_clusters(self.cluster_data, self.var_lab)
        
        # Convert to ClusterModel format
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
                	
        self.merge_mapping = merge_mapping
        return merged_clusters

    def _convert_to_mappers(self, merge_mapping):
        """Convert input cluster data to internal MergeResultMapper objects"""
        output_list = []
        
        for result in self.input_clusters:
            for segment in result.response_segment:
                if segment.micro_cluster is not None:
                    old_cluster_id = list(segment.micro_cluster.keys())[0]
                    merged_cluster_id = None
                    if old_cluster_id in merge_mapping.cluster_to_merged:
                        merged_cluster_id = merge_mapping.cluster_to_merged[old_cluster_id]
                    
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
        """Convert internal data to ClusterModel format"""
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
            response = response_mapping.get(respondent_id, "")
            
            submodels = []
            for item in items:
                segment_key = (respondent_id, item.segment_id)
                segment_response = segment_mapping.get(segment_key, "")
                
                micro_cluster = None
                if item.merged_cluster_id is not None:
                    micro_cluster = {item.merged_cluster_id: ""}
                
                submodel = models.ClusterSubmodel(
                    segment_id=item.segment_id,
                    segment_response=segment_response,
                    descriptive_code=item.descriptive_code,
                    code_description=item.code_description,
                    code_embedding=item.code_embedding,
                    description_embedding=item.description_embedding,
                    meta_cluster=None,
                    macro_cluster=None,
                    micro_cluster=micro_cluster
                )
                submodels.append(submodel)
            
            model = models.ClusterModel(
                respondent_id=respondent_id,
                response=response,
                response_segment=submodels
            )
            result_models.append(model)
            
        if self.verbose:
            print(f"Converted results to {len(result_models)} ClusterModel objects")
        return result_models