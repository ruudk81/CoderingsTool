import asyncio
import functools
import time
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field
from tqdm import tqdm
import instructor
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# Project imports
import models
from config import DEFAULT_LANGUAGE, DEFAULT_MODEL, OPENAI_API_KEY
from modules.utils import qualityFilter
from prompts import CLUSTER_LABELING_PROMPT, RESPONSE_SUMMARY_PROMPT, THEME_SUMMARY_PROMPT

# Patch OpenAI client with instructor for structured output
client = instructor.patch(OpenAI(api_key=OPENAI_API_KEY))


class LabellerConfig(BaseModel):
    """Configuration for the Labeller utility"""
    model: str = DEFAULT_MODEL
    temperature: float = 0.0
    max_retries: int = 3
    batch_size: int = 10
    max_tokens: int = 4000
    timeout: int = 30
    max_samples_per_cluster: int = 10
    min_samples_for_labeling: int = 3
    language: str = DEFAULT_LANGUAGE


class ClusterInfo(BaseModel):
    """Internal model for cluster information"""
    cluster_id: int
    cluster_type: str  # "meta", "meso", "micro"
    items: List[str] = Field(default_factory=list)
    codes: List[str] = Field(default_factory=list)
    descriptions: List[str] = Field(default_factory=list)
    respondent_ids: List[str] = Field(default_factory=list)
    segment_ids: List[str] = Field(default_factory=list)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ClusterContent(BaseModel):
    """Extracted content from a cluster for analysis"""
    cluster_id: int
    descriptive_codes: List[str]
    code_descriptions: List[str]
    unique_codes: List[str]
    unique_descriptions: List[str]
    sample_items: List[Dict[str, str]]  # code, description pairs
    item_count: int
    embeddings: Optional[npt.NDArray[np.float32]] = None  # Store embeddings for centroid calculation
    
    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)


class HierarchicalCluster(BaseModel):
    """Complete hierarchical cluster with all metadata"""
    cluster_id: int
    hierarchy_path: str  # "1/1.1/1.1.1"
    meta_label: str
    meso_label: str
    micro_label: str
    content: ClusterContent
    original_cluster_ids: List[int]  # Pre-merge cluster IDs
    merge_reason: Optional[str] = None
    
    model_config = ConfigDict(from_attributes=True)


class LabelResponse(BaseModel):
    """Response model for LLM labeling"""
    label: str
    confidence: float = Field(ge=0.0, le=1.0)


# New data models for 4-stage labelling system
class InitialLabelResponse(BaseModel):
    """Stage 1: Initial cluster labeling response"""
    cluster_id: int
    label: str
    keywords: List[str]
    theme_summary: str
    confidence: float = Field(ge=0.0, le=1.0)
    
    model_config = ConfigDict(from_attributes=True)


class BatchInitialLabelResponse(BaseModel):
    """Batch response for multiple cluster labels"""
    responses: List[InitialLabelResponse]
    
    model_config = ConfigDict(from_attributes=True)


class MergeAnalysisResponse(BaseModel):
    """Stage 2: Cluster similarity analysis for potential merging"""
    should_merge: bool
    cluster_1_id: int
    cluster_2_id: int
    reason: str
    similarity_score: float = Field(ge=0.0, le=1.0)
    
    model_config = ConfigDict(from_attributes=True)


class MergeRemapResponse(BaseModel):
    """Stage 2: Final merge decisions with remap dictionary"""
    merge_groups: List[List[int]]  # Groups of cluster IDs to merge
    remap_dict: Dict[int, int]  # Old ID -> New ID mapping
    merge_rationale: Dict[str, str]  # New ID -> Reason for merge
    
    model_config = ConfigDict(from_attributes=True)


class HierarchyResponse(BaseModel):
    """Stage 3: Hierarchical structure assignment"""
    cluster_id: int
    meta_level: str  # e.g., "1", "2", "3"
    meso_level: str  # e.g., "1.1", "1.2"
    micro_level: str  # e.g., "1.1.1", "1.1.2"
    hierarchy_path: str  # Full path "1/1.1/1.1.1"
    level_labels: Dict[str, str]  # Level -> Label mapping
    
    model_config = ConfigDict(from_attributes=True)


class HierarchyBatchResponse(BaseModel):
    """Batch response for hierarchy assignments"""
    hierarchies: List[HierarchyResponse]
    meta_labels: Dict[str, str]  # Meta level ID -> Label
    meso_labels: Dict[str, str]  # Meso level ID -> Label
    
    model_config = ConfigDict(from_attributes=True)


class RefinedLabelResponse(BaseModel):
    """Stage 4: Refined labels for mutual exclusivity"""
    hierarchy_level: str  # "meta", "meso", or "micro"
    hierarchy_path: str  # e.g., "1.1" or "1.1.1"
    original_label: str
    refined_label: str
    is_mutually_exclusive: bool
    differentiation_notes: str
    reasoning: str
    
    model_config = ConfigDict(from_attributes=True)


class BatchRefinedLabelResponse(BaseModel):
    """Batch response for refined labels"""
    refined_labels: List[RefinedLabelResponse]
    
    model_config = ConfigDict(from_attributes=True)


class LabelingSummary(BaseModel):
    """Summary of the complete labeling process"""
    total_initial_clusters: int
    clusters_after_merge: int
    merge_ratio: float
    hierarchical_structure: Dict[str, Dict[str, List[str]]]  # meta -> meso -> micro
    quality_metrics: Dict[str, float]
    processing_time: float
    
    model_config = ConfigDict(from_attributes=True)


class Labeller:
    """Labeller for creating hierarchical labels from clusters"""
    
    def __init__(self, config: Optional[LabellerConfig] = None):
        self.config = config or LabellerConfig()
        self.client = client  # OpenAI client patched with instructor
        
    def extract_cluster_content(self, cluster_results: List[models.ClusterModel]) -> Dict[int, ClusterContent]:
        """Extract descriptive codes and descriptions from each cluster
        
        Args:
            cluster_results: List of ClusterModel from clustering step
            
        Returns:
            Dictionary mapping cluster ID to ClusterContent
        """
        cluster_data = defaultdict(lambda: {
            'descriptive_codes': [],
            'code_descriptions': [],
            'embeddings': [],
            'respondent_ids': [],
            'segment_ids': []
        })
        
        # Group items by cluster ID
        for result in cluster_results:
            for segment in result.response_segment:
                if segment.mirco_cluster is not None:
                    cluster_id = list(segment.mirco_cluster.keys())[0]
                    cluster_data[cluster_id]['descriptive_codes'].append(segment.descriptive_code)
                    cluster_data[cluster_id]['code_descriptions'].append(segment.code_description)
                    # Store embeddings based on config (description by default)
                    cluster_data[cluster_id]['embeddings'].append(segment.description_embedding)
                    cluster_data[cluster_id]['respondent_ids'].append(result.respondent_id)
                    cluster_data[cluster_id]['segment_ids'].append(segment.segment_id)
        
        # Convert to ClusterContent objects
        cluster_contents = {}
        for cluster_id, data in cluster_data.items():
            # Create sample items
            sample_items = []
            for i in range(min(10, len(data['descriptive_codes']))):
                sample_items.append({
                    'code': data['descriptive_codes'][i],
                    'description': data['code_descriptions'][i]
                })
            
            cluster_contents[cluster_id] = ClusterContent(
                cluster_id=cluster_id,
                descriptive_codes=data['descriptive_codes'],
                code_descriptions=data['code_descriptions'],
                unique_codes=list(set(data['descriptive_codes'])),
                unique_descriptions=list(set(data['code_descriptions'])),
                sample_items=sample_items,
                item_count=len(data['descriptive_codes'])
            )
            
            # Store embeddings separately for centroid calculation
            cluster_contents[cluster_id].embeddings = np.array(data['embeddings'])
            
        return cluster_contents
    
    def get_representative_items(self, content: ClusterContent, n_items: int = 5) -> List[Dict[str, str]]:
        """Get most representative items using cosine similarity with centroid
        
        Args:
            content: ClusterContent with embeddings
            n_items: Number of representative items to return
            
        Returns:
            List of most representative code/description pairs
        """
        if not hasattr(content, 'embeddings') or len(content.embeddings) == 0:
            return content.sample_items[:n_items]
        
        # Calculate centroid
        centroid = np.mean(content.embeddings, axis=0)
        
        # Calculate cosine similarities
        similarities = cosine_similarity([centroid], content.embeddings)[0]
        
        # Get indices of most similar items
        top_indices = np.argsort(similarities)[-n_items:][::-1]
        
        # Return representative items
        representative_items = []
        for idx in top_indices:
            if idx < len(content.descriptive_codes):
                representative_items.append({
                    'code': content.descriptive_codes[idx],
                    'description': content.code_descriptions[idx],
                    'similarity': float(similarities[idx])
                })
        
        return representative_items
    
    def label_initial_clusters(self, 
                             cluster_results: List[models.ClusterModel], 
                             var_lab: str) -> Dict[int, InitialLabelResponse]:
        """Stage 1: Label each micro-cluster through the lens of var_lab
        
        Args:
            cluster_results: Cluster results from clustering step
            var_lab: Survey question for context
            
        Returns:
            Dictionary mapping cluster ID to InitialLabelResponse
        """
        # Extract cluster content
        cluster_contents = self.extract_cluster_content(cluster_results)
        
        labeled_clusters = {}
        
        # Process clusters in batches for efficiency
        batch_size = self.config.batch_size
        cluster_ids = list(cluster_contents.keys())
        
        with tqdm(total=len(cluster_ids), desc="Labeling clusters") as pbar:
            for i in range(0, len(cluster_ids), batch_size):
                batch_ids = cluster_ids[i:i + batch_size]
                batch_contents = [cluster_contents[cid] for cid in batch_ids]
                
                # Get representative items for each cluster
                batch_representatives = []
                for content in batch_contents:
                    rep_items = self.get_representative_items(content)
                    batch_representatives.append(rep_items)
                
                # Create batch prompt
                prompt = self._create_batch_labeling_prompt(
                    batch_contents, 
                    batch_representatives,
                    var_lab
                )
                
                # Get labels from LLM
                response = self._get_batch_labels(prompt, batch_ids)
                
                # Store results
                for label_response in response.responses:
                    labeled_clusters[label_response.cluster_id] = label_response
                
                pbar.update(len(batch_ids))
        
        return labeled_clusters
    
    def _create_batch_labeling_prompt(self, 
                                     contents: List[ClusterContent],
                                     representatives: List[List[Dict]],
                                     var_lab: str) -> str:
        """Create prompt for batch labeling
        
        Args:
            contents: List of ClusterContent objects
            representatives: List of representative items per cluster
            var_lab: Survey question context
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""You are analyzing survey responses to the question: "{var_lab}"
        
Please label the following clusters based on their content. Each cluster shows the MOST REPRESENTATIVE items,
selected using cosine similarity to the cluster centroid. These are the items that best capture the essence
of each cluster.

For each cluster, provide:
1. A concise, descriptive label that captures the main theme
2. 3-5 keywords that represent the cluster
3. A brief theme summary (1-2 sentences)
4. A confidence score (0.0-1.0)

Focus on creating labels that directly answer or relate to the survey question. Base your labels primarily
on the representative items shown, as they are the most characteristic of each cluster.

Clusters to analyze:
"""
        
        for i, (content, reps) in enumerate(zip(contents, representatives)):
            prompt += f"\n\nCluster {content.cluster_id}:"
            prompt += f"\nSize: {content.item_count} responses"
            prompt += f"\nMost representative items (selected by highest cosine similarity to cluster centroid):"
            
            for j, item in enumerate(reps, 1):
                prompt += f"\n  {j}. Code: {item['code']}"
                prompt += f"\n     Description: {item['description']}"
                if 'similarity' in item:
                    prompt += f"\n     (Centroid similarity: {item['similarity']:.3f})"
            
            # Add some unique codes if available, but emphasize they're less representative
            if len(content.unique_codes) > len(reps):
                prompt += f"\n\nAdditional codes in cluster (less representative):"
                other_codes = [code for code in content.unique_codes 
                              if code not in [r['code'] for r in reps]][:3]
                for code in other_codes:
                    prompt += f"\n  - {code}"
        
        prompt += "\n\nProvide labels for all clusters based primarily on the representative items shown."
        
        return prompt
    
    def _get_batch_labels(self, prompt: str, cluster_ids: List[int]) -> BatchInitialLabelResponse:
        """Get labels from LLM for a batch of clusters
        
        Args:
            prompt: Formatted prompt string
            cluster_ids: List of cluster IDs in the batch
            
        Returns:
            BatchInitialLabelResponse with labels for all clusters
        """
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "You are an expert in thematic analysis and survey response clustering."},
                    {"role": "user", "content": prompt}
                ],
                response_model=BatchInitialLabelResponse,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            # Ensure all cluster IDs are covered
            response_ids = {r.cluster_id for r in response.responses}
            missing_ids = set(cluster_ids) - response_ids
            
            if missing_ids:
                # Fallback for missing labels
                for cluster_id in missing_ids:
                    response.responses.append(
                        InitialLabelResponse(
                            cluster_id=cluster_id,
                            label=f"Cluster {cluster_id}",
                            keywords=["unlabeled"],
                            theme_summary="Failed to generate label",
                            confidence=0.0
                        )
                    )
            
            return response
            
        except Exception as e:
            print(f"Error getting batch labels: {e}")
            # Return fallback response
            return BatchInitialLabelResponse(
                responses=[
                    InitialLabelResponse(
                        cluster_id=cid,
                        label=f"Cluster {cid}",
                        keywords=["error"],
                        theme_summary="Error generating label",
                        confidence=0.0
                    )
                    for cid in cluster_ids
                ]
            )
    
    def analyze_semantic_similarity(self,
                                  labeled_clusters: Dict[int, InitialLabelResponse],
                                  cluster_contents: Dict[int, ClusterContent],
                                  var_lab: str) -> MergeRemapResponse:
        """Stage 2: Identify clusters that are semantically too similar
        
        Sequential approach: Process multiple primary clusters at once, comparing
        each with its subsequent clusters, skipping those already merged.
        
        Args:
            labeled_clusters: Initial labels from Stage 1
            cluster_contents: Cluster content information
            var_lab: Survey question for context
            
        Returns:
            MergeRemapResponse with merge groups and remap dictionary
        """
        cluster_ids = sorted(list(labeled_clusters.keys()))
        merged_into = {}  # Maps cluster ID to the cluster it was merged into
        merge_groups = defaultdict(list)  # Maps primary cluster to list of merged clusters
        similarity_results = []
        
        # Process clusters in batches
        primary_batch_size = 10  # Process 10 primary clusters at once
        total_processed = 0
        
        with tqdm(total=len(cluster_ids), desc="Processing clusters") as pbar:
            for batch_start in range(0, len(cluster_ids), primary_batch_size):
                batch_end = min(batch_start + primary_batch_size, len(cluster_ids))
                primary_batch = cluster_ids[batch_start:batch_end]
                
                # Collect all pairs to analyze for this batch of primary clusters
                all_pairs = []
                for i, primary_cluster in enumerate(primary_batch):
                    # Skip if this cluster has already been merged into another
                    if primary_cluster in merged_into:
                        continue
                    
                    # Get the original index in the full cluster list
                    original_idx = batch_start + i
                    
                    # Compare with all subsequent clusters
                    for secondary_cluster in cluster_ids[original_idx + 1:]:
                        # Skip if secondary cluster already merged
                        if secondary_cluster not in merged_into:
                            all_pairs.append((primary_cluster, secondary_cluster))
                
                # Analyze all pairs in sub-batches
                comparison_batch_size = 20  # Analyze 20 pairs at once
                for pair_batch_start in range(0, len(all_pairs), comparison_batch_size):
                    pair_batch_end = min(pair_batch_start + comparison_batch_size, len(all_pairs))
                    batch_pairs = all_pairs[pair_batch_start:pair_batch_end]
                    
                    if not batch_pairs:
                        continue
                    
                    # Create batch prompt
                    prompt = self._create_similarity_analysis_prompt(
                        batch_pairs,
                        labeled_clusters,
                        cluster_contents,
                        var_lab
                    )
                    
                    # Get similarity analysis
                    results = self._get_similarity_analysis(prompt, batch_pairs)
                    
                    # Process results immediately to update merge status
                    for result in results:
                        similarity_results.append(result)
                        
                        if result.should_merge:
                            # Merge secondary into primary
                            merged_into[result.cluster_2_id] = result.cluster_1_id
                            merge_groups[result.cluster_1_id].append(result.cluster_2_id)
                
                # Update progress for this batch of primary clusters
                pbar.update(len(primary_batch))
        
        # Convert merge_groups to the expected format
        final_merge_groups = []
        for primary, secondaries in merge_groups.items():
            group = [primary] + secondaries
            final_merge_groups.append(sorted(group))
        
        # Create remap dictionary
        remap_dict, merge_rationale = self._create_remap_dictionary(
            final_merge_groups,
            labeled_clusters,
            similarity_results
        )
        
        return MergeRemapResponse(
            merge_groups=final_merge_groups,
            remap_dict=remap_dict,
            merge_rationale=merge_rationale
        )
    
    def _create_similarity_analysis_prompt(self,
                                         batch_pairs: List[Tuple[int, int]],
                                         labeled_clusters: Dict[int, InitialLabelResponse],
                                         cluster_contents: Dict[int, ClusterContent],
                                         var_lab: str) -> str:
        """Create prompt for semantic similarity analysis
        
        Args:
            batch_pairs: List of cluster pairs to analyze
            labeled_clusters: Initial cluster labels
            cluster_contents: Cluster content information
            var_lab: Survey question context
            
        Returns:
            Formatted prompt for similarity analysis
        """
        prompt = f"""You are analyzing clusters of survey responses to the question: "{var_lab}"

Your task is to determine if pairs of clusters are semantically similar enough that they should be merged.
Consider these factors:
1. Do the clusters represent the same underlying theme or concept?
2. Are the keywords and labels conceptually overlapping?
3. Would merging improve clarity without losing important distinctions?
4. Is the distinction between clusters meaningful in the context of the survey question?

Analyze the following cluster pairs:
"""
        
        for pair_idx, (id1, id2) in enumerate(batch_pairs):
            label1 = labeled_clusters[id1]
            label2 = labeled_clusters[id2]
            content1 = cluster_contents[id1]
            content2 = cluster_contents[id2]
            
            prompt += f"\n\nPair {pair_idx + 1}: Cluster {id1} vs Cluster {id2}"
            
            # Cluster 1 details
            prompt += f"\n\nCluster {id1}:"
            prompt += f"\n- Label: {label1.label}"
            prompt += f"\n- Keywords: {', '.join(label1.keywords)}"
            prompt += f"\n- Theme: {label1.theme_summary}"
            prompt += f"\n- Size: {content1.item_count} items"
            prompt += f"\n- Sample codes: {', '.join(content1.unique_codes[:5])}"
            
            # Cluster 2 details
            prompt += f"\n\nCluster {id2}:"
            prompt += f"\n- Label: {label2.label}"
            prompt += f"\n- Keywords: {', '.join(label2.keywords)}"
            prompt += f"\n- Theme: {label2.theme_summary}"
            prompt += f"\n- Size: {content2.item_count} items"
            prompt += f"\n- Sample codes: {', '.join(content2.unique_codes[:5])}"
        
        prompt += """

For each pair, provide:
1. Should these clusters be merged? (true/false)
2. Reasoning for your decision
3. Similarity score (0.0-1.0, where 1.0 is identical)

Base your decision on semantic meaning rather than surface-level similarities."""
        
        return prompt
    
    def _get_similarity_analysis(self, 
                               prompt: str,
                               batch_pairs: List[Tuple[int, int]]) -> List[MergeAnalysisResponse]:
        """Get similarity analysis from LLM
        
        Args:
            prompt: Analysis prompt
            batch_pairs: Cluster pairs being analyzed
            
        Returns:
            List of MergeAnalysisResponse objects
        """
        try:
            # Create a batch request structure
            class BatchMergeAnalysisResponse(BaseModel):
                analyses: List[MergeAnalysisResponse]
            
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "You are an expert in semantic analysis and clustering."},
                    {"role": "user", "content": prompt}
                ],
                response_model=BatchMergeAnalysisResponse,
                temperature=0.1,  # Lower temperature for more consistent analysis
                max_tokens=self.config.max_tokens
            )
            
            # Map responses to correct cluster pairs
            results = []
            for i, analysis in enumerate(response.analyses):
                if i < len(batch_pairs):
                    analysis.cluster_1_id = batch_pairs[i][0]
                    analysis.cluster_2_id = batch_pairs[i][1]
                    results.append(analysis)
            
            return results
            
        except Exception as e:
            print(f"Error in similarity analysis: {e}")
            # Return no-merge decisions as fallback
            return [
                MergeAnalysisResponse(
                    should_merge=False,
                    cluster_1_id=pair[0],
                    cluster_2_id=pair[1],
                    reason="Error in analysis",
                    similarity_score=0.0
                )
                for pair in batch_pairs
            ]
    
    
    def _create_remap_dictionary(self,
                               merge_groups: List[List[int]],
                               labeled_clusters: Dict[int, InitialLabelResponse],
                               similarity_results: List[MergeAnalysisResponse]) -> Tuple[Dict[int, int], Dict[str, str]]:
        """Create remap dictionary from merge groups
        
        Args:
            merge_groups: List of cluster groups to merge
            labeled_clusters: Initial cluster labels
            similarity_results: Similarity analysis results
            
        Returns:
            Tuple of (remap_dict, merge_rationale)
        """
        remap_dict = {}
        merge_rationale = {}
        
        # Create mapping of pairs to their merge reasons
        pair_reasons = {}
        for result in similarity_results:
            if result.should_merge:
                pair_key = tuple(sorted([result.cluster_1_id, result.cluster_2_id]))
                pair_reasons[pair_key] = result.reason
        
        # Process each merge group
        next_id = 0
        all_cluster_ids = set(labeled_clusters.keys())
        
        for group in merge_groups:
            # Choose representative cluster (e.g., largest or highest confidence)
            representative = max(group, key=lambda x: labeled_clusters[x].confidence)
            
            # Create mapping for all clusters in group
            for cluster_id in group:
                remap_dict[cluster_id] = next_id
            
            # Create merge rationale
            reasons = []
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    pair_key = tuple(sorted([group[i], group[j]]))
                    if pair_key in pair_reasons:
                        reasons.append(pair_reasons[pair_key])
            
            merge_rationale[str(next_id)] = f"Merged clusters {group} - " + "; ".join(list(set(reasons))[:3])
            next_id += 1
        
        # Map non-merged clusters to new IDs
        for cluster_id in all_cluster_ids:
            if cluster_id not in remap_dict:
                remap_dict[cluster_id] = next_id
                next_id += 1
        
        return remap_dict, merge_rationale
    
    def apply_cluster_merging(self,
                            cluster_results: List[models.ClusterModel],
                            remap_dict: Dict[int, int]) -> List[models.ClusterModel]:
        """Apply the merge remapping to consolidate clusters
        
        Args:
            cluster_results: Original cluster results
            remap_dict: Mapping from old cluster IDs to new IDs
            
        Returns:
            Updated cluster results with merged clusters
        """
        # Create deep copy of results to avoid modifying originals
        import copy
        merged_results = copy.deepcopy(cluster_results)
        
        # Apply remapping to all clusters
        for result in merged_results:
            for segment in result.response_segment:
                if segment.mirco_cluster is not None:
                    old_id = list(segment.mirco_cluster.keys())[0]
                    if old_id in remap_dict:
                        new_id = remap_dict[old_id]
                        segment.mirco_cluster = {new_id: ""}
        
        return merged_results
    
    def create_hierarchy(self,
                        labeled_clusters: Dict[int, InitialLabelResponse],
                        cluster_contents: Dict[int, ClusterContent],
                        var_lab: str) -> Dict[int, HierarchyResponse]:
        """Stage 3: Organize clusters into 3-level hierarchy
        
        Args:
            labeled_clusters: Labeled clusters (after merging)
            cluster_contents: Cluster content information
            var_lab: Survey question for context
            
        Returns:
            Dictionary mapping cluster ID to HierarchyResponse
        """
        cluster_ids = sorted(labeled_clusters.keys())
        
        # First, group clusters into meta-level categories
        meta_groups = self._create_meta_groups(labeled_clusters, cluster_contents, var_lab)
        
        # Then, create meso-level subcategories within each meta group
        meso_groups = self._create_meso_groups(meta_groups, labeled_clusters, cluster_contents, var_lab)
        
        # Finally, assign micro-level identifiers
        hierarchy_assignments = self._assign_hierarchy_levels(meta_groups, meso_groups, labeled_clusters)
        
        return hierarchy_assignments
    
    def _create_meta_groups(self,
                           labeled_clusters: Dict[int, InitialLabelResponse],
                           cluster_contents: Dict[int, ClusterContent],
                           var_lab: str) -> Dict[str, List[int]]:
        """Create meta-level groupings using LLM
        
        Args:
            labeled_clusters: Labeled clusters
            cluster_contents: Cluster content
            var_lab: Survey question context
            
        Returns:
            Dictionary mapping meta-level ID to list of cluster IDs
        """
        # Create prompt for meta-level grouping
        prompt = self._create_meta_grouping_prompt(labeled_clusters, cluster_contents, var_lab)
        
        # Get meta-level groupings from LLM
        meta_response = self._get_meta_groupings(prompt, list(labeled_clusters.keys()))
        
        # Convert response to dictionary format
        meta_groups = {}
        for i, group in enumerate(meta_response.meta_groups):
            meta_id = str(i + 1)  # Meta levels are "1", "2", "3", etc.
            meta_groups[meta_id] = group.cluster_ids
        
        return meta_groups
    
    def _create_meso_groups(self,
                           meta_groups: Dict[str, List[int]],
                           labeled_clusters: Dict[int, InitialLabelResponse],
                           cluster_contents: Dict[int, ClusterContent],
                           var_lab: str) -> Dict[str, Dict[str, List[int]]]:
        """Create meso-level subcategories within meta groups
        
        Args:
            meta_groups: Meta-level groupings
            labeled_clusters: Labeled clusters
            cluster_contents: Cluster content
            var_lab: Survey question context
            
        Returns:
            Nested dictionary: meta_id -> meso_id -> list of cluster IDs
        """
        meso_groups = {}
        
        for meta_id, cluster_ids in meta_groups.items():
            if len(cluster_ids) <= 3:
                # Small meta groups don't need meso subdivision
                meso_groups[meta_id] = {f"{meta_id}.1": cluster_ids}
            else:
                # Create meso-level subdivisions for larger groups
                prompt = self._create_meso_grouping_prompt(
                    meta_id, cluster_ids, labeled_clusters, cluster_contents, var_lab
                )
                
                meso_response = self._get_meso_groupings(prompt, meta_id, cluster_ids)
                
                # Convert response to dictionary format
                meso_groups[meta_id] = {}
                for i, group in enumerate(meso_response.meso_groups):
                    meso_id = f"{meta_id}.{i + 1}"
                    meso_groups[meta_id][meso_id] = group.cluster_ids
        
        return meso_groups
    
    def _assign_hierarchy_levels(self,
                               meta_groups: Dict[str, List[int]],
                               meso_groups: Dict[str, Dict[str, List[int]]],
                               labeled_clusters: Dict[int, InitialLabelResponse]) -> Dict[int, HierarchyResponse]:
        """Assign final hierarchy levels to all clusters
        
        Args:
            meta_groups: Meta-level groupings
            meso_groups: Meso-level groupings
            labeled_clusters: Labeled clusters
            
        Returns:
            Dictionary mapping cluster ID to HierarchyResponse
        """
        hierarchy_assignments = {}
        
        for meta_id, meta_clusters in meta_groups.items():
            meso_dict = meso_groups.get(meta_id, {})
            
            for meso_id, meso_clusters in meso_dict.items():
                # Assign micro-level IDs within each meso group
                for i, cluster_id in enumerate(sorted(meso_clusters)):
                    micro_id = f"{meso_id}.{i + 1}"
                    
                    hierarchy_assignments[cluster_id] = HierarchyResponse(
                        cluster_id=cluster_id,
                        meta_level=meta_id,
                        meso_level=meso_id,
                        micro_level=micro_id,
                        hierarchy_path=f"{meta_id}/{meso_id}/{micro_id}",
                        level_labels={
                            "meta": f"Meta Category {meta_id}",
                            "meso": f"Subcategory {meso_id}",
                            "micro": labeled_clusters[cluster_id].label
                        }
                    )
        
        return hierarchy_assignments
    
    def _create_meta_grouping_prompt(self,
                                    labeled_clusters: Dict[int, InitialLabelResponse],
                                    cluster_contents: Dict[int, ClusterContent],
                                    var_lab: str) -> str:
        """Create prompt for meta-level grouping
        
        Args:
            labeled_clusters: All labeled clusters
            cluster_contents: Cluster content information
            var_lab: Survey question context
            
        Returns:
            Formatted prompt for meta-level grouping
        """
        prompt = f"""You are organizing survey response clusters for the question: "{var_lab}"

Your task is to group these clusters into high-level meta-categories that represent major themes.
Each meta-category should:
1. Represent a distinct, broad theme related to the survey question
2. Contain clusters that share conceptual similarity
3. Be meaningful and interpretable in the context of the survey

Here are all the clusters to organize:
"""
        
        for cluster_id in sorted(labeled_clusters.keys()):
            label = labeled_clusters[cluster_id]
            content = cluster_contents[cluster_id]
            
            prompt += f"\n\nCluster {cluster_id}:"
            prompt += f"\n- Label: {label.label}"
            prompt += f"\n- Theme: {label.theme_summary}"
            prompt += f"\n- Keywords: {', '.join(label.keywords)}"
            prompt += f"\n- Size: {content.item_count} items"
        
        prompt += """

Group these clusters into 3-7 meta-categories. Each cluster should belong to exactly one meta-category.
Provide:
1. A descriptive name for each meta-category
2. The list of cluster IDs in each category
3. A brief explanation of what unifies the clusters in each category

Consider the survey question context when creating these groupings."""
        
        return prompt
    
    def _get_meta_groupings(self, prompt: str, cluster_ids: List[int]):
        """Get meta-level groupings from LLM
        
        Args:
            prompt: Grouping prompt
            cluster_ids: All cluster IDs to group
            
        Returns:
            Meta grouping response
        """
        try:
            # Define response structure
            class MetaGroup(BaseModel):
                name: str
                cluster_ids: List[int]
                explanation: str
            
            class MetaGroupingResponse(BaseModel):
                meta_groups: List[MetaGroup]
                meta_labels: Dict[str, str]  # Meta ID -> Label
            
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "You are an expert in hierarchical organization and thematic analysis."},
                    {"role": "user", "content": prompt}
                ],
                response_model=MetaGroupingResponse,
                temperature=0.3,
                max_tokens=self.config.max_tokens
            )
            
            # Validate that all clusters are assigned
            assigned_clusters = set()
            for group in response.meta_groups:
                assigned_clusters.update(group.cluster_ids)
            
            missing_clusters = set(cluster_ids) - assigned_clusters
            if missing_clusters:
                # Add missing clusters to a miscellaneous group
                response.meta_groups.append(MetaGroup(
                    name="Other",
                    cluster_ids=list(missing_clusters),
                    explanation="Clusters that don't fit other categories"
                ))
            
            # Create meta labels
            response.meta_labels = {}
            for i, group in enumerate(response.meta_groups):
                meta_id = str(i + 1)
                response.meta_labels[meta_id] = group.name
            
            return response
            
        except Exception as e:
            print(f"Error in meta grouping: {e}")
            # Fallback: create a single meta group
            return type('MetaGroupingResponse', (), {
                'meta_groups': [type('MetaGroup', (), {
                    'name': 'All Clusters',
                    'cluster_ids': cluster_ids,
                    'explanation': 'Fallback grouping'
                })],
                'meta_labels': {'1': 'All Clusters'}
            })()
    
    def _create_meso_grouping_prompt(self,
                                    meta_id: str,
                                    cluster_ids: List[int],
                                    labeled_clusters: Dict[int, InitialLabelResponse],
                                    cluster_contents: Dict[int, ClusterContent],
                                    var_lab: str) -> str:
        """Create prompt for meso-level grouping within a meta category
        
        Args:
            meta_id: Meta-level ID
            cluster_ids: Clusters in this meta group
            labeled_clusters: All labeled clusters
            cluster_contents: Cluster content
            var_lab: Survey question context
            
        Returns:
            Formatted prompt for meso-level grouping
        """
        prompt = f"""You are creating subcategories within a meta-category for the survey question: "{var_lab}"

This is Meta Category {meta_id}. Now create 2-5 subcategories within this meta-category.

Clusters in this meta-category:
"""
        
        for cluster_id in cluster_ids:
            label = labeled_clusters[cluster_id]
            content = cluster_contents[cluster_id]
            
            prompt += f"\n\nCluster {cluster_id}:"
            prompt += f"\n- Label: {label.label}"
            prompt += f"\n- Theme: {label.theme_summary}"
            prompt += f"\n- Keywords: {', '.join(label.keywords)}"
        
        prompt += """

Create logical subcategories that:
1. Group related clusters within this meta-category
2. Represent meaningful distinctions
3. Help organize the responses hierarchically

Provide:
1. A name for each subcategory
2. The cluster IDs in each subcategory
3. Brief explanation of the subcategory"""
        
        return prompt
    
    def _get_meso_groupings(self, prompt: str, meta_id: str, cluster_ids: List[int]):
        """Get meso-level groupings from LLM
        
        Args:
            prompt: Grouping prompt
            meta_id: Parent meta-level ID
            cluster_ids: Clusters to subdivide
            
        Returns:
            Meso grouping response
        """
        try:
            # Define response structure
            class MesoGroup(BaseModel):
                name: str
                cluster_ids: List[int]
                explanation: str
            
            class MesoGroupingResponse(BaseModel):
                meso_groups: List[MesoGroup]
                meso_labels: Dict[str, str]  # Meso ID -> Label
            
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "You are an expert in hierarchical categorization."},
                    {"role": "user", "content": prompt}
                ],
                response_model=MesoGroupingResponse,
                temperature=0.3,
                max_tokens=self.config.max_tokens
            )
            
            # Validate all clusters are assigned
            assigned_clusters = set()
            for group in response.meso_groups:
                assigned_clusters.update(group.cluster_ids)
            
            missing_clusters = set(cluster_ids) - assigned_clusters
            if missing_clusters:
                response.meso_groups.append(MesoGroup(
                    name="Other",
                    cluster_ids=list(missing_clusters),
                    explanation="Remaining clusters"
                ))
            
            # Create meso labels
            response.meso_labels = {}
            for i, group in enumerate(response.meso_groups):
                meso_id = f"{meta_id}.{i + 1}"
                response.meso_labels[meso_id] = group.name
            
            return response
            
        except Exception as e:
            print(f"Error in meso grouping: {e}")
            # Fallback: single meso group
            return type('MesoGroupingResponse', (), {
                'meso_groups': [type('MesoGroup', (), {
                    'name': f'Subgroup {meta_id}.1',
                    'cluster_ids': cluster_ids,
                    'explanation': 'Fallback grouping'
                })],
                'meso_labels': {f'{meta_id}.1': f'Subgroup {meta_id}.1'}
            })()
    
    def get_hierarchy_labels(self,
                           hierarchy_assignments: Dict[int, HierarchyResponse],
                           meta_groupings,
                           meso_groupings) -> HierarchyBatchResponse:
        """Get descriptive labels for all hierarchy levels
        
        Args:
            hierarchy_assignments: Hierarchy assignments for all clusters
            meta_groupings: Meta-level grouping response
            meso_groupings: Meso-level grouping responses
            
        Returns:
            HierarchyBatchResponse with all labels
        """
        hierarchies = list(hierarchy_assignments.values())
        
        # Extract meta and meso labels from grouping responses
        meta_labels = {}
        meso_labels = {}
        
        if hasattr(meta_groupings, 'meta_labels'):
            meta_labels = meta_groupings.meta_labels
        
        for meta_id, meso_response in meso_groupings.items():
            if hasattr(meso_response, 'meso_labels'):
                meso_labels.update(meso_response.meso_labels)
        
        # Update hierarchy responses with proper labels
        for hierarchy in hierarchies:
            if hierarchy.meta_level in meta_labels:
                hierarchy.level_labels["meta"] = meta_labels[hierarchy.meta_level]
            if hierarchy.meso_level in meso_labels:
                hierarchy.level_labels["meso"] = meso_labels[hierarchy.meso_level]
        
        return HierarchyBatchResponse(
            hierarchies=hierarchies,
            meta_labels=meta_labels,
            meso_labels=meso_labels
        )
    
    def refine_labels(self,
                     hierarchy_assignments: Dict[int, HierarchyResponse],
                     labeled_clusters: Dict[int, InitialLabelResponse],
                     var_lab: str) -> Dict[str, RefinedLabelResponse]:
        """Stage 4: Refine labels for mutual exclusivity
        
        Args:
            hierarchy_assignments: Hierarchy assignments from Stage 3
            labeled_clusters: Initial labels from Stage 1
            var_lab: Survey question for context
            
        Returns:
            Dictionary mapping hierarchy path to RefinedLabelResponse
        """
        refined_labels = {}
        
        # Group labels by hierarchy level for refinement
        meta_labels = self._group_labels_by_level(hierarchy_assignments, "meta")
        meso_labels = self._group_labels_by_level(hierarchy_assignments, "meso")
        micro_labels = self._group_labels_by_level(hierarchy_assignments, "micro")
        
        # Refine labels at each level to ensure mutual exclusivity
        with tqdm(total=3, desc="Refining labels at each level") as pbar:
            # Refine meta-level labels
            refined_meta = self._refine_level_labels(meta_labels, "meta", var_lab)
            refined_labels.update(refined_meta)
            pbar.update(1)
            
            # Refine meso-level labels within each meta category
            refined_meso = self._refine_meso_labels(meso_labels, hierarchy_assignments, var_lab)
            refined_labels.update(refined_meso)
            pbar.update(1)
            
            # Refine micro-level labels within each meso category
            refined_micro = self._refine_micro_labels(
                micro_labels, hierarchy_assignments, labeled_clusters, var_lab
            )
            refined_labels.update(refined_micro)
            pbar.update(1)
        
        return refined_labels
    
    def _group_labels_by_level(self,
                              hierarchy_assignments: Dict[int, HierarchyResponse],
                              level: str) -> Dict[str, List[Tuple[str, str]]]:
        """Group labels by hierarchy level
        
        Args:
            hierarchy_assignments: All hierarchy assignments
            level: Level to group by ("meta", "meso", or "micro")
            
        Returns:
            Dictionary mapping parent ID to list of (ID, label) tuples
        """
        grouped_labels = defaultdict(list)
        
        for cluster_id, hierarchy in hierarchy_assignments.items():
            if level == "meta":
                # Group all meta labels together
                grouped_labels["all"].append((
                    hierarchy.meta_level,
                    hierarchy.level_labels.get("meta", f"Meta {hierarchy.meta_level}")
                ))
            elif level == "meso":
                # Group meso labels by their parent meta
                grouped_labels[hierarchy.meta_level].append((
                    hierarchy.meso_level,
                    hierarchy.level_labels.get("meso", f"Meso {hierarchy.meso_level}")
                ))
            elif level == "micro":
                # Group micro labels by their parent meso
                grouped_labels[hierarchy.meso_level].append((
                    hierarchy.micro_level,
                    hierarchy.level_labels.get("micro", f"Micro {hierarchy.micro_level}")
                ))
        
        # Remove duplicates
        for key in grouped_labels:
            grouped_labels[key] = list(set(grouped_labels[key]))
        
        return dict(grouped_labels)
    
    def _refine_level_labels(self,
                           grouped_labels: Dict[str, List[Tuple[str, str]]],
                           level: str,
                           var_lab: str) -> Dict[str, RefinedLabelResponse]:
        """Refine labels at a specific hierarchy level
        
        Args:
            grouped_labels: Labels grouped by parent
            level: Hierarchy level being refined
            var_lab: Survey question context
            
        Returns:
            Dictionary mapping hierarchy path to RefinedLabelResponse
        """
        refined_labels = {}
        
        for parent_id, label_list in grouped_labels.items():
            if len(label_list) <= 1:
                # Single label doesn't need refinement
                for level_id, label in label_list:
                    refined_labels[level_id] = RefinedLabelResponse(
                        hierarchy_level=level,
                        hierarchy_path=level_id,
                        original_label=label,
                        refined_label=label,
                        is_mutually_exclusive=True,
                        differentiation_notes="Single label in category",
                        reasoning="No refinement needed for single label"
                    )
                continue
            
            # Create refinement prompt
            prompt = self._create_refinement_prompt(label_list, level, var_lab, parent_id)
            
            # Get refined labels from LLM
            refined_batch = self._get_refined_labels(prompt, label_list, level)
            
            # Store refined labels
            for refined in refined_batch.refined_labels:
                refined_labels[refined.hierarchy_path] = refined
        
        return refined_labels
    
    def _refine_meso_labels(self,
                           meso_labels: Dict[str, List[Tuple[str, str]]],
                           hierarchy_assignments: Dict[int, HierarchyResponse],
                           var_lab: str) -> Dict[str, RefinedLabelResponse]:
        """Refine meso-level labels within each meta category
        
        Args:
            meso_labels: Meso labels grouped by meta parent
            hierarchy_assignments: Full hierarchy information
            var_lab: Survey question context
            
        Returns:
            Dictionary mapping meso ID to RefinedLabelResponse
        """
        refined_labels = {}
        
        # Get meta labels for context
        meta_labels = {}
        for hierarchy in hierarchy_assignments.values():
            if hierarchy.meta_level not in meta_labels:
                meta_labels[hierarchy.meta_level] = hierarchy.level_labels.get("meta", "")
        
        for meta_id, label_list in meso_labels.items():
            meta_context = meta_labels.get(meta_id, f"Meta Category {meta_id}")
            
            # Create refinement prompt with meta context
            prompt = self._create_meso_refinement_prompt(
                label_list, meta_context, var_lab, meta_id
            )
            
            # Get refined labels
            refined_batch = self._get_refined_labels(prompt, label_list, "meso")
            
            # Store refined labels
            for refined in refined_batch.refined_labels:
                refined_labels[refined.hierarchy_path] = refined
        
        return refined_labels
    
    def _refine_micro_labels(self,
                           micro_labels: Dict[str, List[Tuple[str, str]]],
                           hierarchy_assignments: Dict[int, HierarchyResponse],
                           labeled_clusters: Dict[int, InitialLabelResponse],
                           var_lab: str) -> Dict[str, RefinedLabelResponse]:
        """Refine micro-level labels within each meso category
        
        Args:
            micro_labels: Micro labels grouped by meso parent
            hierarchy_assignments: Full hierarchy information
            labeled_clusters: Original cluster labels with details
            var_lab: Survey question context
            
        Returns:
            Dictionary mapping micro ID to RefinedLabelResponse
        """
        refined_labels = {}
        
        # Get meso labels for context
        meso_labels = {}
        for hierarchy in hierarchy_assignments.values():
            if hierarchy.meso_level not in meso_labels:
                meso_labels[hierarchy.meso_level] = hierarchy.level_labels.get("meso", "")
        
        for meso_id, label_list in micro_labels.items():
            meso_context = meso_labels.get(meso_id, f"Subcategory {meso_id}")
            
            # Create refinement prompt with additional cluster details
            prompt = self._create_micro_refinement_prompt(
                label_list, meso_context, hierarchy_assignments, 
                labeled_clusters, var_lab, meso_id
            )
            
            # Get refined labels
            refined_batch = self._get_refined_labels(prompt, label_list, "micro")
            
            # Store refined labels
            for refined in refined_batch.refined_labels:
                refined_labels[refined.hierarchy_path] = refined
        
        return refined_labels
    
    def _create_refinement_prompt(self,
                                label_list: List[Tuple[str, str]],
                                level: str,
                                var_lab: str,
                                parent_id: str) -> str:
        """Create prompt for label refinement
        
        Args:
            label_list: List of (ID, label) tuples to refine
            level: Hierarchy level
            var_lab: Survey question context
            parent_id: Parent category ID
            
        Returns:
            Formatted refinement prompt
        """
        prompt = f"""You are refining {level}-level labels for survey responses to: "{var_lab}"

Your task is to ensure these labels are mutually exclusive and maximally distinctive.
Current labels that need refinement:
"""
        
        for level_id, label in sorted(label_list):
            prompt += f"\n- {level_id}: {label}"
        
        prompt += f"""

Please refine these labels to:
1. Be mutually exclusive - no overlapping meanings
2. Be clear and distinctive - easy to differentiate
3. Maintain relevance to the survey question
4. Use consistent terminology and style

For each label, provide:
1. The refined label text
2. Whether it's now mutually exclusive (true/false)
3. Notes on how it differs from other labels
4. Reasoning for the refinement

Context: These are {level}-level categories within {parent_id}."""
        
        return prompt
    
    def _create_meso_refinement_prompt(self,
                                     label_list: List[Tuple[str, str]],
                                     meta_context: str,
                                     var_lab: str,
                                     meta_id: str) -> str:
        """Create refinement prompt for meso labels with meta context
        
        Args:
            label_list: Meso labels to refine
            meta_context: Parent meta category label
            var_lab: Survey question
            meta_id: Meta category ID
            
        Returns:
            Formatted prompt
        """
        prompt = f"""You are refining subcategory labels within the meta category "{meta_context}" 
for survey responses to: "{var_lab}"

These subcategories are within Meta Category {meta_id}: {meta_context}

Current subcategory labels:
"""
        
        for level_id, label in sorted(label_list):
            prompt += f"\n- {level_id}: {label}"
        
        prompt += """

Refine these subcategory labels to be mutually exclusive while maintaining their relationship 
to the parent meta category. Each should represent a distinct aspect within the broader theme.

For each label, provide:
1. The refined label text
2. Whether it's mutually exclusive (true/false)
3. How it differs from other subcategories
4. Reasoning for the refinement"""
        
        return prompt
    
    def _create_micro_refinement_prompt(self,
                                      label_list: List[Tuple[str, str]],
                                      meso_context: str,
                                      hierarchy_assignments: Dict[int, HierarchyResponse],
                                      labeled_clusters: Dict[int, InitialLabelResponse],
                                      var_lab: str,
                                      meso_id: str) -> str:
        """Create refinement prompt for micro labels with full context
        
        Args:
            label_list: Micro labels to refine
            meso_context: Parent meso category label
            hierarchy_assignments: Full hierarchy for context
            labeled_clusters: Original cluster details
            var_lab: Survey question
            meso_id: Meso category ID
            
        Returns:
            Formatted prompt
        """
        prompt = f"""You are refining cluster labels within the subcategory "{meso_context}"
for survey responses to: "{var_lab}"

Current cluster labels within {meso_id}: {meso_context}
"""
        
        # Add cluster details for better refinement
        for level_id, label in sorted(label_list):
            prompt += f"\n\n{level_id}: {label}"
            
            # Find the original cluster for this micro level
            for cluster_id, hierarchy in hierarchy_assignments.items():
                if hierarchy.micro_level == level_id:
                    if cluster_id in labeled_clusters:
                        original = labeled_clusters[cluster_id]
                        prompt += f"\n  - Keywords: {', '.join(original.keywords[:3])}"
                        prompt += f"\n  - Theme: {original.theme_summary[:100]}..."
                    break
        
        prompt += """

Refine these cluster labels to be maximally distinctive within their subcategory.
Consider the specific content and keywords when differentiating.

For each label, provide:
1. The refined label text
2. Whether it's mutually exclusive (true/false)
3. Key differentiating features
4. Reasoning for the refinement"""
        
        return prompt
    
    def _get_refined_labels(self,
                          prompt: str,
                          label_list: List[Tuple[str, str]],
                          level: str) -> BatchRefinedLabelResponse:
        """Get refined labels from LLM
        
        Args:
            prompt: Refinement prompt
            label_list: Labels being refined
            level: Hierarchy level
            
        Returns:
            BatchRefinedLabelResponse
        """
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "You are an expert in creating clear, distinctive labels for hierarchical categorization."},
                    {"role": "user", "content": prompt}
                ],
                response_model=BatchRefinedLabelResponse,
                temperature=0.2,  # Lower temperature for consistency
                max_tokens=self.config.max_tokens
            )
            
            # Ensure all labels are covered
            response_paths = {r.hierarchy_path for r in response.refined_labels}
            
            for level_id, original_label in label_list:
                if level_id not in response_paths:
                    response.refined_labels.append(
                        RefinedLabelResponse(
                            hierarchy_level=level,
                            hierarchy_path=level_id,
                            original_label=original_label,
                            refined_label=original_label,
                            is_mutually_exclusive=False,
                            differentiation_notes="Failed to refine",
                            reasoning="Error in refinement process"
                        )
                    )
            
            return response
            
        except Exception as e:
            print(f"Error refining labels: {e}")
            # Return original labels as fallback
            return BatchRefinedLabelResponse(
                refined_labels=[
                    RefinedLabelResponse(
                        hierarchy_level=level,
                        hierarchy_path=level_id,
                        original_label=label,
                        refined_label=label,
                        is_mutually_exclusive=False,
                        differentiation_notes="Refinement failed",
                        reasoning=str(e)
                    )
                    for level_id, label in label_list
                ]
            )
    
    def run_pipeline(self, 
                    cluster_results: List[models.ClusterModel], 
                    var_lab: str) -> List[models.LabelModel]:
        """Execute the complete 4-stage labeling pipeline
        
        Args:
            cluster_results: Cluster results from clustering step
            var_lab: Survey question for context
            
        Returns:
            List of LabelModel objects with hierarchical labels
        """
        start_time = time.time()
        
        print("\n🚀 Starting 4-stage labeling pipeline")
        print(f"Processing {len(cluster_results)} cluster results")
        
        # Stage 1: Initial labeling
        print("\n📝 Stage 1: Initial cluster labeling")
        initial_labels = self.label_initial_clusters(cluster_results, var_lab)
        cluster_contents = self.extract_cluster_content(cluster_results)
        print(f"Labeled {len(initial_labels)} clusters")
        
        # Stage 2: Semantic merging
        print("\n🔄 Stage 2: Semantic merging analysis")
        merge_analysis = self.analyze_semantic_similarity(
            initial_labels, cluster_contents, var_lab
        )
        
        # Apply merging to cluster results
        merged_clusters = self.apply_cluster_merging(cluster_results, merge_analysis.remap_dict)
        
        # Re-extract content and re-label after merging
        merged_contents = self.extract_cluster_content(merged_clusters)
        merged_labels = {}
        
        # Map old labels to new merged cluster IDs
        for old_id, new_id in merge_analysis.remap_dict.items():
            if old_id in initial_labels:
                if new_id not in merged_labels:
                    merged_labels[new_id] = initial_labels[old_id]
                    # Update cluster ID in the label
                    merged_labels[new_id].cluster_id = new_id
        
        print(f"Merged {len(initial_labels)} → {len(merged_labels)} clusters")
        print(f"Merge ratio: {1 - (len(merged_labels) / len(initial_labels)):.2%}")
        
        # Stage 3: Create hierarchy
        print("\n🏗️ Stage 3: Creating hierarchical structure")
        hierarchy_assignments = self.create_hierarchy(merged_labels, merged_contents, var_lab)
        print(f"Created {len(set(h.meta_level for h in hierarchy_assignments.values()))} meta categories")
        print(f"Created {len(set(h.meso_level for h in hierarchy_assignments.values()))} meso categories")
        print(f"Assigned {len(hierarchy_assignments)} micro-level clusters")
        
        # Stage 4: Refine labels
        print("\n✨ Stage 4: Refining labels for mutual exclusivity")
        refined_labels = self.refine_labels(hierarchy_assignments, merged_labels, var_lab)
        print(f"Refined {len(refined_labels)} labels across all levels")
        
        # Convert to LabelModel format
        print("\n📦 Converting to output format")
        label_models = self.to_label_model(
            merged_clusters, 
            hierarchy_assignments, 
            refined_labels,
            merge_analysis
        )
        
        # Calculate summary statistics
        end_time = time.time()
        processing_time = end_time - start_time
        
        summary = LabelingSummary(
            total_initial_clusters=len(initial_labels),
            clusters_after_merge=len(merged_labels),
            merge_ratio=1 - (len(merged_labels) / len(initial_labels)),
            hierarchical_structure=self._get_hierarchy_structure(hierarchy_assignments),
            quality_metrics={
                "avg_confidence": np.mean([label.confidence for label in initial_labels.values()]),
                "mutual_exclusivity_rate": np.mean([
                    label.is_mutually_exclusive 
                    for label in refined_labels.values()
                ])
            },
            processing_time=processing_time
        )
        
        print("\n✅ Pipeline completed successfully")
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Initial clusters: {summary.total_initial_clusters}")
        print(f"Final clusters: {summary.clusters_after_merge}")
        print(f"Merge ratio: {summary.merge_ratio:.2%}")
        
        return label_models
    
    def _get_hierarchy_structure(self, 
                               hierarchy_assignments: Dict[int, HierarchyResponse]) -> Dict[str, Dict[str, List[str]]]:
        """Extract hierarchical structure for summary
        
        Args:
            hierarchy_assignments: Hierarchy assignments
            
        Returns:
            Nested dictionary of meta -> meso -> micro structure
        """
        structure = defaultdict(lambda: defaultdict(list))
        
        for cluster_id, hierarchy in hierarchy_assignments.items():
            structure[hierarchy.meta_level][hierarchy.meso_level].append(hierarchy.micro_level)
        
        # Convert to regular dict
        return {
            meta: dict(meso_dict) 
            for meta, meso_dict in structure.items()
        }
    
    def to_label_model(self,
                      cluster_results: List[models.ClusterModel],
                      hierarchy_assignments: Dict[int, HierarchyResponse],
                      refined_labels: Dict[str, RefinedLabelResponse],
                      merge_analysis: MergeRemapResponse) -> List[models.LabelModel]:
        """Convert results to LabelModel format
        
        Args:
            cluster_results: Merged cluster results
            hierarchy_assignments: Hierarchy assignments
            refined_labels: Refined labels for all levels
            merge_analysis: Merge analysis results
            
        Returns:
            List of LabelModel objects
        """
        label_models = []
        
        # Create mapping from cluster ID to hierarchy
        cluster_to_hierarchy = {}
        for cluster_id, hierarchy in hierarchy_assignments.items():
            cluster_to_hierarchy[cluster_id] = hierarchy
        
        # Process each cluster result
        for result in cluster_results:
            # Create submodels for each segment
            label_submodels = []
            
            for segment in result.response_segment:
                if segment.mirco_cluster is not None:
                    cluster_id = list(segment.mirco_cluster.keys())[0]
                    
                    # Get hierarchy for this cluster
                    if cluster_id in cluster_to_hierarchy:
                        hierarchy = cluster_to_hierarchy[cluster_id]
                        
                        # Get refined labels for each level
                        meta_label = ""
                        meso_label = ""
                        micro_label = ""
                        
                        if hierarchy.meta_level in refined_labels:
                            meta_label = refined_labels[hierarchy.meta_level].refined_label
                        elif hierarchy.meta_level in hierarchy.level_labels:
                            meta_label = hierarchy.level_labels["meta"]
                        
                        if hierarchy.meso_level in refined_labels:
                            meso_label = refined_labels[hierarchy.meso_level].refined_label
                        elif hierarchy.meso_level in hierarchy.level_labels:
                            meso_label = hierarchy.level_labels["meso"]
                        
                        if hierarchy.micro_level in refined_labels:
                            micro_label = refined_labels[hierarchy.micro_level].refined_label
                        elif cluster_id in hierarchy.level_labels:
                            micro_label = hierarchy.level_labels["micro"]
                        
                        # Create label submodel
                        label_submodel = models.LabelSubmodel(
                            segment_id=segment.segment_id,
                            segment_response=segment.segment_response,
                            descriptive_code=segment.descriptive_code,
                            code_description=segment.code_description,
                            code_embedding=segment.code_embedding,
                            description_embedding=segment.description_embedding,
                            meta_cluster=None,  # Not used in new system
                            meso_cluster=None,  # Not used in new system
                            mirco_cluster=segment.mirco_cluster,
                            Theme={hierarchy.meta_level: meta_label},
                            Topic={hierarchy.micro_level: micro_label},
                            Code=None  # Not implemented yet
                        )
                        
                        label_submodels.append(label_submodel)
            
            # Create LabelModel
            if label_submodels:
                label_model = models.LabelModel(
                    respondent_id=result.respondent_id,
                    response=result.response,
                    summary="",  # Note: field is 'summary' not 'response_summary'
                    response_segment=label_submodels
                )
                
                label_models.append(label_model)
        
        return label_models


# Example usage and testing
if __name__ == "__main__":
    """Test the 4-stage labeller with actual cluster data from cache"""
    import sys
    from pathlib import Path
    
    # Add project paths
    sys.path.append(str(Path(__file__).parents[2]))  # Add src directory
    
    from cache_manager import CacheManager
    from cache_config import CacheConfig
    import data_io
    
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
        print("\n=== Running 4-stage labelling pipeline ===")
        labeller = Labeller()
        
        # Run the pipeline
        label_results = labeller.run_pipeline(cluster_results, var_lab)
        
        # Save to cache
        cache_manager.save_to_cache(label_results, filename, 'labels')
        print(f"\nSaved {len(label_results)} label results to cache")
        
        # Print summary of results
        print("\n=== Label Summary ===")
        
        # Count themes and topics
        themes = set()
        topics = set()
        
        for label_model in label_results[:5]:  # Show first 5 for brevity
            print(f"\nRespondent {label_model.respondent_id}:")
            
            for segment in label_model.response_segment[:3]:  # Show first 3 segments
                if segment.Theme:
                    theme_id = list(segment.Theme.keys())[0]
                    theme_label = list(segment.Theme.values())[0]
                    themes.add((theme_id, theme_label))
                    print(f"  Theme: {theme_id} - {theme_label}")
                
                if segment.Topic:
                    topic_id = list(segment.Topic.keys())[0]
                    topic_label = list(segment.Topic.values())[0]
                    topics.add((topic_id, topic_label))
                    print(f"  Topic: {topic_id} - {topic_label}")
                
                print(f"  Segment: {segment.code_description[:50]}...")
        
        print(f"\nTotal unique themes: {len(themes)}")
        print(f"Total unique topics: {len(topics)}")
        
        # Show all themes
        print("\n=== All Themes ===")
        for theme_id, theme_label in sorted(themes):
            print(f"{theme_id}: {theme_label}")
        
    else:
        print("No cached cluster data found.")
        print("Please run the clustering pipeline first:")
        print("  python pipeline.py --force-step clusters")