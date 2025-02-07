import asyncio
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from pydantic import BaseModel, Field, ConfigDict
import instructor
from openai import AsyncOpenAI
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import logging
from tqdm.asyncio import tqdm

# Project imports
import models
from config import OPENAI_API_KEY, DEFAULT_MODEL, DEFAULT_LANGUAGE
from utils import qualityFilter


logger = logging.getLogger(__name__)


# Configuration
class LabellerConfig(BaseModel):
    """Configuration for the Labeller"""
    model: str = DEFAULT_MODEL
    api_key: str = OPENAI_API_KEY
    max_concurrent_requests: int = 10
    batch_size: int = 20
    similarity_threshold: float = 0.95  # Auto-merge threshold
    merge_score_threshold: float = 0.7  # LLM merge threshold
    max_retries: int = 3
    retry_delay: int = 2
    language: str = DEFAULT_LANGUAGE


# Phase 1 Data Models
class InitialLabel(BaseModel):
    """Initial label for a cluster"""
    cluster_id: int
    label: str
    keywords: List[str]
    confidence: float = Field(ge=0.0, le=1.0)
    
    model_config = ConfigDict(from_attributes=True)


class BatchLabelResponse(BaseModel):
    """Batch response for initial labeling"""
    labels: List[InitialLabel]
    
    model_config = ConfigDict(from_attributes=True)


# Phase 2 Data Models
class SimilarityScore(BaseModel):
    """Similarity score between clusters"""
    cluster_id_1: int
    cluster_id_2: int
    score: float = Field(ge=0.0, le=1.0)
    merge_suggested: bool
    reason: str
    
    model_config = ConfigDict(from_attributes=True)


class BatchSimilarityResponse(BaseModel):
    """Batch response for similarity scoring"""
    scores: List[SimilarityScore]
    
    model_config = ConfigDict(from_attributes=True)


class MergeMapping(BaseModel):
    """Mapping of cluster merges"""
    merge_groups: List[List[int]]  # Groups of cluster IDs to merge
    cluster_to_merged: Dict[int, int]  # Original cluster ID → Merged cluster ID
    merge_reasons: Dict[int, str]  # Merged cluster ID → Reason
    
    model_config = ConfigDict(from_attributes=True)


# Phase 3 Data Models
class HierarchyNode(BaseModel):
    """Node in the hierarchical structure"""
    node_id: str  # e.g., "1.2.3"
    level: str  # "theme", "topic", "code"
    label: str
    children: List['HierarchyNode'] = []
    cluster_ids: List[int]  # Original cluster IDs
    
    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)


class HierarchicalStructure(BaseModel):
    """Complete hierarchical structure"""
    themes: List[HierarchyNode]  # Level 1 nodes
    cluster_to_path: Dict[int, str]  # Cluster ID → Path (e.g., "1.2.3")
    
    model_config = ConfigDict(from_attributes=True)


# Phase 4 Data Models
class ThemeSummary(BaseModel):
    """Summary for a theme"""
    theme_id: str
    theme_label: str
    summary: str
    relevance_to_question: str
    
    model_config = ConfigDict(from_attributes=True)


# Internal Data Models
class ClusterData(BaseModel):
    """Internal representation of cluster with extracted data"""
    cluster_id: int
    descriptive_codes: List[str]
    code_descriptions: List[str]
    embeddings: np.ndarray
    centroid: np.ndarray
    size: int
    
    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)


class MergedCluster(BaseModel):
    """Cluster after merging"""
    merged_id: int
    original_ids: List[int]
    label: str
    descriptive_codes: List[str]
    code_descriptions: List[str]
    centroid: np.ndarray
    
    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)


class Labeller:
    """Main labeller class to orchestrate the hierarchical labeling process"""
    
    def __init__(self, config: LabellerConfig = None):
        self.config = config or LabellerConfig()
        self.client = instructor.from_openai(AsyncOpenAI(api_key=self.config.api_key))
        
    def create_hierarchical_labels(self, 
                                 cluster_results: List[models.ClusterModel], 
                                 var_lab: str) -> List[models.LabelModel]:
        """Main method that orchestrates all phases"""
        return asyncio.run(self._create_hierarchical_labels_async(cluster_results, var_lab))
    
    async def _create_hierarchical_labels_async(self, 
                                              cluster_results: List[models.ClusterModel], 
                                              var_lab: str) -> List[models.LabelModel]:
        """Async implementation of the main workflow"""
        logger.info("Starting hierarchical labeling process...")
        
        # Extract cluster data
        cluster_data = self.extract_cluster_data(cluster_results)
        logger.info(f"Extracted data for {len(cluster_data)} clusters")
        
        # Phase 1: Initial labels
        initial_labels = await self.phase1_initial_labels(cluster_data, var_lab)
        logger.info(f"Generated initial labels for {len(initial_labels)} clusters")
        
        # Phase 2: Merge similar clusters
        merge_mapping = await self.phase2_merge_similar(cluster_data, initial_labels, var_lab)
        merged_clusters = self.apply_merging(cluster_data, initial_labels, merge_mapping)
        logger.info(f"Merged to {len(merged_clusters)} clusters")
        
        # Phase 3: Create hierarchy
        hierarchy = await self.phase3_create_hierarchy(merged_clusters, var_lab)
        logger.info("Created hierarchical structure")
        
        # Phase 4: Generate summaries
        summaries = await self.phase4_theme_summaries(hierarchy, var_lab)
        logger.info(f"Generated {len(summaries)} theme summaries")
        
        # Convert to output format
        return self.create_label_models(cluster_results, hierarchy, summaries)
    
    def extract_cluster_data(self, cluster_results: List[models.ClusterModel]) -> Dict[int, ClusterData]:
        """Extract and organize cluster data from model results"""
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
    
    async def phase1_initial_labels(self, 
                                  cluster_data: Dict[int, ClusterData], 
                                  var_lab: str) -> Dict[int, InitialLabel]:
        """Phase 1: Generate initial labels for each cluster"""
        try:
            from phase1_labeller import Phase1Labeller
        except ImportError:
            from .phase1_labeller import Phase1Labeller
        
        phase1 = Phase1Labeller(self.config, self.client)
        return await phase1.label_clusters(cluster_data, var_lab)
    
    async def phase2_merge_similar(self, 
                                 cluster_data: Dict[int, ClusterData],
                                 initial_labels: Dict[int, InitialLabel],
                                 var_lab: str) -> MergeMapping:
        """Phase 2: Merge clusters that are not meaningfully differentiated"""
        try:
            from clusterMerger import ClusterMerger
        except ImportError:
            from .clusterMerger import ClusterMerger
        
        merger = ClusterMerger(self.config, self.client)
        return await merger.merge_similar_clusters(cluster_data, initial_labels, var_lab)
    
    def apply_merging(self,
                     cluster_data: Dict[int, ClusterData],
                     initial_labels: Dict[int, InitialLabel],
                     merge_mapping: MergeMapping) -> Dict[int, MergedCluster]:
        """Apply the merge mapping to create merged clusters"""
        merged_clusters = {}
        next_id = 0
        
        # Process each merge group
        for group in merge_mapping.merge_groups:
            # Collect data from all clusters in the group
            all_codes = []
            all_descriptions = []
            all_embeddings = []
            
            for cluster_id in group:
                cluster = cluster_data[cluster_id]
                all_codes.extend(cluster.descriptive_codes)
                all_descriptions.extend(cluster.code_descriptions)
                all_embeddings.append(cluster.embeddings)
            
            # Combine embeddings and calculate new centroid
            combined_embeddings = np.vstack(all_embeddings)
            new_centroid = np.mean(combined_embeddings, axis=0)
            
            # Use the label from the cluster with highest confidence
            best_cluster_id = max(group, key=lambda cid: initial_labels[cid].confidence)
            label = initial_labels[best_cluster_id].label
            
            merged_clusters[next_id] = MergedCluster(
                merged_id=next_id,
                original_ids=group,
                label=label,
                descriptive_codes=all_codes,
                code_descriptions=all_descriptions,
                centroid=new_centroid
            )
            next_id += 1
        
        return merged_clusters
    
    async def phase3_create_hierarchy(self,
                                    merged_clusters: Dict[int, MergedCluster],
                                    var_lab: str) -> HierarchicalStructure:
        """Phase 3: Create 3-level hierarchical structure"""
        try:
            from phase3_organizer import Phase3Organizer
        except ImportError:
            from .phase3_organizer import Phase3Organizer
        
        phase3 = Phase3Organizer(self.config, self.client)
        return await phase3.create_hierarchy(merged_clusters, var_lab)
    
    async def phase4_theme_summaries(self,
                                   hierarchy: HierarchicalStructure,
                                   var_lab: str) -> List[ThemeSummary]:
        """Phase 4: Generate summaries for each theme"""
        try:
            from phase4_summarizer import Phase4Summarizer
        except ImportError:
            from .phase4_summarizer import Phase4Summarizer
        
        phase4 = Phase4Summarizer(self.config, self.client)
        return await phase4.generate_summaries(hierarchy, var_lab)
    
    def create_label_models(self,
                          cluster_results: List[models.ClusterModel],
                          hierarchy: HierarchicalStructure,
                          summaries: List[ThemeSummary]) -> List[models.LabelModel]:
        """Convert internal representation to output LabelModel format"""
        # Create mapping from theme_id to summary
        summary_map = {s.theme_id: s for s in summaries}
        
        # Process each cluster result
        label_models = []
        
        for result in cluster_results:
            # Create label submodels for each segment
            label_segments = []
            
            for segment in result.response_segment:
                if segment.mirco_cluster is not None:
                    cluster_id = list(segment.mirco_cluster.keys())[0]
                    
                    # Get hierarchy path for this cluster
                    if cluster_id in hierarchy.cluster_to_path:
                        path = hierarchy.cluster_to_path[cluster_id]
                        path_parts = path.split('.')
                        
                        # Find the theme node
                        theme_id = path_parts[0]
                        theme_node = next((t for t in hierarchy.themes if t.node_id == theme_id), None)
                        
                        if theme_node:
                            # Navigate to topic and code
                            topic_node = next((c for c in theme_node.children if c.node_id == '.'.join(path_parts[:2])), None)
                            code_node = None
                            if topic_node:
                                code_node = next((c for c in topic_node.children if c.node_id == path), None)
                            
                            # Create label submodel
                            label_segment = models.LabelSubmodel(
                                segment_id=segment.segment_id,
                                segment_response=segment.segment_response,
                                descriptive_code=segment.descriptive_code,
                                code_description=segment.code_description,
                                code_embedding=segment.code_embedding,
                                description_embedding=segment.description_embedding,
                                mirco_cluster=segment.mirco_cluster,
                                Theme={int(theme_id): theme_node.label} if theme_node else None,
                                Topic={hash('.'.join(path_parts[:2])) % 10000: topic_node.label} if topic_node else None,
                                Code={hash(path) % 10000: code_node.label} if code_node else None
                            )
                            
                            label_segments.append(label_segment)
            
            # Create label model
            if label_segments:
                # Get theme summary if available
                theme_ids = set()
                for segment in label_segments:
                    if segment.Theme:
                        theme_ids.update(segment.Theme.keys())
                
                summary = ""
                for theme_id in theme_ids:
                    if theme_id in summary_map:
                        summary += summary_map[theme_id].summary + "\n"
                
                label_model = models.LabelModel(
                    respondent_id=result.respondent_id,
                    response=result.response,
                    summary=summary.strip(),
                    response_segment=label_segments
                )
                
                label_models.append(label_model)
        
        return label_models


if __name__ == "__main__":
    """Test the labeller with cached cluster data"""
    import sys
    from pathlib import Path
    from collections import defaultdict, Counter
    
    # Add project paths
    sys.path.append(str(Path(__file__).parents[2]))  # Add src directory
    
    from cache_manager import CacheManager
    from config import CacheConfig
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
        print("\n=== Running hierarchical labelling pipeline ===")
        labeller = Labeller()
        
        # Run the pipeline
        label_results = labeller.create_hierarchical_labels(cluster_results, var_lab)
        
        # Save to cache
        cache_key = 'labels_all'
        cache_manager.save_to_cache(label_results, filename, cache_key)
        print(f"\nSaved {len(label_results)} label results to cache with key '{cache_key}'")
        
        # Unpack and analyze the results
        print("\n=== UNPACKING HIERARCHICAL LABELS ===")
        
        # Data structures to collect hierarchy
        themes = {}
        theme_summaries = {}
        theme_topics = defaultdict(lambda: {})
        topic_codes = defaultdict(lambda: defaultdict(list))
        
        # Process each result to extract hierarchy
        for result in label_results:
            # Extract theme summary if available
            if result.summary:
                # Store all summaries we find (they might be theme-specific)
                for segment in result.response_segment:
                    if segment.Theme:
                        theme_id = list(segment.Theme.keys())[0]
                        if theme_id not in theme_summaries:
                            theme_summaries[theme_id] = result.summary
            
            # Extract hierarchical structure
            for segment in result.response_segment:
                if segment.Theme:
                    theme_id, theme_label = list(segment.Theme.items())[0]
                    themes[theme_id] = theme_label
                    
                    if segment.Topic:
                        topic_id, topic_label = list(segment.Topic.items())[0]
                        theme_topics[theme_id][topic_id] = topic_label
                        
                        if segment.Code:
                            code_id, code_label = list(segment.Code.items())[0]
                            topic_codes[theme_id][topic_id].append((code_id, code_label))
        
        # Display the hierarchical structure
        print(f"\nFound {len(themes)} themes:")
        
        for theme_id in sorted(themes.keys()):
            print(f"\n{'='*60}")
            print(f"THEME {theme_id}: {themes[theme_id]}")
            print(f"{'='*60}")
            
            # Theme summary
            if theme_id in theme_summaries:
                print(f"\nSummary:")
                print(f"{theme_summaries[theme_id][:500]}...")
            
            # Topics in this theme
            topics_in_theme = theme_topics[theme_id]
            print(f"\nTopics ({len(topics_in_theme)}):")
            
            for topic_id in sorted(topics_in_theme.keys()):
                topic_label = topics_in_theme[topic_id]
                print(f"\n  TOPIC {topic_id}: {topic_label}")
                
                codes_in_topic = topic_codes[theme_id][topic_id]
                code_counts = Counter(codes_in_topic)
                sorted_code_counts = sorted(code_counts.items())
                
                print(f"  Unique Codes ({len(sorted_code_counts)}):")
                for (code_id, code_label), count in sorted_code_counts:
                    print(f"    CODE {code_id}: {code_label} (#{count})")
        
        # Summary statistics
        print("\n=== SUMMARY STATISTICS ===")
        print(f"Total respondents processed: {len(label_results)}")
        print(f"Total segments processed: {sum(len(r.response_segment) for r in label_results)}")
        print(f"Total themes: {len(themes)}")
        total_topics = sum(len(topics) for topics in theme_topics.values())
        print(f"Total topics: {total_topics}")
        total_codes = len(set([code for theme in topic_codes.values() for topic in theme.values() for code in topic]))
        print(f"Total codes: {total_codes}")
        print(f"Original clusters: {len(unique_clusters)}")
        
    else:
        print("No cached cluster data found.")
        print("Please run the clustering pipeline first:")
        print("  python clusterer.py")