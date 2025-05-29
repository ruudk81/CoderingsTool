"""
Hierarchical Labeller using MapReduce approach with LangChain
Processes micro-clusters into a 3-level hierarchy: Themes, Topics, Keywords
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from pydantic import BaseModel, Field, ConfigDict
import asyncio
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import json

from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Dynamic import path
import sys
from pathlib import Path
import os
try:
    src_dir = str(Path(__file__).resolve().parents[2])
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
except (NameError, AttributeError):
    current_dir = os.getcwd()
    if current_dir.endswith('utils'):
        src_dir = str(Path(current_dir).parents[1])
    else:
        src_dir = os.path.abspath('src')
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

# Import project modules
from config import (
    OPENAI_API_KEY, DEFAULT_MODEL, DEFAULT_LANGUAGE,
    LabellerConfig, DEFAULT_LABELLER_CONFIG
)
from prompts import (
    BATCH_SUMMARY_PROMPT, 
    REDUCE_SUMMARY_PROMPT, 
    HIERARCHICAL_LABELING_PROMPT
)
import models


# Pydantic models for structured outputs
class SubthemeNode(BaseModel):
    node: str
    micro_clusters: List[int]

class ThemeNode(BaseModel):
    node: str
    subthemes: Dict[str, SubthemeNode]

class BatchHierarchy(BaseModel):
    batch_id: str
    hierarchy: Dict[str, ThemeNode]

class UnifiedHierarchy(BaseModel):
    unified_hierarchy: Dict[str, ThemeNode]

class TopicStructure(BaseModel):
    id: str
    name: str
    description: str
    micro_clusters: List[int]

class ThemeStructure(BaseModel):
    id: str
    name: str
    description: str
    topics: List[TopicStructure]

class HierarchicalStructure(BaseModel):
    themes: List[ThemeStructure]


class MicroClusterInfo(BaseModel):
    """Information about a micro-cluster"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    cluster_id: int
    size: int
    representative_codes: List[Tuple[str, float]]  # (code, similarity_score)
    all_codes: List[str]
    centroid: Optional[np.ndarray] = Field(default=None, exclude=True)


class HierarchicalLabeller:
    """
    Processes micro-clusters into a 3-level hierarchy using MapReduce approach
    """
    
    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        language: str = None,
        var_lab: str = None,
        config: LabellerConfig = None,
        verbose: bool = True
    ):
        self.config = config or DEFAULT_LABELLER_CONFIG
        self.api_key = api_key or OPENAI_API_KEY
        self.model = model or self.config.model
        self.language = language or self.config.language
        self.var_lab = var_lab or ""
        self.verbose = verbose
        
        # Extract config values
        self.batch_size = self.config.batch_size
        self.top_k_representatives = self.config.top_k_representatives
        self.temperature = self.config.temperature
        self.max_tokens = self.config.max_tokens
        self.max_retries = self.config.max_retries
        
        # Initialize LangChain components
        self.llm = ChatOpenAI(
            temperature=self.temperature,
            model=self.model,
            openai_api_key=self.api_key,
            max_tokens=self.max_tokens
        )
        
        self.parser = JsonOutputParser()
        
        # Build chains
        self.map_chain = self._build_map_chain()
        self.reduce_chain = self._build_reduce_chain()
        self.labeling_chain = self._build_labeling_chain()
        
        # Retry settings
        self.retry_delay = 2
        
        if self.verbose:
            print(f"Initialized HierarchicalLabeller with model: {self.model}")
            print(f"Language: {self.language}, Batch size: {self.batch_size}")
    
    def _build_map_chain(self):
        """Build LangChain for batch summarization"""
        batch_prompt = PromptTemplate.from_template(BATCH_SUMMARY_PROMPT)
        
        return (
            {
                "batch_clusters": lambda x: x["batch_clusters"],
                "batch_id": lambda x: x["batch_id"],
                "var_lab": lambda x: self.var_lab,
                "language": lambda x: self.language
            }
            | batch_prompt
            | self.llm
            | self.parser
            | RunnableLambda(lambda x: BatchHierarchy(**x))
        )
    
    def _build_reduce_chain(self):
        """Build LangChain for summary reduction"""
        reduce_prompt = PromptTemplate.from_template(REDUCE_SUMMARY_PROMPT)
        
        return (
            {
                "summaries": lambda x: x["summaries"],
                "var_lab": lambda x: self.var_lab,
                "language": lambda x: self.language
            }
            | reduce_prompt
            | self.llm
            | self.parser
            | RunnableLambda(lambda x: UnifiedHierarchy(**x))
        )
    
    def _build_labeling_chain(self):
        """Build LangChain for final hierarchy generation"""
        labeling_prompt = PromptTemplate.from_template(HIERARCHICAL_LABELING_PROMPT)
        
        return (
            {
                "final_summary": lambda x: x["final_summary"],
                "micro_cluster_list": lambda x: x["micro_cluster_list"],
                "var_lab": lambda x: self.var_lab,
                "language": lambda x: self.language
            }
            | labeling_prompt
            | self.llm
            | self.parser
            | RunnableLambda(lambda x: HierarchicalStructure(**x))
        )
    
    def process_hierarchy(self, cluster_models: List[models.ClusterModel]) -> List[models.LabelModel]:
        """Sync wrapper for async processing"""
        return asyncio.run(self.process_hierarchy_async(cluster_models))
    
    async def process_hierarchy_async(self, cluster_models: List[models.ClusterModel]) -> List[models.LabelModel]:
        """Main processing method using MapReduce"""
        if self.verbose:
            print("\n🔄 Starting hierarchical labeling process...")
        
        # Step 1: Extract micro-clusters with representatives
        if self.verbose:
            print("📊 Step 1: Extracting micro-clusters with representatives...")
        micro_clusters = await self._extract_micro_clusters_with_representatives(cluster_models)
        
        if self.verbose:
            print(f"   Found {len(micro_clusters)} unique micro-clusters")
        
        # Step 2: Map phase - create batch hierarchies
        if self.verbose:
            print(f"\n🗂️ Step 2: Map phase - Creating batch hierarchies (batch size: {self.batch_size})...")
        batch_hierarchies = await self._map_phase(micro_clusters)
        
        # Step 3: Reduce phase - unify hierarchies
        if self.verbose:
            print(f"\n🔄 Step 3: Reduce phase - Unifying hierarchies...")
        unified_hierarchy = await self._reduce_phase(batch_hierarchies)
        
        # Step 4: Generate final refined hierarchy
        if self.verbose:
            print(f"\n🏗️ Step 4: Generating final refined hierarchy...")
        hierarchy = await self._generate_hierarchy(unified_hierarchy, micro_clusters)
        
        # Step 5: Apply hierarchy to original responses
        if self.verbose:
            print(f"\n✅ Step 5: Applying hierarchy to responses...")
        label_models = self._apply_hierarchy_to_responses(cluster_models, hierarchy)
        
        if self.verbose:
            print(f"\n✨ Hierarchical labeling complete!")
            self._print_hierarchy_summary(hierarchy)
        
        return label_models
    
    async def _extract_micro_clusters_with_representatives(
        self, 
        cluster_models: List[models.ClusterModel]
    ) -> Dict[int, MicroClusterInfo]:
        """Extract micro-clusters and find representative examples using cosine similarity"""
        micro_clusters = defaultdict(lambda: {
            'codes': [],
            'descriptions': [],
            'embeddings': []
        })
        
        # Collect all segments by micro-cluster
        for model in cluster_models:
            if model.response_segment:
                for segment in model.response_segment:
                    if segment.micro_cluster:
                        for cluster_id, cluster_desc in segment.micro_cluster.items():
                            micro_clusters[cluster_id]['codes'].append(segment.descriptive_code)
                            micro_clusters[cluster_id]['descriptions'].append(segment.code_description)
                            if segment.code_embedding is not None:
                                micro_clusters[cluster_id]['embeddings'].append(segment.code_embedding)
        
        # Process each cluster to find representatives
        cluster_info = {}
        for cluster_id, data in micro_clusters.items():
            if len(data['embeddings']) > 0:
                embeddings = np.array(data['embeddings'])
                centroid = np.mean(embeddings, axis=0)
                
                # Find representative codes
                representatives = self._get_representative_codes(
                    embeddings,
                    data['codes'],
                    centroid,
                    top_k=min(self.top_k_representatives, len(data['codes']))
                )
                
                cluster_info[cluster_id] = MicroClusterInfo(
                    cluster_id=cluster_id,
                    size=len(data['codes']),
                    representative_codes=representatives,
                    all_codes=data['codes'],
                    centroid=centroid
                )
        
        return cluster_info
    
    def _get_representative_codes(
        self,
        cluster_embeddings: np.ndarray,
        cluster_codes: List[str],
        centroid: np.ndarray,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """Find most representative codes using centroid cosine similarity"""
        # Calculate similarities
        similarities = cosine_similarity(cluster_embeddings, centroid.reshape(1, -1)).flatten()
        
        # Get top k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return codes with their similarity scores
        return [(cluster_codes[i], float(similarities[i])) for i in top_indices]
    
    async def _map_phase(self, micro_clusters: Dict[int, MicroClusterInfo]) -> List[BatchHierarchy]:
        """Process micro-clusters in batches to create summaries"""
        # Create batches
        cluster_ids = list(micro_clusters.keys())
        batches = []
        
        for i in range(0, len(cluster_ids), self.batch_size):
            batch_ids = cluster_ids[i:i + self.batch_size]
            batches.append(batch_ids)
        
        if self.verbose:
            print(f"   Created {len(batches)} batches")
        
        # Process batches
        batch_summaries = []
        for batch_idx, batch_ids in enumerate(batches):
            # Prepare batch data
            batch_clusters_text = self._format_batch_clusters(batch_ids, micro_clusters)
            
            # Invoke chain with retries
            result = await self._invoke_with_retries(
                self.map_chain,
                {
                    "batch_clusters": batch_clusters_text,
                    "batch_id": f"batch_{batch_idx}"
                }
            )
            
            batch_summaries.append(result)
            
            if self.verbose:
                print(f"   Processed batch {batch_idx + 1}/{len(batches)}")
        
        return batch_summaries
    
    def _format_batch_clusters(self, batch_ids: List[int], micro_clusters: Dict[int, MicroClusterInfo]) -> str:
        """Format micro-clusters for prompt"""
        formatted_clusters = []
        
        for cluster_id in batch_ids:
            cluster = micro_clusters[cluster_id]
            cluster_text = f"Micro-cluster {cluster_id} ({cluster.size} items):\n"
            
            for code, similarity in cluster.representative_codes:
                cluster_text += f"  - {code} (similarity: {similarity:.3f})\n"
            
            formatted_clusters.append(cluster_text)
        
        return "\n".join(formatted_clusters)
    
    async def _reduce_phase(self, batch_hierarchies: List[BatchHierarchy]) -> UnifiedHierarchy:
        """Hierarchically reduce hierarchies until final unified hierarchy"""
        current_hierarchies = batch_hierarchies
        
        while len(current_hierarchies) > 1:
            # Create batches of hierarchies to reduce
            new_hierarchies = []
            
            for i in range(0, len(current_hierarchies), self.batch_size):
                batch = current_hierarchies[i:i + self.batch_size]
                
                # Format hierarchies for prompt
                hierarchies_text = self._format_hierarchies_for_reduction(batch)
                
                # Invoke reduce chain
                result = await self._invoke_with_retries(
                    self.reduce_chain,
                    {"summaries": hierarchies_text}
                )
                
                new_hierarchies.append(result)
            
            current_hierarchies = new_hierarchies
            
            if self.verbose:
                print(f"   Reduced to {len(current_hierarchies)} hierarchies")
        
        # Return the final unified hierarchy
        return current_hierarchies[0]
    
    def _format_hierarchies_for_reduction(self, hierarchies: List[BatchHierarchy]) -> str:
        """Format hierarchies for reduction prompt"""
        formatted = []
        
        for idx, hierarchy in enumerate(hierarchies):
            formatted.append(f"Hierarchy from batch {hierarchy.batch_id}:")
            # Convert to dict for JSON serialization
            hierarchy_dict = {}
            for theme_id, theme_node in hierarchy.hierarchy.items():
                hierarchy_dict[theme_id] = {
                    "node": theme_node.node,
                    "subthemes": {
                        sub_id: {
                            "node": sub_node.node,
                            "micro_clusters": sub_node.micro_clusters
                        }
                        for sub_id, sub_node in theme_node.subthemes.items()
                    }
                }
            formatted.append(json.dumps(hierarchy_dict, indent=2))
            formatted.append("")  # blank line
        
        return "\n".join(formatted)
    
    async def _generate_hierarchy(
        self, 
        unified_hierarchy: UnifiedHierarchy,
        micro_clusters: Dict[int, MicroClusterInfo]
    ) -> HierarchicalStructure:
        """Generate complete 3-level hierarchy based on unified hierarchy"""
        # Format micro-cluster list
        micro_cluster_list = self._format_micro_cluster_list(micro_clusters)
        
        # Format unified hierarchy for final labeling
        hierarchy_dict = {}
        for theme_id, theme_node in unified_hierarchy.unified_hierarchy.items():
            hierarchy_dict[theme_id] = {
                "node": theme_node.node,
                "subthemes": {
                    sub_id: {
                        "node": sub_node.node,
                        "micro_clusters": sub_node.micro_clusters
                    }
                    for sub_id, sub_node in theme_node.subthemes.items()
                }
            }
        hierarchy_json = json.dumps(hierarchy_dict, indent=2)
        
        # Invoke labeling chain
        hierarchy = await self._invoke_with_retries(
            self.labeling_chain,
            {
                "final_summary": hierarchy_json,
                "micro_cluster_list": micro_cluster_list
            }
        )
        
        return hierarchy
    
    def _format_micro_cluster_list(self, micro_clusters: Dict[int, MicroClusterInfo]) -> str:
        """Format micro-cluster list for hierarchy generation"""
        formatted = []
        
        for cluster_id, cluster in micro_clusters.items():
            cluster_summary = f"Cluster {cluster_id} ({cluster.size} items): "
            # Use first representative code as summary
            if cluster.representative_codes:
                cluster_summary += cluster.representative_codes[0][0]
            formatted.append(cluster_summary)
        
        return "\n".join(formatted)
    
    def _apply_hierarchy_to_responses(
        self, 
        cluster_models: List[models.ClusterModel],
        hierarchy: HierarchicalStructure
    ) -> List[models.LabelModel]:
        """Apply the generated hierarchy to individual responses"""
        # Create mapping from micro-cluster to hierarchy
        micro_to_topic = {}
        topic_to_theme = {}
        theme_dict = {}
        topic_dict = {}
        
        for theme in hierarchy.themes:
            theme_dict[theme.id] = f"{theme.name}: {theme.description}"
            
            for topic in theme.topics:
                topic_dict[topic.id] = f"{topic.name}: {topic.description}"
                topic_to_theme[topic.id] = theme.id
                
                for micro_cluster_id in topic.micro_clusters:
                    micro_to_topic[micro_cluster_id] = topic.id
        
        # Apply to each response
        label_models = []
        
        for cluster_model in cluster_models:
            # Convert to LabelModel
            label_model = cluster_model.to_model(models.LabelModel)
            
            # Apply hierarchy to segments
            if label_model.response_segment:
                for segment in label_model.response_segment:
                    if segment.micro_cluster:
                        # Get micro cluster id
                        micro_id = list(segment.micro_cluster.keys())[0]
                        
                        # Map to hierarchy
                        if micro_id in micro_to_topic:
                            topic_id = micro_to_topic[micro_id]
                            theme_id = topic_to_theme[topic_id]
                            
                            # Set hierarchy
                            segment.Theme = {int(theme_id): theme_dict[theme_id]}
                            segment.Topic = {int(float(topic_id)): topic_dict[topic_id]}
                            segment.Keyword = segment.micro_cluster
            
            label_models.append(label_model)
        
        return label_models
    
    async def _invoke_with_retries(self, chain, inputs: dict):
        """Invoke chain with retry logic"""
        retries = 0
        
        while retries <= self.max_retries:
            try:
                result = await chain.ainvoke(inputs)
                return result
            except Exception as e:
                if self.verbose:
                    print(f"   Retry {retries + 1}: Error in chain execution: {str(e)}")
                retries += 1
                await asyncio.sleep(self.retry_delay * retries)
        
        raise RuntimeError(f"Chain invocation failed after {self.max_retries} retries")
    
    def _print_hierarchy_summary(self, hierarchy: HierarchicalStructure):
        """Print a summary of the generated hierarchy"""
        print(f"\n📊 Hierarchy Summary:")
        print(f"   - {len(hierarchy.themes)} Themes")
        
        total_topics = 0
        total_clusters = 0
        
        for theme in hierarchy.themes:
            total_topics += len(theme.topics)
            for topic in theme.topics:
                total_clusters += len(topic.micro_clusters)
        
        print(f"   - {total_topics} Topics")
        print(f"   - {total_clusters} Micro-clusters mapped")
        
        print(f"\n📋 Theme Overview:")
        for theme in hierarchy.themes:
            print(f"   {theme.id}. {theme.name} ({len(theme.topics)} topics)")


# =============================================================================
# TEST/USAGE SECTION
# =============================================================================

if __name__ == "__main__":
    """Test the hierarchical labeller with cached data"""
    
    # Add parent directory to path for imports
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    
    from cache_manager import CacheManager
    from config import CacheConfig
    import models
    
    # Initialize cache manager
    cache_config = CacheConfig()
    cache_manager = CacheManager(cache_config)
    
    # Test parameters
    filename = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
    var_name = "Q20"
    
    print(f"Loading cluster data for {var_name} from {filename}...")
    
    # Load cluster models from cache (output of step 5)
    cluster_models = cache_manager.load_from_cache(filename, "cluster", models.ClusterModel)
    
    if cluster_models:
        print(f"Loaded {len(cluster_models)} cluster models")
        
        # Initialize labeller
        labeller = HierarchicalLabeller(
            var_lab=var_name,
            verbose=True
        )
        
        # Process hierarchy
        print("\nProcessing hierarchical labels...")
        label_models = labeller.process_hierarchy(cluster_models)
        
        # Print sample results
        if label_models:
            print(f"\n✅ Generated {len(label_models)} label models")
            
            # Show first few labeled segments
            print("\n📝 Sample labeled segments:")
            sample_count = 0
            for model in label_models[:5]:  # First 5 responses
                if model.response_segment:
                    for segment in model.response_segment:
                        if segment.Theme and segment.Topic:
                            sample_count += 1
                            print(f"\nSegment: {segment.code_description}")
                            print(f"Theme: {list(segment.Theme.values())[0]}")
                            print(f"Topic: {list(segment.Topic.values())[0]}")
                            print(f"Keyword: {list(segment.Keyword.values())[0] if segment.Keyword else 'N/A'}")
                            
                            if sample_count >= 5:
                                break
                if sample_count >= 5:
                    break
            
            # Save results to cache
            print("\n💾 Saving label models to cache...")
            success = cache_manager.save_to_cache(
                label_models,
                filename,
                "label"
            )
            
            if success:
                print("✅ Label models saved successfully!")
            else:
                print("❌ Failed to save label models")
    else:
        print("❌ No cluster models found in cache. Please run steps 1-5 first.")