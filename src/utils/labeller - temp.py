import numpy as np
from typing import List, Dict, Tuple, Optional #, Any
from pydantic import BaseModel, Field, ConfigDict
import asyncio
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import json
import hashlib
import datetime

from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

import sys
import os

cwd =  os.getcwd()
base = os.path.basename(cwd)
if base == 'utils':
    import_dir = os.path.abspath(os.path.join(cwd, ".."))
elif base == 'src':
    import_dir = os.path.abspath(os.path.join(cwd))
elif base == 'Coderingstool':
    import_dir = os.path.abspath(os.path.join(cwd, 'src'))
else:
    raise ValueError(f"Unexpected working directory: {cwd}")

if import_dir not in sys.path:
    sys.path.insert(0, import_dir)

from config import OPENAI_API_KEY, DEFAULT_LABELLER_CONFIG, LabellerConfig
from prompts import BATCH_SUMMARY_PROMPT, REDUCE_SUMMARY_PROMPT, HIERARCHICAL_LABELING_PROMPT
import models

# Pydantic models for structured outputs
class SubthemeNode(BaseModel):
    label: str
    micro_clusters: List[int]


class ThemeNode(BaseModel):
    label: str
    subthemes: Dict[str, SubthemeNode]
    direct_clusters: List[int] = Field(default_factory=list)  # Added field

class BatchHierarchy(BaseModel):
    batch_id: str
    hierarchy: Dict[str, ThemeNode]

class UnifiedHierarchy(BaseModel):
    unified_hierarchy: Dict[str, ThemeNode]

class SubthemeStructure(BaseModel):  
    id: str
    label: str  
    description: str
    micro_clusters: List[int]

class ThemeStructure(BaseModel):
    id: str
    label: str  
    description: str
    subthemes: List[SubthemeStructure]  

class HierarchicalStructure(BaseModel):
    themes: List[ThemeStructure]
    
class MicroClusterInfo(BaseModel):
    """Information about a micro-cluster"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    cluster_id: int
    size: int
    representatives: List[Tuple[str, float]]  # (code, similarity_score)
    all_codes: List[str]
    centroid: Optional[np.ndarray] = Field(default=None, exclude=True)

class SystemFingerprintTracker:
    """Track openai system fingerprints to detect backend model changes by openai"""
    
    def __init__(self):
        self.fingerprints = set()
        self.fingerprint_changes = []
    
    def check_fingerprint(self, response):
        """Check if system fingerprint has changed"""
        if hasattr(response, 'system_fingerprint'):
            fingerprint = response.system_fingerprint
            if fingerprint not in self.fingerprints:
                self.fingerprints.add(fingerprint)
                if len(self.fingerprints) > 1:
                    self.fingerprint_changes.append({
                        'timestamp': datetime.now(),
                        'new_fingerprint': fingerprint})
                    return True
        return False
    
    def get_report(self):
        """Get report of fingerprint changes"""
        return {
            'unique_fingerprints': len(self.fingerprints),
            'changes': self.fingerprint_changes}


class HierarchicalLabeller:
    """
    Processes micro-clusters into a 3-level hierarchy using MapReduce approach
    Enhanced with deterministic processing and validation
    """
    
    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        language: str = None,
        var_lab: str = None,
        config: LabellerConfig = None,
        verbose: bool = True,
        cache_manager = None,
        filename: str = None
        ):
   
        self.config = config or DEFAULT_LABELLER_CONFIG 
        self.api_key = api_key or OPENAI_API_KEY
        self.model = model or self.config.model
        self.language = language or self.config.language
        self.var_lab = var_lab or ""
        self.verbose = verbose
        self.cache_manager = cache_manager
        self.filename = filename or "unknown"
        
        # Extract config values
        self.batch_size = self.config.batch_size
        self.top_k_representatives = self.config.top_k_representatives
        self.temperature = 0.0  # Always use 0 for determinism
        self.max_tokens = self.config.max_tokens
        self.max_retries = self.config.max_retries
        self.seed = self.config.seed
        self.use_sequential = self.config.use_sequential_processing
        self.validation_threshold = self.config.validation_threshold

        self.use_seed = True

        self.llm = ChatOpenAI(
                temperature=0.0,   
                model=self.model,
                openai_api_key=self.api_key,
                max_tokens=self.max_tokens,
                seed=self.seed)
        
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
            print(f"Temperature: 0.0 (deterministic), Seed: {self.seed if self.use_seed else 'N/A'}")
            print(f"Sequential processing: {self.use_sequential}")
  
  
    def _build_map_chain(self):
        batch_prompt = PromptTemplate.from_template(BATCH_SUMMARY_PROMPT)
        
        return ({
                "batch_clusters": lambda x: x["batch_clusters"],
                "batch_id": lambda x: x["batch_id"],
                "var_lab": lambda x: self.var_lab,
                "language": lambda x: self.language}
         
            | batch_prompt
            | self.llm
            | self.parser
            | RunnableLambda(lambda x: BatchHierarchy(**x)) )
             

    def _build_reduce_chain(self):
        reduce_prompt = PromptTemplate.from_template(REDUCE_SUMMARY_PROMPT)
        
        return ({
                "summaries": lambda x: x["summaries"],
                "var_lab": lambda x: self.var_lab,
                "language": lambda x: self.language}
            
            | reduce_prompt
            | self.llm
            | self.parser
            | RunnableLambda(lambda x: UnifiedHierarchy(**x)))
   
    def _build_labeling_chain(self):
        labeling_prompt = PromptTemplate.from_template(HIERARCHICAL_LABELING_PROMPT)
        
        return ({
                "final_summary": lambda x: x["final_summary"],
                "micro_cluster_list": lambda x: x["micro_cluster_list"],
                "var_lab": lambda x: self.var_lab,
                "language": lambda x: self.language,
                "all_cluster_ids": lambda x: x["all_cluster_ids"]                 
                }
            
            | labeling_prompt
            | self.llm
            | self.parser
            | RunnableLambda(lambda x: HierarchicalStructure(**x)))
    
     
    def _generate_cache_key(self, cluster_models: List[models.ClusterModel]) -> str:
        """Generate deterministic cache key based on input"""
        
        # Create a deterministic representation of the input
        cluster_data = []
        for model in cluster_models:
            if model.response_segment:
                for segment in model.response_segment:
                    if segment.micro_cluster:
                        cluster_id = list(segment.micro_cluster.keys())[0]
                        cluster_data.append((
                            cluster_id,
                            segment.descriptive_code,
                            segment.code_description
                        ))
        
        # Sort for consistency
        cluster_data.sort()
        
        # Include config parameters in hash
        config_data = {
            'model': self.model,
            'language': self.language,
            'batch_size': self.batch_size,
            'seed': self.seed,
            'var_lab': self.var_lab
        }
        
        # Create hash
        data_str = json.dumps({
            'clusters': cluster_data,
            'config': config_data
        }, sort_keys=True)
        cache_key = hashlib.md5(data_str.encode()).hexdigest()[:12]
        
        return cache_key
    
    def process_hierarchy(self, cluster_models: List[models.ClusterModel]) -> List[models.LabelModel]:
        """Sync wrapper for async processing"""
        return asyncio.run(self.process_hierarchy_async(cluster_models))
    
    async def process_hierarchy_async(self, cluster_models: List[models.ClusterModel]) -> List[models.LabelModel]:
        """Main processing method using MapReduce with caching"""
        if self.verbose:
            print("\n🔄 Starting hierarchical labeling process...")
        
        # Generate cache key based on input
        cache_key = self._generate_cache_key(cluster_models)
        
        # Check if we have cached results
        if self.cache_manager:
            cached_hierarchy = self.cache_manager.load_intermediate_data(
                filename=self.filename,
                cache_key=f"hierarchy_{cache_key}",
                expected_type=dict  # Load as dict then convert
            )
            if cached_hierarchy:
                if self.verbose:
                    print("📦 Using cached hierarchy")
                # Convert dict back to HierarchicalStructure
                hierarchy = HierarchicalStructure(**cached_hierarchy)
                # Apply to responses and return
                return self._apply_hierarchy_to_responses(cluster_models, hierarchy)
        
        # Validate input
        if not cluster_models:
            if self.verbose:
                print("❌ No cluster models provided. Cannot process hierarchy.")
            return []
        
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
            print("\n🔄 Step 3: Reduce phase - Unifying hierarchies...")
        unified_hierarchy = await self._reduce_phase(batch_hierarchies)
        
        # Step 4: Generate final refined hierarchy
        if self.verbose:
            print("\n🏗️ Step 4: Generating final refined hierarchy...")
        hierarchy = await self._generate_hierarchy(unified_hierarchy, micro_clusters)
        
        # Cache final hierarchy
        if self.cache_manager:
            # Convert to dict for caching
            hierarchy_dict = hierarchy.model_dump()
            self.cache_manager.cache_intermediate_data(
                data=hierarchy_dict,
                filename=self.filename,
                cache_key=f"hierarchy_{cache_key}"
            )
        
        # Step 5: Apply hierarchy to original responses
        if self.verbose:
            print("\n✅ Step 5: Applying hierarchy to responses...")
        label_models = self._apply_hierarchy_to_responses(cluster_models, hierarchy)
        
        if self.verbose:
            print("\n✨ Hierarchical labeling complete!")
            self._print_hierarchy_summary(hierarchy)
        
        return label_models
    
    def _create_batches(self, micro_clusters: Dict[int, MicroClusterInfo]) -> List[List[int]]:
        """Create batches with deterministic ordering for consistent results"""
        # Sort cluster IDs to ensure consistent ordering
        sorted_cluster_ids = sorted(micro_clusters.keys())
        
        # Group clusters by size to ensure balanced batches
        clusters_by_size = defaultdict(list)
        for cluster_id in sorted_cluster_ids:
            size = micro_clusters[cluster_id].size
            clusters_by_size[size].append(cluster_id)
        
        # Create batches with size diversity
        batches = []
        current_batch = []
        
        # Flatten sorted by size (largest first for better context)
        all_clusters = []
        for size in sorted(clusters_by_size.keys(), reverse=True):
            all_clusters.extend(sorted(clusters_by_size[size]))
        
        # Create fixed-size batches
        for cluster_id in all_clusters:
            current_batch.append(cluster_id)
            if len(current_batch) >= self.batch_size:
                batches.append(current_batch)
                current_batch = []
        
        # Add remaining clusters
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    async def _extract_micro_clusters_with_representatives(self, cluster_models: List[models.ClusterModel]) -> Dict[int, MicroClusterInfo]:
        """Extract micro-clusters and find representative examples using cosine similarity"""
        
        micro_clusters = defaultdict(lambda: { 'descriptions': [], 'embeddings': [] })
        
        for model in cluster_models:
            if model.response_segment:
                for segment in model.response_segment:
                    if segment.micro_cluster:
                        for cluster_id, cluster_desc in segment.micro_cluster.items():
                            micro_clusters[cluster_id]['descriptions'].append(segment.code_description)
                            if segment.code_embedding is not None:
                                micro_clusters[cluster_id]['embeddings'].append(segment.description_embedding)

        cluster_info = {}
        for cluster_id, data in micro_clusters.items():
            if len(data['embeddings']) > 0:
                embeddings = np.array(data['embeddings'])
                centroid = np.mean(embeddings, axis=0)
                
                # Find representative codes
                representatives = self._get_representatives(
                    embeddings,
                    data['descriptions'],
                    centroid,
                    top_k=min(self.top_k_representatives, len(data['descriptions']))
                    )
                
                cluster_info[cluster_id] = MicroClusterInfo(
                    cluster_id=cluster_id,
                    size=len(data['descriptions']),
                    representatives=representatives,  
                    all_codes=data['descriptions'],
                    centroid=centroid
                )
                
               
        # # debug
        # for info in cluster_info.values():
        #     print(info)
        #     break
        # print()
        
        return cluster_info
    
    def _get_representatives(self, cluster_embeddings: np.ndarray, cluster_descriptions: List[str], centroid: np.ndarray, top_k: int = 3 ) -> List[Tuple[str, float]]:
        """Find most representative codes using centroid cosine similarity"""
   
        similarities = cosine_similarity(cluster_embeddings, centroid.reshape(1, -1)).flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(cluster_descriptions[i], float(similarities[i])) for i in top_indices]
    
    def _format_batch_clusters(self, batch_ids: List[int], micro_clusters: Dict[int, MicroClusterInfo]) -> str:
        """Format micro-clusters with deterministic ordering"""
       
        formatted_clusters = []
        for cluster_id in sorted(batch_ids):
            cluster = micro_clusters[cluster_id]
            cluster_text = f"Micro-cluster {cluster_id:03d} ({cluster.size} items):\n"
            sorted_reps = sorted(cluster.representatives, 
                               key=lambda x: (-x[1], x[0]))  # Sort by similarity desc, then alphabetically
            for i, (code, similarity) in enumerate(sorted_reps):
                cluster_text += f"  {i+1}. {code} (similarity: {similarity:.3f})\n"
            formatted_clusters.append(cluster_text)
        
        return "\n".join(formatted_clusters)
    
    async def _map_phase(self, micro_clusters: Dict[int, MicroClusterInfo]) -> List[BatchHierarchy]:
        """Process micro-clusters in batches to create summaries with deterministic ordering"""
     
        batches = self._create_batches(micro_clusters)
        
        if self.verbose:
            print(f"   Created {len(batches)} batches")
        
        batch_summaries = []
        
        if self.use_sequential:
            # Process batches sequentially for consistency
            for batch_idx, batch_ids in enumerate(batches):
                batch_clusters_text = self._format_batch_clusters(batch_ids, micro_clusters)
                result = await self._invoke_with_retries(
                    self.map_chain, {
                        "batch_clusters": batch_clusters_text,
                        "batch_id": f"batch_{batch_idx:03d}"  # Fixed width for consistent ordering
                        })
                batch_summaries.append(result)
                
                if self.verbose:
                    print(f"   Sequenially processed batch {batch_idx + 1}/{len(batches)}")
        else:
            # Process concurrently  
            tasks = []
            for batch_idx, batch_ids in enumerate(batches):
                batch_clusters_text = self._format_batch_clusters(batch_ids, micro_clusters)
                task = self._invoke_with_retries(
                    self.map_chain,{
                        "batch_clusters": batch_clusters_text,
                        "batch_id": f"batch_{batch_idx:03d}"
                        })
                tasks.append(task)
            
            batch_summaries = await asyncio.gather(*tasks)
            
            if self.verbose:
                print(f"   Concurrently processed batch {batch_idx + 1}/{len(batches)}")
                

        # # debug 
        # for batch in batch_summaries:
        #     print(batch)

        # return batch_summaries
        
        return batch_summaries  
    
    def _format_hierarchies(self, hierarchies: List[BatchHierarchy]) -> str:
        """Format hierarchies for reduction prompt with deterministic ordering"""
        formatted = []
        
        # Sort hierarchies by batch_id for consistency
        sorted_hierarchies = sorted(hierarchies, key=lambda h: h.batch_id)
        
        for hierarchy in sorted_hierarchies:
            formatted.append(f"Hierarchy from {hierarchy.batch_id}:")
            # Convert to dict for JSON serialization with sorted keys
            hierarchy_dict = {}
            for theme_id in sorted(hierarchy.hierarchy.keys()):
                theme_node = hierarchy.hierarchy[theme_id]
                hierarchy_dict[theme_id] = {
                    "label": theme_node.label,
                    "subthemes": {}
                }
                for sub_id in sorted(theme_node.subthemes.keys()):
                    sub_node = theme_node.subthemes[sub_id]
                    hierarchy_dict[theme_id]["subthemes"][sub_id] = {
                        "label": sub_node.label,
                        "micro_clusters": sorted(sub_node.micro_clusters)
                    }
            
            formatted.append(json.dumps(hierarchy_dict, indent=2, sort_keys=True))
            formatted.append("")  # blank line
        
        return "\n".join(formatted)
    
    async def _reduce_phase(self, batch_hierarchies: List[BatchHierarchy]) -> UnifiedHierarchy:
        """Hierarchically reduce hierarchies until final unified hierarchy"""
        
 
        current_hierarchies = batch_hierarchies
        
        while len(current_hierarchies) > 1:
            # Create batches of hierarchies to reduce
            new_hierarchies = []
            
            for i in range(0, len(current_hierarchies), self.batch_size):
                batch = current_hierarchies[i:i + self.batch_size]
                
                # Format hierarchies for prompt
                hierarchies_text = self._format_hierarchies(batch)
                
                # Invoke reduce chain
                result = await self._invoke_with_retries(
                    self.reduce_chain,
                    {"summaries": hierarchies_text}
                )
                
                new_hierarchies.append(result)
            
            current_hierarchies = new_hierarchies
            
            if self.verbose:
                print(f"   Reduced to {len(current_hierarchies)} hierarchies")
        
        final_hierarchy = current_hierarchies[0]
        
        # # debug 
        # for theme_id, theme in final_hierarchy.unified_hierarchy.items():
        #         print(theme.label)
        #         for subtheme in theme.subthemes.values():
        #             print(f"\n{subtheme.label}")
        #             print(f"\n{subtheme.micro_clusters}")
        
        return final_hierarchy

    def _format_micro_cluster_list(self, micro_clusters: Dict[int, MicroClusterInfo]) -> str:
        """Format micro-cluster list with deterministic ordering"""
        formatted = []
        
        # Sort by cluster ID for consistency
        for cluster_id in sorted(micro_clusters.keys()):
            cluster = micro_clusters[cluster_id]
            cluster_summary = f"Cluster {cluster_id:03d} ({cluster.size} items): "
            # Use first representative code as summary
            if cluster.representatives:
                cluster_summary += cluster.representatives[0][0]
            formatted.append(cluster_summary)
        
        return "\n".join(formatted)
    
    async def _generate_hierarchy(self, unified_hierarchy: UnifiedHierarchy, micro_clusters: Dict[int, MicroClusterInfo]) -> HierarchicalStructure:
        """Generate complete 3-level hierarchy with validation"""
        
           
        micro_cluster_list = self._format_micro_cluster_list(micro_clusters)
        hierarchy_dict = {}
        for theme_id in sorted(unified_hierarchy.unified_hierarchy.keys()):
            theme_node = unified_hierarchy.unified_hierarchy[theme_id]
            hierarchy_dict[theme_id] = {
                "label": theme_node.label,
                "subthemes": {}
            }
            for sub_id in sorted(theme_node.subthemes.keys()):
                sub_node = theme_node.subthemes[sub_id]
                hierarchy_dict[theme_id]["subthemes"][sub_id] = {
                    "label": sub_node.label,
                    "micro_clusters": sorted(sub_node.micro_clusters)
                }
        
        hierarchy_json = json.dumps(hierarchy_dict, indent=2, sort_keys=True)
        all_cluster_ids = ", ".join(str(c) for c in sorted(micro_clusters.keys()))
    
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                result = await self._invoke_with_retries(
                    self.labeling_chain, {
                        "final_summary": hierarchy_json,
                        "micro_cluster_list": micro_cluster_list,
                        "all_cluster_ids": all_cluster_ids  # Now this is properly included
                    }
                )
                return result
            except Exception as e:
                if self.verbose:
                    print(f"  Failed on attempt {attempt + 1}: {str(e)}")
                if attempt >= max_attempts - 1:
                    raise
        
        raise RuntimeError("Failed to generate hierarchy after all attempts")
        
        

    

    async def _invoke_with_retries(self, chain, inputs: dict):
        """Invoke chain with retry logic and fingerprint tracking"""
        retries = 0
        
        # Initialize fingerprint tracker if not exists
        if not hasattr(self, 'fingerprint_tracker'):
            self.fingerprint_tracker = SystemFingerprintTracker()
        
        while retries <= self.max_retries:
            try:
                result = await chain.ainvoke(inputs)
                
                # Check for system fingerprint changes
                if hasattr(result, '_raw_response'):
                    if self.fingerprint_tracker.check_fingerprint(result._raw_response):
                        if self.verbose:
                            print("⚠️ Openai descided to change llm model in the backend - results may vary")
                
                return result
            except Exception as e:
                if self.verbose:
                    print(f"   Retry {retries + 1}: Error in chain execution: {str(e)}")
                retries += 1
                await asyncio.sleep(self.retry_delay * retries)
        
        raise RuntimeError(f"Chain invocation failed after {self.max_retries} retries")

   
    def _apply_hierarchy_to_responses(self, cluster_models: List[models.ClusterModel], hierarchy: HierarchicalStructure) -> List[models.LabelModel]:
        """Apply the generated hierarchy to individual responses"""

        # Create mapping from micro-cluster to hierarchy
        micro_to_subtheme = {}
        subtheme_to_theme = {}
        theme_dict = {}
        subtheme_dict = {}
        
        for theme in hierarchy.themes:
            theme_dict[theme.id] = f"{theme.label}: {theme.description}"
            
            for subtheme in theme.subthemes:  
                subtheme_dict[subtheme.id] = f"{subtheme.label}: {subtheme.description}"
                subtheme_to_theme[subtheme.id] = theme.id
                
                for micro_cluster_id in subtheme.micro_clusters:
                    micro_to_subtheme[micro_cluster_id] = subtheme.id
        
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
                        if micro_id in micro_to_subtheme:
                            subtheme_id = micro_to_subtheme[micro_id]
                            theme_id = subtheme_to_theme[subtheme_id]
                            
                            # Set hierarchy
                            theme_id_int = int(theme_id)
                            subtheme_id_float = float(subtheme_id)
                            
                            segment.Theme = {theme_id_int: theme_dict[theme_id]}
                            segment.Topic = {subtheme_id_float: subtheme_dict[subtheme_id]}
                            segment.Keyword = segment.micro_cluster
            
            label_models.append(label_model)
        
        return label_models

    def _print_hierarchy_summary(self, hierarchy: HierarchicalStructure):
        """Print a summary of the generated hierarchy"""
        print("\n📊 Hierarchy Summary:")
        print(f"   - {len(hierarchy.themes)} Themes")
        
        total_topics = 0
        total_clusters = 0
        
        for theme in hierarchy.themes:
            # Change from theme.topics to theme.subthemes
            total_topics += len(theme.subthemes)  
            for topic in theme.subthemes:  # Change here too
                total_clusters += len(topic.micro_clusters)
        
        print(f"   - {total_topics} Topics")
        print(f"   - {total_clusters} Micro-clusters mapped")
        
        print("\n📋 Theme Overview:")
        for theme in hierarchy.themes:
            print(f"   {theme.id}. {theme.label} ({len(theme.subthemes)} topics)")  # And here

# Main Labeller class that wraps HierarchicalLabeller
class Labeller:
    """Main interface for hierarchical labeling"""
    
    def __init__(self, config: LabellerConfig = None, cache_manager=None, filename: str = None):
        self.config = config or LabellerConfig()
        self.cache_manager = cache_manager
        self.filename = filename or "unknown"
        
    def run_pipeline(self, cluster_results: List[models.ClusterModel], var_lab: str) -> List[models.LabelModel]:
        """Run the hierarchical labeling pipeline"""
        # Initialize hierarchical labeller with cache manager
        hierarchical_labeller = HierarchicalLabeller(
            var_lab=var_lab,
            config=self.config,
            verbose=True,
            cache_manager=self.cache_manager,
            filename=self.filename
        )
        
        # Process hierarchy
        return hierarchical_labeller.process_hierarchy(cluster_results)
    
    @staticmethod
    def display_hierarchical_results(
        labeled_results: List[models.LabelModel], 
        cached_theme_summaries: Optional[List] = None,
        total_clusters: int = 0
    ):
        """Display hierarchical labeling results with flexible depth support"""
        if not labeled_results:
            print("No labeled results to display")
            return
        
        # Collect hierarchy structure
        theme_structure = {}
        
        for model in labeled_results:
            if model.response_segment:
                for segment in model.response_segment:
                    if segment.Theme and segment.Topic:
                        theme_id = list(segment.Theme.keys())[0]
                        theme_text = list(segment.Theme.values())[0]
                        topic_id = list(segment.Topic.keys())[0]
                        topic_text = list(segment.Topic.values())[0]
                        
                        if theme_id not in theme_structure:
                            theme_structure[theme_id] = {
                                'text': theme_text,
                                'topics': {},
                                'is_direct': False
                            }
                        
                        # Check if this is a direct assignment (theme == topic)
                        if theme_text == topic_text:
                            theme_structure[theme_id]['is_direct'] = True
                        
                        if topic_id not in theme_structure[theme_id]['topics']:
                            theme_structure[theme_id]['topics'][topic_id] = {
                                'text': topic_text,
                                'count': 0,
                                'examples': []
                            }
                        
                        theme_structure[theme_id]['topics'][topic_id]['count'] += 1
                        
                        # Collect a few examples
                        if len(theme_structure[theme_id]['topics'][topic_id]['examples']) < 3:
                            theme_structure[theme_id]['topics'][topic_id]['examples'].append(
                                segment.code_description
                            )
        
        # Display results
        print("\n" + "="*80)
        print("📊 HIERARCHICAL LABELING RESULTS")
        print("="*80)
        
        if total_clusters > 0:
            print(f"\nProcessed {total_clusters} micro-clusters")
        
        print(f"Generated {len(theme_structure)} themes")
        
        # Display each theme
        for theme_id in sorted(theme_structure.keys()):
            theme = theme_structure[theme_id]
            theme_name = theme['text'].split(':')[0].strip()
            theme_desc = theme['text'].split(':', 1)[1].strip() if ':' in theme['text'] else ''
            
            total_segments = sum(t['count'] for t in theme['topics'].values())
            
            print(f"\n{'='*60}")
            print(f"THEME {theme_id}: {theme_name}")
            if theme_desc:
                print(f"Description: {theme_desc}")
            print(f"Total segments: {total_segments}")
            
            if theme['is_direct']:
                print("Type: Simple theme (direct cluster assignment)")
            else:
                print(f"Topics: {len(theme['topics'])}")
            
            # Display topics (or direct clusters)
            if not theme['is_direct']:
                for topic_id in sorted(theme['topics'].keys()):
                    topic = theme['topics'][topic_id]
                    topic_name = topic['text'].split(':')[0].strip()
                    topic_desc = topic['text'].split(':', 1)[1].strip() if ':' in topic['text'] else ''
                    
                    print(f"\n  📌 TOPIC {topic_id}: {topic_name}")
                    if topic_desc:
                        print(f"     Description: {topic_desc}")
                    print(f"     Segments: {topic['count']}")
                    
                    # Show examples
                    if topic['examples']:
                        print("     Examples:")
                        for i, example in enumerate(topic['examples'], 1):
                            print(f"       {i}. {example}")
            else:
                # For direct themes, just show examples
                all_examples = []
                for topic in theme['topics'].values():
                    all_examples.extend(topic['examples'])
                
                if all_examples:
                    print("\nExamples:")
                    for i, example in enumerate(all_examples[:5], 1):
                        print(f"  {i}. {example}")
        
        print("\n" + "="*80)
        

# =============================================================================
# TEST/USAGE SECTION
# =============================================================================

if __name__ == "__main__":
    """Test the hierarchical labeller with cached data"""
    
    import sys
    from cache_manager import CacheManager
    from config import CacheConfig
    import models
    
    S