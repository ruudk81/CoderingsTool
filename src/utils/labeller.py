"""
Hierarchical Labeller using MapReduce approach with LangChain
Processes micro-clusters into a 3-level hierarchy: Themes, Topics, Keywords
Enhanced with deterministic processing and validation
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
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

from config import OPENAI_API_KEY, LabellerConfig, DEFAULT_LABELLER_CONFIG
from prompts import BATCH_SUMMARY_PROMPT, REDUCE_SUMMARY_PROMPT, HIERARCHICAL_LABELING_PROMPT
import models

# Pydantic models for structured outputs
class SubthemeNode(BaseModel):
    node: str
    micro_clusters: List[int]

class ThemeNode(BaseModel):
    node: str
    subthemes: Dict[str, SubthemeNode]
    direct_clusters: List[int] = Field(default_factory=list)  # Added field

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
    direct_clusters: List[int] = Field(default_factory=list)  # Added field - THIS IS THE MISSING ONE!

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


class SystemFingerprintTracker:
    """Track system fingerprints to detect backend changes"""
    
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
                        'new_fingerprint': fingerprint
                    })
                    return True
        return False
    
    def get_report(self):
        """Get report of fingerprint changes"""
        return {
            'unique_fingerprints': len(self.fingerprints),
            'changes': self.fingerprint_changes
        }


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
       
    def _clean_json_output(self, text: str) -> str:
        """Clean LLM output to ensure valid JSON"""
        import re
        
        # Remove comments (both // and /* */ style)
        text = re.sub(r'//.*?(?=\n|$)', '', text)
        text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
        
        # Remove any text before the first { or [
        match = re.search(r'[\{\[]', text)
        if match:
            text = text[match.start():]
        
        # Remove any text after the last } or ]
        for i in range(len(text)-1, -1, -1):
            if text[i] in '}]':
                text = text[:i+1]
                break
        
        # Remove any checkbox patterns
        text = re.sub(r'□|☑️|✓|✗|❌|✅', '', text)
        
        # Remove any trailing commas before closing braces/brackets
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)
        
        return text.strip()
    
    def _build_map_chain(self):
        """Build LangChain for batch summarization with JSON cleaning"""
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
            | RunnableLambda(lambda x: self._clean_json_output(x.content) if hasattr(x, 'content') else x)
            | self.parser
            | RunnableLambda(lambda x: BatchHierarchy(**x))
        )

    def _build_reduce_chain(self):
        """Build LangChain for summary reduction with JSON cleaning"""
        reduce_prompt = PromptTemplate.from_template(REDUCE_SUMMARY_PROMPT)
        
        return (
            {
                "summaries": lambda x: x["summaries"],
                "var_lab": lambda x: self.var_lab,
                "language": lambda x: self.language
            }
            | reduce_prompt
            | self.llm
            | RunnableLambda(lambda x: self._clean_json_output(x.content) if hasattr(x, 'content') else x)
            | self.parser
            | RunnableLambda(lambda x: UnifiedHierarchy(**x))
        )
   
    def _build_labeling_chain(self):
        """Build LangChain for final hierarchy generation with JSON cleaning"""
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
            | RunnableLambda(lambda x: self._clean_json_output(x.content) if hasattr(x, 'content') else x)
            | self.parser
            | RunnableLambda(lambda x: HierarchicalStructure(**x))
        )
    
    
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
    
    def _create_deterministic_batches(self, micro_clusters: Dict[int, MicroClusterInfo]) -> List[List[int]]:
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
    
    def _format_batch_clusters_deterministic(self, batch_ids: List[int], micro_clusters: Dict[int, MicroClusterInfo]) -> str:
        """Format micro-clusters with deterministic ordering"""
        formatted_clusters = []
        
        # Sort batch IDs to ensure consistent formatting
        for cluster_id in sorted(batch_ids):
            cluster = micro_clusters[cluster_id]
            cluster_text = f"Micro-cluster {cluster_id:03d} ({cluster.size} items):\n"
            
            # Sort representative codes by similarity for consistency
            sorted_reps = sorted(cluster.representative_codes, 
                               key=lambda x: (-x[1], x[0]))  # Sort by similarity desc, then alphabetically
            
            for i, (code, similarity) in enumerate(sorted_reps):
                cluster_text += f"  {i+1}. {code} (similarity: {similarity:.3f})\n"
            
            formatted_clusters.append(cluster_text)
        
        return "\n".join(formatted_clusters)
    
    async def _map_phase(self, micro_clusters: Dict[int, MicroClusterInfo]) -> List[BatchHierarchy]:
        """Process micro-clusters in batches to create summaries with deterministic ordering"""
        # Use deterministic batching
        batches = self._create_deterministic_batches(micro_clusters)
        
        if self.verbose:
            print(f"   Created {len(batches)} deterministic batches")
        
        batch_summaries = []
        
        if self.use_sequential:
            # Process batches sequentially for consistency
            for batch_idx, batch_ids in enumerate(batches):
                # Prepare batch data with consistent formatting
                batch_clusters_text = self._format_batch_clusters_deterministic(batch_ids, micro_clusters)
                
                # Invoke chain with retries
                result = await self._invoke_with_retries(
                    self.map_chain,
                    {
                        "batch_clusters": batch_clusters_text,
                        "batch_id": f"batch_{batch_idx:03d}"  # Fixed width for consistent ordering
                    }
                )
                
                batch_summaries.append(result)
                
                if self.verbose:
                    print(f"   Processed batch {batch_idx + 1}/{len(batches)}")
        else:
            # Process concurrently (original behavior)
            tasks = []
            for batch_idx, batch_ids in enumerate(batches):
                batch_clusters_text = self._format_batch_clusters_deterministic(batch_ids, micro_clusters)
                task = self._invoke_with_retries(
                    self.map_chain,
                    {
                        "batch_clusters": batch_clusters_text,
                        "batch_id": f"batch_{batch_idx:03d}"
                    }
                )
                tasks.append(task)
            
            batch_summaries = await asyncio.gather(*tasks)
        
        return batch_summaries
    
    def _format_hierarchies_for_reduction(self, hierarchies: List[BatchHierarchy]) -> str:
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
                    "node": theme_node.node,
                    "subthemes": {}
                }
                for sub_id in sorted(theme_node.subthemes.keys()):
                    sub_node = theme_node.subthemes[sub_id]
                    hierarchy_dict[theme_id]["subthemes"][sub_id] = {
                        "node": sub_node.node,
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
        
        # Clean the final unified hierarchy
        final_hierarchy = current_hierarchies[0]
        cleaned_hierarchy = self._clean_unified_hierarchy(final_hierarchy)
        
        return cleaned_hierarchy

    
    def _format_micro_cluster_list_deterministic(self, micro_clusters: Dict[int, MicroClusterInfo]) -> str:
        """Format micro-cluster list with deterministic ordering"""
        formatted = []
        
        # Sort by cluster ID for consistency
        for cluster_id in sorted(micro_clusters.keys()):
            cluster = micro_clusters[cluster_id]
            cluster_summary = f"Cluster {cluster_id:03d} ({cluster.size} items): "
            # Use first representative code as summary
            if cluster.representative_codes:
                cluster_summary += cluster.representative_codes[0][0]
            formatted.append(cluster_summary)
        
        return "\n".join(formatted)
    
    def _validate_hierarchy(self, hierarchy: HierarchicalStructure, micro_clusters: Dict[int, MicroClusterInfo]) -> bool:
        """Validate hierarchy with flexible depth support"""
        # Track all assigned micro-clusters
        assigned_clusters = {}
        issues = []
        
        # Check each theme
        for theme in hierarchy.themes:
            if not theme.id or not theme.name:
                issues.append(f"Theme missing id or name: {theme}")
            
            # Check topic assignments
            for topic in theme.topics:
                if not topic.id or not topic.name:
                    issues.append(f"Topic missing id or name: {topic}")
                    
                for cluster_id in topic.micro_clusters:
                    if cluster_id not in assigned_clusters:
                        assigned_clusters[cluster_id] = []
                    assigned_clusters[cluster_id].append((theme.id, topic.id))
            
            # Check direct assignments
            if hasattr(theme, 'direct_clusters') and theme.direct_clusters:
                for cluster_id in theme.direct_clusters:
                    if cluster_id not in assigned_clusters:
                        assigned_clusters[cluster_id] = []
                    assigned_clusters[cluster_id].append((theme.id, None))
        
        # Check for duplicates
        duplicates = {cid: locs for cid, locs in assigned_clusters.items() if len(locs) > 1}
        for cluster_id, locations in duplicates.items():
            loc_strs = []
            for theme_id, topic_id in locations:
                if topic_id:
                    loc_strs.append(f"{theme_id}.{topic_id}")
                else:
                    loc_strs.append(f"{theme_id} (direct)")
            issues.append(f"Micro-cluster {cluster_id} assigned to multiple locations: {', '.join(loc_strs)}")
        
        # Check coverage
        all_cluster_ids = set(micro_clusters.keys())
        assigned_ids = set(assigned_clusters.keys())
        unassigned = all_cluster_ids - assigned_ids
        
        if unassigned:
            issues.append(f"Unassigned micro-clusters: {sorted(unassigned)}")
        
        # Calculate coverage
        coverage = len(assigned_ids - set(duplicates.keys())) / len(all_cluster_ids) if all_cluster_ids else 0
        
        if issues:
            if self.verbose:
                print("❌ Hierarchy validation failed:")
                for issue in issues:
                    print(f"   - {issue}")
                print(f"   Coverage: {coverage:.1%}")
            return False
        
        if coverage < self.validation_threshold:
            if self.verbose:
                print(f"⚠️ Coverage below threshold: {coverage:.1%} < {self.validation_threshold:.1%}")
            return False
        
        if self.verbose:
            print(f"✅ Hierarchy validation passed (coverage: {coverage:.1%})")
        return True
    
    
    def _repair_hierarchy(self, hierarchy: HierarchicalStructure, micro_clusters: Dict[int, MicroClusterInfo]) -> HierarchicalStructure:
        """Intelligently repair hierarchy issues with support for direct assignments"""
        
        # Step 1: Remove ALL duplicates first
        cluster_locations = {}  # cluster_id -> (theme, topic_or_none)
        
        for theme in hierarchy.themes:
            # Check topic clusters
            for topic in theme.topics:
                new_clusters = []
                for cluster_id in topic.micro_clusters:
                    if cluster_id not in cluster_locations:
                        cluster_locations[cluster_id] = (theme, topic)
                        new_clusters.append(cluster_id)
                    else:
                        if self.verbose:
                            prev_theme, prev_topic = cluster_locations[cluster_id]
                            prev_name = prev_topic.name if prev_topic else f"{prev_theme.name} (direct)"
                            print(f"   Removed duplicate cluster {cluster_id} from {topic.name} (keeping in {prev_name})")
                topic.micro_clusters = new_clusters
            
            # Check direct clusters
            if hasattr(theme, 'direct_clusters'):
                new_direct = []
                for cluster_id in theme.direct_clusters:
                    if cluster_id not in cluster_locations:
                        cluster_locations[cluster_id] = (theme, None)
                        new_direct.append(cluster_id)
                    else:
                        if self.verbose:
                            prev_theme, prev_topic = cluster_locations[cluster_id]
                            prev_name = prev_topic.name if prev_topic else f"{prev_theme.name} (direct)"
                            print(f"   Removed duplicate cluster {cluster_id} from {theme.name} direct (keeping in {prev_name})")
                theme.direct_clusters = new_direct
        
        # Step 2: Find unassigned clusters
        all_cluster_ids = set(micro_clusters.keys())
        assigned_ids = set(cluster_locations.keys())
        unassigned = sorted(all_cluster_ids - assigned_ids)
        
        if unassigned and hierarchy.themes:
            if self.verbose:
                print(f"   Found {len(unassigned)} unassigned clusters: {unassigned}")
            
            # Step 3: Smart distribution
            # Prefer assigning to direct clusters for simple themes
            theme_complexities = []
            for theme in hierarchy.themes:
                complexity = {
                    'theme': theme,
                    'has_topics': len(theme.topics) > 0,
                    'topic_count': len(theme.topics),
                    'direct_count': len(theme.direct_clusters) if hasattr(theme, 'direct_clusters') else 0,
                    'total_clusters': sum(len(t.micro_clusters) for t in theme.topics) + 
                                    (len(theme.direct_clusters) if hasattr(theme, 'direct_clusters') else 0)
                }
                theme_complexities.append(complexity)
            
            # Sort by total clusters (ascending) to balance distribution
            theme_complexities.sort(key=lambda x: x['total_clusters'])
            
            # Distribute unassigned clusters
            for i, cluster_id in enumerate(unassigned):
                target = theme_complexities[i % len(theme_complexities)]
                theme = target['theme']
                
                # If theme has no topics, add to direct clusters
                if not target['has_topics']:
                    if not hasattr(theme, 'direct_clusters'):
                        theme.direct_clusters = []
                    theme.direct_clusters.append(cluster_id)
                    target['direct_count'] += 1
                    target['total_clusters'] += 1
                    if self.verbose:
                        print(f"   Added cluster {cluster_id} directly to theme {theme.name}")
                else:
                    # Add to smallest topic
                    smallest_topic = min(theme.topics, key=lambda t: len(t.micro_clusters))
                    smallest_topic.micro_clusters.append(cluster_id)
                    target['total_clusters'] += 1
                    if self.verbose:
                        print(f"   Added cluster {cluster_id} to topic {smallest_topic.name} in theme {theme.name}")
                
                # Re-sort to maintain balance
                theme_complexities.sort(key=lambda x: x['total_clusters'])
        
        return hierarchy
 
    
    def _clean_unified_hierarchy(self, unified: UnifiedHierarchy) -> UnifiedHierarchy:
        """Clean duplicates from unified hierarchy before final generation"""
        seen_clusters = set()
        
        for theme_id, theme in unified.unified_hierarchy.items():
            for sub_id, subtheme in theme.subthemes.items():
                cleaned_clusters = []
                for cluster_id in subtheme.micro_clusters:
                    if cluster_id not in seen_clusters:
                        cleaned_clusters.append(cluster_id)
                        seen_clusters.add(cluster_id)
                    else:
                        if self.verbose:
                            print(f"   Cleaned duplicate cluster {cluster_id} from unified hierarchy")
                subtheme.micro_clusters = cleaned_clusters
        
        return unified
  
    def _pre_validate_llm_output(self, hierarchy: HierarchicalStructure, micro_clusters: Dict[int, MicroClusterInfo]) -> Dict[str, Any]:
        """Quick validation of LLM output to provide feedback"""
        all_cluster_ids = set(micro_clusters.keys())
        assigned_clusters = set()
        duplicates = defaultdict(list)
        
        # Collect all assignments
        for theme in hierarchy.themes:
            # From topics
            for topic in theme.topics:
                for cluster_id in topic.micro_clusters:
                    if cluster_id in assigned_clusters:
                        duplicates[cluster_id].append(f"{theme.id}.{topic.id}")
                    else:
                        assigned_clusters.add(cluster_id)
            
            # From direct assignments
            if hasattr(theme, 'direct_clusters'):
                for cluster_id in theme.direct_clusters:
                    if cluster_id in assigned_clusters:
                        duplicates[cluster_id].append(f"{theme.id} (direct)")
                    else:
                        assigned_clusters.add(cluster_id)
        
        missing = all_cluster_ids - assigned_clusters
        coverage = len(assigned_clusters) / len(all_cluster_ids) if all_cluster_ids else 0
        
        return {
            'total_expected': len(all_cluster_ids),
            'total_assigned': len(assigned_clusters),
            'duplicates': dict(duplicates),
            'missing': sorted(missing),
            'coverage': coverage,
            'is_valid': len(duplicates) == 0 and len(missing) == 0
        }
   
    async def _generate_hierarchy(
        self, 
        unified_hierarchy: UnifiedHierarchy,
        micro_clusters: Dict[int, MicroClusterInfo]
        ) -> HierarchicalStructure:
        """Generate complete 3-level hierarchy with validation"""
        # Format micro-cluster list deterministically
        micro_cluster_list = self._format_micro_cluster_list_deterministic(micro_clusters)
        
        # Format unified hierarchy deterministically
        hierarchy_dict = {}
        for theme_id in sorted(unified_hierarchy.unified_hierarchy.keys()):
            theme_node = unified_hierarchy.unified_hierarchy[theme_id]
            hierarchy_dict[theme_id] = {
                "node": theme_node.node,
                "subthemes": {}
            }
            for sub_id in sorted(theme_node.subthemes.keys()):
                sub_node = theme_node.subthemes[sub_id]
                hierarchy_dict[theme_id]["subthemes"][sub_id] = {
                    "node": sub_node.node,
                    "micro_clusters": sorted(sub_node.micro_clusters)
                }
        
        hierarchy_json = json.dumps(hierarchy_dict, indent=2, sort_keys=True)
        
        # Add total cluster count to prompt
        total_clusters = len(micro_clusters)
        
        # Update prompt to include total count
        prompt_with_count = HIERARCHICAL_LABELING_PROMPT.replace(
            "{total_clusters}", str(total_clusters)
        )
        
        # Invoke labeling chain with retry logic
        max_attempts = 3
        for attempt in range(max_attempts):
            hierarchy = await self._invoke_with_retries(
                self.labeling_chain,
                {
                    "final_summary": hierarchy_json,
                    "micro_cluster_list": micro_cluster_list,
                    "total_clusters": total_clusters  # Pass this if your prompt template needs it
                }
            )
            
            # Pre-validate
            validation = self._pre_validate_llm_output(hierarchy, micro_clusters)
            
            if self.verbose and not validation['is_valid']:
                print(f"\n⚠️ LLM output issues on attempt {attempt + 1}:")
                print(f"   Coverage: {validation['coverage']:.1%}")
                print(f"   Duplicates: {len(validation['duplicates'])}")
                print(f"   Missing: {len(validation['missing'])}")
            
            # Full validation
            if self._validate_hierarchy(hierarchy, micro_clusters):
                return hierarchy
            
            if self.verbose:
                print(f"   Hierarchy validation failed on attempt {attempt + 1}, retrying...")
            
            # Try to repair if this is not the last attempt
            if attempt < max_attempts - 1:
                hierarchy = self._repair_hierarchy(hierarchy, micro_clusters)
                if self._validate_hierarchy(hierarchy, micro_clusters):
                    return hierarchy
        
        # Final repair attempt
        if self.verbose:
            print("   Final repair attempt...")
        hierarchy = self._repair_hierarchy(hierarchy, micro_clusters)
        return hierarchy
    
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
                            print("⚠️ System fingerprint changed - results may vary")
                
                return result
            except Exception as e:
                if self.verbose:
                    print(f"   Retry {retries + 1}: Error in chain execution: {str(e)}")
                retries += 1
                await asyncio.sleep(self.retry_delay * retries)
        
        raise RuntimeError(f"Chain invocation failed after {self.max_retries} retries")

   
    def _apply_hierarchy_to_responses(
        self, 
        cluster_models: List[models.ClusterModel],
        hierarchy: HierarchicalStructure
        ) -> List[models.LabelModel]:
        """Apply the generated hierarchy to individual responses - handles flexible depth"""
        # Create mapping from micro-cluster to hierarchy
        micro_to_topic = {}
        micro_to_theme = {}  # For direct theme assignments
        topic_to_theme = {}
        theme_dict = {}
        topic_dict = {}
        
        # Debug: Print the hierarchy structure
        if self.verbose:
            print("\n🔍 DEBUG: Hierarchy structure from LLM:")
            for theme in hierarchy.themes:
                has_topics = len(theme.topics) > 0
                has_direct = hasattr(theme, 'direct_clusters') and len(theme.direct_clusters) > 0
                
                if has_topics:
                    print(f"Theme {theme.id}: {theme.name} - {len(theme.topics)} topics")
                    for topic in theme.topics:
                        print(f"  Topic {topic.id}: {topic.name} - clusters: {sorted(topic.micro_clusters)}")
                
                if has_direct:
                    print(f"Theme {theme.id}: {theme.name} - direct clusters: {sorted(theme.direct_clusters)}")
        
        for theme in hierarchy.themes:
            theme_dict[theme.id] = f"{theme.name}: {theme.description}"
            
            # Handle topics if they exist
            for topic in theme.topics:
                topic_dict[topic.id] = f"{topic.name}: {topic.description}"
                topic_to_theme[topic.id] = theme.id
                
                for micro_cluster_id in topic.micro_clusters:
                    micro_to_topic[micro_cluster_id] = topic.id
            
            # Handle direct cluster assignments (for simple themes)
            if hasattr(theme, 'direct_clusters') and theme.direct_clusters:
                for micro_cluster_id in theme.direct_clusters:
                    micro_to_theme[micro_cluster_id] = theme.id
        
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
                        
                        # Check if it's a direct theme assignment
                        if micro_id in micro_to_theme:
                            theme_id = micro_to_theme[micro_id]
                            theme_id_int = int(theme_id)
                            
                            # Direct theme assignment - use theme as topic (Option 2)
                            segment.Theme = {theme_id_int: theme_dict[theme_id]}
                            segment.Topic = {theme_id_int: theme_dict[theme_id]}  # Same as theme
                            segment.Keyword = segment.micro_cluster
                        
                        # Or regular topic assignment
                        elif micro_id in micro_to_topic:
                            topic_id = micro_to_topic[micro_id]
                            theme_id = topic_to_theme[topic_id]
                            
                            # Set hierarchy
                            theme_id_int = int(theme_id)
                            topic_id_float = float(topic_id)
                            
                            segment.Theme = {theme_id_int: theme_dict[theme_id]}
                            segment.Topic = {topic_id_float: topic_dict[topic_id]}
                            segment.Keyword = segment.micro_cluster
            
            label_models.append(label_model)
        
        return label_models

    def _print_hierarchy_summary(self, hierarchy: HierarchicalStructure):
        """Print a summary of the generated hierarchy"""
        print("\n📊 Hierarchy Summary:")
        print(f"   - {len(hierarchy.themes)} Themes")
        
        total_topics = 0
        total_clusters = 0
        direct_theme_count = 0
        
        for theme in hierarchy.themes:
            total_topics += len(theme.topics)
            for topic in theme.topics:
                total_clusters += len(topic.micro_clusters)
            
            # Count direct clusters
            if hasattr(theme, 'direct_clusters') and theme.direct_clusters:
                total_clusters += len(theme.direct_clusters)
                if not theme.topics:  # Theme with only direct clusters
                    direct_theme_count += 1
        
        print(f"   - {total_topics} Topics")
        print(f"   - {direct_theme_count} Simple themes (direct assignment)")
        print(f"   - {total_clusters} Micro-clusters mapped")
        
        print("\n📋 Theme Overview:")
        for theme in sorted(hierarchy.themes, key=lambda t: t.id):
            topics_str = f"{len(theme.topics)} topics" if theme.topics else "direct clusters only"
            print(f"   {theme.id}. {theme.name} ({topics_str})")


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
    
    # Add parent directory to path for imports
    import sys
    from pathlib import Path
    
    from cache_manager import CacheManager
    from config import CacheConfig
    import models
    
    import nest_asyncio
    nest_asyncio.apply()
    
    
    # Initialize cache manager
    cache_config = CacheConfig()
    cache_manager = CacheManager(cache_config)
    
    # Test parameters
    filename = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
    var_name = "Q20"
    step_name = "labels"
    
    from utils.data_io import DataLoader
    data_loader = DataLoader()
    var_lab = data_loader.get_varlab(filename=filename, var_name=var_name)
    
    print(f"Loading cluster data for {var_name} from {filename}...")
    
    # Load cluster models from cache (output of step 5)
    cluster_results = cache_manager.load_from_cache(filename, "clusters", models.ClusterModel)
    
    if cluster_results is not None and len(cluster_results) > 0:
        print(f"Loaded {len(cluster_results)} cluster models")
        
        intermediate_dir = cache_manager.config.cache_dir / "intermediate"
        for file in intermediate_dir.glob(f"{filename}_hierarchy_*.pkl"):
            file.unlink()
            print(f"Deleted: {file.name}")

                
        labeller_config = LabellerConfig(
            temperature=0.0,  # Force deterministic
            seed=42,  # Fixed seed
            batch_size=8,  # Consistent batch size
            max_retries=3,
            use_sequential_processing=True,  # Process batches sequentially
            validation_threshold=0.95  # Expect 95% coverage
            )
        
        label_generator = Labeller(
            config=labeller_config,
            cache_manager=cache_manager,
            filename=filename
            )
        
        # Initialize labeller
        labeled_results = label_generator.run_pipeline(cluster_results, var_lab)
        
        validation_passed = True
        total_segments = 0
        labeled_segments = 0
        missing_labels = []
        
        for result in labeled_results:
            if result.response_segment:
                for segment in result.response_segment:
                    total_segments += 1
                    if segment.Theme and segment.Topic:
                        labeled_segments += 1
                    else:
                        missing_labels.append({
                            'respondent_id': result.respondent_id,
                            'segment': segment.segment_response[:50] + '...' if len(segment.segment_response) > 50 else segment.segment_response
                        })
   
    # Count unique clusters for display
    unique_clusters = set()
    if cluster_results:
        for result in cluster_results:
            for segment in result.response_segment:
                if segment.micro_cluster:
                    cluster_id = list(segment.micro_cluster.keys())[0]
                    unique_clusters.add(cluster_id)

    # Display results using the labeller's display function
    Labeller.display_hierarchical_results(labeled_results, None, len(unique_clusters))
        