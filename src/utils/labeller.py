import asyncio
import numpy as np
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field, ConfigDict
import instructor
from openai import AsyncOpenAI
from collections import defaultdict
import logging
from tqdm.asyncio import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# Project imports
import models
from config import OPENAI_API_KEY, DEFAULT_MODEL, DEFAULT_LANGUAGE
from prompts import INITIAL_LABEL_PROMPT, HIERARCHY_CREATION_PROMPT, HIERARCHICAL_THEME_SUMMARY_PROMPT

logger = logging.getLogger(__name__)

# Models for structured data
class LabellerConfig(BaseModel):
    """Configuration for the Labeller"""
    model: str = DEFAULT_MODEL
    api_key: str = OPENAI_API_KEY
    max_concurrent_requests: int = 10   
    batch_size: int = 5  # Reduced from 50 for better concurrency
    max_retries: int = 3
    retry_delay: int = 2
    language: str = DEFAULT_LANGUAGE

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

class ThemeSummary(BaseModel):
    """Summary for a theme"""
    theme_id: str
    theme_label: str
    summary: str
    relevance_to_question: str
    
    model_config = ConfigDict(from_attributes=True)

class ClusterData(BaseModel):
    """Internal representation of cluster with extracted data"""
    cluster_id: int
    descriptive_codes: List[str]
    code_descriptions: List[str]
    embeddings: np.ndarray
    centroid: np.ndarray
    size: int
    # Cached computations for performance
    _representative_items: Optional[List[Dict[str, str]]] = None
    model_config = ConfigDict(from_attributes=True, arbitrary_types_allowed=True)

class Labeller:
    """Main labeller class to orchestrate the hierarchical labeling process"""
    
    def __init__(self, config: LabellerConfig = None):
        self.config = config or LabellerConfig()
        self.client = instructor.from_openai(AsyncOpenAI(api_key=self.config.api_key))
        
    def create_hierarchical_labels(self, 
                                 cluster_results: List[models.ClusterModel], 
                                 var_lab: str) -> Tuple[List[models.LabelModel], List[ThemeSummary]]:
        """Main method that orchestrates all phases"""
        return asyncio.run(self._create_hierarchical_labels_async(cluster_results, var_lab))
    
    def run_pipeline(self, 
                     cluster_results: List[models.ClusterModel], 
                     var_lab: str) -> List[models.LabelModel]:
        """Pipeline-compatible method name"""
        labels, _ = self.create_hierarchical_labels(cluster_results, var_lab)
        return labels
    
    # ===== NESTED PHASE CLASSES =====
    
    class Phase1Labeller:
        """Phase 1: Initial label generation for clusters"""
        
        def __init__(self, config: LabellerConfig, client):
            self.config = config
            self.client = client
            self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        
        async def label_clusters(self, 
                               cluster_data: Dict[int, ClusterData], 
                               var_lab: str) -> Dict[int, InitialLabel]:
            """Generate initial labels for all clusters"""
            logger.info("Phase 1: Generating initial cluster labels...")
            
            # Create batches
            cluster_ids = list(cluster_data.keys())
            batches = self._create_batches(cluster_ids, self.config.batch_size)
            
            # Process batches concurrently
            tasks = []
            for batch_ids in batches:
                batch_data = [cluster_data[cid] for cid in batch_ids]
                task = self._label_batch(batch_data, var_lab, batch_ids)
                tasks.append(task)
            
            # Execute all tasks with progress bar
            results = {}
            with tqdm(total=len(batches), desc="Labeling clusters") as pbar:
                for coro in asyncio.as_completed(tasks):
                    batch_result = await coro
                    results.update(batch_result)
                    pbar.update(1)
            
            return results
        
        async def _label_batch(self,
                              batch_data: List[ClusterData],
                              var_lab: str,
                              batch_ids: List[int]) -> Dict[int, InitialLabel]:
            """Label a batch of clusters"""
            async with self.semaphore:
                try:
                    # Get representative items for each cluster
                    batch_representatives = []
                    for cluster in batch_data:
                        representatives = self._get_representative_items(cluster, n=5)
                        batch_representatives.append(representatives)
                    
                    # Create prompt
                    prompt = self._create_batch_prompt(batch_data, batch_representatives, var_lab)
                    
                    # Get labels from LLM
                    response = await self._get_llm_response(prompt)
                    
                    # Convert to dict
                    result = {}
                    for label in response.labels:
                        if label.cluster_id in batch_ids:
                            result[label.cluster_id] = label
                    
                    # Handle missing labels
                    for cid in batch_ids:
                        if cid not in result:
                            result[cid] = InitialLabel(
                                cluster_id=cid,
                                label=f"Cluster {cid}",
                                keywords=["unlabeled"],
                                confidence=0.0
                            )
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Error labeling batch: {e}")
                    # Return fallback labels
                    return {
                        cid: InitialLabel(
                            cluster_id=cid,
                            label=f"Cluster {cid}",
                            keywords=["error"],
                            confidence=0.0
                        )
                        for cid in batch_ids
                    }
        
        def _get_representative_items(self, cluster: ClusterData, n: int = 5) -> List[Dict[str, str]]:
            """Get most representative items using cosine similarity to centroid"""
            # Return cached items if available
            if cluster._representative_items and len(cluster._representative_items) >= n:
                return cluster._representative_items[:n]
            
            # Calculate similarities to centroid
            similarities = cosine_similarity([cluster.centroid], cluster.embeddings)[0]
            
            # Get top n indices
            top_indices = np.argsort(similarities)[-n:][::-1]
            
            # Create representative items
            representatives = []
            for idx in top_indices:
                if idx < len(cluster.descriptive_codes):
                    representatives.append({
                        'code': cluster.descriptive_codes[idx],
                        'description': cluster.code_descriptions[idx],
                        'similarity': float(similarities[idx])
                    })
            
            # Cache the results
            cluster._representative_items = representatives
            
            return representatives
        
        def _create_batch_prompt(self,
                               batch_data: List[ClusterData],
                               batch_representatives: List[List[Dict]],
                               var_lab: str) -> str:
            """Create prompt for batch labeling"""
            # Start with the template
            prompt = INITIAL_LABEL_PROMPT.replace("{var_lab}", var_lab)
            prompt = prompt.replace("{language}", self.config.language)
            
            # Add cluster information
            clusters_info = []
            for cluster, representatives in zip(batch_data, batch_representatives):
                cluster_text = f"\nCluster {cluster.cluster_id}:"
                cluster_text += f"\n- Size: {cluster.size} responses"
                cluster_text += "\n- Most representative items:"
                
                for i, item in enumerate(representatives, 1):
                    cluster_text += f"\n  {i}. Code: {item['code']}"
                    cluster_text += f"\n     Description: {item['description']}"
                    cluster_text += f"\n     Similarity: {item['similarity']:.3f}"
                
                clusters_info.append(cluster_text)
            
            prompt = prompt.replace("{clusters}", "\n".join(clusters_info))
            
            return prompt
        
        async def _get_llm_response(self, prompt: str) -> BatchLabelResponse:
            """Get response from LLM with retry logic"""
            messages = [
                {"role": "system", "content": "You are an expert in thematic analysis and survey response clustering."},
                {"role": "user", "content": prompt}
            ]
            
            for attempt in range(self.config.max_retries):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.config.model,
                        messages=messages,
                        response_model=BatchLabelResponse,
                        temperature=0.3,
                        max_tokens=4000
                    )
                    return response
                except Exception as e:
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                    else:
                        raise e
        
        def _create_batches(self, items: List[int], batch_size: int) -> List[List[int]]:
            """Create batches from a list of items"""
            return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    
    class Phase2Organizer:
        """Phase 2: Create hierarchical organization of clusters"""
        
        def __init__(self, config: LabellerConfig, client):
            self.config = config
            self.client = client
        
        async def create_hierarchy(self,
                                 cluster_data: Dict[int, ClusterData],
                                 initial_labels: Dict[int, InitialLabel],
                                 var_lab: str) -> HierarchicalStructure:
            """Create 3-level hierarchical structure"""
            logger.info("Phase 2: Creating hierarchical structure...")
            
            # Create themes (level 1)
            themes = await self._create_themes(cluster_data, initial_labels, var_lab)
            logger.info(f"Created {len(themes)} themes")
            
            # Create topics within themes (level 2) - parallelized
            topic_tasks = []
            for theme in themes:
                task = self._create_topics(theme, cluster_data, initial_labels, var_lab)
                topic_tasks.append((theme, task))
            
            # Execute all topic creation tasks in parallel
            with tqdm(total=len(topic_tasks), desc="Creating topics") as pbar:
                # Properly await all tasks in parallel
                topic_results = await asyncio.gather(*[task for _, task in topic_tasks])
                
                # Assign results to themes
                for theme, topics in zip([theme for theme, _ in topic_tasks], topic_results):
                    theme.children = topics
                    pbar.update(1)
                    logger.debug(f"Created {len(topics)} topics for theme {theme.node_id}")
            
            # Assign codes within topics (level 3)
            cluster_to_path = {}
            for theme in themes:
                for topic in theme.children:
                    codes = self._create_codes(topic, cluster_data, initial_labels)
                    topic.children = codes
                    
                    # Track cluster to path mapping
                    for code in codes:
                        for cluster_id in code.cluster_ids:
                            cluster_to_path[cluster_id] = code.node_id
            
            hierarchy = HierarchicalStructure(
                themes=themes,
                cluster_to_path=cluster_to_path
            )
            
            self._validate_hierarchy(hierarchy, cluster_data)
            
            return hierarchy
        
        async def _create_themes(self,
                               cluster_data: Dict[int, ClusterData],
                               initial_labels: Dict[int, InitialLabel],
                               var_lab: str) -> List[HierarchyNode]:
            """Create theme-level groupings using LLM"""
            # Prepare cluster information for LLM
            cluster_info = self._prepare_cluster_info(cluster_data, initial_labels)
            
            # Create prompt
            prompt = self._create_theme_prompt(cluster_info, var_lab)
            
            # Get theme groupings from LLM
            response = await self._get_llm_response(prompt, "themes")
            
            # Convert response to HierarchyNode objects
            themes = []
            for i, theme_data in enumerate(response.get("themes", [])):
                theme_id = str(i + 1)
                theme = HierarchyNode(
                    node_id=theme_id,
                    level="theme",
                    label=theme_data.get("label", f"Theme {theme_id}"),
                    children=[],
                    cluster_ids=theme_data.get("cluster_ids", [])
                )
                themes.append(theme)
            
            # Ensure all clusters are assigned to a theme
            assigned_clusters = set()
            for theme in themes:
                assigned_clusters.update(theme.cluster_ids)
            
            unassigned = set(cluster_data.keys()) - assigned_clusters
            if unassigned:
                # Create a theme for unassigned clusters
                other_theme = HierarchyNode(
                    node_id=str(len(themes) + 1),
                    level="theme",
                    label="Other",
                    children=[],
                    cluster_ids=list(unassigned)
                )
                themes.append(other_theme)
            
            return themes
        
        async def _create_topics(self,
                               theme: HierarchyNode,
                               cluster_data: Dict[int, ClusterData],
                               initial_labels: Dict[int, InitialLabel],
                               var_lab: str) -> List[HierarchyNode]:
            """Create topic-level groupings within a theme"""
            # If theme has few clusters, create a single topic
            if len(theme.cluster_ids) <= 3:
                topic = HierarchyNode(
                    node_id=f"{theme.node_id}.1",
                    level="topic",
                    label=f"{theme.label} - General",
                    children=[],
                    cluster_ids=theme.cluster_ids
                )
                return [topic]
            
            # Prepare cluster information for this theme
            theme_clusters = {cid: cluster_data[cid] for cid in theme.cluster_ids}
            theme_labels = {cid: initial_labels[cid] for cid in theme.cluster_ids if cid in initial_labels}
            cluster_info = self._prepare_cluster_info(theme_clusters, theme_labels)
            
            # Create prompt
            prompt = self._create_topic_prompt(theme, cluster_info, var_lab)
            
            # Get topic groupings from LLM
            response = await self._get_llm_response(prompt, "topics")
            
            # Convert response to HierarchyNode objects
            topics = []
            for i, topic_data in enumerate(response.get("topics", [])):
                topic_id = f"{theme.node_id}.{i + 1}"
                topic = HierarchyNode(
                    node_id=topic_id,
                    level="topic",
                    label=topic_data.get("label", f"Topic {topic_id}"),
                    children=[],
                    cluster_ids=topic_data.get("cluster_ids", [])
                )
                topics.append(topic)
            
            # Ensure all theme clusters are assigned to a topic
            assigned_clusters = set()
            for topic in topics:
                assigned_clusters.update(topic.cluster_ids)
            
            unassigned = set(theme.cluster_ids) - assigned_clusters
            if unassigned:
                other_topic = HierarchyNode(
                    node_id=f"{theme.node_id}.{len(topics) + 1}",
                    level="topic",
                    label="Other",
                    children=[],
                    cluster_ids=list(unassigned)
                )
                topics.append(other_topic)
            
            return topics
        
        def _create_codes(self,
                         topic: HierarchyNode,
                         cluster_data: Dict[int, ClusterData],
                         initial_labels: Dict[int, InitialLabel]) -> List[HierarchyNode]:
            """Create code-level nodes (one per cluster)"""
            codes = []
            
            for i, cluster_id in enumerate(sorted(topic.cluster_ids)):
                code_id = f"{topic.node_id}.{i + 1}"
                
                # Get label from initial labels or fallback
                label = initial_labels[cluster_id].label if cluster_id in initial_labels else f"Cluster {cluster_id}"
                
                code = HierarchyNode(
                    node_id=code_id,
                    level="code",
                    label=label,
                    children=[],
                    cluster_ids=[cluster_id]  # Use current cluster ID
                )
                codes.append(code)
            
            return codes
        
        def _prepare_cluster_info(self, clusters: Dict[int, ClusterData], labels: Dict[int, InitialLabel]) -> List[Dict]:
            """Prepare cluster information for LLM prompts"""
            cluster_info = []
            
            for cluster_id, cluster in clusters.items():
                # Get most common codes and descriptions
                code_counts = defaultdict(int)
                desc_counts = defaultdict(int)
                
                for code in cluster.descriptive_codes:
                    code_counts[code] += 1
                for desc in cluster.code_descriptions:
                    desc_counts[desc] += 1
                
                # Get top 5 most common
                top_codes = sorted(code_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                top_descs = sorted(desc_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                
                # Get label from initial labels
                label = labels[cluster_id].label if cluster_id in labels else f"Cluster {cluster_id}"
                
                info = {
                    "cluster_id": cluster_id,
                    "label": label,
                    "size": cluster.size,  # Use pre-computed size
                    "top_codes": [code for code, _ in top_codes],
                    "top_descriptions": [desc for desc, _ in top_descs]
                }
                cluster_info.append(info)
            
            return cluster_info
        
        def _create_theme_prompt(self, cluster_info: List[Dict], var_lab: str) -> str:
            """Create prompt for theme-level grouping"""
            prompt = HIERARCHY_CREATION_PROMPT.replace("{var_lab}", var_lab)
            prompt = prompt.replace("{level}", "theme")
            prompt = prompt.replace("{language}", self.config.language)
            
            # Add cluster information
            clusters_text = []
            for info in cluster_info:
                cluster_text = f"\nCluster {info['cluster_id']}:"
                cluster_text += f"\n- Label: {info['label']}"
                cluster_text += f"\n- Size: {info['size']} items"
                cluster_text += f"\n- Top codes: {', '.join(info['top_codes'][:3])}"
                clusters_text.append(cluster_text)
            
            prompt = prompt.replace("{clusters}", "\n".join(clusters_text))
            
            return prompt
        
        def _create_topic_prompt(self,
                               theme: HierarchyNode,
                               cluster_info: List[Dict],
                               var_lab: str) -> str:
            """Create prompt for topic-level grouping within a theme"""
            prompt = HIERARCHY_CREATION_PROMPT.replace("{var_lab}", var_lab)
            prompt = prompt.replace("{level}", "topic")
            prompt = prompt.replace("{language}", self.config.language)
            
            # Add theme context
            theme_context = f"You are creating topics within the theme: '{theme.label}'"
            prompt = prompt.replace("You are creating", theme_context)
            
            # Add cluster information
            clusters_text = []
            for info in cluster_info:
                cluster_text = f"\nCluster {info['cluster_id']}:"
                cluster_text += f"\n- Label: {info['label']}"
                cluster_text += f"\n- Size: {info['size']} items"
                cluster_text += f"\n- Top codes: {', '.join(info['top_codes'][:3])}"
                clusters_text.append(cluster_text)
            
            prompt = prompt.replace("{clusters}", "\n".join(clusters_text))
            
            return prompt
        
        async def _get_llm_response(self, prompt: str, response_type: str) -> Dict:
            """Get response from LLM with retry logic"""
            messages = [
                {"role": "system", "content": "You are an expert in hierarchical organization and thematic analysis."},
                {"role": "user", "content": prompt}
            ]
            
            for attempt in range(self.config.max_retries):
                try:
                    # Use regular OpenAI client for JSON response
                    from openai import AsyncOpenAI
                    openai_client = AsyncOpenAI(api_key=self.config.api_key)
                    
                    response = await openai_client.chat.completions.create(
                        model=self.config.model,
                        messages=messages,
                        temperature=0.3,
                        max_tokens=4000,
                        response_format={ "type": "json_object" }
                    )
                    
                    # Parse JSON response
                    content = response.choices[0].message.content
                    return self._parse_json_response(content, response_type)
                    
                except Exception as e:
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                    else:
                        logger.error(f"Failed to get LLM response: {e}")
                        # Return fallback response
                        return self._get_fallback_response(response_type)
        
        def _parse_json_response(self, content: str, response_type: str) -> Dict:
            """Parse JSON response from LLM"""
            import json
            
            try:
                data = json.loads(content)
                return data
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                return self._get_fallback_response(response_type)
        
        def _get_fallback_response(self, response_type: str) -> Dict:
            """Get fallback response when LLM fails"""
            if response_type == "themes":
                return {"themes": [{"label": "All Clusters", "cluster_ids": []}]}
            elif response_type == "topics":
                return {"topics": [{"label": "All Topics", "cluster_ids": []}]}
            else:
                return {}
        
        def _validate_hierarchy(self,
                              hierarchy: HierarchicalStructure,
                              cluster_data: Dict[int, ClusterData]):
            """Validate the hierarchy is complete and consistent"""
            # Check all clusters are mapped
            all_clusters = set(cluster_data.keys())
            mapped_clusters = set(hierarchy.cluster_to_path.keys())
            
            unmapped = all_clusters - mapped_clusters
            if unmapped:
                logger.warning(f"Unmapped clusters: {unmapped}")
            
            # Check hierarchy consistency
            for theme in hierarchy.themes:
                theme_clusters = set()
                for topic in theme.children:
                    topic_clusters = set()
                    for code in topic.children:
                        topic_clusters.update(code.cluster_ids)
                    theme_clusters.update(topic_clusters)
                
                # Basic validation that clusters are properly assigned
                logger.info(f"Theme {theme.node_id} has {len(theme_clusters)} clusters mapped")
    
    class Phase3Summarizer:
        """Phase 3: Generate summaries for themes"""
        
        def __init__(self, config: LabellerConfig, client):
            self.config = config
            self.client = client
            self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        
        async def generate_summaries(self,
                                   hierarchy: HierarchicalStructure,
                                   var_lab: str) -> List[ThemeSummary]:
            """Generate summaries for each theme explaining how it addresses the research question"""
            logger.info("Phase 3: Generating theme summaries...")
            
            # Create tasks for each theme
            tasks = []
            for theme in hierarchy.themes:
                task = self._generate_theme_summary(theme, hierarchy, var_lab)
                tasks.append(task)
            
            # Execute all tasks with progress bar - maintain theme order
            with tqdm(total=len(tasks), desc="Generating summaries") as pbar:
                # Use gather to maintain order corresponding to themes
                summaries = await asyncio.gather(*tasks)
                pbar.update(len(tasks))
            
            # Log summary generation for debugging
            for summary in summaries:
                logger.info(f"Generated summary for theme {summary.theme_id}: '{summary.theme_label}'")
            
            return summaries
        
        async def _generate_theme_summary(self,
                                        theme,
                                        hierarchy: HierarchicalStructure,
                                        var_lab: str) -> ThemeSummary:
            """Generate summary for a single theme"""
            async with self.semaphore:
                try:
                    # Collect information about the theme
                    theme_info = self._collect_theme_info(theme, hierarchy)
                    
                    # Create prompt
                    prompt = self._create_summary_prompt(theme, theme_info, var_lab)
                    
                    # Get summary from LLM
                    response = await self._get_llm_response(prompt)
                    
                    return ThemeSummary(
                        theme_id=theme.node_id,
                        theme_label=theme.label,
                        summary=response.get("summary", ""),
                        relevance_to_question=response.get("relevance", "")
                    )
                    
                except Exception as e:
                    logger.error(f"Error generating summary for theme {theme.node_id}: {e}")
                    return ThemeSummary(
                        theme_id=theme.node_id,
                        theme_label=theme.label,
                        summary=f"Error generating summary: {str(e)}",
                        relevance_to_question="Unable to analyze relevance due to error"
                    )
        
        def _collect_theme_info(self, theme, hierarchy: HierarchicalStructure) -> Dict:
            """Collect comprehensive information about a theme"""
            info = {
                "theme_label": theme.label,
                "topics": [],
                "total_clusters": 0,
                "representative_codes": [],
                "representative_descriptions": []
            }
            
            # Collect information from all topics and codes
            code_frequency = {}
            #desc_samples = []
            
            for topic in theme.children:
                topic_info = {
                    "label": topic.label,
                    "codes": []
                }
                
                for code in topic.children:
                    topic_info["codes"].append(code.label)
                    info["total_clusters"] += len(code.cluster_ids)
                    
                    # Count code frequencies (simplified for this implementation)
                    code_frequency[code.label] = code_frequency.get(code.label, 0) + 1
                
                info["topics"].append(topic_info)
            
            # Get most frequent codes
            sorted_codes = sorted(code_frequency.items(), key=lambda x: x[1], reverse=True)
            info["representative_codes"] = [code for code, _ in sorted_codes[:10]]
            
            return info
        
        def _create_summary_prompt(self, theme, theme_info: Dict, var_lab: str) -> str:
            """Create prompt for theme summary"""
            prompt = HIERARCHICAL_THEME_SUMMARY_PROMPT.replace("{var_lab}", var_lab)
            prompt = prompt.replace("{theme_label}", theme.label)
            prompt = prompt.replace("{language}", self.config.language)
            
            # Add theme structure information
            structure_text = f"\nTheme: {theme_info['theme_label']}"
            structure_text += f"\nTotal response clusters: {theme_info['total_clusters']}"
            structure_text += "\n\nTopics within this theme:"
            
            for topic in theme_info['topics']:
                structure_text += f"\n- {topic['label']}"
                structure_text += f"\n  Codes: {', '.join(topic['codes'][:5])}"
                if len(topic['codes']) > 5:
                    structure_text += f" (and {len(topic['codes']) - 5} more)"
            
            structure_text += "\n\nMost representative codes across the theme:"
            for code in theme_info['representative_codes'][:10]:
                structure_text += f"\n- {code}"
            
            prompt = prompt.replace("{theme_structure}", structure_text)
            
            return prompt
        
        async def _get_llm_response(self, prompt: str) -> Dict:
            """Get response from LLM with retry logic"""
            messages = [
                {"role": "system", "content": "You are an expert in qualitative data analysis and thematic summarization."},
                {"role": "user", "content": prompt}
            ]
            
            for attempt in range(self.config.max_retries):
                try:
                    # Use regular OpenAI client for JSON response
                    from openai import AsyncOpenAI
                    openai_client = AsyncOpenAI(api_key=self.config.api_key)
                    
                    response = await openai_client.chat.completions.create(
                        model=self.config.model,
                        messages=messages,
                        temperature=0.3,
                        max_tokens=1000,
                        response_format={ "type": "json_object" }
                    )
                    
                    # Parse JSON response
                    content = response.choices[0].message.content
                    return self._parse_json_response(content)
                    
                except Exception as e:
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                    else:
                        raise e
        
        def _parse_json_response(self, content: str) -> Dict:
            """Parse JSON response from LLM"""
            import json
            
            try:
                data = json.loads(content)
                return data
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                return {
                    "summary": "Unable to generate summary",
                    "relevance": "Unable to analyze relevance"
                }
    
    # ===== MAIN WORKFLOW METHODS =====
    
    async def _create_hierarchical_labels_async(self, 
                                              cluster_results: List[models.ClusterModel], 
                                              var_lab: str) -> Tuple[List[models.LabelModel], List[ThemeSummary]]:
        """Async implementation of the main workflow"""
        import time
        workflow_start = time.time()
        logger.info("Starting hierarchical labeling process...")
        
        # Extract cluster data
        extract_start = time.time()
        cluster_data = self.extract_cluster_data(cluster_results)
        extract_time = time.time() - extract_start
        logger.info(f"Extracted data for {len(cluster_data)} clusters in {extract_time:.2f}s")
        
        # Phase 1: Initial labels
        phase1_start = time.time()
        initial_labels = await self.phase1_initial_labels(cluster_data, var_lab)
        phase1_time = time.time() - phase1_start
        logger.info(f"Phase 1: Generated initial labels for {len(initial_labels)} clusters in {phase1_time:.2f}s")
        
        # Phase 2: Create hierarchy (merger moved to step 5)
        phase2_start = time.time()
        hierarchy = await self.phase2_create_hierarchy(cluster_data, initial_labels, var_lab)
        phase2_time = time.time() - phase2_start
        logger.info(f"Phase 2: Created hierarchical structure in {phase2_time:.2f}s")
        
        # Phase 3: Generate summaries
        phase3_start = time.time()
        summaries = await self.phase3_theme_summaries(hierarchy, var_lab)
        phase3_time = time.time() - phase3_start
        logger.info(f"Phase 3: Generated {len(summaries)} theme summaries in {phase3_time:.2f}s")
        
        
        # Convert to output format
        convert_start = time.time()
        result = self.create_label_models(cluster_results, hierarchy, summaries)
        convert_time = time.time() - convert_start
        
        total_time = time.time() - workflow_start
        logger.info(f"Total workflow time: {total_time:.2f}s (extract: {extract_time:.2f}s, phase1: {phase1_time:.2f}s, phase2: {phase2_time:.2f}s, phase3: {phase3_time:.2f}s, convert: {convert_time:.2f}s)")
        
        return result, summaries
    
    def extract_cluster_data(self, cluster_results: List[models.ClusterModel]) -> Dict[int, ClusterData]:
        """Extract and organize cluster data from model results"""
        import time
        start_time = time.time()
        
        cluster_data = defaultdict(lambda: {
            'descriptive_codes': [],
            'code_descriptions': [],
            'embeddings': []
        })
        
        # Collect data by cluster ID
        segment_count = 0
        for result in cluster_results:
            for segment in result.response_segment:
                if segment.micro_cluster is not None:
                    segment_count += 1
                    cluster_id = list(segment.micro_cluster.keys())[0]
                    cluster_data[cluster_id]['descriptive_codes'].append(segment.descriptive_code)
                    cluster_data[cluster_id]['code_descriptions'].append(segment.code_description)
                    # Use description embeddings by default
                    cluster_data[cluster_id]['embeddings'].append(segment.description_embedding)
        
        collect_time = time.time() - start_time
        logger.info(f"Collected {segment_count} segments into {len(cluster_data)} clusters in {collect_time:.2f}s")
        
        # Convert to ClusterData objects
        convert_start = time.time()
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
        
        convert_time = time.time() - convert_start
        total_time = time.time() - start_time
        logger.info(f"Converted to ClusterData objects in {convert_time:.2f}s (total: {total_time:.2f}s)")
        
        return clusters
    
    async def phase1_initial_labels(self, 
                                  cluster_data: Dict[int, ClusterData], 
                                  var_lab: str) -> Dict[int, InitialLabel]:
        """Phase 1: Generate initial labels for each cluster"""
        phase1 = self.Phase1Labeller(self.config, self.client)
        return await phase1.label_clusters(cluster_data, var_lab)
    
    async def phase2_create_hierarchy(self,
                                    cluster_data: Dict[int, ClusterData],
                                    initial_labels: Dict[int, InitialLabel],
                                    var_lab: str) -> HierarchicalStructure:
        """Phase 2: Create 3-level hierarchical structure"""
        phase2 = self.Phase2Organizer(self.config, self.client)
        return await phase2.create_hierarchy(cluster_data, initial_labels, var_lab)
    
    
    async def phase3_theme_summaries(self,
                                   hierarchy: HierarchicalStructure,
                                   var_lab: str) -> List[ThemeSummary]:
        """Phase 3: Generate summaries for each theme"""
        phase3 = self.Phase3Summarizer(self.config, self.client)
        return await phase3.generate_summaries(hierarchy, var_lab)
    
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
                if segment.micro_cluster is not None:
                    cluster_id = list(segment.micro_cluster.keys())[0]
                    
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
                                micro_cluster=segment.micro_cluster,
                                Theme={int(theme_id): theme_node.label} if theme_node else None,
                                Topic={hash('.'.join(path_parts[:2])) % 10000: topic_node.label} if topic_node else None,
                                Keyword={hash(path) % 10000: code_node.label} if code_node else None
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
                    # Convert theme_id to string to match summary_map keys
                    theme_id_str = str(theme_id)
                    if theme_id_str in summary_map:
                        summary += summary_map[theme_id_str].summary + "\n"
                
                label_model = models.LabelModel(
                    respondent_id=result.respondent_id,
                    response=result.response,
                    summary=summary.strip(),
                    response_segment=label_segments
                )
                
                label_models.append(label_model)
        
        return label_models

    @staticmethod
    def display_hierarchical_results(label_results: List[models.LabelModel], 
                                   theme_summaries: Optional[List[ThemeSummary]] = None,
                                   unique_clusters_count: Optional[int] = None):
        """Display detailed hierarchical labeling results - extracted from __main__ section"""
        from collections import Counter, defaultdict
        
        print("\n=== UNPACKING HIERARCHICAL LABELS ===")
        
        # Data structures to collect hierarchy and examples  
        themes = {}
        theme_summary_texts = {}
        theme_topics = defaultdict(lambda: {})
        topic_codes = defaultdict(lambda: defaultdict(list))
        theme_response_examples = defaultdict(list)
        topic_response_examples = defaultdict(lambda: defaultdict(list))
        code_response_examples = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        theme_response_counts = defaultdict(int)
        topic_response_counts = defaultdict(lambda: defaultdict(int))
        
        # Use provided theme summaries if available
        if theme_summaries:
            for summary in theme_summaries:
                theme_summary_texts[int(summary.theme_id)] = summary.summary
        
        # Extract themes first
        for result in label_results:
            for segment in result.response_segment:
                if segment.Theme:
                    theme_id, theme_label = list(segment.Theme.items())[0]
                    themes[theme_id] = theme_label
        
        # Process results to extract hierarchy and examples
        for result in label_results:
            for segment in result.response_segment:
                if segment.Theme:
                    theme_id, theme_label = list(segment.Theme.items())[0] 
                    themes[theme_id] = theme_label
                    theme_response_counts[theme_id] += 1
                    
                    # Collect response examples for themes
                    example = {
                        'response': segment.segment_response,
                        'code': segment.descriptive_code,
                        'description': segment.code_description
                    }
                    if len(theme_response_examples[theme_id]) < 5:
                        theme_response_examples[theme_id].append(example)
                    
                    if segment.Topic:
                        topic_id, topic_label = list(segment.Topic.items())[0]
                        theme_topics[theme_id][topic_id] = topic_label
                        topic_response_counts[theme_id][topic_id] += 1
                        
                        if len(topic_response_examples[theme_id][topic_id]) < 3:
                            topic_response_examples[theme_id][topic_id].append(example)
                        
                        if segment.Keyword:
                            code_id, code_label = list(segment.Keyword.items())[0]
                            topic_codes[theme_id][topic_id].append((code_id, code_label))
                            
                            if len(code_response_examples[theme_id][topic_id][code_id]) < 2:
                                code_response_examples[theme_id][topic_id][code_id].append(example)
        
        # Display hierarchical structure  
        print(f"\n{'='*80}")
        print("HIERARCHICAL LABELING RESULTS")
        print(f"{'='*80}")
        print(f"\nFound {len(themes)} themes across {len(label_results)} respondents:")
        
        # Theme distribution
        print("\n📊 THEME DISTRIBUTION:")
        for theme_id in sorted(themes.keys()):
            response_count = theme_response_counts[theme_id]
            percentage = (response_count / sum(len(r.response_segment) for r in label_results)) * 100
            print(f"  Theme {theme_id}: {themes[theme_id]} - {response_count} segments ({percentage:.1f}%)")
        
        # Detailed theme analysis
        for theme_id in sorted(themes.keys()):
            print(f"\n\n{'='*80}")
            print(f"🎯 THEME {theme_id}: {themes[theme_id]}")
            print(f"{'='*80}")
            print(f"Segments in this theme: {theme_response_counts[theme_id]}")
            
            # Theme summary
            if theme_id in theme_summary_texts:
                print("\n📝 Summary:")
                print(f"{theme_summary_texts[theme_id]}")
            else:
                print(f"\n📝 Summary: No summary available for Theme {theme_id}")
            
            # Topics in this theme
            topics_in_theme = theme_topics[theme_id]
            print(f"\n📋 Topics in this theme ({len(topics_in_theme)}):")
            
            for topic_idx, topic_id in enumerate(sorted(topics_in_theme.keys()), 1):
                topic_label = topics_in_theme[topic_id]
                topic_count = topic_response_counts[theme_id][topic_id]
                print(f"\n  {topic_idx}. TOPIC {topic_id}: {topic_label} ({topic_count} segments)")
                
                # Codes in this topic
                codes_in_topic = topic_codes[theme_id][topic_id]
                code_counts = Counter(codes_in_topic)
                sorted_code_counts = sorted(code_counts.items(), key=lambda x: x[1], reverse=True)
                
                print(f"\n     Codes ({len(sorted_code_counts)}):")
                for (code_id, code_label), count in sorted_code_counts[:5]:
                    print(f"       • {code_label} (appears {count}x)")
                    if code_id in code_response_examples[theme_id][topic_id]:
                        for example in code_response_examples[theme_id][topic_id][code_id][:1]:
                            print(f"         Example: \"{example['response'][:80]}...\"")
                
                if len(sorted_code_counts) > 5:
                    print(f"       ... and {len(sorted_code_counts) - 5} more codes")
        
        # Summary statistics
        print(f"\n\n{'='*80}")
        print("📈 SUMMARY STATISTICS & QUALITY METRICS") 
        print(f"{'='*80}")
        
        total_segments = sum(len(r.response_segment) for r in label_results)
        print("\n📊 Overall Statistics:")
        print(f"  • Total respondents processed: {len(label_results)}")
        print(f"  • Total segments processed: {total_segments}")
        print(f"  • Total themes: {len(themes)}")
        total_topics = sum(len(topics) for topics in theme_topics.values())
        print(f"  • Total topics: {total_topics}")
        total_codes = len(set([code for theme in topic_codes.values() for topic in theme.values() for code in topic]))
        print(f"  • Total unique codes: {total_codes}")
        if unique_clusters_count:
            print(f"  • Original clusters: {unique_clusters_count}")
        
        print("\n📏 Distribution Metrics:")
        print(f"  • Average segments per theme: {total_segments / len(themes):.1f}")
        print(f"  • Average topics per theme: {total_topics / len(themes):.1f}")
        print(f"  • Average codes per topic: {total_codes / total_topics:.1f}")
        
        # Theme size distribution
        print("\n📊 Theme Size Distribution:")
        theme_sizes = sorted([(theme_id, theme_response_counts[theme_id]) for theme_id in themes.keys()], 
                           key=lambda x: x[1], reverse=True)
        for theme_id, count in theme_sizes:
            bar_length = int((count / max(theme_response_counts.values())) * 30)
            bar = '█' * bar_length
            print(f"  Theme {theme_id}: {bar} {count} segments")
        
        print("\n✅ Quality Check Complete")
        print(f"{'='*80}")


if __name__ == "__main__":
    """Test the labeller with cached cluster data"""
    #import sys
    #from pathlib import Path
    from collections import Counter #, defaultdict
    
    from cache_manager import CacheManager
    from config import CacheConfig
    from utils import data_io
    
    import models
    import nest_asyncio
    nest_asyncio.apply()

    
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
                if segment.micro_cluster:
                    cluster_id = list(segment.micro_cluster.keys())[0]
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
        label_results, theme_summaries = labeller.create_hierarchical_labels(cluster_results, var_lab)
        
        # Save both results to cache
        cache_key = 'labels_all'
        cache_manager.save_to_cache(label_results, filename, cache_key)
        print(f"\nSaved {len(label_results)} label results to cache with key '{cache_key}'")
        
        # Save summaries separately for easier access
        summary_cache_key = 'theme_summaries'
        cache_manager.save_to_cache(theme_summaries, filename, summary_cache_key)
        print(f"Saved {len(theme_summaries)} theme summaries to cache with key '{summary_cache_key}'")
        
        # Display results using the reusable function
        Labeller.display_hierarchical_results(label_results, theme_summaries, len(unique_clusters))
        
    else:
        print("No cached cluster data found.")
        print("Please run the clustering pipeline first:")
        print("  python clusterer.py")