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
from prompts import INITIAL_LABEL_PROMPT, TOPIC_CREATION_PROMPT, THEME_CREATION_PROMPT, HIERARCHICAL_THEME_SUMMARY_PROMPT

logger = logging.getLogger(__name__)

# Models for structured data
class LabellerConfig(BaseModel):
    """Configuration for the Labeller"""
    model: str = DEFAULT_MODEL
    api_key: str = OPENAI_API_KEY
    max_concurrent_requests: int = 10   
    batch_size: int = 15  # Increased for better context while maintaining concurrency
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
    """Complete hierarchical structure with explicit mappings"""
    themes: List[HierarchyNode]  # Level 1 nodes
    cluster_to_path: Dict[int, str]  # Cluster ID → Path (e.g., "1.2.3")
    # New explicit mapping fields
    cluster_to_topic: Dict[int, str]  # Cluster ID → Topic label
    cluster_to_theme: Dict[int, str]  # Cluster ID → Theme label
    topic_to_theme: Dict[str, str]    # Topic label → Theme label
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
            """Create 3-level hierarchical structure using two-step approach"""
            logger.info("Phase 2: Creating hierarchical structure...")
            
            # Step 1: Create topics from all clusters
            topics = await self._create_topics_from_clusters(cluster_data, initial_labels, var_lab)
            logger.info(f"Created {len(topics)} topics")
            
            # Step 2: Create themes from topics
            themes = await self._create_themes_from_topics(topics, var_lab)
            logger.info(f"Created {len(themes)} themes")
            
            # Step 3: Assign codes within topics (level 3) and build mappings
            cluster_to_path = {}
            cluster_to_topic = {}
            cluster_to_theme = {}
            topic_to_theme = {}
            
            for theme in themes:
                for topic in theme.children:
                    # Track topic to theme mapping
                    topic_to_theme[topic.label] = theme.label
                    
                    codes = self._create_codes(topic, cluster_data, initial_labels)
                    topic.children = codes
                    
                    # Track cluster mappings
                    for code in codes:
                        for cluster_id in code.cluster_ids:
                            cluster_to_path[cluster_id] = code.node_id
                            cluster_to_topic[cluster_id] = topic.label
                            cluster_to_theme[cluster_id] = theme.label
            
            hierarchy = HierarchicalStructure(
                themes=themes,
                cluster_to_path=cluster_to_path,
                cluster_to_topic=cluster_to_topic,
                cluster_to_theme=cluster_to_theme,
                topic_to_theme=topic_to_theme
            )
            
            self._validate_hierarchy(hierarchy, cluster_data)
            
            return hierarchy
        
        async def _create_topics_from_clusters(self,
                                             cluster_data: Dict[int, ClusterData],
                                             initial_labels: Dict[int, InitialLabel],
                                             var_lab: str) -> List[HierarchyNode]:
            """Step 1: Create topics from all clusters"""
            # Prepare cluster information for LLM
            cluster_info = self._prepare_cluster_info(cluster_data, initial_labels)
            
            # Create prompt
            prompt = self._create_topic_prompt(cluster_info, var_lab)
            
            # Get topic groupings from LLM
            response = await self._get_llm_response(prompt, "topics")
            
            # Convert response to HierarchyNode objects
            topics = []
            for i, topic_data in enumerate(response.get("topics", [])):
                topic_id = f"T{i + 1}"
                topic = HierarchyNode(
                    node_id=topic_id,
                    level="topic",
                    label=topic_data.get("label", f"Topic {topic_id}"),
                    children=[],
                    cluster_ids=topic_data.get("cluster_ids", [])
                )
                topics.append(topic)
            
            # Ensure all clusters are assigned to a topic
            assigned_clusters = set()
            for topic in topics:
                assigned_clusters.update(topic.cluster_ids)
            
            unassigned = set(cluster_data.keys()) - assigned_clusters
            if unassigned:
                # Create a topic for unassigned clusters
                other_topic = HierarchyNode(
                    node_id=f"T{len(topics) + 1}",
                    level="topic",
                    label="Overige aspecten",
                    children=[],
                    cluster_ids=list(unassigned)
                )
                topics.append(other_topic)
            
            return topics
        
        async def _create_topics_from_clusters_with_guidance(self,
                                                           cluster_data: Dict[int, ClusterData],
                                                           initial_labels: Dict[int, InitialLabel],
                                                           var_lab: str,
                                                           guidance: str) -> List[HierarchyNode]:
            """Create topics with additional guidance for refinement"""
            # Prepare cluster information for LLM
            cluster_info = self._prepare_cluster_info(cluster_data, initial_labels)
            
            # Create prompt with guidance
            prompt = self._create_topic_prompt(cluster_info, var_lab)
            prompt = guidance + "\n\n" + prompt
            
            # Get topic groupings from LLM
            response = await self._get_llm_response(prompt, "topics")
            
            # Convert response to HierarchyNode objects
            topics = []
            for i, topic_data in enumerate(response.get("topics", [])):
                topic_id = f"T{i + 1}"
                topic = HierarchyNode(
                    node_id=topic_id,
                    level="topic", 
                    label=topic_data.get("label", f"Topic {topic_id}"),
                    children=[],
                    cluster_ids=topic_data.get("cluster_ids", [])
                )
                topics.append(topic)
            
            return topics
        
        async def _create_themes_from_topics(self,
                                           topics: List[HierarchyNode],
                                           var_lab: str) -> List[HierarchyNode]:
            """Step 2: Create themes from topics"""
            # Prepare topic information for LLM
            topic_info = []
            for topic in topics:
                info = {
                    "label": topic.label,
                    "explanation": f"Contains {len(topic.cluster_ids)} clusters",
                    "cluster_ids": topic.cluster_ids
                }
                topic_info.append(info)
            
            # Create prompt
            prompt = self._create_theme_prompt(topic_info, var_lab)
            
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
                    cluster_ids=[]
                )
                
                # Find topics that belong to this theme and assign them as children
                topic_labels = theme_data.get("topic_labels", [])
                for topic in topics:
                    if topic.label in topic_labels:
                        # Update topic node_id to reflect theme hierarchy
                        topic.node_id = f"{theme_id}.{len(theme.children) + 1}"
                        theme.children.append(topic)
                        theme.cluster_ids.extend(topic.cluster_ids)
                
                themes.append(theme)
            
            # Ensure all topics are assigned to a theme
            assigned_topics = set()
            for theme in themes:
                for topic in theme.children:
                    assigned_topics.add(topic.label)
            
            unassigned_topics = [t for t in topics if t.label not in assigned_topics]
            if unassigned_topics:
                # Create a theme for unassigned topics
                other_theme = HierarchyNode(
                    node_id=str(len(themes) + 1),
                    level="theme",
                    label="Overige thema's",
                    children=[],
                    cluster_ids=[]
                )
                for topic in unassigned_topics:
                    topic.node_id = f"{other_theme.node_id}.{len(other_theme.children) + 1}"
                    other_theme.children.append(topic)
                    other_theme.cluster_ids.extend(topic.cluster_ids)
                themes.append(other_theme)
            
            return themes
        
        
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
        
        def _create_theme_prompt(self, topic_info: List[Dict], var_lab: str) -> str:
            """Create prompt for theme-level grouping using topics"""
            prompt = THEME_CREATION_PROMPT.replace("{var_lab}", var_lab)
            prompt = prompt.replace("{language}", self.config.language)
            
            # Add topic information
            topics_text = []
            for info in topic_info:
                topic_text = f"\nTopic: {info['label']}"
                topic_text += f"\n- Explanation: {info['explanation']}"
                topic_text += f"\n- Contains clusters: {', '.join(map(str, info['cluster_ids']))}"
                topics_text.append(topic_text)
            
            prompt = prompt.replace("{topics}", "\n".join(topics_text))
            
            return prompt
        
        def _create_topic_prompt(self, cluster_info: List[Dict], var_lab: str) -> str:
            """Create prompt for topic-level grouping"""
            prompt = TOPIC_CREATION_PROMPT.replace("{var_lab}", var_lab)
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
        
        async def _get_llm_response(self, prompt: str, response_type: str) -> Dict:
            """Get response from LLM with retry logic"""
            messages = [
                {"role": "system", "content": "You are an expert in hierarchical organization and thematic analysis. Always respond in valid JSON format."},
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
            
            # Log mapping statistics
            logger.info(f"Hierarchy mapping statistics:")
            logger.info(f"  - Total clusters: {len(all_clusters)}")
            logger.info(f"  - Mapped clusters: {len(mapped_clusters)}")
            logger.info(f"  - Unique topics: {len(set(hierarchy.cluster_to_topic.values()))}")
            logger.info(f"  - Unique themes: {len(set(hierarchy.cluster_to_theme.values()))}")
            
            # Check hierarchy consistency
            for theme in hierarchy.themes:
                theme_clusters = set()
                for topic in theme.children:
                    topic_clusters = set()
                    for code in topic.children:
                        topic_clusters.update(code.cluster_ids)
                    theme_clusters.update(topic_clusters)
                
                # Basic validation that clusters are properly assigned
                logger.info(f"Theme '{theme.label}' (ID: {theme.node_id}) has {len(theme_clusters)} clusters mapped")
    
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
                {"role": "system", "content": "You are an expert in qualitative data analysis and thematic summarization. Always respond in valid JSON format."},
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
    
    # ===== VALIDATION METHODS =====
    
    def validate_hierarchy_quality(self, 
                                 hierarchy: HierarchicalStructure,
                                 cluster_data: Dict[int, ClusterData],
                                 min_topics_per_theme: int = 2,
                                 max_other_percentage: float = 0.10,
                                 max_theme_percentage: float = 0.20) -> List[str]:
        """Validate hierarchy meets quality standards"""
        issues = []
        
        # Calculate total segments
        total_segments = sum(cluster.size for cluster in cluster_data.values())
        
        # Check for sufficient topic diversity
        single_topic_themes = [t for t in hierarchy.themes if len(t.children) == 1]
        if len(single_topic_themes) > len(hierarchy.themes) * 0.5:
            issues.append(f"Too many single-topic themes: {len(single_topic_themes)}/{len(hierarchy.themes)}")
        
        # Check for oversized themes
        for theme in hierarchy.themes:
            theme_size = sum(cluster_data[cid].size for cid in theme.cluster_ids if cid in cluster_data)
            percentage = theme_size / total_segments
            
            # Check for oversized "other" categories
            if any(word in theme.label.lower() for word in ["overige", "andere", "other", "rest", "miscellaneous", "additional"]):
                if percentage > max_other_percentage:
                    issues.append(f"Oversized catch-all '{theme.label}': {percentage:.1%} (max {max_other_percentage:.0%})")
            
            # Check for any oversized theme
            elif percentage > max_theme_percentage:
                issues.append(f"Oversized theme '{theme.label}': {percentage:.1%} (max {max_theme_percentage:.0%})")
        
        # Check topic distribution
        topics_per_theme = [len(theme.children) for theme in hierarchy.themes]
        avg_topics = sum(topics_per_theme) / len(topics_per_theme) if topics_per_theme else 0
        if avg_topics < 1.5:
            issues.append(f"Insufficient topic diversity: avg {avg_topics:.1f} topics/theme (need 2+)")
        
        # Check for duplicate topic names across themes
        all_topic_names = []
        for theme in hierarchy.themes:
            for topic in theme.children:
                all_topic_names.append(topic.label)
        
        duplicates = [name for name in set(all_topic_names) if all_topic_names.count(name) > 1]
        if duplicates:
            issues.append(f"Duplicate topic names: {', '.join(duplicates)}")
        
        return issues
    
    async def _refine_oversized_categories(self,
                                         hierarchy: HierarchicalStructure,
                                         cluster_data: Dict[int, ClusterData],
                                         initial_labels: Dict[int, InitialLabel],
                                         var_lab: str) -> HierarchicalStructure:
        """Refine hierarchy by splitting oversized categories and redistributing 'Other' topics"""
        logger.info("Refining oversized categories...")
        
        # Find oversized themes
        total_segments = sum(cluster.size for cluster in cluster_data.values())
        oversized_themes = []
        other_theme = None
        
        for theme in hierarchy.themes:
            theme_size = sum(cluster_data[cid].size for cid in theme.cluster_ids if cid in cluster_data)
            percentage = theme_size / total_segments
            
            # Check if it's oversized or a catch-all
            is_catchall = any(word in theme.label.lower() for word in ["overige", "andere", "other", "rest"])
            if is_catchall:
                other_theme = theme
                if percentage > 0.10:
                    oversized_themes.append((theme, percentage))
                    logger.info(f"Found oversized catch-all: '{theme.label}' ({percentage:.1%})")
            elif percentage > 0.20:
                oversized_themes.append((theme, percentage))
                logger.info(f"Found oversized theme: '{theme.label}' ({percentage:.1%})")
        
        # First, try to redistribute "Other" category
        if other_theme and other_theme.cluster_ids:
            logger.info(f"Analyzing 'Other' category with {len(other_theme.cluster_ids)} clusters")
            
            # Analyze what's in the "Other" category
            other_labels = []
            for cid in other_theme.cluster_ids[:10]:  # Sample first 10
                if cid in initial_labels:
                    other_labels.append(f"  - Cluster {cid}: {initial_labels[cid].label}")
            logger.info("Sample clusters in 'Other' category:")
            for label in other_labels:
                logger.info(label)
            
            hierarchy = await self._redistribute_other_category(
                hierarchy, other_theme, cluster_data, initial_labels, var_lab
            )
        
        # Then handle remaining oversized themes
        for theme, percentage in oversized_themes:
            if theme == other_theme:
                continue  # Already handled
                
            if len(theme.children) == 1:
                # Single topic theme - try to split the topic into multiple topics
                logger.info(f"Splitting single-topic theme: {theme.label}")
                
                # Get clusters in this theme
                theme_clusters = theme.cluster_ids
                theme_cluster_data = {cid: cluster_data[cid] for cid in theme_clusters if cid in cluster_data}
                theme_initial_labels = {cid: initial_labels[cid] for cid in theme_clusters if cid in initial_labels}
                
                # Re-run topic creation with instruction to create more topics
                refined_prompt = f"""
                The following clusters were previously grouped into a single oversized topic.
                Please split them into 3-5 more specific topics.
                Avoid creating a single catch-all topic.
                
                Original theme: {theme.label} (containing {percentage:.0%} of all data)
                """
                
                # Create new topics
                phase2 = self.Phase2Organizer(self.config, self.client)
                new_topics = await phase2._create_topics_from_clusters_with_guidance(
                    theme_cluster_data, theme_initial_labels, var_lab, refined_prompt
                )
                
                # Replace the single topic with multiple topics
                if len(new_topics) > 1:
                    theme.children = new_topics
                    # Update node IDs
                    for i, topic in enumerate(theme.children):
                        topic.node_id = f"{theme.node_id}.{i + 1}"
        
        # Rebuild all mappings
        hierarchy = self._rebuild_hierarchy_mappings(hierarchy, cluster_data, initial_labels)
        
        return hierarchy
    
    async def _redistribute_other_category(self,
                                         hierarchy: HierarchicalStructure,
                                         other_theme: HierarchyNode,
                                         cluster_data: Dict[int, ClusterData],
                                         initial_labels: Dict[int, InitialLabel],
                                         var_lab: str) -> HierarchicalStructure:
        """Redistribute clusters from 'Other' category to appropriate themes"""
        logger.info("Redistributing 'Other' category clusters...")
        
        # Get clusters in the "Other" theme
        other_clusters = other_theme.cluster_ids
        if not other_clusters:
            return hierarchy
        
        # Calculate semantic similarity between "Other" clusters and existing themes
        redistributed = []
        remaining = []
        
        for cluster_id in other_clusters:
            if cluster_id not in cluster_data:
                continue
                
            cluster = cluster_data[cluster_id]
            best_theme = None
            best_similarity = 0.0
            
            # Check similarity with each non-"Other" theme
            for theme in hierarchy.themes:
                if theme == other_theme:
                    continue
                    
                # Calculate average similarity to theme's clusters
                theme_embeddings = []
                for tid in theme.cluster_ids:
                    if tid in cluster_data:
                        theme_embeddings.append(cluster_data[tid].centroid)
                
                if theme_embeddings:
                    theme_centroid = np.mean(theme_embeddings, axis=0)
                    similarity = cosine_similarity([cluster.centroid], [theme_centroid])[0][0]
                    
                    if similarity > best_similarity and similarity > 0.6:  # Lower threshold for redistribution
                        best_similarity = similarity
                        best_theme = theme
            
            if best_theme:
                redistributed.append((cluster_id, best_theme, best_similarity))
            else:
                remaining.append(cluster_id)
        
        # Apply redistributions
        for cluster_id, theme, similarity in redistributed:
            logger.info(f"Redistributing cluster {cluster_id} to '{theme.label}' (similarity: {similarity:.3f})")
            other_theme.cluster_ids.remove(cluster_id)
            theme.cluster_ids.append(cluster_id)
            
            # Find appropriate topic within theme
            topic_assigned = False
            for topic in theme.children:
                # Check if this cluster fits with the topic
                if len(topic.cluster_ids) < 10:  # Don't make topics too large
                    topic.cluster_ids.append(cluster_id)
                    topic_assigned = True
                    break
            
            if not topic_assigned and theme.children:
                # Add to smallest topic
                smallest_topic = min(theme.children, key=lambda t: len(t.cluster_ids))
                smallest_topic.cluster_ids.append(cluster_id)
        
        # Handle remaining clusters
        if remaining:
            logger.info(f"{len(remaining)} clusters remain in 'Other' category after redistribution")
            
            # Calculate actual size of remaining clusters
            remaining_size = sum(cluster_data[cid].size for cid in remaining if cid in cluster_data)
            total_size = sum(cluster.size for cluster in cluster_data.values())
            remaining_percentage = remaining_size / total_size
            
            logger.info(f"Remaining clusters represent {remaining_percentage:.1%} of total segments")
            
            # If still too many (>5%), create new themes from them
            if remaining_percentage > 0.05:
                remaining_cluster_data = {cid: cluster_data[cid] for cid in remaining if cid in cluster_data}
                remaining_labels = {cid: initial_labels[cid] for cid in remaining if cid in initial_labels}
                
                # Analyze what's left
                logger.info("Creating new themes from remaining clusters...")
                logger.info("Sample remaining clusters:")
                for i, (cid, label) in enumerate(list(remaining_labels.items())[:5]):
                    logger.info(f"  - Cluster {cid}: {label.label}")
                
                # Create new topics with guidance
                phase2 = self.Phase2Organizer(self.config, self.client)
                guidance = f"""
                These clusters could not be assigned to existing themes.
                Create 3-6 specific, meaningful topics from these clusters.
                Avoid creating generic catch-all topics.
                Focus on finding common patterns or themes.
                """
                
                new_topics = await phase2._create_topics_from_clusters_with_guidance(
                    remaining_cluster_data, remaining_labels, var_lab, guidance
                )
                
                # Always create new themes from these topics
                if len(new_topics) >= 2:
                    # Group into 1-2 new themes
                    new_themes = await phase2._create_themes_from_topics(new_topics[:6], var_lab)
                    
                    # Add new themes to hierarchy
                    max_theme_id = max(int(t.node_id) for t in hierarchy.themes)
                    for i, new_theme in enumerate(new_themes):
                        if any(word in new_theme.label.lower() for word in ["overige", "andere", "other"]):
                            # Rename generic theme names
                            new_theme.label = f"Additional aspects {i+1}"
                        new_theme.node_id = str(max_theme_id + i + 1)
                        hierarchy.themes.append(new_theme)
                    
                    # Remove the original "Other" theme
                    if other_theme in hierarchy.themes:
                        hierarchy.themes.remove(other_theme)
                else:
                    # Keep as reduced "Other" theme but rename it
                    other_theme.cluster_ids = remaining
                    if remaining_percentage < 0.10:
                        other_theme.label = "Miscellaneous aspects"
        
        return hierarchy
    
    def _rebuild_hierarchy_mappings(self,
                                   hierarchy: HierarchicalStructure,
                                   cluster_data: Dict[int, ClusterData],
                                   initial_labels: Dict[int, InitialLabel]) -> HierarchicalStructure:
        """Rebuild all hierarchy mappings after changes"""
        # Create Phase2Organizer instance for code creation
        phase2_instance = self.Phase2Organizer(self.config, self.client)
        
        # Clear existing mappings
        hierarchy.cluster_to_path.clear()
        hierarchy.cluster_to_topic.clear()
        hierarchy.cluster_to_theme.clear()
        hierarchy.topic_to_theme.clear()
        
        # Rebuild mappings
        for theme in hierarchy.themes:
            for topic in theme.children:
                hierarchy.topic_to_theme[topic.label] = theme.label
                
                # Ensure topic has children (codes)
                if not topic.children:
                    codes = phase2_instance._create_codes(topic, cluster_data, initial_labels)
                    topic.children = codes
                
                for code in topic.children:
                    for cluster_id in code.cluster_ids:
                        hierarchy.cluster_to_path[cluster_id] = code.node_id
                        hierarchy.cluster_to_topic[cluster_id] = topic.label
                        hierarchy.cluster_to_theme[cluster_id] = theme.label
        
        return hierarchy
    
    def _merge_small_themes(self,
                           hierarchy: HierarchicalStructure,
                           cluster_data: Dict[int, ClusterData],
                           min_theme_size: float = 0.03) -> HierarchicalStructure:
        """Merge small themes into semantically similar larger themes"""
        total_segments = sum(cluster.size for cluster in cluster_data.values())
        
        # Identify small themes
        small_themes = []
        regular_themes = []
        
        for theme in hierarchy.themes:
            theme_size = sum(cluster_data[cid].size for cid in theme.cluster_ids if cid in cluster_data)
            percentage = theme_size / total_segments
            
            if percentage < min_theme_size and len(theme.children) <= 2:
                small_themes.append((theme, percentage))
                logger.info(f"Found small theme: '{theme.label}' ({percentage:.1%})")
            else:
                regular_themes.append(theme)
        
        if not small_themes:
            return hierarchy
        
        # Try to merge small themes into larger ones
        merged_themes = []
        for small_theme, size in small_themes:
            # Calculate theme centroid
            theme_embeddings = []
            for cid in small_theme.cluster_ids:
                if cid in cluster_data:
                    theme_embeddings.append(cluster_data[cid].centroid)
            
            if not theme_embeddings:
                continue
                
            small_theme_centroid = np.mean(theme_embeddings, axis=0)
            
            # Find best matching larger theme
            best_match = None
            best_similarity = 0.0
            
            for regular_theme in regular_themes:
                # Skip "Other" themes
                if any(word in regular_theme.label.lower() for word in ["overige", "andere", "other"]):
                    continue
                    
                # Calculate regular theme centroid
                regular_embeddings = []
                for cid in regular_theme.cluster_ids:
                    if cid in cluster_data:
                        regular_embeddings.append(cluster_data[cid].centroid)
                
                if regular_embeddings:
                    regular_centroid = np.mean(regular_embeddings, axis=0)
                    similarity = cosine_similarity([small_theme_centroid], [regular_centroid])[0][0]
                    
                    if similarity > best_similarity and similarity > 0.65:
                        best_similarity = similarity
                        best_match = regular_theme
            
            if best_match:
                logger.info(f"Merging '{small_theme.label}' into '{best_match.label}' (similarity: {best_similarity:.3f})")
                
                # Merge topics and clusters
                for topic in small_theme.children:
                    # Update topic node_id
                    topic.node_id = f"{best_match.node_id}.{len(best_match.children) + 1}"
                    best_match.children.append(topic)
                    best_match.cluster_ids.extend(topic.cluster_ids)
                
                merged_themes.append(small_theme)
        
        # Remove merged themes from hierarchy
        for theme in merged_themes:
            if theme in hierarchy.themes:
                hierarchy.themes.remove(theme)
        
        # Rebuild mappings if any merges occurred
        if merged_themes:
            hierarchy = self._rebuild_hierarchy_mappings(hierarchy, cluster_data, {})
        
        return hierarchy
    
    # ===== MAIN WORKFLOW METHODS =====
    
    async def _create_hierarchical_labels_async(self, 
                                              cluster_results: List[models.ClusterModel], 
                                              var_lab: str) -> Tuple[List[models.LabelModel], List[ThemeSummary]]:
        """Async implementation of the main workflow with validation and refinement"""
        import time
        workflow_start = time.time()
        logger.info("Starting hierarchical labeling process...")
        
        # Extract cluster data
        extract_start = time.time()
        cluster_data = self.extract_cluster_data(cluster_results)
        extract_time = time.time() - extract_start
        logger.info(f"Extracted data for {len(cluster_data)} clusters in {extract_time:.2f}s")
        
        # Iterative refinement loop
        max_iterations = 3
        hierarchy = None
        summaries = None
        
        for iteration in range(max_iterations):
            logger.info(f"\n=== Iteration {iteration + 1}/{max_iterations} ===")
            
            # Phase 1: Initial labels
            phase1_start = time.time()
            initial_labels = await self.phase1_initial_labels(cluster_data, var_lab)
            phase1_time = time.time() - phase1_start
            logger.info(f"Phase 1: Generated initial labels for {len(initial_labels)} clusters in {phase1_time:.2f}s")
            
            # Phase 2: Create hierarchy
            phase2_start = time.time()
            hierarchy = await self.phase2_create_hierarchy(cluster_data, initial_labels, var_lab)
            phase2_time = time.time() - phase2_start
            logger.info(f"Phase 2: Created hierarchical structure in {phase2_time:.2f}s")
            
            # Validate hierarchy quality
            quality_issues = self.validate_hierarchy_quality(hierarchy, cluster_data)
            
            if not quality_issues:
                logger.info("✅ Hierarchy passed all quality checks!")
                break
            else:
                logger.warning(f"⚠️  Found {len(quality_issues)} quality issues:")
                for issue in quality_issues:
                    logger.warning(f"   - {issue}")
                
                if iteration < max_iterations - 1:
                    # Try to refine the hierarchy
                    logger.info("Attempting to refine hierarchy...")
                    hierarchy = await self._refine_oversized_categories(hierarchy, cluster_data, initial_labels, var_lab)
                    
                    # Also check for small themes that should be merged
                    hierarchy = self._merge_small_themes(hierarchy, cluster_data)
                else:
                    logger.warning("Max iterations reached, proceeding with current hierarchy")
        
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
        logger.info(f"Total workflow time: {total_time:.2f}s")
        
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
        
        # Create mapping from theme label to theme_id for summary lookup
        theme_label_to_id = {theme.label: theme.node_id for theme in hierarchy.themes}
        
        # Process each cluster result
        label_models = []
        
        for result in cluster_results:
            # Create label submodels for each segment
            label_segments = []
            
            for segment in result.response_segment:
                if segment.micro_cluster is not None:
                    cluster_id = list(segment.micro_cluster.keys())[0]
                    
                    # Use explicit mappings to get labels
                    if cluster_id in hierarchy.cluster_to_path:
                        # Get labels from explicit mappings
                        theme_label = hierarchy.cluster_to_theme.get(cluster_id)
                        topic_label = hierarchy.cluster_to_topic.get(cluster_id)
                        
                        # Get the full path for code extraction
                        path = hierarchy.cluster_to_path[cluster_id]
                        path_parts = path.split('.')
                        
                        # Find nodes for code label
                        theme_id = path_parts[0]
                        theme_node = next((t for t in hierarchy.themes if t.node_id == theme_id), None)
                        
                        code_label = None
                        if theme_node and len(path_parts) >= 3:
                            topic_node = next((c for c in theme_node.children if c.node_id == '.'.join(path_parts[:2])), None)
                            if topic_node:
                                code_node = next((c for c in topic_node.children if c.node_id == path), None)
                                if code_node:
                                    code_label = code_node.label
                        
                        # Extract numeric IDs from hierarchical node_ids for model compatibility
                        topic_id = int(path_parts[1]) if len(path_parts) > 1 else None
                        code_id = int(path_parts[2]) if len(path_parts) > 2 else None
                        
                        label_segment = models.LabelSubmodel(
                            segment_id=segment.segment_id,
                            segment_response=segment.segment_response,
                            descriptive_code=segment.descriptive_code,
                            code_description=segment.code_description,
                            code_embedding=segment.code_embedding,
                            description_embedding=segment.description_embedding,
                            micro_cluster=segment.micro_cluster,
                            Theme={int(theme_id): theme_label} if theme_label else None,
                            Topic={topic_id: topic_label} if topic_label and topic_id is not None else None,
                            Keyword={code_id: code_label} if code_label and code_id is not None else None
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