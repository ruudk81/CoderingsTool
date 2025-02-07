"""
Thematic Labeller for Hierarchical Coding
==========================================
Implements 4-phase hierarchical labeling: Familiarization → Discovery → Assignment → Refinement
Creates Theme → Topic → Keyword hierarchy from survey response clusters.
"""

import asyncio
import hashlib
import json
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import instructor
from openai import AsyncOpenAI

from pydantic import BaseModel, Field

# Import from existing codebase
from config import DEFAULT_LABELLER_CONFIG, LabellerConfig
import models


# Pydantic Models for structured data
class ClusterLabel(BaseModel):
    """Initial cluster label from Phase 1"""
    cluster_id: int
    label: str = Field(description="Concise label (max 4 words)")
    description: str = Field(description="Detailed description of cluster theme")
    representatives: List[Tuple[str, float]] = Field(description="Representative descriptions with similarity scores")


class CodebookEntry(BaseModel):
    """Single entry in the hierarchical codebook"""
    id: str = Field(description="Unique identifier - int for themes, float-like for topics")
    numeric_id: float = Field(description="Numeric ID for model compatibility")
    level: int = Field(description="1 for theme, 2 for topic, 3 for subject")
    label: str = Field(description="Concise label (max 4 words)")
    description: str = Field(description="Clear description")
    parent_id: Optional[str] = Field(default=None, description="ID of parent")
    parent_numeric_id: Optional[float] = Field(default=None, description="Numeric ID of parent")
    direct_clusters: List[int] = Field(default_factory=list, description="Clusters assigned directly to this level")


class Codebook(BaseModel):
    """Complete hierarchical codebook from Phase 2"""
    survey_question: str
    themes: List[CodebookEntry]
    topics: List[CodebookEntry] 
    subjects: List[CodebookEntry]


class ThemeAssignment(BaseModel):
    """Assignment probabilities for a single cluster"""
    cluster_id: int
    theme_assignments: Dict[str, float] = Field(description="Theme ID to probability")
    topic_assignments: Dict[str, float] = Field(description="Topic ID to probability")
    subject_assignments: Dict[str, float] = Field(description="Subject ID to probability")


class BatchHierarchy(BaseModel):
    """Hierarchy from a batch of clusters for MapReduce"""
    batch_id: str
    themes: List[CodebookEntry]
    topics: List[CodebookEntry]
    subjects: List[CodebookEntry]


# Response models for structured LLM outputs
class FamiliarizationResponse(BaseModel):
    """Response from Phase 1 familiarization"""
    label: str = Field(description="Concise thematic label (max 4 words)")
    description: str = Field(description="Clear explanation of cluster theme")


class SingleDiscoveryResponse(BaseModel):
    """Response from Phase 2 single discovery"""
    themes: List[Dict[str, Any]] = Field(description="List of themes with their structure")


class MapDiscoveryResponse(BaseModel):
    """Response from Phase 2 map step"""
    themes: List[Dict[str, Any]]
    topics: List[Dict[str, Any]]
    subjects: List[Dict[str, Any]]


class AssignmentResponse(BaseModel):
    """Response from Phase 3 assignment"""
    theme_assignments: Dict[str, float]
    topic_assignments: Dict[str, float]
    subject_assignments: Dict[str, float]


class RefinementResponse(BaseModel):
    """Response from Phase 4 refinement"""
    quality_issues: List[Dict[str, str]]
    refined_labels: Dict[str, Dict[str, str]]


class ThematicLabeller:
    """Main orchestrator for thematic analysis using instructor"""
    
    def __init__(self, config: LabellerConfig = None, cache_manager=None, filename: str = None):
        # Force fresh config to avoid caching issues
        self.config = config or LabellerConfig()
        self.cache_manager = cache_manager
        self.filename = filename or "unknown"
        self.survey_question = ""
        
        # Setup async OpenAI client with instructor
        self.client = instructor.from_openai(
            AsyncOpenAI(api_key=self.config.api_key or None),
            mode=instructor.Mode.JSON
        )
        
        # Thresholds
        self.map_reduce_threshold = self.config.map_reduce_threshold
        self.batch_size = self.config.batch_size
        
    def _get_representatives(self, cluster: models.ClusterModel) -> List[Tuple[str, float]]:
        """Get top N representative descriptions with similarity scores"""
        embeddings = []
        descriptions = []
        
        for segment in cluster.response_segment:
            if segment.description_embedding is not None:
                embeddings.append(segment.description_embedding)
                descriptions.append(segment.segment_description)
        
        if not embeddings:
            return []
        
        # Calculate centroid
        embeddings_array = np.array(embeddings)
        centroid = np.mean(embeddings_array, axis=0)
        
        # Calculate similarities
        similarities = cosine_similarity(embeddings_array, centroid.reshape(1, -1)).flatten()
        
        # Get top N
        top_k = min(self.config.top_k_representatives, len(descriptions))
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [(descriptions[i], float(similarities[i])) for i in top_indices]
    
    def _extract_micro_clusters(self, cluster_models: List[models.ClusterModel]) -> Dict[int, Dict]:
        """Extract micro-cluster information from cluster models"""
        micro_clusters = {}
        
        for model in cluster_models:
            if model.response_segment:
                for segment in model.response_segment:
                    if segment.micro_cluster:
                        cluster_id = list(segment.micro_cluster.keys())[0]
                        if cluster_id not in micro_clusters:
                            micro_clusters[cluster_id] = {
                                'descriptions': [],
                                'embeddings': [],
                                'codes': []
                            }
                        
                        micro_clusters[cluster_id]['descriptions'].append(segment.segment_description)
                        if segment.description_embedding is not None:
                            micro_clusters[cluster_id]['embeddings'].append(segment.description_embedding)
                        micro_clusters[cluster_id]['codes'].append(segment.segment_label)
        
        return micro_clusters
    
    async def _invoke_with_retries(self, prompt: str, response_model: BaseModel, max_retries: int = None) -> Any:
        """Invoke LLM with retry logic"""
        if max_retries is None:
            max_retries = self.config.max_retries
            
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_model=response_model,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    seed=self.config.seed if hasattr(self.config, 'seed') else 42
                )
                return response
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    # Use min to cap the delay at 10 seconds
                    delay = min(self.config.retry_delay * (attempt + 1), 10)
                    await asyncio.sleep(delay)
                    print(f"Retry {attempt + 1}/{max_retries} after error: {str(e)}")
        
        raise RuntimeError(f"Failed after {max_retries} attempts. Last error: {str(last_error)}")
    
    def _generate_cache_key(self, data: Any) -> str:
        """Generate deterministic cache key"""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()[:12]
    
    async def process_hierarchy_async(self, cluster_models: List[models.ClusterModel], survey_question: str) -> List[models.LabelModel]:
        """Main async processing method"""
        self.survey_question = survey_question
        
        print("\n🔄 Starting hierarchical labeling process...")
        
        # Extract micro-clusters
        micro_clusters = self._extract_micro_clusters(cluster_models)
        print(f"📊 Found {len(micro_clusters)} unique micro-clusters")
        
        # Phase 1: Familiarization
        print("\n📝 Phase 1: Familiarization - Labeling all clusters...")
        import time
        phase1_start = time.time()
        labeled_clusters = await self._phase1_familiarization(micro_clusters)
        phase1_time = time.time() - phase1_start
        print(f"  ✓ Phase 1 completed in {phase1_time:.1f} seconds")
        print(f"  Generated {len(labeled_clusters)} cluster labels")
        
        # Check cache for Phase 2
        cache_key = f"codebook_{self._generate_cache_key(labeled_clusters)}"
        cached_codebook = None
        if self.cache_manager:
            cached_codebook = self.cache_manager.load_intermediate_data(
                self.filename, cache_key, expected_type=dict
            )
        
        if cached_codebook:
            print("📦 Using cached codebook")
            codebook = Codebook(**cached_codebook)
        else:
            # Phase 2: Theme Discovery
            print("\n🔍 Phase 2: Theme Discovery - Building codebook...")
            phase2_start = time.time()
            
            # For discovery, we want to see ALL clusters together for context
            # Only use MapReduce for very large datasets (>100 clusters)
            use_single_call_threshold = 100  # Increased from 30
            
            if len(labeled_clusters) <= use_single_call_threshold:
                print(f"  Using single call to see all {len(labeled_clusters)} clusters together")
                print(f"  This ensures the LLM has full context for creating coherent themes")
                codebook = await self._phase2_discovery_single(labeled_clusters)
            else:
                print(f"  Using MapReduce for {len(labeled_clusters)} clusters (>{use_single_call_threshold})")
                print(f"  Warning: This may result in less coherent themes due to limited context")
                codebook = await self._phase2_discovery_mapreduce(labeled_clusters)
            phase2_time = time.time() - phase2_start
            print(f"  ✓ Phase 2 completed in {phase2_time:.1f} seconds")
            
            # Cache codebook
            if self.cache_manager:
                self.cache_manager.cache_intermediate_data(
                    codebook.model_dump(), self.filename, cache_key
                )
        
        # Phase 3: Assignment
        print("\n🎯 Phase 3: Assignment - Assigning themes to clusters...")
        phase3_start = time.time()
        assignments = await self._phase3_assignment(labeled_clusters, codebook)
        phase3_time = time.time() - phase3_start
        print(f"  ✓ Phase 3 completed in {phase3_time:.1f} seconds")
        
        # Save codebook state before Phase 4 for analysis
        self.codebook_before_phase4 = codebook
        
        # Phase 4: Refinement
        print("\n✨ Phase 4: Refinement - Finalizing labels...")
        phase4_start = time.time()
        final_labels = await self._phase4_refinement(labeled_clusters, assignments, codebook)
        phase4_time = time.time() - phase4_start
        print(f"  ✓ Phase 4 completed in {phase4_time:.1f} seconds")
        
        # Save final labels for analysis
        self.final_labels = final_labels
        
        # Apply to original cluster models
        print("\n✅ Applying hierarchy to responses...")
        result = self._apply_hierarchy_to_responses(cluster_models, final_labels, codebook)
        
        # Print diagnostics for missing clusters
        self._print_assignment_diagnostics(final_labels, micro_clusters)
        
        print("\n🎉 Hierarchical labeling complete!")
        self._print_summary(codebook)
        
        # Save final codebook for analysis
        self.codebook = codebook
        
        return result
    
    def process_hierarchy(self, cluster_models: List[models.ClusterModel], survey_question: str) -> List[models.LabelModel]:
        """Sync wrapper for async processing"""
        # Handle nested event loops
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except ImportError:
            pass
        
        return asyncio.run(self.process_hierarchy_async(cluster_models, survey_question))
    
    async def _phase1_familiarization(self, micro_clusters: Dict[int, Dict]) -> List[ClusterLabel]:
        """Phase 1: Label and describe each cluster"""
        labeled_clusters = []
        
        # Import prompt
        from prompts import PHASE1_FAMILIARIZATION_PROMPT
        
        # Process clusters (can be done concurrently)
        tasks = []
        for cluster_id, cluster_data in sorted(micro_clusters.items()):
            # Calculate representatives
            if cluster_data['embeddings']:
                embeddings = np.array(cluster_data['embeddings'])
                centroid = np.mean(embeddings, axis=0)
                
                similarities = cosine_similarity(embeddings, centroid.reshape(1, -1)).flatten()
                top_k = min(self.config.top_k_representatives, len(cluster_data['descriptions']))
                top_indices = np.argsort(similarities)[-top_k:][::-1]
                
                representatives = [(cluster_data['descriptions'][i], float(similarities[i])) 
                                 for i in top_indices]
            else:
                # Fallback if no embeddings
                representatives = [(desc, 1.0) for desc in cluster_data['descriptions'][:3]]
            
            # Format for prompt
            reps_text = "\n".join([f"{i+1}. {desc} (similarity: {sim:.3f})" 
                                  for i, (desc, sim) in enumerate(representatives)])
            
            prompt = PHASE1_FAMILIARIZATION_PROMPT.format(
                survey_question=self.survey_question,
                cluster_id=cluster_id,
                cluster_size=len(cluster_data['descriptions']),
                representatives=reps_text,
                language=self.config.language
            )
            
            task = self._invoke_with_retries(prompt, FamiliarizationResponse)
            tasks.append((cluster_id, representatives, task))
        
        # Execute tasks concurrently in batches
        print(f"  Processing {len(tasks)} clusters with max {self.config.concurrent_requests} concurrent requests...")
        
        for i in range(0, len(tasks), self.config.concurrent_requests):
            batch = tasks[i:i + self.config.concurrent_requests]
            batch_results = await asyncio.gather(*[task for _, _, task in batch], return_exceptions=True)
            
            for (cluster_id, representatives, _), result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    print(f"Error labeling cluster {cluster_id}: {str(result)}")
                    labeled_clusters.append(ClusterLabel(
                        cluster_id=cluster_id,
                        label="UNLABELED",
                        description="Failed to generate label",
                        representatives=representatives
                    ))
                else:
                    labeled_clusters.append(ClusterLabel(
                        cluster_id=cluster_id,
                        label=result.label,
                        description=result.description,
                        representatives=representatives
                    ))
        
        return labeled_clusters
    
    async def _phase2_discovery_single(self, labeled_clusters: List[ClusterLabel]) -> Codebook:
        """Phase 2: Single call for theme discovery"""
        from prompts import PHASE2_DISCOVERY_SINGLE_PROMPT
        
        # Format clusters for prompt
        cluster_summaries = []
        for cluster in sorted(labeled_clusters, key=lambda x: x.cluster_id):
            summary = f"Cluster {cluster.cluster_id}: {cluster.label}\n"
            summary += f"Description: {cluster.description}\n"
            summary += "Representatives:\n"
            for i, (desc, _) in enumerate(cluster.representatives[:2], 1):
                summary += f"  {i}. {desc}\n"
            cluster_summaries.append(summary)
        
        all_cluster_ids = ", ".join(str(c.cluster_id) for c in labeled_clusters)
        
        prompt = PHASE2_DISCOVERY_SINGLE_PROMPT.format(
            survey_question=self.survey_question,
            language=self.config.language,
            cluster_summaries="\n".join(cluster_summaries),
            all_cluster_ids=all_cluster_ids
        )
        
        result = await self._invoke_with_retries(prompt, SingleDiscoveryResponse)
        
        # Parse result into Codebook - pass labeled_clusters for proper subject labels
        return self._parse_codebook(result.model_dump(), labeled_clusters)
    
    async def _phase2_discovery_mapreduce(self, labeled_clusters: List[ClusterLabel]) -> Codebook:
        """Phase 2: MapReduce for theme discovery"""
        from prompts import PHASE2_DISCOVERY_MAP_PROMPT, PHASE2_DISCOVERY_REDUCE_PROMPT
        
        # Create batches
        batches = []
        sorted_clusters = sorted(labeled_clusters, key=lambda x: x.cluster_id)
        
        for i in range(0, len(sorted_clusters), self.batch_size):
            batch = sorted_clusters[i:i + self.batch_size]
            batches.append(batch)
        
        print(f"  Created {len(batches)} batches of ~{self.batch_size} clusters each")
        
        # Map phase - process batches concurrently
        print(f"  Processing {len(batches)} batches concurrently...")
        batch_tasks = []
        
        for batch_idx, batch in enumerate(batches):
            cluster_summaries = []
            for cluster in batch:
                summary = f"Cluster {cluster.cluster_id}: {cluster.label} - {cluster.description}"
                cluster_summaries.append(summary)
            
            prompt = PHASE2_DISCOVERY_MAP_PROMPT.format(
                survey_question=self.survey_question,
                language=self.config.language,
                batch_id=f"batch_{batch_idx:03d}",
                batch_clusters="\n".join(cluster_summaries)
            )
            
            task = self._invoke_with_retries(prompt, MapDiscoveryResponse)
            batch_tasks.append((batch_idx, task))
        
        # Execute all batch tasks concurrently
        batch_results = await asyncio.gather(*[task for _, task in batch_tasks], return_exceptions=True)
        
        # Process results
        batch_hierarchies = []
        for (batch_idx, _), result in zip(batch_tasks, batch_results):
            if isinstance(result, Exception):
                print(f"Error processing batch {batch_idx}: {str(result)}")
                continue
            
            # Parse themes with proper numeric IDs and levels
            themes = []
            for t in result.themes:
                themes.append(CodebookEntry(
                    id=t['id'],
                    numeric_id=float(t['id'].replace('temp_', '')),  # Convert temp_1 to 1.0
                    level=1,
                    label=t['label'],
                    description=t.get('description', ''),
                    direct_clusters=t.get('direct_clusters', [])
                ))
            
            # Parse topics
            topics = []
            for t in result.topics:
                # Extract numeric ID from temp_1.1 format
                numeric_parts = t['id'].replace('temp_', '').split('.')
                numeric_id = float('.'.join(numeric_parts))
                topics.append(CodebookEntry(
                    id=t['id'],
                    numeric_id=numeric_id,
                    level=2,
                    label=t['label'],
                    description=t.get('description', ''),
                    parent_id=t.get('parent_id'),
                    parent_numeric_id=float(t.get('parent_id', '').replace('temp_', '')) if t.get('parent_id') else None,
                    direct_clusters=t.get('direct_clusters', [])
                ))
            
            # Parse subjects
            subjects = []
            for s in result.subjects:
                # For subjects, use the cluster IDs directly as numeric IDs
                subjects.append(CodebookEntry(
                    id=s['id'],
                    numeric_id=float(s['id'].replace('temp_', '').replace('.', '')),
                    level=3,
                    label=s['label'],
                    description=s.get('description', ''),
                    parent_id=s.get('parent_id'),
                    parent_numeric_id=float('.'.join(s.get('parent_id', '').replace('temp_', '').split('.')[:2])) if s.get('parent_id') else None,
                    direct_clusters=s.get('direct_clusters', [])
                ))
            
                batch_hierarchies.append(BatchHierarchy(
                    batch_id=f"batch_{batch_idx:03d}",
                    themes=themes,
                    topics=topics,
                    subjects=subjects
                ))
        
        # Reduce phase
        print(f"  Reducing {len(batch_hierarchies)} batch hierarchies...")
        
        # Format hierarchies for reduction
        hierarchies_text = []
        for h in batch_hierarchies:
            h_dict = {
                "batch_id": h.batch_id,
                "themes": [t.model_dump() for t in h.themes],
                "topics": [t.model_dump() for t in h.topics],
                "subjects": [s.model_dump() for s in h.subjects]
            }
            hierarchies_text.append(json.dumps(h_dict, indent=2))
        
        prompt = PHASE2_DISCOVERY_REDUCE_PROMPT.format(
            survey_question=self.survey_question,
            language=self.config.language,
            hierarchies="\n\n".join(hierarchies_text)
        )
        
        result = await self._invoke_with_retries(prompt, MapDiscoveryResponse)
        return self._parse_codebook(result.model_dump(), labeled_clusters)
    
    def _parse_codebook(self, result: Dict, labeled_clusters: List[ClusterLabel]) -> Codebook:
        """Parse JSON result into Codebook structure with proper numeric IDs"""
        # Create cluster label lookup for subjects
        cluster_label_lookup = {c.cluster_id: c.label for c in labeled_clusters}
        themes = []
        topics = []
        subjects = []
        
        # Check if this is from MapReduce (has separate theme/topic/subject lists)
        if 'themes' in result and 'topics' in result and 'subjects' in result:
            # Handle MapReduce format
            for theme_data in result['themes']:
                theme_id = theme_data['id']
                # Extract numeric ID (handle both "1" and "temp_1" formats)
                numeric_str = theme_id.replace('temp_', '')
                theme_numeric_id = float(numeric_str)
                
                theme = CodebookEntry(
                    id=str(int(theme_numeric_id)),  # Normalize to "1", "2", etc.
                    numeric_id=theme_numeric_id,
                    level=1,
                    label=theme_data['label'],
                    description=theme_data.get('description', ''),
                    direct_clusters=theme_data.get('direct_clusters', [])
                )
                themes.append(theme)
            
            # Process topics
            for topic_data in result['topics']:
                topic_id = topic_data['id']
                # Extract numeric ID
                numeric_str = topic_id.replace('temp_', '')
                topic_numeric_id = float(numeric_str)
                
                # Find parent theme numeric ID
                parent_id = topic_data.get('parent_id', '')
                parent_numeric_str = parent_id.replace('temp_', '')
                parent_numeric_id = float(parent_numeric_str) if parent_numeric_str else None
                
                topic = CodebookEntry(
                    id=numeric_str,  # Keep as "1.1", "1.2", etc.
                    numeric_id=topic_numeric_id,
                    level=2,
                    label=topic_data['label'],
                    description=topic_data.get('description', ''),
                    parent_id=str(int(parent_numeric_id)) if parent_numeric_id else None,
                    parent_numeric_id=parent_numeric_id,
                    direct_clusters=topic_data.get('direct_clusters', [])
                )
                topics.append(topic)
            
            # Process subjects - use individual cluster labels
            for subject_data in result['subjects']:
                # For subjects, the direct_clusters contain the actual micro-cluster IDs
                for cluster_id in subject_data.get('direct_clusters', []):
                    # Find parent topic
                    parent_id = subject_data.get('parent_id', '')
                    parent_numeric_str = parent_id.replace('temp_', '')
                    parent_numeric_id = float(parent_numeric_str) if parent_numeric_str else None
                    
                    # Use the actual cluster label from Phase 1, not the group label
                    cluster_label = cluster_label_lookup.get(cluster_id, f"Cluster {cluster_id}")
                    subject = CodebookEntry(
                        id=str(cluster_id),
                        numeric_id=float(cluster_id),
                        level=3,
                        label=cluster_label,  # Use individual cluster label!
                        description=subject_data.get('description', ''),
                        parent_id=parent_numeric_str if parent_numeric_str else None,
                        parent_numeric_id=parent_numeric_id,
                        direct_clusters=[cluster_id]
                    )
                    subjects.append(subject)
        
        # Handle single-call format (nested structure)
        elif 'themes' in result:
            for theme_idx, theme_data in enumerate(result['themes'], 1):
                # Use integer IDs for themes
                theme_id = str(theme_idx)
                theme_numeric_id = float(theme_idx)
                
                theme = CodebookEntry(
                    id=theme_id,
                    numeric_id=theme_numeric_id,
                    level=1,
                    label=theme_data['label'],
                    description=theme_data.get('description', ''),
                    direct_clusters=theme_data.get('direct_clusters', [])
                )
                themes.append(theme)
                
                # Extract topics
                if 'topics' in theme_data:
                    for topic_idx, topic_data in enumerate(theme_data['topics'], 1):
                        # Use float IDs for topics (e.g., 1.1, 1.2)
                        topic_id = f"{theme_idx}.{topic_idx}"
                        topic_numeric_id = float(topic_id)
                        
                        topic = CodebookEntry(
                            id=topic_id,
                            numeric_id=topic_numeric_id,
                            level=2,
                            label=topic_data['label'],
                            description=topic_data.get('description', ''),
                            parent_id=theme_id,
                            parent_numeric_id=theme_numeric_id,
                            direct_clusters=topic_data.get('direct_clusters', [])
                        )
                        topics.append(topic)
                        
                        # Extract subjects (keywords) - use individual cluster labels, not group labels
                        if 'subjects' in topic_data:
                            for subject_data in topic_data['subjects']:
                                # Subjects map to micro_clusters directly
                                for cluster_id in subject_data.get('micro_clusters', []):
                                    # Use the actual cluster label from Phase 1, not the group label
                                    cluster_label = cluster_label_lookup.get(cluster_id, f"Cluster {cluster_id}")
                                    subject = CodebookEntry(
                                        id=str(cluster_id),
                                        numeric_id=float(cluster_id),
                                        level=3,
                                        label=cluster_label,  # Use individual cluster label!
                                        description=subject_data.get('description', ''),
                                        parent_id=topic_id,
                                        parent_numeric_id=topic_numeric_id,
                                        direct_clusters=[cluster_id]
                                    )
                                    subjects.append(subject)
        
        # Remove duplicates and clean up hierarchy
        themes = self._remove_duplicate_entries(themes, "themes")
        topics = self._remove_duplicate_entries(topics, "topics") 
        subjects = self._remove_duplicate_entries(subjects, "subjects")
        
        return Codebook(
            survey_question=self.survey_question,
            themes=themes,
            topics=topics,
            subjects=subjects
        )
    
    def _remove_duplicate_entries(self, entries: List[CodebookEntry], entry_type: str) -> List[CodebookEntry]:
        """Remove duplicate entries based on label, keeping the one with more clusters"""
        if not entries:
            return entries
            
        # Group by label
        label_groups = {}
        for entry in entries:
            label = entry.label.strip().lower()
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(entry)
        
        # Keep best entry from each group
        unique_entries = []
        total_duplicates_removed = 0
        
        for label, group in label_groups.items():
            if len(group) == 1:
                unique_entries.append(group[0])
            else:
                # Multiple entries with same label - keep the one with most clusters
                best_entry = max(group, key=lambda e: len(e.direct_clusters))
                unique_entries.append(best_entry)
                group_duplicates = len(group) - 1
                total_duplicates_removed += group_duplicates
                
                removed_ids = [e.id for e in group if e != best_entry]
                print(f"    🧹 Removed {group_duplicates} duplicate {entry_type} with label '{group[0].label}': {removed_ids} (keeping {best_entry.id})")
        
        if total_duplicates_removed > 0:
            print(f"    📊 Total duplicates removed: {total_duplicates_removed} {entry_type}")
        
        return unique_entries
    
    async def _phase3_assignment(self, labeled_clusters: List[ClusterLabel], codebook: Codebook) -> List[ThemeAssignment]:
        """Phase 3: Assign clusters to themes with probabilities"""
        from prompts import PHASE3_ASSIGNMENT_PROMPT
        
        assignments = []
        
        # Format codebook for prompt
        codebook_text = self._format_codebook_for_prompt(codebook)
        
        # Process each cluster
        tasks = []
        for cluster in labeled_clusters:
            # Format representatives
            reps_text = "\n".join([f"- {desc}" for desc, _ in cluster.representatives[:3]])
            
            prompt = PHASE3_ASSIGNMENT_PROMPT.format(
                survey_question=self.survey_question,
                language=self.config.language,
                cluster_id=cluster.cluster_id,
                cluster_label=cluster.label,
                cluster_description=cluster.description,
                representatives=reps_text,
                codebook=codebook_text
            )
            
            task = self._invoke_with_retries(prompt, AssignmentResponse)
            tasks.append((cluster.cluster_id, task))
        
        # Execute tasks concurrently in batches
        print(f"  Processing {len(tasks)} assignments with max {self.config.concurrent_requests} concurrent requests...")
        
        for i in range(0, len(tasks), self.config.concurrent_requests):
            batch = tasks[i:i + self.config.concurrent_requests]
            batch_results = await asyncio.gather(*[task for _, task in batch], return_exceptions=True)
            
            for (cluster_id, _), result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    print(f"Error assigning cluster {cluster_id}: {str(result)}")
                    assignments.append(ThemeAssignment(
                        cluster_id=cluster_id,
                        theme_assignments={"other": 1.0},
                        topic_assignments={"other": 1.0},
                        subject_assignments={"other": 1.0}
                    ))
                else:
                    assignments.append(ThemeAssignment(
                        cluster_id=cluster_id,
                        theme_assignments=result.theme_assignments,
                        topic_assignments=result.topic_assignments,
                        subject_assignments=result.subject_assignments
                    ))
        
        return assignments
    
    def _format_codebook_for_prompt(self, codebook: Codebook) -> str:
        """Format codebook in readable hierarchy"""
        lines = []
        
        for theme in codebook.themes:
            lines.append(f"\nTHEME {theme.id}: {theme.label}")
            if theme.description:
                lines.append(f"  Description: {theme.description}")
            if theme.direct_clusters:
                lines.append(f"  Direct clusters: {theme.direct_clusters}")
            
            # Find related topics
            related_topics = [t for t in codebook.topics if t.parent_id == theme.id]
            for topic in related_topics:
                lines.append(f"  TOPIC {topic.id}: {topic.label}")
                if topic.description:
                    lines.append(f"    Description: {topic.description}")
                if topic.direct_clusters:
                    lines.append(f"    Direct clusters: {topic.direct_clusters}")
                
                # Find related subjects
                related_subjects = [s for s in codebook.subjects if s.parent_id == topic.id]
                for subject in related_subjects:
                    lines.append(f"    SUBJECT {subject.id}: {subject.label}")
                    if subject.description:
                        lines.append(f"      Description: {subject.description}")
                    if subject.direct_clusters:
                        lines.append(f"      Clusters: {subject.direct_clusters}")
        
        return "\n".join(lines)
    
    async def _phase4_refinement(self, labeled_clusters: List[ClusterLabel], 
                                assignments: List[ThemeAssignment], 
                                codebook: Codebook) -> Dict[int, Dict]:
        """Phase 4: Refine and finalize labels"""
        final_labels = {}
        
        # Create lookup dictionaries
        cluster_lookup = {c.cluster_id: c for c in labeled_clusters}
        assignment_lookup = {a.cluster_id: a for a in assignments}
        
        # Process each cluster to create initial assignments
        for cluster_id in cluster_lookup:
            cluster = cluster_lookup[cluster_id]
            assignment = assignment_lookup.get(cluster_id)
            
            if not assignment:
                # Handle missing assignments - assign to 'other'
                print(f"    ⚠️  No assignment found for cluster {cluster_id}, assigning to 'other'")
                final_labels[cluster_id] = {
                    'label': cluster.label,
                    'description': cluster.description,
                    'theme': ('other', 0.0),
                    'topic': ('other', 0.0),
                    'subject': ('other', 0.0)
                }
                continue
            
            # Get best assignments
            theme_id, theme_prob = self._get_best_assignment(assignment.theme_assignments)
            topic_id, topic_prob = self._get_best_assignment(assignment.topic_assignments)
            subject_id, subject_prob = self._get_best_assignment(assignment.subject_assignments)
            
            final_labels[cluster_id] = {
                'label': cluster.label,
                'description': cluster.description,
                'theme': (theme_id, theme_prob),
                'topic': (topic_id, topic_prob),
                'subject': (subject_id, subject_prob)
            }
        
        # Optional LLM-based refinement
        print(f"🔧 DEBUG: self.config.use_llm_refinement = {self.config.use_llm_refinement}")
        if self.config.use_llm_refinement:
            print("    🔄 Performing LLM-based label refinement...")
            refined_labels = await self._llm_refinement(final_labels, codebook)
            return refined_labels
        else:
            print("    📝 Using direct assignments (LLM refinement disabled)")
        
        return final_labels
    
    async def _llm_refinement(self, final_labels: Dict[int, Dict], codebook: Codebook) -> Dict[int, Dict]:
        """Optional LLM-based refinement of labels"""
        from prompts import PHASE4_REFINEMENT_PROMPT
        
        # Create summary of current assignments for LLM review
        assignment_summary = []
        for cluster_id, labels in sorted(final_labels.items()):
            summary = f"Cluster {cluster_id}: {labels['label']}\n"
            summary += f"  Theme: {labels['theme'][0]} (prob: {labels['theme'][1]:.2f})\n"
            summary += f"  Topic: {labels['topic'][0]} (prob: {labels['topic'][1]:.2f})"
            assignment_summary.append(summary)
        
        # Create combined hierarchy with assignments as expected by the prompt
        hierarchy_with_assignments = f"CURRENT ASSIGNMENTS:\n{chr(10).join(assignment_summary)}\n\nCODEBOOK STRUCTURE:\n{self._format_codebook_for_prompt(codebook)}"
        
        prompt = PHASE4_REFINEMENT_PROMPT.format(
            survey_question=self.survey_question,
            language=self.config.language,
            hierarchy_with_assignments=hierarchy_with_assignments
        )
        
        try:
            result = await self._invoke_with_retries(prompt, RefinementResponse)
            
            # Apply refinements if any - but DON'T modify the codebook in place
            if result.refined_labels:
                refinement_count = 0
                
                print(f"    🔧 DEBUG: Phase 4 refinement disabled to prevent hierarchy destruction")
                print(f"    📊 Received refinements for themes, topics, and subjects")
                print(f"    ⚠️  Refinements NOT applied to preserve hierarchy structure")
                
                # DEBUG: Show what would have been refined
                if 'themes' in result.refined_labels and result.refined_labels['themes']:
                    print(f"    📝 Would refine {len(result.refined_labels['themes'])} theme labels")
                
                if 'topics' in result.refined_labels and result.refined_labels['topics']:
                    print(f"    📝 Would refine {len(result.refined_labels['topics'])} topic labels")
                
                if 'subjects' in result.refined_labels and result.refined_labels['subjects']:
                    print(f"    📝 Would refine {len(result.refined_labels['subjects'])} subject labels")
                
                print(f"    💡 Hierarchy preservation prioritized over label refinement")
            
            if result.quality_issues:
                print(f"    ⚠️  Found {len(result.quality_issues)} quality issues (noted but not applied)")
        
        except Exception as e:
            print(f"    ❌ LLM refinement failed: {str(e)}")
        
        # Return original final_labels without modification to preserve hierarchy
        return final_labels
    
    def _get_best_assignment(self, assignments: Dict[str, float], threshold: float = None) -> Tuple[str, float]:
        """Get highest probability assignment above threshold"""
        if threshold is None:
            threshold = self.config.assignment_threshold
            
        if not assignments:
            return ("other", 0.0)
        
        # Sort by probability
        sorted_assignments = sorted(assignments.items(), key=lambda x: x[1], reverse=True)
        best_id, best_prob = sorted_assignments[0]
        
        # Check threshold
        if best_prob < threshold:
            return ("other", best_prob)
        
        return (best_id, best_prob)
    
    def _apply_hierarchy_to_responses(self, cluster_models: List[models.ClusterModel], 
                                     final_labels: Dict[int, Dict], 
                                     codebook: Codebook) -> List[models.LabelModel]:
        """Apply hierarchy to original response models with proper model compatibility"""
        # Create string ID lookups for assignment matching
        theme_str_lookup = {t.id: t for t in codebook.themes}
        topic_str_lookup = {t.id: t for t in codebook.topics}
        
        label_models = []
        
        for cluster_model in cluster_models:
            # Convert to LabelModel using the to_model method
            label_model = cluster_model.to_model(models.LabelModel)
            
            # Force rebuild model to clear any cached schema
            if hasattr(label_model, 'model_rebuild'):
                label_model.model_rebuild()
            
            # Generate a summary for the model (optional)
            segment_count = len(label_model.response_segment) if label_model.response_segment else 0
            label_model.summary = f"Response with {segment_count} segments analyzed"
            
            # Apply hierarchy to segments
            if label_model.response_segment:
                for segment in label_model.response_segment:
                    if segment.micro_cluster:
                        cluster_id = list(segment.micro_cluster.keys())[0]
                        
                        if cluster_id in final_labels:
                            labels = final_labels[cluster_id]
                            
                            # Apply Theme (Dict[int, str])
                            theme_id_str, _ = labels['theme']
                            if theme_id_str in theme_str_lookup:
                                theme = theme_str_lookup[theme_id_str]
                                theme_id_int = int(theme.numeric_id)
                                segment.Theme = {theme_id_int: f"{theme.label}: {theme.description}"}
                            elif theme_id_str == "other":
                                segment.Theme = {999: "Other: Unclassified"}
                            
                            # Apply Topic (Dict[float, str])
                            topic_id_str, _ = labels['topic']
                            if topic_id_str in topic_str_lookup:
                                topic = topic_str_lookup[topic_id_str]
                                topic_id_float = topic.numeric_id  # Already a float from model
                                # Debug: Check the Topic field type
                                topic_field_type = segment.__class__.__annotations__.get('Topic', 'NOT_FOUND')
                                if 'int' in str(topic_field_type):
                                    print(f"🚨 DEBUG: Topic field still expects int! Type: {topic_field_type}")
                                segment.Topic = {topic_id_float: f"{topic.label}: {topic.description}"}
                            elif topic_id_str == "other":
                                segment.Topic = {99.9: "Other: Unclassified"}
                            
                            # Apply Keyword (Dict[int, str]) - use cluster label
                            cluster_label = labels['label']
                            segment.Keyword = {cluster_id: cluster_label}
                        else:
                            # Handle unmapped clusters
                            segment.Theme = {999: "Other: Unmapped cluster"}
                            segment.Topic = {99.9: "Other: Unmapped cluster"}
                            # For unmapped clusters, just use the cluster ID
                            segment.Keyword = {cluster_id: f"Cluster {cluster_id}"}
            
            label_models.append(label_model)
        
        return label_models
    
    def _print_summary(self, codebook: Codebook):
        """Print summary of the generated hierarchy"""
        print(f"\n📊 Codebook Summary:")
        print(f"  - {len(codebook.themes)} Themes")
        print(f"  - {len(codebook.topics)} Topics")
        print(f"  - {len(codebook.subjects)} Subjects")
        
        # Count total assigned clusters
        total_clusters = set()
        for theme in codebook.themes:
            total_clusters.update(theme.direct_clusters)
        for topic in codebook.topics:
            total_clusters.update(topic.direct_clusters)
        for subject in codebook.subjects:
            total_clusters.update(subject.direct_clusters)
        
        print(f"  - {len(total_clusters)} clusters assigned")
        
        # Print the full codebook
        self._display_full_codebook(codebook)
    
    def _display_full_codebook(self, codebook: Codebook):
        """Display the full hierarchical codebook with all labels"""
        print("\n" + "="*80)
        print("📚 FULL CODEBOOK HIERARCHY")
        print("="*80)
        
        # Display themes
        for theme in sorted(codebook.themes, key=lambda x: x.numeric_id):
            print(f"\n🎯 THEME {theme.id}: {theme.label}")
            if theme.description:
                print(f"   Description: {theme.description}")
            if theme.direct_clusters:
                print(f"   Direct clusters: {theme.direct_clusters}")
            
            # Find related topics
            related_topics = [t for t in codebook.topics if t.parent_id == theme.id]
            if related_topics:
                for topic in sorted(related_topics, key=lambda x: x.numeric_id):
                    print(f"\n   📍 TOPIC {topic.id}: {topic.label}")
                    if topic.description:
                        print(f"      Description: {topic.description}")
                    if topic.direct_clusters:
                        print(f"      Direct clusters: {topic.direct_clusters}")
                    
                    # Find related subjects
                    related_subjects = [s for s in codebook.subjects if s.parent_id == topic.id]
                    if related_subjects:
                        for subject in sorted(related_subjects, key=lambda x: x.numeric_id):
                            print(f"\n      🔸 SUBJECT {subject.id}: {subject.label}")
                            if subject.description:
                                print(f"         Description: {subject.description}")
                            if subject.direct_clusters:
                                print(f"         Clusters: {subject.direct_clusters}")
        
        print("\n" + "="*80)
    
    def _print_assignment_diagnostics(self, final_labels: Dict[int, Dict], micro_clusters: Dict[int, Dict]):
        """Print diagnostics about cluster assignments"""
        print("\n🔍 Assignment Diagnostics:")
        
        # Find all cluster IDs
        all_cluster_ids = set(micro_clusters.keys())
        assigned_cluster_ids = set(final_labels.keys())
        missing_cluster_ids = all_cluster_ids - assigned_cluster_ids
        
        print(f"  - Total clusters found: {len(all_cluster_ids)}")
        print(f"  - Clusters assigned: {len(assigned_cluster_ids)}")
        
        if missing_cluster_ids:
            print(f"  - Missing clusters: {sorted(missing_cluster_ids)}")
            print("  ⚠️  Some clusters were not assigned in Phase 3")
        else:
            print("  ✅ All clusters were assigned")
        
        # Count assignment types
        theme_other = sum(1 for labels in final_labels.values() if labels['theme'][0] == 'other')
        topic_other = sum(1 for labels in final_labels.values() if labels['topic'][0] == 'other')
        
        if theme_other > 0 or topic_other > 0:
            print(f"  - Clusters assigned to 'other': {theme_other} themes, {topic_other} topics")


# =============================================================================
# USAGE / TEST SECTION
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.path.append('..')
    
    from utils.cacheManager import CacheManager
    from utils import dataLoader
    from config import CacheConfig, DEFAULT_LABELLER_CONFIG
    import models
    
    # Initialize cache manager
    cache_config = CacheConfig()
    cache_manager = CacheManager(cache_config)
    
    # Test data configuration
    filename = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
    var_name = "Q20"
    
    # Load clusters from cache
    cluster_results = cache_manager.load_from_cache(filename, "clusters", models.ClusterModel)
    
    if cluster_results:
        print(f"✅ Loaded {len(cluster_results)} clustered responses from cache")
        
        # Count unique micro-clusters
        unique_clusters = set()
        for result in cluster_results:
            if result.response_segment:
                for segment in result.response_segment:
                    if segment.micro_cluster:
                        cluster_id = list(segment.micro_cluster.keys())[0]
                        unique_clusters.add(cluster_id)
        
        print(f"📊 Found {len(unique_clusters)} unique micro-clusters to label")
        
        # Get variable label
        data_loader = dataLoader.DataLoader()
        var_lab = data_loader.get_varlab(filename=filename, var_name=var_name)
        print(f"📌 Survey question: {var_lab}")
        
        print("\n=== Running Thematic Labelling ===")
        print("This will execute 4 phases:")
        print("1. Familiarization - Label each micro-cluster")
        print("2. Theme Discovery - Build hierarchical codebook")
        print("3. Assignment - Assign clusters to hierarchy")
        print("4. Refinement - Finalize labels")
        print("=" * 50)
        
        # Initialize thematic labeller with fresh config to avoid caching
        fresh_config = LabellerConfig()
        print(f"🔧 Config check: use_llm_refinement = {fresh_config.use_llm_refinement}")
        labeller = ThematicLabeller(
            config=fresh_config,
            cache_manager=cache_manager,
            filename=filename
        )
        
        # Run the hierarchical labeling process
        try:
            labeled_results = labeller.process_hierarchy(
                cluster_models=cluster_results,
                survey_question=var_lab
            )
            
            print(f"\n✅ Successfully labeled {len(labeled_results)} responses")
            
            # Save to cache
            cache_manager.save_to_cache(labeled_results, filename, "labels")
            print("💾 Saved labeled results to cache")
            
            # Print sample results
            print("\n=== Sample Labeled Response ===")
            for result in labeled_results[:1]:  # Show first result
                print(f"Response ID: {result.respondent_id}")
                if result.response_segment:
                    for i, segment in enumerate(result.response_segment[:2]):  # Show first 2 segments
                        print(f"\nSegment {i+1}:")
                        print(f"  Text: {segment.segment_response}")
                        if segment.Theme:
                            theme_id, theme_label = list(segment.Theme.items())[0]
                            print(f"  Theme: {theme_id} - {theme_label}")
                        if segment.Topic:
                            topic_id, topic_label = list(segment.Topic.items())[0]
                            print(f"  Topic: {topic_id} - {topic_label}")
                        if segment.Keyword:
                            keyword_id, keyword_label = list(segment.Keyword.items())[0]
                            print(f"  Keyword: {keyword_id} - {keyword_label}")
            
            # Print hierarchy statistics
            print("\n=== Hierarchy Statistics ===")
            theme_counts = {}
            topic_counts = {}
            keyword_counts = {}
            
            for result in labeled_results:
                if result.response_segment:
                    for segment in result.response_segment:
                        if segment.Theme:
                            for theme_id in segment.Theme:
                                theme_counts[theme_id] = theme_counts.get(theme_id, 0) + 1
                        if segment.Topic:
                            for topic_id in segment.Topic:
                                topic_counts[topic_id] = topic_counts.get(topic_id, 0) + 1
                        if segment.Keyword:
                            for keyword_id in segment.Keyword:
                                keyword_counts[keyword_id] = keyword_counts.get(keyword_id, 0) + 1
            
            print(f"📊 Final hierarchy:")
            print(f"   - {len(theme_counts)} Themes")
            print(f"   - {len(topic_counts)} Topics") 
            print(f"   - {len(keyword_counts)} Keywords")
            
            print("\n🏆 Top 5 themes by frequency:")
            for theme_id, count in sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"   Theme {theme_id}: {count} segments")
                
        except Exception as e:
            print(f"\n❌ Error during labeling: {str(e)}")
            import traceback
            traceback.print_exc()
            
    else:
        print("❌ No cached clusters found. Please run the clustering pipeline first.")
        print("   Expected cache file: Step 5 clusters")
        print("   Run: python pipeline.py")