
import asyncio
import hashlib
import json
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

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


class ThematicLabeller:
    """Main orchestrator for thematic analysis using LangChain"""
    
    def __init__(self, config: LabellerConfig = None, cache_manager=None, filename: str = None):
        self.config = config or DEFAULT_LABELLER_CONFIG
        self.cache_manager = cache_manager
        self.filename = filename or "unknown"
        self.survey_question = ""
        
        # Setup LLM and parser
        self.llm = self._setup_llm()
        self.parser = JsonOutputParser()
        
        # Build processing chains
        self.familiarization_chain = self._build_familiarization_chain()
        self.discovery_chain = self._build_discovery_chain()
        self.map_chain = self._build_map_chain()
        self.reduce_chain = self._build_reduce_chain()
        self.assignment_chain = self._build_assignment_chain()
        self.refinement_chain = self._build_refinement_chain()
        
        # Thresholds
        self.map_reduce_threshold = 30  # Use MapReduce if more than 30 clusters
        self.batch_size = 10  # Clusters per batch in MapReduce
        
    def _setup_llm(self):
        """Initialize LangChain LLM with configuration"""
        return ChatOpenAI(
            model_name=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            model_kwargs={"seed": self.config.seed if hasattr(self.config, 'seed') else 42},
            openai_api_key=self.config.api_key if hasattr(self.config, 'api_key') else None
        )
    
    def _build_familiarization_chain(self):
        """Build chain for Phase 1: Familiarization"""
        from prompts import PHASE1_FAMILIARIZATION_PROMPT
        
        prompt = PromptTemplate.from_template(PHASE1_FAMILIARIZATION_PROMPT)
        return prompt | self.llm | self.parser
    
    def _build_discovery_chain(self):
        """Build chain for Phase 2: Theme Discovery (single call)"""
        from prompts import PHASE2_DISCOVERY_SINGLE_PROMPT
        
        prompt = PromptTemplate.from_template(PHASE2_DISCOVERY_SINGLE_PROMPT)
        return prompt | self.llm | self.parser
    
    def _build_map_chain(self):
        """Build chain for Phase 2: Theme Discovery (map step)"""
        from prompts import PHASE2_DISCOVERY_MAP_PROMPT
        
        prompt = PromptTemplate.from_template(PHASE2_DISCOVERY_MAP_PROMPT)
        return prompt | self.llm | self.parser
    
    def _build_reduce_chain(self):
        """Build chain for Phase 2: Theme Discovery (reduce step)"""
        from prompts import PHASE2_DISCOVERY_REDUCE_PROMPT
        
        prompt = PromptTemplate.from_template(PHASE2_DISCOVERY_REDUCE_PROMPT)
        return prompt | self.llm | self.parser
    
    def _build_assignment_chain(self):
        """Build chain for Phase 3: Assignment"""
        from prompts import PHASE3_ASSIGNMENT_PROMPT
        
        prompt = PromptTemplate.from_template(PHASE3_ASSIGNMENT_PROMPT)
        return prompt | self.llm | self.parser
    
    def _build_refinement_chain(self):
        """Build chain for Phase 4: Refinement"""
        from prompts import PHASE4_REFINEMENT_PROMPT
        
        prompt = PromptTemplate.from_template(PHASE4_REFINEMENT_PROMPT)
        return prompt | self.llm | self.parser
    
    def _get_representatives(self, cluster: models.ClusterModel) -> List[Tuple[str, float]]:
        """Get top N representative descriptions with similarity scores"""
        # Extract embeddings and descriptions from response segments
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
    
    async def _invoke_with_retries(self, chain, inputs: Dict, max_retries: int = 3) -> Any:
        """Invoke chain with retry logic"""
        last_error = None
        
        for attempt in range(max_retries):
            try:
                result = await chain.ainvoke(inputs)
                return result
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    # Exponential backoff
                    await asyncio.sleep(2 ** attempt)
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
        labeled_clusters = await self._phase1_familiarization(micro_clusters)
        
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
            if len(labeled_clusters) <= self.map_reduce_threshold:
                print(f"  Using single call (≤{self.map_reduce_threshold} clusters)")
                codebook = await self._phase2_discovery_single(labeled_clusters)
            else:
                print(f"  Using MapReduce ({len(labeled_clusters)} clusters)")
                codebook = await self._phase2_discovery_mapreduce(labeled_clusters)
            
            # Cache codebook
            if self.cache_manager:
                self.cache_manager.cache_intermediate_data(
                    codebook.model_dump(), self.filename, cache_key
                )
        
        # Phase 3: Assignment
        print("\n🎯 Phase 3: Assignment - Assigning themes to clusters...")
        assignments = await self._phase3_assignment(labeled_clusters, codebook)
        
        # Phase 4: Refinement
        print("\n✨ Phase 4: Refinement - Finalizing labels...")
        final_labels = await self._phase4_refinement(labeled_clusters, assignments, codebook)
        
        # Apply to original cluster models
        print("\n✅ Applying hierarchy to responses...")
        result = self._apply_hierarchy_to_responses(cluster_models, final_labels, codebook)
        
        print("\n🎉 Hierarchical labeling complete!")
        self._print_summary(codebook)
        
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
            
            inputs = {
                "survey_question": self.survey_question,
                "cluster_id": cluster_id,
                "cluster_size": len(cluster_data['descriptions']),
                "representatives": reps_text,
                "language": self.config.language
            }
            
            task = self._invoke_with_retries(self.familiarization_chain, inputs)
            tasks.append((cluster_id, representatives, task))
        
        # Execute tasks
        for cluster_id, representatives, task in tasks:
            try:
                result = await task
                labeled_clusters.append(ClusterLabel(
                    cluster_id=cluster_id,
                    label=result['label'],
                    description=result['description'],
                    representatives=representatives
                ))
            except Exception as e:
                print(f"Error labeling cluster {cluster_id}: {str(e)}")
                # Create fallback label
                labeled_clusters.append(ClusterLabel(
                    cluster_id=cluster_id,
                    label="UNLABELED",
                    description="Failed to generate label",
                    representatives=representatives
                ))
        
        return labeled_clusters
    
    async def _phase2_discovery_single(self, labeled_clusters: List[ClusterLabel]) -> Codebook:
        """Phase 2: Single call for theme discovery"""
        # Format clusters for prompt
        cluster_summaries = []
        for cluster in sorted(labeled_clusters, key=lambda x: x.cluster_id):
            summary = f"Cluster {cluster.cluster_id}: {cluster.label}\n"
            summary += f"Description: {cluster.description}\n"
            summary += "Representatives:\n"
            for i, (desc, sim) in enumerate(cluster.representatives[:2], 1):
                summary += f"  {i}. {desc}\n"
            cluster_summaries.append(summary)
        
        inputs = {
            "survey_question": self.survey_question,
            "language": self.config.language,
            "cluster_summaries": "\n".join(cluster_summaries),
            "all_cluster_ids": ", ".join(str(c.cluster_id) for c in labeled_clusters)
        }
        
        result = await self._invoke_with_retries(self.discovery_chain, inputs)
        
        # Parse result into Codebook
        return self._parse_codebook(result)
    
    async def _phase2_discovery_mapreduce(self, labeled_clusters: List[ClusterLabel]) -> Codebook:
        """Phase 2: MapReduce for theme discovery"""
        # Create batches
        batches = []
        sorted_clusters = sorted(labeled_clusters, key=lambda x: x.cluster_id)
        
        for i in range(0, len(sorted_clusters), self.batch_size):
            batch = sorted_clusters[i:i + self.batch_size]
            batches.append(batch)
        
        print(f"  Created {len(batches)} batches of ~{self.batch_size} clusters each")
        
        # Map phase
        batch_hierarchies = []
        for batch_idx, batch in enumerate(batches):
            cluster_summaries = []
            for cluster in batch:
                summary = f"Cluster {cluster.cluster_id}: {cluster.label} - {cluster.description}"
                cluster_summaries.append(summary)
            
            inputs = {
                "survey_question": self.survey_question,
                "language": self.config.language,
                "batch_id": f"batch_{batch_idx:03d}",
                "batch_clusters": "\n".join(cluster_summaries)
            }
            
            result = await self._invoke_with_retries(self.map_chain, inputs)
            batch_hierarchies.append(BatchHierarchy(
                batch_id=f"batch_{batch_idx:03d}",
                themes=result.get('themes', []),
                topics=result.get('topics', []),
                subjects=result.get('subjects', [])
            ))
        
        # Reduce phase
        print(f"  Reducing {len(batch_hierarchies)} batch hierarchies...")
        final_hierarchy = await self._reduce_hierarchies(batch_hierarchies)
        
        return final_hierarchy
    
    async def _reduce_hierarchies(self, hierarchies: List[BatchHierarchy]) -> Codebook:
        """Reduce multiple hierarchies into one"""
        # If only one hierarchy, convert directly
        if len(hierarchies) == 1:
            h = hierarchies[0]
            return Codebook(
                survey_question=self.survey_question,
                themes=h.themes,
                topics=h.topics,
                subjects=h.subjects
            )
        
        # Format hierarchies for reduction
        hierarchies_text = []
        for h in hierarchies:
            h_dict = {
                "batch_id": h.batch_id,
                "themes": [t.model_dump() for t in h.themes],
                "topics": [t.model_dump() for t in h.topics],
                "subjects": [s.model_dump() for s in h.subjects]
            }
            hierarchies_text.append(json.dumps(h_dict, indent=2))
        
        inputs = {
            "survey_question": self.survey_question,
            "language": self.config.language,
            "hierarchies": "\n\n".join(hierarchies_text)
        }
        
        result = await self._invoke_with_retries(self.reduce_chain, inputs)
        return self._parse_codebook(result)
    
    def _parse_codebook(self, result: Dict) -> Codebook:
        """Parse JSON result into Codebook structure with proper numeric IDs"""
        themes = []
        topics = []
        subjects = []
        
        # Handle different possible structures in the result
        if 'themes' in result:
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
                        
                        # Extract subjects (keywords)
                        if 'subjects' in topic_data:
                            for subject_data in topic_data['subjects']:
                                # Subjects map to micro_clusters directly
                                for cluster_id in subject_data.get('micro_clusters', []):
                                    subject = CodebookEntry(
                                        id=str(cluster_id),
                                        numeric_id=float(cluster_id),
                                        level=3,
                                        label=subject_data['label'],
                                        description=subject_data.get('description', ''),
                                        parent_id=topic_id,
                                        parent_numeric_id=topic_numeric_id,
                                        direct_clusters=[cluster_id]
                                    )
                                    subjects.append(subject)
        
        return Codebook(
            survey_question=self.survey_question,
            themes=themes,
            topics=topics,
            subjects=subjects
        )
    
    async def _phase3_assignment(self, labeled_clusters: List[ClusterLabel], codebook: Codebook) -> List[ThemeAssignment]:
        """Phase 3: Assign clusters to themes with probabilities"""
        assignments = []
        
        # Format codebook for prompt
        codebook_text = self._format_codebook_for_prompt(codebook)
        
        # Process each cluster
        tasks = []
        for cluster in labeled_clusters:
            # Format representatives
            reps_text = "\n".join([f"- {desc}" for desc, _ in cluster.representatives[:3]])
            
            inputs = {
                "survey_question": self.survey_question,
                "language": self.config.language,
                "cluster_id": cluster.cluster_id,
                "cluster_label": cluster.label,
                "cluster_description": cluster.description,
                "representatives": reps_text,
                "codebook": codebook_text
            }
            
            task = self._invoke_with_retries(self.assignment_chain, inputs)
            tasks.append((cluster.cluster_id, task))
        
        # Execute tasks
        for cluster_id, task in tasks:
            try:
                result = await task
                assignments.append(ThemeAssignment(
                    cluster_id=cluster_id,
                    theme_assignments=result.get('theme_assignments', {}),
                    topic_assignments=result.get('topic_assignments', {}),
                    subject_assignments=result.get('subject_assignments', {})
                ))
            except Exception as e:
                print(f"Error assigning cluster {cluster_id}: {str(e)}")
                # Create empty assignment
                assignments.append(ThemeAssignment(
                    cluster_id=cluster_id,
                    theme_assignments={"other": 1.0},
                    topic_assignments={"other": 1.0},
                    subject_assignments={"other": 1.0}
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
        # For now, we'll use a simple approach
        # In future, this could involve an LLM call to refine labels
        
        final_labels = {}
        
        # Create lookup dictionaries
        cluster_lookup = {c.cluster_id: c for c in labeled_clusters}
        assignment_lookup = {a.cluster_id: a for a in assignments}
        
        # Process each cluster
        for cluster_id in cluster_lookup:
            cluster = cluster_lookup[cluster_id]
            assignment = assignment_lookup.get(cluster_id)
            
            if not assignment:
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
        
        return final_labels
    
    def _get_best_assignment(self, assignments: Dict[str, float], threshold: float = 0.7) -> Tuple[str, float]:
        """Get highest probability assignment above threshold"""
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
        # Create lookup dictionaries using numeric IDs
        theme_lookup = {int(t.numeric_id): t for t in codebook.themes}
        topic_lookup = {t.numeric_id: t for t in codebook.topics}
        
        # Also create string ID lookups for assignment matching
        theme_str_lookup = {t.id: t for t in codebook.themes}
        topic_str_lookup = {t.id: t for t in codebook.topics}
        
        label_models = []
        
        for cluster_model in cluster_models:
            # Convert to LabelModel using the to_model method
            label_model = cluster_model.to_model(models.LabelModel)
            
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
                            theme_id_str, theme_prob = labels['theme']
                            if theme_id_str in theme_str_lookup:
                                theme = theme_str_lookup[theme_id_str]
                                theme_id_int = int(theme.numeric_id)
                                segment.Theme = {theme_id_int: f"{theme.label}: {theme.description}"}
                            elif theme_id_str == "other":
                                segment.Theme = {999: "Other: Unclassified"}
                            
                            # Apply Topic (Dict[float, str]) - keeping as float
                            topic_id_str, topic_prob = labels['topic']
                            if topic_id_str in topic_str_lookup:
                                topic = topic_str_lookup[topic_id_str]
                                topic_id_float = topic.numeric_id  # e.g., 1.1, 1.2, 2.1
                                segment.Topic = {topic_id_float: f"{topic.label}: {topic.description}"}
                            elif topic_id_str == "other":
                                segment.Topic = {99.9: "Other: Unclassified"}
                            
                            # Apply Keyword (Dict[int, str]) - this is the micro_cluster
                            segment.Keyword = segment.micro_cluster
                        else:
                            # Handle unmapped clusters
                            segment.Theme = {999: "Other: Unmapped cluster"}
                            segment.Topic = {9999: "Other: Unmapped cluster"}
                            segment.Keyword = segment.micro_cluster
            
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