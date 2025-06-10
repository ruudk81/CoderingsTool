import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

import asyncio
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from config import LabellerConfig, ModelConfig, DEFAULT_MODEL_CONFIG
import models
from utils.verboseReporter import VerboseReporter


class ClusterLabel(BaseModel):
    """Initial cluster label from Phase 1"""
    cluster_id: int
    label: str = Field(description="Concise label (max 5 words)")
    description: Optional[str] = Field(default=None, description="Natural description from Phase 1")
    representatives: List[Tuple[str, float]] = Field(description="Representative descriptions with similarity scores")

class DescriptiveCodingResponse(BaseModel):
    """Response from Phase 1 descriptive coding"""
    cluster_id: str = Field(description="ID of the cluster being processed")
    segment_label: str = Field(description="Concise thematic label (max 5 words)")
    segment_description: str = Field(description="Natural-sounding description")

class AtomicConcept(BaseModel):
    """Single atomic concept from Phase 2"""
    concept: str = Field(description="Concept name")
    description: str = Field(description="What this concept represents")
    evidence: List[str] = Field(description="Cluster IDs that contain this concept")

class ExtractedAtomicConceptsResponse(BaseModel):
    """Response from Phase 2 atomic concept extraction"""
    analytical_notes: str = Field(description="Working notes from analysis")
    atomic_concepts: List[AtomicConcept] = Field(description="List of atomic concepts identified")

class AtomicConceptGrouped(BaseModel):
    """Atomic concept within a theme"""
    concept_id: str = Field(description="Concept ID like 1.1")
    label: str = Field(description="Atomic concept name")
    description: str = Field(description="What this concept covers")

class ThemeGroup(BaseModel):
    """Theme grouping from Phase 3"""
    theme_id: str = Field(description="Theme ID")
    label: str = Field(description="Theme name")
    description: str = Field(description="What this theme encompasses")
    atomic_concepts: List[AtomicConceptGrouped] = Field(description="Atomic concepts in this theme")

class GroupedConceptsResponse(BaseModel):
    """Response from Phase 3 grouping"""
    themes: List[ThemeGroup] = Field(description="Themes with grouped atomic concepts")
    unassigned_concepts: List[str] = Field(default_factory=list, description="Any concepts that don't fit well")

class RefinedAtomicConcept(BaseModel):
    """Refined atomic concept with examples and statistics"""
    concept_id: str
    label: str
    description: str
    example_quotes: List[str] = Field(description="Representative quotes")
    cluster_count: int = Field(description="Number of clusters assigned")
    percentage: float = Field(description="Percentage of total clusters")

class RefinedTheme(BaseModel):
    """Refined theme with atomic concepts"""
    theme_id: str
    label: str
    description: str
    atomic_concepts: List[RefinedAtomicConcept]

class SummaryStatistics(BaseModel):
    """Summary statistics for the codebook"""
    total_themes: int
    total_concepts: int
    total_clusters: int
    unassigned_clusters: int

class RefinedCodebook(BaseModel):
    """Refined codebook structure"""
    themes: List[RefinedTheme]
    summary_statistics: SummaryStatistics

class LabelRefinementResponse(BaseModel):
    """Response from Phase 4 label refinement"""
    refined_codebook: RefinedCodebook
    refinement_notes: str = Field(description="Key refinements made and rationale")


class ThematicLabeller:
    """Orchestrator for thematic analysis - Simplified 4-phase workflow"""
    
    def __init__(self, config: LabellerConfig = None, model_config: ModelConfig = None, verbose: bool = False, prompt_printer = None): 
        self.config = config or LabellerConfig()
        self.model_config = model_config or DEFAULT_MODEL_CONFIG
        self.survey_question = ""
        self.client = instructor.from_openai(AsyncOpenAI(api_key=self.config.api_key or None), mode=instructor.Mode.JSON)
        self.batch_size = self.config.batch_size
        self.verbose_reporter = VerboseReporter(verbose)
        self.prompt_printer = prompt_printer
        
        # Track which phase prompts have been captured (only capture first prompt per phase)
        self.captured_phase1 = False
        self.captured_phase2 = False
        self.captured_phase3 = False
        self.captured_phase4 = False
        
    def _format_thematic_structure(self, grouped_concepts: GroupedConceptsResponse) -> str:
        """Format grouped concepts for display"""
        formatted = []
        formatted.append("THEMATIC STRUCTURE")
        formatted.append("=" * 50)
        
        for theme in grouped_concepts.themes:
            formatted.append(f"\nTHEME [{theme.theme_id}]: {theme.label}")
            formatted.append(f"  Description: {theme.description}")
            formatted.append("  Atomic Concepts:")
            
            for concept in theme.atomic_concepts:
                formatted.append(f"    [{concept.concept_id}] {concept.label}")
                formatted.append(f"       {concept.description}")
            
        return "\n".join(formatted)
    
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
    
    def _extract_initial_clusters(self, cluster_models: List[models.ClusterModel]) -> Dict[int, Dict]:
        """Extract initial cluster information from cluster models"""
        clusters = {}
        
        for model in cluster_models:
            if model.response_segment:
                for segment in model.response_segment:
                    if segment.initial_cluster is not None:
                        cluster_id = segment.initial_cluster
                        if cluster_id not in clusters:
                            clusters[cluster_id] = {
                                'descriptions': [],
                                'embeddings': [],
                                'codes': []
                            }
                        
                        clusters[cluster_id]['descriptions'].append(segment.segment_description)
                        if segment.description_embedding is not None:
                            clusters[cluster_id]['embeddings'].append(segment.description_embedding)
                        clusters[cluster_id]['codes'].append(segment.segment_label)
        
        return clusters
    
    async def _invoke_with_retries(self, prompt: str, response_model: BaseModel, 
                                   max_retries: int = None, model_override: str = None, phase: str = None) -> Any:
        """Invoke LLM with retry logic and optional model override"""
        if max_retries is None:
            max_retries = self.config.max_retries
        
        # Use model override if provided, otherwise use phase-specific model, otherwise use default
        if model_override:
            model = model_override
        elif phase:
            model = self.model_config.get_model_for_phase(phase)
        else:
            model = self.config.model
            
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    response_model=response_model,
                    temperature=self.model_config.get_temperature_for_phase(phase) if phase else self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    seed=self.model_config.seed
                )
                return response
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    # Use min to cap the delay at 10 seconds
                    delay = min(self.config.retry_delay * (attempt + 1), 10)
                    await asyncio.sleep(delay)
                    self.verbose_reporter.stat_line(f"Retry {attempt + 1}/{max_retries} after error: {str(e)}")
        
        raise RuntimeError(f"Failed after {max_retries} attempts. Last error: {str(last_error)}")
    
    async def process_hierarchy_async(self, cluster_models: List[models.ClusterModel], survey_question: str) -> List[models.LabelModel]:
        """Simplified 4-phase processing workflow"""
        self.survey_question = survey_question
        
        self.verbose_reporter.section_header("HIERARCHICAL LABELING PROCESS (4 PHASES)", emoji="🔄")
        
        # Extract micro-clusters
        initial_clusters = self._extract_initial_clusters(cluster_models)
        self.verbose_reporter.stat_line(f"Found {len(initial_clusters)} unique response segments")
        
        # =============================================================================
        # Phase 1: Descriptive Coding
        # =============================================================================
    
        self.verbose_reporter.step_start("Phase 1: Descriptive Codes", emoji="📝")
        labeled_clusters = await self._phase1_descriptive_coding(initial_clusters)
        self.labeled_clusters = labeled_clusters
        self.verbose_reporter.step_complete(f"Generated {len(labeled_clusters)} descriptive codes")
      
        # =============================================================================
        # Phase 2: Atomic Concepts + Cluster Merging
        # =============================================================================
        
        self.verbose_reporter.step_start("Phase 2: Atomic Concepts + Cluster Merging", emoji="🔍")
        atomic_concepts, merged_clusters = await self._phase2_atomic_concepts_and_merging(labeled_clusters)
        self.atomic_concepts = atomic_concepts
        self.merged_clusters = merged_clusters
        self.verbose_reporter.step_complete(f"Extracted {len(atomic_concepts.atomic_concepts)} concepts, merged to {len(merged_clusters)} clusters")

        # =============================================================================
        # Phase 3: Grouping into Themes
        # =============================================================================

        self.verbose_reporter.step_start("Phase 3: Grouping into Themes", emoji="📚")
        grouped_concepts = await self._phase3_group_concepts_into_themes(atomic_concepts)
        self.verbose_reporter.step_complete("Concepts grouped into themes")
         
        # =============================================================================
        # Phase 4: Label Refinement
        # =============================================================================
        
        self.verbose_reporter.step_start("Phase 4: Label Refinement", emoji="✨")
        refined_codebook = await self._phase4_label_refinement_with_assignments(grouped_concepts, merged_clusters)
        self.verbose_reporter.step_complete("Labels refined with statistics")
        
        # Store results
        self.refined_codebook = refined_codebook
    
        # Print refined codebook
        self._display_refined_codebook(refined_codebook)
        
        # Create mapping from original to merged cluster IDs
        original_to_merged_mapping = self._create_cluster_mapping(labeled_clusters, merged_clusters)
        
        # Create final labels using concept-based structure
        self.final_labels = self._create_final_labels_from_merged_clusters(merged_clusters, original_to_merged_mapping, refined_codebook)
        
        self.verbose_reporter.stat_line("✅ Applying concept assignments to responses...")
        result = self._apply_concept_assignments_to_responses(cluster_models, self.final_labels, refined_codebook)
        self._print_assignment_diagnostics(self.final_labels, initial_clusters)
        self.verbose_reporter.stat_line("🎉 Hierarchical labeling complete!")
        self._print_refined_summary(refined_codebook)

        return result
        
    def _create_final_labels_from_merged_clusters(self, merged_clusters: List[ClusterLabel],
                                                 original_to_merged_mapping: Dict[int, int], 
                                                 refined_codebook: RefinedCodebook) -> Dict[int, Dict]:
        """Create final labels dictionary mapping original cluster IDs to concept assignments from merged clusters"""
        final_labels = {}
        
        # Create lookup dictionaries
        merged_cluster_lookup = {c.cluster_id: c for c in merged_clusters}
        concept_lookup = {}
        theme_lookup = {}
        
        # Build lookup dictionaries from refined codebook
        for theme in refined_codebook.themes:
            theme_lookup[theme.theme_id] = theme
            for concept in theme.atomic_concepts:
                concept_lookup[concept.concept_id] = (concept, theme)
        
        # Reverse the mapping to go from merged back to original
        merged_to_original = {}
        for orig_id, merged_id in original_to_merged_mapping.items():
            if merged_id not in merged_to_original:
                merged_to_original[merged_id] = []
            merged_to_original[merged_id].append(orig_id)
        
        # Process each merged cluster and map back to original IDs
        for merged_cluster in merged_clusters:
            original_cluster_ids = merged_to_original.get(merged_cluster.cluster_id, [merged_cluster.cluster_id])
            
            # Extract concept from cluster description
            concept_name = "Unclassified"
            if "Concept:" in merged_cluster.description:
                concept_name = merged_cluster.description.split("Concept:")[1].split("(")[0].strip()
            
            # Find matching concept in refined codebook
            concept_info = None
            for theme in refined_codebook.themes:
                for concept in theme.atomic_concepts:
                    if concept.label == concept_name or concept_name in concept.label:
                        concept_info = (concept, theme)
                        break
                if concept_info:
                    break
            
            if concept_info:
                concept, theme = concept_info
                label_info = {
                    'theme': (theme.theme_id, theme.label),
                    'concept': (concept.concept_id, concept.label),
                    'confidence': 1.0,
                    'rationale': "Merged cluster assignment"
                }
            else:
                # Fallback for unassigned concepts
                label_info = {
                    'theme': ("99", "Other"),
                    'concept': ("99.1", "Unclassified"),
                    'confidence': 0.0,
                    'rationale': "No concept match found"
                }
            
            # Apply to all original cluster IDs
            for orig_id in original_cluster_ids:
                final_labels[orig_id] = label_info
        
        return final_labels
    
    def _apply_concept_assignments_to_responses(self, cluster_models: List[models.ClusterModel], 
                                               final_labels: Dict[int, Dict], 
                                               refined_codebook: RefinedCodebook) -> List[models.LabelModel]:
        """Apply concept assignments to response segments"""
        # Create lookup dictionaries from refined codebook
        theme_lookup = {theme.theme_id: theme for theme in refined_codebook.themes}
        concept_lookup = {}
        for theme in refined_codebook.themes:
            for concept in theme.atomic_concepts:
                concept_lookup[concept.concept_id] = concept
        
        # Create cluster mappings for the LabelModel
        cluster_mappings = []
        for cluster_id, labels in final_labels.items():
            theme_id_str, theme_label = labels['theme']
            concept_id_str, concept_label = labels['concept']
            confidence = labels['confidence']
            
            mapping = models.ClusterMapping(
                cluster_id=cluster_id,
                cluster_label=concept_label,
                theme_id=theme_id_str,
                topic_id=concept_id_str,  # Using concept as "topic" for compatibility
                code_id=concept_id_str,   # Using concept as "code" for compatibility
                confidence=confidence
            )
            cluster_mappings.append(mapping)
        
        # Convert to hierarchical themes for backward compatibility
        hierarchical_themes = []
        for theme in refined_codebook.themes:
            # Create hierarchical topics from atomic concepts
            hierarchical_topics = []
            for concept in theme.atomic_concepts:
                hierarchical_topic = models.HierarchicalTopic(
                    topic_id=concept.concept_id,
                    numeric_id=float(concept.concept_id.replace('.', '')),
                    label=concept.label,
                    description=concept.description,
                    parent_id=theme.theme_id,
                    level=2,
                    codes=[]  # No codes in the new structure
                )
                hierarchical_topics.append(hierarchical_topic)
            
            hierarchical_theme = models.HierarchicalTheme(
                theme_id=theme.theme_id,
                numeric_id=float(theme.theme_id),
                label=theme.label,
                description=theme.description,
                level=1,
                topics=hierarchical_topics
            )
            hierarchical_themes.append(hierarchical_theme)
        
        label_models = []
        
        for cluster_model in cluster_models:
            # Convert to LabelModel using the to_model method
            label_model = cluster_model.to_model(models.LabelModel)
            
            # Force rebuild model to clear any cached schema
            if hasattr(label_model, 'model_rebuild'):
                label_model.model_rebuild()
            
            # Add hierarchical structure data
            label_model.themes = hierarchical_themes
            label_model.cluster_mappings = cluster_mappings
            
            # Generate a summary for the model
            segment_count = len(label_model.response_segment) if label_model.response_segment else 0
            label_model.summary = f"Response with {segment_count} segments analyzed"
            
            # Apply concept assignments to segments
            if label_model.response_segment:
                for segment in label_model.response_segment:
                    if segment.initial_cluster is not None:
                        cluster_id = segment.initial_cluster
                        
                        if cluster_id in final_labels:
                            labels = final_labels[cluster_id]
                            
                            # Apply Theme (Dict[int, str])
                            theme_id_str, theme_label = labels['theme']
                            if theme_id_str in theme_lookup:
                                theme = theme_lookup[theme_id_str]
                                theme_id_int = int(float(theme.theme_id))
                                description = f": {theme.description}" if theme.description else ""
                                segment.Theme = {theme_id_int: f"{theme.label}{description}"}
                            elif theme_id_str == "99":
                                segment.Theme = {99: "Other: Unclassified"}
                            
                            # Apply Topic (using concept as topic)
                            concept_id_str, concept_label = labels['concept']
                            if concept_id_str in concept_lookup:
                                concept = concept_lookup[concept_id_str]
                                concept_id_float = float(concept_id_str.replace('.', ''))
                                description = f": {concept.description}" if concept.description else ""
                                segment.Topic = {concept_id_float: f"{concept.label}{description}"}
                            elif concept_id_str == "99.1":
                                segment.Topic = {99.1: "Unclassified: No clear concept match"}
                            
                            # Apply Code (same as concept for now)
                            segment.Code = segment.Topic.copy() if hasattr(segment, 'Topic') and segment.Topic else {}
            
            label_models.append(label_model)
        
        return label_models
    
    def _print_refined_summary(self, refined_codebook: RefinedCodebook):
        """Print summary of refined codebook"""
        stats = refined_codebook.summary_statistics
        self.verbose_reporter.summary("Final Results", {
            "Themes": stats.total_themes,
            "Atomic Concepts": stats.total_concepts, 
            "Clusters assigned": stats.total_clusters,
            "Unassigned clusters": stats.unassigned_clusters
        }, emoji="📊")
    
    async def _phase1_descriptive_coding(self, initial_clusters: Dict[int, Dict]) -> List[ClusterLabel]:
        """Phase 1: Descriptive codes - Generate thematic labels for clusters"""
        from prompts import PHASE1_DESCRIPTIVE_CODING_PROMPT
       
        labeled_clusters = []
        tasks = []
        for cluster_id, cluster_data in sorted(initial_clusters.items()):
            # Calculate unique representatives
            if cluster_data['embeddings']:
                embeddings = np.array(cluster_data['embeddings'])
                centroid = np.mean(embeddings, axis=0)
                similarities = cosine_similarity(embeddings, centroid.reshape(1, -1)).flatten()
                top_k = min(5, len(cluster_data['descriptions']))
                sorted_indices = np.argsort(similarities)[::-1]
                seen_descriptions = set()
                representatives = []
                for idx in sorted_indices:
                    desc = cluster_data['descriptions'][idx]
                    if desc not in seen_descriptions:
                        seen_descriptions.add(desc)
                        representatives.append((desc, float(similarities[idx])))
                        if len(representatives) >= top_k:
                            break
            else:
                # Fallback if no embeddings
                representatives = [(desc, 1.0) for desc in cluster_data['descriptions'][:3]]
            
            # Format for prompt
            reps_text = "\n".join([f"{i+1}. {desc} (similarity: {sim:.3f})" for i, (desc, sim) in enumerate(representatives)])
            
            prompt = PHASE1_DESCRIPTIVE_CODING_PROMPT.format(
                survey_question=self.survey_question,
                cluster_id=cluster_id,
                representatives=reps_text,
                language=self.config.language)
            
            # Capture prompt only for the first cluster
            if self.prompt_printer and not self.captured_phase1:
                self.prompt_printer.capture_prompt(
                    step_name="hierarchical_labeling",
                    utility_name="ThematicLabeller",
                    prompt_content=prompt,
                    prompt_type="phase1_descriptive_coding",
                    metadata={
                        "model": self.model_config.get_model_for_phase('phase1_descriptive'),
                        "survey_question": self.survey_question,
                        "language": self.config.language,
                        "phase": "1/4 - Descriptive Codes",
                        "cluster_id": cluster_id
                    }
                )
                self.captured_phase1 = True
            
            task = self._invoke_with_retries(prompt, DescriptiveCodingResponse, phase='phase1_descriptive')
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
                        description="Failed to generate description",
                        representatives=representatives))
                else:
                    labeled_clusters.append(ClusterLabel(
                        cluster_id=cluster_id,
                        label=result.segment_label,
                        description=result.segment_description,
                        representatives=representatives))
        
        return labeled_clusters
    
    async def _phase2_atomic_concepts_and_merging(self, labeled_clusters: List[ClusterLabel]) -> Tuple[ExtractedAtomicConceptsResponse, List[ClusterLabel]]:
        """Phase 2: Extract atomic concepts and merge clusters based on concept assignment"""
        from prompts import PHASE3_EXTRACT_ATOMIC_CONCEPTS_PROMPT
        
        # Format codes for the prompt - use descriptions for better concept extraction
        codes_text = "\n".join([
            f"[ID: {c.cluster_id:2d}] {c.description or c.label}" 
            for c in sorted(labeled_clusters, key=lambda x: x.cluster_id)
        ])
        
        prompt = PHASE3_EXTRACT_ATOMIC_CONCEPTS_PROMPT.format(
            survey_question=self.survey_question,
            codes=codes_text,
            language=self.config.language
        )
        
        # Capture prompt
        if self.prompt_printer and not self.captured_phase2:
            self.prompt_printer.capture_prompt(
                step_name="hierarchical_labeling",
                utility_name="ThematicLabeller",
                prompt_content=prompt,
                prompt_type="phase2_atomic_concepts_and_merging",
                metadata={
                    "model": self.model_config.get_model_for_phase('phase3_themes'),
                    "survey_question": self.survey_question,
                    "language": self.config.language,
                    "phase": "2/4 - Atomic Concepts + Merging"
                }
            )
            self.captured_phase2 = True
        
        # Extract atomic concepts
        atomic_concepts_result = await self._invoke_with_retries(
            prompt, 
            ExtractedAtomicConceptsResponse,
            phase='phase3_themes'
        )
        
        # Merge clusters based on atomic concept evidence
        self.verbose_reporter.stat_line(f"Processing {len(labeled_clusters)} labeled clusters for concept extraction")
        merged_clusters = self._merge_clusters_by_concept_evidence(labeled_clusters, atomic_concepts_result)
        
        self.verbose_reporter.stat_line(f"Extracted {len(atomic_concepts_result.atomic_concepts)} atomic concepts")
        self.verbose_reporter.stat_line(f"Merged {len(labeled_clusters)} clusters into {len(merged_clusters)} concept-based clusters")
        for concept in atomic_concepts_result.atomic_concepts:
            evidence_count = len(concept.evidence)
            self.verbose_reporter.stat_line(f"• {concept.concept} (evidence in {evidence_count} clusters)", bullet="  ")
        
        # Report analytical insights if verbose
        if self.verbose_reporter.enabled and atomic_concepts_result.analytical_notes:
            self.verbose_reporter.stat_line("Analytical notes:", bullet="  ")
            self.verbose_reporter.stat_line(f"{atomic_concepts_result.analytical_notes[:200]}{'...' if len(atomic_concepts_result.analytical_notes) > 200 else ''}", bullet="    ")
        
        return atomic_concepts_result, merged_clusters
    
    def _merge_clusters_by_concept_evidence(self, labeled_clusters: List[ClusterLabel], 
                                          atomic_concepts_result: ExtractedAtomicConceptsResponse) -> List[ClusterLabel]:
        """Merge clusters that are assigned to the same atomic concept based on evidence"""
        # Create mapping from cluster ID to atomic concept
        cluster_to_concept = {}
        total_evidence_entries = 0
        
        for concept in atomic_concepts_result.atomic_concepts:
            self.verbose_reporter.stat_line(f"Concept '{concept.concept}' has evidence: {concept.evidence}", bullet="    ")
            for cluster_id_str in concept.evidence:
                total_evidence_entries += 1
                cluster_id = int(cluster_id_str)
                if cluster_id in cluster_to_concept:
                    # Cluster appears in multiple concepts - keep first assignment
                    self.verbose_reporter.stat_line(f"⚠️ Cluster {cluster_id} appears in multiple concepts, keeping first assignment")
                    continue
                cluster_to_concept[cluster_id] = concept.concept
        
        self.verbose_reporter.stat_line(f"Total evidence entries: {total_evidence_entries}, Unique cluster assignments: {len(cluster_to_concept)}")
        
        # Group clusters by concept
        concept_groups = {}
        unassigned_clusters = []
        
        for cluster in labeled_clusters:
            concept = cluster_to_concept.get(cluster.cluster_id)
            if concept:
                if concept not in concept_groups:
                    concept_groups[concept] = []
                concept_groups[concept].append(cluster)
            else:
                unassigned_clusters.append(cluster)
        
        # Create merged clusters
        merged_clusters = []
        new_cluster_id = 0
        
        # Merge clusters for each concept
        for concept, clusters in concept_groups.items():
            if len(clusters) == 1:
                # Single cluster - just update ID
                cluster = clusters[0]
                merged_cluster = ClusterLabel(
                    cluster_id=new_cluster_id,
                    label=cluster.label,
                    description=f"Concept: {concept}",
                    representatives=cluster.representatives
                )
            else:
                # Multiple clusters - merge them
                all_representatives = []
                all_labels = []
                for cluster in clusters:
                    all_representatives.extend(cluster.representatives)
                    all_labels.append(cluster.label)
                
                # Sort by similarity and take top k
                all_representatives.sort(key=lambda x: x[1], reverse=True)
                top_representatives = all_representatives[:self.config.top_k_representatives]
                
                # Use most common label or first one
                primary_label = all_labels[0] if all_labels else concept
                
                merged_cluster = ClusterLabel(
                    cluster_id=new_cluster_id,
                    label=primary_label,
                    description=f"Concept: {concept} (merged from {len(clusters)} clusters)",
                    representatives=top_representatives
                )
                
                self.verbose_reporter.stat_line(f"Merged {len(clusters)} clusters for concept '{concept}'", bullet="  ")
            
            merged_clusters.append(merged_cluster)
            new_cluster_id += 1
        
        # Add unassigned clusters
        for cluster in unassigned_clusters:
            unassigned_cluster = ClusterLabel(
                cluster_id=new_cluster_id,
                label=cluster.label,
                description="Unassigned to atomic concept",
                representatives=cluster.representatives
            )
            merged_clusters.append(unassigned_cluster)
            new_cluster_id += 1
        
        if unassigned_clusters:
            self.verbose_reporter.stat_line(f"⚠️ {len(unassigned_clusters)} clusters not assigned to any atomic concept")
            self.verbose_reporter.stat_line(f"Unassigned cluster IDs: {[c.cluster_id for c in unassigned_clusters]}")
        
        self.verbose_reporter.stat_line(f"Final result: {len(merged_clusters)} merged clusters ({len(concept_groups)} concept-based + {len(unassigned_clusters)} unassigned)")
        return merged_clusters
    
    async def _phase3_group_concepts_into_themes(self, atomic_concepts_result: ExtractedAtomicConceptsResponse) -> GroupedConceptsResponse:
        """Phase 3: Grouping into themes - Organize atomic concepts into themes"""
        from prompts import PHASE4_GROUP_CONCEPTS_INTO_THEMES_PROMPT
        
        # Format atomic concepts for the prompt
        concepts_text = "\n".join([
            f"- {concept.concept}: {concept.description} (evidence: {', '.join(concept.evidence)})"
            for concept in atomic_concepts_result.atomic_concepts
        ])
        
        prompt = PHASE4_GROUP_CONCEPTS_INTO_THEMES_PROMPT.format(
            survey_question=self.survey_question,
            atomic_concepts=concepts_text,
            language=self.config.language
        )
        
        # Capture prompt
        if self.prompt_printer and not self.captured_phase3:
            self.prompt_printer.capture_prompt(
                step_name="hierarchical_labeling",
                utility_name="ThematicLabeller",
                prompt_content=prompt,
                prompt_type="phase3_group_concepts_into_themes",
                metadata={
                    "model": self.model_config.get_model_for_phase('phase4_codebook'),
                    "survey_question": self.survey_question,
                    "language": self.config.language,
                    "phase": "3/4 - Grouping into Themes"
                }
            )
            self.captured_phase3 = True
        
        result = await self._invoke_with_retries(prompt, GroupedConceptsResponse, phase='phase4_codebook')
        
        # Report statistics
        total_concepts = sum(len(theme.atomic_concepts) for theme in result.themes)
        self.verbose_reporter.stat_line(f"Grouped {total_concepts} concepts into {len(result.themes)} themes")
        for theme in result.themes:
            self.verbose_reporter.stat_line(f"• {theme.label}: {len(theme.atomic_concepts)} concepts", bullet="  ")
        
        if result.unassigned_concepts:
            self.verbose_reporter.stat_line(f"⚠️ {len(result.unassigned_concepts)} concepts remain unassigned", bullet="  ")
        
        return result
    
    async def _phase4_label_refinement_with_assignments(self, grouped_concepts: GroupedConceptsResponse, 
                                                       merged_clusters: List[ClusterLabel]) -> RefinedCodebook:
        """Phase 4: Label refinement with assignment statistics"""
        from prompts import PHASE5_LABEL_REFINEMENT_PROMPT
        
        # Calculate assignment statistics from merged clusters
        concept_assignments = {}
        for cluster in merged_clusters:
            # Extract concept from cluster description
            if "Concept:" in cluster.description:
                concept = cluster.description.split("Concept:")[1].split("(")[0].strip()
                if concept not in concept_assignments:
                    concept_assignments[concept] = []
                concept_assignments[concept].append(str(cluster.cluster_id))
        
        # Create codebook with cluster counts for prompt
        codebook_with_counts = self._format_codebook_with_assignments(grouped_concepts, concept_assignments, merged_clusters)
        
        prompt = PHASE5_LABEL_REFINEMENT_PROMPT.format(
            survey_question=self.survey_question,
            codebook_with_cluster_counts=codebook_with_counts,
            language=self.config.language
        )
        
        # Capture prompt
        if self.prompt_printer and not self.captured_phase4:
            self.prompt_printer.capture_prompt(
                step_name="hierarchical_labeling",
                utility_name="ThematicLabeller",
                prompt_content=prompt,
                prompt_type="phase4_label_refinement_with_stats",
                metadata={
                    "model": self.model_config.get_model_for_phase('phase5_refinement'),
                    "survey_question": self.survey_question,
                    "language": self.config.language,
                    "phase": "4/4 - Label Refinement"
                }
            )
            self.captured_phase4 = True
        
        result = await self._invoke_with_retries(prompt, LabelRefinementResponse, phase='phase5_refinement')
        
        self.verbose_reporter.stat_line(f"Refined {len(result.refined_codebook.themes)} themes")
        self.verbose_reporter.stat_line(f"Total concepts: {result.refined_codebook.summary_statistics.total_concepts}")
        self.verbose_reporter.stat_line(f"Total clusters: {result.refined_codebook.summary_statistics.total_clusters}")
        
        return result.refined_codebook
    
    def _format_codebook_with_assignments(self, grouped_concepts: GroupedConceptsResponse, 
                                        concept_assignments: dict, merged_clusters: List[ClusterLabel]) -> str:
        """Format codebook with cluster assignment counts for refinement prompt"""
        formatted = []
        formatted.append("CODEBOOK WITH CLUSTER ASSIGNMENTS")
        formatted.append("=" * 60)
        
        total_clusters = len(merged_clusters)
        cluster_lookup = {c.cluster_id: c for c in merged_clusters}
        
        for theme in grouped_concepts.themes:
            formatted.append(f"\nTHEME [{theme.theme_id}]: {theme.label}")
            formatted.append(f"Description: {theme.description}")
            formatted.append("")
            
            for concept in theme.atomic_concepts:
                assigned_cluster_ids = concept_assignments.get(concept.concept_id, [])
                cluster_count = len(assigned_cluster_ids)
                percentage = (cluster_count / total_clusters * 100) if total_clusters > 0 else 0
                
                formatted.append(f"  CONCEPT [{concept.concept_id}]: {concept.label}")
                formatted.append(f"  Description: {concept.description}")
                formatted.append(f"  Assigned clusters: {cluster_count} ({percentage:.1f}%)")
                
                # Add example quotes from assigned clusters
                example_quotes = []
                for cluster_id in assigned_cluster_ids[:2]:  # Take first 2 examples
                    if cluster_id in cluster_lookup:
                        cluster = cluster_lookup[cluster_id]
                        if cluster.representatives:
                            example_quotes.append(cluster.representatives[0][0])  # First representative
                
                if example_quotes:
                    formatted.append(f"  Examples: {' | '.join(example_quotes[:2])}")
                formatted.append("")
        
        return "\n".join(formatted)
    
    def _display_refined_codebook(self, refined_codebook: RefinedCodebook):
        """Display the final refined codebook"""
        self.verbose_reporter.section_header("FINAL REFINED CODEBOOK", emoji="📚")
        
        for theme in refined_codebook.themes:
            self.verbose_reporter.stat_line(f"THEME [{theme.theme_id}]: {theme.label}")
            self.verbose_reporter.stat_line(f"  {theme.description}", bullet="  ")
            
            for concept in theme.atomic_concepts:
                percentage_str = f"({concept.percentage:.1f}%)"
                self.verbose_reporter.stat_line(
                    f"[{concept.concept_id}] {concept.label} - {concept.cluster_count} clusters {percentage_str}",
                    bullet="    "
                )
                if concept.example_quotes:
                    self.verbose_reporter.stat_line(f"Examples: {concept.example_quotes[0]}", bullet="      ")
        
        stats = refined_codebook.summary_statistics
        self.verbose_reporter.stat_line(f"\nSummary: {stats.total_themes} themes, {stats.total_concepts} concepts, {stats.total_clusters} clusters")
    
    def process_hierarchy(self, cluster_models: List[models.ClusterModel], survey_question: str) -> List[models.LabelModel]:
        """Sync wrapper for async processing"""
        # Handle nested event loops
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except ImportError:
            pass
        
        return asyncio.run(self.process_hierarchy_async(cluster_models, survey_question))
    
    def _create_cluster_mapping(self, original_clusters: List[ClusterLabel], merged_clusters: List[ClusterLabel]) -> Dict[int, int]:
        """Create mapping from original cluster IDs to merged cluster IDs"""
        mapping = {}
        
        # If no merging occurred, check if we have the merger result stored
        if hasattr(self, 'merger_result'):
            # Use the merger result to create mapping
            merger_result = self.merger_result
            
            # Map merged groups
            for group in merger_result.merged_groups:
                for original_id in group.original_cluster_ids:
                    mapping[original_id] = group.new_cluster_id
            
            # Map unchanged labels
            for unchanged in merger_result.unchanged_labels:
                mapping[unchanged.original_cluster_id] = unchanged.new_cluster_id
        else:
            # Fallback: assume cluster IDs changed sequentially
            original_ids = sorted([c.cluster_id for c in original_clusters])
            merged_ids = sorted([c.cluster_id for c in merged_clusters])
            
            if len(original_ids) == len(merged_ids):
                # No merging occurred, create 1:1 mapping
                for i, orig_id in enumerate(original_ids):
                    mapping[orig_id] = merged_ids[i]
            else:
                # Merging occurred but we don't have detailed info
                # This is a fallback that assumes sequential renumbering
                for orig_cluster in original_clusters:
                    # Try to find merged cluster with same label
                    merged_cluster = next(
                        (c for c in merged_clusters if c.label == orig_cluster.label), 
                        None
                    )
                    if merged_cluster:
                        mapping[orig_cluster.cluster_id] = merged_cluster.cluster_id
        
        return mapping
    
    def _print_assignment_diagnostics(self, final_labels: Dict[int, Dict], initial_clusters: Dict[int, Dict]):
        """Print diagnostics about cluster assignments"""
        print("\n🔍 Assignment Diagnostics:")
        
        # Find all cluster IDs
        all_cluster_ids = set(initial_clusters.keys())
        assigned_cluster_ids = set(final_labels.keys())
        missing_cluster_ids = all_cluster_ids - assigned_cluster_ids
        
        print(f"  - Total clusters found: {len(all_cluster_ids)}")
        print(f"  - Clusters assigned: {len(assigned_cluster_ids)}")
        
        if missing_cluster_ids:
            print(f"  - Missing clusters: {sorted(missing_cluster_ids)}")
            print("  ⚠️  Some clusters were not assigned in Phase 6")
        else:
            print("  ✅ All clusters were assigned")
        
        # Count assignment types
        theme_other = sum(1 for labels in final_labels.values() if labels['theme'][0] == '99')
        concept_other = sum(1 for labels in final_labels.values() if labels['concept'][0] == '99.1')
        
        if theme_other > 0 or concept_other > 0:
            self.verbose_reporter.stat_line(f"Clusters assigned to 'other': {theme_other} themes, {concept_other} concepts")


# =============================================================================
# USAGE / TEST SECTION
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.path.append('..')
    
    from utils.cacheManager import CacheManager
    from utils import dataLoader
    from config import CacheConfig, LabellerConfig
    import models
    
    # Initialize cache manager
    cache_config = CacheConfig()
    cache_manager = CacheManager(cache_config)
    
    # Test data configuration
    filename = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
    var_name = "Q20"
    
    # Load clusters from cache
    cluster_results = cache_manager.load_from_cache(filename, "initial_clusters", models.ClusterModel)
    print(f"✅ Loaded {len(cluster_results)} clustered responses from cache")
    
    # Count unique micro-clusters
    unique_clusters = set()
    for result in cluster_results:
        if result.response_segment:
            for segment in result.response_segment:
                if segment.initial_cluster is not None:
                    cluster_id = segment.initial_cluster
                    unique_clusters.add(cluster_id)
        
    print(f"📊 Found {len(unique_clusters)} unique micro-clusters to label")
        
    # Get variable label
    data_loader = dataLoader.DataLoader()
    var_lab = data_loader.get_varlab(filename=filename, var_name=var_name)
    print(f"📌 Survey question: {var_lab}")
    
    print("\n=== Running Simplified Thematic Labelling (4 Phases) ===")
    print("1. Descriptive Coding - Label each micro-cluster")
    print("2. Atomic Concepts + Merging - Extract concepts and merge clusters")
    print("3. Grouping into Themes - Organize concepts into themes")
    print("4. Label Refinement - Polish labels with statistics")
    print("=" * 50)

    # Initialize thematic labeller with fresh config to avoid caching
    fresh_config = LabellerConfig()
    labeller = ThematicLabeller(
        config=fresh_config,
        verbose=True)
    
    labeled_results = labeller.process_hierarchy(
        cluster_models=cluster_results,
        survey_question=var_lab)
    
    labeled_clusters = labeller.labeled_clusters  # List[ClusterLabel]
    refined_codebook = labeller.refined_codebook  # Refined structure with statistics
    final_labels = labeller.final_labels  # Dict[cluster_id, label_info]
    
    print(f"\n✅ Successfully labeled {len(labeled_results)} responses")
    
    # Save to cache
    cache_manager.save_to_cache(labeled_results, filename, "labels")
    print("💾 Saved labeled results to cache")