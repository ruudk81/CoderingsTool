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
    original_cluster_ids: List[int] = Field(default_factory=list, description="Original cluster IDs that were merged into this cluster")
    assigned_concept: Optional[str] = Field(default=None, description="Assigned concept name from Phase 2")
    assigned_concept_id: Optional[str] = Field(default=None, description="Assigned concept ID for tracking")
    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Confidence score for concept assignment")

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

class ConceptConfidenceScore(BaseModel):
    """Confidence score for cluster-concept assignment"""
    cluster_id: int = Field(description="ID of the cluster being evaluated")
    concept: str = Field(description="Name of the concept being evaluated")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")
    reasoning: str = Field(description="Brief explanation for the score")

class ConfidenceScoringResponse(BaseModel):
    """Response from Phase 2.5 confidence scoring"""
    analytical_notes: str = Field(description="Working notes from confidence evaluation")
    confidence_scores: List[ConceptConfidenceScore] = Field(description="Confidence scores for all cluster-concept pairs")

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
    """Refined atomic concept"""
    concept_id: str
    label: str
    description: str
    stable_id: Optional[str] = Field(default=None, description="Stable ID from Phase 2 for tracking")
    
class RefinedTheme(BaseModel):
    """Refined theme with atomic concepts"""
    theme_id: str
    label: str
    description: str
    atomic_concepts: List[RefinedAtomicConcept]

class RefinedCodebook(BaseModel):
    """Refined codebook structure"""
    themes: List[RefinedTheme]

class LabelRefinementResponse(BaseModel):
    """Response from Phase 4 label refinement"""
    refined_codebook: RefinedCodebook
    

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
        self.captured_phase2_5 = False
        self.captured_phase3 = False
        self.captured_phase4 = False
   
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
        # Phase 2: Atomic Concepts  
        # =============================================================================
        
        self.verbose_reporter.step_start("Phase 2: Atomic Concepts", emoji="🔍")
        atomic_concepts, _ = await self._phase2_atomic_concepts_and_merging(labeled_clusters)
        self.atomic_concepts = atomic_concepts
        
        # Count concepts (including "Other" if present)
        concept_count = len(atomic_concepts.atomic_concepts)
        has_other = any(c.concept == "Other" for c in atomic_concepts.atomic_concepts)
        other_note = " (including 'Other' concept)" if has_other else ""
        
        self.verbose_reporter.step_complete(f"Extracted {concept_count} concepts{other_note}")
        
        # =============================================================================
        # Phase 2.5: Confidence Scoring (if enabled)
        # =============================================================================
        
        confidence_scores = None
        if self.config.use_confidence_scoring:
            self.verbose_reporter.step_start("Phase 2.5: Confidence Scoring", emoji="📊")
            confidence_scores = await self._phase2_5_confidence_scoring(labeled_clusters, atomic_concepts)
            self.verbose_reporter.step_complete("Confidence scores calculated")
        
        # Merge clusters using confidence scores or evidence
        merged_clusters = self._merge_clusters_by_concept_evidence(labeled_clusters, atomic_concepts, confidence_scores)
        self.merged_clusters = merged_clusters
        self.verbose_reporter.stat_line(f"Merged to {len(merged_clusters)} concept-based clusters")

        # =============================================================================
        # Phase 3: Themes
        # =============================================================================

        self.verbose_reporter.step_start("Phase 3: Grouping into Themes", emoji="📚")
        grouped_concepts = await self._phase3_group_concepts_into_themes(atomic_concepts)
        self.grouped_concepts = grouped_concepts
        self.verbose_reporter.step_complete("Concepts grouped into themes")
         
        # =============================================================================
        # Phase 4: Label Refinement
        # =============================================================================
        
        self.verbose_reporter.step_start("Phase 4: Label Refinement", emoji="✨")
        refined_codebook = await self._phase4_label_refinement_with_assignments(grouped_concepts, merged_clusters)
        self.verbose_reporter.step_complete("Labels refined")
      
        self.refined_codebook = refined_codebook
        self._display_refined_codebook(refined_codebook)
        
        # Create mapping from original to merged cluster IDs
        original_to_merged_mapping = self._create_cluster_mapping(labeled_clusters, merged_clusters)
        self.original_to_merged_mapping = original_to_merged_mapping
        
        # Create final labels using concept-based structure
        self.final_labels = self._create_final_labels_from_merged_clusters(merged_clusters, original_to_merged_mapping, refined_codebook)
        
        self.verbose_reporter.stat_line("✅ Applying concept assignments to responses...")
        result = self._apply_concept_assignments_to_responses(cluster_models, self.final_labels, refined_codebook)
        self._print_assignment_diagnostics(self.final_labels, initial_clusters)
        self.verbose_reporter.stat_line("🎉 Hierarchical labeling complete!")
        #self._print_refined_summary(refined_codebook)

        return result
        
    
    def _create_final_labels_from_merged_clusters(self, merged_clusters: List[ClusterLabel], original_to_merged_mapping: Dict[int, int], refined_codebook: RefinedCodebook) -> Dict[int, Dict]:
        """Create final labels dictionary mapping original cluster IDs to concept assignments"""
        final_labels = {}
        
        # Use the Phase 4 tracking if available, otherwise fall back to Phase 3
        if hasattr(self, 'stable_id_to_phase4_ids'):
            stable_id_to_final_ids = self.stable_id_to_phase4_ids.copy()
            self.verbose_reporter.stat_line(f"Using Phase 4 tracking with {len(stable_id_to_final_ids)} mappings")
        elif hasattr(self, 'concept_name_to_ids_phase3'):
            stable_id_to_final_ids = self.concept_name_to_ids_phase3.copy()
            self.verbose_reporter.stat_line(f"Using Phase 3 tracking with {len(stable_id_to_final_ids)} mappings")
        else:
            stable_id_to_final_ids = {}
            self.verbose_reporter.stat_line("⚠️ No tracking found - will use fallback mappings")
        
        # Handle unassigned clusters that were preserved
        for theme in refined_codebook.themes:
            if theme.theme_id == "99" or theme.label.lower() == "other":
                for concept in theme.atomic_concepts:
                    # Check if this is an unassigned cluster concept
                    if "unassigned cluster:" in concept.description.lower():
                        # Extract the stable ID from our mapping
                        for stable_id, final_id in self.concept_id_mapping.items():
                            if final_id == concept.concept_id and stable_id.startswith("unassigned_"):
                                stable_id_to_final_ids[stable_id] = (theme.theme_id, concept.concept_id)
                break
        
        # Create reverse mapping: merged_id → [original_ids]
        merged_to_original = {}
        for orig_id, merged_id in original_to_merged_mapping.items():
            if merged_id not in merged_to_original:
                merged_to_original[merged_id] = []
            merged_to_original[merged_id].append(orig_id)
        
        # Process each merged cluster
        successfully_mapped = 0
        fallback_mapped = 0
        
        for merged_cluster in merged_clusters:
            original_cluster_ids = merged_to_original.get(merged_cluster.cluster_id, [merged_cluster.cluster_id])
            stable_concept_id = merged_cluster.assigned_concept_id
            
            if stable_concept_id and stable_concept_id in stable_id_to_final_ids:
                theme_id, concept_id = stable_id_to_final_ids[stable_concept_id]
                
                # Find theme and concept objects
                theme_obj = next((t for t in refined_codebook.themes if t.theme_id == theme_id), None)
                concept_obj = None
                if theme_obj:
                    concept_obj = next((c for c in theme_obj.atomic_concepts if c.concept_id == concept_id), None)
                
                if theme_obj and concept_obj:
                    label_info = {
                        'theme': (theme_id, theme_obj.label),
                        'concept': (concept_id, concept_obj.label),
                    }
                    successfully_mapped += len(original_cluster_ids)
                else:
                    # Shouldn't happen if tracking is correct
                    label_info = {
                        'theme': (theme_id, "Unknown Theme"),
                        'concept': (concept_id, merged_cluster.assigned_concept or "Unknown"),
                    }
                    self.verbose_reporter.stat_line(
                        f"⚠️ Stable ID '{stable_concept_id}' maps to {theme_id}.{concept_id} but objects not found",
                        bullet="  "
                    )
            else:
                # Fallback
                label_info = {
                    'theme': ("99", "Other"),
                    'concept': ("99.1", "Unclassified"),
                }
                fallback_mapped += len(original_cluster_ids)
                
                if stable_concept_id:
                    self.verbose_reporter.stat_line(
                        f"No mapping for stable ID '{stable_concept_id}' - assigning to Other",
                        bullet="  "
                    )
            
            # Apply to all original cluster IDs
            for orig_id in original_cluster_ids:
                final_labels[orig_id] = label_info
        
        self.verbose_reporter.stat_line(f"Mapping complete: {successfully_mapped} successful, {fallback_mapped} fallback")
        
        return final_labels
    
        
    def _apply_concept_assignments_to_responses(self, cluster_models: List[models.ClusterModel], final_labels: Dict[int, Dict], refined_codebook: RefinedCodebook) -> List[models.LabelModel]:
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
            theme_id_str, _ = labels['theme']  # theme_label not used
            concept_id_str, concept_label = labels['concept']

            
            mapping = models.ClusterMapping(
                cluster_id=cluster_id,
                cluster_label=concept_label,
                theme_id=theme_id_str,
                topic_id=concept_id_str,  # Using concept as "topic" for compatibility
                code_id=concept_id_str,   # Using concept as "code" for compatibility
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
                            theme_id_str, _ = labels['theme']  # theme_label not used here
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
    
    # def _print_refined_summary(self, refined_codebook: RefinedCodebook):
    #     """Print summary of refined codebook"""
    #     stats = refined_codebook.summary_statistics
    #     assigned_clusters = stats.total_clusters - stats.unassigned_clusters
    #     self.verbose_reporter.summary("Final Results", {
    #         "Themes": stats.total_themes,
    #         "Atomic Concepts": stats.total_concepts, 
    #         "Clusters assigned": assigned_clusters,
    #         "Unassigned clusters": stats.unassigned_clusters
    #     }, emoji="📊")
    
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
                        representatives=representatives,
                        original_cluster_ids=[cluster_id]))
                else:
                    labeled_clusters.append(ClusterLabel(
                        cluster_id=cluster_id,
                        label=result.segment_label,
                        description=result.segment_description,
                        representatives=representatives,
                        original_cluster_ids=[cluster_id]))
        
        return labeled_clusters
    
    async def _phase2_atomic_concepts_and_merging(self, labeled_clusters: List[ClusterLabel]) -> Tuple[ExtractedAtomicConceptsResponse, List[ClusterLabel]]:
        """Phase 2: Extract atomic concepts"""
        from prompts import PHASE2_EXTRACT_ATOMIC_CONCEPTS_PROMPT
        
        # get atompic concepts
        codes_text = "\n".join([ f"[ID: {c.cluster_id:2d}] {c.description or c.label}" for c in sorted(labeled_clusters, key=lambda x: x.cluster_id)])
        
        prompt = PHASE2_EXTRACT_ATOMIC_CONCEPTS_PROMPT.format(
            survey_question=self.survey_question,
            codes=codes_text,
            language=self.config.language)

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
                    "phase": "2/4 - Atomic Concepts"})
            self.captured_phase2 = True
        
        atomic_concepts_result = await self._invoke_with_retries(prompt, ExtractedAtomicConceptsResponse, phase='phase3_themes')
        
        self.verbose_reporter.stat_line(f"Extracted {len(atomic_concepts_result.atomic_concepts)} atomic concepts")
        for concept in atomic_concepts_result.atomic_concepts:
            evidence_count = len(concept.evidence)
            self.verbose_reporter.stat_line(f"• {concept.concept} (evidence in {evidence_count} clusters)", bullet="  ")
        
        # Note: Merging is now done separately after optional confidence scoring
        return atomic_concepts_result, []
    
    async def _evaluate_by_concept_batches(self, labeled_clusters: List[ClusterLabel], atomic_concepts_result: ExtractedAtomicConceptsResponse) -> Dict[int, Dict[str, ConceptConfidenceScore]]:
        """Evaluate all clusters against each concept in focused batches"""
        from prompts import PHASE2_5_CONCEPT_FOCUSED_SCORING_PROMPT
        
        confidence_scores_by_cluster = {}
        num_clusters = len(labeled_clusters)
        num_concepts = len(atomic_concepts_result.atomic_concepts)
        
        self.verbose_reporter.stat_line(f"Evaluating {num_clusters} clusters against {num_concepts} concepts using concept-focused batching")
        
        # Prepare all cluster codes once
        all_cluster_codes = "\n".join([
            f"[ID: {c.cluster_id:2d}] {c.description or c.label}"
            for c in sorted(labeled_clusters, key=lambda x: x.cluster_id)
        ])
        
        # Create tasks for each concept
        concept_tasks = []
        for i, concept in enumerate(atomic_concepts_result.atomic_concepts):
            prompt = PHASE2_5_CONCEPT_FOCUSED_SCORING_PROMPT.format(
                survey_question=self.survey_question,
                concept_name=concept.concept,
                concept_description=concept.description,
                all_cluster_codes=all_cluster_codes,
                expected_scores=num_clusters,
                language=self.config.language
            )
            
            # Capture prompt only for the first concept
            if self.prompt_printer and not self.captured_phase2_5:
                self.prompt_printer.capture_prompt(
                    step_name="hierarchical_labeling",
                    utility_name="ThematicLabeller",
                    prompt_content=prompt,
                    prompt_type="phase2_5_concept_focused_scoring",
                    metadata={
                        "model": self.model_config.get_model_for_phase('phase2_5_confidence'),
                        "survey_question": self.survey_question,
                        "language": self.config.language,
                        "phase": f"2.5/4 - Concept {i+1}/{num_concepts} Focused Scoring",
                        "concept_name": concept.concept
                    }
                )
                self.captured_phase2_5 = True
            
            task = self._invoke_with_retries(prompt, ConfidenceScoringResponse, phase='phase2_5_confidence')
            concept_tasks.append((concept.concept, task))
        
        # Execute concept evaluations with concurrency control
        self.verbose_reporter.stat_line(f"Processing {len(concept_tasks)} concepts with max {self.config.concurrent_requests} concurrent requests...")
        
        for i in range(0, len(concept_tasks), self.config.concurrent_requests):
            batch = concept_tasks[i:i + self.config.concurrent_requests]
            batch_results = await asyncio.gather(*[task for _, task in batch], return_exceptions=True)
            
            for (concept_name, _), result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    self.verbose_reporter.stat_line(f"⚠️ Error evaluating concept '{concept_name}': {str(result)}", bullet="  ")
                    continue
                
                # Process scores for this concept
                actual_scores = len(result.confidence_scores)
                if actual_scores == num_clusters:
                    self.verbose_reporter.stat_line(f"✅ Concept '{concept_name}': {actual_scores}/{num_clusters} scores", bullet="  ")
                else:
                    self.verbose_reporter.stat_line(f"⚠️ Concept '{concept_name}': expected {num_clusters}, got {actual_scores}", bullet="  ")
                
                # Store scores by cluster
                for score in result.confidence_scores:
                    cluster_id = score.cluster_id
                    if cluster_id not in confidence_scores_by_cluster:
                        confidence_scores_by_cluster[cluster_id] = {}
                    confidence_scores_by_cluster[cluster_id][concept_name] = score
        
        return confidence_scores_by_cluster
    
    async def _phase2_5_confidence_scoring(self, labeled_clusters: List[ClusterLabel], atomic_concepts_result: ExtractedAtomicConceptsResponse) -> Dict[int, Dict[str, ConceptConfidenceScore]]:
        """Phase 2.5: Score confidence using concept-focused batching approach"""
        # Skip if confidence scoring is disabled
        if not self.config.use_confidence_scoring:
            self.verbose_reporter.stat_line("Confidence scoring disabled, using binary assignment from Phase 2")
            return self._create_binary_confidence_scores(labeled_clusters, atomic_concepts_result)
        
        # Use new concept-focused batching approach
        confidence_scores_by_cluster = await self._evaluate_by_concept_batches(labeled_clusters, atomic_concepts_result)
        
        # Report statistics
        total_scores = sum(len(scores) for scores in confidence_scores_by_cluster.values())
        high_confidence = sum(
            1 for scores in confidence_scores_by_cluster.values()
            for score in scores.values()
            if score.confidence >= self.config.confidence_threshold
        )
        
        expected_total = len(labeled_clusters) * len(atomic_concepts_result.atomic_concepts)
        coverage_rate = (total_scores / expected_total * 100) if expected_total > 0 else 0
        
        self.verbose_reporter.stat_line(
            f"Evaluated {total_scores}/{expected_total} cluster-concept pairs ({coverage_rate:.1f}%): "
            f"{high_confidence} meet threshold ({self.config.confidence_threshold})"
        )
        
        return confidence_scores_by_cluster
    
    async def _evaluate_evidence_clusters(self, labeled_clusters: List[ClusterLabel], atomic_concepts_result: ExtractedAtomicConceptsResponse) -> Dict[int, Dict[str, ConceptConfidenceScore]]:
        """Evaluate clusters that appear in concept evidence"""
        from prompts import PHASE2_5_EVIDENCE_SCORING_PROMPT
        
        # Find clusters with evidence
        clusters_with_evidence = set()
        for concept in atomic_concepts_result.atomic_concepts:
            for cluster_id_str in concept.evidence:
                clusters_with_evidence.add(int(cluster_id_str))
        
        evidence_clusters = [c for c in labeled_clusters if c.cluster_id in clusters_with_evidence]
        
        if not evidence_clusters:
            self.verbose_reporter.stat_line("No evidence clusters to evaluate")
            return {}
        
        self.verbose_reporter.stat_line(f"Evaluating {len(evidence_clusters)} evidence clusters")
        
        # Prepare data
        concepts_text = "\n".join([
            f"- {concept.concept}: {concept.description} (evidence: {concept.evidence})"
            for concept in atomic_concepts_result.atomic_concepts
        ])
        
        codes_text = "\n".join([
            f"[ID: {c.cluster_id:2d}] {c.description or c.label}"
            for c in evidence_clusters
        ])
        
        # Calculate expected scores
        expected_scores = sum(
            sum(1 for cluster_id_str in concept.evidence if int(cluster_id_str) in clusters_with_evidence)
            for concept in atomic_concepts_result.atomic_concepts
        )
        
        prompt = PHASE2_5_EVIDENCE_SCORING_PROMPT.format(
            survey_question=self.survey_question,
            atomic_concepts=concepts_text,
            descriptive_codes=codes_text,
            expected_scores=expected_scores,
            language=self.config.language
        )
        
        # Capture prompt
        if self.prompt_printer and not self.captured_phase2_5:
            self.prompt_printer.capture_prompt(
                step_name="hierarchical_labeling",
                utility_name="ThematicLabeller", 
                prompt_content=prompt,
                prompt_type="phase2_5_evidence_scoring",
                metadata={
                    "model": self.model_config.get_model_for_phase('phase2_5_confidence'),
                    "survey_question": self.survey_question,
                    "language": self.config.language,
                    "phase": "2.5A/4 - Evidence Scoring"
                }
            )
            self.captured_phase2_5 = True
        
        try:
            result = await self._invoke_with_retries(prompt, ConfidenceScoringResponse, phase='phase2_5_confidence')
            
            # Process scores
            confidence_scores = {}
            for score in result.confidence_scores:
                cluster_id = score.cluster_id
                if cluster_id not in confidence_scores:
                    confidence_scores[cluster_id] = {}
                confidence_scores[cluster_id][score.concept] = score
            
            actual_scores = len(result.confidence_scores)
            if actual_scores == expected_scores:
                self.verbose_reporter.stat_line(f"✅ Evidence evaluation: {actual_scores}/{expected_scores} scores", bullet="  ")
            else:
                self.verbose_reporter.stat_line(f"⚠️ Evidence evaluation: expected {expected_scores}, got {actual_scores}", bullet="  ")
            
            return confidence_scores
            
        except Exception as e:
            self.verbose_reporter.stat_line(f"⚠️ Error in evidence evaluation: {str(e)}", bullet="  ")
            return {}
    
    async def _evaluate_unassigned_clusters(self, labeled_clusters: List[ClusterLabel], atomic_concepts_result: ExtractedAtomicConceptsResponse) -> Dict[int, Dict[str, ConceptConfidenceScore]]:
        """Evaluate clusters not in any evidence against all concepts"""
        from prompts import PHASE2_5_UNASSIGNED_SCORING_PROMPT
        
        # Find unassigned clusters
        clusters_with_evidence = set()
        for concept in atomic_concepts_result.atomic_concepts:
            for cluster_id_str in concept.evidence:
                clusters_with_evidence.add(int(cluster_id_str))
        
        unassigned_clusters = [c for c in labeled_clusters if c.cluster_id not in clusters_with_evidence]
        
        if not unassigned_clusters:
            self.verbose_reporter.stat_line("No unassigned clusters to evaluate")
            return {}
        
        self.verbose_reporter.stat_line(f"Evaluating {len(unassigned_clusters)} unassigned clusters against all concepts")
        
        # Prepare data
        concepts_text = "\n".join([
            f"- {concept.concept}: {concept.description}"
            for concept in atomic_concepts_result.atomic_concepts
        ])
        
        unassigned_codes_text = "\n".join([
            f"[ID: {c.cluster_id:2d}] {c.description or c.label}"
            for c in unassigned_clusters
        ])
        
        num_concepts = len(atomic_concepts_result.atomic_concepts)
        num_unassigned = len(unassigned_clusters)
        expected_scores = num_unassigned * num_concepts
        
        prompt = PHASE2_5_UNASSIGNED_SCORING_PROMPT.format(
            survey_question=self.survey_question,
            atomic_concepts=concepts_text,
            unassigned_codes=unassigned_codes_text,
            num_unassigned=num_unassigned,
            num_concepts=num_concepts,
            expected_scores=expected_scores,
            language=self.config.language
        )
        
        try:
            result = await self._invoke_with_retries(prompt, ConfidenceScoringResponse, phase='phase2_5_confidence')
            
            # Process scores
            confidence_scores = {}
            for score in result.confidence_scores:
                cluster_id = score.cluster_id
                if cluster_id not in confidence_scores:
                    confidence_scores[cluster_id] = {}
                confidence_scores[cluster_id][score.concept] = score
            
            actual_scores = len(result.confidence_scores)
            if actual_scores == expected_scores:
                self.verbose_reporter.stat_line(f"✅ Unassigned evaluation: {actual_scores}/{expected_scores} scores", bullet="  ")
            else:
                self.verbose_reporter.stat_line(f"⚠️ Unassigned evaluation: expected {expected_scores}, got {actual_scores}", bullet="  ")
            
            return confidence_scores
            
        except Exception as e:
            self.verbose_reporter.stat_line(f"⚠️ Error in unassigned evaluation: {str(e)}", bullet="  ")
            return {}
    
    def _create_binary_confidence_scores(self, labeled_clusters: List[ClusterLabel], atomic_concepts_result: ExtractedAtomicConceptsResponse) -> Dict[int, Dict[str, ConceptConfidenceScore]]:
        """Create binary confidence scores from Phase 2 evidence (fallback when confidence scoring disabled)"""
        confidence_scores_by_cluster = {}
        
        # Create concept evidence lookup
        concept_evidence = {}
        for concept in atomic_concepts_result.atomic_concepts:
            concept_evidence[concept.concept] = set(int(cid) for cid in concept.evidence)
        
        # Assign binary scores
        for cluster in labeled_clusters:
            cluster_id = cluster.cluster_id
            confidence_scores_by_cluster[cluster_id] = {}
            
            for concept in atomic_concepts_result.atomic_concepts:
                if cluster_id in concept_evidence[concept.concept]:
                    # High confidence for evidence match
                    score = ConceptConfidenceScore(
                        cluster_id=cluster_id,
                        concept=concept.concept,
                        confidence=1.0,
                        reasoning="Direct evidence from Phase 2"
                    )
                else:
                    # Zero confidence for no evidence
                    score = ConceptConfidenceScore(
                        cluster_id=cluster_id,
                        concept=concept.concept,
                        confidence=0.0,
                        reasoning="No evidence in Phase 2"
                    )
                confidence_scores_by_cluster[cluster_id][concept.concept] = score
        
        return confidence_scores_by_cluster

    def _merge_clusters_by_concept_evidence(self, labeled_clusters: List[ClusterLabel], atomic_concepts_result: ExtractedAtomicConceptsResponse, confidence_scores: Optional[Dict[int, Dict[str, ConceptConfidenceScore]]] = None) -> List[ClusterLabel]:
        """Merge clusters that are assigned to the same atomic concept based on evidence or confidence scores"""
        # Create mapping from cluster ID to atomic concept WITH CONCEPT ID
        cluster_to_concept = {}
        concept_id_mapping = {}  # Maps concept name to a stable ID
        total_evidence_entries = 0
        
        # First, create stable IDs for concepts
        for idx, concept in enumerate(atomic_concepts_result.atomic_concepts):
            concept_id = f"concept_{idx+1}"  # Stable ID that won't change
            concept_id_mapping[concept.concept] = concept_id
        
        # Use confidence scores if available, otherwise fall back to evidence
        if confidence_scores:
            # Confidence-based assignment
            assignments_above_threshold = 0
            assignments_below_threshold = 0
            
            for cluster in labeled_clusters:
                cluster_id = cluster.cluster_id
                if cluster_id not in confidence_scores:
                    self.verbose_reporter.stat_line(f"⚠️ No confidence scores for cluster {cluster_id}")
                    continue
                
                # Find concept with highest confidence
                best_concept = None
                best_confidence = 0.0
                best_concept_id = None
                
                for concept_name, score in confidence_scores[cluster_id].items():
                    if score.confidence > best_confidence:
                        best_confidence = score.confidence
                        best_concept = concept_name
                        best_concept_id = concept_id_mapping.get(concept_name)
                
                # Assign if above threshold
                if best_confidence >= self.config.confidence_threshold and best_concept and best_concept_id:
                    cluster_to_concept[cluster_id] = (best_concept, best_concept_id)
                    assignments_above_threshold += 1
                else:
                    assignments_below_threshold += 1
                    if best_concept:
                        self.verbose_reporter.stat_line(
                            f"Cluster {cluster_id} best match '{best_concept}' ({best_confidence:.2f}) below threshold",
                            bullet="  "
                        )
            
            self.verbose_reporter.stat_line(
                f"Confidence-based assignment: {assignments_above_threshold} above threshold, "
                f"{assignments_below_threshold} below threshold"
            )
        else:
            # Original evidence-based assignment
            for concept in atomic_concepts_result.atomic_concepts:
                concept_id = concept_id_mapping[concept.concept]
                for cluster_id_str in concept.evidence:
                    total_evidence_entries += 1
                    cluster_id = int(cluster_id_str)
                    if cluster_id in cluster_to_concept:
                        self.verbose_reporter.stat_line(f"⚠️ Cluster {cluster_id} appears in multiple concepts, keeping first assignment")
                        continue
                    cluster_to_concept[cluster_id] = (concept.concept, concept_id)
        
        self.verbose_reporter.stat_line(f"Total evidence entries: {total_evidence_entries}, Unique cluster assignments: {len(cluster_to_concept)}")
        
        # Store the concept ID mapping for later use
        self.concept_id_mapping = concept_id_mapping
        
        # Group clusters by concept
        concept_groups = {}
        unassigned_clusters = []
        
        for cluster in labeled_clusters:
            concept_info = cluster_to_concept.get(cluster.cluster_id)
            if concept_info:
                concept_name, concept_id = concept_info
                if concept_name not in concept_groups:
                    concept_groups[concept_name] = []
                concept_groups[concept_name].append(cluster)
            else:
                unassigned_clusters.append(cluster)
        
        # Create merged clusters
        merged_clusters = []
        new_cluster_id = 0
        
        # Merge clusters for each concept
        for concept_name, clusters in concept_groups.items():
            concept_id = concept_id_mapping[concept_name]
            
            # Collect all original cluster IDs from the clusters being merged
            all_original_ids = []
            for cluster in clusters:
                all_original_ids.extend(cluster.original_cluster_ids)
            
            if len(clusters) == 1:
                # Single cluster - just update ID but keep original tracking
                cluster = clusters[0]
                # Get confidence score if available
                confidence = None
                if confidence_scores and cluster.cluster_id in confidence_scores:
                    concept_score = confidence_scores[cluster.cluster_id].get(concept_name)
                    if concept_score:
                        confidence = concept_score.confidence
                
                merged_cluster = ClusterLabel(
                    cluster_id=new_cluster_id,
                    label=cluster.label,
                    description=f"Concept: {concept_name}",
                    representatives=cluster.representatives,
                    original_cluster_ids=all_original_ids,
                    assigned_concept=concept_name,
                    assigned_concept_id=concept_id,  # Store the stable ID
                    confidence_score=confidence
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
                primary_label = all_labels[0] if all_labels else concept_name
                
                # Get average confidence score for merged clusters
                confidence = None
                if confidence_scores:
                    confidences = []
                    for cluster in clusters:
                        if cluster.cluster_id in confidence_scores:
                            concept_score = confidence_scores[cluster.cluster_id].get(concept_name)
                            if concept_score:
                                confidences.append(concept_score.confidence)
                    if confidences:
                        confidence = sum(confidences) / len(confidences)
                
                merged_cluster = ClusterLabel(
                    cluster_id=new_cluster_id,
                    label=primary_label,
                    description=f"Concept: {concept_name} (merged from {len(clusters)} clusters)",
                    representatives=top_representatives,
                    original_cluster_ids=all_original_ids,
                    assigned_concept=concept_name,
                    assigned_concept_id=concept_id,  # Store the stable ID
                    confidence_score=confidence
                )
                
                #self.verbose_reporter.stat_line(f"Merged {len(clusters)} clusters for concept '{concept_name}' (ID: {concept_id}, original IDs: {all_original_ids})", bullet="  ")
            
            merged_clusters.append(merged_cluster)
            new_cluster_id += 1
        
        # Keep unassigned clusters separate (don't merge) for promotion to theme level
        if unassigned_clusters:
            self.verbose_reporter.stat_line(f"Preserving {len(unassigned_clusters)} unassigned clusters for 'Other' theme", bullet="  ")
            
            # Keep each unassigned cluster individual
            for cluster in unassigned_clusters:
                # Create a preserved cluster that maintains original identity
                preserved_cluster = ClusterLabel(
                    cluster_id=new_cluster_id,
                    label=cluster.label,  # Keep original label
                    description=f"Unassigned: {cluster.label}",
                    representatives=cluster.representatives,
                    original_cluster_ids=cluster.original_cluster_ids,
                    assigned_concept="Unassigned",  # Mark as unassigned for Phase 3
                    assigned_concept_id=f"unassigned_{cluster.cluster_id}",  # Unique ID for tracking
                    confidence_score=0.0  # Unassigned clusters have 0 confidence
                )
                merged_clusters.append(preserved_cluster)
                new_cluster_id += 1
                
            self.verbose_reporter.stat_line(f"Preserved {len(unassigned_clusters)} unassigned clusters with original labels", bullet="  ")
        
        # Report final statistics
        unassigned_count = len(unassigned_clusters)
        self.verbose_reporter.stat_line(f"Final result: {len(merged_clusters)} clusters ({len(concept_groups)} concept-merged + {unassigned_count} unassigned preserved)")
        
        # Store merged clusters for tracking
        self.merged_clusters_by_concept_id = {cluster.assigned_concept_id: cluster for cluster in merged_clusters}
        
        return merged_clusters
    
    def _create_concept_tracking_for_phase3(self, atomic_concepts_result: ExtractedAtomicConceptsResponse, grouped_concepts: GroupedConceptsResponse):
        """Create a mapping between original concept names and their IDs in the grouped structure"""
        # This mapping tracks: original_concept_name -> (theme_id, concept_id)
        concept_name_to_ids = {}
        
        for theme in grouped_concepts.themes:
            for concept in theme.atomic_concepts:
                # Try to find the original concept name
                # Look for exact matches first
                for original_concept in atomic_concepts_result.atomic_concepts:
                    if original_concept.concept == concept.label:
                        original_concept_id = self.concept_id_mapping.get(original_concept.concept)
                        if original_concept_id:
                            concept_name_to_ids[original_concept_id] = (theme.theme_id, concept.concept_id)
                        break
                else:
                    # If no exact match, try partial matching
                    for original_concept in atomic_concepts_result.atomic_concepts:
                        if (original_concept.concept.lower() in concept.label.lower() or 
                            concept.label.lower() in original_concept.concept.lower()):
                            original_concept_id = self.concept_id_mapping.get(original_concept.concept)
                            if original_concept_id:
                                concept_name_to_ids[original_concept_id] = (theme.theme_id, concept.concept_id)
                            break
        
        return concept_name_to_ids
    
    async def _phase3_group_concepts_into_themes(self, atomic_concepts_result: ExtractedAtomicConceptsResponse) -> GroupedConceptsResponse:
        """Phase 3: Grouping into themes - Organize atomic concepts into themes"""
        from prompts import PHASE3_GROUP_CONCEPTS_INTO_THEMES_PROMPT
        
        # Separate "Other" concepts from meaningful concepts
        meaningful_concepts = []
        other_concepts = []
        
        for concept in atomic_concepts_result.atomic_concepts:
            if concept.concept.lower() in ["other", "miscellaneous", "unclassified"]:
                other_concepts.append(concept)
            else:
                meaningful_concepts.append(concept)
        
        self.verbose_reporter.stat_line(f"Phase 3 input: {len(meaningful_concepts)} meaningful concepts")
        
        # Only process meaningful concepts with the LLM if there are any
        if meaningful_concepts:
            # Format meaningful concepts for the prompt
            concepts_text = "\n".join([
                f"- {concept.concept}: {concept.description} (evidence: {', '.join(concept.evidence)})"
                for concept in meaningful_concepts
            ])
            
            prompt = PHASE3_GROUP_CONCEPTS_INTO_THEMES_PROMPT.format(
                survey_question=self.survey_question,
                atomic_concepts=concepts_text,
                total_concepts=len(meaningful_concepts),
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
            
            result = await self._invoke_with_retries(prompt, GroupedConceptsResponse, phase='phase3_codebook')
            self.concept_name_to_ids_phase3 = self._create_concept_tracking_for_phase3(atomic_concepts_result, result)
            
        else:
            # No meaningful concepts to group - create empty result
            result = GroupedConceptsResponse(themes=[], unassigned_concepts=[])
            self.verbose_reporter.stat_line("No meaningful concepts to group - creating empty themes")
        
        # Add "Other" theme if we have other concepts (from atomic concepts)
        if other_concepts:
            # Create atomic concepts for the "Other" theme
            other_atomic_concepts = []
            for i, concept in enumerate(other_concepts):
                other_atomic_concept = AtomicConceptGrouped(
                    concept_id=f"99.{i+1}",  # Use 99.x for Other concepts
                    label=concept.concept,
                    description=concept.description
                )
                other_atomic_concepts.append(other_atomic_concept)
            
            # Create the "Other" theme
            other_theme = ThemeGroup(
                theme_id="99",
                label="Other",
                description="Miscellaneous responses that don't fit into specific thematic categories",
                atomic_concepts=other_atomic_concepts
            )
            
            # Add to results
            result.themes.append(other_theme)
            self.verbose_reporter.stat_line(f"Added 'Other' theme with {len(other_concepts)} concepts")
        
        # NEW: Add unassigned clusters as concepts under "Other" theme
        # Find unassigned clusters from merged_clusters
        unassigned_clusters = [
            cluster for cluster in self.merged_clusters 
            if cluster.assigned_concept == "Unassigned"
        ]
        
        if unassigned_clusters:
            # Create or find the "Other" theme
            other_theme = next((theme for theme in result.themes if theme.theme_id == "99"), None)
            
            if not other_theme:
                # Create "Other" theme if it doesn't exist
                other_theme = ThemeGroup(
                    theme_id="99",
                    label="Other",
                    description="Unassigned clusters that don't fit into specific thematic categories",
                    atomic_concepts=[]
                )
                result.themes.append(other_theme)
            
            # Add each unassigned cluster as a concept
            existing_concept_count = len(other_theme.atomic_concepts)
            for idx, cluster in enumerate(unassigned_clusters):
                concept = AtomicConceptGrouped(
                    concept_id=f"99.{existing_concept_count + idx + 1}",  # Continue numbering
                    label=cluster.label,  # Use original cluster label
                    description=f"Unassigned cluster: {cluster.description}"
                )
                other_theme.atomic_concepts.append(concept)
                
                # Update tracking for this unassigned cluster
                self.concept_id_mapping[cluster.assigned_concept_id] = f"99.{existing_concept_count + idx + 1}"
            
            self.verbose_reporter.stat_line(f"Added {len(unassigned_clusters)} unassigned clusters as concepts to 'Other' theme")
        
        # Report statistics with proper accounting
        input_concepts = len(atomic_concepts_result.atomic_concepts)
        concepts_in_themes = sum(len(theme.atomic_concepts) for theme in result.themes)
        concepts_unassigned = len(result.unassigned_concepts)
        total_output_concepts = concepts_in_themes + concepts_unassigned
        
        # Show input vs output with breakdown
        meaningful_concepts_in_themes = sum(len(theme.atomic_concepts) for theme in result.themes if theme.theme_id != "99")
        other_concepts_in_themes = sum(len(theme.atomic_concepts) for theme in result.themes if theme.theme_id == "99")
        
        self.verbose_reporter.stat_line(f"Input: {len(meaningful_concepts)} meaningful + {len(other_concepts)} 'Other' = {input_concepts} total concepts")
        self.verbose_reporter.stat_line(f"Output: {meaningful_concepts_in_themes} in meaningful themes + {other_concepts_in_themes} in 'Other' theme + {concepts_unassigned} unassigned = {total_output_concepts} total")
        
        # Show breakdown by theme
        meaningful_themes = [theme for theme in result.themes if theme.theme_id != "99"]
        other_themes = [theme for theme in result.themes if theme.theme_id == "99"]
        
        if meaningful_themes:
            self.verbose_reporter.stat_line(f"Grouped {meaningful_concepts_in_themes} meaningful concepts into {len(meaningful_themes)} themes")
            for theme in meaningful_themes:
                self.verbose_reporter.stat_line(f"• {theme.label}: {len(theme.atomic_concepts)} concepts", bullet="  ")
        
        if other_themes:
            for theme in other_themes:
                self.verbose_reporter.stat_line(f"• {theme.label}: {len(theme.atomic_concepts)} concepts (automatically added)", bullet="  ")
        
        if result.unassigned_concepts:
            self.verbose_reporter.stat_line(f"⚠️ {len(result.unassigned_concepts)} concepts remain unassigned: {result.unassigned_concepts}", bullet="  ")
            
        # Verify no concepts were lost
        if input_concepts != total_output_concepts:
            self.verbose_reporter.stat_line(f"⚠️ WARNING: Concept count mismatch! Input: {input_concepts}, Output: {total_output_concepts}", bullet="  ")
            
            # Find which concepts are missing
            input_concept_names = {concept.concept for concept in atomic_concepts_result.atomic_concepts}
            output_concept_names = set()
            
            # Collect concepts from themes
            for theme in result.themes:
                for concept in theme.atomic_concepts:
                    output_concept_names.add(concept.label)
            
            # Collect concepts from unassigned
            for unassigned_concept in result.unassigned_concepts:
                output_concept_names.add(unassigned_concept)
            
            missing_concepts = input_concept_names - output_concept_names
            if missing_concepts:
                self.verbose_reporter.stat_line(f"⚠️ Missing concepts: {sorted(missing_concepts)}", bullet="  ")
            
            # Also check for concepts in output that weren't in input (renamed concepts)
            extra_concepts = output_concept_names - input_concept_names
            if extra_concepts:
                self.verbose_reporter.stat_line(f"⚠️ New concepts in output: {sorted(extra_concepts)}", bullet="  ")
        
        return result
    
    def _prepare_phase4_tracking_prompt(self, grouped_concepts: GroupedConceptsResponse, concept_assignments: dict, merged_clusters: List[ClusterLabel]) -> str:
        """Prepare codebook for Phase 4 with stable ID tracking"""
        formatted = []
        formatted.append("CODEBOOK WITH CLUSTER ASSIGNMENTS AND TRACKING")
        formatted.append("=" * 60)
        
        cluster_lookup = {c.cluster_id: c for c in merged_clusters}
        
        # Create reverse mapping: concept label → stable ID
        concept_to_stable_id = {}
        for stable_id, (theme_id, concept_id) in self.concept_name_to_ids_phase3.items():
            # Find the concept in grouped_concepts
            for theme in grouped_concepts.themes:
                if theme.theme_id == theme_id:
                    for concept in theme.atomic_concepts:
                        if concept.concept_id == concept_id:
                            concept_to_stable_id[concept.label] = stable_id
                            break
        
        for theme in grouped_concepts.themes:
            formatted.append(f"\nTHEME [{theme.theme_id}]: {theme.label}")
            formatted.append(f"Description: {theme.description}")
            formatted.append("")
            
            for concept in theme.atomic_concepts:
                # Get stable ID for this concept
                stable_id = concept_to_stable_id.get(concept.label, f"unknown_{concept.concept_id}")
                
                # Look up cluster assignments
                assigned_cluster_ids = concept_assignments.get(concept.label, [])
                
                # If no direct match, try partial matching
                if not assigned_cluster_ids:
                    for concept_name, cluster_ids in concept_assignments.items():
                        if (concept_name.lower() in concept.label.lower() or 
                            concept.label.lower() in concept_name.lower()):
                            assigned_cluster_ids = cluster_ids
                            break
                
                cluster_count = len(assigned_cluster_ids)
                
                formatted.append(f"  CONCEPT [{concept.concept_id}]: {concept.label}")
                formatted.append(f"  Stable ID: {stable_id}")  # Include stable ID in prompt
                formatted.append(f"  Description: {concept.description}")
                formatted.append(f"  Assigned clusters: {cluster_count}")
                
                # Add example quotes
                example_quotes = []
                for cluster_id_str in assigned_cluster_ids[:2]:
                    cluster_id = int(cluster_id_str)
                    if cluster_id in cluster_lookup:
                        cluster = cluster_lookup[cluster_id]
                        if cluster.representatives:
                            example_quotes.append(cluster.representatives[0][0])
                
                if example_quotes:
                    formatted.append(f"  Examples: {' | '.join(example_quotes[:2])}")
                formatted.append("")
        
        return "\n".join(formatted)
        
    async def _phase4_label_refinement_with_assignments(self, grouped_concepts: GroupedConceptsResponse,  merged_clusters: List[ClusterLabel]) -> RefinedCodebook:
        """Phase 4: Label refinement with assignment statistics"""
        from prompts import PHASE4_LABEL_REFINEMENT_PROMPT
        
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
        codebook_with_counts = self._prepare_phase4_tracking_prompt(grouped_concepts, concept_assignments, merged_clusters)
        self.codebook = codebook_with_counts 
        
        prompt = PHASE4_LABEL_REFINEMENT_PROMPT.format(
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
        
        # Create Phase 4 tracking using stable IDs from the response
        self.stable_id_to_phase4_ids = {}
        
        label_to_stable_id = {}
        for stable_id, (theme_id, concept_id) in self.concept_name_to_ids_phase3.items():
            # Find the concept label in grouped_concepts
            for theme in grouped_concepts.themes:
                if theme.theme_id == theme_id:
                    for concept in theme.atomic_concepts:
                        if concept.concept_id == concept_id:
                            label_to_stable_id[concept.label.lower()] = stable_id
                            break
        
        for theme in result.refined_codebook.themes:
            for concept in theme.atomic_concepts:
                stable_id = None
                
                # First, check if LLM provided stable_id
                if hasattr(concept, 'stable_id') and concept.stable_id:
                    stable_id = concept.stable_id
                    # self.verbose_reporter.stat_line(
                    #     f"Phase 4 mapping: {stable_id} → {theme.theme_id}.{concept.concept_id} ({concept.label})",
                    #     bullet="  ")
                else:
                    # Fallback: try to find stable ID by matching concept labels
                    # This handles cases where the LLM didn't include stable_id
                    for orig_label, sid in label_to_stable_id.items():
                        if (orig_label in concept.label.lower() or 
                            concept.label.lower() in orig_label):
                            stable_id = sid
                            # self.verbose_reporter.stat_line(
                            #     f"Phase 4 mapping (fallback): {stable_id} → {theme.theme_id}.{concept.concept_id} ({concept.label})",
                            #     bullet="  ")
                            break
                    
                    if not stable_id:
                        self.verbose_reporter.stat_line(
                            f"⚠️ No stable_id found for concept {concept.concept_id} ({concept.label})",
                            bullet="  "
                        )
                
                if stable_id:
                    self.stable_id_to_phase4_ids[stable_id] = (theme.theme_id, concept.concept_id)
    
        self.verbose_reporter.stat_line(f"Refined {len(self.stable_id_to_phase4_ids)} concepts")
        self.verbose_reporter.stat_line(f"Refined {len(result.refined_codebook.themes)} themes")
            
        return result.refined_codebook
    
    
    
    def _format_codebook_with_assignments(self, grouped_concepts: GroupedConceptsResponse, concept_assignments: dict, merged_clusters: List[ClusterLabel]) -> str:
        """Format codebook with cluster assignment counts for refinement prompt"""
        formatted = []
        formatted.append("CODEBOOK WITH CLUSTER ASSIGNMENTS")
        formatted.append("=" * 60)
        
        # total_clusters = len(merged_clusters)
        cluster_lookup = {c.cluster_id: c for c in merged_clusters}
        
        for theme in grouped_concepts.themes:
            formatted.append(f"\nTHEME [{theme.theme_id}]: {theme.label}")
            formatted.append(f"Description: {theme.description}")
            formatted.append("")
            
            for concept in theme.atomic_concepts:
                # Look up by concept label/name since concept_assignments uses names as keys
                assigned_cluster_ids = concept_assignments.get(concept.label, [])
                
                # If no direct match, try partial matching
                if not assigned_cluster_ids:
                    for concept_name, cluster_ids in concept_assignments.items():
                        if (concept_name.lower() in concept.label.lower() or 
                            concept.label.lower() in concept_name.lower()):
                            assigned_cluster_ids = cluster_ids
                            break
                
                cluster_count = len(assigned_cluster_ids)
                
                formatted.append(f"  CONCEPT [{concept.concept_id}]: {concept.label}")
                formatted.append(f"  Description: {concept.description}")
                formatted.append(f"  Assigned clusters: {cluster_count}")
                
                # Add example quotes from assigned clusters
                example_quotes = []
                for cluster_id_str in assigned_cluster_ids[:2]:  # Take first 2 examples
                    cluster_id = int(cluster_id_str)  # Convert string to int
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
            #self.verbose_reporter.stat_line(f"  {theme.description}", bullet="  ")
            
            for concept in theme.atomic_concepts:
                self.verbose_reporter.stat_line(f"[{concept.concept_id}] {concept.label}",bullet="    ")
                #self.verbose_reporter.stat_line(f"{concept.description}",bullet="      ")
        
        # Count totals
        total_themes = len(refined_codebook.themes)
        total_concepts = sum(len(theme.atomic_concepts) for theme in refined_codebook.themes)
        
        self.verbose_reporter.stat_line(f"\nSummary: {total_themes} themes, {total_concepts} concepts")
    
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
        """Create mapping from original cluster IDs to merged cluster IDs using original_cluster_ids tracking"""
        mapping = {}
        
        # Use the original_cluster_ids field to create precise mapping
        for merged_cluster in merged_clusters:
            for original_id in merged_cluster.original_cluster_ids:
                mapping[original_id] = merged_cluster.cluster_id
        
        # Verify that all original clusters are mapped
        all_original_ids = set(c.cluster_id for c in original_clusters)
        mapped_ids = set(mapping.keys())
        missing_ids = all_original_ids - mapped_ids
        
        if missing_ids:
            self.verbose_reporter.stat_line(f"⚠️ Warning: {len(missing_ids)} original cluster IDs not found in mapping: {sorted(missing_ids)}")
            # Create fallback mapping for missing IDs
            for missing_id in missing_ids:
                # Find the original cluster and try to match by label
                orig_cluster = next((c for c in original_clusters if c.cluster_id == missing_id), None)
                if orig_cluster:
                    # Try to find a merged cluster with similar label
                    merged_cluster = next(
                        (c for c in merged_clusters if orig_cluster.label in c.label or c.label in orig_cluster.label),
                        merged_clusters[0] if merged_clusters else None  # Fallback to first merged cluster
                    )
                    if merged_cluster:
                        mapping[missing_id] = merged_cluster.cluster_id
        
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


