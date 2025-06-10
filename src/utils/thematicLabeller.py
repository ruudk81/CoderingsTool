import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

import asyncio
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from config import LabellerConfig
import models
from utils.verboseReporter import VerboseReporter


class CodebookEntry(BaseModel):
    """Single entry in the hierarchical codebook"""
    id: str = Field(description="Unique identifier - int for themes, float-like for topics")
    numeric_id: float = Field(description="Numeric ID for model compatibility")
    level: int = Field(description="1 for theme, 2 for topic, 3 for code")
    label: str = Field(description="Concise label (max 4 words)")
    description: Optional[str] = Field(default="", description="Clear description")
    parent_id: Optional[str] = Field(default=None, description="ID of parent")
    parent_numeric_id: Optional[float] = Field(default=None, description="Numeric ID of parent")
    source_codes: List[int] = Field(default_factory=list, description="For codes: original cluster IDs from Phase 1. For themes/topics: empty")

class Codebook(BaseModel):
    """Complete hierarchical codebook from Phase 4"""
    survey_question: str
    themes: List[CodebookEntry]
    topics: List[CodebookEntry] 
    codes: List[CodebookEntry]

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

class MergedGroup(BaseModel):
    """A group of merged labels"""
    new_cluster_id: int
    merged_label: str
    original_cluster_ids: List[int]

class UnchangedLabel(BaseModel):
    """A label that wasn't merged"""
    new_cluster_id: int
    label: str
    original_cluster_id: int

class LabelMergerResponse(BaseModel):
    """Response from Phase 2 label merger"""
    merged_groups: List[MergedGroup] = Field(default_factory=list)
    unchanged_labels: List[UnchangedLabel] = Field(default_factory=list)

class ExtractedThemesResponse(BaseModel):
    """Response from Phase 3 theme extraction"""
    analytical_notes: str = Field(description="Working notes from analysis")
    themes: List[str] = Field(description="List of main themes identified")
    conceptual_insights: Dict[str, str] = Field(description="Insights for each theme")

class RefinementEntry(BaseModel):
    """Single refinement entry"""
    label: str
    description: str

class LabelRefinementResponse(BaseModel):
    """Response from Phase 5 label refinement"""
    refined_themes: Dict[str, RefinementEntry] = Field(description="Refined theme labels by ID")
    refined_topics: Dict[str, RefinementEntry] = Field(description="Refined topic labels by ID")
    refined_codes: Dict[str, RefinementEntry] = Field(description="Refined code labels by ID")

class AssignmentResponse(BaseModel):
    """Response from Phase 6 assignment"""
    primary_assignment: Dict[str, str] = Field(description="Direct mapping: theme_id, topic_id, code_id")
    confidence: float = Field(description="Confidence in assignment")
    alternatives: Optional[List[Dict]] = Field(default=None, description="Alternative assignments if confidence is low")

class ThemeAssignment(BaseModel):
    """Direct assignment for a single cluster"""
    cluster_id: int
    theme_id: str
    topic_id: str
    code_id: str   
    confidence: float = 1.0
    alternative_assignments: Optional[List[Dict]] = None


class ThematicLabeller:
    """Orchestrator for thematic analysis - Simplified 6-phase workflow"""
    
    def __init__(self, config: LabellerConfig = None, verbose: bool = False, prompt_printer = None): 
        self.config = config or LabellerConfig()
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
        self.captured_phase5 = False
        self.captured_phase6 = False
        
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
                                   max_retries: int = None, model_override: str = None) -> Any:
        """Invoke LLM with retry logic and optional model override"""
        if max_retries is None:
            max_retries = self.config.max_retries
        
        # Use model override if provided, otherwise use default
        model = model_override or self.config.model
            
        last_error = None
        
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=model,
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
                    self.verbose_reporter.stat_line(f"Retry {attempt + 1}/{max_retries} after error: {str(e)}")
        
        raise RuntimeError(f"Failed after {max_retries} attempts. Last error: {str(last_error)}")
    
    async def process_hierarchy_async(self, cluster_models: List[models.ClusterModel], survey_question: str) -> List[models.LabelModel]:
        """Simplified 6-phase processing workflow"""
        self.survey_question = survey_question
        
        self.verbose_reporter.section_header("HIERARCHICAL LABELING PROCESS (6 PHASES)", emoji="🔄")
        
        # Extract micro-clusters
        initial_clusters = self._extract_initial_clusters(cluster_models)
        self.verbose_reporter.stat_line(f"Found {len(initial_clusters)} unique response segments")
        
        # =============================================================================
        # Phase 1: Descriptive Coding
        # =============================================================================
    
        self.verbose_reporter.step_start("Phase 1: Descriptive Coding", emoji="📝")
        labeled_clusters = await self._phase1_descriptive_coding(initial_clusters)
        self.labeled_clusters = labeled_clusters
        self.verbose_reporter.step_complete(f"Generated {len(labeled_clusters)} segment labels")
      
        # =============================================================================
        # Phase 2: Label Merger 
        # =============================================================================
        
        self.verbose_reporter.step_start("Phase 2: Label Merger", emoji="🔗")
        merged_clusters = await self._phase2_label_merger(labeled_clusters)
        self.merged_clusters = merged_clusters
        self.verbose_reporter.step_complete(f"Merged to {len(merged_clusters)} unique labels")
        
        # =============================================================================
        # Phase 3: Theme Discovery 
        # =============================================================================

        self.verbose_reporter.step_start("Phase 3: Theme Discovery", emoji="🔍")
        themes = await self._phase3_extract_themes(merged_clusters)
        self.verbose_reporter.step_complete("Themes discovered")

        # =============================================================================
        # Phase 4: Create codebook
        # =============================================================================

        self.verbose_reporter.step_start("Phase 4: Codebook Creation", emoji="📚")
        self.codebook = await self._phase4_create_codebook(merged_clusters, themes)
        self.verbose_reporter.step_complete("Codebook created")
         
        # =============================================================================
        # Phase 5: Label Refinement  
        # =============================================================================
        
        self.verbose_reporter.step_start("Phase 5: Label Refinement", emoji="✨")
        await self._phase5_label_refinement(self.codebook)
        self.verbose_reporter.step_complete("Labels polished")
        
        # =============================================================================
        # Phase 6: Assignment
        # =============================================================================
        
        self.verbose_reporter.step_start("Phase 6: Assignment", emoji="🎯")
        assignments = await self._phase6_assignment(merged_clusters, self.codebook)
        self.verbose_reporter.step_complete("Themes assigned to clusters")
        
        # Remove empty codes after assignment
        self._remove_empty_codes(self.codebook)
    
        # Print codebook
        self._display_full_codebook(self.codebook)
        
        # Create mapping from original to merged cluster IDs
        original_to_merged_mapping = self._create_cluster_mapping(labeled_clusters, merged_clusters)
        
        # Create final labels using merged clusters but map back to original IDs
        self.final_labels = self._create_final_labels_with_mapping(merged_clusters, assignments, original_to_merged_mapping)
        
        self.verbose_reporter.stat_line("✅ Applying hierarchy to responses...")
        result = self._apply_hierarchy_to_responses(cluster_models, self.final_labels, self.codebook)
        self._print_assignment_diagnostics(self.final_labels, initial_clusters)
        self.verbose_reporter.stat_line("🎉 Hierarchical labeling complete!")
        self._print_summary(self.codebook)

        return result
    
    async def _phase1_descriptive_coding(self, initial_clusters: Dict[int, Dict]) -> List[ClusterLabel]:
        """Phase 1: Descriptive coding """
        from prompts import PHASE1_DESCRIPTIVE_CODING_PROMPT
       
        labeled_clusters = []
        tasks = []
        for cluster_id, cluster_data in sorted(initial_clusters.items()):
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
                        "model": self.config.model,
                        "survey_question": self.survey_question,
                        "language": self.config.language,
                        "phase": "1/6 - Descriptive Coding",
                        "cluster_id": cluster_id
                    }
                )
                self.captured_phase1 = True
            
            task = self._invoke_with_retries(prompt, DescriptiveCodingResponse)
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
    
    async def _phase2_label_merger(self, labeled_clusters: List[ClusterLabel]) -> List[ClusterLabel]:
        """Phase 2: Merge semantically identical descriptions (using descriptions as labels)"""
        from prompts import PHASE2_LABEL_MERGER_PROMPT
        
        # Format descriptions as labels for the merger prompt
        labels_text = "\n".join([
            f"[ID: {c.cluster_id:2d}] {c.description or c.label}" 
            for c in sorted(labeled_clusters, key=lambda x: x.cluster_id)
        ])
        
        prompt = PHASE2_LABEL_MERGER_PROMPT.format(
            survey_question=self.survey_question,
            labels=labels_text,
            language=self.config.language
        )
        
        # Capture prompt
        if self.prompt_printer and not self.captured_phase2:
            self.prompt_printer.capture_prompt(
                step_name="hierarchical_labeling",
                utility_name="ThematicLabeller",
                prompt_content=prompt,
                prompt_type="phase2_label_merger",
                metadata={
                    "model": self.config.model,
                    "survey_question": self.survey_question,
                    "language": self.config.language,
                    "phase": "2/6 - Label Merger"
                }
            )
            self.captured_phase2 = True
        
        try:
            merger_result = await self._invoke_with_retries(prompt, LabelMergerResponse)
            self.merger_result = merger_result  # Store for mapping later
            
            # Create new merged cluster labels
            merged_clusters = []
            
            # Process merged groups
            for group in merger_result.merged_groups:
                # Find representatives and descriptions from all original clusters in the group
                all_representatives = []
                descriptions = []
                for original_id in group.original_cluster_ids:
                    original_cluster = next((c for c in labeled_clusters if c.cluster_id == original_id), None)
                    if original_cluster:
                        all_representatives.extend(original_cluster.representatives)
                        if original_cluster.description:
                            descriptions.append(original_cluster.description)
                
                # Sort by similarity and take top k
                all_representatives.sort(key=lambda x: x[1], reverse=True)
                top_representatives = all_representatives[:self.config.top_k_representatives]
                
                # Use descriptions as the label, and keep the merged_label as description
                primary_description = descriptions[0] if descriptions else group.merged_label
                
                merged_cluster = ClusterLabel(
                    cluster_id=group.new_cluster_id,
                    label=primary_description,  # Use description as the label
                    description=group.merged_label,  # Keep merged label as secondary info
                    representatives=top_representatives
                )
                merged_clusters.append(merged_cluster)
            
            # Process unchanged labels
            for unchanged in merger_result.unchanged_labels:
                original_cluster = next((c for c in labeled_clusters if c.cluster_id == unchanged.original_cluster_id), None)
                if original_cluster:
                    unchanged_cluster = ClusterLabel(
                        cluster_id=unchanged.new_cluster_id,
                        label=original_cluster.description or unchanged.label,  # Use description as the label
                        description=unchanged.label,  # Keep unchanged label as secondary info
                        representatives=original_cluster.representatives
                    )
                    merged_clusters.append(unchanged_cluster)
            
            # Sort by new cluster ID
            merged_clusters.sort(key=lambda x: x.cluster_id)
            
            # Log merging statistics
            original_count = len(labeled_clusters)
            merged_count = len(merged_clusters)
            groups_merged = len(merger_result.merged_groups)
            
            self.verbose_reporter.stat_line(f"Original clusters: {original_count}")
            self.verbose_reporter.stat_line(f"After merging: {merged_count}")
            self.verbose_reporter.stat_line(f"Merged groups: {groups_merged}")
            self.verbose_reporter.stat_line(f"Reduction: {original_count - merged_count} clusters")
            
            return merged_clusters
            
        except Exception as e:
            self.verbose_reporter.stat_line(f"⚠️ Label merger failed: {str(e)}, proceeding with original labels")
            
            # Create fallback merger result for mapping
            self.merger_result = LabelMergerResponse(
                merged_groups=[],
                unchanged_labels=[
                    UnchangedLabel(
                        new_cluster_id=cluster.cluster_id,
                        label=cluster.label,
                        original_cluster_id=cluster.cluster_id
                    ) for cluster in labeled_clusters
                ]
            )
            
            return labeled_clusters
    
    
    async def _phase3_extract_themes(self, labeled_clusters: List[ClusterLabel]) -> ExtractedThemesResponse:
        """Phase 3: Extract themes using better model"""
        from prompts import PHASE3_EXTRACT_THEMES_PROMPT
        
        # Format codes for the prompt
        codes_text = "\n".join([
            f"[source ID: {c.cluster_id:2d}] {c.label}" 
            for c in sorted(labeled_clusters, key=lambda x: x.cluster_id)
        ])
        
        prompt = PHASE3_EXTRACT_THEMES_PROMPT.format(
            survey_question=self.survey_question,
            codes=codes_text,
            language=self.config.language
        )
        
        # Capture prompt
        if self.prompt_printer and not self.captured_phase3:
            self.prompt_printer.capture_prompt(
                step_name="hierarchical_labeling",
                utility_name="ThematicLabeller",
                prompt_content=prompt,
                prompt_type="phase3_extract_themes",
                metadata={
                    "model": self.config.phase3_extract_model,  # Note: using phase3 model
                    "survey_question": self.survey_question,
                    "language": self.config.language,
                    "phase": "3/6 - Extract Themes"
                }
            )
            self.captured_phase3 = True
        
        # Use phase3_extract_model (gpt-4o) for this phase
        result = await self._invoke_with_retries(
            prompt, 
            ExtractedThemesResponse,
            model_override=self.config.phase3_extract_model
        )
        
        self.verbose_reporter.stat_line(f"Extracted {len(result.themes)} main themes")
        for theme in result.themes:
            self.verbose_reporter.stat_line(f"• {theme}", bullet="  ")
        
        # Report analytical insights if verbose
        if self.verbose_reporter.enabled and result.conceptual_insights:
            self.verbose_reporter.stat_line("Key conceptual insights:", bullet="  ")
            for theme, insight in result.conceptual_insights.items():
                self.verbose_reporter.stat_line(f"• {theme}: {insight[:100]}{'...' if len(insight) > 100 else ''}", bullet="    ")
        
        return result
    
    async def _phase4_create_codebook(self, labeled_clusters: List[ClusterLabel], 
                                     themes_result: ExtractedThemesResponse) -> Codebook:
        """Phase 4: Create hierarchical codebook"""
        from prompts import PHASE4_CREATE_CODEBOOK_PROMPT
        
        # Format themes and codes
        themes_text = "\n".join([f"- {theme}" for theme in themes_result.themes])
        codes_text = "\n".join([
            f"[source ID: {c.cluster_id:2d}] {c.label}" 
            for c in sorted(labeled_clusters, key=lambda x: x.cluster_id)
        ])
        
        prompt = PHASE4_CREATE_CODEBOOK_PROMPT.format(
            survey_question=self.survey_question,
            themes=themes_text,
            merged_clusters=codes_text,
            language=self.config.language
        )
        
        # Capture prompt
        if self.prompt_printer and not self.captured_phase4:
            self.prompt_printer.capture_prompt(
                step_name="hierarchical_labeling",
                utility_name="ThematicLabeller",
                prompt_content=prompt,
                prompt_type="phase4_create_codebook",
                metadata={
                    "model": self.config.model,  # Back to default model
                    "survey_question": self.survey_question,
                    "language": self.config.language,
                    "phase": "4/6 - Create Codebook"
                }
            )
            self.captured_phase4 = True
        
        # Create a simple response model for parsing
        class CodebookResponse(BaseModel):
            themes: List[Dict[str, Any]]
        
        result = await self._invoke_with_retries(prompt, CodebookResponse)
        
        # Parse result into Codebook
        codebook = self._parse_codebook(result.model_dump())
        
        # Report statistics
        self.verbose_reporter.stat_line(f"Created codebook with {len(codebook.themes)} themes")
        self.verbose_reporter.stat_line(f"Organized into {len(codebook.topics)} topics")
        self.verbose_reporter.stat_line(f"Containing {len(codebook.codes)} codes")
        
        return codebook
    
    async def _phase5_label_refinement(self, codebook: Codebook) -> None:
        """Phase 5: Refine all labels for clarity and consistency"""
        from prompts import PHASE5_LABEL_REFINEMENT_PROMPT
        
        codebook_text = self.format_codebook_prompt(codebook)
        
        prompt = PHASE5_LABEL_REFINEMENT_PROMPT.format(
            survey_question=self.survey_question,
            codebook=codebook_text,
            language=self.config.language)
        
        # Capture Phase 5 prompt
        if self.prompt_printer and not self.captured_phase5:
            self.prompt_printer.capture_prompt(
                step_name="hierarchical_labeling",
                utility_name="ThematicLabeller",
                prompt_content=prompt,
                prompt_type="phase5_label_refinement",
                metadata={
                    "model": self.config.model,
                    "survey_question": self.survey_question,
                    "language": self.config.language,
                    "phase": "5/6 - Label Refinement"
                }
            )
            self.captured_phase5 = True
        
        try:
            result = await self._invoke_with_retries(prompt, LabelRefinementResponse)
            
            # Apply refinements
            refinement_count = 0
            
            # Create lookups
            theme_lookup = {t.id: t for t in codebook.themes}
            topic_lookup = {t.id: t for t in codebook.topics}
            code_lookup = {c.id: c for c in codebook.codes}
            
            # Apply theme refinements
            for theme_id, refinement in result.refined_themes.items():
                if theme_id in theme_lookup:
                    theme = theme_lookup[theme_id]
                    old_label = theme.label
                    old_description = theme.description
                    
                    # Update if either label or description changed
                    if theme.label != refinement.label or theme.description != refinement.description:
                        theme.label = refinement.label
                        theme.description = refinement.description
                        refinement_count += 1
                        
                        if old_label != refinement.label:
                            self.verbose_reporter.stat_line(
                                f"🏷️  Theme {theme_id}: '{old_label}' → '{refinement.label}'", 
                                bullet="      "
                            )
                        if old_description != refinement.description:
                            self.verbose_reporter.stat_line(
                                f"📝  Theme {theme_id} description updated", 
                                bullet="      "
                            )
            
            # Apply topic refinements
            for topic_id, refinement in result.refined_topics.items():
                if topic_id in topic_lookup:
                    topic = topic_lookup[topic_id]
                    old_label = topic.label
                    old_description = topic.description
                    
                    if topic.label != refinement.label or topic.description != refinement.description:
                        topic.label = refinement.label
                        topic.description = refinement.description
                        refinement_count += 1
                        
                        if old_label != refinement.label:
                            self.verbose_reporter.stat_line(
                                f"🏷️  Topic {topic_id}: '{old_label}' → '{refinement.label}'", 
                                bullet="      "
                            )
                        if old_description != refinement.description:
                            self.verbose_reporter.stat_line(
                                f"📝  Topic {topic_id} description updated", 
                                bullet="      "
                            )
            
            # Apply code refinements
            for code_id, refinement in result.refined_codes.items():
                if code_id in code_lookup:
                    code = code_lookup[code_id]
                    old_label = code.label
                    old_description = code.description
                    
                    if code.label != refinement.label or code.description != refinement.description:
                        code.label = refinement.label
                        code.description = refinement.description
                        refinement_count += 1
                        
                        if old_label != refinement.label:
                            self.verbose_reporter.stat_line(
                                f"🏷️  Code {code_id}: '{old_label}' → '{refinement.label}'", 
                                bullet="      "
                            )
                        if old_description != refinement.description:
                            self.verbose_reporter.stat_line(
                                f"📝  Code {code_id} description updated", 
                                bullet="      "
                            )
            
            self.verbose_reporter.stat_line(f"✨ Applied {refinement_count} label/description refinements")
            
        except Exception as e:
            self.verbose_reporter.stat_line(f"⚠️ Label refinement failed: {str(e)}, keeping original labels")
        
        
    async def _phase6_assignment(self, labeled_clusters: List[ClusterLabel], codebook: Codebook) -> List[ThemeAssignment]:
        """Phase 6: Assignment - Assign clusters to the refined hierarchy"""
        assignments = []
        from prompts import PHASE6_ASSIGNMENT_PROMPT
       
        codebook_text = self.format_codebook_prompt(codebook)
        
        # Process each cluster
        tasks = []
        for cluster in labeled_clusters:
            # Format representatives
            seen = set()
            unique_texts = []
            for text, _ in cluster.representatives:
                if text not in seen:
                    seen.add(text)
                    unique_texts.append(text)
            representatives = '- ' + '\n- '.join(unique_texts)
            
            prompt = PHASE6_ASSIGNMENT_PROMPT.format(
                survey_question=self.survey_question,
                cluster_id=cluster.cluster_id,
                cluster_label=cluster.label,
                cluster_representatives=representatives,
                codebook=codebook_text,
                language=self.config.language)
            
            # Capture Phase 6 prompt only for the first cluster
            if self.prompt_printer and not self.captured_phase6:
                self.prompt_printer.capture_prompt(
                    step_name="hierarchical_labeling",
                    utility_name="ThematicLabeller",
                    prompt_content=prompt,
                    prompt_type="phase6_assignment",
                    metadata={
                        "model": self.config.model,
                        "survey_question": self.survey_question,
                        "language": self.config.language,
                        "phase": "6/6 - Assignment",
                        "cluster_id": cluster.cluster_id
                    }
                )
                self.captured_phase6 = True
            
            task = self._invoke_with_retries(prompt, AssignmentResponse)
            tasks.append((cluster, task))
        
        # Execute in batches
        print(f"  Assigning {len(tasks)} clusters to hierarchy...")
        assigned_count = 0
        other_count = 0
        
        for i in range(0, len(tasks), self.config.concurrent_requests):
            batch = tasks[i:i + self.config.concurrent_requests]
            batch_results = await asyncio.gather(*[task for _, task in batch], return_exceptions=True)
            
            for (cluster, _), result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    print(f"    ⚠️ Error assigning cluster {cluster.cluster_id}: {str(result)}")
                    # Fallback to "other"
                    assignment = ThemeAssignment(
                        cluster_id=cluster.cluster_id,
                        theme_id="99",
                        topic_id="99.1",
                        code_id="99.1.1",
                        confidence=0.0
                    )
                else:
                    # Direct assignment from LLM
                    assignment = ThemeAssignment(
                        cluster_id=cluster.cluster_id,
                        theme_id=result.primary_assignment['theme_id'],
                        topic_id=result.primary_assignment['topic_id'],
                        code_id=result.primary_assignment['code_id'],
                        confidence=result.confidence,
                        alternative_assignments=result.alternatives
                    )
                    
                    # Check if assigned to "other"
                    if result.primary_assignment['code_id'] == "99.1.1":
                        other_count += 1
                    else:
                        assigned_count += 1
                        
                assignments.append(assignment)
        
        print(f"    ✓ Assigned {assigned_count} clusters to specific codes")
        if other_count > 0:
            print(f"    ℹ️ {other_count} clusters assigned to 'other'")
        
        # Update codebook with direct assignments
        self._update_codebook_with_direct_assignments(codebook, assignments)
        
        # Validate all clusters were assigned
        self._validate_assignments(labeled_clusters, assignments)
        
        return assignments
    
    def process_hierarchy(self, cluster_models: List[models.ClusterModel], survey_question: str) -> List[models.LabelModel]:
        """Sync wrapper for async processing"""
        # Handle nested event loops
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except ImportError:
            pass
        
        return asyncio.run(self.process_hierarchy_async(cluster_models, survey_question))

    def _parse_codebook(self, result: Dict) -> Codebook:
        """Parse JSON result into Codebook structure with cluster assignments from Phase 4"""
        themes = []
        topics = []
        codes = []
 
        if 'themes' in result:
            for theme_idx, theme_data in enumerate(result['themes'], 1):
                theme_id = str(theme_idx)
                theme_numeric_id = float(theme_idx)
                
                theme = CodebookEntry(
                    id=theme_id,
                    numeric_id=theme_numeric_id,
                    level=1,
                    label=theme_data['label'],
                    description=theme_data.get('description', ''),
                    source_codes=[]  # Themes don't have source codes
                )
                themes.append(theme)
                
                if 'topics' in theme_data:
                    for topic_idx, topic_data in enumerate(theme_data['topics'], 1):
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
                            source_codes=[]  # Topics don't have source codes
                        )
                        topics.append(topic)
                        
                        if 'codes' in topic_data:
                            for code_idx, code_data in enumerate(topic_data['codes'], 1):
                                code_id = f"{topic_id}.{code_idx}"
                                code_numeric_id = float(f"{topic_id}{code_idx}")
                                
                                source_codes = code_data.get('source_codes', [])
                                
                                code = CodebookEntry(
                                    id=code_id,
                                    numeric_id=code_numeric_id,
                                    level=3,
                                    label=code_data['label'],
                                    description=code_data.get('description', ''),
                                    parent_id=topic_id,
                                    parent_numeric_id=topic_numeric_id,
                                    source_codes=source_codes   
                                )
                                codes.append(code)
                        
        return Codebook(
            survey_question=self.survey_question,
            themes=themes,
            topics=topics,
            codes=codes)
    
    def format_codebook_prompt(self, codebook: Codebook, include_sources: bool = True) -> str:
        """Format Pydantic codebook for readable prompt with clear ID/label distinction"""
        formatted = []
        formatted.append("HIERARCHICAL CODEBOOK")
        formatted.append("=" * 80)
        formatted.append("Format: [ID] LABEL → Description")
        formatted.append("Note: Focus on LABEL meanings, not ID numbers!")
        formatted.append("=" * 80)
        
        # Group topics and codes by parent
        topics_by_theme = {}
        codes_by_topic = {}
        
        for topic in codebook.topics:
            if topic.parent_id not in topics_by_theme:
                topics_by_theme[topic.parent_id] = []
            topics_by_theme[topic.parent_id].append(topic)
        
        for code in codebook.codes:
            if code.parent_id not in codes_by_topic:
                codes_by_topic[code.parent_id] = []
            codes_by_topic[code.parent_id].append(code)
        
        # Build formatted string with clear visual separation
        for theme in codebook.themes:
            formatted.append(f"\n{'='*60}")
            formatted.append(f"THEME [ID: {theme.id}] → \"{theme.label}\"")
            if theme.description:
                formatted.append(f"   Description: {theme.description}")
            formatted.append(f"{'='*60}")
            
            for topic in topics_by_theme.get(theme.id, []):
                formatted.append(f"\n   TOPIC [ID: {topic.id}] → \"{topic.label}\"")
                if topic.description:
                    formatted.append(f"      Description: {topic.description}")
                formatted.append(f"   {'-'*50}")
                
                for code in codes_by_topic.get(topic.id, []):
                    formatted.append(f"      CODE [ID: {code.id}] → \"{code.label}\"")
                    if code.description:
                        formatted.append(f"         Description: {code.description}")
                    
                    if include_sources and code.source_codes:
                        # Include which clusters this code represents (use merged clusters)
                        source_labels = []
                        for cluster_id in code.source_codes:
                            if hasattr(self, 'merged_clusters'):
                                cluster = next((c for c in self.merged_clusters if c.cluster_id == cluster_id), None)
                                if cluster:
                                    source_labels.append(f"source_code {cluster_id}: \"{cluster.label}\"")
                                else:
                                    source_labels.append(f"source_code {cluster_id}")
                        
                        if source_labels:
                            formatted.append(f"         Source data: {' | '.join(source_labels)}")
                            if len(code.source_codes) > 5:
                                formatted.append(f"                      ... and {len(code.source_codes) - 5} more clusters")
                    formatted.append("")
        
        formatted.append("\n" + "="*80)
        formatted.append("REMINDER: Evaluate based on LABEL meanings, not ID numbers!")
        formatted.append("="*80)
        
        return "\n".join(formatted)
    
    def _update_codebook_with_direct_assignments(self, codebook: Codebook, assignments: List[ThemeAssignment]):
        """Update codebook with direct cluster assignments"""
        # Clear existing assignments
        for code in codebook.codes:
            code.source_codes = []
        
        # Track assignments
        assigned_clusters = set()
        multi_assignments = 0
        
        for assignment in assignments:
            cluster_id = assignment.cluster_id
            
            # Find and update the assigned code
            for code in codebook.codes:   
                if code.id == assignment.code_id:  
                    code.source_codes.append(cluster_id)
                    assigned_clusters.add(cluster_id)
                    break
            
            # Handle alternatives if confidence is low
            if assignment.alternative_assignments and assignment.confidence < 0.8:
                alternatives_used = []
                for alt in assignment.alternative_assignments:
                    if alt.get('confidence', 0) >= self.config.assignment_threshold:
                        for code in codebook.codes:   
                            if code.id == alt['code_id']:   
                                code.source_codes.append(cluster_id)
                                alternatives_used.append(alt['code_id'])  
                                break
                
                if alternatives_used:
                    multi_assignments += 1
                    print(f"    📌 Cluster {cluster_id} also assigned to alternatives: {alternatives_used}")
        
        print(f"    ✓ Assigned {len(assigned_clusters)} segments")
        if multi_assignments > 0:
            print(f"    ✓ {multi_assignments} clusters assigned to multiple codes via alternatives")  
    
    def _remove_empty_codes(self, codebook: Codebook) -> None:
       """Remove codes that have no clusters assigned"""
       codes_with_clusters = []
       codes_removed = []
       
       for code in codebook.codes:
           if code.source_codes:
               codes_with_clusters.append(code)
           else:
               codes_removed.append(code)

       codebook.codes = codes_with_clusters   
       
       # Also clean up empty topics (topics with no codes)
       topics_with_codes = []
       for topic in codebook.topics:
           # Check if any code has this topic as parent
           has_codes = any(c.parent_id == topic.id for c in codebook.codes)
           if has_codes:
               topics_with_codes.append(topic)
  
       codebook.topics = topics_with_codes
       
       # Also clean up empty themes (themes with no topics)
       themes_with_topics = []
       for theme in codebook.themes:
           # Check if any topic has this theme as parent
           has_topics = any(t.parent_id == theme.id for t in codebook.topics)
           if has_topics:
               themes_with_topics.append(theme)
       
       codebook.themes = themes_with_topics
       
       # Print summary
       if codes_removed:
           print(f"    🧹 Removed {len(codes_removed)} empty codes")
           # Optionally print which ones were removed
           for code in codes_removed[:5]:  # Show first 5
               print(f"       - {code.id}: {code.label}")
           if len(codes_removed) > 5:
               print(f"       ... and {len(codes_removed) - 5} more")
     
    def _create_final_labels(self, labeled_clusters: List[ClusterLabel], assignments: List[ThemeAssignment]) -> Dict[int, Dict]:
        """Create final labels dictionary from assignments"""
        final_labels = {}
        
        # Create lookup dictionaries
        cluster_lookup = {c.cluster_id: c for c in labeled_clusters}
        assignment_lookup = {a.cluster_id: a for a in assignments}
        
        # Process each cluster with direct assignments
        for cluster_id in cluster_lookup:
            cluster = cluster_lookup[cluster_id]
            assignment = assignment_lookup.get(cluster_id)
            
            if not assignment:
                # Handle missing assignments
                print(f"    ⚠️  No assignment found for cluster {cluster_id}, assigning to 'other'")
                final_labels[cluster_id] = {
                    'label': cluster.label,
                    'theme': ('99', 0.0),
                    'topic': ('99.1', 0.0),
                    'code': ('99.1.1', 0.0)
                }
            else:
                # Use direct assignments
                final_labels[cluster_id] = {
                    'label': cluster.label,
                    'theme': (assignment.theme_id, assignment.confidence),
                    'topic': (assignment.topic_id, assignment.confidence),
                    'code': (assignment.code_id, assignment.confidence)
                }
        
        return final_labels
    
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
    
    def _create_final_labels_with_mapping(self, merged_clusters: List[ClusterLabel], 
                                         assignments: List[ThemeAssignment], 
                                         original_to_merged_mapping: Dict[int, int]) -> Dict[int, Dict]:
        """Create final labels dictionary mapping original cluster IDs to assignments via merged clusters"""
        final_labels = {}
        
        # Create lookup dictionaries
        merged_cluster_lookup = {c.cluster_id: c for c in merged_clusters}
        assignment_lookup = {a.cluster_id: a for a in assignments}
        
        # Reverse the mapping to go from merged back to original
        merged_to_original = {}
        for orig_id, merged_id in original_to_merged_mapping.items():
            if merged_id not in merged_to_original:
                merged_to_original[merged_id] = []
            merged_to_original[merged_id].append(orig_id)
        
        # Process each merged cluster assignment and map back to original IDs
        for merged_cluster_id, assignment in assignment_lookup.items():
            merged_cluster = merged_cluster_lookup.get(merged_cluster_id)
            if not merged_cluster:
                continue
            
            # Get all original cluster IDs that map to this merged cluster
            original_ids = merged_to_original.get(merged_cluster_id, [merged_cluster_id])
            
            # Assign the same theme/topic/code to all original clusters that were merged
            for original_id in original_ids:
                final_labels[original_id] = {
                    'label': merged_cluster.label,  # Use the merged label
                    'theme': (assignment.theme_id, assignment.confidence),
                    'topic': (assignment.topic_id, assignment.confidence),
                    'code': (assignment.code_id, assignment.confidence)
                }
        
        return final_labels
    
    def _validate_assignments(self, labeled_clusters: List[ClusterLabel], assignments: List[ThemeAssignment]):
        """Validate that all clusters were assigned"""
        expected_ids = {c.cluster_id for c in labeled_clusters}
        assigned_ids = {a.cluster_id for a in assignments}
        
        # Get valid code IDs from current codebook
        valid_code_ids = {c.id for c in self.codebook.codes if hasattr(self, 'codebook')}  
     
        
        missing_ids = expected_ids - assigned_ids
        extra_ids = assigned_ids - expected_ids
        
        # Check if assignments point to non-existent codes 
        invalid_assignments = []
        for assignment in assignments:
            if assignment.code_id not in valid_code_ids and assignment.code_id != "99.1.1":   
                invalid_assignments.append(assignment.cluster_id)
        
        if missing_ids:
            print(f"    ⚠️ WARNING: {len(missing_ids)} clusters not assigned: {sorted(missing_ids)}")
        
        if extra_ids:
            print(f"    ⚠️ WARNING: {len(extra_ids)} unexpected assignments: {sorted(extra_ids)}")
        
        if invalid_assignments:
            print(f"    ⚠️ WARNING: {len(invalid_assignments)} clusters assigned to removed codes")
            
        if not missing_ids and not extra_ids and not invalid_assignments:
            print(f"    ✅ All {len(expected_ids)} clusters successfully assigned to valid codes")
    
    def _apply_hierarchy_to_responses(self, cluster_models: List[models.ClusterModel], 
                                     final_labels: Dict[int, Dict], 
                                     codebook: Codebook) -> List[models.LabelModel]:
        """Apply hierarchy to original response models with proper model compatibility"""
        # Create string ID lookups for assignment matching
        theme_str_lookup = {t.id: t for t in codebook.themes}
        topic_str_lookup = {t.id: t for t in codebook.topics}
        code_str_lookup = {c.id: c for c in codebook.codes}
        
        # Build hierarchical structure for the LabelModel
        hierarchical_themes = []
        for theme in codebook.themes:
            hier_theme = models.HierarchicalTheme(
                theme_id=theme.id,
                numeric_id=theme.numeric_id,
                label=theme.label,
                description=theme.description or "",
                level=theme.level,
                topics=[]
            )
            
            # Add topics to theme
            for topic in codebook.topics:
                if topic.parent_id == theme.id:
                    hier_topic = models.HierarchicalTopic(
                        topic_id=topic.id,
                        numeric_id=topic.numeric_id,
                        label=topic.label,
                        description=topic.description or "",
                        parent_id=topic.parent_id,
                        level=topic.level,
                        codes=[]
                    )
                    
                    # Add codes to topic
                    for code in codebook.codes:
                        if code.parent_id == topic.id:
                            hier_code = models.HierarchicalCode(
                                code_id=code.id,
                                numeric_id=code.numeric_id,
                                label=code.label,
                                description=code.description or "",
                                parent_id=code.parent_id,
                                level=code.level
                            )
                            hier_topic.codes.append(hier_code)
                    
                    hier_theme.topics.append(hier_topic)
            
            hierarchical_themes.append(hier_theme)
        
        # Build cluster mappings
        cluster_mappings = []
        for cluster_id, labels in final_labels.items():
            theme_id_str, confidence = labels['theme']
            topic_id_str, _ = labels['topic']
            code_id_str, _ = labels['code']
            
            mapping = models.ClusterMapping(
                cluster_id=cluster_id,
                cluster_label=labels['label'],
                theme_id=theme_id_str,
                topic_id=topic_id_str,
                code_id=code_id_str,
                confidence=confidence
            )
            cluster_mappings.append(mapping)
        
        label_models = []
        
        for cluster_model in cluster_models:
            # Convert to LabelModel using the to_model method
            label_model = cluster_model.to_model(models.LabelModel)
            
            # Force rebuild model to clear any cached schema
            if hasattr(label_model, 'model_rebuild'):
                label_model.model_rebuild()
            
            # Add hierarchical structure data (same for all models as it's survey-level)
            label_model.themes = hierarchical_themes
            label_model.cluster_mappings = cluster_mappings
            
            # Generate a summary for the model (optional)
            segment_count = len(label_model.response_segment) if label_model.response_segment else 0
            label_model.summary = f"Response with {segment_count} segments analyzed"
            
            # Apply hierarchy to segments
            if label_model.response_segment:
                for segment in label_model.response_segment:
                    if segment.initial_cluster is not None:
                        cluster_id = segment.initial_cluster
                        
                        if cluster_id in final_labels:
                            labels = final_labels[cluster_id]
                            
                            # Apply Theme (Dict[int, str])
                            theme_id_str, _ = labels['theme']
                            if theme_id_str in theme_str_lookup:
                                theme = theme_str_lookup[theme_id_str]
                                theme_id_int = int(theme.numeric_id)
                                description = f": {theme.description}" if theme.description else ""
                                segment.Theme = {theme_id_int: f"{theme.label}{description}"}
                            elif theme_id_str == "other":
                                segment.Theme = {999: "Other: Unclassified"}
                            
                            # Apply Topic (Dict[float, str])
                            topic_id_str, _ = labels['topic']
                            if topic_id_str in topic_str_lookup:
                                topic = topic_str_lookup[topic_id_str]
                                topic_id_float = topic.numeric_id  # Already a float from model
                                description = f": {topic.description}" if topic.description else ""
                                segment.Topic = {topic_id_float: f"{topic.label}{description}"}
                            elif topic_id_str == "other":
                                segment.Topic = {99.9: "Other: Unclassified"}
                            
                            # Apply Code (Dict[float, str])
                            code_id_str, _ = labels['code']
                            if code_id_str in code_str_lookup:
                                code = code_str_lookup[code_id_str]
                                code_id_float = code.numeric_id
                                description = f": {code.description}" if code.description else ""
                                segment.Code = {code_id_float: f"{code.label}{description}"}
                            elif code_id_str == "99.1.1":
                                segment.Code = {99.11: "Other: Unclassified"}
                        else:
                            # Handle unmapped clusters - assign to "other" category
                            segment.Theme = {999: "Other: Unmapped cluster"}
                            segment.Topic = {99.9: "Other: Unmapped cluster"}
                            segment.Code = {99.11: "Other: Unmapped cluster"}
            
            label_models.append(label_model)
        
        return label_models
    
    def _print_summary(self, codebook: Codebook):
        """Print summary of the generated hierarchy"""
        self.verbose_reporter.summary("Codebook Summary", {
            "Themes": len(codebook.themes),
            "Topics": len(codebook.topics),
            "Codes": len(codebook.codes),
            "Clusters assigned": len(set(cluster for code in codebook.codes for cluster in code.source_codes))
        }, emoji="📊")
        
    def _display_full_codebook(self, codebook: Codebook):
        """Display the full hierarchical codebook with all labels"""
        if not self.verbose_reporter.enabled:
            return
        self.verbose_reporter.section_header("FULL CODEBOOK HIERARCHY", emoji="📚")
        
        # Display themes
        for theme in sorted(codebook.themes, key=lambda x: x.numeric_id):
            self.verbose_reporter.stat_line(f"🎯 THEME {theme.id}: {theme.label}", bullet="")
            if theme.description:
                self.verbose_reporter.stat_line(f"Description: {theme.description}\n", bullet="   ")
            
            # Find related topics
            related_topics = [t for t in codebook.topics if t.parent_id == theme.id]
            if related_topics:
                for topic in sorted(related_topics, key=lambda x: x.numeric_id):
                    self.verbose_reporter.stat_line(f"📍 TOPIC {topic.id}: {topic.label}", bullet="   ")
                    if topic.description:
                        self.verbose_reporter.stat_line(f"Description: {topic.description}\n", bullet="      ")
                    
                    related_codes = [c for c in codebook.codes if c.parent_id == topic.id]
                    if related_codes:   
                        for code in sorted(related_codes, key=lambda x: x.numeric_id):  
                            self.verbose_reporter.stat_line(f"🔸 CODE {code.id}: {code.label}", bullet="       " )  
                            if code.description:
                                self.verbose_reporter.stat_line(f"Description: {code.description}", bullet="         " )  
                                
                            if code.source_codes:
                                self.verbose_reporter.stat_line(f"Micro-clusters: {code.source_codes}\n", bullet="         " )  
        
        print("\n" + "="*80)
    
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
        theme_other = sum(1 for labels in final_labels.values() if labels['theme'][0] == 'other')
        topic_other = sum(1 for labels in final_labels.values() if labels['topic'][0] == 'other')
        
        if theme_other > 0 or topic_other > 0:
            self.verbose_reporter.stat_line(f"Clusters assigned to 'other': {theme_other} themes, {topic_other} topics")


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
    
    print("\n=== Running Simplified Thematic Labelling (6 Phases) ===")
    print("1. Descriptive Coding - Label each micro-cluster")
    print("2. Label Merger - Merge semantically identical labels")
    print("3. Extract Themes - Identify main themes (using gpt-4o)")
    print("4. Create Codebook - Build hierarchical structure")
    print("5. Label Refinement - Polish all labels")
    print("6. Assignment - Assign clusters to hierarchy")
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
    codebook = labeller.codebook  # Complete hierarchical structure
    final_labels = labeller.final_labels  # Dict[cluster_id, label_info]
    
    print(f"\n✅ Successfully labeled {len(labeled_results)} responses")
    
    # Save to cache
    cache_manager.save_to_cache(labeled_results, filename, "labels")
    print("💾 Saved labeled results to cache")