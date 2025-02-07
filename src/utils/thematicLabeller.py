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


# Models for structured data
class CodebookEntry(BaseModel):
    """Single entry in the hierarchical codebook"""
    id: str = Field(description="Unique identifier - int for themes, float-like for topics")
    numeric_id: float = Field(description="Numeric ID for model compatibility")
    level: int = Field(description="1 for theme, 2 for topic, 3 for code")
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
    codes: List[CodebookEntry]


class ClusterLabel(BaseModel):
    """Initial cluster label from Phase 1"""
    cluster_id: int
    label: str = Field(description="Concise label (max 4 words)")
    description: str = Field(description="Detailed description of cluster theme")
    representatives: List[Tuple[str, float]] = Field(description="Representative descriptions with similarity scores")


class DescriptiveCodingResponse(BaseModel):
    """Response from Phase 1 descriptive coding"""
    label: str = Field(description="Concise thematic label (max 4 words)")
    description: str = Field(description="Clear explanation of cluster theme")
    
    
class DiscoveryResponse(BaseModel):
    """Response from Phase 2  discovery"""
    themes: List[Dict[str, Any]] = Field(description="List of themes with their structure")


class ThemeJudgmentResponse(BaseModel):
    """Response from Phase 3 theme judger"""
    is_logical: bool = Field(description="Whether the structure is meaningful and logical")
    feedback: Optional[str] = Field(description="Instructions for improvement if not logical")
    specific_issues: Optional[List[str]] = Field(description="Specific structural issues identified")

class ThemeReviewResponse(BaseModel):
    """Response from Phase 4 theme review"""
    themes: List[Dict[str, Any]] = Field(description="Improved theme structure")


class LabelRefinementResponse(BaseModel):
    """Response from Phase 4 label refinement"""
    refined_themes: Dict[str, str] = Field(description="Refined theme labels by ID")
    refined_topics: Dict[str, str] = Field(description="Refined topic labels by ID")
    refined_codes : Dict[str, str] = Field(description="Refined code labels by ID")
    

class AssignmentResponse(BaseModel):
    """Response from Phase 5 assignment"""
    primary_assignment: Dict[str, str] = Field(description="Direct mapping: theme_id, topic_id, code_id")  # Changed
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
    """Orchestrator for thematic analysis"""
    
    def __init__(self, config: LabellerConfig = None, verbose: bool = False): 
        self.config = config or LabellerConfig()
        self.survey_question = ""
        self.client = instructor.from_openai(AsyncOpenAI(api_key=self.config.api_key or None), mode=instructor.Mode.JSON)
        self.batch_size = self.config.batch_size
        self.max_rediscovery_attempts = 3  # Maximum attempts for theme discovery
        self.verbose_reporter = VerboseReporter(verbose)
        
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
                    self.verbose_reporter.stat_line(f"Retry {attempt + 1}/{max_retries} after error: {str(e)}")
        
        raise RuntimeError(f"Failed after {max_retries} attempts. Last error: {str(last_error)}")
    
    async def process_hierarchy_async(self, cluster_models: List[models.ClusterModel], survey_question: str) -> List[models.LabelModel]:
        """Enhanced async processing method with new phases"""
        self.survey_question = survey_question
        
        self.verbose_reporter.section_header("HIERARCHICAL LABELING PROCESS", emoji="🔄")
        
        # Extract micro-clusters
        micro_clusters = self._extract_micro_clusters(cluster_models)
        self.verbose_reporter.stat_line(f"Found {len(micro_clusters)} unique response segments")
        
        # =============================================================================
        # Phase 1: Descriptive Coding
        # =============================================================================
    
        import time
        self.verbose_reporter.step_start("Phase 1: Descriptive Coding", emoji="📝")
        phase1_start = time.time()
        labeled_clusters = await self._phase1_descriptive_coding(micro_clusters)
        
        self.labeled_clusters = labeled_clusters
        
        phase1_time = time.time() - phase1_start
        self.verbose_reporter.step_complete(f"Generated {len(labeled_clusters)} segment labels")
        

        # =============================================================================
        # Phase 2: Theme Discovery 
        # =============================================================================
    
        self.verbose_reporter.step_start("Phase 2: Theme Discovery", emoji="🔍")
        phase2_start = time.time()
        
        codebook = await self._phase2_discovery(labeled_clusters)
            
        phase2_time = time.time() - phase2_start
        self.verbose_reporter.step_complete("Codebook structure created")
            
        self.codebook = codebook
            
        # =============================================================================
        # Phase 3 and 4: Theme Review  
        # =============================================================================
          
        review_attempt = 0
        max_review_attempts = 4
        
        while review_attempt < max_review_attempts:
            self.verbose_reporter.step_start(f"Phase 3: Theme Judger (attempt {review_attempt + 1}/{max_review_attempts})", emoji="⚖️")
            phase3_start = time.time()
            
            judgment = await self._phase3_theme_judger(codebook)
            
            phase3_time = time.time() - phase3_start
            self.verbose_reporter.step_complete("Codebook evaluation completed")
            
            if judgment.is_logical:
                self.verbose_reporter.stat_line("✅ Codebook structure approved!")
                break
            else:
                self.verbose_reporter.stat_line("⚠️ Structure needs improvement")
                if judgment.specific_issues:
                    self.verbose_reporter.stat_line("Issues identified:")
                    for issue in judgment.specific_issues: #[:3]
                        self.verbose_reporter.stat_line(f"- {issue}", bullet="  ")
                
                review_attempt += 1
                
                if review_attempt >= max_review_attempts:
                    self.verbose_reporter.stat_line("⚠️ Maximum review attempts reached, proceeding with current structure")
                    break
                
                # Phase 4: Review and improve structure
                self.verbose_reporter.step_start("Phase 4: Theme Review", emoji="🔄")
                phase4_start = time.time()
                
                codebook = await self._phase4_theme_review(codebook, judgment)
                
                phase4_time = time.time() - phase4_start
                self.verbose_reporter.step_complete("📝 Codebook structure has been revised")
                
                self.codebook = codebook
                
         
        # =============================================================================
        # Phase 5: Label Refinement  
        # =============================================================================
        
        self.verbose_reporter.step_start("Phase 5: Label Refinement", emoji="✨")
        phase5_start = time.time()
        await self._phase5_label_refinement(codebook)
        phase5_time = time.time() - phase5_start
        self.verbose_reporter.step_complete("Labels polished")
        
        # =============================================================================
        # Phase 6: Assignment
        # =============================================================================
        
        self.verbose_reporter.step_start("Phase 6: Assignment", emoji="🎯")
        phase6_start = time.time()
        assignments = await self._phase6_assignment(labeled_clusters, codebook)
        phase6_time = time.time() - phase6_start
        self.verbose_reporter.step_complete("Themes assigned to clusters")
        
        # Remove empty codes  after assignment
        self._remove_empty_codes(codebook)
    
        # Print codebook
        self._display_full_codebook(codebook)
        self.final_labels = self._create_final_labels(labeled_clusters, assignments)
        
        self.verbose_reporter.stat_line("✅ Applying hierarchy to responses...")
        result = self._apply_hierarchy_to_responses(cluster_models, self.final_labels, codebook)
        self._print_assignment_diagnostics(self.final_labels, micro_clusters)
        self.verbose_reporter.stat_line("🎉 Enhanced hierarchical labeling complete!")
        self._print_summary(codebook)

        return result
    
    async def _phase1_descriptive_coding(self, micro_clusters: Dict[int, Dict]) -> List[ClusterLabel]:
        """Phase 1: Descriptive coding """
        from prompts import PHASE1_DESCRIPTIVE_CODING_PROMPT
       
        labeled_clusters = []
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
            reps_text = "\n".join([f"{i+1}. {desc} (similarity: {sim:.3f})" for i, (desc, sim) in enumerate(representatives)])
            
            prompt = PHASE1_DESCRIPTIVE_CODING_PROMPT.format(
                survey_question=self.survey_question,
                cluster_id=cluster_id,
                representatives=reps_text,
                language=self.config.language)
            
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
                        description="Failed to generate label",
                        representatives=representatives))
                else:
                    labeled_clusters.append(ClusterLabel(
                        cluster_id=cluster_id,
                        label=result.label,
                        description=result.description,
                        representatives=representatives))
        
        return labeled_clusters
    
    async def _phase2_discovery(self, labeled_clusters: List[ClusterLabel], improvement_instructions: Optional[str] = None) -> Codebook:
        """Phase 2: Single call for theme discovery with optional improvement instructions"""
        from prompts import PHASE2_DISCOVERY_PROMPT
        
        # Format clusters for prompt
        cluster_summaries = []
        for cluster in sorted(labeled_clusters, key=lambda x: x.cluster_id):
            #summary = f"Cluster {cluster.cluster_id}: {cluster.label} - {cluster.description}\n"
            summary = f"Cluster {cluster.cluster_id}: {cluster.label}\n"
            cluster_summaries.append(summary)
        
        all_cluster_ids = ", ".join(str(c.cluster_id) for c in labeled_clusters)
        
        # Add improvement instructions if this is a retry
        improvement_text = ""
        if improvement_instructions:
            improvement_text = f"\n\nIMPORTANT - Previous attempt feedback:\n{improvement_instructions}\n\nPlease address these issues in your new hierarchy."
        
        prompt = PHASE2_DISCOVERY_PROMPT.format(
            survey_question=self.survey_question,
            language=self.config.language,
            cluster_summaries="\n".join(cluster_summaries),
            all_cluster_ids=all_cluster_ids,
            improvement_instructions=improvement_text
            )
        
        result = await self._invoke_with_retries(prompt, DiscoveryResponse)
        result_dict = result.model_dump()
        
        # Parse result into Codebook
        return self._parse_codebook(result_dict)
    
    async def _phase3_theme_judger(self, codebook: Codebook) -> ThemeJudgmentResponse:
        """Phase 3: Judge the quality of the theme structure"""
        from prompts import PHASE3_THEME_JUDGER_PROMPT
        
        # Format codebook for evaluation
        lines = []
        for theme in codebook.themes:
            lines.append(f"\nTHEME {theme.id}: {theme.label}")
            if theme.description:
                lines.append(f"  Description: {theme.description}")
            
            related_topics = [t for t in codebook.topics if t.parent_id == theme.id]
            for topic in related_topics:
                lines.append(f"  TOPIC {topic.id}: {topic.label}")
                if topic.description:
                    lines.append(f"    Description: {topic.description}")
                
                related_codes  = [s for s in codebook.codes if s.parent_id == topic.id]
                for code in related_codes :
                    lines.append(f"    code {code.id}: {code.label}")
        
        codebook_text = "\n".join(lines)
        
        prompt = PHASE3_THEME_JUDGER_PROMPT.format(
            survey_question=self.survey_question,
            codebook=codebook_text,
            language=self.config.language)
        
        try:
            result = await self._invoke_with_retries(prompt, ThemeJudgmentResponse)
            
            # if result.is_logical:
            #     print("    ✅ Codebook structure is logical and meaningful")
            # else:
            #     print("    ⚠️ Codebook structure needs improvement")
            #     if result.specific_issues:
            #         print("    Issues identified:")
            #         for issue in result.specific_issues[:3]:
            #             print(f"      - {issue}")
            
            return result
            
        except Exception as e:
            print(f"    ❌ Theme judgment failed: {str(e)}, assuming structure is acceptable")
            return ThemeJudgmentResponse(is_logical=True)
    
    async def _phase4_theme_review(self, current_codebook: Codebook, judgment: ThemeJudgmentResponse) -> Codebook:
        """Phase 4: Theme Review - Improve codebook based on feedback"""
        from prompts import PHASE4_THEME_REVIEW_PROMPT
       
        codebook_text = self._format_codebook_for_review(current_codebook)
        issues_text = "\n".join([f"- {issue}" for issue in (judgment.specific_issues or [])])
        
        prompt = PHASE4_THEME_REVIEW_PROMPT.format(
            survey_question=self.survey_question,
            current_codebook=codebook_text,
            improvement_feedback=judgment.feedback,
            specific_issues=issues_text,
            language=self.config.language
            )
        
        result = await self._invoke_with_retries(prompt, ThemeReviewResponse)
        improved_codebook = self._parse_codebook(result.model_dump())
  
        return improved_codebook
    
    async def _phase5_label_refinement(self, codebook: Codebook) -> None:
        """Phase 5: Refine all labels for clarity and consistency"""
        from prompts import PHASE5_LABEL_REFINEMENT_PROMPT
        
        codebook_text = self._format_codebook(codebook)
        
        prompt = PHASE5_LABEL_REFINEMENT_PROMPT.format(
            survey_question=self.survey_question,
            codebook=codebook_text,
            language=self.config.language)
        
        try:
            result = await self._invoke_with_retries(prompt, LabelRefinementResponse)
            
            # Apply refinements
            refinement_count = 0
            
            # Create lookups
            theme_lookup = {t.id: t for t in codebook.themes}
            topic_lookup = {t.id: t for t in codebook.topics}
            code_lookup = {s.id: s for s in codebook.codes}
            
            # Apply theme refinements
            for theme_id, refined_label in result.refined_themes.items():
                if theme_id in theme_lookup and theme_lookup[theme_id].label != refined_label:
                    old_label = theme_lookup[theme_id].label
                    theme_lookup[theme_id].label = refined_label
                    refinement_count += 1
                    print(f"      🏷️  Theme {theme_id}: '{old_label}' → '{refined_label}'")
            
            # Apply topic refinements
            for topic_id, refined_label in result.refined_topics.items():
                if topic_id in topic_lookup and topic_lookup[topic_id].label != refined_label:
                    old_label = topic_lookup[topic_id].label
                    topic_lookup[topic_id].label = refined_label
                    refinement_count += 1
                    print(f"      🏷️  Topic {topic_id}: '{old_label}' → '{refined_label}'")
            
            # Apply code refinements
            for code_id, refined_label in result.refined_codes .items():
                if code_id in code_lookup and code_lookup[code_id].label != refined_label:
                    old_label = code_lookup[code_id].label
                    code_lookup[code_id].label = refined_label
                    refinement_count += 1
                    print(f"      🏷️  code {code_id}: '{old_label}' → '{refined_label}'")
            
            print(f"    ✨ Applied {refinement_count} label refinements")
            
        except Exception as e:
            print(f"    ⚠️ Label refinement failed: {str(e)}, keeping original labels")
    
    async def _phase6_assignment(self, labeled_clusters: List[ClusterLabel], codebook: Codebook) -> List[ThemeAssignment]:
        """Phase 6: Assignment - Assign clusters to the refined hierarchy"""
        assignments = []
        from prompts import PHASE6_ASSIGNMENT_PROMPT
        codebook_text = self._format_codebook(codebook)
        
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
                cluster_description=cluster.description,
                cluster_representatives=representatives,
                codebook=codebook_text,
                language=self.config.language)
            
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
        """Parse JSON result into Codebook structure without cluster assignments"""
        themes = []
        topics = []
        codes = []
        
        # Handle single-call format (nested structure)
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
                    direct_clusters=[]  # Always empty - no assignments yet
                )
                themes.append(theme)
                
                # Extract topics
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
                            direct_clusters=[]  # Always empty - no assignments yet
                        )
                        topics.append(topic)
                        
                        # Extract codes
                        if 'codes' in topic_data:  # Changed from 'codes '
                            for code_idx, code_data in enumerate(topic_data['codes'], 1):
                                code_id = f"{topic_id}.{code_idx}"
                                code_numeric_id = float(f"{topic_id}{code_idx}")
                                
                                code = CodebookEntry(
                                    id=code_id,
                                    numeric_id=code_numeric_id,
                                    level=3,
                                    label=code_data['label'],
                                    description=code_data.get('description', ''),
                                    parent_id=topic_id,
                                    parent_numeric_id=topic_numeric_id,
                                    direct_clusters=[]  # Empty - will be filled in Phase 5
                                )
                                codes.append(code)
                    
        return Codebook(
            survey_question=self.survey_question,
            themes=themes,
            topics=topics,
            codes=codes)
  
    
    def _format_codebook(self, codebook: Codebook) -> str:
        """Format codebook in readable hierarchy"""
        lines = []
        
        for theme in codebook.themes:
            lines.append(f"\nTHEME {theme.id}: {theme.label}")
            related_topics = [t for t in codebook.topics if t.parent_id == theme.id]
            for topic in related_topics:
                lines.append(f"  TOPIC {topic.id}: {topic.label}")
                related_codes = [c for c in codebook.codes if c.parent_id == topic.id]   
                for code in related_codes:  # Changed
                    lines.append(f"    CODE {code.id}: {code.label}")   
                        
        return "\n".join(lines)
    
    def _format_codebook_for_review(self, codebook: Codebook) -> str:
        """Format codebook with full details for Phase 4 review"""
        lines = []
        
        for theme in codebook.themes:
            lines.append(f"\nTHEME {theme.id}: {theme.label}")
            lines.append(f"  Description: {theme.description}")
            
            related_topics = [t for t in codebook.topics if t.parent_id == theme.id]
            for topic in related_topics:
                lines.append(f"\n  TOPIC {topic.id}: {topic.label}")
                lines.append(f"    Description: {topic.description}")
                
                related_codes = [c for c in codebook.codes if c.parent_id == topic.id]
                for code in related_codes:
                    lines.append(f"\n    CODE {code.id}: {code.label}")
                    lines.append(f"      Description: {code.description}")
                    # Optionally add cluster count if available
                    if code.direct_clusters:
                        lines.append(f"      Clusters: {len(code.direct_clusters)} responses")
        
        return "\n".join(lines)
    
    def _update_codebook_with_direct_assignments(self, codebook: Codebook, assignments: List[ThemeAssignment]):
        """Update codebook with direct cluster assignments"""
        # Clear existing assignments
        for code in codebook.codes:  # Changed
            code.direct_clusters = []
        
        # Track assignments
        assigned_clusters = set()
        multi_assignments = 0
        
        for assignment in assignments:
            cluster_id = assignment.cluster_id
            
            # Find and update the assigned code
            for code in codebook.codes:   
                if code.id == assignment.code_id:  
                    code.direct_clusters.append(cluster_id)
                    assigned_clusters.add(cluster_id)
                    break
            
            # Handle alternatives if confidence is low
            if assignment.alternative_assignments and assignment.confidence < 0.8:
                alternatives_used = []
                for alt in assignment.alternative_assignments:
                    if alt.get('confidence', 0) >= self.config.assignment_threshold:
                        for code in codebook.codes:   
                            if code.id == alt['code_id']:   
                                code.direct_clusters.append(cluster_id)
                                alternatives_used.append(alt['code_id'])  
                                break
                
                if alternatives_used:
                    multi_assignments += 1
                    print(f"    📌 Cluster {cluster_id} also assigned to alternatives: {alternatives_used}")
        
        print(f"    ✓ Assigned {len(assigned_clusters)} segments")
        if multi_assignments > 0:
            print(f"    ✓ {multi_assignments} clusters assigned to multiple codes via alternatives")  
    
    def _format_cluster_descriptions(self) -> str:
        """Format cluster descriptions for the review prompt"""
        lines = []
        lines.append("Cluster ID | Label | Description")
        lines.append("-" * 80)
        
        # self.labeled_clusters should exist from Phase 2
        if hasattr(self, 'labeled_clusters') and self.labeled_clusters:
            for cluster in sorted(self.labeled_clusters, key=lambda x: x.cluster_id):
                # Truncate long descriptions to fit nicely
                desc = cluster.description[:60] + "..." if len(cluster.description) > 60 else cluster.description
                lines.append(f"{cluster.cluster_id:3d} | {cluster.label:30s} | {desc}")
        else:
            lines.append("No cluster descriptions available")
        
        return "\n".join(lines)
    
    
    def _get_all_cluster_ids(self, codebook: Codebook) -> set:
        """Get all cluster IDs from a codebook"""
        all_clusters = set()
        for code in codebook.codes:
            all_clusters.update(code.direct_clusters)
        return all_clusters
    
    def _add_missing_clusters_to_other(self, codebook: Codebook, missing_clusters: set):
        """Add missing clusters to an 'other' category"""
        other_code = self._get_or_create_other_code(codebook)
        other_code.direct_clusters.extend(list(missing_clusters))
    
    def _get_or_create_other_code(self, codebook: Codebook) -> CodebookEntry:
        """Get or create an 'other' code for unclassified clusters"""
        # Look for existing "other" theme
        other_theme = None
        for theme in codebook.themes:
            if theme.label.lower() in ["other", "anders", "overig"]:
                other_theme = theme
                break
             
        if not other_theme:
            # Create "other" theme
            other_theme = CodebookEntry(
                id="99",
                numeric_id=99.0,
                level=1,
                label="Other",
                description="Responses that don't fit other categories",
                direct_clusters=[]
            )
            codebook.themes.append(other_theme)
        
        # Look for "other" topic under this theme
        other_topic = None
        for topic in codebook.topics:
            if topic.parent_id == other_theme.id:
                other_topic = topic
                break
        
        if not other_topic:
            # Create "other" topic
            other_topic = CodebookEntry(
                id="99.1",
                numeric_id=99.1,
                level=2,
                label="Miscellaneous",
                description="Various other responses",
                parent_id=other_theme.id,
                parent_numeric_id=other_theme.numeric_id,
                direct_clusters=[]
            )
            codebook.topics.append(other_topic)
        
        # Look for "other" codes
        other_code = None
        for code in codebook.codes:  
            if code.parent_id == other_topic.id:
                other_code = code
                break
        
        if not other_code:
            # Create "other" code
            other_code = CodebookEntry(
                id="99.1.1",
                numeric_id=99.11,
                level=3,
                label="Unclassified",
                description="Responses not fitting other categories",
                parent_id=other_topic.id,
                parent_numeric_id=other_topic.numeric_id,
                direct_clusters=[]
            )
            codebook.codes.append(other_code)  
        
        return other_code
    
    def _remove_empty_codes(self, codebook: Codebook) -> None:
       """Remove codes that have no clusters assigned"""
       # Find codes with clusters
       codes_with_clusters = []
       codes_removed = []
       
       for code in codebook.codes:  # Changed
           if code.direct_clusters:
               codes_with_clusters.append(code)
           else:
               codes_removed.append(code)
       
       # Update the codebook
       codebook.codes = codes_with_clusters  # Changed
       
       # Also clean up empty topics (topics with no codes)
       topics_with_codes = []
       for topic in codebook.topics:
           # Check if any code has this topic as parent
           has_codes = any(c.parent_id == topic.id for c in codebook.codes)  # Changed
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
    
     
    def _create_final_labels(self, labeled_clusters: List[ClusterLabel], 
                            assignments: List[ThemeAssignment]) -> Dict[int, Dict]:
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
                    'description': cluster.description,
                    'theme': ('99', 0.0),
                    'topic': ('99.1', 0.0),
                    'code': ('99.1.1', 0.0)
                }
            else:
                # Use direct assignments
                final_labels[cluster_id] = {
                    'label': cluster.label,
                    'description': cluster.description,
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
        self.verbose_reporter.summary("Codebook Summary", {
            "Themes": len(codebook.themes),
            "Topics": len(codebook.topics),
            "Codes": len(codebook.codes),
            "Clusters assigned": len(set(cluster for code in codebook.codes for cluster in code.direct_clusters))
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
                self.verbose_reporter.stat_line(f"Description: {theme.description}", bullet="   ")
            
            # Find related topics
            related_topics = [t for t in codebook.topics if t.parent_id == theme.id]
            if related_topics:
                for topic in sorted(related_topics, key=lambda x: x.numeric_id):
                    self.verbose_reporter.stat_line(f"📍 TOPIC {topic.id}: {topic.label}", bullet="   ")
                    if topic.description:
                        self.verbose_reporter.stat_line(f"Description: {topic.description}", bullet="      ")
                    
                    # Find related codes
                    related_codes = [c for c in codebook.codes if c.parent_id == topic.id]  # Changed
                    if related_codes:  # Changed
                        for code in sorted(related_codes, key=lambda x: x.numeric_id):  # Changed
                            print(f"\n      🔸 CODE {code.id}: {code.label}")  # Changed
                            if code.description:
                                print(f"         Description: {code.description}")
                            if code.direct_clusters:
                                print(f"         Micro-clusters: {code.direct_clusters}")
        
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
            print("  ⚠️  Some clusters were not assigned in Phase 5")
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
    cluster_results = cache_manager.load_from_cache(filename, "clusters", models.ClusterModel)
    print(f"✅ Loaded {len(cluster_results)} clustered responses from cache")
    
    # for result in cluster_results:
    #     for segment in result.response_segment:
    #         print(segment.segment_description)

    
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
    print("1. Descriptive Coding - Label each micro-cluster")
    print("2. Theme Discovery - Build hierarchical codebook")
    print("3. Theme Reviewer -  Assessment")
    print("4. Theme review - Refine codebook structure")
    print("5. Assignment - Assign clusters to hierarchy")
    print("4. Assignment - Assign clusters to hierarchy")
    print("=" * 50)

    # Initialize thematic labeller with fresh config to avoid caching
    fresh_config = LabellerConfig()
    labeller = ThematicLabeller(
        config=fresh_config)
    
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
    
    print("📊 Final hierarchy:")
    print(f"   - {len(theme_counts)} Themes")
    print(f"   - {len(topic_counts)} Topics") 
    print(f"   - {len(keyword_counts)} Keywords")
    
    print("\n🏆 Top 5 themes by frequency:")
    for theme_id, count in sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"   Theme {theme_id}: {count} segments")
 
    ######################
    
    #phase 1 descriptive codes
    for cluster in labeller.labeled_clusters:
        print(f"Cluster {cluster.cluster_id}: {cluster.label} - {cluster.description}\n")

    #phase 2 codebook
    lines = []
    for theme in codebook.themes:
        lines.append(f"\nTHEME {theme.id}: {theme.label}")
        related_topics = [t for t in codebook.topics if t.parent_id == theme.id]
        for topic in related_topics:
            lines.append(f"  TOPIC {topic.id}: {topic.label}")
            related_codes  = [s for s in codebook.codes if s.parent_id == topic.id]
            for code in related_codes :
                lines.append(f"    code {code.id}: {code.label}")
                #lines.append(f"    code {code.id}: {code.description}")
                
    print("\n".join(lines))
    
    
    # print("\n=== PHASE 2 CODEBOOK ===")
    # print(f"Themes: {len(labeller.codebook_phase2.themes)}")
    # print(f"Topics: {len(labeller.codebook_phase2.topics)}")
    # print(f"codes : {len(labeller.codebook_phase2.codes )}")
    # 
    # print("\n=== PHASE 3 CODEBOOK (after refinement) ===")
    # print(f"Themes: {len(labeller.codebook_phase2.themes)}")
    # print(f"Topics: {len(labeller.codebook_phase2.topics)}")
    # print(f"codes : {len(labeller.codebook_phase2.codes )}")
    
    #phase 2 codebook
    # print("phase 2 codebook:")
    lines = []
    for theme in labeller.codebook_phase2.themes:
        lines.append(f"\nTHEME {theme.id}: {theme.label}")
        related_topics = [t for t in labeller.codebook_phase2.topics if t.parent_id == theme.id]
        for topic in related_topics:
            lines.append(f"  TOPIC {topic.id}: {topic.label}")
            related_codes  = [s for s in labeller.codebook_phase2.codes  if s.parent_id == topic.id]
            for code in related_codes :
                lines.append(f"    code {code.id}: {code.label}")
                #lines.append(f"    code {code.id}: {code.description}")
                
    # print("\n".join(lines))
    # 
    # #phase 3 codebook
    # print("\nphase 3 codebook:")
    lines = []
    for theme in labeller.codebook_phase2.themes:
        lines.append(f"\nTHEME {theme.id}: {theme.label}")
        related_topics = [t for t in labeller.codebook_phase2.topics if t.parent_id == theme.id]
        for topic in related_topics:
            lines.append(f"  TOPIC {topic.id}: {topic.label}")
            related_codes  = [s for s in labeller.codebook_phase2.codes  if s.parent_id == topic.id]
            for code in related_codes :
                lines.append(f"    code {code.id}: {code.label}")
                #lines.append(f"    code {code.id}: {code.description}")
                
    # print("\n".join(lines))
    # 
    # 
    # # See what changed
    # print("\n=== CHANGES ===")
    # Compare theme labels
    phase2_themes = {t.id: t.label for t in labeller.codebook_phase2.themes}
    phase2_themes = {t.id: t.label for t in labeller.codebook_phase2.themes}
    
    for theme_id in phase2_themes:
        if theme_id in phase2_themes and phase2_themes[theme_id] != phase2_themes[theme_id]:
            pass
            # print(f"Theme {theme_id}: '{phase2_themes[theme_id]}' → '{phase2_themes[theme_id]}'")
    
    

    ######################