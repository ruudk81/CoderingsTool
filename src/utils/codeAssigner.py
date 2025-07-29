import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

import asyncio
import nest_asyncio
from typing import Dict, List, Optional, Union
import instructor
from openai import AsyncOpenAI
import tiktoken
from pydantic import BaseModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import OPENAI_API_KEY, DEFAULT_LANGUAGE, ModelConfig, CodeAssignmentConfig, DEFAULT_CODE_ASSIGNMENT_CONFIG, EmbeddingConfig
from prompts import CODE_ASSIGNMENT_PROMPT
import models
from .verboseReporter import VerboseReporter
from utils.embedder import Embedder

async_client = instructor.patch(AsyncOpenAI(api_key=OPENAI_API_KEY))

class CodeAssignmentResponse(BaseModel):
    idea_id: str
    idea: str
    assigned_codes: List[str]
    assigned_themes: List[str]
    assignment_confidence: float
    assignment_rationale: str

class CodeAssigner:
    def __init__(
        self,
        cluster_models: List[models.ClusterModel],
        codebook: List[models.Codebook],
        var_lab: str,
        code_to_theme_mapping: Optional[Dict[str, str]] = None,
        config: Optional[CodeAssignmentConfig] = None,
        verbose: bool = False,
        prompt_printer = None):
        
        self.cluster_models = cluster_models
        self.codebook = codebook
        self.var_lab = var_lab
        self.config = config or DEFAULT_CODE_ASSIGNMENT_CONFIG
        self.client = async_client
        self.language = DEFAULT_LANGUAGE
        self._results: List[models.CodeAssignedModel] = []
        self.verbose_reporter = VerboseReporter(verbose)
        self.model_config = ModelConfig()
        self.prompt_printer = prompt_printer
        self._captured_prompt = False
        
        # Initialize embedder for similarity calculations
        embedding_config = EmbeddingConfig()
        self.embedder = Embedder(config=embedding_config, verbose=False)
        
        # Cache for code embeddings
        self._code_embeddings = None
        
        # Theme mapping for code-to-theme assignments
        self.code_to_theme_mapping = code_to_theme_mapping or {}
        
        # Initialize tokenizer
        try:
            self.encoding = tiktoken.encoding_for_model(self.config.model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")
            print(f"Using cl100k_base encoding as fallback for {self.config.model}")


    def _get_code_embeddings(self):
        """Generate embeddings for all codes in the codebook for similarity matching"""
        if self._code_embeddings is None:
            code_texts = [f"{code.code}: {code.definition}" for code in self.codebook]
            
            # Create temporary models for embedding generation - need EmbeddingsModel
            temp_models = []
            for i, code_text in enumerate(code_texts):
                temp_model = models.EmbeddingsModel(
                    respondent_id=f"code_{i}",
                    response=code_text,
                    response_ideas=[models.EmbeddingsSubmodel(
                        idea_id=f"code_{i}_1",
                        idea=code_text
                    )],
                    idea_count=1
                )
                temp_models.append(temp_model)
            
            # Generate embeddings (synchronous method that uses asyncio internally)
            embedded_codes = self.embedder.get_embeddings_with_tracking(temp_models, "Code embeddings")
            
            # Extract embeddings array
            embeddings = []
            for model in embedded_codes:
                if hasattr(model, 'response_ideas') and model.response_ideas and len(model.response_ideas) > 0:
                    # Use the EmbeddingsSubmodel structure
                    embedding = model.response_ideas[0].idea_embedding
                    if embedding is not None:
                        embeddings.append(embedding)
                    else:
                        # Fallback zero embedding if embedding failed
                        embeddings.append(np.zeros(1536))  # text-embedding-3-large dimension
                else:
                    embeddings.append(np.zeros(1536))
            
            self._code_embeddings = np.array(embeddings)
        
        return self._code_embeddings

    def _find_similar_codes(self, idea_embedding: np.ndarray, top_k: int = 5) -> List[models.Codebook]:
        """Find the top_k most similar codes to an idea based on embedding similarity"""
        if self._code_embeddings is None:
            raise ValueError("Code embeddings not initialized. Call _get_code_embeddings first.")
        
        # Calculate cosine similarity
        similarities = cosine_similarity([idea_embedding], self._code_embeddings)[0]
        
        # Get top_k most similar indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return corresponding codes
        return [self.codebook[i] for i in top_indices]

    def _assign_themes_to_codes(self, assigned_codes: List[str]) -> List[str]:
        """Map assigned codes to their themes using cached mapping"""
        themes = []
        for code in assigned_codes:
            theme = self.code_to_theme_mapping.get(code)
            if theme and theme not in themes:
                themes.append(theme)
        return themes

    def _extract_all_ideas(self) -> List[tuple]:
        """Extract all individual ideas with their embeddings for processing"""
        all_ideas = []
        
        for model in self.cluster_models:
            # ClusterModel has response_ideas with ClusterSubmodel objects that include embeddings
            if hasattr(model, 'response_ideas') and model.response_ideas:
                for idea_submodel in model.response_ideas:
                    if hasattr(idea_submodel, 'idea_embedding') and idea_submodel.idea_embedding is not None:
                        all_ideas.append((
                            model.respondent_id,
                            idea_submodel.idea_id,
                            idea_submodel.idea,
                            idea_submodel.idea_embedding
                        ))
                    else:
                        # If no embedding, skip this idea (shouldn't happen in normal flow)
                        self.verbose_reporter.stat_line(f"Warning: No embedding for idea {idea_submodel.idea_id}")
            else:
                self.verbose_reporter.stat_line(f"Warning: No response_ideas found for respondent {model.respondent_id}")
        
        return all_ideas

    async def _process_idea_assignment(self, idea_data: tuple) -> CodeAssignmentResponse:
        """Process a single idea assignment with candidate codes"""
        respondent_id, idea_id, idea_text, idea_embedding = idea_data
        
        # Find most similar codes (configurable top_k)
        similar_codes = self._find_similar_codes(idea_embedding, top_k=self.config.top_k_similar_codes)
        
        # Format candidate codes for prompt
        candidate_codes_text = "\n".join([
            f"Code: {code.code}\nDefinition: {code.definition}\n" 
            for code in similar_codes
        ])
        
        # Create prompt
        prompt = CODE_ASSIGNMENT_PROMPT.format(
            language=self.language,
            var_lab=self.var_lab,
            idea_id=idea_id,
            idea_text=idea_text,
            candidate_codes=candidate_codes_text
        )
        
        # Capture prompt for debugging if enabled
        if self.prompt_printer and not self._captured_prompt:
            self.prompt_printer.capture_prompt(
                step_name="code_assignment",
                utility_name="CodeAssigner",
                prompt_content=prompt
            )
            self._captured_prompt = True
        
        # Make API call with retries
        for attempt in range(self.config.retries):
            try:
                # Note: The LLM response model doesn't include themes, so we need a temporary model
                from pydantic import BaseModel
                class LLMCodeAssignmentResponse(BaseModel):
                    idea_id: str
                    idea: str
                    assigned_codes: List[str]
                    assignment_confidence: float
                    assignment_rationale: str
                
                llm_response = await self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    seed=self.model_config.seed,
                    response_model=LLMCodeAssignmentResponse
                )
                
                # Add theme assignments based on assigned codes
                assigned_themes = self._assign_themes_to_codes(llm_response.assigned_codes)
                
                return CodeAssignmentResponse(
                    idea_id=llm_response.idea_id,
                    idea=llm_response.idea,
                    assigned_codes=llm_response.assigned_codes,
                    assigned_themes=assigned_themes,
                    assignment_confidence=llm_response.assignment_confidence,
                    assignment_rationale=llm_response.assignment_rationale
                )
                
            except Exception as e:
                if attempt < self.config.retries - 1:
                    await asyncio.sleep(self.config.retry_delay)
                    continue
                else:
                    self.verbose_reporter.stat_line(f"Failed to process idea {idea_id} after {self.config.retries} attempts: {e}")
                    # Return fallback response
                    fallback_code = similar_codes[0].code if similar_codes else "Unknown"
                    fallback_themes = self._assign_themes_to_codes([fallback_code]) if fallback_code != "Unknown" else []
                    return CodeAssignmentResponse(
                        idea_id=idea_id,
                        idea=idea_text,
                        assigned_codes=[fallback_code],
                        assigned_themes=fallback_themes,
                        assignment_confidence=0.1,
                        assignment_rationale="Failed to process - assigned most similar code as fallback"
                    )

    async def _process_batch(self, batch: List[tuple], batch_index: int = 0) -> List[CodeAssignmentResponse]:
        """Process a batch of ideas concurrently (following qualityFilter/ideaExtractor pattern)"""
        # Create tasks for all ideas in this batch - no semaphore limits like other processors
        tasks = [self._process_idea_assignment(idea_data) for idea_data in batch]
        
        # Process all ideas in batch concurrently (no limits)
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle results and exceptions
        batch_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create fallback result for failed idea
                respondent_id, idea_id, idea_text, idea_embedding = batch[i]
                self.verbose_reporter.stat_line(f"Failed to process idea {idea_id}: {str(result)}")
                batch_results.append(CodeAssignmentResponse(
                    idea_id=idea_id,
                    idea=idea_text,
                    assigned_codes=["Processing_Error"],
                    assigned_themes=[],
                    assignment_confidence=0.0,
                    assignment_rationale="Processing failed - assigned error code"
                ))
            else:
                batch_results.append(result)
        
        return batch_results

    def _create_batches(self, all_ideas: List[tuple]) -> List[List[tuple]]:
        """Create batches for processing"""
        batches = []
        for i in range(0, len(all_ideas), self.config.batch_size):
            batch = all_ideas[i:i + self.config.batch_size]
            batches.append(batch)
        return batches

    def _merge_results_into_models(self, assignment_results: List[CodeAssignmentResponse]) -> List[models.CodeAssignedModel]:
        """Merge assignment results back into model structure"""
        # Create lookup for assignments by idea_id
        assignments_lookup = {result.idea_id: result for result in assignment_results}
        
        coded_models = []
        
        for original_model in self.cluster_models:
            # Convert to CodeAssignedModel
            coded_model = original_model.to_model(models.CodeAssignedModel)
            
            # Update response_ideas with assignments
            if coded_model.response_ideas:
                updated_ideas = []
                for idea_submodel in coded_model.response_ideas:
                    # Convert to AssignedIdeaSubmodel (extends ClusterSubmodel)
                    assigned_idea = models.AssignedIdeaSubmodel(
                        idea_id=idea_submodel.idea_id,
                        idea=idea_submodel.idea,
                        # Copy cluster information if available
                        initial_cluster=getattr(idea_submodel, 'initial_cluster', None),
                        idea_embedding=getattr(idea_submodel, 'idea_embedding', None)
                    )
                    
                    # Add assignment data if available
                    if idea_submodel.idea_id in assignments_lookup:
                        assignment = assignments_lookup[idea_submodel.idea_id]
                        assigned_idea.assigned_codes = assignment.assigned_codes
                        assigned_idea.assigned_themes = assignment.assigned_themes
                        assigned_idea.assignment_confidence = assignment.assignment_confidence
                        assigned_idea.assignment_rationale = assignment.assignment_rationale
                    else:
                        # Fallback if no assignment found
                        assigned_idea.assigned_codes = ["Unassigned"]
                        assigned_idea.assigned_themes = []
                        assigned_idea.assignment_confidence = 0.0
                        assigned_idea.assignment_rationale = "No assignment found"
                    
                    updated_ideas.append(assigned_idea)
                
                coded_model.response_ideas = updated_ideas
            
            coded_models.append(coded_model)
        
        return coded_models

    async def assign_codes(self) -> List[models.CodeAssignedModel]:
        """Main method to assign codes to all ideas"""
        self.verbose_reporter.section_header("CODE ASSIGNMENT PROCESSING")
        
        # Initialize code embeddings
        self.verbose_reporter.stat_line("Generating code embeddings for similarity matching...")
        self._get_code_embeddings()
        
        # Extract all ideas
        all_ideas = self._extract_all_ideas()
        total_ideas = len(all_ideas)
        
        if total_ideas == 0:
            self.verbose_reporter.stat_line("No ideas found for code assignment")
            return []
        
        self.verbose_reporter.stat_line(f"Processing {total_ideas} ideas with {len(self.codebook)} available codes")
        
        # Create batches
        batches = self._create_batches(all_ideas)
        total_batches = len(batches)
        
        # Process all batches concurrently (following qualityFilter/ideaExtractor pattern)
        self.verbose_reporter.stat_line(f"Processing {total_ideas} ideas in {total_batches} batches concurrently...")
        
        batch_tasks = [self._process_batch(batch, i) for i, batch in enumerate(batches)]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Collect results from all batches
        all_results = []
        total_failures = 0
        for i, batch_result in enumerate(batch_results):
            if isinstance(batch_result, Exception):
                self.verbose_reporter.stat_line(f"Batch {i+1} processing failed: {str(batch_result)}")
                total_failures += 1
                continue
            
            # Add all results from this batch
            all_results.extend(batch_result)
        
        if total_failures > 0:
            self.verbose_reporter.stat_line(f"{total_failures} out of {total_batches} batches failed completely")
        
        # Merge results back into model structure
        self._results = self._merge_results_into_models(all_results)
        
        # Report summary
        self.verbose_reporter.summary("CODE ASSIGNMENT COMPLETED", {
            "Total ideas processed": len(all_results),
            "Average confidence": f"{np.mean([r.assignment_confidence for r in all_results]):.2f}" if all_results else "N/A"
        })
        
        return self._results

    def assign(self) -> List[models.CodeAssignedModel]:
        """Synchronous wrapper for assign_codes"""
        if nest_asyncio:
            nest_asyncio.apply()
        
        return asyncio.run(self.assign_codes())