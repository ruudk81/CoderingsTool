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
from .verboseReporter import VerboseReporter, ProcessingStats
from utils.embedder import Embedder

async_client = instructor.patch(AsyncOpenAI(api_key=OPENAI_API_KEY))

class CodeAssignmentResponse(BaseModel):
    idea_id: str
    idea: str
    assigned_codes: List[str]
    assignment_confidence: float
    assignment_rationale: str

class CodeAssigner:
    def __init__(
        self,
        ideas_extracted_models: List[models.IdeasExtractedModel],
        codebook: List[models.Codebook],
        var_lab: str,
        config: Optional[CodeAssignmentConfig] = None,
        verbose: bool = False,
        prompt_printer = None):
        
        self.ideas_extracted_models = ideas_extracted_models
        self.codebook = codebook
        self.var_lab = var_lab
        self.config = config or DEFAULT_CODE_ASSIGNMENT_CONFIG
        self.client = async_client
        self.language = DEFAULT_LANGUAGE
        self._results: List[models.CodeAssignedModel] = []
        self.verbose_reporter = VerboseReporter(verbose)
        self._stats = ProcessingStats()
        self.model_config = ModelConfig()
        self.prompt_printer = prompt_printer
        self._captured_prompt = False
        
        # Initialize embedder for similarity calculations
        embedding_config = EmbeddingConfig()
        self.embedder = Embedder(config=embedding_config, verbose=False)
        
        # Cache for code embeddings
        self._code_embeddings = None
        
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
                    response_ideas=[models.IdeasExtractedSubmodel(
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
                if hasattr(model, 'idea_embeddings') and model.idea_embeddings and len(model.idea_embeddings) > 0:
                    # Use the new structure with idea_embeddings
                    embedding = model.idea_embeddings[0].idea_embedding
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

    def _extract_all_ideas(self) -> List[tuple]:
        """Extract all individual ideas with their embeddings for processing"""
        all_ideas = []
        
        for model in self.ideas_extracted_models:
            # Check if model has idea_embeddings (EmbeddingsModel structure)
            if hasattr(model, 'idea_embeddings') and model.idea_embeddings:
                for idea_submodel in model.idea_embeddings:
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
            # Fallback to response_ideas if idea_embeddings not available
            elif hasattr(model, 'response_ideas') and model.response_ideas:
                for idea_submodel in model.response_ideas:
                    # Check if this is an EmbeddingsSubmodel with embedding
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
            self.prompt_printer.capture_prompt("Code Assignment", prompt, print_now=True)
            self._captured_prompt = True
        
        # Make API call with retries
        for attempt in range(self.config.retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    seed=self.model_config.seed,
                    response_model=CodeAssignmentResponse
                )
                
                self._stats.increment_api_calls()
                return response
                
            except Exception as e:
                if attempt < self.config.retries - 1:
                    await asyncio.sleep(self.config.retry_delay)
                    continue
                else:
                    self.verbose_reporter.stat_line(f"Failed to process idea {idea_id} after {self.config.retries} attempts: {e}")
                    # Return fallback response
                    return CodeAssignmentResponse(
                        idea_id=idea_id,
                        idea=idea_text,
                        assigned_codes=[similar_codes[0].code] if similar_codes else ["Unknown"],
                        assignment_confidence=0.1,
                        assignment_rationale="Failed to process - assigned most similar code as fallback"
                    )

    async def _process_batch(self, batch: List[tuple]) -> List[CodeAssignmentResponse]:
        """Process a batch of ideas concurrently"""
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        async def process_with_semaphore(idea_data):
            async with semaphore:
                return await self._process_idea_assignment(idea_data)
        
        tasks = [process_with_semaphore(idea_data) for idea_data in batch]
        return await asyncio.gather(*tasks)

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
        
        for original_model in self.ideas_extracted_models:
            # Convert to CodeAssignedModel
            coded_model = original_model.to_model(models.CodeAssignedModel)
            
            # Update response_ideas with assignments
            if coded_model.response_ideas:
                updated_ideas = []
                for idea_submodel in coded_model.response_ideas:
                    # Convert to AssignedCodeSubmodel
                    assigned_idea = models.AssignedCodeSubmodel(
                        idea_id=idea_submodel.idea_id,
                        idea=idea_submodel.idea
                    )
                    
                    # Add assignment data if available
                    if idea_submodel.idea_id in assignments_lookup:
                        assignment = assignments_lookup[idea_submodel.idea_id]
                        assigned_idea.assigned_codes = assignment.assigned_codes
                        assigned_idea.assignment_confidence = assignment.assignment_confidence
                        assigned_idea.assignment_rationale = assignment.assignment_rationale
                    else:
                        # Fallback if no assignment found
                        assigned_idea.assigned_codes = ["Unassigned"]
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
        
        # Process all batches
        all_results = []
        for i, batch in enumerate(batches, 1):
            self.verbose_reporter.stat_line(f"Processing batch {i}/{total_batches} ({len(batch)} ideas)")
            batch_results = await self._process_batch(batch)
            all_results.extend(batch_results)
        
        # Merge results back into model structure
        self._results = self._merge_results_into_models(all_results)
        
        # Report summary
        self.verbose_reporter.summary("CODE ASSIGNMENT COMPLETED", {
            "Total ideas processed": len(all_results),
            "API calls made": self._stats.api_calls,
            "Average confidence": f"{np.mean([r.assignment_confidence for r in all_results]):.2f}" if all_results else "N/A"
        })
        
        return self._results

    def assign(self) -> List[models.CodeAssignedModel]:
        """Synchronous wrapper for assign_codes"""
        if nest_asyncio:
            nest_asyncio.apply()
        
        return asyncio.run(self.assign_codes())