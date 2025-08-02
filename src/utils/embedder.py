import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import os
import numpy as np
import asyncio
from typing import List, Tuple
from dataclasses import dataclass
from openai import AsyncOpenAI

# === MODELS ========================================================================================================
import models

# === CONFIG ========================================================================================================
from config import OPENAI_API_KEY, EmbeddingConfig, DEFAULT_EMBEDDING_CONFIG

# === UTILS ========================================================================================================
from .verboseReporter import VerboseReporter, ProcessingStats


@dataclass
class ResponseData:
    respondent_id: str
    segment_id: str
    text_to_embed: str
    array_index: int  

class Embedder:
    """Embedder with ID tracking through async operations"""
    
    def __init__(
        self, 
        config: EmbeddingConfig = None,
        provider: str = "openai",
        client: any = None, 
        embedding_model: str = None, 
        var_lab: str = None,
        verbose: bool = False):
        
        self.config = config or DEFAULT_EMBEDDING_CONFIG
        self.client = client or AsyncOpenAI(api_key=os.getenv(OPENAI_API_KEY))
        self.embedding_model = embedding_model or self.config.embedding_model
        self.var_lab = var_lab
        self.verbose = verbose
        self.verbose_reporter = VerboseReporter(verbose, capture_logging=True)
        self.stats = ProcessingStats()
        
        self.verbose_reporter.stat_line("Initialized Embedder")
    
    def _get_ResponseData(self, data: List[models.IdeasExtractedModel]) -> List[ResponseData]:
        """Create segment identifiers for tracking"""
        response_data = []
        
        for respondent_data in data:
            if respondent_data.response_ideas:
                for response_idea in respondent_data.response_ideas:
                    response_item = ResponseData(
                        respondent_id=str(respondent_data.respondent_id),
                        segment_id=str(response_idea.idea_id),
                        text_to_embed=response_idea.idea ,
                        array_index=len(response_data)
                    )
                    response_data.append(response_item)
        
        return response_data
    
    async def _generate_question_aware_embeddings(self, response_embeddings: np.ndarray, question: str) -> np.ndarray:
        """Generate question-aware embeddings by combining response and question embeddings"""
        # Generate question embedding
        question_response = await self.client.embeddings.create(input=[question], model=self.embedding_model)
        question_embedding = np.array(question_response.data[0].embedding, dtype=np.float32)
        
        # Create domain anchor (average of all response embeddings)
        domain_anchor = np.mean(response_embeddings, axis=0)
        
        # Combine embeddings using configured weights
        question_aware_embeddings = []
        for response_emb in response_embeddings:
            combined = (self.config.response_weight * response_emb + 
                       self.config.question_weight * question_embedding + 
                       self.config.domain_anchor_weight * domain_anchor)
            question_aware_embeddings.append(combined)
        
        return np.array(question_aware_embeddings)

    async def _process_embeddings_with_id_tracking(self, data: List[models.EmbeddingsModel]) -> List[models.EmbeddingsModel]:
        """Process embeddings with explicit ID tracking"""
        
        # Generate segment identifiers
        response_data = self._get_ResponseData(data)
        
        if not response_data:
            return data
        
        # Extract texts for embedding
        texts_to_embed = [response_item.text_to_embed for response_item in response_data]
        
        self.verbose_reporter.stat_line(f"Processing {len(texts_to_embed)}  embeddings with ID tracking")
        
        # Create batches
        batch_size = self.config.batch_size
        batches = []
        batch_identifiers = []
        
        for i in range(0, len(texts_to_embed), batch_size):
            batch_texts = texts_to_embed[i:i+batch_size]
            batch_ids = response_data[i:i+batch_size]
            batches.append(batch_texts)
            batch_identifiers.append(batch_ids)
        
        # Process batches concurrently
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        async def process_batch_with_tracking(batch_texts: List[str], batch_ids: List[ResponseData]) -> List[Tuple[ResponseData, np.ndarray]]:
            async with semaphore:
                response = await self.client.embeddings.create(
                    input=batch_texts, 
                    model=self.embedding_model
                )
                
                # Pair embeddings with their identifiers
                results = []
                for identifier, embedding_data in zip(batch_ids, response.data):
                    embedding_array = np.array(embedding_data.embedding, dtype=np.float32)
                    results.append((identifier, embedding_array))
                
                return results
        
        # Process all batches
        tasks = [
            process_batch_with_tracking(batch_texts, batch_ids)
            for batch_texts, batch_ids in zip(batches, batch_identifiers)
        ]
        
        batch_results = await asyncio.gather(*tasks)
        
        # Flatten results while preserving exact original order using array_index
        all_embeddings = [None] * len(response_data)
        all_identifiers = [None] * len(response_data)
        
        for batch_result in batch_results:
            for identifier, embedding in batch_result:
                # Use array_index to restore exact original order
                original_index = identifier.array_index
                all_embeddings[original_index] = embedding
                all_identifiers[original_index] = identifier
        
        # Apply question-aware processing if enabled and processing descriptions
        if (self.config.use_question_aware and 
            self.var_lab):
            
            self.verbose_reporter.stat_line("Applying question-aware embedding transformation...")
            all_embeddings = await self._generate_question_aware_embeddings(
                np.array(all_embeddings), 
                self.var_lab
            )
        
        # Create lookup
        embedding_lookup = {}
        for identifier, embedding in zip(all_identifiers, all_embeddings):
            key = (identifier.respondent_id, identifier.segment_id)
            embedding_lookup[key] = embedding
        
        # Apply embeddings back to data using ID lookup and convert to EmbeddingsModel
        updated_count = 0
        result = []
        
        for respondent_data in data:
            # Create EmbeddingsSubmodel objects with embeddings
            embeddings_submodels = []
            
            if hasattr(respondent_data, 'response_ideas') and respondent_data.response_ideas:
                for response_idea in respondent_data.response_ideas:
                    key = (str(respondent_data.respondent_id), str(response_idea.idea_id))
                    embedding_data = {
                        'idea_id': response_idea.idea_id,
                        'idea': response_idea.idea
                    }
                    if key in embedding_lookup:
                        embedding_data['idea_embedding'] = embedding_lookup[key]
                        updated_count += 1
                    
                    embeddings_submodels.append(models.EmbeddingsSubmodel(**embedding_data))
            
            # Create EmbeddingsModel with proper structure
            embeddings_model = models.EmbeddingsModel(
                respondent_id=respondent_data.respondent_id,
                response=respondent_data.response,
                response_type=getattr(respondent_data, 'response_type', None),
                quality_filter=getattr(respondent_data, 'quality_filter', None),
                quality_filter_code=getattr(respondent_data, 'quality_filter_code', None),
                response_ideas=embeddings_submodels,
                idea_count=len(embeddings_submodels)
            )
            result.append(embeddings_model)
        
        self.verbose_reporter.stat_line(f"Successfully applied {updated_count} embeddings using ID tracking")
        
        return result
   
    def get_embeddings_with_tracking(self, data: List[models.EmbeddingsModel], var_lab: str = None) -> List[models.EmbeddingsModel]:
        """Generate both code and description embeddings with ID tracking"""
        if var_lab is not None:
            self.var_lab = var_lab
 
        result = asyncio.run(self._process_embeddings_with_id_tracking(data)) 
        
        self.verbose_reporter.step_start("Generating Embeddings with IDs Tracked", emoji="🔗")

        return result

