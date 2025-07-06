"""
Enhanced embedder that tracks actual respondent_id and segment_id through async operations
instead of relying on array indices.
"""

import os, sys
sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

import os
import numpy as np
import asyncio
from typing import List, Dict, Tuple
from openai import AsyncOpenAI
import models
from config import OPENAI_API_KEY, EmbeddingConfig, DEFAULT_EMBEDDING_CONFIG
from .verboseReporter import VerboseReporter, ProcessingStats
from dataclasses import dataclass


@dataclass
class SegmentIdentifier:
    """Explicit tracking of segment identity through async operations"""
    respondent_id: str
    segment_id: str
    text_to_embed: str
    array_index: int  # For mapping back to results


class EnhancedEmbedder:
    """Embedder with explicit ID tracking through async operations"""
    
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
        self.verbose_reporter = VerboseReporter(verbose)
        self.stats = ProcessingStats()
        
        self.verbose_reporter.stat_line(f"Initialized Enhanced Embedder with explicit ID tracking")
    
    def _generate_segment_identifiers(self, data: List[models.EmbeddingsModel], 
                                    embedding_type: str) -> List[SegmentIdentifier]:
        """Create explicit segment identifiers for tracking"""
        identifiers = []
        
        for resp_item in data:
            if resp_item.response_segment:
                for segment in resp_item.response_segment:
                    if embedding_type == "description":
                        text_to_embed = segment.segment_description
                    else:  # code
                        text_to_embed = segment.segment_label.replace("_", " ").title()
                        if self.var_lab:
                            text_to_embed = self.var_lab + text_to_embed
                    
                    identifier = SegmentIdentifier(
                        respondent_id=str(resp_item.respondent_id),
                        segment_id=str(segment.segment_id),
                        text_to_embed=text_to_embed,
                        array_index=len(identifiers)
                    )
                    identifiers.append(identifier)
        
        return identifiers
    
    async def _generate_question_aware_embeddings(self, response_embeddings: np.ndarray, question: str) -> np.ndarray:
        """Generate question-aware embeddings by combining response and question embeddings"""
        # Generate question embedding
        question_response = await self.client.embeddings.create(
            input=[question], 
            model=self.embedding_model
        )
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

    async def _process_embeddings_with_id_tracking(self, 
                                                  data: List[models.EmbeddingsModel],
                                                  embedding_type: str) -> List[models.EmbeddingsModel]:
        """Process embeddings with explicit ID tracking"""
        
        # Generate segment identifiers
        segment_identifiers = self._generate_segment_identifiers(data, embedding_type)
        
        if not segment_identifiers:
            return data
        
        # Extract texts for embedding
        texts_to_embed = [identifier.text_to_embed for identifier in segment_identifiers]
        
        self.verbose_reporter.stat_line(f"Processing {len(texts_to_embed)} {embedding_type} embeddings with ID tracking")
        
        # Create batches
        batch_size = self.config.batch_size
        batches = []
        batch_identifiers = []
        
        for i in range(0, len(texts_to_embed), batch_size):
            batch_texts = texts_to_embed[i:i+batch_size]
            batch_ids = segment_identifiers[i:i+batch_size]
            batches.append(batch_texts)
            batch_identifiers.append(batch_ids)
        
        # Process batches concurrently
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        async def process_batch_with_tracking(batch_texts: List[str], 
                                            batch_ids: List[SegmentIdentifier]) -> List[Tuple[SegmentIdentifier, np.ndarray]]:
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
        all_embeddings = [None] * len(segment_identifiers)
        all_identifiers = [None] * len(segment_identifiers)
        
        for batch_result in batch_results:
            for identifier, embedding in batch_result:
                # Use array_index to restore exact original order
                original_index = identifier.array_index
                all_embeddings[original_index] = embedding
                all_identifiers[original_index] = identifier
        
        # Apply question-aware processing if enabled and processing descriptions
        if (self.config.use_question_aware and 
            embedding_type == "description" and 
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
        
        # Apply embeddings back to data using ID lookup
        updated_count = 0
        for resp_item in data:
            if resp_item.response_segment:
                for segment in resp_item.response_segment:
                    key = (str(resp_item.respondent_id), str(segment.segment_id))
                    if key in embedding_lookup:
                        if embedding_type == "description":
                            segment.description_embedding = embedding_lookup[key]
                        else:  # code
                            segment.code_embedding = embedding_lookup[key]
                        updated_count += 1
        
        self.verbose_reporter.stat_line(f"Successfully applied {updated_count} {embedding_type} embeddings using ID tracking")
        
        return data
    
    def get_code_embeddings_with_tracking(self, data: List[models.EmbeddingsModel]) -> List[models.EmbeddingsModel]:
        """Generate code embeddings with explicit ID tracking"""
        self.verbose_reporter.step_start("Generating Code Embeddings (ID Tracked)", emoji="🔤")
        result = asyncio.run(self._process_embeddings_with_id_tracking(data, "code"))
        self.verbose_reporter.step_complete("Code embeddings generated with ID tracking")
        return result
    
    def get_description_embeddings_with_tracking(self, data: List[models.EmbeddingsModel], var_lab: str = None) -> List[models.EmbeddingsModel]:
        """Generate description embeddings with explicit ID tracking"""
        if var_lab is not None:
            self.var_lab = var_lab
            
        self.verbose_reporter.step_start("Generating Description Embeddings (ID Tracked)", emoji="📝")
        result = asyncio.run(self._process_embeddings_with_id_tracking(data, "description"))
        self.verbose_reporter.step_complete("Description embeddings generated with ID tracking")
        return result
    
    def get_combined_embeddings_with_tracking(self, data: List[models.EmbeddingsModel], var_lab: str = None) -> List[models.EmbeddingsModel]:
        """Generate both code and description embeddings with ID tracking"""
        if var_lab is not None:
            self.var_lab = var_lab
        
        self.verbose_reporter.step_start("Generating Combined Embeddings (ID Tracked)", emoji="🔗")
        
        # Process code embeddings first
        result = asyncio.run(self._process_embeddings_with_id_tracking(data, "code"))
        
        # Then process description embeddings on the same data
        result = asyncio.run(self._process_embeddings_with_id_tracking(result, "description"))
        
        self.verbose_reporter.step_complete("Combined embeddings generated with ID tracking")
        return result