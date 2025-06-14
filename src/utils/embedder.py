import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

import os
import numpy as np
import asyncio
from typing import List, Dict, Optional
from openai import AsyncOpenAI
import models
from config import OPENAI_API_KEY, EmbeddingConfig, DEFAULT_EMBEDDING_CONFIG
from .verboseReporter import VerboseReporter, ProcessingStats
from .tfidfEmbedder import TfidfEmbedder

class Embedder:
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
        
        # Initialize TF-IDF embedder if ensemble mode is enabled
        self.tfidf_embedder = None
        if self.config.use_ensemble:
            self.tfidf_embedder = TfidfEmbedder(config=self.config.tfidf, verbose=verbose)
            self.verbose_reporter.stat_line(f"Initialized ensemble mode with weights: OpenAI {self.config.openai_weight}, TF-IDF {self.config.tfidf_weight}")
        
        
        self.verbose_reporter.stat_line(f"Initialized Embedder with {provider} provider and {self.embedding_model} model")
    
    async def generate_description_batches(self, data: List[models.EmbeddingsModel], batch_size: int = None):
        batch_size = batch_size or self.config.batch_size
        all_segments = []
        segment_indices = []
        
        if self.var_lab:
            for resp_idx, resp_item in enumerate(data):
                for seg_idx, segment in enumerate(resp_item.response_segment):
                    text_to_embed = self.var_lab + segment.segment_description
                    all_segments.append(text_to_embed)
                    segment_indices.append((resp_idx, seg_idx))
        else:
            for resp_idx, resp_item in enumerate(data):
                for seg_idx, segment in enumerate(resp_item.response_segment):
                    text_to_embed = segment.segment_description
                    all_segments.append(text_to_embed)
                    segment_indices.append((resp_idx, seg_idx))
        
        
        batches_responses = [all_segments[i:i+batch_size] for i in range(0, len(all_segments), batch_size)]
        batches_indices = [segment_indices[i:i+batch_size] for i in range(0, len(segment_indices), batch_size)] 
        
        return batches_responses, batches_indices
        
    async def generate_code_batches(self, data: List[models.EmbeddingsModel], batch_size: int = None):
        batch_size = batch_size or self.config.batch_size
        all_segments = []
        segment_indices = []
        
        if self.var_lab:
            for resp_idx, resp_item in enumerate(data):
                for seg_idx, segment in enumerate(resp_item.response_segment):
                    text_to_embed = self.var_lab + segment.segment_label.replace("_", " ").title() # Code syntax (capitals + underscore) needs to be converted
                    all_segments.append(text_to_embed)
                    segment_indices.append((resp_idx, seg_idx))
        else:
            for resp_idx, resp_item in enumerate(data):
                for seg_idx, segment in enumerate(resp_item.response_segment):
                    text_to_embed = segment.segment_label.replace("_", " ").title() # Code syntax (capitals + underscore) needs to be converted
                    all_segments.append(text_to_embed)
                    segment_indices.append((resp_idx, seg_idx))
    
        batches_responses = [all_segments[i:i+batch_size] for i in range(0, len(all_segments), batch_size)]
        batches_indices = [segment_indices[i:i+batch_size] for i in range(0, len(segment_indices), batch_size)]
        
        return batches_responses, batches_indices

    async def process_batches(self, responses: List[str]):
        response = await self.client.embeddings.create(input=responses, model=self.embedding_model)
        return [item.embedding for item in response.data]
        
    def get_code_embeddings(self, data: List[models.EmbeddingsModel], var_lab: str = None, max_concurrent: int = None):
        max_concurrent = max_concurrent or self.config.max_concurrent_requests
        if var_lab is not None:
            self.var_lab = var_lab
        
        self.verbose_reporter.step_start("Generating Code Embeddings", emoji="🔤")
        result = asyncio.run(self._get_code_embeddings_async(data, max_concurrent))
        self.verbose_reporter.step_complete("Code embeddings generated")
        return result
      
    def get_description_embeddings(self, data: List[models.EmbeddingsModel], var_lab: str = None, max_concurrent: int = None):
        max_concurrent = max_concurrent or self.config.max_concurrent_requests
        if var_lab is not None:
            self.var_lab = var_lab
        
        self.verbose_reporter.step_start("Generating Description Embeddings", emoji="📝")
        result = asyncio.run(self._get_description_embeddings_async(data, max_concurrent))
        self.verbose_reporter.step_complete("Description embeddings generated")
        return result
        
    async def _get_description_embeddings_async(self, data: List[models.EmbeddingsModel], max_concurrent: int = None):
        max_concurrent = max_concurrent or self.config.max_concurrent_requests
        batches_responses, batches_indices = await self.generate_description_batches(data)
        return await self._process_embeddings(data, batches_responses, batches_indices, max_concurrent, is_description=True)
    
    async def _get_code_embeddings_async(self, data: List[models.EmbeddingsModel], max_concurrent: int = None):
        max_concurrent = max_concurrent or self.config.max_concurrent_requests
        batches_responses, batches_indices = await self.generate_code_batches(data)
        return await self._process_embeddings(data, batches_responses, batches_indices, max_concurrent, is_description=False)
    
    async def _process_embeddings(self, data: List[models.EmbeddingsModel], batches_responses, batches_indices, max_concurrent: int = None, is_description: bool = False):
        max_concurrent = max_concurrent or self.config.max_concurrent_requests
        # Calculate total segments
        total_segments = sum(len(batch) for batch in batches_responses)
        batch_size = len(batches_responses[0]) if batches_responses else 0
        
        self.verbose_reporter.stat_line(f"Total segments to embed: {total_segments}")
        self.verbose_reporter.stat_line(f"Batch size: {batch_size}, Total batches: {len(batches_responses)}")
        self.verbose_reporter.stat_line(f"Max concurrent requests: {max_concurrent}")
        
        # For ensemble mode, we need to collect all embeddings first
        if self.config.use_ensemble and is_description:
            # Collect all texts and indices for batch processing
            all_texts = []
            all_indices = []
            for batch_texts, batch_indices in zip(batches_responses, batches_indices):
                all_texts.extend(batch_texts)
                all_indices.extend(batch_indices)
            
            # Get all OpenAI embeddings
            self.verbose_reporter.stat_line("Generating OpenAI embeddings for ensemble...")
            openai_embeddings = []
            
            semaphore = asyncio.Semaphore(max_concurrent)
            async def process_openai_batch(batch_responses):
                async with semaphore:
                    return await self.process_batches(batch_responses)
            
            tasks = [process_openai_batch(batch) for batch in batches_responses]
            batch_results = await asyncio.gather(*tasks)
            
            # Flatten results
            for batch_embeddings in batch_results:
                for embedding in batch_embeddings:
                    openai_embeddings.append(np.array(embedding, dtype=np.float32))
            
            openai_embeddings = np.array(openai_embeddings)
            
            # Generate ensemble embeddings
            self.verbose_reporter.stat_line("Generating ensemble embeddings...")
            ensemble_embeddings = await self._generate_ensemble_embeddings(all_texts, openai_embeddings)
            
            # Assign embeddings back to data
            for (resp_idx, seg_idx), embedding in zip(all_indices, ensemble_embeddings):
                data[resp_idx].response_segment[seg_idx].description_embedding = embedding
            
            self.verbose_reporter.stat_line(f"Completed {len(all_indices)} ensemble embeddings")
        
        else:
            # Original non-ensemble processing
            semaphore = asyncio.Semaphore(max_concurrent)
            processed_count = 0
            
            async def process_batch(batch_responses, batch_indices, batch_num):
                nonlocal processed_count
                async with semaphore:
                    batch_embeddings = await self.process_batches(batch_responses)
                    
                    for (resp_idx, seg_idx), embedding in zip(batch_indices, batch_embeddings):
                        # save as array
                        embedding_array = np.array(embedding, dtype=np.float32)
                        
                        if is_description:
                            data[resp_idx].response_segment[seg_idx].description_embedding = embedding_array
                        else:
                            data[resp_idx].response_segment[seg_idx].code_embedding = embedding_array
                    
                    processed_count += len(batch_responses)
                    self.verbose_reporter.stat_line(f"Progress: {processed_count}/{total_segments} segments processed ({processed_count/total_segments*100:.1f}%)")

            # Create and run tasks
            tasks = [
                process_batch(batch_responses, batch_indices, i+1) 
                for i, (batch_responses, batch_indices) in enumerate(zip(batches_responses, batches_indices))]
            
            await asyncio.gather(*tasks)
        
        return data
    
    def combine_embeddings(self, code_embeddings: list, description_embeddings: list) -> list:
        self.verbose_reporter.step_start("Combining Embeddings", emoji="🔗")
        
        # Count total segments
        total_segments = sum(len(resp.response_segment) for resp in code_embeddings if resp.response_segment)
        
        self.verbose_reporter.stat_line(f"Responses with embeddings: {len(code_embeddings)}")
        self.verbose_reporter.stat_line(f"Total segments to combine: {total_segments}")
        
        # Create a dictionary to easily look up description embeddings by respondent_id and segment_id
        description_map = {}
        for resp in description_embeddings:
            resp_id = resp.respondent_id
            description_map[resp_id] = {}
            for segment in resp.response_segment:
                seg_id = segment.segment_id
                description_map[resp_id][seg_id] = segment.description_embedding
        
        # Start with the code embeddings as the base
        combined_embeddings = code_embeddings.copy()  # Make a shallow copy to avoid modifying original
        
        # Add description embeddings where they exist
        combined_count = 0
        for resp in combined_embeddings:
            resp_id = resp.respondent_id
            if resp_id in description_map:
                for segment in resp.response_segment:
                    seg_id = segment.segment_id
                    if seg_id in description_map[resp_id]:
                        segment.description_embedding = description_map[resp_id][seg_id]
                        combined_count += 1
        
        self.verbose_reporter.stat_line(f"Successfully combined: {combined_count} segment embeddings")
        
        # Check embedding dimensions
        if combined_embeddings and combined_embeddings[0].response_segment:
            first_seg = combined_embeddings[0].response_segment[0]
            if hasattr(first_seg, 'code_embedding') and first_seg.code_embedding is not None:
                self.verbose_reporter.stat_line(f"Embedding dimensions: {first_seg.code_embedding.shape[0]}")
        
        self.verbose_reporter.step_complete("Embeddings combined")
        return combined_embeddings

    async def _async_embed_word_clusters(self, clusters: Dict[int, List[str]]) -> np.ndarray:
   
        client = self.client 
        
        cluster_texts = []
        for cluster_id, words in clusters.items():
            full_cluster_text = " ".join(words)
            cluster_texts.append(full_cluster_text)
        
        tasks = []
        for cluster_text in cluster_texts:
            task = client.embeddings.create(
                input=cluster_text,
                model=self.embedding_model
            )
            tasks.append(task)
    
        responses = await asyncio.gather(*tasks)
    
        embeddings = [response.data[0].embedding for response in responses]
        return np.array(embeddings)
    
    def embed_word_clusters(self, clusters: Dict[int, List[str]]) -> np.ndarray: 
        return asyncio.run(self._async_embed_word_clusters(clusters))
    
    async def _async_embed_words(self, words: List[str], batch_size: int = None, max_concurrent: int = None) -> Dict[str, np.ndarray]:
        batch_size = batch_size or self.config.batch_size
        max_concurrent = max_concurrent or self.config.max_concurrent_requests
        semaphore = asyncio.Semaphore(max_concurrent)
        result_dict = {}
        
        word_batches = [words[i:i+batch_size] for i in range(0, len(words), batch_size)]
        
        async def process_batch(batch_words):
            async with semaphore:
                self.verbose_reporter.stat_line(f"Processing batch of {len(batch_words)} words")
                response = await self.client.embeddings.create(
                    input=batch_words,
                    model=self.embedding_model
                )
                
                # Map each word to its embedding
                for i, item in enumerate(response.data):
                    word = batch_words[i]
                    embedding = np.array(item.embedding, dtype=np.float32)
                    result_dict[word] = embedding
                
                self.verbose_reporter.stat_line(f"Completed batch of {len(batch_words)} words")
        
        # Create and run tasks
        tasks = [process_batch(batch) for batch in word_batches]
        
        self.verbose_reporter.stat_line(f"Starting to process {len(tasks)} batches with max {max_concurrent} concurrent requests")
        await asyncio.gather(*tasks)
        self.verbose_reporter.stat_line(f"Completed embeddings for {len(words)} words")
        
        ordered_embeddings = [result_dict[word] for word in words]
        
        return ordered_embeddings
    
    def embed_words(self, words: List[str], batch_size: int = None, max_concurrent: int = None) -> Dict[str, np.ndarray]:
       
        return asyncio.run(self._async_embed_words(words, batch_size, max_concurrent))
    
    def prepare_tfidf_model(self, all_descriptions: List[str]) -> None:
        """Fit TF-IDF model on all descriptions (called once before processing)"""
        if not self.config.use_ensemble:
            return
        
        self.verbose_reporter.step_start("Preparing TF-IDF model", emoji="🔧")
        
        # Try to load existing model
        if self.tfidf_embedder.load():
            self.verbose_reporter.stat_line("Loaded existing TF-IDF model from cache")
        else:
            # Fit new model
            self.tfidf_embedder.fit(all_descriptions)
            self.tfidf_embedder.save()
        
        self.verbose_reporter.step_complete("TF-IDF model ready")
    
    async def _generate_ensemble_embeddings(
        self, 
        texts: List[str], 
        openai_embeddings: np.ndarray
    ) -> np.ndarray:
        """Generate ensemble embeddings by combining OpenAI and TF-IDF"""
        
        # Generate TF-IDF embeddings
        tfidf_embeddings = self.tfidf_embedder.transform(texts)
        
        # Report dimensions
        self.verbose_reporter.stat_line(f"OpenAI embedding dims: {openai_embeddings.shape[1]}")
        self.verbose_reporter.stat_line(f"TF-IDF embedding dims: {tfidf_embeddings.shape[1]}")
        
        if self.config.ensemble_combination == "weighted_concat":
            # Concatenate embeddings with weights
            ensemble_embeddings = self._weighted_concatenate(
                openai_embeddings, 
                tfidf_embeddings,
                self.config.openai_weight,
                self.config.tfidf_weight
            )
        else:  # weighted_average
            # Average embeddings (requires same dimensions)
            ensemble_embeddings = self._weighted_average(
                openai_embeddings,
                tfidf_embeddings,
                self.config.openai_weight,
                self.config.tfidf_weight
            )
        
        # No dimension reduction here - let UMAP handle it
        
        self.verbose_reporter.stat_line(f"Final ensemble embedding dims: {ensemble_embeddings.shape[1]}")
        return ensemble_embeddings
    
    def _weighted_concatenate(
        self,
        openai_emb: np.ndarray,
        tfidf_emb: np.ndarray,
        openai_weight: float,
        tfidf_weight: float
    ) -> np.ndarray:
        """Concatenate embeddings with weights applied"""
        # Apply weights
        weighted_openai = openai_emb * openai_weight
        weighted_tfidf = tfidf_emb * tfidf_weight
        
        # Normalize TF-IDF to similar scale as OpenAI
        tfidf_norm = np.linalg.norm(weighted_tfidf, axis=1, keepdims=True)
        openai_norm = np.linalg.norm(weighted_openai, axis=1, keepdims=True)
        
        # Avoid division by zero
        tfidf_norm = np.where(tfidf_norm == 0, 1, tfidf_norm)
        openai_norm = np.where(openai_norm == 0, 1, openai_norm)
        
        # Scale TF-IDF to match OpenAI magnitude
        scale_factor = np.mean(openai_norm) / np.mean(tfidf_norm)
        weighted_tfidf = weighted_tfidf * scale_factor
        
        # Concatenate
        return np.concatenate([weighted_openai, weighted_tfidf], axis=1)
    
    def _weighted_average(
        self,
        openai_emb: np.ndarray,
        tfidf_emb: np.ndarray,
        openai_weight: float,
        tfidf_weight: float
    ) -> np.ndarray:
        """Average embeddings with weights (requires dimension alignment)"""
        # Reduce TF-IDF to match OpenAI dimensions
        if tfidf_emb.shape[1] != openai_emb.shape[1]:
            target_dim = openai_emb.shape[1]
            tfidf_emb = self.tfidf_embedder.reduce_dimensions(tfidf_emb, target_dim)
        
        # Weighted average
        return (openai_emb * openai_weight + tfidf_emb * tfidf_weight) / (openai_weight + tfidf_weight)
    
