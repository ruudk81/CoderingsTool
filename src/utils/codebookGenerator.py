import os
import sys
sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass
import logging
import numpy as np

from openai import AsyncOpenAI
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from prompts import SYSTEM_MESSAGE_CODEBOOK, INITIAL_CODEBOOK_GENERATION, REVIEW_CODEBOOK_GENERATION
from config import EmbeddingConfig, DEFAULT_LANGUAGE, OPENAI_API_KEY, ModelConfig
import models
from utils.verboseReporter import VerboseReporter

try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import logging
logging.getLogger("httpx").disabled = True


@dataclass
class ClusterBatch:
    """Represents a batch of clusters to process together"""
    batch_id: int
    cluster_ids: List[int]
    cluster_data: Dict[int, Dict]  # cluster_id -> {'ideas': [], 'embeddings': []}


class CodebookDataProcessor:
    """Handles data preparation for codebook generation"""
    
    def __init__(self, 
                 cluster_results: List[models.ClusterModel], 
                 embedded_text: List[models.EmbeddingsModel],
                 starter_codes: List[Dict[str, str]], 
                 var_lab: str, 
                 k: int = 5):
        
        self.language = DEFAULT_LANGUAGE
        self.cluster_results = cluster_results
        self.embedded_text = embedded_text  
        self.codebook = starter_codes.copy()  # Growing codebook
        self.var_lab = var_lab
        self.k = k
        
    def prepare_cluster_text(self) -> Dict[int, Dict]:
        """Prepare cluster data with ideas and embeddings"""
        # Create embedding map from embedded_text
        embedding_map = {}
        for result in self.embedded_text:
            if hasattr(result, 'idea_embeddings') and result.idea_embeddings:
                for idea in result.idea_embeddings:
                    embedding_map[idea.idea_id] = {
                        'idea': idea.idea,
                        'embedding': idea.idea_embedding
                    }
        
        # Group ideas by cluster
        clusters = {}
        total_ideas = 0
        missing_embeddings = 0
        
        for result in self.cluster_results:
            ideas_list = result.response_ideas or []
            
            for idea in ideas_list:
                if idea.initial_cluster is not None and idea.initial_cluster != -1:
                    cluster_id = idea.initial_cluster
                    
                    if idea.idea_id in embedding_map:
                        embedding_data = embedding_map[idea.idea_id]
                        
                        if cluster_id not in clusters: 
                            clusters[cluster_id] = {'ideas': [], 'embeddings': []}
                        
                        clusters[cluster_id]['ideas'].append(embedding_data['idea'])
                        clusters[cluster_id]['embeddings'].append(embedding_data['embedding'])
                        total_ideas += 1
                    else:
                        missing_embeddings += 1
        
        # Filter out empty clusters
        clusters = {cid: cdata for cid, cdata in clusters.items() if len(cdata['embeddings']) > 0}
        
        if missing_embeddings > 0:
            logger.warning(f"Missing embeddings for {missing_embeddings} ideas")
        
        return clusters
            
    async def embed_codebook_texts(self, code_texts: List[str]) -> List[np.ndarray]:
        """Embed codebook texts for similarity comparison"""
        try:
            embedding_config = EmbeddingConfig()
            client = AsyncOpenAI(api_key=os.environ.get(OPENAI_API_KEY))
            
            response = await client.embeddings.create(
                model=embedding_config.embedding_model, 
                input=code_texts
            )
            
            embeddings = []
            for embedding_data in response.data:
                embeddings.append(np.array(embedding_data.embedding, dtype=np.float32))
                
            return embeddings
            
        except Exception as e:
            logger.error(f"Error embedding codebook: {str(e)}")
            return []
    
    def find_k_nearest_codes(self, cluster_embedding: np.ndarray, codebook_embeddings: List[np.ndarray]) -> List[Dict]:
        """Find k nearest codes to a cluster embedding"""
        if not codebook_embeddings:
            return []
            
        codebook_array = np.array(codebook_embeddings)
        similarities = cosine_similarity(cluster_embedding.reshape(1, -1), codebook_array)[0]
        top_k_indices = np.argsort(similarities)[-self.k:][::-1]
        
        seen = set()
        nearest_codes = []
        
        for idx in top_k_indices:
            if idx < len(self.codebook):
                code = self.codebook[idx]
                code_text = code.get('code', '')
                
                if code_text not in seen:
                    seen.add(code_text)
                    nearest_codes.append(code)
                    
                    if len(nearest_codes) >= self.k:
                        break
                
        return nearest_codes


class InductiveCodebookGenerator:
    """Main class for inductive codebook generation using iterative GATOS methodology"""
    
    def __init__(
        self,
        cluster_results: List[models.ClusterModel], 
        embedded_text: List[models.EmbeddingsModel],
        starter_codes: List[Dict[str, str]], 
        var_lab: str, 
        k: int = 5,
        verbose: bool = False, 
        prompt_printer = None,
        batch_size: int = 5,  # Process clusters in batches
        max_concurrent_requests: int = 3  # Limit concurrent API calls
    ):
        self.cluster_results = cluster_results
        self.embedded_text = embedded_text
        self.starter_codes = starter_codes
        self.var_lab = var_lab
        self.k = k
        self.verbose = verbose
        self.prompt_printer = prompt_printer
        self.batch_size = batch_size
        self.max_concurrent_requests = max_concurrent_requests
        
        # Initialize data processor
        self.data_processor = CodebookDataProcessor(
            cluster_results=cluster_results,
            embedded_text=embedded_text, 
            starter_codes=starter_codes,
            var_lab=var_lab,
            k=k
        )
        
        # Initialize model config
        self.model_config = ModelConfig()
        
        # Track statistics
        self.stats = {
            'total_clusters': 0,
            'new_codes_added': 0,
            'no_new_codes_needed': 0,
            'errors': 0
        }
        
    def create_cluster_batches(self, clusters: Dict[int, Dict]) -> List[ClusterBatch]:
        """Create batches of clusters for efficient processing"""
        cluster_ids = sorted(clusters.keys())
        batches = []
        
        for i in range(0, len(cluster_ids), self.batch_size):
            batch_cluster_ids = cluster_ids[i:i + self.batch_size]
            batch_cluster_data = {cid: clusters[cid] for cid in batch_cluster_ids}
            
            batches.append(ClusterBatch(
                batch_id=i // self.batch_size,
                cluster_ids=batch_cluster_ids,
                cluster_data=batch_cluster_data
            ))
        
        return batches
        
    async def _call_llm_for_code_generation(self, code_text: str, cluster_text: str) -> Dict[str, Any]:
        """Call LLM to determine if new code is needed"""
        # Initialize LLM
        llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=self.model_config.get_model_for_phase("phase1_descriptive"),
            temperature=0.0
        )
        
        # Create prompts
        system_message = SYSTEM_MESSAGE_CODEBOOK.format(language=DEFAULT_LANGUAGE)
        
        # Stage 1: Initial code generation
        initial_prompt_text = INITIAL_CODEBOOK_GENERATION.format(
            language=DEFAULT_LANGUAGE,
            survey_question=self.var_lab,
            code_text=code_text,
            cluster_text=cluster_text,
            **{"data type": "survey responses"}
        )
        
        initial_prompt = PromptTemplate.from_template(system_message + "\n\n" + initial_prompt_text)
        
        # Get initial suggestion
        initial_chain = initial_prompt | llm | StrOutputParser()
        initial_response = await initial_chain.ainvoke({})
        
        # Parse initial response
        code = None
        definition = None
        lines = initial_response.strip().split('\n')
        
        for line in lines:
            if line.strip().startswith("Code:"):
                code = line.replace("Code:", "").strip()
            elif line.strip().startswith("Definition:"):
                definition = line.replace("Definition:", "").strip()
        
        # Stage 2: Review the suggestion
        if code and definition:
            review_code_text = f"Suggested new code:\n- {code}: {definition}\n\nExisting codes:\n{code_text}"
        else:
            review_code_text = f"No new code suggested.\n\nExisting codes:\n{code_text}"
        
        review_prompt_text = REVIEW_CODEBOOK_GENERATION.format(
            language=DEFAULT_LANGUAGE,
            survey_question=self.var_lab,
            code_text=review_code_text,
            cluster_text=cluster_text
        )
        
        review_prompt = PromptTemplate.from_template(system_message + "\n\n" + review_prompt_text)
        
        # Get final decision
        review_chain = review_prompt | llm | StrOutputParser()
        final_response = await review_chain.ainvoke({})
        
        # Parse final response
        needs_new_code = False
        final_code = None
        final_definition = None
        
        # Look for the logical recommendation
        if "no new codes needed" in final_response.lower():
            needs_new_code = False
        else:
            # Extract code and definition from final response
            lines = final_response.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith("Code:"):
                    final_code = line.replace("Code:", "").strip()
                    needs_new_code = True
                elif line.strip().startswith("Definition:"):
                    final_definition = line.replace("Definition:", "").strip()
        
        return {
            'needs_new_code': needs_new_code,
            'code': final_code,
            'definition': final_definition,
            'initial_response': initial_response,
            'final_response': final_response
        }
    
    async def process_cluster(self, cluster_id: int, cluster_data: Dict, 
                            codebook_snapshot: List[Dict], semaphore: asyncio.Semaphore) -> Dict[str, Any]:
        """Process a single cluster asynchronously with rate limiting"""
        async with semaphore:  # Limit concurrent requests
            try:
                # Get codebook embeddings for this snapshot
                code_texts = [f"{code['code']}: {code['definition']}" for code in codebook_snapshot]
                codebook_embeddings = await self.data_processor.embed_codebook_texts(code_texts)
                
                if not codebook_embeddings:
                    return {
                        'cluster_id': cluster_id,
                        'status': 'embedding_error',
                        'needs_new_code': False
                    }
                
                # Calculate cluster embedding
                cluster_embedding = np.mean(cluster_data['embeddings'], axis=0)
                
                # Find k nearest codes
                nearest_codes = self.data_processor.find_k_nearest_codes(cluster_embedding, codebook_embeddings)
                
                # Prepare texts for LLM
                cluster_text = "\n".join([f"- {idea}" for idea in cluster_data['ideas']])
                code_text = "\n".join([
                    f"- {code['code']}: {code['definition']}" 
                    for code in nearest_codes
                ]) if nearest_codes else "No existing codes in codebook"
                
                # Capture prompt for first cluster only
                if self.prompt_printer and cluster_id == 0:
                    prompt_content = f"Existing codes:\n{code_text}\n\nCluster text:\n{cluster_text}"
                    self.prompt_printer.capture_prompt(
                        step_name="gatos_codebook", 
                        utility_name="InductiveCodebookGenerator",
                        prompt_content=prompt_content,
                        prompt_type="GATOS Codebook Generation"
                    )
                
                # Call LLM
                result = await self._call_llm_for_code_generation(code_text, cluster_text)
                
                return {
                    'cluster_id': cluster_id,
                    'status': 'success',
                    'needs_new_code': result['needs_new_code'],
                    'code': result.get('code'),
                    'definition': result.get('definition')
                }
                
            except Exception as e:
                logger.error(f"Error processing cluster {cluster_id}: {str(e)}")
                return {
                    'cluster_id': cluster_id,
                    'status': 'error',
                    'needs_new_code': False,
                    'error': str(e)
                }
    
    async def process_batch_async(self, batch: ClusterBatch, current_codebook: List[Dict]) -> List[Dict[str, Any]]:
        """Process a batch of clusters asynchronously"""
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        # Create tasks for all clusters in batch
        tasks = []
        for cluster_id in batch.cluster_ids:
            cluster_data = batch.cluster_data[cluster_id]
            task = self.process_cluster(cluster_id, cluster_data, current_codebook, semaphore)
            tasks.append(task)
        
        # Process all clusters in batch concurrently
        results = await asyncio.gather(*tasks)
        return results
    
    def generate(self) -> Dict[str, Any]:
        """Generate inductive codebook using batch processing with async operations"""
        verbose_reporter = VerboseReporter(self.verbose)
        verbose_reporter.section_header("INDUCTIVE CODEBOOK GENERATION (BATCH + ASYNC)")
        
        # Prepare clusters
        clusters = self.data_processor.prepare_cluster_text()
        
        if not clusters:
            verbose_reporter.stat_line("No valid clusters found with embeddings")
            return {
                'codebook': self.data_processor.codebook,
                'cluster_assignments': {},
                'stats': self.stats
            }
        
        self.stats['total_clusters'] = len(clusters)
        
        # Create batches
        batches = self.create_cluster_batches(clusters)
        verbose_reporter.stat_line(
            f"Processing {len(clusters)} clusters in {len(batches)} batches "
            f"(batch size: {self.batch_size}, max concurrent: {self.max_concurrent_requests})"
        )
        
        # Track cluster assignments
        cluster_to_code = {}
        
        # Process batches iteratively
        for batch_idx, batch in enumerate(batches):
            if self.verbose:
                verbose_reporter.stat_line(
                    f"Processing batch {batch_idx + 1}/{len(batches)} "
                    f"({len(batch.cluster_ids)} clusters)"
                )
            
            # Get current codebook snapshot for this batch
            current_codebook = self.data_processor.codebook.copy()
            
            # Process batch asynchronously
            batch_results = asyncio.run(self.process_batch_async(batch, current_codebook))
            
            # Process results and update codebook
            for result in batch_results:
                cluster_id = result['cluster_id']
                
                if result['status'] == 'success':
                    if result['needs_new_code'] and result.get('code') and result.get('definition'):
                        # Add new code to codebook
                        new_code = {
                            'code': result['code'],
                            'definition': result['definition'],
                            'cluster_origin': str(cluster_id)
                        }
                        self.data_processor.codebook.append(new_code)
                        cluster_to_code[cluster_id] = result['code']
                        self.stats['new_codes_added'] += 1
                        
                        if self.verbose:
                            print(f"  New code added for cluster {cluster_id}: '{result['code']}'")
                    else:
                        cluster_to_code[cluster_id] = "existing_code"
                        self.stats['no_new_codes_needed'] += 1
                else:
                    cluster_to_code[cluster_id] = result['status']
                    self.stats['errors'] += 1
            
            if self.verbose:
                verbose_reporter.stat_line(
                    f"Batch {batch_idx + 1} complete. "
                    f"Total codes: {len(self.data_processor.codebook)} "
                    f"(+{self.stats['new_codes_added']} new)"
                )
        
        # Final summary
        verbose_reporter.summary("CODEBOOK GENERATION COMPLETE", {
            "Initial codes": len(self.starter_codes),
            "New codes added": self.stats['new_codes_added'],
            "Final codebook size": len(self.data_processor.codebook),
            "Clusters processed": len(clusters),
            "Batches processed": len(batches),
            "Processing errors": self.stats['errors']
        })
        
        return {
            'codebook': self.data_processor.codebook,
            'cluster_assignments': cluster_to_code,
            'stats': {
                'initial_codes': len(self.starter_codes),
                'new_codes': self.stats['new_codes_added'],
                'total_codes': len(self.data_processor.codebook),
                'clusters_processed': len(clusters),
                'batches_processed': len(batches),
                'no_new_codes_needed': self.stats['no_new_codes_needed'],
                'errors': self.stats['errors']
            }
        }