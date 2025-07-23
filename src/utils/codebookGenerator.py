import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

import asyncio
import time
import os
import numpy as np
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging
import tiktoken

from openai import AsyncOpenAI
from sklearn.metrics.pairwise import cosine_similarity

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

import models
from prompts import SYSTEM_MESSAGE_CODEBOOK, INITIAL_CODEBOOK_GENERATION, REVIEW_CODEBOOK_GENERATION
from config import ModelConfig, EmbeddingConfig, DEFAULT_LANGUAGE, OPENAI_API_KEY
from utils.verboseReporter import VerboseReporter
from utils.embedder import Embedder

try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CodeRecommendation(BaseModel):
    """Model for LangChain code generation recommendation"""
    needs_new_code: bool = Field(description="Whether a new code is needed")
    code: Optional[str] = Field(default=None, description="The recommended code")
    definition: Optional[str] = Field(default=None, description="Code definition")
    reasoning: str = Field(description="Reasoning for the recommendation")
    evaluation: Dict[str, str] = Field(default_factory=dict, description="Evaluation criteria results")


class InductiveCodebookGenerator:
    """Main class for inductive codebook generation using LangChain"""
    
    def __init__(self, 
                 cluster_results: List[models.ClusterModel], 
                 embedded_text: List[models.EmbeddingsModel],
                 starter_codes: List[Dict[str, str]], 
                 var_lab: str, 
                 k: int = 5,
                 verbose: bool = False, 
                 prompt_printer = None):
        
        self.language = DEFAULT_LANGUAGE
        self.cluster_results = cluster_results
        self.embedded_text = embedded_text  
        self.codebook = starter_codes.copy()  # Growing codebook
        self.var_lab = var_lab
        self.k = k
        self.verbose = verbose
        self.verbose_reporter = VerboseReporter(verbose)
        self.prompt_printer = prompt_printer
        self.model_config = ModelConfig()
        self.embedder = Embedder(config=EmbeddingConfig(), verbose=False)
        
        # Initialize LangChain components
        self.llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model=self.model_config.get_model_for_phase("phase1_descriptive"),
            temperature=0.3
        )
        
        # Track statistics
        self.stats = {
            'total_clusters': 0,
            'new_codes_added': 0,
            'no_new_codes_needed': 0,
            'errors': 0
        }
        
    def _prepare_clusters(self) -> Dict[int, Dict]:
        """Prepare cluster data with embeddings"""
        # Create mapping from idea_id to embedding
        embedding_map = {}
        for result in self.embedded_text:
            if hasattr(result, 'idea_embeddings') and result.idea_embeddings:
                for idea in result.idea_embeddings:
                    embedding_map[idea.idea_id] = {
                        'idea': idea.idea,
                        'embedding': idea.idea_embedding
                    }
        
        # Group ideas by cluster with their embeddings
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
                        if self.verbose:
                            self.verbose_reporter.stat_line(f"Warning: No embedding found for idea_id {idea.idea_id}")
        
        # Remove empty clusters
        clusters = {cid: cdata for cid, cdata in clusters.items() if len(cdata['embeddings']) > 0}
        
        self.stats['total_clusters'] = len(clusters)
        self.verbose_reporter.stat_line(f"Prepared {len(clusters)} clusters with {total_ideas} total ideas")
        if missing_embeddings > 0:
            self.verbose_reporter.stat_line(f"Missing embeddings for {missing_embeddings} ideas")
            
        return clusters
        
    async def _embed_codebook_texts(self, code_texts: List[str]) -> List[np.ndarray]:
        """Embed codebook texts for similarity matching"""
        try:
            embedding_config = EmbeddingConfig()
            client = AsyncOpenAI(api_key=OPENAI_API_KEY)
            
            response = await client.embeddings.create(
                model=embedding_config.embedding_model, 
                input=code_texts
            )
            
            embeddings = []
            for embedding_data in response.data:
                embeddings.append(np.array(embedding_data.embedding, dtype=np.float32))
                
            return embeddings
            
        except Exception as e:
            self.verbose_reporter.stat_line(f"Error embedding codebook: {str(e)}")
            return []
    
    def _find_k_nearest_codes(self, cluster_embedding: np.ndarray, codebook_embeddings: List[np.ndarray]) -> List[Dict]:
        """Find k nearest codes for a cluster embedding"""
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
        
    async def _process_cluster_batch(self, 
                                   cluster_batch: List[tuple], 
                                   codebook_embeddings: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Process a batch of clusters concurrently"""
        tasks = []
        
        for cluster_id, cluster_data in cluster_batch:
            # Calculate mean embedding for cluster
            cluster_embedding = np.mean(cluster_data['embeddings'], axis=0)
            
            # Find nearest codes
            nearest_codes = self._find_k_nearest_codes(cluster_embedding, codebook_embeddings)
            
            # Prepare texts for prompt
            cluster_text = "\n".join([f"- {idea}" for idea in cluster_data['ideas']])
            code_text = "\n".join([f"- {code['code']}: {code['definition']}" for code in nearest_codes]) if nearest_codes else "No existing codes"
            
            # Create task for async processing
            task = self._process_single_cluster(cluster_id, cluster_text, code_text)
            tasks.append(task)
        
        # Process batch concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Pair results with cluster IDs
        batch_results = []
        for i, (cluster_id, _) in enumerate(cluster_batch):
            if isinstance(results[i], Exception):
                logger.error(f"Error processing cluster {cluster_id}: {results[i]}")
                batch_results.append({
                    'cluster_id': cluster_id,
                    'error': str(results[i]),
                    'decision': None
                })
                self.stats['errors'] += 1
            else:
                batch_results.append({
                    'cluster_id': cluster_id,
                    'decision': results[i]
                })
        
        return batch_results
        
    async def _process_single_cluster(self, cluster_id: int, cluster_text: str, code_text: str) -> Optional[CodeRecommendation]:
        """Process a single cluster through the LangChain pipeline"""
        try:
            # System prompt
            system_message = SYSTEM_MESSAGE_CODEBOOK.format(language=self.language)
            
            # Stage 1: Initial code generation
            initial_prompt = INITIAL_CODEBOOK_GENERATION.format(
                code_text=code_text,
                survey_question=self.var_lab,
                cluster_text=cluster_text,
                language=self.language
            )
            
            # Capture prompt for first cluster
            if self.prompt_printer and cluster_id == 0:
                self.prompt_printer.capture_prompt(
                    step_name="gatos_codebook",
                    utility_name="InductiveCodebookGenerator",
                    prompt_content=system_message + "\n\n" + initial_prompt,
                    prompt_type="LangChain Codebook Generation - Initial"
                )
            
            # Get initial response
            initial_response = await self.llm.ainvoke(system_message + "\n\n" + initial_prompt)
            initial_text = initial_response.content if hasattr(initial_response, 'content') else str(initial_response)
            
            # Parse initial response
            lines = initial_text.strip().split('\n')
            suggested_code = None
            suggested_definition = None
            
            for line in lines:
                if line.strip().startswith("Code:"):
                    suggested_code = line.replace("Code:", "").strip()
                elif line.strip().startswith("Definition:"):
                    suggested_definition = line.replace("Definition:", "").strip()
            
            # Stage 2: Review and final decision
            if suggested_code and suggested_definition:
                review_code_text = f"Suggested new code:\n- {suggested_code}: {suggested_definition}\n\nExisting codes:\n{code_text}"
            else:
                review_code_text = f"No new code suggested.\n\nExisting codes:\n{code_text}"
            
            review_prompt = REVIEW_CODEBOOK_GENERATION.format(
                code_text=review_code_text,
                survey_question=self.var_lab,
                cluster_text=cluster_text,
                language=self.language
            )
            
            # Add instruction for JSON output
            review_prompt += "\n\nReturn your final recommendation in the following JSON format:\n"
            review_prompt += "{\n"
            review_prompt += '  "needs_new_code": true/false,\n'
            review_prompt += '  "code": "The code name (if needs_new_code is true)",\n'
            review_prompt += '  "definition": "The code definition (if needs_new_code is true)",\n'
            review_prompt += '  "reasoning": "Your reasoning for this decision",\n'
            review_prompt += '  "evaluation": {\n'
            review_prompt += '    "parsimony": "How this meets the parsimony criterion",\n'
            review_prompt += '    "abstraction_level": "How this meets the abstraction level criterion",\n'
            review_prompt += '    "non_redundancy": "How this meets the non-redundancy criterion"\n'
            review_prompt += '  }\n'
            review_prompt += "}"
            
            # Get final decision
            final_response = await self.llm.ainvoke(system_message + "\n\n" + review_prompt)
            final_text = final_response.content if hasattr(final_response, 'content') else str(final_response)
            
            # Parse as JSON
            try:
                output_parser = JsonOutputParser(pydantic_object=CodeRecommendation)
                result = output_parser.parse(final_text)
            except Exception as parse_error:
                # Fallback: try to extract JSON from the response
                import json
                import re
                json_match = re.search(r'\{.*\}', final_text, re.DOTALL)
                if json_match:
                    try:
                        json_data = json.loads(json_match.group())
                        result = CodeRecommendation(**json_data)
                    except:
                        logger.error(f"Failed to parse JSON for cluster {cluster_id}: {parse_error}")
                        # Return a default "no new code" response
                        result = CodeRecommendation(
                            needs_new_code=False,
                            reasoning="Failed to parse response",
                            evaluation={}
                        )
                else:
                    result = CodeRecommendation(
                        needs_new_code=False,
                        reasoning="Failed to parse response",
                        evaluation={}
                    )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in LangChain processing for cluster {cluster_id}: {str(e)}")
            return None
    
    def generate(self) -> Dict[str, Any]:
        """Generate inductive codebook by processing clusters iteratively"""
        self.verbose_reporter.section_header("INDUCTIVE CODEBOOK GENERATION (LANGCHAIN)")
        start_time = time.time()
        
        # Prepare clusters
        clusters = self._prepare_clusters()
        
        if not clusters:
            self.verbose_reporter.stat_line("No valid clusters found with embeddings")
            return {
                'codebook': self.codebook,
                'cluster_assignments': {},
                'stats': self.stats
            }
        
        cluster_ids = sorted(clusters.keys())
        self.verbose_reporter.stat_line(f"Processing {len(cluster_ids)} clusters with batch processing")
        
        # Track cluster assignments
        cluster_to_code = {}
        
        # Process in batches for efficiency
        batch_size = 5  # Process 5 clusters concurrently
        
        for i in range(0, len(cluster_ids), batch_size):
            batch_cluster_ids = cluster_ids[i:i + batch_size]
            batch_data = [(cid, clusters[cid]) for cid in batch_cluster_ids]
            
            if self.verbose and i > 0:
                self.verbose_reporter.stat_line(
                    f"Progress: {i}/{len(cluster_ids)} clusters processed "
                    f"({self.stats['new_codes_added']} new codes added)"
                )
            
            # Get current codebook embeddings
            code_texts = [f"{code['code']}: {code['definition']}" for code in self.codebook]
            codebook_embeddings = asyncio.run(self._embed_codebook_texts(code_texts))
            
            if not codebook_embeddings:
                self.verbose_reporter.stat_line(f"Warning: Failed to embed codebook for batch starting at cluster {batch_cluster_ids[0]}")
                continue
                
            # Process batch
            batch_results = asyncio.run(self._process_cluster_batch(batch_data, codebook_embeddings))
            
            # Update codebook based on results
            for result in batch_results:
                cluster_id = result['cluster_id']
                decision = result.get('decision')
                
                if decision and decision.needs_new_code and decision.code and decision.definition:
                    # Add new code to codebook
                    new_code = {
                        'code': decision.code,
                        'definition': decision.definition,
                        'cluster_origin': str(cluster_id)
                    }
                    self.codebook.append(new_code)
                    cluster_to_code[cluster_id] = decision.code
                    self.stats['new_codes_added'] += 1
                    
                    if self.verbose:
                        print(f"  New code added for cluster {cluster_id}: '{decision.code}'")
                        
                elif decision:
                    # No new code needed
                    cluster_to_code[cluster_id] = "existing_code"
                    self.stats['no_new_codes_needed'] += 1
                else:
                    # Error processing
                    cluster_to_code[cluster_id] = "processing_error"
        
        elapsed_time = time.time() - start_time
        
        # Final reporting
        self.verbose_reporter.summary("CODEBOOK GENERATION COMPLETE", {
            "Initial codes": len(self.codebook) - self.stats['new_codes_added'],
            "New codes added": self.stats['new_codes_added'],
            "Final codebook size": len(self.codebook),
            "Clusters processed": len(cluster_ids),
            "Processing errors": self.stats['errors'],
            "Time elapsed": f"{elapsed_time:.2f}s"
        })
        
        return {
            'codebook': self.codebook,
            'cluster_assignments': cluster_to_code,
            'stats': {
                'initial_codes': len(self.codebook) - self.stats['new_codes_added'],
                'new_codes': self.stats['new_codes_added'],
                'total_codes': len(self.codebook),
                'clusters_processed': len(cluster_ids),
                'no_new_codes_needed': self.stats['no_new_codes_needed'],
                'errors': self.stats['errors']
            }
        }