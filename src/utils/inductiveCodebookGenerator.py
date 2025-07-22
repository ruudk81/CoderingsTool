import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import asyncio
import time
import numpy as np
from typing import List, Dict, Optional, Any
from sklearn.metrics.pairwise import cosine_similarity
import instructor
from openai import AsyncOpenAI

# === MODELS ========================================================================================================
import models

# === CONFIG ========================================================================================================
from config import ModelConfig, EmbeddingConfig
from utils.verboseReporter import VerboseReporter
from utils.embedder import Embedder
from prompts import INDUCTIVE_CODEBOOK_GENERATION_PROMPT

# === UTILS ========================================================================================================
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

# === CONSTANTS ========================================================================================================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

class InductiveCodebookGenerator:
    """
    Performs iterative inductive codebook generation using GATOS methodology.
    For each cluster, finds k nearest neighbor codes and decides whether to create new codes.
    """
    
    def __init__(self, 
                 cluster_results: List[models.ClusterModel], 
                 starter_codes: List[Dict[str, str]], 
                 var_lab: str, 
                 k: int = 5,
                 verbose: bool = False, 
                 prompt_printer = None):
        """
        Initialize the inductive codebook generator.
        
        Args:
            cluster_results: Results from clustering step (List[ClusterModel])
            starter_codes: Initial speculative codes
            var_lab: Survey question/variable label
            k: Number of nearest neighbors to find (default: 5)
            verbose: Enable verbose output
            prompt_printer: Optional prompt capture utility
        """
        self.cluster_results = cluster_results
        self.codebook = starter_codes.copy()  # Growing codebook
        self.var_lab = var_lab
        self.k = k
        self.verbose = verbose
        self.verbose_reporter = VerboseReporter(verbose)
        self.prompt_printer = prompt_printer
        self.model_config = ModelConfig()
        self.client = instructor.patch(AsyncOpenAI(api_key=OPENAI_API_KEY))
        self.embedder = Embedder(config=EmbeddingConfig(), verbose=False)
        
        # Track statistics
        self.stats = {
            'total_clusters': 0,
            'new_codes_added': 0,
            'no_new_codes_needed': 0
        }
        
    def _prepare_clusters(self) -> Dict[int, Dict]:
        """
        Group ideas by cluster ID with their embeddings from ClusterModel.
        
        Returns:
            Dict mapping cluster_id to {'ideas': List[str], 'embeddings': List[np.ndarray]}
        """
        clusters = {}
        total_ideas = 0
        
        for result in self.cluster_results:
            # Use idea_embeddings which contains the embedding data (same as response_ideas but with embeddings)
            if hasattr(result, 'idea_embeddings') and result.idea_embeddings:
                embeddings_list = result.idea_embeddings
            else:
                # Fallback to response_ideas (should have embeddings from ClusterSubmodel)
                embeddings_list = result.response_ideas or []
                
            for idea in embeddings_list:
                if idea.initial_cluster is not None and idea.initial_cluster != -1:
                    cluster_id = idea.initial_cluster
                    
                    if cluster_id not in clusters:
                        clusters[cluster_id] = {
                            'ideas': [],
                            'embeddings': []
                        }
                    
                    clusters[cluster_id]['ideas'].append(idea.idea)
                    
                    # Extract embedding (should be available from clustering step)
                    if hasattr(idea, 'idea_embedding') and idea.idea_embedding is not None:
                        clusters[cluster_id]['embeddings'].append(idea.idea_embedding)
                    else:
                        self.verbose_reporter.stat_line(f"Warning: No embedding for idea in cluster {cluster_id}")
                        # Continue without this idea
                        clusters[cluster_id]['ideas'].pop()  # Remove the idea we just added
                        continue
                    
                    total_ideas += 1
        
        # Remove clusters with no valid embeddings
        clusters = {cid: cdata for cid, cdata in clusters.items() if len(cdata['embeddings']) > 0}
        
        self.stats['total_clusters'] = len(clusters)
        self.verbose_reporter.stat_line(f"Prepared {len(clusters)} clusters with {total_ideas} total ideas")
        return clusters
    
    async def _embed_codebook_texts(self, code_texts: List[str]) -> List[np.ndarray]:
        """
        Embed codebook texts using the existing embedder.
        
        Args:
            code_texts: List of code text descriptions
            
        Returns:
            List of embedding arrays
        """
        # Create dummy response models for embedder
        dummy_responses = []
        for i, text in enumerate(code_texts):
            dummy_responses.append(models.IdeasExtractedModel(
                respondent_id=f"code_{i}",
                response=text,
                response_ideas=[models.IdeasExtractedSubmodel(
                    idea_id=f"code_{i}_1",
                    idea=text
                )],
                idea_count=1
            ))
        
        # Use embedder to get embeddings
        try:
            embedded = await self.embedder.get_embeddings_async(dummy_responses)
            embeddings = []
            
            for item in embedded:
                if hasattr(item, 'idea_embeddings') and item.idea_embeddings:
                    embeddings.append(item.idea_embeddings[0].idea_embedding)
                elif hasattr(item, 'response_ideas') and item.response_ideas:
                    # Try to get from response_ideas if available
                    idea = item.response_ideas[0]
                    if hasattr(idea, 'idea_embedding') and idea.idea_embedding is not None:
                        embeddings.append(idea.idea_embedding)
                    else:
                        raise ValueError(f"No embedding found for code {item.respondent_id}")
                else:
                    raise ValueError(f"No ideas found for code {item.respondent_id}")
            
            return embeddings
            
        except Exception as e:
            self.verbose_reporter.stat_line(f"Error embedding codebook: {str(e)}")
            return []
    
    def _find_k_nearest_codes(self, 
                             cluster_embedding: np.ndarray, 
                             codebook_embeddings: List[np.ndarray]) -> List[Dict]:
        """
        Find k nearest codes using cosine similarity.
        
        Args:
            cluster_embedding: Mean embedding of cluster ideas
            codebook_embeddings: List of codebook embeddings
            
        Returns:
            List of nearest code dictionaries
        """
        if not codebook_embeddings:
            return []
            
        # Convert to numpy array for cosine similarity calculation
        codebook_array = np.array(codebook_embeddings)
        
        # Calculate similarities
        similarities = cosine_similarity(
            cluster_embedding.reshape(1, -1), 
            codebook_array
        )[0]
        
        # Get top k indices
        top_k_indices = np.argsort(similarities)[-self.k:][::-1]
        
        # Remove duplicates and collect codes
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
    
    async def _process_cluster(self, 
                              cluster_id: int, 
                              cluster_data: Dict,
                              codebook_embeddings: List[np.ndarray]) -> Optional[models.CodeGenerationDecision]:
        """
        Process a single cluster through the 6-step GATOS decision process.
        
        Args:
            cluster_id: ID of the cluster
            cluster_data: Dict with 'ideas' and 'embeddings' keys
            codebook_embeddings: Current codebook embeddings
            
        Returns:
            CodeGenerationDecision or None if error
        """
        try:
            # Calculate mean embedding for cluster
            cluster_embedding = np.mean(cluster_data['embeddings'], axis=0)
            
            # Find k nearest codes
            nearest_codes = self._find_k_nearest_codes(cluster_embedding, codebook_embeddings)
            
            # Format cluster ideas and codes for prompt
            cluster_text = "\n".join([f"- {idea}" for idea in cluster_data['ideas']])
            
            if nearest_codes:
                codes_text = "\n".join([
                    f"- {code['code']}: {code['definition']}" 
                    for code in nearest_codes
                ])
            else:
                codes_text = "No existing codes in codebook"
            
            # Build prompt using GATOS format
            prompt = INDUCTIVE_CODEBOOK_GENERATION_PROMPT.format(
                data_type="survey response",
                data_collection_context=f"a survey asking: {self.var_lab}",
                codes=codes_text,
                text=cluster_text,
                redundancy_example="Example: 'Product quality' and 'Quality of product' are redundant because they refer to the same concept"
            )
            
            # Capture first prompt for debugging
            if self.prompt_printer and cluster_id == 0:
                self.prompt_printer.capture_prompt(
                    step_name="gatos_codebook",
                    utility_name="InductiveCodebookGenerator",
                    prompt_content=prompt,
                    prompt_type="Inductive Codebook Generation"
                )
            
            # Get structured response
            response = await self.client.chat.completions.create(
                model=self.model_config.get_model_for_phase("phase1_descriptive"),
                messages=[{"role": "user", "content": prompt}],
                response_model=models.CodeGenerationDecision,
                temperature=0.3,
                max_retries=3
            )
            
            return response
            
        except Exception as e:
            self.verbose_reporter.stat_line(f"Error processing cluster {cluster_id}: {str(e)}")
            return None
    
    def generate(self) -> Dict[str, Any]:
        """
        Main entry point - performs iterative inductive codebook generation.
        
        Returns:
            Dict with 'codebook', 'cluster_assignments', and 'stats'
        """
        self.verbose_reporter.section_header("INDUCTIVE CODEBOOK GENERATION")
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
        self.verbose_reporter.stat_line(f"Processing {len(cluster_ids)} clusters iteratively")
        
        # Track cluster assignments
        cluster_to_code = {}
        
        # Process each cluster sequentially (as per GATOS methodology)
        for i, cluster_id in enumerate(cluster_ids):
            if self.verbose and i % 10 == 0 and i > 0:
                self.verbose_reporter.stat_line(
                    f"Progress: {i}/{len(cluster_ids)} clusters processed "
                    f"({self.stats['new_codes_added']} new codes added)"
                )
            
            # Get current codebook embeddings
            code_texts = [f"{code['code']}: {code['definition']}" for code in self.codebook]
            codebook_embeddings = asyncio.run(self._embed_codebook_texts(code_texts))
            
            if not codebook_embeddings:
                self.verbose_reporter.stat_line(f"Warning: Failed to embed codebook for cluster {cluster_id}")
                continue
                
            # Process cluster
            decision = asyncio.run(self._process_cluster(
                cluster_id, clusters[cluster_id], codebook_embeddings
            ))
            
            if decision:
                if decision.create_new_code and decision.new_code and decision.new_definition:
                    # Add new code to codebook
                    new_code = {
                        'code': decision.new_code,
                        'definition': decision.new_definition,
                        'cluster_origin': cluster_id
                    }
                    self.codebook.append(new_code)
                    cluster_to_code[cluster_id] = decision.new_code
                    self.stats['new_codes_added'] += 1
                    
                    if self.verbose:
                        print(f"  New code added for cluster {cluster_id}: '{decision.new_code}'")
                        
                else:
                    # No new code needed
                    cluster_to_code[cluster_id] = "existing_code"
                    self.stats['no_new_codes_needed'] += 1
            else:
                # Error processing cluster
                cluster_to_code[cluster_id] = "processing_error"
        
        elapsed_time = time.time() - start_time
        
        # Final reporting
        self.verbose_reporter.summary("CODEBOOK GENERATION COMPLETE", {
            "Initial codes": len(self.codebook) - self.stats['new_codes_added'],
            "New codes added": self.stats['new_codes_added'],
            "Final codebook size": len(self.codebook),
            "Clusters processed": len(cluster_ids),
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
                'no_new_codes_needed': self.stats['no_new_codes_needed']
            }
        }