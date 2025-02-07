import asyncio
from typing import List, Dict
import numpy as np
from tqdm.asyncio import tqdm
import logging
from sklearn.metrics.pairwise import cosine_similarity

try:
    # When running as a script
    from labeller import (
        LabellerConfig, InitialLabel, BatchLabelResponse, ClusterData
    )
except ImportError:
    # When imported as a module
    from .labeller import (
        LabellerConfig, InitialLabel, BatchLabelResponse, ClusterData
    )
from prompts import INITIAL_LABEL_PROMPT

logger = logging.getLogger(__name__)


class Phase1Labeller:
    """Phase 1: Initial label generation for clusters"""
    
    def __init__(self, config: LabellerConfig, client):
        self.config = config
        self.client = client
        self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)
    
    async def label_clusters(self, 
                           cluster_data: Dict[int, ClusterData], 
                           var_lab: str) -> Dict[int, InitialLabel]:
        """Generate initial labels for all clusters"""
        logger.info("Phase 1: Generating initial cluster labels...")
        
        # Create batches
        cluster_ids = list(cluster_data.keys())
        batches = self._create_batches(cluster_ids, self.config.batch_size)
        
        # Process batches concurrently
        tasks = []
        for batch_ids in batches:
            batch_data = [cluster_data[cid] for cid in batch_ids]
            task = self._label_batch(batch_data, var_lab, batch_ids)
            tasks.append(task)
        
        # Execute all tasks with progress bar
        results = {}
        with tqdm(total=len(batches), desc="Labeling clusters") as pbar:
            for coro in asyncio.as_completed(tasks):
                batch_result = await coro
                results.update(batch_result)
                pbar.update(1)
        
        return results
    
    async def _label_batch(self,
                          batch_data: List[ClusterData],
                          var_lab: str,
                          batch_ids: List[int]) -> Dict[int, InitialLabel]:
        """Label a batch of clusters"""
        async with self.semaphore:
            try:
                # Get representative items for each cluster
                batch_representatives = []
                for cluster in batch_data:
                    representatives = self._get_representative_items(cluster, n=5)
                    batch_representatives.append(representatives)
                
                # Create prompt
                prompt = self._create_batch_prompt(batch_data, batch_representatives, var_lab)
                
                # Get labels from LLM
                response = await self._get_llm_response(prompt)
                
                # Convert to dict
                result = {}
                for label in response.labels:
                    if label.cluster_id in batch_ids:
                        result[label.cluster_id] = label
                
                # Handle missing labels
                for cid in batch_ids:
                    if cid not in result:
                        result[cid] = InitialLabel(
                            cluster_id=cid,
                            label=f"Cluster {cid}",
                            keywords=["unlabeled"],
                            confidence=0.0
                        )
                
                return result
                
            except Exception as e:
                logger.error(f"Error labeling batch: {e}")
                # Return fallback labels
                return {
                    cid: InitialLabel(
                        cluster_id=cid,
                        label=f"Cluster {cid}",
                        keywords=["error"],
                        confidence=0.0
                    )
                    for cid in batch_ids
                }
    
    def _get_representative_items(self, cluster: ClusterData, n: int = 5) -> List[Dict[str, str]]:
        """Get most representative items using cosine similarity to centroid"""
        # Calculate similarities to centroid
        similarities = cosine_similarity([cluster.centroid], cluster.embeddings)[0]
        
        # Get top n indices
        top_indices = np.argsort(similarities)[-n:][::-1]
        
        # Create representative items
        representatives = []
        for idx in top_indices:
            if idx < len(cluster.descriptive_codes):
                representatives.append({
                    'code': cluster.descriptive_codes[idx],
                    'description': cluster.code_descriptions[idx],
                    'similarity': float(similarities[idx])
                })
        
        return representatives
    
    def _create_batch_prompt(self,
                           batch_data: List[ClusterData],
                           batch_representatives: List[List[Dict]],
                           var_lab: str) -> str:
        """Create prompt for batch labeling"""
        # Start with the template
        prompt = INITIAL_LABEL_PROMPT.replace("{var_lab}", var_lab)
        prompt = prompt.replace("{language}", self.config.language)
        
        # Add cluster information
        clusters_info = []
        for cluster, representatives in zip(batch_data, batch_representatives):
            cluster_text = f"\nCluster {cluster.cluster_id}:"
            cluster_text += f"\n- Size: {cluster.size} responses"
            cluster_text += "\n- Most representative items:"
            
            for i, item in enumerate(representatives, 1):
                cluster_text += f"\n  {i}. Code: {item['code']}"
                cluster_text += f"\n     Description: {item['description']}"
                cluster_text += f"\n     Similarity: {item['similarity']:.3f}"
            
            clusters_info.append(cluster_text)
        
        prompt = prompt.replace("{clusters}", "\n".join(clusters_info))
        
        return prompt
    
    async def _get_llm_response(self, prompt: str) -> BatchLabelResponse:
        """Get response from LLM with retry logic"""
        messages = [
            {"role": "system", "content": "You are an expert in thematic analysis and survey response clustering."},
            {"role": "user", "content": prompt}
        ]
        
        for attempt in range(self.config.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    response_model=BatchLabelResponse,
                    temperature=0.3,
                    max_tokens=4000
                )
                return response
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    raise e
    
    def _create_batches(self, items: List[int], batch_size: int) -> List[List[int]]:
        """Create batches from a list of items"""
        return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


if __name__ == "__main__":
    """Test Phase 1 with actual cached cluster data"""
    import sys
    from pathlib import Path
    import json
    import pickle
    
    # Add project paths
    sys.path.append(str(Path(__file__).parents[2]))  # Add src directory
    
    from openai import AsyncOpenAI
    import instructor
    from config import OPENAI_API_KEY, DEFAULT_MODEL
    from cache_manager import CacheManager
    from cache_config import CacheConfig
    import data_io
    import models
    
    # Initialize cache manager
    cache_config = CacheConfig()
    cache_manager = CacheManager(cache_config)
    
    # File and variable info
    filename = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
    var_name = "Q20"
    
    # Load cluster results from cache
    cluster_results = cache_manager.load_from_cache(filename, "clusters", models.ClusterModel)
    
    if cluster_results:
        print(f"Loaded {len(cluster_results)} cluster results from cache")
        
        # Extract cluster data from the results (same as in labeller.py)
        from labeller import Labeller
        temp_labeller = Labeller()
        cluster_data = temp_labeller.extract_cluster_data(cluster_results)
        print(f"Extracted data for {len(cluster_data)} clusters")
        
        # Get variable label
        data_loader = data_io.DataLoader()
        var_lab = data_loader.get_varlab(filename=filename, var_name=var_name)
        print(f"Variable label: {var_lab}")
        
        # Initialize configuration
        config = LabellerConfig(
            api_key=OPENAI_API_KEY,
            model=DEFAULT_MODEL,
            batch_size=10  # Process in larger batches for real data
        )
        
        # Initialize phase 1 labeller
        client = instructor.from_openai(AsyncOpenAI(api_key=config.api_key))
        phase1 = Phase1Labeller(config, client)
        
        async def run_test():
            """Run the test"""
            print("=== Testing Phase 1: Initial Label Generation with Real Data ===")
            print(f"Variable label: {var_lab}")
            print(f"Number of clusters: {len(cluster_data)}")
            
            try:
                # Generate labels
                labels = await phase1.label_clusters(cluster_data, var_lab)
                
                # Display results (sample)
                print("\n=== Results (showing first 10) ===")
                for i, (cluster_id, label) in enumerate(labels.items()):
                    # if i >= 10:
                    #     print(f"\n... and {len(labels) - 10} more")
                    #     break
                    print(f"\nCluster {cluster_id}:")
                    print(f"  Label: {label.label}")
                    print(f"  Keywords: {', '.join(label.keywords[:5])}")
                    print(f"  Confidence: {label.confidence:.2f}")
                    
                    # Show representative items
                    cluster = cluster_data[cluster_id]
                    reps = phase1._get_representative_items(cluster, n=3)
                    print("  Representative items:")
                    for j, item in enumerate(reps, 1):
                        print(f"    {j}. {item['code'][:50]}... ({item['similarity']:.3f})")
                
                # Save to cache for phase 2
                cache_key = 'phase1_labels'
                cache_manager.cache_intermediate_data(labels, filename, cache_key)
                print(f"\nSaved results to cache with key '{cache_key}'")
                
                # Also save to JSON for inspection
                output_file = Path("phase1_test_results.json")
                output_data = {
                    str(cluster_id): {
                        "label": label.label,
                        "keywords": label.keywords,
                        "confidence": label.confidence
                    }
                    for cluster_id, label in labels.items()
                }
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                
                print(f"Results saved to: {output_file}")
                
            except Exception as e:
                print(f"Error during testing: {e}")
                import traceback
                traceback.print_exc()
        
        # Run the test
        asyncio.run(run_test())
    else:
        print("No cached cluster data found.")
        print("Please run the clustering pipeline first:")
        print("  python clusterer.py")