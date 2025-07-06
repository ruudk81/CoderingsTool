"""
Clean clusterer - minimal implementation to test if basic clustering works correctly.
Just segment_id + description → embeddings → UMAP → HDBSCAN → clusters
"""

import os, sys
sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

import numpy as np
from typing import List, Dict, Tuple
from openai import OpenAI
from umap import UMAP
import hdbscan
from config import OPENAI_API_KEY, ClusteringConfig, DEFAULT_CLUSTERING_CONFIG
import models
from collections import Counter
import asyncio


class CleanClusterer:
    """Minimal clusterer for testing semantic coherence"""
    
    def __init__(self, config: ClusteringConfig = None):
        self.config = config or DEFAULT_CLUSTERING_CONFIG
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        
        # Initialize UMAP with same config as main clusterer
        umap_config = self.config.umap
        self.umap_model = UMAP(
            n_neighbors=umap_config.n_neighbors,
            n_components=umap_config.n_components,
            min_dist=umap_config.min_dist,
            metric=umap_config.metric,
            random_state=umap_config.random_state,
            n_jobs=umap_config.n_jobs,
            low_memory=umap_config.low_memory,
            transform_seed=umap_config.transform_seed
        )
        
        # Initialize HDBSCAN with same config as main clusterer
        hdbscan_config = self.config.hdbscan
        hdbscan_params = {
            'metric': hdbscan_config.metric,
            'cluster_selection_method': hdbscan_config.cluster_selection_method,
            'prediction_data': hdbscan_config.prediction_data,
            'approx_min_span_tree': hdbscan_config.approx_min_span_tree,
            'gen_min_span_tree': hdbscan_config.gen_min_span_tree
        }
        if hdbscan_config.min_cluster_size is not None:
            hdbscan_params['min_cluster_size'] = hdbscan_config.min_cluster_size
        if hdbscan_config.min_samples is not None:
            hdbscan_params['min_samples'] = hdbscan_config.min_samples
            
        self.hdbscan_model = hdbscan.HDBSCAN(**hdbscan_params)
        
        print(f"Clean Clusterer initialized with:")
        print(f"  UMAP: {umap_config.n_neighbors} neighbors, {umap_config.n_components} components")
        print(f"  HDBSCAN: min_cluster_size={hdbscan_config.min_cluster_size}, min_samples={hdbscan_config.min_samples}")
    
    def extract_segments(self, cluster_results: List[models.ClusterModel]) -> List[Tuple[str, str]]:
        """Extract (segment_id, description) pairs from cluster results"""
        segments = []
        
        for resp in cluster_results:
            if resp.response_segment:
                for seg in resp.response_segment:
                    segments.append((seg.segment_id, seg.segment_description))
        
        print(f"Extracted {len(segments)} segments for clean clustering")
        return segments
    
    def generate_embeddings(self, descriptions: List[str]) -> np.ndarray:
        """Generate embeddings for descriptions"""
        print("Generating embeddings...")
        
        # Use same embedding model as main pipeline
        response = self.client.embeddings.create(
            input=descriptions,
            model="text-embedding-3-large"
        )
        
        embeddings = []
        for item in response.data:
            embeddings.append(np.array(item.embedding, dtype=np.float32))
        
        embeddings_array = np.array(embeddings)
        print(f"Generated embeddings: {embeddings_array.shape}")
        return embeddings_array
    
    def cluster_segments(self, segments: List[Tuple[str, str]]) -> Dict[str, int]:
        """Clean clustering: segments → embeddings → UMAP → HDBSCAN → clusters"""
        
        if not segments:
            return {}
        
        segment_ids = [seg_id for seg_id, _ in segments]
        descriptions = [desc for _, desc in segments]
        
        print(f"\nCLEAN CLUSTERING PIPELINE")
        print(f"Input: {len(segments)} segments")
        
        # Step 1: Generate embeddings
        embeddings = self.generate_embeddings(descriptions)
        
        # Step 2: UMAP dimensionality reduction
        print("Applying UMAP dimensionality reduction...")
        reduced_embeddings = self.umap_model.fit_transform(embeddings)
        print(f"Reduced embeddings: {reduced_embeddings.shape}")
        
        # Step 3: HDBSCAN clustering
        print("Applying HDBSCAN clustering...")
        cluster_labels = self.hdbscan_model.fit_predict(reduced_embeddings)
        
        # Count clusters
        cluster_counts = Counter(cluster_labels)
        num_clusters = len([c for c in cluster_counts.keys() if c != -1])
        noise_count = cluster_counts.get(-1, 0)
        
        print(f"Clustering results:")
        print(f"  Clusters found: {num_clusters}")
        print(f"  Noise points: {noise_count}")
        print(f"  Clustered points: {len(segments) - noise_count}")
        
        # Create segment_id → cluster mapping
        segment_cluster_map = {}
        for seg_id, cluster_id in zip(segment_ids, cluster_labels):
            segment_cluster_map[seg_id] = cluster_id if cluster_id != -1 else None
        
        return segment_cluster_map
    
    def analyze_clusters(self, segments: List[Tuple[str, str]], 
                        segment_cluster_map: Dict[str, int]) -> None:
        """Analyze semantic coherence of clusters"""
        
        print(f"\n{'='*60}")
        print("CLEAN CLUSTERING ANALYSIS")
        print(f"{'='*60}")
        
        # Group segments by cluster
        clusters = {}
        for seg_id, description in segments:
            cluster_id = segment_cluster_map.get(seg_id)
            if cluster_id is not None:
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append((seg_id, description))
        
        # Analyze each cluster
        for cluster_id in sorted(clusters.keys()):
            cluster_segments = clusters[cluster_id]
            print(f"\nCLUSTER {cluster_id} ({len(cluster_segments)} segments):")
            
            # Show all segments in cluster
            for seg_id, description in cluster_segments:
                print(f"  {seg_id}: {description}")
            
            # Analyze semantic topics in cluster
            topics = set()
            for _, description in cluster_segments:
                desc_lower = description.lower()
                if any(word in desc_lower for word in ['temperatuur', 'warm', 'koud', 'graden']):
                    topics.add('temperature')
                elif any(word in desc_lower for word in ['kantine', 'eten', 'voedsel', 'maaltijd', 'lunch']):
                    topics.add('food')
                elif any(word in desc_lower for word in ['prijs', 'duur', 'goedkoop', 'kosten']):
                    topics.add('price')
                elif any(word in desc_lower for word in ['gezond', 'ongezond', 'snoep', 'suiker', 'zout']):
                    topics.add('health')
                elif any(word in desc_lower for word in ['school', 'leerling', 'docent', 'onderwijs']):
                    topics.add('education')
                else:
                    topics.add('other')
            
            print(f"  → Semantic topics: {sorted(topics)}")
            if len(topics) > 1:
                print(f"  ⚠️  MIXED CLUSTER: {len(topics)} different semantic topics!")
            else:
                print(f"  ✅ COHERENT CLUSTER: Single semantic topic")


def run_clean_clustering_test(cluster_results: List[models.ClusterModel], 
                             config: ClusteringConfig = None) -> Dict[str, int]:
    """Run clean clustering test on cluster results"""
    
    print("\n" + "="*80)
    print("CLEAN CLUSTERER TEST")
    print("="*80)
    
    # Initialize clean clusterer
    clean_clusterer = CleanClusterer(config)
    
    # Extract segments
    segments = clean_clusterer.extract_segments(cluster_results)
    
    # Perform clean clustering
    segment_cluster_map = clean_clusterer.cluster_segments(segments)
    
    # Analyze results
    clean_clusterer.analyze_clusters(segments, segment_cluster_map)
    
    return segment_cluster_map


# Function to add to pipeline.py
def compare_with_original_clusters(original_cluster_results: List[models.ClusterModel], 
                                 clean_cluster_map: Dict[str, int]) -> None:
    """Compare clean clustering results with original clustering"""
    
    print(f"\n{'='*80}")
    print("COMPARISON: ORIGINAL vs CLEAN CLUSTERING")
    print(f"{'='*80}")
    
    # Build original cluster map
    original_cluster_map = {}
    for resp in original_cluster_results:
        if resp.response_segment:
            for seg in resp.response_segment:
                original_cluster_map[seg.segment_id] = seg.initial_cluster
    
    # Compare assignments
    agreements = 0
    disagreements = 0
    
    for seg_id in original_cluster_map:
        original_cluster = original_cluster_map[seg_id]
        clean_cluster = clean_cluster_map.get(seg_id)
        
        # For comparison, we just check if both assigned to clusters or both to noise
        original_clustered = original_cluster is not None and original_cluster != -1
        clean_clustered = clean_cluster is not None
        
        if original_clustered == clean_clustered:
            agreements += 1
        else:
            disagreements += 1
    
    total = agreements + disagreements
    agreement_rate = agreements / total * 100 if total > 0 else 0
    
    print(f"Clustering agreement: {agreements}/{total} ({agreement_rate:.1f}%)")
    print(f"Disagreements: {disagreements}")
    
    if agreement_rate < 70:
        print("⚠️  Low agreement - significant differences between approaches")
    else:
        print("✅ High agreement - clustering approaches are consistent")