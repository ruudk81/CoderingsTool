"""
Codebook Display Utility - Similarity-based hierarchical clustering for codebook presentation

This module provides functionality to display codebooks in a semantically organized manner,
grouping similar codes together using hierarchical clustering based on embedding similarity.
"""

import numpy as np
from typing import List, Dict, Optional, Any
from sklearn.metrics.pairwise import cosine_similarity
from hdbscan import HDBSCAN
from config import ModelConfig
import models



class CodebookDisplayer:
    """
    Displays codebooks using similarity-based hierarchical clustering.
    
    Groups codes by semantic similarity using embeddings and HDBSCAN clustering
    to create a more intuitive codebook presentation.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    
    def cluster_codes_by_similarity(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Cluster code embeddings using HDBSCAN for hierarchical grouping.
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            Array of cluster labels (-1 for noise/outliers)
        """
        if len(embeddings) < 2:
            return np.array([0] * len(embeddings))
        
        # Convert to numpy array
        embedding_matrix = np.array(embeddings)
        
        # Use HDBSCAN with very permissive parameters for similarity-based ordering
        clusterer = HDBSCAN(
            min_cluster_size=2,              # Minimum required by HDBSCAN
            min_samples=1,                   # Minimal samples required for core points
            metric='euclidean',              # Works well with normalized embeddings
            cluster_selection_epsilon=0.8,   # Very permissive - allow loose similarity groupings
            cluster_selection_method='eom',  # Excess of mass method
            allow_single_cluster=True        # Allow all codes in one cluster if they're similar
        )
        
        cluster_labels = clusterer.fit_predict(embedding_matrix)
        
        if self.verbose:
            unique_clusters = len(set(cluster_labels))
            noise_points = np.sum(cluster_labels == -1)
            print(f"Clustering results: {unique_clusters} clusters, {noise_points} outliers")
        
        return cluster_labels
    
    def calculate_similarity_matrix(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Calculate cosine similarity matrix between all code embeddings.
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            Similarity matrix
        """
        if not embeddings:
            return np.array([])
        
        embedding_matrix = np.array(embeddings)
        return cosine_similarity(embedding_matrix)
    
    def organize_codes_by_clusters(self, 
                                   codes: List[models.CodebookEntry], 
                                   cluster_labels: np.ndarray,
                                   similarity_matrix: np.ndarray) -> List[Dict[str, Any]]:
        """
        Organize codes into cluster groups with similarity information.
        
        Args:
            codes: List of CodebookEntry objects
            cluster_labels: Cluster assignment for each code
            similarity_matrix: Cosine similarity matrix
            
        Returns:
            List of cluster dictionaries with codes and metadata
        """
        clusters = {}
        
        # Group codes by cluster
        for i, (code, cluster_id) in enumerate(zip(codes, cluster_labels)):
            if cluster_id not in clusters:
                clusters[cluster_id] = {
                    'cluster_id': cluster_id,
                    'codes': [],
                    'indices': []
                }
            clusters[cluster_id]['codes'].append(code)
            clusters[cluster_id]['indices'].append(i)
        
        # Calculate cluster statistics
        organized_clusters = []
        for cluster_id, cluster_data in clusters.items():
            indices = cluster_data['indices']
            
            # Calculate average internal similarity for cluster
            if len(indices) > 1:
                cluster_similarities = []
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        sim = similarity_matrix[indices[i], indices[j]]
                        cluster_similarities.append(sim)
                avg_similarity = np.mean(cluster_similarities) if cluster_similarities else 0
            else:
                avg_similarity = 1.0  # Single item clusters have perfect internal similarity
            
            cluster_data['avg_similarity'] = avg_similarity
            cluster_data['size'] = len(cluster_data['codes'])
            organized_clusters.append(cluster_data)
        
        # Sort clusters by size (largest first) and similarity
        organized_clusters.sort(key=lambda x: (-x['size'], -x['avg_similarity']))
        
        return organized_clusters
    
    def display_clustered_codebook(self, 
                                   codebook_main: models.CodebookModel, 
                                   generator: Optional[Any] = None,
                                   reasoning_results: Optional[Any] = None) -> None:
        """
        Display codebook organized by semantic similarity clusters using cached embeddings.
        
        Args:
            codebook_main: The main codebook model with codes
            generator: Generator object for accessing cached embeddings from shared_codebook
            reasoning_results: CodeGeneratorReasoningResults with final_embeddings (fallback source)
        """
        if not codebook_main or not codebook_main.codes:
            print("CLUSTERED CODEBOOK DISPLAY")
            print("No codes available to display.")
            return
        
        codes = codebook_main.codes
        embeddings = None
        
        # Try to get cached embeddings from generator
        if generator and hasattr(generator, 'shared_codebook'):
            try:
                current_version = generator.shared_codebook._version
                embedding_cache = generator.shared_codebook._embedding_cache
                
                # First try current version
                cached_embeddings = embedding_cache.get(current_version)
                if cached_embeddings and len(cached_embeddings) == len(codes):
                    embeddings = cached_embeddings
                    if self.verbose:
                        print(f"Using cached code embeddings from current version {current_version}")
                else:
                    # Try most recent available version with compatible size
                    if embedding_cache:
                        available_versions = sorted(embedding_cache.keys(), reverse=True)
                        if self.verbose:
                            print(f"Current version {current_version} not found, trying recent versions: {available_versions}")
                        
                        for version in available_versions:
                            candidate_embeddings = embedding_cache.get(version)
                            if candidate_embeddings and len(candidate_embeddings) == len(codes):
                                embeddings = candidate_embeddings
                                if self.verbose:
                                    print(f"Using cached embeddings from version {version} (compatible size: {len(candidate_embeddings)})")
                                break
                            elif self.verbose and candidate_embeddings:
                                print(f"Version {version} has {len(candidate_embeddings)} embeddings, need {len(codes)} - skipping")
                        
                        if embeddings is None and self.verbose:
                            print(f"WARNING: No compatible cached embeddings found in generator")
                            print(f"Current codebook has {len(codes)} codes")
                            for v in available_versions[:3]:  # Show first few versions
                                emb_count = len(embedding_cache[v]) if embedding_cache[v] else 0
                                print(f"  Version {v}: {emb_count} embeddings")
                    elif self.verbose:
                        print("WARNING: No cached embeddings available in generator")
                        
            except Exception as e:
                if self.verbose:
                    print(f"WARNING: Could not access cached embeddings from generator: {e}")
        
        # Fallback: Try reasoning results if generator embeddings not available
        if embeddings is None and reasoning_results:
            try:
                if hasattr(reasoning_results, 'final_embeddings') and reasoning_results.final_embeddings:
                    cached_embeddings = reasoning_results.final_embeddings
                    if len(cached_embeddings) == len(codes):
                        # Convert from list of lists back to numpy arrays
                        embeddings = [np.array(emb) for emb in cached_embeddings]
                        if self.verbose:
                            print(f"Using final embeddings from reasoning results ({len(embeddings)} embeddings)")
                    elif self.verbose:
                        print(f"WARNING: Reasoning results embedding count mismatch: {len(cached_embeddings)} vs {len(codes)} codes")
                elif self.verbose:
                    print("WARNING: No final_embeddings found in reasoning results")
            except Exception as e:
                if self.verbose:
                    print(f"WARNING: Could not access embeddings from reasoning results: {e}")
        
        # If no embeddings available, fall back to simple display
        if embeddings is None or not embeddings:
            print("CODEBOOK (Simple Display - No Cached Embeddings)")
            print("Note: Run the pipeline to generate embeddings for clustered display")
            print("=" * 60)
            for i, entry in enumerate(codes, 1):
                print(f"{i:2d}. {entry.code.upper()}")
                definition = entry.definition
                if len(definition) > 100:
                    definition = definition[:97] + "..."
                print(f"    {definition}")
                print()
            return
        
        # Perform clustering with cached embeddings
        cluster_labels = self.cluster_codes_by_similarity(embeddings)
        similarity_matrix = self.calculate_similarity_matrix(embeddings)
        organized_clusters = self.organize_codes_by_clusters(codes, cluster_labels, similarity_matrix)
        
        # Display clustered results
        print("CLUSTERED CODEBOOK DISPLAY (Using Cached Embeddings)")
        print("=" * 80)
        print(f"Total codes: {len(codes)}")
        print(f"Semantic clusters: {len(organized_clusters)}")
        print("=" * 80)
        
        overall_idx = 1
        for cluster in organized_clusters:
            cluster_id = cluster['cluster_id']
            cluster_size = cluster['size']
            avg_sim = cluster['avg_similarity']
            
            # Cluster header
            if cluster_id == -1:
                cluster_name = ">> Similar Codes (Loose Grouping)"
            else:
                cluster_name = f">> Cluster {cluster_id + 1}"
            
            print(f"\n{cluster_name} ({cluster_size} codes, avg similarity: {avg_sim:.3f})")
            print("-" * 60)
            
            # Display codes in cluster
            for code in cluster['codes']:
                print(f"{overall_idx:2d}. {code.code.upper()}")
                definition = code.definition
                if len(definition) > 100:
                    definition = definition[:97] + "..."
                print(f"    {definition}")
                overall_idx += 1
                print()
        
        print("=" * 80)


def display_clustered_codebook(codebook_main: models.CodebookModel, 
                              generator: Optional[Any] = None,
                              model_config: Optional[ModelConfig] = None,
                              verbose: bool = True,
                              reasoning_results: Optional[Any] = None) -> None:
    """
    Convenience function to display codebook with similarity-based clustering.
    
    Args:
        codebook_main: The main codebook model with codes
        generator: Generator object for accessing cached embeddings from shared_codebook
        model_config: Not used (kept for compatibility)
        verbose: Enable verbose output
        reasoning_results: CodeGeneratorReasoningResults with final_embeddings (fallback source)
    """
    displayer = CodebookDisplayer(verbose)
    displayer.display_clustered_codebook(codebook_main, generator, reasoning_results)