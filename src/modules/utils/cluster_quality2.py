"""Quality metrics for clustering evaluation"""

import numpy as np
from sklearn.metrics import silhouette_score
from typing import Dict, List, Optional
from datetime import datetime
import warnings


class ClusterQualityMetrics:
    """Calculate quality metrics for clustering results"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        
    def calculate_silhouette_score(self, embeddings: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate silhouette score for clustering quality
        
        Args:
            embeddings: Array of embeddings (n_samples, n_features)
            labels: Cluster labels for each point
            
        Returns:
            Silhouette score (-1 to 1, higher is better)
        """
        # Filter out noise points (-1 labels)
        mask = labels != -1
        if np.sum(mask) < 2:
            return 0.0
            
        # Need at least 2 clusters for silhouette score
        unique_labels = np.unique(labels[mask])
        if len(unique_labels) < 2:
            return 0.0
            
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                score = silhouette_score(embeddings[mask], labels[mask])
            return float(score)
        except Exception as e:
            if self.verbose:
                print(f"Error calculating silhouette score: {e}")
            return 0.0
    
    def calculate_noise_ratio(self, labels: np.ndarray) -> float:
        """
        Calculate the ratio of points marked as noise
        
        Args:
            labels: Cluster labels (-1 indicates noise)
            
        Returns:
            Ratio of noise points (0 to 1, lower is better)
        """
        noise_count = np.sum(labels == -1)
        total_count = len(labels)
        return noise_count / total_count if total_count > 0 else 0.0
    
    def calculate_mean_cluster_size(self, labels: np.ndarray) -> float:
        """
        Calculate average cluster size (excluding noise)
        
        Args:
            labels: Cluster labels
            
        Returns:
            Mean number of items per cluster
        """
        # Filter out noise
        valid_labels = labels[labels != -1]
        if len(valid_labels) == 0:
            return 0.0
            
        unique_labels, counts = np.unique(valid_labels, return_counts=True)
        return float(np.mean(counts))
    
    def calculate_cluster_count(self, labels: np.ndarray) -> int:
        """
        Count number of clusters (excluding noise)
        
        Args:
            labels: Cluster labels
            
        Returns:
            Number of clusters found
        """
        unique_labels = np.unique(labels)
        # Don't count -1 (noise) as a cluster
        return len(unique_labels[unique_labels != -1])
    
    def calculate_embedding_coverage(self, labels: np.ndarray) -> float:
        """
        Calculate percentage of embeddings successfully clustered
        
        Args:
            labels: Cluster labels
            
        Returns:
            Coverage ratio (0 to 1, higher is better)
        """
        clustered_count = np.sum(labels != -1)
        total_count = len(labels)
        return clustered_count / total_count if total_count > 0 else 0.0
    
    def calculate_quality_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate overall quality score from individual metrics
        
        Args:
            metrics: Dictionary of individual metrics
            
        Returns:
            Overall quality score (0 to 1, higher is better)
        """
        # Weights for different metrics
        weights = {
            'silhouette': 0.4,
            'coverage': 0.3,
            'noise': 0.3  # Negative weight
        }
        
        # Normalize silhouette score from [-1, 1] to [0, 1]
        normalized_silhouette = (metrics.get('silhouette_score', 0) + 1) / 2
        
        # Calculate weighted score
        quality_score = (
            weights['silhouette'] * normalized_silhouette +
            weights['coverage'] * metrics.get('embedding_coverage', 0) +
            weights['noise'] * (1 - metrics.get('noise_ratio', 0))
        )
        
        return max(0.0, min(1.0, quality_score))
    
    def determine_quality_status(self, quality_score: float, min_threshold: float = 0.3) -> str:
        """
        Determine quality status based on score
        
        Args:
            quality_score: Overall quality score (0-1)
            min_threshold: Minimum acceptable quality
            
        Returns:
            Quality status string
        """
        if quality_score >= 0.6:
            return "good"
        elif quality_score >= min_threshold:
            return "acceptable"
        else:
            return "poor"
    
    def calculate_all_metrics(self, 
                            embeddings: np.ndarray, 
                            labels: np.ndarray,
                            min_quality_threshold: float = 0.3) -> Dict[str, any]:
        """
        Calculate all quality metrics for clustering results
        
        Args:
            embeddings: Array of embeddings
            labels: Cluster labels
            min_quality_threshold: Minimum acceptable quality score
            
        Returns:
            Dictionary containing all metrics
        """
        # Calculate individual metrics
        silhouette = self.calculate_silhouette_score(embeddings, labels)
        noise_ratio = self.calculate_noise_ratio(labels)
        mean_size = self.calculate_mean_cluster_size(labels)
        cluster_count = self.calculate_cluster_count(labels)
        coverage = self.calculate_embedding_coverage(labels)
        
        # Create metrics dictionary
        metrics = {
            "silhouette_score": silhouette,
            "noise_ratio": noise_ratio,
            "mean_cluster_size": mean_size,
            "cluster_count": cluster_count,
            "embedding_coverage": coverage
        }
        
        # Calculate overall quality
        quality_score = self.calculate_quality_score(metrics)
        quality_status = self.determine_quality_status(quality_score, min_quality_threshold)
        
        # Add overall metrics
        metrics.update({
            "quality_score": quality_score,
            "quality_status": quality_status,
            "timestamp": datetime.now().isoformat()
        })
        
        if self.verbose:
            self._print_metrics_summary(metrics)
        
        return metrics
    
    def _print_metrics_summary(self, metrics: Dict[str, any]):
        """Print a summary of the quality metrics"""
        print("\n=== Clustering Quality Metrics ===")
        print(f"Silhouette Score: {metrics['silhouette_score']:.3f}")
        print(f"Noise Ratio: {metrics['noise_ratio']:.2%}")
        print(f"Mean Cluster Size: {metrics['mean_cluster_size']:.1f}")
        print(f"Cluster Count: {metrics['cluster_count']}")
        print(f"Embedding Coverage: {metrics['embedding_coverage']:.2%}")
        print(f"Overall Quality Score: {metrics['quality_score']:.3f}")
        print(f"Quality Status: {metrics['quality_status']}")
        print("================================\n")


# Test section
if __name__ == "__main__":
    """Test the quality metrics with actual embeddings"""
    import sys
    from pathlib import Path
    
    # Add project paths
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root / "src"))
    
    from cache_manager import CacheManager
    from cache_config import CacheConfig
    import models
    
    # Initialize cache manager
    cache_config = CacheConfig()
    cache_manager = CacheManager(cache_config)
    
    # Load embeddings from cache
    filename = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
    embedded_data = cache_manager.load_from_cache(filename, "embeddings", models.EmbeddingsModel)
    
    if embedded_data:
        print(f"Loaded {len(embedded_data)} items from cache")
        
        # Extract embeddings based on type
        embeddings_list = []
        for item in embedded_data:
            if item.response_segment:
                for segment in item.response_segment:
                    # Use description embeddings by default
                    if segment.description_embedding is not None:
                        embeddings_list.append(segment.description_embedding)
        
        if embeddings_list:
            embeddings_array = np.array(embeddings_list)
            print(f"Extracted {len(embeddings_array)} embeddings")
            
            # Test 1: Create synthetic well-separated clusters
            print("\n=== Test 1: Well-separated clusters ===")
            n_samples = len(embeddings_array)
            good_labels = np.array([i % 5 for i in range(n_samples)])  # 5 equal clusters
            
            metrics_calculator = ClusterQualityMetrics(verbose=True)
            good_metrics = metrics_calculator.calculate_all_metrics(
                embeddings_array, 
                good_labels,
                min_quality_threshold=0.3
            )
            
            # Test 2: Create noisy clustering
            print("\n=== Test 2: Noisy clustering ===")
            noisy_labels = np.array([-1 if i % 3 == 0 else i % 5 for i in range(n_samples)])
            
            noisy_metrics = metrics_calculator.calculate_all_metrics(
                embeddings_array,
                noisy_labels,
                min_quality_threshold=0.3
            )
            
            # Test 3: All noise
            print("\n=== Test 3: All noise ===")
            all_noise_labels = np.full(n_samples, -1)
            
            noise_metrics = metrics_calculator.calculate_all_metrics(
                embeddings_array,
                all_noise_labels,
                min_quality_threshold=0.3
            )
            
            # Test 4: Actual clustering with HDBSCAN
            print("\n=== Test 4: Actual HDBSCAN clustering ===")
            import hdbscan
            from umap import UMAP
            
            # Reduce dimensions first
            reducer = UMAP(n_components=10, random_state=42, n_jobs=1)
            reduced_embeddings = reducer.fit_transform(embeddings_array)
            
            # Cluster with HDBSCAN
            clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean')
            actual_labels = clusterer.fit_predict(reduced_embeddings)
            
            actual_metrics = metrics_calculator.calculate_all_metrics(
                reduced_embeddings,
                actual_labels,
                min_quality_threshold=0.3
            )
            
        else:
            print("No embeddings found in data")
    else:
        print("No cached embeddings found. Please run the pipeline first.")
        print("You can also test with synthetic data by modifying this script.")