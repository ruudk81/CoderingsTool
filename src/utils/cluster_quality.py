"""Quality metrics for clustering evaluation."""
import numpy as np
from sklearn.metrics import silhouette_score
from collections import Counter
from typing import Dict, Any, List
import hdbscan

class ClusterQualityAnalyzer:
    """Analyze cluster quality metrics."""
    
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        """
        Initialize with embeddings and cluster labels.
        
        Args:
            embeddings: The embeddings used for clustering
            labels: Cluster assignments (-1 for noise)
        """
        self.embeddings = embeddings
        self.labels = labels
        
    def calculate_silhouette(self) -> float:
        """Calculate silhouette score (excluding noise points)."""
        # Filter out noise points
        mask = self.labels != -1
        if mask.sum() < 2:
            return 0.0
            
        filtered_embeddings = self.embeddings[mask]
        filtered_labels = self.labels[mask]
        
        # Need at least 2 clusters for silhouette
        unique_labels = np.unique(filtered_labels)
        if len(unique_labels) < 2:
            return 0.0
            
        return silhouette_score(filtered_embeddings, filtered_labels)
        
    def calculate_noise_ratio(self) -> float:
        """Calculate ratio of points classified as noise."""
        return (self.labels == -1).sum() / len(self.labels)
        
    def calculate_mean_cluster_size(self) -> float:
        """Calculate mean cluster size (excluding noise)."""
        cluster_counts = Counter(self.labels)
        # Remove noise if present
        cluster_counts.pop(-1, None)
        
        if not cluster_counts:
            return 0.0
            
        return np.mean(list(cluster_counts.values()))
        
    def calculate_size_variance(self) -> float:
        """Calculate variance in cluster sizes."""
        cluster_counts = Counter(self.labels)
        cluster_counts.pop(-1, None)
        
        if not cluster_counts:
            return 0.0
            
        sizes = list(cluster_counts.values())
        return np.var(sizes)
        
    def calculate_coverage(self) -> float:
        """Calculate percentage of points in clusters (not noise)."""
        return 1.0 - self.calculate_noise_ratio()
        
    def get_full_report(self) -> Dict[str, float]:
        """Get comprehensive quality metrics."""
        return {
            'silhouette_score': self.calculate_silhouette(),
            'noise_ratio': self.calculate_noise_ratio(),
            'coverage': self.calculate_coverage(),
            'mean_cluster_size': self.calculate_mean_cluster_size(),
            'size_variance': self.calculate_size_variance(),
            'num_clusters': len(np.unique(self.labels[self.labels != -1]))
        }
        
    def calculate_quality_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate overall quality score from metrics.
        
        Higher is better. Weights different aspects:
        - Silhouette score (cluster separation)
        - Coverage (points in clusters vs noise)
        - Noise ratio (penalized)
        """
        # Define weights for different metrics
        weights = {
            'silhouette': 0.4,
            'coverage': 0.3,
            'noise': 0.3  # Negative weight
        }
        
        # Normalize and combine
        silhouette = (metrics['silhouette_score'] + 1) / 2  # Normalize from [-1,1] to [0,1]
        coverage = metrics['coverage']
        noise_penalty = 1 - metrics['noise_ratio']
        
        score = (weights['silhouette'] * silhouette + 
                weights['coverage'] * coverage + 
                weights['noise'] * noise_penalty)
                
        return score

# Test section
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    
    # Create clustered data
    cluster1 = np.random.normal(0, 0.5, (50, 10))
    cluster2 = np.random.normal(3, 0.5, (50, 10))
    cluster3 = np.random.normal(6, 0.5, (30, 10))
    noise = np.random.uniform(-2, 8, (20, 10))
    
    embeddings = np.vstack([cluster1, cluster2, cluster3, noise])
    
    # Create labels (last 20 are noise)
    labels = np.array([0]*50 + [1]*50 + [2]*30 + [-1]*20)
    
    # Analyze quality
    analyzer = ClusterQualityAnalyzer(embeddings, labels)
    report = analyzer.get_full_report()
    quality_score = analyzer.calculate_quality_score(report)
    
    print("Quality Report:")
    for metric, value in report.items():
        print(f"  {metric}: {value:.3f}")
    print(f"\nOverall Quality Score: {quality_score:.3f}")