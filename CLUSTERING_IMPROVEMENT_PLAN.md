# Clustering Improvement Plan

## Current Implementation Analysis

### Strengths
1. Uses HDBSCAN for automatic cluster detection
2. UMAP for dimensionality reduction
3. Implements hierarchical clustering (meta, meso, micro)
4. Weighted keyword extraction for cluster interpretation
5. Handles both code and description embeddings

### Weaknesses
1. **Hardcoded parameters**:
   - UMAP: n_neighbors=5, n_components=10
   - HDBSCAN: min_cluster_size not set (defaults to 5)
   - Fixed n-gram range (1, 3)

2. **No quality metrics**:
   - No silhouette score
   - No Davies-Bouldin score
   - No cluster stability analysis

3. **Limited algorithm options**:
   - Only HDBSCAN for clustering
   - Only UMAP for dimensionality reduction

4. **Poor outlier handling**:
   - Outliers are just assigned sequential IDs
   - No strategy for re-assignment

5. **No visualization**:
   - No 2D/3D plots of clusters
   - No dendrogram for hierarchical view

## Improvement Roadmap

### Phase 2.1: Configuration System
```python
@dataclass
class ClusteringConfig:
    # Dimensionality reduction
    reducer_type: str = "umap"  # umap, tsne, pca
    n_components: int = 10
    n_neighbors: int = 15
    min_dist: float = 0.1
    
    # Clustering
    clusterer_type: str = "hdbscan"  # hdbscan, dbscan, kmeans
    min_cluster_size: int = 5
    min_samples: int = 3
    metric: str = "euclidean"
    
    # Text processing
    ngram_range: Tuple[int, int] = (1, 3)
    max_features: int = 1000
    
    # Meta-clustering
    meta_min_cluster_size: int = 2
    
    # Quality thresholds
    min_silhouette_score: float = 0.3
    max_noise_ratio: float = 0.5
```

### Phase 2.2: Quality Metrics
```python
class ClusterQualityAnalyzer:
    def calculate_silhouette_score(self, embeddings, labels):
        """Calculate silhouette score for cluster quality"""
    
    def calculate_davies_bouldin_score(self, embeddings, labels):
        """Calculate Davies-Bouldin score"""
    
    def calculate_calinski_harabasz_score(self, embeddings, labels):
        """Calculate Calinski-Harabasz score"""
    
    def analyze_cluster_stability(self, embeddings, labels):
        """Bootstrap analysis for cluster stability"""
    
    def get_optimal_parameters(self, embeddings):
        """Grid search for optimal parameters"""
```

### Phase 2.3: Alternative Algorithms
```python
class ClusteringFactory:
    @staticmethod
    def create_reducer(config: ClusteringConfig):
        if config.reducer_type == "umap":
            return UMAP(...)
        elif config.reducer_type == "tsne":
            return TSNE(...)
        elif config.reducer_type == "pca":
            return PCA(...)
    
    @staticmethod
    def create_clusterer(config: ClusteringConfig):
        if config.clusterer_type == "hdbscan":
            return HDBSCAN(...)
        elif config.clusterer_type == "dbscan":
            return DBSCAN(...)
        elif config.clusterer_type == "kmeans":
            return KMeans(...)
```

### Phase 2.4: Outlier Handling
```python
class OutlierHandler:
    def reassign_outliers(self, embeddings, labels, strategy="nearest"):
        """Reassign outliers to nearest clusters"""
    
    def create_micro_clusters(self, outlier_embeddings):
        """Create small clusters from outliers"""
    
    def merge_small_clusters(self, labels, min_size=3):
        """Merge clusters below threshold"""
```

### Phase 2.5: Visualization
```python
class ClusterVisualizer:
    def plot_2d_clusters(self, embeddings, labels):
        """2D scatter plot with Plotly"""
    
    def plot_3d_clusters(self, embeddings, labels):
        """3D interactive plot"""
    
    def create_dendrogram(self, linkage_matrix):
        """Hierarchical clustering dendrogram"""
    
    def plot_quality_metrics(self, metrics_history):
        """Plot quality metrics over parameters"""
```

### Phase 2.6: Integration
1. Update pipeline.py to use ClusteringConfig
2. Add command-line arguments for key parameters
3. Save quality metrics to cache database
4. Generate HTML report with visualizations

## Implementation Priority

1. **High Priority** (Week 1):
   - Create ClusteringConfig class
   - Add basic quality metrics
   - Make min_cluster_size configurable

2. **Medium Priority** (Week 2):
   - Implement algorithm alternatives
   - Add outlier reassignment
   - Basic visualization

3. **Low Priority** (Week 3):
   - Advanced parameter optimization
   - Comprehensive reporting
   - Interactive visualizations

## Success Metrics
- ✓ Silhouette score > 0.4
- ✓ Noise ratio < 30%
- ✓ Configurable parameters via CLI
- ✓ At least 2 clustering algorithms
- ✓ Basic cluster visualization
- ✓ Quality metrics in cache database