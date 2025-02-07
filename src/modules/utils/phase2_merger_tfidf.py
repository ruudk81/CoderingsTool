import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix
import hdbscan
from typing import List, Dict, Set, Tuple
import logging

try:
    # When running as a script
    from labeller import (
        LabellerConfig, ClusterData, InitialLabel, MergeMapping
    )
except ImportError:
    # When imported as a module
    from .labeller import (
        LabellerConfig, ClusterData, InitialLabel, MergeMapping
    )

logger = logging.getLogger(__name__)


class Phase2Merger:
    """Phase 2: Merge clusters using TF-IDF and embeddings"""
    
    def __init__(self, config: LabellerConfig, client=None):
        self.config = config
        self.client = client  # Not needed for TF-IDF approach
        self.verbose = True
        
        # Initialize TF-IDF vectorizer
        self.vectorizer_model = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words=None,
            min_df=1,
            max_df=1.0
        )
    
    async def merge_similar_clusters(self,
                                   cluster_data: Dict[int, ClusterData],
                                   initial_labels: Dict[int, InitialLabel],
                                   var_lab: str) -> MergeMapping:
        """Main method to identify and merge similar clusters using TF-IDF"""
        logger.info("Phase 2: Analyzing cluster similarity using TF-IDF approach...")
        
        # Since this is not async, we can call the sync version directly
        return self._merge_clusters_tfidf(cluster_data, initial_labels, var_lab)
    
    def _merge_clusters_tfidf(self,
                             cluster_data: Dict[int, ClusterData],
                             initial_labels: Dict[int, InitialLabel],
                             var_lab: str) -> MergeMapping:
        """Merge clusters using TF-IDF and weighted embeddings"""
        
        # Step 1: Prepare cluster text data
        cluster_ids = sorted(cluster_data.keys())
        cluster_texts = []
        
        for cluster_id in cluster_ids:
            cluster = cluster_data[cluster_id]
            # Combine descriptive codes and descriptions for each cluster
            cluster_text = " ".join(cluster.descriptive_codes + cluster.code_descriptions)
            cluster_texts.append(cluster_text)
        
        logger.info(f"Processing {len(cluster_ids)} clusters with TF-IDF")
        
        # Step 2: Calculate TF-IDF
        X = self.vectorizer_model.fit_transform(cluster_texts)
        words = self.vectorizer_model.get_feature_names_out()
        tf = X.toarray()
        tf = np.divide(tf, tf.sum(axis=1).reshape(-1, 1) + 1e-10)  # Avoid division by zero
        
        # Calculate IDF
        df = np.where(X.toarray() > 0, 1, 0).sum(axis=0)
        n_samples = X.shape[0]
        idf = np.log((n_samples + 1) / (df + 1)) + 1
        c_tf_idf = csr_matrix(np.multiply(tf, idf))
        
        # Step 3: Extract keywords for each cluster
        nr_candidate_words = 20
        clusters_dict = self._extract_keywords(c_tf_idf, words, nr_candidate_words)
        
        if self.verbose:
            logger.info(f"Generated keywords for {len(clusters_dict)} clusters")
            for cluster_idx in list(clusters_dict.keys())[:3]:
                cluster_id = cluster_ids[cluster_idx]
                keywords = clusters_dict[cluster_idx][:5]
                logger.info(f"Cluster {cluster_id} keywords: {', '.join(keywords)}")
        
        # Step 4: Weight keywords using embeddings
        weighted_clusters_dict = self._weight_keywords_with_embeddings(
            clusters_dict, cluster_texts, c_tf_idf, words
        )
        
        # Step 5: Apply weights to TF-IDF and create meta-clusters
        weighted_c_tf_idf = self._apply_weights_to_tfidf(
            tf, idf, weighted_clusters_dict, words
        )
        
        # Normalize for clustering
        normalized_vectors = normalize(weighted_c_tf_idf.toarray())
        
        # Step 6: Use HDBSCAN to find groups of similar clusters
        meta_clusterer = hdbscan.HDBSCAN(
            min_cluster_size=2,
            metric='cosine',  # Use cosine similarity
            cluster_selection_method='eom'
        )
        
        merge_labels = meta_clusterer.fit_predict(normalized_vectors)
        
        # Step 7: Create merge groups from HDBSCAN results
        merge_groups = self._create_merge_groups(cluster_ids, merge_labels)
        
        # Step 8: Create merge mapping
        merge_mapping = self._create_merge_mapping(merge_groups, initial_labels)
        
        logger.info(f"TF-IDF merging: {len(cluster_ids)} → {len(merge_groups)} clusters")
        
        return merge_mapping
    
    def _extract_keywords(self, c_tf_idf, words, n):
        """Extract top keywords for each cluster"""
        matrix = c_tf_idf
        
        # Find top n words
        indices = []
        for le, ri in zip(matrix.indptr[:-1], matrix.indptr[1:]):
            n_row_pick = min(n, ri - le)
            if n_row_pick > 0:
                row_indices = matrix.indices[le + np.argpartition(matrix.data[le:ri], -n_row_pick)[-n_row_pick:]]
                values = list(row_indices) + [None] * (n - len(row_indices))
                indices.append(values)
            else:
                indices.append([None] * n)
        indices = np.array(indices)
        
        top_values = []
        for row, idx_row in enumerate(indices):
            values = np.array([matrix[row, idx] if idx is not None else 0 for idx in idx_row])
            top_values.append(values)
        scores = np.array(top_values)
        
        # Sort indices and scores
        sorted_indices = np.argsort(scores, 1)
        indices = np.take_along_axis(indices, sorted_indices, axis=1)
        scores = np.take_along_axis(scores, sorted_indices, axis=1)
        
        # Create clusters_dict with top keywords
        clusters_dict = {}
        for cluster_idx in range(c_tf_idf.shape[0]):
            cluster_words = [
                words[word_index] if word_index is not None and score > 0 else ""
                for word_index, score in zip(indices[cluster_idx][::-1], scores[cluster_idx][::-1])
            ]
            cluster_words = [word for word in cluster_words if word][:n]
            clusters_dict[cluster_idx] = cluster_words
        
        return clusters_dict
    
    def _weight_keywords_with_embeddings(self, clusters_dict, cluster_texts, c_tf_idf, words):
        """Weight keywords using embeddings"""
        # Get embeddings
        try:
            from modules.utils import embedder
            get_embedding = embedder.Embedder()
        except ImportError:
            # Fallback for testing
            logger.warning("Could not import embedder, using random embeddings for testing")
            class MockEmbedder:
                def embed_words(self, texts):
                    return np.random.rand(len(texts), 768)
            get_embedding = MockEmbedder()
        
        # Get embeddings for cluster texts
        cluster_embeddings = get_embedding.embed_words(cluster_texts)
        
        # Get unique words from all clusters
        vocab = list(set([word for words in clusters_dict.values() for word in words]))
        word_embeddings = get_embedding.embed_words(vocab)
        
        # Calculate similarity between clusters and words
        sim = cosine_similarity(cluster_embeddings, word_embeddings)
        
        # Create weighted keywords dictionary
        updated_clusters_dict = {}
        
        for i, cluster_idx in enumerate(sorted(clusters_dict.keys())):
            indices = [vocab.index(word) for word in clusters_dict[cluster_idx] if word in vocab]
            if indices:
                values = sim[i, indices]
                sorted_indices = np.argsort(values)
                word_indices = [indices[idx] for idx in sorted_indices]
                scores = np.sort(values)
                updated_clusters_dict[cluster_idx] = [
                    (vocab[idx], score) for score, idx in 
                    zip(scores[::-1], word_indices[::-1])
                ]
            else:
                updated_clusters_dict[cluster_idx] = []
        
        return updated_clusters_dict
    
    def _apply_weights_to_tfidf(self, tf, idf, weighted_clusters_dict, words):
        """Apply weights to TF-IDF matrix"""
        weights = np.ones_like(tf)
        
        for cluster_idx, keyword_weights in weighted_clusters_dict.items():
            for keyword, similarity in keyword_weights:
                if keyword in words:
                    word_idx = np.where(words == keyword)[0][0]
                    weights[cluster_idx, word_idx] = similarity
        
        tf_weighted = tf * weights
        weighted_c_tf_idf = csr_matrix(np.multiply(tf_weighted, idf))
        
        return weighted_c_tf_idf
    
    def _create_merge_groups(self, cluster_ids, merge_labels):
        """Create merge groups from HDBSCAN labels"""
        # Group clusters by their merge label
        label_to_clusters = {}
        
        for idx, label in enumerate(merge_labels):
            cluster_id = cluster_ids[idx]
            
            if label == -1:
                # Noise points remain unmerged (single-item groups)
                label_to_clusters[f"single_{cluster_id}"] = [cluster_id]
            else:
                if label not in label_to_clusters:
                    label_to_clusters[label] = []
                label_to_clusters[label].append(cluster_id)
        
        # Convert to list of groups
        merge_groups = list(label_to_clusters.values())
        
        return merge_groups
    
    def _create_merge_mapping(self, merge_groups, initial_labels):
        """Create the final merge mapping"""
        # Create cluster to merged mapping
        cluster_to_merged = {}
        merge_reasons = {}
        
        for merged_id, group in enumerate(merge_groups):
            for cluster_id in group:
                cluster_to_merged[cluster_id] = merged_id
            
            if len(group) > 1:
                # Get labels for merged clusters
                labels = [initial_labels[cid].label for cid in group if cid in initial_labels]
                merge_reasons[merged_id] = f"TF-IDF similarity: {', '.join(labels[:3])}"
        
        return MergeMapping(
            merge_groups=merge_groups,
            cluster_to_merged=cluster_to_merged,
            merge_reasons=merge_reasons
        )


# Test section
if __name__ == "__main__":
    import asyncio
    import json
    from pathlib import Path
    import sys
    sys.path.insert(0, r'C:\Users\rkn\Python_apps\Coderingstool\src')
    from cache_config import CacheConfig
    from cache_manager import CacheManager
    from config import OPENAI_API_KEY, DEFAULT_MODEL
    import models
    import data_io
    
    # Load test data
    cache_config = CacheConfig()
    cache_manager = CacheManager(cache_config)
    
    filename = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
    var_name = "Q20"
    
    # Load clusters from cache
    cluster_results = cache_manager.load_from_cache(filename, "clusters", models.ClusterModel)
    phase1_labels = cache_manager.load_intermediate_data(filename, "phase1_labels")
    
    if cluster_results and phase1_labels:
        print(f"Loaded {len(cluster_results)} cluster results from cache")
        print(f"Loaded {len(phase1_labels)} phase 1 labels from cache")
        
        # Extract cluster data
        try:
            # Import Labeller to use its extract method
            from labeller import Labeller
            temp_labeller = Labeller()
            cluster_data = temp_labeller.extract_cluster_data(cluster_results)
            print(f"Extracted data for {len(cluster_data)} clusters")
        except Exception as e:
            print(f"Error extracting cluster data: {e}")
            raise
        
        # Convert labels if needed
        initial_labels = {}
        for cluster_id, label_data in phase1_labels.items():
            if hasattr(label_data, 'label'):
                initial_labels[cluster_id] = label_data
            elif isinstance(label_data, dict):
                from labeller import InitialLabel
                initial_labels[cluster_id] = InitialLabel(**label_data)
            else:
                initial_labels[cluster_id] = label_data
        
        # Get variable label
        data_loader = data_io.DataLoader()
        var_lab = data_loader.get_varlab(filename=filename, var_name=var_name)
        print(f"Variable label: {var_lab}")
        
        # Initialize configuration
        config = LabellerConfig(
            api_key=OPENAI_API_KEY,
            model=DEFAULT_MODEL,
            batch_size=10
        )
        
        # Initialize phase 2 merger (no client needed for TF-IDF)
        phase2 = Phase2Merger(config)
        
        async def run_test():
            """Run the test"""
            print("\n=== Testing Phase 2: TF-IDF Cluster Merging ===")
            print(f"Variable label: {var_lab}")
            print(f"Number of clusters: {len(cluster_data)}")
            
            try:
                # Run TF-IDF merge process
                merge_mapping = await phase2.merge_similar_clusters(
                    cluster_data, initial_labels, var_lab
                )
                
                # Display results
                print("\n=== Merge Results ===")
                print(f"Number of merge groups: {len(merge_mapping.merge_groups)}")
                
                # Show merged groups
                print("\nMerged groups (showing groups with >1 cluster):")
                for i, group in enumerate(merge_mapping.merge_groups):
                    if len(group) > 1:
                        labels = [initial_labels[cid].label for cid in group if cid in initial_labels]
                        print(f"  Group {i}: Clusters {group}")
                        print(f"    Labels: {labels}")
                
                # Calculate statistics
                original_count = len(cluster_data)
                merged_count = len(merge_mapping.merge_groups)
                reduction = (1 - merged_count/original_count) * 100
                
                print(f"\nStatistics:")
                print(f"  Original clusters: {original_count}")
                print(f"  Merged clusters: {merged_count}")
                print(f"  Reduction: {reduction:.1f}%")
                
                # Save to cache
                cache_key = 'phase2_merge_mapping_tfidf'
                cache_data = {
                    'merge_mapping': merge_mapping,
                    'cluster_data': cluster_data,
                    'initial_labels': initial_labels
                }
                cache_manager.cache_intermediate_data(cache_data, filename, cache_key)
                print(f"\nSaved results to cache with key '{cache_key}'")
                
                # Save to JSON
                output_data = {
                    "merge_groups": merge_mapping.merge_groups,
                    "cluster_mapping": merge_mapping.cluster_to_merged,
                    "merge_reasons": merge_mapping.merge_reasons,
                    "statistics": {
                        "original": original_count,
                        "merged": merged_count,
                        "reduction_percent": reduction
                    }
                }
                
                output_file = Path("phase2_tfidf_results.json")
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
        print("Missing required cached data.")
        print("Please ensure you have run:")
        print("  1. python clusterer.py")
        print("  2. python phase1_labeller.py")