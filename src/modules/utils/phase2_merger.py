import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix
import hdbscan
from typing import List, Dict, Set, Tuple
import logging
import spacy

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
        
        # Initialize vectorizer with specific parameters
        stop_words = self._get_stop_words()
        self.vectorizer_model = CountVectorizer(
            stop_words=stop_words,
            ngram_range=(1, 2),  # Use smaller ngrams for more specific matching
            min_df=1,
            max_df=0.8,  # Ignore terms that appear in more than 80% of clusters
            token_pattern=r'\b\w+\b'  # Better token pattern
        )
    
    def _get_stop_words(self):
        """Get stop words based on language"""
        if self.config.language == "nl":
            try:
                return list(spacy.load("nl_core_news_lg").Defaults.stop_words)
            except:
                logger.warning("Dutch language model not found. Using None.")
                return None
        else:
            return 'english'
    
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
            # Combine descriptive codes and descriptions, filtering out NAs
            codes = [code.replace("_", " ").lower() for code in cluster.descriptive_codes if code.lower() != "na"]
            descriptions = [desc.lower() for desc in cluster.code_descriptions if desc.lower() != "na"]
            
            # Give more weight to codes by repeating them
            weighted_codes = codes * 2  # Repeat codes to give them more weight
            cluster_text = " ".join(weighted_codes + descriptions)
            cluster_texts.append(cluster_text)
        
        logger.info(f"Processing {len(cluster_ids)} clusters with TF-IDF")
        
        # Step 2: Calculate TF-IDF
        X = self.vectorizer_model.fit_transform(cluster_texts)
        words = self.vectorizer_model.get_feature_names_out()
        tf = X.toarray()
        
        # Normalize TF (avoid division by zero)
        tf_sums = tf.sum(axis=1).reshape(-1, 1)
        tf_sums[tf_sums == 0] = 1  # Avoid division by zero
        tf = np.divide(tf, tf_sums)
        
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
            clusters_dict, cluster_texts, c_tf_idf, words, cluster_data, cluster_ids
        )
        
        # Step 5: Apply weights to TF-IDF and create meta-clusters
        weighted_c_tf_idf = self._apply_weights_to_tfidf(
            tf, idf, weighted_clusters_dict, words
        )
        
        # Normalize for clustering (L2 normalization makes euclidean distance equivalent to cosine)
        normalized_vectors = normalize(weighted_c_tf_idf.toarray(), norm='l2', axis=1)
        
        # Step 6: Use HDBSCAN to find groups of similar clusters
        # Use more conservative parameters to prevent over-merging
        meta_clusterer = hdbscan.HDBSCAN(
            min_cluster_size=4,  # Increased to be less aggressive
            min_samples=3,  # Require more samples for core points
            metric='euclidean',  # Use euclidean (cosine not supported by BallTree)
            cluster_selection_method='eom',
            cluster_selection_epsilon=0.3,  # Tighter epsilon for less merging
            algorithm='best'  # Let HDBSCAN choose the best algorithm
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
    
    def _weight_keywords_with_embeddings(self, clusters_dict, cluster_texts, c_tf_idf, words, 
                                        cluster_data, cluster_ids):
        """Weight keywords using embeddings - following working code logic"""
        # Get embeddings using the proper import pattern
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
        
        # Step 1: Get representative responses for each cluster
        nr_repr_docs = 10
        nr_candidate_words = 20
        representative_responses = []
        repr_resp_indices = []
        start_idx = 0
        
        # Create a DataFrame-like structure for easier processing
        cluster_responses_dict = {}
        for cluster_idx, cluster_id in enumerate(cluster_ids):
            cluster = cluster_data[cluster_id]
            # Get actual responses from the cluster
            responses = []
            for code, desc in zip(cluster.descriptive_codes, cluster.code_descriptions):
                if code.lower() != "na" and desc.lower() != "na":
                    responses.append(f"{code.replace('_', ' ').lower()} {desc}")
            cluster_responses_dict[cluster_idx] = responses
        
        # Get representative responses following the working code logic
        for cluster_idx in sorted(clusters_dict.keys()):
            cluster_responses = cluster_responses_dict.get(cluster_idx, [])
            
            if len(cluster_responses) > nr_candidate_words:
                sampled_indices = np.random.choice(range(len(cluster_responses)), 
                                                  size=nr_candidate_words, 
                                                  replace=False)
                cluster_responses = [cluster_responses[i] for i in sampled_indices]
            
            if len(cluster_responses) > nr_repr_docs:
                resp_vectors = self.vectorizer_model.transform(cluster_responses)
                # Use direct indexing as in the working code
                cluster_vector = c_tf_idf[cluster_idx]
                similarities = cosine_similarity(resp_vectors, cluster_vector)
                # Note: The working code uses argsort()[nr_repr_docs:] to get top items
                top_indices = np.argsort(similarities.flatten())[-nr_repr_docs:]
                selected_responses = [cluster_responses[idx] for idx in top_indices]
            else:
                selected_responses = cluster_responses
            
            representative_responses.extend(selected_responses)
            repr_resp_indices.append(list(range(start_idx, start_idx + len(selected_responses))))
            start_idx += len(selected_responses)
        
        # Get embeddings for representative responses
        if representative_responses:
            repr_embeddings = get_embedding.embed_words(representative_responses)
            temp_cluster_embeddings = []
            for indices in repr_resp_indices:
                if indices:  # Check if indices is not empty
                    embeddings_slice = repr_embeddings[indices[0]:indices[-1] + 1]
                    temp_cluster_embeddings.append(np.mean(embeddings_slice, axis=0))
                else:
                    # If no representative responses, use a zero vector
                    temp_cluster_embeddings.append(np.zeros(768))
            temp_cluster_embeddings = np.array(temp_cluster_embeddings)
        else:
            # Fallback if no representative responses
            temp_cluster_embeddings = np.zeros((len(clusters_dict), 768))
        
        # Get unique words from all clusters
        vocab = list(set([word for words in clusters_dict.values() for word in words]))
        if vocab:
            word_embeddings = get_embedding.embed_words(vocab)
            
            # Calculate similarity between clusters and words
            sim = cosine_similarity(temp_cluster_embeddings, word_embeddings)
            
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
        else:
            updated_clusters_dict = {idx: [] for idx in clusters_dict.keys()}
        
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
                cache_key = 'phase2_merge_mapping'
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