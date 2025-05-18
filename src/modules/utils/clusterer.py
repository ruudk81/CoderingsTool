# data
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from typing import List, Any, Optional
import numpy.typing as npt
from collections import defaultdict, Counter

# cluster
from umap import UMAP
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import hdbscan

# semantics
import spacy 
from sklearn.feature_extraction.text import CountVectorizer 
from scipy.sparse import csr_matrix

# config
import models
import warnings  # hard coded warning in umap about hidden stat
warnings.filterwarnings("ignore", message="n_jobs value.*overridden to 1 by setting random_state")
from config import DEFAULT_LANGUAGE

# clustering improvements
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))  # Add src directory

# Try both import methods to handle different environments
try:
    from clustering_config import ClusteringConfig
    from cluster_quality import ClusterQualityAnalyzer
except ImportError:
    from .clustering_config import ClusteringConfig
    from .cluster_quality import ClusterQualityAnalyzer


# Structured formats 
class ResultMapper(BaseModel):
    # input
    respondent_id: Any
    segment_id: str
    descriptive_code: str
    code_description: str
    code_embedding: npt.NDArray[np.float32]
    description_embedding: npt.NDArray[np.float32]
    # add
    reduced_code_embedding: Optional[npt.NDArray[np.float32]] = None
    reduced_description_embedding: Optional[npt.NDArray[np.float32]] = None
    initial_code_cluster: Optional[int] = None
    initial_description_cluster: Optional[int] = None
    updated_code_cluster: Optional[int] = None
    updated_description_cluster: Optional[int] = None
    meta_cluster: Optional[int] = None
    # config
    model_config = ConfigDict(arbitrary_types_allowed=True)  # for arrays with embeddings


# Main utils 
class ClusterGenerator:
    def __init__(
        self,
        input_list: List[models.EmbeddingsModel] = None,
        var_lab=None,
        dim_reduction_model=None,
        cluster_model=None,
        vectorizer_model=None,
        embedding_type: str = "code",  # Can be "code" OR "description" 
        config: ClusteringConfig = None,  # NEW: optional config
        verbose: bool = True):
        
        self.var_lab = var_lab if var_lab else ""
        self.embedding_type = embedding_type
        self.output_list: List[ResultMapper] = []
        self.verbose = verbose
        # Store the original input list to preserve response data
        self.original_input_list = input_list if input_list else []
        
        # Store config and quality metrics
        self.config = config or ClusteringConfig()
        self.quality_analyzer = None  # Will be initialized after clustering
        self.clustering_attempts = []  # Track attempts for reporting
        
        # Override embedding_type from config if provided
        if self.config and self.config.embedding_type:
            self.embedding_type = self.config.embedding_type
       
        if input_list:
            self.populate_from_input_list(input_list)
        
        # Initialize dimensionality reduction model
        if dim_reduction_model is None:
            # Use config for parameters if available
            data_size = len(self.output_list) if self.output_list else 1000
            reducer_params = self.config.get_reducer_params(data_size)
            self.dim_reduction_model = UMAP(**reducer_params)
            if self.verbose:
                print("Using default dimensionality reduction: UMAP")
                print(f"Parameters: {reducer_params}")
        else:
            self.dim_reduction_model = dim_reduction_model
            
        # Initialize clustering model
        if cluster_model is None:
            # Use config for parameters
            data_size = len(self.output_list) if self.output_list else 1000
            cluster_params = self.config.get_auto_params(data_size)
            self.cluster_model = hdbscan.HDBSCAN(**cluster_params)
            if self.verbose:
                print("Using default clustering: HDBSCAN")
                print(f"Parameters: {cluster_params}")
        else:
            self.cluster_model = cluster_model
        
        # Initialize vectorizer model
        if vectorizer_model is None:
            stop_words = self._get_stop_words()
            self.vectorizer_model = CountVectorizer(
                stop_words=stop_words,
                ngram_range=(1, 3),
                min_df=1)    
            if self.verbose:
                print("Using default vectorizing: SCIKIT-LEARN")
        else:
            self.vectorizer_model = vectorizer_model

    def _get_stop_words(self):
        if DEFAULT_LANGUAGE == "Dutch":
            try:
                return list(spacy.load("nl_core_news_lg").Defaults.stop_words)
            except:
                print("Warning: Dutch language model not found. Using English stop words.")
                return 'english'
        else:
            return 'english'

    def populate_from_input_list(self, input_list: List[models.EmbeddingsModel]) -> None:
        if self.verbose:
            print("Populating output list from input models...")
        
        self.output_list = []
        
        for response_item in input_list:
            if response_item.response_segment:
                for segment_item in response_item.response_segment:
                    # Check if segment_item has the required attributes
                    if (hasattr(segment_item, 'descriptive_code') and 
                        hasattr(segment_item, 'code_description') and
                        hasattr(segment_item, 'code_embedding') and
                        hasattr(segment_item, 'description_embedding')):
                        
                        self.output_list.append(ResultMapper(
                            respondent_id=response_item.respondent_id,
                            segment_id=segment_item.segment_id,
                            descriptive_code=segment_item.descriptive_code or "NA",
                            code_description=segment_item.code_description or "NA",
                            code_embedding=segment_item.code_embedding,
                            description_embedding=segment_item.description_embedding
                        ))
        
        # if self.verbose:
        #     print(f"Populated {len(self.output_list)} items in output list")
        #     # Debug print a few items
        #     for item in self.output_list[:2]:
        #         print(f"Respondent ID: {item.respondent_id}")
        #         print(f"Descriptive code: {item.descriptive_code}")
        #         print(f"Code description: {item.code_description}")

    def add_reduced_embeddings(self) -> None:
        # Reduces dimensionality of embeddings and adds them to output_list.
        if self.verbose:
            print("Reducing dimensionality of embeddings...")
        
        # Process code embeddings if needed
        if self.embedding_type == "code":
            code_embeddings_array = np.array([item.code_embedding for item in self.output_list])
            reduced_code_embeddings = self.dim_reduction_model.fit_transform(code_embeddings_array)
            
            # Add reduced embeddings to output list
            for i, item in enumerate(self.output_list):
                item.reduced_code_embedding = reduced_code_embeddings[i]
        
        # Process description embeddings if needed
        if self.embedding_type == "description":
            description_embeddings_array = np.array([item.description_embedding for item in self.output_list])
            reduced_description_embeddings = self.dim_reduction_model.fit_transform(description_embeddings_array)
            
            # Add reduced embeddings to output list
            for i, item in enumerate(self.output_list):
                item.reduced_description_embedding = reduced_description_embeddings[i]
                
        if self.verbose:
            print("Completed dimensionality reduction")

    def add_initial_clusters(self) -> None:
        # Performs initial clustering on reduced embeddings and adds cluster labels to output_list.
        if self.verbose:
            print("Clustering reduced embeddings...")
        
        max_attempts = 3
        current_attempt = 1

        # Cluster code embeddings if needed
        if self.embedding_type == "code":
            reduced_code_embeddings = np.array([item.reduced_code_embedding for item in self.output_list])
            
            while current_attempt <= max_attempts:
                if self.verbose and current_attempt > 1:
                    print(f"\nAttempt {current_attempt}/{max_attempts}...")
                
                initial_code_clusters = self.cluster_model.fit_predict(reduced_code_embeddings)
                
                # Calculate quality metrics
                metrics = self.calculate_clustering_quality(reduced_code_embeddings, initial_code_clusters)
                
                if self.verbose:
                    print(f"Quality score: {metrics['quality_score']:.3f}")
                
                # Check if quality is acceptable
                if metrics['quality_score'] >= self.config.min_quality_score:
                    break
                
                # Try with adjusted parameters if quality is low
                if current_attempt < max_attempts:
                    current_min_cluster_size = self.cluster_model.min_cluster_size
                    new_min_cluster_size = max(2, current_min_cluster_size - 2)
                    if self.verbose:
                        print(f"Adjusting min_cluster_size from {current_min_cluster_size} to {new_min_cluster_size}")
                    
                    # Create new clusterer with adjusted parameters
                    cluster_params = self.config.get_auto_params(len(self.output_list))
                    cluster_params['min_cluster_size'] = new_min_cluster_size
                    self.cluster_model = hdbscan.HDBSCAN(**cluster_params)
                
                current_attempt += 1
            
            # Check if we need micro-clusters for outliers
            if metrics['noise_ratio'] > 0.09:
                initial_code_clusters = self._create_micro_clusters(reduced_code_embeddings, initial_code_clusters)
                # Recalculate metrics after micro-clustering
                metrics = self.calculate_clustering_quality(reduced_code_embeddings, initial_code_clusters)
            
            # Add cluster labels to output list
            for i, item in enumerate(self.output_list):
                item.initial_code_cluster = initial_code_clusters[i]
            
            if self.verbose:
                cluster_counts = Counter(initial_code_clusters)
                print(f"Code clusters: {len(set(initial_code_clusters))} clusters found")
                for cluster_id, count in sorted(cluster_counts.items()):
                    print(f"Code Cluster {cluster_id}: {count} items")
                
                # Show a sample of codes per cluster
                self._print_sample_items_per_cluster(initial_code_clusters, "descriptive_code", "Code")
                
        # Cluster description embeddings if needed
        if self.embedding_type == "description":
            reduced_description_embeddings = np.array([item.reduced_description_embedding for item in self.output_list])
            current_attempt = 1  # Reset for description clustering
            
            while current_attempt <= max_attempts:
                if self.verbose and current_attempt > 1:
                    print(f"\nAttempt {current_attempt}/{max_attempts}...")
                
                initial_description_clusters = self.cluster_model.fit_predict(reduced_description_embeddings)
                
                # Calculate quality metrics
                metrics = self.calculate_clustering_quality(reduced_description_embeddings, initial_description_clusters)
                
                if self.verbose:
                    print(f"Quality score: {metrics['quality_score']:.3f}")
                
                # Check if quality is acceptable
                if metrics['quality_score'] >= self.config.min_quality_score:
                    break
                
                # Try with adjusted parameters if quality is low
                if current_attempt < max_attempts:
                    current_min_cluster_size = self.cluster_model.min_cluster_size
                    new_min_cluster_size = max(2, current_min_cluster_size - 2)
                    if self.verbose:
                        print(f"Adjusting min_cluster_size from {current_min_cluster_size} to {new_min_cluster_size}")
                    
                    # Create new clusterer with adjusted parameters
                    cluster_params = self.config.get_auto_params(len(self.output_list))
                    cluster_params['min_cluster_size'] = new_min_cluster_size
                    self.cluster_model = hdbscan.HDBSCAN(**cluster_params)
                
                current_attempt += 1
            
            # Check if we need micro-clusters for outliers
            if metrics['noise_ratio'] > 0.09:
                initial_description_clusters = self._create_micro_clusters(reduced_description_embeddings, initial_description_clusters)
                # Recalculate metrics after micro-clustering
                metrics = self.calculate_clustering_quality(reduced_description_embeddings, initial_description_clusters)
            
            # Add cluster labels to output list
            for i, item in enumerate(self.output_list):
                item.initial_description_cluster = initial_description_clusters[i]
            
            if self.verbose:
                cluster_counts = Counter(initial_description_clusters)
                print(f"Description clusters: {len(set(initial_description_clusters))} clusters found")
                for cluster_id, count in sorted(cluster_counts.items()):
                    print(f"Description Cluster {cluster_id}: {count} items")
                
                # Show a sample of descriptions per cluster
                self._print_sample_items_per_cluster(initial_description_clusters, "code_description", "Description")

    def _print_sample_items_per_cluster(self, clusters, attribute, label_prefix, sample_size=5):
        # Helper method to print samples from each cluster
        cluster_to_items = defaultdict(list)
        for cluster, item in zip(clusters, self.output_list):
            value = getattr(item, attribute)
            if value and value.lower() != "na":
                cluster_to_items[cluster].append(value)
                
        for cluster in sorted(cluster_to_items.keys()):
            print(f"{label_prefix} Cluster {cluster}:")
            for item_text in cluster_to_items[cluster][:sample_size]:
                print(f" - {item_text}")
                
    def add_refined_clusters(self, embedding_type: str = "code") -> None:
        # refines clustering by reducing the nr of clusters
        if self.verbose:
            print(f"Refining {embedding_type} clusters...")
            
        # Step 1: Prepare the data
        respondent_ids = [item.respondent_id for item in self.output_list]
        segment_ids = [item.segment_id for item in self.output_list]
        
        if embedding_type == "code":
            items = [item.descriptive_code.replace("_", " ").lower() for item in self.output_list]
            clusters = [item.initial_code_cluster for item in self.output_list]
        elif embedding_type == "description":
            items = [item.code_description for item in self.output_list]
            clusters = [item.initial_description_cluster for item in self.output_list]
        else:
            raise ValueError("embedding_type must be 'code' or 'description'")
        
        # Filter out outliers and invalid items
        filtered_respondent_ids = []
        filtered_segment_ids = []
        filtered_items = []
        filtered_clusters = []
        
        for r_id, s_id, cluster, item in zip(respondent_ids, segment_ids, clusters, items):
            if cluster != -1 and item.lower() != "na":
                filtered_respondent_ids.append(r_id)
                filtered_segment_ids.append(s_id)
                filtered_items.append(item)
                filtered_clusters.append(cluster)
        
        # Remap cluster IDs to be sequential
        unique_clusters = sorted(set(filtered_clusters))
        mapping = {original: new for new, original in enumerate(unique_clusters)}
        remapped_clusters = [mapping[cluster] for cluster in filtered_clusters]
        
        # Update output_list with remapped clusters
        updated_clusters = {}
        for r_id, s_id, cluster in zip(filtered_respondent_ids, filtered_segment_ids, remapped_clusters):
            updated_clusters[(r_id, s_id)] = cluster
        
        for item in self.output_list:
            key = (item.respondent_id, item.segment_id)
            if embedding_type == "code":
                item.updated_code_cluster = updated_clusters.get(key, None)
            else:
                item.updated_description_cluster = updated_clusters.get(key, None)
        
        # Display updated clusters
        if self.verbose and embedding_type == "description":
            for item in self.output_list:
                print(f"{item.initial_description_cluster} : {item.updated_description_cluster}")
        
        # Step 2: Calculate keywords using TF-IDF
        nr_candidate_words = 20
        
        # Prepare data for generating keywords
        responses = filtered_items
        clusters = remapped_clusters
        
        cluster_df = pd.DataFrame({
            "cluster": clusters,
            "response": responses
        })
        
        # Concatenate all responses for each cluster
        responses_per_cluster = (cluster_df
            .groupby("cluster")["response"]
            .agg(lambda x: " ".join(x))
            .tolist())
        
        # Calculate c-TF-IDF
        X = self.vectorizer_model.fit_transform(responses_per_cluster)
        words = self.vectorizer_model.get_feature_names_out()
        tf = X.toarray()
        tf = np.divide(tf, tf.sum(axis=1).reshape(-1, 1))
        df = np.where(X.toarray() > 0, 1, 0).sum(axis=0)
        n_samples = X.shape[0]
        idf = np.log((n_samples + 1) / (df + 1)) + 1
        c_tf_idf = csr_matrix(np.multiply(tf, idf))
        matrix = c_tf_idf
        n = nr_candidate_words
        
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
            cluster_words = [word for word in cluster_words if word][:nr_candidate_words]
            clusters_dict[cluster_idx] = cluster_words
        
        if self.verbose:
            print(f"Generated keywords for {len(clusters_dict)} clusters")
            for cluster in list(clusters_dict.keys())[:2]:
                print(f"Cluster {cluster} keywords: {', '.join(clusters_dict[cluster][:5])}")
        
        # Step 3: Apply weights to keywords
        nr_repr_docs = 10
        representative_responses = []
        repr_resp_indices = []
        start_idx = 0
        
        # Get representative responses for each cluster
        for cluster in sorted(clusters_dict.keys()):
            cluster_responses = cluster_df[cluster_df.cluster == cluster].response.values
            
            if len(cluster_responses) > nr_candidate_words:
                sampled_indices = np.random.choice(range(len(cluster_responses)), 
                                                  size=nr_candidate_words, 
                                                  replace=False)
                cluster_responses = cluster_responses[sampled_indices]
            
            if len(cluster_responses) > nr_repr_docs:
                resp_vectors = self.vectorizer_model.transform(cluster_responses)
                # Use direct indexing as in the working code
                cluster_vector = c_tf_idf[cluster]
                similarities = cosine_similarity(resp_vectors, cluster_vector)
                top_indices = np.argsort(similarities.flatten())[nr_repr_docs:]
                selected_responses = [cluster_responses[idx] for idx in top_indices]
            else:
                selected_responses = cluster_responses
            
            representative_responses.extend(selected_responses)
            repr_resp_indices.append(list(range(start_idx, start_idx + len(selected_responses))))
            start_idx += len(selected_responses)
        
        # Get embeddings for responses and keywords
        from modules.utils import embedder
        get_embedding = embedder.Embedder()
        
        repr_embeddings = get_embedding.embed_words(representative_responses)
        temp_cluster_embeddings = [
            np.mean(repr_embeddings[indices[0]:indices[-1] + 1], axis=0)
            for indices in repr_resp_indices
        ]
        
        # Get unique words from all clusters
        vocab = list(set([word for words in clusters_dict.values() for word in words]))
        word_embeddings = get_embedding.embed_words(vocab)
        
        # Calculate similarity between clusters and words
        sim = cosine_similarity(temp_cluster_embeddings, word_embeddings)
        
        # Create weighted keywords dictionary
        #top_n_words = 10
        updated_clusters_dict = {}
        
        for i, cluster in enumerate(sorted(clusters_dict.keys())):
            indices = [vocab.index(word) for word in clusters_dict[cluster] if word in vocab]
            if indices:
                values = sim[i, indices]
                sorted_indices = np.argsort(values)
                word_indices = [indices[idx] for idx in sorted_indices]
                scores = np.sort(values)
                updated_clusters_dict[cluster] = [
                    (vocab[idx], score) for score, idx in 
                    zip(scores[::-1], word_indices[::-1])
                ]
            else:
                updated_clusters_dict[cluster] = []
        
        # Step 4: Create meta-clusters using weighted TF-IDF
        unique_clusters = sorted(updated_clusters_dict.keys())
        cluster_id_to_index = {cluster_id: i for i, cluster_id in enumerate(unique_clusters)}
        
        # Apply weights to TF-IDF
        weights = np.ones_like(tf)
        for cluster_id, keyword_weights in updated_clusters_dict.items():
            if cluster_id not in cluster_id_to_index:
                continue
            cluster_idx = cluster_id_to_index[cluster_id]
            for keyword, similarity in keyword_weights:
                if keyword in words:
                    word_idx = np.where(words == keyword)[0][0]
                    weights[cluster_idx, word_idx] = similarity
        
        tf_weighted = tf * weights
        weighted_c_tf_idf = csr_matrix(np.multiply(tf_weighted, idf))
        
        # Normalize for clustering
        normalized_vectors = normalize(weighted_c_tf_idf.toarray())
        
        # Create meta-clusters
        meta_clusterer = hdbscan.HDBSCAN(
            min_cluster_size=2,
            metric='euclidean',
            cluster_selection_method='eom')
        
        meta_cluster_labels = meta_clusterer.fit_predict(normalized_vectors)
        
        # Process meta-cluster results
        unique_clusters = sorted(set(clusters))
        non_noise_meta_labels = sorted(label for label in set(meta_cluster_labels) if label != -1)
        
        # Create mapping from cluster to meta-cluster
        cluster_to_meta = {}
        for idx, meta in enumerate(meta_cluster_labels):
            if meta != -1:
                cluster_to_meta[idx] = meta
        
        # Handle outliers
        next_id = max(non_noise_meta_labels) + 1 if non_noise_meta_labels else 0
        noise_clusters = [i for i, label in enumerate(meta_cluster_labels) if label == -1]
        
        for noise_cluster in noise_clusters:
            cluster_to_meta[noise_cluster] = next_id
            next_id += 1
        
        # Group clusters by meta-cluster
        meta_to_clusters = {}
        for original_cluster, meta_cluster in cluster_to_meta.items():
            if meta_cluster not in meta_to_clusters:
                meta_to_clusters[meta_cluster] = []
            meta_to_clusters[meta_cluster].append(original_cluster)
        
        # Update output_list with meta-clusters
        for item in self.output_list:
            if embedding_type == "code" and item.updated_code_cluster is not None:
                item.meta_cluster = cluster_to_meta.get(item.updated_code_cluster, None)
            elif embedding_type == "description" and item.updated_description_cluster is not None:
                item.meta_cluster = cluster_to_meta.get(item.updated_description_cluster, None)
        
        # Print hierarchy
        if self.verbose:
            for meta, micros in meta_to_clusters.items():
                if len(micros) > 1:
                    print(f"\n📚 Meta-cluster: {meta}")
                    for micro in micros:
                        print(f"  Cluster: {micro}")
                        if micro in updated_clusters_dict:
                            for word, score in updated_clusters_dict[micro][:5]:
                                print(f"    - {word}: {score:.2f}")
                        # Print the responses for this cluster
                        if micro < len(responses_per_cluster):
                            print(f"  {responses_per_cluster[micro]}")
                        print("\n")

    def run_pipeline(self, embedding_type: str = None) -> None:
        
        if embedding_type:
            self.embedding_type = embedding_type
            
        if self.verbose:
            print(f"\n🚀 Running clustering pipeline with embedding type: {self.embedding_type}")
            #print("=" * 80)
            
        if not self.output_list:
            raise ValueError("Output list is empty. Please populate it first.")
        
        # Step 1: Reduce dimensionality of embeddings
        if self.verbose:
            print("\n📊 STEP 1: Dimensionality Reduction")
            #print("-" * 80)
        self.add_reduced_embeddings()
        
        # Step 2: Perform initial clustering
        if self.verbose:
            print("\n🔍 STEP 2: Initial Clustering")
            #print("-" * 80)
        self.add_initial_clusters()
        
        # Step 3: Refine clusters and create meta-clusters
        if self.verbose:
            print("\n🧩 STEP 3: Cluster Refinement and Meta-Clustering")
            #print("-" * 80)
            
        if self.embedding_type == "code":
            if self.verbose:
                print("\n⚙️ Processing code embeddings:")
            self.add_refined_clusters(embedding_type="code")
            
        if self.embedding_type == "description":
            if self.verbose:
                print("\n📝 Processing description embeddings:")
            self.add_refined_clusters(embedding_type="description")
            
        if self.verbose:
            print("\n✅ Pipeline completed successfully")
            #print("=" * 80)
            
            # Print some summary statistics
            codes_with_clusters = sum(1 for item in self.output_list 
                                    if item.updated_code_cluster is not None)
            descriptions_with_clusters = sum(1 for item in self.output_list 
                                           if item.updated_description_cluster is not None)
            meta_clusters = len(set(item.meta_cluster for item in self.output_list 
                                   if item.meta_cluster is not None))
            
            print("\nSUMMARY:")
            print(f"- Total items processed: {len(self.output_list)}")
            if self.embedding_type == "code":
                print(f"- Code items clustered: {codes_with_clusters}")
            if self.embedding_type == "description":
                print(f"- Description items clustered: {descriptions_with_clusters}")
            print(f"- Meta-clusters created: {meta_clusters}")
            #print("=" * 80)
    
    def calculate_clustering_quality(self, embeddings: np.ndarray, labels: np.ndarray) -> Dict:
        """Calculate quality metrics for the clustering results"""
        # Initialize quality analyzer with embeddings and labels
        self.quality_analyzer = ClusterQualityAnalyzer(embeddings, labels)
        
        # Get full quality report
        metrics = self.quality_analyzer.get_full_report()
        
        # Calculate overall quality score
        metrics['overall_quality'] = self.quality_analyzer.calculate_quality_score(metrics)
        
        self.clustering_attempts.append(metrics)
        return metrics
    
    def _create_micro_clusters(self, embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Create micro-clusters from outliers if noise ratio is too high"""
        noise_mask = labels == -1
        noise_count = np.sum(noise_mask)
        
        if noise_count == 0:
            return labels
            
        if self.verbose:
            print(f"\nCreating micro-clusters from {noise_count} outliers...")
        
        # Get outlier embeddings
        outlier_embeddings = embeddings[noise_mask]
        
        # Use smaller min_cluster_size for outliers
        micro_clusterer = hdbscan.HDBSCAN(
            min_cluster_size=2,
            min_samples=1,
            metric='euclidean'
        )
        
        micro_labels = micro_clusterer.fit_predict(outlier_embeddings)
        
        # Assign new cluster IDs (continuing from existing max cluster ID)
        max_cluster_id = np.max(labels[labels != -1]) if np.any(labels != -1) else -1
        new_labels = labels.copy()
        
        outlier_indices = np.where(noise_mask)[0]
        for i, micro_label in enumerate(micro_labels):
            if micro_label != -1:
                new_labels[outlier_indices[i]] = max_cluster_id + 1 + micro_label
        
        new_noise_count = np.sum(new_labels == -1)
        if self.verbose:
            print(f"Reduced outliers from {noise_count} to {new_noise_count}")
            
        return new_labels
            
    def to_cluster_model(self) -> List[models.ClusterModel]:
       
        if not self.output_list:
            raise ValueError("Output list is empty. Nothing to convert.")
            
        # Group output items by respondent_id
        items_by_respondent = defaultdict(list)
        for item in self.output_list:
            items_by_respondent[item.respondent_id].append(item)
        
        # Create mapping of respondent_id to original response data
        response_mapping = {}
        segment_mapping = {}
        for original_item in self.original_input_list:
            response_mapping[original_item.respondent_id] = original_item.response
            if original_item.response_segment:
                for segment in original_item.response_segment:
                    segment_key = (original_item.respondent_id, segment.segment_id)
                    segment_mapping[segment_key] = segment.segment_response
        
        # Create ClusterModel instances
        result_models = []
        
        for respondent_id, items in items_by_respondent.items():
            # Get the original response from mapping
            response = response_mapping.get(respondent_id, "")
            
            # Create submodels for each segment
            submodels = []
            for item in items:
                # Get original segment response
                segment_key = (respondent_id, item.segment_id)
                segment_response = segment_mapping.get(segment_key, "")
                
                # Create cluster mappings
                meta_cluster = {item.meta_cluster: ""} if item.meta_cluster is not None else None
                
                # Fix for micro_cluster - only include non-None values and use empty string for values
                micro_cluster = {}
                if item.updated_code_cluster is not None and self.embedding_type == "code":
                    micro_cluster[item.updated_code_cluster] = ""
                if item.updated_description_cluster is not None and self.embedding_type == "description":
                    micro_cluster[item.updated_description_cluster] = ""
                
                # Use None if no valid clusters were added
                if not micro_cluster:
                    micro_cluster = None
                
                # Create submodel
                submodel = models.ClusterSubmodel(
                    segment_id=item.segment_id,
                    segment_response=segment_response,
                    descriptive_code=item.descriptive_code,
                    code_description=item.code_description,
                    code_embedding=item.code_embedding,
                    description_embedding=item.description_embedding,
                    meta_cluster=meta_cluster,
                    mirco_cluster=micro_cluster  # Note: the typo in the model ("mirco" vs "micro")
                )
                
                submodels.append(submodel)
            
            # Create ClusterModel
            model = models.ClusterModel(
                respondent_id=respondent_id,
                response=response,
                response_segment=submodels
            )
            
            result_models.append(model)
        
        return result_models    

# Example usage and testing
if __name__ == "__main__":
    import data_io
    import csvHandler
    from clustering_config import ClusteringConfig
    
    csv_handler = csvHandler.CsvHandler()
    data_loader = data_io.DataLoader()
    
    filename = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
    var_name = "Q20"
    var_lab = data_loader.get_varlab(filename=filename, var_name=var_name)
   
    input_list = csv_handler.load_from_csv(filename, 'embeddings', models.EmbeddingsModel)

    # Create configuration (uses defaults: description embeddings, Dutch language)
    config = ClusteringConfig()
    
    # Or customize:
    # config = ClusteringConfig(embedding_type="code", language="en")
    
    clusterer = ClusterGenerator(
        input_list=input_list,
        var_lab=var_lab,
        config=config,  # Pass the config object
        verbose=True)
    
    clusterer.run_pipeline()
    cluster_results = clusterer.to_cluster_model()
    csv_handler.save_to_csv(cluster_results, filename, 'clusters')
    
    # Print quality metrics
    if clusterer.clustering_attempts:
        print("\nClustering Quality Metrics:")
        final_metrics = clusterer.clustering_attempts[-1]  # Get last attempt (final result)
        for key, value in final_metrics.items():
            print(f"  {key}: {value:.3f}")
    
    # Print cluster summary
    meta_cluster_counts = defaultdict(int)
    meta_cluster_codes = defaultdict(list)
    meta_cluster_descriptions = defaultdict(list)
    
    for response_items in cluster_results:
        for segment_items in response_items.response_segment:
            if segment_items.meta_cluster is not None:
                meta_id = list(segment_items.meta_cluster.keys())[0]  # Get the meta-cluster ID
                meta_cluster_counts[meta_id] += 1
                meta_cluster_codes[meta_id].append(segment_items.descriptive_code)
                meta_cluster_descriptions[meta_id].append(segment_items.code_description)

    print(f"\nFound {len(meta_cluster_counts)} meta-clusters in results")
    print(f"Clustering parameters used: min_samples={clusterer.min_samples}, min_cluster_size={clusterer.min_cluster_size}")
    
    # Show retry attempts if any
    if len(clusterer.clustering_attempts) > 1:
        print(f"\nClustering attempts: {len(clusterer.clustering_attempts)}")
        for i, attempt in enumerate(clusterer.clustering_attempts):
            print(f"  Attempt {i+1}: quality={attempt.get('overall_quality', 0):.3f}")
    for meta_id, count in sorted(meta_cluster_counts.items()):
        print(f"\n📚 Meta-cluster {meta_id}: {count} items")
      
            
        sample_size = min(5, len(meta_cluster_codes[meta_id]))
        for i in range(sample_size):
            print(f"  - {meta_cluster_codes[meta_id][i]}")    
        
        
        sample_size = min(5, len(meta_cluster_descriptions[meta_id]))
        for i in range(sample_size):
            print(f"  - {meta_cluster_descriptions[meta_id][i]}")


# Test section
if __name__ == "__main__":
    """Test the clusterer with actual embeddings"""
    import sys
    from pathlib import Path
    
    # Add project paths
    project_root = Path(__file__).parents[2]
    sys.path.append(str(project_root))
    
    from cache_manager import CacheManager
    from cache_config import CacheConfig
    from clustering_config import ClusteringConfig
    import models
    
    # Initialize cache manager
    cache_config = CacheConfig()
    cache_manager = CacheManager(cache_config)
    
    # Load embeddings from cache
    filename = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
    embedded_data = cache_manager.load_from_cache(filename, "embeddings", models.EmbeddingsModel)
    
    if embedded_data:
        print(f"Loaded {len(embedded_data)} items from cache")
        
        # Test 1: Default clustering with description embeddings
        print("\n=== Test 1: Default clustering (description embeddings) ===")
        clusterer = ClusterGenerator(
            input_list=embedded_data,
            var_lab="Test variable",
            embedding_type="description",
            config=ClusteringConfig(),
            verbose=True
        )
        
        clusterer.run_pipeline()
        
        # Check quality metrics
        if clusterer.clustering_attempts:
            print("\nQuality metrics from attempts:")
            for i, metrics in enumerate(clusterer.clustering_attempts):
                print(f"Attempt {i+1}: Quality score = {metrics['quality_score']:.3f}")
        
        # Test 2: Code embeddings
        print("\n=== Test 2: Code embeddings ===")
        clusterer_code = ClusterGenerator(
            input_list=embedded_data,
            var_lab="Test variable",
            embedding_type="code",
            config=ClusteringConfig(embedding_type="code"),
            verbose=True
        )
        
        clusterer_code.run_pipeline()
        
        # Test 3: Strict quality requirements
        print("\n=== Test 3: Strict quality requirements ===")
        strict_config = ClusteringConfig(
            min_quality_score=0.7,
            max_noise_ratio=0.05
        )
        
        clusterer_strict = ClusterGenerator(
            input_list=embedded_data,
            var_lab="Test variable",
            config=strict_config,
            verbose=True
        )
        
        clusterer_strict.run_pipeline()
        
    else:
        print("No cached embeddings found. Please run the pipeline first.")