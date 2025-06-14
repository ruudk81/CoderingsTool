#!/usr/bin/env python3
"""
Debug script to investigate c-TF-IDF similarity issues.

This script demonstrates the enhanced debugging functionality for comparing
c-TF-IDF and embedding-based similarities.

Usage:
    python debug_similarity.py

This will run the similarity diagnostics on existing clustering results.
"""

import sys
import os
sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

import numpy as np
from typing import List
import pandas as pd

# Import our modules
from config import DEFAULT_CLUSTERING_CONFIG, ClusteringConfig
from utils.clusterer import ClusterGenerator
from utils.verboseReporter import VerboseReporter
from models import EmbeddingsModel, DescriptiveSubmodel


def create_test_data() -> List[EmbeddingsModel]:
    """Create synthetic test data for debugging"""
    reporter = VerboseReporter(True)
    reporter.step_start("Creating synthetic test data", "🔧")
    
    # Create test responses with ensemble embeddings
    test_responses = []
    
    # Simulate different clusters with known characteristics
    cluster_themes = [
        {
            "theme": "quality_issues",
            "responses": [
                "product quality was poor",
                "bad quality control",
                "defective items received",
                "quality below expectations"
            ]
        },
        {
            "theme": "delivery_problems", 
            "responses": [
                "delivery was very late",
                "shipping took too long",
                "package arrived damaged",
                "delayed delivery service"
            ]
        },
        {
            "theme": "customer_service",
            "responses": [
                "excellent customer support",
                "helpful staff members",
                "good customer service experience",
                "friendly customer representatives"
            ]
        }
    ]
    
    # Generate synthetic embeddings (simulating ensemble: 0.6 response + 0.3 question + 0.1 domain)
    np.random.seed(42)  # For reproducibility
    
    for resp_id, cluster_data in enumerate(cluster_themes):
        for seg_id, response_text in enumerate(cluster_data["responses"]):
            # Create synthetic ensemble embeddings
            # Base embedding for the response
            base_embedding = np.random.normal(0, 1, 1536)  # OpenAI embedding size
            
            # Add some cluster structure by biasing toward cluster center
            cluster_center = np.random.normal(resp_id * 2, 0.5, 1536)  # Different centers per cluster
            
            # Simulate ensemble: response (60%) + question (30%) + domain (10%)
            response_embedding = base_embedding
            question_embedding = np.random.normal(0, 0.8, 1536)  # Question context
            domain_embedding = np.random.normal(0, 0.5, 1536)  # Domain context
            
            ensemble_embedding = (
                0.6 * response_embedding + 
                0.3 * question_embedding + 
                0.1 * domain_embedding +
                0.2 * cluster_center  # Add cluster structure
            )
            
            # Normalize (similar to what OpenAI embeddings would be)
            ensemble_embedding = ensemble_embedding / np.linalg.norm(ensemble_embedding)
            
            # Create submodel
            submodel = DescriptiveSubmodel(
                segment_id=f"{resp_id}_{seg_id}",
                segment_response=response_text,
                segment_label=f"{cluster_data['theme']} {seg_id}",
                segment_description=f"This is about {cluster_data['theme']}: {response_text}",
                code_embedding=ensemble_embedding.astype(np.float32),
                description_embedding=ensemble_embedding.astype(np.float32)
            )
            
            # Create response model
            response_model = EmbeddingsModel(
                respondent_id=f"resp_{resp_id}_{seg_id}",
                response=response_text,
                response_segment=[submodel]
            )
            
            test_responses.append(response_model)
    
    # Add some noise/outlier responses
    outlier_responses = [
        "completely random unrelated text",
        "this has nothing to do with the survey",
        "random noise response here",
        "another unrelated comment"
    ]
    
    for i, outlier_text in enumerate(outlier_responses):
        # Create random embeddings for outliers (less structured)
        random_embedding = np.random.normal(0, 1, 1536)
        random_embedding = random_embedding / np.linalg.norm(random_embedding)
        
        submodel = DescriptiveSubmodel(
            segment_id=f"outlier_{i}",
            segment_response=outlier_text,
            segment_label=f"outlier {i}",
            segment_description=f"Outlier response: {outlier_text}",
            code_embedding=random_embedding.astype(np.float32),
            description_embedding=random_embedding.astype(np.float32)
        )
        
        response_model = EmbeddingsModel(
            respondent_id=f"outlier_{i}",
            response=outlier_text,
            response_segment=[submodel]
        )
        
        test_responses.append(response_model)
    
    reporter.stat_line(f"Created {len(test_responses)} test responses")
    reporter.stat_line(f"- {len(cluster_themes)} themed clusters")
    reporter.stat_line(f"- {len(outlier_responses)} outlier responses")
    reporter.step_complete("Test data creation completed")
    
    return test_responses


def run_similarity_debugging():
    """Run the similarity debugging demonstration"""
    reporter = VerboseReporter(True)
    reporter.section_header("C-TF-IDF SIMILARITY DEBUGGING DEMO", "🔍")
    
    # Create test data
    test_data = create_test_data()
    
    # Configure clustering with diagnostics enabled
    config = ClusteringConfig()
    config.enable_similarity_diagnostics = True  # Enable the new debugging feature
    config.verbose = True
    config.embedding_type = "description"  # Use description embeddings
    
    # Enable noise rescue to see the comparison
    config.noise_rescue.enabled = True
    config.noise_rescue.use_ctfidf_rescue = True
    config.noise_rescue.ctfidf_similarity_threshold = 0.1  # Lower threshold for more rescues
    
    # Create clusterer with debugging enabled
    reporter.step_start("Initializing clusterer with diagnostics", "⚙️")
    clusterer = ClusterGenerator(
        input_list=test_data,
        config=config,
        embedding_type="description",
        verbose=True
    )
    
    # Run the full pipeline with diagnostics
    reporter.step_start("Running clustering pipeline with similarity diagnostics", "🚀")
    clusterer.run_pipeline()
    
    # The diagnostics will be automatically run during the pipeline
    if hasattr(clusterer, 'diagnostic_results'):
        reporter.section_header("DIAGNOSTIC RESULTS SUMMARY", "📊")
        
        results = clusterer.diagnostic_results
        
        # Show verification result
        if 'verification_result' in results:
            verification = results['verification_result']
            if verification.get('matrices_identical', False):
                reporter.stat_line("✅ c-TF-IDF implementation verified correct")
            else:
                reporter.stat_line("⚠️ c-TF-IDF implementation differences detected")
        
        # Show embedding stats
        if 'embedding_stats' in results:
            stats = results['embedding_stats']
            reporter.stat_line(f"Embedding analysis:")
            reporter.stat_line(f"  - Shape: {stats['shape']}")
            reporter.stat_line(f"  - Average norm: {stats['avg_norm']:.4f}")
            reporter.stat_line(f"  - Mean: {stats['mean']:.4f}")
        
        # Show noise analysis
        if 'noise_ratio' in results:
            reporter.stat_line(f"Noise ratio: {results['noise_ratio']:.1%}")
        
        # Show key recommendations
        if 'recommendations' in results:
            reporter.sample_list("Key findings", results['recommendations'][:5])
    
    reporter.section_header("DEBUGGING DEMO COMPLETE", "✅")
    reporter.stat_line("This demo shows how to:")
    reporter.stat_line("1. ✅ Verify c-TF-IDF implementation against BERTopic")
    reporter.stat_line("2. ✅ Compare c-TF-IDF vs embedding similarities side-by-side")
    reporter.stat_line("3. ✅ Analyze the semantic gap between text and embeddings")
    reporter.stat_line("4. ✅ Get actionable recommendations for improvement")
    reporter.stat_line("")
    reporter.stat_line("To use with real data:")
    reporter.stat_line("- Set config.enable_similarity_diagnostics = True")
    reporter.stat_line("- Run your normal clustering pipeline")
    reporter.stat_line("- Review the detailed similarity comparison output")


if __name__ == "__main__":
    try:
        run_similarity_debugging()
    except Exception as e:
        reporter = VerboseReporter(True)
        reporter.stat_line(f"❌ Debug script failed: {e}")
        import traceback
        reporter.stat_line(f"Traceback: {traceback.format_exc()}")