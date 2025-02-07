"""start in src"""

# ===  MODULES ========================================================================================================
import sys
#import os
import time
import random
import argparse
#import logging
import nest_asyncio
nest_asyncio.apply()


# === MODELS ========================================================================================================
import models

# === CONFIG ========================================================================================================
from utils import data_io
from cache_manager import CacheManager
from config import CacheConfig, ProcessingConfig

# Initialize cache manager
cache_config = CacheConfig()
processing_config = ProcessingConfig()
cache_manager = CacheManager(cache_config)

# Pipeline configuration
filename             = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
id_column            = "DLNMID"
var_name             = "Q20"
data_loader          = data_io.DataLoader()
var_lab              = data_loader.get_varlab(filename = filename, var_name = var_name)

# We'll configure logging after parsing args

# === COMMAND LINE ARGUMENTS ========================================================================================
parser = argparse.ArgumentParser(description='CoderingsTool Pipeline')
parser.add_argument('--force-recalculate', action='store_true', help='Force recalculation of all steps')
parser.add_argument('--force-step', type=str, help='Force recalculation of specific step')
parser.add_argument('--cleanup', action='store_true', help='Clean up old cache files')
parser.add_argument('--stats', action='store_true', help='Show cache statistics')
parser.add_argument('--show-metrics', action='store_true', help='Show clustering quality metrics')

# General pipeline arguments
parser.add_argument('--verbose', 
                   action='store_true',
                   default=True,  # Set to True by default for Spyder 
                   help='Enable verbose logging for debugging')

# Clustering specific arguments
parser.add_argument('--embedding-type', 
                   choices=['description', 'code'], 
                   default='description',
                   help='Type of embeddings to use for clustering (default: description)')
parser.add_argument('--language', 
                   choices=['nl', 'en'], 
                   default='nl',
                   help='Language for clustering (default: nl)')
parser.add_argument('--min-quality-score', 
                   type=float, 
                   default=0.3,
                   help='Minimum acceptable clustering quality (default: 0.3)')
parser.add_argument('--max-noise-ratio', 
                   type=float, 
                   default=0.5,
                   help='Maximum noise ratio before micro-clustering (default: 0.5)')

args = parser.parse_args()


# === STEP 1 ========================================================================================================
"""get data"""
step_name = "data"
force_recalc = args.force_recalculate or args.force_step == step_name

if not force_recalc and cache_manager.is_cache_valid(filename, step_name, processing_config):
    raw_text_list = cache_manager.load_from_cache(filename, step_name, models.ResponseModel)
    print(f"Loaded {len(raw_text_list)} items from cache for step: {step_name}")
else:    
    start_time       = time.time()
    raw_text_df      = data_loader.get_variable_with_IDs(filename = filename, id_column = id_column,var_name = var_name)
    raw_unstructued  = list(zip([int(id_int) for id_int in raw_text_df[id_column].tolist()], raw_text_df[var_name].tolist()))
    raw_text_list    = [models.ResponseModel(respondent_id=resp_id, response=resp if resp is not None else "" ) for resp_id, resp in raw_unstructued]
    end_time         = time.time()
    elapsed_time     = end_time - start_time
    
    cache_manager.save_to_cache(raw_text_list, filename, step_name, processing_config, elapsed_time)
    print(f"\n\n'Import data' completed in {elapsed_time:.2f} seconds.\n")


# === STEP 2 ========================================================================================================
"""preprocess data"""
from utils import textNormalizer, spellChecker, textFinalizer

step_name = "preprocessed"
force_recalc = args.force_recalculate or args.force_step == step_name

if not force_recalc and cache_manager.is_cache_valid(filename, step_name, processing_config):
    preprocessed_text = cache_manager.load_from_cache(filename, step_name, models.PreprocessModel)
    print(f"Loaded {len(preprocessed_text)} items from cache for step: {step_name}")
else: 
    text_normalizer       = textNormalizer.TextNormalizer()
    spell_checker         = spellChecker.SpellChecker()
    text_finalizer        = textFinalizer.TextFinalizer()
    
    start_time            = time.time()
    preprocess_text       = [item.to_model(models.PreprocessModel) for item in raw_text_list]
    normalized_text       = text_normalizer.normalize_responses(preprocess_text)
    normal_no_missing     = [item.to_model(models.PreprocessModel) for item in raw_text_list if item.response != '<NA>']
    corrected_text        = spell_checker.spell_check(normal_no_missing, var_lab)
    preprocessed_text     = text_finalizer.finalize_responses(corrected_text)
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    cache_manager.save_to_cache(preprocessed_text, filename, step_name, processing_config, elapsed_time)
    print(f"\n\n'Preprocessing phase' completed in {elapsed_time:.2f} seconds.\n")

#debug print
idx = 1 
for response in preprocessed_text[:10]:  # Limit debug output
    print(f"{idx}. {response.response}")
    idx += 1
        

# === STEP 3 ========================================================================================================
"""describe and segment data"""
from utils import qualityFilter, segmentDescriber

step_name = "segmented_descriptions"
force_recalc = args.force_recalculate or args.force_step == step_name

if not force_recalc and cache_manager.is_cache_valid(filename, step_name, processing_config):
    encoded_text = cache_manager.load_from_cache(filename, step_name, models.DescriptiveModel)
    print(f"Loaded {len(encoded_text)} items from cache for step: {step_name}")
else: 
    start_time            = time.time()
    config                = qualityFilter.GraderConfig()
    grader                = qualityFilter.Grader(preprocessed_text, var_lab, config)
    graded_text           = grader.grade()
    grading_summary       = grader.summary()
    filtered_text         = grader.filter()
    encoder               = segmentDescriber.SegmentDescriber()
    encoded_text          = encoder.generate_codes(filtered_text, var_lab)
    end_time              = time.time()
    elapsed_time          = end_time - start_time
  
    cache_manager.save_to_cache(encoded_text, filename, step_name, processing_config, elapsed_time)
    print(f"\n\n'Segmentation phase' completed in {elapsed_time:.2f} seconds.\n")

    print("\nSummary:")
    for key, value in grading_summary.items(): 
        print(f"{key}: {value}")
        
    print("\nMeaningless responses")
    random_graded_text = random.sample(graded_text, min(20, len(graded_text)))
    for text in random_graded_text:
        if text.quality_filter:
            print(text.response)
    
#debug print
random_encoded_text = random.sample(encoded_text, min(10, len(encoded_text)))
for result in random_encoded_text:
    print(f"\nRespondent ID: {result.respondent_id}")
    print(f"Response: {result.response}")
    print("Descriptive Codes:")
    codes = result.response_segment or []
    for code in codes:
        print(f"  - Segment: {code.segment_response}")
        print(f"    Code: {code.descriptive_code}")
        print(f"    Description: {code.code_description}")    


# === STEP 4 ========================================================================================================
"""get embeddings"""
from utils import embedder  

step_name = "embeddings"
force_recalc = args.force_recalculate or args.force_step == step_name

if not force_recalc and cache_manager.is_cache_valid(filename, step_name, processing_config):
    embedded_text = cache_manager.load_from_cache(filename, step_name, models.EmbeddingsModel)
    print(f"Loaded {len(embedded_text)} items from cache for step: {step_name}")
else:
    start_time              = time.time()
    get_embeddings          = embedder.Embedder()
    input_data              = [item.to_model(models.EmbeddingsModel) for item in encoded_text]
    code_embeddings         = get_embeddings.get_code_embeddings(input_data)
    description_embeddings  = get_embeddings.get_description_embeddings(input_data, var_lab)
    embedded_text           = get_embeddings.combine_embeddings(code_embeddings, description_embeddings)
    end_time                = time.time()
    elapsed_time            = end_time - start_time
    
    cache_manager.save_to_cache(embedded_text, filename, step_name, processing_config, elapsed_time)
    print(f"\n'Get embeddings' completed in {elapsed_time:.2f} seconds.")

#debug print 
for result in embedded_text[:1]:
    print(f"\nRespondent ID: {result.respondent_id}")
    print(f"Response: {result.response}")
    print("Descriptive Codes:")
    codes = result.response_segment or []
    for code in codes:
        print(f"  - Segment: {code.segment_response}")
        print(f"    Code: {code.descriptive_code}")
        print(f"    Description: {code.code_description}")
        print(f"    Code embedding shape: {code.code_embedding.shape if code.code_embedding is not None else None}")
        print(f"    Description embedding shape: {code.description_embedding.shape if code.description_embedding is not None else None}")
    print("\n")


# === STEP 5 ========================================================================================================
"get clusters"
from utils import clusterer, clusterMerger2

step_name = "clusters"
force_recalc = args.force_recalculate or args.force_step == step_name

if not force_recalc and cache_manager.is_cache_valid(filename, step_name, processing_config):
    cluster_results = cache_manager.load_from_cache(filename, step_name, models.ClusterModel)
    print(f"Loaded {len(cluster_results)} items from cache for step: {step_name}")
else:
    start_time = time.time()
    
    # Step 5a: Initial clustering
    print(f"\nClustering with embedding_type={args.embedding_type}")
    
    # Create clusterer - no config needed for simplified version
    cluster_gen = clusterer.ClusterGenerator(
        input_list=embedded_text, 
        var_lab=var_lab, 
        embedding_type=args.embedding_type,
        verbose=True )
    
    cluster_gen.run_pipeline()
    initial_clusters = cluster_gen.to_cluster_model()
    
    for result in initial_clusters:
        print(result)
        break

    # Quality metrics are already displayed by the simplified clusterer
    print("\nInitial clustering completed successfully")
    
    # Step 5b: Merge similar clusters
    print("\nMerging similar clusters...")
    # Create merger config with verbose flag from command line args
    
    from utils import clusterer, clusterMerger3
    merger = clusterMerger3.ClusterMerger(
        input_list=initial_clusters, 
        var_lab=var_lab)
    cluster_results, merge_mapping = merger.merge_clusters()
    print("\nCluster merging completed successfully")
    
    # Save merge mapping data for later use by labeller
    cache_key = 'cluster_merge_mapping'
    cache_data = {
        'merge_mapping': merge_mapping,
        'cluster_data': merger.cluster_data,
        'initial_labels': merger.initial_labels
    }
    cache_manager.cache_intermediate_data(cache_data, filename, cache_key)
    print(f"Saved merge mapping to cache with key '{cache_key}'")
    
    # Display merge statistics
    total_initial = len(merger.cluster_data)
    total_final = len(set(merge_mapping.cluster_to_merged.values()))
    merged_groups = [g for g in merge_mapping.merge_groups if len(g) > 1]
    reduction = (total_initial - total_final) / total_initial * 100 if total_initial > 0 else 0
    
    print(f"Initial clusters: {total_initial}")
    print(f"Final clusters: {total_final}")
    print(f"Merged clusters: {len(merged_groups)}")
    print(f"Reduction: {reduction:.1f}%")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Save clustering results to cache
    cache_manager.save_to_cache(cluster_results, filename, step_name, processing_config, elapsed_time)
    
    print(f"\n'Get clusters' completed in {elapsed_time:.2f} seconds.")

# debug print
from collections import defaultdict
micro_cluster_counts = defaultdict(int)
micro_cluster_codes = defaultdict(list)
micro_cluster_descriptions = defaultdict(list)

for response_items in cluster_results:
    for segment_items in response_items.response_segment:
        if segment_items.mirco_cluster is not None:
            micro_id = list(segment_items.mirco_cluster.keys())[0]  # Get the micro-cluster ID
            micro_cluster_counts[micro_id] += 1
            micro_cluster_codes[micro_id].append(segment_items.descriptive_code)
            micro_cluster_descriptions[micro_id].append(segment_items.code_description)

print(f"Found {len(micro_cluster_counts)} micro-clusters in results")
for micro_id, count in sorted(micro_cluster_counts.items()):  # Show first 10 clusters
    print(f"\n🔬 Micro-cluster {micro_id}: {count} items")
    
    sample_size = min(3, len(micro_cluster_codes[micro_id]))
    for i in range(sample_size):
        print(f"  - {micro_cluster_codes[micro_id][i]}: {micro_cluster_descriptions[micro_id][i][:50]}...")


# === STEP 6 ========================================================================================================
"""get labels"""
from utils import labeller

step_name = "labels"
force_recalc = args.force_recalculate or args.force_step == step_name

if not force_recalc and cache_manager.is_cache_valid(filename, step_name, processing_config):
    labeled_results = cache_manager.load_from_cache(filename, step_name, models.LabelModel)
    print(f"Loaded {len(labeled_results)} items from cache for step: {step_name}")
else:
    start_time = time.time()
    
    # Initialize the new labeller with config
    labeller_config = labeller.LabellerConfig()
    label_generator = labeller.Labeller(config=labeller_config)
    
    # Run the new 4-stage pipeline
    labeled_results = label_generator.run_pipeline(cluster_results, var_lab)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    cache_manager.save_to_cache(labeled_results, filename, step_name, processing_config, elapsed_time)
    print(f"\n'Get labels' completed in {elapsed_time:.2f} seconds.")

# debug print
print("\nLabel Summary:")
theme_counts = defaultdict(int)
topic_counts = defaultdict(int)

for result in labeled_results:
    for segment in result.response_segment or []:
        # Check for Theme (meta-level)
        if hasattr(segment, 'Theme') and segment.Theme:
            theme_id = list(segment.Theme.keys())[0]
            theme_counts[theme_id] += 1
        
        # Check for Topic (micro-level) 
        if hasattr(segment, 'Topic') and segment.Topic:
            topic_id = list(segment.Topic.keys())[0]
            topic_counts[topic_id] += 1

print(f"Found {len(theme_counts)} themes in results")
for theme_id, count in sorted(theme_counts.items())[:10]:  # Show first 10
    print(f"Theme {theme_id}: {count} items")
    # Show theme label
    for result in labeled_results:
        for segment in result.response_segment or []:
            if hasattr(segment, 'Theme') and segment.Theme and theme_id in segment.Theme:
                print(f"  Label: {segment.Theme[theme_id]}")
                break
        else:
            continue
        break

print(f"\nFound {len(topic_counts)} topics in results")
# Show sample of topics
for topic_id, count in sorted(topic_counts.items())[:5]:  # Show first 5
    print(f"Topic {topic_id}: {count} items")
    # Show topic label
    for result in labeled_results:
        for segment in result.response_segment or []:
            if hasattr(segment, 'Topic') and segment.Topic and topic_id in segment.Topic:
                print(f"  Label: {segment.Topic[topic_id]}")
                break
        else:
            continue
        break


