"""start in src"""

# ===  MODULES ========================================================================================================
#import sys
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
# idx = 1 
# for response in preprocessed_text[:10]:  # Limit debug output
#     print(f"{idx}. {response.response}")
#     idx += 1
        

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
from utils import clusterer, clusterMerger

step_name = "clusters"
force_recalc = args.force_recalculate or args.force_step == step_name

if not force_recalc and cache_manager.is_cache_valid(filename, step_name, processing_config):
    cluster_results = cache_manager.load_from_cache(filename, step_name, models.ClusterModel)
    print(f"Loaded {len(cluster_results)} items from cache for step: {step_name}")
else:
    start_time = time.time()
    print(f"\nClustering with embedding_type={args.embedding_type}")
    cluster_gen = clusterer.ClusterGenerator(
        input_list=embedded_text, 
        var_lab=var_lab, 
        embedding_type=args.embedding_type,
        verbose=True )
    cluster_gen.run_pipeline()
    initial_clusters = cluster_gen.to_cluster_model()
    print("\nInitial clustering completed successfully")
    
    # # debug print 
    # for result in initial_clusters:
    #     print(result)
    #     break

    print("\nMerging similar clusters...")
    merger = clusterMerger.ClusterMerger(
        input_list=initial_clusters, 
        var_lab=var_lab)
    cluster_results, merge_mapping = merger.merge_clusters()
    print("\nCluster merging completed successfully")
    
    # Save merge mapping data for later use by labeller
    cache_key = 'cluster_merge_mapping'
    cache_data = {
        'merge_mapping': merge_mapping,
        'cluster_data': merger.cluster_data}
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
        if segment_items.micro_cluster is not None:
            micro_id = list(segment_items.micro_cluster.keys())[0]  # Get the micro-cluster ID
            micro_cluster_counts[micro_id] += 1
            micro_cluster_codes[micro_id].append(segment_items.descriptive_code)
            micro_cluster_descriptions[micro_id].append(segment_items.code_description)

print(f"Found {len(micro_cluster_counts)} micro-clusters in results")
for micro_id, count in sorted(micro_cluster_counts.items()):  # Show first 10 clusters
    print(f"\n🔬 Micro-cluster {micro_id}: {count} items")
    
    sample_size = min(3, len(micro_cluster_codes[micro_id]))
    for i in range(sample_size):
        #print(f"  - {micro_cluster_codes[micro_id][i]}: {micro_cluster_descriptions[micro_id][i][:50]}...")
        print(f"  - {micro_cluster_descriptions[micro_id][i]}")


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
    labeller_config = labeller.LabellerConfig()
    label_generator = labeller.Labeller(config=labeller_config)
    labeled_results = label_generator.run_pipeline(cluster_results, var_lab)
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    cache_manager.save_to_cache(labeled_results, filename, step_name, processing_config, elapsed_time)
    print(f"\n'Get labels' completed in {elapsed_time:.2f} seconds.")

# debug print
from collections import  Counter, defaultdict
print("\n=== UNPACKING HIERARCHICAL LABELS ===")

# Data structures to collect hierarchy
themes = {}
theme_summaries = {}
theme_topics = defaultdict(lambda: {})
topic_codes = defaultdict(lambda: defaultdict(list))

# Process each result to extract hierarchy
for result in labeled_results:
    # Extract theme summary if available
    if result.summary:
        # Store all summaries we find (they might be theme-specific)
        for segment in result.response_segment:
            if segment.Theme:
                theme_id = list(segment.Theme.keys())[0]
                if theme_id not in theme_summaries:
                    theme_summaries[theme_id] = result.summary
    
    # Extract hierarchical structure
    for segment in result.response_segment:
        if segment.Theme:
            theme_id, theme_label = list(segment.Theme.items())[0]
            themes[theme_id] = theme_label
            
            if segment.Topic:
                topic_id, topic_label = list(segment.Topic.items())[0]
                theme_topics[theme_id][topic_id] = topic_label
                
                if segment.Keyword:
                    code_id, code_label = list(segment.Keyword.items())[0]
                    topic_codes[theme_id][topic_id].append((code_id, code_label))

# Display the hierarchical structure
print(f"\nFound {len(themes)} themes:")

for theme_id in sorted(themes.keys()):
    print(f"\n{'='*60}")
    print(f"THEME {theme_id}: {themes[theme_id]}")
    print(f"{'='*60}")
    
    # Theme summary
    if theme_id in theme_summaries:
        print("\nSummary:")
        print(f"{theme_summaries[theme_id][:500]}...")
    
    # Topics in this theme
    topics_in_theme = theme_topics[theme_id]
    print(f"\nTopics ({len(topics_in_theme)}):")
    
    for topic_id in sorted(topics_in_theme.keys()):
        topic_label = topics_in_theme[topic_id]
        print(f"\n  TOPIC {topic_id}: {topic_label}")
        
        codes_in_topic = topic_codes[theme_id][topic_id]
        code_counts = Counter(codes_in_topic)
        sorted_code_counts = sorted(code_counts.items())
        
        print(f"  Unique Codes ({len(sorted_code_counts)}):")
        for (code_id, code_label), count in sorted_code_counts:
            print(f"    CODE {code_id}: {code_label} (#{count})")

# Summary statistics
print("\n=== SUMMARY STATISTICS ===")
print(f"Total respondents processed: {len(labeled_results)}")
print(f"Total segments processed: {sum(len(r.response_segment) for r in labeled_results)}")
print(f"Total themes: {len(themes)}")
total_topics = sum(len(topics) for topics in theme_topics.values())
print(f"Total topics: {total_topics}")
total_codes = len(set([code for theme in topic_codes.values() for topic in theme.values() for code in topic]))
print(f"Total codes: {total_codes}")

if cluster_results:
    print(f"Loaded {len(cluster_results)} cluster results from cache")
    
    # Count total clusters
    unique_clusters = set()
    for result in cluster_results:
        for segment in result.response_segment:
            if segment.micro_cluster:
                cluster_id = list(segment.micro_cluster.keys())[0]
                unique_clusters.add(cluster_id) 

print(f"Original clusters: {len(unique_clusters)}")



