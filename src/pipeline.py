import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None
import random
import re

# ===  MODULES ========================================================================================================
import time
import pandas as pd
import nest_asyncio
nest_asyncio.apply()

# === MODELS ========================================================================================================
import models

# === CONFIG ========================================================================================================
from utils import dataLoader
from utils.cacheManager import CacheManager
from config import CacheConfig

# Initialize cache manager
cache_config = CacheConfig()
cache_manager = CacheManager(cache_config)

# === PIPELINE CONFIGURATION ========================================================================================
# Test data 
filename = "M250285 input voor coderen - met Q18Q19.sav"
id_column = "respondentid"
var_name = "q19"
#var_name = "Q18Q19"

# Pipeline behavior flags
FORCE_RECALCULATE_ALL = False  # Set to True to bypass all cache and recalculate everything
FORCE_STEP = "data"  # Set to step name (e.g., "initial_clusters") to recalculate specific step
VERBOSE = True  # Enable verbose output for debugging in Spyder
PROMPT_PRINTER = False  # Enable prompt printing for LLM calls
DEBUG_CLUSTER_TRACKING = False  # Enable detailed cluster ID tracking diagnostics

# Clustering parameters
EMBEDDING_TYPE = "description"  # Options: "description" or "code"
LANGUAGE = "nl"  # Options: "nl" or "en" (currently not used)
USE_QUESTION_AWARE_EMBEDDINGS = False  # Enable question-aware embeddings

# Initialize data loader and get variable label
data_loader = dataLoader.DataLoader(verbose=False)
var_lab = data_loader.get_varlab(filename=filename, var_name=var_name)

# Display configuration
print("=" * 80)
print("CODERINGSTOOL PIPELINE")
print("=" * 80)
print(f"📊 Data file: {filename}")
print(f"📌 Variable: {var_name} - {var_lab}")
print(f"🔧 Force recalculate: {'ALL' if FORCE_RECALCULATE_ALL else FORCE_STEP or 'None'}")
print(f"🎯 Embedding type: {EMBEDDING_TYPE}")
print(f"🔀 Question-aware embeddings: {USE_QUESTION_AWARE_EMBEDDINGS}")
print(f"💬 Verbose mode: {VERBOSE}")
print(f"🤖 Prompt printer: {PROMPT_PRINTER}")
print(f"🔍 Debug cluster tracking: {DEBUG_CLUSTER_TRACKING}")
print("=" * 80)


# === STEP 1 ========================================================================================================
"""get data"""
from utils.verboseReporter import VerboseReporter

step_name = "data"
force_recalc = FORCE_RECALCULATE_ALL or FORCE_STEP == step_name

verbose_reporter = VerboseReporter(VERBOSE)
data_loader = dataLoader.DataLoader(verbose=VERBOSE)

if not force_recalc and cache_manager.is_cache_valid(filename, step_name):
    raw_text_list = cache_manager.load_from_cache(filename, step_name, models.ResponseModel)
    verbose_reporter.summary("DATA FROM CACHE", {"Input": f"{len(raw_text_list)} responses"})
else:
    verbose_reporter.section_header("DATA LOADING SUMMARY")
    start_time       = time.time()
    # loading data from spss file
    raw_text_df      = data_loader.get_variable_with_IDs(filename = filename, id_column = id_column,var_name = var_name)
    raw_unstructued  = list(zip([int(id_int) for id_int in raw_text_df[id_column].tolist()], raw_text_df[var_name].tolist()))
    raw_text_list = []
    # structuring data NaN=system missing; Numeric=undefined user missing; String=response 
    for resp_id, resp in raw_unstructued:
        if pd.isna(resp):
            response_type = 'nan'
        elif isinstance(resp, (int, float)):
            response_type = 'numeric'
        elif isinstance(resp, str):
            response_type = 'string'
        else:
            response_type = 'unknown'
        raw_text_list.append(models.ResponseModel(respondent_id=resp_id,  response=resp, response_type=response_type))
    end_time         = time.time()
    elapsed_time     = end_time - start_time
    cache_manager.save_to_cache(raw_text_list, filename, step_name, elapsed_time)
    
    print("\n=== RAW DATA TYPE ANALYSIS ===")
    type_counts = {'nan': 0, 'numeric': 0, 'string': 0, 'unknown': 0}
    for item in raw_text_list:
        type_counts[item.response_type] += 1
    for data_type, count in type_counts.items():
        print(f"{data_type}: {count} items")
    print(f"\n\n'Import data' completed in {elapsed_time:.2f} seconds.\n")
    
# debug 
# import random
# n_samples = 5
# indices = random.sample(range(len(raw_text_list)), n_samples)
# for i in indices:
#     print("Raw structured:", raw_text_list[i])
#     print("---")        

# === STEP 2 ========================================================================================================
"""preprocess data"""

from utils import textNormalizer, spellChecker, textFinalizer
from utils.verboseReporter import VerboseReporter
from utils.promptPrinter import promptPrinter

FORCE = False

step_name        = "preprocessed"
if  FORCE:
    FORCE_STEP   = step_name

force_recalc = FORCE_RECALCULATE_ALL or FORCE_STEP == step_name
verbose_reporter = VerboseReporter(VERBOSE)
prompt_printer = promptPrinter(enabled=PROMPT_PRINTER, print_realtime=True)   

code_meanings = {
    99999997: "User missing: Don't know/only expressing uncertainty", 
    99999998: "System missing: NA",
    99999999: "No answer: Empty strings/Single Characters/Only Numbers"}


if not force_recalc and cache_manager.is_cache_valid(filename, step_name):
    preprocessed_text = cache_manager.load_from_cache(filename, step_name, models.PreprocessedModel)
    code_counts = {}
    for item in preprocessed_text:
        code = item.quality_filter_code
        if code is not None:
            code_counts[code] = code_counts.get(code, 0) + 1
    verbose_reporter.summary("PREPROCESSED RESPONSES FROM CACHE", {"• Input" : f"{len(raw_text_list)} responses"})
    for code, count in code_counts.items():
        verbose_reporter.stat_line(f"{code_meanings.get(code, 'Unknown code')} = {count} responses")
    verbose_reporter.stat_line(f"Output: {len(preprocessed_text) - sum(code_counts.values())}")
else:    
    verbose_reporter.section_header("PREPROCESSING PHASE")
    # intialize utils
    text_normalizer       = textNormalizer.TextNormalizer(verbose=VERBOSE)
    spell_checker         = spellChecker.SpellChecker(verbose=VERBOSE, prompt_printer=prompt_printer)
    text_finalizer        = textFinalizer.TextFinalizer(verbose=VERBOSE)
    start_time            = time.time()
    # preprocess strings 
    all_responses = []
    string_responses = []
    non_string_responses = []
    for item in raw_text_list:
        preprocess_item = item.to_model(models.PreprocessedModel)
        all_responses.append(preprocess_item)
        if item.response_type == 'string':
            string_responses.append(preprocess_item)
        else:
            non_string_responses.append(preprocess_item)
    if string_responses:
        normalized_text = text_normalizer.normalize_responses(string_responses)
        normal_no_missing = [item for item in normalized_text if isinstance(item.response, str) and item.response != '<NA>']
        corrected_text = spell_checker.spell_check(normal_no_missing, var_lab)
        finalized_text = text_finalizer.finalize_responses(corrected_text)
    else:
        finalized_text = [] 
    # Combine processed strings with non-string responses
    # Create a mapping of respondent_id to processed response
    processed_map = {item.respondent_id: item for item in finalized_text}
    processed_map.update({item.respondent_id: item for item in non_string_responses})
    # create list for the qualityFilter/grader
    preprocessed_text = []
    for original in raw_text_list:
        if original.respondent_id in processed_map:
            item = processed_map[original.respondent_id]
            # Add initial quality filter code based on response type and content
            # Convert to PreprocessedModel to add quality_filter_code
            desc_item = item.to_model(models.PreprocessedModel)
            # Categorize based on type and content
            if item.response == 'nan':
                desc_item.quality_filter_code = 99999998  # System missing
                desc_item.quality_filter = True
            elif isinstance(item.response, int):
                # Check if it's a known missing code
                if item.response in [99999997, 99999998, 99999999]:
                    desc_item.quality_filter_code = int(item.response)
                    desc_item.quality_filter = True
                else:
                    # Regular numeric response - will be evaluated by qualityFilter
                    desc_item.quality_filter_code = None
                    desc_item.quality_filter = None
            elif isinstance(item.response, str):
                if item.response.strip() == '':
                    desc_item.quality_filter_code = 99999999   
                    desc_item.quality_filter = True
                else:
                    # Text response - will be evaluated by qualityFilter
                    desc_item.quality_filter_code = None
                    desc_item.quality_filter = None
            preprocessed_text.append(desc_item)
        else:
            preprocessed_text.append(models.PreprocessedModel(
                respondent_id=original.respondent_id,
                response='<NA>',
                response_type='nan',
                quality_filter_code=99999998,  # no answer, etc. only numbers, 1 character or empty
                quality_filter=True))
    end_time = time.time()
    elapsed_time = end_time - start_time

    cache_manager.save_to_cache(preprocessed_text, filename, step_name, elapsed_time)
    
    print("\n=== QUALITY FILTER CODE SUMMARY ===")
    code_counts = {}
    for item in preprocessed_text:
        code = item.quality_filter_code
        if code is not None:
            code_counts[code] = code_counts.get(code, 0) + 1
    
    code_meanings = {
        99999997: "User missing: Don't know/only expressing uncertainty", 
        99999998: "System missing: NA",
        99999999: "No answer: Empty strings/Single Characters/Only Numbers"}
    
    for code, count in sorted(code_counts.items()):
        meaning = code_meanings.get(code, "Unknown code")
        print(f"Code {code}: {count} items - {meaning}")
    
    print(f"Total items with codes: {sum(code_counts.values())}")
    print(f"Total items without codes: {len(preprocessed_text) - sum(code_counts.values())}")
    print(f"\n\n'Preprocessing phase' completed in {elapsed_time:.2f} seconds.\n")

#debug
# import random
# n_samples = 5
# indices = random.sample(range(len(preprocessed_text)), n_samples)
# for i in indices:
#     if preprocessed_text[i].quality_filter_code == None:
#         print("Preprocessed:", preprocessed_text[i].response)
#         print("---")    

# === STEP 3 ========================================================================================================
"""quality filter"""
from utils import qualityFilter

FORCE = False

step_name        = "quality_filter"
if  FORCE:
    FORCE_STEP   = step_name

verbose_reporter = VerboseReporter(VERBOSE)
prompt_printer   = promptPrinter(enabled=PROMPT_PRINTER, print_realtime=True) 
force_recalc     = FORCE_RECALCULATE_ALL or FORCE_STEP == step_name

if not force_recalc and cache_manager.is_cache_valid(filename, step_name):
    quality_filtered_text = cache_manager.load_from_cache(filename, step_name, models.QualityFilteredModel)
    input_len = len([item.response for item in quality_filtered_text if item.quality_filter_code != 99999998] )
    filtered_len = len([item.quality_filter for item in quality_filtered_text if item.quality_filter and item.quality_filter_code != 99999998])
    code_counts = {}
    for item in quality_filtered_text:
        code = item.quality_filter_code
        if code is not None:
            code_counts[code] = code_counts.get(code, 0) + 1
    verbose_reporter.summary("QUALIFIED RESPONESES FROM CACHE", {"• Input" : f"{input_len} responses"})
    for code, count in code_counts.items():
            if code != 99999998:
                verbose_reporter.stat_line(f"{code_meanings.get(code, 'Unknown code')} = {count} responses")
    verbose_reporter.stat_line(f"Output: {len(preprocessed_text) - sum(code_counts.values())}")
else:
    verbose_reporter.section_header("QUALITY FILTERING PHASE")
    start_time = time.time()
    grader = qualityFilter.Grader(preprocessed_text, var_lab, verbose=VERBOSE, prompt_printer=prompt_printer)
    quality_filtered_text = grader.grade()
    grading_summary = grader.summary()
    end_time = time.time()
    elapsed_time = end_time - start_time
    cache_manager.save_to_cache(quality_filtered_text, filename, step_name, elapsed_time)
    
    print("\n=== MISSING CODE SUMMARY ===")
    code_counts = {}
    for item in quality_filtered_text:
        code = item.quality_filter_code
        if code is not None:
            code_counts[code] = code_counts.get(code, 0) + 1
    code_meanings = {
        99999997: "User missing: Don't know/only expressing uncertainty", 
        99999998: "System missing: NA",
        99999999: "No answer: Empty strings/Single Characters/Only numbers/Nonsensical/gibberish/meaningless content"}
    for code, count in sorted(code_counts.items()):
        meaning = code_meanings.get(code, "Unknown code")
        print(f"Code {code}: {count} items - {meaning}")
    print(f"Total items with codes: {sum(code_counts.values())}")
    print(f"Total items without codes: {len(preprocessed_text) - sum(code_counts.values())}\n")
    print(f"\n\n'Quality filtering phase' completed in {elapsed_time:.2f} seconds.\n")

#debug
# import random
# n_samples = 5
# indices = random.sample(range(len(quality_filtered_text)), n_samples)
# for i in indices:
#     print("Filtered:", quality_filtered_text[i])
#     print("---")    


# === STEP 4 ========================================================================================================
"""describe and segment data"""
from utils import segmentDescriber

FORCE = False

step_name        = "segmented_descriptions"
if  FORCE:
    FORCE_STEP   = step_name

verbose_reporter = VerboseReporter(VERBOSE)
prompt_printer   = promptPrinter(enabled=PROMPT_PRINTER, print_realtime=True)  
force_recalc     = FORCE_RECALCULATE_ALL or FORCE_STEP == step_name

if not force_recalc and cache_manager.is_cache_valid(filename, step_name):
    encoded_text = cache_manager.load_from_cache(filename, step_name, models.SegmentedModel)
    segments = len([segment.segment_id for item in encoded_text for segment in item.response_segment])
    verbose_reporter.summary("SEGMENTED RESPONSES FROM CACHE", {f"Input: {len(encoded_text)} filtered responses → Output": f"{segments} response segments"})
else: 
    verbose_reporter.section_header("SEGMENTATION & DESCRIPTION PHASE")
    start_time = time.time()
    # Filter out items that were marked as meaningless in quality filtering
    filtered_text = [item for item in quality_filtered_text if not item.quality_filter]
    encoder = segmentDescriber.SegmentDescriber(verbose=VERBOSE, prompt_printer=prompt_printer)
    encoded_text = encoder.generate_codes(filtered_text, var_lab)
    end_time = time.time()
    elapsed_time = end_time - start_time
    cache_manager.save_to_cache(encoded_text, filename, step_name, elapsed_time)
    print(f"\n\n'Segmentation phase' completed in {elapsed_time:.2f} seconds.\n")
    

# #debug 1 - per response-label/code
# import random
# sampled_items = random.sample(encoded_text, 1)
# print(f"Q: {var_lab}\n")
# for item in sampled_items:
#     print(f"A: {item.response}\n")
#     for segment in item.response_segment:
#         print(f"-    {segment.segment_label}")

# # debug 2 - per description
# import random
# sampled_items = random.sample(encoded_text, 1)
# print(f"Q: {var_lab}\n")
# for item in sampled_items:
#     print(f"A: {item.response}\n")
#     for segment in item.response_segment:
#         print(f"-    {segment.segment_description}")
        
# # debug 2 - per description tokens
# import spacy
# nlp = spacy.load("nl_core_news_lg")

# import random
# sampled_items = random.sample(encoded_text, 1)
# print(f"Q: {var_lab}\n")
# for item in sampled_items:
#     print(f"A: {item.response}\n")
#     for segment in item.response_segment:
#         print(f"-    {segment.segment_description}")
#         doc = nlp(segment.segment_description)
#         topic_tokens = []
#         for token in doc:
#             if (token.pos_ in ['NOUN', 'PROPN'] or
#                 (token.pos_ == 'ADJ' and token.head.pos_ in ['NOUN', 'PROPN'])) and \
#                not token.is_stop and not token.is_punct:
#                 topic_tokens.append(token.lemma_)
#         print(" ".join(topic_tokens))
    
# # debug 4 - example outputs
# import random
# n_samples = 5
# sampled_items = random.sample(encoded_text, n_samples)
# for item in sampled_items:
#     for segment in item.response_segment:
#         print(segment.segment_description)
# print("\n")    
    

# === STEP 5 ========================================================================================================
"""get initial clusters"""
from utils import embedder, clusterer

FORCE = True

step_name        = "initial_clusters"
if  FORCE:
    FORCE_STEP   = step_name

verbose_reporter = VerboseReporter(VERBOSE)
force_recalc     = FORCE_RECALCULATE_ALL or FORCE_STEP == step_name

if not force_recalc and cache_manager.is_cache_valid(filename, step_name):
    initial_cluster_results = cache_manager.load_from_cache(filename, step_name, models.ClusterModel)
    cluster_ids = set([segment.initial_cluster for result in initial_cluster_results for segment in result.response_segment if segment.initial_cluster is not None])
    num_initial_clusters = len(cluster_ids)
    total_segments = sum(len(resp.response_segment) for resp in initial_cluster_results if resp.response_segment)
    verbose_reporter.summary("INITIAL CLUSTERS FROM CACHE", {"Input": f"{len(encoded_text)} responses","Total segments": f"{total_segments}", "Initial clusters": f"{num_initial_clusters}"})
else:
    verbose_reporter.section_header("INITIAL CLUSTERING PHASE")
    start_time = time.time()
    # Step 5a: Generate embeddings
    print("\nEmbedding CODES and DESCRIPTIONS of response segments")
    
    # Configure embeddings for question-aware mode if enabled
    from config import EmbeddingConfig
    # Create a fresh config instance to avoid modifying the shared default
    embedding_config = EmbeddingConfig()
    
    # Override question-aware setting based on pipeline flag
    embedding_config.use_question_aware = USE_QUESTION_AWARE_EMBEDDINGS
    
    if USE_QUESTION_AWARE_EMBEDDINGS:
        print(f"Question-aware mode enabled: {embedding_config.response_weight:.1f} Response + {embedding_config.question_weight:.1f} Question + {embedding_config.domain_anchor_weight:.1f} Domain")
    else:
        print("Question-aware mode disabled")
    
    get_embeddings = embedder.Embedder(config=embedding_config, verbose=VERBOSE)
    
    # Note: Question-aware mode will automatically use var_lab when generating embeddings
    
    input_data = [item.to_model(models.ClusterModel) for item in encoded_text]
    code_embeddings = get_embeddings.get_code_embeddings(input_data)
    description_embeddings = get_embeddings.get_description_embeddings(input_data, var_lab)
    embedded_text = get_embeddings.combine_embeddings(code_embeddings, description_embeddings)
    # Step 5b: Generate initial clusters
    print(f"\nClustering with embedding_type={EMBEDDING_TYPE}")
    cluster_gen = clusterer.ClusterGenerator(
        input_list=embedded_text, 
        var_lab=var_lab, 
        embedding_type=EMBEDDING_TYPE,
        verbose=VERBOSE)
    cluster_gen.run_pipeline()
    
    # Debug: Check alignment before converting to cluster model
    if DEBUG_CLUSTER_TRACKING:
        from debug_cluster_alignment import debug_cluster_alignment
        debug_cluster_alignment(cluster_gen, sample_size=10)
    
    initial_cluster_results = cluster_gen.to_cluster_model()
    end_time = time.time()
    elapsed_time = end_time - start_time
    cache_manager.save_to_cache(initial_cluster_results, filename, step_name, elapsed_time)
    print(f"\n'Get initial clusters' completed in {elapsed_time:.2f} seconds.")
    

#debug  1 - print random clusters
import random
cluster_ids = list(set([segment.initial_cluster for result in initial_cluster_results for segment in result.response_segment if segment.initial_cluster is not None]))
sampled_cluster = random.sample(cluster_ids, 1)[0]
print(f"\nCluster {sampled_cluster}:\n")
cluster_segments = []
for item in initial_cluster_results:
    for segment in item.response_segment:
        if segment.initial_cluster == sampled_cluster:
            cluster_segments.append(segment.segment_description)
            #cluster_segments.append(segment.segment_label)
sampled_segments = random.sample(cluster_segments, min(10, len(cluster_segments)))
for segment_desc in sampled_segments:
    print(f"-    {segment_desc}")
    
    
#debug  2  print all clusters
# cluster_ids = set([segment.initial_cluster for result in initial_cluster_results for segment in result.response_segment if segment.initial_cluster is not None])
# for x in range(1, round(len(cluster_ids) / 20) + 1):
#     y = x * 20
#     print(f"\n=== Showing clusters {y-20} to {min(y, len(cluster_ids)-1)} ===\n")

#     for z in range(y - 20, y):
#         if z < len(cluster_ids):
#             print(f"\nCluster {z}")
#             for item in initial_cluster_results:
#                 for subitem in item.response_segment:
#                     if subitem.initial_cluster == z:
#                         print(subitem.segment_description)
#     input("\n🔸 Press Enter to continue to the next batch of clusters...")


# === STEP 6 ========================================================================================================
PROMPT_PRINTER = False 
"""thematic labeling"""
if DEBUG_CLUSTER_TRACKING:
    from utils.thematicLabeller_diagnostic import DiagnosticThematicLabeller
    from config import DEFAULT_LABELLER_CONFIG
    
    verbose_reporter = VerboseReporter(VERBOSE)
    prompt_printer   = promptPrinter(enabled=PROMPT_PRINTER, print_realtime=True)
  
    
    thematic_labeller = DiagnosticThematicLabeller(config=DEFAULT_LABELLER_CONFIG, verbose=VERBOSE, prompt_printer=prompt_printer)
    labeled_results = thematic_labeller.process_hierarchy(cluster_models=initial_cluster_results, survey_question=var_lab)
    
    # Print comprehensive diagnostic summary
    thematic_labeller.print_diagnostic_summary()
    
else:
    from utils.thematicLabeller import ThematicLabeller
    from config import DEFAULT_LABELLER_CONFIG

    verbose_reporter = VerboseReporter(VERBOSE)
    prompt_printer   = promptPrinter(enabled=PROMPT_PRINTER, print_realtime=True)  # Real-time printing during pipeline

    thematic_labeller = ThematicLabeller(config=DEFAULT_LABELLER_CONFIG, verbose=VERBOSE, prompt_printer=prompt_printer)
    labeled_results = thematic_labeller.process_hierarchy(cluster_models=initial_cluster_results, survey_question=var_lab)
    cache_manager.save_to_cache(labeled_results, filename, step_name, elapsed_time)


# debug
print("\nINITIAL CLUSTERS")  
cluster_summaries = []
for cluster in sorted(thematic_labeller.labeled_clusters, key=lambda x: x.cluster_id):
        summary = f"[source ID: {cluster.cluster_id:2d}] {cluster.description}"  # Use actual cluster_id with padding
        cluster_summaries.append(summary)
cluster_summaries_text = "\n".join(cluster_summaries)

sampled_results = random.sample(labeled_results, 1)
for result in sampled_results:
    print(f"Q: {var_lab}\n")  # Assuming var_lab is defined elsewhere
    print(f"A: {result.response}\n")  # Assuming result has a 'response' attribute
    for item in result.response_segment:
        #print(f"-    {item.segment_description}")
        if isinstance(item.initial_cluster, int):
            cluster = re.sub(r'\[.*?\]', '', cluster_summaries[item.initial_cluster]).strip()
            print(f"-  cluster: {cluster}")
        if isinstance(item.Code, dict):
            concept = item.Code.values()
            print(f"-  concept: {next(iter(item.Code.values()))}\n")
        
# debug  
# print("\nAtomic concepts")  
# for concept in thematic_labeller.atomic_concepts.atomic_concepts:
#     print(concept.concept)


# debug  
# print("\nMERGED CLUSTERS")  
# merged_summaries = []
# for cluster in sorted(thematic_labeller.merged_clusters, key=lambda x: x.cluster_id):
#         summary = f"[source ID: {cluster.cluster_id:2d}] {cluster.label}"  # Use actual cluster_id with padding
#         merged_summaries.append(summary)
# merged_summaries_text = "\n".join(merged_summaries)
# print(merged_summaries_text)

# debug  
# codebook = thematic_labeller.refined_codebook
# lines_final = []
# for theme in codebook.themes:
#     lines_final.append(f"{theme.theme_id}. {theme.label.upper()}")
#     for atomic_concept in theme.atomic_concepts:
#         lines_final.append(f"{atomic_concept.concept_id}. {atomic_concept.label.upper()}")
#         lines_final.append(f"{atomic_concept.description}")
# print("\n==== CODEBOOK (After all phases) ===")    
# print("\n".join(lines_final))

# === STEP 6 ========================================================================================================

# step_name = "labels"
# verbose_reporter = VerboseReporter(VERBOSE)
# prompt_printer   = promptPrinter(enabled=PROMPT_PRINTER, print_realtime=True)  # Real-time printing during pipeline
# force_recalc = FORCE_RECALCULATE_ALL or FORCE_STEP == step_name


# if not force_recalc and cache_manager.is_cache_valid(filename, step_name):
#     labeled_results = cache_manager.load_from_cache(filename, step_name, models.LabelModel)
#     if labeled_results and labeled_results[0].themes:
#         # Get complete hierarchical structure from first result (same for all)
#         first_result = labeled_results[0]

#         # Count themes, topics, and codes from hierarchical structure
#         theme_count = len(first_result.themes)
#         topic_count = sum(len(theme.topics) for theme in first_result.themes)
#         code_count = sum(len(topic.codes) for theme in first_result.themes for topic in theme.topics)

#         # Get cluster mappings
#         cluster_count = len(first_result.cluster_mappings) if first_result.cluster_mappings else 0

#         verbose_reporter.summary("HIERARCHICAL STRUCTURE", {
#             "Themes": theme_count,
#             "Topics": topic_count,
#             "Codes": code_count,
#             "Mapped Clusters": cluster_count
#             })

#         print("\nExample Theme Structure:")
#         for theme in first_result.themes:  
#             print(f"Theme {theme.theme_id}: {theme.label}")
#             for topic in theme.topics: 
#                 print(f"  Topic {topic.topic_id}: {topic.label}")
#                 for code in topic.codes: 
#                     print(f"    Code {code.code_id}: {code.label}")
    
    
# else:
#     verbose_reporter.section_header("HIERARCHICAL LABELING PHASE")
#     start_time = time.time()
    
#     if DEBUG_CLUSTER_TRACKING:
#         from utils.thematicLabeller_diagnostic import DiagnosticThematicLabeller
#         print("\n" + "="*60)
#         print("🔍 DIAGNOSTIC MODE: Running with cluster ID tracking")
#         print("="*60)
#         thematic_labeller = DiagnosticThematicLabeller(config=DEFAULT_LABELLER_CONFIG, verbose=VERBOSE, prompt_printer=prompt_printer)
#         labeled_results = thematic_labeller.process_hierarchy(cluster_models=initial_cluster_results, survey_question=var_lab)
#         # Print comprehensive diagnostic summary
#         thematic_labeller.print_diagnostic_summary()
#     else:
#         thematic_labeller = ThematicLabeller(config=DEFAULT_LABELLER_CONFIG, verbose=VERBOSE, prompt_printer=prompt_printer)
#         labeled_results = thematic_labeller.process_hierarchy(cluster_models=initial_cluster_results, survey_question=var_lab)
    
#     end_time = time.time()
#     elapsed_time = end_time - start_time
#     cache_manager.save_to_cache(labeled_results, filename, step_name, elapsed_time)
#     verbose_reporter.stat_line(f"'Hierarchical labeling' completed in {elapsed_time:.2f} seconds.")


# # debug
# print("\nINITIAL CLUSTERS")  
# cluster_summaries = []
# for cluster in sorted(thematic_labeller.labeled_clusters, key=lambda x: x.cluster_id):
#         summary = f"[source ID: {cluster.cluster_id:2d}] {cluster.description}"  # Use actual cluster_id with padding
#         cluster_summaries.append(summary)
# cluster_summaries_text = "\n".join(cluster_summaries)
# print(cluster_summaries_text)
# print("\nMERGED CLUSTERS")  
# merged_summaries = []
# for cluster in sorted(thematic_labeller.merged_clusters, key=lambda x: x.cluster_id):
#         summary = f"[source ID: {cluster.cluster_id:2d}] {cluster.label}"  # Use actual cluster_id with padding
#         merged_summaries.append(summary)
# merged_summaries_text = "\n".join(merged_summaries)
# print(merged_summaries_text)


# === STEP 7 ========================================================================================================
"""export results"""
from utils.resultsExporter import ResultsExporter

step_name = "results"
verbose_reporter = VerboseReporter(VERBOSE)
force_recalc = FORCE_RECALCULATE_ALL or FORCE_STEP == step_name

if not force_recalc and cache_manager.is_cache_valid(filename, step_name):
    export_results = cache_manager.load_from_cache(filename, step_name, dict)
    verbose_reporter.summary("EXPORT RESULTS FROM CACHE", {
        "SPSS file": export_results.get('spss_file', 'Not found'),
        "Excel file": export_results.get('excel_file', 'Not found')
    })
else:
    verbose_reporter.section_header("RESULTS EXPORT PHASE")
    start_time = time.time()
    
    # Initialize results exporter
    results_exporter = ResultsExporter(verbose=VERBOSE)
    
    # Export results to SPSS and Excel
    export_results = results_exporter.export_results(
        labeled_results=labeled_results,
        filename=filename,
        id_column=id_column,
        var_name=var_name
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Cache the export results (file paths)
    cache_manager.save_to_cache(export_results, filename, step_name, elapsed_time)
    
    verbose_reporter.stat_line(f"'Results export' completed in {elapsed_time:.2f} seconds.")

print("\n" + "=" * 80)
print("PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 80)
print("📊 Final output files:")
print(f"   • SPSS: {export_results.get('spss_file', 'Not generated')}")
print(f"   • Excel: {export_results.get('excel_file', 'Not generated')}")
print(f"📁 Export directory: {export_results.get('export_directory', 'Unknown')}")
print("=" * 80)

