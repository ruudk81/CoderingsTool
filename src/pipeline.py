import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# ===  MODULES ========================================================================================================
import time
import asyncio
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
# #var_name = "Q18Q19"

# filename = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
# id_column = "DLNMID"
# var_name = "Q20"

# Pipeline behavior flags
FORCE_RECALCULATE_ALL = False  # Set to True to bypass all cache and recalculate everything
FORCE_STEP = None  # # Options: "data", "preprocessed", "quality_filter", "extracted_ideas", "embeddings", "initial_clusters", "gatos_codebook", "theme_identification"
VERBOSE = True  # Enable verbose output for debugging in Spyder
PROMPT_PRINTER = False  # Enable prompt printing for LLM calls

# STEP 5 CACHING IMPROVEMENT:
# - Split Step 5 into 5a (embeddings) and 5b (clustering) for independent caching
# - embedded_text is now cached as EmbeddingsModel and can be loaded separately for Step 6
# - Granular control: FORCE_STEP = "embeddings" or "initial_clusters"

# Clustering parameters
LANGUAGE = "nl"  # Options: "nl" or "en" (currently not used)

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
print(f"💬 Verbose mode: {VERBOSE}")
print(f"🤖 Prompt printer: {PROMPT_PRINTER}")
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
    processed_map = {item.respondent_id: item for item in finalized_text}
    processed_map.update({item.respondent_id: item for item in non_string_responses})
    preprocessed_text = []
    for original in raw_text_list:
        if original.respondent_id in processed_map:
            item = processed_map[original.respondent_id]
            desc_item = item.to_model(models.PreprocessedModel)
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

# # debug
# import random
# n_samples = 5
# indices = random.sample(range(len(quality_filtered_text)), n_samples)
# for i in indices:
#     print("Filtered:", quality_filtered_text[i])
#     print("---")    


# === STEP 4 ========================================================================================================
"""Extract initial ideas"""
from utils import ideaExtractor

FORCE = False

step_name        = "extracted_ideas"
if  FORCE:
    FORCE_STEP   = step_name

verbose_reporter = VerboseReporter(VERBOSE)
prompt_printer   = promptPrinter(enabled=PROMPT_PRINTER, print_realtime=True)  
force_recalc     = FORCE_RECALCULATE_ALL or FORCE_STEP == step_name

if not force_recalc and cache_manager.is_cache_valid(filename, step_name):
    encoded_text = cache_manager.load_from_cache(filename, step_name, models.IdeasExtractedModel)
    segments = sum(item.idea_count for item in encoded_text)
    verbose_reporter.summary("IDEAS EXPRESSED AND EXTRACTED FROM RESPONSES IN CACHE", {f"Input: {len(encoded_text)} filtered responses → Output": f"{segments} response segments"})
else: 
    verbose_reporter.section_header("EXTRACTION OF IDEAS EXPRESSD PHASE")
    start_time = time.time()
    # Filter out items that were marked as meaningless in quality filtering
    filtered_text = [item for item in quality_filtered_text if not item.quality_filter]
    encoder = ideaExtractor.IdeaExtractor(
        responses=filtered_text,
        var_lab=var_lab,
        verbose=VERBOSE,
        prompt_printer=prompt_printer
    )
    encoded_text = encoder.extract()
    end_time = time.time()
    elapsed_time = end_time - start_time
    cache_manager.save_to_cache(encoded_text, filename, step_name, elapsed_time)
    print(f"\n\n'Idea extraction phase' completed in {elapsed_time:.2f} seconds.\n")
    

# for text in encoded_text:
#     print(text)
#     break

# # debug - example outputs
# import random
# n_samples = 1
# sampled_items = random.sample(encoded_text, n_samples)
# for item in sampled_items:
#     print(item.response)
#     for segment in item.response_ideas:
#         print(f"- {segment.idea}")

# === STEP 5 =======================================================================================================
"""Generate embeddings"""
from config import EmbeddingConfig
from utils.embedder import Embedder

FORCE = False

step_name = "embeddings"
if  FORCE:
    FORCE_STEP   = step_name

verbose_reporter = VerboseReporter(VERBOSE)
force_recalc = FORCE_RECALCULATE_ALL or FORCE_STEP == step_name

if not force_recalc and cache_manager.is_cache_valid(filename, step_name):
    embedded_text = cache_manager.load_from_cache(filename, step_name, models.EmbeddingsModel)
    total_embeddings = sum(len(resp.response_ideas) for resp in embedded_text if resp.response_ideas)
    verbose_reporter.summary("EMBEDDINGS FROM CACHE", {
        "Input": f"{len(encoded_text)} responses", 
        "Total embeddings": f"{total_embeddings}"
    })
else:
    verbose_reporter.section_header("EMBEDDING GENERATION PHASE")
    start_time = time.time()
    print("\nEmbedding of extracted ideas")

    embedding_config = EmbeddingConfig()
    get_embeddings = Embedder(config=embedding_config, verbose=VERBOSE) 
    input_data = [item.to_model(models.EmbeddingsModel) for item in encoded_text]
    embedded_text = get_embeddings.get_embeddings_with_tracking(input_data, var_lab)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    cache_manager.save_to_cache(embedded_text, filename, step_name, elapsed_time)
    print(f"\n'Embedding generation' completed in {elapsed_time:.2f} seconds.")

#debug 
# import random
# n_samples = 1
# sampled_items = random.sample(embedded_text, n_samples)
# for item in sampled_items:
#     print(f"{item.response}\n")
#     for segment in item.response_ideas:
#         print(f"- {segment.idea}")

# === STEP 6 =======================================================================================================
"""Generate initial clusters"""
from utils.clusterer import Clusterer

FORCE = False

step_name = "initial_clusters"
if  FORCE:
    FORCE_STEP   = step_name

verbose_reporter = VerboseReporter(VERBOSE)
force_recalc = FORCE_RECALCULATE_ALL or FORCE_STEP == step_name

if not force_recalc and cache_manager.is_cache_valid(filename, step_name):
    initial_cluster_results = cache_manager.load_from_cache(filename, step_name, models.ClusterModel)
    cluster_ids = set([segment.initial_cluster for result in initial_cluster_results for segment in result.response_ideas if segment.initial_cluster is not None])
    num_initial_clusters = len(cluster_ids)
    total_segments = sum(len(resp.response_ideas) for resp in initial_cluster_results if resp.response_ideas)
    verbose_reporter.summary("INITIAL CLUSTERS FROM CACHE", {
        "Input": f"{len(embedded_text)} responses",
        "Total segments": f"{total_segments}", 
        "Initial clusters": f"{num_initial_clusters}"
    })
else:
    verbose_reporter.section_header("INITIAL CLUSTERING PHASE")
    start_time = time.time()
    print("\nClustering embedded ideas")
    
    clusterer = Clusterer(embedded_text)
    clusterer.run()
    initial_cluster_results = clusterer.to_cluster_model()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    cache_manager.save_to_cache(initial_cluster_results, filename, step_name, elapsed_time)
    print(f"\n'Initial clustering' completed in {elapsed_time:.2f} seconds.")

#debug - print random clusters  
# import random
# cluster_ids = list(set([
#     response_idea.initial_cluster 
#     for result in initial_cluster_results 
#     for response_idea in result.response_ideas   
#     if response_idea.initial_cluster is not None]))
# sampled_cluster = random.sample(cluster_ids, 1)[0]
# print(f"\nCluster {sampled_cluster}:\n")
# cluster_segments = []
# for result in initial_cluster_results:
#     for response_idea in result.response_ideas:   
#         if response_idea.initial_cluster == sampled_cluster:
#             cluster_segments.append(response_idea.idea)
# sampled_segments = random.sample(cluster_segments, min(10, len(cluster_segments)))
# for segment_desc in sampled_segments:
#     print(f"-    {segment_desc}")
    
    
# #debug - print all clusters
# cluster_ids = list(set([
#     response_idea.initial_cluster 
#     for result in initial_cluster_results 
#     for response_idea in result.response_ideas  # This has initial_cluster
#     if response_idea.initial_cluster is not None]))
# for x in range(1, round(len(cluster_ids) / 20) + 1):
#     y = x * 20
#     print(f"\n=== Showing clusters {y-20} to {min(y, len(cluster_ids)-1)} ===\n")

#     for z in range(y - 20, y):
#         if z < len(cluster_ids):
#             print(f"\nCluster {z}")
#             for item in initial_cluster_results:
#                 for subitem in item.response_ideas:
#                     if subitem.initial_cluster == z:
#                         print(subitem.idea)
#     input("\n🔸 Press Enter to continue to the next batch of clusters...")
        

# === STEP 7 ========================================================================================================
"""Codebook Generation"""
from utils import speculativeStarterCodes
from utils import codebookGenerator as codebookGenerator

FORCE = False

step_name = "codebook_generation"
if  FORCE:
    FORCE_STEP      = step_name
    PROMPT_PRINTER  = True

verbose_reporter = VerboseReporter(VERBOSE)
prompt_printer = promptPrinter(enabled=PROMPT_PRINTER, print_realtime=True)
force_recalc = FORCE_RECALCULATE_ALL or FORCE_STEP == step_name

if not force_recalc and cache_manager.is_cache_valid(filename, step_name):
    codebook = cache_manager.load_from_cache(filename, step_name, models.Codebook)
    verbose_reporter.section_header("CODEBOOK FROM CACHE")
else:
    verbose_reporter.section_header("CODEBOOK GENERATION PHASE")
    start_time = time.time()
  
    starter_generator = speculativeStarterCodes.SpeculativeStarterCodes(
        var_lab=var_lab, 
        verbose=VERBOSE, 
        prompt_printer=prompt_printer
    )
    starter_codes = starter_generator.generate()
  
    if not starter_codes:
        print("Error: Failed to generate starter codes. Cannot proceed with codebook generation.")
        codebook = models.Codebook(
            code=[],
            definition= [],
            topic = [],
            theme = [])
    else:
        generator = codebookGenerator.InductiveCodebookGenerator(
             cluster_results=initial_cluster_results,
             embedded_text=embedded_text,
             starter_codes=starter_codes,
             var_lab=var_lab,
             k=5,
             verbose=True,
             batch_size=10,
             max_concurrent_requests=5,
             prompt_printer=prompt_printer  )
        results = generator.generate()
        
    idx = 1
    codebook = []
    for key, value in results.items():
        if key == 'codebook':
            for item in value:
                print(f"{idx}: {item['code']}")
                codebook_entry = models.Codebook(
                    code = item['code'],
                    definition = item['definition'],
                    topic = None,  # keep empty for now
                    theme = None  # keep empty for now
                )
                codebook.append(codebook_entry)
                idx += 1 

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Cache the results
    cache_manager.save_to_cache(codebook, filename, step_name, elapsed_time)
    print(f"\n'GATOS codebook generation' completed in {elapsed_time:.2f} seconds.\n")

#debug - decisions
# import random
# step3 = results.get('step3_recommendations', {})
# validation = results.get('validation_details', {})
# available_ids = list(step3.keys())
# sampled_id = random.choice(available_ids)

# rec = step3[sampled_id]
# print(f"ID: {sampled_id}")
# print(f"Cluster Theme: {rec.cluster_core_theme}")
# print(f"Best Matching Codes: {rec.best_matching_codes}")
# print(f"Coverage %: {rec.coverage_assessment.percentage}")
# print(f"Coverage Rationale: {rec.coverage_assessment.rationale}")
# print(f"Decision (Step 3): {rec.decision}")
# print(f"New Code Name: {rec.action_details.new_code_name}")
# print(f"New Code Definition: {rec.action_details.new_code_definition}")
# print(f"Justification: {rec.justification}")   
# print("\n") 
  
# val = validation.get(sampled_id)
# if val: 
#     print(f"Decision (Validation): {val['decision']}")
#     print(f"Decision Rationale: {val['decision_rationale']}")
#     reasoning = val.get('reasoning', {})
#     print("Reasoning:")
#     print(f"  Parsimony: {reasoning.get('parsimony', '—')}")
#     print(f"  Redundancy: {reasoning.get('redundancy', '—')}")
#     print(f"  Justification: {reasoning.get('justification', '—')}")


# for key, value in results.items():
#     if key == 'cluster_assignments':
#         for cluster_id, cluster_assignment in value.items():
#             print(cluster_assignment)
            
 
# for id_, rec in step3.items():
#     print(f"ID: {id_}")
#     print(f"Cluster Theme: {rec.cluster_core_theme}")
#     print(f"Best Matching Codes: {rec.best_matching_codes}")
#     print(f"Coverage %: {rec.coverage_assessment.percentage}")
#     print(f"Coverage Rationale: {rec.coverage_assessment.rationale}")
#     print(f"Decision (Step 3): {rec.decision}")
#     print(f"New Code Name: {rec.action_details.new_code_name}")
#     print(f"New Code Definition: {rec.action_details.new_code_definition}")
#     print(f"Justification: {rec.justification}")
    
#     # Fetch corresponding validation details
#     val = validation.get(id_)
#     if val:
#         print(f"Decision (Validation): {val['decision']}")
#         print(f"Decision Rationale: {val['decision_rationale']}")
#         reasoning = val.get('reasoning', {})
#         print("Reasoning:")
#         print(f"  Parsimony: {reasoning.get('parsimony', '—')}")
#         print(f"  Redundancy: {reasoning.get('redundancy', '—')}")
#         print(f"  Justification: {reasoning.get('justification', '—')}")
#     else:
#         print("No validation details found.")
#     print("-" * 80)
            

# for key, value in results.items():
#     if key == 'step3_recommendations':
#         for id_, rec  in value.items():
#             print(f"ID: {id_}")
#             print(f"Cluster Theme: {rec.cluster_core_theme}")
#             print(f"Best Matching Codes: {rec.best_matching_codes}")
#             print(f"Coverage %: {rec.coverage_assessment.percentage}")
#             print(f"Coverage Rationale: {rec.coverage_assessment.rationale}")
#             print(f"Decision: {rec.decision}")
#             print(f"New Code Name: {rec.action_details.new_code_name}")
#             print(f"New Code Definition: {rec.action_details.new_code_definition}")
#             print(f"Justification: {rec.justification}")
#             print("-" * 80)
#             break

# for key, value in results.items():
#     if key == 'validation_details':
#         for id_, rec in value.items():
#             print(f"ID: {id_}")
#             print(f"Decision: {rec['decision']}")
#             print(f"Decision_rationale: {rec['decision_rationale']}")
#             reasoning = rec.get('reasoning', {})
#             print("Reasoning:")
#             print(f"-Parsimony: {reasoning.get('parsimony', '—')}")
#             print(f"-Redundancy: {reasoning.get('redundancy', '—')}")
#             print(f"-Justification: {reasoning.get('justification', '—')}")
#             print("-" * 80)
#             break

# idx = 1 
# for key, value in results.items():
#     if key == 'step4_validated_codes':
#         for id_, info in value.items(): 
#             print(f"ID: {idx}")
#             print(f"Code: {info['code']}")
#             print(f"Definition: {info['definition']}\n")
#             idx += 1 
    
# for key, value in results.items():
#     print(key)


# === STEP 8 ========================================================================================================
"""Theme Identification"""
from utils import themeIdentifier

FORCE = True

step_name = "theme_identification"
if  FORCE:
    FORCE_STEP      = step_name
    PROMPT_PRINTER  = False

verbose_reporter = VerboseReporter(VERBOSE)
prompt_printer = promptPrinter(enabled=PROMPT_PRINTER, print_realtime=True)
force_recalc = FORCE_RECALCULATE_ALL or FORCE_STEP == step_name

if not force_recalc and cache_manager.is_cache_valid(filename, step_name):
    enriched_codebook = cache_manager.load_from_cache(filename, step_name, list)
    verbose_reporter.summary("THEME IDENTIFICATION FROM CACHE", {
        "Total codes": len(enriched_codebook),
        "With domains": len([c for c in enriched_codebook if c.topic]),
        "With themes": len([c for c in enriched_codebook if c.theme])
    })
else:
    verbose_reporter.section_header("THEME IDENTIFICATION PHASE")
    start_time = time.time()
    
    if not codebook:
        print("Error: No codes available for theme identification.")
        enriched_codebook = []
    else:
        theme_identifier_instance = themeIdentifier.ThemeIdentifier(
            codebook=codebook,
            var_lab=var_lab,
            verbose=VERBOSE,
            prompt_printer=prompt_printer
        )
        
        # Use the new hierarchical approach
        hierarchical_results = asyncio.run(theme_identifier_instance.identify_themes_hierarchical())
        
        print(f"\n🔍 DEBUG: Hierarchy contains {len(hierarchical_results['codebook'])} codes")
        print(f"Expected: 64 codes")
        
        for result in hierarchical_results['codebook']:
                #original_code = next((c for c in codebook if c.code == result['code']), None)
                #print(f"Code: {original_code.code}")
                #print(f"Definition: {original_code.definition}")
                print(f"Domain: {result['domain']}")
                print(f"Theme: {result['theme']}\n")










for key, value in hierarchical_results.items():
    print(key)
        
for result in hierarchical_results['codebook']:
        #original_code = next((c for c in codebook if c.code == result['code']), None)
        #print(f"Code: {original_code.code}")
        #print(f"Definition: {original_code.definition}")
        print(f"Domain: {result['domain']}")
        print(f"Theme: {result['theme']}\n")

        
        
        # # Build enriched codebook with domains and themes
        # enriched_codebook = []
        # for result in hierarchical_results['codebook']:
        #     # Find the original codebook entry
        #     original_code = next((c for c in codebook if c.code == result['code']), None)
            
        #     if original_code:
        #         enriched_entry = models.Codebook(
        #             code=original_code.code,
        #             definition=original_code.definition,
        #             topic=result['domain'],  # Domain goes in topic field
        #             theme=result['theme']    # Theme goes in theme field
        #         )
        #         enriched_codebook.append(enriched_entry)
        
        # # Validate the hierarchical results
        # if hierarchical_results['hierarchy']:
        #     validation_report = theme_identifier_instance.validate_hierarchy_completeness(hierarchical_results['hierarchy'])
        #     if validation_report['coverage_percentage'] < 95:
        #         print(f"⚠️  Warning: Only {validation_report['coverage_percentage']:.1f}% code coverage achieved")
        #         if validation_report['missing_codes']:
        #             print(f"   Missing codes: {validation_report['missing_codes']}")
        #     else:
        #         print(f"✅ Excellent coverage: {validation_report['coverage_percentage']:.1f}%")
            
        #     # Also store the traditional theme results for backward compatibility
        #     theme_results = {
        #         'suggested_themes': [],
        #         'theme_analysis': hierarchical_results['hierarchy'],
        #         'hierarchical_structure': hierarchical_results['hierarchy'],
        #         'coverage_lookup': hierarchical_results['coverage_lookup'],
        #         'validation_report': validation_report
        #     }
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Cache the enriched codebook
    cache_manager.save_to_cache(enriched_codebook, filename, step_name, elapsed_time)
    print(f"\n'Hierarchical theme identification' completed in {elapsed_time:.2f} seconds.\n")

# Update the main codebook with enriched data
if enriched_codebook:
    codebook = enriched_codebook

# Print final summary
print("\n" + "=" * 80)
print("GATOS PIPELINE COMPLETED")
print("=" * 80)
print("📊 Final Results:")
if 'enriched_codebook' in locals() and enriched_codebook:
    print(f"   • Generated codes: {len(enriched_codebook)}")
    codes_with_domains = len([c for c in enriched_codebook if c.topic])
    codes_with_themes = len([c for c in enriched_codebook if c.theme])
    print(f"   • Codes with domains: {codes_with_domains}")
    print(f"   • Codes with themes: {codes_with_themes}")
    
    # Show unique themes and domains
    unique_themes = set(c.theme for c in enriched_codebook if c.theme)
    unique_domains = set(c.topic for c in enriched_codebook if c.topic)
    
    print(f"   • Total themes: {len(unique_themes)}")
    print(f"   • Total domains: {len(unique_domains)}")
    
    if unique_themes:
        print("📋 Themes identified:")
        for i, theme in enumerate(sorted(unique_themes)[:5]):  # Show first 5 themes
            theme_codes = [c for c in enriched_codebook if c.theme == theme]
            print(f"   {i+1}. {theme} ({len(theme_codes)} codes)")
        if len(unique_themes) > 5:
            print(f"   ... and {len(unique_themes) - 5} more themes")
else:
    print("   • No hierarchical results available")
print("=" * 80)
# """export results"""
# from utils.resultsExporter import ResultsExporter

# step_name = "results"
# verbose_reporter = VerboseReporter(VERBOSE)
# force_recalc = FORCE_RECALCULATE_ALL or FORCE_STEP == step_name

# if not force_recalc and cache_manager.is_cache_valid(filename, step_name):
#     export_results = cache_manager.load_from_cache(filename, step_name, dict)
#     verbose_reporter.summary("EXPORT RESULTS FROM CACHE", {
#         "SPSS file": export_results.get('spss_file', 'Not found'),
#         "Excel file": export_results.get('excel_file', 'Not found')
#     })
# else:
#     verbose_reporter.section_header("RESULTS EXPORT PHASE")
#     start_time = time.time()
    
#     # Initialize results exporter
#     results_exporter = ResultsExporter(verbose=VERBOSE)
    
#     # Export results to SPSS and Excel
#     export_results = results_exporter.export_results(
#         labeled_results=labeled_results,
#         filename=filename,
#         id_column=id_column,
#         var_name=var_name
#     )
    
#     end_time = time.time()
#     elapsed_time = end_time - start_time
    
#     # Cache the export results (file paths)
#     cache_manager.save_to_cache(export_results, filename, step_name, elapsed_time)
    
#     verbose_reporter.stat_line(f"'Results export' completed in {elapsed_time:.2f} seconds.")

# print("\n" + "=" * 80)
# print("PIPELINE COMPLETED SUCCESSFULLY")
# print("=" * 80)
# print("📊 Final output files:")
# print(f"   • SPSS: {export_results.get('spss_file', 'Not generated')}")
# print(f"   • Excel: {export_results.get('excel_file', 'Not generated')}")
# print(f"📁 Export directory: {export_results.get('export_directory', 'Unknown')}")
# print("=" * 80)

