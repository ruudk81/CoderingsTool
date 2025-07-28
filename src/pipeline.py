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
FORCE_STEP = None  # # Options: "data", "preprocessed", "quality_filter", "extracted_ideas", "embeddings", "initial_clusters", "gatos_codebook", "theme_identification", "code_assignment"
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
import random
cluster_ids = list(set([
    response_idea.initial_cluster 
    for result in initial_cluster_results 
    for response_idea in result.response_ideas   
    if response_idea.initial_cluster is not None]))
sampled_cluster = random.sample(cluster_ids, 1)[0]
print(f"\nCluster {sampled_cluster}:\n")
cluster_segments = []
for result in initial_cluster_results:
    for response_idea in result.response_ideas:   
        if response_idea.initial_cluster == sampled_cluster:
            cluster_segments.append(response_idea.idea)
sampled_segments = random.sample(cluster_segments, min(10, len(cluster_segments)))
for segment_desc in sampled_segments:
    print(f"-    {segment_desc}")
    
    
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
"""Code Generation"""
from utils import speculativeStarterCodes
from utils import codebookGenerator as codebookGenerator

FORCE = True

step_name = "codebook_generation"
if  FORCE:
    FORCE_STEP      = step_name
    PROMPT_PRINTER  = True

verbose_reporter = VerboseReporter(VERBOSE)
prompt_printer = promptPrinter(enabled=PROMPT_PRINTER, print_realtime=True)
force_recalc = FORCE_RECALCULATE_ALL or FORCE_STEP == step_name

if not force_recalc and cache_manager.is_cache_valid(filename, step_name):
    codebook_models = cache_manager.load_from_cache(filename, step_name, models.CodebookModel)
    if codebook_models and len(codebook_models) > 0:
        codebook_model = codebook_models[0]  # Extract the single model from the list
        verbose_reporter.summary("CODEBOOK FROM CACHE", {
            "Total codes": len(codebook_model.codes),
            "Source variable": codebook_model.source_variable
        })
        # Extract legacy codebook list for backward compatibility
        codebook = [models.Codebook(code=entry.code, definition=entry.definition) 
                    for entry in codebook_model.codes]
    else:
        print("ERROR: Failed to load codebook from cache")
        codebook_model = models.CodebookModel(codes=[], source_variable=var_name)
        codebook = []
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
        codebook_model = models.CodebookModel(
            codes=[],
            generation_metadata={"error": "Failed to generate starter codes"},
            source_variable=var_name
        )
        codebook = []
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
        
        # Convert results to new CodebookModel structure
        codebook_entries = []
        codebook = []  # Legacy format for backward compatibility
        
        if results and isinstance(results, dict):
            idx = 1
            for key, value in results.items():
                if key == 'codebook':
                    for item in value:
                        print(f"{idx}. {item['code']} : {item['definition']}")
                        
                        # New structured format
                        codebook_entry = models.CodebookEntry(
                            code=item['code'],
                            definition=item['definition'],
                            source_clusters=None  # Could be enhanced later with cluster tracing
                        )
                        codebook_entries.append(codebook_entry)
                        
                        # Legacy format for backward compatibility
                        legacy_entry = models.Codebook(
                            code=item['code'],
                            definition=item['definition'],
                            topic=None,
                            theme=None
                        )
                        codebook.append(legacy_entry)
                        idx += 1
        else:
            print("Warning: Codebook generator returned no results")
        
        # Create structured codebook model
        codebook_model = models.CodebookModel(
            codes=codebook_entries,
            generation_metadata={
                "methodology": "Inductive codebook generation from clusters",
                "starter_codes_count": len(starter_codes) if starter_codes else 0,
                "total_codes_generated": len(codebook_entries),
                "k_parameter": 5,
                "generation_success": len(codebook_entries) > 0
            },
            source_variable=var_name
        )

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Debug: Check what we're trying to cache
    if 'codebook_model' not in locals():
        print("ERROR: codebook_model was not created!")
        codebook_model = models.CodebookModel(
            codes=[],
            generation_metadata={"error": "Failed to create codebook model"},
            source_variable=var_name
        )
    
    print(f"DEBUG: Attempting to cache CodebookModel with {len(codebook_model.codes)} codes")
    
    # Cache the structured codebook model (wrap in list as cache manager expects List[T])
    cache_manager.save_to_cache([codebook_model], filename, step_name, elapsed_time)
    print(f"\n'codebook generation' completed in {elapsed_time:.2f} seconds.\n")

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
from utils.themeIdentifier import ThemeIdentifier

FORCE = True

step_name = "theme_identification"
if  FORCE:
    FORCE_STEP      = step_name
    PROMPT_PRINTER  = True

verbose_reporter = VerboseReporter(VERBOSE)
prompt_printer = promptPrinter(enabled=PROMPT_PRINTER, print_realtime=True)
force_recalc = FORCE_RECALCULATE_ALL or FORCE_STEP == step_name

if not force_recalc and cache_manager.is_cache_valid(filename, step_name):
    theme_enriched_codebooks = cache_manager.load_from_cache(filename, step_name, models.ThemeEnrichedCodebookModel)
    if theme_enriched_codebooks and len(theme_enriched_codebooks) > 0:
        theme_enriched_codebook = theme_enriched_codebooks[0]  # Extract the single model from the list
        verbose_reporter.summary("THEME IDENTIFICATION FROM CACHE", {
            "Total codes": len(theme_enriched_codebook.codes),
            "With themes": len([c for c in theme_enriched_codebook.codes if c.theme]),
            "Themes identified": len(theme_enriched_codebook.themes_summary) if theme_enriched_codebook.themes_summary else 0
        })
        # Extract legacy codebook for backward compatibility
        enriched_codebook = [models.Codebook(code=entry.code, definition=entry.definition, 
                                            topic=entry.theme, theme=entry.theme) 
                            for entry in theme_enriched_codebook.codes]
    else:
        print("ERROR: Failed to load theme enriched codebook from cache")
        theme_enriched_codebook = models.ThemeEnrichedCodebookModel(
            codes=[], 
            source_variable=var_name,
            themes_summary=[],
            code_to_theme_mapping={},
            theme_methodology="Error loading from cache"
        )
        enriched_codebook = []
else:
    verbose_reporter.section_header("THEME IDENTIFICATION PHASE")
    start_time = time.time()
    
    if not codebook:
        print("Error: No codes available for theme identification.")
        theme_enriched_codebook = models.ThemeEnrichedCodebookModel(
            codes=[],
            generation_metadata={"error": "No codes available"},
            source_variable=var_name,
            themes_summary=[],
            code_to_theme_mapping={},
            theme_methodology="No theme identification performed"
        )
        enriched_codebook = []
    else:
        theme_identifier = ThemeIdentifier(
            codebook=codebook,
            var_lab=var_lab,
            verbose=VERBOSE,
            prompt_printer=prompt_printer
        )
        
        async def run_theme_identification():
            return await theme_identifier.identify_themes_by_clustering()
            
        result = asyncio.run(run_theme_identification())    
        
        # Process theme results into structured format
        enriched_entries = []
        code_to_theme_mapping = {}
        themes = result['themes']

        # Build code-to-theme mapping
        for theme in themes:
            theme_name = theme['theme_name']
            theme_desc = theme.get('theme_description', '')
            cluster_id = theme.get('cluster_id', -1)
            is_misc = theme.get('is_miscellaneous', False)
            
            print(f"\n🟣 Theme: {theme_name}")
            print(f"   Cluster ID: {cluster_id}")
            print("   Codes:")
            
            for code_info in theme['codes']:
                code_name = code_info['code_name']
                code_to_theme_mapping[code_name] = theme_name
                print(f"     - Code {code_info['code_number']}: {code_name}")
        
        # Enrich codebook entries with theme information
        for entry in codebook_model.codes:
            theme_name = code_to_theme_mapping.get(entry.code)
            theme_info = None
            theme_cluster_id = None
            is_misc = False
            
            if theme_name:
                # Find theme details
                for theme in themes:
                    if theme['theme_name'] == theme_name:
                        theme_info = theme.get('theme_description', '')
                        theme_cluster_id = theme.get('cluster_id', -1)
                        is_misc = theme.get('is_miscellaneous', False)
                        break
            
            enriched_entry = models.ThemeEnrichedCodebookEntry(
                code=entry.code,
                definition=entry.definition,
                source_clusters=entry.source_clusters,
                theme=theme_name,
                theme_description=theme_info,
                theme_cluster_id=theme_cluster_id,
                is_miscellaneous=is_misc
            )
            enriched_entries.append(enriched_entry)
        
        # Create structured theme-enriched codebook
        theme_enriched_codebook = models.ThemeEnrichedCodebookModel(
            codes=enriched_entries,
            generation_metadata=codebook_model.generation_metadata,
            source_variable=codebook_model.source_variable,
            themes_summary=themes,
            code_to_theme_mapping=code_to_theme_mapping,
            theme_methodology=result.get('methodology', 'Clustering-based theme identification')
        )
        
        # Create legacy enriched codebook for backward compatibility
        enriched_codebook = [models.Codebook(code=entry.code, definition=entry.definition,
                                           topic=entry.theme, theme=entry.theme)
                           for entry in enriched_entries]
      
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Cache the structured theme-enriched codebook (wrap in list as cache manager expects List[T])
    cache_manager.save_to_cache([theme_enriched_codebook], filename, step_name, elapsed_time)
    print(f"\n'Hierarchical theme identification' completed in {elapsed_time:.2f} seconds.\n")

# Update the main codebook with enriched data (for backward compatibility)
if enriched_codebook:
    codebook = enriched_codebook

# === STEP 9 ========================================================================================================
"""Code Assignment"""
from utils import codeAssigner

FORCE = False

step_name = "code_assignment"
if  FORCE:
    FORCE_STEP      = step_name
    PROMPT_PRINTER  = True

verbose_reporter = VerboseReporter(VERBOSE)
prompt_printer = promptPrinter(enabled=PROMPT_PRINTER, print_realtime=True)
force_recalc = FORCE_RECALCULATE_ALL or FORCE_STEP == step_name

if not force_recalc and cache_manager.is_cache_valid(filename, step_name):
    code_assigned_results = cache_manager.load_from_cache(filename, step_name, models.CodeAssignedModel)
    total_ideas = sum(len(resp.response_ideas) for resp in code_assigned_results if resp.response_ideas)
    total_assignments = sum(len([idea for idea in resp.response_ideas if idea and idea.assigned_codes]) for resp in code_assigned_results if resp.response_ideas)
    verbose_reporter.summary("CODE ASSIGNMENTS FROM CACHE", {
        "Input responses": len(code_assigned_results),
        "Ideas processed": total_ideas,
        "Code assignments": total_assignments,
        "Theme assignments": sum(len([idea for idea in resp.response_ideas if idea and idea.assigned_themes]) for resp in code_assigned_results if resp.response_ideas)
    })
else:
    verbose_reporter.section_header("CODE ASSIGNMENT PHASE")
    start_time = time.time()
    
    # Check inputs from both Step 6 (clusters) and Step 8 (enriched codebook)
    if not theme_enriched_codebook or not theme_enriched_codebook.codes:
        print("Error: No enriched codebook available for code assignment.")
        code_assigned_results = []
    elif not initial_cluster_results:
        print("Error: No cluster results available for code assignment.")
        code_assigned_results = []
    else:
        print(f"\nAssigning codes and themes from {len(theme_enriched_codebook.codes)} enriched codes to {sum(len(resp.response_ideas) for resp in initial_cluster_results if resp.response_ideas)} ideas")
        
        # Convert cluster results to use embedded text (they should have embeddings from Step 5)
        ideas_with_embeddings = []
        for cluster_result in initial_cluster_results:
            if hasattr(cluster_result, 'response_ideas') and cluster_result.response_ideas:
                for idea in cluster_result.response_ideas:
                    if hasattr(idea, 'idea_embedding') and idea.idea_embedding is not None:
                        ideas_with_embeddings.append(cluster_result)
                        break
        
        if not ideas_with_embeddings:
            print("Error: No embedded ideas found in cluster results.")
            code_assigned_results = []
        else:
            code_assigner_instance = codeAssigner.CodeAssigner(
                ideas_extracted_models=initial_cluster_results,  # Use cluster results with embeddings
                codebook=[models.Codebook(code=entry.code, definition=entry.definition) 
                         for entry in theme_enriched_codebook.codes],  # Legacy format for compatibility
                var_lab=var_lab,
                verbose=VERBOSE,
                prompt_printer=prompt_printer
            )
            
            # Pass theme mapping to the assigner for theme assignment
            code_assigner_instance.code_to_theme_mapping = theme_enriched_codebook.code_to_theme_mapping
            
            assignment_results = code_assigner_instance.assign()
            
            # Convert to CodeAssignedModel (extends ClusterModel)
            code_assigned_results = []
            for i, cluster_result in enumerate(initial_cluster_results):
                if i < len(assignment_results):
                    assigned_result = assignment_results[i]
                    
                    # Convert to CodeAssignedModel structure
                    code_assigned_model = models.CodeAssignedModel(
                        respondent_id=cluster_result.respondent_id,
                        response=cluster_result.response,
                        quality_filter=cluster_result.quality_filter,
                        quality_filter_code=cluster_result.quality_filter_code,
                        response_ideas=assigned_result.response_ideas if hasattr(assigned_result, 'response_ideas') else None,
                        idea_count=assigned_result.idea_count if hasattr(assigned_result, 'idea_count') else 0,
                        assignment_metadata={
                            "codebook_used": f"{len(theme_enriched_codebook.codes)} codes with themes",
                            "theme_methodology": theme_enriched_codebook.theme_methodology,
                            "assignment_timestamp": start_time
                        }
                    )
                    code_assigned_results.append(code_assigned_model)
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Cache the results
    cache_manager.save_to_cache(code_assigned_results, filename, step_name, elapsed_time)
    print(f"\n'Code assignment' completed in {elapsed_time:.2f} seconds.\n")

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

if 'code_assigned_results' in locals() and code_assigned_results:
    total_responses = len(code_assigned_results)
    total_ideas = sum(len(resp.response_ideas) for resp in code_assigned_results if resp.response_ideas)
    total_assignments = sum(len([idea for idea in resp.response_ideas if idea and idea.assigned_codes]) for resp in code_assigned_results if resp.response_ideas)
    print(f"   • Code assignments: {total_assignments} assignments for {total_ideas} ideas across {total_responses} responses")
    
    # Show average confidence
    all_confidences = []
    for resp in code_assigned_results:
        if resp.response_ideas:
            for idea in resp.response_ideas:
                if idea and idea.assignment_confidence is not None:
                    all_confidences.append(idea.assignment_confidence)
    
    if all_confidences:
        avg_confidence = sum(all_confidences) / len(all_confidences)
        print(f"   • Average assignment confidence: {avg_confidence:.2f}")

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

