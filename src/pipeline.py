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
# filename = "M250285 input voor coderen - met Q18Q19.sav"
# id_column = "respondentid"
# var_name = "q19"
# #var_name = "Q18Q19"

filename = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
id_column = "DLNMID"
var_name = "Q20"

# filename = "M250480 Associatiemonitor ASN Bank net databestand.sav"
# id_column = "DLNMID"
# var_name = "Qd1_combined"

# Pipeline behavior flags
FORCE_RECALCULATE_ALL = False  # Set to True to bypass all cache and recalculate everything
FORCE_STEP = "gatos_codebook"  # # Options: "data", "preprocessed", "quality_filter", "extracted_ideas", "embeddings", "initial_clusters", "gatos_codebook", "theme_identification", "code_assignment"
VERBOSE = False  # Enable verbose output for debugging in Spyder
PROMPT_PRINTER = False  # Enable prompt printing for LLM calls

# Clustering parameters
LANGUAGE = "nl"  # Options: "nl" or "en" (currently not used)

# Initialize data loader and get variable label
data_loader = dataLoader.DataLoader(verbose=False)
var_lab = data_loader.get_varlab(filename=filename, var_name=var_name)

# Display configuration
print("=" * 80)
print("CODERINGSTOOL PIPELINE")
print("=" * 80)
print(f"Data file: {filename}")
print(f"Variable: {var_name} - {var_lab}")
print(f"Force recalculate: {'ALL' if FORCE_RECALCULATE_ALL else FORCE_STEP or 'None'}")
print(f"Verbose mode: {VERBOSE}")
print(f"Prompt printer: {PROMPT_PRINTER}")
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
from utils import verboseReporter
from utils import promptPrinter

FORCE = False
VERBOSE = True

step_name        = "preprocessed"
if  FORCE:
    FORCE_STEP   = step_name

force_recalc = FORCE_RECALCULATE_ALL or FORCE_STEP == step_name
verbose_reporter = verboseReporter.VerboseReporter(VERBOSE)
prompt_printer = promptPrinter.PromptPrinter(enabled=PROMPT_PRINTER, print_realtime=True)   

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
    
    # Quality filter summary
    if VERBOSE:
        print()  # Empty line
        print("=== QUALITY FILTER CODE SUMMARY ===")
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
        print()  # Empty line
    
    # Show consolidated sample corrections from all preprocessing steps
    if VERBOSE:
        print()
        print("📋 Sample preprocessing corrections:")
        
        # Collect samples from all processing steps
        all_samples = []
        
        # From text normalizer
        if hasattr(text_normalizer, 'transformation_examples') and text_normalizer.transformation_examples:
            all_samples.extend(text_normalizer.transformation_examples)
        
        # From spell checker (most important for user)
        if hasattr(spell_checker, 'correction_examples') and spell_checker.correction_examples:
            all_samples.extend(spell_checker.correction_examples)
        
        # From text finalizer
        if hasattr(text_finalizer, 'transformation_examples') and text_finalizer.transformation_examples:
            all_samples.extend(text_finalizer.transformation_examples)
        
        # Show one random sample from spell checker (most relevant) if available
        if hasattr(spell_checker, 'correction_examples') and spell_checker.correction_examples:
            import random
            sample = random.choice(spell_checker.correction_examples)
            print(f'  "{sample[0]}" -> "{sample[1]}"')
        elif all_samples:
            import random
            sample = random.choice(all_samples)
            print(f'  "{sample[0]}" -> "{sample[1]}"')
        else:
            print("  No corrections made")
        print()
    
    print(f"\n'Preprocessing phase' completed in {elapsed_time:.2f} seconds.\n")

    
# === STEP 3 ========================================================================================================
"""quality filter"""
from utils import qualityFilter

FORCE = False
VERBOSE = True

step_name        = "quality_filter"
if  FORCE:
    FORCE_STEP   = step_name

verbose_reporter = verboseReporter.VerboseReporter(VERBOSE)
prompt_printer = promptPrinter.PromptPrinter(enabled=PROMPT_PRINTER, print_realtime=True)   
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
"""Response segments/ideas"""
from utils import ideaExtractor

FORCE = False
VERBOSE = True

step_name        = "extracted_ideas"
if  FORCE:
    FORCE_STEP   = step_name

verbose_reporter = verboseReporter.VerboseReporter(VERBOSE)
prompt_printer = promptPrinter.PromptPrinter(enabled=PROMPT_PRINTER, print_realtime=True)   
force_recalc     = FORCE_RECALCULATE_ALL or FORCE_STEP == step_name

if not force_recalc and cache_manager.is_cache_valid(filename, step_name):
    encoded_text = cache_manager.load_from_cache(filename, step_name, models.IdeasExtractedModel)
    segments = sum(item.idea_count for item in encoded_text)
    verbose_reporter.summary("IDEAS EXPRESSED AND EXTRACTED FROM RESPONSES IN CACHE", {f"Input: {len(encoded_text)} filtered responses -> Output": f"{segments} response segments"})
else: 
    verbose_reporter.section_header("EXTRACTION OF IDEAS EXPRESSED PHASE")
    start_time = time.time()
    filtered_text = [item for item in quality_filtered_text if not item.quality_filter]
    verbose_reporter.stat_line(f"Input: {len(quality_filtered_text)} quality-filtered responses")
    verbose_reporter.stat_line(f"Processing: {len(filtered_text)} meaningful responses (excluded {len(quality_filtered_text) - len(filtered_text)} filtered responses)")
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

# debug - example outputs
import random
n_samples = 1
sampled_items = random.sample(encoded_text, n_samples)
for item in sampled_items:
    print(item.response)
    for segment in item.response_ideas:
        print(f"- {segment.idea}")

# === STEP 5 =======================================================================================================
"""Generate embeddings"""
from config import EmbeddingConfig
from utils.embedder import Embedder

FORCE = False

step_name = "embeddings"
if  FORCE:
    FORCE_STEP   = step_name

verbose_reporter = verboseReporter.VerboseReporter(VERBOSE)

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
    verbose_reporter.step_start("Generating Embeddings", emoji="🔗")


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
"""Reduce data/get clusters"""
from utils.clusterer import Clusterer

FORCE = False
VERBOSE = True

step_name = "initial_clusters"
if  FORCE:
    FORCE_STEP   = step_name

verbose_reporter = verboseReporter.VerboseReporter(VERBOSE)
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
    
    clusterer = Clusterer(embedded_text, verbose=VERBOSE)
    clusterer.run()
    initial_cluster_results = clusterer.to_cluster_model()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    cache_manager.save_to_cache(initial_cluster_results, filename, step_name, elapsed_time)
    print(f"\n'Initial clustering' completed in {elapsed_time:.2f} seconds.")

# # #debug - print random clusters  
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
    
    
#debug - print all clusters
cluster_ids = list(set([
    response_idea.initial_cluster 
    for result in initial_cluster_results 
    for response_idea in result.response_ideas  # This has initial_cluster
    if response_idea.initial_cluster is not None]))
for x in range(1, round(len(cluster_ids) / 20) + 1):
    y = x * 20
    print(f"\n=== Showing clusters {y-20} to {min(y, len(cluster_ids)-1)} ===\n")

    for z in range(y - 20, y):
        if z < len(cluster_ids):
            print(f"\nCluster {z}")
            for item in initial_cluster_results:
                for subitem in item.response_ideas:
                    if subitem.initial_cluster == z:
                        print(subitem.idea)
    input("\n🔸 Press Enter to continue to the next batch of clusters...")
        

# === STEP 7 ========================================================================================================
"""Generate codes"""
from utils import speculativeStarterCodes
from utils import codeGenerator as codeGenerator

FORCE = True
VERBOSE = True
PROMPT_PRINTER  = True

step_name = "codebook_generation"
if  FORCE:
    FORCE_STEP      = step_name

verbose_reporter = verboseReporter.VerboseReporter(VERBOSE)
prompt_printer = promptPrinter.PromptPrinter(enabled=PROMPT_PRINTER, print_realtime=True)   
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
    
    # Phase 1: Generate starter codes
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
        # Phase 2: Inductive code generation
        generator = codeGenerator.InductiveCodeGenerator(
             cluster_results=initial_cluster_results,
             starter_codes=starter_codes,
             var_lab=var_lab,
             k=5,
             verbose=True,
             batch_size=10,
             max_concurrent_requests=5,
             prompt_printer=prompt_printer  )
        results = generator.generate()
        
        codebook_entries = []
        codebook = []  # Legacy format for backward compatibility
        
        if results and isinstance(results, dict):
            # Display final codebook summary
            if VERBOSE and 'codebook' in results:
                verbose_reporter.empty_line()
                print("📊 FINAL CODEBOOK SUMMARY")
                verbose_reporter.stat_line(f"Total codes: {len(results['codebook'])}")
                
                # Show sample codes (first 10)
                verbose_reporter.empty_line()
                print("📋 Complete codebook:")
                
            idx = 1
            for key, value in results.items():
                if key == 'codebook':
                    for item in value:
                        if VERBOSE:
                            definition = item['definition']
                            if len(definition) > 100:
                                definition = definition[:97] + "..."
                            print(f"  {idx}. \"{item['code']}\" - {definition}")
                        
                        codebook_entry = models.CodebookEntry(
                            code=item['code'],
                            definition=item['definition'],
                            source_clusters=None  
                        )
                        codebook_entries.append(codebook_entry)
                        
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
    
    if 'codebook_model' not in locals():
        print("ERROR: codebook_model was not created!")
        codebook_model = models.CodebookModel(
            codes=[],
            generation_metadata={"error": "Failed to create codebook model"},
            source_variable=var_name
        )
    
    cache_manager.save_to_cache([codebook_model], filename, step_name, elapsed_time)
    print(f"\n'codebook generation' completed in {elapsed_time:.2f} seconds.\n")

# #debug 
# idx = 1
# for entry in codebook:
#     print(idx)
#     print(entry.code)
#     print(entry.definition)
#     print("\n")
#     idx += 1

from utils.codeGenerator_displayResults import display_cluster_analysis, display_summary_statistics
if 'results' in locals():
    
    # display code generations statistics
    display_summary_statistics(results)
    
    # debug 1: Display a random cluster with full details
    display_cluster_analysis(results)
    
    # debug 2: Display specific types of clusters
    #from utils.resultsDisplay import find_clusters_by_decision
    # new_code_clusters = find_clusters_by_decision(results, 'create_new')
    # if new_code_clusters:
    #     print("\n" + "="*80 + "\nEXAMPLE: NEW CODE CREATION\n" + "="*80)
    #     display_cluster_analysis(results, new_code_clusters[0])
    
    # modified_clusters = find_clusters_by_decision(results, 'modify_existing')
    # if modified_clusters:
    #     print("\n" + "="*80 + "\nEXAMPLE: CODE MODIFICATION\n" + "="*80)
    #     display_cluster_analysis(results, modified_clusters[0])
    
    #debug 3 : Display multiple clusters at once
    # from utils.resultsDisplay import display_multiple_clusters
    # print("\n" + "="*80 + "\nMULTIPLE CLUSTER ANALYSIS\n" + "="*80)
    # display_multiple_clusters(results, max_clusters=3)
else:
    print("No results found. Please run the generator first: results = generator.generate()")



# === STEP 8 ========================================================================================================
"""Identify themes"""
from utils.themeIdentifier import ThemeIdentifier

FORCE = True
VERBOSE = False
PROMPT_PRINTER  = False

step_name = "theme_identification"
if  FORCE:
    FORCE_STEP      = step_name

verbose_reporter = verboseReporter.VerboseReporter(VERBOSE)
prompt_printer = promptPrinter.PromptPrinter(enabled=PROMPT_PRINTER, print_realtime=True)   
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
        enriched_codebook = [models.Codebook(
            code=entry.code, 
            definition=entry.definition, 
            theme=entry.theme,
            theme_description=entry.theme_description
        ) for entry in theme_enriched_codebook.codes]
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
            print(f"  definition {theme_desc}")
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
                # Find theme details with normalized matching
                theme_name_normalized = theme_name.strip().lower()
                for theme in themes:
                    if theme['theme_name'].strip().lower() == theme_name_normalized:
                        theme_info = theme.get('theme_description', '')
                        theme_cluster_id = theme.get('cluster_id', -1)
                        is_misc = theme.get('is_miscellaneous', False)
                        break
                
                # Log if theme not found
                if theme_info is None:
                    print(f"Warning: Theme '{theme_name}' not found in themes list for code '{entry.code}'")
            
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
        enriched_codebook = [models.Codebook(
            code=entry.code, 
            definition=entry.definition,
            theme=entry.theme,
            theme_description=entry.theme_description
        ) for entry in enriched_entries]
      
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Cache the structured theme-enriched codebook (wrap in list as cache manager expects List[T])
    cache_manager.save_to_cache([theme_enriched_codebook], filename, step_name, elapsed_time)
    print(f"\n'Hierarchical theme identification' completed in {elapsed_time:.2f} seconds.\n")

# Update the main codebook with enriched data (for backward compatibility)
if enriched_codebook:
    codebook = enriched_codebook

# # Display theme-enriched codebook summary
# if theme_enriched_codebook and theme_enriched_codebook.codes:
#     print("\n=== THEME-ENRICHED CODEBOOK SUMMARY ===")
#     themes_found = {}
#     for entry in theme_enriched_codebook.codes:
#         if entry.theme:
#             if entry.theme not in themes_found:
#                 themes_found[entry.theme] = {
#                     'description': entry.theme_description,
#                     'codes': []
#                 }
#             themes_found[entry.theme]['codes'].append(entry.code)
    
#     for idx, (theme_name, theme_info) in enumerate(themes_found.items(), 1):
#         print(f"\n{idx}. {theme_name}")
#         if theme_info['description']:
#             print(f"   Description: {theme_info['description']}")
#         print(f"   Codes ({len(theme_info['codes'])}):")
#         for code in theme_info['codes']:
#             print(f"   - {code}")
    
#     # Show codes without themes
#     no_theme_codes = [entry.code for entry in theme_enriched_codebook.codes if not entry.theme]
#     if no_theme_codes:
#         print(f"\nUnthemed codes ({len(no_theme_codes)}):")
#         for code in no_theme_codes:
#             print(f"   - {code}")


# === STEP 9 ========================================================================================================
"""Assign codes (and themes)"""
from utils import codeAssigner

FORCE = True

step_name = "code_assignment"
if  FORCE:
    FORCE_STEP      = step_name
    PROMPT_PRINTER  = False

verbose_reporter = verboseReporter.VerboseReporter(VERBOSE)
prompt_printer = promptPrinter.PromptPrinter(enabled=PROMPT_PRINTER, print_realtime=True)   
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
    
    if not theme_enriched_codebook or not theme_enriched_codebook.codes:
        print("Error: No enriched codebook available for code assignment.")
        code_assigned_results = []
    elif not initial_cluster_results:
        print("Error: No cluster results available for code assignment.")
        code_assigned_results = []
    else:
        print(f"\nAssigning codes and themes from {len(theme_enriched_codebook.codes)} enriched codes to {sum(len(resp.response_ideas) for resp in initial_cluster_results if resp.response_ideas)} ideas")
  
        code_assigner_instance = codeAssigner.CodeAssigner(
            cluster_models=initial_cluster_results,  # Use cluster results from Step 6 (includes embeddings)
            codebook=[models.Codebook(
                code=entry.code, 
                definition=entry.definition,
                theme=entry.theme,
                theme_description=entry.theme_description
            ) for entry in theme_enriched_codebook.codes],  # Include theme information
            var_lab=var_lab,
            code_to_theme_mapping=theme_enriched_codebook.code_to_theme_mapping,  # Pass theme mapping for assignment
            verbose=VERBOSE,
            prompt_printer=prompt_printer)
        code_assigned_results = code_assigner_instance.assign()
     
        for result in code_assigned_results:
            if not hasattr(result, 'assignment_metadata') or result.assignment_metadata is None:
                result.assignment_metadata = {}
            result.assignment_metadata.update({
                "codebook_used": f"{len(theme_enriched_codebook.codes)} codes with themes",
                "theme_methodology": theme_enriched_codebook.theme_methodology,
                "assignment_timestamp": start_time
            })
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    cache_manager.save_to_cache(code_assigned_results, filename, step_name, elapsed_time)
    print(f"\n'Code assignment' completed in {elapsed_time:.2f} seconds.\n")


from utils.pipelineSummarizer import PipelineSummarizer
summarizer = PipelineSummarizer(verbose=True)
summarizer.generate_summary(
    code_assigned_results=code_assigned_results if 'code_assigned_results' in locals() else None,
    theme_enriched_codebook=theme_enriched_codebook if 'theme_enriched_codebook' in locals() else None,
    enriched_codebook=enriched_codebook if 'enriched_codebook' in locals() else None
)


# for result in code_assigned_results:
#     for idea in result.response_ideas:
#         print(idea.assigned_themes)
#     break
    

#debug
import random
sampled_result = random.choice(code_assigned_results)
print(f"Respondent ID: {sampled_result.respondent_id}")
print(f"Response: {sampled_result.response}")
#print(f"Idea count: {sampled_result.idea_count}")
#print(f"Codebook: {sampled_result.assignment_metadata.get('codebook_used')}")
#print("---- Assigned Codes ----")
for idea in sampled_result.response_ideas:
    print("-" * 40)
    print(f"Idea ID: {idea.idea_id}")
    print(f"Idea: {idea.idea}")
    print(f"Assigned Codes: {', '.join(idea.assigned_codes)}")
    print(f"Assigned Themes: {', '.join(idea.assigned_themes)}")
    print(f"Assignment Confidence: {idea.assignment_confidence}")
    print(f"Rationale: {idea.assignment_rationale}")
    print("-" * 40)



