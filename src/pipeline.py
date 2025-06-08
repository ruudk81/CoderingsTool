"""
CoderingsTool Pipeline
=====================
To run: python pipeline.py (from src directory)
"""

# ===  MODULES ========================================================================================================
import time
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
filename = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
id_column = "DLNMID"
var_name = "Q20"

# Pipeline behavior flags
FORCE_RECALCULATE_ALL = True  # Set to True to bypass all cache and recalculate everything
FORCE_STEP = None  # Set to step name (e.g., "embeddings") to recalculate specific step
VERBOSE = True  # Enable verbose output for debugging in Spyder
PROMPT_PRINTER = True  # Enable prompt printing for LLM calls

# Clustering parameters
EMBEDDING_TYPE = "description"  # Options: "description" or "code"
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
print(f"🎯 Embedding type: {EMBEDDING_TYPE}")
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
    raw_text_df      = data_loader.get_variable_with_IDs(filename = filename, id_column = id_column,var_name = var_name)
    raw_unstructued  = list(zip([int(id_int) for id_int in raw_text_df[id_column].tolist()], raw_text_df[var_name].tolist()))
    raw_text_list    = [models.ResponseModel(respondent_id=resp_id, response=resp if resp is not None else "" ) for resp_id, resp in raw_unstructued]
    end_time         = time.time()
    elapsed_time     = end_time - start_time
    cache_manager.save_to_cache(raw_text_list, filename, step_name, elapsed_time)
    print(f"\n\n'Import data' completed in {elapsed_time:.2f} seconds.\n")


# === STEP 2 ========================================================================================================
"""preprocess data"""
from utils import textNormalizer, spellChecker, textFinalizer
from utils.verboseReporter import VerboseReporter
from utils.promptPrinter import promptPrinter

step_name = "preprocessed"
force_recalc = FORCE_RECALCULATE_ALL or FORCE_STEP == step_name
verbose_reporter = VerboseReporter(VERBOSE)
prompt_printer = promptPrinter(enabled=PROMPT_PRINTER, print_realtime=True)  # Real-time printing during pipeline

if not force_recalc and cache_manager.is_cache_valid(filename, step_name):
    preprocessed_text = cache_manager.load_from_cache(filename, step_name, models.PreprocessModel)
    verbose_reporter.summary("PREPROCESSED RESPONSES FROM CACHE", {f"Input: {len(raw_text_list)} responses → Output": f"{len(preprocessed_text)} responses", "Overall success rate": f"{(len(preprocessed_text) / len(raw_text_list) * 100):.1f}%"})
else: 
    verbose_reporter.section_header("PREPROCESSING PHASE")
  
    text_normalizer       = textNormalizer.TextNormalizer(verbose=VERBOSE)
    spell_checker         = spellChecker.SpellChecker(verbose=VERBOSE, prompt_printer=prompt_printer)
    text_finalizer        = textFinalizer.TextFinalizer(verbose=VERBOSE)
  
    start_time            = time.time()
    preprocess_text       = [item.to_model(models.PreprocessModel) for item in raw_text_list]
    normalized_text       = text_normalizer.normalize_responses(preprocess_text)
    normal_no_missing     = [item.to_model(models.PreprocessModel) for item in raw_text_list if item.response != '<NA>']
    corrected_text        = spell_checker.spell_check(normal_no_missing, var_lab)
    preprocessed_text     = text_finalizer.finalize_responses(corrected_text)
    end_time = time.time()
    elapsed_time = end_time - start_time

    cache_manager.save_to_cache(preprocessed_text, filename, step_name, elapsed_time)
    print(f"\n\n'Preprocessing phase' completed in {elapsed_time:.2f} seconds.\n")

# === STEP 3 ========================================================================================================
"""describe and segment data"""
from utils import qualityFilter, segmentDescriber

step_name        = "segmented_descriptions"
verbose_reporter = VerboseReporter(VERBOSE)
prompt_printer   = promptPrinter(enabled=PROMPT_PRINTER, print_realtime=True)  # Real-time printing during pipeline
force_recalc     = FORCE_RECALCULATE_ALL or FORCE_STEP == step_name


if not force_recalc and cache_manager.is_cache_valid(filename, step_name):
    encoded_text = cache_manager.load_from_cache(filename, step_name, models.DescriptiveModel)
    filtering_rate = 100 * (1-(len(encoded_text)/len(preprocessed_text)))
    verbose_reporter.summary("SEGMENTED RESPONSES FROM CAHCE", {f"Input: {len(preprocessed_text)} responses → Output: {len(encoded_text)} coded responses": "", "Filtering rate": f"{filtering_rate:.1f}%"})

else: 
    verbose_reporter.section_header("SEGMENTATION & DESCRIPTION PHASE")
    
    start_time            = time.time()
    grader                = qualityFilter.Grader(preprocessed_text, var_lab, verbose=VERBOSE, prompt_printer=prompt_printer)
    graded_text           = grader.grade()
    grading_summary       = grader.summary()
    filtered_text         = grader.filter()
    encoder               = segmentDescriber.SegmentDescriber(verbose=VERBOSE, prompt_printer=prompt_printer)
    encoded_text          = encoder.generate_codes(filtered_text, var_lab)
    end_time              = time.time()
    elapsed_time          = end_time - start_time

    cache_manager.save_to_cache(encoded_text, filename, step_name, elapsed_time)
    print(f"\n\n'Segmentation phase' completed in {elapsed_time:.2f} seconds.\n")


# === STEP 4 ========================================================================================================
"""get embeddings"""
from utils import embedder  

step_name        = "embeddings"
get_embeddings   = embedder.Embedder(verbose=VERBOSE)
verbose_reporter = VerboseReporter(VERBOSE)
force_recalc     = FORCE_RECALCULATE_ALL or FORCE_STEP == step_name

if not force_recalc and cache_manager.is_cache_valid(filename, step_name):
    embedded_text = cache_manager.load_from_cache(filename, step_name, models.EmbeddingsModel)
    total_segments = sum(len(resp.response_segment) for resp in embedded_text if resp.response_segment)
    verbose_reporter.summary("EMBEDDINGS FROM CACHE", {f"Input: {len(encoded_text)} responses → Output": f"{len(embedded_text)} embedded responses","Total segments embedded": f"{total_segments}", "Embedding model": f"{get_embeddings.embedding_model}"})   
    
else:
    verbose_reporter.section_header("EMBEDDINGS GENERATION PHASE")
    
    start_time              = time.time()
    input_data              = [item.to_model(models.EmbeddingsModel) for item in encoded_text]
    code_embeddings         = get_embeddings.get_code_embeddings(input_data)
    description_embeddings  = get_embeddings.get_description_embeddings(input_data, var_lab)
    embedded_text           = get_embeddings.combine_embeddings(code_embeddings, description_embeddings)
    end_time                = time.time()
    elapsed_time            = end_time - start_time

    cache_manager.save_to_cache(embedded_text, filename, step_name, elapsed_time)
    print(f"\n'Get embeddings' completed in {elapsed_time:.2f} seconds.")

    
# === STEP 5 ========================================================================================================
"get clusters"
from utils import clusterer, clusterMerger

step_name        = "clusters"
verbose_reporter = VerboseReporter(VERBOSE)
prompt_printer   = promptPrinter(enabled=PROMPT_PRINTER, print_realtime=True)  # Real-time printing during pipeline
force_recalc     = FORCE_RECALCULATE_ALL or FORCE_STEP == step_name

if not force_recalc and cache_manager.is_cache_valid(filename, step_name):
    cluster_results = cache_manager.load_from_cache(filename, step_name, models.ClusterModel)
    micro_ids = set([list(segment.micro_cluster.keys())[0] for result in cluster_results for segment in result.response_segment])
    num_micro_clusters = len(micro_ids)
    verbose_reporter.summary("CLUSTERS FROM CACHE", {"Input": f"{total_segments} response segments","Output": f"{num_micro_clusters} clusters"})
    
else:
    start_time = time.time()
    print(f"\nClustering with embedding_type={EMBEDDING_TYPE}")
    cluster_gen = clusterer.ClusterGenerator(
        input_list=embedded_text, 
        var_lab=var_lab, 
        embedding_type=EMBEDDING_TYPE,
        verbose=VERBOSE )
    cluster_gen.run_pipeline()
    initial_clusters = cluster_gen.to_cluster_model()
    print("\nInitial clustering completed successfully")

    print("\nMerging similar clusters...")
    merger = clusterMerger.ClusterMerger(
        input_list=initial_clusters, 
        var_lab=var_lab,
        config=cluster_gen.config,  # Use same config as ClusterGenerator
        verbose=VERBOSE,
        prompt_printer=prompt_printer)
    cluster_results, merge_mapping = merger.merge_clusters()
    print("\nCluster merging completed successfully")

    cache_key = 'cluster_merge_mapping'
    cache_data = {
        'merge_mapping': merge_mapping,
        'cluster_data': merger.cluster_data}
    cache_manager.cache_intermediate_data(cache_data, filename, cache_key)
    print(f"Saved merge mapping to cache with key '{cache_key}'")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    cache_manager.save_to_cache(cluster_results, filename, step_name, elapsed_time)
    print(f"\n'Get clusters' completed in {elapsed_time:.2f} seconds.")

# === STEP 6 ========================================================================================================
"""get labels"""
from utils.thematicLabeller import ThematicLabeller
from config import DEFAULT_LABELLER_CONFIG

#debug
thematic_labeller = ThematicLabeller(config=DEFAULT_LABELLER_CONFIG, verbose=VERBOSE)
labeled_results = thematic_labeller.process_hierarchy(cluster_models=cluster_results, survey_question=var_lab)

# print(thematic_labeller.unused_codes_text)

# cluster_summaries = []
# for cluster in sorted(thematic_labeller.labeled_clusters, key=lambda x: x.cluster_id):
#         summary = f"[source ID: {cluster.cluster_id:2d}] {cluster.label}"  # Use actual cluster_id with padding
#         cluster_summaries.append(summary)
# cluster_summaries_text = "\n".join(cluster_summaries)
# print(cluster_summaries_text)

codebook_initial = thematic_labeller.initial_codebook
lines_initial = []
for theme in codebook_initial.themes:
    lines_initial.append(f"{theme.id}. {theme.label.upper()}")
    
    related_topics = [t for t in codebook_initial.topics if t.parent_id == theme.id]
    for topic in related_topics:
        lines_initial.append(f"   {topic.id} {topic.label}")
        
        related_codes = [c for c in codebook_initial.codes if c.parent_id == topic.id]
        for code in related_codes:
            # Include source_codes in the display
            source_info = f" → clusters: {code.source_codes}" if code.source_codes else " → no clusters"
            lines_initial.append(f"      {code.id} {code.label}{source_info}")

codebook_final = thematic_labeller.codebook
lines_final = []
for theme in codebook_final.themes:
    lines_final.append(f"{theme.id}. {theme.label.upper()}")
    
    related_topics = [t for t in codebook_final.topics if t.parent_id == theme.id]
    for topic in related_topics:
        lines_final.append(f"   {topic.id} {topic.label}")
        
        related_codes = [c for c in codebook_final.codes if c.parent_id == topic.id]
        for code in related_codes:
            # Include source_codes in the display
            source_info = f" → clusters: {code.source_codes}" if code.source_codes else " → no clusters"
            lines_final.append(f"      {code.id} {code.label}{source_info}")
    
print("\n=== INITIAL CODEBOOK (Phase 2) ===")
print("\n".join(lines_initial))
total_sources = sum(len(code.source_codes) for code in codebook_initial.codes)
print(f"Total clusters assigned: {total_sources}")

            
print("\n==== FINAL CODEBOOK (After all phases) ===")    
print("\n".join(lines_final))
total_sources = sum(len(code.source_codes) for code in codebook_final.codes)
print(f"Total clusters assigned: {total_sources}")

###############

# === STEP 6 ========================================================================================================
"""hierarchical labeling"""
from utils.thematicLabeller import ThematicLabeller
from config import DEFAULT_LABELLER_CONFIG

step_name = "labels"
verbose_reporter = VerboseReporter(VERBOSE)
prompt_printer   = promptPrinter(enabled=PROMPT_PRINTER, print_realtime=True)  # Real-time printing during pipeline
force_recalc = FORCE_RECALCULATE_ALL or FORCE_STEP == step_name

if not force_recalc and cache_manager.is_cache_valid(filename, step_name):
    labeled_results = cache_manager.load_from_cache(filename, step_name, models.LabelModel)
    if labeled_results and labeled_results[0].themes:
        # Get complete hierarchical structure from first result (same for all)
        first_result = labeled_results[0]

        # Count themes, topics, and codes from hierarchical structure
        theme_count = len(first_result.themes)
        topic_count = sum(len(theme.topics) for theme in first_result.themes)
        code_count = sum(len(topic.codes) for theme in first_result.themes for topic in theme.topics)

        # Get cluster mappings
        cluster_count = len(first_result.cluster_mappings) if first_result.cluster_mappings else 0

        verbose_reporter.summary("HIERARCHICAL STRUCTURE", {
            "Themes": theme_count,
            "Topics": topic_count,
            "Codes": code_count,
            "Mapped Clusters": cluster_count
            })

        print("\nExample Theme Structure:")
        for theme in first_result.themes:  
            print(f"Theme {theme.theme_id}: {theme.label}")
            for topic in theme.topics: 
                print(f"  Topic {topic.topic_id}: {topic.label}")
                for code in topic.codes: 
                    print(f"    Code {code.code_id}: {code.label}")
    
    
else:
    verbose_reporter.section_header("HIERARCHICAL LABELING PHASE")
    
    start_time = time.time()
    thematic_labeller = ThematicLabeller(config=DEFAULT_LABELLER_CONFIG, verbose=VERBOSE, prompt_printer=prompt_printer)
    labeled_results = thematic_labeller.process_hierarchy(cluster_models=cluster_results, survey_question=var_lab)
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    cache_manager.save_to_cache(labeled_results, filename, step_name, elapsed_time)
    verbose_reporter.stat_line(f"'Hierarchical labeling' completed in {elapsed_time:.2f} seconds.")


# === PROMPT SUMMARY ========================================================================================================
"""Print all captured prompts after pipeline completion"""
if PROMPT_PRINTER and prompt_printer:
    print("\n" + "="*80)
    print("PIPELINE COMPLETE - PROMPT SUMMARY")
    print("="*80)
    
    # Print summary statistics
    prompt_printer.print_summary()
    
    # Print all prompts again for review
    prompt_printer.print_all_prompts()
    
    # Optional: Save prompts to file
    # prompt_printer.save_prompts("pipeline_prompts.json")


