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
from utils import data_io
from utils.cache_manager import CacheManager
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

# Clustering parameters
EMBEDDING_TYPE = "description"  # Options: "description" or "code"
LANGUAGE = "nl"  # Options: "nl" or "en" (currently not used)

# Initialize data loader and get variable label
data_loader = data_io.DataLoader(verbose=False)
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
print("=" * 80)


# === STEP 1 ========================================================================================================
"""get data"""
from utils.verbose_reporter import VerboseReporter

step_name = "data"
force_recalc = FORCE_RECALCULATE_ALL or FORCE_STEP == step_name

verbose_reporter = VerboseReporter(VERBOSE)
data_loader = data_io.DataLoader(verbose=VERBOSE)

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
from utils.verbose_reporter import VerboseReporter

step_name = "preprocessed"
force_recalc = FORCE_RECALCULATE_ALL or FORCE_STEP == step_name
verbose_reporter = VerboseReporter(VERBOSE)

if not force_recalc and cache_manager.is_cache_valid(filename, step_name):
    preprocessed_text = cache_manager.load_from_cache(filename, step_name, models.PreprocessModel)
    verbose_reporter.summary("PREPROCESSED RESPONSES FROM CACHE", {f"Input: {len(raw_text_list)} responses → Output": f"{len(preprocessed_text)} responses", "Overall success rate": f"{(len(preprocessed_text) / len(raw_text_list) * 100):.1f}%"})
else: 
    verbose_reporter.section_header("PREPROCESSING PHASE")
  
    text_normalizer       = textNormalizer.TextNormalizer(verbose=VERBOSE)
    spell_checker         = spellChecker.SpellChecker(verbose=VERBOSE)
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
force_recalc     = FORCE_RECALCULATE_ALL or FORCE_STEP == step_name


if not force_recalc and cache_manager.is_cache_valid(filename, step_name):
    encoded_text = cache_manager.load_from_cache(filename, step_name, models.DescriptiveModel)
    filtering_rate = 100 * (1-(len(encoded_text)/len(preprocessed_text)))
    verbose_reporter.summary("SEGMENTED RESPONSES FROM CAHCE", {f"Input: {len(preprocessed_text)} responses → Output: {len(encoded_text)} coded responses": "", "Filtering rate": f"{filtering_rate:.1f}%"})

else: 
    verbose_reporter.section_header("SEGMENTATION & DESCRIPTION PHASE")
    
    start_time            = time.time()
    grader                = qualityFilter.Grader(preprocessed_text, var_lab, verbose=VERBOSE)
    graded_text           = grader.grade()
    grading_summary       = grader.summary()
    filtered_text         = grader.filter()
    encoder               = segmentDescriber.SegmentDescriber(verbose=VERBOSE)
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
        verbose=VERBOSE)
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
from utils import labeller

step_name = "labels"
force_recalc = FORCE_RECALCULATE_ALL or FORCE_STEP == step_name

if not force_recalc and cache_manager.is_cache_valid(filename, step_name):
    labeled_results = cache_manager.load_from_cache(filename, step_name, models.LabelModel)
    print(f"Loaded {len(labeled_results)} items from cache for step: {step_name}")
else:
    start_time = time.time()
    labeller_config = labeller.LabellerConfig()
    label_generator = labeller.Labeller(config=labeller_config)
    labeled_results = label_generator.run_pipeline(cluster_results, var_lab)
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    cache_manager.save_to_cache(labeled_results, filename, step_name, elapsed_time)
    print(f"\n'Get labels' completed in {elapsed_time:.2f} seconds.")

# Count unique clusters for display
unique_clusters = set()
if cluster_results:
    for result in cluster_results:
        for segment in result.response_segment:
            if segment.micro_cluster:
                cluster_id = list(segment.micro_cluster.keys())[0]
                unique_clusters.add(cluster_id)

# Load cached theme summaries if available for better display
cached_theme_summaries = cache_manager.load_from_cache(filename, "theme_summaries", labeller.ThemeSummary)

# Display results using the labeller's display function
labeller.Labeller.display_hierarchical_results(labeled_results, cached_theme_summaries, len(unique_clusters))



