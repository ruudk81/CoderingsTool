"""start in src"""

# ===  MODULES ========================================================================================================
import sys
import os
import time
import random
import nest_asyncio
nest_asyncio.apply()


# === MODELS ========================================================================================================
project_paths = [
    r'C:\Users\rkn\Python_apps\Coderingstool\src',
    r'C:\Users\rkn\Python_apps\Coderingstool\src\modules',
    r'C:\Users\rkn\Python_apps\Coderingstool\src\modules\utils']

for path in project_paths:
    sys.path.insert(0, path)
    
import models

# === CONFIG ========================================================================================================
from modules.utils import data_io, csvHandler

filename             = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
id_column            = "DLNMID"
var_name             = "Q20"
csv_handler          = csvHandler.CsvHandler()
filepath             = csv_handler.get_filepath(filename, 'data')
data_loader          = data_io.DataLoader()
var_lab              = data_loader.get_varlab(filename = filename, var_name = var_name)

# === STEP 1 ========================================================================================================
"""get data"""
if  os.path.exists(filepath):
    raw_text_list    = csv_handler.load_from_csv(filename, 'data', models.ResponseModel)
else:    
    start_time       = time.time()
    raw_text_df      = data_loader.get_variable_with_IDs(filename = filename, id_column = id_column,var_name = var_name)
    raw_unstructued  = list(zip([int(id_int) for id_int in raw_text_df[id_column].tolist()], raw_text_df[var_name].tolist()))
    raw_text_list    = [models.ResponseModel(respondent_id=resp_id, response=resp if resp is not None else "" ) for resp_id, resp in raw_unstructued]
    end_time         = time.time()
    elapsed_time     = end_time - start_time
    
    csv_handler.save_to_csv(raw_text_list, filename, 'data' )  
    print(f"\n\n'Import data' completed in {elapsed_time:.2f} seconds.\n")


# === STEP 2 ========================================================================================================
"""preprocess data"""
from modules.utils import textNormalizer, spellChecker, textFinalizer

if  os.path.exists(filepath):
    preprocessed_text     = csv_handler.load_from_csv(filename, 'preprocessed', models.PreprocessModel)
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
    
    csv_handler.save_to_csv(preprocessed_text, filename, 'preprocessed' )  
    print(f"\n\n'Preprocessing phase' completed in {elapsed_time:.2f} seconds.\n")

#debug print
idx = 1 
for response in preprocessed_text:
    print(f"{idx}. {response.response}")
    idx += 1
        

# === STEP 3 ========================================================================================================
"""describe and segment data"""
from modules.utils import qualityFilter, segmentDescriber

if  os.path.exists(filepath):
    encoded_text          = csv_handler.load_from_csv(filename, 'segmented_descriptions', models.DescriptiveModel)
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
  
    csv_handler.save_to_csv(encoded_text, filename, 'segmented_descriptions' )  
    print(f"\n\n'Preprocessing phase' completed in {elapsed_time:.2f} seconds.\n")

    print("\nSummary:")
    for key, value in grading_summary.items(): 
        print(f"{key}: {value}")
        
    print("\nMeaningless responses")
    random_graded_text = random.sample(graded_text, min(20, len(graded_text)))
    for text in random_graded_text:
        #print(text.respondent_id)
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
from modules.utils import embedder  

if  os.path.exists(filepath):
    embedded_text           = csv_handler.load_from_csv(filename, 'embeddings', models.EmbeddingsModel)
else:
    start_time              = time.time()
    get_embeddings          = embedder.Embedder()
    input_data              = [item.to_model(models.EmbeddingsModel) for item in encoded_text]
    code_embeddings         = get_embeddings.get_code_embeddings(input_data)
    description_embeddings  = get_embeddings.get_description_embeddings(input_data, var_lab)
    embedded_text           = get_embeddings.combine_embeddings(code_embeddings, description_embeddings)
    end_time                = time.time()
    elapsed_time            = end_time - start_time
    
    csv_handler.save_to_csv(embedded_text, filename, 'embeddings' )  
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
        print(f"    Code embedding: {code.code_embedding}")
        print(f"    Description embedding: {code.description_embedding}")
    print("\n")


# === STEP 5 ========================================================================================================
"get clusters"
from modules.utils import clusterer  

if  os.path.exists(filepath):
    embedded_text           = csv_handler.load_from_csv(filename, 'embeddings', models.ClusterModel)
else:
    start_time              = time.time()
    clusterer               = clusterer.ClusterGenerator(input_list=embedded_text, var_lab=var_lab, embedding_type="code", verbose=True)
    clusterer.run_pipeline()
    cluster_results         = clusterer.to_cluster_model()
    end_time                = time.time()
    elapsed_time            = end_time - start_time

    csv_handler.save_to_csv(cluster_results, filename, 'clusters')
    print(f"\n'Get clusters' completed in {elapsed_time:.2f} seconds.")

# debug print
from collections import defaultdict
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

print(f"Found {len(meta_cluster_counts)} meta-clusters in results")
for meta_id, count in sorted(meta_cluster_counts.items()):
    print(f"\n📚 Meta-cluster {meta_id}: {count} items")
  
        
    sample_size = min(5, len(meta_cluster_codes[meta_id]))
    for i in range(sample_size):
        print(f"  - {meta_cluster_codes[meta_id][i]}")    
    
    
    sample_size = min(5, len(meta_cluster_descriptions[meta_id]))
    for i in range(sample_size):
        print(f"  - {meta_cluster_descriptions[meta_id][i]}")


