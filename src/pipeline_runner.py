"""
Pipeline Runner for Streamlit Integration
Provides callable functions for each pipeline step that can be used in Streamlit app
"""

import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

import time
import asyncio
import pandas as pd
import nest_asyncio
from typing import List, Dict, Optional, Any, Tuple, Union
import streamlit as st

nest_asyncio.apply()

# === MODELS ========================================================================================================
import models

# === CONFIG ========================================================================================================
from utils import dataLoader
from utils.cacheManager import CacheManager
from config import (
    CacheConfig, 
    ModelConfig,
    SpellCheckConfig,
    QualityFilterConfig, 
    SegmentationConfig,
    EmbeddingConfig,
    HDBSCANConfig,
    CodeDesignerConfig,
    CodeAssignmentConfig
)

# === UTILS ========================================================================================================
from utils.verboseReporter import VerboseReporter
from utils.cached_resources import get_openai_client, get_tiktoken_encoding, get_spacy_nlp_conditional, get_embedder_for_provider
from utils.session_manager import get_session_manager
from utils.streamlit_debug import DebugCapture, VerboseCapture, PromptCapture, SampleGenerator, StepSamplers

# Cached resource functions for heavy pipeline components
@st.cache_resource
def _get_cached_embedder(config_hash: str, model_config_hash: str, provider: str):
    """Cache embedder instances for session-wide reuse"""
    Embedder = _get_embedder()
    return Embedder(
        config=None,  # Will be set when called
        model_config=None,  # Will be set when called  
        provider=provider,
        verbose=False
    )

@st.cache_resource
def _get_cached_clusterer(config_hash: str):
    """Cache clusterer class for session-wide reuse"""
    return _get_clusterer()

@st.cache_resource  
def _get_cached_theme_identifier():
    """Cache theme identifier class for session-wide reuse"""
    return _get_theme_identifier()

# Cached data functions for processing results
@st.cache_data(show_spinner="Loading SPSS data...")
def _cache_spss_data(filename: str, id_column: str, var_name: str, encoding: str = None):
    """Cache SPSS file parsing results for fast reruns with encoding support (single variable)"""
    from utils.dataLoader import DataLoader
    data_loader = DataLoader(verbose=False)
    
    try:
        # Try with specified encoding or let DataLoader handle encoding detection
        result = data_loader.get_variable_with_IDs(filename=filename, id_column=id_column, var_name=var_name, encoding=encoding)
        
        # Store successful encoding info for user feedback
        successful_encoding = data_loader.get_last_successful_encoding()
        if successful_encoding:
            st.session_state['last_encoding_used'] = successful_encoding
            st.session_state['encoding_success_message'] = f"✅ File loaded successfully with {successful_encoding} encoding"
        
        return result
        
    except ValueError as e:
        error_msg = str(e)
        if "encoding" in error_msg.lower() or "byte sequence" in error_msg.lower():
            # This is an encoding error, provide helpful message
            st.error(f"🔴 Encoding Error: {error_msg}")
            st.info("💡 Try specifying a different encoding in the advanced options, or contact support if the issue persists.")
        raise

@st.cache_data(show_spinner="Loading and merging SPSS data...")
def _cache_multiple_spss_data(filename: str, id_column: str, var_names: tuple, 
                              merge_strategy: str = "concatenate", separator: str = " ",
                              skip_empty: bool = True, encoding: str = None):
    """Cache multiple SPSS variables merged results for fast reruns with encoding support"""
    from utils.dataLoader import DataLoader
    data_loader = DataLoader(verbose=False)
    
    try:
        # Convert tuple back to list for processing
        var_names_list = list(var_names)
        
        # Try with specified encoding or let DataLoader handle encoding detection
        result = data_loader.get_multiple_variables_with_IDs(
            filename=filename, 
            id_column=id_column, 
            var_names=var_names_list,
            merge_strategy=merge_strategy,
            separator=separator,
            skip_empty=skip_empty,
            encoding=encoding
        )
        
        # Store successful encoding info for user feedback
        successful_encoding = data_loader.get_last_successful_encoding()
        if successful_encoding:
            st.session_state['last_encoding_used'] = successful_encoding
            st.session_state['encoding_success_message'] = f"✅ File loaded successfully with {successful_encoding} encoding"
        
        return result
        
    except ValueError as e:
        error_msg = str(e)
        if "encoding" in error_msg.lower() or "byte sequence" in error_msg.lower():
            # This is an encoding error, provide helpful message
            st.error(f"🔴 Encoding Error: {error_msg}")
            st.info("💡 Try specifying a different encoding in the advanced options, or contact support if the issue persists.")
        raise

@st.cache_data(show_spinner="Loading cached embeddings...")
def _cache_embedding_results(content_hash: str, provider: str, model_name: str):
    """Cache embedding generation results by content hash"""
    # This will be populated by the actual embedding process
    # Returns None if not cached, which signals to generate new embeddings
    return None

@st.cache_data(show_spinner="Loading cached spell corrections...")  
def _cache_spell_correction_results(content_hash: str, model_name: str, config_hash: str):
    """Cache spell correction results by content and config hash"""
    # This will be populated by the actual spell correction process
    # Returns None if not cached, which signals to run spell correction
    return None

# Lazy loading functions to improve startup performance
def _get_text_normalizer():
    from utils import textNormalizer
    return textNormalizer

def _get_spell_checker():
    from utils import spellChecker
    return spellChecker

def _get_text_finalizer():
    from utils import textFinalizer
    return textFinalizer

def _get_quality_filter():
    from utils import qualityFilter
    return qualityFilter

def _get_idea_extractor():
    from utils import ideaExtractor
    return ideaExtractor

def _get_embedder():
    from utils.embedder import Embedder
    return Embedder

def _get_clusterer():
    from utils.clusterer import Clusterer
    return Clusterer

def _get_code_generator():
    from utils import codeGenerator
    return codeGenerator

def _get_theme_identifier():
    from utils.themeIdentifier import ThemeIdentifier
    return ThemeIdentifier

def _get_code_assigner():
    from utils import codeAssigner
    return codeAssigner

def _get_code_assignment_exporter():
    from utils.codeAssignmentExporter import CodeAssignmentExporter
    return CodeAssignmentExporter

def _get_code_generator_reasoning_results():
    from utils.codeGenerator import CodeGeneratorReasoningResults
    return CodeGeneratorReasoningResults

def _get_theme_organizer_reasoning():
    from utils.themeOrganizerReasoning import ThemeOrganizerReasoning
    return ThemeOrganizerReasoning

class StreamlitPipelineRunner:
    """Pipeline runner optimized for Streamlit with session state management"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.cache_config = CacheConfig()
        self.cache_manager = CacheManager(self.cache_config)
        self.data_loader = dataLoader.DataLoader(verbose=False)
        
    def create_verbose_reporter(self, streamlit_container=None) -> VerboseReporter:
        """Create verbose reporter that can optionally stream to Streamlit"""
        # TODO: Implement streaming to Streamlit container
        return VerboseReporter(self.verbose)
    
    def step_1_load_data(self, filename: str, id_column: str, var_name: str = None, var_names: list = None,
                        force_recalc: bool = False, 
                        streamlit_container=None, encoding: str = None) -> List[models.ResponseModel]:
        """Step 1: Load data from SPSS file (single or multiple variables)"""
        
        step_name = "data"
        verbose_reporter = self.create_verbose_reporter(streamlit_container)
        
        if streamlit_container:
            if var_names and len(var_names) > 1:
                streamlit_container.text(f"🔄 Loading and merging {len(var_names)} variables from SPSS file...")
            else:
                streamlit_container.text("🔄 Loading data from SPSS file...")
        
        if not force_recalc and self.cache_manager.is_cache_valid(filename, step_name):
            raw_text_list = self.cache_manager.load_from_cache(filename, step_name, models.ResponseModel)
            verbose_reporter.summary("DATA FROM CACHE", {"Input": f"{len(raw_text_list)} responses"})
        else:
            verbose_reporter.section_header("DATA LOADING SUMMARY")
            start_time = time.time()
            
            # Use provided encoding or fall back to session state, then to auto-detect (None)
            if encoding is None:
                encoding = st.session_state.get('file_encoding', 'auto')
                encoding = None if encoding == 'auto' else encoding
            
            # Determine loading mode: single or multiple variables
            if var_names and len(var_names) > 1:
                # Multiple variables mode - get merge configuration from session state
                merge_config = st.session_state.get('merge_config', {})
                merge_strategy = merge_config.get('strategy', 'concatenate')
                separator = merge_config.get('separator', ' ')
                skip_empty = merge_config.get('skip_empty', True)
                
                # Use tuple for caching (lists are not hashable)
                var_names_tuple = tuple(var_names)
                raw_text_df = _cache_multiple_spss_data(
                    filename, id_column, var_names_tuple, 
                    merge_strategy, separator, skip_empty, encoding
                )
                text_column = 'merged_text'
            else:
                # Single variable mode (backward compatibility)
                if var_names and len(var_names) == 1:
                    var_name = var_names[0]
                elif not var_name:
                    raise ValueError("Either var_name or var_names must be provided")
                
                raw_text_df = _cache_spss_data(filename, id_column, var_name, encoding)
                text_column = var_name
            
            raw_unstructured = list(zip([int(id_int) for id_int in raw_text_df[id_column].tolist()], raw_text_df[text_column].tolist()))
            raw_text_list = []
            
            # Structure data NaN=system missing; Numeric=undefined user missing; String=response 
            for resp_id, resp in raw_unstructured:
                if pd.isna(resp):
                    response_type = 'nan'
                elif isinstance(resp, (int, float)):
                    response_type = 'numeric'
                elif isinstance(resp, str):
                    response_type = 'string'
                else:
                    response_type = 'unknown'
                raw_text_list.append(models.ResponseModel(respondent_id=resp_id, response=resp, response_type=response_type))
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.cache_manager.save_to_cache(raw_text_list, filename, step_name, elapsed_time)
            
            if streamlit_container:
                streamlit_container.success(f"✅ Loaded {len(raw_text_list)} responses in {elapsed_time:.2f}s")
        
        return raw_text_list
    
    def step_2_preprocess(self, raw_text_list: List[models.ResponseModel], filename: str, var_lab: str,
                         model_config: Optional[ModelConfig] = None,
                         spellcheck_config: Optional[SpellCheckConfig] = None,
                         force_recalc: bool = False,
                         streamlit_container=None,
                         debug_capture: Optional[DebugCapture] = None) -> List[models.PreprocessedModel]:
        """Step 2: Preprocess text data"""
        
        step_name = "preprocessed"
        verbose_reporter = self.create_verbose_reporter(streamlit_container)
        
        if streamlit_container:
            streamlit_container.text("🔄 Preprocessing text data...")
        
        if not force_recalc and self.cache_manager.is_cache_valid(filename, step_name):
            preprocessed_text = self.cache_manager.load_from_cache(filename, step_name, models.PreprocessedModel)
            verbose_reporter.summary("PREPROCESSED RESPONSES FROM CACHE", {"Input": f"{len(raw_text_list)} responses"})
        else:
            verbose_reporter.section_header("PREPROCESSING PHASE")
            start_time = time.time()
            
            # Initialize utils (lazy loaded)
            textNormalizer = _get_text_normalizer()
            spellChecker = _get_spell_checker()
            textFinalizer = _get_text_finalizer()
            
            text_normalizer = textNormalizer.TextNormalizer(verbose=self.verbose)
            spell_checker = spellChecker.SpellChecker(
                config=spellcheck_config,
                model_config=model_config,
                verbose=self.verbose
            )
            text_finalizer = textFinalizer.TextFinalizer(verbose=self.verbose)
            
            # Preprocess strings
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
                
                # Debug capture: spell correction samples
                if debug_capture and debug_capture.show_samples:
                    if hasattr(spell_checker, 'correction_examples') and spell_checker.correction_examples:
                        sample_gen = SampleGenerator(debug_capture, step_name)
                        sample_gen.generate_samples(
                            spell_checker.correction_examples,
                            "spell_corrections", 
                            StepSamplers.sample_spell_corrections
                        )
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
                        if item.response in [99999997, 99999998, 99999999]:
                            desc_item.quality_filter_code = int(item.response)
                            desc_item.quality_filter = True
                        else:
                            desc_item.quality_filter_code = None
                            desc_item.quality_filter = None
                    elif isinstance(item.response, str):
                        if item.response.strip() == '':
                            desc_item.quality_filter_code = 99999999   
                            desc_item.quality_filter = True
                        else:
                            desc_item.quality_filter_code = None
                            desc_item.quality_filter = None
                    preprocessed_text.append(desc_item)
                else:
                    preprocessed_text.append(models.PreprocessedModel(
                        respondent_id=original.respondent_id,
                        response='<NA>',
                        response_type='nan',
                        quality_filter_code=99999998,
                        quality_filter=True))
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.cache_manager.save_to_cache(preprocessed_text, filename, step_name, elapsed_time)
            
            if streamlit_container:
                streamlit_container.success(f"✅ Preprocessed {len(preprocessed_text)} responses in {elapsed_time:.2f}s")
        
        return preprocessed_text
    
    def step_3_quality_filter(self, preprocessed_text: List[models.PreprocessedModel], filename: str, var_lab: str,
                             model_config: Optional[ModelConfig] = None,
                             quality_filter_config: Optional[QualityFilterConfig] = None,
                             force_recalc: bool = False,
                             streamlit_container=None) -> List[models.QualityFilteredModel]:
        """Step 3: Quality filter responses"""
        
        step_name = "quality_filter"
        verbose_reporter = self.create_verbose_reporter(streamlit_container)
        
        if streamlit_container:
            streamlit_container.text("🔄 Filtering low-quality responses...")
        
        if not force_recalc and self.cache_manager.is_cache_valid(filename, step_name):
            quality_filtered_text = self.cache_manager.load_from_cache(filename, step_name, models.QualityFilteredModel)
            verbose_reporter.summary("QUALIFIED RESPONSES FROM CACHE", {"Input": f"{len(preprocessed_text)} responses"})
        else:
            verbose_reporter.section_header("QUALITY FILTERING PHASE")
            start_time = time.time()
            
            # Lazy load quality filter
            qualityFilter = _get_quality_filter()
            grader = qualityFilter.Grader(
                preprocessed_text, 
                var_lab, 
                config=quality_filter_config,
                model_config=model_config,
                verbose=self.verbose
            )
            quality_filtered_text = grader.grade()
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.cache_manager.save_to_cache(quality_filtered_text, filename, step_name, elapsed_time)
            
            if streamlit_container:
                filtered_count = len([item for item in quality_filtered_text if not item.quality_filter])
                streamlit_container.success(f"✅ Quality filtering completed. {filtered_count} meaningful responses retained in {elapsed_time:.2f}s")
        
        return quality_filtered_text
    
    def step_4_extract_ideas(self, quality_filtered_text: List[models.QualityFilteredModel], filename: str, var_lab: str,
                           model_config: Optional[ModelConfig] = None,
                           segmentation_config: Optional[SegmentationConfig] = None,
                           force_recalc: bool = False,
                           streamlit_container=None,
                           debug_capture: Optional[DebugCapture] = None) -> List[models.IdeasExtractedModel]:
        """Step 4: Extract ideas from responses"""
        
        step_name = "extracted_ideas"
        verbose_reporter = self.create_verbose_reporter(streamlit_container)
        
        if streamlit_container:
            streamlit_container.text("🔄 Extracting ideas from responses...")
        
        if not force_recalc and self.cache_manager.is_cache_valid(filename, step_name):
            encoded_text = self.cache_manager.load_from_cache(filename, step_name, models.IdeasExtractedModel)
            segments = sum(item.idea_count for item in encoded_text)
            verbose_reporter.summary("IDEAS FROM CACHE", {"Input": f"{len(encoded_text)} responses", "Output": f"{segments} segments"})
        else:
            verbose_reporter.section_header("EXTRACTION OF IDEAS EXPRESSED PHASE")
            start_time = time.time()
            
            filtered_text = [item for item in quality_filtered_text if not item.quality_filter]
            
            # Lazy load idea extractor
            ideaExtractor = _get_idea_extractor()
            encoder = ideaExtractor.IdeaExtractor(
                responses=filtered_text,
                var_lab=var_lab,
                config=segmentation_config,
                model_config=model_config,
                verbose=self.verbose
            )
            encoded_text = encoder.extract()
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.cache_manager.save_to_cache(encoded_text, filename, step_name, elapsed_time)
            
            # Debug capture: idea extraction samples
            if debug_capture and debug_capture.show_samples and encoded_text:
                sample_gen = SampleGenerator(debug_capture, step_name)
                sample_gen.generate_samples(
                    encoded_text,
                    "idea_extractions",
                    StepSamplers.sample_idea_extractions
                )
            
            if streamlit_container:
                segments = sum(item.idea_count for item in encoded_text)
                streamlit_container.success(f"✅ Extracted {segments} ideas from {len(filtered_text)} responses in {elapsed_time:.2f}s")
        
        return encoded_text
    
    def step_5_generate_embeddings(self, encoded_text: List[models.IdeasExtractedModel], filename: str, var_lab: str,
                                  model_config: Optional[ModelConfig] = None,
                                  embedding_config: Optional[EmbeddingConfig] = None,
                                  provider: str = "openai",
                                  force_recalc: bool = False,
                                  streamlit_container=None) -> List[models.EmbeddingsModel]:
        """Step 5: Generate embeddings"""
        
        step_name = "embeddings"
        verbose_reporter = self.create_verbose_reporter(streamlit_container)
        
        if streamlit_container:
            streamlit_container.text(f"🔄 Generating embeddings using {provider}...")
        
        if not force_recalc and self.cache_manager.is_cache_valid(filename, step_name):
            embedded_text = self.cache_manager.load_from_cache(filename, step_name, models.EmbeddingsModel)
            total_embeddings = sum(len(resp.response_ideas) for resp in embedded_text if resp.response_ideas)
            verbose_reporter.summary("EMBEDDINGS FROM CACHE", {"Input": f"{len(encoded_text)} responses", "Total embeddings": f"{total_embeddings}"})
        else:
            verbose_reporter.section_header("EMBEDDING GENERATION PHASE")
            start_time = time.time()
            
            # Lazy load embedder
            Embedder = _get_embedder()
            get_embeddings = Embedder(
                config=embedding_config,
                model_config=model_config,
                provider=provider,
                verbose=self.verbose
            )
            input_data = [item.to_model(models.EmbeddingsModel) for item in encoded_text]
            embedded_text = get_embeddings.get_embeddings_with_tracking(input_data, var_lab)
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.cache_manager.save_to_cache(embedded_text, filename, step_name, elapsed_time)
            
            if streamlit_container:
                total_embeddings = sum(len(resp.response_ideas) for resp in embedded_text if resp.response_ideas)
                streamlit_container.success(f"✅ Generated {total_embeddings} embeddings in {elapsed_time:.2f}s")
        
        return embedded_text
    
    def step_6_cluster(self, embedded_text: List[models.EmbeddingsModel], filename: str,
                      hdbscan_config: Optional[HDBSCANConfig] = None,
                      force_recalc: bool = False,
                      streamlit_container=None,
                      debug_capture: Optional[DebugCapture] = None) -> List[models.ClusterModel]:
        """Step 6: Cluster embeddings"""
        
        step_name = "initial_clusters"
        verbose_reporter = self.create_verbose_reporter(streamlit_container)
        
        if streamlit_container:
            streamlit_container.text("🔄 Clustering similar responses...")
        
        if not force_recalc and self.cache_manager.is_cache_valid(filename, step_name):
            initial_cluster_results = self.cache_manager.load_from_cache(filename, step_name, models.ClusterModel)
            cluster_ids = set([segment.initial_cluster for result in initial_cluster_results for segment in result.response_ideas if segment.initial_cluster is not None])
            verbose_reporter.summary("CLUSTERS FROM CACHE", {"Input": f"{len(embedded_text)} responses", "Clusters": f"{len(cluster_ids)}"})
        else:
            verbose_reporter.section_header("INITIAL CLUSTERING PHASE")
            start_time = time.time()
            
            # Lazy load clusterer
            Clusterer = _get_clusterer()
            clusterer = Clusterer(embedded_text, hdbscan_config=hdbscan_config, verbose=self.verbose)
            clusterer.run()
            initial_cluster_results = clusterer.to_cluster_model()
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.cache_manager.save_to_cache(initial_cluster_results, filename, step_name, elapsed_time)
            
            # Debug capture: cluster content samples
            if debug_capture and debug_capture.show_samples and initial_cluster_results:
                sample_gen = SampleGenerator(debug_capture, step_name)
                sample_gen.generate_samples(
                    initial_cluster_results,
                    "cluster_contents",
                    StepSamplers.sample_cluster_contents
                )
            
            if streamlit_container:
                cluster_ids = set([segment.initial_cluster for result in initial_cluster_results for segment in result.response_ideas if segment.initial_cluster is not None])
                streamlit_container.success(f"✅ Created {len(cluster_ids)} clusters in {elapsed_time:.2f}s")
        
        return initial_cluster_results
    
    def step_7_generate_codebook(self, initial_cluster_results: List[models.ClusterModel], filename: str, var_name: str, var_lab: str,
                               model_config: Optional[ModelConfig] = None,
                               code_designer_config: Optional[CodeDesignerConfig] = None,
                               use_speculative_starter_codes: bool = False,
                               force_recalc: bool = False,
                               streamlit_container=None) -> Tuple[models.CodebookModel, Optional[Any]]:
        """Step 7: Generate codebook"""
        
        step_name = "codebook_generation"
        verbose_reporter = self.create_verbose_reporter(streamlit_container)
        
        if streamlit_container:
            streamlit_container.text("🔄 Generating codebook from clusters...")
        
        reasoning_results = None
        if not force_recalc and self.cache_manager.is_cache_valid(filename, step_name):
            codebook_models = self.cache_manager.load_from_cache(filename, step_name, models.CodebookModel)
            if codebook_models and len(codebook_models) > 0:
                codebook_main = codebook_models[0]
                verbose_reporter.summary("CODEBOOK FROM CACHE", {"Total codes": len(codebook_main.codes)})
                
                # Try to load reasoning from cache too
                try:
                    CodeGeneratorReasoningResults = _get_code_generator_reasoning_results()
                    reasoning_models = self.cache_manager.load_from_cache(
                        filename, f"{step_name}_reasoning", CodeGeneratorReasoningResults
                    )
                    if reasoning_models and len(reasoning_models) > 0:
                        reasoning_results = reasoning_models[0]
                except Exception:
                    pass  # Reasoning not available from cache
            else:
                codebook_main = models.CodebookModel(codes=[], source_variable=var_name)
        else:
            verbose_reporter.section_header("CODEBOOK GENERATION PHASE")
            start_time = time.time()
            
            # Generate starter codes if requested
            starter_codes = []
            if use_speculative_starter_codes:
                from utils import speculativeStarterCodes
                starter_generator = speculativeStarterCodes.SpeculativeStarterCodes(
                    var_lab=var_lab, 
                    verbose=self.verbose
                )
                starter_codes = starter_generator.generate()
            
            # Generate codebook using inductive code generator (lazy loaded)
            codeGenerator = _get_code_generator()
            generator = codeGenerator.InductiveCodeGenerator(
                cluster_results=initial_cluster_results,
                starter_codes=starter_codes,
                var_lab=var_lab,
                config=code_designer_config,
                model_config=model_config,
                verbose=True,
                verbose_detailed=False
            )
            results = generator.generate()
            reasoning_results = results  # Store reasoning results for return
            
            codebook_entries = []
            CodeGeneratorReasoningResults = _get_code_generator_reasoning_results()
            if results and isinstance(results, CodeGeneratorReasoningResults):
                final_codebook = results.codebook
                for item in final_codebook:
                    codebook_entry = models.CodebookEntry(
                        code=item['code'],
                        definition=item['definition'],
                        source_cluster=item['source_cluster_id']
                    )
                    codebook_entries.append(codebook_entry)
            
            codebook_main = models.CodebookModel(
                codes=codebook_entries,
                generation_metadata={
                    "methodology": "Inductive codebook generation from clusters",
                    "starter_codes_count": len(starter_codes) if starter_codes else 0,
                    "total_codes_generated": len(codebook_entries),
                },
                source_variable=var_name
            )
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.cache_manager.save_to_cache([codebook_main], filename, step_name, elapsed_time)
            
            # Always cache reasoning results for export consistency
            if reasoning_results:
                try:
                    self.cache_manager.save_to_cache([reasoning_results], filename, f"{step_name}_reasoning", elapsed_time)
                except Exception as e:
                    print(f"⚠️ Warning: Failed to cache reasoning results: {e}")
            
            if streamlit_container:
                streamlit_container.success(f"✅ Generated {len(codebook_entries)} codes in {elapsed_time:.2f}s")
        
        return codebook_main, reasoning_results
    
    def step_8_identify_themes(self, codebook_main: models.CodebookModel, filename: str, var_name: str, var_lab: str,
                             force_recalc: bool = False,
                             streamlit_container=None) -> models.ThemeEnrichedCodebookModel:
        """Step 8: Identify themes"""
        
        step_name = "theme_identification"
        verbose_reporter = self.create_verbose_reporter(streamlit_container)
        
        if streamlit_container:
            streamlit_container.text("🔄 Identifying themes in codebook...")
        
        if not force_recalc and self.cache_manager.is_cache_valid(filename, step_name):
            theme_enriched_codebooks = self.cache_manager.load_from_cache(filename, step_name, models.ThemeEnrichedCodebookModel)
            if theme_enriched_codebooks and len(theme_enriched_codebooks) > 0:
                theme_enriched_codebook = theme_enriched_codebooks[0]
                verbose_reporter.summary("THEMES FROM CACHE", {"Total codes": len(theme_enriched_codebook.codes)})
            else:
                theme_enriched_codebook = models.ThemeEnrichedCodebookModel(
                    codes=[], source_variable=var_name, themes_summary=[], code_to_theme_mapping={}, theme_methodology="Error loading from cache"
                )
        else:
            verbose_reporter.section_header("THEME IDENTIFICATION PHASE")
            start_time = time.time()
            
            if not codebook_main.codes:
                theme_enriched_codebook = models.ThemeEnrichedCodebookModel(
                    codes=[], source_variable=var_name, themes_summary=[], code_to_theme_mapping={}, theme_methodology="No codes available"
                )
            else:
                codebook = [{"code": entry.code, "definition": entry.definition} for entry in codebook_main.codes]
                
                # Lazy load theme identifier
                ThemeIdentifier = _get_theme_identifier()
                theme_identifier = ThemeIdentifier(
                    codebook=codebook,
                    var_lab=var_lab,
                    verbose=self.verbose
                )
                
                async def run_theme_identification():
                    return await theme_identifier.identify_themes_by_clustering()
                
                result = asyncio.run(run_theme_identification())
                
                # Process theme results
                enriched_entries = []
                code_to_theme_mapping = {}
                themes = result['themes']
                
                # Build code-to-theme mapping
                for theme in themes:
                    theme_name = theme['theme_name']
                    for code_info in theme['codes']:
                        code_name = code_info['code_name']
                        code_to_theme_mapping[code_name] = theme_name
                
                # Enrich codebook entries with theme information
                for entry in codebook_main.codes:
                    theme_name = code_to_theme_mapping.get(entry.code)
                    theme_info = None
                    theme_cluster_id = None
                    is_misc = False
                    
                    if theme_name:
                        theme_name_normalized = theme_name.strip().lower()
                        for theme in themes:
                            if theme['theme_name'].strip().lower() == theme_name_normalized:
                                theme_info = theme.get('theme_description', '')
                                theme_cluster_id = theme.get('cluster_id', -1)
                                is_misc = theme.get('is_miscellaneous', False)
                                break
                    
                    enriched_entry = models.ThemeEnrichedCodebookEntry(
                        code=entry.code,
                        definition=entry.definition,
                        source_cluster=entry.source_cluster,
                        theme=theme_name,
                        theme_description=theme_info,
                        theme_cluster_id=theme_cluster_id,
                        is_miscellaneous=is_misc
                    )
                    enriched_entries.append(enriched_entry)
                
                theme_enriched_codebook = models.ThemeEnrichedCodebookModel(
                    codes=enriched_entries,
                    generation_metadata=codebook_main.generation_metadata,
                    source_variable=codebook_main.source_variable,
                    themes_summary=themes,
                    code_to_theme_mapping=code_to_theme_mapping,
                    theme_methodology=result.get('methodology', 'Clustering-based theme identification')
                )
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.cache_manager.save_to_cache([theme_enriched_codebook], filename, step_name, elapsed_time)
            
            if streamlit_container:
                theme_count = len(set(entry.theme for entry in theme_enriched_codebook.codes if entry.theme))
                streamlit_container.success(f"✅ Identified {theme_count} themes in {elapsed_time:.2f}s")
        
        return theme_enriched_codebook
    
    def step_8b_organize_themes_reasoning(self, codebook_main: models.CodebookModel, filename: str, var_name: str, var_lab: str,
                                        model_config: Optional[ModelConfig] = None,
                                        reasoning_effort: str = "high",
                                        text_verbosity: str = "medium",
                                        force_recalc: bool = False,
                                        streamlit_container=None) -> models.ThemeEnrichedCodebookModel:
        """Step 8b: Organize themes using OpenAI reasoning models (alternative to step 8)"""
        
        step_name = "theme_organization_reasoning"
        verbose_reporter = self.create_verbose_reporter(streamlit_container)
        
        if streamlit_container:
            streamlit_container.text("🔄 Organizing themes with reasoning model...")
        
        if not force_recalc and self.cache_manager.is_cache_valid(filename, step_name):
            theme_enriched_codebooks = self.cache_manager.load_from_cache(filename, step_name, models.ThemeEnrichedCodebookModel)
            if theme_enriched_codebooks and len(theme_enriched_codebooks) > 0:
                theme_enriched_codebook = theme_enriched_codebooks[0]
                verbose_reporter.summary("THEMES FROM CACHE", {"Total codes": len(theme_enriched_codebook.codes)})
            else:
                theme_enriched_codebook = models.ThemeEnrichedCodebookModel(
                    codes=[], source_variable=var_name, themes_summary=[], code_to_theme_mapping={}, 
                    theme_methodology="Error loading from cache"
                )
        else:
            verbose_reporter.section_header("THEME ORGANIZATION WITH REASONING MODEL")
            start_time = time.time()
            
            if not codebook_main.codes:
                theme_enriched_codebook = models.ThemeEnrichedCodebookModel(
                    codes=[], source_variable=var_name, themes_summary=[], code_to_theme_mapping={}, 
                    theme_methodology="No codes available"
                )
            else:
                # Prepare codebook for reasoning model
                codebook = [{"code": entry.code, "definition": entry.definition} for entry in codebook_main.codes]
                
                # Use model from model_config if available, otherwise default
                model_name = "gpt-5"
                if model_config and hasattr(model_config, 'model'):
                    model_name = model_config.model
                
                # Initialize theme organizer (lazy loaded)
                ThemeOrganizerReasoning = _get_theme_organizer_reasoning()
                theme_organizer = ThemeOrganizerReasoning(
                    codebook=codebook,
                    var_lab=var_lab,
                    verbose=self.verbose,
                    model=model_name,
                    reasoning_effort=reasoning_effort,
                    text_verbosity=text_verbosity
                )
                
                async def run_theme_organization():
                    return await theme_organizer.organize_themes_reasoning()
                
                result = asyncio.run(run_theme_organization())
                
                # Process theme results
                enriched_entries = []
                code_to_theme_mapping = result.get('code_to_theme_mapping', {})
                themes = result.get('themes', [])
                
                # Enrich codebook entries with theme information
                for entry in codebook_main.codes:
                    theme_name = code_to_theme_mapping.get(entry.code)
                    theme_info = None
                    theme_cluster_id = None
                    is_misc = False
                    
                    if theme_name:
                        theme_name_normalized = theme_name.strip().lower()
                        for theme in themes:
                            if theme['theme_name'].strip().lower() == theme_name_normalized:
                                theme_info = theme.get('theme_description', '')
                                theme_cluster_id = theme.get('cluster_id', 'reasoning_theme')
                                is_misc = theme.get('is_miscellaneous', False)
                                break
                    
                    enriched_entry = models.ThemeEnrichedCodebookEntry(
                        code=entry.code,
                        definition=entry.definition,
                        source_cluster=entry.source_cluster,
                        theme=theme_name,
                        theme_description=theme_info,
                        theme_cluster_id=theme_cluster_id,
                        is_miscellaneous=is_misc
                    )
                    enriched_entries.append(enriched_entry)
                
                theme_enriched_codebook = models.ThemeEnrichedCodebookModel(
                    codes=enriched_entries,
                    generation_metadata=codebook_main.generation_metadata,
                    source_variable=codebook_main.source_variable,
                    themes_summary=themes,
                    code_to_theme_mapping=code_to_theme_mapping,
                    theme_methodology=result.get('methodology', 'Single-prompt reasoning model organization')
                )
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.cache_manager.save_to_cache([theme_enriched_codebook], filename, step_name, elapsed_time)
            
            if streamlit_container:
                theme_count = len(set(entry.theme for entry in theme_enriched_codebook.codes if entry.theme))
                streamlit_container.success(f"✅ Organized {theme_count} themes using reasoning model in {elapsed_time:.2f}s")
        
        return theme_enriched_codebook
    
    def step_9a_assign_codes(self, initial_cluster_results: List[models.ClusterModel], 
                           theme_enriched_codebook: models.ThemeEnrichedCodebookModel, 
                           filename: str, var_lab: str, method: str = "direct_llm",
                           model_config: Optional[ModelConfig] = None,
                           code_assignment_config: Optional[CodeAssignmentConfig] = None,
                           force_recalc: bool = False,
                           streamlit_container=None,
                           debug_capture: Optional[DebugCapture] = None) -> List[models.CodeAssignedModel]:
        """Step 9a: Assign codes to ideas"""
        
        step_name = "code_assignment_direct" if method == "direct_llm" else "code_assignment"
        verbose_reporter = self.create_verbose_reporter(streamlit_container)
        
        if streamlit_container:
            method_name = "Direct LLM" if method == "direct_llm" else "Embedding Similarity"
            streamlit_container.text(f"🔄 Assigning codes using {method_name}...")
        
        if not force_recalc and self.cache_manager.is_cache_valid(filename, step_name):
            code_assigned_results = self.cache_manager.load_from_cache(filename, step_name, models.CodeAssignedModel)
            total_ideas = sum(len(resp.response_ideas) for resp in code_assigned_results if resp.response_ideas)
            verbose_reporter.summary("CODE ASSIGNMENTS FROM CACHE", {"Input responses": len(code_assigned_results), "Ideas processed": total_ideas})
        else:
            verbose_reporter.section_header("CODE ASSIGNMENT PHASE")
            start_time = time.time()
            
            if not theme_enriched_codebook or not theme_enriched_codebook.codes:
                raise ValueError("No enriched codebook available for code assignment")
            elif not initial_cluster_results:
                raise ValueError("No cluster results available for code assignment")
            
            # Lazy load code assigner
            codeAssigner = _get_code_assigner()
            
            if method == "direct_llm":
                # Direct LLM processing
                code_assigner_instance = codeAssigner.CodeAssigner(
                    cluster_models=initial_cluster_results,
                    codebook=[models.Codebook(
                        code=entry.code, 
                        definition=entry.definition,
                        theme=entry.theme,
                        theme_description=entry.theme_description
                    ) for entry in theme_enriched_codebook.codes],
                    var_lab=var_lab,
                    code_to_theme_mapping=theme_enriched_codebook.code_to_theme_mapping,
                    cached_idea_embeddings=None,
                    config=code_assignment_config,
                    model_config=model_config,
                    verbose=self.verbose
                )
            else:
                # Embedding similarity method
                cached_ideas = codeAssigner.EmbeddingLoader.load_idea_embeddings_from_cache(
                    self.cache_manager, filename
                )
                if not cached_ideas:
                    raise ValueError("No cached idea embeddings found. Run embedding step first.")
                
                code_assigner_instance = codeAssigner.CodeAssigner(
                    cluster_models=[],
                    codebook=[models.Codebook(
                        code=entry.code, 
                        definition=entry.definition,
                        theme=entry.theme,
                        theme_description=entry.theme_description
                    ) for entry in theme_enriched_codebook.codes],
                    var_lab=var_lab,
                    code_to_theme_mapping=theme_enriched_codebook.code_to_theme_mapping,
                    cached_idea_embeddings=cached_ideas,
                    config=code_assignment_config,
                    model_config=model_config,
                    verbose=self.verbose
                )
            
            code_assigned_results = code_assigner_instance.assign()
            
            # Add metadata
            for result in code_assigned_results:
                if not hasattr(result, 'assignment_metadata') or result.assignment_metadata is None:
                    result.assignment_metadata = {}
                result.assignment_metadata.update({
                    "codebook_used": f"{len(theme_enriched_codebook.codes)} codes with themes",
                    "assignment_method": method,
                    "theme_methodology": theme_enriched_codebook.theme_methodology,
                    "assignment_timestamp": start_time
                })
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.cache_manager.save_to_cache(code_assigned_results, filename, step_name, elapsed_time)
            
            # Debug capture: code assignment samples
            if debug_capture and debug_capture.show_samples and code_assigned_results:
                sample_gen = SampleGenerator(debug_capture, step_name)
                sample_gen.generate_samples(
                    code_assigned_results,
                    "code_assignments",
                    StepSamplers.sample_code_assignments
                )
            
            if streamlit_container:
                total_ideas = sum(len(resp.response_ideas) for resp in code_assigned_results if resp.response_ideas)
                total_assignments = sum(len([idea for idea in resp.response_ideas if idea and idea.assigned_codes]) for resp in code_assigned_results if resp.response_ideas)
                streamlit_container.success(f"✅ Assigned codes to {total_assignments}/{total_ideas} ideas in {elapsed_time:.2f}s")
        
        return code_assigned_results
    
    def step_10_export_excel(self, code_assigned_results: List[models.CodeAssignedModel],
                           theme_enriched_codebook: models.ThemeEnrichedCodebookModel,
                           filename: str, var_name: str, export_dir: Optional[str] = None,
                           include_rationale: bool = True,
                           streamlit_container=None) -> str:
        """Step 10: Export results to Excel"""
        
        if streamlit_container:
            streamlit_container.text("🔄 Exporting results to Excel...")
        
        start_time = time.time()
        
        # Lazy load code assignment exporter
        CodeAssignmentExporter = _get_code_assignment_exporter()
        exporter = CodeAssignmentExporter(verbose=self.verbose)
        excel_path = exporter.export_to_excel(
            code_assigned_results=code_assigned_results,
            theme_enriched_codebook=theme_enriched_codebook,
            filename=filename,
            var_name=var_name,
            export_dir=export_dir
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        if streamlit_container:
            streamlit_container.success(f"✅ Excel export completed in {elapsed_time:.2f}s")
            streamlit_container.info(f"📁 File saved to: {excel_path}")
        
        return excel_path
    
    def step_10_export_excel_consistent(self, code_assigned_results: List[models.CodeAssignedModel],
                                        theme_enriched_codebook: models.ThemeEnrichedCodebookModel,
                                        filename: str, var_name: str, export_dir: Optional[str] = None,
                                        reasoning_results: Optional[Any] = None,
                                        streamlit_container=None) -> str:
        """Step 10: Export results to Excel with consistent 15-column format (always tries to include reasoning)"""
        
        if streamlit_container:
            streamlit_container.text("🔄 Exporting results with consistent format...")
        
        verbose_reporter = self.create_verbose_reporter(streamlit_container)
        verbose_reporter.section_header("EXCEL EXPORT WITH CONSISTENT FORMAT")
        
        # Try to get reasoning data from parameter, cache, or fallback to empty reasoning
        final_reasoning_results = reasoning_results
        
        if final_reasoning_results is None:
            try:
                CodeGeneratorReasoningResults = _get_code_generator_reasoning_results()
                reasoning_models = self.cache_manager.load_from_cache(
                    filename, "codebook_generation_reasoning", CodeGeneratorReasoningResults
                )
                if reasoning_models and len(reasoning_models) > 0:
                    final_reasoning_results = reasoning_models[0]
                    verbose_reporter.stat_line("✅ Loaded reasoning data from cache")
            except Exception as e:
                verbose_reporter.warning(f"⚠️ No reasoning data available: {e}")
        else:
            verbose_reporter.stat_line("✅ Using reasoning data from step 7")
        
        # Always use the with_reasoning export for consistent format
        # If no reasoning data available, it will show empty reasoning columns
        CodeAssignmentExporter = _get_code_assignment_exporter()
        exporter = CodeAssignmentExporter(verbose=self.verbose)
        
        if final_reasoning_results:
            output_path = exporter.export_to_excel_with_reasoning(
                code_assigned_results=code_assigned_results,
                theme_enriched_codebook=theme_enriched_codebook,
                reasoning_results=final_reasoning_results,
                filename=filename,
                var_name=var_name,
                export_dir=export_dir
            )
            verbose_reporter.stat_line("📊 Excel with reasoning exported (15 columns)")
        else:
            # Fallback to regular export when no reasoning available
            verbose_reporter.warning("⚠️ No reasoning data available - using regular export format")
            output_path = exporter.export_to_excel(
                code_assigned_results=code_assigned_results,
                theme_enriched_codebook=theme_enriched_codebook,
                filename=filename,
                var_name=var_name,
                export_dir=export_dir
            )
            verbose_reporter.stat_line("📊 Regular Excel exported (12 columns)")
        
        if streamlit_container:
            streamlit_container.success(f"✅ Export completed: {output_path}")
        
        return output_path
    
    def step_10_export_excel_with_reasoning(self, code_assigned_results: List[models.CodeAssignedModel],
                                           theme_enriched_codebook: models.ThemeEnrichedCodebookModel,
                                           filename: str, var_name: str, export_dir: Optional[str] = None,
                                           reasoning_results: Optional[Any] = None,
                                           streamlit_container=None) -> str:
        """Step 10: Export Excel with reasoning data from step 7"""
        
        verbose_reporter = self.create_verbose_reporter(streamlit_container)
        
        if streamlit_container:
            streamlit_container.text("🔄 Exporting results with reasoning data...")
        
        verbose_reporter.section_header("EXCEL EXPORT WITH REASONING PHASE")
        
        # Use passed reasoning_results or try to load from cache
        if reasoning_results is not None:
            verbose_reporter.stat_line("✅ Using reasoning data passed from step 7")
        else:
            try:
                CodeGeneratorReasoningResults = _get_code_generator_reasoning_results()
                reasoning_models = self.cache_manager.load_from_cache(
                    filename, "codebook_generation_reasoning", CodeGeneratorReasoningResults
                )
                if reasoning_models and len(reasoning_models) > 0:
                    reasoning_results = reasoning_models[0]
                    verbose_reporter.stat_line("✅ Loaded step 7 reasoning data from cache")
                else:
                    verbose_reporter.warning("⚠️ No reasoning data found in cache - using regular export")
            except Exception as e:
                verbose_reporter.warning(f"⚠️ Failed to load reasoning data: {e} - using regular export")
        
        # Create exporter (lazy loaded)
        CodeAssignmentExporter = _get_code_assignment_exporter()
        exporter = CodeAssignmentExporter(verbose=self.verbose)
        
        # Export with or without reasoning data
        if reasoning_results:
            output_path = exporter.export_to_excel_with_reasoning(
                code_assigned_results=code_assigned_results,
                theme_enriched_codebook=theme_enriched_codebook,
                reasoning_results=reasoning_results,
                filename=filename,
                var_name=var_name,
                export_dir=export_dir
            )
            verbose_reporter.stat_line("📊 Excel with reasoning exported successfully")
        else:
            # Fallback to regular export
            output_path = exporter.export_to_excel(
                code_assigned_results=code_assigned_results,
                theme_enriched_codebook=theme_enriched_codebook,
                filename=filename,
                var_name=var_name,
                export_dir=export_dir
            )
            verbose_reporter.stat_line("📊 Regular Excel export completed (no reasoning data)")
        
        if streamlit_container:
            streamlit_container.success(f"✅ Export completed: {output_path}")
        
        return output_path

# Global pipeline runner instance for Streamlit
@st.cache_resource
def get_pipeline_runner():
    """Get cached pipeline runner instance"""
    return StreamlitPipelineRunner()