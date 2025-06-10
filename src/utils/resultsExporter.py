import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

import os
import pandas as pd
import numpy as np
import pyreadstat
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import json

import models
from .verboseReporter import VerboseReporter
from .dataLoader import DataLoader
from config import ExportConfig, DEFAULT_EXPORT_CONFIG

class ResultsExporter:
    """
    Export pipeline results to SPSS (.sav) and Excel formats
    
    This class handles:
    1. Creating codes columns for each respondent (themes, topics, codes)
    2. Adding codes to original SPSS file and saving as new file
    3. Creating comprehensive Excel report with multiple tabs
    """
    
    def __init__(self, config: ExportConfig = None, verbose: bool = True):
        self.config = config or DEFAULT_EXPORT_CONFIG
        self.verbose = verbose
        self.verbose_reporter = VerboseReporter(verbose)
        self.data_loader = DataLoader(verbose=False)
        
    def export_results(self, 
                      labeled_results: List[models.LabelModel],
                      filename: str,
                      id_column: str,
                      var_name: str) -> Dict[str, str]:
        """
        Export complete results to both SPSS and Excel formats
        
        Args:
            labeled_results: List of LabelModel objects from step 6
            filename: Original SPSS filename
            id_column: ID column name in SPSS file
            var_name: Variable name that was analyzed
            
        Returns:
            Dict with paths to created files
        """
        self.verbose_reporter.section_header("EXPORTING RESULTS")
        
        # Extract data structure from results
        if not labeled_results or not labeled_results[0].themes:
            raise ValueError("No hierarchical structure found in labeled results")
            
        first_result = labeled_results[0]
        hierarchical_structure = {
            'themes': first_result.themes,
            'cluster_mappings': first_result.cluster_mappings
        }
        
        # Create respondent codes mapping
        respondent_codes = self._create_respondent_codes_mapping(labeled_results, hierarchical_structure)
        
        # Create export directory
        export_dir = self._setup_export_directory(filename, var_name)
        
        # Export to SPSS
        spss_path = self._export_to_spss(
            respondent_codes, filename, id_column, var_name, export_dir)
        
        # Export to Excel  
        excel_path = self._export_to_excel(
            respondent_codes, hierarchical_structure, filename, var_name, export_dir)
        
        self.verbose_reporter.stat_line(f"✅ Results exported successfully")
        self.verbose_reporter.stat_line(f"📊 SPSS file: {spss_path}")
        self.verbose_reporter.stat_line(f"📈 Excel file: {excel_path}")
        
        return {
            'spss_file': spss_path,
            'excel_file': excel_path,
            'export_directory': export_dir
        }
    
    def _create_respondent_codes_mapping(self, 
                                       labeled_results: List[models.LabelModel],
                                       hierarchical_structure: Dict) -> Dict[int, Dict[str, Any]]:
        """
        Create mapping of respondent_id to their assigned codes
        
        For each respondent, determine:
        - If quality_filter=True: use quality_filter_code for all three columns
        - If quality_filter=False: determine theme/topic/code from cluster assignments
        """
        respondent_codes = {}
        themes = hierarchical_structure['themes']
        cluster_mappings = {mapping.cluster_id: mapping for mapping in hierarchical_structure['cluster_mappings']}
        
        self.verbose_reporter.step_start("Creating respondent codes mapping")
        
        for result in labeled_results:
            respondent_id = result.respondent_id
            
            # Initialize codes for this respondent
            codes = {
                'theme_code': None,
                'topic_code': None, 
                'code_code': None,
                'quality_filter': result.quality_filter,
                'quality_filter_code': result.quality_filter_code
            }
            
            # Check if this respondent was filtered out
            if result.quality_filter:
                # Use quality filter code for all three columns
                filter_code = result.quality_filter_code
                codes.update({
                    'theme_code': filter_code,
                    'topic_code': filter_code,
                    'code_code': filter_code
                })
            else:
                # Find assigned codes from cluster mappings
                assigned_codes = self._find_respondent_assigned_codes(result, cluster_mappings, themes)
                codes.update(assigned_codes)
            
            respondent_codes[respondent_id] = codes
        
        # Report statistics
        total_respondents = len(respondent_codes)
        filtered_count = sum(1 for codes in respondent_codes.values() if codes['quality_filter'])
        coded_count = total_respondents - filtered_count
        
        self.verbose_reporter.stat_line(f"Total respondents: {total_respondents}")
        self.verbose_reporter.stat_line(f"Filtered respondents: {filtered_count}")
        self.verbose_reporter.stat_line(f"Coded respondents: {coded_count}")
        
        return respondent_codes
    
    def _find_respondent_assigned_codes(self, 
                                      result: models.LabelModel,
                                      cluster_mappings: Dict[int, models.ClusterMapping],
                                      themes: List[models.HierarchicalTheme]) -> Dict[str, float]:
        """
        Find the assigned hierarchical codes for a respondent based on their segments
        """
        assigned_codes = {
            'theme_code': None,
            'topic_code': None,
            'code_code': None
        }
        
        if not result.response_segment:
            return assigned_codes
        
        # Get all cluster assignments for this respondent's segments
        segment_clusters = []
        for segment in result.response_segment:
            if hasattr(segment, 'initial_cluster') and segment.initial_cluster is not None:
                segment_clusters.append(segment.initial_cluster)
        
        if not segment_clusters:
            return assigned_codes
        
        # Find the most frequent cluster assignment (or take first if tied)
        most_common_cluster = max(set(segment_clusters), key=segment_clusters.count)
        
        # Look up the hierarchical assignment for this cluster
        if most_common_cluster in cluster_mappings:
            mapping = cluster_mappings[most_common_cluster]
            
            # Convert IDs to numeric codes
            theme_id = mapping.theme_id
            topic_id = mapping.topic_id  
            code_id = mapping.code_id
            
            # Find numeric codes from hierarchical structure
            for theme in themes:
                if theme.theme_id == theme_id:
                    assigned_codes['theme_code'] = theme.numeric_id
                    
                    for topic in theme.topics:
                        if topic.topic_id == topic_id:
                            assigned_codes['topic_code'] = topic.numeric_id
                            
                            for code in topic.codes:
                                if code.code_id == code_id:
                                    assigned_codes['code_code'] = code.numeric_id
                                    break
                            break
                    break
        
        return assigned_codes
    
    def _setup_export_directory(self, filename: str, var_name: str) -> str:
        """Create and return export directory path"""
        # Get base data directory
        base_data_dir = self.data_loader.data_dir
        export_dir = self.config.get_export_dir(base_data_dir)
        
        # Create subdirectory if enabled
        if self.config.create_subdirs:
            base_filename = Path(filename).stem
            subdir_name = f"{base_filename}_{var_name}"
            export_dir = os.path.join(export_dir, subdir_name)
        
        # Create directory
        Path(export_dir).mkdir(parents=True, exist_ok=True)
        
        self.verbose_reporter.stat_line(f"Export directory: {export_dir}")
        return export_dir
    
    def _export_to_spss(self, 
                       respondent_codes: Dict[int, Dict[str, Any]],
                       filename: str,
                       id_column: str,
                       var_name: str,
                       export_dir: str) -> str:
        """
        Export codes to SPSS by adding columns to original data
        """
        self.verbose_reporter.step_start("Exporting to SPSS")
        
        # Load original SPSS data
        original_df, meta = self.data_loader.load_sav(filename)
        
        # Create new columns for codes
        new_columns = {
            f"{var_name}_THEME": [],
            f"{var_name}_TOPIC": [],
            f"{var_name}_CODE": []
        }
        
        # Map codes to original data
        for _, row in original_df.iterrows():
            respondent_id = int(row[id_column])
            
            if respondent_id in respondent_codes:
                codes = respondent_codes[respondent_id]
                new_columns[f"{var_name}_THEME"].append(codes['theme_code'])
                new_columns[f"{var_name}_TOPIC"].append(codes['topic_code'])
                new_columns[f"{var_name}_CODE"].append(codes['code_code'])
            else:
                # Respondent not in analysis - mark as system missing
                new_columns[f"{var_name}_THEME"].append(99999998)
                new_columns[f"{var_name}_TOPIC"].append(99999998)
                new_columns[f"{var_name}_CODE"].append(99999998)
        
        # Add new columns to dataframe
        for col_name, values in new_columns.items():
            original_df[col_name] = values
        
        # Create output filename
        base_name = Path(filename).stem
        output_filename = f"{base_name}{self.config.spss_suffix}.sav"
        output_path = os.path.join(export_dir, output_filename)
        
        # Save to SPSS format
        pyreadstat.write_sav(original_df, output_path)
        
        self.verbose_reporter.stat_line(f"Added {len(new_columns)} code columns")
        self.verbose_reporter.stat_line(f"SPSS file saved: {output_filename}")
        
        return output_path
    
    def _export_to_excel(self,
                        respondent_codes: Dict[int, Dict[str, Any]],
                        hierarchical_structure: Dict,
                        filename: str,
                        var_name: str,
                        export_dir: str) -> str:
        """
        Export comprehensive Excel report with multiple tabs
        """
        self.verbose_reporter.step_start("Exporting to Excel")
        
        # Create output filename
        base_name = Path(filename).stem
        output_filename = f"{base_name}_{var_name}{self.config.excel_suffix}.xlsx"
        output_path = os.path.join(export_dir, output_filename)
        
        # Create Excel writer
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            
            # Tab 1: Codebook
            if self.config.enable_codebook_tab:
                self._create_codebook_tab(writer, hierarchical_structure)
            
            # Tab 2: Dendrogram (hierarchical structure view)
            if self.config.enable_dendrogram_tab:
                self._create_dendrogram_tab(writer, hierarchical_structure)
            
            # Tab 3: Frequency charts data
            if self.config.enable_frequency_tab:
                self._create_frequency_tab(writer, respondent_codes, hierarchical_structure)
            
            # Tab 4: Wordcloud data (frequencies for each theme)
            if self.config.enable_wordcloud_tab:
                self._create_wordcloud_tab(writer, respondent_codes, hierarchical_structure)
        
        self.verbose_reporter.stat_line(f"Excel file saved: {output_filename}")
        return output_path
    
    def _create_codebook_tab(self, writer, hierarchical_structure: Dict):
        """Create codebook tab with full hierarchical structure"""
        themes = hierarchical_structure['themes']
        
        codebook_data = []
        
        for theme in themes:
            # Add theme row
            codebook_data.append({
                'Level': 'Theme',
                'ID': theme.theme_id,
                'Numeric_ID': theme.numeric_id,
                'Label': theme.label,
                'Description': theme.description,
                'Parent_ID': '',
                'Full_Path': theme.label
            })
            
            for topic in theme.topics:
                # Add topic row
                codebook_data.append({
                    'Level': 'Topic',
                    'ID': topic.topic_id,
                    'Numeric_ID': topic.numeric_id,
                    'Label': topic.label,
                    'Description': topic.description,
                    'Parent_ID': theme.theme_id,
                    'Full_Path': f"{theme.label} > {topic.label}"
                })
                
                for code in topic.codes:
                    # Add code row
                    codebook_data.append({
                        'Level': 'Code',
                        'ID': code.code_id,
                        'Numeric_ID': code.numeric_id,
                        'Label': code.label,
                        'Description': code.description,
                        'Parent_ID': topic.topic_id,
                        'Full_Path': f"{theme.label} > {topic.label} > {code.label}"
                    })
        
        codebook_df = pd.DataFrame(codebook_data)
        codebook_df.to_excel(writer, sheet_name='Codebook', index=False)
    
    def _create_dendrogram_tab(self, writer, hierarchical_structure: Dict):
        """Create dendrogram tab showing hierarchical relationships"""
        themes = hierarchical_structure['themes']
        
        # Create a structured view for dendrogram visualization
        dendrogram_data = []
        
        for theme in themes:
            theme_row = [theme.label, '', '', theme.numeric_id, 'Theme']
            dendrogram_data.append(theme_row)
            
            for topic in theme.topics:
                topic_row = ['', topic.label, '', topic.numeric_id, 'Topic']
                dendrogram_data.append(topic_row)
                
                for code in topic.codes:
                    code_row = ['', '', code.label, code.numeric_id, 'Code']
                    dendrogram_data.append(code_row)
        
        dendrogram_df = pd.DataFrame(dendrogram_data, 
                                   columns=['Theme', 'Topic', 'Code', 'Numeric_ID', 'Level'])
        dendrogram_df.to_excel(writer, sheet_name='Hierarchy', index=False)
    
    def _create_frequency_tab(self, writer, respondent_codes: Dict, hierarchical_structure: Dict):
        """Create frequency analysis tab"""
        themes = hierarchical_structure['themes']
        
        # Count frequencies for each level
        theme_counts = {}
        topic_counts = {}
        code_counts = {}
        
        for respondent_id, codes in respondent_codes.items():
            # Count themes
            theme_code = codes['theme_code']
            if theme_code is not None:
                theme_counts[theme_code] = theme_counts.get(theme_code, 0) + 1
            
            # Count topics
            topic_code = codes['topic_code']
            if topic_code is not None:
                topic_counts[topic_code] = topic_counts.get(topic_code, 0) + 1
            
            # Count codes
            code_code = codes['code_code']
            if code_code is not None:
                code_counts[code_code] = code_counts.get(code_code, 0) + 1
        
        # Create frequency dataframes
        total_respondents = len(respondent_codes)
        
        # Theme frequencies
        theme_freq_data = []
        for theme in themes:
            count = theme_counts.get(theme.numeric_id, 0)
            percentage = (count / total_respondents) * 100 if total_respondents > 0 else 0
            theme_freq_data.append({
                'Level': 'Theme',
                'ID': theme.theme_id,
                'Label': theme.label,
                'Frequency': count,
                'Percentage': percentage
            })
        
        # Topic frequencies
        topic_freq_data = []
        for theme in themes:
            for topic in theme.topics:
                count = topic_counts.get(topic.numeric_id, 0)
                percentage = (count / total_respondents) * 100 if total_respondents > 0 else 0
                topic_freq_data.append({
                    'Level': 'Topic',
                    'ID': topic.topic_id,
                    'Label': topic.label,
                    'Frequency': count,
                    'Percentage': percentage
                })
        
        # Code frequencies
        code_freq_data = []
        for theme in themes:
            for topic in theme.topics:
                for code in topic.codes:
                    count = code_counts.get(code.numeric_id, 0)
                    percentage = (count / total_respondents) * 100 if total_respondents > 0 else 0
                    code_freq_data.append({
                        'Level': 'Code',
                        'ID': code.code_id,
                        'Label': code.label,
                        'Frequency': count,
                        'Percentage': percentage
                    })
        
        # Combine all frequencies
        all_freq_data = theme_freq_data + topic_freq_data + code_freq_data
        freq_df = pd.DataFrame(all_freq_data)
        freq_df.to_excel(writer, sheet_name='Frequencies', index=False)
    
    def _create_wordcloud_tab(self, writer, respondent_codes: Dict, hierarchical_structure: Dict):
        """Create wordcloud data tab with code frequencies by theme"""
        themes = hierarchical_structure['themes']
        
        wordcloud_data = []
        
        for theme in themes:
            # Get all codes under this theme
            theme_code_frequencies = {}
            
            for topic in theme.topics:
                for code in topic.codes:
                    # Count how many respondents were assigned to this code
                    count = sum(1 for codes in respondent_codes.values() 
                              if codes['code_code'] == code.numeric_id)
                    
                    if count > 0:
                        theme_code_frequencies[code.label] = count
            
            # Create wordcloud entries for this theme
            for code_label, frequency in theme_code_frequencies.items():
                wordcloud_data.append({
                    'Theme': theme.label,
                    'Code_Label': code_label,
                    'Frequency': frequency,
                    'Weight': frequency  # For wordcloud sizing
                })
        
        if wordcloud_data:
            wordcloud_df = pd.DataFrame(wordcloud_data)
            wordcloud_df.to_excel(writer, sheet_name='Wordcloud_Data', index=False)
        else:
            # Create empty dataframe with proper columns
            empty_df = pd.DataFrame(columns=['Theme', 'Code_Label', 'Frequency', 'Weight'])
            empty_df.to_excel(writer, sheet_name='Wordcloud_Data', index=False)