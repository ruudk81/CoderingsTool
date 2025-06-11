import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

import os
import pandas as pd
import numpy as np
import pyreadstat
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import io
from openpyxl.drawing.image import Image
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils.dataframe import dataframe_to_rows
import xlsxwriter

import models
from .verboseReporter import VerboseReporter
from .dataLoader import DataLoader
from config import ExportConfig, DEFAULT_EXPORT_CONFIG

class ResultsExporter:
    """
    Export pipeline results to SPSS (.sav) and Excel formats
    
    This class handles:
    1. Creating codes columns for each respondent (themes and concepts)
    2. Adding codes to original SPSS file and saving as new file
    3. Creating comprehensive Excel report with multiple tabs
    
    Modified to support 2-level hierarchy (Themes → Concepts)
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
            respondent_codes, hierarchical_structure, filename, id_column, var_name, export_dir)
        
        # Export to Excel  
        excel_path = self._export_to_excel(
            respondent_codes, hierarchical_structure, filename, var_name, export_dir)
        
        self.verbose_reporter.stat_line("✅ Results exported successfully")
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
        Create mapping of respondent_id to their assigned codes using binary approach
        
        Creates binary variables for:
        - Each theme (1 if respondent has any segment in theme, 0 otherwise)
        - Each concept (1 if respondent has any segment in concept, 0 otherwise)  
        - Quality filter codes (1 if respondent has that filter code, 0 otherwise)
        """
        respondent_codes = {}
        themes = hierarchical_structure['themes']
        cluster_mappings = {mapping.cluster_id: mapping for mapping in hierarchical_structure['cluster_mappings']}
        
        self.verbose_reporter.step_start("Creating respondent codes mapping (binary approach)")
        
        # Create list of all possible themes and concepts for binary variables
        all_themes = [(theme.theme_id, theme.numeric_id) for theme in themes]
        all_concepts = []
        for theme in themes:
            for concept in theme.topics:  # topics are concepts in 2-level hierarchy
                all_concepts.append((concept.topic_id, concept.numeric_id, theme.theme_id))
        
        self.verbose_reporter.stat_line(f"Creating binary variables for {len(all_themes)} themes and {len(all_concepts)} concepts")
        
        for result in labeled_results:
            respondent_id = result.respondent_id
            
            # Initialize binary codes for this respondent
            codes = {
                'quality_filter': result.quality_filter,
                'quality_filter_code': result.quality_filter_code,
                'themes_present': set(),  # Set of theme IDs present in segments
                'concepts_present': set(),  # Set of concept IDs present in segments
            }
            
            # Check if this respondent was filtered out
            if result.quality_filter:
                # Quality filtered - no themes/concepts present
                codes['themes_present'] = set()
                codes['concepts_present'] = set()
            else:
                # Find all themes/concepts from ALL segments (not just most common)
                theme_concept_pairs = self._find_all_segment_assignments(result, cluster_mappings, themes)
                codes['themes_present'] = {pair[0] for pair in theme_concept_pairs}
                codes['concepts_present'] = {pair[1] for pair in theme_concept_pairs}
            
            respondent_codes[respondent_id] = codes
        
        # Report statistics
        total_respondents = len(respondent_codes)
        filtered_count = sum(1 for codes in respondent_codes.values() if codes['quality_filter'])
        coded_count = total_respondents - filtered_count
        
        # Count respondents with multiple themes/concepts
        multi_theme_count = sum(1 for codes in respondent_codes.values() 
                               if not codes['quality_filter'] and len(codes['themes_present']) > 1)
        multi_concept_count = sum(1 for codes in respondent_codes.values() 
                                 if not codes['quality_filter'] and len(codes['concepts_present']) > 1)
        
        self.verbose_reporter.stat_line(f"Total respondents: {total_respondents}")
        self.verbose_reporter.stat_line(f"Filtered respondents: {filtered_count}")
        self.verbose_reporter.stat_line(f"Coded respondents: {coded_count}")
        self.verbose_reporter.stat_line(f"Respondents with multiple themes: {multi_theme_count}")
        self.verbose_reporter.stat_line(f"Respondents with multiple concepts: {multi_concept_count}")
        
        return respondent_codes
    
    def _find_all_segment_assignments(self, 
                                     result: models.LabelModel,
                                     cluster_mappings: Dict[int, models.ClusterMapping],
                                     themes: List[models.HierarchicalTheme]) -> List[Tuple[str, str]]:
        """
        Find ALL theme/concept assignments for a respondent's segments (not just most common)
        
        Returns:
            List of (theme_id, concept_id) tuples for all segments
        """
        theme_concept_pairs = []
        
        if not result.response_segment:
            return theme_concept_pairs
        
        # Process ALL segments, not just the most common
        for segment in result.response_segment:
            if hasattr(segment, 'initial_cluster') and segment.initial_cluster is not None:
                cluster_id = segment.initial_cluster
                
                if cluster_id in cluster_mappings:
                    mapping = cluster_mappings[cluster_id]
                    theme_id = mapping.theme_id
                    concept_id = mapping.topic_id  # concept stored as topic_id
                    
                    # Verify this theme/concept exists in our structure
                    theme_found = False
                    for theme in themes:
                        if theme.theme_id == theme_id:
                            for concept in theme.topics:
                                if concept.topic_id == concept_id:
                                    theme_concept_pairs.append((theme_id, concept_id))
                                    theme_found = True
                                    break
                            if theme_found:
                                break
        
        return theme_concept_pairs
    
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
                       hierarchical_structure: Dict,
                       filename: str,
                       id_column: str,
                       var_name: str,
                       export_dir: str) -> str:
        """
        Export codes to SPSS using binary variables approach
        Creates binary variables for each theme, concept, and quality filter code
        """
        self.verbose_reporter.step_start("Exporting to SPSS (binary variables)")
        
        # Load original SPSS data
        original_df, meta = self.data_loader.load_sav(filename)
        
        # Get themes and concepts from hierarchical structure
        themes_list = hierarchical_structure['themes']
        
        # Extract themes and concepts for binary variables
        themes = [(theme.theme_id, theme.label) for theme in themes_list]
        concepts = []
        for theme in themes_list:
            for concept in theme.topics:  # topics are concepts in 2-level hierarchy
                concepts.append((concept.topic_id, concept.label, theme.theme_id))
        
        # Create binary columns for themes
        new_columns = {}
        
        # 1. Theme binary variables
        for theme_id, theme_label in themes:
            col_name = f"{var_name}_THEME_{theme_id}"
            new_columns[col_name] = []
        
        # 2. Concept binary variables  
        for concept_id, concept_label, theme_id in concepts:
            # Use format like Q1_CONCEPT_1_2 for theme 1, concept 2
            col_name = f"{var_name}_CONCEPT_{concept_id.replace('.', '_')}"
            new_columns[col_name] = []
        
        # 3. Quality filter binary variables
        quality_filter_columns = {
            f"{var_name}_USER_MISSING_97": [],
            f"{var_name}_SYSTEM_MISSING_98": [],
            f"{var_name}_NO_ANSWER_99": []
        }
        new_columns.update(quality_filter_columns)
        
        self.verbose_reporter.stat_line(f"Creating {len(new_columns)} binary variables")
        
        # Map codes to original data
        for _, row in original_df.iterrows():
            respondent_id = int(row[id_column])
            
            if respondent_id in respondent_codes:
                codes = respondent_codes[respondent_id]
                
                # Set theme binary variables
                for theme_id, theme_label in themes:
                    col_name = f"{var_name}_THEME_{theme_id}"
                    value = 1 if theme_id in codes['themes_present'] else 0
                    new_columns[col_name].append(value)
                
                # Set concept binary variables
                for concept_id, concept_label, theme_id in concepts:
                    col_name = f"{var_name}_CONCEPT_{concept_id.replace('.', '_')}"
                    value = 1 if concept_id in codes['concepts_present'] else 0
                    new_columns[col_name].append(value)
                
                # Set quality filter binary variables
                if codes['quality_filter']:
                    filter_code = codes['quality_filter_code']
                    new_columns[f"{var_name}_USER_MISSING_97"].append(1 if filter_code == 99999997 else 0)
                    new_columns[f"{var_name}_SYSTEM_MISSING_98"].append(1 if filter_code == 99999998 else 0)
                    new_columns[f"{var_name}_NO_ANSWER_99"].append(1 if filter_code == 99999999 else 0)
                else:
                    # Not filtered - all quality filter variables = 0
                    new_columns[f"{var_name}_USER_MISSING_97"].append(0)
                    new_columns[f"{var_name}_SYSTEM_MISSING_98"].append(0)
                    new_columns[f"{var_name}_NO_ANSWER_99"].append(0)
            else:
                # Respondent not in analysis - set all theme/concept to 0, system missing to 1
                for theme_id, theme_label in themes:
                    col_name = f"{var_name}_THEME_{theme_id}"
                    new_columns[col_name].append(0)
                
                for concept_id, concept_label, theme_id in concepts:
                    col_name = f"{var_name}_CONCEPT_{concept_id.replace('.', '_')}"
                    new_columns[col_name].append(0)
                
                # Mark as system missing
                new_columns[f"{var_name}_USER_MISSING_97"].append(0)
                new_columns[f"{var_name}_SYSTEM_MISSING_98"].append(1)
                new_columns[f"{var_name}_NO_ANSWER_99"].append(0)
        
        # Add new columns to dataframe
        for col_name, values in new_columns.items():
            original_df[col_name] = values
        
        # Create output filename
        base_name = Path(filename).stem
        output_filename = f"{base_name}{self.config.spss_suffix}.sav"
        output_path = os.path.join(export_dir, output_filename)
        
        # Save to SPSS format
        pyreadstat.write_sav(original_df, output_path)
        
        self.verbose_reporter.stat_line(f"Added {len(new_columns)} binary variables")
        self.verbose_reporter.stat_line(f"  - {len(themes)} theme variables")
        self.verbose_reporter.stat_line(f"  - {len(concepts)} concept variables") 
        self.verbose_reporter.stat_line(f"  - 3 quality filter variables")
        self.verbose_reporter.stat_line(f"SPSS file saved: {output_filename}")
        
        return output_path
    
    def _export_to_excel(self,
                        respondent_codes: Dict[int, Dict[str, Any]],
                        hierarchical_structure: Dict,
                        filename: str,
                        var_name: str,
                        export_dir: str) -> str:
        """
        Export comprehensive Excel report with embedded visualizations
        """
        self.verbose_reporter.step_start("Exporting to Excel with visualizations")
        
        # Create output filename
        base_name = Path(filename).stem
        output_filename = f"{base_name}_{var_name}{self.config.excel_suffix}.xlsx"
        output_path = os.path.join(export_dir, output_filename)
        
        # Use xlsxwriter for better chart support
        workbook = xlsxwriter.Workbook(output_path)
        
        try:
            # Tab 1: Codebook
            if self.config.enable_codebook_tab:
                self._create_enhanced_codebook_tab(workbook, hierarchical_structure)
            
            # Tab 2: Hierarchy with visual dendrogram
            if self.config.enable_dendrogram_tab:
                self._create_enhanced_dendrogram_tab(workbook, hierarchical_structure, export_dir)
            
            # Tab 3: Frequencies with embedded charts
            if self.config.enable_frequency_tab:
                self._create_enhanced_frequency_tab(workbook, respondent_codes, hierarchical_structure, export_dir)
            
            # Tab 4: Wordcloud with embedded images
            if self.config.enable_wordcloud_tab:
                self._create_enhanced_wordcloud_tab(workbook, respondent_codes, hierarchical_structure, export_dir)
        
        finally:
            workbook.close()
        
        self.verbose_reporter.stat_line(f"Excel file with visualizations saved: {output_filename}")
        return output_path
    
    def _create_enhanced_codebook_tab(self, workbook, hierarchical_structure: Dict):
        """Create enhanced codebook tab with styled formatting for 2-level hierarchy"""
        themes = hierarchical_structure['themes']
        
        # Create worksheet
        worksheet = workbook.add_worksheet('Codebook')
        
        # Define formats
        header_format = workbook.add_format({
            'bold': True,
            'font_size': 12,
            'bg_color': '#366092',
            'font_color': 'white',
            'align': 'center',
            'valign': 'vcenter',
            'border': 1
        })
        
        theme_format = workbook.add_format({
            'bold': True,
            'font_size': 11,
            'bg_color': '#D9E1F2',
            'border': 1,
            'align': 'left'
        })
        
        concept_format = workbook.add_format({
            'font_size': 10,
            'bg_color': '#F2F2F2',
            'border': 1,
            'align': 'left',
            'indent': 1
        })
        
        # Headers for 2-level hierarchy
        headers = ['Level', 'ID', 'Numeric_ID', 'Label', 'Description', 'Parent_ID', 'Full_Path']
        for col, header in enumerate(headers):
            worksheet.write(0, col, header, header_format)
        
        # Data
        row = 1
        for theme in themes:
            # Theme row
            worksheet.write(row, 0, 'Theme', theme_format)
            worksheet.write(row, 1, theme.theme_id, theme_format)
            worksheet.write(row, 2, theme.numeric_id, theme_format)
            worksheet.write(row, 3, theme.label, theme_format)
            worksheet.write(row, 4, theme.description, theme_format)
            worksheet.write(row, 5, '', theme_format)
            worksheet.write(row, 6, theme.label, theme_format)
            row += 1
            
            for topic in theme.topics:  # Topics are actually concepts in 2-level hierarchy
                # Concept row
                worksheet.write(row, 0, 'Concept', concept_format)
                worksheet.write(row, 1, topic.topic_id, concept_format)
                worksheet.write(row, 2, topic.numeric_id, concept_format)
                worksheet.write(row, 3, topic.label, concept_format)
                worksheet.write(row, 4, topic.description, concept_format)
                worksheet.write(row, 5, theme.theme_id, concept_format)
                worksheet.write(row, 6, f"{theme.label} > {topic.label}", concept_format)
                row += 1
        
        # Adjust column widths
        worksheet.set_column('A:A', 8)   # Level
        worksheet.set_column('B:B', 10)  # ID
        worksheet.set_column('C:C', 12)  # Numeric_ID
        worksheet.set_column('D:D', 25)  # Label
        worksheet.set_column('E:E', 40)  # Description
        worksheet.set_column('F:F', 12)  # Parent_ID
        worksheet.set_column('G:G', 50)  # Full_Path
    
    def _create_enhanced_dendrogram_tab(self, workbook, hierarchical_structure: Dict, export_dir: str):
        """Create dendrogram tab with visual hierarchy for 2-level structure"""
        themes = hierarchical_structure['themes']
        
        # Create worksheet
        worksheet = workbook.add_worksheet('Hierarchy')
        
        # Create tree diagram using matplotlib
        fig, ax = plt.subplots(1, 1, figsize=(self.config.chart_width, self.config.chart_height))
        fig.patch.set_facecolor('white')
        
        # Create hierarchical tree structure for 2 levels
        y_pos = 0
        y_positions = {}
        
        for theme_idx, theme in enumerate(themes):
            # Theme level
            theme_y = y_pos
            y_positions[theme.theme_id] = theme_y
            
            # Draw theme box
            ax.text(0, theme_y, theme.label, fontsize=14, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
            
            concept_start_y = y_pos
            for concept_idx, concept in enumerate(theme.topics):  # Topics are concepts
                y_pos -= 1
                concept_y = y_pos
                y_positions[concept.topic_id] = concept_y
                
                # Draw concept box
                ax.text(1, concept_y, concept.label, fontsize=12,
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgreen', alpha=0.6))
                
                # Draw line from theme to concept
                ax.plot([0.4, 0.9], [theme_y, concept_y], 'k-', alpha=0.5)
            
            y_pos -= 1  # Space between themes
        
        ax.set_xlim(-0.5, 2)
        ax.set_ylim(y_pos - 1, 1)
        ax.set_title('Hierarchical Structure: Themes → Concepts', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # Save chart
        chart_path = os.path.join(export_dir, 'hierarchy_chart.png')
        plt.tight_layout()
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Add structured data table
        header_format = workbook.add_format({
            'bold': True, 'bg_color': '#366092', 'font_color': 'white',
            'align': 'center', 'border': 1
        })
        
        headers = ['Theme', 'Concept', 'Numeric_ID', 'Level']
        for col, header in enumerate(headers):
            worksheet.write(0, col, header, header_format)
        
        # Data rows
        row = 1
        for theme in themes:
            worksheet.write(row, 0, theme.label, workbook.add_format({'bold': True, 'bg_color': '#D9E1F2'}))
            worksheet.write(row, 1, '', workbook.add_format({'bg_color': '#D9E1F2'}))
            worksheet.write(row, 2, theme.numeric_id, workbook.add_format({'bg_color': '#D9E1F2'}))
            worksheet.write(row, 3, 'Theme', workbook.add_format({'bg_color': '#D9E1F2'}))
            row += 1
            
            for concept in theme.topics:  # Topics are concepts
                worksheet.write(row, 0, '', workbook.add_format({'bg_color': '#F2F2F2'}))
                worksheet.write(row, 1, concept.label, workbook.add_format({'bg_color': '#F2F2F2'}))
                worksheet.write(row, 2, concept.numeric_id, workbook.add_format({'bg_color': '#F2F2F2'}))
                worksheet.write(row, 3, 'Concept', workbook.add_format({'bg_color': '#F2F2F2'}))
                row += 1
        
        # Insert chart image
        try:
            worksheet.insert_image('F2', chart_path, {'x_scale': 0.8, 'y_scale': 0.8})
        except Exception as e:
            self.verbose_reporter.stat_line(f"Warning: Could not insert hierarchy chart: {e}")
        
        # Adjust column widths
        worksheet.set_column('A:D', 20)
        worksheet.set_column('E:P', 25)  # Space for chart
    
    def _create_enhanced_frequency_tab(self, workbook, respondent_codes: Dict, hierarchical_structure: Dict, export_dir: str):
        """Create frequency analysis tab with embedded bar charts for 2-level hierarchy"""
        themes = hierarchical_structure['themes']
        
        # Create worksheet
        worksheet = workbook.add_worksheet('Frequencies')
        
        # Count frequencies for each level using binary approach
        theme_counts = {}
        concept_counts = {}
        
        for respondent_id, codes in respondent_codes.items():
            # Count themes present (binary approach)
            for theme_id in codes['themes_present']:
                theme_counts[theme_id] = theme_counts.get(theme_id, 0) + 1
            
            # Count concepts present (binary approach)  
            for concept_id in codes['concepts_present']:
                concept_counts[concept_id] = concept_counts.get(concept_id, 0) + 1
        
        total_respondents = len(respondent_codes)
        
        # Prepare data for charts
        theme_data = []
        concept_data = []
        
        for theme in themes:
            count = theme_counts.get(theme.theme_id, 0)
            percentage = (count / total_respondents) * 100 if total_respondents > 0 else 0
            if count > 0:  # Only include non-zero frequencies
                theme_data.append((theme.label, count, percentage))
        
        for theme in themes:
            for concept in theme.topics:  # Topics are concepts
                count = concept_counts.get(concept.topic_id, 0)
                percentage = (count / total_respondents) * 100 if total_respondents > 0 else 0
                if count > 0:
                    concept_data.append((concept.label, count, percentage))
        
        # Create charts
        chart_paths = []
        
        # Theme frequency chart
        if theme_data:
            fig, ax = plt.subplots(figsize=(12, 6))
            labels, counts, percentages = zip(*sorted(theme_data, key=lambda x: x[1], reverse=True))
            bars = ax.bar(range(len(labels)), counts, color='skyblue', alpha=0.8)
            ax.set_xlabel('Themes', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title('Theme Frequencies', fontsize=14, fontweight='bold')
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels([label[:20] + '...' if len(label) > 20 else label for label in labels], 
                             rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, count, pct in zip(bars, counts, percentages):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            theme_chart_path = os.path.join(export_dir, 'theme_frequencies.png')
            plt.savefig(theme_chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            chart_paths.append(('Theme Frequencies', theme_chart_path))
        
        # Concept frequency chart (top 20)
        if concept_data:
            fig, ax = plt.subplots(figsize=(14, 8))
            sorted_concepts = sorted(concept_data, key=lambda x: x[1], reverse=True)[:20]
            labels, counts, percentages = zip(*sorted_concepts)
            bars = ax.bar(range(len(labels)), counts, color='lightgreen', alpha=0.8)
            ax.set_xlabel('Concepts', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title('Top 20 Concept Frequencies', fontsize=14, fontweight='bold')
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels([label[:25] + '...' if len(label) > 25 else label for label in labels], 
                             rotation=45, ha='right')
            
            for bar, count, pct in zip(bars, counts, percentages):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            concept_chart_path = os.path.join(export_dir, 'concept_frequencies.png')
            plt.savefig(concept_chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            chart_paths.append(('Concept Frequencies', concept_chart_path))
        
        # Add data table
        header_format = workbook.add_format({
            'bold': True, 'bg_color': '#366092', 'font_color': 'white',
            'align': 'center', 'border': 1
        })
        
        headers = ['Level', 'ID', 'Label', 'Frequency', 'Percentage']
        for col, header in enumerate(headers):
            worksheet.write(0, col, header, header_format)
        
        # Add all frequency data
        row = 1
        all_freq_data = []
        
        # Add theme data
        for theme in themes:
            count = theme_counts.get(theme.theme_id, 0)
            percentage = (count / total_respondents) * 100 if total_respondents > 0 else 0
            all_freq_data.append(['Theme', theme.theme_id, theme.label, count, percentage])
        
        # Add concept data
        for theme in themes:
            for concept in theme.topics:  # Topics are concepts
                count = concept_counts.get(concept.topic_id, 0)
                percentage = (count / total_respondents) * 100 if total_respondents > 0 else 0
                all_freq_data.append(['Concept', concept.topic_id, concept.label, count, percentage])
        
        # Write data
        for data_row in all_freq_data:
            for col, value in enumerate(data_row):
                if col == 4:  # Percentage column
                    worksheet.write(row, col, f"{value:.1f}%")
                else:
                    worksheet.write(row, col, value)
            row += 1
        
        # Insert charts
        chart_row = row + 3
        for i, (title, chart_path) in enumerate(chart_paths):
            try:
                worksheet.write(chart_row, 0, title, workbook.add_format({'bold': True, 'font_size': 14}))
                worksheet.insert_image(chart_row + 1, 0, chart_path, {'x_scale': 0.7, 'y_scale': 0.7})
                chart_row += 35  # Space for next chart
            except Exception as e:
                self.verbose_reporter.stat_line(f"Warning: Could not insert chart {title}: {e}")
        
        # Adjust column widths
        worksheet.set_column('A:E', 20)
    
    def _create_enhanced_wordcloud_tab(self, workbook, respondent_codes: Dict, hierarchical_structure: Dict, export_dir: str):
        """Create wordcloud tab with embedded wordcloud images for each theme"""
        themes = hierarchical_structure['themes']
        
        # Create worksheet
        worksheet = workbook.add_worksheet('Wordcloud_Data')
        
        # Add data table first
        header_format = workbook.add_format({
            'bold': True, 'bg_color': '#366092', 'font_color': 'white',
            'align': 'center', 'border': 1
        })
        
        headers = ['Theme', 'Concept_Label', 'Frequency', 'Weight']
        for col, header in enumerate(headers):
            worksheet.write(0, col, header, header_format)
        
        wordcloud_data = []
        theme_wordclouds = {}
        
        row = 1
        for theme in themes:
            theme_concept_frequencies = {}
            
            for concept in theme.topics:  # Topics are concepts
                count = sum(1 for codes in respondent_codes.values() 
                          if concept.topic_id in codes['concepts_present'])
                
                if count > 0:
                    theme_concept_frequencies[concept.label] = count
                    
                    # Add to data table
                    worksheet.write(row, 0, theme.label)
                    worksheet.write(row, 1, concept.label)
                    worksheet.write(row, 2, count)
                    worksheet.write(row, 3, count)  # Weight same as frequency
                    row += 1
            
            # Create wordcloud for this theme if it has data
            if theme_concept_frequencies:
                theme_wordclouds[theme.label] = theme_concept_frequencies
        
        # Generate wordcloud images
        wordcloud_paths = []
        
        for theme_label, concept_frequencies in theme_wordclouds.items():
            if len(concept_frequencies) > 0:
                try:
                    # Create wordcloud
                    wordcloud = WordCloud(
                        width=self.config.wordcloud_width,
                        height=self.config.wordcloud_height,
                        max_words=self.config.max_wordcloud_words,
                        background_color='white',
                        colormap='viridis',
                        relative_scaling=0.5,
                        random_state=42
                    ).generate_from_frequencies(concept_frequencies)
                    
                    # Create matplotlib figure
                    fig, ax = plt.subplots(figsize=(12, 8))
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis('off')
                    ax.set_title(f'Wordcloud: {theme_label}', fontsize=16, fontweight='bold', pad=20)
                    
                    # Save wordcloud
                    wordcloud_path = os.path.join(export_dir, f'wordcloud_{theme_label.replace(" ", "_")}.png')
                    plt.tight_layout()
                    plt.savefig(wordcloud_path, dpi=300, bbox_inches='tight', facecolor='white')
                    plt.close()
                    
                    wordcloud_paths.append((theme_label, wordcloud_path))
                    
                except Exception as e:
                    self.verbose_reporter.stat_line(f"Warning: Could not create wordcloud for {theme_label}: {e}")
        
        # Insert wordcloud images
        if wordcloud_paths:
            chart_row = row + 3
            for theme_label, wordcloud_path in wordcloud_paths:
                try:
                    worksheet.write(chart_row, 0, f'Wordcloud: {theme_label}', 
                                  workbook.add_format({'bold': True, 'font_size': 14}))
                    worksheet.insert_image(chart_row + 1, 0, wordcloud_path, 
                                         {'x_scale': 0.6, 'y_scale': 0.6})
                    chart_row += 30  # Space for next wordcloud
                except Exception as e:
                    self.verbose_reporter.stat_line(f"Warning: Could not insert wordcloud for {theme_label}: {e}")
        
        # Adjust column widths
        worksheet.set_column('A:D', 20)
        worksheet.set_column('E:O', 25)  # Space for wordclouds