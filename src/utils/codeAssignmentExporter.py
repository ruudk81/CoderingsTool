import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import os
from typing import List, Dict, Optional
from pathlib import Path
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows

# === MODELS ========================================================================================================
import models

# === UTILS ========================================================================================================
from .verboseReporter import VerboseReporter

class CodeAssignmentExporter:
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.verbose_reporter = VerboseReporter(verbose, capture_logging=True)
        
    def export_to_excel(self,
                       code_assigned_results: List[models.CodeAssignedModel],
                       theme_enriched_codebook: models.ThemeEnrichedCodebookModel,
                       filename: str,
                       var_name: str,
                       export_dir: Optional[str] = None) -> str:
        """
        Export code assignment results to Excel with all requested columns:
        - respondent_id
        - idea_id
        - initial_cluster_id (parent cluster)
        - source_cluster_id (sub-cluster)
        - code_label
        - code_description
        - assignment_rationale
        - theme_name
        - theme_description
        """
        
        self.verbose_reporter.section_header("EXPORTING CODE ASSIGNMENTS TO EXCEL")
        
        # Prepare data for export
        export_data = []
        
        # Create comprehensive code mapping from enriched codebook
        code_to_info = {}
        for code_entry in theme_enriched_codebook.codes:
            code_to_info[code_entry.code] = {
                'theme_name': code_entry.theme or 'No Theme',
                'theme_description': code_entry.theme_description or 'No Description',
                'code_description': code_entry.definition,
                'source_cluster': code_entry.source_cluster  # This is the sub-cluster ID like "12-1"
            }
        
        # Process each result
        for result in code_assigned_results:
            respondent_id = result.respondent_id
            
            if result.response_ideas:
                for idea in result.response_ideas:
                    # Process each assigned code
                    if idea.assigned_codes:
                        for code in idea.assigned_codes:
                            code_info = code_to_info.get(code, {
                                'theme_name': 'Unknown Theme',
                                'theme_description': 'Unknown Description',
                                'code_description': 'Unknown Code',
                                'source_cluster': None
                            })
                            
                            # Get source cluster info from the enriched codebook
                            source_cluster = code_info['source_cluster']
                            
                            # Extract parent cluster from source cluster
                            if source_cluster and isinstance(source_cluster, str) and '-' in source_cluster:
                                parent_cluster = source_cluster.split('-')[0]
                            else:
                                parent_cluster = source_cluster
                            
                            row_data = {
                                'respondent_id': respondent_id,
                                'original_response': result.response,
                                'idea_id': idea.idea_id,
                                'idea_text': idea.idea,
                                'initial_cluster_id': parent_cluster,
                                'source_cluster_id': source_cluster,  # This is the sub-cluster from enriched codebook
                                'code_label': code,
                                'code_description': code_info['code_description'],
                                'assignment_rationale': idea.assignment_rationale if hasattr(idea, 'assignment_rationale') else '',
                                'assignment_confidence': idea.assignment_confidence if hasattr(idea, 'assignment_confidence') else None,
                                'theme_name': code_info['theme_name'],
                                'theme_description': code_info['theme_description']
                            }
                            export_data.append(row_data)
                    else:
                        # No codes assigned - still export the row with empty code fields
                        # Get cluster info from the idea itself as fallback
                        initial_cluster = idea.initial_cluster if hasattr(idea, 'initial_cluster') else None
                        if initial_cluster and isinstance(initial_cluster, str) and '-' in initial_cluster:
                            parent_cluster = initial_cluster.split('-')[0]
                            source_cluster = initial_cluster
                        else:
                            parent_cluster = initial_cluster
                            source_cluster = initial_cluster
                            
                        row_data = {
                            'respondent_id': respondent_id,
                            'original_response': result.response,
                            'idea_id': idea.idea_id,
                            'idea_text': idea.idea,
                            'initial_cluster_id': parent_cluster,
                            'source_cluster_id': source_cluster,
                            'code_label': 'No Code Assigned',
                            'code_description': '',
                            'assignment_rationale': idea.assignment_rationale if hasattr(idea, 'assignment_rationale') else '',
                            'assignment_confidence': idea.assignment_confidence if hasattr(idea, 'assignment_confidence') else None,
                            'theme_name': '',
                            'theme_description': ''
                        }
                        export_data.append(row_data)
        
        # Convert to DataFrame
        df = pd.DataFrame(export_data)
        
        # Create export directory if not provided
        if export_dir is None:
            export_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'exports')
        
        Path(export_dir).mkdir(parents=True, exist_ok=True)
        
        # Create output filename
        base_name = Path(filename).stem
        output_filename = f"{base_name}_{var_name}_code_assignments.xlsx"
        output_path = os.path.join(export_dir, output_filename)
        
        # Export to Excel with formatting
        self._write_formatted_excel(df, output_path, var_name)
        
        # Report statistics
        self.verbose_reporter.stat_line(f"Total rows exported: {len(export_data)}")
        self.verbose_reporter.stat_line(f"Unique respondents: {df['respondent_id'].nunique()}")
        self.verbose_reporter.stat_line(f"Unique ideas: {df['idea_id'].nunique()}")
        self.verbose_reporter.stat_line(f"Unique codes assigned: {df[df['code_label'] != 'No Code Assigned']['code_label'].nunique()}")
        self.verbose_reporter.stat_line(f"Excel file saved: {output_path}")
        
        return output_path
    
    def _write_formatted_excel(self, df: pd.DataFrame, output_path: str, var_name: str):
        """Write DataFrame to Excel with formatting"""
        
        # Create workbook and worksheet
        wb = Workbook()
        ws = wb.active
        ws.title = f"{var_name}_Assignments"
        
        # Define styles
        header_font = Font(bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        header_alignment = Alignment(horizontal="center", vertical="center")
        border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )
        
        # Write headers
        headers = [
            'Respondent ID',
            'Original Response',
            'Idea ID',
            'Idea Text',
            'Initial Cluster ID',
            'Source Cluster ID',
            'Code Label',
            'Code Description',
            'Assignment Rationale',
            'Assignment Confidence',
            'Theme Name',
            'Theme Description'
        ]
        
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col, value=header)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = header_alignment
            cell.border = border
        
        # Write data
        for r_idx, row in enumerate(dataframe_to_rows(df, index=False, header=False), 2):
            for c_idx, value in enumerate(row, 1):
                cell = ws.cell(row=r_idx, column=c_idx, value=value)
                cell.border = border
                
                # Format confidence values
                if c_idx == 10 and value is not None:  # Assignment confidence column (shifted by 1 due to new column)
                    try:
                        cell.value = float(value)
                        cell.number_format = '0.00'
                    except:
                        pass
        
        # Adjust column widths
        column_widths = {
            'A': 15,  # Respondent ID
            'B': 50,  # Original Response
            'C': 15,  # Idea ID
            'D': 50,  # Idea Text
            'E': 18,  # Initial Cluster ID
            'F': 18,  # Source Cluster ID
            'G': 30,  # Code Label
            'H': 50,  # Code Description
            'I': 60,  # Assignment Rationale
            'J': 20,  # Assignment Confidence
            'K': 30,  # Theme Name
            'L': 50   # Theme Description
        }
        
        for col, width in column_widths.items():
            ws.column_dimensions[col].width = width
        
        # Freeze the header row
        ws.freeze_panes = 'A2'
        
        # Add summary sheet
        summary_ws = wb.create_sheet(title="Summary")
        
        # Summary statistics
        summary_data = [
            ["Summary Statistics", ""],
            ["", ""],
            ["Total Assignments", len(df)],
            ["Unique Respondents", df['respondent_id'].nunique()],
            ["Unique Ideas", df['idea_id'].nunique()],
            ["Unique Codes", df[df['code_label'] != 'No Code Assigned']['code_label'].nunique()],
            ["Unique Themes", df[df['theme_name'] != '']['theme_name'].nunique()],
            ["", ""],
            ["Code Frequency", "Count"],
        ]
        
        # Add code frequency
        code_freq = df[df['code_label'] != 'No Code Assigned']['code_label'].value_counts()
        for code, count in code_freq.items():
            summary_data.append([code, count])
        
        # Write summary data
        for row_idx, row_data in enumerate(summary_data, 1):
            for col_idx, value in enumerate(row_data, 1):
                cell = summary_ws.cell(row=row_idx, column=col_idx, value=value)
                if row_idx == 1 or (row_idx == 9 and col_idx == 1):
                    cell.font = Font(bold=True, size=14)
                elif row_idx in [3, 4, 5, 6, 7]:
                    if col_idx == 1:
                        cell.font = Font(bold=True)
        
        # Adjust summary column widths
        summary_ws.column_dimensions['A'].width = 30
        summary_ws.column_dimensions['B'].width = 15
        
        # Save workbook
        wb.save(output_path)