import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import os
import pandas as pd
import pyreadstat

# === UTILS ========================================================================================================
from .verboseReporter import VerboseReporter, ProcessingStats

class DataLoader:
    def __init__(self, data_dir: str = None, verbose: bool = False):
        self.verbose = verbose
        self.verbose_reporter = VerboseReporter(verbose, capture_logging=True)
        self.stats = ProcessingStats()
        self._last_successful_encoding = None
        current_dir = os.getcwd()
        if data_dir is None:
            if os.path.basename(current_dir) == 'utils':
                data_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..', 'data'))
            elif os.path.basename(current_dir) == 'modules':
                data_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'data'))
            else:
                data_dir = os.path.abspath(os.path.join(current_dir, '..', 'data'))  
        self.data_dir = data_dir
        
    def get_file_path(self, filename: str) -> str:
        if not filename.lower().endswith('.sav'):
            filename = f"{filename}.sav"
        
        return os.path.join(self.data_dir, filename)
    
    def get_last_successful_encoding(self) -> str:
        """Get the encoding that was successfully used for the last file load"""
        return self._last_successful_encoding
      
    def load_sav(self, filename: str, encoding: str = None):
        """Load SPSS file with robust encoding detection"""
        filepath = self.get_file_path(filename)
        self.verbose_reporter.step_start("Extracting Variable Data")
        
        # Common encodings to try, in order of preference
        encodings_to_try = [
            'utf-8',           # Most modern, preferred
            'windows-1252',    # Common Windows default
            'iso-8859-1',      # Latin-1, very permissive
            'cp1252',          # Windows Western European
            'iso-8859-15',     # Latin-9 (includes Euro symbol)
            'windows-1250',    # Central European
            None               # Let pyreadstat auto-detect
        ]
        
        # If user specified encoding, try it first
        if encoding:
            encodings_to_try.insert(0, encoding)
        
        self.verbose_reporter.stat_line(f"Loading file: {os.path.basename(filepath)}")
        self.verbose_reporter.stat_line(f"File size: {os.path.getsize(filepath) / 1024 / 1024:.1f} MB")
        
        last_error = None
        for enc in encodings_to_try:
            try:
                if enc:
                    self.verbose_reporter.stat_line(f"Trying encoding: {enc}")
                    df, meta = pyreadstat.read_sav(filepath, apply_value_formats=True, encoding=enc)
                else:
                    self.verbose_reporter.stat_line("Trying auto-detection")
                    df, meta = pyreadstat.read_sav(filepath, apply_value_formats=True)
                
                # Success!
                encoding_used = enc or "auto-detected"
                self.verbose_reporter.stat_line(f"✅ Successfully loaded with encoding: {encoding_used}")
                self.verbose_reporter.stat_line(f"Rows loaded: {len(df):,}")
                self.verbose_reporter.stat_line(f"Variables loaded: {len(df.columns)}")
                self.verbose_reporter.stat_line(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
                
                # Store successful encoding for future reference
                self._last_successful_encoding = enc
                
                return df, meta
                
            except Exception as e:
                error_msg = str(e).lower()
                if "encoding" in error_msg or "byte sequence" in error_msg or "decode" in error_msg:
                    # This is an encoding error, try next encoding
                    if enc:
                        self.verbose_reporter.stat_line(f"❌ Encoding {enc} failed: {str(e)}")
                    last_error = e
                    continue
                else:
                    # This is a different error, don't continue trying encodings
                    self.verbose_reporter.stat_line(f"ERROR: {str(e)}")
                    raise ValueError(f"Error loading .sav file '{filepath}': {str(e)}")
        
        # If we get here, all encodings failed
        self.verbose_reporter.stat_line("ERROR: All encoding attempts failed")
        error_msg = f"Error loading .sav file '{filepath}': Unable to read with any encoding. Last error: {str(last_error)}"
        raise ValueError(error_msg)
    
    def list_variables(self, filename: str, encoding: str = None):
        """List all variables in SPSS file with encoding support"""
        df, meta = self.load_sav(filename, encoding)
        
        variables = {}
        for var_name in meta.column_names:
            var_label = meta.column_labels[meta.column_names.index(var_name)]
            variables[var_name] = var_label 
            
        return variables
    
    def list_variables_with_types(self, filename: str, encoding: str = None):
        """List all variables with their types (string vs numeric)"""
        df, meta = self.load_sav(filename, encoding)
        
        variables_with_types = {}
        for var_name in meta.column_names:
            var_label = meta.column_labels[meta.column_names.index(var_name)]
            # Check if variable is string type (object dtype in pandas)
            is_string = df[var_name].dtype == 'object'
            variables_with_types[var_name] = {
                'label': var_label,
                'is_string': is_string,
                'dtype': str(df[var_name].dtype)
            }
            
        return variables_with_types
    
    def get_variable(self, filename: str, var_name: str, encoding: str = None):
        """Get specific variable with encoding support"""
        df, _ = self.load_sav(filename, encoding)
        if var_name not in df.columns:
            raise ValueError(f"Variable '{var_name}' not found in file '{filename}'")
        variable = df[var_name]
        return variable
    
    def get_variable_with_IDs(self, filename: str, id_column: str, var_name: str, encoding: str = None):
        """Get variable with IDs with encoding support"""
        df, meta = self.load_sav(filename, encoding)
        
        if var_name not in df.columns:
            self.verbose_reporter.stat_line(f"ERROR: Variable '{var_name}' not found")
            raise ValueError(f"Variable '{var_name}' not found in file '{filename}'")
        if id_column not in df.columns:
            self.verbose_reporter.stat_line(f"ERROR: ID column '{id_column}' not found")
            raise ValueError(f"ID column '{id_column}' not found in file '{filename}'")
        
        variable = df[[id_column, var_name]]
        
        # Report statistics
        var_label = meta.column_labels[meta.column_names.index(var_name)]
        self.verbose_reporter.stat_line(f"Variable: {var_name}")
        self.verbose_reporter.stat_line(f"Label: {var_label}")
        self.verbose_reporter.stat_line(f"Non-null values: {variable[var_name].notna().sum():,}")
        self.verbose_reporter.stat_line(f"Null values: {variable[var_name].isna().sum():,}")
        self.verbose_reporter.stat_line(f"Unique values: {variable[var_name].nunique():,}")
        
        # Sample non-null values
        non_null_values = variable[variable[var_name].notna()][var_name]
        if len(non_null_values) > 0:
            sample_values = non_null_values.head(5).tolist()
            self.verbose_reporter.sample_list("Sample responses", sample_values)
       
        return variable
        
    def get_multiple_variables_with_IDs(self, filename: str, id_column: str, var_names: list, 
                                       merge_strategy: str = "concatenate", separator: str = " ",
                                       skip_empty: bool = True, encoding: str = None):
        """Get multiple variables merged into single column with IDs with encoding support"""
        df, meta = self.load_sav(filename, encoding)
        
        # Validate all variables exist
        missing_vars = [var for var in var_names if var not in df.columns]
        if missing_vars:
            self.verbose_reporter.stat_line(f"ERROR: Variables not found: {missing_vars}")
            raise ValueError(f"Variables not found in file '{filename}': {missing_vars}")
        
        if id_column not in df.columns:
            self.verbose_reporter.stat_line(f"ERROR: ID column '{id_column}' not found")
            raise ValueError(f"ID column '{id_column}' not found in file '{filename}'")
        
        # Create working dataframe with ID and selected variables
        working_df = df[[id_column] + var_names].copy()
        
        # Apply merge strategy
        if merge_strategy == "concatenate":
            merged_values = self._merge_concatenate(working_df, var_names, separator, skip_empty)
        elif merge_strategy == "first_available":
            merged_values = self._merge_first_available(working_df, var_names, skip_empty)
        elif merge_strategy == "prioritized":
            merged_values = self._merge_prioritized(working_df, var_names, skip_empty)
        elif merge_strategy == "all_combined":
            merged_values = self._merge_all_combined(working_df, var_names, separator, skip_empty, meta)
        else:
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            id_column: working_df[id_column],
            'merged_text': merged_values
        })
        
        # Report statistics
        combined_label = self._create_combined_label(meta, var_names, merge_strategy)
        self.verbose_reporter.stat_line(f"Variables merged: {len(var_names)} ({', '.join(var_names)})")
        self.verbose_reporter.stat_line(f"Combined label: {combined_label}")
        self.verbose_reporter.stat_line(f"Merge strategy: {merge_strategy}")
        
        # Calculate coverage statistics
        total_responses = len(result_df)
        non_empty_responses = result_df['merged_text'].notna().sum()
        coverage_pct = (non_empty_responses / total_responses) * 100 if total_responses > 0 else 0
        
        self.verbose_reporter.stat_line(f"Total responses: {total_responses:,}")
        self.verbose_reporter.stat_line(f"Non-empty merged responses: {non_empty_responses:,}")
        self.verbose_reporter.stat_line(f"Coverage: {coverage_pct:.1f}%")
        
        # Individual variable statistics
        for var in var_names:
            var_coverage = working_df[var].notna().sum()
            var_pct = (var_coverage / total_responses) * 100 if total_responses > 0 else 0
            self.verbose_reporter.stat_line(f"  {var}: {var_coverage:,} responses ({var_pct:.1f}%)")
        
        # Sample merged values
        non_null_merged = result_df[result_df['merged_text'].notna()]['merged_text']
        if len(non_null_merged) > 0:
            sample_values = non_null_merged.head(5).tolist()
            self.verbose_reporter.sample_list("Sample merged responses", sample_values)
        
        return result_df
    
    def _merge_concatenate(self, df, var_names, separator, skip_empty):
        """Concatenate all non-empty values with separator"""
        merged = []
        for _, row in df.iterrows():
            parts = []
            for var in var_names:
                value = row[var]
                if pd.notna(value) and str(value).strip():
                    parts.append(str(value).strip())
                elif not skip_empty:
                    parts.append("")
            
            if parts:
                merged.append(separator.join(parts))
            else:
                merged.append(None)
        return merged
    
    def _merge_first_available(self, df, var_names, skip_empty):
        """Use first available non-empty value"""
        merged = []
        for _, row in df.iterrows():
            result = None
            for var in var_names:
                value = row[var]
                if pd.notna(value) and str(value).strip():
                    result = str(value).strip()
                    break
            merged.append(result)
        return merged
    
    def _merge_prioritized(self, df, var_names, skip_empty):
        """Same as first_available but explicit about priority order"""
        return self._merge_first_available(df, var_names, skip_empty)
    
    def _merge_all_combined(self, df, var_names, separator, skip_empty, meta):
        """Include all responses with variable labels"""
        merged = []
        for _, row in df.iterrows():
            parts = []
            for var in var_names:
                value = row[var]
                if pd.notna(value) and str(value).strip():
                    var_label = meta.column_labels[meta.column_names.index(var)] or var
                    parts.append(f"{var_label}: {str(value).strip()}")
                elif not skip_empty:
                    var_label = meta.column_labels[meta.column_names.index(var)] or var
                    parts.append(f"{var_label}: [empty]")
            
            if parts:
                merged.append(separator.join(parts))
            else:
                merged.append(None)
        return merged
    
    def _create_combined_label(self, meta, var_names, merge_strategy):
        """Create a combined label for the merged variables"""
        labels = []
        for var in var_names:
            label = meta.column_labels[meta.column_names.index(var)]
            labels.append(label or var)
        
        if merge_strategy == "concatenate":
            return f"Combined: {' + '.join(labels)}"
        elif merge_strategy in ["first_available", "prioritized"]:
            return f"First of: {' / '.join(labels)}"
        elif merge_strategy == "all_combined":
            return f"All combined: {' & '.join(labels)}"
        else:
            return f"Merged ({merge_strategy}): {' | '.join(labels)}"
    
    def get_varlab(self, filename: str, var_name: str, encoding: str = None):
        """Get variable label with encoding support"""
        df, meta = self.load_sav(filename, encoding)
        var_label = meta.column_labels[meta.column_names.index(var_name)]
        return var_label
        
    def save_as_csv_data(self, df, file_path, index=False):
        df.to_csv(file_path, index=index)
    
    def load_csv_data(self, file_path, delimiter=","):
        df = pd.read_csv(file_path, delimiter=delimiter)
        return df

