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

