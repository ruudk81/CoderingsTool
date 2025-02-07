import os
import pandas as pd
import pyreadstat
from .verboseReporter import VerboseReporter, ProcessingStats

class DataLoader:
    def __init__(self, data_dir: str = None, verbose: bool = False):
        self.verbose = verbose
        self.verbose_reporter = VerboseReporter(verbose)
        self.stats = ProcessingStats()
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
      
    def load_sav(self, filename: str):
        filepath = self.get_file_path(filename)
        self.verbose_reporter.step_start("Extracting Variable Data")
        
        try:
            self.verbose_reporter.stat_line(f"Loading file: {os.path.basename(filepath)}")
            self.verbose_reporter.stat_line(f"File size: {os.path.getsize(filepath) / 1024 / 1024:.1f} MB")
            
            df, meta = pyreadstat.read_sav(filepath, apply_value_formats=True)
            
            self.verbose_reporter.stat_line(f"Rows loaded: {len(df):,}")
            self.verbose_reporter.stat_line(f"Variables loaded: {len(df.columns)}")
            self.verbose_reporter.stat_line(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
            
         
            return df, meta
        except Exception as e:
            self.verbose_reporter.stat_line(f"ERROR: {str(e)}")
            raise ValueError(f"Error loading .sav file '{filepath}': {str(e)}")
    
    def list_variables(self, filename: str):
        df, meta = self.load_sav(filename)
        
        variables = {}
        for var_name in meta.column_names:
            var_label = meta.column_labels[meta.column_names.index(var_name)]
            variables[var_name] = var_label 
            
        return variables
    
    def get_variable(self, filename: str, var_name: str):
        df, _ = self.load_sav(filename)
        if var_name not in df.columns:
            raise ValueError(f"Variable '{var_name}' not found in file '{filename}'")
        variable = df[var_name]
        return variable
    
    def get_variable_with_IDs(self, filename: str, id_column: str, var_name: str):
         
        df, meta = self.load_sav(filename)
        
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
        
    def get_varlab(self, filename: str, var_name: str):
        df, meta = self.load_sav(filename)
        var_label = meta.column_labels[meta.column_names.index(var_name)]
        return var_label
        
    def save_as_csv_data(self, df, file_path, index=False):
        df.to_csv(file_path, index=index)
    
    def load_csv_data(self, file_path, delimiter=","):
        df = pd.read_csv(file_path, delimiter=delimiter)
        return df

# Example:
if __name__ == "__main__":
    loader = DataLoader()
    
    variables_dict = loader.list_variables(
        filename = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav")
    
    survey_data_df = loader.get_variable_with_IDs(
       filename = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav",
       id_column = "DLNMID",
       var_name = "Q20")
    
    var_label = loader.get_varlab(
      filename = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav",
      var_name = "Q20")
   

