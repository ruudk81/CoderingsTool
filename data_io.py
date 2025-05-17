import os
import pandas as pd
import pyreadstat

class DataLoader:
    def __init__(self, data_dir: str = None):
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
        try:
            df, meta = pyreadstat.read_sav(filepath, apply_value_formats=True)
            return df, meta
        except Exception as e:
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
        df, _ = self.load_sav(filename)
        if var_name not in df.columns:
            raise ValueError(f"Variable '{var_name}' not found in file '{filename}'")
        if id_column not in df.columns:
            raise ValueError(f"ID column '{id_column}' not found in file '{filename}'")
        variable = df[[id_column, var_name]]
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
   

