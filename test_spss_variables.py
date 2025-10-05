import pyreadstat
import pandas as pd

# Load the SPSS file
filepath = r"C:\Users\rkn\Python_apps\CoderingsTool\data\M250127 Flitspeiling NAVOtop 0meting_153832.sav"
df, meta = pyreadstat.read_sav(filepath, apply_value_formats=True)

print("=" * 80)
print("SPSS FILE VARIABLE ANALYSIS")
print("=" * 80)

# Print all variables with their types
for var_name in meta.column_names:
    var_label = meta.column_labels[meta.column_names.index(var_name)]
    pandas_dtype = df[var_name].dtype

    # Check if pyreadstat provides original variable types
    original_type = "N/A"
    if hasattr(meta, 'original_variable_types') and meta.original_variable_types:
        original_type = meta.original_variable_types.get(var_name, "N/A")

    readstat_type = "N/A"
    if hasattr(meta, 'readstat_variable_types') and meta.readstat_variable_types:
        readstat_type = meta.readstat_variable_types.get(var_name, "N/A")

    # Check for value labels (indicates categorical variable)
    has_value_labels = var_name in meta.variable_value_labels if hasattr(meta, 'variable_value_labels') else False

    # Get sample values
    non_null_values = df[var_name].dropna()
    sample_value = non_null_values.iloc[0] if len(non_null_values) > 0 else "No data"

    print(f"\nVariable: {var_name}")
    print(f"  Label: {var_label}")
    print(f"  Pandas dtype: {pandas_dtype}")
    print(f"  Original SPSS type: {original_type}")
    print(f"  Readstat type: {readstat_type}")
    print(f"  Has value labels: {has_value_labels}")
    print(f"  Sample value: {sample_value}")
    print(f"  Detected as string (dtype=='object'): {pandas_dtype == 'object'}")

print("\n" + "=" * 80)
print("LOOKING FOR Q10 SPECIFICALLY:")
print("=" * 80)
if 'Q10' in df.columns:
    print(f"Q10 pandas dtype: {df['Q10'].dtype}")
    print(f"Q10 sample values:")
    print(df['Q10'].dropna().head(5))
else:
    print("Q10 not found in dataset!")
    print("Available variables starting with Q:")
    q_vars = [v for v in meta.column_names if v.startswith('Q')]
    for qv in q_vars[:10]:  # Show first 10
        print(f"  {qv}: {meta.column_labels[meta.column_names.index(qv)]}")
