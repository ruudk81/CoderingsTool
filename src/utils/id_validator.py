import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

import pandas as pd
from typing import List, Dict, Set, Tuple
import models

def validate_id_consistency(
    original_df: pd.DataFrame,
    id_column: str,
    labeled_results: List[models.LabelModel],
    verbose: bool = True
) -> Dict[str, any]:
    """
    Validate that IDs are consistent between original SPSS data and pipeline results.
    
    Returns a dictionary with validation results and diagnostics.
    """
    # Extract IDs from original SPSS file
    original_ids = set(original_df[id_column].astype(int).tolist())
    
    # Extract IDs from pipeline results
    pipeline_ids = {result.respondent_id for result in labeled_results}
    
    # Compute differences
    ids_only_in_original = original_ids - pipeline_ids
    ids_only_in_pipeline = pipeline_ids - original_ids
    ids_in_both = original_ids & pipeline_ids
    
    # Create detailed report
    validation_results = {
        'original_count': len(original_ids),
        'pipeline_count': len(pipeline_ids),
        'matched_count': len(ids_in_both),
        'only_in_original': ids_only_in_original,
        'only_in_pipeline': ids_only_in_pipeline,
        'match_percentage': (len(ids_in_both) / len(original_ids) * 100) if original_ids else 0
    }
    
    if verbose:
        print("\n🔍 ID VALIDATION REPORT")
        print("=" * 50)
        print(f"Original SPSS IDs: {validation_results['original_count']}")
        print(f"Pipeline IDs: {validation_results['pipeline_count']}")
        print(f"Matched IDs: {validation_results['matched_count']} ({validation_results['match_percentage']:.1f}%)")
        print(f"IDs only in SPSS: {len(validation_results['only_in_original'])}")
        print(f"IDs only in pipeline: {len(validation_results['only_in_pipeline'])}")
        
        if validation_results['only_in_original']:
            print("\n⚠️  Sample IDs in SPSS but not in pipeline:")
            sample_ids = list(validation_results['only_in_original'])[:10]
            for id in sample_ids:
                print(f"  - {id}")
            if len(validation_results['only_in_original']) > 10:
                print(f"  ... and {len(validation_results['only_in_original']) - 10} more")
        
        if validation_results['only_in_pipeline']:
            print("\n❌ ERROR: IDs in pipeline but not in SPSS:")
            sample_ids = list(validation_results['only_in_pipeline'])[:10]
            for id in sample_ids:
                print(f"  - {id}")
            if len(validation_results['only_in_pipeline']) > 10:
                print(f"  ... and {len(validation_results['only_in_pipeline']) - 10} more")
    
    return validation_results


def check_id_format_consistency(
    original_df: pd.DataFrame,
    id_column: str,
    raw_text_list: List[models.ResponseModel]
) -> Dict[str, any]:
    """
    Check if ID formats are consistent between loading and original data.
    """
    # Get original ID types
    original_id_types = set()
    for id_val in original_df[id_column]:
        original_id_types.add(type(id_val).__name__)
    
    # Get pipeline ID types
    pipeline_id_types = set()
    for item in raw_text_list:
        pipeline_id_types.add(type(item.respondent_id).__name__)
    
    # Check for potential conversion issues
    issues = []
    
    # Check if original has floats that might lose precision
    sample_ids = original_df[id_column].head(10).tolist()
    for id_val in sample_ids:
        if isinstance(id_val, float) and id_val != int(id_val):
            issues.append(f"Float ID {id_val} will be truncated to {int(id_val)}")
    
    return {
        'original_types': original_id_types,
        'pipeline_types': pipeline_id_types,
        'conversion_issues': issues
    }