#!/usr/bin/env python3
"""
Segment ID Validation Utility

This utility validates that all segment IDs in the pipeline are unique and properly formatted.
Run this to verify the segment ID duplication fix is working correctly.
"""

import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

from typing import List, Dict, Set
from collections import Counter, defaultdict
sys.path.append('..')
import models

def validate_segment_ids(data_list: List[models.DescriptiveModel]) -> Dict:
    """
    Validate segment IDs across the entire dataset.
    
    Returns:
        dict: Validation results with statistics and issues
    """
    all_segment_ids = []
    respondent_segment_map = defaultdict(list)
    format_analysis = {
        'compound_format': 0,  # respondent_X format
        'simple_format': 0,    # just numbers
        'other_format': 0      # other patterns
    }
    
    # Collect all segment IDs
    for response_item in data_list:
        respondent_id = response_item.respondent_id
        
        if response_item.response_segment:
            for segment in response_item.response_segment:
                segment_id = segment.segment_id
                all_segment_ids.append(segment_id)
                respondent_segment_map[respondent_id].append(segment_id)
                
                # Analyze format
                if '_' in str(segment_id) and str(segment_id).split('_')[0] == str(respondent_id):
                    format_analysis['compound_format'] += 1
                elif str(segment_id).isdigit():
                    format_analysis['simple_format'] += 1
                else:
                    format_analysis['other_format'] += 1
    
    # Count duplicates
    segment_counts = Counter(all_segment_ids)
    duplicates = {seg_id: count for seg_id, count in segment_counts.items() if count > 1}
    
    # Analyze respondent patterns
    respondents_with_duplicates = []
    for resp_id, seg_ids in respondent_segment_map.items():
        if len(seg_ids) != len(set(seg_ids)):
            respondents_with_duplicates.append(resp_id)
    
    return {
        'total_segments': len(all_segment_ids),
        'unique_segments': len(set(all_segment_ids)),
        'duplicate_count': len(all_segment_ids) - len(set(all_segment_ids)),
        'duplicates': duplicates,
        'format_analysis': format_analysis,
        'respondents_with_duplicates': respondents_with_duplicates,
        'sample_segment_ids': all_segment_ids[:10],
        'respondent_count': len(respondent_segment_map)
    }

def print_validation_report(results: Dict, verbose: bool = True):
    """Print a formatted validation report."""
    print("📋 SEGMENT ID VALIDATION REPORT")
    print("=" * 50)
    
    # Overall statistics
    print(f"📊 Overall Statistics:")
    print(f"  • Total segments: {results['total_segments']}")
    print(f"  • Unique segments: {results['unique_segments']}")
    print(f"  • Duplicate count: {results['duplicate_count']}")
    print(f"  • Respondents: {results['respondent_count']}")
    
    # Duplication status
    if results['duplicate_count'] == 0:
        print(f"✅ SUCCESS: No duplicate segment IDs found!")
    else:
        print(f"❌ ISSUE: {results['duplicate_count']} duplicate segment IDs found")
        
        if verbose and results['duplicates']:
            print(f"\n🔍 Duplicate Details:")
            for seg_id, count in list(results['duplicates'].items())[:5]:  # Show first 5
                print(f"  • '{seg_id}': appears {count} times")
            if len(results['duplicates']) > 5:
                print(f"  • ... and {len(results['duplicates']) - 5} more duplicates")
    
    # Format analysis
    print(f"\n📝 Format Analysis:")
    total = results['total_segments']
    for format_type, count in results['format_analysis'].items():
        percentage = (count / total * 100) if total > 0 else 0
        status = "✅" if format_type == 'compound_format' and count > 0 else "⚠️ " if format_type != 'compound_format' and count > 0 else "🔸"
        print(f"  {status} {format_type.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
    
    # Sample IDs
    if verbose and results['sample_segment_ids']:
        print(f"\n📋 Sample Segment IDs:")
        for seg_id in results['sample_segment_ids']:
            print(f"  • '{seg_id}'")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if results['duplicate_count'] == 0:
        print(f"  ✅ Segment ID system is working correctly")
        if results['format_analysis']['compound_format'] == results['total_segments']:
            print(f"  ✅ All segment IDs use proper compound format")
        else:
            print(f"  ⚠️  Some segment IDs don't use compound format - verify they're unique")
    else:
        print(f"  ❌ Fix segment ID generation to eliminate duplicates")
        print(f"  💡 Consider using compound format: 'respondent_id_segment_number'")

def validate_from_cache_or_file(cache_step: str = "segmented_descriptions"):
    """
    Load data from cache and validate segment IDs.
    
    Args:
        cache_step: Which pipeline step to validate from cache
    """
    try:
        # Try to import and load from cache
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        
        from config import DEFAULT_CACHE_CONFIG  
        from cache import CacheDatabase
        
        cache_db = CacheDatabase(DEFAULT_CACHE_CONFIG)
        
        # Try to load from cache
        cached_data = cache_db.load_step_data("M250285 input voor coderen - met Q18Q19.sav", cache_step)
        
        if cached_data is not None:
            print(f"📂 Loaded {len(cached_data)} items from cache step: {cache_step}")
            results = validate_segment_ids(cached_data)
            print_validation_report(results, verbose=True)
            return results
        else:
            print(f"❌ No cached data found for step: {cache_step}")
            return None
            
    except Exception as e:
        print(f"❌ Error loading from cache: {e}")
        print(f"   Details: {str(e)}")
        return None

if __name__ == "__main__":
    print("🔍 Segment ID Validation Utility")
    print("=" * 40)
    
    # Try to validate from cache
    result = validate_from_cache_or_file("segmented_descriptions")
    
    if result is None:
        print("\n💡 To validate segment IDs:")
        print("  1. Run the pipeline to generate segmented data")
        print("  2. Run this script again to validate the results")
        print("  3. Or use: validate_segment_ids(your_data_list)")