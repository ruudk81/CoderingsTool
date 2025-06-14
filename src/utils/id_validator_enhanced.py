"""
Enhanced ID validator to diagnose segment ID uniqueness issues
"""
from typing import List, Dict, Any
from collections import Counter
import models

class SegmentIdValidator:
    """Validates and reports on segment ID uniqueness"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def validate_segment_ids(self, data: List[Any]) -> Dict[str, Any]:
        """
        Validate segment ID uniqueness across a dataset
        
        Args:
            data: List of model objects with response_segment attribute
            
        Returns:
            Dict with validation results
        """
        all_segment_ids = []
        segment_info = []
        
        # Collect all segment IDs
        for item in data:
            if hasattr(item, 'response_segment') and item.response_segment:
                for segment in item.response_segment:
                    if hasattr(segment, 'segment_id'):
                        segment_id = segment.segment_id
                        all_segment_ids.append(segment_id)
                        segment_info.append({
                            'segment_id': segment_id,
                            'respondent_id': item.respondent_id,
                            'segment_response': getattr(segment, 'segment_response', 'N/A')[:50] + '...'
                        })
        
        # Analyze duplicates
        id_counts = Counter(all_segment_ids)
        duplicates = {seg_id: count for seg_id, count in id_counts.items() if count > 1}
        
        # Prepare results
        total_segments = len(all_segment_ids)
        unique_segments = len(set(all_segment_ids))
        duplicate_count = total_segments - unique_segments
        
        results = {
            'total_segments': total_segments,
            'unique_segments': unique_segments,
            'duplicate_count': duplicate_count,
            'duplicates': duplicates,
            'success_rate': unique_segments / total_segments if total_segments > 0 else 0.0,
            'all_segment_ids': all_segment_ids,
            'segment_info': segment_info
        }
        
        if self.verbose:
            self._print_validation_report(results)
            
        return results
    
    def _print_validation_report(self, results: Dict[str, Any]):
        """Print a detailed validation report"""
        print("\n" + "=" * 60)
        print("🔍 SEGMENT ID VALIDATION REPORT")
        print("=" * 60)
        
        print(f"📊 Total segments: {results['total_segments']}")
        print(f"🎯 Unique segment IDs: {results['unique_segments']}")
        print(f"⚠️  Duplicate segments: {results['duplicate_count']}")
        print(f"✅ Success rate: {results['success_rate']:.1%}")
        
        if results['duplicates']:
            print(f"\n❌ DUPLICATE ANALYSIS:")
            print(f"Found {len(results['duplicates'])} segment IDs with duplicates:")
            
            # Show top 10 most duplicated IDs
            sorted_duplicates = sorted(results['duplicates'].items(), key=lambda x: x[1], reverse=True)
            for seg_id, count in sorted_duplicates[:10]:
                print(f"  '{seg_id}': appears {count} times")
                
                # Show examples of segments with this ID
                examples = [info for info in results['segment_info'] if info['segment_id'] == seg_id][:3]
                for example in examples:
                    print(f"    Respondent {example['respondent_id']}: {example['segment_response']}")
                if len(examples) < count:
                    print(f"    ... and {count - len(examples)} more")
                print()
        else:
            print(f"\n✅ PERFECT: All segment IDs are unique!")
            print(f"Example IDs: {results['all_segment_ids'][:10]}")
            if len(results['all_segment_ids']) > 10:
                print(f"... and {len(results['all_segment_ids']) - 10} more")
        
        print("=" * 60)
    
    def generate_id_format_report(self, data: List[Any]) -> Dict[str, Any]:
        """Analyze the format patterns of segment IDs"""
        all_segment_ids = []
        
        for item in data:
            if hasattr(item, 'response_segment') and item.response_segment:
                for segment in item.response_segment:
                    if hasattr(segment, 'segment_id'):
                        all_segment_ids.append(segment.segment_id)
        
        # Analyze patterns
        patterns = {
            'simple_numeric': 0,      # "1", "2", "3"
            'compound_format': 0,     # "123_1", "456_2"
            'other_format': 0         # anything else
        }
        
        compound_examples = []
        simple_examples = []
        other_examples = []
        
        for seg_id in all_segment_ids:
            seg_id_str = str(seg_id)
            if '_' in seg_id_str and seg_id_str.count('_') == 1:
                parts = seg_id_str.split('_')
                if len(parts) == 2 and parts[1].isdigit():
                    patterns['compound_format'] += 1
                    if len(compound_examples) < 5:
                        compound_examples.append(seg_id_str)
                else:
                    patterns['other_format'] += 1
                    if len(other_examples) < 5:
                        other_examples.append(seg_id_str)
            elif seg_id_str.isdigit():
                patterns['simple_numeric'] += 1
                if len(simple_examples) < 5:
                    simple_examples.append(seg_id_str)
            else:
                patterns['other_format'] += 1
                if len(other_examples) < 5:
                    other_examples.append(seg_id_str)
        
        format_report = {
            'patterns': patterns,
            'examples': {
                'compound': compound_examples,
                'simple': simple_examples,
                'other': other_examples
            },
            'total_analyzed': len(all_segment_ids)
        }
        
        if self.verbose:
            self._print_format_report(format_report)
            
        return format_report
    
    def _print_format_report(self, report: Dict[str, Any]):
        """Print segment ID format analysis"""
        print("\n" + "=" * 60)
        print("📋 SEGMENT ID FORMAT ANALYSIS")
        print("=" * 60)
        
        total = report['total_analyzed']
        for pattern_name, count in report['patterns'].items():
            percentage = (count / total * 100) if total > 0 else 0
            print(f"{pattern_name}: {count} ({percentage:.1f}%)")
            
            examples = report['examples'].get(pattern_name.split('_')[0], [])
            if examples:
                print(f"  Examples: {', '.join(examples)}")
            print()
        
        print("=" * 60)

# Convenience function for quick validation
def validate_segment_ids(data: List[Any], verbose: bool = True) -> Dict[str, Any]:
    """Quick function to validate segment IDs"""
    validator = SegmentIdValidator(verbose=verbose)
    return validator.validate_segment_ids(data)