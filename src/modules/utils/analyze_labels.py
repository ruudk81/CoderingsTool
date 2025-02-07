"""Utility to load and analyze labeller results from cache"""
import sys
from pathlib import Path
from collections import defaultdict
from typing import List, Dict

# Add project paths
sys.path.append(str(Path(__file__).parents[2]))  # Add src directory

import models
from cache_manager import CacheManager
from cache_config import CacheConfig


def load_and_analyze_labels(filename="M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav", 
                          cache_key='labels_test'):
    """Load labeller results from cache and analyze hierarchical structure"""
    
    # Initialize cache manager
    cache_config = CacheConfig()
    cache_manager = CacheManager(cache_config)
    
    # Load label results from cache
    label_results = cache_manager.load_from_cache(filename, cache_key, models.LabelModel)
    
    if not label_results:
        print(f"No cached label data found for key '{cache_key}'")
        return None
    
    print(f"Loaded {len(label_results)} label results from cache")
    
    # Analyze the hierarchical structure
    analysis = analyze_hierarchical_structure(label_results)
    
    # Print the analysis
    print_hierarchical_analysis(analysis)
    
    return analysis


def analyze_hierarchical_structure(label_results: List[models.LabelModel]) -> Dict:
    """Analyze the hierarchical structure of label results"""
    
    # Initialize data structures
    themes = {}
    theme_summaries = {}
    theme_topics = defaultdict(lambda: defaultdict(set))  # theme_id -> topic_id -> set of code_ids
    theme_topic_codes = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # theme_id -> topic_id -> code_id -> count
    
    # Process each result
    for result in label_results:
        # Extract theme summaries (if available)
        if result.summary:
            # Since summaries might be theme-specific, we'll store all we find
            theme_summaries[result.respondent_id] = result.summary
        
        # Process each segment
        for segment in result.response_segment:
            if segment.Theme:
                theme_id, theme_label = list(segment.Theme.items())[0]
                themes[theme_id] = theme_label
                
                if segment.Topic:
                    topic_id, topic_label = list(segment.Topic.items())[0]
                    theme_topics[theme_id][topic_id] = topic_label
                    
                    if segment.Code:
                        code_id, code_label = list(segment.Code.items())[0]
                        theme_topic_codes[theme_id][topic_id][code_id] = {
                            'label': code_label,
                            'count': theme_topic_codes[theme_id][topic_id].get(code_id, {}).get('count', 0) + 1
                        }
    
    # Compile the analysis
    analysis = {
        'themes': themes,
        'theme_summaries': theme_summaries,
        'theme_structure': {}
    }
    
    # Build hierarchical structure with counts
    for theme_id, theme_label in themes.items():
        theme_data = {
            'label': theme_label,
            'summary': next((s for s in theme_summaries.values() if s), ''),  # Get first available summary
            'topics': {}
        }
        
        for topic_id, topic_label in theme_topics[theme_id].items():
            topic_data = {
                'label': topic_label,
                'codes': {}
            }
            
            for code_id, code_data in theme_topic_codes[theme_id][topic_id].items():
                topic_data['codes'][code_id] = code_data
            
            topic_data['code_count'] = len(topic_data['codes'])
            topic_data['segment_count'] = sum(c['count'] for c in topic_data['codes'].values())
            
            theme_data['topics'][topic_id] = topic_data
        
        theme_data['topic_count'] = len(theme_data['topics'])
        theme_data['total_codes'] = sum(t['code_count'] for t in theme_data['topics'].values())
        theme_data['total_segments'] = sum(t['segment_count'] for t in theme_data['topics'].values())
        
        analysis['theme_structure'][theme_id] = theme_data
    
    return analysis


def print_hierarchical_analysis(analysis: Dict):
    """Print the hierarchical analysis in a formatted way"""
    
    if not analysis:
        return
    
    print("\n" + "="*80)
    print("HIERARCHICAL LABEL ANALYSIS")
    print("="*80)
    
    theme_structure = analysis['theme_structure']
    
    for theme_id in sorted(theme_structure.keys()):
        theme_data = theme_structure[theme_id]
        
        print(f"\n{'='*60}")
        print(f"THEME {theme_id}: {theme_data['label']}")
        print(f"{'='*60}")
        
        # Print theme summary
        if theme_data['summary']:
            print(f"\nSummary:")
            print(f"{theme_data['summary'][:500]}...")
        
        # Print theme statistics
        print(f"\nStatistics:")
        print(f"- Topics: {theme_data['topic_count']}")
        print(f"- Total Codes: {theme_data['total_codes']}")
        print(f"- Total Segments: {theme_data['total_segments']}")
        
        # Print topics
        print(f"\nTopics:")
        for topic_id in sorted(theme_data['topics'].keys()):
            topic_data = theme_data['topics'][topic_id]
            print(f"\n  TOPIC {topic_id}: {topic_data['label']}")
            print(f"  - Codes: {topic_data['code_count']}")
            print(f"  - Segments: {topic_data['segment_count']}")
            
            # Print codes
            print(f"  Codes:")
            for code_id in sorted(topic_data['codes'].keys()):
                code_data = topic_data['codes'][code_id]
                print(f"    CODE {code_id}: {code_data['label']} (Count: {code_data['count']})")
        
        print()


def get_theme_summary(analysis: Dict, theme_id: int) -> Dict:
    """Get detailed summary for a specific theme"""
    
    if not analysis or theme_id not in analysis['theme_structure']:
        return None
    
    theme_data = analysis['theme_structure'][theme_id]
    
    summary = {
        'theme_id': theme_id,
        'theme_label': theme_data['label'],
        'summary': theme_data['summary'],
        'topic_count': theme_data['topic_count'],
        'total_codes': theme_data['total_codes'],
        'total_segments': theme_data['total_segments'],
        'topics': []
    }
    
    for topic_id, topic_data in theme_data['topics'].items():
        topic_summary = {
            'topic_id': topic_id,
            'topic_label': topic_data['label'],
            'code_count': topic_data['code_count'],
            'segment_count': topic_data['segment_count'],
            'codes': []
        }
        
        for code_id, code_data in topic_data['codes'].items():
            code_summary = {
                'code_id': code_id,
                'code_label': code_data['label'],
                'count': code_data['count']
            }
            topic_summary['codes'].append(code_summary)
        
        summary['topics'].append(topic_summary)
    
    return summary


if __name__ == "__main__":
    # Run the analysis
    analysis = load_and_analyze_labels()
    
    # Example: Get summary for a specific theme
    if analysis and analysis['theme_structure']:
        first_theme_id = list(analysis['theme_structure'].keys())[0]
        theme_summary = get_theme_summary(analysis, first_theme_id)
        
        print(f"\n\nDetailed summary for Theme {first_theme_id}:")
        print(f"Theme: {theme_summary['theme_label']}")
        print(f"Topics: {theme_summary['topic_count']}")
        print(f"Total Codes: {theme_summary['total_codes']}")
        print(f"Total Segments: {theme_summary['total_segments']}")