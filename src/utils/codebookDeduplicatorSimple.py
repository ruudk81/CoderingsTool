import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime
from dataclasses import dataclass
import logging

# NLP and similarity
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Local imports
from config import DEFAULT_LANGUAGE
from utils.codeGenerator import CodeGeneratorReasoningResults

# Setup logging
logger = logging.getLogger(__name__)

# === SIMPLIFIED DEDUPLICATION ========================================================================================================

@dataclass
class SimplifiedDeduplicationResults:
    """Results from simplified deduplication based on code names/definitions only"""
    original_codebook: List[Dict[str, str]]
    deduplicated_codebook: List[Dict[str, str]]
    merge_decisions: List[Dict[str, Any]]
    similarity_matrix: Dict[str, Dict[str, float]]
    processing_stats: Dict[str, Any]
    timestamp: str

def deduplicate_codebook_simple(
    reasoning_results: CodeGeneratorReasoningResults,
    similarity_threshold: float = 0.9,
    language: str = DEFAULT_LANGUAGE
) -> SimplifiedDeduplicationResults:
    """
    Simplified codebook deduplication that works directly with the codebook,
    without requiring cluster-to-code mappings.
    """
    start_time = datetime.now()
    
    # Extract codebook
    if not hasattr(reasoning_results, 'codebook') or not reasoning_results.codebook:
        return SimplifiedDeduplicationResults(
            original_codebook=[],
            deduplicated_codebook=[],
            merge_decisions=[],
            similarity_matrix={},
            processing_stats={'error': 'No codebook found in reasoning results'},
            timestamp=start_time.isoformat()
        )
    
    original_codebook = reasoning_results.codebook
    print(f"Starting deduplication with {len(original_codebook)} codes")
    
    # Extract all ideas from step1_inputs
    all_ideas_text = []
    if hasattr(reasoning_results, 'step1_inputs') and reasoning_results.step1_inputs:
        for cluster_id, step1_data in reasoning_results.step1_inputs.items():
            if 'cluster_text' in step1_data:
                cluster_text = step1_data['cluster_text']
                if cluster_text:
                    ideas = [idea.strip() for idea in cluster_text.split('\n') if idea.strip()]
                    clean_ideas = [idea[2:].strip() if idea.startswith('- ') else idea for idea in ideas]
                    all_ideas_text.extend(clean_ideas)
    
    print(f"Collected {len(all_ideas_text)} total ideas from {len(reasoning_results.step1_inputs)} clusters")
    
    # Create a combined text representation for each code
    # Use code name + definition as the basis for similarity
    code_texts = []
    code_names = []
    for code_entry in original_codebook:
        if isinstance(code_entry, dict):
            code_name = code_entry.get('code', '')
            code_def = code_entry.get('definition', '')
            # Combine code name and definition for similarity analysis
            combined_text = f"{code_name} {code_def}"
            code_texts.append(combined_text)
            code_names.append(code_name)
    
    # Compute TF-IDF similarity between codes
    if len(code_texts) < 2:
        print("Less than 2 codes found, no deduplication needed")
        return SimplifiedDeduplicationResults(
            original_codebook=original_codebook,
            deduplicated_codebook=original_codebook,
            merge_decisions=[],
            similarity_matrix={},
            processing_stats={
                'original_code_count': len(original_codebook),
                'deduplicated_code_count': len(original_codebook),
                'codes_merged': 0,
                'note': 'Less than 2 codes - no deduplication needed'
            },
            timestamp=start_time.isoformat()
        )
    
    # Use TfidfVectorizer with Dutch/English stop words based on language
    stop_words = 'dutch' if 'dutch' in language.lower() or 'nederland' in language.lower() else 'english'
    
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=None,  # Don't use built-in stop words, handle via spaCy
        token_pattern=r'\b\w+\b',
        min_df=1,
        max_df=0.95
    )
    
    tfidf_matrix = vectorizer.fit_transform(code_texts)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Find similar codes
    merge_decisions = []
    codes_to_remove = set()
    
    for i in range(len(code_names)):
        if code_names[i] in codes_to_remove:
            continue
            
        for j in range(i + 1, len(code_names)):
            if code_names[j] in codes_to_remove:
                continue
                
            similarity = similarity_matrix[i, j]
            
            if similarity >= similarity_threshold:
                # Decide which code to keep (prefer shorter/cleaner names)
                if len(code_names[i]) <= len(code_names[j]):
                    primary_idx, secondary_idx = i, j
                else:
                    primary_idx, secondary_idx = j, i
                
                merge_decision = {
                    'primary_code': code_names[primary_idx],
                    'secondary_code': code_names[secondary_idx],
                    'similarity': float(similarity),
                    'reason': f"High similarity ({similarity:.3f}) based on code name and definition"
                }
                merge_decisions.append(merge_decision)
                codes_to_remove.add(code_names[secondary_idx])
    
    # Create deduplicated codebook
    deduplicated_codebook = []
    for code_entry in original_codebook:
        code_name = code_entry.get('code', '')
        if code_name not in codes_to_remove:
            # Check if this code absorbed others
            merged_codes = [d['secondary_code'] for d in merge_decisions if d['primary_code'] == code_name]
            if merged_codes:
                # Update definition to mention merged codes
                updated_entry = code_entry.copy()
                updated_entry['definition'] = f"{code_entry.get('definition', '')} [Merged: {', '.join(merged_codes)}]"
                deduplicated_codebook.append(updated_entry)
            else:
                deduplicated_codebook.append(code_entry)
    
    # Create similarity matrix dict
    similarity_dict = {}
    for i, code1 in enumerate(code_names):
        similarity_dict[code1] = {}
        for j, code2 in enumerate(code_names):
            similarity_dict[code1][code2] = float(similarity_matrix[i, j])
    
    processing_time = (datetime.now() - start_time).total_seconds()
    
    return SimplifiedDeduplicationResults(
        original_codebook=original_codebook,
        deduplicated_codebook=deduplicated_codebook,
        merge_decisions=merge_decisions,
        similarity_matrix=similarity_dict,
        processing_stats={
            'original_code_count': len(original_codebook),
            'deduplicated_code_count': len(deduplicated_codebook),
            'codes_merged': len(merge_decisions),
            'processing_time_seconds': processing_time,
            'similarity_threshold': similarity_threshold,
            'language': language,
            'total_ideas_found': len(all_ideas_text),
            'methodology': 'Simplified TF-IDF on code names and definitions'
        },
        timestamp=start_time.isoformat()
    )

def print_simple_deduplication_report(results: SimplifiedDeduplicationResults):
    """Print a formatted report of simplified deduplication results"""
    stats = results.processing_stats
    
    print(f"\n{'='*60}")
    print(f"SIMPLIFIED CODEBOOK DEDUPLICATION REPORT")
    print(f"{'='*60}")
    print(f"Timestamp: {results.timestamp}")
    print(f"Methodology: {stats.get('methodology', 'unknown')}")
    print(f"Language: {stats.get('language', 'unknown')}")
    print(f"Similarity Threshold: {stats.get('similarity_threshold', 'unknown')}")
    print(f"\nOriginal Codes: {stats.get('original_code_count', 0)}")
    print(f"Final Codes: {stats.get('deduplicated_code_count', 0)}")
    print(f"Codes Merged: {stats.get('codes_merged', 0)}")
    print(f"Total Ideas Found: {stats.get('total_ideas_found', 0)}")
    
    if 'error' in stats:
        print(f"\n⚠️  ERROR: {stats['error']}")
    
    if results.merge_decisions:
        print(f"\nMERGE DECISIONS:")
        print(f"{'-'*40}")
        for i, decision in enumerate(results.merge_decisions, 1):
            print(f"{i}. Merged '{decision['secondary_code']}' → '{decision['primary_code']}'")
            print(f"   Similarity: {decision['similarity']:.3f}")
            print(f"   Reason: {decision['reason']}")
            print()
    else:
        print(f"\nNo codes were merged.")
    
    print(f"Processing completed in {stats.get('processing_time_seconds', 0):.2f} seconds")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    print("Simplified Codebook Deduplicator")
    print("Usage: from utils.codebookDeduplicatorSimple import deduplicate_codebook_simple, print_simple_deduplication_report")