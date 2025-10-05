import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
from typing import List, Dict, Any
from collections import defaultdict
import logging
from dataclasses import dataclass
from datetime import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# Local imports
from config import DEFAULT_LANGUAGE
from utils.codeGenerator import CodeGeneratorReasoningResults
from utils.ctfidf import ClassTfidfTransformer

# Setup logging
logger = logging.getLogger(__name__)

# === DATACLASSES ========================================================================================================

@dataclass
class IdeaCodeMapping:
    """Maps ideas to their final assigned codes"""
    cluster_id: str
    theme_id: str
    original_ideas: List[str]  # The 30 sampled ideas
    final_code: str
    final_definition: str

@dataclass
class DeduplicationResults:
    """Final results of codebook deduplication"""
    original_codebook: List[Dict[str, str]]
    deduplicated_codebook: List[Dict[str, str]]
    merge_groups: Dict[int, List[str]]  # Group ID -> List of codes in that group
    similarity_pairs: List[Dict[str, Any]]  # Pairs above threshold with similarity scores
    ctfidf_matrix: Any  # The c-TF-IDF matrix for analysis
    processing_stats: Dict[str, Any]
    timestamp: str

# === CORE IMPLEMENTATION ========================================================================================================

class CodebookDeduplicator:
    """
    Deduplicates codebook using BERTopic's c-TF-IDF approach:
    1. Group all ideas by code (treating each code as a "class")
    2. Create c-TF-IDF representations
    3. Find codes with cosine similarity above threshold
    4. Merge similar codes into groups
    """
    
    def __init__(self, similarity_threshold: float = 0.9, language: str = DEFAULT_LANGUAGE):
        """
        Initialize deduplicator.
        
        Args:
            similarity_threshold: Minimum cosine similarity to merge codes (0.0-1.0)
            language: Language for text processing
        """
        self.similarity_threshold = similarity_threshold
        self.language = language.lower()
        self.nlp = self._load_spacy_model()
        self.vectorizer = CountVectorizer(
            lowercase=True,
            token_pattern=r'\b\w+\b',
            min_df=1,
            max_df=0.95
        )
        self.ctfidf = ClassTfidfTransformer()
        
    def _load_spacy_model(self):
        """Load appropriate spaCy model"""
        try:
            if 'dutch' in self.language or 'nederland' in self.language:
                return spacy.load('nl_core_news_lg')
            else:
                return spacy.load('en_core_web_lg')
        except OSError:
            logger.warning("Large spaCy model not found, falling back to medium model")
            try:
                if 'dutch' in self.language or 'nederland' in self.language:
                    return spacy.load('nl_core_news_md')
                else:
                    return spacy.load('en_core_web_md')
            except OSError:
                logger.error("No suitable spaCy model found")
                raise
    
    def deduplicate(self, reasoning_results: CodeGeneratorReasoningResults) -> DeduplicationResults:
        """Main deduplication method following BERTopic's approach"""
        start_time = datetime.now()
        
        # Step 1: Extract idea->code mappings
        idea_mappings = self._extract_idea_mappings(reasoning_results)
        
        if not idea_mappings:
            return self._create_empty_results(start_time)
        
        # Step 2: Create documents per code (class)
        code_documents = self._create_code_documents(idea_mappings)
        code_names = list(code_documents.keys())
        
        if len(code_names) < 2:
            return self._create_no_merge_results(reasoning_results, start_time)
        
        # Step 3: Create c-TF-IDF representations
        documents = list(code_documents.values())
        bow_matrix = self.vectorizer.fit_transform(documents)
        self.ctfidf.fit(bow_matrix)
        ctfidf_matrix = self.ctfidf.transform(bow_matrix)
        
        # Step 4: Compute similarity matrix
        similarity_matrix = cosine_similarity(ctfidf_matrix)
        
        # Step 5: Find similar pairs above threshold
        similarity_pairs = []
        for i in range(len(code_names)):
            for j in range(i + 1, len(code_names)):
                similarity = similarity_matrix[i, j]
                if similarity >= self.similarity_threshold:
                    similarity_pairs.append({
                        'code1': code_names[i],
                        'code2': code_names[j],
                        'similarity': float(similarity)
                    })
        
        # Step 6: Build connected components (merge groups)
        merge_groups = self._build_connected_components(code_names, similarity_pairs)
        
        # Step 7: Create deduplicated codebook
        deduplicated_codebook = self._create_deduplicated_codebook(
            reasoning_results.codebook, 
            merge_groups,
            idea_mappings
        )
        
        # Step 8: Compile results
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return DeduplicationResults(
            original_codebook=reasoning_results.codebook,
            deduplicated_codebook=deduplicated_codebook,
            merge_groups=merge_groups,
            similarity_pairs=similarity_pairs,
            ctfidf_matrix=ctfidf_matrix,
            processing_stats={
                'original_code_count': len(reasoning_results.codebook),
                'deduplicated_code_count': len(deduplicated_codebook),
                'codes_merged': len(reasoning_results.codebook) - len(deduplicated_codebook),
                'processing_time_seconds': processing_time,
                'similarity_threshold': self.similarity_threshold,
                'similar_pairs_found': len(similarity_pairs),
                'merge_groups_created': len(merge_groups),
                'language': self.language,
                'total_idea_mappings': len(idea_mappings)
            },
            timestamp=start_time.isoformat()
        )
    
    def _extract_idea_mappings(self, reasoning_results: CodeGeneratorReasoningResults) -> List[IdeaCodeMapping]:
        """Extract idea->code mappings from reasoning results"""
        mappings = []
        
        logger.info(f"Extracting idea mappings from {len(reasoning_results.step1_inputs)} clusters")
        
        for cluster_id, step1_data in reasoning_results.step1_inputs.items():
            # Extract ideas from cluster_text format
            sampled_ideas = None
            
            if 'cluster_text' in step1_data:
                cluster_text = step1_data['cluster_text']
                if cluster_text:
                    ideas = [idea.strip() for idea in cluster_text.split('\n') if idea.strip()]
                    sampled_ideas = [idea[2:].strip() if idea.startswith('- ') else idea for idea in ideas]
            elif 'ideas' in step1_data:
                sampled_ideas = step1_data['ideas']
            
            if not sampled_ideas:
                continue
            
            # Get the validated code for this cluster
            code_info = self._get_code_for_cluster(cluster_id, reasoning_results)
            
            if code_info:
                mapping = IdeaCodeMapping(
                    cluster_id=str(cluster_id),
                    theme_id=code_info.get('theme_id', ''),
                    original_ideas=sampled_ideas,
                    final_code=code_info['code'],
                    final_definition=code_info['definition']
                )
                mappings.append(mapping)
        
        logger.info(f"Extracted {len(mappings)} idea mappings")
        return mappings
    
    def _get_code_for_cluster(self, cluster_id: str, reasoning_results: CodeGeneratorReasoningResults) -> Dict[str, str]:
        """Get the final code for a cluster"""
        # Check step4_validations first
        if hasattr(reasoning_results, 'step4_validations') and reasoning_results.step4_validations:
            if cluster_id in reasoning_results.step4_validations:
                validation_data = reasoning_results.step4_validations[cluster_id]
                if isinstance(validation_data, dict) and 'code_validation' in validation_data:
                    code_validation = validation_data['code_validation']
                    if 'validated_code' in code_validation:
                        validated_code = code_validation['validated_code']
                        return {
                            'code': validated_code.get('code', ''),
                            'definition': validated_code.get('definition', ''),
                            'theme_id': code_validation.get('theme_name', '')
                        }
        
        # # Fallback: check codebook for matching source_cluster_id
        # if hasattr(reasoning_results, 'codebook') and reasoning_results.codebook:
        #     for code_entry in reasoning_results.codebook:
        #         if isinstance(code_entry, dict):
        #             source_cluster = str(code_entry.get('source_cluster_id', ''))
        #             if source_cluster == str(cluster_id):
        #                 return {
        #                     'code': code_entry.get('code', ''),
        #                     'definition': code_entry.get('definition', ''),
        #                     'theme_id': code_entry.get('theme_id', '')
        #                 }
        
        return None
    
    def _build_connected_components(self, code_names: List[str], similarity_pairs: List[Dict[str, Any]]) -> Dict[int, List[str]]:
        """Build connected components from similarity pairs (transitive closure)"""
        # Create adjacency list
        adjacency = defaultdict(set)
        for pair in similarity_pairs:
            code1, code2 = pair['code1'], pair['code2']
            adjacency[code1].add(code2)
            adjacency[code2].add(code1)
        
        # Find connected components using DFS
        visited = set()
        components = {}
        component_id = 0
        
        for code in code_names:
            if code not in visited:
                # Start new component
                component = []
                stack = [code]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        component.append(current)
                        # Add neighbors to stack
                        for neighbor in adjacency[current]:
                            if neighbor not in visited:
                                stack.append(neighbor)
                
                components[component_id] = component
                component_id += 1
        
        return components
    
    def _create_code_documents(self, idea_mappings: List[IdeaCodeMapping]) -> Dict[str, str]:
        """Create a single document per code by joining all its ideas"""
        code_documents = defaultdict(list)
        
        for mapping in idea_mappings:
            # Process ideas through spaCy
            processed_ideas = []
            for idea in mapping.original_ideas:
                doc = self.nlp(idea.lower())
                # Extract meaningful tokens (nouns, adjectives, verbs)
                tokens = [
                    token.lemma_ for token in doc
                    if not token.is_stop and not token.is_punct and not token.is_space
                    and token.pos_ in {'NOUN', 'ADJ', 'VERB'}
                    and len(token.lemma_) > 2
                ]
                processed_ideas.append(' '.join(tokens))
            
            # Join all processed ideas for this code
            code_documents[mapping.final_code].extend(processed_ideas)
        
        # Create final documents by joining all ideas per code
        return {code: ' '.join(ideas) for code, ideas in code_documents.items()}
    
    def _create_deduplicated_codebook(
        self, 
        original_codebook: List[Dict[str, str]], 
        merge_groups: Dict[int, List[str]],
        idea_mappings: List[IdeaCodeMapping]
    ) -> List[Dict[str, str]]:
        """Create deduplicated codebook by selecting primary code from each group"""
        
        # Count ideas per code for deciding which to keep
        code_idea_counts = defaultdict(int)
        for mapping in idea_mappings:
            code_idea_counts[mapping.final_code] += len(mapping.original_ideas)
        
        # Create code lookup
        code_lookup = {entry['code']: entry for entry in original_codebook}
        
        deduplicated_codebook = []
        
        for group_id, codes_in_group in merge_groups.items():
            if len(codes_in_group) == 1:
                # Single code in group - keep as is
                code = codes_in_group[0]
                if code in code_lookup:
                    deduplicated_codebook.append(code_lookup[code])
            else:
                # Multiple codes - select primary based on idea count
                primary_code = max(codes_in_group, key=lambda c: code_idea_counts.get(c, 0))
                secondary_codes = [c for c in codes_in_group if c != primary_code]
                
                # Update definition to note merged codes
                if primary_code in code_lookup:
                    entry = code_lookup[primary_code].copy()
                    if secondary_codes:
                        entry['definition'] = f"{entry['definition']} [Merged: {', '.join(secondary_codes)}]"
                    deduplicated_codebook.append(entry)
        
        return deduplicated_codebook
    
    def _create_empty_results(self, start_time: datetime) -> DeduplicationResults:
        """Create empty results when no mappings found"""
        return DeduplicationResults(
            original_codebook=[],
            deduplicated_codebook=[],
            merge_groups={},
            similarity_pairs=[],
            ctfidf_matrix=None,
            processing_stats={'error': 'No idea mappings found'},
            timestamp=start_time.isoformat()
        )
    
    def _create_no_merge_results(self, reasoning_results: CodeGeneratorReasoningResults, start_time: datetime) -> DeduplicationResults:
        """Create results when no merging needed"""
        return DeduplicationResults(
            original_codebook=reasoning_results.codebook,
            deduplicated_codebook=reasoning_results.codebook,
            merge_groups={i: [code['code']] for i, code in enumerate(reasoning_results.codebook)},
            similarity_pairs=[],
            ctfidf_matrix=None,
            processing_stats={
                'original_code_count': len(reasoning_results.codebook),
                'deduplicated_code_count': len(reasoning_results.codebook),
                'codes_merged': 0,
                'similarity_threshold': self.similarity_threshold,
                'note': 'Less than 2 codes - no deduplication needed'
            },
            timestamp=start_time.isoformat()
        )

# === UTILITY FUNCTIONS ========================================================================================================

def deduplicate_codebook(
    reasoning_results: CodeGeneratorReasoningResults,
    similarity_threshold: float = 0.9,
    language: str = DEFAULT_LANGUAGE
) -> DeduplicationResults:
    """
    Main entry point for codebook deduplication using BERTopic's c-TF-IDF approach.
    
    Args:
        reasoning_results: Results from code generation
        similarity_threshold: Minimum cosine similarity to merge codes (0.0-1.0)
        language: Language for processing
    """
    deduplicator = CodebookDeduplicator(similarity_threshold, language)
    return deduplicator.deduplicate(reasoning_results)

def print_deduplication_report(results: DeduplicationResults):
    """Print a formatted report of deduplication results"""
    stats = results.processing_stats
    
    print(f"\n{'='*60}")
    print("CODEBOOK DEDUPLICATION REPORT")
    print(f"{'='*60}")
    print(f"Timestamp: {results.timestamp}")
    print(f"Language: {stats.get('language', 'unknown')}")
    print(f"Similarity Threshold: {stats.get('similarity_threshold', 'unknown')}")
    print(f"\nOriginal Codes: {stats.get('original_code_count', 0)}")
    print(f"Final Codes: {stats.get('deduplicated_code_count', 0)}")
    print(f"Codes Merged: {stats.get('codes_merged', 0)}")
    print(f"Reduction: {stats.get('codes_merged', 0) / max(stats.get('original_code_count', 1), 1) * 100:.1f}%")
    print(f"Similar Pairs Found: {stats.get('similar_pairs_found', 0)}")
    print(f"Merge Groups Created: {stats.get('merge_groups_created', 0)}")
    
    if 'error' in stats:
        print(f"\n⚠️  ERROR: {stats['error']}")
    
    if results.similarity_pairs:
        print(f"\nSIMILARITY PAIRS (threshold ≥ {stats.get('similarity_threshold', 0)}):")
        print(f"{'-'*40}")
        for pair in results.similarity_pairs[:10]:  # Show first 10 pairs
            print(f"{pair['code1']} ↔ {pair['code2']} (similarity: {pair['similarity']:.3f})")
        if len(results.similarity_pairs) > 10:
            print(f"... and {len(results.similarity_pairs) - 10} more pairs")
    
    if results.merge_groups:
        print(f"\nMERGE GROUPS ({len(results.merge_groups)} groups):")
        print(f"{'-'*40}")
        for group_id, codes in results.merge_groups.items():
            if len(codes) > 1:
                print(f"Group {group_id}: {' + '.join(codes)}")
    
    print(f"\nProcessing completed in {stats.get('processing_time_seconds', 0):.2f} seconds")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    print("Codebook Deduplicator - Using BERTopic's c-TF-IDF approach")
    print("Usage: from utils.codebookDeduplicator import deduplicate_codebook")