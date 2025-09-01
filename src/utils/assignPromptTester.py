#!/usr/bin/env python3
"""
Simple Prompt Tester for CodeAssigner Assignment Prompts
"""

import os
import sys
import random
import pickle
import numpy as np
from typing import Optional, Union, List, Dict

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import CacheConfig, DEFAULT_LANGUAGE, EmbeddingConfig, get_embedding_dimensions
from utils.codeAssigner import CodeAssigner, EmbeddingLoader
from prompts import CODE_ASSIGNMENT_PROMPT
import models
from sklearn.metrics.pairwise import cosine_similarity

class AssignPromptTester:
    """Simple prompt tester for code assignment prompts"""
    
    def __init__(self, var_lab: str, language: str = DEFAULT_LANGUAGE, num_samples: int = 5):
        self.var_lab = var_lab
        self.language = language
        self.num_samples = num_samples
        self.cache_config = CacheConfig()
        
        # Load data
        self.codebook = self._load_enriched_codebook()
        self.cached_ideas = self._load_cached_ideas()
        
        if not self.codebook:
            print("ERROR: No enriched codebook found!")
            return
        
        if not self.cached_ideas:
            print("ERROR: No cached ideas found!")
            return
        
        print(f"SUCCESS: Loaded {len(self.codebook)} codes and {len(self.cached_ideas)} ideas")
        
        # Prepare code embeddings for similarity matching
        self._prepare_code_embeddings()
    
    def _load_enriched_codebook(self):
        """Load enriched codebook from cache"""
        cache_dir = self.cache_config.cache_dir
        codebook_files = list(cache_dir.glob("008_theme_identification_*.pkl"))
        
        if not codebook_files:
            print("ERROR: No enriched codebook found")
            return None
        
        codebook_file = max(codebook_files, key=lambda f: f.stat().st_mtime)
        
        try:
            with open(codebook_file, 'rb') as f:
                data = pickle.load(f)
            print(f"SUCCESS: Loaded enriched codebook: {codebook_file.name}")
            
            # Extract the enriched codebook from the model
            if isinstance(data, list) and len(data) > 0:
                enriched_model = data[0]
                if hasattr(enriched_model, 'enriched_codebook'):
                    return enriched_model.enriched_codebook
                else:
                    print("ERROR: No enriched_codebook attribute found")
                    return None
            else:
                print("ERROR: Unexpected data format")
                return None
                
        except Exception as e:
            print(f"ERROR: Error loading enriched codebook: {e}")
            return None
    
    def _load_cached_ideas(self):
        """Load cached idea embeddings"""
        cache_dir = self.cache_config.cache_dir
        cache_manager = type('CacheManager', (), {
            'cache_dir': cache_dir,
            'load_from_cache': lambda self, filename, step, model_class: self._load_embeddings()
        })()
        
        def _load_embeddings():
            embedding_files = list(cache_dir.glob("005_embeddings_*.pkl"))
            if not embedding_files:
                return []
            
            embedding_file = max(embedding_files, key=lambda f: f.stat().st_mtime)
            
            try:
                with open(embedding_file, 'rb') as f:
                    data = pickle.load(f)
                return data
            except Exception as e:
                print(f"ERROR loading embeddings: {e}")
                return []
        
        cache_manager._load_embeddings = _load_embeddings
        
        # Use EmbeddingLoader to extract ideas
        filename = "dummy"  # Not used in this context
        return EmbeddingLoader.load_idea_embeddings_from_cache(cache_manager, filename)
    
    def _prepare_code_embeddings(self):
        """Prepare code embeddings for similarity calculations"""
        print("Preparing code embeddings for similarity matching...")
        
        # Format codes for embedding (using definition only as per EmbeddingLoader)
        code_texts = [code.definition for code in self.codebook]
        
        # For testing purposes, create dummy embeddings
        # In real usage, these would come from the actual embedding generation
        embedding_config = EmbeddingConfig()
        dim = get_embedding_dimensions(embedding_config.embedding_model)
        
        # Generate random embeddings for demonstration
        # In real usage, you'd load these from cache or generate them
        np.random.seed(42)  # For consistent results
        self._code_embeddings = np.random.normal(0, 1, (len(self.codebook), dim))
        
        print(f"Generated {len(self._code_embeddings)} code embeddings with dimension {dim}")
    
    def _find_similar_codes(self, idea_embedding: np.ndarray, top_k: int = 5) -> List:
        """Find the top_k most similar codes to an idea"""
        if self._code_embeddings is None:
            return self.codebook[:top_k]  # Fallback
        
        similarities = cosine_similarity([idea_embedding], self._code_embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.codebook[i] for i in top_indices]
    
    def _sample_ideas(self, num_samples: int = None) -> List[Dict]:
        """Sample random ideas for testing"""
        if num_samples is None:
            num_samples = self.num_samples
        
        sample_size = min(num_samples, len(self.cached_ideas))
        return random.sample(self.cached_ideas, sample_size)
    
    def test_single_random_idea(self):
        """Test assignment prompt for ONE randomly selected idea"""
        print("\n" + "="*80)
        print("SINGLE RANDOM IDEA PROMPT TEST")
        print("="*80)
        
        if not self.cached_ideas:
            print("ERROR: No cached ideas available")
            return
        
        # Pick one random idea
        random_idea = random.choice(self.cached_ideas)
        
        print(f"Selected Random Idea:")
        print(f"  ID: {random_idea['idea_id']}")
        print(f"  Respondent: {random_idea['respondent_id']}")
        print(f"  Position: {self.cached_ideas.index(random_idea) + 1} of {len(self.cached_ideas)}")
        
        # Get idea details
        idea_id = random_idea['idea_id']
        idea_text = random_idea['idea']
        idea_embedding = np.array(random_idea['embedding'])
        
        print(f"\nIdea Text ({len(idea_text)} chars):")
        print(f"  '{idea_text}'")
        print(f"Embedding Shape: {idea_embedding.shape}")
        
        # Find similar codes
        top_k = 5
        similar_codes = self._find_similar_codes(idea_embedding, top_k=top_k)
        
        print(f"\nTop {top_k} Similar Codes:")
        for j, code in enumerate(similar_codes, 1):
            print(f"  {j}. {code.code}")
            print(f"     Definition: {code.definition}")
            if hasattr(code, 'theme') and code.theme:
                print(f"     Theme: {code.theme}")
        
        # Format candidate codes for prompt
        candidate_codes_text = "\\n".join([
            f"Code: {code.definition}\\n"
            for code in similar_codes
        ])
        
        # Create the prompt
        prompt = CODE_ASSIGNMENT_PROMPT.format(
            language=self.language,
            var_lab=self.var_lab,
            idea_id=idea_id,
            idea_text=idea_text,
            candidate_codes=candidate_codes_text
        )
        
        print(f"\n{'='*60}")
        print("FORMATTED PROMPT:")
        print(f"{'='*60}")
        print(prompt)
        print("="*60)
        
        return random_idea, prompt
    
    def quick_random_test(self, show_codes: bool = True, show_full_prompt: bool = True):
        """Quick random test suitable for pipeline integration"""
        if not self.cached_ideas:
            return None, None
        
        # Pick random idea
        random_idea = random.choice(self.cached_ideas)
        idea_id = random_idea['idea_id']
        idea_text = random_idea['idea']
        idea_embedding = np.array(random_idea['embedding'])
        
        print(f"\n🎯 RANDOM PROMPT DEBUG:")
        print(f"  Random Idea: {idea_id} (Respondent: {random_idea['respondent_id']})")
        print(f"  Idea: {idea_text}")
        
        if show_codes:
            # Find similar codes
            similar_codes = self._find_similar_codes(idea_embedding, top_k=5)
            print(f"  Top 5 Similar Codes: {[code.code for code in similar_codes]}")
        
        if show_full_prompt:
            # Create prompt
            similar_codes = self._find_similar_codes(idea_embedding, top_k=5)
            candidate_codes_text = "\\n".join([
                f"Code: {code.definition}\\n"
                for code in similar_codes
            ])
            
            prompt = CODE_ASSIGNMENT_PROMPT.format(
                language=self.language,
                var_lab=self.var_lab,
                idea_id=idea_id,
                idea_text=idea_text,
                candidate_codes=candidate_codes_text
            )
            
            print(f"\n{'─'*40}")
            print("FULL PROMPT:")
            print(f"{'─'*40}")
            print(prompt)
            print("─"*40)
            
            return random_idea, prompt
        
        return random_idea, None
    
    def test_assignment_prompts(self, specific_idea_id: Optional[str] = None):
        """Test code assignment prompts for sampled ideas"""
        print("\n" + "="*80)
        print("CODE ASSIGNMENT PROMPT TESTING")
        print("="*80)
        
        if specific_idea_id:
            # Find specific idea
            test_ideas = [idea for idea in self.cached_ideas if idea['idea_id'] == specific_idea_id]
            if not test_ideas:
                print(f"ERROR: Idea ID '{specific_idea_id}' not found")
                return
            print(f"\nTesting specific idea: {specific_idea_id}")
        else:
            # Sample random ideas
            test_ideas = self._sample_ideas()
            print(f"\nTesting {len(test_ideas)} randomly sampled ideas:")
        
        for i, idea_data in enumerate(test_ideas, 1):
            print(f"\n{'-'*60}")
            print(f"IDEA {i}: {idea_data['idea_id']}")
            print(f"{'-'*60}")
            
            # Get idea details
            idea_id = idea_data['idea_id']
            idea_text = idea_data['idea']
            idea_embedding = np.array(idea_data['embedding'])
            respondent_id = idea_data['respondent_id']
            
            print(f"Respondent ID: {respondent_id}")
            print(f"Idea Text: {idea_text}")
            print(f"Idea Length: {len(idea_text)} characters")
            print(f"Embedding Shape: {idea_embedding.shape}")
            
            # Find similar codes
            top_k = 5  # Match CodeAssigner default
            similar_codes = self._find_similar_codes(idea_embedding, top_k=top_k)
            
            print(f"\nTop {top_k} Similar Codes:")
            for j, code in enumerate(similar_codes, 1):
                print(f"  {j}. Code: {code.code}")
                print(f"     Definition: {code.definition}")
                if hasattr(code, 'theme') and code.theme:
                    print(f"     Theme: {code.theme}")
            
            # Format candidate codes for prompt (matching CodeAssigner format)
            candidate_codes_text = "\\n".join([
                f"Code: {code.definition}\\n"
                for code in similar_codes
            ])
            
            # Create the prompt
            prompt = CODE_ASSIGNMENT_PROMPT.format(
                language=self.language,
                var_lab=self.var_lab,
                idea_id=idea_id,
                idea_text=idea_text,
                candidate_codes=candidate_codes_text
            )
            
            print(f"\n{'='*40}")
            print("FORMATTED PROMPT:")
            print(f"{'='*40}")
            print(prompt)
            
            if i < len(test_ideas):
                input(f"\n-->  Press Enter for next idea ({i+1}/{len(test_ideas)})...")
        
        print("\nSUCCESS: All assignment prompts tested!")
    
    def list_available_ideas(self, limit: int = 20):
        """List available ideas for testing"""
        print(f"\n{'='*60}")
        print("AVAILABLE IDEAS FOR TESTING")
        print(f"{'='*60}")
        
        print(f"Total ideas available: {len(self.cached_ideas)}")
        print(f"Showing first {min(limit, len(self.cached_ideas))} ideas:")
        
        for i, idea_data in enumerate(self.cached_ideas[:limit], 1):
            idea_id = idea_data['idea_id']
            idea_text = idea_data['idea']
            respondent_id = idea_data['respondent_id']
            
            # Truncate long ideas for display
            display_text = idea_text[:100] + "..." if len(idea_text) > 100 else idea_text
            
            print(f"{i:3d}. ID: {idea_id} | Resp: {respondent_id}")
            print(f"     Text: {display_text}")
    
    def show_codebook_summary(self):
        """Show summary of available codebook"""
        print(f"\n{'='*60}")
        print("CODEBOOK SUMMARY")
        print(f"{'='*60}")
        
        print(f"Total codes: {len(self.codebook)}")
        
        # Group by theme if available
        themes = {}
        for code in self.codebook:
            theme = getattr(code, 'theme', 'No Theme')
            if theme not in themes:
                themes[theme] = []
            themes[theme].append(code)
        
        print(f"Themes: {len(themes)}")
        
        for theme, codes in themes.items():
            print(f"\nTheme: {theme} ({len(codes)} codes)")
            for code in codes[:3]:  # Show first 3 codes per theme
                print(f"  - {code.code}: {code.definition[:80]}...")
            if len(codes) > 3:
                print(f"    ... and {len(codes) - 3} more codes")


def main(var_lab=None, idea_id=None, num_samples=5):
    """Main function with optional parameters
    
    Args:
        var_lab: Survey question (default: None to prompt user)
        idea_id: Specific idea ID to test (default: None for random sampling)
        num_samples: Number of ideas to sample for testing (default: 5)
    """
    print("Assignment Prompt Tester")
    
    # Handle var_lab - use parameter or prompt
    if var_lab is None:
        var_lab = input("Enter survey question (var_lab): ").strip()
        if not var_lab:
            print("ERROR: var_lab is required")
            return
    
    # Display testing info
    if idea_id is not None:
        print(f"Testing specific idea: {idea_id}")
    else:
        print(f"Testing {num_samples} random ideas")
    
    tester = AssignPromptTester(var_lab=var_lab, num_samples=num_samples)
    
    if not tester.codebook or not tester.cached_ideas:
        return
    
    print("\nOptions:")
    print("1. Test assignment prompts (random ideas)")
    print("2. Test assignment prompts (specific idea ID)")
    print("3. List available ideas")
    print("4. Show codebook summary")
    print("5. Test all (prompts + summaries)")
    print("6. Quick random test (single idea)")
    
    choice = input("\nChoose (1-6): ").strip()
    
    if choice == '1':
        tester.test_assignment_prompts()
    elif choice == '2':
        if idea_id is None:
            idea_id = input("Enter idea ID: ").strip()
        tester.test_assignment_prompts(specific_idea_id=idea_id)
    elif choice == '3':
        tester.list_available_ideas()
    elif choice == '4':
        tester.show_codebook_summary()
    elif choice == '5':
        tester.show_codebook_summary()
        input("\n-->  Press Enter for ideas list...")
        tester.list_available_ideas()
        input("\n-->  Press Enter for prompt testing...")
        tester.test_assignment_prompts()
    elif choice == '6':
        tester.test_single_random_idea()
    else:
        print("ERROR: Invalid choice")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test CodeAssigner assignment prompts')
    parser.add_argument('--var-lab', help='Survey question (default: prompt user)')
    parser.add_argument('--idea-id', help='Specific idea ID to test (default: random sampling)')
    parser.add_argument('--num-samples', type=int, default=5, help='Number of ideas to sample (default: 5)')
    
    args = parser.parse_args()
    
    main(var_lab=args.var_lab, idea_id=args.idea_id, num_samples=args.num_samples)