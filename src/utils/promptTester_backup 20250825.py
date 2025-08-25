#!/usr/bin/env python3
"""
Simple Prompt Tester for CodeGenerator 4-Chain Prompts
"""

import os
import sys
import random
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import CacheConfig, DEFAULT_LANGUAGE
import models
from prompts import (
    CLUSTER_SUMMARY_PROMPT,
    CANDIDATE_CODE_SELECTION_PROMPT,
    CODE_GENERATION_PROMPT,
    VALIDATION_PROMPT
)


class SimplePromptTester:
    """Simple prompt tester for Step 7 codeGenerator prompts"""
    
    def __init__(self, var_lab: str, language: str = DEFAULT_LANGUAGE):
        self.var_lab = var_lab
        self.language = language
        self.cache_config = CacheConfig()
        
        # Load data
        self.initial_cluster_results = self._load_cluster_results()
        self.codebook_reasoning = self._load_codebook_reasoning()
        
        if not self.initial_cluster_results:
            print("ERROR: ERROR: No cluster results found!")
            return
        
        if not self.codebook_reasoning:
            print("ERROR: ERROR: No codebook reasoning found!")
            return
        
        # Sample a random cluster ID to use throughout
        self.cluster_id = self._sample_cluster_id()
        print(f"TARGET: Using cluster ID: {self.cluster_id}")
        available_steps = []
        for step in ['step1', 'step2', 'step3', 'step4']:
            field_name = f'{step}_summaries' if step == 'step1' else f'{step}_analysis' if step == 'step2' else f'{step}_recommendations' if step == 'step3' else f'{step}_validations'
            field_data = getattr(self.codebook_reasoning, field_name, {})
            if self.cluster_id in field_data:
                available_steps.append(step)
        print(f"   Available steps for this cluster: {available_steps}")
    
    def _load_cluster_results(self):
        """Load initial_cluster_results from cache"""
        cache_dir = self.cache_config.cache_dir
        cluster_files = list(cache_dir.glob("006_initial_clusters_*.pkl"))
        
        if not cluster_files:
            print("ERROR: No cluster results found")
            return None
        
        cluster_file = max(cluster_files, key=lambda f: f.stat().st_mtime)
        
        try:
            with open(cluster_file, 'rb') as f:
                data = pickle.load(f)
            print(f"SUCCESS: Loaded cluster results: {cluster_file.name}")
            return data
        except Exception as e:
            print(f"ERROR: Error loading cluster results: {e}")
            return None
    
    def _load_codebook_reasoning(self):
        """Load codebook reasoning from cache and reconstruct Pydantic model"""
        cache_dir = self.cache_config.cache_dir
        reasoning_files = list(cache_dir.glob("999_codebook_generation_reasoning_*.pkl"))
        
        if not reasoning_files:
            print("ERROR: No codebook reasoning found")
            return None
        
        reasoning_file = max(reasoning_files, key=lambda f: f.stat().st_mtime)
        
        try:
            with open(reasoning_file, 'rb') as f:
                data = pickle.load(f)
            print(f"SUCCESS: Loaded codebook reasoning: {reasoning_file.name}")
            
            # Handle list format (extract first element)
            if isinstance(data, list) and len(data) > 0:
                dict_data = data[0]
            else:
                dict_data = data
            
            # Convert dictionary back to Pydantic model
            if isinstance(dict_data, dict):
                return models.CodeGeneratorReasoningResults.model_validate(dict_data)
            else:
                # Already a Pydantic model
                return dict_data
                
        except Exception as e:
            print(f"ERROR: Error loading codebook reasoning: {e}")
            return None
    
    def _sample_cluster_id(self):
        """Sample a random cluster ID that has data in ALL steps"""
        # First get all possible cluster IDs
        all_cluster_ids = list(set([
            response_idea['initial_cluster'] 
            for result in self.initial_cluster_results 
            for response_idea in result['response_ideas']   
            if response_idea['initial_cluster'] is not None and response_idea['initial_cluster'] != -1]))
        
        # Filter to only clusters that have data in all 4 steps
        complete_clusters = []
        for cluster_id in all_cluster_ids:
            if (cluster_id in self.codebook_reasoning.step1_summaries and
                cluster_id in self.codebook_reasoning.step2_analysis and
                cluster_id in self.codebook_reasoning.step3_recommendations and
                cluster_id in getattr(self.codebook_reasoning, 'step4_validations', {})):
                complete_clusters.append(cluster_id)
        
        if not complete_clusters:
            print("WARNING:  Warning: No clusters found with complete 4-step data, using any cluster with step1 data")
            # Fallback to any cluster with step1 data
            step1_clusters = list(self.codebook_reasoning.step1_summaries.keys())
            return random.choice(step1_clusters) if step1_clusters else all_cluster_ids[0]
        
        return random.sample(complete_clusters, 1)[0]
    
    def _get_cluster_text(self):
        """Get cluster text for the sampled cluster"""
        cluster_segments = []
        for result in self.initial_cluster_results:
            for response_idea in result['response_ideas']:   
                if response_idea['initial_cluster'] == self.cluster_id:
                    cluster_segments.append(response_idea['idea'])
        
        sampled_segments = random.sample(cluster_segments, min(10, len(cluster_segments)))
        return "\n".join(sampled_segments)
    
    def test_prompt_1(self):
        """Test Prompt 1: Cluster Summary"""
        print("\n" + "="*80)
        print("PROMPT 1: CLUSTER SUMMARY")
        print("="*80)
        
        # Use EXACT same inputs as actual pipeline
        step1_inputs = getattr(self.codebook_reasoning, 'step1_inputs', {})
        if self.cluster_id in step1_inputs:
            step1_input = step1_inputs[self.cluster_id]
            print(f"\n[USING ACTUAL PIPELINE INPUTS]\n")
        else:
            # Fallback: reconstruct (but show warning)
            cluster_text = self._get_cluster_text()
            step1_input = {
                "language": self.language,
                "survey_question": self.var_lab,
                "cluster_text": cluster_text
            }
            print(f"\n[WARNING: Reconstructing inputs - may not match actual pipeline]\n")
        
        print(f"Input Summary:")
        print(f"  Language: {step1_input.get('language', 'Unknown')}")
        print(f"  Survey Question: {step1_input.get('survey_question', 'Unknown')}")
        cluster_text = step1_input.get('cluster_text', '')
        if cluster_text:
            ideas = cluster_text.split('\n')
            print(f"  Cluster Ideas: {len(ideas)} ideas, {len(cluster_text)} characters")
        
        prompt = CLUSTER_SUMMARY_PROMPT.format(**step1_input)
        print(f"\n{'-'*60}\nFORMATTED PROMPT:\n{'-'*60}\n")
        print(prompt)
    
    def test_prompt_2(self):
        """Test Prompt 2: Candidate Code Selection"""
        print("\n" + "="*80)
        print("PROMPT 2: CANDIDATE CODE SELECTION")
        print("="*80)
        
        # Use EXACT same inputs as actual pipeline
        step2_inputs = getattr(self.codebook_reasoning, 'step2_inputs', {})
        if self.cluster_id in step2_inputs:
            step2_input = step2_inputs[self.cluster_id]
            print(f"\n[USING ACTUAL PIPELINE INPUTS]\n")
        else:
            print(f"\n[ERROR: No Step 2 inputs found for cluster {self.cluster_id}]")
            print("Cannot display Prompt 2 without actual pipeline inputs")
            return
        
        print(f"Input Summary:")
        print(f"  Language: {step2_input.get('language', 'Unknown')}")
        print(f"  Survey Question: {step2_input.get('survey_question', 'Unknown')}")
        print(f"  Cluster Summary: {step2_input.get('cluster_summary', 'Unknown')}")
        
        code_text = step2_input.get('code_text', '')
        if code_text:
            code_lines = [line.strip() for line in code_text.split('\n') if line.strip() and line.strip().startswith('-')]
            print(f"  Nearest Codes: {len(code_lines)} codes, {len(code_text)} characters")
        
        prompt = CANDIDATE_CODE_SELECTION_PROMPT.format(**step2_input)
        print(f"\n{'-'*60}\nFORMATTED PROMPT:\n{'-'*60}\n")
        print(prompt)
    
    def test_prompt_3(self):
        """Test Prompt 3: Code Generation"""
        print("\n" + "="*80)
        print("PROMPT 3: CODE GENERATION")
        print("="*80)
        
        # Use EXACT same inputs as actual pipeline
        step3_inputs = getattr(self.codebook_reasoning, 'step3_inputs', {})
        if self.cluster_id in step3_inputs:
            step3_input = step3_inputs[self.cluster_id]
            print(f"\n[USING ACTUAL PIPELINE INPUTS]\n")
        else:
            print(f"\n[ERROR: No Step 3 inputs found for cluster {self.cluster_id}]")
            print("Cannot display Prompt 3 without actual pipeline inputs")
            return
        
        print(f"Input Summary:")
        print(f"  Language: {step3_input.get('language', 'Unknown')}")
        print(f"  Survey Question: {step3_input.get('survey_question', 'Unknown')}")
        print(f"  Cluster Summary: {step3_input.get('cluster_summary', 'Unknown')}")
        
        candidate_codes = step3_input.get('candidate_codes', '')
        if candidate_codes:
            code_lines = [line.strip() for line in candidate_codes.split('\n') if line.strip() and line.strip().startswith('-')]
            print(f"  Candidate Codes: {len(code_lines)} codes, {len(candidate_codes)} characters")
        
        prompt = CODE_GENERATION_PROMPT.format(**step3_input)
        print(f"\n{'-'*60}\nFORMATTED PROMPT:\n{'-'*60}\n")
        print(prompt)
    
    def test_prompt_4(self):
        """Test Prompt 4: Validation"""
        print("\n" + "="*80)
        print("PROMPT 4: VALIDATION")
        print("="*80)
        
        # Use EXACT same inputs as actual pipeline
        step4_inputs = getattr(self.codebook_reasoning, 'step4_inputs', {})
        if self.cluster_id in step4_inputs:
            step4_input = step4_inputs[self.cluster_id]
            print(f"\n[USING ACTUAL PIPELINE INPUTS]\n")
        else:
            print(f"\n[ERROR: No Step 4 inputs found for cluster {self.cluster_id}]")
            print("Cannot display Prompt 4 without actual pipeline inputs")
            return
        
        print(f"Input Summary:")
        print(f"  Language: {step4_input.get('language', 'Unknown')}")
        print(f"  Survey Question: {step4_input.get('survey_question', 'Unknown')}")
        print(f"  Cluster Summary: {step4_input.get('cluster_summary', 'Unknown')}")
        
        candidate_codes = step4_input.get('candidate_codes', '')
        if candidate_codes:
            code_lines = [line.strip() for line in candidate_codes.split('\n') if line.strip() and line.strip().startswith('-')]
            print(f"  Candidate Codes: {len(code_lines)} codes, {len(candidate_codes)} characters")
        
        step3_recommendation = step4_input.get('step3_recommendation', '')
        if step3_recommendation:
            print(f"  Step 3 Recommendation: {len(step3_recommendation)} characters")
        
        prompt = VALIDATION_PROMPT.format(**step4_input)
        print(f"\n{'-'*60}\nFORMATTED PROMPT:\n{'-'*60}\n")
        print(prompt)
    
    def test_all_prompts(self):
        """Test all 4 prompts in sequence"""
        if not self.initial_cluster_results or not self.codebook_reasoning:
            print("ERROR: Cannot test: Missing data")
            return
        
        self.test_prompt_1()
        input("\n-->  Press Enter for Prompt 2...")
        
        self.test_prompt_2()
        input("\n-->  Press Enter for Prompt 3...")
        
        self.test_prompt_3()
        input("\n-->  Press Enter for Prompt 4...")
        
        self.test_prompt_4()
        
        print("\nSUCCESS: All prompts tested!")


def main():
    """Simple main function"""
    print("TESTER: Simple CodeGenerator Prompt Tester")
    
    # You need to provide var_lab when calling this
    var_lab = input("Enter survey question (var_lab): ").strip()
    if not var_lab:
        print("ERROR: var_lab is required")
        return
    
    tester = SimplePromptTester(var_lab=var_lab)
    
    if not tester.initial_cluster_results or not tester.codebook_reasoning:
        return
    
    print("\nOptions:")
    print("1. Test Prompt 1 only")
    print("2. Test Prompt 2 only") 
    print("3. Test Prompt 3 only")
    print("4. Test Prompt 4 only")
    print("5. Test all prompts")
    
    choice = input("\nChoose (1-5): ").strip()
    
    if choice == '1':
        tester.test_prompt_1()
    elif choice == '2':
        tester.test_prompt_2()
    elif choice == '3':
        tester.test_prompt_3()
    elif choice == '4':
        tester.test_prompt_4()
    elif choice == '5':
        tester.test_all_prompts()
    else:
        print("ERROR: Invalid choice")


if __name__ == "__main__":
    main()