#!/usr/bin/env python3
"""
Prompt Tester for CodeGenerator 4-Chain Prompts
This utility allows testing of individual prompts or the full chain
for Step 7 (Code Generation) in the pipeline.
"""

import os
import sys
import random
import pickle
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import CacheConfig, DEFAULT_LANGUAGE
from prompts import (
    CLUSTER_SUMMARY_PROMPT,
    CANDIDATE_CODE_SELECTION_PROMPT,
    CODE_GENERATION_PROMPT,
    VALIDATION_PROMPT
)


class CodeGeneratorPromptTester:
    """Test the 4-chain prompts used in codeGenerator for Step 7"""
    
    def __init__(self, language: str = DEFAULT_LANGUAGE, var_lab: str = None):
        self.language = language
        self.cache_config = CacheConfig()
        self.cluster_data = None
        self.codebook = None
        self.var_lab = var_lab
        self.reasoning_data = None
        
        # Load cached data
        self._load_cluster_data()
        self._load_codebook()
        self._load_reasoning_cache()
    
    def _load_cluster_data(self):
        """Load cluster results from cache"""
        cache_dir = self.cache_config.cache_dir
        
        # Look for cluster results file (step 6 - initial clusters)
        cluster_files = list(cache_dir.glob("006_initial_clusters_*.pkl"))
        
        if not cluster_files:
            # Try alternative naming pattern
            cluster_files = list(cache_dir.glob("005_clusters_*.pkl"))
        
        if not cluster_files:
            print("❌ ERROR: No cluster data found in cache!")
            print(f"   Searched in: {cache_dir}")
            print("   Looking for: 006_initial_clusters_*.pkl or 005_clusters_*.pkl")
            print("   Please run the pipeline through step 6 first.")
            return
        
        # Use the most recent cluster file
        cluster_file = max(cluster_files, key=lambda f: f.stat().st_mtime)
        
        try:
            with open(cluster_file, 'rb') as f:
                self.cluster_data = pickle.load(f)
            print(f"✅ Loaded cluster data from: {cluster_file.name}")
            
            # Show var_lab if provided
            if self.var_lab:
                print(f"   Survey question: {self.var_lab}")
            else:
                print("   ⚠️  Warning: var_lab not provided - prompts will show 'None'")
        except Exception as e:
            print(f"❌ ERROR loading cluster data: {e}")
    
    def _load_codebook(self):
        """Load codebook from cache if available"""
        cache_dir = self.cache_config.cache_dir
        
        # Look for codebook files (step 7)
        codebook_files = list(cache_dir.glob("007_codebook_*.pkl"))
        
        if not codebook_files:
            print("⚠️  WARNING: No codebook found in cache")
            print("   Will use 'No candidate codes available' for prompts 2-4")
            return
        
        # Use the most recent codebook file
        codebook_file = max(codebook_files, key=lambda f: f.stat().st_mtime)
        
        try:
            with open(codebook_file, 'rb') as f:
                codebook_data = pickle.load(f)
                # Extract codes from the codebook data structure
                if isinstance(codebook_data, dict) and 'codes' in codebook_data:
                    self.codebook = codebook_data['codes']
                elif isinstance(codebook_data, list):
                    self.codebook = codebook_data
                else:
                    # Try to extract codes from the structure
                    self.codebook = self._extract_codes_from_data(codebook_data)
                    
            print(f"✅ Loaded codebook from: {codebook_file.name}")
            if self.codebook:
                print(f"   Total codes: {len(self.codebook)}")
        except Exception as e:
            print(f"⚠️  WARNING loading codebook: {e}")
            print("   Will use 'No candidate codes available' for prompts 2-4")
    
    def _extract_codes_from_data(self, data: Any) -> List[Dict[str, str]]:
        """Try to extract codes from various data structures"""
        codes = []
        
        # If it's a list of objects with code attributes
        if isinstance(data, list) and data:
            for item in data:
                if hasattr(item, 'codes'):
                    codes.extend(item.codes)
                elif hasattr(item, 'code') and hasattr(item, 'definition'):
                    codes.append({'code': item.code, 'definition': item.definition})
        
        return codes
    
    def _load_reasoning_cache(self):
        """Load codebook reasoning cache if available"""
        cache_dir = self.cache_config.cache_dir
        
        # Look for reasoning cache files
        reasoning_files = list(cache_dir.glob("999_codebook_generation_reasoning_*.pkl"))
        
        if not reasoning_files:
            print("⚠️  WARNING: No reasoning cache found")
            print("   Will use synthetic examples for prompts 2-4")
            return
        
        # Use the most recent reasoning file
        reasoning_file = max(reasoning_files, key=lambda f: f.stat().st_mtime)
        
        try:
            with open(reasoning_file, 'rb') as f:
                self.reasoning_data = pickle.load(f)
            print(f"✅ Loaded reasoning cache from: {reasoning_file.name}")
            
            # Show some stats
            if hasattr(self.reasoning_data, 'step1_summaries'):
                print(f"   Total clusters with reasoning: {len(self.reasoning_data.step1_summaries)}")
                # Update var_lab if not provided and available in reasoning
                if not self.var_lab and hasattr(self.reasoning_data, 'var_lab'):
                    self.var_lab = self.reasoning_data.var_lab
                    print(f"   Survey question from reasoning: {self.var_lab}")
        except Exception as e:
            print(f"⚠️  WARNING loading reasoning cache: {e}")
            print("   Will use synthetic examples for prompts 2-4")
    
    def _get_random_cluster(self) -> Dict[str, Any]:
        """Get a random cluster with its responses"""
        if not self.cluster_data:
            raise RuntimeError("No cluster data available")
        
        # Collect all clusters with their IDs
        cluster_map = {}
        for result in self.cluster_data:
            # Handle ClusterModel objects
            if hasattr(result, 'response_ideas'):
                for response_idea in result.response_ideas:
                    if hasattr(response_idea, 'initial_cluster') and response_idea.initial_cluster is not None and response_idea.initial_cluster != -1:
                        cluster_id = response_idea.initial_cluster
                        if cluster_id not in cluster_map:
                            cluster_map[cluster_id] = []
                        cluster_map[cluster_id].append(response_idea.idea)
            # Handle dict format (fallback)
            elif isinstance(result, dict) and 'response_ideas' in result:
                for response_idea in result['response_ideas']:
                    if response_idea.get('initial_cluster') is not None and response_idea.get('initial_cluster') != -1:
                        cluster_id = response_idea['initial_cluster']
                        if cluster_id not in cluster_map:
                            cluster_map[cluster_id] = []
                        cluster_map[cluster_id].append(response_idea.get('idea', ''))
        
        if not cluster_map:
            raise RuntimeError("No clustered responses found")
        
        # Select random cluster
        cluster_id = random.choice(list(cluster_map.keys()))
        responses = cluster_map[cluster_id]
        
        # Sample up to 10 responses from the cluster
        sampled_responses = random.sample(responses, min(10, len(responses)))
        
        return {
            'cluster_id': cluster_id,
            'responses': sampled_responses,
            'total_responses': len(responses)
        }
    
    def _format_cluster_text(self, responses: List[str]) -> str:
        """Format responses as cluster text"""
        return "\n".join(responses)
    
    def _generate_cluster_summary(self, themes: List[str], analyst_note: str = None) -> str:
        """Generate cluster summary from themes (mimics codeGenerator logic)"""
        if len(themes) == 1:
            theme = themes[0]
            summary = theme.replace("\\n", "\n").strip()
            if analyst_note:
                return f"Analyst note: {analyst_note}\n\n{summary}"
            return summary
        else:
            summary_parts = [f"Theme {i+1}: {theme}" for i, theme in enumerate(themes)]
            summary = "\n".join(summary_parts)
            if analyst_note:
                return f"Analyst note: {analyst_note}\n\n{summary}"
            return summary
    
    def _get_real_cluster_data(self, cluster_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Get real data from reasoning cache for a specific cluster"""
        if not self.reasoning_data or not hasattr(self.reasoning_data, 'step1_summaries'):
            return None
        
        # If no cluster_id specified, pick a random one that has complete data
        if cluster_id is None:
            # Find clusters that have data in all steps
            complete_clusters = []
            for cid in self.reasoning_data.step1_summaries.keys():
                if (cid in self.reasoning_data.step2_analysis and 
                    cid in self.reasoning_data.step3_recommendations and
                    cid in self.reasoning_data.step4_validations):
                    complete_clusters.append(cid)
            
            if not complete_clusters:
                return None
            
            cluster_id = random.choice(complete_clusters)
        
        # Extract data for this cluster
        result = {'cluster_id': cluster_id}
        
        # Step 1 data
        if cluster_id in self.reasoning_data.step1_summaries:
            step1 = self.reasoning_data.step1_summaries[cluster_id]
            result['themes'] = step1.get('themes', [])
            result['analyst_note'] = step1.get('analyst_note', '')
            result['cluster_summary'] = self._generate_cluster_summary(
                result['themes'], 
                result['analyst_note']
            )
        
        # Step 2 data
        if cluster_id in self.reasoning_data.step2_analysis:
            result['candidate_codes'] = self.reasoning_data.step2_analysis[cluster_id]
        
        # Step 3 data
        if cluster_id in self.reasoning_data.step3_recommendations:
            result['code_recommendation'] = self.reasoning_data.step3_recommendations[cluster_id]
        
        # Step 4 data
        if cluster_id in self.reasoning_data.step4_validations:
            result['validation'] = self.reasoning_data.step4_validations[cluster_id]
        
        return result
    
    def _format_candidate_codes(self, codes: List[Dict[str, str]]) -> str:
        """Format candidate codes for prompt input"""
        if not codes:
            return "No existing codes in codebook"
        
        return "\n".join([
            f"- {code.get('code', 'Unknown')}: {code.get('definition', 'No definition')}"
            for code in codes
        ])
    
    def _format_code_recommendation(self, recommendation: Dict[str, Any]) -> str:
        """Format code recommendation from step 3 for step 4 input"""
        parts = []
        
        # Extract cluster analysis
        if 'cluster_analysis' in recommendation:
            analysis = recommendation['cluster_analysis']
            parts.append(f"Number of themes: {analysis.get('number_of_themes', 'Unknown')}")
        
        # Extract coding decisions
        if 'coding_decisions' in recommendation:
            for decision in recommendation['coding_decisions']:
                parts.append(f"\nTheme {decision.get('theme_number', '?')}: {decision.get('theme_description', 'Unknown theme')}")
                parts.append(f"Decision: {decision.get('decision', 'unknown')}")
                
                action = decision.get('action_details', {})
                if decision.get('decision') == 'use_existing' and action.get('codes_to_use'):
                    parts.append(f"Codes to use: {', '.join(action['codes_to_use'])}")
                elif decision.get('decision') == 'modify_existing':
                    parts.append(f"Code to modify: {action.get('codes_to_modify', 'Unknown')}")
                    parts.append(f"New name: {action.get('modified_code_name', 'Unknown')}")
                    parts.append(f"New definition: {action.get('modified_code_definition', 'Unknown')}")
                elif decision.get('decision') == 'create_new':
                    parts.append(f"New code: {action.get('new_code_name', 'Unknown')}")
                    parts.append(f"Definition: {action.get('new_code_definition', 'Unknown')}")
                
                parts.append(f"Justification: {decision.get('justification', 'No justification provided')}")
        
        return "\n".join(parts)
    
    def test_cluster_summary_prompt(self, cluster_id: Optional[int] = None):
        """Test Prompt 1: Cluster Summary"""
        print("\n" + "="*80)
        print("PROMPT 1: CLUSTER SUMMARY")
        print("="*80 + "\n")
        
        if not self.cluster_data:
            print("❌ Cannot test: No cluster data available")
            return
        
        # Get cluster data
        cluster = self._get_random_cluster()
        cluster_text = self._format_cluster_text(cluster['responses'])
        
        # Populate prompt
        prompt = CLUSTER_SUMMARY_PROMPT.format(
            language=self.language,
            survey_question=self.var_lab,
            cluster_text=cluster_text
        )
        
        print(f"Cluster ID: {cluster['cluster_id']} (sampled {len(cluster['responses'])}/{cluster['total_responses']} responses)")
        print("-"*80)
        print(prompt)
        print("-"*80)
    
    def test_candidate_selection_prompt(self, cluster_id: Optional[int] = None):
        """Test Prompt 2: Candidate Code Selection"""
        print("\n" + "="*80)
        print("PROMPT 2: CANDIDATE CODE SELECTION")
        print("="*80 + "\n")
        
        if not self.cluster_data:
            print("❌ Cannot test: No cluster data available")
            return
        
        # Try to get real data first
        real_data = self._get_real_cluster_data(cluster_id)
        
        if real_data and 'cluster_summary' in real_data:
            # Use real cluster summary from step 1
            cluster_summary = real_data['cluster_summary']
            cluster_id = real_data['cluster_id']
            print(f"Using real data from cluster {cluster_id}")
        else:
            # Fallback to synthetic data
            print("Using synthetic data (no reasoning cache available)")
            sample_themes = [
                "Respondents express concern about environmental sustainability",
                "Focus on reducing carbon footprint through daily actions"
            ]
            cluster_summary = self._generate_cluster_summary(
                sample_themes,
                "Cluster shows unified concern about environmental impact"
            )
        
        # Get code text from existing codebook
        if self.codebook:
            # Sample some codes
            sample_codes = random.sample(self.codebook, min(10, len(self.codebook)))
            code_text = "\n".join([
                f"- {code.get('code', code)}: {code.get('definition', 'No definition')}"
                for code in sample_codes
            ])
        else:
            code_text = "No existing codes in codebook"
        
        # Populate prompt
        prompt = CANDIDATE_CODE_SELECTION_PROMPT.format(
            language=self.language,
            survey_question=self.var_lab,
            cluster_summary=cluster_summary,
            code_text=code_text
        )
        
        print(prompt)
        print("-"*80)
    
    def test_code_generation_prompt(self, cluster_id: Optional[int] = None):
        """Test Prompt 3: Code Generation"""
        print("\n" + "="*80)
        print("PROMPT 3: CODE GENERATION/RECOMMENDATION")
        print("="*80 + "\n")
        
        if not self.cluster_data:
            print("❌ Cannot test: No cluster data available")
            return
        
        # Try to get real data first
        real_data = self._get_real_cluster_data(cluster_id)
        
        if real_data and 'cluster_summary' in real_data and 'candidate_codes' in real_data:
            # Use real data from steps 1 & 2
            cluster_summary = real_data['cluster_summary']
            candidate_codes = self._format_candidate_codes(real_data['candidate_codes'])
            cluster_id = real_data['cluster_id']
            print(f"Using real data from cluster {cluster_id}")
        else:
            # Fallback to synthetic data
            print("Using synthetic data (no reasoning cache available)")
            sample_themes = ["Concerns about work-life balance in remote settings"]
            cluster_summary = self._generate_cluster_summary(sample_themes)
            
            # Sample candidate codes
            if self.codebook and len(self.codebook) >= 3:
                sample_codes = random.sample(self.codebook, 3)
                candidate_codes = "\n".join([
                    f"- {code.get('code', code)}: {code.get('definition', 'No definition')}"
                    for code in sample_codes
                ])
            else:
                candidate_codes = "No codes selected"
        
        # Populate prompt
        prompt = CODE_GENERATION_PROMPT.format(
            language=self.language,
            survey_question=self.var_lab,
            cluster_summary=cluster_summary,
            candidate_codes=candidate_codes
        )
        
        print(prompt)
        print("-"*80)
    
    def test_validation_prompt(self, cluster_id: Optional[int] = None):
        """Test Prompt 4: Validation"""
        print("\n" + "="*80)
        print("PROMPT 4: VALIDATION")
        print("="*80 + "\n")
        
        if not self.cluster_data:
            print("❌ Cannot test: No cluster data available")
            return
        
        # Try to get real data first
        real_data = self._get_real_cluster_data(cluster_id)
        
        if (real_data and 'cluster_summary' in real_data and 
            'candidate_codes' in real_data and 'code_recommendation' in real_data):
            # Use real data from steps 1-3
            cluster_summary = real_data['cluster_summary']
            candidate_codes = self._format_candidate_codes(real_data['candidate_codes'])
            step3_recommendation = self._format_code_recommendation(real_data['code_recommendation'])
            cluster_id = real_data['cluster_id']
            print(f"Using real data from cluster {cluster_id}")
        else:
            # Fallback to synthetic data
            print("Using synthetic data (no reasoning cache available)")
            sample_themes = ["Preference for flexible working hours"]
            cluster_summary = self._generate_cluster_summary(sample_themes)
            
            candidate_codes = "- WORK_FLEXIBILITY: Desire for flexible work arrangements\n- TIME_MANAGEMENT: Issues related to managing time effectively"
            
            step3_recommendation = """Number of themes: 1

Theme 1: Preference for flexible working hours
Decision: use_existing
Codes to use: WORK_FLEXIBILITY
Justification: The existing code 'WORK_FLEXIBILITY' accurately captures the theme about preferring flexible working hours."""
        
        # Populate prompt with correct parameter names
        prompt = VALIDATION_PROMPT.format(
            language=self.language,
            survey_question=self.var_lab,
            cluster_summary=cluster_summary,
            candidate_codes=candidate_codes,
            step3_recommendation=step3_recommendation
        )
        
        print(prompt)
        print("-"*80)
    
    def test_prompt_chain(self):
        """Test all 4 prompts as a chain"""
        print("\n" + "="*80)
        print("TESTING FULL 4-PROMPT CHAIN")
        print("="*80)
        
        if not self.cluster_data:
            print("❌ Cannot test: No cluster data available")
            return
        
        # Try to get a cluster with complete reasoning data
        real_data = self._get_real_cluster_data()
        
        if real_data:
            cluster_id = real_data['cluster_id']
            print(f"\n📊 Testing with real data from Cluster ID: {cluster_id}")
            print("   Data source: Reasoning cache")
        else:
            # Fallback to getting a cluster from raw data
            cluster = self._get_random_cluster()
            cluster_id = cluster['cluster_id']
            print(f"\n📊 Testing with Cluster ID: {cluster_id}")
            print(f"   Total responses in cluster: {cluster['total_responses']}")
            print(f"   Sampled responses: {len(cluster['responses'])}")
            print("   Data source: Raw cluster data + synthetic examples")
        
        # Test each prompt in sequence using the same cluster
        print("\n" + "-"*35 + " PROMPT 1 " + "-"*35)
        self.test_cluster_summary_prompt(cluster_id)
        
        input("\n➡️  Press Enter to continue to Prompt 2...")
        print("\n" + "-"*35 + " PROMPT 2 " + "-"*35)
        self.test_candidate_selection_prompt(cluster_id)
        
        input("\n➡️  Press Enter to continue to Prompt 3...")
        print("\n" + "-"*35 + " PROMPT 3 " + "-"*35)
        self.test_code_generation_prompt(cluster_id)
        
        input("\n➡️  Press Enter to continue to Prompt 4...")
        print("\n" + "-"*35 + " PROMPT 4 " + "-"*35)
        self.test_validation_prompt(cluster_id)
        
        print("\n✅ Chain testing complete!")


def main(var_lab: str = None):
    """Main function for standalone execution"""
    print("🔬 CodeGenerator Prompt Tester")
    print("   Testing 4-chain prompts for Step 7\n")
    
    tester = CodeGeneratorPromptTester(var_lab=var_lab)
    
    if not tester.cluster_data:
        print("\n❌ Exiting: No data available for testing")
        return
    
    while True:
        print("\n" + "="*50)
        print("Select test option:")
        print("="*50)
        print("1. Test Prompt 1 (Cluster Summary)")
        print("2. Test Prompt 2 (Candidate Selection)")
        print("3. Test Prompt 3 (Code Generation)")
        print("4. Test Prompt 4 (Validation)")
        print("5. Test Full Chain (All 4 prompts)")
        print("0. Exit")
        print("-"*50)
        
        choice = input("Enter your choice (0-5): ").strip()
        
        if choice == '0':
            print("\n👋 Goodbye!")
            break
        elif choice == '1':
            tester.test_cluster_summary_prompt()
        elif choice == '2':
            tester.test_candidate_selection_prompt()
        elif choice == '3':
            tester.test_code_generation_prompt()
        elif choice == '4':
            tester.test_validation_prompt()
        elif choice == '5':
            tester.test_prompt_chain()
        else:
            print("❌ Invalid choice. Please try again.")
        
        if choice in ['1', '2', '3', '4']:
            input("\n➡️  Press Enter to return to menu...")


if __name__ == "__main__":
    main()