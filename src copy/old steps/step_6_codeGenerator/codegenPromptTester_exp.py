#!/usr/bin/env python3
"""
Experimental Prompt Tester for CodeGenerator 4-Chain Prompts.

Displays the exact prompts used by the experimental codeGenerator pipeline.
Requires pre-loaded codebook_reasoning from the caller (no stale pickle loading).
Uses facet-based prompts from prompts_exp.py.
"""

import random
from typing import Optional, Union

from config import DEFAULT_LANGUAGE
from development.step_6_codeGenerator.prompts_exp import (
    CLUSTER_SUMMARY_PROMPT, CODING_DECISION_PROMPT,
    CODING_MODIFICATION_PROMPT, CODE_CREATION_PROMPT, VALIDATION_PROMPT
)
from development.step_6_codeGenerator.codeGenerator_exp import CodeGeneratorReasoningResults


class SimplePromptTester:
    """Prompt tester for experimental CodeGenerator 4-chain prompts.

    Args:
        codebook_reasoning: Pre-loaded CodeGeneratorReasoningResults from CacheManager.
        cluster_id: Specific cluster to inspect, or None for random selection.
        var_lab: Survey question label.
        language: Response language (default from config).
    """

    def __init__(
        self,
        codebook_reasoning: CodeGeneratorReasoningResults,
        cluster_id: Optional[Union[int, str]],
        var_lab: str,
        language: str = DEFAULT_LANGUAGE,
    ):
        self.var_lab = var_lab
        self.language = language
        self.codebook_reasoning = codebook_reasoning

        if cluster_id is None:
            self.cluster_id = self._sample_cluster_id()
            print(f"TARGET: Using cluster ID: {self.cluster_id}")
        else:
            self.cluster_id = str(cluster_id) if isinstance(cluster_id, int) else cluster_id

    def _sample_cluster_id(self):
        """Sample a random cluster ID that has data in ALL steps."""
        # Prefer clusters with complete 4-step data
        complete_clusters = []
        for cid in self.codebook_reasoning.step1_summaries:
            if (cid in self.codebook_reasoning.step2_analysis and
                cid in self.codebook_reasoning.step3_recommendations and
                cid in getattr(self.codebook_reasoning, 'step4_validations', {})):
                complete_clusters.append(cid)

        if complete_clusters:
            return random.choice(complete_clusters)

        # Fallback to any cluster with step1 data
        step1_clusters = list(self.codebook_reasoning.step1_summaries.keys())
        if step1_clusters:
            print("WARNING: No clusters with complete 4-step data, using cluster with step1 data")
            return random.choice(step1_clusters)

        # Last resort: step3_recommendations keys
        return random.choice(list(self.codebook_reasoning.step3_recommendations.keys()))

    def test_prompt_1(self):
        """Test Prompt 1: Cluster Summary"""
        print("\n" + "="*80)
        print("PROMPT 1: CLUSTER SUMMARY")
        print("="*80)

        step1_inputs = getattr(self.codebook_reasoning, 'step1_inputs', {})

        if self.cluster_id in step1_inputs:
            step1_input = step1_inputs[self.cluster_id].copy()
            parent_cluster = str(self.cluster_id).split('-')[0]
            step1_input['cluster_id'] = parent_cluster
            print("\n[USING ACTUAL PIPELINE INPUTS]\n")
        else:
            print(f"\n[ERROR: No Step 1 inputs found for cluster {self.cluster_id}]")
            print("Cannot display Prompt 1 without actual pipeline inputs")
            return

        print("Input Summary:")
        print(f"  Language: {step1_input.get('language', 'Unknown')}")
        print(f"  Survey Question: {step1_input.get('survey_question', 'Unknown')}")
        cluster_text = step1_input.get('cluster_text', '')
        if cluster_text:
            ideas = cluster_text.split('\n')
            print(f"  Cluster Ideas: {len(ideas)} ideas, {len(cluster_text)} characters")

        # Detect which route was used based on captured params
        data_unit = step1_input.get('data_unit', 'cluster')
        print(f"  Route: {data_unit}")

        prompt = CLUSTER_SUMMARY_PROMPT.format(**step1_input)
        print(f"\n{'-'*60}\nFORMATTED PROMPT:\n{'-'*60}\n")
        print(prompt)

    def test_prompt_2(self):
        """Test Prompt 2: Candidate Code Selection"""
        print("\n" + "="*80)
        print("PROMPT 2: CANDIDATE CODE SELECTION")
        print("="*80)

        step2_inputs = getattr(self.codebook_reasoning, 'step2_inputs', {})
        if self.cluster_id in step2_inputs:
            step2_input = step2_inputs[self.cluster_id]
            print("\n[USING ACTUAL PIPELINE INPUTS]\n")
        else:
            print(f"\n[ERROR: No Step 2 inputs found for cluster {self.cluster_id}]")
            print("Cannot display Prompt 2 without actual pipeline inputs")
            return

        print("Input Summary:")
        print(f"  Language: {step2_input.get('language', 'Unknown')}")
        print(f"  Survey Question: {step2_input.get('survey_question', 'Unknown')}")
        print(f"  Cluster Summary: {step2_input.get('cluster_summary', 'Unknown')}")

        code_text = step2_input.get('code_text', '')
        if code_text:
            code_lines = [line.strip() for line in code_text.split('\n') if line.strip().startswith('Code:')]
            print(f"  Nearest Codes: {len(code_lines)} codes, {len(code_text)} characters")

        prompt = CODING_DECISION_PROMPT.format(**step2_input)
        print(f"\n{'-'*60}\nFORMATTED PROMPT:\n{'-'*60}\n")
        print(prompt)

    def test_prompt_3(self):
        """Test Prompt 3: Code Generation"""
        print("\n" + "="*80)
        print("PROMPT 3: CODE GENERATION")
        print("="*80)

        step3_inputs = getattr(self.codebook_reasoning, 'step3_inputs', {})
        if self.cluster_id in step3_inputs:
            step3_input = step3_inputs[self.cluster_id]
            print("\n[USING ACTUAL PIPELINE INPUTS]\n")
        else:
            print(f"\n[ERROR: No Step 3 inputs found for cluster {self.cluster_id}]")
            print("Cannot display Prompt 3 without actual pipeline inputs")
            return

        print("Input Summary:")
        print(f"  Language: {step3_input.get('language', 'Unknown')}")
        print(f"  Survey Question: {step3_input.get('survey_question', 'Unknown')}")
        print(f"  Cluster Summary: {step3_input.get('cluster_summary', 'Unknown')}")

        coding_decision = step3_input.get('coding_decision', '')
        if coding_decision:
            if coding_decision.upper().startswith("MODIFY"):
                prompt_template = CODING_MODIFICATION_PROMPT
                template_name = "CODING_MODIFICATION_PROMPT"
            else:
                prompt_template = CODE_CREATION_PROMPT
                template_name = "CODE_CREATION_PROMPT"
            print(f"  Coding Decision: {coding_decision}")
            print(f"  Template: {template_name}")

        prompt = prompt_template.format(**step3_input)
        print(f"\n{'-'*60}\nFORMATTED PROMPT:\n{'-'*60}\n")
        print(prompt)

    def test_prompt_4(self):
        """Test Prompt 4: Validation"""
        print("\n" + "="*80)
        print("PROMPT 4: VALIDATION")
        print("="*80)

        step4_inputs = getattr(self.codebook_reasoning, 'step4_inputs', {})
        if self.cluster_id in step4_inputs:
            step4_input = step4_inputs[self.cluster_id]
            print("\n[USING ACTUAL PIPELINE INPUTS]\n")
        else:
            print(f"\n[ERROR: No Step 4 inputs found for cluster {self.cluster_id}]")
            print("Cannot display Prompt 4 without actual pipeline inputs")
            return

        print("Input Summary:")
        print(f"  Language: {step4_input.get('language', 'Unknown')}")
        print(f"  Survey Question: {step4_input.get('survey_question', 'Unknown')}")
        print(f"  Cluster Summary: {step4_input.get('cluster_summary', 'Unknown')}")

        code_text = step4_input.get('code_text', '')
        if code_text:
            code_lines = [line.strip() for line in code_text.split('\n') if line.strip().startswith('Code:')]
            print(f"  Existing Codes: {len(code_lines)} codes, {len(code_text)} characters")

        step3_recommendation = step4_input.get('step3_recommendation', '')
        if step3_recommendation:
            print(f"  Step 3 Recommendation: {len(step3_recommendation)} characters")

        prompt = VALIDATION_PROMPT.format(**step4_input)
        print(f"\n{'-'*60}\nFORMATTED PROMPT:\n{'-'*60}\n")
        print(prompt)

    def test_all_prompts(self):
        """Test all 4 prompts in sequence"""
        self.test_prompt_1()
        self.test_prompt_2()
        self.test_prompt_3()
        self.test_prompt_4()
        print("\nAll prompts tested!")
