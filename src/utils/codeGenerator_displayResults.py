import random
from typing import Optional, List


def display_cluster_analysis(codebook_reasoning, cluster_id: Optional[int] = None, show_detailed_reasoning: bool = False, debug_mode: bool = False) -> None:
    
    print("\n" + "="*80 + "\nCLUSTER REASONING ANALYSIS\n" + "="*80)
    
    # Access data directly from codebook_reasoning object
    step1_summaries = getattr(codebook_reasoning, 'step1_summaries', {})
    step2_analysis = getattr(codebook_reasoning, 'step2_analysis', {})  
    step3_recommendations = getattr(codebook_reasoning, 'step3_recommendations', {})
    step4_validations = getattr(codebook_reasoning, 'step4_validations', {})
    #cluster_assignments = getattr(codebook_reasoning, 'cluster_assignments', {})
    
    #  inputs 
    step1_inputs = getattr(codebook_reasoning, 'step1_inputs', {})
    #step2_inputs = getattr(codebook_reasoning, 'step2_inputs', {})
    #step3_inputs = getattr(codebook_reasoning, 'step3_inputs', {})
    #step4_inputs = getattr(codebook_reasoning, 'step4_inputs', {})
    
    # Select cluster to display
    if cluster_id is None:
        if step3_recommendations:
            available_ids = list(step3_recommendations.keys())
            cluster_id = random.choice(available_ids)
        else:
            print("No clusters with recommendations found.")
            return
    
    # Verify cluster exists
    if cluster_id not in step3_recommendations:
        print(f"Cluster {cluster_id} not found in results.")
        return
    
    print(f"\n📋 Cluster Analysis (ID: {cluster_id})")
    print("-" * 40)
    
    # 1. CLUSTER THEME(S)
    if step1_summaries and cluster_id in step1_summaries:
        step1_data = step1_summaries[cluster_id]
        themes = step1_data.get("themes", [])
        
        print("\n📝 CLUSTER THEME(S):")
        if len(themes) > 1: 
            for i, theme in enumerate(themes, 1):
                if isinstance(theme, dict) and 'theme_name' in theme:
                    print(f"   {i}. {theme['theme_name']}")
                else:
                    print(f"   {i}. {str(theme)[:80]}...")
        elif len(themes) == 1:
            theme = themes[0]
            if isinstance(theme, dict) and 'theme_name' in theme:
                print(f"   {theme['theme_name']}")
            else:
                print(f"   {str(theme)[:80]}...")
        else:
            print("   [No themes identified]")
    else:
        print("\n📝 1. CLUSTER THEME(S): [Not available]")
    
    # 2. CLUSTER IDEAS (from step1_inputs, like promptTester does)
    if step1_inputs and cluster_id in step1_inputs:
        cluster_text = step1_inputs[cluster_id].get('cluster_ideas', '')  # Changed from cluster_text to cluster_ideas
        if cluster_text:
            # Parse cluster_text the same way as promptTester
            ideas = [idea.strip() for idea in cluster_text.split('\n') if idea.strip()]
            # Remove the "- " prefix if present
            clean_ideas = [idea[2:].strip() if idea.startswith('- ') else idea for idea in ideas]
            print(f"\n💡 CLUSTER IDEAS ({len(clean_ideas)} responses):")
            for i, idea in enumerate(clean_ideas[:5], 1):  # Show first 5
                print(f"   {i}. {idea}")
            if len(clean_ideas) > 5:
                print(f"   ... and {len(clean_ideas) - 5} more ideas")
        else:
            print("\n💡 CLUSTER IDEAS: [No cluster_text found]")
    else:
        print("\n💡 CLUSTER IDEAS: [Not available]")
    
    # 3. CANDIDATE CODES (from step2_analysis)
    if step2_analysis:
        # Try to find candidate codes for this cluster
        candidate_codes = None
        
        if cluster_id in step2_analysis:
            candidate_codes = step2_analysis[cluster_id]
        elif 0 in step2_analysis:
            # Fallback: use data from cluster 0 (seems to contain all candidate codes)
            candidate_codes = step2_analysis[0]
        
        if candidate_codes:
            print(f"\n🔍 CANDIDATE CODES ({len(candidate_codes)} found):")
            for i, code_data in enumerate(candidate_codes, 1):
                if isinstance(code_data, dict):
                    code_name = code_data.get('code', 'Unknown')
                    definition = code_data.get('definition', 'No definition')
                    print(f"   {i}. {code_name}")
                    if show_detailed_reasoning:
                        print(f"      Definition: {definition}")
                else:
                    print(f"   {i}. {str(code_data)}")
                
                if i >= 10 and len(candidate_codes) > 10:
                    print(f"   ... and {len(candidate_codes) - 10} more candidate codes")
                    break
        else:
            # Debug: Show available cluster IDs
            available_ids = list(step2_analysis.keys())
            print(f"\n🔍 CANDIDATE CODES: [No candidate codes found. Available IDs: {available_ids[:5]}{'...' if len(available_ids) > 5 else ''}]")
    else:
        print("\n🔍 CANDIDATE CODES: [step2_analysis is empty]")
    
    # 4. RECOMMENDED CHANGES TO CODEBOOK
    if step3_recommendations and cluster_id in step3_recommendations:
        gen_result = step3_recommendations[cluster_id]
        print("\n📊 RECOMMENDED CHANGES TO CODEBOOK:\n")
        
        if 'coding_decisions' in gen_result:
            for i, decision in enumerate(gen_result['coding_decisions'], 1):
                decision_type = decision.get('decision', 'Unknown')
                theme_number = decision.get('theme_number', i)
                final_code_label = decision.get('final_code_label', 'Unknown')
                final_code_description = decision.get('final_code_description', 'Unknown')
                source_code = decision.get('source_code', None)
                justification = decision.get('justification', 'No justification provided')
                
                print(f"\nDecision {theme_number}: {decision_type.upper()}")
                print(f"Final code: {final_code_label}")
                
                if show_detailed_reasoning:
                    print(f"Definition: {final_code_description}")
                    if source_code:
                        print(f"Source code: {source_code}")
                
                print(f"Reasoning: {justification}")
        else:
            print("   [No coding decisions found]")
    else:
        print("\n📊 RECOMMENDED CHANGES TO CODEBOOK: [Not available]")
    
    # 5. VALIDATION WITH REASONING
    if step4_validations and cluster_id in step4_validations:
        val_result = step4_validations[cluster_id]
        print("\n✅ VALIDATION:")
        
        if 'code_validations' in val_result:
            for i, validation in enumerate(val_result['code_validations'], 1):
                #theme_desc = validation.get('theme_description', 'Unknown theme')
                #original_rec = validation.get('original_recommendation', 'Not available')
                decision = validation.get('decision', 'Unknown')
                rationale = validation.get('decision_rationale', 'Not provided')
                
                # print(f"\n   Theme {i}: {theme_desc}")
                # print(f"   Original recommendation: {original_rec}")
                print(f"\nValidation decision ({i}): {decision}")
                print(f"Reasoning: {rationale}")
                
                if show_detailed_reasoning:
                    # Show original recommendation
                    orig_rec = validation.get('original_recommendation', {})
                    if orig_rec:
                        print(f"   Original recommendation: {orig_rec.get('code', 'Unknown')}")
                        print(f"   Original definition: {orig_rec.get('definition', 'Unknown')}")
                
                # Show final validated code
                validated_code = validation.get('validated_code')
                if validated_code:
                    if isinstance(validated_code, list):
                        print("   Final validated codes (SPLIT):")
                        for j, code in enumerate(validated_code, 1):
                            print(f"     {j}. {code.get('code', 'Unknown')}")
                            if show_detailed_reasoning:
                                print(f"        Definition: {code.get('definition', 'No definition')}")
                    # else:
                    #     # print(f"\nFinal validated code: {validated_code.get('code', 'Unknown')}")
                    #     # if show_detailed_reasoning:
                    #     #    print(f"Definition: {validated_code.get('definition', 'No definition')}")
        else:
            print("   [No validation results found]")
    else:
        print("\n✅ VALIDATION: [Not available]")
    
    # Final Validated Codes (from step4_validations)
    if step4_validations and cluster_id in step4_validations:
        val_result = step4_validations[cluster_id]
        if 'code_validations' in val_result:
            all_codes = []
            for validation in val_result['code_validations']:
                validated_code = validation.get('validated_code')
                if validated_code:
                    if isinstance(validated_code, list):
                        # Multiple codes (SPLIT case)
                        for code in validated_code:
                            if isinstance(code, dict):
                                all_codes.append(code)
                    elif isinstance(validated_code, dict):
                        # Single code
                        all_codes.append(validated_code)
            
            if all_codes:
                print(f"\n📌 Final Validated Codes ({len(all_codes)}):")
                for i, code_info in enumerate(all_codes, 1):
                    code_name = code_info.get('code', 'Unknown')
                    definition = code_info.get('definition', 'No definition')
                    print(f"   {i}. {code_name}")
                    if show_detailed_reasoning:
                        print(f"      Definition: {definition}")
            else:
                print("\n📌 Final Validated Codes: [No validated codes found]")
        else:
            print("\n📌 Final Validated Codes: [No code_validations found]")
    else:
        print("\n📌 Final Validated Codes: [Not available in step4_validations]")
    
    print("\n" + "="*80)


def display_summary_statistics(codebook_reasoning) -> None:
   
    # Access data directly from codebook_reasoning object
    stats = codebook_reasoning.stats
    step3_recommendations = codebook_reasoning.step3_recommendations
    step4_validations = codebook_reasoning.step4_validations
    cluster_assignments = codebook_reasoning.cluster_assignments
    
    print("\n" + "="*80 + "\nPIPELINE SUMMARY STATISTICS\n" + "="*80)
    
    # Step 3 Decision breakdown
    if step3_recommendations:
        all_decisions = []
        for gen_result in step3_recommendations.values():
            if 'coding_decisions' in gen_result:
                for decision in gen_result['coding_decisions']:
                    all_decisions.append(decision.get('decision', 'unknown'))
        
        print("\nStep 3 Decisions:")
        print(f"• Create new: {all_decisions.count('create_new')}")
        print(f"• Modify existing: {all_decisions.count('modify_existing')}")  
        print(f"• Use existing: {all_decisions.count('use_existing')}")
        print(f"• Total decisions: {len(all_decisions)}")
        print(f"• Total clusters: {len(step3_recommendations)}")
    
    # Step 4 Validation breakdown
    if step4_validations:
        all_val_decisions = []
        for val_result in step4_validations.values():
            if 'code_validations' in val_result:
                for validation in val_result['code_validations']:
                    all_val_decisions.append(validation.get('decision', 'unknown'))
        
        print("\nStep 4 Validation:")
        print(f"• Approved: {all_val_decisions.count('APPROVE')}")
        print(f"• Revised: {all_val_decisions.count('REVISE')}")
        print(f"• Rejected: {all_val_decisions.count('REJECT')}")
        print(f"• Split: {all_val_decisions.count('SPLIT')}")
    
    # Cluster status breakdown
    if cluster_assignments:
        statuses = [assignment['status'] for assignment in cluster_assignments.values()]
        print("\nCluster Processing Status:")
        print(f"• Completed: {statuses.count('completed')}")
        print(f"• No themes found: {statuses.count('no_themes_found')}")
        print(f"• Failed: {len(statuses) - statuses.count('completed') - statuses.count('no_themes_found')}")
    
    # Performance stats
    if stats:
        print("\nPerformance:")
        if 'processing_time' in stats:
            print(f"• Total processing time: {stats['processing_time']:.2f}s")
        print(f"• New codes added: {stats.get('new_codes_added', 0)}")
        print(f"• Codes modified: {stats.get('codes_modified', 0)}")
        print(f"• No new codes needed: {stats.get('no_new_codes_needed', 0)}")
        
        # Additional stats
        if 'initial_codes' in stats and 'final_codes' in stats:
            print(f"• Codebook growth: {stats['initial_codes']} → {stats['final_codes']} codes")
        if 'avg_time_per_cluster' in stats:
            print(f"• Average time per cluster: {stats['avg_time_per_cluster']:.2f}s")


def display_multiple_clusters(codebook_reasoning, cluster_ids: List[int] = None, 
                            max_clusters: int = 5, debug_mode: bool = False) -> None:
  
    step3_recommendations = codebook_reasoning.step3_recommendations
    
    if cluster_ids is None:
        # Random selection
        available_ids = list(step3_recommendations.keys())
        num_to_show = min(max_clusters, len(available_ids))
        cluster_ids = random.sample(available_ids, num_to_show)
    
    print(f"\nDisplaying {len(cluster_ids)} clusters:")
    
    for idx, cluster_id in enumerate(cluster_ids, 1):
        print(f"\n{'='*20} CLUSTER {idx}/{len(cluster_ids)} {'='*20}")
        display_cluster_analysis(codebook_reasoning, cluster_id, show_detailed_reasoning=False, debug_mode=debug_mode)


def find_clusters_by_decision(codebook_reasoning, decision_type: str) -> List[int]:
   
    matching_clusters = []
    step3_recommendations = codebook_reasoning.step3_recommendations
    
    for cluster_id, gen_result in step3_recommendations.items():
        # Check if any decision in this cluster matches the target type
        if 'coding_decisions' in gen_result:
            for decision in gen_result['coding_decisions']:
                if decision.get('decision') == decision_type:
                    matching_clusters.append(cluster_id)
                    break  # Found one match, move to next cluster
    
    return matching_clusters