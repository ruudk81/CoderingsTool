import random
from typing import Optional, List, Union


def display_cluster_analysis(codebook_reasoning, cluster_id: Optional[Union[int, str]] = None, show_detailed_reasoning: bool = False, debug_mode: bool = False) -> None:
    
    step3_recommendations = getattr(codebook_reasoning, 'step3_recommendations', {})
   
    if cluster_id is None:
        if step3_recommendations:
            available_ids = list(step3_recommendations.keys())
            cluster_id = random.choice(available_ids)
        else:
            print("No clusters with recommendations found.")
            return
    else:
        cluster_id = cluster_id
    
    print("\n" + "="*80 + f"\nCLUSTER REASONING ANALYSIS (cluster: {cluster_id})\n" + "="*80)
    
    
    _display_single_cluster(codebook_reasoning, cluster_id, show_detailed_reasoning, debug_mode)
        
def _display_single_cluster(codebook_reasoning, cluster_id: Union[int, str], show_detailed_reasoning: bool, debug_mode: bool) -> None:
    """Display analysis for a single cluster (helper function)"""
    step1_inputs = getattr(codebook_reasoning, 'step1_inputs', {})
    step1_summaries = getattr(codebook_reasoning, 'step1_summaries', {})
    step2_analysis = getattr(codebook_reasoning, 'step2_analysis', {})  
    step3_recommendations = getattr(codebook_reasoning, 'step3_recommendations', {})
    step4_validations = getattr(codebook_reasoning, 'step4_validations', {})
  
    # 1. CLUSTER THEME(S)
    #cluster_id = '14-2'
    if step1_summaries and cluster_id in step1_summaries:
        step1_data = step1_summaries[cluster_id]
        analysis = step1_data.get("analysis", [])
        theme_id = step1_data.get("theme_id", [])
        theme_label = step1_data.get("theme_label", [])
        theme_description = step1_data.get("theme_description", [])
        
        print("\n🧠 CLUSTER ANALYSIS:")
        if analysis:
            print(f"{analysis}")
        else:
            print("[No analysis]")

        print("\n🔍 CLUSTER THEME:")
        if theme_id > 0: 
            print(f"Theme: {theme_id}")
            print(f"Label: {theme_label}")
            print(f"Description: {theme_description}")
        else:
            print("[No themes identified]")
    else:
        print("\n🔍 1. CLUSTER THEME(S): [Not available]")
    
    # 2. CLUSTER IDEAS 
    if step1_inputs and cluster_id in step1_inputs:
        cluster_text = step1_inputs[cluster_id].get('cluster_text', '')  
        if cluster_text:
            ideas = [idea.strip() for idea in cluster_text.split('\n') if idea.strip()]
            clean_ideas = [idea[2:].strip() if idea.startswith('- ') else idea for idea in ideas]
            print(f"\n💡 CLUSTER IDEAS ({len(clean_ideas)} responses):")
            for i, idea in enumerate(clean_ideas, 1):  
                print(f"   {i}. {idea}")
            # for i, idea in enumerate(clean_ideas[:10], 1):  # Show first 10
            #     print(f"   {i}. {idea}")
            # if len(clean_ideas) > 10:
            #     print(f"   ... and {len(clean_ideas) - 10} more ideas")
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
        if candidate_codes:
            print(f"\n🔍 CANDIDATE CODES ({len(candidate_codes)} found):")
            for i, code_data in enumerate(candidate_codes, 1):
                code_name = code_data.get('code', 'Unknown')
                definition = code_data.get('definition', 'No definition')
                print(f"{i}. {code_name}")
                if show_detailed_reasoning:
                        print(f"-  {definition}")
        else:
            # # Debug: Show available cluster IDs
            # available_ids = list(step2_analysis.keys())
            print("\n🔍 CANDIDATE CODES: [No candidate found]")
    else:
        print("\n🔍 CANDIDATE CODES: [Not available]")
    
    #show_detailed_reasoning = False
    
    # 4. RECOMMENDED CHANGES TO CODEBOOK
    if step3_recommendations and cluster_id in step3_recommendations:
        gen_result = step3_recommendations[cluster_id]
        print("\n🎯 RECOMMENDED CHANGES TO CODEBOOK:")
        
        if 'coding_decisions' in gen_result:
            for i, decision in enumerate(gen_result['coding_decisions'], 1):
                decision_type = decision.get('decision', 'Unknown')
                theme_number = decision.get('theme_number', i)
                final_code_label = decision.get('final_code_label', 'Unknown')
                final_code_description = decision.get('final_code_definition', 'No description available')  # Fixed field name
                source_code = decision.get('source_code', None)
                justification = decision.get('justification', 'No justification provided')
                
                if theme_number <= 1:
                    if decision_type == "create": 
                        print(f"{decision_type.upper()}: {final_code_label}") 
                    if decision_type == "modify": 
                        source_display = source_code if source_code else "[Unknown Source]"
                        print(f"{decision_type.upper()}: {source_display} -> {final_code_label}")    
                    if decision_type == "use": 
                        print(f"{decision_type.upper()}: {final_code_label}")                          
                else:
                    print(f"\nDecision: {theme_number}")
                    if decision_type == "create": 
                        print(f"{decision_type.upper()}: {final_code_label}") 
                    if decision_type == "modify": 
                        source_display = source_code if source_code else "[Unknown Source]"
                        print(f"{decision_type.upper()}: {source_display} -> {final_code_label}")    
                    if decision_type == "use": 
                        print(f"{decision_type.upper()}: {final_code_label}") 
                
                if show_detailed_reasoning:
                    print(f"Definition: {final_code_description}")
                    if source_code:
                        print(f"Source code: {source_code}")
                
                print(f"\nReasoning: {justification}")
        else:
            print("[No recommendations found]")
    else:
        print("\n🎯 RECOMMENDED CHANGES TO CODEBOOK: [Not available]")
    
    
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
                
                if i > 1:
                    print(f"\n\ndecision ({i}): {decision}") 
                    print(f"\nReasoning: {rationale}")
                else: 
                    print(f"decision: {decision}") 
                    print(f"\nReasoning: {rationale}")
                
                if show_detailed_reasoning:
                    # Show original recommendation
                    orig_rec = validation.get('original_recommendation', {})
                    if orig_rec:
                        print(f"\nOriginal recommendation: {orig_rec.get('code', 'Unknown')}")
                        print(f"Original definition: {orig_rec.get('definition', 'Unknown')}")
                
                # Show final validated code
                validated_code = validation.get('validated_code')
                if validated_code:
                    print("\n")
                    for key, value in validated_code.items():
                        print(f"Validated {key}: {value}")
        else:
            print("[No validation results found]")
    else:
        print("\n✅ VALIDATION: [Not available]")
    print("\n" + "="*80)


def display_summary_statistics(codebook_reasoning) -> None:
   
    # Access data directly from codebook_reasoning object
    stats = codebook_reasoning.stats
    step3_recommendations = codebook_reasoning.step3_recommendations
    step4_validations = codebook_reasoning.step4_validations
    
    print("\n" + "="*80 + "\nPIPELINE SUMMARY STATISTICS\n" + "="*80)
    
    # Processing statues 
    print("\nCluster Processing Status:")
    if stats:
        for key, value in stats.items():
            if key == "clusters_found":
                print(f"• Total clusters found: {value}")
            if key == "processing_success_rate":
                print(f"• Completion: {value}%")     
     
    
    # Step 3 Decision breakdown
    if step3_recommendations:
        all_decisions = []
        for gen_result in step3_recommendations.values():
            if 'coding_decisions' in gen_result:
                for decision in gen_result['coding_decisions']:
                    all_decisions.append(decision.get('decision', 'unknown'))
        
        print("\nStep 3 Decisions:")
        print(f"• Create new: {all_decisions.count('create')}")
        print(f"• Modify existing: {all_decisions.count('modify')}")  
        print(f"• Use existing: {all_decisions.count('use')}")
        print(f"• Total decisions: {len(all_decisions)}")
    
    # Step 4 Validation breakdown
    if step4_validations:
        all_val_decisions = []
        for val_result in step4_validations.values():
            if 'code_validations' in val_result:
                for validation in val_result['code_validations']:
                    all_val_decisions.append(validation.get('decision', 'unknown'))
        
        print("\nStep 4 Validation:")
        print(f"• Approved: {all_val_decisions.count('APPROVE')}")
        print(f"• Rejected: {all_val_decisions.count('REJECT')}")
        print(f"• Total validations: {len(all_val_decisions)}")


def display_multiple_clusters(codebook_reasoning, cluster_ids: List[Union[int, str]] = None, max_clusters: int = 5, debug_mode: bool = False) -> None:
  
    #debug_mode = True
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


def find_clusters_by_decision(codebook_reasoning, decision_type: str) -> List[Union[int, str]]:
    """step 3 only: create; modify and use"""
   
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