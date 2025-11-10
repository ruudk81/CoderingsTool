import random
from typing import Optional, List, Union, Dict, Any


def get_cluster_analysis(codebook_reasoning, cluster_id: Optional[Union[int, str]] = None) -> Dict[str, Any]:
    """Extract cluster analysis data and return as structured dictionary.

    Args:
        codebook_reasoning: CodeGeneratorReasoningResults object containing analysis data
        cluster_id: Specific cluster ID to analyze (e.g., '43-1'). If None, selects random cluster.

    Returns:
        Dictionary containing structured cluster analysis data with keys:
        - cluster_id: Full cluster identifier
        - main_cluster_id: Main cluster number
        - ideas: Dict with 'count' and 'ideas_list'
        - analysis: Dict with cluster analysis and theme info
        - candidate_codes: List of candidate code dicts
        - recommendations: Dict with decision type, justification, and final code
        - validation: Dict with verdict, validated decision, and rationale
    """
    step1_inputs = getattr(codebook_reasoning, 'step1_inputs', {})
    step1_summaries = getattr(codebook_reasoning, 'step1_summaries', {})
    step2_analysis = getattr(codebook_reasoning, 'step2_analysis', {})
    step3_recommendations = getattr(codebook_reasoning, 'step3_recommendations', {})
    step4_validations = getattr(codebook_reasoning, 'step4_validations', {})

    # Select cluster_id if not provided
    if cluster_id is None:
        if step3_recommendations:
            available_ids = list(step3_recommendations.keys())
            cluster_id = random.choice(available_ids)
        else:
            return {'error': 'No clusters with recommendations found'}

    cluster_id = str(cluster_id)
    main_cluster_id = cluster_id.split('-')[0]

    # Initialize result structure
    result = {
        'cluster_id': cluster_id,
        'main_cluster_id': main_cluster_id,
        'ideas': {'count': 0, 'ideas_list': []},
        'analysis': {'text': None},
        'theme': {'theme_id': None, 'theme_label': None, 'theme_description': None},
        'candidate_codes': [],
        'recommendations': {'decision_type': None, 'source_code': None, 'justification': None, 'final_code_label': None},
        'validation': {'verdict': None, 'validated_decision': None, 'validated_code': None, 'rationale': None}
    }

    # 1. Extract cluster ideas
    if step1_inputs and cluster_id in step1_inputs:
        cluster_text = step1_inputs[cluster_id].get('cluster_text', '')
        if cluster_text:
            ideas = [idea.strip() for idea in cluster_text.split('\n') if idea.strip()]
            clean_ideas = [idea[2:].strip() if idea.startswith('- ') else idea for idea in ideas]
            result['ideas']['count'] = len(clean_ideas)
            result['ideas']['ideas_list'] = clean_ideas

    # 2. Extract cluster analysis and theme (separated)
    if step1_summaries and cluster_id in step1_summaries:
        step1_data = step1_summaries[cluster_id]
        result['analysis']['text'] = step1_data.get("analysis", None)
        result['theme']['theme_id'] = step1_data.get("theme_id", None)
        result['theme']['theme_label'] = step1_data.get("theme_label", None)
        result['theme']['theme_description'] = step1_data.get("theme_description", None)

        # Extract new fields from full theme object
        if 'themes' in step1_data and step1_data['themes']:
            first_theme = step1_data['themes'][0]
            # Handle both Pydantic objects and dicts
            if hasattr(first_theme, 'abstraction_level'):
                result['theme']['abstraction_level'] = first_theme.abstraction_level
                if hasattr(first_theme, 'assignment_examples'):
                    examples = first_theme.assignment_examples
                    result['theme']['inclusion_examples'] = examples.inclusion if hasattr(examples, 'inclusion') else []
                    result['theme']['exclusion_examples'] = examples.exclusion if hasattr(examples, 'exclusion') else []
                    if hasattr(examples, 'near_neighbor'):
                        neighbor = examples.near_neighbor
                        result['theme']['near_neighbor_label'] = neighbor.label if hasattr(neighbor, 'label') else None
                        result['theme']['tell_apart_rule'] = neighbor.tell_apart_rule if hasattr(neighbor, 'tell_apart_rule') else None
            elif isinstance(first_theme, dict):
                result['theme']['abstraction_level'] = first_theme.get('abstraction_level', None)
                if 'assignment_examples' in first_theme:
                    examples = first_theme['assignment_examples']
                    result['theme']['inclusion_examples'] = examples.get('inclusion', [])
                    result['theme']['exclusion_examples'] = examples.get('exclusion', [])
                    if 'near_neighbor' in examples:
                        neighbor = examples['near_neighbor']
                        result['theme']['near_neighbor_label'] = neighbor.get('label', None)
                        result['theme']['tell_apart_rule'] = neighbor.get('tell_apart_rule', None)

    # 3. Extract candidate codes
    if step2_analysis and cluster_id in step2_analysis:
        step2_data = step2_analysis[cluster_id]
        decision = step2_data.get('coding_decision', {})
        if 'matched_candidates' in decision:
            result['candidate_codes'] = decision['matched_candidates']

    # 4. Extract recommendations
    decision_data = None
    if step2_analysis and cluster_id in step2_analysis:
        step2_data = step2_analysis[cluster_id]
        if isinstance(step2_data, dict) and 'coding_decision' in step2_data:
            decision_data = step2_data['coding_decision']

    generated_code_data = None
    if step3_recommendations and cluster_id in step3_recommendations:
        generated_code_data = step3_recommendations[cluster_id]

    if decision_data:
        result['recommendations']['decision_type'] = decision_data.get('decision', None)
        result['recommendations']['source_code'] = decision_data.get('source_code', None)
        result['recommendations']['justification'] = decision_data.get('justification', None)

        # Extract modify_parameters if present
        if 'modify_parameters' in decision_data:
            modify_params = decision_data['modify_parameters']
            result['recommendations']['modify_instruction'] = modify_params.get('modify_instruction', None)
            result['recommendations']['motive_comparison'] = modify_params.get('motive_comparison', None)
            result['recommendations']['abstraction_level_action'] = modify_params.get('abstraction_level_action', None)
            result['recommendations']['inclusion_update'] = modify_params.get('inclusion_update', None)
            result['recommendations']['exclusion_update'] = modify_params.get('exclusion_update', None)
            result['recommendations']['parent_theme_label'] = modify_params.get('parent_theme_label', None)

    if generated_code_data:
        result['recommendations']['final_code_label'] = generated_code_data.get('code_label_proposal', None)

    # 5. Extract validation with rationale included
    if step4_validations and cluster_id in step4_validations:
        val_result = step4_validations[cluster_id]
        if 'code_validation' in val_result:
            validation = val_result['code_validation']
            result['validation']['verdict'] = validation.get('verdict', None)
            result['validation']['validated_decision'] = validation.get('validated_decision', None)
            result['validation']['validated_code'] = validation.get('validated_code', None)
            result['validation']['rationale'] = validation.get('decision_rationale', None)

    return result


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
    
    main_cluster_id = cluster_id.split('-')[0]
    
    print("\n" + "="*80 + f"\nCLUSTER {main_cluster_id} REASONING ANALYSIS\n" + "="*80)
    
    _display_single_cluster(codebook_reasoning, cluster_id, show_detailed_reasoning, debug_mode)
        
def _display_single_cluster(codebook_reasoning, cluster_id: Union[int, str], show_detailed_reasoning: bool, debug_mode: bool) -> None:
    """Display analysis for a single cluster (helper function)"""
    step1_inputs = getattr(codebook_reasoning, 'step1_inputs', {})
    step1_summaries = getattr(codebook_reasoning, 'step1_summaries', {})
    step2_analysis = getattr(codebook_reasoning, 'step2_analysis', {})  
    step3_recommendations = getattr(codebook_reasoning, 'step3_recommendations', {})
    step4_validations = getattr(codebook_reasoning, 'step4_validations', {})
    
    if False: #debug
        cluster_id = '43-1'
        show_detailed_reasoning = False

    main_cluster_id = cluster_id.split('-')[0] 
    
    # 1. CLUSTER IDEAS 
    if step1_inputs and cluster_id in step1_inputs:
        cluster_text = step1_inputs[cluster_id].get('cluster_text', '')  
        if cluster_text:
            ideas = [idea.strip() for idea in cluster_text.split('\n') if idea.strip()]
            clean_ideas = [idea[2:].strip() if idea.startswith('- ') else idea for idea in ideas]
            print(f"\n💡 CLUSTER {main_cluster_id} IDEAS ({len(clean_ideas)} responses):")
            for i, idea in enumerate(clean_ideas, 1):  
                print(f"   {i}. {idea}")
            # for i, idea in enumerate(clean_ideas[:5], 1):  # Show first 10
            #     print(f"   {i}. {idea}")
            # if len(clean_ideas) > 5:
            #     print(f"   ... and {len(clean_ideas) - 5} more ideas")
        else:
            print(f"\n💡 CLUSTER {main_cluster_id} IDEAS: [No cluster_text found]")
    else:
        print(f"\n💡 CLUSTER {main_cluster_id} IDEAS: [Not available]")
    
    # 2. CLUSTER THEME(S)
    if step1_summaries and cluster_id in step1_summaries:
        step1_data = step1_summaries[cluster_id]
        analysis = step1_data.get("analysis", [])
        
        theme_id = step1_data.get("theme_id", [])
        theme_label = step1_data.get("theme_label", [])
        theme_description = step1_data.get("theme_description", [])
        
        print(f"\n🧠 CLUSTER {main_cluster_id} ANALYSIS:")
        if analysis:
            print(f"{analysis}")
        else:
            print("[No analysis]")

        print(f"\n🔍 CLUSTER {cluster_id} THEME:")
        if theme_id > 0:
            #print(f"Theme: {theme_id}")
            print(f"Label: {theme_label}")
            print(f"Description: {theme_description}")

            # Display new fields if available
            if 'themes' in step1_data and step1_data['themes']:
                first_theme = step1_data['themes'][0]
                
                if False: # modification parameters
                    # Abstraction level
                    abstraction_level = None
                    if hasattr(first_theme, 'abstraction_level'):
                        abstraction_level = first_theme.abstraction_level
                    elif isinstance(first_theme, dict):
                        abstraction_level = first_theme.get('abstraction_level')
    
                    if abstraction_level:
                        print(f"Abstraction Level: {abstraction_level}")
    
                    # Assignment examples
                    assignment_examples = None
                    if hasattr(first_theme, 'assignment_examples'):
                        assignment_examples = first_theme.assignment_examples
                    elif isinstance(first_theme, dict):
                        assignment_examples = first_theme.get('assignment_examples')

                    if assignment_examples:
                        # Inclusion examples
                        inclusion = []
                        if hasattr(assignment_examples, 'inclusion'):
                            inclusion = assignment_examples.inclusion
                        elif isinstance(assignment_examples, dict):
                            inclusion = assignment_examples.get('inclusion', [])

                        if inclusion:
                            print("Inclusion Examples:")
                            for example in inclusion:
                                print(f"  • {example}")

                        # Exclusion examples
                        exclusion = []
                        if hasattr(assignment_examples, 'exclusion'):
                            exclusion = assignment_examples.exclusion
                        elif isinstance(assignment_examples, dict):
                            exclusion = assignment_examples.get('exclusion', [])

                        if exclusion:
                            print("Exclusion Examples:")
                            for example in exclusion:
                                print(f"  • {example}")

                        # Near neighbor
                        near_neighbor = None
                        if hasattr(assignment_examples, 'near_neighbor'):
                            near_neighbor = assignment_examples.near_neighbor
                        elif isinstance(assignment_examples, dict):
                            near_neighbor = assignment_examples.get('near_neighbor')

                        if near_neighbor:
                            neighbor_label = None
                            tell_apart = None
                            if hasattr(near_neighbor, 'label'):
                                neighbor_label = near_neighbor.label
                                tell_apart = near_neighbor.tell_apart_rule
                            elif isinstance(near_neighbor, dict):
                                neighbor_label = near_neighbor.get('label')
                                tell_apart = near_neighbor.get('tell_apart_rule')

                            if neighbor_label:
                                print(f"Near Neighbor: {neighbor_label}")
                                if tell_apart:
                                    print(f"  How to tell apart: {tell_apart}")
        else:
            print("[No themes identified]")
    else:
        print(f"\n🔍 1. CLUSTER {main_cluster_id} THEME(S): [Not available]")
    
    # 3. CANDIDATE CODES (from step2_analysis)
    if step2_analysis:
        # Try to find candidate codes for this cluster
        candidate_codes = None
        candidate_codes = []
        
        if cluster_id in step2_analysis:
            step2_data = step2_analysis[cluster_id]
            decision = step2_data['coding_decision']
            if 'matched_candidates' in decision:
                candidate_codes.extend(decision['matched_candidates'])

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
    
    
    # 4. RECOMMENDED CHANGES TO CODEBOOK  
    # Combine information from step2 (decisions) and step3 (generated codes)
    decision_data = None
    if step2_analysis and cluster_id in step2_analysis:
        step2_data = step2_analysis[cluster_id]
        if isinstance(step2_data, dict) and 'coding_decision' in step2_data:
            decision_data = step2_data['coding_decision']
    
    generated_code_data = None
    if step3_recommendations and cluster_id in step3_recommendations:
        generated_code_data = step3_recommendations[cluster_id]
       
    
    if decision_data and generated_code_data: 
        print("\n🎯 RECOMMENDED CHANGES TO CODEBOOK:")
        
        # Process single decision with generated code
        decision_type = decision_data.get('decision', 'Unknown')
        source_code = decision_data.get('source_code', None)
        justification = decision_data.get('justification', 'No justification provided')
        
        final_code_label = generated_code_data.get('code_label_proposal', 'Unknown')
        
        if decision_type ==  "CREATE": 
            print(f"{decision_type.upper()} new code") 
        if decision_type == "MODIFY": 
            source_display = source_code if source_code else "[Unknown Source]"
            print(f"{decision_type.upper()} this existing code: {source_display}")    
        if decision_type == "USE": 
            source_display = source_code if source_code else "[Unknown Source]"
            print(f"{decision_type.upper()} this existing code: {source_display}")      

        print(f"\nReasoning: {justification}")

        # Display modify_parameters if present
        if False: 
            if 'modify_parameters' in decision_data:
                modify_params = decision_data['modify_parameters']
                print("\nModification Details:")
                if modify_params.get('modify_instruction') and modify_params['modify_instruction'] != 'none':
                    print(f"  Type: {modify_params['modify_instruction']}")
                if modify_params.get('motive_comparison'):
                    print(f"  Motive: {modify_params['motive_comparison']}")
                if modify_params.get('abstraction_level_action') and modify_params['abstraction_level_action'] != 'none':
                    print(f"  Abstraction action: {modify_params['abstraction_level_action']}")
                if modify_params.get('inclusion_update'):
                    print(f"  Inclusion updates: {modify_params['inclusion_update']}")
                if modify_params.get('exclusion_update'):
                    print(f"  Exclusion updates: {modify_params['exclusion_update']}")
                if modify_params.get('parent_theme_label'):
                    print(f"  Parent theme: {modify_params['parent_theme_label']}")

        print(f"\nRecommended code: {final_code_label}")

        # else:
        #     print("[No recommendations found]")
    else:
        print("\n🎯 RECOMMENDED CHANGES TO CODEBOOK: [Not available]")
    
    
    # 5. VALIDATION WITH REASONING
    if step4_validations and cluster_id in step4_validations:
        val_result = step4_validations[cluster_id]
        print("\n✅ VALIDATION:")
        
        if 'code_validation' in val_result:
            validation = val_result['code_validation']
            #theme_desc = validation.get('theme_description', 'Unknown theme')
            #original_rec = validation.get('original_recommendation', 'Not available')
            verdict = validation.get('verdict', 'Unknown')  # APPROVE/REJECT
            validated_decision = validation.get('validated_decision', 'Unknown')  # USE/CREATE/MODIFY (final decision)
            rationale = validation.get('decision_rationale', 'Not provided')
            
            # print(f"\n   Theme {i}: {theme_desc}")
            # print(f"   Original recommendation: {original_rec}")
            
            print(f"Verdict: {verdict}") 
            print(f"Final Decision: {validated_decision}")
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
                    if key != 'Validated assignment_examples':
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
     
    
    # Step 2 Decision breakdown (decisions are in step2_analysis now)
    step2_analysis = getattr(codebook_reasoning, 'step2_analysis', {})
    if step2_analysis:
        all_decisions = []
        for step2_data in step2_analysis.values():
            if isinstance(step2_data, dict) and 'coding_decision' in step2_data:
                decision = step2_data['coding_decision']
                all_decisions.append(decision.get('decision', 'unknown'))
        
        print("\nStep 2 Decisions:")
        print(f"• Create new: {all_decisions.count('create')}")
        print(f"• Modify existing: {all_decisions.count('modify')}")  
        print(f"• Use existing: {all_decisions.count('use')}")
        print(f"• Total decisions: {len(all_decisions)}")
    
    # Step 3 Generated codes breakdown  
    if step3_recommendations:
        all_generated = []
        for gen_result in step3_recommendations.values():
            if 'generated_code' in gen_result:
                code = gen_result['generated_code']
                all_generated.append(code.get('code_label', 'unknown'))
        
        print("\nStep 3 Generated codes:")
        print(f"• Total codes generated: {len(all_generated)}")
    
    # Step 4 Validation breakdown
    if step4_validations:
        all_verdicts = []
        all_final_decisions = []
        for val_result in step4_validations.values():
            if 'code_validation' in val_result:
                validation = val_result['code_validation']
                all_verdicts.append(validation.get('verdict', 'unknown'))
                all_final_decisions.append(validation.get('validated_decision', 'unknown'))
        
        print("\nStep 4 Validation:")
        print(f"• Verdict - Approved: {all_verdicts.count('APPROVE')}")
        print(f"• Verdict - Rejected: {all_verdicts.count('REJECT')}")
        print(f"• Final Decision - USE: {all_final_decisions.count('use')}")
        print(f"• Final Decision - CREATE: {all_final_decisions.count('create')}")
        print(f"• Final Decision - MODIFY: {all_final_decisions.count('modify')}")
        print(f"• Total validations: {len(all_verdicts)}")


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
    """step 2 decisions: create; modify and use"""
   
    matching_clusters = []
    step2_analysis = getattr(codebook_reasoning, 'step2_analysis', {})
    
    for cluster_id, step2_data in step2_analysis.items():
        # Check if any decision in this cluster matches the target type
        if isinstance(step2_data, dict) and 'coding_decisions' in step2_data:
            for decision in step2_data['coding_decisions']:
                if decision.get('decision') == decision_type:
                    matching_clusters.append(cluster_id)
                    break  # Found one match, move to next cluster
    
    return matching_clusters