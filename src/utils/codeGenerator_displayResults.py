import random
from typing import Dict, Any, Optional, List


def display_cluster_analysis(results: Dict[str, Any], cluster_id: Optional[int] = None, show_detailed_reasoning: bool = False) -> None:
    
    print("\n" + "="*80 + "\nSTEP 7 RESULTS ANALYSIS\n" + "="*80)
    
    # Extract data from results
    cluster_data = results.get('cluster_data', {})
    step2 = results.get('step2_summaries', {})
    step3 = results.get('step3_recommendations', {})
    step4 = results.get('step4_validated_codes', {})
    validation = results.get('validation_details', {})
    candidate_codes_data = results.get('candidate_codes_data', {})
    
    # Select cluster to display
    if cluster_id is None:
        if step3:
            available_ids = list(step3.keys())
            cluster_id = random.choice(available_ids)
        else:
            print("No clusters with recommendations found.")
            return
    
    # Verify cluster exists
    if cluster_id not in step3:
        print(f"Cluster {cluster_id} not found in results.")
        return
    
    print(f"\n📋 Cluster Analysis (ID: {cluster_id})")
    print("-" * 40)
    
    # Step 2: Cluster Summary
    if step2 and cluster_id in step2:
        print("\n📝 STEP 2 - Cluster Theme Summary:")
        summary_text = step2[cluster_id]
        # Format the summary with proper indentation
        print(f'"{summary_text}"')
    else:
        print("\n📝 STEP 2 - Cluster Theme Summary: [Not available]")
    
    # Cluster Ideas
    if cluster_data and cluster_id in cluster_data:
        cluster_info = cluster_data[cluster_id]
        ideas = cluster_info.get('ideas', [])
        print(f"\n💡 Cluster Ideas ({len(ideas)} total):")
        
        # Show up to 10 ideas
        for i, idea in enumerate(ideas[:10], 1):
            if len(idea) > 80:
                print(f"  {i}. {idea[:77]}...")
            else:
                print(f"  {i}. {idea}")
        
        if len(ideas) > 10:
            print(f"  ... and {len(ideas) - 10} more")
    
    # Candidate Codes (from Step 1)
    if candidate_codes_data and cluster_id in candidate_codes_data:
        codes = candidate_codes_data[cluster_id]
        print("\n🔍 Candidate Codes:")
        
        # Show up to 5 candidate codes
        for i, code in enumerate(codes[:5], 1):
            if isinstance(code, dict):
                code_name = code.get('code', 'Unknown')
                definition = code.get('definition', 'No definition')
                if len(definition) > 60:
                    print(f"  {i}. {code_name}: {definition[:57]}...")
                else:
                    print(f"  {i}. {code_name}: {definition}")
            else:
                print(f"  {i}. {code}")
        
        if len(codes) > 5:
            print(f"  ... and {len(codes) - 5} more candidate codes")
    else:
        print("\n🔍 Candidate Codes:")
        print("  No candidate codes")
  
    # Step 3: Recommendation
    rec = step3[cluster_id]
    print("\n📋 STEP 3 - Recommendation:")
    print(f"• Decision: {rec.decision}")
    print(f"• Core Theme: {rec.cluster_core_theme}")
    
    # Action details based on decision type
    if hasattr(rec, 'action_details') and rec.action_details:
        print("• Action Details:")
        
        if rec.decision == 'use_existing' and hasattr(rec.action_details, 'codes_to_use'):
            codes_list = rec.action_details.codes_to_use
            print(f"  - Codes to use: {', '.join(codes_list)}")
            
        elif rec.decision == 'modify_existing' and hasattr(rec.action_details, 'codes_to_modify'):
            print(f"  - Code to modify: {rec.action_details.codes_to_modify}")
            print(f"  - Modified code name: {rec.action_details.modified_code_name}")
            print(f"  - Modified definition: {rec.action_details.modified_code_definition}")
            
        elif rec.decision == 'create_new' and hasattr(rec.action_details, 'new_code_name'):
            print(f"  - New code name: {rec.action_details.new_code_name}")
            print(f"  - New code definition: {rec.action_details.new_code_definition}")
    
    # Justification
    print("• Justification:")
    # Format justification with proper indentation
    justification_lines = rec.justification.split('\n')
    for line in justification_lines:
        if line.strip():
            print(f"  {line.strip()}")
    
    # Step 4: Validation
    val = validation.get(cluster_id)
    if val:
        print("\n✅ STEP 4 - Validation:")
        print(f"• Decision: {val['decision']}")
        print("• Decision Rationale:")
        rationale_lines = val['decision_rationale'].split('\n')
        for line in rationale_lines:
            if line.strip():
                print(f"  {line.strip()}")
        
        # Show detailed reasoning if requested
        if show_detailed_reasoning and 'reasoning' in val:
            reasoning = val['reasoning']
            print("\n• Detailed Evaluation:")
            print(f"  - Semantic Fit & Coverage: {reasoning.get('semantic_fit_reasoning', 'N/A')}")
            print(f"  - Atomicity: {reasoning.get('atomicity_reasoning', 'N/A')}")
            print(f"  - Parsimony: {reasoning.get('parsimony_reasoning', 'N/A')}")
            print(f"  - Non-redundancy: {reasoning.get('redundancy_reasoning', 'N/A')}")
            print(f"  - Justification Alignment: {reasoning.get('justification_reasoning', 'N/A')}")
    
    # Final Validated Code
    validated_code = step4.get(cluster_id)
    if validated_code:
        print("\n📌 Final Validated Code:")
        print(f"• Code: {validated_code['code']}")
        print("• Definition: {validated_code['definition']}")
    else:
        print("\n📌 Final Validated Code: [Not generated/validated]")
    
    print("\n" + "="*80)


def display_summary_statistics(results: Dict[str, Any]) -> None:
    """
    Display summary statistics for the entire pipeline run
    
    Args:
        results: The results dictionary from generator.generate()
    """
    stats = results.get('stats', {})
    step3 = results.get('step3_recommendations', {})
    validation = results.get('validation_details', {})
    
    print("\n" + "="*80 + "\nPIPELINE SUMMARY STATISTICS\n" + "="*80)
    
    # Decision breakdown
    if step3:
        decisions = [rec.decision for rec in step3.values()]
        print("\nStep 3 Decisions:")
        print(f"• Create new: {decisions.count('create_new')}")
        print(f"• Modify existing: {decisions.count('modify_existing')}")
        print(f"• Use existing: {decisions.count('use_existing')}")
        print(f"• Total clusters: {len(decisions)}")
    
    # Validation breakdown
    if validation:
        val_decisions = [v['decision'] for v in validation.values()]
        print("\nStep 4 Validation:")
        print(f"• Approved: {val_decisions.count('APPROVE')}")
        print(f"• Revised: {val_decisions.count('REVISE')}")
        print(f"• Rejected: {val_decisions.count('REJECT')}")
    
    # Performance stats
    if stats:
        print("\nPerformance:")
        #print(f"• Total processing time: {stats.get('total_time', 0):.2f}s")
        print(f"• New codes added: {stats.get('new_codes_added', 0)}")
        print(f"• Codes modified: {stats.get('codes_modified', 0)}")
        print(f"• No new codes needed: {stats.get('no_new_codes_needed', 0)}")


def display_multiple_clusters(results: Dict[str, Any], cluster_ids: List[int] = None, 
                            max_clusters: int = 5) -> None:
    """
    Display analysis for multiple clusters
    
    Args:
        results: The results dictionary from generator.generate()
        cluster_ids: List of specific cluster IDs to display (None for random selection)
        max_clusters: Maximum number of clusters to display
    """
    step3 = results.get('step3_recommendations', {})
    
    if cluster_ids is None:
        # Random selection
        available_ids = list(step3.keys())
        num_to_show = min(max_clusters, len(available_ids))
        cluster_ids = random.sample(available_ids, num_to_show)
    
    print(f"\nDisplaying {len(cluster_ids)} clusters:")
    
    for idx, cluster_id in enumerate(cluster_ids, 1):
        print(f"\n{'='*20} CLUSTER {idx}/{len(cluster_ids)} {'='*20}")
        display_cluster_analysis(results, cluster_id, show_detailed_reasoning=False)


def find_clusters_by_decision(results: Dict[str, Any], decision_type: str) -> List[int]:
    """
    Find all clusters with a specific decision type
    
    Args:
        results: The results dictionary from generator.generate()
        decision_type: 'create_new', 'modify_existing', or 'use_existing'
    
    Returns:
        List of cluster IDs matching the decision type
    """
    step3 = results.get('step3_recommendations', {})
    matching_clusters = []
    
    for cluster_id, rec in step3.items():
        if rec.decision == decision_type:
            matching_clusters.append(cluster_id)
    
    return matching_clusters