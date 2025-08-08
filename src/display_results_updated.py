"""
Updated display code that can be directly pasted into your notebook or script
after running: results = generator.generate()
"""

# Unpack and display Step 7 results (code generation) - ENHANCED VERSION
if 'results' in locals():
    import random
    
    print("\n" + "="*80 + "\nSTEP 7 RESULTS ANALYSIS\n" + "="*80)
    
    # Extract data from results
    cluster_data = results.get('cluster_data', {})   
    cluster_assignments = results.get('cluster_assignments', {})
    step2 = results.get('step2_summaries', {})
    step3 = results.get('step3_recommendations', {})
    step4 = results.get('step4_validated_codes', {})
    validation = results.get('validation_details', {})
    candidate_codes = results.get('candidate_codes_data', {})
  
    # Sample a random cluster
    if step3:
        available_ids = list(step3.keys())
        sampled_id = random.choice(available_ids)
        
        print(f"\n📋 Random Sample Cluster Analysis:")
        print("-" * 40)
        print(f"Cluster ID: {sampled_id}")
        
        # STEP 2 - Cluster Theme Summary
        if step2 and sampled_id in step2:
            print(f"\n📝 Prompt 2 - Cluster Theme Summary:")
            print(f'"{step2[sampled_id]}"')
        else:
            print(f"\n📝 Prompt 2 - Cluster Theme Summary: [Not available - update codeGenerator.py to capture]")
        
        # Cluster Ideas
        if cluster_data and sampled_id in cluster_data:
            cluster_info = cluster_data[sampled_id]
            ideas = cluster_info.get('ideas', [])
            print(f"\n💡 Cluster Ideas ({len(ideas)} total):")
            for i, idea in enumerate(ideas[:10], 1):
                print(f"  {i}. {idea[:80]}..." if len(idea) > 80 else f"  {i}. {idea}")
            if len(ideas) > 10:
                print(f"  ... and {len(ideas) - 10} more")
        
        # Candidate codes
        if candidate_codes and sampled_id in candidate_codes:
            codes = candidate_codes[sampled_id]
            print(f"\n🔍 Candidate Codes:")
            for i, code in enumerate(codes[:5], 1):  # Show up to 5 candidate codes
                if isinstance(code, dict):
                    print(f"  {i}. {code.get('code', 'Unknown')}: {code.get('definition', 'No definition')[:60]}...")
                else:
                    print(f"  {i}. {code}")
            if len(codes) > 5:
                print(f"  ... and {len(codes) - 5} more candidate codes")
        
        # STEP 3 - Recommendation
        rec = step3[sampled_id]
        print(f"\n📋 Prompt 3 - Recommendation:")
        print(f"• Decision: {rec.decision}")
        print(f"• Core Theme: {rec.cluster_core_theme}")
        
        # Action details based on decision type
        print("• Action Details:")
        if hasattr(rec, 'action_details') and rec.action_details:
            if rec.decision == 'use_existing' and hasattr(rec.action_details, 'codes_to_use'):
                print(f"  - Codes to use: {', '.join(rec.action_details.codes_to_use)}")
            elif rec.decision == 'modify_existing':
                print(f"  - Code to modify: {rec.action_details.codes_to_modify}")
                print(f"  - Modified code name: {rec.action_details.modified_code_name}")
                print(f"  - Modified definition: {rec.action_details.modified_code_definition}")
            elif rec.decision == 'create_new':
                print(f"  - New code name: {rec.action_details.new_code_name}")
                print(f"  - New code definition: {rec.action_details.new_code_definition}")
        
        print(f"• Justification:")
        justification_lines = rec.justification.split('\n')
        for line in justification_lines:
            if line.strip():
                print(f"  {line.strip()}")
        
        # STEP 4 - Validation
        val = validation.get(sampled_id)
        if val:
            print(f"\n✅ Prompt 4 - Validation:")
            print(f"• Decision: {val['decision']}")
            print(f"• Decision Rationale:")
            rationale_lines = val['decision_rationale'].split('\n')
            for line in rationale_lines:
                if line.strip():
                    print(f"  {line.strip()}")
            
            # Optional: Show detailed evaluation
            # if 'reasoning' in val:
            #     reasoning = val['reasoning']
            #     print("• Detailed Evaluation:")
            #     print(f"  - Semantic Fit & Coverage: {reasoning.get('semantic_fit_reasoning', 'N/A')}")
            #     print(f"  - Atomicity: {reasoning.get('atomicity_reasoning', 'N/A')}")
            #     print(f"  - Parsimony: {reasoning.get('parsimony_reasoning', 'N/A')}")
            #     print(f"  - Non-redundancy: {reasoning.get('redundancy_reasoning', 'N/A')}")
            #     print(f"  - Justification Alignment: {reasoning.get('justification_reasoning', 'N/A')}")
       
        # Final validated code
        validated_code = step4.get(sampled_id)
        if validated_code:
            print(f"\n📌 Final Validated Code:")
            print(f"• Code: {validated_code['code']}")
            print(f"• Definition: {validated_code['definition']}")
        else:
            print(f"\n📌 Final Validated Code: [Not generated/validated]")
            
    else:
        print("No clusters with recommendations found in results.")
else:
    print("No results found. Please run: results = generator.generate()")