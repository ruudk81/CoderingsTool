# Enhanced Step 7 results display - place this AFTER your existing debug display in the pipeline

# Unpack and display Step 7 results (code generation) - ENHANCED VERSION  
if VERBOSE and 'results' in locals():
    import random
    
    print("\n" + "="*80 + "\nSTEP 7 ENHANCED RESULTS ANALYSIS\n" + "="*80)
    
    # Get the data
    step3 = results.get('step3_recommendations', {})
    validation = results.get('validation_details', {}) 
    step4 = results.get('step4_validated_codes', {})
    cluster_data = results.get('cluster_data', {})  # Raw cluster ideas and embeddings
    cluster_assignments = results.get('cluster_assignments', {})
    
    # Sample a random cluster
    if step3:
        available_ids = list(step3.keys())
        sampled_id = random.choice(available_ids)
        
        print(f"\n🔍 Enhanced Cluster Analysis (ID: {sampled_id})")
        print("-" * 50)
        
        # Show cluster content (raw ideas/responses)
        if cluster_data and sampled_id in cluster_data:
            cluster_info = cluster_data[sampled_id] 
            ideas = cluster_info.get('ideas', [])
            print(f"\n📝 Cluster Content ({len(ideas)} ideas):")
            for i, idea in enumerate(ideas[:3], 1):  # Show first 3 ideas
                truncated_idea = idea[:80] + "..." if len(idea) > 80 else idea
                print(f"  {i}. {truncated_idea}")
            if len(ideas) > 3:
                print(f"  ... and {len(ideas) - 3} more ideas")
        else:
            print(f"\n📝 Cluster Content: [Not available in results]")
        
        # Show Step 3 recommendation details
        rec = step3[sampled_id]
        print(f"\n🎯 Step 3 Recommendation:")
        print(f"  Core Theme: {rec.cluster_core_theme}")
        print(f"  Decision: {rec.decision}")
        
        # Extract Step 2 summary from Step 3 recommendation (it's embedded in the justification/context)
        print(f"\n📋 Step 2 Summary (embedded in Step 3):")
        print(f"  [Summary would be extracted from Step 3 context - currently embedded in recommendation]")
        
        # Show candidate codes - these would need to be captured from Step 1 processing
        print(f"\n📚 Candidate Codes (Step 1):")
        print(f"  [Candidate codes not directly stored - would need to be captured during processing]")
        
        # Action details
        if hasattr(rec, 'action_details') and rec.action_details:
            print(f"\n⚙️ Action Details:")
            if rec.decision == 'use_existing' and rec.action_details.codes_to_use:
                print(f"  Codes to use: {', '.join(rec.action_details.codes_to_use)}")
            elif rec.decision == 'modify_existing':
                print(f"  Code to modify: {rec.action_details.codes_to_modify}")
                print(f"  Modified name: {rec.action_details.modified_code_name}")
                print(f"  Modified definition: {rec.action_details.modified_code_definition}")
            elif rec.decision == 'create_new':
                print(f"  New code name: {rec.action_details.new_code_name}")
                print(f"  New definition: {rec.action_details.new_code_definition}")
        
        print(f"\n💭 Justification:")
        print(f"  {rec.justification}")
        
        # Validation details with all 5 criteria
        val = validation.get(sampled_id)
        if val:
            print(f"\n✅ Step 4 Validation:")
            print(f"  Decision: {val['decision']}")
            print(f"  Rationale: {val['decision_rationale']}")
            
            reasoning = val.get('reasoning', {})
            print(f"\n📊 Detailed Evaluation (5 Criteria):")
            print(f"  🎯 Semantic Fit & Coverage: {reasoning.get('semantic_fit_reasoning', 'N/A')}")
            print(f"  ⚛️  Atomicity: {reasoning.get('atomicity_reasoning', 'N/A')}")
            print(f"  💎 Parsimony: {reasoning.get('parsimony_reasoning', 'N/A')}")
            print(f"  🔄 Non-redundancy: {reasoning.get('redundancy_reasoning', 'N/A')}")
            print(f"  🤝 Justification Alignment: {reasoning.get('justification_reasoning', 'N/A')}")
        else:
            print(f"\n❌ No validation performed for this cluster (use_existing decision)")
        
        # Final validated code
        validated_code = step4.get(sampled_id)
        if validated_code:
            print(f"\n🏆 Final Validated Code:")
            print(f"  Code: {validated_code['code']}")
            print(f"  Definition: {validated_code['definition']}")
        else:
            final_assignment = cluster_assignments.get(sampled_id, 'No assignment')
            print(f"\n🏷️  Final Assignment: {final_assignment}")

        print("\n" + "="*80)