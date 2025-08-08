import random
from typing import Optional, List, Union, Dict
import models


def display_cluster_analysis(results: Union[models.CodeGeneratorReasoningResults, Dict], 
                           cluster_id: Optional[int] = None, 
                           show_detailed_reasoning: bool = False) -> None:
    """Display analysis using cached reasoning results or dictionary format"""
    
    print("\n" + "="*80 + "\nCODEBOOK REASONING ANALYSIS\n" + "="*80)
    
    # Handle both Pydantic model and dictionary formats
    if isinstance(results, dict):
        # New dictionary format from generate()
        step1_summaries = results.get('step1_summaries', {})
        step2_candidate_codes = results.get('candidate_codes_data', {})
        step3 = results.get('step3_recommendations', {})
        step4_validations = results.get('validation_details', {})
        step4_validated_codes = results.get('step4_validated_codes', {})
        cluster_data = results.get('cluster_data', {})  # Get cluster ideas data
    else:
        # Legacy Pydantic model format
        step1_summaries = results.step1_summaries
        step2_candidate_codes = results.step2_analysis
        step3 = results.step3_recommendations
        step4_validations = results.step4_validations
        step4_validated_codes = results.step4_validated_codes
        cluster_data = {}  # Not available in legacy format
    
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
    
    # Step 1: Cluster Summary (with themes)
    if step1_summaries and cluster_id in step1_summaries:
        print("\n📝 STEP 1 - Cluster Summary & Theme Analysis:")
        summary_data = step1_summaries[cluster_id]
        
        # Try to parse JSON string format
        if isinstance(summary_data, str):
            try:
                import json
                parsed_data = json.loads(summary_data)
                if isinstance(parsed_data, dict):
                    summary_data = parsed_data
            except (json.JSONDecodeError, TypeError):
                # If parsing fails, treat as legacy string format
                pass
        
        if isinstance(summary_data, dict) and 'cluster_summary' in summary_data:
            # Multi-theme JSON format (parsed or direct dict)
            print(f'Summary: "{summary_data["cluster_summary"]}"')
            themes = summary_data.get('themes', [])
            if themes:
                if len(themes) == 1:
                    print(f"Theme: {themes[0]}")
                else:
                    print(f"Themes identified ({len(themes)}):")
                    for i, theme in enumerate(themes, 1):
                        print(f"  Theme {i}: {theme}")
            else:
                print("Themes: No coherent themes identified")
        else:
            # Legacy string format
            print(f'Summary: "{summary_data}"')
    else:
        print("\n📝 STEP 1 - Cluster Summary & Theme Analysis: [Not available]")
    
    # Cluster Ideas (show actual ideas for debugging)
    if cluster_data and cluster_id in cluster_data:
        ideas = cluster_data[cluster_id].get('ideas', [])
        print(f"\n💡 Cluster Ideas ({len(ideas)} total):")
        # Show first 5 ideas for debugging, with option to show more
        ideas_to_show = ideas[:5] if len(ideas) > 5 else ideas
        for i, idea in enumerate(ideas_to_show, 1):
            # Truncate long ideas for readability
            display_idea = idea[:100] + "..." if len(idea) > 100 else idea
            print(f"  {i}. {display_idea}")
        if len(ideas) > 5:
            print(f"  ... and {len(ideas) - 5} more ideas")
    else:
        print(f"\n💡 Cluster Ideas: [Not available in this format]")
    
    # Step 2: Candidate Codes 
    if step2_candidate_codes and cluster_id in step2_candidate_codes:
        codes = step2_candidate_codes[cluster_id]
        print("\n🔍 STEP 2 - Candidate Code Selection:")
        
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
        print("\n🔍 STEP 2 - Candidate Code Selection:")
        print("  No candidate codes")
  
    # Step 3: Code Generation (multi-theme support)
    rec = step3[cluster_id]
    print("\n📋 STEP 3 - Code Generation:")
    
    if isinstance(rec, dict) and 'coding_decisions' in rec:
        # Multi-theme JSON format
        cluster_analysis = rec.get('cluster_analysis', {})
        num_themes = cluster_analysis.get('number_of_themes', len(rec['coding_decisions']))
        print(f"• Themes identified: {num_themes}")
        
        coding_decisions = rec['coding_decisions']
        for i, decision in enumerate(coding_decisions, 1):
            print(f"\n  Theme {decision.get('theme_number', i)}: {decision.get('theme_description', 'Unknown theme')}")
            print(f"  • Decision: {decision.get('decision', 'Unknown').replace('_', ' ').title()}")
            
            action_details = decision.get('action_details', {})
            if action_details.get('codes_to_use'):
                print(f"    - Codes to use: {', '.join(action_details['codes_to_use'])}")
            elif action_details.get('codes_to_modify'):
                print(f"    - Code to modify: {action_details['codes_to_modify']}")
                print(f"    - Modified code name: {action_details.get('modified_code_name', '')}")
                print(f"    - Modified definition: {action_details.get('modified_code_definition', '')}")
            elif action_details.get('new_code_name'):
                print(f"    - New code name: {action_details['new_code_name']}")
                print(f"    - New code definition: {action_details.get('new_code_definition', '')}")
            
            print(f"  • Justification: {decision.get('justification', '')}")
        
        overall_justification = rec.get('overall_justification', '')
        if overall_justification:
            print(f"\n• Overall Justification: {overall_justification}")
            
    else:
        # Legacy single-theme format
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
    
    # Step 4: Validation (multi-theme support)
    val = step4_validations.get(cluster_id)
    if val:
        print("\n✅ STEP 4 - Validation:")
        
        if isinstance(val, dict) and 'theme_validations' in val:
            # Multi-theme validation format
            theme_validations = val['theme_validations']
            print(f"• Themes validated: {len(theme_validations)}")
            
            for theme_val in theme_validations:
                theme_num = theme_val.get('theme_number', 1)
                theme_desc = theme_val.get('theme_description', 'Unknown theme')
                decision = theme_val.get('decision', 'Unknown')
                
                print(f"\n  Theme {theme_num}: {theme_desc}")
                print(f"  • Decision: {decision}")
                print(f"  • Rationale: {theme_val.get('decision_rationale', '')}")
                
                if show_detailed_reasoning and theme_val.get('evaluation'):
                    evaluation = theme_val['evaluation']
                    print("  • Detailed Evaluation:")
                    print(f"    - Semantic Fit: {evaluation.get('semantic_fit', 'N/A')}")
                    print(f"    - Atomicity: {evaluation.get('atomicity', 'N/A')}")
                    print(f"    - Parsimony: {evaluation.get('parsimony', 'N/A')}")
                    print(f"    - Redundancy: {evaluation.get('redundancy', 'N/A')}")
            
            overall_summary = val.get('overall_summary', '')
            if overall_summary:
                print(f"\n• Overall Summary: {overall_summary}")
        else:
            # Legacy single-theme validation format
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
    
    else:
        print("\n✅ STEP 4 - Validation: [No validation performed]")
    
    # Final Validated Codes (check both sources)
    validated_codes_to_show = []
    
    # First check step4_validated_codes (new format)
    if step4_validated_codes and cluster_id in step4_validated_codes:
        validated_codes_to_show = step4_validated_codes[cluster_id]
    # Fallback to validation details (legacy format)  
    elif val and 'validated_code' in val and val['validated_code']:
        validated_codes_to_show = [val['validated_code']]
    
    if validated_codes_to_show:
        if len(validated_codes_to_show) == 1:
            validated_code = validated_codes_to_show[0]
            print("\n📌 Final Validated Code:")
            print(f"• Code: {validated_code.get('code', 'N/A')}")
            print(f"• Definition: {validated_code.get('definition', 'N/A')}")
        else:
            print(f"\n📌 Final Validated Codes ({len(validated_codes_to_show)} themes):")
            for i, validated_code in enumerate(validated_codes_to_show, 1):
                theme_num = validated_code.get('theme_number', i)
                theme_desc = validated_code.get('theme_description', '')
                print(f"  Theme {theme_num}: {theme_desc}")
                print(f"  • Code: {validated_code.get('code', 'N/A')}")
                print(f"  • Definition: {validated_code.get('definition', 'N/A')}")
                if i < len(validated_codes_to_show):
                    print()
    else:
        print("\n📌 Final Validated Codes: [Not generated/validated]")
    
    print("\n" + "="*80)


def display_summary_statistics(results: Union[models.CodeGeneratorReasoningResults, Dict]) -> None:
    """
    Display summary statistics for the entire pipeline run
    
    Args:
        results: The cached reasoning results or dictionary format
    """
    # Handle both Pydantic model and dictionary formats
    if isinstance(results, dict):
        # New dictionary format from generate()
        stats = results.get('stats', {})
        step3 = results.get('step3_recommendations', {})
        validation = results.get('validation_details', {})
    else:
        # Legacy Pydantic model format
        stats = results.stats
        step3 = results.step3_recommendations
        validation = results.step4_validations
    
    print("\n" + "="*80 + "\nPIPELINE SUMMARY STATISTICS\n" + "="*80)
    
    # Decision breakdown
    if step3:
        decisions = []
        # Handle multi-theme format where decisions are in coding_decisions arrays
        for rec in step3.values():
            if isinstance(rec, dict) and 'coding_decisions' in rec:
                # New multi-theme format - collect all decisions from all themes
                for decision in rec['coding_decisions']:
                    decisions.append(decision.get('decision', 'unknown'))
            elif hasattr(rec, 'decision'):
                # Legacy single-theme format
                decisions.append(rec.decision)
            elif isinstance(rec, dict) and 'decision' in rec:
                # Single-theme dict format
                decisions.append(rec['decision'])
        
        print("\nStep 3 Decisions:")
        print(f"• Create new: {decisions.count('create_new')}")
        print(f"• Modify existing: {decisions.count('modify_existing')}")
        print(f"• Use existing: {decisions.count('use_existing')}")
        print(f"• Total themes processed: {len(decisions)}")
        print(f"• Total clusters: {len(step3)}")
    
    # Validation breakdown
    if validation:
        val_decisions = []
        # Handle multi-theme validation format where decisions are in code_validations arrays
        for v in validation.values():
            if isinstance(v, dict) and 'code_validations' in v:
                # New multi-theme format - collect all validation decisions
                for val in v['code_validations']:
                    val_decisions.append(val.get('decision', 'UNKNOWN'))
            elif isinstance(v, dict) and 'decision' in v:
                # Single-theme format
                val_decisions.append(v['decision'])
                
        print("\nStep 4 Validation:")
        print(f"• Approved: {val_decisions.count('APPROVE')}")
        print(f"• Revised: {val_decisions.count('REVISE')}")
        print(f"• Rejected: {val_decisions.count('REJECT')}")
        print(f"• Merged: {val_decisions.count('MERGE')}")
        print(f"• Split: {val_decisions.count('SPLIT')}")
    
    # Performance stats
    if stats:
        print("\nPerformance:")
        #print(f"• Total processing time: {stats.get('total_time', 0):.2f}s")
        print(f"• New codes added: {stats.get('new_codes_added', 0)}")
        print(f"• Codes modified: {stats.get('codes_modified', 0)}")
        print(f"• No new codes needed: {stats.get('no_new_codes_needed', 0)}")


def display_multiple_clusters(results: Union[models.CodeGeneratorReasoningResults, Dict], 
                            cluster_ids: List[int] = None, 
                            max_clusters: int = 5) -> None:
    """
    Display analysis for multiple clusters
    
    Args:
        results: The cached reasoning results or dictionary format
        cluster_ids: List of specific cluster IDs to display (None for random selection)
        max_clusters: Maximum number of clusters to display
    """
    # Handle both formats
    if isinstance(results, dict):
        step3 = results.get('step3_recommendations', {})
    else:
        step3 = results.step3_recommendations
    
    if cluster_ids is None:
        # Random selection
        available_ids = list(step3.keys())
        num_to_show = min(max_clusters, len(available_ids))
        cluster_ids = random.sample(available_ids, num_to_show)
    
    print(f"\nDisplaying {len(cluster_ids)} clusters:")
    
    for idx, cluster_id in enumerate(cluster_ids, 1):
        print(f"\n{'='*20} CLUSTER {idx}/{len(cluster_ids)} {'='*20}")
        display_cluster_analysis(results, cluster_id, show_detailed_reasoning=False)


def find_clusters_by_decision(results: Union[models.CodeGeneratorReasoningResults, Dict], 
                             decision_type: str) -> List[int]:
    """
    Find all clusters with a specific decision type
    
    Args:
        results: The cached reasoning results or dictionary format
        decision_type: 'create_new', 'modify_existing', or 'use_existing'
    
    Returns:
        List of cluster IDs matching the decision type
    """
    # Handle both formats
    if isinstance(results, dict):
        step3 = results.get('step3_recommendations', {})
    else:
        step3 = results.step3_recommendations
    matching_clusters = []
    
    for cluster_id, rec in step3.items():
        # Handle multi-theme format where decisions are in coding_decisions arrays
        if isinstance(rec, dict) and 'coding_decisions' in rec:
            # New multi-theme format - check if any theme has the matching decision
            for decision in rec['coding_decisions']:
                if decision.get('decision') == decision_type:
                    matching_clusters.append(cluster_id)
                    break  # Only add cluster once, even if multiple themes match
        elif hasattr(rec, 'decision') and rec.decision == decision_type:
            # Legacy single-theme format
            matching_clusters.append(cluster_id)
        elif isinstance(rec, dict) and rec.get('decision') == decision_type:
            # Single-theme dict format
            matching_clusters.append(cluster_id)
    
    return matching_clusters


def load_and_display_reasoning(cache_manager, filename: str, cluster_id: int = None, 
                              step_name: str = "codebook_generation"):
    """
    Load cached reasoning results and display them
    
    Args:
        cache_manager: The cache manager instance
        filename: The cache filename
        cluster_id: Specific cluster to display (None for random)
        step_name: The step name
    """
    try:
        reasoning_models = cache_manager.load_from_cache(
            filename, f"{step_name}_reasoning", models.CodeGeneratorReasoningResults
        )
        
        if reasoning_models and len(reasoning_models) > 0:
            results = reasoning_models[0]
            print("📁 Displaying from cached reasoning results...")
            display_cluster_analysis(results, cluster_id)
        else:
            print("ERROR: No reasoning cache found.")
            print("   Enable CACHE_CODEGENERATOR_REASONING=True in pipeline and re-run to create cache.")
            
    except Exception as e:
        print(f"ERROR: Error loading reasoning cache: {e}")
        print("   Make sure the cache file exists and was created with the current model structure.")