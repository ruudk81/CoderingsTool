"""
Enhanced results display script for Step 7 (code generation)

This script demonstrates how to use the new display utilities to show
comprehensive cluster analysis results.
"""

from utils.resultsDisplay import (
    display_cluster_analysis, 
    display_summary_statistics,
    display_multiple_clusters,
    find_clusters_by_decision
)

# Enhanced display code for Step 7 results
if 'results' in locals():
    # Display summary statistics first
    display_summary_statistics(results)
    
    # Display a random cluster with full details
    print("\n" + "="*80 + "\nRANDOM CLUSTER SAMPLE\n" + "="*80)
    display_cluster_analysis(results)
    
    # Optional: Display specific types of clusters
    # Example 1: Show a cluster where new code was created
    new_code_clusters = find_clusters_by_decision(results, 'create_new')
    if new_code_clusters:
        print("\n" + "="*80 + "\nEXAMPLE: NEW CODE CREATION\n" + "="*80)
        display_cluster_analysis(results, new_code_clusters[0])
    
    # Example 2: Show a cluster where existing code was modified
    modified_clusters = find_clusters_by_decision(results, 'modify_existing')
    if modified_clusters:
        print("\n" + "="*80 + "\nEXAMPLE: CODE MODIFICATION\n" + "="*80)
        display_cluster_analysis(results, modified_clusters[0])
    
    # Example 3: Display multiple clusters at once
    # print("\n" + "="*80 + "\nMULTIPLE CLUSTER ANALYSIS\n" + "="*80)
    # display_multiple_clusters(results, max_clusters=3)
else:
    print("No results found. Please run the generator first: results = generator.generate()")