"""
Pipeline Summary Analysis Module

Provides simple count analysis for CoderingsTool pipeline results
"""

from typing import List, Dict, Any, Optional


class PipelineSummarizer:
    """Generates count summaries for themes and codes"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        
    def generate_summary(self, 
                        code_assigned_results: List[Any],
                        theme_enriched_codebook: Optional[Any] = None,
                        enriched_codebook: Optional[Any] = None) -> None:
        """
        Generate simple count summary of themes and codes
        
        Args:
            code_assigned_results: List of CodeAssignedModel objects
            theme_enriched_codebook: ThemeEnrichedCodebookModel object (unused for now)
            enriched_codebook: Legacy codebook format (unused for now)
        """
        
        if not code_assigned_results:
            print("\nNo code assignment results available for count analysis.")
            return
            
        # Calculate frequencies
        code_frequency = {}
        theme_frequency = {}
        
        # Count assignments
        total_ideas = 0
        for resp in code_assigned_results:
            if resp.response_ideas:
                for idea in resp.response_ideas:
                    total_ideas += 1
                    
                    # Count code assignments
                    if idea and idea.assigned_codes:
                        for code in idea.assigned_codes:
                            code_frequency[code] = code_frequency.get(code, 0) + 1
                    
                    # Count theme assignments
                    if idea and idea.assigned_themes:
                        for theme in idea.assigned_themes:
                            theme_frequency[theme] = theme_frequency.get(theme, 0) + 1
        
        # Print results
        self._print_counts(theme_frequency, code_frequency, total_ideas)
    
    def _print_counts(self, theme_frequency: Dict[str, int], code_frequency: Dict[str, int], total_ideas: int) -> None:
        """Print theme and code counts in descending order"""
        
        print("\n" + "=" * 80)
        print("📊 THEME AND CODE COUNT ANALYSIS")
        print("=" * 80)
        
        # Print theme counts
        if theme_frequency:
            sorted_themes = sorted(theme_frequency.items(), key=lambda x: x[1], reverse=True)
            print(f"\n📋 THEME COUNTS (Total: {len(sorted_themes)} themes assigned to {sum(theme_frequency.values())} ideas)")
            print("-" * 70)
            
            for i, (theme, count) in enumerate(sorted_themes, 1):
                percentage = (count / total_ideas * 100) if total_ideas > 0 else 0
                print(f"{i:3d}. {theme:<50} {count:5d} ideas ({percentage:5.1f}%)")
        else:
            print("\n📋 No themes assigned")
        
        # Print code counts  
        if code_frequency:
            sorted_codes = sorted(code_frequency.items(), key=lambda x: x[1], reverse=True)
            print(f"\n🏷️  CODE COUNTS (Total: {len(sorted_codes)} codes assigned to {sum(code_frequency.values())} ideas)")
            print("-" * 70)
            
            for i, (code, count) in enumerate(sorted_codes, 1):
                percentage = (count / total_ideas * 100) if total_ideas > 0 else 0
                print(f"{i:3d}. {code:<50} {count:5d} ideas ({percentage:5.1f}%)")
        else:
            print("\n🏷️  No codes assigned")
        
        # Summary statistics
        print(f"\n📈 SUMMARY:")
        print("-" * 70)
        print(f"   Total ideas extracted: {total_ideas}")
        print(f"   Unique themes assigned: {len(theme_frequency)}")
        print(f"   Unique codes assigned: {len(code_frequency)}")
        
        if total_ideas > 0:
            avg_codes = sum(code_frequency.values()) / total_ideas
            avg_themes = sum(theme_frequency.values()) / total_ideas
            print(f"   Average codes per idea: {avg_codes:.2f}")
            print(f"   Average themes per idea: {avg_themes:.2f}")
        
        print("=" * 80)