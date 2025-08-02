import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

"""
Pipeline Summary Analysis Module

Provides simple count analysis for CoderingsTool pipeline results
"""

# === MODULES ========================================================================================================
# Standard library imports
from typing import List, Dict, Any, Optional

# === MODELS ========================================================================================================
# No models imports needed for this module

# === CONFIG ========================================================================================================
# No config imports needed for this module

# === UTILS ========================================================================================================
from verboseReporter import VerboseReporter


class PipelineSummarizer:
    """Generates count summaries for themes and codes"""
    
    def __init__(self, verbose: bool = True, prompt_printer = None):
        self.verbose_reporter = VerboseReporter(verbose, capture_logging=True)
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
            self.verbose_reporter.warning("No code assignment results available for count analysis.")
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
        
        self.verbose_reporter.section_header("THEME AND CODE COUNT ANALYSIS", emoji="📊")
        
        # Print theme counts
        if theme_frequency:
            sorted_themes = sorted(theme_frequency.items(), key=lambda x: x[1], reverse=True)
            self.verbose_reporter.stat_line(f"📋 THEME COUNTS (Total: {len(sorted_themes)} themes assigned to {sum(theme_frequency.values())} ideas)")
            
            for i, (theme, count) in enumerate(sorted_themes, 1):
                percentage = (count / total_ideas * 100) if total_ideas > 0 else 0
                self.verbose_reporter.stat_line(f"{i:3d}. {theme:<50} {count:5d} ideas ({percentage:5.1f}%)")
        else:
            self.verbose_reporter.stat_line("📋 No themes assigned")
        
        # Print code counts  
        if code_frequency:
            sorted_codes = sorted(code_frequency.items(), key=lambda x: x[1], reverse=True)
            self.verbose_reporter.stat_line(f"🏷️  CODE COUNTS (Total: {len(sorted_codes)} codes assigned to {sum(code_frequency.values())} ideas)")
            
            for i, (code, count) in enumerate(sorted_codes, 1):
                percentage = (count / total_ideas * 100) if total_ideas > 0 else 0
                self.verbose_reporter.stat_line(f"{i:3d}. {code:<50} {count:5d} ideas ({percentage:5.1f}%)")
        else:
            self.verbose_reporter.stat_line("🏷️  No codes assigned")
        
        # Summary statistics
        summary_stats = {
            "Total ideas extracted": total_ideas,
            "Unique themes assigned": len(theme_frequency),
            "Unique codes assigned": len(code_frequency)
        }
        
        if total_ideas > 0:
            avg_codes = sum(code_frequency.values()) / total_ideas
            avg_themes = sum(theme_frequency.values()) / total_ideas
            summary_stats["Average codes per idea"] = f"{avg_codes:.2f}"
            summary_stats["Average themes per idea"] = f"{avg_themes:.2f}"
        
        self.verbose_reporter.summary("SUMMARY", summary_stats, emoji="📈")