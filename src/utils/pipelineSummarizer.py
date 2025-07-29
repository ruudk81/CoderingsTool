"""
Pipeline Summary Analysis Module

Provides comprehensive analysis and reporting for CoderingsTool pipeline results
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict


class PipelineSummarizer:
    """Generates detailed summaries and analytics for pipeline results"""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        
    def generate_summary(self, 
                        code_assigned_results: List[Any],
                        theme_enriched_codebook: Optional[Any] = None,
                        enriched_codebook: Optional[List[Any]] = None) -> None:
        """
        Generate comprehensive summary of pipeline results
        
        Args:
            code_assigned_results: List of CodeAssignedModel objects
            theme_enriched_codebook: ThemeEnrichedCodebookModel object
            enriched_codebook: Legacy codebook format (optional)
        """
        
        if not code_assigned_results:
            print("No code assignment results available for analysis.")
            return
            
        # Extract basic statistics
        stats = self._calculate_basic_stats(code_assigned_results)
        frequencies = self._calculate_frequencies(code_assigned_results)
        
        # Print enhanced summary sections
        self._print_header("DETAILED ANALYSIS SUMMARY")
        
        # 1. Theme Analysis
        if frequencies['theme_frequency'] and theme_enriched_codebook:
            self._print_theme_analysis(frequencies, theme_enriched_codebook)
        
        # 2. Code Coverage
        if theme_enriched_codebook and frequencies['code_frequency']:
            self._print_code_coverage(frequencies, theme_enriched_codebook)
        
        # 3. Response Coverage
        self._print_response_coverage(code_assigned_results, stats)
        
        # 4. Confidence Distribution
        if stats['all_confidences']:
            self._print_confidence_distribution(stats['all_confidences'])
        
        # 5. Theme Reach
        if frequencies['theme_frequency']:
            self._print_theme_reach(code_assigned_results, frequencies, stats)
        
        # 6. Export Summary
        self._print_export_summary(frequencies, stats)
        
        print("\n" + "=" * 80)
    
    def _calculate_basic_stats(self, code_assigned_results: List[Any]) -> Dict[str, Any]:
        """Calculate basic statistics from results"""
        total_responses = len(code_assigned_results)
        total_ideas = sum(len(resp.response_ideas) for resp in code_assigned_results if resp.response_ideas)
        total_assignments = sum(
            len([idea for idea in resp.response_ideas if idea and idea.assigned_codes]) 
            for resp in code_assigned_results if resp.response_ideas
        )
        
        # Extract all confidence scores
        all_confidences = []
        for resp in code_assigned_results:
            if resp.response_ideas:
                for idea in resp.response_ideas:
                    if idea and idea.assignment_confidence is not None:
                        all_confidences.append(idea.assignment_confidence)
        
        return {
            'total_responses': total_responses,
            'total_ideas': total_ideas,
            'total_assignments': total_assignments,
            'all_confidences': all_confidences
        }
    
    def _calculate_frequencies(self, code_assigned_results: List[Any]) -> Dict[str, Dict]:
        """Calculate code and theme frequencies"""
        code_frequency = {}
        theme_frequency = {}
        
        for resp in code_assigned_results:
            if resp.response_ideas:
                for idea in resp.response_ideas:
                    if idea and idea.assigned_codes:
                        for code in idea.assigned_codes:
                            code_frequency[code] = code_frequency.get(code, 0) + 1
                    if idea and idea.assigned_themes:
                        for theme in idea.assigned_themes:
                            theme_frequency[theme] = theme_frequency.get(theme, 0) + 1
        
        return {
            'code_frequency': code_frequency,
            'theme_frequency': theme_frequency
        }
    
    def _print_header(self, title: str) -> None:
        """Print a formatted section header"""
        print("\n" + "=" * 80)
        print(f"📊 {title}")
        print("=" * 80)
    
    def _print_theme_analysis(self, frequencies: Dict, theme_enriched_codebook: Any) -> None:
        """Print hierarchical theme analysis"""
        print("\n🎯 THEME ANALYSIS (Hierarchical View):")
        print("-" * 70)
        
        # Build theme-to-codes mapping
        theme_to_codes = defaultdict(list)
        for entry in theme_enriched_codebook.codes:
            if entry.theme:
                theme_to_codes[entry.theme].append(entry.code)
        
        # Sort themes by assignment frequency
        theme_items = sorted(frequencies['theme_frequency'].items(), key=lambda x: x[1], reverse=True)
        
        for theme, total_assignments in theme_items:
            print(f"\n   📌 {theme} ({total_assignments} total assignments)")
            
            # Show codes under this theme
            if theme in theme_to_codes:
                theme_codes = theme_to_codes[theme]
                print(f"      Codes in theme ({len(theme_codes)}):")
                
                # Sort codes by frequency within theme
                code_freq_in_theme = [
                    (code, frequencies['code_frequency'].get(code, 0)) 
                    for code in theme_codes
                ]
                code_freq_in_theme.sort(key=lambda x: x[1], reverse=True)
                
                # Show top 5 codes
                for code, freq in code_freq_in_theme[:5]:
                    pct = (freq / total_assignments * 100) if total_assignments > 0 else 0
                    print(f"        • {code}: {freq} ({pct:.1f}% of theme)")
                
                if len(theme_codes) > 5:
                    remaining_freq = sum(f for _, f in code_freq_in_theme[5:])
                    print(f"        • ... {len(theme_codes) - 5} more codes: {remaining_freq} assignments")
    
    def _print_code_coverage(self, frequencies: Dict, theme_enriched_codebook: Any) -> None:
        """Print code coverage analysis"""
        print("\n\n📈 CODE COVERAGE ANALYSIS:")
        print("-" * 70)
        
        total_codes = len(theme_enriched_codebook.codes)
        used_codes = len(frequencies['code_frequency'])
        unused_codes_count = total_codes - used_codes
        
        print(f"   Total codes in codebook: {total_codes}")
        print(f"   Codes actually used: {used_codes} ({used_codes/total_codes*100:.1f}%)")
        print(f"   Codes never assigned: {unused_codes_count} ({unused_codes_count/total_codes*100:.1f}%)")
        
        # Find unused codes
        all_codes = {entry.code for entry in theme_enriched_codebook.codes}
        used_codes_set = set(frequencies['code_frequency'].keys())
        unused_codes_list = sorted(all_codes - used_codes_set)
        
        if unused_codes_list:
            print(f"\n   Unused codes ({len(unused_codes_list)}):")
            for i, code in enumerate(unused_codes_list[:10]):
                print(f"      - {code}")
            if len(unused_codes_list) > 10:
                print(f"      ... and {len(unused_codes_list) - 10} more")
    
    def _print_response_coverage(self, code_assigned_results: List[Any], stats: Dict) -> None:
        """Print response coverage analysis"""
        print("\n\n📋 RESPONSE COVERAGE ANALYSIS:")
        print("-" * 70)
        
        responses_with_codes = 0
        responses_without_codes = 0
        ideas_with_codes = 0
        ideas_without_codes = 0
        
        for resp in code_assigned_results:
            has_any_code = False
            if resp.response_ideas:
                for idea in resp.response_ideas:
                    if idea and idea.assigned_codes and len(idea.assigned_codes) > 0:
                        ideas_with_codes += 1
                        has_any_code = True
                    else:
                        ideas_without_codes += 1
            
            if has_any_code:
                responses_with_codes += 1
            else:
                responses_without_codes += 1
        
        total_responses = stats['total_responses']
        total_ideas = stats['total_ideas']
        
        print(f"   Responses with at least one code: {responses_with_codes} ({responses_with_codes/total_responses*100:.1f}%)")
        print(f"   Responses with no codes: {responses_without_codes} ({responses_without_codes/total_responses*100:.1f}%)")
        print(f"   Ideas successfully coded: {ideas_with_codes} ({ideas_with_codes/total_ideas*100:.1f}%)")
        print(f"   Ideas without codes: {ideas_without_codes} ({ideas_without_codes/total_ideas*100:.1f}%)")
    
    def _print_confidence_distribution(self, all_confidences: List[float]) -> None:
        """Print confidence distribution analysis"""
        print("\n\n🎯 ASSIGNMENT CONFIDENCE DISTRIBUTION:")
        print("-" * 70)
        
        confidence_bins = {
            "Excellent (0.9-1.0)": 0,
            "Good (0.7-0.8)": 0,
            "Moderate (0.5-0.6)": 0,
            "Poor (0.3-0.4)": 0,
            "Very Poor (0.0-0.2)": 0
        }
        
        # Categorize confidences
        for conf in all_confidences:
            if conf >= 0.9:
                confidence_bins["Excellent (0.9-1.0)"] += 1
            elif conf >= 0.7:
                confidence_bins["Good (0.7-0.8)"] += 1
            elif conf >= 0.5:
                confidence_bins["Moderate (0.5-0.6)"] += 1
            elif conf >= 0.3:
                confidence_bins["Poor (0.3-0.4)"] += 1
            else:
                confidence_bins["Very Poor (0.0-0.2)"] += 1
        
        # Print distribution
        for category, count in confidence_bins.items():
            pct = (count / len(all_confidences) * 100)
            bar = "█" * int(pct / 2)  # Visual bar chart
            print(f"   {category}: {count:5d} ({pct:5.1f}%) {bar}")
        
        # Print statistics
        print(f"\n   Mean confidence: {np.mean(all_confidences):.3f}")
        print(f"   Median confidence: {np.median(all_confidences):.3f}")
        print(f"   Std deviation: {np.std(all_confidences):.3f}")
    
    def _print_theme_reach(self, code_assigned_results: List[Any], frequencies: Dict, stats: Dict) -> None:
        """Print theme reach analysis"""
        print("\n\n📊 THEME REACH (Unique Responses):")
        print("-" * 70)
        
        # Track unique responses per theme
        theme_to_responses = defaultdict(set)
        
        for resp in code_assigned_results:
            if resp.response_ideas:
                themes_in_response = set()
                for idea in resp.response_ideas:
                    if idea and idea.assigned_themes:
                        for theme in idea.assigned_themes:
                            themes_in_response.add(theme)
                
                for theme in themes_in_response:
                    theme_to_responses[theme].add(resp.respondent_id)
        
        # Sort by unique response count
        theme_reach = [(theme, len(responses)) for theme, responses in theme_to_responses.items()]
        theme_reach.sort(key=lambda x: x[1], reverse=True)
        
        print("   Themes by number of unique responses covered:")
        for i, (theme, unique_count) in enumerate(theme_reach[:10]):
            pct = (unique_count / stats['total_responses'] * 100)
            total_assigns = frequencies['theme_frequency'].get(theme, 0)
            avg_per_response = total_assigns / unique_count if unique_count > 0 else 0
            print(f"      {i+1:2d}. {theme}: {unique_count} responses ({pct:.1f}%), "
                  f"{total_assigns} total assignments (avg {avg_per_response:.1f}/response)")
    
    def _print_export_summary(self, frequencies: Dict, stats: Dict) -> None:
        """Print export summary"""
        print("\n\n💾 EXPORT SUMMARY:")
        print("-" * 70)
        
        code_freq = frequencies['code_frequency']
        theme_freq = frequencies['theme_frequency']
        
        print(f"   Total unique codes assigned: {len(code_freq)}")
        print(f"   Total unique themes assigned: {len(theme_freq)}")
        print(f"   Total code assignments made: {sum(code_freq.values())}")
        print(f"   Total theme assignments made: {sum(theme_freq.values())}")
        
        if stats['total_ideas'] > 0:
            print(f"   Average codes per idea: {sum(code_freq.values()) / stats['total_ideas']:.2f}")
            print(f"   Average themes per idea: {sum(theme_freq.values()) / stats['total_ideas']:.2f}")