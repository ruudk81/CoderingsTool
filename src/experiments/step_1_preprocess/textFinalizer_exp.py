import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import re
from typing import List, Union

# === MODELS ========================================================================================================
import models

# === UTILS ========================================================================================================
from utils.verboseReporter import VerboseReporter, ProcessingStats


class TextFinalizer:
    
    def __init__(self, verbose: bool = False, prompt_printer = None):
        self.verbose_reporter = VerboseReporter(verbose, capture_logging=True)
        
        # Configuration reporting
        if self.verbose_reporter.enabled:
            self.verbose_reporter.empty_line()
            self.verbose_reporter.stat_line("Text finalizer configuration:")
            self.verbose_reporter.stat_line("Capitalization: First letter uppercase", indent=1)
            self.verbose_reporter.stat_line("Punctuation: Ensure ending punctuation", indent=1)
            self.verbose_reporter.stat_line("Cleanup: Remove duplicate punctuation and fix spacing", indent=1)

    @staticmethod
    def capitalize_first_letter(text: str) -> str:
        if not text or len(text) == 0:
            return text
        return text[0].upper() + text[1:]
    
    @staticmethod
    def ensure_ending_punctuation(text: str) -> str:
        if not text or len(text) == 0:
            return text
        if text[-1] in '.!?':
            return text
        return text + '.'
    
    @staticmethod
    def remove_duplicate_punctuation(text: str) -> str:
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'\?{2,}', '?', text)
        text = re.sub(r'!{2,}', '!', text)
        text = re.sub(r'\s+', ' ', text)
        return text
    
    @staticmethod
    def fix_spacing_after_punctuation(text: str) -> str:
        text = re.sub(r'([.!?])([a-zA-Z])', r'\1 \2', text)
        text = re.sub(r'\s+([.!?,;:])', r'\1', text)
        return text
    
    def finalize_response(self, text: Union[str, None, object]) -> str:
        try:
            if not isinstance(text, str) or not text:
                return text
                
            text = text.lower()
            text = self.capitalize_first_letter(text)
            text = self.ensure_ending_punctuation(text)
            text = self.remove_duplicate_punctuation(text)
            text = self.fix_spacing_after_punctuation(text)
                
            return text
        except Exception as e:
            # Improved error reporting
            truncated_text = str(text)[:50] + "..." if len(str(text)) > 50 else str(text)
            self.verbose_reporter.error(f"Error finalizing text '{truncated_text}': {e}")
            return text
        
        
    def finalize_with_tracking(self, data: models.PreprocessedModel) -> models.PreprocessedModel:
        
        finalized_text = self.finalize_response(data.response)

        return models.PreprocessedModel(respondent_id=data.respondent_id, response =finalized_text)
 
    def finalize_responses(self, data: List[models.PreprocessedModel]) -> List[models.PreprocessedModel]:
        stats = ProcessingStats()
        stats.start_timing()
        stats.input_count = len(data)

        # Always show main progress
        self.verbose_reporter.empty_line()
        self.verbose_reporter.stat_line(f"Processing {len(data)} responses for finalization...")
             
        # Track changes and examples
        capitalization_fixes = 0
        punctuation_additions = 0
        format_cleanup = 0
        spacing_fixes = 0
        transformation_examples = []
        
        # Calculate initial quality metrics
        total_length_before = 0
        responses_needing_fixes = 0
        
        results = []
        for i, item in enumerate(data):
            # Verbose progress indicators for large datasets
            if self.verbose_reporter.enabled and len(data) > 1000 and i % 500 == 0 and i > 0:
                self.verbose_reporter.progress_line(i, len(data), "finalizing")
            
            original = item.response
            finalized = self.finalize_with_tracking(item)
            results.append(finalized)
            
            # Track quality metrics
            if isinstance(original, str) and original:
                total_length_before += len(original)
                needs_fix = False
                
                # Track specific changes
                if len(original) > 0 and original[0] != original[0].upper():
                    capitalization_fixes += 1
                    needs_fix = True
                if not original.endswith(('.', '!', '?')):
                    punctuation_additions += 1
                    needs_fix = True
                if re.search(r'\.{2,}|\?{2,}|!{2,}', original):
                    format_cleanup += 1
                    needs_fix = True
                if re.search(r'([.!?])([a-zA-Z])', original):
                    spacing_fixes += 1
                    needs_fix = True
                
                if needs_fix:
                    responses_needing_fixes += 1
                
                # Collect transformation examples for debugging
                if (len(transformation_examples) < 5 and 
                    original != finalized.response and
                    isinstance(finalized.response, str)):
                    transformation_examples.append((original, finalized.response))
        
        stats.output_count = len(results)
        stats.end_timing()

        # Store statistics as instance attributes for app display
        self.stats = {
            'input_count': stats.input_count,
            'output_count': stats.output_count,
            'capitalization_fixes': capitalization_fixes,
            'punctuation_additions': punctuation_additions,
            'format_cleanup': format_cleanup,
            'spacing_fixes': spacing_fixes
        }

        # Always show main completion stats (as was before)
        self.verbose_reporter.empty_line()
        self.verbose_reporter.stat_line(f"Finalization completed: {stats.input_count} → {stats.output_count} responses")
        
        # Verbose detailed transformation statistics
        if self.verbose_reporter.enabled:
            if capitalization_fixes > 0:
                self.verbose_reporter.stat_line(f"Capitalization fixes: {capitalization_fixes} responses")
            if punctuation_additions > 0:
                self.verbose_reporter.stat_line(f"Punctuation additions: {punctuation_additions} responses")
            if format_cleanup > 0:
                self.verbose_reporter.stat_line(f"Format cleanup: {format_cleanup} responses")
            if spacing_fixes > 0:
                self.verbose_reporter.stat_line(f"Spacing fixes: {spacing_fixes} responses")
            
                      
            # Store examples for end-of-phase summary (don't show here)
            self.transformation_examples = transformation_examples if transformation_examples else []
        
        return results

