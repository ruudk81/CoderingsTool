import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

# === MODULES ========================================================================================================
import re
from typing import List, Union, Optional
from pydantic import BaseModel, Field, field_validator

# === MODELS ========================================================================================================
import models

# === UTILS ========================================================================================================
from .verboseReporter import VerboseReporter, ProcessingStats


class NormalizerConfig(BaseModel):
    custom_symbols: str = Field(default="'#%&:;<=>@[\]^_{|}~-", description="Symbols to remove during normalization")
    na_placeholder: str = Field(default="<NA>", description="Placeholder for invalid/empty text")
    min_length: int = Field(default=1, description="Minimum valid text length")
    
    @field_validator('min_length')
    def validate_min_length(cls, v):
        if v < 1:
            raise ValueError("min_length must be at least 1")
        return v

class TextNormalizer:
    def __init__(self, config: Optional[NormalizerConfig] = None, verbose: bool = False, prompt_printer = None):
        self.config = config if config is not None else NormalizerConfig()
        self.verbose_reporter = VerboseReporter(verbose, capture_logging=True)
        
        # Report configuration if verbose
        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line("Text normalizer configuration:")
            self.verbose_reporter.stat_line(f"  • Minimum length: {self.config.min_length} characters")
            self.verbose_reporter.stat_line(f"  • NA placeholder: '{self.config.na_placeholder}'")
            self.verbose_reporter.stat_line(f"  • Custom symbols: '{self.config.custom_symbols}'")
    
    #TODO: language recognition
    
    def replace_slash(self, text: str) -> str: 
        return re.sub(r'\s*/\s*|/', ' of ', text) #TODO: remove or if DUTCH
    
    # def remove_symbols(self, text: str) -> str:
    #     escaped_punctuation = re.escape(self.config.custom_symbols)
    #     text = re.sub(f"[{escaped_punctuation}]", " ", text)
    #     return text.strip()
    
    def normalize_whitespace(self, text: str) -> str:
        text = " ".join(text.split())
        text = re.sub(r"\s+([,.;!?])", r"\1", text)
        return text.strip()
    
    def handle_empty(self, text: Union[str, None, object]) -> str:
        if not isinstance(text, str) or not text or len(text.strip()) <= self.config.min_length:
            return self.config.na_placeholder
        return text
    
    def normalize_response(self, text: Union[str, None, object]) -> str:
        try:
            if not isinstance(text, str):
                return self.config.na_placeholder
                
            text = text.lower()
            text = self.replace_slash(text)
            #text = self.remove_symbols(text)
            text = self.normalize_whitespace(text)
            text = self.handle_empty(text)
            
            return text
        except Exception as e:
            # Improved error reporting
            truncated_text = str(text)[:50] + "..." if len(str(text)) > 50 else str(text)
            self.verbose_reporter.error(f"Error processing text '{truncated_text}': {e}")
            return self.config.na_placeholder
    
    def normalize_with_tracking(self, data: models.PreprocessedModel) -> models.PreprocessedModel:
        
        normalized_text = self.normalize_response(data.response)

        return models.PreprocessedModel(respondent_id= data.respondent_id, response = normalized_text)
 
    def normalize_responses(self, data: List[models.PreprocessedModel]) -> List[models.PreprocessedModel]:
        stats = ProcessingStats()
        stats.start_timing()
        stats.input_count = len(data)
        
        # Always show main progress
        print(f"Processing {len(data)} responses for normalization...")
        
        # Verbose configuration details
        if self.verbose_reporter.enabled:
            self.verbose_reporter.stat_line(f"Configuration: min_length={self.config.min_length}, placeholder='{self.config.na_placeholder}'")
        
        # Track changes and examples
        symbol_changes = 0
        case_changes = 0
        whitespace_changes = 0
        invalid_filtered = 0
        slash_changes = 0
        transformation_examples = []
        
        # Calculate initial quality metrics
        total_length_before = 0
        valid_responses_before = 0
        
        results = []
        for i, item in enumerate(data):
            # Verbose progress indicators for large datasets
            if self.verbose_reporter.enabled and len(data) > 1000 and i % 500 == 0 and i > 0:
                self.verbose_reporter.progress_line(i, len(data), "normalizing")
            
            original = item.response
            normalized = self.normalize_with_tracking(item)
            results.append(normalized)
            
            # Track quality metrics
            if isinstance(original, str):
                total_length_before += len(original)
                valid_responses_before += 1
                
                # Track specific changes
                if original != original.lower():
                    case_changes += 1
                if re.search(r'\s{2,}', original):
                    whitespace_changes += 1
                if '/' in original:
                    slash_changes += 1
                
                # Collect transformation examples for debugging
                if (len(transformation_examples) < 5 and 
                    original != normalized.response and 
                    normalized.response != self.config.na_placeholder):
                    transformation_examples.append((original, normalized.response))
            
            if normalized.response == self.config.na_placeholder:
                invalid_filtered += 1
        
        stats.output_count = len(results) - invalid_filtered
        stats.end_timing()
        
        # Calculate quality metrics
        valid_results = [r for r in results if r.response != self.config.na_placeholder]
        avg_length_before = total_length_before / max(1, valid_responses_before)
        avg_length_after = sum(len(r.response) for r in valid_results) / max(1, len(valid_results))
        retention_rate = (len(valid_results) / len(data)) * 100 if data else 0
        
        # Performance metrics
        total_time = stats.get_duration()
        avg_time_per_response = total_time / len(data) if data else 0
        responses_per_second = len(data) / total_time if total_time > 0 else 0
        
        # Always show main completion stats (as was before)
        print(f"Normalization completed: {stats.input_count} → {stats.output_count} responses")
        
        # Verbose detailed transformation statistics
        if self.verbose_reporter.enabled:
            if case_changes > 0:
                self.verbose_reporter.stat_line(f"Case normalization: {case_changes} responses updated")
            if whitespace_changes > 0:
                self.verbose_reporter.stat_line(f"Whitespace cleanup: {whitespace_changes} responses updated")
            if slash_changes > 0:
                self.verbose_reporter.stat_line(f"Slash replacement: {slash_changes} responses updated")
            if invalid_filtered > 0:
                self.verbose_reporter.stat_line(f"Invalid responses filtered: {invalid_filtered} responses")
            
            # Quality metrics
            self.verbose_reporter.stat_line(f"Quality metrics:")
            self.verbose_reporter.stat_line(f"  • Average length before: {avg_length_before:.1f} characters")
            self.verbose_reporter.stat_line(f"  • Average length after: {avg_length_after:.1f} characters")
            self.verbose_reporter.stat_line(f"  • Data retention rate: {retention_rate:.1f}%")
            
            # Performance metrics
            self.verbose_reporter.stat_line(f"Performance metrics:")
            self.verbose_reporter.stat_line(f"  • Total time: {total_time:.2f}s")
            self.verbose_reporter.stat_line(f"  • Average per response: {avg_time_per_response:.3f}s")
            self.verbose_reporter.stat_line(f"  • Responses per second: {responses_per_second:.1f}")
            
            # Show transformation examples in verbose mode
            if transformation_examples:
                self.verbose_reporter.correction_samples(transformation_examples[:3])
        
        return results
    
