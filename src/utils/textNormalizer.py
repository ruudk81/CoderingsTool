import os, sys; sys.path.extend([p for p in [os.getcwd().split('coderingsTool')[0] + suffix for suffix in ['', 'coderingsTool', 'coderingsTool/src', 'coderingsTool/src/utils']] if p not in sys.path]) if 'coderingsTool' in os.getcwd() else None

import re
from typing import List, Union, Optional 
from pydantic import BaseModel, Field, field_validator
import models
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
    def __init__(self, config: Optional[NormalizerConfig] = None, verbose: bool = False):
        self.config = config if config is not None else NormalizerConfig()
        self.verbose_reporter = VerboseReporter(verbose)
    
    #TODO: language recognition
    
    def replace_slash(self, text: str) -> str: 
        return re.sub(r'\s*/\s*|/', ' of ', text) #TODO: remove or if DUTCH
    
    def remove_symbols(self, text: str) -> str:
        escaped_punctuation = re.escape(self.config.custom_symbols)
        text = re.sub(f"[{escaped_punctuation}]", " ", text)
        return text.strip()
    
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
            text = self.remove_symbols(text)
            text = self.normalize_whitespace(text)
            text = self.handle_empty(text)
            
            return text
        except Exception as e:
            print(f"Error processing text: {e}")
            return self.config.na_placeholder
    
    def normalize_with_tracking(self, data: models.PreprocessModel) -> models.PreprocessModel:
        
        normalized_text = self.normalize_response(data.response)

        return models.PreprocessModel(respondent_id= data.respondent_id, response = normalized_text)
 
    def normalize_responses(self, data: List[models.PreprocessModel]) -> List[models.PreprocessModel]:
        stats = ProcessingStats()
        stats.start_timing()
        stats.input_count = len(data)
        
        self.verbose_reporter.step_start("Text Normalization")
        
        # Track changes
        symbol_changes = 0
        case_changes = 0
        whitespace_changes = 0
        invalid_filtered = 0
        
        results = []
        for item in data:
            original = item.response
            normalized = self.normalize_with_tracking(item)
            results.append(normalized)
            
            # Track what changed
            if original != original.lower():
                case_changes += 1
            if any(symbol in original for symbol in self.config.custom_symbols):
                symbol_changes += 1
            if re.search(r'\s{2,}', original or ''):
                whitespace_changes += 1
            if normalized.response == self.config.na_placeholder:
                invalid_filtered += 1
        
        stats.output_count = len(results) - invalid_filtered
        stats.end_timing()
        
        # Report statistics
        self.verbose_reporter.stat_line(f"Started with {stats.input_count} responses")
        if symbol_changes > 0:
            self.verbose_reporter.stat_line(f"Symbol removal: {symbol_changes} responses updated")
        if case_changes > 0:
            self.verbose_reporter.stat_line(f"Case normalization: {case_changes} responses updated")
        if whitespace_changes > 0:
            self.verbose_reporter.stat_line(f"Whitespace cleanup: {whitespace_changes} responses updated")
        if invalid_filtered > 0:
            self.verbose_reporter.stat_line(f"Invalid responses filtered: {invalid_filtered} responses")
        
        self.verbose_reporter.step_complete(f"Completed with {stats.output_count} valid responses")
        
        return results
    
# Example / test section
if __name__ == "__main__":
    
    import models
    from utils import csvHandler
    
    filename     = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
    id_column    = "DLNMID"
    var_name     = "Q20"
    
    csv_handler = csvHandler.CsvHandler()
    structured_raw_text = csv_handler.load_from_csv(filename, 'data', models.ResponseModel)

    normalizer = TextNormalizer()
    
    preprocess_data = [item.to_model(models.PreprocessModel) for item in structured_raw_text]
    results_list = normalizer.normalize_responses(preprocess_data)
    print("Results as list:")
    for result in results_list:
        print(f"ID: {result.respondent_id}, Text: {result.response}")  