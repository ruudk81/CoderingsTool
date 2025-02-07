import re
from typing import List, Union
import models
from .verboseReporter import VerboseReporter, ProcessingStats

class TextFinalizer:
    
    def __init__(self, verbose: bool = False):
        self.verbose_reporter = VerboseReporter(verbose)

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
        
        text = text.lower()
        text = self.capitalize_first_letter(text)
        text = self.ensure_ending_punctuation(text)
        text = self.remove_duplicate_punctuation(text)
        text = self.fix_spacing_after_punctuation(text)
            
        return text
        
        
    def finalize_with_tracking(self, data: models.PreprocessModel) -> models.PreprocessModel:
        
        finalized_text = self.finalize_response(data.response)

        return models.PreprocessModel(respondent_id=data.respondent_id, response =finalized_text)
 
    def finalize_responses(self, data: List[models.PreprocessModel]) -> List[models.PreprocessModel]:
        stats = ProcessingStats()
        stats.start_timing()
        stats.input_count = len(data)
        
        self.verbose_reporter.step_start("Text Finalization")
        
        # Track changes
        capitalization_fixes = 0
        punctuation_additions = 0
        format_cleanup = 0
        
        results = []
        for item in data:
            original = item.response
            finalized = self.finalize_with_tracking(item)
            results.append(finalized)
            
            # Track what changed
            if original and len(original) > 0:
                if original[0] != original[0].upper():
                    capitalization_fixes += 1
                if not original.endswith(('.', '!', '?')):
                    punctuation_additions += 1
                if re.search(r'\.{2,}|\?{2,}|!{2,}|\s{2,}', original):
                    format_cleanup += 1
        
        stats.output_count = len(results)
        stats.end_timing()
        
        # Report statistics
        if capitalization_fixes > 0:
            self.verbose_reporter.stat_line(f"Capitalization fixes: {capitalization_fixes} responses")
        if punctuation_additions > 0:
            self.verbose_reporter.stat_line(f"Punctuation additions: {punctuation_additions} responses")
        if format_cleanup > 0:
            self.verbose_reporter.stat_line(f"Format cleanup: {format_cleanup} responses")
        
        self.verbose_reporter.step_complete("Finalization completed")
        
        return results

# Example / test section
if __name__ == "__main__":
    
    import models
    from utils import csvHandler
    
    filename     = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
    id_column    = "DLNMID"
    var_name     = "Q20"
    
    csv_handler = csvHandler.CsvHandler()
    preprocess_data = csv_handler.load_from_csv(filename, 'data', models.PreprocessModel)

    finalizer = TextFinalizer()
    
    
    results_list = finalizer.finalize_responses(preprocess_data)
  
    print("Results as list:")
    for result in results_list:
        print(f"ID: {result.respondent_id}, Text: {result.response}")