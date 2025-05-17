import re
from typing import List, Union
import models

class TextFinalizer:

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
        return [self.finalize_with_tracking(item) for item in data]

# Example / test section
if __name__ == "__main__":
    
    import models
    from modules.utils import csvHandler
    
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