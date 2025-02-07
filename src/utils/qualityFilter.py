import asyncio
import functools
import nest_asyncio #for Spyder
from typing import Dict, List, Optional, Union
from pydantic import BaseModel
import instructor
from openai import OpenAI

from config import DEFAULT_MODEL, OPENAI_API_KEY, DEFAULT_LANGUAGE
from prompts import GRADER_INSTRUCTIONS
import models

# Patch OpenAI client with instructor for structured output
client = instructor.patch(OpenAI(api_key=OPENAI_API_KEY)) 

class GraderConfig(BaseModel):
    model: str = DEFAULT_MODEL
    batch_size: int = 20 #TODO: DEFAULT_BATCHSIZE
    temperature: float = 0.0 #TODO: DEFAULT_TEMPERATURE
    max_tokens: int = 4000 #TODO: DEFAULT_TOKENS
    language: str = DEFAULT_LANGUAGE
    retries: int = 3 #TODO: DEFAULT_RETRIES

class Grader:
    def __init__(
        self, 
        responses: List[models.DescriptiveModel], 
        var_lab: str,
        config: Optional[GraderConfig] = None):
        
        self.responses = responses
        self.question = var_lab
        self.config = config or GraderConfig()
        self.client = client
        self.grader_instructions = GRADER_INSTRUCTIONS 
        self._results: List[models.DescriptiveModel] = [] 

    def _batch(self) -> List[List[tuple]]:
        indexed = [(i, r.respondent_id, r.response) for i, r in enumerate(self.responses)]
        return [indexed[i:i + self.config.batch_size] for i in range(0, len(indexed), self.config.batch_size)]

    def _build_prompt(self, var_lab: str, batch: List[tuple]) -> str:
        
        responses_text = "\n".join(f"respondent_id: {rid}, response: \"{response}\"" for _, rid, response in batch)
        return self.grader_instructions.format(
            language=self.config.language,
            var_lab=var_lab,
            responses=responses_text)

    async def _call_openai_api(self, prompt: str) -> List[models.DescriptiveModel]:
        tries = 0
        max_tries = self.config.retries
        
        while tries < max_tries:
            tries += 1
            try:
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(
                    None,
                    functools.partial(
                        self.client.chat.completions.create,
                        model=self.config.model,
                        response_model=List[models.DescriptiveModel],
                        max_retries=3,   
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens
                    )
                )
                return response
                
            except Exception as e:
                print(f"\nAPI call failed on attempt {tries}/{max_tries}:")
                print(f"Error: {str(e)}")
                
                if tries >= max_tries:
                    raise
                
                # Exponential backoff
                await asyncio.sleep(2 ** tries)
                continue

    async def _grade_batch(self, batch: List[tuple]) -> List[models.DescriptiveModel]:
        prompt = self._build_prompt(self.question, batch)
        response_data = await self._call_openai_api(prompt)
        return response_data

    async def _process_all_batches(self):
        batches = self._batch()
        tasks = [self._grade_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        total_failures = 0
        for i, batch_result in enumerate(batch_results):
            if isinstance(batch_result, Exception):
                print(f"Batch {i+1} processing failed after all retries: {str(batch_result)}")
                total_failures += 1
                continue

            self._results.extend(batch_result)
             
        if total_failures > 0:
            print(f"{total_failures} out of {len(batches)} batches failed completely after all retries")

    def grade(self) -> List[models.DescriptiveModel]:
        nest_asyncio.apply()
        asyncio.run(self._process_all_batches())
        return self._results

    def filter(self) -> List[models.DescriptiveModel]:
        return [r for r in self._results if not r.quality_filter]

    def summary(self) -> Dict[str, Union[int, float]]:
        total = len(self._results)
        meaningless = sum(1 for r in self._results if r.quality_filter)
        meaningful = total - meaningless

        return {
            "total_responses": total,
            "meaningful_responses": meaningful,
            "meaningless_responses": meaningless,
            "meaningful_percentage": round((meaningful / total) * 100, 2) if total > 0 else 0}

# example/test section
if __name__ == "__main__":
    
    from utils import data_io, csvHandler
    
    filename     = "M241030 Koninklijke Vezet Kant en Klaar 2024 databestand.sav"
    id_column    = "DLNMID"
    var_name     = "Q20"

    csv_handler          = csvHandler.CsvHandler()
    filepath             = csv_handler.get_filepath(filename, 'preprocessed')
    data_loader          = data_io.DataLoader()
    var_lab              = data_loader.get_varlab(filename = filename, var_name = var_name)

    preprocessed_text     = csv_handler.load_from_csv(filename, 'preprocessed', models.PreprocessModel)

    config = GraderConfig(
        language="Dutch",
        batch_size=20,
        retries=3 )

    grader = Grader(preprocessed_text, var_lab, config)
    all_results = grader.grade()
    filtered = grader.filter()

    summary = grader.summary()
    print("\nSummary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nAll results:")
    for r in all_results:
        if r.quality_filter:
        #print(r)
            print(f"{r.respondent_id}: {r.response} ")

    # print("\nMeaningful responses only:")
    # for r in filtered:
    #     print(f"{r.respondent_id}: {r.response}")